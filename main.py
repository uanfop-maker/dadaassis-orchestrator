from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from job_manager import ensure_dirs, create_job, pick_job, complete_job, fail_job, get_job, list_pending, list_recent, cost_summary, sweep_timeouts
from orchestrator import route, call_fable, call_gemini, dispatch_to_opus, circuit_record_success, circuit_record_failure

_BUILD_SHA = os.getenv("BUILD_SHA", "dev")

app = FastAPI(
    title="DaDaAssis Orchestrator",
    description="Multi-Agent Orchestrator：管理 Job Queue、路由到 cc-opus 或 Fable、Timeout Sweeper",
    version="1.0.0",
)

INTER_SERVICE_SECRET_IN = os.getenv("INTER_SERVICE_SECRET_B", "")  # cc-opus → cc-v2 (orchestrator)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = "https://api.telegram.org"


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    ensure_dirs()
    _start_sweeper()
    print(f"[ORCH] DaDaAssis Orchestrator started sha={_BUILD_SHA}", flush=True)


def _start_sweeper() -> None:
    def _loop() -> None:
        while True:
            time.sleep(60)
            try:
                affected = sweep_timeouts()
                if affected:
                    print(f"[SWEEPER] {len(affected)} timed-out jobs processed", flush=True)
            except Exception as exc:
                print(f"[SWEEPER] error: {exc}", flush=True)
    threading.Thread(target=_loop, daemon=True, name="job-sweeper").start()


# ──────────────────────────────────────────────────────────────────────────────
# Health / Version
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sha": _BUILD_SHA}


@app.get("/version")
async def version() -> dict:
    return {"sha": _BUILD_SHA}


# ──────────────────────────────────────────────────────────────────────────────
# Job 分派 API（cc-v2 / Claude Code 呼叫此端點）
# ──────────────────────────────────────────────────────────────────────────────

class DispatchRequest(BaseModel):
    chat_id: int
    message_thread_id: int | None = None
    prompt: str
    task_type: str | None = None        # 留空讓 Orchestrator 自動決定
    explicit_cmd: str | None = None     # /deep /write /story 等顯式指令
    priority: int = Field(default=3, ge=1, le=5)
    extra: dict | None = None


@app.post("/dispatch")
async def dispatch(req: DispatchRequest, x_trace_id: str | None = Header(default=None, alias="X-Trace-Id")) -> dict:
    """
    Claude Code (cc-v2) 呼叫此端點分派任務。
    Orchestrator 自動路由到 cc-opus / Fable / local，並回傳 job_id。
    """
    trace_id = x_trace_id or str(uuid.uuid4())
    target = route(req.prompt, explicit_cmd=req.explicit_cmd)
    task_type = req.task_type or (
        "deep_reasoning" if target == "opus"
        else "creative" if target == "fable"
        else "local"
    )

    job = create_job(
        chat_id=req.chat_id,
        prompt=req.prompt,
        task_type=task_type,
        target=target,
        message_thread_id=req.message_thread_id,
        extra={**(req.extra or {}), "trace_id": trace_id, "priority": req.priority},
    )

    if job is None:
        return {"status": "duplicate", "message": "相同任務處理中"}

    if target == "fable":
        # Fable：同步呼叫，立即回傳結果並 push 到 Telegram
        j = pick_job(job["job_id"])
        if j:
            result, usage = call_fable(req.prompt, trace_id=trace_id)
            complete_job(j["job_id"], result, usage)
            if TELEGRAM_BOT_TOKEN:
                asyncio.create_task(_push_result_to_telegram(j, result))
            return {"status": "done", "job_id": j["job_id"], "target": "fable", "result": result}

    elif target == "gemini":
        # Gemini：同步呼叫，大型上下文分析
        j = pick_job(job["job_id"])
        if j:
            result, usage = call_gemini(req.prompt, trace_id=trace_id)
            complete_job(j["job_id"], result, usage)
            # Push to Telegram if token set
            if TELEGRAM_BOT_TOKEN:
                asyncio.create_task(_push_result_to_telegram(j, result))
            return {"status": "done", "job_id": j["job_id"], "target": "gemini", "result": result}

    elif target == "opus":
        # cc-opus：非同步 HTTP push
        j = pick_job(job["job_id"])
        if j:
            ok = dispatch_to_opus(j)
            if not ok:
                # 熔斷降級：改為 local
                fail_job(j["job_id"], "circuit breaker open, fallback to local")
                return {"status": "fallback_local", "job_id": j["job_id"], "message": "cc-opus 不可用，請改用本地模式"}
        return {"status": "dispatched", "job_id": job["job_id"], "target": "opus"}

    # local：由呼叫方自行處理
    return {"status": "local", "job_id": job["job_id"], "target": "local"}


# ──────────────────────────────────────────────────────────────────────────────
# Callback（cc-opus 完成後呼叫）
# ──────────────────────────────────────────────────────────────────────────────

class CallbackRequest(BaseModel):
    job_id: str
    status: str          # "done" | "failed"
    result: str | None = None
    error: str | None = None
    usage: dict | None = None
    elapsed_ms: int | None = None


@app.post("/callback")
async def job_callback(
    req: CallbackRequest,
    x_dadaassis_auth: str | None = Header(default=None, alias="X-DaDaAssis-Auth"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> dict:
    if INTER_SERVICE_SECRET_IN and x_dadaassis_auth != INTER_SERVICE_SECRET_IN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    print(f"[CALLBACK] job={req.job_id} status={req.status} trace={x_trace_id}", flush=True)

    if req.status == "done":
        ok = complete_job(req.job_id, req.result or "", req.usage)
        if not ok:
            print(f"[CALLBACK] job={req.job_id} not in running, discarded (idempotent)", flush=True)
            return {"accepted": True, "idempotent_discard": True}
        circuit_record_success()
        # 嘗試把結果推送到 Telegram
        job = get_job(req.job_id)
        if job and TELEGRAM_BOT_TOKEN:
            asyncio.create_task(_push_result_to_telegram(job, req.result or ""))
        return {"accepted": True, "idempotent_discard": False}
    else:
        job = fail_job(req.job_id, req.error or "unknown error")
        circuit_record_failure()
        return {"accepted": True, "new_status": (job or {}).get("status")}


async def _push_result_to_telegram(job: dict, result: str) -> None:
    import httpx
    chat_id = job.get("chat_id")
    thread_id = job.get("message_thread_id")
    if not chat_id:
        return
    params = {"chat_id": chat_id, "text": result, "parse_mode": "Markdown"}
    if thread_id:
        params["message_thread_id"] = thread_id
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=params)
    except Exception as exc:
        print(f"[PUSH_TG] error job={job.get('job_id')} err={exc}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Job 查詢
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/job/{job_id}")
async def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "target": job.get("target"),
        "result": job.get("result"),
        "attempt": job.get("attempt"),
        "trace_id": job.get("trace_id"),
    }


@app.get("/jobs/pending")
async def list_pending_jobs() -> dict:
    jobs = list_pending()
    return {"count": len(jobs), "jobs": [{"job_id": j["job_id"], "task_type": j["task_type"], "target": j["target"]} for j in jobs]}


@app.get("/jobs/history")
async def jobs_history(status: str = "done", limit: int = 20) -> dict:
    if status not in ("done", "dead", "running", "pending"):
        raise HTTPException(status_code=400, detail="status must be done|dead|running|pending")
    jobs = list_recent(status, min(limit, 100))
    return {
        "status": status,
        "count": len(jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "target": j.get("target"),
                "task_type": j.get("task_type"),
                "chat_id": j.get("chat_id"),
                "created_at": j.get("created_at"),
                "completed_at": j.get("completed_at"),
                "attempt": j.get("attempt"),
                "result_preview": (j.get("result") or "")[:80],
            }
            for j in jobs
        ],
    }


@app.get("/costs")
async def cost_stats() -> dict:
    return cost_summary()


@app.get("/circuit")
async def circuit_status() -> dict:
    from orchestrator import _load_circuit
    return _load_circuit()


# ──────────────────────────────────────────────────────────────────────────────
# Team Dispatch（Harper + Benjamin + Lucas → Grok 整合）
# ──────────────────────────────────────────────────────────────────────────────

HARPER_ENDPOINT = os.getenv("CC_HARPER_ENDPOINT", "http://cc-harper:8080")
BENJAMIN_ENDPOINT = os.getenv("CC_BENJAMIN_ENDPOINT", "http://cc-benjamin:8080")
LUCAS_ENDPOINT = os.getenv("CC_LUCAS_ENDPOINT", "http://cc-lucas:8080")
INTER_SERVICE_SECRET_OUT = os.getenv("INTER_SERVICE_SECRET_A", "")

_team_results: dict[str, dict] = {}  # team_id → {role: result, ...}
_team_locks: dict[str, asyncio.Lock] = {}


class TeamDispatchRequest(BaseModel):
    chat_id: int
    message_thread_id: int | None = None
    prompt: str
    use_opus_synthesis: bool = False  # True → Opus 4.8 整合，預設 Sonnet


@app.post("/team-dispatch")
async def team_dispatch(req: TeamDispatchRequest) -> dict:
    """Fan-out 到 Harper + Benjamin + Lucas，等全部回傳後 Grok 整合，結果 push 到 TG。"""
    team_id = "team_" + str(uuid.uuid4())[:8]
    _team_results[team_id] = {}
    _team_locks[team_id] = asyncio.Lock()

    self_host = os.getenv("SELF_HOST", "cc-orchestrator")
    self_port = os.getenv("SELF_PORT", "8080")
    callback_url = f"http://{self_host}:{self_port}/team-callback"

    asyncio.create_task(_run_team(req, team_id, callback_url))
    return {"status": "dispatched", "team_id": team_id}


async def _run_team(req: TeamDispatchRequest, team_id: str, callback_url: str) -> None:
    import httpx as _httpx

    agents = [
        ("harper", HARPER_ENDPOINT),
        ("benjamin", BENJAMIN_ENDPOINT),
        ("lucas", LUCAS_ENDPOINT),
    ]

    async def _dispatch_agent(role: str, endpoint: str) -> None:
        job_id = f"{team_id}_{role}"
        payload = {
            "job_id": job_id,
            "team_id": team_id,
            "prompt": req.prompt,
            "callback_url": callback_url,
        }
        headers = {"Content-Type": "application/json", "X-DaDaAssis-Auth": INTER_SERVICE_SECRET_OUT}
        async with _httpx.AsyncClient(timeout=20) as client:
            try:
                r = await client.post(f"{endpoint}/job", json=payload, headers=headers)
                if r.status_code not in (200, 202):
                    print(f"[TEAM] {role} dispatch failed status={r.status_code}", flush=True)
            except Exception as exc:
                print(f"[TEAM] {role} dispatch err={exc}", flush=True)
                async with _team_locks[team_id]:
                    _team_results[team_id][role] = f"（{role} 不可用：{exc}）"

    await asyncio.gather(*[_dispatch_agent(role, ep) for role, ep in agents])

    # 等待所有 agent 回傳（最多 120 秒）
    for _ in range(60):
        await asyncio.sleep(2)
        async with _team_locks[team_id]:
            done = len(_team_results[team_id])
        if done >= 3:
            break

    async with _team_locks[team_id]:
        results = dict(_team_results[team_id])

    if not results:
        await _push_team_result(req.chat_id, req.message_thread_id, "（所有 agent 無回應）")
        return

    # Grok 整合
    synthesis = await _grok_synthesis(req.prompt, results, req.use_opus_synthesis)
    await _push_team_result(req.chat_id, req.message_thread_id, synthesis)

    # 清理
    _team_results.pop(team_id, None)
    _team_locks.pop(team_id, None)


async def _grok_synthesis(prompt: str, results: dict, use_opus: bool) -> str:
    parts = []
    for role in ("harper", "benjamin", "lucas"):
        val = results.get(role, "（未回應）")
        parts.append(f"**{role.capitalize()}**：\n{val}")

    synthesis_prompt = f"""你是 Grok，DaDaAssis 4人AI核心團隊的總協調官。

使用者的問題：
{prompt}

團隊分析：
{chr(10).join(parts)}

你的任務：
1. 整合三位專家的觀點
2. 解決內部衝突，取得最佳答案
3. 輸出一份流暢、有條理的最終回答

請用繁體中文輸出，格式清晰，直接呈現最終答案。"""

    if use_opus:
        from orchestrator import dispatch_to_opus as _opus
        result, _ = call_gemini(synthesis_prompt)
        return result

    # 預設：OpenRouter Sonnet 同步呼叫
    from orchestrator import OPENROUTER_API_KEY, OPENROUTER_URL, SONNET_MODEL
    import requests
    if not OPENROUTER_API_KEY:
        return f"（Grok 整合：API key 未設定）\n\n{'---'.join(parts)}"
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://dadaassis.zeabur.app", "X-Title": "DaDaAssis Grok"},
            json={"model": SONNET_MODEL, "messages": [{"role": "user", "content": synthesis_prompt}], "max_tokens": 3000},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"（Grok 整合失敗：{exc}）\n\n{'---'.join(parts)}"


class TeamCallbackRequest(BaseModel):
    job_id: str
    team_id: str
    role: str
    status: str
    result: str


@app.post("/team-callback")
async def team_callback(
    req: TeamCallbackRequest,
    x_auth: str | None = Header(default=None, alias="X-DaDaAssis-Auth"),
) -> dict:
    if INTER_SERVICE_SECRET_IN and x_auth != INTER_SERVICE_SECRET_IN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    print(f"[TEAM-CB] team={req.team_id} role={req.role} status={req.status}", flush=True)
    if req.team_id in _team_locks:
        async with _team_locks[req.team_id]:
            _team_results[req.team_id][req.role] = req.result if req.status == "done" else f"（{req.role} 失敗：{req.result}）"
    return {"accepted": True}


async def _push_team_result(chat_id: int, thread_id: int | None, result: str) -> None:
    import httpx as _httpx
    if not TELEGRAM_BOT_TOKEN:
        return
    params = {"chat_id": chat_id, "text": result, "parse_mode": "Markdown"}
    if thread_id:
        params["message_thread_id"] = thread_id
    async with _httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=params)
        except Exception as exc:
            print(f"[TEAM] push_tg err={exc}", flush=True)

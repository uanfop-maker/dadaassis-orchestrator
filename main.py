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
from orchestrator import (
    route, call_fable, call_gemini, dispatch_to_opus, dispatch_to_fable, dispatch_to_persona,
    circuit_record_success, circuit_record_failure, PERSONA_TARGETS,
)

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
        else "persona" if target in PERSONA_TARGETS
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
        # cc-fable：非同步 HTTP push，跟 opus 同一套模式（獨立服務，用完即 clear）
        j = pick_job(job["job_id"])
        if j:
            ok = dispatch_to_fable(j)
            if not ok:
                # 熔斷降級：改為 local
                fail_job(j["job_id"], "circuit breaker open, fallback to local")
                return {"status": "fallback_local", "job_id": j["job_id"], "message": "cc-fable 不可用，請改用本地模式"}
        return {"status": "dispatched", "job_id": job["job_id"], "target": "fable"}

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

    elif target in PERSONA_TARGETS:
        # Harper/Benjamin/Lucas 單一角色 solo 派工：非同步 HTTP push，跟 opus 同一套模式
        j = pick_job(job["job_id"])
        if j:
            ok = dispatch_to_persona(target, j)
            if not ok:
                fail_job(j["job_id"], f"{target} circuit breaker open or dispatch failed, fallback to local")
                return {"status": "fallback_local", "job_id": j["job_id"], "message": f"{target} 目前不可用，請改用本地模式"}
        return {"status": "dispatched", "job_id": job["job_id"], "target": target}

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

    # 熔斷器要記哪個 key：opus/fable 各自獨立 key；persona job 記自己的 key（+ 共享的 OAuth key）
    job_before = get_job(req.job_id)
    job_target = (job_before or {}).get("target", "opus")
    circuit_key = job_target if job_target in PERSONA_TARGETS or job_target == "fable" else "opus"

    if req.status == "done":
        ok = complete_job(req.job_id, req.result or "", req.usage)
        if not ok:
            print(f"[CALLBACK] job={req.job_id} not in running, discarded (idempotent)", flush=True)
            return {"accepted": True, "idempotent_discard": True}
        circuit_record_success(circuit_key)
        if circuit_key in PERSONA_TARGETS:
            circuit_record_success("persona_oauth")
        # 嘗試把結果推送到 Telegram
        job = get_job(req.job_id)
        if job and TELEGRAM_BOT_TOKEN:
            asyncio.create_task(_push_result_to_telegram(job, req.result or ""))
        return {"accepted": True, "idempotent_discard": False}
    else:
        # persona worker 目前把失敗原因放在 result（不是 error），兩邊都吃避免漏記
        error_detail = req.error or req.result or "unknown error"
        job = fail_job(req.job_id, error_detail)
        circuit_record_failure(circuit_key)
        if circuit_key in PERSONA_TARGETS and "OAUTH_LIMIT" in error_detail:
            # 三個 persona 共用同一組 Max 訂閱 credential，一個撞牆代表全部要停
            circuit_record_failure("persona_oauth")
            print(f"[CALLBACK] persona_oauth circuit tripped by {job_target} job={req.job_id}", flush=True)
        return {"accepted": True, "new_status": (job or {}).get("status")}


TELEGRAM_MAX_CHARS = 4000  # Telegram 硬限制 4096，留緩衝給 chunk 標記


def _chunk_text(text: str, max_chars: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """依段落邊界切割長文字，避免超過 Telegram sendMessage 長度上限（4096）。"""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n\n", 0, max_chars)
        if split_at <= 0:
            split_at = remaining.rfind("\n", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].rstrip("\n"))
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def _send_telegram_message(chat_id, thread_id, text: str, label: str = "") -> None:
    """發送一則訊息，處理長度切割 + Markdown 解析失敗 fallback，並確實記錄失敗（不吞錯）。"""
    import httpx

    chunks = _chunk_text(text)
    async with httpx.AsyncClient(timeout=30) as client:
        for i, chunk in enumerate(chunks):
            prefix = f"[{i + 1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
            params = {"chat_id": chat_id, "text": prefix + chunk, "parse_mode": "Markdown"}
            if thread_id:
                params["message_thread_id"] = thread_id
            try:
                resp = await client.post(f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=params)
                if resp.status_code != 200:
                    # legacy Markdown 對 LLM 產出的文字很容易解析失敗（未配對的 * _ 等），降級成純文字重試
                    params.pop("parse_mode", None)
                    resp2 = await client.post(f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=params)
                    if resp2.status_code != 200:
                        print(
                            f"[PUSH_TG] FAILED {label} chunk={i + 1}/{len(chunks)} "
                            f"status={resp.status_code} body={resp.text[:300]} "
                            f"plain_status={resp2.status_code} plain_body={resp2.text[:300]}",
                            flush=True,
                        )
            except Exception as exc:
                print(f"[PUSH_TG] EXCEPTION {label} chunk={i + 1}/{len(chunks)} err={exc}", flush=True)


async def _push_result_to_telegram(job: dict, result: str) -> None:
    chat_id = job.get("chat_id")
    thread_id = job.get("message_thread_id")
    if not chat_id:
        return
    await _send_telegram_message(chat_id, thread_id, result, label=f"job={job.get('job_id')}")


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
    keys = ["opus", "persona_oauth", *PERSONA_TARGETS]
    return {key: _load_circuit(key) for key in keys}


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
        return await _opus_synthesis(synthesis_prompt, parts)

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


OPUS_SYNTHESIS_MAX_ATTEMPTS = 3
OPUS_SYNTHESIS_POLL_TIMEOUT_SEC = 90  # per attempt


async def _opus_synthesis(synthesis_prompt: str, parts: list[str]) -> str:
    """use_opus_synthesis=True 時真的呼叫 cc-opus（先前這裡誤呼叫了 Gemini，import 了 dispatch_to_opus 沒用）。

    ganamia 定的規則：opus 是 synthesis 的主力，重試最多 3 次；3 次都失敗才讓 Fable
    出現當最終備援——Fable 平常不參與（太貴），不是每次 synthesis 都跑一輪。3 次都
    失敗以外的情況，Fable 只在 ganamia 明確指定時才出現（既有的 /write /story /poem
    顯式指令路由，跟這裡無關，不受影響）。

    cc-opus 是非同步 job/callback 模式，這裡借用既有的 job_manager + 通用 /callback
    端點，用輪詢等結果，讓呼叫端看起來像同步呼叫。
    """
    for attempt in range(1, OPUS_SYNTHESIS_MAX_ATTEMPTS + 1):
        result = await _try_opus_synthesis_once(synthesis_prompt, attempt)
        if result is not None:
            return result

    print(f"[TEAM] opus synthesis failed {OPUS_SYNTHESIS_MAX_ATTEMPTS}x, escalating to cc-fable", flush=True)
    fable_job = create_job(chat_id=0, prompt=synthesis_prompt, task_type="team_synthesis_fallback", target="fable")
    fj = pick_job(fable_job["job_id"]) if fable_job else None
    if fj and dispatch_to_fable(fj):
        cur = None
        for _ in range(OPUS_SYNTHESIS_POLL_TIMEOUT_SEC // 2):
            await asyncio.sleep(2)
            cur = get_job(fj["job_id"])
            if cur and cur["status"] in ("done", "dead"):
                break
        if cur and cur.get("status") == "done" and cur.get("result"):
            return cur["result"]
        fail_job(fj["job_id"], "fable dispatch timeout")

    # cc-fable 也不可用時的最終備援：直接同步呼叫 OpenRouter
    print("[TEAM] cc-fable dispatch/timeout failed, final fallback to inline call_fable", flush=True)
    fable_text, _usage = call_fable(synthesis_prompt, trace_id="team-synthesis-fable-fallback")
    return fable_text


async def _try_opus_synthesis_once(synthesis_prompt: str, attempt: int) -> str | None:
    """一次 opus synthesis 嘗試。成功回傳結果文字，失敗/逾時回傳 None（讓外層決定要不要重試）。"""
    job = create_job(chat_id=0, prompt=synthesis_prompt, task_type="team_synthesis", target="opus")
    j = pick_job(job["job_id"]) if job else None
    if not j:
        print(f"[TEAM] opus synthesis attempt={attempt}: job creation/pick failed", flush=True)
        return None

    if not dispatch_to_opus(j):
        fail_job(j["job_id"], "opus dispatch failed or circuit open")
        print(f"[TEAM] opus synthesis attempt={attempt}: dispatch failed job={j['job_id']}", flush=True)
        return None

    cur = None
    for _ in range(OPUS_SYNTHESIS_POLL_TIMEOUT_SEC // 2):
        await asyncio.sleep(2)
        cur = get_job(j["job_id"])
        if cur and cur["status"] in ("done", "dead"):
            break

    if cur and cur.get("status") == "done" and cur.get("result"):
        print(f"[TEAM] opus synthesis attempt={attempt}: done job={j['job_id']}", flush=True)
        return cur["result"]

    print(f"[TEAM] opus synthesis attempt={attempt}: timeout/failed job={j['job_id']} status={(cur or {}).get('status')}", flush=True)
    return None


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
    if not TELEGRAM_BOT_TOKEN:
        return
    await _send_telegram_message(chat_id, thread_id, result, label="team")

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
        # Fable：同步呼叫，立即回傳結果
        j = pick_job(job["job_id"])
        if j:
            result, usage = call_fable(req.prompt, trace_id=trace_id)
            complete_job(j["job_id"], result, usage)
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

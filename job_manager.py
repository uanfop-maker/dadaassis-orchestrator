from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

# Volume 掛載後路徑為 /data/_jobs/；本地開發走 APP_DIR/_jobs/
_APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
JOBS_ROOT = DATA_DIR / "_jobs"
STATE_DIR = DATA_DIR / "_state"

DIRS = {
    "pending": JOBS_ROOT / "pending",
    "running": JOBS_ROOT / "running",
    "done":    JOBS_ROOT / "done",
    "dead":    JOBS_ROOT / "dead",
}


def ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _job_id(chat_id: int | str, prompt: str) -> str:
    """去重 ID：chat + prompt摘要 + 5分鐘時間桶，防止同一請求重複進隊。"""
    bucket = int(time.time()) // 300
    raw = f"{chat_id}:{prompt[:120]}:{bucket}"
    return "job_" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def create_job(
    chat_id: int | str,
    prompt: str,
    task_type: str,
    target: str,
    message_thread_id: int | str | None = None,
    extra: dict | None = None,
    deadline_secs: int = 300,
) -> dict[str, Any] | None:
    """建立並寫入 pending/ job，若同 ID 已存在任何目錄則回 None（去重）。"""
    ensure_dirs()
    job_id = _job_id(chat_id, prompt)
    # 去重：掃四個目錄
    for d in DIRS.values():
        if (d / f"{job_id}.json").exists():
            return None

    now = int(time.time())
    job: dict[str, Any] = {
        "job_id": job_id,
        "trace_id": str(uuid.uuid4()),
        "status": "pending",
        "task_type": task_type,
        "target": target,
        "chat_id": int(chat_id),
        "message_thread_id": message_thread_id,
        "prompt": prompt,
        "extra": extra or {},
        "created_at": now,
        "dispatched_at": None,
        "deadline_ts": None,
        "attempt": 0,
        "result": None,
        "error_history": [],
        "usage": None,
    }
    path = DIRS["pending"] / f"{job_id}.json"
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    return job


def pick_job(job_id: str) -> dict[str, Any] | None:
    """把 pending → running（原子 rename），回傳更新後的 job。"""
    src = DIRS["pending"] / f"{job_id}.json"
    dst = DIRS["running"] / f"{job_id}.json"
    if not src.exists():
        return None
    job = json.loads(src.read_text())
    now = int(time.time())
    job["status"] = "running"
    job["attempt"] = job.get("attempt", 0) + 1
    job["dispatched_at"] = now
    job["deadline_ts"] = now + 300
    dst.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    src.unlink(missing_ok=True)
    return job


def complete_job(job_id: str, result: str, usage: dict | None = None) -> bool:
    """running → done。若不在 running/ 則表示已超時重派，回 False（冪等保護）。"""
    src = DIRS["running"] / f"{job_id}.json"
    if not src.exists():
        return False
    job = json.loads(src.read_text())
    job["status"] = "done"
    job["result"] = result
    job["usage"] = usage
    job["completed_at"] = int(time.time())
    dst = DIRS["done"] / f"{job_id}.json"
    dst.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    src.unlink(missing_ok=True)
    _append_cost_ledger(job)
    return True


def fail_job(job_id: str, error: str) -> dict[str, Any] | None:
    """running → failed：attempt < 3 則重回 pending，否則 → dead。回傳更新後的 job。"""
    src = DIRS["running"] / f"{job_id}.json"
    if not src.exists():
        return None
    job = json.loads(src.read_text())
    job["error_history"].append({"attempt": job["attempt"], "error": error, "ts": int(time.time())})
    if job["attempt"] < 3:
        job["status"] = "pending"
        job["dispatched_at"] = None
        job["deadline_ts"] = None
        dst = DIRS["pending"] / f"{job_id}.json"
    else:
        job["status"] = "dead"
        dst = DIRS["dead"] / f"{job_id}.json"
    dst.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    src.unlink(missing_ok=True)
    return job


def sweep_timeouts() -> list[dict[str, Any]]:
    """掃 running/ 中超時的 job，移回 pending 或 dead；回傳受影響的 job list。"""
    ensure_dirs()
    now = int(time.time())
    affected = []
    for f in DIRS["running"].glob("*.json"):
        try:
            job = json.loads(f.read_text())
            if job.get("deadline_ts") and now > job["deadline_ts"]:
                job_id = job["job_id"]
                result = fail_job(job_id, "timeout")
                if result:
                    affected.append(result)
        except Exception:
            pass
    # 清理 done/ dead/ 超過 7 天的舊 job
    cutoff = now - 7 * 86400
    for state in ("done", "dead"):
        for f in DIRS[state].glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except Exception:
                pass
    return affected


def get_job(job_id: str) -> dict[str, Any] | None:
    for d in DIRS.values():
        f = d / f"{job_id}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def list_pending() -> list[dict[str, Any]]:
    jobs = []
    for f in DIRS["pending"].glob("*.json"):
        try:
            jobs.append(json.loads(f.read_text()))
        except Exception:
            pass
    return sorted(jobs, key=lambda j: j.get("created_at", 0))


def _append_cost_ledger(job: dict) -> None:
    try:
        ledger = STATE_DIR / "cost_ledger.jsonl"
        usage = job.get("usage") or {}
        line = {
            "job_id": job["job_id"],
            "task_type": job.get("task_type"),
            "target": job.get("target"),
            "ts": int(time.time()),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": usage.get("cost_usd", 0),
        }
        with ledger.open("a") as fp:
            fp.write(json.dumps(line) + "\n")
    except Exception:
        pass

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

# ──────────────────────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FABLE_MODEL = os.getenv("FABLE_MODEL", "anthropic/claude-fable-5")
SONNET_MODEL = os.getenv("SONNET_MODEL", "anthropic/claude-sonnet-4-6")
OPUS_ENDPOINT = os.getenv("CC_OPUS_ENDPOINT", "http://cc-opus.zeabur.internal:8080")
INTER_SERVICE_SECRET_OUT = os.getenv("INTER_SERVICE_SECRET_A", "")  # cc-v2 → cc-opus

# 觸發 cc-opus 的關鍵詞
OPUS_KEYWORDS = [
    "深入分析", "深度分析", "架構設計", "trade-off", "tradeoff",
    "比較評估", "研究報告", "系統設計", "技術評估", "root cause",
    "根本原因", "為什麼", "should we", "compare", "evaluate",
]
FABLE_KEYWORDS = [
    "寫一篇", "寫一個", "創意", "故事", "詩", "文案", "仿寫",
    "寫作", "描述", "生成文章", "幫我寫", "write a", "create a story",
    "compose", "poem", "creative",
]

# ──────────────────────────────────────────────────────────────────────────────
# 熔斷器（cc-opus 專用）
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
_CIRCUIT_FILE = DATA_DIR / "_state" / "circuit.json"
_CIRCUIT_SILENCE_SECS = 300  # OPEN 後靜默 5 分鐘再試


def _load_circuit() -> dict:
    try:
        if _CIRCUIT_FILE.exists():
            return json.loads(_CIRCUIT_FILE.read_text())
    except Exception:
        pass
    return {"state": "CLOSED", "fails": 0, "open_at": 0}


def _save_circuit(c: dict) -> None:
    try:
        _CIRCUIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CIRCUIT_FILE.write_text(json.dumps(c))
    except Exception:
        pass


def circuit_record_success() -> None:
    c = _load_circuit()
    c["state"] = "CLOSED"
    c["fails"] = 0
    _save_circuit(c)


def circuit_record_failure() -> None:
    c = _load_circuit()
    c["fails"] = c.get("fails", 0) + 1
    if c["fails"] >= 3:
        c["state"] = "OPEN"
        c["open_at"] = int(time.time())
    _save_circuit(c)


def circuit_allow() -> bool:
    """是否允許發出請求（CLOSED or HALF-OPEN）。"""
    c = _load_circuit()
    if c["state"] == "CLOSED":
        return True
    if c["state"] == "OPEN":
        if int(time.time()) - c.get("open_at", 0) >= _CIRCUIT_SILENCE_SECS:
            # 進入 HALF-OPEN
            c["state"] = "HALF-OPEN"
            _save_circuit(c)
            return True
        return False
    # HALF-OPEN：允許一次試探
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator 路由決策
# ──────────────────────────────────────────────────────────────────────────────

def _classify(text: str, explicit_cmd: str | None = None) -> str:
    """回傳 'opus' | 'fable' | 'local'。"""
    if explicit_cmd in ("/deep", "/research"):
        return "opus"
    if explicit_cmd in ("/write", "/story", "/poem"):
        return "fable"
    lower = text.lower()
    for kw in OPUS_KEYWORDS:
        if kw.lower() in lower:
            return "opus"
    for kw in FABLE_KEYWORDS:
        if kw.lower() in lower:
            return "fable"
    # 長文本暗示需要深度推理
    if len(text) > 800:
        return "opus"
    return "local"


def route(
    text: str,
    explicit_cmd: str | None = None,
    fallback_local: bool = True,
) -> str:
    """決定任務路由，若 cc-opus 熔斷則降級 local。"""
    target = _classify(text, explicit_cmd)
    if target == "opus" and not circuit_allow():
        return "local"
    return target


# ──────────────────────────────────────────────────────────────────────────────
# Fable 呼叫（同步，cc-v2 直接呼叫 OpenRouter）
# ──────────────────────────────────────────────────────────────────────────────

def call_fable(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    trace_id: str | None = None,
) -> tuple[str, dict | None]:
    """呼叫 Fable 5，回傳 (text, usage_dict)。失敗回 (fallback_text, None)。"""
    if not OPENROUTER_API_KEY:
        return _fable_fallback(prompt), None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dadaassis.zeabur.app",
        "X-Title": "DaDaAssis Fable",
    }
    if trace_id:
        headers["X-Trace-Id"] = trace_id

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json={"model": FABLE_MODEL, "messages": messages, "max_tokens": 2000},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        return content, usage
    except Exception as exc:
        print(f"[ORCH] Fable call failed trace={trace_id} err={exc}", flush=True)
        return _fable_fallback(prompt), None


def _fable_fallback(prompt: str) -> str:
    return f"（Fable 模型暫不可用，已改用一般模式）\n\n{prompt}"


# ──────────────────────────────────────────────────────────────────────────────
# cc-opus 非同步分派（HTTP push）
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_to_opus(
    job: dict[str, Any],
    callback_url: str | None = None,
) -> bool:
    """
    向 cc-opus POST job，成功回 True，失敗更新熔斷器並回 False。
    callback_url 預設為 cc-v2 內網。
    """
    if not callback_url:
        cc_v2_host = os.getenv("CC_V2_INTERNAL_HOST", "cc-v2.zeabur.internal")
        cc_v2_port = os.getenv("CC_V2_INTERNAL_PORT", "8080")
        callback_url = f"http://{cc_v2_host}:{cc_v2_port}/callback"

    payload = {
        "job_id": job["job_id"],
        "task_type": job.get("task_type", "deep_reasoning"),
        "prompt": job.get("prompt", ""),
        "context": job.get("extra", {}),
        "callback_url": callback_url,
        "deadline_ts": job.get("deadline_ts", int(time.time()) + 300),
        "attempt": job.get("attempt", 1),
    }
    headers = {
        "Content-Type": "application/json",
        "X-DaDaAssis-Auth": INTER_SERVICE_SECRET_OUT,
        "X-Trace-Id": job.get("trace_id", str(uuid.uuid4())),
    }
    try:
        resp = requests.post(
            f"{OPUS_ENDPOINT}/job",
            headers=headers,
            json=payload,
            timeout=30,  # 含冷啟動時間
        )
        if resp.status_code in (200, 202):
            circuit_record_success()
            return True
        circuit_record_failure()
        return False
    except Exception as exc:
        print(f"[ORCH] dispatch_to_opus failed job={job.get('job_id')} err={exc}", flush=True)
        circuit_record_failure()
        return False


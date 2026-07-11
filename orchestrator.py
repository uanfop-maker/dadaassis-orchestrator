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
OPUS_ENDPOINT = os.getenv("CC_OPUS_ENDPOINT", "http://cc-opus:8080")
INTER_SERVICE_SECRET_OUT = os.getenv("INTER_SERVICE_SECRET_A", "")  # cc-v2 → cc-opus / persona workers
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Persona worker（Harper/Benjamin/Lucas）端點——同一組端點也被 main.py 的 /team-dispatch 使用
PERSONA_ENDPOINTS = {
    "harper": os.getenv("CC_HARPER_ENDPOINT", "http://cc-harper:8080"),
    "benjamin": os.getenv("CC_BENJAMIN_ENDPOINT", "http://cc-benjamin:8080"),
    "lucas": os.getenv("CC_LUCAS_ENDPOINT", "http://cc-lucas:8080"),
}
PERSONA_TARGETS = set(PERSONA_ENDPOINTS.keys())

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
GEMINI_KEYWORDS = [
    "診斷", "分析程式碼", "code review", "看一下這段", "parse", "解析",
    "文件解讀", "api doc", "大量資料", "長文本",
]
# 個別 persona 的自動判定關鍵詞（單一角色 solo 派工，非 /team-dispatch 的三人合議）
PERSONA_KEYWORDS = {
    "harper": [
        "查一下", "查資料", "最新消息", "最新資訊", "查詢一下", "情報",
        "現在的行情", "search for", "look up", "latest news",
    ],
    "benjamin": [
        "驗證邏輯", "程式碼審查", "debug 一下", "驗算",
        "檢查這段程式", "review this code", "算一下對不對",
    ],
    "lucas": [
        "唱反調", "反方觀點", "挑戰這個", "有沒有風險", "潑冷水",
        "devil's advocate", "找漏洞", "反過來想",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 熔斷器（per-key：opus / harper / benjamin / lucas / persona_oauth）
# ──────────────────────────────────────────────────────────────────────────────
#
# persona_oauth 是額外的「共享」熔斷器：harper/benjamin/lucas 共用同一組 Max 訂閱
# OAuth credential，任一 persona 回報 OAuth 類錯誤（額度/認證失效）時，三個 persona
# 都要一起降級，不能各自為政繼續打同一組已經出問題的 credential。

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
_CIRCUIT_SILENCE_SECS = 300  # OPEN 後靜默 5 分鐘再試


def _circuit_file(key: str = "opus") -> Path:
    # "opus" 沿用原本的檔名，避免既有部署升級時熔斷狀態被重置
    name = "circuit.json" if key == "opus" else f"circuit_{key}.json"
    return DATA_DIR / "_state" / name


def _load_circuit(key: str = "opus") -> dict:
    try:
        f = _circuit_file(key)
        if f.exists():
            return json.loads(f.read_text())
    except Exception:
        pass
    return {"state": "CLOSED", "fails": 0, "open_at": 0}


def _save_circuit(c: dict, key: str = "opus") -> None:
    try:
        f = _circuit_file(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(c))
    except Exception:
        pass


def circuit_record_success(key: str = "opus") -> None:
    c = _load_circuit(key)
    c["state"] = "CLOSED"
    c["fails"] = 0
    _save_circuit(c, key)


def circuit_record_failure(key: str = "opus") -> None:
    c = _load_circuit(key)
    c["fails"] = c.get("fails", 0) + 1
    if c["fails"] >= 3:
        c["state"] = "OPEN"
        c["open_at"] = int(time.time())
    _save_circuit(c, key)


def circuit_allow(key: str = "opus") -> bool:
    """是否允許發出請求（CLOSED or HALF-OPEN）。"""
    c = _load_circuit(key)
    if c["state"] == "CLOSED":
        return True
    if c["state"] == "OPEN":
        if int(time.time()) - c.get("open_at", 0) >= _CIRCUIT_SILENCE_SECS:
            # 進入 HALF-OPEN
            c["state"] = "HALF-OPEN"
            _save_circuit(c, key)
            return True
        return False
    # HALF-OPEN：允許一次試探
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator 路由決策
# ──────────────────────────────────────────────────────────────────────────────

_PERSONA_EXPLICIT_CMDS = {"/harper": "harper", "/benjamin": "benjamin", "/lucas": "lucas"}


def _classify(text: str, explicit_cmd: str | None = None) -> str:
    """回傳 'opus' | 'fable' | 'gemini' | 'harper' | 'benjamin' | 'lucas' | 'local'。

    顯式指令優先權最高，其次是關鍵詞分類；分類順序固定，避免不同規則搶同一句話。
    """
    if explicit_cmd in _PERSONA_EXPLICIT_CMDS:
        return _PERSONA_EXPLICIT_CMDS[explicit_cmd]
    if explicit_cmd in ("/deep", "/research"):
        return "opus"
    if explicit_cmd in ("/write", "/story", "/poem"):
        return "fable"
    if explicit_cmd in ("/diagnose", "/gemini", "/search"):
        return "gemini"
    lower = text.lower()
    for kw in GEMINI_KEYWORDS:
        if kw.lower() in lower:
            return "gemini"
    for kw in OPUS_KEYWORDS:
        if kw.lower() in lower:
            return "opus"
    for persona, kws in PERSONA_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lower:
                return persona
    for kw in FABLE_KEYWORDS:
        if kw.lower() in lower:
            return "fable"
    # 長文本暗示需要深度推理
    if len(text) > 800:
        return "gemini" if GEMINI_API_KEY else "opus"
    return "local"


def route(
    text: str,
    explicit_cmd: str | None = None,
    fallback_local: bool = True,
) -> str:
    """決定任務路由，若目標服務熔斷則降級。"""
    target = _classify(text, explicit_cmd)
    if target in PERSONA_TARGETS:
        # persona 專用熔斷器 + 三個 persona 共享的 OAuth 熔斷器，任一開啟就降級 local
        if not circuit_allow(target) or not circuit_allow("persona_oauth"):
            return "local"
        return target
    if target == "opus" and not circuit_allow("opus"):
        return "gemini" if GEMINI_API_KEY else "local"
    if target == "gemini" and not GEMINI_API_KEY:
        return "opus" if circuit_allow("opus") else "local"
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
# Gemini 呼叫（同步，大型上下文分析 / 診斷專用）
# ──────────────────────────────────────────────────────────────────────────────

def call_gemini(
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    trace_id: str | None = None,
) -> tuple[str, dict | None]:
    """呼叫 Gemini 2.5 Pro：優先直連 Google API，429/5xx 時 fallback OpenRouter。"""
    # 1. 直連 Google API
    if GEMINI_API_KEY:
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "已理解。"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, json={"contents": contents, "generationConfig": {"maxOutputTokens": 8192}, "tools": [{"google_search": {}}]}, timeout=timeout)
            if resp.status_code not in (429, 500, 502, 503):
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                meta = data.get("usageMetadata", {})
                usage = {"input_tokens": meta.get("promptTokenCount", 0), "output_tokens": meta.get("candidatesTokenCount", 0), "model": GEMINI_MODEL}
                print(f"[ORCH] Gemini/direct done trace={trace_id} in={usage['input_tokens']} out={usage['output_tokens']}", flush=True)
                return text, usage
            print(f"[ORCH] Gemini/direct rate-limited ({resp.status_code}), falling back to OpenRouter trace={trace_id}", flush=True)
        except Exception as exc:
            print(f"[ORCH] Gemini/direct err trace={trace_id} err={exc}, trying OpenRouter", flush=True)

    # 2. Fallback: OpenRouter Gemini
    if OPENROUTER_API_KEY:
        or_model = f"google/{GEMINI_MODEL}" if not GEMINI_MODEL.startswith("google/") else GEMINI_MODEL
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://dadaassis.zeabur.app", "X-Title": "DaDaAssis Gemini"},
                json={"model": or_model, "messages": messages, "max_tokens": 8192},
                timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage_raw = data.get("usage", {})
            usage = {"input_tokens": usage_raw.get("prompt_tokens", 0), "output_tokens": usage_raw.get("completion_tokens", 0), "model": or_model}
            print(f"[ORCH] Gemini/openrouter done trace={trace_id} in={usage['input_tokens']} out={usage['output_tokens']}", flush=True)
            return text, usage
        except Exception as exc:
            print(f"[ORCH] Gemini/openrouter err trace={trace_id} err={exc}", flush=True)
            return f"（Gemini 不可用：{exc}）", None

    return "（Gemini 不可用：未設定任何 API Key）", None


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
        self_host = os.getenv("SELF_HOST", "cc-orchestrator")
        self_port = os.getenv("SELF_PORT", "8080")
        callback_url = f"http://{self_host}:{self_port}/callback"

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


# ──────────────────────────────────────────────────────────────────────────────
# Persona worker 非同步分派（HTTP push）——單一角色 solo 派工，非 /team-dispatch 三人合議
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_to_persona(
    persona: str,
    job: dict[str, Any],
    callback_url: str | None = None,
) -> bool:
    """
    向指定 persona worker（harper/benjamin/lucas）POST job，走跟 dispatch_to_opus
    相同的 fire-and-forget 模式：cc-v2/orchestrator 不等結果，worker 完成後回呼
    callback_url（預設 orchestrator 自己的 /callback，跟 opus 共用同一個 handler）。

    成功回 True；失敗會同時記錄該 persona 專屬熔斷器 + 三個 persona 共用的
    persona_oauth 熔斷器（因為目前 harper/benjamin/lucas 是同一組 Max 訂閱
    credential，個別失敗有可能是共用資源出問題，寧可保守一起降級）。
    """
    endpoint = PERSONA_ENDPOINTS.get(persona)
    if not endpoint:
        return False

    if not callback_url:
        self_host = os.getenv("SELF_HOST", "cc-orchestrator")
        self_port = os.getenv("SELF_PORT", "8080")
        callback_url = f"http://{self_host}:{self_port}/callback"

    payload = {
        "job_id": job["job_id"],
        "team_id": job["job_id"],  # solo 派工沒有 team 概念，借用 job_id 滿足 worker 的必填欄位
        "prompt": job.get("prompt", ""),
        "context": job.get("extra", {}),
        "callback_url": callback_url,
        "attempt": job.get("attempt", 1),
    }
    headers = {
        "Content-Type": "application/json",
        "X-DaDaAssis-Auth": INTER_SERVICE_SECRET_OUT,
        "X-Trace-Id": job.get("trace_id", str(uuid.uuid4())),
    }
    try:
        resp = requests.post(
            f"{endpoint}/job",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 202):
            circuit_record_success(persona)
            return True
        circuit_record_failure(persona)
        return False
    except Exception as exc:
        print(f"[ORCH] dispatch_to_persona({persona}) failed job={job.get('job_id')} err={exc}", flush=True)
        circuit_record_failure(persona)
        return False

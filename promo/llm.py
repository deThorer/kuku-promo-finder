"""One swappable interface to any language model.

Change provider/model in .env only - no code changes:
    LLM_PROVIDER = gemini | groq | openrouter | openai | anthropic | ollama | openai_compatible
    LLM_MODEL    = model name for that provider

Standard library only (no SDKs), so it runs anywhere Python 3.10+ runs.
Every prompt and raw response is saved to outputs/llm_logs/ for auditing.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OPENAI_STYLE = {
    # provider: (default base URL, env var holding the key)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "openai_compatible": (None, "LLM_API_KEY"),  # set LLM_BASE_URL too
}


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.split(" #")[0].strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """The model's DAILY quota is used up (retrying today is pointless)."""


class ModelBusy(LLMError):
    """The model kept answering 'overloaded' (HTTP 500/502/503/504) through every retry."""


def _is_daily_quota(detail: str) -> bool:
    d = detail.lower()
    return "per day" in d or "perday" in d or "limit: 0" in d or "requests_per_day" in d


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float | None = None, log_dir: str | Path | None = "outputs/llm_logs"):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.model = model or os.getenv("LLM_MODEL", "")
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0.4"))
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "4"))  # seconds between calls (free-tier RPM)
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
        self.timeout = float(os.getenv("LLM_TIMEOUT", "90"))  # seconds to wait for one answer
        self.log_dir = Path(log_dir) if log_dir else None
        self._last_call = 0.0
        # models to switch to (same provider) when the current one's daily quota is used up
        fb = os.getenv("LLM_FALLBACK_MODELS", "")
        self.fallbacks = [m.strip() for m in fb.split(",") if m.strip() and m.strip() != self.model]
        self.exhausted: list[str] = []
        if not self.model:
            raise LLMError("Set LLM_MODEL in .env (e.g. the current Gemini Flash model name from Google AI Studio).")

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, json_mode: bool = True, tag: str = "call") -> str:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        while True:
            try:
                if self.provider == "gemini":
                    text = self._gemini(system, user, json_mode)
                elif self.provider == "anthropic":
                    text = self._anthropic(system, user)
                elif self.provider in OPENAI_STYLE:
                    text = self._openai_style(system, user, json_mode)
                else:
                    raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'")
                break
            except (QuotaExhausted, ModelBusy) as err:
                busy = isinstance(err, ModelBusy)
                self.exhausted.append(self.model)
                if not self.fallbacks:
                    if busy:
                        raise ModelBusy(f"{self.model} is overloaded and no fallback model is left. "
                                        f"Rerun the same command in a few minutes.") from None
                    raise QuotaExhausted(f"daily quota used up for: {', '.join(self.exhausted)}. "
                                         f"Add more models to LLM_FALLBACK_MODELS or wait for the reset.") from None
                nxt = self.fallbacks.pop(0)
                why = "is overloaded or not answering" if busy else "has used its daily quota"
                print(f"  [llm] {self.model} {why} -> switching to {nxt}")
                self.model = nxt
            finally:
                self._last_call = time.time()
        self._log(tag, system, user, text)
        return text

    def complete_json(self, system: str, user: str, tag: str = "call") -> dict:
        text = self.complete(system, user, json_mode=True, tag=tag)
        try:
            return extract_json(text)
        except ValueError:
            text = self.complete(system, user + "\n\nReturn ONLY valid JSON. No prose, no code fences.",
                                 json_mode=True, tag=tag + "_jsonretry")
            return extract_json(text)

    # --------------------------------------------------------------- providers
    def _gemini(self, system: str, user: str, json_mode: bool) -> str:
        key = _need_env("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        cfg: dict = {"temperature": self.temperature}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": cfg,
        }
        data = self._post(url, body, {"x-goog-api-key": key})
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise LLMError(f"Gemini returned no text: {json.dumps(data)[:500]}")
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def _openai_style(self, system: str, user: str, json_mode: bool) -> str:
        base, key_env = OPENAI_STYLE[self.provider]
        base = os.getenv("LLM_BASE_URL") or base
        if not base:
            raise LLMError("Set LLM_BASE_URL for provider openai_compatible")
        headers = {}
        if key_env:
            headers["Authorization"] = f"Bearer {_need_env(key_env)}"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post(base.rstrip("/") + "/chat/completions", body, headers)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError(f"No text in response: {json.dumps(data)[:500]}")

    def _anthropic(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": _need_env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}
        data = self._post("https://api.anthropic.com/v1/messages", body, headers)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    # ------------------------------------------------------------------ plumbing
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        hdrs = {"Content-Type": "application/json", "User-Agent": "promo-finder/0.1", **headers}
        delay = 5.0
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                full = e.read().decode("utf-8", errors="replace")
                detail = full[:600]
                if e.code == 429 and _is_daily_quota(full):
                    raise QuotaExhausted(f"daily quota used up for {self.model}") from None
                if e.code in (500, 502, 503, 504) and attempt >= self.max_retries:
                    raise ModelBusy(f"{self.model} overloaded (HTTP {e.code}) after {attempt} tries") from None
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After")
                    sleep = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    print(f"  [llm] HTTP {e.code}, retrying in {sleep:.0f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(sleep)
                    delay *= 2
                    continue
                raise LLMError(f"HTTP {e.code} from {self.provider}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                    http.client.HTTPException) as e:
                reason = getattr(e, "reason", None) or type(e).__name__
                if attempt < self.max_retries:
                    print(f"  [llm] no answer from {self.model} ({reason}), retrying in {delay:.0f}s "
                          f"(attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                # a model that keeps timing out is treated like an overloaded one -> fallback model
                raise ModelBusy(f"{self.model} gave no answer after {attempt} tries ({reason})") from None
        raise LLMError("Retries exhausted")

    def _log(self, tag: str, system: str, user: str, response: str) -> None:
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        rec = {"time": stamp, "provider": self.provider, "model": self.model, "tag": tag,
               "system": system, "user": user, "response": response}
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        (self.log_dir / f"{stamp}_{safe_tag}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")


def _need_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise LLMError(f"Missing {name} in .env")
    return val


def extract_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ``` fences or added prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Model did not return valid JSON: {text[:300]}")

"""Minimal OpenAI-compatible chat client using the Python standard library."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agp_research.config import DEFAULT_ENV_PATH, setting


@dataclass
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60
    cache_dir: Path | None = None
    log_path: Path | None = None
    last_call: dict[str, Any] = field(default_factory=dict, init=False)

    @classmethod
    def from_environment(
        cls, env_path: str | Path = DEFAULT_ENV_PATH
    ) -> "OpenAICompatibleClient | None":
        api_key = setting("OPENAI_API_KEY", env_path=env_path)
        if not api_key:
            return None
        timeout_text = setting("AGP_LLM_TIMEOUT", "60", env_path=env_path)
        try:
            timeout = int(timeout_text or "60")
        except ValueError as error:
            raise ValueError("AGP_LLM_TIMEOUT must be a whole number of seconds") from error
        if timeout <= 0:
            raise ValueError("AGP_LLM_TIMEOUT must be greater than zero")
        cache_text = setting("AGP_LLM_CACHE_DIR", ".agp_cache/llm", env_path=env_path)
        log_text = setting(
            "AGP_LLM_LOG_PATH", ".agp_logs/llm_requests.jsonl", env_path=env_path
        )
        return cls(
            api_key=api_key,
            model=setting("AGP_MODEL", "gpt-4.1-mini", env_path=env_path)
            or "gpt-4.1-mini",
            base_url=(
                setting(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1", env_path=env_path
                )
                or "https://api.openai.com/v1"
            ).rstrip("/"),
            timeout=timeout,
            cache_dir=Path(cache_text) if cache_text else None,
            log_path=Path(log_text) if log_text else None,
        )

    def complete_json(self, system: str, user: str) -> dict:
        content = self._request(system, user, json_mode=True)
        return json.loads(content)

    def complete_text(self, system: str, user: str) -> str:
        return self._request(system, user, json_mode=False)

    def _request(self, system: str, user: str, *, json_mode: bool) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        cache_key = self._cache_key(payload, system, user, json_mode)
        cached = self._read_cache(cache_key)
        if cached is not None:
            self._record_call(
                cache_key=cache_key,
                json_mode=json_mode,
                cache_hit=True,
                elapsed_seconds=0.0,
                usage=cached.get("usage"),
                status="ok",
            )
            return str(cached["content"])

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            self._record_call(
                cache_key=cache_key,
                json_mode=json_mode,
                cache_hit=False,
                elapsed_seconds=time.perf_counter() - started,
                usage=None,
                status=f"http_error_{error.code}",
            )
            raise RuntimeError(f"LLM API returned HTTP {error.code}: {detail}") from error
        content = str(result["choices"][0]["message"]["content"])
        usage = result.get("usage")
        self._write_cache(cache_key, content, usage)
        self._record_call(
            cache_key=cache_key,
            json_mode=json_mode,
            cache_hit=False,
            elapsed_seconds=time.perf_counter() - started,
            usage=usage,
            status="ok",
        )
        return content

    def _cache_key(
        self, payload: dict, system: str, user: str, json_mode: bool
    ) -> str:
        identity = {
            "version": 1,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": payload["temperature"],
            "json_mode": json_mode,
            "system": system,
            "user": user,
        }
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _read_cache(self, cache_key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) and "content" in data else None

    def _write_cache(self, cache_key: str, content: str, usage: Any) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{cache_key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"content": content, "usage": usage}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _record_call(
        self,
        *,
        cache_key: str,
        json_mode: bool,
        cache_hit: bool,
        elapsed_seconds: float,
        usage: Any,
        status: str,
    ) -> None:
        usage_data = usage if isinstance(usage, dict) else {}
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
            "model": self.model,
            "base_url": self.base_url,
            "json_mode": json_mode,
            "cache_hit": cache_hit,
            "network_request": not cache_hit,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
            "prompt_tokens": usage_data.get("prompt_tokens"),
            "completion_tokens": usage_data.get("completion_tokens"),
            "total_tokens": usage_data.get("total_tokens"),
        }
        self.last_call = record
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

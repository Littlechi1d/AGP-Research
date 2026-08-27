"""Minimal OpenAI-compatible chat client using the Python standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from agp_research.config import DEFAULT_ENV_PATH, setting


@dataclass
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60

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
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {error.code}: {detail}") from error
        return result["choices"][0]["message"]["content"]

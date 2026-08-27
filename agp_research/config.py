"""Small dependency-free loader for project settings stored in .env."""

from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Read simple KEY=VALUE settings without modifying os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    settings: dict[str, str] = {}
    for line_number, original_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")

        key, value = (part.strip() for part in line.split("=", 1))
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid .env variable name on line {line_number}: {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        settings[key] = value
    return settings


def setting(
    name: str,
    default: str | None = None,
    *,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> str | None:
    """Return a shell variable first, then .env value, then the default."""
    if name in os.environ:
        return os.environ[name]
    return read_env_file(env_path).get(name, default)

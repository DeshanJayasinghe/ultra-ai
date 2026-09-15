from __future__ import annotations

from os import environ
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only before dependencies install.
    load_dotenv = None


def load_environment_file(*, start: Path | None = None) -> bool:
    """Load the nearest WearWise AI .env file without overriding real env vars."""

    current = (start or Path.cwd()).resolve()
    search_roots = [current, *current.parents]

    for root in search_roots:
        env_path = root / ".env"
        pyproject_path = root / "pyproject.toml"
        if env_path.exists() and pyproject_path.exists():
            if load_dotenv is None:
                return _load_simple_dotenv(env_path)
            return load_dotenv(dotenv_path=env_path, override=False)

    return False


def _load_simple_dotenv(env_path: Path) -> bool:
    loaded = False
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value
            loaded = True

    return loaded

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dotenv_values(dotenv_path: str | None = None) -> dict[str, str]:
    path = Path(dotenv_path) if dotenv_path is not None else _repo_root() / ".env"
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_alpaca_credentials(dotenv_path: str | None = None) -> dict[str, str]:
    dotenv_values = load_dotenv_values(dotenv_path)
    api_key = os.getenv("ALPACA_API_KEY") or dotenv_values.get("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET") or dotenv_values.get("ALPACA_API_SECRET")
    api_base_url = os.getenv("ALPACA_API_BASE_URL") or dotenv_values.get(
        "ALPACA_API_BASE_URL"
    )

    return {
        "API_KEY": api_key or "",
        "API_SECRET": api_secret or "",
        "API_BASE_URL": api_base_url or "https://paper-api.alpaca.markets",
    }


def get_tiingo_credentials(dotenv_path: str | None = None) -> dict[str, str]:
    dotenv_values = load_dotenv_values(dotenv_path)
    api_key = os.getenv("TIINGO_API_KEY") or dotenv_values.get("TIINGO_API_KEY")

    return {
        "API_KEY": api_key or "",
    }
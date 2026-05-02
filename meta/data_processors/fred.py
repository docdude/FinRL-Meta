from __future__ import annotations

import os

import pandas as pd
import requests

from meta.data_processors._credentials import load_dotenv_values

FRED_SERIES_PREFIX = "fred:"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def is_fred_series_ticker(ticker: str | None) -> bool:
    return bool(ticker and ticker.startswith(FRED_SERIES_PREFIX))


def resolve_fred_api_key(api_key: str | None = None) -> str:
    dotenv_values = load_dotenv_values()
    return (
        api_key
        or os.getenv("FRED_API_KEY")
        or dotenv_values.get("FRED_API_KEY")
    )


def fetch_fred_series_df(
    series_ticker: str,
    market_dates: list[str],
    cache_dir: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch and align a FRED series to the supplied market dates."""
    if not market_dates:
        raise ValueError("Cannot align FRED series without market dates.")

    if not is_fred_series_ticker(series_ticker):
        raise ValueError(
            f"FRED ticker must start with '{FRED_SERIES_PREFIX}'."
        )

    series_id = series_ticker.split(":", 1)[1]
    start_date = min(market_dates)
    end_date = max(market_dates)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"fred_{series_id}_{start_date}_{end_date}.csv",
    )
    if os.path.exists(cache_path):
        rf_df = pd.read_csv(cache_path)
    else:
        response = requests.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "api_key": resolve_fred_api_key(api_key),
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        rf_df = pd.DataFrame(payload.get("observations", []))
        rf_df = rf_df[["date", "value"]].rename(columns={"value": "close"})
        rf_df.to_csv(cache_path, index=False)

    if "date" not in rf_df.columns or "close" not in rf_df.columns:
        raise ValueError(
            f"FRED series '{series_id}' returned an unexpected schema."
        )

    rf_df["date"] = pd.to_datetime(rf_df["date"], errors="coerce")
    rf_df["close"] = pd.to_numeric(rf_df["close"], errors="coerce")
    rf_df = rf_df.dropna(subset=["date"]).sort_values(by=["date"])

    market_index = pd.to_datetime(pd.Index(market_dates), errors="raise")
    rf_df = rf_df.set_index("date").reindex(market_index).ffill().bfill()
    if rf_df["close"].isna().any():
        raise ValueError(
            f"FRED series '{series_id}' could not be aligned to market dates."
        )

    rf_df = rf_df.reset_index().rename(columns={"index": "date"})
    rf_df["date"] = rf_df["date"].dt.strftime("%Y-%m-%d")
    rf_df["tic"] = series_ticker
    return rf_df[["date", "tic", "close"]]

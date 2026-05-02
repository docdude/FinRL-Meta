from __future__ import annotations

from pathlib import Path

from meta.data_processors._base import DataSource
from meta.data_processors._base import IndicatorLib
from meta.data_processors._credentials import get_alpaca_credentials
from meta.data_processors._credentials import get_tiingo_credentials
from meta.env_market_impact.envs.market_data import MarketDataPreparator


def _default_cache_dir() -> str:
    return str(Path(__file__).resolve().parents[3] / "data")


def get_vec_alpaca_data_source_kwargs(
    dotenv_path: str | None = None,
) -> dict[str, str | None]:
    credentials = get_alpaca_credentials(dotenv_path)
    if not credentials["API_KEY"] or not credentials["API_SECRET"]:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "in the environment or the repo-root .env file."
        )
    return {
        "API": None,
        **credentials,
    }


def get_vec_tiingo_data_source_kwargs(
    dotenv_path: str | None = None,
) -> dict[str, str | None]:
    credentials = get_tiingo_credentials(dotenv_path)
    if not credentials["API_KEY"]:
        raise ValueError(
            "Missing Tiingo credentials. Set TIINGO_API_KEY in the environment "
            "or the repo-root .env file."
        )
    return {
        "TIINGO_API_KEY": credentials["API_KEY"],
    }


def build_vec_market_data_preparator(
    tickers: list[str],
    start_date: str,
    end_date: str,
    tech_indicators: list[str],
    train_ratio: float,
    benchmark_ticker: str = "SPY",
    rf_ticker: str | None = "fred:DTB3",
    rf_constant: float = 0.0,
    indicator_lib: IndicatorLib = IndicatorLib.TALIB,
    cache_dir: str | None = None,
    time_interval: str = "1d",
    dotenv_path: str | None = None,
) -> MarketDataPreparator:
    return MarketDataPreparator(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=tech_indicators,
        train_ratio=train_ratio,
        benchmark_ticker=benchmark_ticker,
        rf_ticker=rf_ticker,
        rf_constant=rf_constant,
        data_source=DataSource.alpaca,
        indicator_lib=indicator_lib,
        cache_dir=cache_dir or _default_cache_dir(),
        time_interval=time_interval,
        data_source_kwargs=get_vec_alpaca_data_source_kwargs(dotenv_path),
    )


def build_vec_tiingo_market_data_preparator(
    tickers: list[str],
    start_date: str,
    end_date: str,
    tech_indicators: list[str],
    train_ratio: float,
    benchmark_ticker: str = "SPY",
    rf_ticker: str | None = "fred:DTB3",
    rf_constant: float = 0.0,
    indicator_lib: IndicatorLib = IndicatorLib.TALIB,
    cache_dir: str | None = None,
    time_interval: str = "1d",
    dotenv_path: str | None = None,
) -> MarketDataPreparator:
    return MarketDataPreparator(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=tech_indicators,
        train_ratio=train_ratio,
        benchmark_ticker=benchmark_ticker,
        rf_ticker=rf_ticker,
        rf_constant=rf_constant,
        data_source=DataSource.tiingo,
        indicator_lib=indicator_lib,
        cache_dir=cache_dir or _default_cache_dir(),
        time_interval=time_interval,
        data_source_kwargs=get_vec_tiingo_data_source_kwargs(dotenv_path),
    )
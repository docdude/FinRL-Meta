"""Market data downloading, caching, preprocessing, and env-config creation.

Consolidates all the data-preparation steps into a single
:class:`MarketDataPreparator` class.  Construction downloads (or loads
from cache), preprocesses, and splits the data -- the instance is
ready to use immediately.

Typical usage
-------------
>>> prep = MarketDataPreparator(
...     tickers=NAS_100_TICKER,
...     start_date="2020-01-01",
...     end_date="2025-07-01",
...     tech_indicators=INDICATORS,
...     benchmark_ticker="QQEW",
...     train_ratio=0.9,
... )
>>> train_config = prep.create_env_config(Split.TRAIN)
>>> trade_config = prep.create_env_config(Split.TRADE)
"""

import enum
import hashlib
import os

import numpy as np
import pandas as pd
import requests

from .utils import get_logger
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource
from meta.data_processors._base import IndicatorLib
from meta.data_processors.fred import fetch_fred_series_df
from meta.data_processors.fred import is_fred_series_ticker

log = get_logger()


class Split(enum.Enum):
    """Identifies a data split for :meth:`MarketDataPreparator.create_env_config`."""

    TRAIN = "train"
    TRADE = "trade"


class MarketDataPreparator:
    """Download, cache, preprocess, split, and reshape market data.

    All heavy work (download / cache lookup, indicator computation,
    train-trade split) happens during construction so the instance is
    immediately ready to use.

    Parameters
    ----------
    tickers : list[str]
        Tradeable ticker symbols.
    start_date, end_date : str
        Date range in ``"YYYY-MM-DD"`` format.
    tech_indicators : list[str]
        Technical-indicator column names to compute.
    train_ratio : float
        Fraction of unique dates assigned to the training set
        (e.g. ``0.9``).
    benchmark_ticker : str
        Ticker used as the portfolio benchmark (e.g. ``"SPY"``,
        ``"QQEW"``).
    rf_ticker : str | None
        Ticker for the risk-free rate proxy.  Defaults to ``"^IRX"``
        (13-week T-bill yield). When ``None``, a constant synthetic
        risk-free series is used instead. Prefix with ``"fred:"`` to
        fetch a FRED series independently of the market-data provider,
        e.g. ``"fred:DTB3"``.
    rf_constant : float
        Constant risk-free rate value used when ``rf_ticker`` is ``None``.
    data_source : DataSource
        Which market-data provider to use.  Defaults to
        ``DataSource.yahoofinance``.
    indicator_lib : IndicatorLib
        Which technical-indicator library to use.  Defaults to
        ``IndicatorLib.TALIB``.
    cache_dir : str
        Directory for pickle caches of the downloaded data.
    time_interval : str
        Bar interval forwarded to the underlying data processor.
    data_source_kwargs : dict | None
        Optional provider-specific keyword arguments forwarded to
        :class:`meta.data_processor.DataProcessor`.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        tech_indicators: list[str],
        train_ratio: float,
        benchmark_ticker: str = "SPY",
        rf_ticker: str | None = "^IRX",
        rf_constant: float = 0.0,
        data_source: DataSource = DataSource.yahoofinance,
        indicator_lib: IndicatorLib = IndicatorLib.TALIB,
        cache_dir: str = "data",
        time_interval: str = "1d",
        data_source_kwargs: dict | None = None,
        fred_api_key: str | None = None,
    ) -> None:
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.tech_indicators = list(tech_indicators)
        self.train_ratio = train_ratio
        self.benchmark_ticker = benchmark_ticker
        self.rf_ticker = rf_ticker
        self.rf_constant = float(rf_constant)
        self.data_source = data_source
        self.indicator_lib = indicator_lib
        self.cache_dir = cache_dir
        self.time_interval = time_interval
        self.data_source_kwargs = dict(data_source_kwargs or {})
        self.fred_api_key = fred_api_key

        # Fetch / load, then split
        self._prepare()
        self._split()

    @property
    def universe_size(self) -> int:
        """Return the number of tickers in the universe."""
        return len(self.tickers)

    def create_env_config(self, split: Split) -> dict:
        """Build the config dict consumed by :class:`MACEStockTradingEnv`.

        Parameters
        ----------
        split : Split
            Which data split to use (``Split.TRAIN`` or ``Split.TRADE``).

        Returns
        -------
        dict
            Keys: ``date_list``, ``price_array``, ``tech_array``,
            ``volatility_array``, ``volume_array``, ``adv20_array``,
            ``tbill_rates``, ``tic_list``.
        """
        df, rf_df = self._get_split(split)

        df = df.sort_values(["date", "tic"], ignore_index=True)
        date_list = df["date"].unique()
        tic_list = df["tic"].unique()

        price_array = []
        tech_array = []
        volatility_array = []
        volume_array = []
        adv20_array = []

        for date in date_list:
            date_df = df[df["date"] == date]
            price_array.append(date_df["close"].values)
            tech_array.append(date_df[self.tech_indicators].values.flatten())
            volatility_array.append(date_df["volatility"].values)
            volume_array.append(date_df["volume"].values)
            adv20_array.append(date_df["adv20"].values)

        rf_df = rf_df[rf_df["date"].isin(date_list)]
        tbill_rates = rf_df["close"].values
        if tbill_rates.shape[0] != len(date_list):
            raise ValueError(
                "Risk-free rate series length does not match market dates. "
                "Check rf_ticker/rf_constant configuration."
            )

        return {
            "date_list": date_list,
            "price_array": np.array(price_array),
            "tech_array": np.array(tech_array),
            "volatility_array": np.array(volatility_array),
            "volume_array": np.array(volume_array),
            "adv20_array": np.array(adv20_array),
            "tbill_rates": tbill_rates,
            "tic_list": tic_list,
        }

    def get_benchmark_df(self, split: Split) -> pd.DataFrame:
        """Return the benchmark DataFrame for the given split."""
        if split is Split.TRAIN:
            return self._train_benchmark_df
        return self._trade_benchmark_df

    def _get_split(self, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(market_df, rf_df)`` for the requested split."""
        if split is Split.TRAIN:
            return self._train_df, self._train_rf_df
        return self._trade_df, self._trade_rf_df

    def _prepare(self) -> None:
        """Download (or load from cache) and separate market / benchmark / rf."""
        cache_path = self._get_cache_path()
        legacy_cache_path = self._get_legacy_cache_path()

        if os.path.exists(cache_path):
            log.info(f"Loading cached data from {cache_path}")
            processed_df = pd.read_pickle(cache_path)
        elif os.path.exists(legacy_cache_path):
            log.info(f"Loading legacy cached data from {legacy_cache_path}")
            processed_df = pd.read_pickle(legacy_cache_path)
        else:
            log.info("No cache found. Fetching and processing data...")
            processed_df = self._fetch_data()
            log.info(f"Saving data to cache: {cache_path}")
            processed_df.to_pickle(cache_path)

        self._resolve_effective_universe(processed_df)
        processed_df = self._filter_complete_panel_dates(processed_df)

        self._market_df = (
            processed_df[processed_df["tic"].isin(self.tickers)]
            .sort_values(by=["date", "tic"])
            .reset_index(drop=True)
        )
        self._benchmark_df = (
            processed_df[processed_df["tic"] == self.benchmark_ticker]
            .sort_values(by=["date", "tic"])
            .reset_index(drop=True)
        )
        if self.rf_ticker is None:
            self._rf_df = self._build_constant_rf_df()
        elif is_fred_series_ticker(self.rf_ticker):
            try:
                self._rf_df = fetch_fred_series_df(
                    series_ticker=self.rf_ticker,
                    market_dates=self._market_df["date"]
                    .drop_duplicates()
                    .sort_values()
                    .tolist(),
                    cache_dir=self.cache_dir,
                    api_key=self.fred_api_key,
                )
            except (requests.RequestException, ValueError, KeyError) as error:
                log.warning(
                    "FRED risk-free fetch failed for %s; using constant %.4f instead: %s",
                    self.rf_ticker,
                    self.rf_constant,
                    error,
                )
                self._rf_df = self._build_constant_rf_df()
        else:
            self._rf_df = (
                processed_df[processed_df["tic"] == self.rf_ticker]
                .sort_values(by=["date", "tic"])
                .reset_index(drop=True)
            )

    def _build_constant_rf_df(self) -> pd.DataFrame:
        rf_dates = (
            self._market_df[["date"]]
            .drop_duplicates()
            .sort_values(by=["date"])
            .reset_index(drop=True)
        )
        rf_dates["tic"] = "RF_CONST"
        rf_dates["close"] = self.rf_constant
        return rf_dates

    def _resolve_effective_universe(self, df: pd.DataFrame) -> None:
        """Drop requested market tickers that have no provider data at all."""
        available_tickers = set(df["tic"].unique())

        if self.benchmark_ticker not in available_tickers:
            raise ValueError(
                f"Benchmark ticker '{self.benchmark_ticker}' is missing from the provider data."
            )

        if (
            self.rf_ticker is not None
            and not is_fred_series_ticker(self.rf_ticker)
            and self.rf_ticker not in available_tickers
        ):
            raise ValueError(
                f"Risk-free ticker '{self.rf_ticker}' is missing from the provider data."
            )

        missing_tickers = [tic for tic in self.tickers if tic not in available_tickers]
        if missing_tickers:
            preview = ", ".join(missing_tickers[:10])
            if len(missing_tickers) > 10:
                preview += ", ..."
            log.warning(
                "Dropping %s requested market ticker(s) with no provider data: %s",
                len(missing_tickers),
                preview,
            )
            self.tickers = [tic for tic in self.tickers if tic in available_tickers]

        if not self.tickers:
            raise ValueError(
                "None of the requested market tickers have provider data after preprocessing."
            )

    def _filter_complete_panel_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only dates with a rectangular market panel plus benchmark / rf rows."""
        if df.empty:
            return df

        valid_dates = set(df["date"].unique())

        market_counts = (
            df[df["tic"].isin(self.tickers)]
            .groupby("date")["tic"]
            .nunique()
        )
        complete_market_dates = set(
            market_counts[market_counts == len(self.tickers)].index
        )
        valid_dates &= complete_market_dates

        benchmark_dates = set(
            df[df["tic"] == self.benchmark_ticker]["date"].unique()
        )
        valid_dates &= benchmark_dates

        if self.rf_ticker is not None and not is_fred_series_ticker(
            self.rf_ticker
        ):
            rf_dates = set(df[df["tic"] == self.rf_ticker]["date"].unique())
            valid_dates &= rf_dates

        filtered = df[df["date"].isin(valid_dates)].copy()
        dropped_dates = sorted(set(df["date"].unique()) - valid_dates)
        if dropped_dates:
            log.info(
                "Dropping %s incomplete panel date(s); first=%s last=%s",
                len(dropped_dates),
                dropped_dates[0],
                dropped_dates[-1],
            )

        return filtered.sort_values(by=["date", "tic"]).reset_index(drop=True)

    def _split(self) -> None:
        """Split the prepared data into train / trade sets by date."""
        unique_dates = sorted(self._market_df["date"].unique())
        if not unique_dates:
            raise ValueError(
                "No dates remain after provider coverage filtering. "
                "Check for globally missing tickers or benchmark coverage gaps."
            )

        split_index = min(
            int(len(unique_dates) * self.train_ratio),
            len(unique_dates) - 1,
        )
        split_date = unique_dates[split_index]

        self._train_df = self._market_df[self._market_df.date < split_date]
        self._trade_df = self._market_df[self._market_df.date >= split_date]
        self._train_benchmark_df = self._benchmark_df[
            self._benchmark_df.date < split_date
        ]
        self._trade_benchmark_df = self._benchmark_df[
            self._benchmark_df.date >= split_date
        ].reset_index(drop=True)
        self._train_rf_df = self._rf_df[self._rf_df.date < split_date]
        self._trade_rf_df = self._rf_df[self._rf_df.date >= split_date].reset_index(
            drop=True
        )

        log.info(
            f"Training from {self._train_df.date.min()} to {self._train_df.date.max()}"
        )
        log.info(
            f"Trading from {self._trade_df.date.min()} to {self._trade_df.date.max()}"
        )

    def _get_cache_path(self) -> str:
        """Deterministic cache-file path based on request parameters."""
        os.makedirs(self.cache_dir, exist_ok=True)
        tickers_str = ",".join(sorted(self.tickers))
        indicators_str = ",".join(self.tech_indicators)
        rf_key = self._get_processed_rf_cache_key()
        cache_key = (
            f"{tickers_str}_{self.start_date}_{self.end_date}"
            f"_{self.benchmark_ticker}_{rf_key}"
            f"_{self.data_source.value}_{self.indicator_lib.name}_{self.time_interval}"
            f"_{indicators_str}"
        )
        filename = hashlib.md5(cache_key.encode()).hexdigest() + ".pickle"
        return os.path.join(self.cache_dir, filename)

    def _get_legacy_cache_path(self) -> str:
        tickers_str = ",".join(sorted(self.tickers))
        rf_key = self._get_processed_rf_cache_key()
        cache_key = (
            f"{tickers_str}_{self.start_date}_{self.end_date}"
            f"_{self.benchmark_ticker}_{rf_key}"
        )
        filename = hashlib.md5(cache_key.encode()).hexdigest() + ".pickle"
        return os.path.join(self.cache_dir, filename)

    def _get_processed_rf_cache_key(self) -> str:
        if self.rf_ticker is None or is_fred_series_ticker(self.rf_ticker):
            return f"const:{self.rf_constant}"
        return self.rf_ticker

    def _fetch_data(self) -> pd.DataFrame:
        """Download raw data, compute derived columns, and clean NaNs."""
        p = DataProcessor(
            data_source=self.data_source,
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.time_interval,
            **self.data_source_kwargs,
        )
        symbols = self.tickers + [self.benchmark_ticker]
        if self.rf_ticker is not None and not is_fred_series_ticker(
            self.rf_ticker
        ):
            symbols.append(self.rf_ticker)
        p.download_data(ticker_list=symbols)
        p.clean_data()
        p.add_technical_indicator(
            self.tech_indicators,
            select_stockstats_talib=self.indicator_lib,
            drop_na_timesteps=False,
        )

        df = p.dataframe
        df = df.sort_values(by=["time", "tic"]).reset_index(drop=True)
        df = df.rename(columns={"time": "date"})

        df["daily_return"] = df.groupby("tic")["close"].pct_change()
        df["volatility"] = df.groupby("tic")["daily_return"].transform(
            lambda x: x.shift(1).rolling(window=20, min_periods=2).std()
        )
        df["adv20"] = df.groupby("tic")["volume"].transform(
            lambda x: x.shift(1).rolling(window=20, min_periods=1).mean()
        )
        df = df.drop(["index"], axis=1, errors="ignore")

        # First date where all derived indicators are valid
        indicator_cols = self.tech_indicators + [
            "volatility",
            "adv20",
            "daily_return",
        ]
        df_valid = df.dropna(subset=indicator_cols)
        if df_valid.empty:
            raise ValueError(
                "No rows with complete indicator data found. "
                "Check data and indicator settings."
            )

        valid_start_date = df_valid["date"].min()
        log.info(
            f"Filtering data to start from {valid_start_date}, "
            "where all indicators are valid."
        )
        df = df[df["date"] >= valid_start_date].copy()

        # Forward-fill remaining NaNs within each stock, then zero-fill
        df = df.groupby("tic", group_keys=False).apply(lambda x: x.ffill())
        df = df.fillna(0).reset_index(drop=True)
        return df

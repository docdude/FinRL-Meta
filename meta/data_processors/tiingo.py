from __future__ import annotations

from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import pandas_market_calendars as tc
except:
    print(
        "Cannot import pandas_market_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for tiingo processor.")

from meta.config import TIME_ZONE_SELFDEFINED
from meta.config import USE_TIME_ZONE_SELFDEFINED
from meta.data_processors._base import _Base
from meta.data_processors._base import calc_time_zone
from meta.data_processors._credentials import get_tiingo_credentials


class Tiingo(_Base):
    BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        credentials = get_tiingo_credentials(kwargs.get("DOTENV_PATH"))
        self.api_key = (
            kwargs.get("TIINGO_API_KEY")
            or kwargs.get("api_key")
            or kwargs.get("token")
            or credentials["API_KEY"]
        )
        if not self.api_key:
            raise ValueError(
                "Missing Tiingo credentials. Set TIINGO_API_KEY in the environment or repo-root .env."
            )
        self.max_retry = int(kwargs.get("max_retry", 3))
        self.session = requests.Session()

    @staticmethod
    def _extract_error_detail(response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            text = getattr(response, "text", "").strip()
            return text or None

        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                value = payload.get(key)
                if value:
                    return str(value)

        if payload:
            return str(payload)
        return None

    def _build_http_error(
        self,
        ticker: str,
        response,
    ) -> requests.HTTPError:
        message_parts = [
            f"Tiingo request failed for {ticker} with status {response.status_code}."
        ]

        detail = self._extract_error_detail(response)
        if detail:
            message_parts.append(f"Detail: {detail}")

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            message_parts.append(f"Retry-After: {retry_after}")

        return requests.HTTPError(" ".join(message_parts), response=response)

    def _validate_time_interval(self, time_interval: Optional[str] = None) -> str:
        interval = (time_interval or self.time_interval).strip()
        supported = {"1d", "1D", "daily", "1day", "1Day"}
        if interval not in supported:
            raise ValueError(
                "Tiingo processor currently supports daily bars only. "
                f"Got '{interval}'."
            )
        return "1d"

    def _fetch_ticker(self, ticker: str) -> pd.DataFrame:
        params = {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "format": "json",
            "resampleFreq": "daily",
            "token": self.api_key,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retry + 1):
            try:
                response = self.session.get(
                    self.BASE_URL.format(ticker=ticker),
                    params=params,
                    timeout=60,
                )
                if response.status_code >= 400:
                    raise self._build_http_error(ticker, response)
                payload = response.json()
                if not payload:
                    return pd.DataFrame()
                return pd.DataFrame(payload)
            except Exception as error:
                last_error = error
                if attempt == self.max_retry:
                    raise
        if last_error is not None:
            raise last_error
        return pd.DataFrame()

    def download_data(
        self,
        ticker_list: List[str],
        save_path: str = "./data/dataset.csv",
    ):
        self._validate_time_interval()
        self.time_zone = calc_time_zone(
            ticker_list,
            TIME_ZONE_SELFDEFINED,
            USE_TIME_ZONE_SELFDEFINED,
        )

        frames = []
        failures = 0
        for ticker in ticker_list:
            temp_df = self._fetch_ticker(ticker)
            if temp_df.empty:
                failures += 1
                continue

            result = pd.DataFrame()
            result["date"] = pd.to_datetime(temp_df["date"])
            result["open"] = temp_df["adjOpen"]
            result["high"] = temp_df["adjHigh"]
            result["low"] = temp_df["adjLow"]
            result["close"] = temp_df["adjClose"]
            result["adjusted_close"] = temp_df["adjClose"]
            result["volume"] = temp_df["adjVolume"]
            result["tic"] = ticker
            result["day"] = result["date"].dt.dayofweek
            frames.append(result)

        if failures == len(ticker_list):
            raise ValueError("no data is fetched.")

        self.dataframe = pd.concat(frames, ignore_index=True)
        self.dataframe["date"] = self.dataframe["date"].dt.strftime("%Y-%m-%d")
        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def clean_data(self):
        df = self.dataframe.copy()
        df = df.rename(columns={"date": "time"})
        tic_list = np.unique(df.tic.values)
        trading_days = self.get_trading_days(start=self.start_date, end=self.end_date)
        time_list = trading_days

        new_df = pd.DataFrame()
        for tic in tic_list:
            print(("Clean data for ") + tic)
            tmp_df = pd.DataFrame(
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ],
                index=time_list,
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjusted_close",
                        "volume",
                    ]
                ]

            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_close = tmp_df.iloc[i]["close"]
                        first_valid_adjclose = tmp_df.iloc[i]["adjusted_close"]
                        tmp_df.iloc[0] = [
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            first_valid_adjclose,
                            0.0,
                        ]
                        break

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    previous_adjusted_close = tmp_df.iloc[i - 1]["adjusted_close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_adjusted_close,
                        0.0,
                    ]

            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])
            print(("Data clean for ") + tic + (" is finished."))

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})
        print("Data clean all finished!")
        self.dataframe = new_df

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        return [str(day)[:10] for day in df]
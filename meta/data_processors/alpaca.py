import re
from typing import List
from typing import Optional

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pytz
from alpaca_trade_api.rest import TimeFrame

try:
    import pandas_market_calendars as tc
except:
    print(
        "Cannot import pandas_market_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for alpaca processor.")
# from basic_processor import _Base
from meta.data_processors._base import _Base
from meta.data_processors._base import calc_time_zone
from meta.data_processors._credentials import get_alpaca_credentials

from meta.config import (
    ALPACA_API_BASE_URL,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
    TIME_ZONE_PARIS,
    TIME_ZONE_BERLIN,
    TIME_ZONE_JAKARTA,
    TIME_ZONE_SELFDEFINED,
    USE_TIME_ZONE_SELFDEFINED,
    BINANCE_BASE_URL,
)


class Alpaca(_Base):
    _INTERVAL_MAP = {
        "1D": TimeFrame.Day,
        "1d": TimeFrame.Day,
        "1Day": TimeFrame.Day,
        "day": TimeFrame.Day,
        "1H": TimeFrame.Hour,
        "1h": TimeFrame.Hour,
        "1Hour": TimeFrame.Hour,
        "1W": TimeFrame.Week,
        "1wk": TimeFrame.Week,
        "1Week": TimeFrame.Week,
        "1M": TimeFrame.Month,
        "1mo": TimeFrame.Month,
        "1Month": TimeFrame.Month,
        "1m": TimeFrame.Minute,
        "1Min": TimeFrame.Minute,
        "1Minute": TimeFrame.Minute,
    }

    # def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, api=None):
    #     if api is None:
    #         try:
    #             self.api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
    #         except BaseException:
    #             raise ValueError("Wrong Account Info!")
    #     else:
    #         self.api = api
    def __init__(
        self,
        data_source: str = "alpaca",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        time_interval: str = "1d",
        **kwargs,
    ):
        start_date = start_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        end_date = end_date or start_date
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        self.time_interval = str(self.convert_interval(time_interval))
        self.data_feed = kwargs.get("DATA_FEED") or kwargs.get("data_feed") or "sip"
        self.trim_to_common_panel = bool(
            kwargs.get(
                "TRIM_TO_COMMON_PANEL",
                kwargs.get("trim_to_common_panel", False),
            )
        )
        api = kwargs.get("API", kwargs.get("api"))
        credentials = get_alpaca_credentials(kwargs.get("DOTENV_PATH"))
        api_key = kwargs.get("API_KEY") or kwargs.get("api_key") or credentials["API_KEY"]
        api_secret = kwargs.get("API_SECRET") or kwargs.get("api_secret") or credentials["API_SECRET"]
        api_base_url = (
            kwargs.get("API_BASE_URL")
            or kwargs.get("api_base_url")
            or credentials["API_BASE_URL"]
            or ALPACA_API_BASE_URL
        )
        if not api_key and ALPACA_API_KEY != "xxx":
            api_key = ALPACA_API_KEY
        if not api_secret and ALPACA_API_SECRET != "xxx":
            api_secret = ALPACA_API_SECRET

        if api is None:
            if not api_key or not api_secret:
                raise ValueError(
                    "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                    "ALPACA_API_SECRET in the environment or repo-root .env."
                )
            try:
                self.api = tradeapi.REST(
                    api_key,
                    api_secret,
                    api_base_url,
                    "v2",
                )
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    @classmethod
    def convert_interval(cls, time_interval: str) -> TimeFrame:
        time_interval = time_interval.strip()
        if time_interval in cls._INTERVAL_MAP:
            return cls._INTERVAL_MAP[time_interval]

        minute_match = re.match(r"^(\d+)(m|Min|Minute)$", time_interval, re.IGNORECASE)
        if minute_match:
            amount = int(minute_match.group(1))
            if amount == 60:
                return TimeFrame.Hour
            if amount > 59:
                raise ValueError(
                    "alpaca_trade_api does not support minute multipliers above 59. "
                    f"Use an hourly interval instead of '{time_interval}'."
                )
            return TimeFrame(amount=amount, unit=TimeFrame.Minute.unit)

        hour_match = re.match(r"^(\d+)(h|H|Hour)$", time_interval, re.IGNORECASE)
        if hour_match:
            amount = int(hour_match.group(1))
            return TimeFrame(amount=amount, unit=TimeFrame.Hour.unit)

        day_match = re.match(r"^(\d+)(d|D|Day)$", time_interval, re.IGNORECASE)
        if day_match:
            amount = int(day_match.group(1))
            return TimeFrame(amount=amount, unit=TimeFrame.Day.unit)

        supported = ", ".join(sorted(cls._INTERVAL_MAP))
        raise ValueError(
            f"Unsupported Alpaca time interval '{time_interval}'. Supported: {supported}"
        )

    @staticmethod
    def _is_daily_interval(time_interval: str) -> bool:
        return time_interval.endswith("Day") or time_interval.endswith("Week") or time_interval.endswith("Month")

    @staticmethod
    def _interval_step_minutes(time_interval: str) -> int:
        minute_match = re.match(r"^(\d+)Min$", time_interval)
        if minute_match:
            return int(minute_match.group(1))
        hour_match = re.match(r"^(\d+)Hour$", time_interval)
        if hour_match:
            return int(hour_match.group(1)) * 60
        return 1

    def _build_expected_times(self, trading_days: list[str]) -> list[str]:
        if self._is_daily_interval(self.time_interval):
            return trading_days

        times: list[str] = []
        step_minutes = self._interval_step_minutes(self.time_interval)
        for day in trading_days:
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize(self.time_zone)
            close_time = pd.Timestamp(day + " 16:00:00").tz_localize(self.time_zone)
            while current_time < close_time:
                times.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
                current_time += pd.Timedelta(minutes=step_minutes)
        return times

    def download_data(
        self,
        ticker_list,
        start_date=None,
        end_date=None,
        time_interval=None,
        save_path: str = "./data/dataset.csv",
    ) -> pd.DataFrame:
        self.time_zone = calc_time_zone(
            ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        )
        start_date = pd.Timestamp(start_date or self.start_date, tz=self.time_zone)
        end_date = pd.Timestamp(end_date or self.end_date, tz=self.time_zone) + pd.Timedelta(days=1)
        timeframe = self.convert_interval(time_interval or self.time_interval)
        time_interval = str(timeframe)
        self.time_interval = time_interval

        frames = []
        if self._is_daily_interval(time_interval):
            for tic in ticker_list:
                barset = self.api.get_bars(
                    tic,
                    timeframe,
                    start=start_date.date().isoformat(),
                    end=end_date.date().isoformat(),
                    limit=10_000,
                    feed=self.data_feed,
                ).df
                if barset.empty:
                    continue
                barset["tic"] = tic
                frames.append(barset.reset_index())
            print(
                f"Daily data through {end_date.date().isoformat()} fetched successfully"
            )
        else:
            date = start_date
            while date != end_date:
                start_time = (date + pd.Timedelta("09:30:00")).isoformat()
                end_time = (date + pd.Timedelta("15:59:00")).isoformat()
                for tic in ticker_list:
                    barset = self.api.get_bars(
                        tic,
                        timeframe,
                        start=start_time,
                        end=end_time,
                        limit=500,
                        feed=self.data_feed,
                    ).df
                    if barset.empty:
                        continue
                    barset["tic"] = tic
                    frames.append(barset.reset_index())
                print(("Data before ") + end_time + " is successfully fetched")
                date = date + pd.Timedelta(days=1)
                if date.isoformat()[-14:-6] == "01:00:00":
                    date = date - pd.Timedelta("01:00:00")
                elif date.isoformat()[-14:-6] == "23:00:00":
                    date = date + pd.Timedelta("01:00:00")
                if date.isoformat()[-14:-6] != "00:00:00":
                    raise ValueError("Timezone Error")

        data_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data_df.empty:
            raise ValueError("No Alpaca bars returned for the requested range.")

        if self._is_daily_interval(time_interval):
            data_df["time"] = data_df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d"))
        else:
            data_df["time"] = data_df["timestamp"].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
            )
        self.dataframe = data_df

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def clean_data(self):
        df = self.dataframe.copy()
        tic_list = np.unique(df.tic.values)

        trading_days = self.get_trading_days(
            start=self.start_date,
            end=max(self.end_date, str(df["time"].max())[:10]),
        )
        times = self._build_expected_times(trading_days)
        if self.trim_to_common_panel:
            available_times = {
                tic: set(df.loc[df.tic == tic, "time"].astype(str).tolist())
                for tic in tic_list
            }
            common_time_set = None
            for time_set in available_times.values():
                if common_time_set is None:
                    common_time_set = set(time_set)
                else:
                    common_time_set &= time_set

            if not common_time_set:
                raise ValueError(
                    "No common Alpaca timestamps across the requested tickers. "
                    "Try a shorter date range or remove symbols with sparse history."
                )

            first_common_time = min(common_time_set)
            last_common_time = max(common_time_set)
            times = [
                time for time in times if first_common_time <= time <= last_common_time
            ]

            if not times:
                raise ValueError(
                    "Unable to build a common Alpaca panel from the requested date range."
                )

            print(
                "Trimmed Alpaca panel to common coverage window",
                f"{times[0]} -> {times[-1]}",
                "to avoid synthetic leading or trailing fills.",
            )

        frames = []
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "The price of the first row for ticker ",
                    tic,
                    " is NaN. ",
                    "It will filled with the first valid price.",
                )
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_price = tmp_df.iloc[i]["close"]
                        tmp_df.iloc[0] = [
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            0.0,
                        ]
                        break

            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "Missing data for ticker: ",
                    tic,
                    " . The prices are all NaN. Fill with 0.",
                )
                tmp_df.iloc[0] = [0.0, 0.0, 0.0, 0.0, 0.0]

            # forward filling row by row
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["adjusted_close"] = tmp_df["close"]
            tmp_df = tmp_df[
                ["open", "high", "low", "close", "adjusted_close", "volume"]
            ]
            tmp_df["tic"] = tic
            frames.append(tmp_df)

        new_df = pd.concat(frames, axis=0)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        print("Data clean finished!")

        self.dataframe = new_df

    # def add_technical_indicator(
    #     self,
    #     df,
    #     tech_indicator_list=[
    #         "macd",
    #         "boll_ub",
    #         "boll_lb",
    #         "rsi_30",
    #         "dx_30",
    #         "close_30_sma",
    #         "close_60_sma",
    #     ],
    # ):
    #     df = df.rename(columns={"time": "date"})
    #     df = df.copy()
    #     df = df.sort_values(by=["tic", "date"])
    #     stock = Sdf.retype(df.copy())
    #     unique_ticker = stock.tic.unique()
    #     tech_indicator_list = tech_indicator_list
    #
    #     for indicator in tech_indicator_list:
    #         indicator_df = pd.DataFrame()
    #         for i in range(len(unique_ticker)):
    #             # print(unique_ticker[i], i)
    #             temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
    #             temp_indicator = pd.DataFrame(temp_indicator)
    #             temp_indicator["tic"] = unique_ticker[i]
    #             # print(len(df[df.tic == unique_ticker[i]]['date'].to_list()))
    #             temp_indicator["date"] = df[df.tic == unique_ticker[i]][
    #                 "date"
    #             ].to_list()
    #             indicator_df = indicator_df.append(temp_indicator, ignore_index=True)
    #         df = df.merge(
    #             indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
    #         )
    #     df = df.sort_values(by=["date", "tic"])
    #     df = df.rename(columns={"date": "time"})
    #     print("Succesfully add technical indicators")
    #     return df

    # def add_vix(self, data):
    #     vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
    #     cleaned_vix = self.clean_data(vix_df)
    #     vix = cleaned_vix[["time", "close"]]
    #     vix = vix.rename(columns={"close": "VIXY"})
    #
    #     df = data.copy()
    #     df = df.merge(vix, on="time")
    #     df = df.sort_values(["time", "tic"]).reset_index(drop=True)
    #     return df

    # def calculate_turbulence(self, data, time_period=252):
    #     # can add other market assets
    #     df = data.copy()
    #     df_price_pivot = df.pivot(index="date", columns="tic", values="close")
    #     # use returns to calculate turbulence
    #     df_price_pivot = df_price_pivot.pct_change()
    #
    #     unique_date = df.date.unique()
    #     # start after a fixed time period
    #     start = time_period
    #     turbulence_index = [0] * start
    #     # turbulence_index = [0]
    #     count = 0
    #     for i in range(start, len(unique_date)):
    #         current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
    #         # use one year rolling window to calcualte covariance
    #         hist_price = df_price_pivot[
    #             (df_price_pivot.index < unique_date[i])
    #             & (df_price_pivot.index >= unique_date[i - time_period])
    #         ]
    #         # Drop tickers which has number missing values more than the "oldest" ticker
    #         filtered_hist_price = hist_price.iloc[
    #             hist_price.isna().sum().min() :
    #         ].dropna(axis=1)
    #
    #         cov_temp = filtered_hist_price.cov()
    #         current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
    #             filtered_hist_price, axis=0
    #         )
    #         temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
    #             current_temp.values.T
    #         )
    #         if temp > 0:
    #             count += 1
    #             if count > 2:
    #                 turbulence_temp = temp[0][0]
    #             else:
    #                 # avoid large outlier because of the calculation just begins
    #                 turbulence_temp = 0
    #         else:
    #             turbulence_temp = 0
    #         turbulence_index.append(turbulence_temp)
    #
    #     turbulence_index = pd.DataFrame(
    #         {"date": df_price_pivot.index, "turbulence": turbulence_index}
    #     )
    #     return turbulence_index
    #
    # def add_turbulence(self, data, time_period=252):
    #     """
    #     add turbulence index from a precalcualted dataframe
    #     :param data: (df) pandas dataframe
    #     :return: (df) pandas dataframe
    #     """
    #     df = data.copy()
    #     turbulence_index = self.calculate_turbulence(df, time_period=time_period)
    #     df = df.merge(turbulence_index, on="date")
    #     df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    #     return df

    # def df_to_array(self, df, tech_indicator_list, if_vix):
    #     df = df.copy()
    #     unique_ticker = df.tic.unique()
    #     if_first_time = True
    #     for tic in unique_ticker:
    #         if if_first_time:
    #             price_array = df[df.tic == tic][["close"]].values
    #             tech_array = df[df.tic == tic][tech_indicator_list].values
    #             if if_vix:
    #                 turbulence_array = df[df.tic == tic]["VIXY"].values
    #             else:
    #                 turbulence_array = df[df.tic == tic]["turbulence"].values
    #             if_first_time = False
    #         else:
    #             price_array = np.hstack(
    #                 [price_array, df[df.tic == tic][["close"]].values]
    #             )
    #             tech_array = np.hstack(
    #                 [tech_array, df[df.tic == tic][tech_indicator_list].values]
    #             )
    #     print("Successfully transformed into array")
    #     return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        # df = nyse.sessions_in_range(
        #     pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        # )
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))

        return [str(day)[:10] for day in df]

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        timeframe = self.convert_interval(time_interval)
        normalized_interval = str(timeframe)
        frames = []
        for tic in ticker_list:
            barset = self.api.get_bars(
                tic,
                timeframe,
                limit=limit,
                feed=self.data_feed,
            ).df
            if barset.empty:
                continue
            barset["tic"] = tic
            frames.append(barset.reset_index())

        if not frames:
            raise ValueError("No Alpaca bars returned for latest-data request.")

        data_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
        timestamp_col = "timestamp" if "timestamp" in data_df.columns else "time"
        start_time = data_df[timestamp_col].min()
        end_time = data_df[timestamp_col].max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        normalized_frames = []
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i][timestamp_col]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            normalized_frames.append(tmp_df)

        new_df = pd.concat(normalized_frames, axis=0)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_bars(
            "VIXY",
            timeframe,
            limit=1,
            feed=self.data_feed,
        ).df
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb

    def get_portfolio_history(self, start, end):
        trading_days = self.get_trading_days(start, end)
        frames = []
        for day in trading_days:
            frames.append(
                self.api.get_portfolio_history(
                    date_start=day, timeframe="5Min"
                ).df.iloc[:79]
            )
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        equities = df.equity.values
        cumu_returns = equities / equities[0]
        cumu_returns = cumu_returns[~np.isnan(cumu_returns)]
        return cumu_returns

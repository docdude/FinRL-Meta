from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from stockstats import StockDataFrame as Sdf

# import alpaca_trade_api as tradeapi


class AlpacaProcessor:
    def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, client=None):
        if client is None:
            try:
                self.client = StockHistoricalDataClient(API_KEY, API_SECRET)
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.client = client

    def convert_interval(self, time_interval: str) -> TimeFrame:
        """
        Convert FinRL/common time interval strings to Alpaca TimeFrame.
        
        Alpaca TimeFrame supports: Minute, Hour, Day, Week, Month with multipliers.
        Examples: TimeFrame.Minute, TimeFrame(amount=5, unit=TimeFrame.Minute.unit)
        
        Supported inputs (matching Yahoo style + Alpaca native):
            Minutes: 1m, 1Min, 2m, 5m, 5Min, 15m, 15Min, 30m, 30Min, 60m, 90m
            Hours: 1h, 1H, 1Hour
            Days: 1d, 1D, 1Day, day
            Weeks: 1wk, 1W, 1Week
            Months: 1mo, 1M, 1Month
        """
        time_interval = time_interval.strip()
        
        # Direct Alpaca TimeFrame mappings
        direct_map = {
            # Days
            "1D": TimeFrame.Day,
            "1d": TimeFrame.Day,
            "1Day": TimeFrame.Day,
            "day": TimeFrame.Day,
            # Hours
            "1H": TimeFrame.Hour,
            "1h": TimeFrame.Hour,
            "1Hour": TimeFrame.Hour,
            # Weeks
            "1W": TimeFrame.Week,
            "1wk": TimeFrame.Week,
            "1Week": TimeFrame.Week,
            # Months
            "1M": TimeFrame.Month,
            "1mo": TimeFrame.Month,
            "1Month": TimeFrame.Month,
            # Minutes (1 min)
            "1m": TimeFrame.Minute,
            "1Min": TimeFrame.Minute,
            "1Minute": TimeFrame.Minute,
        }
        
        if time_interval in direct_map:
            return direct_map[time_interval]
        
        # Handle minute multipliers: 2m, 5m, 15m, 30m, 60m, 90m, etc.
        # Match patterns like "5m", "15Min", "30Minute"
        minute_match = re.match(r'^(\d+)(m|Min|Minute)$', time_interval, re.IGNORECASE)
        if minute_match:
            amount = int(minute_match.group(1))
            return TimeFrame(amount=amount, unit=TimeFrame.Minute.unit)
        
        # Match patterns like "2h", "4H", "4Hour"
        hour_match = re.match(r'^(\d+)(h|H|Hour)$', time_interval, re.IGNORECASE)
        if hour_match:
            amount = int(hour_match.group(1))
            return TimeFrame(amount=amount, unit=TimeFrame.Hour.unit)
        
        # Match patterns like "5d", "5D", "5Day"
        day_match = re.match(r'^(\d+)(d|D|Day)$', time_interval, re.IGNORECASE)
        if day_match:
            amount = int(day_match.group(1))
            return TimeFrame(amount=amount, unit=TimeFrame.Day.unit)
        
        # Default to Day with warning
        print(f"Warning: Unrecognized time_interval '{time_interval}', defaulting to 1Day")
        return TimeFrame.Day

    def _fetch_data_for_ticker(self, ticker, start_date, end_date, timeframe):
        """Fetch data for a single ticker using pre-converted TimeFrame."""
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=DataFeed.SIP,  # Use SIP consolidated feed (requires paid account)
        )
        bars = self.client.get_stock_bars(request_params).df

        return bars

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        """
        Downloads data using Alpaca's tradeapi.REST method.

        Parameters:
        - ticker_list : list of strings, each string is a ticker
        - start_date : string in the format 'YYYY-MM-DD'
        - end_date : string in the format 'YYYY-MM-DD'
        - time_interval: string representing the interval ('1D', '1Min', etc.)

        Returns:
        - pd.DataFrame with the requested data
        """
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        NY = "America/New_York"
        start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
        end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)
        
        # Convert interval once for all tickers (optimization)
        timeframe = self.convert_interval(time_interval) if isinstance(time_interval, str) else time_interval
        
        data_list = []
        # Use ThreadPoolExecutor to fetch data for multiple tickers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    timeframe,  # Pass pre-converted TimeFrame
                )
                for ticker in ticker_list
            ]
        for future in futures:

            bars = future.result()
            # fix start
            # Reorganize the dataframes to be in original alpaca_trade_api structure
            # Rename the existing 'symbol' column if it exists
            if not bars.empty:

                # Now reset the index
                bars.reset_index(inplace=True)

                # Set 'timestamp' as the new index
                if "level_1" in bars.columns:
                    bars.rename(columns={"level_1": "timestamp"}, inplace=True)
                if "level_0" in bars.columns:
                    bars.rename(columns={"level_0": "symbol"}, inplace=True)

                bars.set_index("timestamp", inplace=True)

                # Reorder and rename columns as needed
                bars = bars[
                    [
                        "close",
                        "high",
                        "low",
                        "trade_count",
                        "open",
                        "volume",
                        "vwap",
                        "symbol",
                    ]
                ]

                data_list.append(bars)
            else:
                print("empty")

        # Combine the data
        data_df = pd.concat(data_list, axis=0)

        # Convert the timezone
        data_df = data_df.tz_convert(NY)

        # If time_interval is less than a day, filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")

        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={"index": "timestamp", "symbol": "tic"}
        )

        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])

        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        return data_df

    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                print(
                    f"The price of the first row for ticker {tic} is NaN. It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                print(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic

        return tmp_df

    def clean_data(self, df):
        print("Data cleaning started")
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        print("align start and end dates")
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        # ... (generating 'times' series, same as in your existing code)

        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index - handle both daily and minute intervals
        print("produce full timestamp index")
        NY = "America/New_York"
        
        # Check if using daily or intraday data based on time_interval
        daily_patterns = ["1D", "1d", "1Day", "day", "1W", "1wk", "1Week", "1M", "1mo", "1Month"]
        is_daily = self.time_interval in daily_patterns
        
        if is_daily:
            # For daily/weekly/monthly data, just use trading days
            times = [pd.Timestamp(day).tz_localize(NY) for day in trading_days]
        else:
            # For intraday data, generate timestamps per trading day
            # Determine interval in minutes
            interval_minutes = 1  # default
            match = re.match(r'^(\d+)(m|Min|Minute)$', self.time_interval, re.IGNORECASE)
            if match:
                interval_minutes = int(match.group(1))
            
            bars_per_day = 390 // interval_minutes  # 390 minutes in trading day
            
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(bars_per_day):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=interval_minutes)

        print("Start processing tickers")

        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        print("ticker list complete")

        print("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})
        
        # Add 'date' column for compatibility with data_split() and other FinRL functions
        # Keep 'timestamp' for internal processor methods, add 'date' as string YYYY-MM-DD
        if 'date' not in new_df.columns:
            new_df['date'] = new_df['timestamp'].dt.strftime('%Y-%m-%d')

        print("Data clean finished!")

        return new_df

    def add_technical_indicator(
        self,
        df,
        tech_indicator_list=[
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    ):
        print("Started adding Indicators")

        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype
        
        # Preserve 'date' column if it exists (may be dropped by Sdf.retype or merge operations)
        has_date_col = 'date' in df.columns
        if has_date_col:
            date_backup = df[['timestamp', 'tic', 'date']].copy()

        # Convert df to stock data format just once
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()

        # Convert timestamp to a consistent datatype (timezone-naive) before entering the loop
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

        print("Running Loop")
        for indicator in tech_indicator_list:
            indicator_dfs = []
            for tic in unique_ticker:
                tic_data = stock[stock.tic == tic]
                indicator_series = tic_data[indicator]

                tic_timestamps = df.loc[df.tic == tic, "timestamp"]

                indicator_df = pd.DataFrame(
                    {
                        "tic": tic,
                        "_merge_ts": tic_timestamps.values,
                        indicator: indicator_series.values,
                    }
                )
                indicator_dfs.append(indicator_df)

            # Concatenate all intermediate dataframes at once
            indicator_df = pd.concat(indicator_dfs, ignore_index=True)

            # Merge the indicator data frame
            df = df.merge(
                indicator_df[["tic", "_merge_ts", indicator]],
                left_on=["tic", "timestamp"],
                right_on=["tic", "_merge_ts"],
                how="left",
            ).drop(columns="_merge_ts")

        print("Restore Timestamps")
        # Restore the original data type of the 'timestamp' column
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["timestamp"] = df["timestamp"].dt.tz_convert(original_timestamp_dtype.tz)
        else:
            df["timestamp"] = df["timestamp"].astype(original_timestamp_dtype)
        
        # Restore 'date' column if it was originally present
        if has_date_col and 'date' not in df.columns:
            # Convert timestamp for merge (handle tz-naive)
            date_backup['_ts_naive'] = date_backup['timestamp'].dt.tz_convert(None) if date_backup['timestamp'].dt.tz else date_backup['timestamp']
            df['_ts_naive'] = df['timestamp'].dt.tz_convert(None) if df['timestamp'].dt.tz else df['timestamp']
            df = df.merge(date_backup[['_ts_naive', 'tic', 'date']], on=['_ts_naive', 'tic'], how='left')
            df = df.drop(columns=['_ts_naive'])

        print("Finished adding Indicators")
        return df

    # Allows to multithread the add_vix function for quicker execution
    def download_and_clean_data(self):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)

    def add_vix(self, data):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]
        
        # Always merge on timestamp (datetime), not date (string) to avoid type mismatch
        vix = vix.rename(columns={"close": "VIXY"})

        data = data.copy()
        data = data.merge(vix, on="timestamp")
        
        # Sort by appropriate column
        sort_column = "date" if "date" in data.columns else "timestamp"
        data = data.sort_values([sort_column, "tic"]).reset_index(drop=True)

        return data

    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )

        # print("turbulence_index\n", turbulence_index)

        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        # df = nyse.sessions_in_range(
        #     pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        # )
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            request_params = StockBarsRequest(
                symbol_or_symbols=[tic], timeframe=TimeFrame.Minute, limit=limit
            )

            barset = self.client.get_stock_bars(request_params).df
            # Reorganize the dataframes to be in original alpaca_trade_api structure
            # Rename the existing 'symbol' column if it exists
            if "symbol" in barset.columns:
                barset.rename(columns={"symbol": "symbol_old"}, inplace=True)

            # Now reset the index
            barset.reset_index(inplace=True)

            # Set 'timestamp' as the new index
            if "level_0" in barset.columns:
                barset.rename(columns={"level_0": "symbol"}, inplace=True)
            if "level_1" in barset.columns:
                barset.rename(columns={"level_1": "timestamp"}, inplace=True)
            barset.set_index("timestamp", inplace=True)

            # Reorder and rename columns as needed
            barset = barset[
                [
                    "close",
                    "high",
                    "low",
                    "trade_count",
                    "open",
                    "volume",
                    "vwap",
                    "symbol",
                ]
            ]

            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
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
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        request_params = StockBarsRequest(
            symbol_or_symbols="VIXY", timeframe=TimeFrame.Minute, limit=1
        )
        turb_df = self.client.get_stock_bars(request_params).df
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb

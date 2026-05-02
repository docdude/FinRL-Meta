import pandas as pd

from meta.data_processors.alpaca import Alpaca


def test_alpaca_clean_data_keeps_adjusted_close_and_rectangular_panel() -> None:
    processor = Alpaca(
        data_source="alpaca",
        start_date="2025-12-31",
        end_date="2026-01-01",
        time_interval="1d",
        api=object(),
    )
    processor.dataframe = pd.DataFrame(
        {
            "time": ["2025-12-31", "2026-01-02", "2025-12-31"],
            "open": [100.0, 101.0, 200.0],
            "high": [101.0, 102.0, 201.0],
            "low": [99.0, 100.0, 199.0],
            "close": [100.5, 101.5, 200.5],
            "volume": [1000.0, 1100.0, 2000.0],
            "tic": ["AAPL", "AAPL", "MSFT"],
        }
    )

    processor.clean_data()

    cleaned = processor.dataframe.sort_values(["time", "tic"]).reset_index(drop=True)

    assert "adjusted_close" in cleaned.columns

    counts = cleaned.groupby("time")["tic"].nunique().to_dict()
    assert counts == {"2025-12-31": 2, "2026-01-02": 2}

    msft_last_row = cleaned[
        (cleaned["time"] == "2026-01-02") & (cleaned["tic"] == "MSFT")
    ].iloc[0]
    assert msft_last_row["close"] == 200.5
    assert msft_last_row["adjusted_close"] == 200.5
    assert msft_last_row["volume"] == 0.0
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta.data_processors._base import DataSource
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ad hoc trade diagnostics for a scalar MACE backtest run. "
            "Outputs are written next to the selected summary file."
        )
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to a backtest_summary.json file.",
    )
    parser.add_argument(
        "--agent",
        help="Optional DRL agent filter, for example ddpg or ppo.",
    )
    parser.add_argument(
        "--impact-model",
        help="Optional impact-model filter, for example 'Baseline Impact Model'.",
    )
    parser.add_argument(
        "--match-token",
        help=(
            "Optional substring used to disambiguate a single run, for example "
            "the short uid from the result filename such as b84cf636."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top traded tickers to include in price-marker charts.",
    )
    parser.add_argument(
        "--horizons",
        default="5,20,63",
        help="Comma-separated forward-return horizons in trading days.",
    )
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        help="MarketDataPreparator start date used to rebuild the cached panel.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-01-01",
        help="MarketDataPreparator end date used to rebuild the cached panel.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="MarketDataPreparator train ratio used for the source run.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        help="Override the benchmark ticker if it differs from the summary payload.",
    )
    return parser.parse_args()


def _load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_backtest(
    summary_payload: dict,
    agent: str | None,
    impact_model: str | None,
    match_token: str | None,
) -> dict:
    candidates = list(summary_payload.get("backtests", []))
    if agent:
        candidates = [
            backtest
            for backtest in candidates
            if str(backtest.get("drl_agent", "")).lower() == agent.lower()
        ]
    if impact_model:
        candidates = [
            backtest
            for backtest in candidates
            if str(backtest.get("impact_model", "")).lower() == impact_model.lower()
        ]
    if match_token:
        candidates = [
            backtest
            for backtest in candidates
            if match_token in json.dumps(backtest, sort_keys=True)
        ]

    if not candidates:
        raise ValueError("No backtest matched the requested filters.")
    if len(candidates) > 1:
        sample_runs = [
            Path(backtest["results_csv_test"]).stem.replace("_test", "")
            for backtest in candidates[:6]
        ]
        raise ValueError(
            "Filters matched multiple backtests. Add --match-token to pick one. "
            f"Examples: {sample_runs}"
        )
    return candidates[0]


def _resolve_artifact_path(summary_path: Path, artifact_path: str) -> Path:
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate

    env_market_impact_dir = summary_path.resolve().parents[2]
    repo_root = summary_path.resolve().parents[4]

    for base_dir in (summary_path.parent, env_market_impact_dir, repo_root):
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return (env_market_impact_dir / candidate).resolve()


def _build_market_data(
    *,
    start_date: str,
    end_date: str,
    train_ratio: float,
    benchmark_ticker: str,
) -> MarketDataPreparator:
    prep = MarketDataPreparator(
        tickers=NAS_100_TICKER,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=INDICATORS,
        train_ratio=train_ratio,
        benchmark_ticker=benchmark_ticker,
        data_source=DataSource.yahoofinance,
    )
    return prep


def _build_lookup_frames(prep: MarketDataPreparator) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    market_df = prep._market_df.sort_values(["date", "tic"], ignore_index=True)
    ticker_list = market_df["tic"].drop_duplicates().tolist()
    price_df = (
        market_df.pivot(index="date", columns="tic", values="close")
        .reindex(columns=ticker_list)
        .sort_index()
    )
    price_df.index = pd.to_datetime(price_df.index)
    price_df.index.name = "date"

    rf_df = prep._rf_df.drop_duplicates(subset=["date"]).sort_values("date")
    tbill_df = rf_df.set_index("date")["close"].reindex(price_df.index)
    tbill_df.index = pd.to_datetime(tbill_df.index)
    tbill_df.index.name = "date"
    tbill_df.name = "tbill_rate"
    return price_df, tbill_df, ticker_list


def _price_frame(config: dict) -> pd.DataFrame:
    return pd.DataFrame(
        config["price_array"],
        index=pd.Index(pd.to_datetime(config["date_list"]), name="date"),
        columns=list(config["tic_list"]),
    )


def _tbill_series(config: dict) -> pd.Series:
    return pd.Series(
        config["tbill_rates"],
        index=pd.Index(pd.to_datetime(config["date_list"]), name="date"),
        name="tbill_rate",
    )


def _daily_rf_rate(percent_rate: float) -> float:
    return float((1.0 + percent_rate / 100.0) ** (1.0 / 252.0) - 1.0)


def _signed_trade_shares(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype=float)

    signed = trades_df["shares"].where(trades_df["side"].eq("buy"), -trades_df["shares"])
    return signed.groupby(trades_df["ticker"]).sum().sort_index()


def _attach_tickers(trades_df: pd.DataFrame, ticker_list: list[str]) -> pd.DataFrame:
    if trades_df.empty:
        attached = trades_df.copy()
        attached["ticker"] = pd.Series(dtype=object)
        attached["date"] = pd.to_datetime(attached["date"])
        return attached

    attached = trades_df.copy()
    attached["ticker"] = attached["stock_idx"].astype(int).map(dict(enumerate(ticker_list)))
    attached["date"] = pd.to_datetime(attached["date"])
    attached["signed_shares"] = attached["shares"].where(
        attached["side"].eq("buy"), -attached["shares"]
    )
    return attached


def _approx_hold_counterfactual(
    *,
    results_df: pd.DataFrame,
    price_df: pd.DataFrame,
    tbill_df: pd.Series,
    holdings: pd.Series,
    initial_cash: float,
    anchor_date: pd.Timestamp,
    anchor_value: float,
) -> pd.Series:
    positions = holdings.reindex(price_df.columns, fill_value=0.0).astype(float)
    results_dates = pd.to_datetime(results_df["date"])
    start_loc = price_df.index.get_loc(anchor_date)
    relevant_dates = price_df.index[start_loc : start_loc + len(results_dates)]
    if len(relevant_dates) != len(results_dates) or not relevant_dates.equals(
        pd.Index(results_dates)
    ):
        raise ValueError("Result dates do not align with the reconstructed trade dates.")

    cash_path = []
    running_cash = float(initial_cash)
    for offset, date in enumerate(relevant_dates):
        if offset > 0:
            running_cash *= 1.0 + _daily_rf_rate(float(tbill_df.loc[date]))
        cash_path.append(running_cash)

    holdings_value = price_df.loc[relevant_dates].mul(positions, axis=1).sum(axis=1)
    raw_portfolio = holdings_value + pd.Series(cash_path, index=relevant_dates)
    residual = float(anchor_value) - float(raw_portfolio.iloc[0])
    return raw_portfolio + residual


def _event_study(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "side",
                "horizon",
                "events",
                "avg_raw_return",
                "avg_relative_return",
                "avg_directional_edge",
                "weighted_directional_edge",
                "directional_hit_rate",
            ]
        )

    future_returns = {
        horizon: price_df.shift(-horizon).div(price_df).sub(1.0)
        for horizon in horizons
    }

    event_rows = []
    for trade in trades_df.itertuples(index=False):
        if pd.isna(trade.ticker) or trade.ticker not in price_df.columns:
            continue
        trade_date = pd.Timestamp(trade.date)
        if trade_date not in price_df.index:
            continue
        for horizon, future_df in future_returns.items():
            raw_return = future_df.at[trade_date, trade.ticker]
            if pd.isna(raw_return):
                continue
            date_universe_mean = future_df.loc[trade_date].mean(skipna=True)
            relative_return = float(raw_return) - float(date_universe_mean)
            directional_edge = relative_return if trade.side == "buy" else -relative_return
            event_rows.append(
                {
                    "side": trade.side,
                    "horizon": horizon,
                    "raw_return": float(raw_return),
                    "relative_return": relative_return,
                    "directional_edge": directional_edge,
                    "notional": float(trade.notional),
                }
            )

    events_df = pd.DataFrame(event_rows)
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "side",
                "horizon",
                "events",
                "avg_raw_return",
                "avg_relative_return",
                "avg_directional_edge",
                "weighted_directional_edge",
                "directional_hit_rate",
            ]
        )

    summaries = []
    for (side, horizon), group in events_df.groupby(["side", "horizon"], sort=True):
        weights = group["notional"].to_numpy(dtype=float)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones_like(weights)
        summaries.append(
            {
                "side": side,
                "horizon": int(horizon),
                "events": int(len(group)),
                "avg_raw_return": float(group["raw_return"].mean()),
                "avg_relative_return": float(group["relative_return"].mean()),
                "avg_directional_edge": float(group["directional_edge"].mean()),
                "weighted_directional_edge": float(
                    np.average(group["directional_edge"], weights=weights)
                ),
                "directional_hit_rate": float((group["directional_edge"] > 0).mean()),
            }
        )
    return pd.DataFrame(summaries).sort_values(["side", "horizon"]).reset_index(drop=True)


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _counterfactual_metrics(results_df: pd.DataFrame, hold_series: pd.Series) -> dict:
    actual = results_df.copy()
    actual["date"] = pd.to_datetime(actual["date"])
    aligned_hold = hold_series.reindex(actual["date"])
    return {
        "actual_total_return": float(
            actual["portfolio_value"].iloc[-1] / actual["portfolio_value"].iloc[0] - 1.0
        ),
        "benchmark_total_return": float(
            actual["benchmark_value"].iloc[-1] / actual["benchmark_value"].iloc[0] - 1.0
        ),
        "approx_hold_total_return": float(
            aligned_hold.iloc[-1] / aligned_hold.iloc[0] - 1.0
        ),
        "actual_minus_hold": float(
            actual["portfolio_value"].iloc[-1] / actual["portfolio_value"].iloc[0]
            - aligned_hold.iloc[-1] / aligned_hold.iloc[0]
        ),
    }


def _plot_counterfactual(
    results_df: pd.DataFrame,
    hold_series: pd.Series,
    output_path: Path,
    *,
    title: str,
    hold_label: str,
) -> None:
    plotted = results_df.copy()
    plotted["date"] = pd.to_datetime(plotted["date"])
    aligned_hold = hold_series.reindex(plotted["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plotted["date"],
        plotted["portfolio_value"],
        label="Actual portfolio",
        color="#0a6c74",
        linewidth=2.2,
    )
    ax.plot(
        plotted["date"],
        plotted["benchmark_value"],
        label="Benchmark",
        color="#8a5a44",
        linewidth=1.8,
        linestyle="--",
    )
    ax.plot(
        plotted["date"],
        aligned_hold,
        label=hold_label,
        color="#d97706",
        linewidth=2.0,
    )
    ax.set_title(title)
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_event_study(
    event_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if event_df.empty:
        ax.text(0.5, 0.5, "No trades available", ha="center", va="center")
        ax.axis("off")
    else:
        horizons = sorted(event_df["horizon"].unique())
        x = np.arange(len(horizons))
        width = 0.36
        for offset, side, color in [(-width / 2, "buy", "#15803d"), (width / 2, "sell", "#b91c1c")]:
            side_df = (
                event_df[event_df["side"] == side]
                .set_index("horizon")
                .reindex(horizons)
            )
            values = side_df["weighted_directional_edge"].fillna(0.0).to_numpy()
            ax.bar(x + offset, values, width=width, label=side.title(), color=color, alpha=0.85)
            for idx, value in enumerate(values):
                ax.text(
                    x[idx] + offset,
                    value + (0.0008 if value >= 0 else -0.0012),
                    _format_pct(value),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
        ax.axhline(0.0, color="#111827", linewidth=1.0)
        ax.set_xticks(x, [str(horizon) for horizon in horizons])
        ax.set_xlabel("Forward horizon (trading days)")
        ax.set_ylabel("Notional-weighted directional edge")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_top_tickers(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    top_n: int,
) -> list[str]:
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No trades available", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return []

    grouped_notional = (
        trades_df.groupby("ticker")["notional"].sum().sort_values(ascending=False)
    )
    top_tickers = list(grouped_notional.head(top_n).index)
    cols = 2
    rows = math.ceil(len(top_tickers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(4.5 * rows, 5.5)), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, ticker in zip(axes, top_tickers):
        series = price_df[ticker]
        ax.plot(series.index, series.values, color="#334155", linewidth=1.5)
        ticker_trades = trades_df[trades_df["ticker"] == ticker].copy()
        ticker_trades = (
            ticker_trades.groupby(["date", "side"], as_index=False)["notional"].sum()
        )
        buy_rows = ticker_trades[ticker_trades["side"] == "buy"]
        sell_rows = ticker_trades[ticker_trades["side"] == "sell"]
        for rows_df, color, marker in [
            (buy_rows, "#16a34a", "^"),
            (sell_rows, "#dc2626", "v"),
        ]:
            if rows_df.empty:
                continue
            marker_prices = series.reindex(rows_df["date"])
            sizes = 30.0 + 170.0 * (rows_df["notional"] / rows_df["notional"].max())
            ax.scatter(
                rows_df["date"],
                marker_prices,
                s=sizes,
                color=color,
                marker=marker,
                alpha=0.8,
            )
        ax.set_title(f"{ticker} | traded ${grouped_notional.loc[ticker]:,.0f}")
        ax.grid(alpha=0.2)

    for ax in axes[len(top_tickers) :]:
        ax.axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return top_tickers


def _load_run_artifacts(summary_path: Path, backtest: dict) -> dict[str, pd.DataFrame]:
    artifact_keys = {
        "train_results": "results_csv_train",
        "test_results": "results_csv_test",
        "test_blank_results": "results_csv_test_blank",
        "train_trades": "trades_csv_train",
        "test_trades": "trades_csv_test",
        "test_blank_trades": "trades_csv_test_blank",
    }
    loaded = {}
    for label, artifact_key in artifact_keys.items():
        artifact_path = _resolve_artifact_path(summary_path, backtest[artifact_key])
        loaded[label] = pd.read_csv(artifact_path)
    return loaded


def main() -> int:
    args = parse_args()
    summary_path = args.summary_json.resolve()
    summary_payload = _load_summary(summary_path)
    backtest = _select_backtest(
        summary_payload,
        agent=args.agent,
        impact_model=args.impact_model,
        match_token=args.match_token,
    )

    benchmark_ticker = args.benchmark_ticker or summary_payload.get("benchmark_ticker", "QQEW")
    prep = _build_market_data(
        start_date=args.start_date,
        end_date=args.end_date,
        train_ratio=args.train_ratio,
        benchmark_ticker=benchmark_ticker,
    )
    train_config = prep.create_env_config(Split.TRAIN)
    price_df, tbill_df, full_tickers = _build_lookup_frames(prep)

    trade_tickers = list(full_tickers)
    train_tickers = list(train_config["tic_list"])
    artifacts = _load_run_artifacts(summary_path, backtest)

    train_trades = _attach_tickers(artifacts["train_trades"], train_tickers)
    test_trades = _attach_tickers(artifacts["test_trades"], trade_tickers)
    test_blank_trades = _attach_tickers(artifacts["test_blank_trades"], trade_tickers)

    inherited_holdings = _signed_trade_shares(train_trades)
    test_results = artifacts["test_results"].copy()
    test_results["date"] = pd.to_datetime(test_results["date"])
    test_anchor_date = pd.Timestamp(test_results["date"].iloc[0])
    test_hold = _approx_hold_counterfactual(
        results_df=test_results,
        price_df=price_df,
        tbill_df=tbill_df,
        holdings=inherited_holdings,
        initial_cash=float(test_results["cash"].iloc[0]),
        anchor_date=test_anchor_date,
        anchor_value=float(test_results["portfolio_value"].iloc[0]),
    )

    blank_results = artifacts["test_blank_results"].copy()
    blank_results["date"] = pd.to_datetime(blank_results["date"])
    first_live_step = blank_results[blank_results["step"] == 0].iloc[0]
    day0_holdings = _signed_trade_shares(test_blank_trades[test_blank_trades["step"] <= 0])
    blank_live_results = blank_results[blank_results["date"] >= pd.Timestamp(first_live_step["date"])]
    blank_hold = _approx_hold_counterfactual(
        results_df=blank_live_results,
        price_df=price_df,
        tbill_df=tbill_df,
        holdings=day0_holdings,
        initial_cash=float(first_live_step["cash"]),
        anchor_date=pd.Timestamp(first_live_step["date"]),
        anchor_value=float(first_live_step["portfolio_value"]),
    )

    horizons = [int(piece.strip()) for piece in args.horizons.split(",") if piece.strip()]
    test_event = _event_study(test_trades, price_df, horizons)
    blank_event = _event_study(test_blank_trades, price_df, horizons)

    run_slug = Path(backtest["results_csv_test"]).stem.replace("_test", "")
    output_dir = summary_path.parent / "adhoc_diagnostics" / run_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_counterfactual(
        test_results,
        test_hold,
        output_dir / "regular_counterfactual.png",
        title=(
            f"{run_slug}: regular OOS vs benchmark vs inherited-book hold\n"
            "Hold counterfactual is anchored to step -1 and ignores latent carried impact state"
        ),
        hold_label="Approx inherited-book hold",
    )

    blank_plot_frame = blank_results.copy()
    blank_plot_frame = blank_plot_frame[blank_plot_frame["date"] >= blank_hold.index[0]]
    _plot_counterfactual(
        blank_plot_frame,
        blank_hold,
        output_dir / "blank_counterfactual.png",
        title=(
            f"{run_slug}: blank-slate OOS vs benchmark vs frozen day-0 basket\n"
            "Hold counterfactual is anchored to step 0 and ignores latent impact state after day 0"
        ),
        hold_label="Approx frozen day-0 basket",
    )

    _plot_event_study(
        test_event,
        output_dir / "regular_event_study.png",
        title=f"{run_slug}: regular OOS trade event study",
    )
    _plot_event_study(
        blank_event,
        output_dir / "blank_event_study.png",
        title=f"{run_slug}: blank-slate OOS trade event study",
    )

    top_regular = _plot_top_tickers(
        test_trades,
        price_df.loc[test_results["date"].min() : test_results["date"].max()],
        output_dir / "regular_top_tickers.png",
        title=f"{run_slug}: regular OOS top traded tickers",
        top_n=args.top_n,
    )
    top_blank = _plot_top_tickers(
        test_blank_trades,
        price_df.loc[blank_results["date"].min() : blank_results["date"].max()],
        output_dir / "blank_top_tickers.png",
        title=f"{run_slug}: blank-slate OOS top traded tickers",
        top_n=args.top_n,
    )

    test_metrics = _counterfactual_metrics(test_results, test_hold)
    blank_metrics = _counterfactual_metrics(blank_plot_frame, blank_hold)
    residual_regular = float(
        test_results["portfolio_value"].iloc[0]
        - (
            float(test_results["cash"].iloc[0])
            + price_df.loc[test_anchor_date]
            .mul(inherited_holdings.reindex(price_df.columns, fill_value=0.0))
            .sum()
        )
    )
    residual_blank = float(
        float(first_live_step["portfolio_value"])
        - (
            float(first_live_step["cash"])
            + price_df.loc[pd.Timestamp(first_live_step["date"])]
            .mul(day0_holdings.reindex(price_df.columns, fill_value=0.0))
            .sum()
        )
    )
    summary = {
        "run_slug": run_slug,
        "drl_agent": backtest.get("drl_agent"),
        "impact_model": backtest.get("impact_model"),
        "benchmark_ticker": benchmark_ticker,
        "notes": [
            "Approximate hold counterfactuals are anchored to the observed portfolio value at the first plotted date.",
            "They use reconstructed holdings, raw close prices, and T-bill cash accrual, but they do not recreate latent carried market-impact state.",
            "Positive directional edge in the event study means buys outperformed the same-day universe average or sells underperformed it.",
        ],
        "regular": {
            **test_metrics,
            "trade_count": int(len(test_trades)),
            "traded_tickers": int(test_trades["ticker"].nunique()),
            "top_tickers": top_regular,
            "anchor_residual_value": residual_regular,
            "event_study": test_event.to_dict(orient="records"),
        },
        "blank": {
            **blank_metrics,
            "trade_count": int(len(test_blank_trades)),
            "traded_tickers": int(test_blank_trades["ticker"].nunique()),
            "top_tickers": top_blank,
            "anchor_residual_value": residual_blank,
            "event_study": blank_event.to_dict(orient="records"),
        },
    }
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from typing import Optional

import pandas as pd


DEFAULT_MODEL_SIGNATURES: dict[str, dict[str, Any]] = {
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "ent_coef": 0.01,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.01,
    },
    "ddpg": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "buffer_size": 50000,
    },
    "sac": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "buffer_size": 50000,
    },
    "td3": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "buffer_size": 50000,
    },
}


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected

    if isinstance(expected, (int, float)):
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False

    return actual == expected


def matches_default_model_signature(metadata: dict[str, Any]) -> bool:
    agent = str(metadata.get("drl_agent", "")).lower()
    expected_signature = DEFAULT_MODEL_SIGNATURES.get(agent)
    if expected_signature is None:
        return False

    model_kwargs = metadata.get("model_kwargs") or {}
    for key, expected_value in expected_signature.items():
        actual_value = model_kwargs.get(key, metadata.get(key))
        if actual_value is None or not _values_match(actual_value, expected_value):
            return False
    return True


def infer_run_scope(metadata: dict[str, Any]) -> str:
    training_engine = str(metadata.get("training_engine") or "").lower()
    env_type = str(metadata.get("env_type") or "").lower()
    num_envs = metadata.get("num_envs")
    requested_num_envs = metadata.get("requested_num_envs")

    if "vec" in training_engine or env_type.endswith("_vec"):
        return "Vec"
    if isinstance(num_envs, (int, float)) and int(num_envs) > 1:
        return "Vec"
    if isinstance(requested_num_envs, (int, float)) and int(requested_num_envs) > 1:
        return "Vec"
    return "Scalar"


def infer_summary_suite_kind(backtests: list[dict[str, Any]]) -> str:
    scopes = {infer_run_scope(bt) for bt in backtests}
    pair_counts = Counter(
        (bt.get("drl_agent"), bt.get("impact_model")) for bt in backtests
    )
    duplicate_group_count = sum(count > 1 for count in pair_counts.values())
    unique_impacts = {bt.get("impact_model") for bt in backtests}

    if "Vec" in scopes:
        if duplicate_group_count > 0:
            return "vec_comparison"
        return "vec_suite"

    if len(unique_impacts) == 1 and duplicate_group_count > 0:
        return "hpo_comparison"
    if duplicate_group_count > 0:
        return "reference_comparison"
    return "backtest_suite"


def get_best_epoch_stats(
    epoch_stats: list[dict[str, Any]],
    metric_key: str = "annualized_sharpe",
) -> Optional[dict[str, Any]]:
    valid_rows = [row for row in epoch_stats if row.get(metric_key) is not None]
    if not valid_rows:
        return None
    return dict(max(valid_rows, key=lambda row: row[metric_key]))


def _suite_role_for_metadata(
    metadata: dict[str, Any],
    suite_kind: str,
) -> str:
    is_default = matches_default_model_signature(metadata)
    scope = infer_run_scope(metadata)

    if suite_kind == "hpo_comparison":
        return "Default (HPO Suite)" if is_default else "HPO Tuned"
    if suite_kind == "reference_comparison":
        return "Reference Default" if is_default else "Reference Alt"
    if suite_kind in {"vec_comparison", "vec_suite"} or scope == "Vec":
        return "Vec Run"
    if is_default:
        return "Reference Default"
    return "Custom"


def enrich_backtests_metadata(
    backtests: list[dict[str, Any]],
    *,
    suite_kind: Optional[str] = None,
) -> list[dict[str, Any]]:
    resolved_suite_kind = suite_kind or infer_summary_suite_kind(backtests)
    enriched_backtests: list[dict[str, Any]] = []

    for metadata in backtests:
        enriched = dict(metadata)
        run_scope = enriched.get("run_scope") or infer_run_scope(enriched)
        suite_role = enriched.get("suite_role") or _suite_role_for_metadata(
            enriched,
            resolved_suite_kind,
        )
        epoch_stats_train = enriched.get("epoch_stats_train") or []
        epoch_stats_test_blank = enriched.get("epoch_stats_test_blank") or []

        enriched.setdefault("run_scope", run_scope)
        enriched.setdefault("suite_kind", resolved_suite_kind)
        enriched.setdefault("suite_role", suite_role)
        enriched.setdefault("suite_label", f"{run_scope} | {suite_role}")

        if epoch_stats_train:
            enriched.setdefault("final_epoch_stats_train", dict(epoch_stats_train[-1]))
        if epoch_stats_test_blank:
            enriched.setdefault(
                "final_epoch_stats_test_blank",
                dict(epoch_stats_test_blank[-1]),
            )
            best_test_blank = get_best_epoch_stats(epoch_stats_test_blank)
            if best_test_blank is not None:
                enriched.setdefault("best_epoch_stats_test_blank", best_test_blank)

        enriched_backtests.append(enriched)

    return enriched_backtests


def enrich_summary_payload(summary_data: dict[str, Any]) -> dict[str, Any]:
    enriched_summary = dict(summary_data)
    backtests = enriched_summary.get("backtests", [])
    suite_kind = enriched_summary.get("summary_suite_kind") or infer_summary_suite_kind(
        backtests
    )
    enriched_summary["summary_suite_kind"] = suite_kind
    enriched_summary["backtests"] = enrich_backtests_metadata(
        backtests,
        suite_kind=suite_kind,
    )
    return enriched_summary


def prepare_summary_payload(
    *,
    benchmark_ticker: str,
    backtests: list[dict[str, Any]],
    extra_summary_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark_ticker": benchmark_ticker,
        "backtests": backtests,
    }
    if extra_summary_fields:
        payload.update(extra_summary_fields)
    return enrich_summary_payload(payload)


def load_summary_payload(summary_path: str | Path) -> dict[str, Any]:
    with open(summary_path, "r") as file:
        return enrich_summary_payload(json.load(file))


def _select_backtest(
    backtests: list[dict[str, Any]],
    *,
    agent: str,
    impact_model: Optional[str] = None,
    suite_role: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    candidates = [
        backtest
        for backtest in backtests
        if str(backtest.get("drl_agent", "")).lower() == agent.lower()
    ]
    if impact_model is not None:
        candidates = [
            backtest
            for backtest in candidates
            if backtest.get("impact_model") == impact_model
        ]
    if suite_role is not None:
        exact_role_matches = [
            backtest
            for backtest in candidates
            if backtest.get("suite_role") == suite_role
        ]
        if exact_role_matches:
            candidates = exact_role_matches

    if not candidates:
        return None

    def _selection_key(backtest: dict[str, Any]) -> tuple[float, float]:
        best_stats = backtest.get("best_epoch_stats_test_blank") or {}
        final_stats = backtest.get("final_epoch_stats_test_blank") or {}
        return (
            float(best_stats.get("annualized_sharpe", float("-inf"))),
            float(final_stats.get("annualized_sharpe", float("-inf"))),
        )

    return max(candidates, key=_selection_key)


def compare_summary_payloads(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    impact_model: Optional[str] = None,
    left_role: Optional[str] = None,
    right_role: Optional[str] = None,
    agents: Optional[list[str]] = None,
) -> pd.DataFrame:
    left_backtests = left_summary.get("backtests", [])
    right_backtests = right_summary.get("backtests", [])

    if agents is None:
        left_agents = {str(bt.get("drl_agent", "")).lower() for bt in left_backtests}
        right_agents = {
            str(bt.get("drl_agent", "")).lower() for bt in right_backtests
        }
        agents = sorted(left_agents & right_agents)

    rows: list[dict[str, Any]] = []
    for agent in agents:
        left_bt = _select_backtest(
            left_backtests,
            agent=agent,
            impact_model=impact_model,
            suite_role=left_role,
        )
        right_bt = _select_backtest(
            right_backtests,
            agent=agent,
            impact_model=impact_model,
            suite_role=right_role,
        )
        if left_bt is None or right_bt is None:
            continue

        left_final = left_bt.get("final_epoch_stats_test_blank") or {}
        right_final = right_bt.get("final_epoch_stats_test_blank") or {}
        left_best = left_bt.get("best_epoch_stats_test_blank") or {}
        right_best = right_bt.get("best_epoch_stats_test_blank") or {}

        rows.append(
            {
                "agent": agent.upper(),
                "impact_model": impact_model or right_bt.get("impact_model"),
                "left_label": left_bt.get("suite_label"),
                "right_label": right_bt.get("suite_label"),
                "left_final_sharpe": left_final.get("annualized_sharpe"),
                "right_final_sharpe": right_final.get("annualized_sharpe"),
                "delta_final_sharpe": (
                    None
                    if left_final.get("annualized_sharpe") is None
                    or right_final.get("annualized_sharpe") is None
                    else right_final["annualized_sharpe"]
                    - left_final["annualized_sharpe"]
                ),
                "left_final_return": left_final.get("annualized_return"),
                "right_final_return": right_final.get("annualized_return"),
                "delta_final_return": (
                    None
                    if left_final.get("annualized_return") is None
                    or right_final.get("annualized_return") is None
                    else right_final["annualized_return"]
                    - left_final["annualized_return"]
                ),
                "left_final_cost": left_final.get("total_trading_cost"),
                "right_final_cost": right_final.get("total_trading_cost"),
                "delta_final_cost": (
                    None
                    if left_final.get("total_trading_cost") is None
                    or right_final.get("total_trading_cost") is None
                    else right_final["total_trading_cost"]
                    - left_final["total_trading_cost"]
                ),
                "left_final_turnover": left_final.get("avg_daily_turnover"),
                "right_final_turnover": right_final.get("avg_daily_turnover"),
                "delta_final_turnover": (
                    None
                    if left_final.get("avg_daily_turnover") is None
                    or right_final.get("avg_daily_turnover") is None
                    else right_final["avg_daily_turnover"]
                    - left_final["avg_daily_turnover"]
                ),
                "left_best_epoch": left_best.get("epoch"),
                "right_best_epoch": right_best.get("epoch"),
                "left_best_sharpe": left_best.get("annualized_sharpe"),
                "right_best_sharpe": right_best.get("annualized_sharpe"),
                "delta_best_sharpe": (
                    None
                    if left_best.get("annualized_sharpe") is None
                    or right_best.get("annualized_sharpe") is None
                    else right_best["annualized_sharpe"]
                    - left_best["annualized_sharpe"]
                ),
                "left_best_cost": left_best.get("total_trading_cost"),
                "right_best_cost": right_best.get("total_trading_cost"),
                "delta_best_cost": (
                    None
                    if left_best.get("total_trading_cost") is None
                    or right_best.get("total_trading_cost") is None
                    else right_best["total_trading_cost"]
                    - left_best["total_trading_cost"]
                ),
                "left_best_turnover": left_best.get("avg_daily_turnover"),
                "right_best_turnover": right_best.get("avg_daily_turnover"),
                "delta_best_turnover": (
                    None
                    if left_best.get("avg_daily_turnover") is None
                    or right_best.get("avg_daily_turnover") is None
                    else right_best["avg_daily_turnover"]
                    - left_best["avg_daily_turnover"]
                ),
            }
        )

    return pd.DataFrame(rows)
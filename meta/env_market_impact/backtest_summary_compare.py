from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .backtest_summary_utils import compare_summary_payloads
from .backtest_summary_utils import load_summary_payload


class BacktestSummaryComparator:
    def __init__(self, left_summary_path: str, right_summary_path: str):
        self.left_summary_path = Path(left_summary_path)
        self.right_summary_path = Path(right_summary_path)
        self.left_summary = load_summary_payload(self.left_summary_path)
        self.right_summary = load_summary_payload(self.right_summary_path)

    def compare_agents(
        self,
        *,
        impact_model: Optional[str] = None,
        left_role: Optional[str] = None,
        right_role: Optional[str] = None,
        agents: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return compare_summary_payloads(
            self.left_summary,
            self.right_summary,
            impact_model=impact_model,
            left_role=left_role,
            right_role=right_role,
            agents=agents,
        )

    def save_html(
        self,
        output_path: str,
        *,
        impact_model: Optional[str] = None,
        left_role: Optional[str] = None,
        right_role: Optional[str] = None,
        agents: Optional[list[str]] = None,
        title: str = "Backtest Summary Comparison",
    ) -> str:
        comparison_df = self.compare_agents(
            impact_model=impact_model,
            left_role=left_role,
            right_role=right_role,
            agents=agents,
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            f"<title>{title}</title>"
            "<style>body{font-family:Arial,sans-serif;margin:20px;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #d1d5db;padding:8px;text-align:left;font-size:13px;}"
            "th{background:#e5eef9;}tr:nth-child(even){background:#f8fafc;}"
            "</style></head><body>"
            f"<h1>{title}</h1>"
            f"<p>Left: {self.left_summary_path}</p>"
            f"<p>Right: {self.right_summary_path}</p>"
            f"{comparison_df.to_html(index=False, float_format=lambda x: f'{x:.3f}')}"
            "</body></html>"
        )
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path
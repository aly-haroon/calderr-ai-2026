"""
visualization.py

Given the result of a code-execution step, decide whether/how to render a
chart automatically. This module handles the case where the LLM's generated
code did NOT already produce a matplotlib figure (code_executor.py passes
one through if it exists) — here we build a sensible default chart purely
from the *shape* of the result, as a convenience fallback.

Chart selection heuristic:
    * Series/DataFrame with a single numeric column, few unique index values
      (<= 10) -> bar chart (good for aggregates like groupby().mean())
    * Series/DataFrame with a datetime-like index -> line chart (time series)
    * Single numeric column with many rows -> histogram (distribution)
    * DataFrame with exactly two numeric columns -> scatter plot (correlation)
    * Series that looks like proportions of a whole and has <= 6 categories
      -> pie chart
    * Anything else -> no chart (plain table/value is clearer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MAX_BAR_CATEGORIES = 15
MAX_PIE_CATEGORIES = 6
HISTOGRAM_MIN_ROWS = 20


class VisualizationError(Exception):
    """Raised when a chart cannot be safely built from the given result."""


@dataclass
class ChartResult:
    """A generated chart plus metadata about why it was chosen."""

    figure: Any
    chart_type: str
    reason: str


def _is_datetime_like(index: pd.Index) -> bool:
    return isinstance(index, pd.DatetimeIndex) or "datetime" in str(index.dtype).lower()


def build_chart_from_result(result: Any) -> ChartResult | None:
    """
    Attempt to automatically build a chart from a code-execution result.

    Args:
        result: The `result` variable produced by generated pandas code.
            Typically a DataFrame, Series, or scalar.

    Returns:
        A ChartResult, or None if no sensible chart could be inferred
        (e.g. the result is a plain scalar/string).
    """
    try:
        if isinstance(result, pd.Series):
            return _chart_from_series(result)
        if isinstance(result, pd.DataFrame):
            return _chart_from_dataframe(result)
        return None  # scalars / strings: no chart, just display the value
    except VisualizationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VisualizationError(f"Could not build a chart automatically: {exc}") from exc


def _chart_from_series(series: pd.Series) -> ChartResult | None:
    if series.empty:
        return None

    numeric = pd.api.types.is_numeric_dtype(series)

    if _is_datetime_like(series.index) and numeric:
        fig, ax = plt.subplots(figsize=(8, 4))
        series.plot(kind="line", ax=ax, marker="o")
        ax.set_title("Trend over time")
        ax.set_ylabel(series.name or "value")
        fig.tight_layout()
        return ChartResult(fig, "line", "Datetime index detected -> line chart")

    if numeric and series.nunique() <= MAX_PIE_CATEGORIES and (series >= 0).all():
        fig, ax = plt.subplots(figsize=(6, 6))
        series.plot(kind="pie", ax=ax, autopct="%1.1f%%")
        ax.set_ylabel("")
        ax.set_title("Proportion breakdown")
        fig.tight_layout()
        return ChartResult(fig, "pie", "Few non-negative numeric categories -> pie chart")

    if numeric and len(series) <= MAX_BAR_CATEGORIES:
        fig, ax = plt.subplots(figsize=(8, 4))
        series.plot(kind="bar", ax=ax, color="#2563eb")
        ax.set_title("Comparison by category")
        fig.tight_layout()
        return ChartResult(fig, "bar", "Numeric series with few categories -> bar chart")

    if numeric and len(series) >= HISTOGRAM_MIN_ROWS:
        fig, ax = plt.subplots(figsize=(8, 4))
        series.plot(kind="hist", ax=ax, bins=20, color="#2563eb", edgecolor="white")
        ax.set_title("Distribution")
        fig.tight_layout()
        return ChartResult(fig, "histogram", "Long numeric series -> histogram")

    return None


def _chart_from_dataframe(df: pd.DataFrame) -> ChartResult | None:
    if df.empty:
        return None

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if _is_datetime_like(df.index) and numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[numeric_cols].plot(kind="line", ax=ax, marker="o")
        ax.set_title("Trend over time")
        fig.tight_layout()
        return ChartResult(fig, "line", "Datetime index with numeric columns -> line chart")

    if len(numeric_cols) == 2 and len(df) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7, color="#2563eb")
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f"{numeric_cols[1]} vs {numeric_cols[0]}")
        fig.tight_layout()
        return ChartResult(fig, "scatter", "Exactly two numeric columns -> scatter plot")

    if len(numeric_cols) >= 1 and len(df) <= MAX_BAR_CATEGORIES:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[numeric_cols].plot(kind="bar", ax=ax)
        ax.set_title("Comparison by category")
        fig.tight_layout()
        return ChartResult(fig, "bar", "Few rows with numeric columns -> bar chart")

    if len(numeric_cols) == 1 and len(df) >= HISTOGRAM_MIN_ROWS:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[numeric_cols[0]].plot(kind="hist", ax=ax, bins=20, color="#2563eb", edgecolor="white")
        ax.set_title(f"Distribution of {numeric_cols[0]}")
        fig.tight_layout()
        return ChartResult(fig, "histogram", "Single numeric column, many rows -> histogram")

    return None

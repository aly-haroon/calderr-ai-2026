"""
schema_analyzer.py

Responsible for inspecting a pandas DataFrame and producing a structured,
serializable summary of its shape and contents. This summary is later:
    * shown to the user in the Streamlit "Schema" panel, and
    * injected into the LLM prompt so the Groq code generator knows what
      columns/types it is allowed to reference.

Keeping this logic isolated makes it easy to unit test independently of
Streamlit or the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnInfo:
    """Metadata describing a single column."""

    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class SchemaReport:
    """Full schema report for a DataFrame."""

    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    numerical_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    sample_rows: list[dict[str, Any]]
    total_missing_cells: int
    memory_usage_kb: float

    def to_prompt_string(self) -> str:
        """
        Render a compact, LLM-friendly text description of the schema.
        This is injected directly into the code-generation prompt so the
        model knows exact column names and dtypes and does not hallucinate
        columns that do not exist.
        """
        lines = [
            f"Rows: {self.row_count}, Columns: {self.column_count}",
            "Column details:",
        ]
        for col in self.columns:
            lines.append(
                f"  - {col.name} (dtype={col.dtype}, "
                f"missing={col.missing_count}, unique={col.unique_count})"
            )
        lines.append(f"Numerical columns: {', '.join(self.numerical_columns) or 'none'}")
        lines.append(f"Categorical columns: {', '.join(self.categorical_columns) or 'none'}")
        if self.datetime_columns:
            lines.append(f"Datetime columns: {', '.join(self.datetime_columns)}")
        return "\n".join(lines)


class SchemaAnalyzerError(Exception):
    """Raised when a DataFrame cannot be analyzed (e.g. empty dataset)."""


def analyze_schema(df: pd.DataFrame, sample_size: int = 5) -> SchemaReport:
    """
    Build a SchemaReport describing `df`.

    Args:
        df: The DataFrame to inspect.
        sample_size: Number of sample rows to include for preview purposes.

    Returns:
        A populated SchemaReport.

    Raises:
        SchemaAnalyzerError: If the DataFrame is empty or has no columns.
    """
    if df is None or df.empty:
        raise SchemaAnalyzerError("The uploaded dataset is empty. Please upload a valid CSV.")
    if df.shape[1] == 0:
        raise SchemaAnalyzerError("The uploaded dataset has no columns.")

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    # Categorical = everything that is not numeric and not datetime
    categorical_columns = [
        c for c in df.columns if c not in numerical_columns and c not in datetime_columns
    ]

    columns: list[ColumnInfo] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        columns.append(
            ColumnInfo(
                name=str(col),
                dtype=str(series.dtype),
                missing_count=missing,
                missing_pct=round((missing / len(df)) * 100, 2) if len(df) else 0.0,
                unique_count=int(series.nunique(dropna=True)),
                is_numeric=col in numerical_columns,
                is_categorical=col in categorical_columns,
                sample_values=series.dropna().head(3).tolist(),
            )
        )

    sample_rows = df.head(sample_size).to_dict(orient="records")

    return SchemaReport(
        row_count=len(df),
        column_count=df.shape[1],
        columns=columns,
        numerical_columns=[str(c) for c in numerical_columns],
        categorical_columns=[str(c) for c in categorical_columns],
        datetime_columns=[str(c) for c in datetime_columns],
        sample_rows=sample_rows,
        total_missing_cells=int(df.isna().sum().sum()),
        memory_usage_kb=round(df.memory_usage(deep=True).sum() / 1024, 2),
    )

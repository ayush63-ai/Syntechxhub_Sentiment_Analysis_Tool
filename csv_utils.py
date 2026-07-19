"""CSV upload helpers for batch sentiment analysis."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TEXT_COLUMN_ALIASES = (
    "Tweet Text",
    "text",
    "review",
    "reviews",
    "feedback",
    "comment",
    "message",
    "content",
)


def _normalize_column_name(column: str) -> str:
    return " ".join(column.strip().lower().replace("_", " ").split())


def preferred_text_column(columns: Iterable[object]) -> str | None:
    """Return the best text column for common review and Xquik exports."""
    original_columns = [str(column) for column in columns]
    normalized_to_original = {
        _normalize_column_name(column): column
        for column in original_columns
    }

    for alias in TEXT_COLUMN_ALIASES:
        match = normalized_to_original.get(_normalize_column_name(alias))
        if match is not None:
            return match

    return original_columns[0] if original_columns else None


def filter_nonempty_text_rows(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Keep only rows with analyzable text in the selected column."""
    filtered_df = df.copy()
    filtered_df[text_column] = filtered_df[text_column].fillna("").astype(str).str.strip()
    return filtered_df[filtered_df[text_column] != ""].reset_index(drop=True)

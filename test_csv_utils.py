"""Tests for batch CSV upload helpers."""

import pandas as pd

from csv_utils import filter_nonempty_text_rows, preferred_text_column


def test_preferred_text_column_selects_xquik_tweet_text_header():
    columns = ["Tweet ID", "Tweet Text", "Created At"]

    assert preferred_text_column(columns) == "Tweet Text"


def test_preferred_text_column_falls_back_to_first_column():
    columns = ["unknown", "other"]

    assert preferred_text_column(columns) == "unknown"


def test_filter_nonempty_text_rows_strips_blank_reviews():
    df = pd.DataFrame(
        {
            "Tweet Text": [" Great launch ", "", None, "Needs work"],
            "Tweet ID": ["1", "2", "3", "4"],
        }
    )

    filtered_df = filter_nonempty_text_rows(df, "Tweet Text")

    assert filtered_df["Tweet Text"].tolist() == ["Great launch", "Needs work"]
    assert filtered_df["Tweet ID"].tolist() == ["1", "4"]

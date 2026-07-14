"""Tests for multi-seed robustness aggregation."""

import pandas as pd
import pytest

from src.analysis.text_robustness_multiseed import (
    summarize_runs,
)


def test_summarize_runs_computes_mean():
    raw = pd.DataFrame(
        [
            {
                "seed": 1,
                "word_dropout_rate": 0.5,
                "remaining_word_fraction": 0.5,
                "accuracy": 0.4,
                "macro_f1": 0.3,
                "weighted_f1": 0.35,
                "macro_f1_drop": 0.1,
                "neutral_prediction_fraction": 0.5,
            },
            {
                "seed": 2,
                "word_dropout_rate": 0.5,
                "remaining_word_fraction": 0.5,
                "accuracy": 0.6,
                "macro_f1": 0.5,
                "weighted_f1": 0.55,
                "macro_f1_drop": 0.2,
                "neutral_prediction_fraction": 0.7,
            },
        ]
    )

    summary = summarize_runs(raw)
    row = summary.iloc[0]

    assert row["accuracy_mean"] == pytest.approx(0.5)
    assert row["macro_f1_mean"] == pytest.approx(0.4)
    assert (
        row["neutral_prediction_fraction_mean"]
        == pytest.approx(0.6)
    )


def test_identical_runs_have_zero_standard_deviation():
    raw = pd.DataFrame(
        [
            {
                "seed": 1,
                "word_dropout_rate": 1.0,
                "remaining_word_fraction": 0.0,
                "accuracy": 0.48,
                "macro_f1": 0.09,
                "weighted_f1": 0.31,
                "macro_f1_drop": 0.28,
                "neutral_prediction_fraction": 1.0,
            },
            {
                "seed": 2,
                "word_dropout_rate": 1.0,
                "remaining_word_fraction": 0.0,
                "accuracy": 0.48,
                "macro_f1": 0.09,
                "weighted_f1": 0.31,
                "macro_f1_drop": 0.28,
                "neutral_prediction_fraction": 1.0,
            },
        ]
    )

    summary = summarize_runs(raw)

    assert summary.iloc[0]["macro_f1_std"] == pytest.approx(
        0.0
    )

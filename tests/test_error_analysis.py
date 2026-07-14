"""Tests for text-baseline error analysis."""

import pandas as pd
import pytest

from src.analysis.error_analysis import (
    build_confusion_pairs,
    build_length_summary,
    build_per_class_summary,
    prepare_predictions,
    validate_predictions_frame,
)


def make_predictions() -> pd.DataFrame:
    """Create a small synthetic prediction table."""
    return pd.DataFrame(
        [
            {
                "Dialogue_ID": 0,
                "Utterance_ID": 0,
                "Speaker": "A",
                "Utterance": "I am happy",
                "Emotion": "joy",
                "Predicted_Emotion": "joy",
            },
            {
                "Dialogue_ID": 0,
                "Utterance_ID": 1,
                "Speaker": "B",
                "Utterance": "This is very bad",
                "Emotion": "sadness",
                "Predicted_Emotion": "anger",
            },
            {
                "Dialogue_ID": 1,
                "Utterance_ID": 0,
                "Speaker": "A",
                "Utterance": "No",
                "Emotion": "anger",
                "Predicted_Emotion": "neutral",
            },
            {
                "Dialogue_ID": 1,
                "Utterance_ID": 1,
                "Speaker": "B",
                "Utterance": (
                    "I do not know what is happening here today"
                ),
                "Emotion": "neutral",
                "Predicted_Emotion": "neutral",
            },
        ]
    )


def test_validation_rejects_missing_columns():
    incomplete = pd.DataFrame(
        [{"Utterance": "Hello"}]
    )

    with pytest.raises(
        ValueError,
        match="Missing required prediction columns",
    ):
        validate_predictions_frame(incomplete)


def test_prepare_predictions_adds_correctness():
    prepared = prepare_predictions(
        make_predictions()
    )

    assert prepared["Correct"].tolist() == [
        True,
        False,
        False,
        True,
    ]

    assert prepared.loc[2, "Word_Count"] == 1


def test_confusion_pairs_only_include_errors():
    prepared = prepare_predictions(
        make_predictions()
    )

    pairs = build_confusion_pairs(prepared)

    assert len(pairs) == 2
    assert pairs["Count"].sum() == 2


def test_per_class_summary_computes_recall():
    prepared = prepare_predictions(
        make_predictions()
    )

    summary = build_per_class_summary(
        prepared
    )

    joy = summary[
        summary["Emotion"] == "joy"
    ].iloc[0]

    assert joy["Recall"] == pytest.approx(1.0)


def test_length_summary_counts_all_samples():
    prepared = prepare_predictions(
        make_predictions()
    )

    summary = build_length_summary(
        prepared
    )

    assert summary["Samples"].sum() == 4

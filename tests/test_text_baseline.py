"""Tests for the MELD text-only baseline."""

import pytest

from src.models.text_baseline import (
    build_text_pipeline,
    compute_metrics,
)


def test_compute_metrics_for_perfect_predictions():
    true_labels = ["joy", "sadness", "neutral"]
    predicted_labels = ["joy", "sadness", "neutral"]

    metrics = compute_metrics(
        true_labels,
        predicted_labels,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)


def test_text_pipeline_can_fit_and_predict():
    texts = [
        "I am very happy",
        "This is wonderful",
        "I feel terrible",
        "This is awful",
    ]

    labels = [
        "joy",
        "joy",
        "sadness",
        "sadness",
    ]

    model = build_text_pipeline(
        min_df=1,
        max_features=100,
    )

    model.fit(texts, labels)
    predictions = model.predict(texts)

    assert len(predictions) == len(texts)
    assert set(predictions).issubset(
        {"joy", "sadness"}
    )

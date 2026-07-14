"""Tests for text word-dropout robustness utilities."""

import pytest

from src.analysis.text_robustness import (
    apply_word_dropout,
)


def test_zero_dropout_preserves_text():
    result = apply_word_dropout(
        text="this is a test",
        dropout_rate=0.0,
        seed=42,
    )

    assert result == "this is a test"


def test_complete_dropout_removes_all_words():
    result = apply_word_dropout(
        text="this is a test",
        dropout_rate=1.0,
        seed=42,
    )

    assert result == ""


def test_dropout_is_deterministic():
    first = apply_word_dropout(
        text="one two three four five",
        dropout_rate=0.5,
        seed=42,
    )

    second = apply_word_dropout(
        text="one two three four five",
        dropout_rate=0.5,
        seed=42,
    )

    assert first == second


def test_partial_dropout_keeps_at_least_one_word():
    result = apply_word_dropout(
        text="one two",
        dropout_rate=0.99,
        seed=42,
    )

    assert len(result.split()) >= 1


def test_invalid_dropout_rate_is_rejected():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        apply_word_dropout(
            text="example",
            dropout_rate=1.5,
            seed=42,
        )

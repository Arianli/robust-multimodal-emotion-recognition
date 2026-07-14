"""Tests for the context-aware MELD text baseline."""

import pandas as pd
import pytest

from src.models.context_text_baseline import (
    add_dialogue_context,
)


def make_dialogues() -> pd.DataFrame:
    """Create two small dialogues."""
    return pd.DataFrame(
        [
            {
                "Dialogue_ID": 0,
                "Utterance_ID": 0,
                "Speaker": "A",
                "Utterance": "Hello",
                "Emotion": "neutral",
            },
            {
                "Dialogue_ID": 0,
                "Utterance_ID": 1,
                "Speaker": "B",
                "Utterance": "How are you?",
                "Emotion": "neutral",
            },
            {
                "Dialogue_ID": 0,
                "Utterance_ID": 2,
                "Speaker": "A",
                "Utterance": "I am upset",
                "Emotion": "anger",
            },
            {
                "Dialogue_ID": 1,
                "Utterance_ID": 0,
                "Speaker": "C",
                "Utterance": "New dialogue",
                "Emotion": "neutral",
            },
        ]
    )


def test_context_includes_previous_turns():
    prepared = add_dialogue_context(
        make_dialogues(),
        context_window=2,
    )

    target = prepared[
        (prepared["Dialogue_ID"] == 0)
        & (prepared["Utterance_ID"] == 2)
    ].iloc[0]

    assert "Hello" in target["Context_Text"]
    assert "How are you?" in target["Context_Text"]
    assert "I am upset" in target["Context_Text"]


def test_context_does_not_cross_dialogues():
    prepared = add_dialogue_context(
        make_dialogues(),
        context_window=3,
    )

    target = prepared[
        (prepared["Dialogue_ID"] == 1)
        & (prepared["Utterance_ID"] == 0)
    ].iloc[0]

    assert "New dialogue" in target["Context_Text"]
    assert "Hello" not in target["Context_Text"]
    assert "How are you?" not in target["Context_Text"]


def test_zero_context_only_contains_current_turn():
    prepared = add_dialogue_context(
        make_dialogues(),
        context_window=0,
    )

    target = prepared[
        (prepared["Dialogue_ID"] == 0)
        & (prepared["Utterance_ID"] == 2)
    ].iloc[0]

    assert "I am upset" in target["Context_Text"]
    assert "Hello" not in target["Context_Text"]


def test_negative_context_window_is_rejected():
    with pytest.raises(
        ValueError,
        match="context_window must be non-negative",
    ):
        add_dialogue_context(
            make_dialogues(),
            context_window=-1,
        )

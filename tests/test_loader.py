"""Tests for the MELD metadata loader."""

import pandas as pd
import pytest

from src.data.loader import load_split, validate_meld_frame


VALID_ROW = {
    "Utterance": "This is a test utterance.",
    "Speaker": "Speaker A",
    "Emotion": "neutral",
    "Sentiment": "neutral",
    "Dialogue_ID": 0,
    "Utterance_ID": 0,
    "Season": 1,
    "Episode": 1,
    "StartTime": "00:00:00,000",
    "EndTime": "00:00:01,000",
}


def test_load_split_from_local_directory(tmp_path):
    expected = pd.DataFrame([VALID_ROW])
    expected.to_csv(tmp_path / "train_sent_emo.csv", index=False)

    actual = load_split("train", source=tmp_path)

    assert len(actual) == 1
    assert actual.loc[0, "Emotion"] == "neutral"
    assert actual.loc[0, "Dialogue_ID"] == 0


def test_validate_rejects_missing_columns():
    incomplete = pd.DataFrame(
        [{"Utterance": "Missing most required fields."}]
    )

    with pytest.raises(ValueError, match="Missing required MELD columns"):
        validate_meld_frame(incomplete)


def test_validate_rejects_unknown_emotion():
    invalid_row = VALID_ROW.copy()
    invalid_row["Emotion"] = "confused"

    with pytest.raises(ValueError, match="Unknown MELD emotion labels"):
        validate_meld_frame(pd.DataFrame([invalid_row]))


def test_load_split_rejects_invalid_split_name():
    with pytest.raises(ValueError, match="Unknown split"):
        load_split("validation")

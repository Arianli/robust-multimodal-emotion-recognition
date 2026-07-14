"""Tests for MELD metadata auditing."""

import pandas as pd

from src.data.audit import (
    build_audit_report,
    summarize_split,
)


def make_frame(
    dialogue_id: int,
    utterances: list[str],
    start_second: int = 0,
) -> pd.DataFrame:
    """Create a small MELD-style metadata table."""
    rows = []

    for utterance_id, utterance in enumerate(
        utterances
    ):
        second = start_second + utterance_id

        rows.append(
            {
                "Utterance": utterance,
                "Speaker": "Speaker A",
                "Emotion": "neutral",
                "Sentiment": "neutral",
                "Dialogue_ID": dialogue_id,
                "Utterance_ID": utterance_id,
                "Season": 1,
                "Episode": 1,
                "StartTime": f"00:00:{second:02d},000",
                "EndTime": f"00:00:{second:02d},900",
            }
        )

    return pd.DataFrame(rows)


def test_summarize_split_counts_dialogues():
    first = make_frame(
        dialogue_id=0,
        utterances=["Hello", "Hi"],
    )
    second = make_frame(
        dialogue_id=1,
        utterances=["Goodbye"],
        start_second=10,
    )

    dataframe = pd.concat(
        [first, second],
        ignore_index=True,
    )

    summary = summarize_split(dataframe)

    assert summary["rows"] == 3
    assert summary["dialogues"] == 2
    assert summary["duplicate_records"] == 0


def test_same_numeric_id_is_not_content_overlap():
    train = make_frame(
        dialogue_id=0,
        utterances=["Training dialogue"],
    )
    dev = make_frame(
        dialogue_id=0,
        utterances=["Different development dialogue"],
        start_second=10,
    )
    test = make_frame(
        dialogue_id=0,
        utterances=["Different test dialogue"],
        start_second=20,
    )

    report = build_audit_report(
        {
            "train": train,
            "dev": dev,
            "test": test,
        }
    )

    overlap = report[
        "cross_split_overlap_counts"
    ]["train_vs_dev"]

    assert overlap["exact_clip_overlaps"] == 0
    assert overlap["exact_dialogue_overlaps"] == 0


def test_identical_dialogue_content_is_detected():
    train = make_frame(
        dialogue_id=0,
        utterances=["Same dialogue"],
    )
    dev = make_frame(
        dialogue_id=99,
        utterances=["Same dialogue"],
    )
    test = make_frame(
        dialogue_id=1,
        utterances=["Different dialogue"],
        start_second=20,
    )

    report = build_audit_report(
        {
            "train": train,
            "dev": dev,
            "test": test,
        }
    )

    overlap = report[
        "cross_split_overlap_counts"
    ]["train_vs_dev"]

    assert overlap["exact_clip_overlaps"] == 1
    assert overlap["exact_dialogue_overlaps"] == 1

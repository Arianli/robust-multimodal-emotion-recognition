"""Utilities for loading and validating MELD metadata."""

from pathlib import Path

import pandas as pd


DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD"
)

SPLIT_FILENAMES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

REQUIRED_COLUMNS = {
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
}

EXPECTED_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
}


def validate_meld_frame(dataframe: pd.DataFrame) -> None:
    """Validate the schema and emotion labels of a MELD metadata table."""
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required MELD columns: {missing}")

    observed_emotions = set(dataframe["Emotion"].dropna().unique())
    unknown_emotions = observed_emotions - EXPECTED_EMOTIONS

    if unknown_emotions:
        unknown = ", ".join(sorted(unknown_emotions))
        raise ValueError(f"Unknown MELD emotion labels: {unknown}")


def load_split(
    split: str,
    source: str | Path = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Load and validate one MELD metadata split."""
    if split not in SPLIT_FILENAMES:
        valid_splits = ", ".join(SPLIT_FILENAMES)
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: {valid_splits}"
        )

    filename = SPLIT_FILENAMES[split]

    if str(source).startswith(("http://", "https://")):
        location = f"{str(source).rstrip('/')}/{filename}"
    else:
        location = Path(source) / filename

    dataframe = pd.read_csv(location)
    validate_meld_frame(dataframe)

    return dataframe


def load_all_splits(
    source: str | Path = DEFAULT_BASE_URL,
) -> dict[str, pd.DataFrame]:
    """Load and validate the train, development, and test splits."""
    return {
        split: load_split(split, source)
        for split in SPLIT_FILENAMES
    }

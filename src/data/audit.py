"""Audit the MELD metadata splits and save reproducible statistics."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data.loader import load_all_splits


CLIP_KEY_COLUMNS = [
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
]


def normalize_text(value: object) -> str:
    """Normalize text before constructing content fingerprints."""
    return " ".join(str(value).strip().lower().split())


def get_clip_keys(dataframe: pd.DataFrame) -> set[tuple]:
    """Return exact source-video clip identifiers."""
    return set(
        dataframe[CLIP_KEY_COLUMNS]
        .astype(str)
        .itertuples(index=False, name=None)
    )


def build_dialogue_fingerprints(
    dataframe: pd.DataFrame,
) -> set[str]:
    """Create content-based fingerprints for complete dialogues.

    Dialogue_ID is only interpreted inside its own dataset split.
    The fingerprint instead uses the ordered utterance content,
    speakers, episode information, and clip timestamps.
    """
    fingerprints = set()

    for _, dialogue in dataframe.groupby(
        "Dialogue_ID",
        sort=False,
    ):
        ordered = dialogue.sort_values("Utterance_ID")

        rows = []

        for row in ordered.itertuples(index=False):
            rows.append(
                "|".join(
                    [
                        str(row.Season),
                        str(row.Episode),
                        str(row.StartTime),
                        str(row.EndTime),
                        normalize_text(row.Speaker),
                        normalize_text(row.Utterance),
                    ]
                )
            )

        payload = "\n".join(rows)
        fingerprint = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        fingerprints.add(fingerprint)

    return fingerprints


def summarize_split(dataframe: pd.DataFrame) -> dict:
    """Compute basic statistics for one MELD metadata split."""
    dialogue_lengths = dataframe.groupby(
        "Dialogue_ID"
    ).size()

    duplicate_records = dataframe.duplicated(
        subset=["Dialogue_ID", "Utterance_ID"]
    ).sum()

    emotion_counts = (
        dataframe["Emotion"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "rows": int(len(dataframe)),
        "dialogues": int(
            dataframe["Dialogue_ID"].nunique()
        ),
        "speakers": int(
            dataframe["Speaker"].nunique()
        ),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_records": int(
            duplicate_records
        ),
        "average_utterances_per_dialogue": float(
            dialogue_lengths.mean()
        ),
        "median_utterances_per_dialogue": float(
            dialogue_lengths.median()
        ),
        "maximum_utterances_per_dialogue": int(
            dialogue_lengths.max()
        ),
        "emotion_counts": {
            emotion: int(count)
            for emotion, count in emotion_counts.items()
        },
    }


def find_cross_split_overlaps(
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:
    """Find exact clip and complete-dialogue overlaps."""
    clip_keys = {
        name: get_clip_keys(dataframe)
        for name, dataframe in splits.items()
    }

    dialogue_fingerprints = {
        name: build_dialogue_fingerprints(dataframe)
        for name, dataframe in splits.items()
    }

    overlaps = {}

    for first, second in combinations(splits, 2):
        overlaps[f"{first}_vs_{second}"] = {
            "exact_clip_overlaps": len(
                clip_keys[first] & clip_keys[second]
            ),
            "exact_dialogue_overlaps": len(
                dialogue_fingerprints[first]
                & dialogue_fingerprints[second]
            ),
        }

    return overlaps


def build_audit_report(
    splits: dict[str, pd.DataFrame],
) -> dict:
    """Create the complete MELD metadata audit report."""
    return {
        "splits": {
            name: summarize_split(dataframe)
            for name, dataframe in splits.items()
        },
        "cross_split_overlap_counts":
            find_cross_split_overlaps(splits),
    }


def save_audit_outputs(
    splits: dict[str, pd.DataFrame],
    report: dict,
    output_dir: Path,
) -> None:
    """Save the JSON audit and emotion distribution table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        output_dir / "data_audit_summary.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    distribution = pd.DataFrame(
        {
            name: dataframe["Emotion"].value_counts()
            for name, dataframe in splits.items()
        }
    ).fillna(0).astype(int).sort_index()

    distribution.to_csv(
        output_dir / "emotion_distribution.csv"
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit the official MELD metadata splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_audit"),
        help="Directory used to save audit outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Load MELD, run the audit, and save the results."""
    args = parse_arguments()

    splits = load_all_splits()
    report = build_audit_report(splits)

    save_audit_outputs(
        splits=splits,
        report=report,
        output_dir=args.output_dir,
    )

    print(json.dumps(report, indent=2))
    print(f"\nAudit outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

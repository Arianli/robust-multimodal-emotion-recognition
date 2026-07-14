"""Analyze systematic errors made by the MELD text-only baseline."""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Dialogue_ID",
    "Utterance_ID",
    "Speaker",
    "Utterance",
    "Emotion",
    "Predicted_Emotion",
}


def validate_predictions_frame(dataframe: pd.DataFrame) -> None:
    """Validate the prediction table created by the text baseline."""
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required prediction columns: {missing}"
        )


def prepare_predictions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add correctness, word-count, and length-bucket features."""
    validate_predictions_frame(dataframe)

    prepared = dataframe.copy()

    prepared["Correct"] = (
        prepared["Emotion"]
        == prepared["Predicted_Emotion"]
    )

    prepared["Word_Count"] = (
        prepared["Utterance"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    prepared["Length_Bucket"] = pd.cut(
        prepared["Word_Count"],
        bins=[-1, 3, 7, 15, float("inf")],
        labels=[
            "1-3 words",
            "4-7 words",
            "8-15 words",
            "16+ words",
        ],
    )

    return prepared


def build_confusion_pairs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Count the most common incorrect true/predicted label pairs."""
    errors = dataframe[~dataframe["Correct"]]

    confusion_pairs = (
        errors.groupby(
            ["Emotion", "Predicted_Emotion"]
        )
        .size()
        .reset_index(name="Count")
        .sort_values(
            ["Count", "Emotion"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    confusion_pairs["Percent_of_All_Errors"] = (
        confusion_pairs["Count"]
        / max(len(errors), 1)
    )

    return confusion_pairs


def build_per_class_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute support, recall, and error rate for each emotion."""
    summary = (
        dataframe.groupby("Emotion")
        .agg(
            Support=("Emotion", "size"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    summary["Errors"] = (
        summary["Support"] - summary["Correct"]
    )

    summary["Recall"] = (
        summary["Correct"] / summary["Support"]
    )

    summary["Error_Rate"] = 1.0 - summary["Recall"]

    return summary.sort_values(
        "Recall",
        ascending=True,
    ).reset_index(drop=True)


def build_length_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Measure model performance across utterance-length buckets."""
    summary = (
        dataframe.groupby(
            "Length_Bucket",
            observed=False,
        )
        .agg(
            Samples=("Correct", "size"),
            Correct=("Correct", "sum"),
            Average_Word_Count=("Word_Count", "mean"),
        )
        .reset_index()
    )

    summary["Accuracy"] = (
        summary["Correct"] / summary["Samples"]
    )

    return summary


def sample_misclassified_examples(
    dataframe: pd.DataFrame,
    examples_per_pair: int = 3,
) -> pd.DataFrame:
    """Select deterministic examples for each confusion pair."""
    errors = dataframe[~dataframe["Correct"]].copy()

    samples = []

    grouped = errors.groupby(
        ["Emotion", "Predicted_Emotion"],
        sort=False,
    )

    for _, group in grouped:
        sample_size = min(
            examples_per_pair,
            len(group),
        )

        sampled = group.sample(
            n=sample_size,
            random_state=42,
        )

        samples.append(sampled)

    if not samples:
        return errors

    result = pd.concat(
        samples,
        ignore_index=True,
    )

    columns = [
        "Dialogue_ID",
        "Utterance_ID",
        "Speaker",
        "Utterance",
        "Emotion",
        "Predicted_Emotion",
        "Word_Count",
    ]

    return result[columns]


def build_summary(
    dataframe: pd.DataFrame,
    confusion_pairs: pd.DataFrame,
    per_class: pd.DataFrame,
) -> dict:
    """Build a compact JSON summary of the main findings."""
    total = len(dataframe)
    correct = int(dataframe["Correct"].sum())
    errors = total - correct

    return {
        "total_predictions": total,
        "correct_predictions": correct,
        "incorrect_predictions": errors,
        "accuracy": correct / max(total, 1),
        "top_confusion_pairs": (
            confusion_pairs.head(10).to_dict(
                orient="records"
            )
        ),
        "lowest_recall_classes": (
            per_class.head(3).to_dict(
                orient="records"
            )
        ),
    }


def run_error_analysis(
    predictions_path: Path,
    output_dir: Path,
) -> dict:
    """Run the complete baseline error analysis."""
    dataframe = pd.read_csv(predictions_path)
    prepared = prepare_predictions(dataframe)

    confusion_pairs = build_confusion_pairs(
        prepared
    )

    per_class = build_per_class_summary(
        prepared
    )

    length_summary = build_length_summary(
        prepared
    )

    examples = sample_misclassified_examples(
        prepared
    )

    summary = build_summary(
        dataframe=prepared,
        confusion_pairs=confusion_pairs,
        per_class=per_class,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_pairs.to_csv(
        output_dir / "confusion_pairs.csv",
        index=False,
    )

    per_class.to_csv(
        output_dir / "per_class_error_summary.csv",
        index=False,
    )

    length_summary.to_csv(
        output_dir / "length_bucket_performance.csv",
        index=False,
    )

    examples.to_csv(
        output_dir / "misclassified_examples.csv",
        index=False,
    )

    (
        output_dir / "error_analysis_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze errors from the MELD text-only baseline."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/text_baseline/test_predictions.csv"
        ),
        help="Path to the saved test prediction table.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/text_baseline/error_analysis"
        ),
        help="Directory used to save analysis outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run error analysis from the command line."""
    args = parse_arguments()

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.predictions}"
        )

    summary = run_error_analysis(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2))
    print(
        f"\nAnalysis saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

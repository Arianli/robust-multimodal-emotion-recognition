"""Train and evaluate a TF-IDF text-only baseline on MELD."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.data.loader import load_all_splits


def build_text_pipeline(
    min_df: int = 2,
    max_features: int = 30000,
) -> Pipeline:
    """Build the TF-IDF and logistic-regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def compute_metrics(
    true_labels,
    predicted_labels,
) -> dict[str, float]:
    """Compute the main classification metrics."""
    return {
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_split_outputs(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    labels: list[str],
    split_name: str,
    output_dir: Path,
) -> dict[str, float]:
    """Save metrics, reports, predictions, and confusion matrices."""
    true_labels = dataframe["Emotion"].to_numpy()

    metrics = compute_metrics(
        true_labels=true_labels,
        predicted_labels=predictions,
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=labels,
    )

    prediction_table = dataframe[
        [
            "Dialogue_ID",
            "Utterance_ID",
            "Speaker",
            "Utterance",
            "Emotion",
        ]
    ].copy()

    prediction_table["Predicted_Emotion"] = predictions
    prediction_table["Correct"] = (
        prediction_table["Emotion"]
        == prediction_table["Predicted_Emotion"]
    )

    prediction_table.to_csv(
        output_dir / f"{split_name}_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )

    (
        output_dir / f"{split_name}_classification_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    return metrics


def run_experiment(output_dir: Path) -> dict:
    """Train the model and evaluate it on development and test data."""
    splits = load_all_splits()

    train_frame = splits["train"]
    dev_frame = splits["dev"]
    test_frame = splits["test"]

    labels = sorted(
        train_frame["Emotion"].unique().tolist()
    )

    model = build_text_pipeline()

    model.fit(
        train_frame["Utterance"],
        train_frame["Emotion"],
    )

    dev_predictions = model.predict(
        dev_frame["Utterance"]
    )
    test_predictions = model.predict(
        test_frame["Utterance"]
    )

    majority_label = (
        train_frame["Emotion"]
        .value_counts()
        .idxmax()
    )

    majority_dev_predictions = np.full(
        len(dev_frame),
        majority_label,
        dtype=object,
    )
    majority_test_predictions = np.full(
        len(test_frame),
        majority_label,
        dtype=object,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_metrics = {
        "dev": save_split_outputs(
            dataframe=dev_frame,
            predictions=dev_predictions,
            labels=labels,
            split_name="dev",
            output_dir=output_dir,
        ),
        "test": save_split_outputs(
            dataframe=test_frame,
            predictions=test_predictions,
            labels=labels,
            split_name="test",
            output_dir=output_dir,
        ),
    }

    majority_metrics = {
        "label": majority_label,
        "dev": compute_metrics(
            dev_frame["Emotion"],
            majority_dev_predictions,
        ),
        "test": compute_metrics(
            test_frame["Emotion"],
            majority_test_predictions,
        ),
    }

    results = {
        "model": "tfidf_logistic_regression",
        "labels": labels,
        "tfidf_logistic_regression": model_metrics,
        "majority_baseline": majority_metrics,
    }

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        model,
        output_dir / "text_baseline.joblib",
    )

    return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the MELD text-only baseline."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/text_baseline"),
        help="Directory used to save experiment outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete text-only baseline experiment."""
    args = parse_arguments()

    results = run_experiment(
        output_dir=args.output_dir
    )

    print(json.dumps(results, indent=2))
    print(
        f"\nOutputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

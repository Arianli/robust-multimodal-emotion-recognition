"""Multi-seed evaluation of text robustness under word dropout."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.analysis.text_robustness import corrupt_utterances
from src.data.loader import load_split
from src.models.sentence_embedding_baseline import (
    DEFAULT_MODEL_NAME,
    encode_texts,
)
from src.models.text_baseline import compute_metrics


def summarize_runs(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate robustness runs across corruption seeds."""
    summary = (
        raw_results.groupby(
            [
                "word_dropout_rate",
                "remaining_word_fraction",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            macro_f1_drop_mean=("macro_f1_drop", "mean"),
            neutral_prediction_fraction_mean=(
                "neutral_prediction_fraction",
                "mean",
            ),
        )
        .fillna(0.0)
    )

    return summary


def run_multiseed_experiment(
    classifier_path: Path,
    output_dir: Path,
    model_name: str,
    dropout_rates: list[float],
    seeds: list[int],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate several random corruption masks."""
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found: {classifier_path}"
        )

    if 0.0 not in dropout_rates:
        raise ValueError(
            "dropout_rates must include 0.0"
        )

    test_frame = load_split("test")
    true_labels = test_frame["Emotion"]

    classifier = joblib.load(classifier_path)

    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    rows = []

    for seed in seeds:
        print(f"\nStarting corruption seed: {seed}")

        seed_rows = []

        for dropout_rate in dropout_rates:
            print(
                f"  Evaluating dropout: "
                f"{dropout_rate:.0%}"
            )

            corrupted_texts = corrupt_utterances(
                utterances=test_frame["Utterance"],
                dropout_rate=dropout_rate,
                seed=seed,
            )

            embeddings = encode_texts(
                encoder=encoder,
                texts=corrupted_texts,
                batch_size=batch_size,
            )

            predictions = classifier.predict(
                embeddings
            )

            metrics = compute_metrics(
                true_labels=true_labels,
                predicted_labels=predictions,
            )

            prediction_distribution = (
                pd.Series(predictions)
                .value_counts(normalize=True)
            )

            seed_rows.append(
                {
                    "seed": seed,
                    "word_dropout_rate": dropout_rate,
                    "remaining_word_fraction":
                        1.0 - dropout_rate,
                    **metrics,
                    "neutral_prediction_fraction": float(
                        prediction_distribution.get(
                            "neutral",
                            0.0,
                        )
                    ),
                }
            )

        clean_macro_f1 = next(
            row["macro_f1"]
            for row in seed_rows
            if row["word_dropout_rate"] == 0.0
        )

        clean_accuracy = next(
            row["accuracy"]
            for row in seed_rows
            if row["word_dropout_rate"] == 0.0
        )

        for row in seed_rows:
            row["accuracy_drop"] = (
                clean_accuracy - row["accuracy"]
            )
            row["macro_f1_drop"] = (
                clean_macro_f1 - row["macro_f1"]
            )

        rows.extend(seed_rows)

    raw_results = pd.DataFrame(rows)
    summary = summarize_runs(raw_results)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_results.to_csv(
        output_dir / "multiseed_raw_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "multiseed_summary.csv",
        index=False,
    )

    json_output = {
        "model": model_name,
        "classifier": str(classifier_path),
        "seeds": seeds,
        "dropout_rates": dropout_rates,
        "summary": json.loads(
            summary.to_json(orient="records")
        ),
        "raw_results": json.loads(
            raw_results.to_json(orient="records")
        ),
    }

    (
        output_dir / "multiseed_results.json"
    ).write_text(
        json.dumps(json_output, indent=2),
        encoding="utf-8",
    )

    return raw_results, summary


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed MELD text corruption evaluation."
        )
    )

    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=Path(
            "results/sentence_embedding_baseline/"
            "embedding_classifier.joblib"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/robustness/"
            "text_word_dropout_multiseed"
        ),
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.1,
            0.3,
            0.5,
            0.7,
            1.0,
        ],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            13,
            42,
            73,
            101,
            2026,
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete multi-seed experiment."""
    args = parse_arguments()

    _, summary = run_multiseed_experiment(
        classifier_path=args.classifier_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        dropout_rates=args.dropout_rates,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )

    print("\nMulti-seed robustness summary:")
    print(summary.to_string(index=False))
    print(
        f"\nOutputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

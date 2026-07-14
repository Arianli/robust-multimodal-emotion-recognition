"""Evaluate sentence-embedding robustness under word dropout."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data.loader import load_split
from src.models.sentence_embedding_baseline import (
    DEFAULT_MODEL_NAME,
    encode_texts,
)
from src.models.text_baseline import compute_metrics


def apply_word_dropout(
    text: str,
    dropout_rate: float,
    seed: int,
) -> str:
    """Randomly remove words from one utterance."""
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError(
            "dropout_rate must be between 0 and 1"
        )

    tokens = str(text).split()

    if not tokens:
        return ""

    if dropout_rate == 0.0:
        return " ".join(tokens)

    if dropout_rate == 1.0:
        return ""

    generator = np.random.default_rng(seed)

    keep_mask = (
        generator.random(len(tokens))
        >= dropout_rate
    )

    if not keep_mask.any():
        keep_mask[
            generator.integers(0, len(tokens))
        ] = True

    return " ".join(
        token
        for token, keep in zip(tokens, keep_mask)
        if keep
    )


def corrupt_utterances(
    utterances: pd.Series,
    dropout_rate: float,
    seed: int,
) -> list[str]:
    """Apply deterministic word dropout to a full split."""
    return [
        apply_word_dropout(
            text=text,
            dropout_rate=dropout_rate,
            seed=seed + index,
        )
        for index, text in enumerate(
            utterances.fillna("")
        )
    ]


def run_robustness_experiment(
    classifier_path: Path,
    output_dir: Path,
    model_name: str,
    dropout_rates: list[float],
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate the classifier at several corruption levels."""
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found: {classifier_path}"
        )

    if 0.0 not in dropout_rates:
        raise ValueError(
            "dropout_rates must include 0.0 as the clean baseline"
        )

    test_frame = load_split("test")

    classifier = joblib.load(classifier_path)

    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    rows = []

    for dropout_rate in dropout_rates:
        print(
            f"\nEvaluating word dropout rate: "
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
            true_labels=test_frame["Emotion"],
            predicted_labels=predictions,
        )

        rows.append(
            {
                "word_dropout_rate": dropout_rate,
                "remaining_word_fraction":
                    1.0 - dropout_rate,
                **metrics,
            }
        )

    results = pd.DataFrame(rows)

    clean_row = results[
        results["word_dropout_rate"] == 0.0
    ].iloc[0]

    results["accuracy_drop"] = (
        clean_row["accuracy"]
        - results["accuracy"]
    )

    results["macro_f1_drop"] = (
        clean_row["macro_f1"]
        - results["macro_f1"]
    )

    results["weighted_f1_drop"] = (
        clean_row["weighted_f1"]
        - results["weighted_f1"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir / "word_dropout_results.csv",
        index=False,
    )

    summary = {
        "model": model_name,
        "classifier": str(classifier_path),
        "seed": seed,
        "results": results.to_dict(
            orient="records"
        ),
    }

    (
        output_dir / "word_dropout_results.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate text-model robustness under word dropout."
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
            "results/robustness/text_word_dropout"
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
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    """Run the word-dropout robustness experiment."""
    args = parse_arguments()

    results = run_robustness_experiment(
        classifier_path=args.classifier_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        dropout_rates=args.dropout_rates,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("\nRobustness results:")
    print(results.to_string(index=False))
    print(
        f"\nOutputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

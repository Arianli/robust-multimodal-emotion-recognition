"""Frozen sentence-embedding baseline for MELD emotion recognition."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.data.loader import load_all_splits
from src.models.text_baseline import save_split_outputs


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode_texts(
    encoder,
    texts: Sequence[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts into normalized semantic vectors."""
    embeddings = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def build_classifier() -> LogisticRegression:
    """Build the classifier placed on top of frozen embeddings."""
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )


def run_embedding_experiment(
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
) -> dict:
    """Train and evaluate the sentence-embedding baseline."""
    splits = load_all_splits()

    train_frame = splits["train"]
    dev_frame = splits["dev"]
    test_frame = splits["test"]

    labels = sorted(
        train_frame["Emotion"].unique().tolist()
    )

    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    print("Encoding training utterances...")
    train_embeddings = encode_texts(
        encoder,
        train_frame["Utterance"].fillna(""),
        batch_size,
    )

    print("Encoding development utterances...")
    dev_embeddings = encode_texts(
        encoder,
        dev_frame["Utterance"].fillna(""),
        batch_size,
    )

    print("Encoding test utterances...")
    test_embeddings = encode_texts(
        encoder,
        test_frame["Utterance"].fillna(""),
        batch_size,
    )

    print("Training logistic-regression classifier...")
    classifier = build_classifier()

    classifier.fit(
        train_embeddings,
        train_frame["Emotion"],
    )

    dev_predictions = classifier.predict(
        dev_embeddings
    )

    test_predictions = classifier.predict(
        test_embeddings
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    dev_metrics = save_split_outputs(
        dataframe=dev_frame,
        predictions=dev_predictions,
        labels=labels,
        split_name="dev",
        output_dir=output_dir,
    )

    test_metrics = save_split_outputs(
        dataframe=test_frame,
        predictions=test_predictions,
        labels=labels,
        split_name="test",
        output_dir=output_dir,
    )

    results = {
        "model": "frozen_sentence_embedding_logistic_regression",
        "encoder": model_name,
        "embedding_dimension": int(
            train_embeddings.shape[1]
        ),
        "batch_size": batch_size,
        "train_samples": int(
            train_embeddings.shape[0]
        ),
        "dev": dev_metrics,
        "test": test_metrics,
    }

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        classifier,
        output_dir / "embedding_classifier.joblib",
    )

    return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen sentence-embedding baseline on MELD."
        )
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/sentence_embedding_baseline"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete experiment."""
    args = parse_arguments()

    results = run_embedding_experiment(
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    print(json.dumps(results, indent=2))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

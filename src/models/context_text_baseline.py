"""Context-aware TF-IDF baseline for MELD emotion recognition."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import load_all_splits
from src.models.text_baseline import (
    build_text_pipeline,
    save_split_outputs,
)


def add_dialogue_context(
    dataframe: pd.DataFrame,
    context_window: int = 3,
) -> pd.DataFrame:
    """Add preceding dialogue turns to each utterance.

    Args:
        dataframe: A MELD metadata split.
        context_window: Maximum number of previous utterances to include.

    Returns:
        A copy of the dataframe with a Context_Text column.
    """
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    prepared_groups = []

    for _, dialogue in dataframe.groupby(
        "Dialogue_ID",
        sort=False,
    ):
        ordered = dialogue.sort_values(
            "Utterance_ID"
        ).copy()

        utterances = ordered["Utterance"].fillna("").astype(str)
        speakers = ordered["Speaker"].fillna("Unknown").astype(str)

        context_texts = []

        for position in range(len(ordered)):
            start = max(
                0,
                position - context_window,
            )

            previous_turns = []

            for previous_position in range(
                start,
                position,
            ):
                previous_turns.append(
                    (
                        f"[PREVIOUS SPEAKER={speakers.iloc[previous_position]}] "
                        f"{utterances.iloc[previous_position]}"
                    )
                )

            current_turn = (
                f"[CURRENT SPEAKER={speakers.iloc[position]}] "
                f"{utterances.iloc[position]}"
            )

            if previous_turns:
                combined = (
                    " ".join(previous_turns)
                    + " "
                    + current_turn
                )
            else:
                combined = current_turn

            context_texts.append(combined)

        ordered["Context_Text"] = context_texts
        prepared_groups.append(ordered)

    return pd.concat(
        prepared_groups,
        ignore_index=True,
    )


def run_context_experiment(
    output_dir: Path,
    context_window: int,
) -> dict:
    """Train and evaluate the context-aware baseline."""
    splits = load_all_splits()

    train_frame = add_dialogue_context(
        splits["train"],
        context_window=context_window,
    )

    dev_frame = add_dialogue_context(
        splits["dev"],
        context_window=context_window,
    )

    test_frame = add_dialogue_context(
        splits["test"],
        context_window=context_window,
    )

    labels = sorted(
        train_frame["Emotion"].unique().tolist()
    )

    model = build_text_pipeline()

    model.fit(
        train_frame["Context_Text"],
        train_frame["Emotion"],
    )

    dev_predictions = model.predict(
        dev_frame["Context_Text"]
    )

    test_predictions = model.predict(
        test_frame["Context_Text"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "model": "context_tfidf_logistic_regression",
        "context_window": context_window,
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

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        model,
        output_dir / "context_text_baseline.joblib",
    )

    return metrics


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a context-aware MELD text baseline."
        )
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of preceding utterances to include.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/context_text_baseline"
        ),
        help="Directory used to save experiment outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the context-aware experiment."""
    args = parse_arguments()

    results = run_context_experiment(
        output_dir=args.output_dir,
        context_window=args.context_window,
    )

    print(json.dumps(results, indent=2))
    print(
        f"\nOutputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

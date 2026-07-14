"""Compare utterance-only and context-aware text baselines."""

import json
from pathlib import Path

import pandas as pd


BASELINE_PATH = Path(
    "results/text_baseline/metrics.json"
)

CONTEXT_PATH = Path(
    "results/context_text_baseline/metrics.json"
)

OUTPUT_PATH = Path(
    "results/text_model_comparison.csv"
)


def main() -> None:
    """Load both experiment files and create a comparison table."""
    baseline = json.loads(
        BASELINE_PATH.read_text(
            encoding="utf-8"
        )
    )

    context = json.loads(
        CONTEXT_PATH.read_text(
            encoding="utf-8"
        )
    )

    baseline_test = baseline[
        "tfidf_logistic_regression"
    ]["test"]

    context_test = context["test"]

    rows = [
        {
            "Model": "Utterance-only TF-IDF",
            "Accuracy": baseline_test["accuracy"],
            "Macro_F1": baseline_test["macro_f1"],
            "Weighted_F1": baseline_test["weighted_f1"],
        },
        {
            "Model": "Context TF-IDF (3 turns)",
            "Accuracy": context_test["accuracy"],
            "Macro_F1": context_test["macro_f1"],
            "Weighted_F1": context_test["weighted_f1"],
        },
    ]

    table = pd.DataFrame(rows)

    baseline_macro_f1 = rows[0]["Macro_F1"]
    context_macro_f1 = rows[1]["Macro_F1"]

    table["Macro_F1_Change_vs_Baseline"] = [
        0.0,
        context_macro_f1 - baseline_macro_f1,
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

"""Generate Phase 4 evaluation plots and the README comparison table.

Reads results/unified_predictions.parquet (schema: src/evaluation/metrics.py) and
produces, per system, a confusion matrix heatmap plus one combined precision-recall
curve overlay, then prints the comparison_table() as markdown for the README.

Systems: baseline_nli, track_a_deberta, approach_1_modernbert, track_b_modernbert.

Usage:
    python scripts/generate_evaluation_plots.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.evaluation.metrics import (  # noqa: E402
    comparison_table,
    load_predictions,
    metrics_for_system,
    system_predictions,
)

SYSTEMS = ["baseline_nli", "track_a_deberta", "approach_1_modernbert", "track_b_modernbert"]
LABELS = ["faithful", "hallucinated"]
RESULTS_DIR = REPO_ROOT / "results"

# Fixed categorical order (not cycled) per system, paired with a distinct line style
# so identity survives on top of color for colorblind readers.
SYSTEM_STYLE = {
    "baseline_nli": {"color": "#2a78d6", "linestyle": "-"},
    "track_a_deberta": {"color": "#1baf7a", "linestyle": "--"},
    "approach_1_modernbert": {"color": "#eda100", "linestyle": "-."},
    "track_b_modernbert": {"color": "#008300", "linestyle": ":"},
}

BASELINE_SCORE_CAVEAT = (
    "Note: baseline_nli's y_score is NOT a calibrated probability like the other three\n"
    "systems' softmax outputs (per ADR-015) -- it's a one-dimensional reduction of a\n"
    "coupled two-threshold decision rule. Its PR curve shape is still meaningful (higher\n"
    "score = more hallucination-like), but its threshold values are not directly\n"
    "comparable to the other systems' threshold values."
)


def plot_confusion_matrices(df) -> None:
    for system in SYSTEMS:
        cm = metrics_for_system(df, system)["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(system)
        out_path = RESULTS_DIR / f"confusion_matrix_{system}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_pr_curve_comparison(df) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for system in SYSTEMS:
        subset = system_predictions(df, system)
        precision, recall, _ = precision_recall_curve(
            subset["y_true"].to_numpy(), subset["y_score"].to_numpy(), pos_label=1
        )
        style = SYSTEM_STYLE[system]
        ax.plot(recall, precision, label=system, linewidth=2, **style)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Comparison Across Systems")
    ax.legend(loc="lower left")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    fig.text(
        0.5,
        -0.05,
        "baseline_nli's y_score is not a calibrated probability like the other systems'\n"
        "softmax outputs (ADR-015) -- curve shape is meaningful, thresholds are not comparable.",
        ha="center",
        va="top",
        fontsize=8,
        wrap=True,
    )

    out_path = RESULTS_DIR / "pr_curve_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print()
    print(BASELINE_SCORE_CAVEAT)


def print_markdown_table(df) -> None:
    table = comparison_table(df, systems=SYSTEMS)
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines = [header, separator]
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            value = row[column]
            if column in ("precision", "recall", "f1", "accuracy"):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    print()
    print("=== Comparison table (markdown) ===")
    print("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()

    plot_confusion_matrices(df)
    plot_pr_curve_comparison(df)
    print_markdown_table(df)


if __name__ == "__main__":
    main()

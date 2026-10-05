"""Write evaluation outputs under ``reports/ml/<dataset>/<task>/<model>/`` and build ``reports/ml/summary.csv``.

Only metrics, tables and plots are written here. Trained model files never go under ``reports/``.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)
MAX_ROC_POINTS = 2000
MAX_ANNOTATED_CLASSES = 20


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _plot_confusion(cm_counts: np.ndarray, cm_norm: np.ndarray, labels: list[str], title: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping %s", path.name)
        return False
    n = len(labels)
    size = max(5.0, 0.55 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if n <= MAX_ANNOTATED_CLASSES:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm_counts[i, j]:,}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float | None, title: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label="ROC-AUC = n/a" if auc is None else f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_pr(precision: np.ndarray, recall: np.ndarray, ap: float | None, title: str, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 4.5))
    label = "Average precision = n/a" if ap is None else f"Average precision = {ap:.4f}"
    ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_report(report_dir: Path, evaluation: dict[str, Any], y_true: np.ndarray, proba: np.ndarray, title: str) -> list[str]:
    """Write all evaluation files for one trained model and return their names."""
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluation["metrics"]
    labels = metrics["confusion_matrix"]["labels"]
    cm = np.array(metrics["confusion_matrix"]["counts"])
    cm_norm = np.array(metrics["confusion_matrix"]["row_normalized"])

    _json(report_dir / "metrics.json", metrics)
    _json(report_dir / "classification_report.json", metrics["classification_report"])
    (report_dir / "classification_report.txt").write_text(evaluation["classification_report_text"])
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true\\predicted").to_csv(report_dir / "confusion_matrix.csv")
    pd.DataFrame(cm_norm, index=labels, columns=labels).rename_axis("true\\predicted").to_csv(report_dir / "confusion_matrix_normalized.csv")
    written = ["metrics.json", "classification_report.json", "classification_report.txt", "confusion_matrix.csv", "confusion_matrix_normalized.csv"]

    if _plot_confusion(cm, cm_norm, labels, f"{title} - confusion matrix", report_dir / "confusion_matrix.png"):
        written.append("confusion_matrix.png")

    if len(labels) == 2 and len(np.unique(y_true)) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, proba[:, 1])
        if len(fpr) > MAX_ROC_POINTS:
            keep = np.unique(np.linspace(0, len(fpr) - 1, MAX_ROC_POINTS).astype(int))
            fpr, tpr, thresholds = fpr[keep], tpr[keep], thresholds[keep]
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(report_dir / "roc_curve.csv", index=False)
        written.append("roc_curve.csv")
        if _plot_roc(fpr, tpr, metrics["roc_auc"].get("binary"), f"{title} - ROC", report_dir / "roc_curve.png"):
            written.append("roc_curve.png")

        # Precision-recall is especially useful for imbalanced binary intrusion datasets.
        precision, recall, pr_thresholds = precision_recall_curve(y_true, proba[:, 1])
        if len(precision) > MAX_ROC_POINTS:
            keep = np.unique(np.linspace(0, len(precision) - 1, MAX_ROC_POINTS).astype(int))
            precision, recall = precision[keep], recall[keep]
            # Thresholds has one fewer element than precision/recall.
            pr_thresholds = np.array([])
        pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": np.pad(
                pr_thresholds.astype(float),
                (0, max(0, len(precision) - len(pr_thresholds))),
                constant_values=np.nan,
            )[:len(precision)],
        }).to_csv(report_dir / "precision_recall_curve.csv", index=False)
        written.append("precision_recall_curve.csv")
        if _plot_pr(precision, recall, metrics["roc_auc"].get("average_precision"),
                    f"{title} - Precision-Recall", report_dir / "precision_recall_curve.png"):
            written.append("precision_recall_curve.png")
    return written


SUMMARY_FIELDS = ["dataset", "task", "model", "average", "accuracy", "precision", "recall", "f1_score", "roc_auc", "n_test", "created_at_utc"]


def update_summary(reports_root: Path) -> Path:
    """Rebuild ``summary.csv`` from every ``<dataset>/<task>/<model>/metrics.json`` found (values come from those files)."""
    rows = []
    for metrics_path in sorted(Path(reports_root).glob("*/*/*/metrics.json")):
        dataset, task, model = metrics_path.parent.parts[-3:]
        m = json.loads(metrics_path.read_text())
        meta_path = metrics_path.parent / "run_info.json"
        created = json.loads(meta_path.read_text()).get("created_at_utc", "") if meta_path.exists() else ""
        auc = m["roc_auc"]
        rows.append({
            "dataset": dataset, "task": task, "model": model, "average": m["average"], "accuracy": m["accuracy"],
            "precision": m["precision"], "recall": m["recall"], "f1_score": m["f1_score"],
            "roc_auc": auc.get("binary") if "binary" in auc else auc.get("macro_ovr"),
            "n_test": m["n_samples"], "created_at_utc": created,
        })
    path = Path(reports_root) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path

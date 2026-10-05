"""Evaluation metrics, always computed from predictions (nothing here is a fixed number).

Binary:      precision/recall/F1 are for the positive (attack) class; ROC-AUC and PR-AUC use P(attack).
Multiclass:  precision/recall/F1 are macro-averaged (weighted averages are also reported); ROC-AUC is one-vs-rest
             per class, macro- and support-weighted averaged over the classes that have both positives and
             negatives in the evaluated data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


def predict_labels(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1)


def _prf(y_true, y_pred, average: str, labels: list[int]) -> dict[str, float]:
    kw = {"labels": labels, "average": average, "zero_division": 0}
    if average == "binary":
        kw = {"pos_label": 1, "average": "binary", "zero_division": 0}
    return {
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1_score": float(f1_score(y_true, y_pred, **kw)),
    }


def roc_auc_summary(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """ROC-AUC where it is defined; otherwise ``None`` with the reason (never a made-up value)."""
    n_classes = len(class_names)
    if n_classes == 2:
        if len(np.unique(y_true)) < 2:
            return {"binary": None, "note": "only one class present in the evaluated data; ROC-AUC undefined"}
        return {"binary": float(roc_auc_score(y_true, proba[:, 1])), "average_precision": float(average_precision_score(y_true, proba[:, 1]))}
    per_class: dict[str, float | None] = {}
    support, aucs = [], []
    for c, name in enumerate(class_names):
        positives = y_true == c
        if positives.sum() == 0 or positives.all():
            per_class[name] = None
            continue
        auc = float(roc_auc_score(positives, proba[:, c]))
        per_class[name] = auc
        aucs.append(auc)
        support.append(int(positives.sum()))
    skipped = [n for n, v in per_class.items() if v is None]
    return {
        "macro_ovr": float(np.mean(aucs)) if aucs else None,
        "weighted_ovr": float(np.average(aucs, weights=support)) if aucs else None,
        "per_class": per_class,
        "classes_skipped_no_positives_or_negatives": skipped,
    }


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Full metric set for one evaluated split. Returns JSON-serialisable data plus the text report."""
    y_true = np.asarray(y_true)
    y_pred = predict_labels(proba)
    labels = list(range(len(class_names)))
    binary = len(class_names) == 2
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    primary_avg = "binary" if binary else "macro"
    primary = _prf(y_true, y_pred, primary_avg, labels)
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "average": primary_avg,
        "positive_class": class_names[1] if binary else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        **primary,
        "macro": _prf(y_true, y_pred, "macro", labels),
        "weighted": _prf(y_true, y_pred, "weighted", labels),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": roc_auc_summary(y_true, proba, class_names),
        "confusion_matrix": {"labels": class_names, "counts": cm.tolist(), "row_normalized": cm_norm.tolist()},
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
        "support": {name: int((y_true == i).sum()) for i, name in enumerate(class_names)},
    }
    text = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0)
    return {"metrics": metrics, "classification_report_text": text}


def headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact view used for logs, validation summaries and the summary table."""
    auc = metrics["roc_auc"]
    return {
        "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"],
        "f1_score": metrics["f1_score"], "average": metrics["average"],
        "roc_auc": auc.get("binary") if "binary" in auc else auc.get("macro_ovr"),
    }

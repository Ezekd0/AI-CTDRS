"""Saving and (server-side) loading of trained-model bundles.

Layout:  ml/artifacts/<dataset>/<task>/<model>/
    model.joblib | model.json | model.pt     trained model
    preprocessor.joblib, preprocessing_report.json   copied from the dataset's preprocessing step
    feature_names.json, label_encoder.joblib, label_mapping.json
    metadata.json                            training metadata (params, seed, data hashes, versions, history)
    metrics.json                             test-set evaluation (same file as under reports/ml/)

SECURITY: model files are pickle/joblib/torch files and must never be served to a browser. Frontend-facing code
may only use ``public_summary`` / ``list_public_summaries`` (metrics and descriptive metadata, no file paths);
``load_bundle`` is for server-side inference code and refuses paths outside the artifacts root.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from ml.training.models import load_model

SCHEMA_VERSION = 1
PUBLIC_METADATA_FIELDS = ("run_id", "created_at_utc", "dataset", "task", "model", "seed", "n_features", "class_names", "split")


def _json_default(obj: Any) -> Any:
    """JSON encoder for numpy scalars/arrays that library history objects may contain."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def run_dir(root: Path, dataset_slug: str, task: str, model: str) -> Path:
    return Path(root) / dataset_slug / task / model


def library_versions(model: str) -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    if model == "xgboost":
        import xgboost

        versions["xgboost"] = xgboost.__version__
    if model == "lstm":
        import torch

        versions["torch"] = torch.__version__
    return versions


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_bundle(
    directory: Path,
    model,
    feature_names: list[str],
    label_encoder: LabelEncoder,
    class_names: list[str],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    preprocessing_files: dict[str, Path],
) -> dict[str, str]:
    """Write one model bundle; returns {artifact: file name}."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = model.save(directory)
    (directory / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    joblib.dump(label_encoder, directory / "label_encoder.joblib")
    (directory / "label_mapping.json").write_text(json.dumps({str(i): n for i, n in enumerate(class_names)}, indent=2))
    copied = {}
    for key, src in preprocessing_files.items():
        if key in ("preprocessor", "preprocessing_report"):
            shutil.copy2(src, directory / src.name)
            copied[key] = src.name
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    files = {"model": model_path.name, "feature_names": "feature_names.json", "label_encoder": "label_encoder.joblib",
             "label_mapping": "label_mapping.json", "metrics": "metrics.json", "metadata": "metadata.json", **copied}
    (directory / "metadata.json").write_text(json.dumps({**metadata, "artifact_files": files}, indent=2, default=_json_default))
    return files


# ----------------------------------------------------------------------------
# Safe access
# ----------------------------------------------------------------------------
def _inside(root: Path, path: Path) -> Path:
    resolved, root = path.resolve(), root.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("path is outside the artifacts root")
    return resolved


def public_summary(directory: Path) -> dict[str, Any]:
    """Frontend-safe description of a trained model: descriptive metadata and test metrics only."""
    meta = json.loads((directory / "metadata.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    out = {k: meta.get(k) for k in PUBLIC_METADATA_FIELDS}
    auc = metrics["roc_auc"]
    out["metrics"] = {
        "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"],
        "f1_score": metrics["f1_score"], "average": metrics["average"],
        "roc_auc": auc.get("binary") if "binary" in auc else auc.get("macro_ovr"),
        "confusion_matrix": metrics["confusion_matrix"],
    }
    return out


def list_public_summaries(artifacts_root: Path) -> list[dict[str, Any]]:
    root = Path(artifacts_root)
    return [public_summary(p.parent) for p in sorted(root.glob("*/*/*/metadata.json")) if (p.parent / "metrics.json").exists()]


@dataclass
class LoadedBundle:
    model: Any
    feature_names: list[str]
    label_encoder: LabelEncoder
    class_names: list[str]
    metadata: dict[str, Any]
    lime_background: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


def load_bundle(artifacts_root: Path, dataset_slug: str, task: str, model_name: str) -> LoadedBundle:
    """Server-side only. Accept either a public dataset key or its artifact slug.

    The training registry intentionally exposes ``cicids2018`` while storing artifacts
    under ``cse_cicids2018``. Resolving that alias here keeps every server consumer on
    the same artifact-selection contract.
    """
    root = Path(artifacts_root)
    try:
        from ml.training.datasets import resolve_dataset
        artifact_slug = resolve_dataset(dataset_slug).slug
    except ValueError:
        artifact_slug = dataset_slug
    directory = _inside(root, run_dir(root, artifact_slug, task, model_name))
    meta = json.loads((directory / "metadata.json").read_text())
    class_names = meta["class_names"]
    model = load_model(meta["model"], directory / meta["artifact_files"]["model"], len(class_names), meta["n_features"])
    background_path = directory / 'lime_background.npy'
    background = np.load(background_path, allow_pickle=False, mmap_mode="r") if background_path.exists() else None
    return LoadedBundle(
        model=model,
        feature_names=json.loads((directory / "feature_names.json").read_text()),
        label_encoder=joblib.load(directory / "label_encoder.joblib"),
        class_names=class_names,
        metadata=meta,
        lime_background=background,
    )


def command_line() -> str:
    return " ".join([Path(sys.executable).name, "-m", "ml.training.train", *sys.argv[1:]])

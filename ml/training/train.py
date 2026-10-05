"""Training command for the AI-CTDRS detection models.

    python -m ml.training.train --dataset cicids2017 --model xgboost
    python -m ml.training.train --dataset nsl-kdd --model random_forest
    python -m ml.training.train --dataset cicids2018 --model lstm --task binary
    python -m ml.training.train --dataset all --model all

Datasets: cicids2017 | cicids2018 | nsl-kdd | all        Models: random_forest | xgboost | lstm | all
Tasks:    binary | multiclass | both (default)

Every metric is computed from the model's predictions on the held-out test split at run time. Outputs:
    ml/artifacts/<dataset>/<task>/<model>/   model, preprocessing artifacts, feature names, label encoder, metadata
    reports/ml/<dataset>/<task>/<model>/     metrics, classification report, confusion matrix, ROC, run_info.json
    reports/ml/summary.csv                   one row per trained model found under reports/ml/
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation.metrics import evaluate_predictions, headline
from ml.explainability.shap_explainer import explain_global, write_json
from ml.evaluation.reporting import update_summary, write_report
from ml.training.artifacts import command_line, git_commit, library_versions, run_dir, save_bundle, utc_now
from ml.training.datasets import (
    DATASETS, TASKS, build_task_data, class_weights, ensure_preprocessed, load_dataset, resolve_dataset, sha256_file,
)
from ml.training.models import DEFAULTS, MODEL_NAMES, build_model, merge_params, resolve_model

logger = logging.getLogger("ml.training")
CLASS_WEIGHT_MODES = ("balanced", "sqrt", "none")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_param(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--param expects KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # plain string, e.g. device=cpu
    return key.strip(), value


def train_one(
    dataset: str,
    model_name: str,
    task: str,
    *,
    processed_root: Path,
    artifacts_root: Path,
    reports_root: Path,
    seed: int = 42,
    val_size: float = 0.1,
    max_train_rows: int | None = None,
    class_weight_mode: str = "balanced",
    param_overrides: dict[str, Any] | None = None,
    preprocess_if_missing: bool = False,
    raw_dir: str | None = None,
    _loaded=None,
) -> dict[str, Any]:
    """Train and evaluate one (dataset, model, task); write artifacts and reports; return a short result."""
    spec, model_name = resolve_dataset(dataset), resolve_model(model_name)
    params = merge_params(model_name, param_overrides)
    started = time.perf_counter()

    data = _loaded
    if data is None:
        if preprocess_if_missing and not (processed_root / spec.slug / "train.csv.gz").exists():
            ensure_preprocessed(spec, processed_root, artifacts_root, raw_dir)
        data = load_dataset(spec, processed_root, artifacts_root)
    td = build_task_data(data, task, val_size=val_size, seed=seed, max_train_rows=max_train_rows)

    seed_everything(seed)
    weights = class_weights(td.y_train, td.n_classes, class_weight_mode)
    sample_weight = None if weights is None else weights[td.y_train]
    model = build_model(model_name, td.n_classes, len(td.feature_names), seed, params)

    logger.info("[%s | %s | %s] training on %d rows (%d features, %d classes), validation %d rows",
                spec.display_name, task, model_name, len(td.y_train), len(td.feature_names), td.n_classes, len(td.y_val))
    fit_start = time.perf_counter()
    fit_info = model.fit(td.X_train, td.y_train, td.X_val, td.y_val, sample_weight=sample_weight, class_weights=weights)
    fit_seconds = time.perf_counter() - fit_start

    validation = None
    if td.has_validation:
        validation = headline(evaluate_predictions(td.y_val, model.predict_proba(td.X_val), td.class_names)["metrics"])
    test_proba = model.predict_proba(td.X_test)
    evaluation = evaluate_predictions(td.y_test, test_proba, td.class_names)
    metrics = evaluation["metrics"]

    run_id, created = uuid.uuid4().hex, utc_now()
    metadata = {
        "schema_version": 1, "run_id": run_id, "created_at_utc": created, "command": command_line(),
        "dataset": spec.key, "dataset_display_name": spec.display_name, "task": task, "model": model_name,
        "model_params": params, "seed": seed, "class_weight_mode": class_weight_mode,
        "n_features": len(td.feature_names), "class_names": td.class_names,
        "split": td.info,
        "data_files": {k: {"name": p.name, "sha256": sha256_file(p)} for k, p in data.files.items()},
        "training": {"fit_seconds": round(fit_seconds, 3), "total_seconds": round(time.perf_counter() - started, 3), **fit_info},
        "validation_metrics": validation,
        "libraries": library_versions(model_name),
        "git_commit": git_commit(),
    }
    artifact_dir = run_dir(artifacts_root, spec.slug, task, model_name)
    report_dir = run_dir(reports_root, spec.slug, task, model_name)
    files = save_bundle(artifact_dir, model, td.feature_names, td.label_encoder, td.class_names, metadata, metrics, data.files)
    # LIME needs a representative training background; never fall back to the single explained row.
    if model_name in {'random_forest', 'xgboost'}:
        background_size = min(500, len(td.X_train))
        rng = np.random.default_rng(seed)
        background_idx = rng.choice(len(td.X_train), size=background_size, replace=False)
        np.save(artifact_dir / 'lime_background.npy', td.X_train[background_idx].astype(np.float32), allow_pickle=False)
        metadata['lime_background_samples'] = int(background_size)
        metadata_path = artifact_dir / 'metadata.json'
        metadata_payload = json.loads(metadata_path.read_text())
        metadata_payload['lime_background_samples'] = int(background_size)
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x))
    # SHAP is generated only for supported tree models and always from real held-out test rows.
    if model_name in {"random_forest", "xgboost"}:
        shap_global = explain_global(model, model_name, td.X_test, td.feature_names, td.class_names)
        write_json(artifact_dir / "shap_global.json", shap_global)
        files["shap_global"] = "shap_global.json"
    report_files = write_report(report_dir, evaluation, td.y_test, test_proba, f"{spec.display_name} / {task} / {model_name}")
    (report_dir / "run_info.json").write_text(json.dumps(
        {"run_id": run_id, "created_at_utc": created, "dataset": spec.key, "task": task, "model": model_name, "seed": seed,
         "n_train": td.info["n_train"], "n_val": td.info["n_val"], "n_test": td.info["n_test"]}, indent=2))
    update_summary(reports_root)

    result = {"dataset": spec.key, "task": task, "model": model_name, **headline(metrics),
              "artifacts": str(artifact_dir), "reports": str(report_dir), "artifact_files": files, "report_files": report_files}
    logger.info("[%s | %s | %s] test accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f (%s) -> %s",
                spec.display_name, task, model_name, result["accuracy"], result["precision"], result["recall"],
                result["f1_score"], result["average"], report_dir)
    return result


def run_many(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    datasets = list(DATASETS) if args.dataset == "all" else [resolve_dataset(args.dataset).key]
    models = list(MODEL_NAMES) if args.model == "all" else [resolve_model(args.model)]
    tasks = list(TASKS) if args.task == "both" else [args.task]
    overrides = dict(args.param or [])
    unknown = [k for k in overrides if not any(k in DEFAULTS[m] for m in models)]
    if unknown:
        raise SystemExit(f"error: --param {unknown} not recognised by {models}")

    results, failures = [], []
    for ds in datasets:
        spec = resolve_dataset(ds)
        try:
            if args.preprocess_if_missing and not (args.processed_root / spec.slug / "train.csv.gz").exists():
                ensure_preprocessed(spec, args.processed_root, args.artifacts_root, args.raw_dir)
            loaded = load_dataset(spec, args.processed_root, args.artifacts_root)
        except Exception as exc:  # noqa: BLE001 - report and continue with other datasets
            logger.error("[%s] cannot load data: %s", spec.display_name, exc)
            failures.append(f"{spec.key}: {exc}")
            continue
        for model_name in models:
            for task in tasks:
                try:
                    results.append(train_one(
                        ds, model_name, task, processed_root=args.processed_root, artifacts_root=args.artifacts_root,
                        reports_root=args.reports_dir, seed=args.seed, val_size=args.val_size,
                        max_train_rows=args.max_train_rows, class_weight_mode=args.class_weight,
                        param_overrides={k: v for k, v in overrides.items() if k in DEFAULTS[model_name]},
                        _loaded=loaded,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[%s | %s | %s] failed", spec.display_name, task, model_name)
                    failures.append(f"{spec.key}/{task}/{model_name}: {exc}")
    return results, failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ml.training.train", description="Train and evaluate IDS models")
    p.add_argument("--dataset", required=True, choices=[*DATASETS, "all"], help="dataset to train on")
    p.add_argument("--model", required=True, choices=[*MODEL_NAMES, "all"], help="model to train")
    p.add_argument("--task", default="both", choices=[*TASKS, "both"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.1, help="stratified validation fraction taken from the training split (0 disables)")
    p.add_argument("--max-train-rows", type=int, default=None, help="stratified cap on training rows (quick dev runs)")
    p.add_argument("--class-weight", default="balanced", choices=CLASS_WEIGHT_MODES)
    p.add_argument("--param", action="append", type=parse_param, metavar="KEY=VALUE", help="override a model hyper-parameter (repeatable)")
    p.add_argument("--processed-root", type=Path, default=Path("ml/data/processed"))
    p.add_argument("--artifacts-root", type=Path, default=Path("ml/artifacts"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports/ml"))
    p.add_argument("--preprocess-if-missing", action="store_true", help="run the dataset's preprocessing step if processed files are absent")
    p.add_argument("--raw-dir", default=None, help="raw data dir for --preprocess-if-missing (single dataset only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.raw_dir and args.dataset == "all":
        raise SystemExit("error: --raw-dir only makes sense with a single --dataset")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results, failures = run_many(args)
    for r in results:
        print(f"{r['dataset']:<11} {r['task']:<10} {r['model']:<14} acc={r['accuracy']:.4f} "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1_score']:.4f} ({r['average']})")
    if failures:
        print("\nFAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

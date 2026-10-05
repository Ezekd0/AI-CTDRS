"""Dataset registry and task data for model training.

Training consumes the OUTPUT of the per-dataset preprocessing pipelines (``train.csv.gz`` /
``test.csv.gz`` plus the fitted ``preprocessor.joblib``). Datasets are never concatenated.

Splits
  * CICIDS2017 / CSE-CIC-IDS2018: the preprocessing step already made a stratified train/test split.
  * NSL-KDD: the official KDDTrain+ / KDDTest+ split is used as is.
  * Here, a stratified VALIDATION split is carved out of the training portion only (used for early
    stopping and model selection). The test split is touched exactly once, for final evaluation.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

TASKS = ("binary", "multiclass")
ATTACK_CLASS_NAME = "ATTACK"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # name used on the command line
    slug: str  # directory name used by the preprocessing pipelines
    display_name: str
    preprocessor_module: str
    preprocessor_class: str
    encoder_attr: str  # LabelEncoder attribute on the fitted preprocessor (multiclass classes)
    multiclass_id_column: str  # processed-CSV column with the encoded multiclass id
    multiclass_name_column: str  # processed-CSV column with the class name
    binary_column: str
    raw_dir: str
    preprocess_command: str


DATASETS: dict[str, DatasetSpec] = {
    "cicids2017": DatasetSpec(
        key="cicids2017", slug="cicids2017", display_name="CICIDS2017",
        preprocessor_module="ml.preprocessing.cicids2017_preprocessor", preprocessor_class="CICIDS2017Preprocessor",
        encoder_attr="label_encoder_", multiclass_id_column="label_id", multiclass_name_column="label",
        binary_column="is_attack", raw_dir="ml/data/raw/CIC-IDS2017",
        preprocess_command="python -m ml.preprocessing.cicids2017_preprocessor --raw-dir ml/data/raw/CIC-IDS2017",
    ),
    "cicids2018": DatasetSpec(
        key="cicids2018", slug="cse_cicids2018", display_name="CSE-CIC-IDS2018",
        preprocessor_module="ml.preprocessing.cse_cicids2018_preprocessor", preprocessor_class="CSECICIDS2018Preprocessor",
        encoder_attr="label_encoder_", multiclass_id_column="label_id", multiclass_name_column="label",
        binary_column="is_attack", raw_dir="ml/data/raw/CSE-CIC-IDS2018",
        preprocess_command="python -m ml.preprocessing.cse_cicids2018_preprocessor --raw-dir ml/data/raw/CSE-CIC-IDS2018",
    ),
    "nsl-kdd": DatasetSpec(
        key="nsl-kdd", slug="nsl_kdd", display_name="NSL-KDD",
        preprocessor_module="ml.preprocessing.nsl_kdd_preprocessor", preprocessor_class="NSLKDDPreprocessor",
        encoder_attr="category_encoder_", multiclass_id_column="attack_category_id", multiclass_name_column="attack_category",
        binary_column="is_attack", raw_dir="ml/data/raw/NSL-KDD",
        preprocess_command="python -m ml.preprocessing.nsl_kdd_preprocessor --raw-dir ml/data/raw/NSL-KDD",
    ),
}
_ALIASES = {"cicids2017": "cicids2017", "cicids2018": "cicids2018", "csecicids2018": "cicids2018", "nslkdd": "nsl-kdd"}


def resolve_dataset(name: str) -> DatasetSpec:
    """'NSL-KDD', 'nsl_kdd', 'cse-cic-ids2018' ... -> DatasetSpec."""
    squashed = "".join(ch for ch in name.lower() if ch.isalnum())
    if squashed not in _ALIASES:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    return DATASETS[_ALIASES[squashed]]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    train: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str]
    multiclass_encoder: LabelEncoder
    benign_label: str
    files: dict[str, Path]  # train, test, preprocessor, preprocessing_report (if present)


def ensure_preprocessed(spec: DatasetSpec, processed_root: Path, artifacts_root: Path, raw_dir: str | Path | None = None) -> None:
    """Run the dataset's own preprocessing pipeline (used by --preprocess-if-missing)."""
    module = importlib.import_module(spec.preprocessor_module)
    logger.info("Running %s preprocessing from %s", spec.display_name, raw_dir or spec.raw_dir)
    module.run_pipeline(
        raw_dir=raw_dir or spec.raw_dir,
        processed_dir=processed_root / spec.slug,
        artifacts_dir=artifacts_root / spec.slug,
    )


def load_dataset(spec: DatasetSpec, processed_root: str | Path, artifacts_root: str | Path) -> LoadedDataset:
    processed_dir, artifacts_dir = Path(processed_root) / spec.slug, Path(artifacts_root) / spec.slug
    paths = {
        "train": processed_dir / "train.csv.gz",
        "test": processed_dir / "test.csv.gz",
        "preprocessor": artifacts_dir / "preprocessor.joblib",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec.display_name}: preprocessed files not found: {missing}.\n"
            f"Run the preprocessing step first:\n  {spec.preprocess_command}\n"
            f"or pass --preprocess-if-missing."
        )
    report = artifacts_dir / "preprocessing_report.json"
    if report.exists():
        paths["preprocessing_report"] = report

    module = importlib.import_module(spec.preprocessor_module)
    pre = getattr(module, spec.preprocessor_class).load(paths["preprocessor"])
    feature_names = list(pre.feature_names_)
    encoder = getattr(pre, spec.encoder_attr)
    benign = pre.config.benign_label

    def read(path: Path) -> pd.DataFrame:
        header = pd.read_csv(path, nrows=0).columns
        absent = [c for c in feature_names if c not in header]
        if absent:
            raise ValueError(f"{path.name} is missing {len(absent)} feature columns recorded by the preprocessor, e.g. {absent[:5]}")
        dtypes = {c: np.float32 for c in feature_names}
        return pd.read_csv(path, dtype=dtypes)

    train, test = read(paths["train"]), read(paths["test"])
    logger.info("Loaded %s: train=%d test=%d features=%d", spec.display_name, len(train), len(test), len(feature_names))
    return LoadedDataset(spec, train, test, feature_names, encoder, benign, paths)


# ----------------------------------------------------------------------------
# Stratified helpers
# ----------------------------------------------------------------------------
def stratified_holdout(strata: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split indices into (keep, holdout) stratified on ``strata``.

    Classes with fewer than 2 rows cannot be stratified and stay entirely in ``keep``. If the data are too
    small to stratify, no holdout is made and the report says so (nothing is silently degraded).
    """
    strata = np.asarray(strata)
    idx = np.arange(len(strata))
    counts = pd.Series(strata).value_counts()
    rare_classes = counts[counts < 2].index
    rare_mask = np.isin(strata, rare_classes)
    eligible = idx[~rare_mask]
    n_classes = int((counts >= 2).sum())
    n_hold = int(round(fraction * len(eligible)))
    info: dict[str, Any] = {"requested_fraction": fraction, "rare_classes_kept_in_train": [int(c) for c in rare_classes]}
    if fraction <= 0 or n_hold < n_classes or len(eligible) - n_hold < n_classes:
        info.update(stratified=False, n_holdout=0, reason="holdout disabled or too small to stratify")
        return idx, np.array([], dtype=int), info
    keep, hold = train_test_split(eligible, test_size=n_hold, random_state=seed, stratify=strata[eligible])
    keep = np.sort(np.concatenate([keep, idx[rare_mask]]))
    info.update(stratified=True, n_holdout=int(len(hold)))
    return keep, np.sort(hold), info


def stratified_subsample(strata: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """Stratified subsample of at most ``n_max`` rows (rare classes with <2 rows are always kept)."""
    strata = np.asarray(strata)
    idx = np.arange(len(strata))
    if n_max >= len(idx):
        return idx
    counts = pd.Series(strata).value_counts()
    rare_mask = np.isin(strata, counts[counts < 2].index)
    eligible = idx[~rare_mask]
    budget = n_max - int(rare_mask.sum())
    n_classes = int((counts >= 2).sum())
    if budget < n_classes:
        raise ValueError(f"max_train_rows={n_max} is too small to keep every class (need at least {n_classes + int(rare_mask.sum())})")
    chosen, _ = train_test_split(eligible, train_size=budget, random_state=seed, stratify=strata[eligible])
    return np.sort(np.concatenate([chosen, idx[rare_mask]]))


def distribution(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(class_names)) if len(y) else np.zeros(len(class_names), dtype=int)
    return {name: int(c) for name, c in zip(class_names, counts)}


# ----------------------------------------------------------------------------
# Task data
# ----------------------------------------------------------------------------
@dataclass
class TaskData:
    task: str
    feature_names: list[str]
    class_names: list[str]
    label_encoder: LabelEncoder
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def has_validation(self) -> bool:
        return len(self.y_val) > 0


def binary_label_encoder(benign_label: str) -> LabelEncoder:
    """0 = benign, 1 = attack. ``classes_`` is set explicitly so the order is not alphabetical."""
    enc = LabelEncoder()
    enc.classes_ = np.array([benign_label, ATTACK_CLASS_NAME], dtype=object)
    return enc


def _check_encoder_matches_data(df: pd.DataFrame, id_col: str, name_col: str, encoder: LabelEncoder, where: str) -> None:
    pairs = df[[id_col, name_col]].drop_duplicates()
    ids = pairs[id_col].to_numpy()
    if (ids < 0).any() or (ids >= len(encoder.classes_)).any():
        bad = sorted(set(pairs.loc[(pairs[id_col] < 0) | (pairs[id_col] >= len(encoder.classes_)), name_col]))
        raise ValueError(f"{where}: rows with classes the preprocessor never saw in training: {bad}")
    expected = encoder.classes_[ids]
    if not np.array_equal(expected.astype(str), pairs[name_col].astype(str).to_numpy()):
        raise ValueError(f"{where}: '{id_col}' does not agree with the preprocessor's label encoder; re-run preprocessing")


def build_task_data(
    data: LoadedDataset, task: str, val_size: float = 0.1, seed: int = 42, max_train_rows: int | None = None
) -> TaskData:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    spec = data.spec
    id_col, name_col = spec.multiclass_id_column, spec.multiclass_name_column
    for split_name, df in (("train", data.train), ("test", data.test)):
        _check_encoder_matches_data(df, id_col, name_col, data.multiclass_encoder, f"{spec.display_name} {split_name}")

    if task == "multiclass":
        encoder = data.multiclass_encoder
        y_train_all, y_test = data.train[id_col].to_numpy(np.int64), data.test[id_col].to_numpy(np.int64)
    else:
        encoder = binary_label_encoder(data.benign_label)
        y_train_all, y_test = data.train[spec.binary_column].to_numpy(np.int64), data.test[spec.binary_column].to_numpy(np.int64)
    class_names = [str(c) for c in encoder.classes_]
    strata_all = data.train[id_col].to_numpy(np.int64)  # always the finest label, for both tasks

    rows = np.arange(len(data.train))
    if max_train_rows is not None:
        rows = stratified_subsample(strata_all, max_train_rows, seed)
    train_idx, val_idx, split_info = stratified_holdout(strata_all[rows], val_size, seed)
    train_rows, val_rows = rows[train_idx], rows[val_idx]

    X = data.train[data.feature_names].to_numpy(np.float32)
    X_train, X_val = X[train_rows], X[val_rows]
    y_train, y_val = y_train_all[train_rows], y_train_all[val_rows]
    X_test = data.test[data.feature_names].to_numpy(np.float32)

    present = np.unique(y_train)
    if not np.array_equal(present, np.arange(len(class_names))):
        raise ValueError(f"Training rows must contain every class 0..{len(class_names) - 1}; found {present.tolist()}")

    info = {
        "task": task,
        "stratified_on": id_col,
        "validation_split": split_info,
        "max_train_rows": max_train_rows,
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "train_distribution": distribution(y_train, class_names),
        "val_distribution": distribution(y_val, class_names),
        "test_distribution": distribution(y_test, class_names),
        "test_split_origin": "official split" if spec.key == "nsl-kdd" else "stratified split made by the preprocessing step",
    }
    return TaskData(task, data.feature_names, class_names, encoder, X_train, y_train, X_val, y_val, X_test, y_test, info)


def class_weights(y: np.ndarray, n_classes: int, mode: str = "balanced") -> np.ndarray | None:
    """Per-class weights normalised so the average SAMPLE weight is 1 (keeps the effective learning rate).

    ``balanced`` = n / (K * count); ``sqrt`` = its square root (milder for extreme imbalance); ``none`` -> None.
    """
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1.0))
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError("class weight mode must be one of: balanced, sqrt, none")
    return w / float((w * counts).sum() / counts.sum())

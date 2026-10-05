"""Dedicated, reusable preprocessing pipeline for CICIDS2017.

Design rules (to avoid leakage):
  * Row-level cleaning that needs no learned statistics (column/label
    normalisation, de-duplication, inf -> NaN) happens BEFORE the split.
    De-duplicating first prevents identical flows landing in both train and test.
  * Everything learned from data (median imputation, constant/correlated
    feature removal, scaling, label encoding) is fitted on the TRAINING split only.
  * Flow identifiers (Flow ID, IPs, source port, timestamp) are never features.

The same fitted object is used at inference through ``transform_features``.

CLI:
    python -m ml.preprocessing.cicids2017_preprocessor \
        --raw-dir ml/data/raw/CIC-IDS2017
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

BENIGN_LABEL = "BENIGN"
LABEL_CANDIDATES = ("label", "class", "attack_cat")  # normalised names
DEFAULT_IDENTIFIER_COLUMNS = (
    "flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip",
    "source_port", "src_port", "timestamp", "unnamed_0",
)


@dataclass
class PreprocessConfig:
    test_size: float = 0.2
    random_state: int = 42
    split_strategy: str = "source_file"
    scale: bool = True
    correlation_threshold: float | None = 0.98  # None disables correlated-feature removal
    min_class_samples: int = 2  # classes below this cannot be stratified and are dropped (reported)
    benign_label: str = BENIGN_LABEL
    identifier_columns: tuple[str, ...] = DEFAULT_IDENTIFIER_COLUMNS


# ----------------------------------------------------------------------------
# Stateless helpers
# ----------------------------------------------------------------------------
def normalize_column_name(name: Any) -> str:
    """' Flow Bytes/s' -> 'flow_bytes_s'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names in place; duplicates get a ``_dupN`` suffix."""
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        name = normalize_column_name(col) or "unnamed"
        if name in seen:
            seen[name] += 1
            name = f"{name}_dup{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def clean_label(value: Any) -> str | None:
    """Fix whitespace and the corrupted dash in 'Web Attack \ufffd Brute Force'."""
    if pd.isna(value):
        return None
    text = re.sub(r"[^A-Za-z0-9]+", " ", str(value)).strip()
    if not text:
        return None
    return BENIGN_LABEL if text.upper() == "BENIGN" else text


def detect_label_column(df: pd.DataFrame) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Looked for {LABEL_CANDIDATES}; got {list(df.columns)[:10]}...")


def class_distribution(labels: pd.Series) -> dict[str, dict[str, float]]:
    counts = labels.value_counts()
    total = int(counts.sum())
    return {str(k): {"count": int(v), "percent": round(100.0 * v / total, 4)} for k, v in counts.items()}


def load_raw_csvs(raw_dir: str | Path, nrows_per_file: int | None = None) -> pd.DataFrame:
    """Load every CSV in ``raw_dir`` and stack them (all share the CICIDS2017 schema)."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=nrows_per_file, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, nrows=nrows_per_file, low_memory=False, encoding="latin-1")
        cleaned = normalize_columns(df)
        cleaned.attrs["_source_file"] = f.name
        frames.append(cleaned)
        logger.info("Loaded %s: %d rows, %d columns", f.name, len(df), df.shape[1])
    if len({tuple(sorted(f.columns)) for f in frames}) > 1:
        logger.warning("CSV files do not share identical columns; missing values will be NaN")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["_source_file_groups"] = [f.attrs.get("_source_file") for f in frames for _ in range(len(f))]
    return combined


# ----------------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------------
class CICIDS2017Preprocessor:
    """prepare() -> split -> fit(train) -> transform_features()/encode_labels()."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.label_column_: str | None = None
        self.imputer_columns_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.dropped_columns_: dict[str, list[str]] = {}
        self.split_groups_: pd.Series | None = None

    # -- step 1: stateless cleaning (safe before the split) -------------------
    def prepare(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        cfg = self.config
        df = normalize_columns(raw.copy())
        source_groups = raw.attrs.get("_source_file_groups")
        source_series = pd.Series(source_groups, index=df.index, dtype="string") if source_groups is not None and len(source_groups) == len(df) else None
        label_col = detect_label_column(df)
        self.label_column_ = label_col
        report: dict[str, Any] = {"label_column": label_col, "rows_loaded": int(len(df))}

        df[label_col] = df[label_col].map(clean_label)
        n = len(df)
        df = df.dropna(subset=[label_col])
        report["rows_missing_label_removed"] = n - len(df)

        n = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = n - len(df)

        report["class_distribution_before_rare_filter"] = class_distribution(df[label_col])
        counts = df[label_col].value_counts()
        rare = sorted(counts[counts < cfg.min_class_samples].index.tolist())
        if rare:
            logger.warning("Dropping classes with < %d samples: %s", cfg.min_class_samples, rare)
            df = df[~df[label_col].isin(rare)]
        report["rare_classes_dropped"] = rare

        identifiers = [c for c in df.columns if c in set(cfg.identifier_columns)]
        self.dropped_columns_["identifiers"] = identifiers
        y = df[label_col].reset_index(drop=True)
        self.split_groups_ = source_series.loc[df.index].reset_index(drop=True) if source_series is not None else None
        X = df.drop(columns=[label_col, *identifiers]).reset_index(drop=True)

        X = X.apply(pd.to_numeric, errors="coerce")
        report["infinite_cells_replaced"] = int(np.isinf(X.to_numpy(dtype=float)).sum())
        X = X.replace([np.inf, -np.inf], np.nan)
        report["missing_cells_before_imputation"] = int(X.isna().sum().sum())
        report["rows_after_cleaning"] = int(len(X))
        report["class_distribution"] = class_distribution(y)
        logger.info("Class distribution: %s", report["class_distribution"])
        return X, y, report

    # -- step 2: learn everything from TRAIN only -------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CICIDS2017Preprocessor":
        cfg = self.config
        cols = [c for c in X_train.columns if X_train[c].notna().any()]
        self.dropped_columns_["all_missing_or_non_numeric"] = [c for c in X_train.columns if c not in cols]

        self.imputer_ = SimpleImputer(strategy="median").fit(X_train[cols])
        self.imputer_columns_ = cols
        imputed = pd.DataFrame(self.imputer_.transform(X_train[cols]), columns=cols, index=X_train.index)

        keep = [c for c in cols if imputed[c].nunique() > 1]
        self.dropped_columns_["constant"] = [c for c in cols if c not in keep]

        self.dropped_columns_["correlated"] = []
        if cfg.correlation_threshold is not None and len(keep) > 1:
            upper = np.triu(imputed[keep].corr().abs().to_numpy(), k=1)
            drop = {keep[j] for j in range(len(keep)) if (upper[:, j] > cfg.correlation_threshold).any()}
            self.dropped_columns_["correlated"] = [c for c in keep if c in drop]
            keep = [c for c in keep if c not in drop]

        self.feature_names_ = keep
        self.scaler_ = StandardScaler().fit(imputed[keep]) if cfg.scale else None
        self.label_encoder_ = LabelEncoder().fit(y_train)
        logger.info("Fitted preprocessor: %d features, %d classes", len(keep), len(self.label_encoder_.classes_))
        return self

    # -- inference-safe transforms -------------------------------------------
    def _check_fitted(self) -> None:
        if self.feature_names_ is None or self.imputer_ is None:
            raise RuntimeError("Preprocessor is not fitted")

    def transform_features(self, df: pd.DataFrame, scaled: bool = True) -> pd.DataFrame:
        """Raw flow records (CICIDS2017 columns, any naming style) -> model-ready numeric frame.

        Extra columns are ignored; missing columns are imputed with the training median.
        Pass ``scaled=False`` to get original-unit values (e.g. for display or SHAP/LIME).
        """
        self._check_fitted()
        frame = normalize_columns(df.copy())
        missing = [c for c in self.imputer_columns_ if c not in frame.columns]
        if missing:
            logger.warning("Input is missing %d expected columns; imputing: %s", len(missing), missing[:5])
        raw_frame = frame.reindex(columns=self.imputer_columns_)
        coerced = raw_frame.apply(pd.to_numeric, errors="coerce")
        invalid = raw_frame.notna() & coerced.isna()
        if bool(invalid.any().any()):
            bad = [str(c) for c in invalid.columns[invalid.any()].tolist()[:5]]
            raise ValueError(f"Non-numeric values supplied for numeric features: {bad}")
        X = coerced
        X = X.replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(self.imputer_.transform(X), columns=self.imputer_columns_, index=X.index)[self.feature_names_]
        if scaled and self.scaler_ is not None:
            out = pd.DataFrame(self.scaler_.transform(out), columns=self.feature_names_, index=out.index)
        return out

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        self._check_fitted()
        cleaned = labels.map(clean_label)
        return self.label_encoder_.transform(cleaned)

    def binary_labels(self, labels: pd.Series) -> np.ndarray:
        """0 = BENIGN, 1 = ATTACK."""
        cleaned = labels.map(clean_label)
        return (cleaned != self.config.benign_label).astype(int).to_numpy()

    def decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.label_encoder_.inverse_transform(encoded)

    # -- persistence -------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "label_column": self.label_column_,
                "imputer_columns": self.imputer_columns_,
                "feature_names": self.feature_names_,
                "imputer": self.imputer_,
                "scaler": self.scaler_,
                "label_encoder": self.label_encoder_,
                "dropped_columns": self.dropped_columns_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CICIDS2017Preprocessor":
        state = joblib.load(path)
        cfg = dict(state["config"])
        cfg["identifier_columns"] = tuple(cfg["identifier_columns"])
        obj = cls(PreprocessConfig(**cfg))
        obj.label_column_ = state["label_column"]
        obj.imputer_columns_ = state["imputer_columns"]
        obj.feature_names_ = state["feature_names"]
        obj.imputer_ = state["imputer"]
        obj.scaler_ = state["scaler"]
        obj.label_encoder_ = state["label_encoder"]
        obj.dropped_columns_ = state["dropped_columns"]
        return obj


# ----------------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------------
def run_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path = "ml/data/processed/cicids2017",
    artifacts_dir: str | Path = "ml/artifacts/cicids2017",
    config: PreprocessConfig | None = None,
    nrows_per_file: int | None = None,
) -> dict[str, Any]:
    """Load -> clean -> split -> fit on train -> save processed data + artifacts + report."""
    cfg = config or PreprocessConfig()
    pre = CICIDS2017Preprocessor(cfg)
    X, y, report = pre.prepare(load_raw_csvs(raw_dir, nrows_per_file))

    if cfg.split_strategy != "source_file":
        raise ValueError("CICIDS2017 only supports leakage-safe source_file splitting")
    groups = pre.split_groups_
    if groups is None or len(groups) != len(X):
        raise ValueError("Source-file provenance is unavailable; refusing an unsafe row-random split")
    # Classes represented by only one source file cannot be evaluated out-of-file without leaking that file.
    # Keep those source files in training and split only groups that can safely contribute to the test set.
    group_frame = pd.DataFrame({"group": groups.to_numpy(), "label": y.to_numpy()})
    group_counts = group_frame.groupby("label")["group"].nunique()
    protected_groups = set(group_frame.loc[group_frame["label"].map(group_counts) < 2, "group"])
    eligible_mask = ~groups.isin(protected_groups).to_numpy()
    eligible_idx = np.flatnonzero(eligible_mask)
    if len(eligible_idx) == 0:
        raise ValueError("No source-file groups can be held out safely; each class occurs in only one source file")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    rel_train, rel_test = next(splitter.split(X.iloc[eligible_idx], y.iloc[eligible_idx], groups=groups.iloc[eligible_idx]))
    test_idx = eligible_idx[rel_test]
    train_idx = np.setdiff1d(np.arange(len(X)), test_idx, assume_unique=False)
    train_classes, test_classes = set(y.iloc[train_idx]), set(y.iloc[test_idx])
    if not set(train_classes).issuperset(set(y.unique())):
        raise ValueError("Leakage-safe source-file split removed a class from training; adjust source files")
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    pre.fit(X_tr, y_tr)

    def build(Xs: pd.DataFrame, ys: pd.Series) -> pd.DataFrame:
        out = pre.transform_features(Xs).reset_index(drop=True)
        out["label"] = ys.to_numpy()
        out["label_id"] = pre.encode_labels(ys)
        out["is_attack"] = pre.binary_labels(ys)
        return out

    train_df, test_df = build(X_tr, y_tr), build(X_te, y_te)
    processed_dir, artifacts_dir = Path(processed_dir), Path(artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv.gz", index=False, compression="gzip")
    test_df.to_csv(processed_dir / "test.csv.gz", index=False, compression="gzip")
    pre.save(artifacts_dir / "preprocessor.joblib")

    report.update(
        {
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "split_strategy": cfg.split_strategy,
            "train_source_files": sorted(set(pre.split_groups_.iloc[train_idx].tolist())),
            "test_source_files": sorted(set(pre.split_groups_.iloc[test_idx].tolist())),
            "n_features": len(pre.feature_names_),
            "feature_names": pre.feature_names_,
            "dropped_columns": pre.dropped_columns_,
            "label_classes": pre.label_encoder_.classes_.tolist(),
            "train_class_distribution": class_distribution(y_tr),
            "test_class_distribution": class_distribution(y_te),
        }
    )
    (artifacts_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Saved processed data to %s and artifacts to %s", processed_dir, artifacts_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CICIDS2017 preprocessing pipeline")
    parser.add_argument("--raw-dir", default="ml/data/raw/CIC-IDS2017")
    parser.add_argument("--processed-dir", default="ml/data/processed/cicids2017")
    parser.add_argument("--artifacts-dir", default="ml/artifacts/cicids2017")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows-per-file", type=int, default=None, help="Optional row cap for quick dev runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = PreprocessConfig(test_size=args.test_size, random_state=args.seed)
    report = run_pipeline(args.raw_dir, args.processed_dir, args.artifacts_dir, cfg, args.nrows_per_file)
    print(json.dumps({k: report[k] for k in ("rows_loaded", "duplicates_removed", "n_train", "n_test", "n_features")}, indent=2))


if __name__ == "__main__":
    main()

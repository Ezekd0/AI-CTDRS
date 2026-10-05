"""Dedicated, reusable preprocessing pipeline for CSE-CIC-IDS2018.

This is a SEPARATE pipeline from ``cicids2017_preprocessor``. It does not assume the 2018
schema matches 2017: columns are discovered per file, schema differences are recorded in the
report, and nothing relies on a hard-coded feature list. See
``docs/cicids2017_vs_cse_cicids2018_preprocessing.md`` for the differences.

Leakage rules (same principle as the 2017 pipeline):
  * Cleaning that learns nothing (name/label normalisation, header-row removal, inf -> NaN,
    de-duplication) happens BEFORE the split.
  * Everything learned (imputer, constant/correlated/high-missing feature removal, scaler,
    label encoder) is fitted on the TRAINING split only.
  * Flow identifiers (Flow ID, IPs, source port, timestamp) are never features.

CLI:
    python -m ml.preprocessing.cse_cicids2018_preprocessor --raw-dir ml/data/raw/CSE-CIC-IDS2018
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
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
LABEL_COLUMN = "label"  # standard name used in cleaned frames
LABEL_CANDIDATES = ("label", "class", "attack_cat")  # normalised names searched in raw files
DEFAULT_IDENTIFIER_COLUMNS = (
    "flow_id", "src_ip", "source_ip", "dst_ip", "destination_ip",
    "src_port", "source_port", "timestamp", "unnamed_0",
)
# keys are lower-case cleaned label text
DEFAULT_LABEL_ALIASES = {"benign": BENIGN_LABEL, "infilteration": "Infiltration"}


@dataclass
class CSECICIDS2018Config:
    test_size: float = 0.2
    random_state: int = 42
    split_strategy: str = "source_file"
    scale: bool = True
    correlation_threshold: float | None = 0.98
    min_class_samples: int = 2
    max_missing_fraction: float = 0.5  # feature columns missing in more of train than this are dropped
    max_rows_per_class: int | None = None  # optional seeded down-sampling cap (dataset is very large)
    chunksize: int = 500_000
    benign_label: str = BENIGN_LABEL
    identifier_columns: tuple[str, ...] = DEFAULT_IDENTIFIER_COLUMNS
    label_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_ALIASES))


# ----------------------------------------------------------------------------
# Stateless helpers
# ----------------------------------------------------------------------------
def normalize_column_name(name: Any) -> str:
    """'Flow Byts/s' -> 'flow_byts_s'."""
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


def clean_label(value: Any, aliases: dict[str, str] | None = None) -> str | None:
    """Collapse punctuation/whitespace, then apply the (case-insensitive) alias map."""
    if pd.isna(value):
        return None
    text = re.sub(r"[^A-Za-z0-9]+", " ", str(value)).strip()
    if not text:
        return None
    return (aliases or DEFAULT_LABEL_ALIASES).get(text.lower(), text)


def detect_label_column(df: pd.DataFrame) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Looked for {LABEL_CANDIDATES}; got {list(df.columns)[:10]}...")


def class_distribution(labels: pd.Series) -> dict[str, dict[str, float]]:
    counts = labels.value_counts()
    total = int(counts.sum())
    return {str(k): {"count": int(v), "percent": round(100.0 * v / total, 4)} for k, v in counts.items()}


def _numeric_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric float32 and turn +/-inf into NaN."""
    out = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# CSV discovery and loading
# ----------------------------------------------------------------------------
def discover_csvs(raw_dir: str | Path) -> list[Path]:
    """Find every CSV under ``raw_dir`` (recursively, sorted for reproducibility)."""
    files = sorted(Path(raw_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return files


def clean_chunk(chunk: pd.DataFrame, cfg: CSECICIDS2018Config, stats: dict[str, Any]) -> pd.DataFrame:
    """Per-chunk cleaning: normalise names, drop repeated header rows / unlabeled rows,
    drop identifiers, coerce features to float32, inf -> NaN. Returns features + ``label``."""
    chunk = normalize_columns(chunk)
    stats.setdefault("columns", list(chunk.columns))
    stats["rows_read"] = stats.get("rows_read", 0) + len(chunk)
    label_col = detect_label_column(chunk)

    # Some 2018 files embed copies of the header row inside the data.
    is_header = chunk[label_col].astype(str).str.strip().str.lower() == label_col
    stats["repeated_header_rows_removed"] = stats.get("repeated_header_rows_removed", 0) + int(is_header.sum())
    chunk = chunk.loc[~is_header]

    labels = chunk[label_col].map(lambda v: clean_label(v, cfg.label_aliases))
    missing = labels.isna()
    stats["rows_missing_label_removed"] = stats.get("rows_missing_label_removed", 0) + int(missing.sum())
    chunk, labels = chunk.loc[~missing], labels.loc[~missing]

    id_set = set(cfg.identifier_columns)
    ids = [c for c in chunk.columns if c in id_set]
    stats["identifier_columns_present"] = sorted(set(stats.get("identifier_columns_present", [])) | set(ids))

    feats = chunk.drop(columns=[label_col, *ids]).apply(pd.to_numeric, errors="coerce").astype("float32")
    stats["infinite_cells_replaced"] = stats.get("infinite_cells_replaced", 0) + int(np.isinf(feats.to_numpy()).sum())
    feats = feats.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    feats[LABEL_COLUMN] = labels.to_numpy()
    return feats


def _load_file(path: Path, cfg: CSECICIDS2018Config, nrows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    last_err: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        stats: dict[str, Any] = {"file": path.name}
        try:
            parts = [
                clean_chunk(c, cfg, stats)
                for c in pd.read_csv(path, nrows=nrows, chunksize=cfg.chunksize, low_memory=False, encoding=encoding)
            ]
            break
        except UnicodeDecodeError as err:  # retry the whole file with a permissive encoding
            last_err = err
    else:
        raise last_err  # type: ignore[misc]
    if not parts:
        raise ValueError(f"{path.name} contains no rows")
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed_within_file"] = n - len(df)
    logger.info("Loaded %s: %d rows read, %d kept", path.name, stats["rows_read"], len(df))
    return df, stats


def summarize_schema(file_stats: list[dict[str, Any]]) -> dict[str, Any]:
    sets = [set(s["columns"]) for s in file_stats]
    common, union = set.intersection(*sets), set.union(*sets)
    return {
        "n_files": len(file_stats),
        "common_columns": sorted(common),
        "columns_not_in_all_files": sorted(union - common),
        "per_file": file_stats,
    }


def load_raw_csvs(
    raw_dir: str | Path, config: CSECICIDS2018Config | None = None, nrows_per_file: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Discover + load all CSVs. Returns (features + ``label`` frame, schema/load report)."""
    cfg = config or CSECICIDS2018Config()
    frames, stats = [], []
    for path in discover_csvs(raw_dir):
        df, s = _load_file(path, cfg, nrows_per_file)
        df.attrs["_source_file"] = str(path.relative_to(Path(raw_dir)))
        frames.append(df)
        stats.append(s)
    schema = summarize_schema(stats)
    if schema["columns_not_in_all_files"]:
        logger.warning("Files do not share identical columns: %s", schema["columns_not_in_all_files"])
    frame = pd.concat(frames, ignore_index=True, sort=False)
    feature_cols = [c for c in frame.columns if c != LABEL_COLUMN]
    frame[feature_cols] = frame[feature_cols].astype("float32")
    keys = ("rows_read", "repeated_header_rows_removed", "rows_missing_label_removed",
            "infinite_cells_replaced", "duplicates_removed_within_file")
    schema["totals"] = {k: int(sum(s.get(k, 0) for s in stats)) for k in keys}
    frame.attrs["_source_file_groups"] = [f.attrs.get("_source_file") for f in frames for _ in range(len(f))]
    return frame, schema


# ----------------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------------
class CSECICIDS2018Preprocessor:
    """load_raw_csvs() -> prepare() -> split -> fit(train) -> transform_features()/encode_labels()."""

    def __init__(self, config: CSECICIDS2018Config | None = None) -> None:
        self.config = config or CSECICIDS2018Config()
        self.imputer_columns_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.dropped_columns_: dict[str, list[str]] = {}
        self.split_groups_: pd.Series | None = None

    # -- step 1: stateless dataset-level cleaning ------------------------------
    def prepare(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        """``frame`` is the output of ``load_raw_csvs`` (numeric features + ``label``)."""
        cfg = self.config
        source_groups = frame.attrs.get("_source_file_groups")
        source_series = pd.Series(source_groups, index=frame.index, dtype="string") if source_groups is not None and len(source_groups) == len(frame) else None
        if LABEL_COLUMN not in frame.columns:
            raise ValueError("frame must contain a 'label' column (use load_raw_csvs)")
        report: dict[str, Any] = {"rows_after_chunk_cleaning": int(len(frame))}

        n = len(frame)
        df = frame.drop_duplicates()  # content duplicates (identifiers already removed)
        report["duplicates_removed_across_files"] = n - len(df)

        report["class_distribution_before_filters"] = class_distribution(df[LABEL_COLUMN])
        counts = df[LABEL_COLUMN].value_counts()
        rare = sorted(counts[counts < cfg.min_class_samples].index.tolist())
        if rare:
            logger.warning("Dropping classes with < %d samples: %s", cfg.min_class_samples, rare)
            df = df[~df[LABEL_COLUMN].isin(rare)]
        report["rare_classes_dropped"] = rare

        if cfg.max_rows_per_class:
            rng = np.random.default_rng(cfg.random_state)
            keep = []
            for idx in df.groupby(LABEL_COLUMN, sort=True).indices.values():
                keep.append(rng.choice(idx, cfg.max_rows_per_class, replace=False) if len(idx) > cfg.max_rows_per_class else idx)
            df = df.iloc[np.sort(np.concatenate(keep))]
        report["max_rows_per_class"] = cfg.max_rows_per_class

        y = df[LABEL_COLUMN].reset_index(drop=True)
        self.split_groups_ = source_series.loc[df.index].reset_index(drop=True) if source_series is not None else None
        X = df.drop(columns=[LABEL_COLUMN]).reset_index(drop=True)
        report["missing_cells_before_imputation"] = int(X.isna().sum().sum())
        report["rows_after_cleaning"] = int(len(X))
        report["class_distribution"] = class_distribution(y)
        logger.info("Class distribution: %s", report["class_distribution"])
        return X, y, report

    # -- step 2: learn everything from TRAIN only ------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CSECICIDS2018Preprocessor":
        cfg = self.config
        missing_frac = X_train.isna().mean()
        cols = [c for c in X_train.columns if missing_frac[c] <= cfg.max_missing_fraction and missing_frac[c] < 1.0]
        self.dropped_columns_["high_missing"] = [c for c in X_train.columns if c not in cols]

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
        """Raw CSE-CIC-IDS2018-style flow records -> model-ready numeric frame.

        Accepts raw headers (any spacing/case); extra columns (IPs, timestamp, label) are ignored;
        missing columns are imputed with the training median. ``scaled=False`` keeps original units.
        """
        self._check_fitted()
        frame = normalize_columns(df.copy())
        missing = [c for c in self.imputer_columns_ if c not in frame.columns]
        if missing:
            logger.warning("Input is missing %d expected columns; imputing: %s", len(missing), missing[:5])
        raw_frame = frame.reindex(columns=self.imputer_columns_)
        numeric_raw = raw_frame.apply(pd.to_numeric, errors="coerce")
        invalid = raw_frame.notna() & numeric_raw.isna()
        if bool(invalid.any().any()):
            bad = [str(c) for c in invalid.columns[invalid.any()].tolist()[:5]]
            raise ValueError(f"Non-numeric values supplied for numeric features: {bad}")
        X = _numeric_float32(raw_frame)
        out = pd.DataFrame(self.imputer_.transform(X), columns=self.imputer_columns_, index=X.index)[self.feature_names_]
        if scaled and self.scaler_ is not None:
            out = pd.DataFrame(self.scaler_.transform(out), columns=self.feature_names_, index=out.index)
        return out

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        self._check_fitted()
        return self.label_encoder_.transform(labels.map(lambda v: clean_label(v, self.config.label_aliases)))

    def binary_labels(self, labels: pd.Series) -> np.ndarray:
        """0 = BENIGN, 1 = ATTACK."""
        cleaned = labels.map(lambda v: clean_label(v, self.config.label_aliases))
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
    def load(cls, path: str | Path) -> "CSECICIDS2018Preprocessor":
        state = joblib.load(path)
        cfg = dict(state["config"])
        cfg["identifier_columns"] = tuple(cfg["identifier_columns"])
        obj = cls(CSECICIDS2018Config(**cfg))
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
    processed_dir: str | Path = "ml/data/processed/cse_cicids2018",
    artifacts_dir: str | Path = "ml/artifacts/cse_cicids2018",
    config: CSECICIDS2018Config | None = None,
    nrows_per_file: int | None = None,
) -> dict[str, Any]:
    """Discover -> load/clean -> split -> fit on train -> save processed data, artifacts, report."""
    cfg = config or CSECICIDS2018Config()
    frame, schema = load_raw_csvs(raw_dir, cfg, nrows_per_file)
    pre = CSECICIDS2018Preprocessor(cfg)
    X, y, report = pre.prepare(frame)
    del frame

    if cfg.split_strategy != "source_file":
        raise ValueError("CSE-CIC-IDS2018 only supports leakage-safe source_file splitting")
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
            "schema": schema,
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
    parser = argparse.ArgumentParser(description="CSE-CIC-IDS2018 preprocessing pipeline")
    parser.add_argument("--raw-dir", default="ml/data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--processed-dir", default="ml/data/processed/cse_cicids2018")
    parser.add_argument("--artifacts-dir", default="ml/artifacts/cse_cicids2018")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows-per-file", type=int, default=None, help="Optional row cap per file for dev runs")
    parser.add_argument("--max-rows-per-class", type=int, default=None, help="Optional seeded cap per class")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = CSECICIDS2018Config(test_size=args.test_size, random_state=args.seed, max_rows_per_class=args.max_rows_per_class)
    report = run_pipeline(args.raw_dir, args.processed_dir, args.artifacts_dir, cfg, args.nrows_per_file)
    print(json.dumps({k: report[k] for k in ("rows_after_cleaning", "n_train", "n_test", "n_features")}, indent=2))


if __name__ == "__main__":
    main()

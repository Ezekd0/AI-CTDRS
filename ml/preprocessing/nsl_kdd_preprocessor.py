"""Dedicated, reusable preprocessing pipeline for NSL-KDD (KDDTrain+ / KDDTest+).

NSL-KDD is a different kind of dataset from CICIDS2017/CSE-CIC-IDS2018: files usually have NO header,
there are 41 connection features (3 categorical, the rest numeric) plus a label and a difficulty
level, labels are specific attack names that roll up into 4 categories, and the official train/test
split is fixed. Nothing here is shared with, or assumes, the CICIDS schemas.

Rules:
  * KDDTrain+ and KDDTest+ are the official split; they are NOT re-split or merged.
  * Everything learned (imputer, one-hot categories, log/scale statistics, label encoders, constant
    column removal) is fitted on KDDTrain+ ONLY and applied unchanged to KDDTest+ and inference data.
  * The difficulty level is kept as metadata for evaluation; it is never a feature.
  * KDDTest+ is not de-duplicated (benchmark set); overlap with train is reported, not removed.

CLI:
    python -m ml.preprocessing.nsl_kdd_preprocessor --raw-dir ml/data/raw/NSL-KDD
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_srv_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]
BINARY_COLUMNS: list[str] = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
NUMERIC_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
LABEL_CANDIDATES = ("label", "class", "attack", "outcome")
DIFFICULTY_CANDIDATES = ("difficulty_level", "difficulty", "difficulty_score")

# Widely used community mapping of NSL-KDD attack names to the four KDD categories.
# Override/extend via NSLKDDConfig.attack_category_map if your files contain other names.
_CATEGORY_GROUPS = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable",
            "mailbomb", "worm"],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy",
            "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"],
    "U2R": ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"],
}
DEFAULT_ATTACK_CATEGORY_MAP: dict[str, str] = {name: cat for cat, names in _CATEGORY_GROUPS.items() for name in names}


@dataclass
class NSLKDDConfig:
    scale: bool = True  # StandardScaler on continuous numeric columns (not binary flags / one-hot)
    log_transform: bool = True  # log1p on heavy-tailed columns before scaling
    log_transform_columns: tuple[str, ...] = ("duration", "src_bytes", "dst_bytes")
    drop_duplicates_train: bool = True
    benign_label: str = "normal"
    attack_category_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ATTACK_CATEGORY_MAP))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def normalize_column_name(name: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [normalize_column_name(c) or "unnamed" for c in df.columns]
    return df


def clean_label(value: Any) -> str | None:
    """'Neptune.' / ' neptune ' -> 'neptune'."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower().rstrip(".").strip()
    return text or None


def class_distribution(labels: pd.Series) -> dict[str, dict[str, float]]:
    counts = labels.value_counts()
    total = int(counts.sum())
    return {str(k): {"count": int(v), "percent": round(100.0 * v / total, 4)} for k, v in counts.items()}


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _log1p(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = np.log1p(out[c].clip(lower=0))
    return out


# ----------------------------------------------------------------------------
# Discovery and loading
# ----------------------------------------------------------------------------
def find_split_files(raw_dir: str | Path) -> tuple[Path, Path]:
    """Locate KDDTrain+ and KDDTest+ (.txt/.csv). The 20-percent and -21 variants are ignored;
    pass them explicitly to ``run_pipeline`` if you want them."""
    train = test = None
    for p in sorted(Path(raw_dir).rglob("*")):
        name = p.name.lower()
        if p.suffix.lower() not in (".txt", ".csv"):
            continue
        if name.startswith("kddtrain+") and "20percent" not in name and train is None:
            train = p
        elif name.startswith("kddtest+") and test is None:
            test = p
    if train is None or test is None:
        raise FileNotFoundError(f"Could not find both KDDTrain+ and KDDTest+ (.txt/.csv) under {raw_dir}")
    return train, test


def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    """Load one NSL-KDD file (headerless 42/43-column layout, or a file with a header row).

    Returns all-string columns: the 41 features, ``label`` and (if present) ``difficulty_level``.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
    has_header = not _is_number(first.split(",")[0].strip().strip('"'))
    if has_header:
        df = normalize_columns(pd.read_csv(path, dtype=str, skipinitialspace=True, encoding="utf-8-sig"))
        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        if label_col is None:
            raise ValueError(f"{path.name}: no label column among {LABEL_CANDIDATES}")
        df = df.rename(columns={label_col: "label"})
        diff = next((c for c in DIFFICULTY_CANDIDATES if c in df.columns), None)
        if diff:
            df = df.rename(columns={diff: "difficulty_level"})
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing expected NSL-KDD columns: {missing[:6]}")
    else:
        df = pd.read_csv(path, header=None, dtype=str, skipinitialspace=True, encoding="utf-8-sig")
        n = df.shape[1]
        if n == len(FEATURE_COLUMNS) + 2:
            df.columns = [*FEATURE_COLUMNS, "label", "difficulty_level"]
        elif n == len(FEATURE_COLUMNS) + 1:
            df.columns = [*FEATURE_COLUMNS, "label"]
        else:
            raise ValueError(f"{path.name}: expected 42 or 43 columns, found {n}; not an NSL-KDD text/CSV file")
    keep = [*FEATURE_COLUMNS, "label"] + (["difficulty_level"] if "difficulty_level" in df.columns else [])
    logger.info("Loaded %s: %d rows (header=%s)", path.name, len(df), has_header)
    return df[keep]


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Types and validity only (no learned state): numeric coercion, inf -> NaN, categorical
    strip/lowercase with a sentinel for missing. Returns the 41 features in canonical order."""
    frame = normalize_columns(df.copy())
    missing_cols = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing_cols:
        logger.warning("Input is missing %d expected columns (will be imputed): %s", len(missing_cols), missing_cols[:6])
    feats = frame.reindex(columns=FEATURE_COLUMNS)

    raw_num = feats[NUMERIC_COLUMNS]
    num = raw_num.apply(pd.to_numeric, errors="coerce").astype("float64")
    invalid = int((num.isna() & raw_num.notna()).sum().sum())
    inf = int(np.isinf(num.to_numpy()).sum())
    num = num.replace([np.inf, -np.inf], np.nan)

    cat = feats[CATEGORICAL_COLUMNS].apply(lambda s: s.astype("string").str.strip().str.lower())
    cat = cat.replace("", pd.NA)
    cat_missing = int(cat.isna().sum().sum())
    cat = cat.fillna("unknown").astype(object)

    out = pd.concat([num, cat], axis=1)[FEATURE_COLUMNS]
    return out, {"invalid_numeric_cells": invalid, "infinite_cells_replaced": inf, "missing_categorical_cells": cat_missing}


# ----------------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------------
class NSLKDDPreprocessor:
    """prepare() both splits -> fit(train) -> transform_features()/label encoders."""

    def __init__(self, config: NSLKDDConfig | None = None) -> None:
        self.config = config or NSLKDDConfig()
        self.imputer_: SimpleImputer | None = None
        self.onehot_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.category_encoder_: LabelEncoder | None = None
        self.type_encoder_: LabelEncoder | None = None
        self.numeric_features_: list[str] | None = None
        self.scaled_columns_: list[str] = []
        self.log_columns_: list[str] = []
        self.feature_names_: list[str] | None = None
        self.dropped_columns_: dict[str, list[str]] = {}

    # -- labels ---------------------------------------------------------------
    def label_to_category(self, labels: pd.Series) -> pd.Series:
        """Specific attack name -> normal/DoS/Probe/R2L/U2R. Unknown names raise ValueError."""
        cfg = self.config
        cleaned = labels.map(clean_label)
        cats = cleaned.map(lambda l: cfg.benign_label if l == cfg.benign_label else cfg.attack_category_map.get(l))
        bad = sorted(set(cleaned[cats.isna()].dropna()))
        if bad:
            raise ValueError(f"Unmapped NSL-KDD labels {bad}; add them to NSLKDDConfig.attack_category_map")
        return cats

    def binary_labels(self, labels: pd.Series) -> np.ndarray:
        """0 = normal, 1 = attack."""
        return (labels.map(clean_label) != self.config.benign_label).astype(int).to_numpy()

    @staticmethod
    def _encode(encoder: LabelEncoder, values: pd.Series, unknown: int = -1) -> np.ndarray:
        mapping = {c: i for i, c in enumerate(encoder.classes_)}
        return values.map(lambda v: mapping.get(v, unknown)).to_numpy(dtype=int)

    def encode_categories(self, labels_or_categories: pd.Series, from_attack_names: bool = False) -> np.ndarray:
        self._check_fitted()
        cats = self.label_to_category(labels_or_categories) if from_attack_names else labels_or_categories
        return self._encode(self.category_encoder_, cats)

    def encode_attack_types(self, labels: pd.Series) -> np.ndarray:
        """Specific attack name -> id; names never seen in KDDTrain+ get -1."""
        self._check_fitted()
        return self._encode(self.type_encoder_, labels.map(clean_label))

    def decode_categories(self, encoded: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.category_encoder_.inverse_transform(encoded)

    # -- step 1: stateless cleaning ----------------------------------------------
    def prepare(self, raw: pd.DataFrame, deduplicate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Returns (clean 41-column features, targets/metadata frame, report)."""
        df = normalize_columns(raw.copy())
        if "label" not in df.columns:
            raise ValueError("raw frame needs a 'label' column (use load_nsl_kdd)")
        report: dict[str, Any] = {"rows_loaded": int(len(df))}

        df["label"] = df["label"].map(clean_label)
        n = len(df)
        df = df.dropna(subset=["label"])
        report["rows_missing_label_removed"] = n - len(df)
        self.label_to_category(df["label"])  # validates every label up front

        n = len(df)
        if deduplicate:
            df = df.drop_duplicates(subset=[c for c in [*FEATURE_COLUMNS, "label"] if c in df.columns])
        report["duplicates_removed"] = n - len(df)
        df = df.reset_index(drop=True)

        X, stats = clean_features(df)
        meta = pd.DataFrame({
            "label": df["label"],
            "attack_category": self.label_to_category(df["label"]),
            "is_attack": self.binary_labels(df["label"]),
        })
        if "difficulty_level" in df.columns:
            meta["difficulty_level"] = pd.to_numeric(df["difficulty_level"], errors="coerce")
        report.update(stats)
        report["rows_after_cleaning"] = int(len(X))
        report["attack_category_distribution"] = class_distribution(meta["attack_category"])
        report["attack_type_distribution"] = class_distribution(meta["label"])
        return X, meta, report

    # -- step 2: learn from KDDTrain+ only ----------------------------------------
    def fit(self, X_train: pd.DataFrame, meta_train: pd.DataFrame) -> "NSLKDDPreprocessor":
        cfg = self.config
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train[NUMERIC_COLUMNS])
        imputed = pd.DataFrame(self.imputer_.transform(X_train[NUMERIC_COLUMNS]), columns=NUMERIC_COLUMNS, index=X_train.index)

        keep = [c for c in NUMERIC_COLUMNS if imputed[c].nunique() > 1]
        self.dropped_columns_["constant"] = [c for c in NUMERIC_COLUMNS if c not in keep]
        self.numeric_features_ = keep

        self.log_columns_ = [c for c in cfg.log_transform_columns if c in keep] if cfg.log_transform else []
        self.scaled_columns_ = [c for c in keep if c not in BINARY_COLUMNS] if cfg.scale else []
        self.scaler_ = None
        if self.scaled_columns_:
            self.scaler_ = StandardScaler().fit(_log1p(imputed[keep], self.log_columns_)[self.scaled_columns_])

        self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[CATEGORICAL_COLUMNS])
        self.feature_names_ = keep + self.onehot_.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()

        self.category_encoder_ = LabelEncoder().fit(meta_train["attack_category"])
        self.type_encoder_ = LabelEncoder().fit(meta_train["label"])
        logger.info("Fitted: %d features (%d numeric), %d categories, %d attack types", len(self.feature_names_),
                    len(keep), len(self.category_encoder_.classes_), len(self.type_encoder_.classes_))
        return self

    # -- inference-safe transform ---------------------------------------------------
    def _check_fitted(self) -> None:
        if self.feature_names_ is None or self.imputer_ is None:
            raise RuntimeError("Preprocessor is not fitted")

    def transform_features(self, df: pd.DataFrame, scaled: bool = True) -> pd.DataFrame:
        """Raw NSL-KDD-style records -> model-ready frame (same columns/order as training).

        Accepts the 41 feature columns in any header style; label/difficulty/extra columns are ignored;
        missing columns are imputed (numeric median / 'unknown' category); unseen categorical values
        become all-zero one-hot rows. ``scaled=False`` keeps numeric columns in original units
        (no log, no scaling) for display or explanation.
        """
        self._check_fitted()
        X, _ = clean_features(df)
        num = pd.DataFrame(self.imputer_.transform(X[NUMERIC_COLUMNS]), columns=NUMERIC_COLUMNS, index=X.index)
        num = num[self.numeric_features_]
        if scaled:
            num = _log1p(num, self.log_columns_)
            if self.scaler_ is not None:
                num[self.scaled_columns_] = self.scaler_.transform(num[self.scaled_columns_])
        cat = pd.DataFrame(
            self.onehot_.transform(X[CATEGORICAL_COLUMNS]),
            columns=self.onehot_.get_feature_names_out(CATEGORICAL_COLUMNS),
            index=X.index,
        )
        return pd.concat([num, cat], axis=1)[self.feature_names_]

    # -- persistence -----------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "imputer": self.imputer_, "onehot": self.onehot_, "scaler": self.scaler_,
                "category_encoder": self.category_encoder_, "type_encoder": self.type_encoder_,
                "numeric_features": self.numeric_features_, "scaled_columns": self.scaled_columns_,
                "log_columns": self.log_columns_, "feature_names": self.feature_names_,
                "dropped_columns": self.dropped_columns_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NSLKDDPreprocessor":
        s = joblib.load(path)
        cfg = dict(s["config"])
        cfg["log_transform_columns"] = tuple(cfg["log_transform_columns"])
        obj = cls(NSLKDDConfig(**cfg))
        obj.imputer_, obj.onehot_, obj.scaler_ = s["imputer"], s["onehot"], s["scaler"]
        obj.category_encoder_, obj.type_encoder_ = s["category_encoder"], s["type_encoder"]
        obj.numeric_features_, obj.scaled_columns_ = s["numeric_features"], s["scaled_columns"]
        obj.log_columns_, obj.feature_names_ = s["log_columns"], s["feature_names"]
        obj.dropped_columns_ = s["dropped_columns"]
        return obj


# ----------------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------------
def run_pipeline(
    raw_dir: str | Path | None = None,
    processed_dir: str | Path = "ml/data/processed/nsl_kdd",
    artifacts_dir: str | Path = "ml/artifacts/nsl_kdd",
    config: NSLKDDConfig | None = None,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load KDDTrain+/KDDTest+ -> clean -> fit on train -> transform both -> save data, artifacts, report."""
    cfg = config or NSLKDDConfig()
    if train_path is None or test_path is None:
        if raw_dir is None:
            raise ValueError("Provide raw_dir or both train_path and test_path")
        found_train, found_test = find_split_files(raw_dir)
        train_path, test_path = train_path or found_train, test_path or found_test
    train_raw, test_raw = load_nsl_kdd(train_path), load_nsl_kdd(test_path)

    pre = NSLKDDPreprocessor(cfg)
    X_tr, m_tr, rep_tr = pre.prepare(train_raw, deduplicate=cfg.drop_duplicates_train)
    X_te, m_te, rep_te = pre.prepare(test_raw, deduplicate=False)
    pre.fit(X_tr, m_tr)

    h_tr = pd.util.hash_pandas_object(X_tr, index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(X_te, index=False).to_numpy()
    overlap = int(np.isin(h_te, h_tr).sum())
    seen_types = set(pre.type_encoder_.classes_)
    unseen_types = sorted(set(m_te["label"]) - seen_types)
    unseen_cat_values = {
        col: sorted(set(X_te[col]) - set(cats))
        for col, cats in zip(CATEGORICAL_COLUMNS, pre.onehot_.categories_)
        if set(X_te[col]) - set(cats)
    }

    def build(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        out = pre.transform_features(X).reset_index(drop=True)
        meta = meta.reset_index(drop=True)
        out["label"] = meta["label"]
        out["attack_category"] = meta["attack_category"]
        out["attack_category_id"] = pre.encode_categories(meta["attack_category"])
        out["attack_type_id"] = pre.encode_attack_types(meta["label"])
        out["is_attack"] = meta["is_attack"]
        if "difficulty_level" in meta:
            out["difficulty_level"] = meta["difficulty_level"]
        return out

    train_df, test_df = build(X_tr, m_tr), build(X_te, m_te)
    processed_dir, artifacts_dir = Path(processed_dir), Path(artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv.gz", index=False, compression="gzip")
    test_df.to_csv(processed_dir / "test.csv.gz", index=False, compression="gzip")
    pre.save(artifacts_dir / "preprocessor.joblib")

    report = {
        "files": {"train": Path(train_path).name, "test": Path(test_path).name},
        "train": rep_tr,
        "test": rep_te,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(pre.feature_names_),
        "feature_names": pre.feature_names_,
        "dropped_columns": pre.dropped_columns_,
        "log_transformed_columns": pre.log_columns_,
        "scaled_columns": pre.scaled_columns_,
        "categorical_cardinalities": {c: int(len(k)) for c, k in zip(CATEGORICAL_COLUMNS, pre.onehot_.categories_)},
        "attack_category_classes": pre.category_encoder_.classes_.tolist(),
        "attack_type_classes_train": pre.type_encoder_.classes_.tolist(),
        "test_attack_types_unseen_in_train": unseen_types,
        "test_rows_with_unseen_attack_type": int((~m_te["label"].isin(seen_types)).sum()),
        "test_unseen_categorical_values": unseen_cat_values,
        "test_rows_identical_to_train_features": overlap,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items() if k != "attack_category_map"},
    }
    (artifacts_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Saved processed data to %s and artifacts to %s", processed_dir, artifacts_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="NSL-KDD preprocessing pipeline")
    parser.add_argument("--raw-dir", default="ml/data/raw/NSL-KDD")
    parser.add_argument("--train-file", default=None, help="Explicit path (e.g. KDDTrain+_20Percent.txt)")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--processed-dir", default="ml/data/processed/nsl_kdd")
    parser.add_argument("--artifacts-dir", default="ml/artifacts/nsl_kdd")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--no-log-transform", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = NSLKDDConfig(scale=not args.no_scale, log_transform=not args.no_log_transform)
    report = run_pipeline(args.raw_dir, args.processed_dir, args.artifacts_dir, cfg, args.train_file, args.test_file)
    print(json.dumps({k: report[k] for k in ("n_train", "n_test", "n_features", "test_attack_types_unseen_in_train")}, indent=2))


if __name__ == "__main__":
    main()

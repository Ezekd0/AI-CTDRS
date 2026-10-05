"""Tests use SMALL SYNTHETIC CSVs that mimic the CICIDS2017 schema (messy headers, inf/NaN,
duplicates, identifiers, a corrupted label dash). They are test fixtures only; no statistics
from the real dataset are implied."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.preprocessing.cicids2017_preprocessor import (
    CICIDS2017Preprocessor,
    PreprocessConfig,
    detect_label_column,
    load_raw_csvs,
    run_pipeline,
)

N_FILE1, N_DUP, N_FILE2 = 100, 5, 46  # 100 + 5 duplicated rows ; 30 + 25 + 1 rare
EXPECTED_CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "Web Attack Brute Force"]


def _block(rng, label, n):
    pkts = rng.integers(1, 100, n)
    return pd.DataFrame({
        "Flow ID": [f"f{rng.integers(10**9)}" for _ in range(n)],
        " Source IP": "10.0.0.1",
        " Source Port": rng.integers(1024, 65535, n),
        " Destination Port": rng.integers(1, 65535, n),
        " Timestamp": "7/7/2017 8:59",
        " Flow Duration": rng.integers(1, 1_000_000, n).astype(float),
        " Total Fwd Packets": pkts,
        " Total Fwd Packets Copy": pkts * 2,  # perfectly correlated -> should be dropped
        " Flow Bytes/s": rng.random(n) * 1e4,
        " Constant Col": 5,  # zero variance -> should be dropped
        " Label": label,
    })


@pytest.fixture()
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "raw"
    d.mkdir()
    f1 = pd.concat([_block(rng, "BENIGN", 50), _block(rng, "DoS Hulk", 20), _block(rng, "PortScan", 15), _block(rng, "Web Attack \ufffd Brute Force", 15)], ignore_index=True)
    f1.loc[0, " Flow Bytes/s"] = np.inf
    f1.loc[1, " Flow Bytes/s"] = -np.inf
    f1.loc[2, " Flow Duration"] = np.nan
    f1 = pd.concat([f1, f1.head(N_DUP)], ignore_index=True)  # exact duplicates
    f2 = pd.concat(
        [_block(rng, "BENIGN", 10), _block(rng, "DoS Hulk", 10), _block(rng, "PortScan", 10), _block(rng, "Web Attack \ufffd Brute Force", 15), _block(rng, "Heartbleed", 1)],
        ignore_index=True,
    )
    f1.to_csv(d / "monday.csv", index=False)
    f2.to_csv(d / "tuesday.csv", index=False)
    return d


def test_dataset_loads_and_columns_normalized(raw_dir):
    raw = load_raw_csvs(raw_dir)
    assert len(raw) == N_FILE1 + N_DUP + N_FILE2
    assert "destination_port" in raw.columns and "flow_bytes_s" in raw.columns
    assert all(c == c.strip().lower() and " " not in c for c in raw.columns)


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csvs(tmp_path)


def test_labels_detected_and_cleaned(raw_dir):
    raw = load_raw_csvs(raw_dir)
    assert detect_label_column(raw) == "label"
    _, y, report = CICIDS2017Preprocessor().prepare(raw)
    assert sorted(y.unique()) == EXPECTED_CLASSES
    assert report["rare_classes_dropped"] == ["Heartbleed"]
    assert report["duplicates_removed"] == N_DUP


def test_missing_label_column_raises():
    with pytest.raises(ValueError):
        detect_label_column(pd.DataFrame({"a": [1]}))


def test_pipeline_outputs_valid_numeric_features(raw_dir, tmp_path):
    out, art = tmp_path / "proc", tmp_path / "art"
    report = run_pipeline(raw_dir, out, art, PreprocessConfig(test_size=0.25))
    train = pd.read_csv(out / "train.csv.gz")
    feats = train.drop(columns=["label", "label_id", "is_attack"])
    assert all(pd.api.types.is_numeric_dtype(t) for t in feats.dtypes)
    assert np.isfinite(feats.to_numpy()).all()
    for dropped in ("flow_id", "source_ip", "source_port", "timestamp", "constant_col", "total_fwd_packets_copy"):
        assert dropped not in feats.columns
    assert report["infinite_cells_replaced"] == 2
    assert (art / "preprocessor.joblib").exists() and (art / "preprocessing_report.json").exists()


def test_train_test_schema_compatible(raw_dir, tmp_path):
    out = tmp_path / "proc"
    run_pipeline(raw_dir, out, tmp_path / "art")
    train, test = pd.read_csv(out / "train.csv.gz"), pd.read_csv(out / "test.csv.gz")
    assert list(train.columns) == list(test.columns)
    assert set(test["label"]) <= set(train["label"])
    assert set(train["is_attack"]) == {0, 1}


def test_split_is_reproducible(raw_dir, tmp_path):
    run_pipeline(raw_dir, tmp_path / "a", tmp_path / "a_art")
    run_pipeline(raw_dir, tmp_path / "b", tmp_path / "b_art")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "train.csv.gz"), pd.read_csv(tmp_path / "b" / "train.csv.gz"))


def test_fit_uses_train_statistics_only(raw_dir):
    pre = CICIDS2017Preprocessor()
    X, y, _ = pre.prepare(load_raw_csvs(raw_dir))
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    pre.fit(X.iloc[tr], y.iloc[tr])
    j = pre.imputer_columns_.index("flow_duration")
    assert pre.imputer_.statistics_[j] == pytest.approx(X.iloc[tr]["flow_duration"].median())
    assert pre.scaler_.mean_[0] == pytest.approx(pre.transform_features(X.iloc[tr], scaled=False).iloc[:, 0].mean())


def test_inference_transform_and_roundtrip(raw_dir, tmp_path):
    pre = CICIDS2017Preprocessor()
    raw = load_raw_csvs(raw_dir)
    X, y, _ = pre.prepare(raw)
    pre.fit(X, y)
    path = pre.save(tmp_path / "pre.joblib")
    loaded = CICIDS2017Preprocessor.load(path)
    sample = raw.head(10).drop(columns=["flow_duration"])  # missing column is imputed, not an error
    a, b = pre.transform_features(sample), loaded.transform_features(sample)
    assert list(a.columns) == pre.feature_names_
    assert np.isfinite(a.to_numpy()).all()
    pd.testing.assert_frame_equal(a, b)
    assert set(loaded.decode_labels(loaded.encode_labels(y))) == set(y)

def test_source_file_groups_never_cross_train_test(raw_dir, tmp_path):
    report = run_pipeline(raw_dir, tmp_path / "proc", tmp_path / "art")
    assert set(report["train_source_files"]).isdisjoint(report["test_source_files"])
    assert report["split_strategy"] == "source_file"

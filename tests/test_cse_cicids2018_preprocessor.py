"""Tests use SMALL SYNTHETIC CSVs that mimic quirks of the 2018 files (abbreviated headers,
a file with extra identifier columns, a repeated header row, 'Benign' casing, the 'Infilteration'
spelling, inf/NaN, timestamp-only duplicates). They are fixtures only; no real-dataset
statistics are implied."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.preprocessing.cse_cicids2018_preprocessor import (
    CSECICIDS2018Config,
    CSECICIDS2018Preprocessor,
    detect_label_column,
    discover_csvs,
    load_raw_csvs,
    run_pipeline,
)

EXPECTED_CLASSES = ["BENIGN", "Bot", "DDOS attack HOIC", "Infiltration"]


def _block(rng, label, n, ids=False, extra=False):
    d = {}
    if ids:
        d.update({"Flow ID": [f"f{rng.integers(10**9)}" for _ in range(n)], "Src IP": "172.31.0.1",
                  "Src Port": rng.integers(1024, 65535, n), "Dst IP": "172.31.0.2"})
    d.update({
        "Dst Port": rng.integers(1, 65535, n),
        "Protocol": rng.choice([6, 17], n),
        "Timestamp": "02/03/2018 08:47:38",
        "Flow Duration": rng.integers(1, 1_000_000, n).astype(float),
        "Tot Fwd Pkts": rng.integers(1, 100, n),
        "TotLen Fwd Pkts": rng.integers(0, 10_000, n),
        "Flow Byts/s": rng.random(n) * 1e4,
        "Flow Pkts/s": rng.random(n) * 1e3,
        "Bwd PSH Flags": 0,  # constant -> dropped
    })
    if extra:
        d["Extra Only B"] = rng.random(n)
    d["Label"] = label
    return pd.DataFrame(d)


@pytest.fixture()
def raw_dir(tmp_path):
    rng = np.random.default_rng(1)
    d = tmp_path / "raw"
    (d / "sub").mkdir(parents=True)  # discovery must be recursive
    a = pd.concat([_block(rng, "Benign", 50), _block(rng, "Bot", 20), _block(rng, "DDOS attack-HOIC", 15), _block(rng, "Infilteration", 15)], ignore_index=True)
    a.loc[10, "Flow Byts/s"] = np.inf
    a.loc[11, "Flow Byts/s"] = -np.inf
    a.loc[12, "Flow Pkts/s"] = np.nan
    dups = a.head(4).copy()
    dups["Timestamp"] = "03/03/2018 09:00:00"  # differs ONLY by timestamp
    a = pd.concat([a, dups], ignore_index=True)
    lines = a.to_csv(index=False).splitlines()
    lines.insert(30, lines[0])  # repeated header row inside the data
    (d / "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv").write_text("\n".join(lines) + "\n")

    b = pd.concat(
        [_block(rng, "Benign", 10, True, True), _block(rng, "Bot", 10, True, True), _block(rng, "DDOS attack-HOIC", 10, True, True),
         _block(rng, "Infilteration", 25, True, True), _block(rng, "SQL Injection", 1, True, True)],
        ignore_index=True,
    )
    b.to_csv(d / "sub" / "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", index=False)
    return d


def test_csv_discovery_recursive_and_empty(raw_dir, tmp_path):
    assert [p.name for p in discover_csvs(raw_dir)] == [
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    ]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_csvs(empty)


def test_schema_discovered_not_assumed(raw_dir):
    frame, schema = load_raw_csvs(raw_dir)
    assert {"flow_id", "src_ip", "src_port", "dst_ip", "extra_only_b"} <= set(schema["columns_not_in_all_files"])
    assert "dst_port" in frame.columns and "tot_fwd_pkts" in frame.columns  # abbreviated 2018 names
    assert "total_fwd_packets" not in frame.columns and "destination_port" not in frame.columns
    assert not {"flow_id", "src_ip", "timestamp"} & set(frame.columns)  # identifiers removed
    assert schema["per_file"][1]["identifier_columns_present"] == ["dst_ip", "flow_id", "src_ip", "src_port", "timestamp"]
    assert schema["per_file"][0]["identifier_columns_present"] == ["timestamp"]


def test_repeated_header_rows_inf_and_timestamp_duplicates(raw_dir):
    frame, schema = load_raw_csvs(raw_dir)
    t = schema["totals"]
    assert t["repeated_header_rows_removed"] == 1
    assert t["infinite_cells_replaced"] == 2
    assert t["duplicates_removed_within_file"] == 4  # identical once the timestamp is dropped
    feats = frame.drop(columns="label")
    assert (feats.dtypes == "float32").all()
    assert not np.isinf(feats.to_numpy()).any()
    assert "Label" not in set(frame["label"])


def test_labels_detected_aliased_and_rare_dropped(raw_dir):
    frame, _ = load_raw_csvs(raw_dir)
    assert detect_label_column(pd.DataFrame({"label": [1]})) == "label"
    with pytest.raises(ValueError):
        detect_label_column(pd.DataFrame({"a": [1]}))
    _, y, report = CSECICIDS2018Preprocessor().prepare(frame)
    assert sorted(y.unique()) == EXPECTED_CLASSES  # 'Benign'->BENIGN, 'Infilteration'->Infiltration
    assert report["rare_classes_dropped"] == ["SQL Injection"]
    assert sum(v["count"] for v in report["class_distribution"].values()) == report["rows_after_cleaning"]


def test_class_cap_is_applied_and_seeded(raw_dir):
    frame, _ = load_raw_csvs(raw_dir)
    cfg = CSECICIDS2018Config(max_rows_per_class=20)
    _, y1, rep = CSECICIDS2018Preprocessor(cfg).prepare(frame)
    _, y2, _ = CSECICIDS2018Preprocessor(cfg).prepare(frame)
    assert y1.value_counts().max() <= 20
    assert max(v["count"] for v in rep["class_distribution_before_filters"].values()) > 20
    pd.testing.assert_series_equal(y1, y2)


def test_pipeline_outputs_valid_numeric_features(raw_dir, tmp_path):
    out, art = tmp_path / "proc", tmp_path / "art"
    report = run_pipeline(raw_dir, out, art, CSECICIDS2018Config(test_size=0.25))
    train = pd.read_csv(out / "train.csv.gz")
    feats = train.drop(columns=["label", "label_id", "is_attack"])
    assert all(pd.api.types.is_numeric_dtype(t) for t in feats.dtypes)
    assert np.isfinite(feats.to_numpy()).all()
    assert not {"flow_id", "src_ip", "timestamp", "bwd_psh_flags"} & set(feats.columns)
    assert (art / "preprocessor.joblib").exists() and (art / "preprocessing_report.json").exists()
    assert report["n_train"] + report["n_test"] == report["rows_after_cleaning"]


def test_high_missing_columns_dropped(raw_dir):
    frame, _ = load_raw_csvs(raw_dir)  # 'extra_only_b' exists in only one of the two files
    pre = CSECICIDS2018Preprocessor(CSECICIDS2018Config(max_missing_fraction=0.3))
    X, y, _ = pre.prepare(frame)
    pre.fit(X, y)
    assert "extra_only_b" in pre.dropped_columns_["high_missing"]
    assert "extra_only_b" not in pre.feature_names_


def test_train_test_schema_compatible_and_reproducible(raw_dir, tmp_path):
    run_pipeline(raw_dir, tmp_path / "a", tmp_path / "a_art")
    run_pipeline(raw_dir, tmp_path / "b", tmp_path / "b_art")
    train, test = pd.read_csv(tmp_path / "a" / "train.csv.gz"), pd.read_csv(tmp_path / "a" / "test.csv.gz")
    assert list(train.columns) == list(test.columns)
    assert set(test["label"]) <= set(train["label"])
    assert set(train["is_attack"]) == {0, 1}
    pd.testing.assert_frame_equal(train, pd.read_csv(tmp_path / "b" / "train.csv.gz"))


def test_fit_uses_train_statistics_only(raw_dir):
    frame, _ = load_raw_csvs(raw_dir)
    pre = CSECICIDS2018Preprocessor()
    X, y, _ = pre.prepare(frame)
    tr, _ = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    pre.fit(X.iloc[tr], y.iloc[tr])
    j = pre.imputer_columns_.index("flow_duration")
    assert pre.imputer_.statistics_[j] == pytest.approx(X.iloc[tr]["flow_duration"].median(), rel=1e-5)


def test_inference_with_raw_headers_missing_column_and_roundtrip(raw_dir, tmp_path):
    frame, _ = load_raw_csvs(raw_dir)
    pre = CSECICIDS2018Preprocessor()
    X, y, _ = pre.prepare(frame)
    pre.fit(X, y)
    loaded = CSECICIDS2018Preprocessor.load(pre.save(tmp_path / "pre.joblib"))
    raw_b = pd.read_csv(next((raw_dir / "sub").glob("*.csv"))).head(8)  # raw, un-normalised headers + IPs
    sample = raw_b.drop(columns=["Flow Duration"])  # missing column is imputed, not an error
    a, b = pre.transform_features(sample), loaded.transform_features(sample)
    assert list(a.columns) == pre.feature_names_
    assert np.isfinite(a.to_numpy()).all()
    pd.testing.assert_frame_equal(a, b)
    assert set(loaded.decode_labels(loaded.encode_labels(y))) == set(y)
    assert set(loaded.binary_labels(pd.Series(["Benign", "Infilteration"]))) == {0, 1}

def test_source_file_groups_never_cross_train_test(raw_dir, tmp_path):
    report = run_pipeline(raw_dir, tmp_path / "proc", tmp_path / "art")
    assert set(report["train_source_files"]).isdisjoint(report["test_source_files"])
    assert report["split_strategy"] == "source_file"

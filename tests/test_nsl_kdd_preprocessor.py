"""Tests use SMALL SYNTHETIC files in the NSL-KDD layout (headerless, 41 features + label +
difficulty). They are fixtures only; no real-dataset statistics are implied."""
import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.nsl_kdd_preprocessor import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    NSLKDDConfig,
    NSLKDDPreprocessor,
    find_split_files,
    load_nsl_kdd,
    run_pipeline,
)

TRAIN_SPEC = {"normal": 60, "neptune": 30, "satan": 20, "guess_passwd": 15, "buffer_overflow": 10}
TEST_SPEC = {"normal": 25, "neptune": 15, "satan": 10, "mscan": 8, "worm": 5, "guess_passwd": 5, "buffer_overflow": 3}
BINARY = {"land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"}


def _frame(rng, spec):
    rows = []
    for label, n in spec.items():
        d = {}
        for c in FEATURE_COLUMNS:
            if c == "protocol_type":
                d[c] = rng.choice(["tcp", "udp", "icmp"], n)
            elif c == "service":
                d[c] = np.full(n, "telnet") if label == "mscan" else rng.choice(["http", "ftp", "smtp", "private"], n)
            elif c == "flag":
                d[c] = rng.choice(["SF", "S0", "REJ"], n)
            elif c in ("num_outbound_cmds", "is_host_login"):
                d[c] = np.zeros(n, dtype=int)  # constant -> should be dropped
            elif c in BINARY:
                d[c] = rng.integers(0, 2, n)
            elif c.endswith("rate"):
                d[c] = np.round(rng.random(n), 4)
            elif c in ("duration", "src_bytes", "dst_bytes"):
                d[c] = rng.integers(0, 10**7, n)
            else:
                d[c] = rng.integers(0, 500, n)
        d["label"] = label
        d["difficulty_level"] = rng.integers(1, 22, n)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def files(tmp_path):
    rng = np.random.default_rng(3)
    d = tmp_path / "raw"
    d.mkdir()
    train = _frame(rng, TRAIN_SPEC)
    train_file = pd.concat([train, train.head(3)], ignore_index=True)  # 3 exact duplicates
    test = _frame(rng, TEST_SPEC)
    test = pd.concat([test, train.iloc[[5]]], ignore_index=True)  # one row identical to a train row
    test.loc[2, "src_bytes"] = "abc"  # invalid numeric
    test.loc[3, "duration"] = "inf"  # infinite
    test.loc[4, "flag"] = ""  # missing categorical
    train_file.to_csv(d / "KDDTrain+.txt", header=False, index=False)
    test.to_csv(d / "KDDTest+.txt", header=False, index=False)
    train_file.head(10).to_csv(d / "KDDTrain+_20Percent.txt", header=False, index=False)
    return d


def test_find_split_files_ignores_20percent_and_errors_when_missing(files, tmp_path):
    train, test = find_split_files(files)
    assert train.name == "KDDTrain+.txt" and test.name == "KDDTest+.txt"
    with pytest.raises(FileNotFoundError):
        find_split_files(tmp_path / "nothing")


def test_load_headerless_and_header_variants(files, tmp_path):
    df = load_nsl_kdd(files / "KDDTrain+.txt")
    assert list(df.columns) == [*FEATURE_COLUMNS, "label", "difficulty_level"]
    assert len(df) == sum(TRAIN_SPEC.values()) + 3
    with_header = _frame(np.random.default_rng(0), {"normal": 4, "neptune": 4}).rename(columns=str.title)
    with_header.to_csv(tmp_path / "hdr.csv", index=False)
    assert list(load_nsl_kdd(tmp_path / "hdr.csv").columns) == [*FEATURE_COLUMNS, "label", "difficulty_level"]


def test_wrong_layout_raises(tmp_path):
    pd.DataFrame(np.ones((3, 10))).to_csv(tmp_path / "bad.txt", header=False, index=False)
    with pytest.raises(ValueError):
        load_nsl_kdd(tmp_path / "bad.txt")


def test_label_mapping_binary_and_unknown_label(files):
    pre = NSLKDDPreprocessor()
    cats = pre.label_to_category(pd.Series(["Normal.", "neptune", "mscan", "worm", "guess_passwd", "perl"]))
    assert cats.tolist() == ["normal", "DoS", "Probe", "DoS", "R2L", "U2R"]
    assert pre.binary_labels(pd.Series(["normal", "worm"])).tolist() == [0, 1]
    raw = load_nsl_kdd(files / "KDDTrain+.txt")
    raw.loc[0, "label"] = "totally_new_attack"
    with pytest.raises(ValueError, match="totally_new_attack"):
        pre.prepare(raw)


def test_invalid_values_counted_and_train_only_dedupe(files):
    pre = NSLKDDPreprocessor()
    _, _, rep_tr = pre.prepare(load_nsl_kdd(files / "KDDTrain+.txt"), deduplicate=True)
    X_te, meta_te, rep_te = pre.prepare(load_nsl_kdd(files / "KDDTest+.txt"), deduplicate=False)
    assert rep_tr["duplicates_removed"] == 3
    assert rep_te["duplicates_removed"] == 0 and len(X_te) == sum(TEST_SPEC.values()) + 1
    assert (rep_te["invalid_numeric_cells"], rep_te["infinite_cells_replaced"], rep_te["missing_categorical_cells"]) == (1, 1, 1)
    assert set(meta_te["attack_category"]) == {"normal", "DoS", "Probe", "R2L", "U2R"}


def test_pipeline_outputs_binary_and_multiclass_targets(files, tmp_path):
    out, art = tmp_path / "proc", tmp_path / "art"
    rep = run_pipeline(files, out, art)
    train, test = pd.read_csv(out / "train.csv.gz"), pd.read_csv(out / "test.csv.gz")
    target_cols = ["label", "attack_category", "attack_category_id", "attack_type_id", "is_attack", "difficulty_level"]
    feats = train.drop(columns=target_cols)
    assert list(train.columns) == list(test.columns)
    assert np.isfinite(feats.to_numpy()).all() and np.isfinite(test.drop(columns=target_cols).to_numpy()).all()
    assert {"protocol_type_tcp", "service_http", "flag_sf"} <= set(feats.columns)
    assert not {"num_outbound_cmds", "is_host_login"} & set(feats.columns)
    assert rep["n_train"] == sum(TRAIN_SPEC.values()) and rep["n_test"] == len(test)  # official split kept
    for df in (train, test):
        assert ((df["attack_category"] != "normal").astype(int) == df["is_attack"]).all()
    assert set(rep["attack_category_classes"]) == {"DoS", "Probe", "R2L", "U2R", "normal"}
    assert rep["test_attack_types_unseen_in_train"] == ["mscan", "worm"]
    assert (test.loc[test["label"].isin(["mscan", "worm"]), "attack_type_id"] == -1).all()
    assert (test.loc[test["label"] == "mscan", "attack_category_id"] >= 0).all()
    assert rep["test_rows_identical_to_train_features"] == 1
    assert rep["test_unseen_categorical_values"] == {"service": ["telnet"], "flag": ["unknown"]}  # "unknown" = the blanked flag
    assert (art / "preprocessor.joblib").exists() and (art / "preprocessing_report.json").exists()


def test_scaling_applies_only_to_continuous_columns(files, tmp_path):
    out = tmp_path / "proc"
    run_pipeline(files, out, tmp_path / "art")
    train = pd.read_csv(out / "train.csv.gz")
    for c in ("count", "src_bytes", "serror_rate"):
        assert train[c].mean() == pytest.approx(0, abs=1e-8) and train[c].std(ddof=0) == pytest.approx(1, rel=1e-6)
    assert set(train["logged_in"].unique()) <= {0, 1}  # binary flag untouched
    assert set(train["service_http"].unique()) <= {0.0, 1.0}  # one-hot untouched


def test_fit_uses_train_statistics_only(files):
    pre = NSLKDDPreprocessor()
    X, meta, _ = pre.prepare(load_nsl_kdd(files / "KDDTrain+.txt"), deduplicate=True)
    pre.fit(X, meta)
    j = NUMERIC_COLUMNS.index("duration")
    assert pre.imputer_.statistics_[j] == pytest.approx(X["duration"].median())
    k = pre.scaled_columns_.index("duration")
    assert pre.scaler_.mean_[k] == pytest.approx(np.log1p(X["duration"]).mean())
    assert "telnet" not in pre.onehot_.categories_[CATEGORICAL_COLUMNS.index("service")]  # test-only value


def test_same_transformation_applies_to_inference_data(files, tmp_path):
    out, art = tmp_path / "proc", tmp_path / "art"
    run_pipeline(files, out, art)
    pre = NSLKDDPreprocessor.load(art / "preprocessor.joblib")
    # raw KDDTest+ through the loaded preprocessor == the batch-processed test features
    batch = pd.read_csv(out / "test.csv.gz")[pre.feature_names_]
    via_inference = pre.transform_features(load_nsl_kdd(files / "KDDTest+.txt"))
    pd.testing.assert_frame_equal(via_inference.reset_index(drop=True), batch, rtol=1e-9, atol=1e-9)

    # a single partial record: any header casing, unseen service, most columns absent
    rec = pd.DataFrame([{"Duration": 0, "Protocol_Type": "TCP", "Service": "Telnet", "Flag": "SF",
                         "Src_Bytes": 181, "Dst_Bytes": 5450}])
    row = pre.transform_features(rec)
    assert list(row.columns) == pre.feature_names_ and len(row) == 1
    assert np.isfinite(row.to_numpy()).all()
    assert row["protocol_type_tcp"].iloc[0] == 1
    assert row[[c for c in row.columns if c.startswith("service_")]].to_numpy().sum() == 0  # unseen -> all zero
    assert pre.transform_features(rec, scaled=False)["src_bytes"].iloc[0] == 181  # original units


def test_label_encoders_and_unseen_attack_types(files, tmp_path):
    run_pipeline(files, tmp_path / "proc", tmp_path / "art")
    pre = NSLKDDPreprocessor.load(tmp_path / "art" / "preprocessor.joblib")
    ids = pre.encode_attack_types(pd.Series(["neptune", "mscan"]))
    assert ids[0] >= 0 and ids[1] == -1
    cat_ids = pre.encode_categories(pd.Series(["neptune", "normal", "mscan"]), from_attack_names=True)
    assert pre.decode_categories(cat_ids).tolist() == ["DoS", "normal", "Probe"]

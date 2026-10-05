"""Training-pipeline tests on SMALL SYNTHETIC data (fixtures only; no real-dataset statistics are implied).

Raw synthetic files go through the real preprocessing pipelines, then through ``train_one``. Assertions check
structure and internal consistency (e.g. the confusion matrix reproduces the reported accuracy); none of them
pin a metric value, because metrics are always computed at run time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.evaluation.metrics import evaluate_predictions
from ml.preprocessing import cicids2017_preprocessor as c17
from ml.preprocessing import nsl_kdd_preprocessor as nsl
from ml.training.artifacts import list_public_summaries, load_bundle, public_summary
from ml.training.datasets import (
    binary_label_encoder, build_task_data, class_weights, load_dataset, resolve_dataset, stratified_holdout,
)
from ml.training.sequences import sequence_shape, to_sequences
from ml.training.train import main, train_one

NSL_TRAIN = {"normal": 150, "neptune": 90, "satan": 60, "guess_passwd": 40, "buffer_overflow": 25}
NSL_TEST = {"normal": 60, "neptune": 40, "satan": 30, "guess_passwd": 20, "buffer_overflow": 6}
BINARY_FLAGS = {"land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"}


def _nsl_frame(rng, spec):
    shift = {"normal": 0, "neptune": 3, "satan": 6, "guess_passwd": 9, "buffer_overflow": 12}
    rows = []
    for label, n in spec.items():
        d = {}
        for c in nsl.FEATURE_COLUMNS:
            if c == "protocol_type":
                d[c] = rng.choice(["tcp", "udp", "icmp"], n)
            elif c == "service":
                d[c] = rng.choice(["http", "ftp", "smtp", "private"], n)
            elif c == "flag":
                d[c] = rng.choice(["SF", "S0", "REJ"], n)
            elif c in BINARY_FLAGS:
                d[c] = rng.integers(0, 2, n)
            elif c.endswith("rate"):
                d[c] = np.round(np.clip(rng.normal(0.1 * shift[label] / 3, 0.15, n), 0, 1), 4)
            else:
                d[c] = np.round(rng.normal(50 + 20 * shift[label], 25, n)).clip(0).astype(int)
        d["label"], d["difficulty_level"] = label, rng.integers(1, 22, n)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def _c17_frame(rng, label, n, shift):
    return pd.DataFrame({
        "Flow ID": [f"f{i}" for i in range(n)], " Source IP": "10.0.0.1", " Source Port": rng.integers(1024, 65535, n),
        " Destination Port": rng.integers(1, 65535, n), " Timestamp": "7/7/2017 8:59",
        " Flow Duration": np.abs(rng.normal(1e5 * (1 + shift), 5e4, n)),
        " Total Fwd Packets": np.abs(rng.normal(10 + 30 * shift, 8, n)).round(),
        " Flow Bytes/s": np.abs(rng.normal(1e3 * (1 + 2 * shift), 400, n)),
        " Idle Mean": rng.random(n) * 100, " Label": label,
    })


@pytest.fixture()
def roots(tmp_path):
    return {"processed": tmp_path / "processed", "artifacts": tmp_path / "artifacts", "reports": tmp_path / "reports" / "ml"}


@pytest.fixture()
def nsl_ready(tmp_path, roots):
    rng = np.random.default_rng(1)
    raw = tmp_path / "raw_nsl"
    raw.mkdir()
    _nsl_frame(rng, NSL_TRAIN).to_csv(raw / "KDDTrain+.txt", header=False, index=False)
    _nsl_frame(rng, NSL_TEST).to_csv(raw / "KDDTest+.txt", header=False, index=False)
    nsl.run_pipeline(raw, roots["processed"] / "nsl_kdd", roots["artifacts"] / "nsl_kdd")
    return roots


@pytest.fixture()
def c17_ready(tmp_path, roots):
    rng = np.random.default_rng(2)
    raw = tmp_path / "raw_c17"
    raw.mkdir()
    a = pd.concat([_c17_frame(rng, "BENIGN", 150, 0), _c17_frame(rng, "DoS Hulk", 75, 1), _c17_frame(rng, "PortScan", 50, 2), _c17_frame(rng, "Bot", 20, 3)], ignore_index=True)
    b = pd.concat([_c17_frame(rng, "BENIGN", 150, 0), _c17_frame(rng, "DoS Hulk", 75, 1), _c17_frame(rng, "PortScan", 50, 2), _c17_frame(rng, "Bot", 20, 3)], ignore_index=True)
    # Two capture groups with the same evaluated classes let the leakage-safe source-file split run.
    a.to_csv(raw / "capture_a.csv", index=False)
    b.to_csv(raw / "capture_b.csv", index=False)
    c17.run_pipeline(raw, roots["processed"] / "cicids2017", roots["artifacts"] / "cicids2017")
    return roots


def _run(dataset, model, task, roots, **kw):
    return train_one(dataset, model, task, processed_root=roots["processed"], artifacts_root=roots["artifacts"],
                     reports_root=roots["reports"], **kw)


def _check_outputs(slug, task, model, roots, n_classes):
    adir, rdir = roots["artifacts"] / slug / task / model, roots["reports"] / slug / task / model
    for name in ("feature_names.json", "label_encoder.joblib", "label_mapping.json", "metadata.json", "metrics.json",
                 "preprocessor.joblib", "preprocessing_report.json"):
        assert (adir / name).exists(), name
    for name in ("metrics.json", "classification_report.txt", "classification_report.json", "confusion_matrix.csv", "run_info.json"):
        assert (rdir / name).exists(), name
    metrics = json.loads((rdir / "metrics.json").read_text())
    cm = np.array(metrics["confusion_matrix"]["counts"])
    assert cm.shape == (n_classes, n_classes)
    assert cm.sum() == metrics["n_samples"]
    assert metrics["accuracy"] == pytest.approx(np.trace(cm) / cm.sum())
    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert 0.0 <= metrics[key] <= 1.0
    return adir, rdir, metrics


def test_nsl_kdd_random_forest_binary_and_multiclass(nsl_ready):
    pre = nsl.NSLKDDPreprocessor.load(nsl_ready["artifacts"] / "nsl_kdd" / "preprocessor.joblib")

    r = _run("nsl-kdd", "random_forest", "binary", nsl_ready)
    adir, rdir, m = _check_outputs("nsl_kdd", "binary", "random_forest", nsl_ready, 2)
    assert m["average"] == "binary" and 0.0 <= m["roc_auc"]["binary"] <= 1.0
    assert (rdir / "roc_curve.csv").exists()
    assert json.loads((adir / "feature_names.json").read_text()) == pre.feature_names_
    assert list(joblib.load(adir / "label_encoder.joblib").classes_) == ["normal", "ATTACK"]

    _run("nsl-kdd", "random_forest", "multiclass", nsl_ready)
    adir, rdir, m = _check_outputs("nsl_kdd", "multiclass", "random_forest", nsl_ready, 5)
    assert m["average"] == "macro" and 0.0 <= m["roc_auc"]["macro_ovr"] <= 1.0
    assert m["confusion_matrix"]["labels"] == list(pre.category_encoder_.classes_)
    assert set(m["classification_report"]) >= set(m["confusion_matrix"]["labels"])
    assert r["model"] == "random_forest" and "accuracy" in r

    summary = pd.read_csv(nsl_ready["reports"] / "summary.csv")
    assert set(summary["task"]) == {"binary", "multiclass"} and len(summary) == 2


def test_cicids2017_multiclass_and_stratified_validation(c17_ready):
    loaded = load_dataset(resolve_dataset("cicids2017"), c17_ready["processed"], c17_ready["artifacts"])
    td = build_task_data(loaded, "multiclass", val_size=0.15, seed=7)
    assert td.info["validation_split"]["stratified"] and td.has_validation
    # every class keeps proportional presence in train and validation; test is untouched
    for name, n_val in td.info["val_distribution"].items():
        assert n_val >= 1 and td.info["train_distribution"][name] > n_val
    assert td.info["n_test"] == len(loaded.test)
    assert "Heartbleed" not in td.class_names  # single-sample class is removed by the preprocessor

    _run("cicids2017", "random_forest", "multiclass", c17_ready, val_size=0.15)
    _check_outputs("cicids2017", "multiclass", "random_forest", c17_ready, len(td.class_names))


def test_same_seed_gives_identical_metrics_and_different_seed_is_recorded(nsl_ready):
    _run("nsl-kdd", "random_forest", "multiclass", nsl_ready, seed=11, param_overrides={"n_estimators": 30})
    path = nsl_ready["reports"] / "nsl_kdd" / "multiclass" / "random_forest" / "metrics.json"
    first = path.read_text()
    _run("nsl-kdd", "random_forest", "multiclass", nsl_ready, seed=11, param_overrides={"n_estimators": 30})
    assert path.read_text() == first
    meta = json.loads((nsl_ready["artifacts"] / "nsl_kdd" / "multiclass" / "random_forest" / "metadata.json").read_text())
    assert meta["seed"] == 11 and meta["model_params"]["n_estimators"] == 30
    assert all(len(v["sha256"]) == 64 for v in meta["data_files"].values())
    assert "validation_metrics" in meta and "libraries" in meta


def test_metadata_is_not_exposed_with_paths_and_reports_hold_no_model_files(nsl_ready):
    _run("nsl-kdd", "random_forest", "binary", nsl_ready, param_overrides={"n_estimators": 20})
    adir = nsl_ready["artifacts"] / "nsl_kdd" / "binary" / "random_forest"
    summary = public_summary(adir)
    blob = json.dumps(summary)
    assert str(nsl_ready["artifacts"]) not in blob and "command" not in summary and ".joblib" not in blob
    assert len(list_public_summaries(nsl_ready["artifacts"])) == 1
    forbidden = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".h5", ".bin"}
    assert not [p for p in nsl_ready["reports"].rglob("*") if p.suffix in forbidden]


def test_bundle_reload_reproduces_predictions_and_blocks_path_escape(nsl_ready):
    _run("nsl-kdd", "random_forest", "multiclass", nsl_ready, param_overrides={"n_estimators": 20})
    bundle = load_bundle(nsl_ready["artifacts"], "nsl_kdd", "multiclass", "random_forest")
    test = pd.read_csv(nsl_ready["processed"] / "nsl_kdd" / "test.csv.gz")
    proba = bundle.predict_proba(test[bundle.feature_names].to_numpy())
    assert proba.shape == (len(test), len(bundle.class_names))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    cm = np.array(json.loads((nsl_ready["reports"] / "nsl_kdd" / "multiclass" / "random_forest" / "metrics.json").read_text())
                  ["confusion_matrix"]["counts"])
    assert (proba.argmax(axis=1) == test["attack_category_id"].to_numpy()).sum() == np.trace(cm)
    with pytest.raises(ValueError):
        load_bundle(nsl_ready["artifacts"], "..", "..", "..")


def test_cli_entry_point_and_param_validation(nsl_ready):
    argv = ["--dataset", "nsl-kdd", "--model", "random_forest", "--task", "binary", "--param", "n_estimators=15",
            "--processed-root", str(nsl_ready["processed"]), "--artifacts-root", str(nsl_ready["artifacts"]),
            "--reports-dir", str(nsl_ready["reports"])]
    assert main(argv) == 0
    assert (nsl_ready["reports"] / "nsl_kdd" / "binary" / "random_forest" / "metrics.json").exists()
    with pytest.raises(SystemExit):
        main([*argv, "--param", "not_a_param=1"])


def test_missing_preprocessed_data_gives_actionable_error(roots):
    with pytest.raises(FileNotFoundError, match="ml.preprocessing"):
        _run("cicids2017", "random_forest", "binary", roots)


def test_metrics_are_computed_from_predictions():
    y = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3)[y]
    m = evaluate_predictions(y, perfect, ["a", "b", "c"])["metrics"]
    assert m["accuracy"] == 1.0 and m["roc_auc"]["macro_ovr"] == 1.0
    wrong = np.eye(3)[(y + 1) % 3]
    assert evaluate_predictions(y, wrong, ["a", "b", "c"])["metrics"]["accuracy"] == 0.0
    single = evaluate_predictions(np.zeros(4, dtype=int), np.tile([0.9, 0.1], (4, 1)), ["x", "y"])["metrics"]
    assert single["roc_auc"]["binary"] is None  # undefined -> None, never invented


def test_stratified_holdout_keeps_rare_classes_in_train_and_class_weights_normalised():
    strata = np.array([0] * 50 + [1] * 30 + [2] * 1)
    keep, hold, info = stratified_holdout(strata, 0.2, seed=0)
    assert info["stratified"] and 2 not in strata[hold] and 2 in strata[keep]
    assert set(keep).isdisjoint(hold) and len(keep) + len(hold) == len(strata)
    w = class_weights(strata, 3, "balanced")
    assert (w * np.bincount(strata)).sum() / len(strata) == pytest.approx(1.0)
    assert w[2] > w[1] > w[0]
    assert class_weights(strata, 3, "none") is None
    _, none_held, info = stratified_holdout(np.array([0, 1, 0, 1]), 0.1, seed=0)
    assert not info["stratified"] and len(none_held) == 0


def test_binary_label_encoder_order_is_benign_then_attack():
    enc = binary_label_encoder("BENIGN")
    assert list(enc.inverse_transform([0, 1])) == ["BENIGN", "ATTACK"]
    assert list(enc.transform(["ATTACK", "BENIGN"])) == [1, 0]


def test_sequence_reshaping():
    assert sequence_shape(10, 4) == (3, 4, 2)
    X = np.arange(20, dtype=np.float32).reshape(2, 10)
    seq = to_sequences(X, 4)
    assert seq.shape == (2, 3, 4) and seq.dtype == np.float32
    np.testing.assert_array_equal(seq.reshape(2, -1)[:, :10], X)
    assert (seq.reshape(2, -1)[:, 10:] == 0).all()


def test_xgboost_binary_multiclass_and_reload(nsl_ready):
    pytest.importorskip("xgboost")
    small = {"n_estimators": 30, "early_stopping_rounds": 5}
    for task, n in (("binary", 2), ("multiclass", 5)):
        _run("nsl-kdd", "xgboost", task, nsl_ready, param_overrides=small)
        adir, _, m = _check_outputs("nsl_kdd", task, "xgboost", nsl_ready, n)
        assert (adir / "model.json").exists()
        bundle = load_bundle(nsl_ready["artifacts"], "nsl_kdd", task, "xgboost")
        test = pd.read_csv(nsl_ready["processed"] / "nsl_kdd" / "test.csv.gz")
        proba = bundle.predict_proba(test[bundle.feature_names].to_numpy())
        pred = proba.argmax(axis=1)
        truth = test["is_attack" if task == "binary" else "attack_category_id"].to_numpy()
        assert (pred == truth).mean() == pytest.approx(m["accuracy"])


def test_lstm_binary_multiclass_and_reload(nsl_ready):
    pytest.importorskip("torch")
    small = {"epochs": 3, "batch_size": 64, "hidden_size": 16, "features_per_step": 8, "device": "cpu"}
    for task, n in (("binary", 2), ("multiclass", 5)):
        _run("nsl-kdd", "lstm", task, nsl_ready, param_overrides=small)
        adir, _, m = _check_outputs("nsl_kdd", task, "lstm", nsl_ready, n)
        assert (adir / "model.pt").exists()
        bundle = load_bundle(nsl_ready["artifacts"], "nsl_kdd", task, "lstm")
        test = pd.read_csv(nsl_ready["processed"] / "nsl_kdd" / "test.csv.gz")
        proba = bundle.predict_proba(test[bundle.feature_names].to_numpy())
        truth = test["is_attack" if task == "binary" else "attack_category_id"].to_numpy()
        assert (proba.argmax(axis=1) == truth).mean() == pytest.approx(m["accuracy"])

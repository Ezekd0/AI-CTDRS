from pathlib import Path

from app.api.v1.routes.model_evaluations import _load_run


def test_load_run_exposes_measured_metrics(tmp_path: Path):
    run = tmp_path / "cicids2017" / "binary" / "random_forest"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        '{"n_samples": 10, "average": "binary", "accuracy": 0.9, '
        '"precision": 0.8, "recall": 0.7, "f1_score": 0.75, '
        '"balanced_accuracy": 0.8, "roc_auc": {"binary": 0.91, "average_precision": 0.88}, '
        '"classification_report": {}, "confusion_matrix": {"labels": ["benign", "ATTACK"], "counts": [[4,1],[1,4]], "row_normalized": [[0.8,0.2],[0.2,0.8]]}, "support": {"benign":5,"ATTACK":5}}'
    )
    (run / "run_info.json").write_text('{"created_at_utc": "2026-10-03T00:00:00Z"}')
    result = _load_run(run)
    assert result["accuracy"] == 0.9
    assert result["f1_score"] == 0.75
    assert result["roc_auc"] == 0.91
    assert result["average_precision"] == 0.88

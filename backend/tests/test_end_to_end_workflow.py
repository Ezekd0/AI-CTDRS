"""Integration coverage for the production detection chain.

The fixture trains a real RandomForest on a tiny numeric network-feature fixture,
then exercises the same artifact loader, persistence, explanations, alerting,
response simulation, dashboard and report code used by the API.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Alert, Base, Detection, Explanation, ResponseAction, User
from app.db.session import get_engine
from app.modules.prediction.service import predict
from app.schemas.common import PredictionRequest
from app.api.v1.routes.dashboard import dashboard
from app.api.v1.routes.detections import detection_audit
from app.api.v1.routes.reports import create_detection_report
from app.api.v1.routes.responses import ResponseActionRequest, ResponseActionType, execute_response_action


def _write_artifact(root: Path) -> None:
    directory = root / 'nsl_kdd' / 'binary' / 'random_forest'
    directory.mkdir(parents=True)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)
    joblib.dump(model, directory / 'model.joblib')
    joblib.dump(FunctionTransformer(validate=False), directory / 'preprocessor.joblib')
    encoder = LabelEncoder().fit(['normal', 'ATTACK'])
    joblib.dump(encoder, directory / 'label_encoder.joblib')
    (directory / 'feature_names.json').write_text(json.dumps(['x1', 'x2']))
    (directory / 'metrics.json').write_text(json.dumps({
        'accuracy': 1, 'precision': 1, 'recall': 1, 'f1_score': 1, 'average': 'binary',
        'roc_auc': {'binary': 1}, 'confusion_matrix': {'labels': ['normal', 'ATTACK'], 'counts': [[4, 0], [0, 4]]},
    }))
    (directory / 'metadata.json').write_text(json.dumps({
        'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest',
        'class_names': ['normal', 'ATTACK'], 'n_features': 2, 'artifact_files': {'model': 'model.joblib'},
    }))


def test_complete_detection_chain(tmp_path, monkeypatch):
    monkeypatch.setenv('DATABASE_URL', f"sqlite:///{tmp_path / 'integration.db'}")
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path / 'artifacts'))
    monkeypatch.setenv('REPORTS_ROOT', str(tmp_path / 'reports'))
    monkeypatch.setenv('ALERT_PROBABILITY_THRESHOLD', '0.5')
    get_settings.cache_clear()
    get_engine.cache_clear()
    _write_artifact(tmp_path / 'artifacts')
    Base.metadata.create_all(get_engine())

    user = User(id='integration-user', email='integration@example.com', role='analyst', is_active=True)
    with Session(get_engine()) as db:
        db.add(user)
        db.commit()
    user_id = 'integration-user'

    # 1-6: input -> validation/selection -> preprocessing -> prediction -> confidence -> persistence.
    request = PredictionRequest(
        dataset='nsl-kdd', task='binary', model='random_forest',
        features={'x1': 3, 'x2': 1}, generate_shap=True, generate_lime=False,
    )
    result = predict(request, user=type('UserRef', (), {'id': user_id})())
    detection_id = result['detection_id']
    assert result['prediction'] == 'ATTACK'
    assert 0 <= result['probability'] <= 1

    with Session(get_engine()) as db:
        detection = db.get(Detection, detection_id)
        assert detection is not None
        assert detection.transformed_features == [3.0, 1.0]
        assert db.scalars(select(Alert).where(Alert.detection_id == detection_id)).first() is not None
        assert db.scalars(select(Explanation).where(Explanation.detection_id == detection_id, Explanation.method == 'shap')).first() is not None

        # 13-14: controlled response simulation -> persisted audit event.
        operator = db.get(User, user_id)
        response = execute_response_action(
            detection_id,
            ResponseActionRequest(action=ResponseActionType.simulate_ip_block, reason='Integration test', confirmed=True),
            operator,
            db,
        )
        assert response['status'] == 'simulated'

        audit = detection_audit(detection_id, operator, db)
        assert any(item['event'] == 'response_action' for item in audit['items'])

        # 10: dashboard reflects the newly persisted detection.
        dash = dashboard(start=None, end=None, user=operator, db=db)
        assert dash['statistics']['total_detections'] == 1

        # 15: report contains the same prediction/evidence chain.
        report = create_detection_report(detection_id, user=operator, db=db)
        assert report['status'] == 'generated'
        assert report['report']['detection']['prediction'] == 'ATTACK'


@pytest.mark.skipif(not __import__('importlib').util.find_spec('lime'), reason='LIME dependency is not installed in this test environment')
def test_lime_explanation_dependency_is_real():
    """Keep LIME execution as a real dependency check; no fake explanation is substituted."""
    from ml.explainability.lime_explainer import explain_local
    assert callable(explain_local)

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.routes import predict as prediction_route
from app.core.config import get_settings
from app.db.models import Base, Detection, Explanation
from app.db.session import get_engine
from app.main import create_app
from app.modules.explainability.worker import worker
from app.modules.explainability import service as explanation_service


@pytest.mark.parametrize('error,status', [
    (TimeoutError('server-side limit'), 'timed_out:'),
    (RuntimeError('explainer failed'), 'unavailable:'),
])
def test_explanation_failure_preserves_real_prediction(tmp_path, monkeypatch, error, status):
    root = Path(__file__).resolve().parents[2] / 'ml' / 'artifacts'
    monkeypatch.setenv('ARTIFACTS_ROOT', str(root))
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "prediction.db"}')
    monkeypatch.setenv('REPORTS_ROOT', str(tmp_path / 'reports'))
    get_settings.cache_clear()
    get_engine.cache_clear()
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(worker, 'run', fail)
    try:
        Base.metadata.create_all(get_engine())
        app = create_app()
        dependency = prediction_route.router.routes[0].dependant.dependencies[0].call
        app.dependency_overrides[dependency] = lambda: type('Admin', (), {'id': None, 'role': 'administrator'})()
        response = TestClient(app).post('/api/predict', json={
            'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest',
            'features': {'duration': 0, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF', 'src_bytes': 491, 'dst_bytes': 0},
            'generate_shap': True, 'generate_lime': True,
        })
        assert response.status_code == 201, response.text
        result = response.json()
        assert result['prediction'] in ('normal', 'ATTACK')
        assert all(value.startswith(status) for value in result['explanation'].values())
        with Session(get_engine()) as db:
            detection = db.get(Detection, result['detection_id'])
            assert len(detection.transformed_features) == 121
            assert detection.explanation_status == result['explanation']
            assert not db.scalars(select(Explanation).where(Explanation.detection_id == detection.id)).all()
    finally:
        get_engine().dispose()
        get_engine.cache_clear()
        get_settings.cache_clear()


def test_failed_explanation_persistence_does_not_rollback_prediction_or_lime(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'ml' / 'artifacts'
    monkeypatch.setenv('ARTIFACTS_ROOT', str(root))
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "prediction.db"}')
    monkeypatch.setenv('REPORTS_ROOT', str(tmp_path / 'reports'))
    get_settings.cache_clear()
    get_engine.cache_clear()
    original = explanation_service._persist
    def fail_shap(db, detection, method, payload):
        result = original(db, detection, method, payload)
        db.flush()
        if method == 'shap':
            raise RuntimeError('Explanation persistence failed')
        return result
    monkeypatch.setattr(explanation_service, '_persist', fail_shap)
    try:
        Base.metadata.create_all(get_engine())
        app = create_app()
        dependency = prediction_route.router.routes[0].dependant.dependencies[0].call
        app.dependency_overrides[dependency] = lambda: type('Admin', (), {'id': None, 'role': 'administrator'})()
        response = TestClient(app).post('/api/predict', json={
            'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest',
            'features': {'duration': 0, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF', 'src_bytes': 491, 'dst_bytes': 0},
            'generate_shap': True, 'generate_lime': True,
        })
        assert response.status_code == 201, response.text
        result = response.json()
        assert result['explanation']['shap'].startswith('unavailable:')
        assert result['explanation']['lime'] == 'generated'
        with Session(get_engine()) as db:
            detection = db.get(Detection, result['detection_id'])
            assert detection is not None
            assert detection.explanation_status == result['explanation']
            rows = db.scalars(select(Explanation).where(Explanation.detection_id == detection.id)).all()
            assert [row.method for row in rows] == ['lime']
    finally:
        get_engine().dispose()
        get_engine.cache_clear()
        get_settings.cache_clear()

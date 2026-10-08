"""Exercise the shipped NSL-KDD bundle through the existing HTTP endpoint."""
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.v1.routes import predict as prediction_route
from app.core.config import get_settings
from app.db.models import Base, Detection
from app.db.session import get_engine
from app.main import create_app


def test_shipped_bundle_predicts_and_persists(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'ml' / 'artifacts'
    monkeypatch.setenv('ARTIFACTS_ROOT', str(root))
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path / "prediction.db"}')
    monkeypatch.setenv('REPORTS_ROOT', str(tmp_path / 'reports'))
    get_settings.cache_clear()
    get_engine.cache_clear()
    try:
        Base.metadata.create_all(get_engine())
        app = create_app()
        client = TestClient(app)
        request = {
            'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest',
            'features': {'duration': 0, 'protocol_type': 'tcp', 'service': 'ftp_data',
                         'flag': 'SF', 'src_bytes': 491, 'dst_bytes': 0},
            'generate_shap': True, 'generate_lime': True,
        }
        assert client.post('/api/predict', json=request).status_code == 401
        dependency = prediction_route.router.routes[0].dependant.dependencies[0].call
        app.dependency_overrides[dependency] = lambda: type('Admin', (), {'id': None, 'role': 'administrator'})()
        response = client.post('/api/predict', json=request)
        assert response.status_code == 201, response.text
        result = response.json()
        assert result['prediction'] in ('normal', 'ATTACK')
        assert 0 <= result['probability'] <= 1
        assert result['explanation'] == {'shap': 'generated', 'lime': 'generated'}
        with Session(get_engine()) as db:
            detection = db.get(Detection, result['detection_id'])
            assert detection is not None
            assert len(detection.transformed_features) == 121
    finally:
        get_engine().dispose()
        get_engine.cache_clear()
        get_settings.cache_clear()

"""Detection failures must remain visible to an authorized cross-origin browser."""
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.config import get_settings
from app.api.v1.routes import predict as prediction_route

ORIGIN = 'https://ai-ctdrs-c6du.vercel.app'
FEATURES = {'duration': 10, 'src_bytes': 1024, 'dst_bytes': 2048, 'src_packets': 20, 'dst_packets': 15}


def test_prediction_failure_has_cors_headers(monkeypatch):
    monkeypatch.setenv('CORS_ORIGINS', ORIGIN)
    get_settings.cache_clear()
    try:
        app = create_app()
        dependency = prediction_route.router.routes[0].dependant.dependencies[0].call
        app.dependency_overrides[dependency] = lambda: type('Admin', (), {'id': 'admin', 'role': 'administrator'})()
        def fail(request, user):
            assert request.features == FEATURES
            raise FileNotFoundError('Saved preprocessing artifact not found')
        monkeypatch.setattr(prediction_route, 'predict', fail)
        client = TestClient(app, raise_server_exceptions=False)
        preflight = client.options('/api/predict', headers={'Origin': ORIGIN, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization,content-type'})
        assert preflight.status_code == 200
        response = client.post('/api/predict', headers={'Origin': ORIGIN}, json={'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest', 'features': FEATURES})
        assert response.status_code == 500
        assert response.json() == {'detail': 'Internal server error'}
        assert response.headers['access-control-allow-origin'] == ORIGIN
        assert response.headers['access-control-allow-credentials'] == 'true'
        denied = client.post('/api/predict', headers={'Origin': 'https://untrusted.example'}, json={'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest', 'features': FEATURES})
        assert 'access-control-allow-origin' not in denied.headers
    finally:
        get_settings.cache_clear()


def test_actual_prediction_with_missing_artifacts_returns_readable_500(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('CORS_ORIGINS', ORIGIN)
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path / 'missing-artifacts'))
    get_settings.cache_clear()
    try:
        app = create_app()
        dependency = prediction_route.router.routes[0].dependant.dependencies[0].call
        app.dependency_overrides[dependency] = lambda: type('Admin', (), {'id': 'admin', 'role': 'administrator'})()
        response = TestClient(app, raise_server_exceptions=False).post(
            '/api/predict', headers={'Origin': ORIGIN},
            json={'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest',
                  'features': FEATURES, 'generate_shap': True, 'generate_lime': True},
        )
        assert response.status_code == 500
        assert response.headers['access-control-allow-origin'] == ORIGIN
        assert 'nsl_kdd/binary/random_forest/metadata.json' in caplog.text
        assert 'FileNotFoundError' in caplog.text
    finally:
        get_settings.cache_clear()

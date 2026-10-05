from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.models import Base, User
from app.db.session import get_db
from app.core.security import hash_password, create_access_token


def _client_and_db():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = Session(engine)
    users = [
        User(id='viewer', email='viewer@example.com', password_hash=hash_password('ViewerPassword123!'), role='viewer', is_active=True),
        User(id='analyst', email='analyst@example.com', password_hash=hash_password('AnalystPassword123!'), role='analyst', is_active=True),
        User(id='admin', email='admin@example.com', password_hash=hash_password('AdminPassword123!'), role='administrator', is_active=True),
    ]
    db.add_all(users); db.commit()
    def override_db():
        yield db
    app.dependency_overrides[get_db] = override_db
    return TestClient(app), db


def teardown_function():
    app.dependency_overrides.clear()


def test_unauthorized_api_access_is_rejected():
    client, db = _client_and_db()
    try:
        for path in ['/api/datasets', '/api/detections', '/api/alerts', '/api/explain/shap/global', '/api/reports', '/api/responses']:
            response = client.get(path)
            assert response.status_code == 401, (path, response.status_code, response.text)
    finally:
        db.close()


def test_invalid_jwt_is_rejected():
    client, db = _client_and_db()
    try:
        response = client.get('/api/datasets', headers={'Authorization': 'Bearer definitely.not.a.jwt'})
        assert response.status_code == 401
    finally:
        db.close()


def test_viewer_cannot_predict_or_escalate_privileges():
    client, db = _client_and_db()
    try:
        viewer = db.get(User, 'viewer')
        token, _, _ = create_access_token(viewer)
        headers = {'Authorization': f'Bearer {token}'}
        response = client.post('/api/predict', json={'dataset': 'nsl-kdd', 'task': 'binary', 'model': 'random_forest', 'features': {'x': 1}}, headers=headers)
        assert response.status_code == 403
        response = client.patch('/api/auth/users/analyst/role', json={'role': 'administrator'}, headers=headers)
        assert response.status_code == 403
    finally:
        db.close()


def test_analyst_cannot_escalate_privileges():
    client, db = _client_and_db()
    try:
        analyst = db.get(User, 'analyst')
        token, _, _ = create_access_token(analyst)
        response = client.patch('/api/auth/users/viewer/role', json={'role': 'administrator'}, headers={'Authorization': f'Bearer {token}'})
        assert response.status_code == 403
    finally:
        db.close()

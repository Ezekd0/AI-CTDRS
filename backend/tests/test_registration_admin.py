"""Registration and admin workflow; optionally run against an isolated migrated PostgreSQL DB."""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password, verify_password
from app.db.models import Base, User
from app.db.session import get_db
from app.main import app

PASSWORD = 'NewAccountPassword123!'


@pytest.fixture
def workflow():
    url = os.environ.get('CTDRS_TEST_DATABASE_URL')
    engine = create_engine(url) if url else create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    if not url:
        Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([
            User(id='registration-admin', email='registration-admin@example.com', role='administrator', password_hash=hash_password(PASSWORD)),
            User(id='registration-legacy', email='registration-legacy@example.com', role='analyst', password_hash=hash_password(PASSWORD)),
        ])
        db.commit()
    def override():
        with Session(engine) as db:
            yield db
    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            yield client, engine
    finally:
        app.dependency_overrides.clear()
        with Session(engine) as db:
            for user in db.scalars(select(User).where(User.email.like('registration-%'))):
                db.delete(user)
            db.commit()
        engine.dispose()


def login(client, email):
    response = client.post('/api/auth/login', json={'email': email, 'password': PASSWORD})
    assert response.status_code == 200, response.text
    assert response.json()['token_type'] == 'bearer'
    return {'Authorization': f"Bearer {response.json()['access_token']}"}


def test_registration_persistence_login_and_admin(workflow):
    client, engine = workflow
    body = {'full_name': '  Registration Test User  ', 'email': 'Registration-New@Example.com', 'password': PASSWORD, 'role': 'administrator'}
    response = client.post('/api/auth/register', json=body)
    assert response.status_code == 201, response.text
    user = response.json()
    assert user['full_name'] == 'Registration Test User'
    assert user['role'] == 'viewer' and user['is_active'] is True
    assert user['created_at']
    assert 'password_hash' not in user
    with Session(engine) as db:
        saved = db.get(User, user['id'])
        assert saved.email == 'registration-new@example.com'
        assert saved.full_name == 'Registration Test User'
        assert saved.password_hash.startswith('$argon2')
        assert verify_password(PASSWORD, saved.password_hash)
    assert client.post('/api/auth/register', json=body).status_code == 409
    viewer = login(client, body['email'])
    assert client.get('/api/auth/me', headers=viewer).json()['id'] == user['id']
    assert client.get('/api/dashboard', headers=viewer).status_code == 200
    assert client.get('/api/admin/overview').status_code == 401
    for path in ['/api/admin/overview', '/api/admin/users']:
        assert client.get(path, headers=viewer).status_code == 403
    assert client.patch(f"/api/admin/users/{user['id']}/status", json={'is_active': False}, headers=viewer).status_code == 403
    legacy = login(client, 'registration-legacy@example.com')
    assert client.get('/api/auth/me', headers=legacy).json()['full_name'] is None
    assert client.get('/api/admin/users', headers=legacy).status_code == 403
    admin = login(client, 'registration-admin@example.com')
    overview = client.get('/api/admin/overview', headers=admin)
    assert overview.status_code == 200, overview.text
    assert overview.json()['total_users'] >= 3
    assert overview.json()['system_status']['database'] == 'connected'
    users = client.get('/api/admin/users', params={'search': 'Registration Test', 'role': 'viewer', 'is_active': True}, headers=admin)
    assert users.status_code == 200
    assert users.json()['total'] == 1 and users.json()['items'][0]['id'] == user['id']
    assert client.get('/api/admin/users', params={'search': '%'}, headers=admin).json()['total'] == 0
    assert client.get('/api/admin/users', params={'limit': 1}, headers=admin).json()['total'] >= 3
    assert len(client.get('/api/admin/users', params={'limit': 1}, headers=admin).json()['items']) == 1
    assert client.patch('/api/admin/users/missing/status', json={'is_active': False}, headers=admin).status_code == 404
    assert client.patch('/api/admin/users/registration-admin/status', json={'is_active': False}, headers=admin).status_code == 409
    assert client.patch(f"/api/admin/users/{user['id']}/status", json={'is_active': False}, headers=admin).status_code == 200
    assert client.get('/api/auth/me', headers=viewer).status_code == 401
    assert client.post('/api/auth/login', json={'email': body['email'], 'password': PASSWORD}).status_code == 401
    assert client.patch(f"/api/admin/users/{user['id']}/status", json={'is_active': True}, headers=admin).status_code == 200
    viewer = login(client, body['email'])
    assert client.get('/api/dashboard', headers=viewer).status_code == 200
    # Existing role management remains available and authorization uses the live DB role.
    assert client.patch(f"/api/auth/users/{user['id']}/role", json={'role': 'administrator'}, headers=admin).status_code == 200
    assert client.get('/api/admin/users', headers=viewer).status_code == 200
    assert client.patch(f"/api/auth/users/{user['id']}/role", json={'role': 'viewer'}, headers=admin).status_code == 200
    assert client.get('/api/admin/users', headers=viewer).status_code == 403


@pytest.mark.parametrize('changes', [{'full_name': '   '}, {'full_name': 'x' * 201}, {'email': 'invalid'}, {'password': 'short'}, {'password': 'x' * 129}])
def test_registration_validation(workflow, changes):
    client, _ = workflow
    body = {'full_name': 'Registration User', 'email': 'registration-invalid@example.com', 'password': PASSWORD, **changes}
    assert client.post('/api/auth/register', json=body).status_code == 422


def test_legacy_registration_contract(workflow):
    client, _ = workflow
    assert client.post('/api/auth/register', json={'email': 'registration-oldclient@example.com', 'password': PASSWORD}).status_code == 201


def test_concurrent_duplicate_registration(workflow):
    if not os.environ.get('CTDRS_TEST_DATABASE_URL'):
        pytest.skip('Concurrent registration requires isolated PostgreSQL')
    client, _ = workflow
    body = {'full_name': 'Concurrent User', 'email': 'registration-concurrent@example.com', 'password': PASSWORD}
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: client.post('/api/auth/register', json=body), range(2)))
    assert sorted(r.status_code for r in responses) == [201, 409]

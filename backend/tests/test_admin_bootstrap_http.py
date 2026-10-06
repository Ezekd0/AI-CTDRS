import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from app.core.config import Settings, get_settings
from app.core.security import verify_password
from app.db.models import Base, User
from app.db.session import get_db
from app.main import app

KEY = 'test-bootstrap-key-with-32-characters'
PASSWORD = 'test-admin-password-1234'

@pytest.fixture
def bootstrap():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    settings = Settings(admin_bootstrap_enabled=True, admin_bootstrap_key=KEY, main_admin_email='ospa@example.com')
    def database():
        with Session(engine) as db:
            yield db
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            yield client, engine, settings
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_creation_login_authorization_and_one_time(bootstrap):
    client, engine, settings = bootstrap
    url = '/api/auth/bootstrap-admin'
    assert client.get(url).status_code == 200
    for headers in [{}, {'Authorization': 'Bearer wrong'}]:
        assert client.post(url, headers=headers, json={'password': PASSWORD}).status_code == 403
    headers = {'Authorization': 'Bearer ' + KEY}
    invalid = client.post(url, headers=headers, json={'password': 'pvt123'})
    assert invalid.status_code == 422 and 'pvt123' not in invalid.text
    assert client.post(url, headers=headers, json={'password': PASSWORD, 'email': 'other@example.com'}).status_code == 422
    response = client.post(url, headers=headers, json={'password': PASSWORD})
    assert response.status_code == 201, response.text
    assert response.json()['full_name'] == 'Ospa Admin'
    assert response.json()['email'] == 'ospa@example.com'
    assert PASSWORD not in response.text and 'password_hash' not in response.text
    with Session(engine) as db:
        user = db.scalar(select(User))
        assert user.password_hash.startswith('$argon2') and verify_password(PASSWORD, user.password_hash)
    token = client.post('/api/auth/login', json={'email': 'ospa@example.com', 'password': PASSWORD})
    assert token.status_code == 200 and token.json()['role'] == 'administrator'
    admin = {'Authorization': 'Bearer ' + token.json()['access_token']}
    for path in ['/api/admin/overview', '/api/admin/users', '/api/network/reports']:
        assert client.get(path, headers=admin).status_code == 200
    assert client.post(url, headers=headers, json={'password': PASSWORD + 'new'}).status_code == 409
    settings.main_admin_email = 'another@example.com'
    assert client.post(url, headers=headers, json={'password': PASSWORD}).status_code == 409
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(User)) == 1
        assert verify_password(PASSWORD, db.scalar(select(User)).password_hash)
    settings.admin_bootstrap_enabled = False
    assert client.get(url).status_code == 404
    assert client.post(url, headers=headers, json={'password': PASSWORD}).status_code == 404
    assert client.post('/api/auth/login', json={'email': 'ospa@example.com', 'password': PASSWORD}).status_code == 200


def test_no_promotion_of_registered_user(bootstrap):
    client, engine, _ = bootstrap
    assert client.post('/api/auth/register', json={'email': 'ospa@example.com', 'password': PASSWORD}).status_code == 201
    assert client.post('/api/auth/bootstrap-admin', headers={'Authorization': 'Bearer ' + KEY}, json={'password': PASSWORD}).status_code == 409
    with Session(engine) as db:
        assert db.scalar(select(User)).role == 'viewer'


@pytest.mark.parametrize('values', [
    {'main_admin_email': 'supergmail.com'},
    {'admin_bootstrap_enabled': True},
    {'admin_bootstrap_enabled': True, 'main_admin_email': 'ospa@example.com', 'admin_bootstrap_key': 'short'},
    {'main_admin_full_name': ' '},
])
def test_configuration_rejects_invalid_bootstrap(values):
    with pytest.raises(ValidationError):
        Settings(**values)

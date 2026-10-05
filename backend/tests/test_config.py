import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_simulation_mode_defaults_true():
    s = Settings(database_url="sqlite://", secret_key="x" * 32, _env_file=None)
    assert s.response_simulation_mode is True


def test_short_secret_rejected():
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite://", secret_key="short", _env_file=None)


def test_weak_jwt_algorithm_rejected():
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite://", secret_key="x" * 32, jwt_algorithm="none", _env_file=None)

def test_unknown_jwt_algorithm_rejected():
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite://", secret_key="x" * 32, jwt_algorithm="HS1", _env_file=None)

def test_wildcard_cors_rejected():
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite://", secret_key="x" * 32, cors_origins="*", _env_file=None)

def test_cors_requires_explicit_http_origin():
    with pytest.raises(ValidationError):
        Settings(database_url="sqlite://", secret_key="x" * 32, cors_origins="localhost:5173", _env_file=None)

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
import pytest
from pydantic import ValidationError

from app.db.models import Base, User
from app.core.security import verify_password
from scripts.create_admin import bootstrap_admin


def test_bootstrap_creates_and_promotes_without_duplicate_users():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        values = dict(email='Admin@Example.com', password='first-secure-password', full_name='Admin Person')
        assert bootstrap_admin(db, **values) == 'created'
        user = db.scalar(select(User))
        user_id = user.id
        assert user.email == 'admin@example.com'
        assert user.full_name == 'Admin Person'
        assert user.role == 'administrator' and user.is_active
        assert verify_password(values['password'], user.password_hash)
        user.role = 'viewer'
        user.is_active = False
        db.commit()
        values.update(password='second-secure-password', full_name='Updated Admin')
        assert bootstrap_admin(db, **values) == 'updated'
        assert user.id == user_id
        assert user.role == 'administrator' and user.is_active
        assert user.full_name == 'Updated Admin'
        assert verify_password(values['password'], user.password_hash)
        assert not verify_password('first-secure-password', user.password_hash)
        assert bootstrap_admin(db, **values) == 'updated'
        assert db.scalar(select(func.count()).select_from(User)) == 1


@pytest.mark.parametrize('overrides', [{'email': 'invalid'}, {'password': 'short'}, {'full_name': ' '}])
def test_invalid_bootstrap_input_does_not_create_user(overrides):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        values = dict(email='admin@example.com', password='secure-password-123', full_name='Admin')
        values.update(overrides)
        with pytest.raises(ValidationError):
            bootstrap_admin(db, **values)
        assert db.scalar(select(func.count()).select_from(User)) == 0

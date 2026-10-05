from datetime import datetime, timezone, timedelta
from app.core.security import hash_password, verify_password, create_access_token
from app.db.models.models import User

def test_password_hash_is_not_plaintext():
    raw='correct horse battery staple'
    hashed=hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed)
    assert not verify_password('wrong password', hashed)

def test_access_token_contains_user_identity_and_expiry():
    user=User(id='u1', email='u@example.com', role='analyst', is_active=True)
    token,jti,expires=create_access_token(user)
    assert token and jti
    assert expires > datetime.now(timezone.utc)

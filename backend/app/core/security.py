from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import json
import uuid
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.config import get_settings
from app.db.session import get_db
from app.db.models.models import User, RevokedToken

pwd_context = PasswordHasher()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/api/auth/login')
ROLES = {'administrator', 'analyst', 'viewer'}

class JWTError(ValueError):
    pass

def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')

def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))

def _encode_hs256(payload: dict, secret: str) -> str:
    header = {'alg': 'HS256', 'typ': 'JWT'}
    header_part = _b64e(json.dumps(header, separators=(',', ':'), sort_keys=True).encode())
    payload_part = _b64e(json.dumps(payload, separators=(',', ':'), sort_keys=True, default=lambda x: int(x.timestamp()) if isinstance(x, datetime) else x).encode())
    signing_input = f'{header_part}.{payload_part}'.encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f'{header_part}.{payload_part}.{_b64e(signature)}'

def _decode_hs256(token: str, secret: str, verify_exp: bool = True) -> dict:
    try:
        header_part, payload_part, signature_part = token.split('.')
        header = json.loads(_b64d(header_part))
        if header.get('alg') != 'HS256' or header.get('typ') != 'JWT':
            raise JWTError('Unsupported JWT header')
        expected = hmac.new(secret.encode(), f'{header_part}.{payload_part}'.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64d(signature_part)):
            raise JWTError('Invalid signature')
        payload = json.loads(_b64d(payload_part))
        if verify_exp and ('exp' not in payload or datetime.now(timezone.utc).timestamp() >= float(payload['exp'])):
            raise JWTError('Expired token')
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JWTError('Malformed token') from exc

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False

def create_access_token(user: User) -> tuple[str, str, datetime]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.access_token_expire_minutes)
    jti = str(uuid.uuid4())
    payload = {'sub': user.id, 'email': user.email, 'role': user.role, 'jti': jti, 'iat': now, 'exp': expires}
    return _encode_hs256(payload, settings.secret_key), jti, expires

def decode_access_token(token: str, *, verify_exp: bool = True) -> dict:
    settings = get_settings()
    if settings.jwt_algorithm != 'HS256':
        raise JWTError('Unsupported configured JWT algorithm')
    return _decode_hs256(token, settings.secret_key, verify_exp=verify_exp)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired authentication token', headers={'WWW-Authenticate': 'Bearer'})
    try:
        payload = decode_access_token(token)
        user_id = payload.get('sub'); jti = payload.get('jti')
        if not user_id or not jti:
            raise JWTError('Missing token identity')
    except JWTError:
        raise credentials
    if db.scalar(select(RevokedToken).where(RevokedToken.jti == jti)):
        raise credentials
    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise credentials
    return user

def require_roles(*roles: str):
    if not roles or not set(roles) <= ROLES:
        raise ValueError('Invalid role dependency')
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Insufficient permissions')
        return user
    return dependency

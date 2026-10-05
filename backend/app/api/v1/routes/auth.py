from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.db.session import get_db
from app.db.models.models import User, RevokedToken
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponse, UserResponse
from app.core.security import hash_password, verify_password, create_access_token, get_current_user, oauth2_scheme, decode_access_token, JWTError
from app.core.config import get_settings

router = APIRouter(prefix='/auth', tags=['Authentication'])

@router.post('/register', response_model=UserResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail='An account with this email already exists')
    user = User(full_name=body.full_name, email=email, password_hash=hash_password(body.password), role='viewer', is_active=True)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if db.scalar(select(User).where(User.email == email)):
            raise HTTPException(status_code=409, detail='An account with this email already exists')
        raise
    db.refresh(user)
    return user

@router.post('/login', response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not user.password_hash or not user.is_active or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail='Invalid email or password', headers={'WWW-Authenticate':'Bearer'})
    token, _, expires = create_access_token(user)
    return AuthResponse(access_token=token, expires_in=max(0, int((expires - datetime.now(timezone.utc)).total_seconds())), role=user.role)

@router.post('/logout', status_code=204)
def logout(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    settings = get_settings()
    try:
        payload = decode_access_token(token, verify_exp=False)
    except JWTError:
        raise HTTPException(status_code=401, detail='Invalid authentication token')
    jti, exp = payload.get('jti'), payload.get('exp')
    if not jti or not exp:
        raise HTTPException(status_code=401, detail='Invalid authentication token')
    db.merge(RevokedToken(jti=jti, expires_at=datetime.fromtimestamp(exp, timezone.utc)))
    db.commit()

@router.get('/me', response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user

@router.get('/status')
def status():
    return {'authentication': 'configured'}

from pydantic import BaseModel
from typing import Literal
from app.core.security import require_roles

class RoleUpdate(BaseModel):
    role: Literal['administrator', 'analyst', 'viewer']

@router.patch('/users/{user_id}/role', response_model=UserResponse)
def update_role(user_id: str, body: RoleUpdate, db: Session = Depends(get_db), admin: User = Depends(require_roles('administrator'))):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    target.role = body.role
    db.commit(); db.refresh(target)
    return target

"""Administrative reads and account activation using existing role authorization."""
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import require_roles
from app.db.models.models import Detection, User
from app.db.session import get_db
from app.schemas.auth import UserResponse

router = APIRouter(prefix='/admin', tags=['Administration'], dependencies=[Depends(require_roles('administrator'))])


@router.get('/overview')
def overview(db: Session = Depends(get_db)):
    count = lambda statement: db.scalar(statement) or 0
    since = datetime.now(timezone.utc) - timedelta(days=7)
    users = db.scalars(select(User).order_by(User.created_at.desc(), User.id).limit(5)).all()
    detections = db.scalars(select(Detection).order_by(Detection.created_at.desc(), Detection.id).limit(5)).all()
    return {
        'total_users': count(select(func.count()).select_from(User)),
        'active_users': count(select(func.count()).select_from(User).where(User.is_active.is_(True))),
        'registrations_last_7_days': count(select(func.count()).select_from(User).where(User.created_at >= since)),
        'total_detections': count(select(func.count()).select_from(Detection)),
        'recent_registrations': [UserResponse.model_validate(user) for user in users],
        'recent_detections': [{'id': row.id, 'prediction': row.prediction, 'created_at': row.created_at} for row in detections],
        'system_status': {'api': 'ok', 'database': 'connected', 'response_mode': get_settings().response_mode},
    }


class UserList(BaseModel):
    items: list[UserResponse]
    total: int


@router.get('/users', response_model=UserList)
def users(
    search: str = Query('', max_length=200),
    role: Literal['administrator', 'analyst', 'viewer'] | None = None,
    is_active: bool | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    filters = []
    if search.strip():
        # Escape LIKE metacharacters so searches remain literal and parameterized.
        term = search.strip().replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        filters.append(or_(User.email.ilike(f'%{term}%', escape='\\'), User.full_name.ilike(f'%{term}%', escape='\\')))
    if role is not None:
        filters.append(User.role == role)
    if is_active is not None:
        filters.append(User.is_active == is_active)
    total = db.scalar(select(func.count()).select_from(User).where(*filters)) or 0
    items = db.scalars(select(User).where(*filters).order_by(User.created_at.desc(), User.id).offset(offset).limit(limit)).all()
    return UserList(items=[UserResponse.model_validate(user) for user in items], total=total)


class StatusUpdate(BaseModel):
    is_active: bool


@router.patch('/users/{user_id}/status', response_model=UserResponse)
def update_status(user_id: str, body: StatusUpdate, db: Session = Depends(get_db)):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    # Administrator accounts cannot be disabled through this surface, avoiding lockout.
    if target.role == 'administrator' and not body.is_active:
        raise HTTPException(status_code=409, detail='Administrator accounts cannot be disabled here')
    target.is_active = body.is_active
    db.commit()
    db.refresh(target)
    return target

from __future__ import annotations

from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import get_settings
from app.core.security import require_roles
from app.db.models.models import Alert, Detection, ResponseAction, User
from app.db.session import get_db

router = APIRouter(prefix='/responses', tags=['Incident Response'])

class ResponseActionType(str, Enum):
    create_alert = 'create_alert'
    mark_investigation = 'mark_investigation'
    simulate_ip_block = 'simulate_ip_block'
    simulate_host_isolation = 'simulate_host_isolation'
    escalate_incident = 'escalate_incident'
    close_incident = 'close_incident'

class ResponseActionRequest(BaseModel):
    action: ResponseActionType
    reason: str = Field(..., min_length=3, max_length=1000)
    confirmed: bool = False

def _severity(detection: Detection) -> str:
    p = float(detection.probability or 0)
    return 'critical' if p >= .9 else 'high' if p >= .75 else 'medium' if p >= .5 else 'low'

def _action_status(action: ResponseActionType) -> str:
    if action in {ResponseActionType.simulate_ip_block, ResponseActionType.simulate_host_isolation}:
        return 'simulated'
    return 'completed'

@router.get('')
def response_history(
    detection_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=250),
    user=Depends(require_roles('administrator', 'analyst')),
    db: Session = Depends(get_db),
):
    q = select(ResponseAction).options(joinedload(ResponseAction.user)).order_by(ResponseAction.created_at.desc()).limit(limit)
    if detection_id:
        q = q.where(ResponseAction.detection_id == detection_id)
    rows = db.execute(q).unique().scalars().all()
    return {'mode': get_settings().response_mode, 'items': [
        {'id': r.id, 'incident': r.detection_id, 'user': r.user.email if r.user else None, 'action': r.action_type, 'timestamp': r.created_at.isoformat(), 'status': r.status, 'reason': r.reason, 'details': r.details or {}} for r in rows
    ], 'count': len(rows)}

@router.post('/{detection_id}/actions')
def execute_response_action(
    detection_id: str,
    body: ResponseActionRequest,
    user: User = Depends(require_roles('administrator', 'analyst')),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    if settings.response_mode != 'SIMULATION':
        raise HTTPException(503, 'Response integrations are not enabled; system is simulation-only.')
    if not body.confirmed:
        raise HTTPException(400, 'Confirmation is required before executing a response action.')

    detection = db.execute(select(Detection).options(joinedload(Detection.alerts)).where(Detection.id == detection_id)).unique().scalar_one_or_none()
    if detection is None:
        raise HTTPException(404, 'Incident/detection not found')

    action = body.action
    status = _action_status(action)
    details = {'response_mode': 'SIMULATION', 'incident_id': detection_id, 'external_effect': False}

    if action == ResponseActionType.create_alert:
        alert = Alert(detection_id=detection_id, user_id=user.id, severity=_severity(detection), status='new', message=f'Response-created alert for detection {detection_id}.')
        db.add(alert)
        details['alert_created'] = True
    elif action == ResponseActionType.mark_investigation:
        for alert in detection.alerts:
            alert.status = 'investigation'
        details['incident_status'] = 'investigation'
    elif action == ResponseActionType.simulate_ip_block:
        details['simulation'] = 'IP block recorded for laboratory simulation only; no network control was performed.'
    elif action == ResponseActionType.simulate_host_isolation:
        details['simulation'] = 'Host isolation recorded for laboratory simulation only; no host control was performed.'
    elif action == ResponseActionType.escalate_incident:
        for alert in detection.alerts:
            alert.status = 'escalated'
        details['incident_status'] = 'escalated'
    elif action == ResponseActionType.close_incident:
        for alert in detection.alerts:
            if alert.status != 'closed':
                alert.status = 'closed'
        details['incident_status'] = 'closed'

    record = ResponseAction(detection_id=detection_id, user_id=user.id, action_type=action.value, status=status, reason=body.reason.strip(), details=details)
    db.add(record)
    db.commit()
    db.refresh(record)
    return {'id': record.id, 'incident': record.detection_id, 'user': user.email, 'action': record.action_type, 'timestamp': record.created_at.isoformat(), 'status': record.status, 'reason': record.reason, 'details': record.details, 'mode': settings.response_mode}

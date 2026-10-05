from __future__ import annotations

from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.security import require_roles
from app.db.models.models import Alert, AlertAuditLog, Detection, User
from app.db.session import get_db

router = APIRouter(prefix='/alerts', tags=['Alerts'])

class AlertStatus(str, Enum):
    NEW = 'NEW'
    INVESTIGATING = 'INVESTIGATING'
    RESOLVED = 'RESOLVED'
    FALSE_POSITIVE = 'FALSE_POSITIVE'

class AlertAction(str, Enum):
    acknowledge = 'acknowledge'
    assign = 'assign'
    investigate = 'investigate'
    resolve = 'resolve'
    false_positive = 'false_positive'

class AlertActionRequest(BaseModel):
    action: AlertAction
    confirmed: bool = False
    analyst_id: str | None = Field(default=None, max_length=36)
    reason: str = Field(default='Alert workflow update', min_length=3, max_length=1000)

def _severity(d: Detection, alert: Alert) -> str:
    return str(alert.severity or ('critical' if d.probability >= .9 else 'high' if d.probability >= .75 else 'medium' if d.probability >= .5 else 'low')).lower()

def _item(a: Alert) -> dict:
    d = a.detection
    return {
        'id': a.id,
        'detection_id': a.detection_id,
        'severity': _severity(d, a),
        'attack_type': d.prediction,
        'source': _source(d.raw_features),
        'timestamp': a.created_at.isoformat(),
        'status': a.status.upper(),
        'assigned_analyst': {'id': a.assigned_analyst.id, 'email': a.assigned_analyst.email} if a.assigned_analyst else None,
        'message': a.message,
        'updated_at': a.updated_at.isoformat(),
    }

def _source(features):
    if not isinstance(features, dict): return None
    norm = {str(k).strip().lower().replace('-', '_').replace(' ', '_'): v for k,v in features.items()}
    for k in ('source_ip','src_ip','srcip'):
        if k in norm: return norm[k]
    return None

@router.get('')
def list_alerts(
    status: AlertStatus | None = Query(None),
    severity: str | None = Query(None),
    attack_type: str | None = Query(None),
    assigned_analyst_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=250),
    user=Depends(require_roles('administrator', 'analyst', 'viewer')),
    db: Session = Depends(get_db),
):
    q = select(Alert).options(joinedload(Alert.detection), joinedload(Alert.assigned_analyst)).order_by(Alert.created_at.desc()).limit(limit)
    if status: q = q.where(Alert.status == status.value.lower())
    if severity: q = q.where(Alert.severity == severity.lower())
    if attack_type: q = q.join(Detection, Detection.id == Alert.detection_id).where(Detection.prediction == attack_type)
    if assigned_analyst_id: q = q.where(Alert.assigned_analyst_id == assigned_analyst_id)
    rows = db.execute(q).unique().scalars().all()
    return {'items': [_item(a) for a in rows], 'count': len(rows)}

@router.get('/analysts')
def list_analysts(user=Depends(require_roles('administrator', 'analyst')), db: Session = Depends(get_db)):
    rows = db.execute(select(User).where(User.role.in_(['administrator','analyst']), User.is_active == True).order_by(User.email)).scalars().all()
    return {'items': [{'id': x.id, 'email': x.email, 'role': x.role} for x in rows], 'count': len(rows)}

@router.get('/{alert_id}/audit')
def alert_audit(alert_id: str, user=Depends(require_roles('administrator','analyst','viewer')), db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert: raise HTTPException(404, 'Alert not found')
    rows = db.execute(select(AlertAuditLog).options(joinedload(AlertAuditLog.user)).where(AlertAuditLog.alert_id == alert_id).order_by(AlertAuditLog.created_at.desc())).unique().scalars().all()
    return {'items': [{'id': r.id, 'action': r.action, 'timestamp': r.created_at.isoformat(), 'user': r.user.email if r.user else None, 'previous_status': r.previous_status, 'new_status': r.new_status, 'details': r.details or {}} for r in rows], 'count': len(rows)}

@router.post('/{alert_id}/actions')
def alert_action(alert_id: str, body: AlertActionRequest, user: User = Depends(require_roles('administrator','analyst')), db: Session = Depends(get_db)):
    if not body.confirmed: raise HTTPException(400, 'Confirmation is required before changing an alert.')
    alert = db.execute(select(Alert).options(joinedload(Alert.detection), joinedload(Alert.assigned_analyst)).where(Alert.id == alert_id)).unique().scalar_one_or_none()
    if not alert: raise HTTPException(404, 'Alert not found')
    previous = alert.status.upper()
    new_status = previous
    details = {'reason': body.reason.strip()}
    if body.action == AlertAction.acknowledge:
        details['acknowledged'] = True
    elif body.action == AlertAction.assign:
        if not body.analyst_id: raise HTTPException(400, 'analyst_id is required for assignment')
        analyst = db.get(User, body.analyst_id)
        if not analyst or not analyst.is_active or analyst.role not in ('administrator','analyst'): raise HTTPException(400, 'Selected user is not an active authorized analyst')
        alert.assigned_analyst_id = analyst.id
        details['assigned_analyst'] = analyst.email
    elif body.action == AlertAction.investigate:
        new_status = AlertStatus.INVESTIGATING.value
        alert.status = new_status.lower()
    elif body.action == AlertAction.resolve:
        new_status = AlertStatus.RESOLVED.value
        alert.status = new_status.lower()
    elif body.action == AlertAction.false_positive:
        new_status = AlertStatus.FALSE_POSITIVE.value
        alert.status = new_status.lower()
    audit = AlertAuditLog(alert_id=alert.id, user_id=user.id, action=body.action.value, previous_status=previous, new_status=new_status, details=details)
    db.add(audit); db.commit(); db.refresh(alert); db.refresh(audit)
    return {'alert': _item(alert), 'audit': {'id': audit.id, 'action': audit.action, 'timestamp': audit.created_at.isoformat(), 'user': user.email, 'previous_status': previous, 'new_status': new_status, 'details': details}}

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, joinedload

from app.core.security import require_roles
from app.db.models.models import Alert, AlertAuditLog, Detection, Explanation, ResponseAction
from app.db.session import get_db

router = APIRouter(prefix='/detections', tags=['Detections'])


def _value(features: dict[str, Any] | None, *names: str) -> Any:
    if not isinstance(features, dict):
        return None
    normalized = {str(k).strip().lower().replace('-', '_').replace(' ', '_'): v for k, v in features.items()}
    for name in names:
        key = name.lower().replace('-', '_').replace(' ', '_')
        if key in normalized:
            return normalized[key]
    return None


def _event(d: Detection, alert: Alert | None = None) -> dict[str, Any]:
    f = d.raw_features or {}
    severity = (alert.severity if alert else None) or ('critical' if d.probability >= 0.9 else 'high' if d.probability >= 0.75 else 'medium' if d.probability >= 0.5 else 'low')
    return {
        'id': d.id,
        'timestamp': d.created_at.isoformat(),
        'source_ip': _value(f, 'source_ip', 'src_ip', 'srcip', 'Source IP', 'Src IP'),
        'destination_ip': _value(f, 'destination_ip', 'dst_ip', 'dest_ip', 'dstip', 'Destination IP', 'Dst IP'),
        'protocol': _value(f, 'protocol', 'proto', 'Protocol'),
        'predicted_attack': d.prediction,
        'confidence': d.probability,
        'severity': str(severity).lower(),
        'response_status': 'pending' if not d.response_actions else d.response_actions[-1].status,
        'dataset': d.dataset_id,
        'model': d.model_id,
    }


@router.get('')
def list_detections(
    limit: int = Query(50, ge=1, le=200),
    user=Depends(require_roles('administrator', 'analyst', 'viewer')),
    db: Session = Depends(get_db),
):
    items = db.execute(select(Detection).options(joinedload(Detection.alerts), joinedload(Detection.response_actions)).order_by(Detection.created_at.desc()).limit(limit)).unique().scalars().all()
    return {'items': [_event(d, d.alerts[0] if d.alerts else None) for d in items], 'count': len(items)}


@router.get('/live')
def live_detections(
    severity: str | None = Query(None),
    attack_type: str | None = Query(None),
    dataset: str | None = Query(None),
    model: str | None = Query(None),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(100, ge=1, le=250),
    user=Depends(require_roles('administrator', 'analyst', 'viewer')),
    db: Session = Depends(get_db),
):
    conditions = []
    if start:
        conditions.append(Detection.created_at >= (start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start))
    if end:
        conditions.append(Detection.created_at <= (end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end))
    if dataset:
        conditions.append(Detection.dataset_id == dataset)
    if model:
        conditions.append(Detection.model_id == model)
    if attack_type:
        conditions.append(Detection.prediction == attack_type)
    if severity:
        s = severity.lower()
        conditions.append(or_(Alert.severity == s, and_(Alert.id.is_(None), Detection.probability >= {'critical': .9, 'high': .75, 'medium': .5, 'low': 0}.get(s, -1))))

    query = select(Detection).options(joinedload(Detection.alerts), joinedload(Detection.response_actions)).outerjoin(Alert, Alert.detection_id == Detection.id).where(*conditions).order_by(Detection.created_at.desc()).limit(limit)
    detections = db.execute(query).unique().scalars().all()
    return {'items': [_event(d, next(iter(d.alerts), None)) for d in detections], 'count': len(detections)}


@router.get('/{detection_id}/audit')
def detection_audit(detection_id: str, user=Depends(require_roles('administrator', 'analyst', 'viewer')), db: Session = Depends(get_db)):
    detection = db.execute(select(Detection).options(joinedload(Detection.alerts), joinedload(Detection.response_actions), joinedload(Detection.explanations)).where(Detection.id == detection_id)).unique().scalar_one_or_none()
    if detection is None:
        raise HTTPException(404, 'Detection not found')
    events = [{'timestamp': detection.created_at.isoformat(), 'event': 'detection_created', 'status': 'recorded', 'details': 'Detection persisted by the prediction service.'}]
    events += [{'timestamp': a.created_at.isoformat(), 'event': 'alert', 'status': a.status, 'details': a.message} for a in detection.alerts]
    alert_ids = [a.id for a in detection.alerts]
    if alert_ids:
        audit_rows = db.execute(select(AlertAuditLog).where(AlertAuditLog.alert_id.in_(alert_ids)).order_by(AlertAuditLog.created_at.desc())).scalars().all()
        events += [{'timestamp': a.created_at.isoformat(), 'event': f'alert_{a.action}', 'status': a.new_status or a.previous_status or 'recorded', 'details': (a.details or {}).get('reason', a.action)} for a in audit_rows]
    events += [{'timestamp': r.created_at.isoformat(), 'event': 'response_action', 'status': r.status, 'details': f'{r.action_type}: {r.reason}'} for r in detection.response_actions]
    events += [{'timestamp': e.created_at.isoformat(), 'event': f'explanation_{e.method}', 'status': 'generated', 'details': 'Explanation generated and persisted by the backend.'} for e in detection.explanations]
    events.sort(key=lambda x: x['timestamp'], reverse=True)
    return {'items': events, 'count': len(events)}


@router.get('/{detection_id}')
def get_detection(detection_id: str, user=Depends(require_roles('administrator', 'analyst', 'viewer')), db: Session = Depends(get_db)):
    detection = db.execute(select(Detection).options(joinedload(Detection.alerts), joinedload(Detection.response_actions), joinedload(Detection.explanations)).where(Detection.id == detection_id)).unique().scalar_one_or_none()
    if detection is None:
        raise HTTPException(404, 'Detection not found')
    item = _event(detection, next(iter(detection.alerts), None))
    item.update({
        'prediction_index': detection.prediction_index,
        'probabilities': detection.probabilities,
        'raw_features': detection.raw_features,
        'original_feature_values': detection.original_feature_values,
        'transformed_features': detection.transformed_features,
        'feature_names': detection.feature_names,
        'explanation_status': detection.explanation_status,
        'alerts': [{'id': a.id, 'severity': a.severity, 'status': a.status, 'message': a.message, 'created_at': a.created_at.isoformat()} for a in detection.alerts],
        'response_actions': [{'id': a.id, 'action_type': a.action_type, 'status': a.status, 'reason': a.reason, 'details': a.details or {}, 'created_at': a.created_at.isoformat()} for a in detection.response_actions],
        'explanations': [{'id': e.id, 'method': e.method, 'created_at': e.created_at.isoformat()} for e in detection.explanations],
    })
    return item

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import get_settings
from app.core.security import require_roles
from app.db.models.models import Alert, AlertAuditLog, Detection, Explanation, Report, ResponseAction
from app.db.session import get_db

router = APIRouter(prefix='/reports', tags=['Reports'])


def _report_payload(d: Detection) -> dict[str, Any]:
    return {
        'detection': {
            'id': d.id,
            'dataset': d.dataset_id,
            'model': d.model.name,
            'task': d.model.task,
            'prediction': d.prediction,
            'prediction_index': d.prediction_index,
            'confidence': d.probability,
            'probabilities': d.probabilities,
            'created_at': d.created_at.isoformat(),
        },
        'alerts': [
            {'id': a.id, 'severity': a.severity, 'status': a.status, 'message': a.message, 'created_at': a.created_at.isoformat()}
            for a in d.alerts
        ],
        'alert_audit': [
            {'id': a.id, 'alert_id': a.alert_id, 'action': a.action, 'previous_status': a.previous_status,
             'new_status': a.new_status, 'details': a.details or {}, 'created_at': a.created_at.isoformat()}
            for alert in d.alerts
            for a in alert.audit_logs
        ],
        'explanations': [
            {'id': e.id, 'method': e.method, 'payload': e.payload, 'created_at': e.created_at.isoformat()}
            for e in d.explanations
        ],
        'response_actions': [
            {'id': r.id, 'action': r.action_type, 'status': r.status, 'reason': r.reason, 'details': r.details or {}, 'created_at': r.created_at.isoformat()}
            for r in d.response_actions
        ],
        'features': {
            'raw': d.raw_features,
            'original': d.original_feature_values,
            'transformed': d.transformed_features,
            'names': d.feature_names,
        },
    }


@router.get('')
def list_reports(
    detection_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=250),
    user=Depends(require_roles('administrator', 'analyst', 'viewer')),
    db: Session = Depends(get_db),
):
    q = select(Report).options(joinedload(Report.detection)).order_by(Report.created_at.desc()).limit(limit)
    if detection_id:
        q = q.where(Report.detection_id == detection_id)
    rows = db.execute(q).unique().scalars().all()
    return {'items': [
        {'id': r.id, 'detection_id': r.detection_id, 'report_type': r.report_type, 'status': r.status,
         'storage_key': r.storage_key, 'created_at': r.created_at.isoformat(), 'metadata': r.metadata_json or {}}
        for r in rows
    ], 'count': len(rows)}


@router.post('/detections/{detection_id}', status_code=201)
def create_detection_report(
    detection_id: str,
    user=Depends(require_roles('administrator', 'analyst')),
    db: Session = Depends(get_db),
):
    d = db.execute(select(Detection).options(
        joinedload(Detection.model), joinedload(Detection.alerts), joinedload(Detection.explanations), joinedload(Detection.response_actions)
    ).where(Detection.id == detection_id)).unique().scalar_one_or_none()
    if d is None:
        raise HTTPException(404, 'Detection not found')

    payload = _report_payload(d)
    settings = get_settings()
    root = settings.resolved_reports_root / 'detections'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{detection_id}_report.json'
    path.write_text(json.dumps(payload, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x))

    row = Report(user_id=user.id, detection_id=d.id, report_type='detection', status='generated',
                 storage_key=f'reports/{path.relative_to(settings.resolved_reports_root).as_posix()}',
                 metadata_json={'prediction': d.prediction, 'confidence': d.probability, 'explanation_methods': [e.method for e in d.explanations]})
    db.add(row); db.commit(); db.refresh(row)
    return {'id': row.id, 'detection_id': d.id, 'report_type': row.report_type, 'status': row.status,
            'storage_key': row.storage_key, 'created_at': row.created_at.isoformat(), 'report': payload}

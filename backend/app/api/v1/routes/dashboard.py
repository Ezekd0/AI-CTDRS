from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import require_roles
from app.db.session import get_db
from app.db.models.models import Alert, Detection, Model

router = APIRouter(prefix='/dashboard', tags=['Dashboard'])


def _range(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    finish = end or now
    begin = start or (finish - timedelta(days=30))
    if begin.tzinfo is None:
        begin = begin.replace(tzinfo=timezone.utc)
    if finish.tzinfo is None:
        finish = finish.replace(tzinfo=timezone.utc)
    if begin > finish:
        raise ValueError('start must be before end')
    return begin, finish


@router.get('')
def dashboard(
    start: datetime | None = Query(None, description='Inclusive UTC start time'),
    end: datetime | None = Query(None, description='Inclusive UTC end time'),
    user=Depends(require_roles('administrator', 'analyst', 'viewer')),
    db: Session = Depends(get_db),
):
    begin, finish = _range(start, end)
    dfilter = (Detection.created_at >= begin, Detection.created_at <= finish)
    afilter = (Alert.created_at >= begin, Alert.created_at <= finish)

    total_detections = db.scalar(select(func.count(Detection.id)).where(*dfilter)) or 0
    severity_rows = db.execute(
        select(Alert.severity, func.count(Alert.id)).where(*afilter).group_by(Alert.severity)
    ).all()
    severities = {str(k).lower(): int(v) for k, v in severity_rows}

    attack_rows = db.execute(
        select(Detection.prediction, func.count(Detection.id))
        .where(*dfilter)
        .group_by(Detection.prediction)
        .order_by(func.count(Detection.id).desc())
    ).all()

    # The schema has no separate raw-event table; total_events therefore means
    # persisted prediction events (Detection rows), which are the production event source.
    trend_rows = db.execute(
        select(Detection.created_at).where(*dfilter).order_by(Detection.created_at.asc())
    ).scalars().all()
    trend = Counter(x.date().isoformat() for x in trend_rows)
    days = []
    cursor = begin.date()
    while cursor <= finish.date():
        key = cursor.isoformat()
        days.append({'date': key, 'detections': trend.get(key, 0)})
        cursor += timedelta(days=1)

    recent_detections = db.execute(
        select(Detection).where(*dfilter).order_by(Detection.created_at.desc()).limit(10)
    ).scalars().all()
    recent_alerts = db.execute(
        select(Alert).where(*afilter).order_by(Alert.created_at.desc()).limit(10)
    ).scalars().all()

    active_model = db.execute(
        select(Model).where(Model.is_active.is_(True)).order_by(Model.updated_at.desc()).limit(1)
    ).scalar_one_or_none()

    return {
        'range': {'start': begin.isoformat(), 'end': finish.isoformat()},
        'statistics': {
            'total_events': total_detections,
            'total_detections': total_detections,
            'critical_alerts': severities.get('critical', 0),
            'high_risk_alerts': severities.get('high', 0),
            'medium_risk_alerts': severities.get('medium', 0),
            'low_risk_alerts': severities.get('low', 0),
        },
        'detection_trend': days,
        'attack_distribution': [
            {'attack': str(label), 'count': int(count)} for label, count in attack_rows
        ],
        'recent_incidents': [
            {
                'id': d.id,
                'prediction': d.prediction,
                'probability': d.probability,
                'dataset_id': d.dataset_id,
                'model_id': d.model_id,
                'created_at': d.created_at.isoformat(),
            } for d in recent_detections
        ],
        'recent_alerts': [
            {
                'id': a.id,
                'detection_id': a.detection_id,
                'severity': a.severity,
                'status': a.status,
                'message': a.message,
                'created_at': a.created_at.isoformat(),
            } for a in recent_alerts
        ],
        'active_model': None if active_model is None else {
            'id': active_model.id,
            'name': active_model.name,
            'dataset_id': active_model.dataset_id,
            'task': active_model.task,
            'version': active_model.version,
        },
    }

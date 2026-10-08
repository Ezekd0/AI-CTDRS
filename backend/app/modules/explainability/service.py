"""Server-side SHAP/LIME generation and persistence for detections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.models import Detection, Explanation
from app.modules.explainability.worker import worker


def _safe_detection_id(detection_id: str) -> str:
    if not detection_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in detection_id):
        raise ValueError('Invalid detection_id')
    return detection_id


def detection_path(detection_id: str) -> Path:
    return get_settings().resolved_reports_root / 'detections' / f'{_safe_detection_id(detection_id)}.json'


def load_detection(detection_id: str) -> dict[str, Any]:
    path = detection_path(detection_id)
    if not path.exists():
        raise FileNotFoundError(detection_id)
    return json.loads(path.read_text())


def save_detection(detection_id: str, payload: dict[str, Any]) -> Path:
    path = detection_path(detection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x))
    return path


def _persist(db: Session, detection: Detection, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = db.execute(select(Explanation).where(Explanation.detection_id == detection.id, Explanation.method == method)).scalar_one_or_none()
    if row is None:
        db.add(Explanation(detection_id=detection.id, method=method, payload=payload))
    else:
        row.payload = payload
    status = dict(detection.explanation_status or {})
    status[method] = 'generated'
    detection.explanation_status = status
    return payload


def _generate(db: Session, detection: Detection, method: str, timeout: float) -> dict[str, Any]:
    if detection.model.name not in {'random_forest', 'xgboost'}:
        raise ValueError(f'{method.upper()} supports Random Forest and XGBoost only')
    settings = get_settings()
    result = worker.run(method, {
        'root': str(settings.resolved_artifacts_root),
        'dataset': detection.dataset_id, 'task': detection.model.task,
        'model': detection.model.name, 'features': detection.transformed_features,
        'names': detection.feature_names, 'originals': detection.original_feature_values,
    }, timeout=timeout)
    result.update({'detection_id': detection.id, 'dataset': detection.dataset_id,
                   'task': detection.model.task, 'model': detection.model.name})
    return _persist(db, detection, method, result)


def generate_shap_for_detection(db: Session, detection: Detection, timeout: float = 20.0) -> dict[str, Any]:
    return _generate(db, detection, 'shap', timeout)


def generate_lime_for_detection(db: Session, detection: Detection, timeout: float = 20.0) -> dict[str, Any]:
    return _generate(db, detection, 'lime', timeout)


def explain_detection(detection_id: str) -> dict[str, Any]:
    """Backward-compatible file/API helper; DB persistence is preferred."""
    from app.db.session import get_engine
    from sqlalchemy.orm import joinedload
    with Session(get_engine()) as db:
        detection = db.execute(
            select(Detection).options(joinedload(Detection.model)).where(Detection.id == detection_id)
        ).scalar_one_or_none()
        if detection is None:
            raise FileNotFoundError(detection_id)
        result = generate_shap_for_detection(db, detection)
        db.commit()
        return result

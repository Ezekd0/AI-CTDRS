from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import get_settings
from app.core.security import require_roles
from app.db.models.models import Detection, Explanation
from app.db.session import get_db
from app.modules.explainability.service import generate_shap_for_detection, generate_lime_for_detection

router = APIRouter(prefix='/explain', tags=['Explainability'])


def _safe(v: str):
    if not v or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in v):
        raise HTTPException(400, 'Invalid selector')
    return v


def _detection(db: Session, detection_id: str) -> Detection:
    d = db.execute(
        select(Detection).options(joinedload(Detection.model), joinedload(Detection.dataset)).where(Detection.id == detection_id)
    ).scalar_one_or_none()
    if d is None:
        raise HTTPException(404, 'Detection not found')
    if not d.transformed_features or not d.feature_names:
        raise HTTPException(422, 'Detection does not contain the feature data required for explanation')
    return d


@router.get('/shap/global')
def get_global_shap(dataset: str = Query(...), task: str = Query(...), model: str = Query(...), user=Depends(require_roles('administrator', 'analyst', 'viewer'))):
    from ml.training.datasets import resolve_dataset
    try:
        slug = resolve_dataset(dataset).slug
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    p = get_settings().resolved_artifacts_root / slug / _safe(task) / _safe(model) / 'shap_global.json'
    if not p.exists():
        raise HTTPException(404, 'Global SHAP explanation has not been generated for this model run')
    return json.loads(p.read_text())


def _stored_explanation(db: Session, detection_id: str, method: str):
    row = db.execute(select(Explanation).where(Explanation.detection_id == detection_id, Explanation.method == method)).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, f'{method.upper()} explanation has not been generated for this detection')
    return row.payload


@router.get('/shap/{detection_id}')
def get_shap_detection(detection_id: str, user=Depends(require_roles('administrator', 'analyst', 'viewer')), db: Session = Depends(get_db)):
    _detection(db, detection_id)
    return _stored_explanation(db, detection_id, 'shap')


@router.post('/shap/{detection_id}')
def create_shap_detection(detection_id: str, user=Depends(require_roles('administrator', 'analyst')), db: Session = Depends(get_db)):
    d = _detection(db, detection_id)
    try:
        result = generate_shap_for_detection(db, d)
        db.commit()
        return result
    except FileNotFoundError:
        raise HTTPException(404, 'Model artifact not found for this detection')
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f'SHAP explanation failed: {exc}')


@router.get('/lime/{detection_id}')
def get_lime_detection(detection_id: str, user=Depends(require_roles('administrator', 'analyst', 'viewer')), db: Session = Depends(get_db)):
    _detection(db, detection_id)
    return _stored_explanation(db, detection_id, 'lime')


@router.post('/lime/{detection_id}')
def create_lime_detection(detection_id: str, user=Depends(require_roles('administrator', 'analyst')), db: Session = Depends(get_db)):
    d = _detection(db, detection_id)
    try:
        result = generate_lime_for_detection(db, d)
        db.commit()
        return result
    except FileNotFoundError:
        raise HTTPException(404, 'Model artifact not found for this detection')
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f'LIME explanation failed: {exc}')

from __future__ import annotations
import json, uuid, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sqlalchemy import select
from app.db.models import Dataset, Model, Detection, Explanation, Alert
from app.db.session import get_engine
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.modules.prediction.runtime import cached_bundle, cached_preprocessor
from time import monotonic
import logging

logger = logging.getLogger(__name__)
EXPLANATION_BUDGET_SECONDS = 20.0

SUPPORTED = {'cicids2017', 'cicids2018', 'nsl-kdd'}
MODELS = {'random_forest', 'xgboost'}


def _artifact_slug(dataset: str) -> str:
    from ml.training.datasets import resolve_dataset
    return resolve_dataset(dataset).slug


def _severity(probability: float) -> str:
    return 'critical' if probability >= 0.9 else 'high' if probability >= 0.75 else 'medium' if probability >= 0.5 else 'low'


def _should_alert(prediction: str, probability: float, threshold: float) -> bool:
    normalized = prediction.strip().lower().rstrip('.')
    benign = {'benign', 'normal'}
    return probability >= threshold and normalized not in benign


def _validate_features(features: dict[str, Any]) -> None:
    if not isinstance(features, dict) or not features:
        raise ValueError('At least one network feature is required')
    for name, value in features.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Feature names must be non-empty strings')
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f'Feature {name!r} must be a scalar value')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Feature {name!r} must be finite')

def _safe_component(value: str) -> str:
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in value):
        raise ValueError('Invalid artifact selector')
    return value

def _load_preprocessor(dataset: str, task: str, model: str):
    return cached_preprocessor(get_settings().resolved_artifacts_root, _safe_component(dataset),
                               _safe_component(task), _safe_component(model))

def _normalize_feature_key(name: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in name.strip().lower()).strip('_')


def _transform(preprocessor, features: dict[str, Any], expected: list[str]) -> tuple[np.ndarray, list[Any]]:
    expected_normalized = {_normalize_feature_key(name): name for name in expected}
    provided_normalized = {_normalize_feature_key(str(name)) for name in features}
    recognized = provided_normalized & set(expected_normalized)
    if not recognized:
        raise ValueError('Input does not contain any recognized feature names for the selected model')
    frame = pd.DataFrame([features])
    if hasattr(preprocessor, 'transform_features'):
        out = preprocessor.transform_features(frame, scaled=True)
    elif hasattr(preprocessor, 'transform'):
        out = preprocessor.transform(frame)
    else:
        raise ValueError('Unsupported preprocessing artifact')
    if hasattr(out, 'to_numpy'): arr = out.to_numpy(dtype=np.float32)
    else: arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 1 or arr.shape[1] != len(expected):
        raise ValueError(f'Preprocessed feature count mismatch: expected {len(expected)}, got {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError('Preprocessing produced non-finite feature values')
    normalized_features = {_normalize_feature_key(str(k)): v for k, v in features.items()}
    originals = [normalized_features.get(_normalize_feature_key(name)) for name in expected]
    return arr, originals

def predict(request, user=None) -> dict[str, Any]:
    s = get_settings()
    user_id = getattr(user, 'id', None) if user is not None else None
    if request.dataset not in SUPPORTED or request.model not in MODELS: raise ValueError('Unsupported dataset or model')
    if len(request.features) > s.max_prediction_features: raise ValueError(f'Too many input features; maximum is {s.max_prediction_features}')
    _validate_features(request.features)
    bundle = cached_bundle(s.resolved_artifacts_root, request.dataset, request.task, request.model)
    X, originals = _transform(_load_preprocessor(request.dataset, request.task, request.model), request.features, bundle.feature_names)
    probabilities = np.asarray(bundle.predict_proba(X))[0]
    idx = int(np.argmax(probabilities))
    probability_map = {str(name): float(probabilities[i]) for i, name in enumerate(bundle.class_names)}
    detection_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat(timespec='seconds')
    payload = {
        'detection_id': detection_id, 'dataset': request.dataset, 'task': request.task, 'model': request.model,
        'prediction': bundle.class_names[idx], 'prediction_index': idx, 'probability': float(probabilities[idx]),
        'probabilities': probability_map, 'original_feature_values': originals, 'raw_features': request.features,
        'transformed_features': X[0].tolist(), 'feature_names': bundle.feature_names, 'created_at_utc': created,
    }
    # Persist the real prediction and its input to PostgreSQL (or the configured SQLAlchemy URL).
    explanation = {'shap': None, 'lime': None}
    with Session(get_engine()) as db:
        dataset_row = db.get(Dataset, request.dataset)
        if dataset_row is None:
            from ml.training.datasets import resolve_dataset
            spec = resolve_dataset(request.dataset)
            dataset_row = Dataset(id=spec.key, name=spec.display_name)
            db.add(dataset_row); db.flush()
        model_row = db.scalar(select(Model).where(Model.dataset_id == dataset_row.id, Model.task == request.task, Model.name == request.model))
        if model_row is None:
            model_row = Model(dataset_id=dataset_row.id, task=request.task, name=request.model, artifact_key=f'{_artifact_slug(request.dataset)}/{request.task}/{request.model}')
            db.add(model_row); db.flush()
        detection_row = Detection(id=detection_id, user_id=user_id, dataset_id=dataset_row.id, model_id=model_row.id,
            prediction=bundle.class_names[idx], prediction_index=idx, probability=float(probabilities[idx]),
            probabilities=probability_map, raw_features=request.features, original_feature_values=originals,
            transformed_features=X[0].tolist(), feature_names=bundle.feature_names, explanation_status={})
        db.add(detection_row)
        if _should_alert(bundle.class_names[idx], float(probabilities[idx]), s.alert_probability_threshold):
            db.add(Alert(
                detection_id=detection_id,
                severity=_severity(float(probabilities[idx])),
                status='new',
                message=f'Automated detection alert: {bundle.class_names[idx]} at {probabilities[idx]:.2%} confidence.',
            ))
        db.commit()
    root = s.resolved_reports_root / 'detections'; root.mkdir(parents=True, exist_ok=True)
    (root / f'{detection_id}.json').write_text(json.dumps(payload, indent=2))

    if request.generate_shap or request.generate_lime:
        from app.modules.explainability.service import generate_shap_for_detection, generate_lime_for_detection
        deadline = monotonic() + EXPLANATION_BUDGET_SECONDS
        for method, requested, generate in (
            ('shap', request.generate_shap, generate_shap_for_detection),
            ('lime', request.generate_lime, generate_lime_for_detection),
        ):
            if not requested:
                continue
            status = 'unavailable: explanations are disabled on this server'
            if s.enable_explanations:
                try:
                    # Independent transactions: failed explanation persistence cannot
                    # roll back the already committed prediction or another explanation.
                    with Session(get_engine()) as db:
                        persisted = db.get(Detection, detection_id)
                        generate(db, persisted, timeout=deadline - monotonic())
                        db.commit()
                    status = 'generated'
                except TimeoutError as exc:
                    status = f'timed_out: {exc}'
                except Exception as exc:
                    logger.warning('%s explanation unavailable for %s: %s', method, detection_id, exc)
                    status = f'unavailable: {exc}'
            explanation[method] = status
        try:
            with Session(get_engine()) as db:
                persisted = db.get(Detection, detection_id)
                persisted.explanation_status = explanation
                db.commit()
        except Exception:
            logger.exception('Could not persist explanation status for %s', detection_id)
    return {**payload, 'explanation': explanation}

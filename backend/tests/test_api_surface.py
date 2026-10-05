"""Executable API coverage for the production resource surface."""
from __future__ import annotations

import json
import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.main import app
from app.db.models import Alert, Base, Detection, Dataset, Model, User
from app.db.session import get_db, get_engine
from app.core.config import get_settings
from app.core.security import create_access_token


def _setup(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'api.db'}"
    artifact_root = tmp_path / 'artifacts'
    import os
    os.environ['DATABASE_URL'] = db_url
    os.environ['ARTIFACTS_ROOT'] = str(artifact_root)
    os.environ['REPORTS_ROOT'] = str(tmp_path / 'reports')
    get_settings.cache_clear(); get_engine.cache_clear()
    Base.metadata.create_all(get_engine())
    with Session(get_engine()) as db:
        db.add_all([
            User(id='admin', email='admin@example.com', password_hash='x', role='administrator', is_active=True),
            User(id='analyst', email='analyst@example.com', password_hash='x', role='analyst', is_active=True),
            User(id='viewer', email='viewer@example.com', password_hash='x', role='viewer', is_active=True),
        ])
        db.commit()
    def override_db():
        with Session(get_engine()) as db:
            yield db
    app.dependency_overrides[get_db] = override_db
    return TestClient(app), artifact_root


def _token(user_id):
    with Session(get_engine()) as db:
        return create_access_token(db.get(User, user_id))[0]


def _artifact(root):
    d = root / 'nsl_kdd' / 'binary' / 'random_forest'; d.mkdir(parents=True)
    X = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]], dtype=np.float32)
    y = np.array([0,0,0,1,1,1])
    joblib.dump(RandomForestClassifier(n_estimators=20, random_state=42).fit(X,y), d/'model.joblib')
    joblib.dump(FunctionTransformer(validate=False), d/'preprocessor.joblib')
    joblib.dump(LabelEncoder().fit(['normal','ATTACK']), d/'label_encoder.joblib')
    (d/'feature_names.json').write_text(json.dumps(['x1','x2']))
    (d/'metrics.json').write_text(json.dumps({'accuracy':1,'precision':1,'recall':1,'f1_score':1,'average':'binary','roc_auc':{'binary':1},'confusion_matrix':{'labels':['normal','ATTACK'],'counts':[[3,0],[0,3]]}}))
    (d/'metadata.json').write_text(json.dumps({'dataset':'nsl-kdd','task':'binary','model':'random_forest','class_names':['normal','ATTACK'],'n_features':2,'artifact_files':{'model':'model.joblib'}}))


def test_api_resource_surface_and_prediction(tmp_path):
    client, root = _setup(tmp_path)
    try:
        viewer = {'Authorization': f'Bearer {_token("viewer") }'}
        analyst = {'Authorization': f'Bearer {_token("analyst") }'}
        for path in ['/api/datasets','/api/models','/api/detections','/api/alerts','/api/reports']:
            r = client.get(path, headers=viewer)
            assert r.status_code == 200, (path, r.text)
        r = client.get('/api/responses', headers=analyst)
        assert r.status_code == 200, ("/api/responses", r.text)
        _artifact(root)
        r = client.post('/api/predict', headers=analyst, json={
            'dataset':'nsl-kdd','task':'binary','model':'random_forest',
            'features':{'x1':2,'x2':1}, 'generate_shap':False, 'generate_lime':False,
        })
        assert r.status_code == 201, r.text
        detection_id = r.json()['detection_id']
        assert client.get(f'/api/detections/{detection_id}', headers=viewer).status_code == 200
        # GET is read-only: it does not silently generate a missing explanation.
        assert client.get(f'/api/explain/shap/{detection_id}', headers=viewer).status_code == 404
        assert client.post(f'/api/explain/shap/{detection_id}', headers=viewer).status_code == 403
        assert client.get('/api/dashboard', headers=viewer).status_code == 200
        assert client.get('/api/reports', headers=viewer).status_code == 200
        response = client.post(f'/api/responses/{detection_id}/actions', headers=analyst, json={'action':'simulate_ip_block','reason':'API integration test','confirmed':True})
        assert response.status_code == 200, response.text
        assert response.json()['status'] == 'simulated'
        report = client.post(f'/api/reports/detections/{detection_id}', headers=analyst)
        assert report.status_code == 201, report.text
        assert report.json()['report']['detection']['prediction'] == 'ATTACK'
    finally:
        app.dependency_overrides.clear()


def test_prediction_rejects_unknown_only_features(tmp_path):
    client, root = _setup(tmp_path)
    try:
        _artifact(root)
        analyst = {'Authorization': f'Bearer {_token("analyst") }'}
        r = client.post('/api/predict', headers=analyst, json={
            'dataset':'nsl-kdd','task':'binary','model':'random_forest','features':{'not_a_feature':123},
        })
        assert r.status_code == 422, r.text
    finally:
        app.dependency_overrides.clear()

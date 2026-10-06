from fastapi import APIRouter
from app.api.v1.routes import health, model_evaluations, explainability, predict, detections, models, datasets, alerts, reports, responses, auth, dashboard, training, admin, network
api_router=APIRouter(prefix='/api')
for r in (health, model_evaluations, explainability, predict, detections, models, datasets, alerts, reports, responses, auth, dashboard, training, admin, network): api_router.include_router(r.router)

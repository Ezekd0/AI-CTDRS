from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import get_settings
router=APIRouter(tags=['System'])
class HealthResponse(BaseModel): status:str; app:str; version:str; environment:str
@router.get('/health', response_model=HealthResponse)
def health():
    s=get_settings(); return HealthResponse(status='ok',app=s.app_name,version=s.app_version,environment=s.environment)

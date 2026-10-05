from fastapi import APIRouter, Depends
from app.core.security import require_roles
from app.core.config import get_settings
from ml.training.artifacts import list_public_summaries
router=APIRouter(prefix='/models',tags=['Models'])
@router.get('')
def models(user=Depends(require_roles('administrator', 'analyst', 'viewer'))): return {'items': list_public_summaries(get_settings().resolved_artifacts_root)}

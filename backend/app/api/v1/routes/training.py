from fastapi import APIRouter, Depends, HTTPException
from app.core.security import require_roles
router = APIRouter(prefix='/models', tags=['Models'])
@router.post('/train', status_code=202)
def train(user=Depends(require_roles('administrator'))):
    raise HTTPException(status_code=501, detail='Training job integration is not configured in this deployment')

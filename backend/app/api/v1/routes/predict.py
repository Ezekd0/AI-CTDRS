from fastapi import Depends
from app.core.security import require_roles
from fastapi import APIRouter
from app.schemas.common import PredictionRequest, PredictionResponse
from app.modules.prediction.service import predict
router = APIRouter(prefix='/predict', tags=['Prediction'])
@router.post('', response_model=PredictionResponse, status_code=201)
def create_prediction(request: PredictionRequest, user=Depends(require_roles('administrator', 'analyst'))):
    return predict(request, user=user)

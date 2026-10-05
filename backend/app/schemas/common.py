from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal['random_forest', 'xgboost']
TaskName = Literal['binary', 'multiclass']
DatasetName = Literal['cicids2017', 'cicids2018', 'nsl-kdd']

class ErrorResponse(BaseModel):
    detail: str

class ModelRef(BaseModel):
    dataset: DatasetName
    task: TaskName
    model: ModelName

class PredictionRequest(ModelRef):
    features: dict[str, Any] = Field(..., min_length=1)
    generate_shap: bool = False
    generate_lime: bool = False

class ExplanationStatus(BaseModel):
    shap: str | None = None
    lime: str | None = None

class PredictionResponse(BaseModel):
    detection_id: str
    dataset: str
    task: str
    model: str
    prediction: str
    prediction_index: int
    probability: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    explanation: ExplanationStatus
    created_at_utc: str

class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    detection_id: str
    dataset: str
    task: str
    model: str
    prediction: str
    probability: float
    created_at_utc: str

class ModelSummary(BaseModel):
    model: str
    dataset: str
    task: str
    class_names: list[str]
    n_features: int
    metrics: dict[str, Any] | None = None

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'

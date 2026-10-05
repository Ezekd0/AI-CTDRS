"""Environment-driven application configuration."""
from functools import lru_cache
from pathlib import Path
from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False, populate_by_name=True)
    app_name: str = 'AI-CTDRS'
    app_version: str = '1.0.0'
    environment: str = 'development'
    log_level: str = 'INFO'
    database_url: str = Field(..., min_length=1)
    secret_key: str = Field(..., min_length=32, validation_alias=AliasChoices('JWT_SECRET_KEY', 'SECRET_KEY'))
    jwt_algorithm: str = 'HS256'
    access_token_expire_minutes: int = Field(30, gt=0)
    cors_origins: str = 'http://localhost:5173'
    artifacts_root: Path = Path('ml/artifacts')
    reports_root: Path = Path('reports')
    max_prediction_features: int = Field(1000, gt=0, le=10000)
    enable_explanations: bool = True
    enable_api_docs: bool = False
    response_mode: str = 'SIMULATION'
    alert_probability_threshold: float = Field(0.75, ge=0, le=1)

    @field_validator('jwt_algorithm')
    @classmethod
    def _jwt_algorithm_secure(cls, v: str) -> str:
        algorithm = v.strip().upper()
        if algorithm not in {'HS256'}:
            raise ValueError('JWT_ALGORITHM must be HS256')
        return algorithm

    @field_validator('cors_origins')
    @classmethod
    def _cors_origins_safe(cls, v: str) -> str:
        origins = [x.strip() for x in v.split(',') if x.strip()]
        if not origins:
            raise ValueError('CORS_ORIGINS must contain at least one explicit origin')
        if '*' in origins:
            raise ValueError('CORS_ORIGINS cannot contain * when credentials are enabled')
        for origin in origins:
            if not (origin.startswith('http://') or origin.startswith('https://')):
                raise ValueError('CORS_ORIGINS must contain explicit http(s) origins')
        return ','.join(origins)

    @field_validator('response_mode')
    @classmethod
    def _response_mode_simulation_default(cls, v: str) -> str:
        mode = v.strip().upper()
        if mode != 'SIMULATION':
            raise ValueError('Only SIMULATION response mode is enabled; external response integrations are not configured.')
        return mode

    @field_validator('secret_key')
    @classmethod
    def _secret_long_enough(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError('SECRET_KEY must be at least 32 characters')
        return v

    @property
    def response_simulation_mode(self) -> bool:
        return self.response_mode == 'SIMULATION'

    @property
    def cors_origin_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(',') if x.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def resolved_artifacts_root(self) -> Path:
        return (self.project_root / self.artifacts_root).resolve() if not self.artifacts_root.is_absolute() else self.artifacts_root.resolve()

    @property
    def resolved_reports_root(self) -> Path:
        return (self.project_root / self.reports_root).resolve() if not self.reports_root.is_absolute() else self.reports_root.resolve()

@lru_cache
def get_settings() -> Settings:
    return Settings()

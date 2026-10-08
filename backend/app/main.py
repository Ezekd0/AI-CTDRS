import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.db.session import get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.resolved_reports_root.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s %s (%s)", settings.app_name, settings.app_version, settings.environment)
    if settings.enable_explanations and settings.environment.lower() in {'production', 'prod'}:
        from app.modules.explainability.worker import worker
        worker.start_warmup(settings.resolved_artifacts_root)
    yield
    logger.info("Stopping %s", settings.app_name)


def create_app() -> FastAPI:
    s = get_settings()
    setup_logging(s.log_level)
    docs_enabled = s.environment.lower() not in {"production", "prod"} or s.enable_api_docs
    app = FastAPI(
        title=s.app_name,
        version=s.app_version,
        description="AI-CTDRS production API",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Cache-Control", "no-store" if request.url.path.startswith("/api/") else "no-cache")
        if s.environment.lower() in {"production", "prod"} and request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    app.include_router(api_router)

    @app.get("/health", tags=["System"])
    def health():
        try:
            with get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            logger.exception("Health check database failure")
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        return {
            "status": "ok",
            "app": s.app_name,
            "version": s.app_version,
            "environment": s.environment,
            "response_simulation_mode": s.response_simulation_mode,
        }

    @app.get("/api/v1/health", tags=["System"], include_in_schema=False)
    def legacy_health():
        return health()

    @app.get("/health/ready", tags=["System"])
    def readiness():
        try:
            with get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        artifact_root = s.resolved_artifacts_root
        has_artifacts = any(artifact_root.glob("*/*/*/metadata.json")) if artifact_root.exists() else False
        if not has_artifacts:
            raise HTTPException(status_code=503, detail="No trained model artifacts are installed")
        return {"status": "ready"}

    register_exception_handlers(app)
    return app


app = create_app()

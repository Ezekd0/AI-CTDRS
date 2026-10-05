from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
import logging

logger = logging.getLogger(__name__)

def register_exception_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={'detail': 'Request validation failed', 'errors': exc.errors()})

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={'detail': str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception('Unhandled API error: %s', exc)
        return JSONResponse(status_code=500, content={'detail': 'Internal server error'})

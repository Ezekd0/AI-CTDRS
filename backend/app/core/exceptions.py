from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)

def register_exception_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={'detail': 'Request validation failed', 'errors': [{'loc': error['loc'], 'type': error['type'], 'msg': error['msg']} for error in exc.errors()]})

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={'detail': str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception('Unhandled API error: %s', exc)
        # ServerErrorMiddleware handles these outside the normal CORS middleware.
        # Preserve the configured origin policy so browsers can read the HTTP 500.
        headers = {}
        origin = request.headers.get('origin')
        if origin in get_settings().cors_origin_list:
            headers = {'Access-Control-Allow-Origin': origin,
                       'Access-Control-Allow-Credentials': 'true', 'Vary': 'Origin'}
        return JSONResponse(status_code=500, content={'detail': 'Internal server error'}, headers=headers)

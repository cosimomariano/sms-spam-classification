from __future__ import annotations
from fastapi import FastAPI
from app.api.routes.api import router as api_router
from app.core.config import settings
from app.core.error_handlers import register_error_handlers
from app.core.logging import configure_logging
from app.core.openapi import load_contract_openapi

configure_logging()
app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(api_router)
register_error_handlers(app)
app.openapi = load_contract_openapi
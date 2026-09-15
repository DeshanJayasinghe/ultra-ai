from fastapi import FastAPI

from wearwise_ai.core.logging import configure_logging
from wearwise_ai.core.middleware import CorrelationIdMiddleware
from wearwise_ai.routes.health import router as health_router


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="WearWise AI",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(health_router)

    return app

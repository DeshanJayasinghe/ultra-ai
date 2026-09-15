from fastapi import APIRouter, Request

from wearwise_ai.core.config import get_settings

router = APIRouter()


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def readiness() -> dict[str, str]:
    settings = get_settings()

    if not settings.service_name:
        return {"status": "unavailable"}

    return {"status": "ok"}


@router.get("/internal/v1/service/manifest")
def service_manifest(request: Request) -> dict[str, object]:
    settings = get_settings()

    return {
        "data": {
            "service": settings.service_name,
            "environment": settings.environment,
            "model_manifest_path": settings.model_manifest_path,
        },
        "meta": {
            "request_id": request.state.correlation_id,
        },
    }

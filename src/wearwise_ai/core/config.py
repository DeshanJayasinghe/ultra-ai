from functools import lru_cache
from os import getenv

from pydantic import BaseModel, Field


class Settings(BaseModel):
    environment: str = Field(default="local")
    service_name: str = Field(default="wearwise-ai")
    model_manifest_path: str = Field(default="models/manifest.json")
    artifact_root: str = Field(default=".")


@lru_cache
def get_settings() -> Settings:
    return Settings(
        environment=getenv("WEARWISE_ENVIRONMENT", "local"),
        service_name=getenv("WEARWISE_SERVICE_NAME", "wearwise-ai"),
        model_manifest_path=getenv("WEARWISE_MODEL_MANIFEST_PATH", "models/manifest.json"),
        artifact_root=getenv("WEARWISE_AI_ARTIFACT_ROOT", "."),
    )

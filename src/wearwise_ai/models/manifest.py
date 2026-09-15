from pydantic import BaseModel, Field


class ModelManifestEntry(BaseModel):
    name: str
    version: str
    task: str
    enabled: bool = Field(default=True)

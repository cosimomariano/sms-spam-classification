from __future__ import annotations

from app.generated.models import HealthResponse, ModelMetadataResponse
from app.services.artifacts_service import ArtifactsService


class ModelMetadataService:
    def __init__(self, artifacts_service: ArtifactsService) -> None:
        self.artifacts_service = artifacts_service

    def health(self) -> HealthResponse:
        try:
            manifest = self.artifacts_service.load_manifest()
            self.artifacts_service.load_bundle() 
            return HealthResponse(status='UP', model_loaded=True, model_version=manifest['model_version'])
        except Exception:
            return HealthResponse(status='DEGRADED', model_loaded=False, model_version=None)

    def model_metadata(self) -> ModelMetadataResponse:
        return ModelMetadataResponse(**self.artifacts_service.load_manifest())
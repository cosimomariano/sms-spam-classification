from __future__ import annotations

from fastapi import APIRouter, Depends

from app.generated.models import (
    HealthResponse, 
    ModelMetadataResponse, 
    PredictRequest, 
    PredictResponse
)
from app.services.artifacts_service import ArtifactsService
from app.services.metadata_service import ModelMetadataService
from app.services.inference_service import InferenceService
from ml.pipeline.memoization import MemoizationService

## Router per lo smistamento delle chiamate alle rotte
router = APIRouter()

## Dipendenze 
def get_artifacts_service() -> ArtifactsService:
    return ArtifactsService()

def get_metadata_service(
    artifacts: ArtifactsService = Depends(get_artifacts_service)
) -> ModelMetadataService:
    return ModelMetadataService(artifacts)

def get_inference_service(
    artifacts: ArtifactsService = Depends(get_artifacts_service)
) -> InferenceService:
    return InferenceService(artifacts, MemoizationService())


## Rotte
@router.get('/health', response_model=HealthResponse, tags=['system'])
def health(
    service: ModelMetadataService = Depends(get_metadata_service)
) -> HealthResponse:
    """Verifica lo stato di salute del servizio e il caricamento del modello."""
    return service.health()


@router.get('/model-metadata', response_model=ModelMetadataResponse, tags=['system'])
def model_metadata(
    service: ModelMetadataService = Depends(get_metadata_service)
) -> ModelMetadataResponse:
    """Recupera i metadati del modello attualmente in uso per il serving."""
    return service.model_metadata()


@router.post('/predict', response_model=PredictResponse, tags=['prediction'])
def predict(
    request: PredictRequest, 
    service: InferenceService = Depends(get_inference_service)
) -> PredictResponse:
    """Classifica un messaggio testuale come SMS o HAM in tempo reale."""
    return service.predict(request.text)
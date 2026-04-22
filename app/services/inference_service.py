from __future__ import annotations
from app.generated.models import PredictResponse
from app.services.artifacts_service import ArtifactsService
from ml.pipeline.cleaning import TextCleaningService
from ml.pipeline.memoization import MemoizationService

class InferenceService:
    def __init__(self, artifacts_service: ArtifactsService, memoization: MemoizationService) -> None:
        self.artifacts_service = artifacts_service
        self.cleaner = TextCleaningService(memoization)

    def predict(self, text: str) -> PredictResponse:
        pipeline_model = self.artifacts_service.load_bundle()
        manifest = self.artifacts_service.load_manifest()
        
        clean_text = self.cleaner.clean_text(text)
        
        prediction = str(pipeline_model.predict([clean_text])[0])
        
        return PredictResponse(prediction=prediction, model_name=manifest['model_name'])
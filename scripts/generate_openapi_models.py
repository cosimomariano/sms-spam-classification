from __future__ import annotations
from pathlib import Path
MODELS = "\n# AUTO-GENERATED FROM openapi/openapi.yaml. Do not edit manually.\nfrom __future__ import annotations\n\nfrom typing import Any\nfrom pydantic import BaseModel, ConfigDict, Field\n\n\nclass PredictRequest(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n    text: str = Field(..., min_length=1, max_length=5000)\n\n\nclass PredictResponse(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n    prediction: str\n    model_name: str = Field(..., alias='model_name')\n\n\nclass HealthResponse(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n    status: str\n    model_loaded: bool = Field(..., alias='model_loaded')\n    model_version: str | None = Field(None, alias='model_version')\n\n\nclass ModelMetadataResponse(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n    model_name: str = Field(..., alias='model_name')\n    model_version: str = Field(..., alias='model_version')\n    trained_at: str = Field(..., alias='trained_at')\n    artifact_path: str = Field(..., alias='artifact_path')\n    training_dataset_hash: str = Field(..., alias='training_dataset_hash')\n    feature_config: dict[str, Any] = Field(..., alias='feature_config')\n    metrics: dict[str, Any]\n    memoization_formula: str = Field(..., alias='memoization_formula')\n\n\nclass ErrorResponse(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n    detail: str\n    error_code: str = Field(..., alias='error_code')\n"

def main() -> None:
    target = Path(__file__).resolve().parents[1] / 'app' / 'generated' / 'models.py'
    target.write_text(MODELS, encoding='utf-8')
    print(f'Generated {target}')

if __name__ == '__main__':
    main()
# AUTO-GENERATED FROM openapi/openapi.yaml. Do not edit manually.
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    prediction: str
    model_name: str = Field(..., alias='model_name')


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    status: str
    model_loaded: bool = Field(..., alias='model_loaded')
    model_version: str | None = Field(None, alias='model_version')


class ModelMetadataResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    model_name: str = Field(..., alias='model_name')
    model_version: str = Field(..., alias='model_version')
    trained_at: str = Field(..., alias='trained_at')
    artifact_path: str = Field(..., alias='artifact_path')
    training_dataset_hash: str = Field(..., alias='training_dataset_hash')
    feature_config: dict[str, Any] = Field(..., alias='feature_config')
    metrics: dict[str, Any]
    memoization_formula: str = Field(..., alias='memoization_formula')


class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    detail: str
    error_code: str = Field(..., alias='error_code')

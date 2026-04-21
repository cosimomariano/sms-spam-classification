from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

from app.core.config import settings
from app.core.exceptions import ModelNotAvailableError

@lru_cache(maxsize=1)
def _read_manifest_from_disk(path: Path) -> dict:
    if not path.exists():
        raise ModelNotAvailableError("Selected model manifest not found.")
    return json.loads(path.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def _read_bundle_from_disk(path: Path):
    if not path.exists():
        raise ModelNotAvailableError(
            f"Active model artifact not found: {path}. "
            f"Retrain the pipeline or regenerate the manifest."
        )
    return joblib.load(path)


class ArtifactsService:
    def load_manifest(self) -> dict:
        return _read_manifest_from_disk(settings.selected_model_manifest_path)

    def load_bundle(self):
        # Chiamiamo l'artefatto 'bundle' perché contiene sia il modello che il vectorizer
        manifest = self.load_manifest()
        bundle_path = Path(manifest["artifact_path"]).resolve()
        return _read_bundle_from_disk(bundle_path)

    def clear_cache(self) -> None:
        _read_manifest_from_disk.cache_clear()
        _read_bundle_from_disk.cache_clear()
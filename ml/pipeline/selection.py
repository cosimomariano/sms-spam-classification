from __future__ import annotations

import json
from datetime import datetime, timezone
from app.core.config import settings
from pathlib import Path

class ModelSelectionService:
    def select_best(self, results: list) -> dict:
        # Prendo il modello che ha avuto le migliori performance e salvo l'artefatto con i metadati associati in formato JSON
        best_result = max(results, key=self._ranking_key)
        manifest = self._build_manifest(best_result)

        self._save_manifest(manifest)

        return manifest

    def _ranking_key(self, result):
        metrics = result.metrics
        return metrics["f1"], metrics["accuracy"]

    def _build_manifest(self, result) -> dict:
        return {
            "model_name": result.model_name,
            "model_version": self._build_model_version(result),
            "trained_at": self._current_timestamp(),
            "artifact_path": self._build_relative_artifact_path(result.artifact_path),
            "training_dataset_hash": result.training_dataset_hash,
            "feature_config": result.feature_config,
            "metrics": result.metrics,
            "memoization_formula": settings.memoization_formula,
            "random_seed": result.random_seed,
        }

    from pathlib import Path

    def _build_relative_artifact_path(self, artifact_path: Path | str) -> str:
        artifact_path = Path(artifact_path).resolve()
        artifacts_root = settings.artifacts_dir.resolve()

        try:
            return artifact_path.relative_to(artifacts_root).as_posix()
        except ValueError:
            return Path(artifact_path.name).as_posix()

    def _build_model_version(self, result) -> str:
        return f"{result.model_name}-{settings.pipeline_version}"

    def _current_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _save_manifest(self, manifest: dict) -> None:
        content = json.dumps(manifest, indent=2, ensure_ascii=False)
        settings.selected_model_manifest_path.write_text(
            content,
            encoding="utf-8",
        )
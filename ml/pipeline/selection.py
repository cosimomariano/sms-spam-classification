from __future__ import annotations

import json
from datetime import datetime, timezone

from app.core.config import settings


class ModelSelectionService:
    def select_best(self, results: list) -> dict:
        best = max(results, key=lambda item: (item.metrics['f1'], item.metrics['accuracy']))
        manifest = {
            'model_name': best.model_name,
            'model_version': f"{best.model_name}-{settings.pipeline_version}",
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'artifact_path': str(best.artifact_path),
            'training_dataset_hash': best.training_dataset_hash,
            'feature_config': best.feature_config,
            'metrics': best.metrics,
            'memoization_formula': settings.memoization_formula,
            'random_seed': best.random_seed,
        }
        settings.selected_model_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        return manifest

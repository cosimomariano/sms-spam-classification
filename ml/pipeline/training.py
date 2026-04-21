from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from ml.models.factories import build_candidate_models, feature_config
from ml.pipeline.memoization import MemoizationService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelResult:
    model_name: str
    metrics: dict[str, float]
    artifact_path: Path
    training_dataset_hash: str
    feature_config: dict
    random_seed: int


class TrainingService:
    def __init__(self, memoization: MemoizationService) -> None:
        self.memoization = memoization

    def _split_data(self, cleaned_path: Path):
        cleaned_hash = self.memoization.file_sha256(cleaned_path)
        df = pd.read_csv(cleaned_path)
        X_train, X_test, y_train, y_test = train_test_split(df['clean_text'].fillna("").astype(str).str.strip(), 
                                                            df['label'].fillna("").astype(str).str.strip(), 
                                                            test_size=settings.test_size, 
                                                            random_state=settings.random_seed, 
                                                            stratify=df['label'])
        return X_train, X_test, y_train, y_test, cleaned_hash

    def _train_one(self, model_name, model, X_train, X_test, y_train, y_test, cleaned_hash):
        cfg = feature_config()
        key, payload = self.memoization.build_key(
            step_name=f'train_{model_name}',
            data_descriptor={'cleaned_dataset_hash': cleaned_hash},
            config_descriptor=cfg,
            params_descriptor={'model_name': model_name},
            random_descriptor={'seed': settings.random_seed},
        )
        model_path = settings.models_dir / f'{model_name}_{key}.joblib'
        metrics_path = settings.metrics_dir / f'{model_name}_{key}.json'
        if self.memoization.can_reuse(key=key, expected_payload=payload, required_artifacts=[model_path, metrics_path]):
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            logger.info('Skipping model training due to step memorization.', extra={'event': 'memoization_skip', 'extra_fields': {'step': f'train_{model_name}'}})
            return ModelResult(model_name, metrics, model_path, cleaned_hash, cfg, settings.random_seed)
        logger.info('Training model started.', extra={'event': 'step_started', 'extra_fields': {'step': f'train_{model_name}'}})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision_score(y_test, predictions, pos_label='spam', zero_division=0)),
            'recall': float(recall_score(y_test, predictions, pos_label='spam', zero_division=0)),
            'f1': float(f1_score(y_test, predictions, pos_label='spam', zero_division=0)),
        }
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        self.memoization.persist(key=key, payload=payload)
        logger.info('Training model completed.', extra={'event': 'step_completed', 'extra_fields': {'step': f'train_{model_name}', 'metrics': metrics}})
        return ModelResult(model_name, metrics, model_path, cleaned_hash, cfg, settings.random_seed)

    def train_all(self, cleaned_path: Path) -> list[ModelResult]:
        X_train, X_test, y_train, y_test, cleaned_hash = self._split_data(cleaned_path)
        results = []
        with ThreadPoolExecutor(max_workers=settings.parallel_workers) as executor:
            futures = {executor.submit(self._train_one, name, model, X_train, X_test, y_train, y_test, cleaned_hash): name for name, model in build_candidate_models().items()}
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda x: x.model_name)

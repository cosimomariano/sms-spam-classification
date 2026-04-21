from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from app.core.config import settings
from ml.pipeline.memoization import MemoizationService

logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
NUMBER_PATTERN = re.compile(r'\b\d+\b')
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
MULTISPACE_PATTERN = re.compile(r'\s+')
STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with'}


class TextCleaningService:
    def __init__(self, memoization: MemoizationService) -> None:
        self.memoization = memoization

    @staticmethod
    def clean_text(text: str) -> str:
        text = (text or '').lower()
        text = URL_PATTERN.sub(' url ', text)
        text = NUMBER_PATTERN.sub(' number ', text)
        text = NON_ALPHA_PATTERN.sub(' ', text)
        text = MULTISPACE_PATTERN.sub(' ', text).strip()
        return ' '.join(t for t in text.split(' ') if t and t not in STOPWORDS)

    def clean_dataset(self, normalized_path: Path) -> Path:
        target = settings.cleaned_dataset_path
        data_descriptor = {'normalized_dataset_hash': self.memoization.file_sha256(normalized_path)}
        config_descriptor = {'batch_size': settings.batch_size, 'cleaning_version': '1'}
        key, payload = self.memoization.build_key(step_name='clean_dataset', data_descriptor=data_descriptor, config_descriptor=config_descriptor, params_descriptor={})
        if self.memoization.can_reuse(key=key, expected_payload=payload, required_artifacts=[target]):
            logger.info('Skipping cleaning due to step memorization.', extra={'event': 'memoization_skip', 'extra_fields': {'step': 'clean_dataset'}})
            return target
        if target.exists():
            target.unlink()
        total_rows = 0
        for idx, chunk in enumerate(pd.read_csv(normalized_path, chunksize=settings.batch_size), start=1):
            cleaned = chunk.copy()
            cleaned['clean_text'] = cleaned['message'].map(self.clean_text)
            total_rows += len(cleaned)
            cleaned.to_csv(target, mode='a', index=False, header=not target.exists())
            logger.info('Cleaned batch processed.', extra={'event': 'batch_processed', 'extra_fields': {'step': 'clean_dataset', 'batch_index': idx, 'batch_rows': len(cleaned)}})
        self.memoization.persist(key=key, payload=payload)
        logger.info('Cleaning completed.', extra={'event': 'step_completed', 'extra_fields': {'step': 'clean_dataset', 'rows': total_rows}})
        return target

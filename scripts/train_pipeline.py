from __future__ import annotations

import logging

from app.core.logging import configure_logging
from ml.pipeline.cleaning import TextCleaningService
from ml.pipeline.ingestion import BatchIngestionService
from ml.pipeline.memoization import MemoizationService
from ml.pipeline.selection import ModelSelectionService
from ml.pipeline.training import TrainingService


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    memoization = MemoizationService()
    ingestion = BatchIngestionService(memoization)
    cleaning = TextCleaningService(memoization)
    training = TrainingService(memoization)
    selection = ModelSelectionService()
    logger.info('Pipeline started.', extra={'event': 'pipeline_started', 'extra_fields': {}})
    normalized = ingestion.normalize_dataset()
    cleaned = cleaning.clean_dataset(normalized)
    results = training.train_all(cleaned)
    manifest = selection.select_best(results)
    logger.info('Pipeline completed.', extra={'event': 'pipeline_completed', 'extra_fields': {'selected_model': manifest['model_name']}})
    print(f"Training completed. Active model: {manifest['model_name']}")


if __name__ == '__main__':
    main()

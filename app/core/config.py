from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='APP_', env_file='.env', extra='ignore')

    app_name: str = 'SMS Spam Classification API'
    app_version: str = '1.0.0'
    environment: str = 'local'

    # Percorsi utilizzati
    raw_dataset_path: Path = BASE_DIR / 'data' / 'raw' / 'sms-spam-collection.csv'
    normalized_dataset_path: Path = BASE_DIR / 'data' / 'processed' / 'dataset_normalized.csv'

    models_dir: Path = BASE_DIR / 'artifacts' / 'models'
    metrics_dir: Path = BASE_DIR / 'artifacts' / 'metrics'
    memo_dir: Path = BASE_DIR / 'artifacts' / 'memo'

    # Iperparametri e parametri associati alla pipeline
    batch_size: int = 500
    test_size: float = 0.2
    random_seed: int = 42
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    parallel_workers: int = 3

    pipeline_version: str = '2.0.0'

settings = Settings()
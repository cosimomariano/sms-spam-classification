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


    # Iperparametri e parametri associati alla pipeline
    batch_size: int = 500

settings = Settings()
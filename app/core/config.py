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
    cleaned_dataset_path: Path = BASE_DIR / 'data' / 'processed' / 'dataset_cleaned.csv'

    artifacts_dir: Path = BASE_DIR / 'artifacts'
    models_dir: Path = BASE_DIR / 'artifacts' / 'models'
    metrics_dir: Path = BASE_DIR / 'artifacts' / 'metrics'
    memo_dir: Path = BASE_DIR / 'artifacts' / 'memo'
    selected_model_manifest_path: Path = BASE_DIR / 'artifacts' / 'models' / 'selected_model.json'

    # Iperparametri e parametri associati alla pipeline
    batch_size: int = 500
    test_size: float = 0.2
    random_seed: int = 42
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    parallel_workers: int = 3

    pipeline_version: str = '2.0.0'
    memoization_formula: str = 'K = H(D,C,P,V[,R])'
    max_input_text_length: int = 5000
    cleaning_version: str = '1'

    def model_post_init(self, __context: dict) -> None:
        """Eseguito in automatico da Pydantic dopo l'inizializzazione della classe."""
        directories = [
            self.artifacts_dir, 
            self.models_dir, 
            self.metrics_dir, 
            self.memo_dir, 
            self.normalized_dataset_path.parent, 
            self.cleaned_dataset_path.parent
        ]
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
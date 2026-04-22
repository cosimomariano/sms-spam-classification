from __future__ import annotations
from pathlib import Path
import yaml
from app.core.config import settings

def load_contract_openapi() -> dict:
    contract_path = Path(__file__).resolve().parents[2] / 'openapi' / 'openapi.yaml'
    schema = yaml.safe_load(contract_path.read_text(encoding='utf-8'))
    schema['info']['title'] = settings.app_name
    schema['info']['version'] = settings.app_version
    return schema
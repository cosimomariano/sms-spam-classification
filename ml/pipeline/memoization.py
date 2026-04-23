from __future__ import annotations
import hashlib
import json

from pathlib import Path
from typing import Any
from app.core.config import settings

class MemoizationService:
    """Formula utilizzata per lo step memorization -> K = H(D,C,P,V[,R])."""

    @staticmethod
    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as fh:
            ## Leggo a blocchi di 8192 byte e calcolo l'hash senza caricare il tutto in RAM
            for chunk in iter(lambda: fh.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def stable_hash(payload: dict[str, Any]) -> str:
        # Faccio il dump del dizionario ordinandolo prima per chiave ASC per garantire 
        # che due dizionari identici producano la stessa stringa e di conseguenza stesso hash
        # indipendentemente dall'ordine di inserimento adottato
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    # Costruisco la chiave per lo step memoization
    def build_key(self, *, step_name: str, data_descriptor: dict[str, Any], config_descriptor: dict[str, Any], params_descriptor: dict[str, Any], random_descriptor: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        payload = {
            'step_name': step_name,
            'D': data_descriptor,
            'C': config_descriptor,
            'P': params_descriptor,
            'V': {'pipeline_version': settings.pipeline_version}, # La versione della pipeline è configurata da config.py per garantire facile modifica
        }

        if random_descriptor:
            payload['R'] = random_descriptor

        return self.stable_hash(payload), payload

    def memo_metadata_path(self, key: str) -> Path:
        # Ritorno il path di salvataggio dei metadati dello step
        return settings.memo_dir / f'{key}.json'

    def can_reuse(self, *, key: str, expected_payload: dict[str, Any], required_artifacts: list[Path]) -> bool:
        meta = self.memo_metadata_path(key)
        if not meta.exists() or not all(p.exists() for p in required_artifacts):
            return False
        return json.loads(meta.read_text(encoding='utf-8')) == expected_payload

    def persist(self, *, key: str, payload: dict[str, Any]) -> None:
        #Creo la dir di cache se non esiste
        settings.memo_dir.mkdir(parents=True, exist_ok=True)
        self.memo_metadata_path(key).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
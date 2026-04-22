from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from app.core.config import settings
from ml.pipeline.constants import LABEL_COLUMN, TEXT_COLUMN, VALID_LABELS
from ml.pipeline.memoization import MemoizationService

logger = logging.getLogger(__name__)

class BatchIngestionService:
    def __init__(self, memoization: MemoizationService) -> None:
        self.memoization = memoization

    def normalize_dataset(self) -> Path:
        # Path del file system per il salvataggio/lettura del dataset csv
        raw_path = settings.raw_dataset_path
        target = settings.normalized_dataset_path

        # Applicazione del memoizazion pattern e riutilizzo dello step eventuale
        data_descriptor = {'raw_dataset_hash': self.memoization.file_sha256(raw_path)}
        config_descriptor = {'batch_size': settings.batch_size, 
                             'text_column': TEXT_COLUMN, 
                             'label_column': LABEL_COLUMN}
        key, payload = self.memoization.build_key(step_name='normalize_dataset', 
                                                  data_descriptor=data_descriptor, 
                                                  config_descriptor=config_descriptor, 
                                                  params_descriptor={})
        
        # Controllo se posso riutilizzare uno step gia effettuato e ritornare il dataset gia normalizzato in precedenza
        if self.memoization.can_reuse(key=key, expected_payload=payload, required_artifacts=[target]):
            logger.info('Skipping normalization due to step memorization.', 
                        extra={'event': 'memoization_skip', 'extra_fields': {'step': 'normalize_dataset'}})
            return target
        
        # Verifico se è presente un file residuo non conforme all'attuale run e lo rimuovo per evitare duplicati eventuali
        if target.exists():
            target.unlink()

        total_rows = 0

        # Leggo il dataset a chunk sulla base del parametro batch_size della configurazione
        for idx, chunk in enumerate(pd.read_csv(raw_path, chunksize=settings.batch_size), start=1):
            normalized = chunk.rename(columns={chunk.columns[0]: LABEL_COLUMN, chunk.columns[1]: TEXT_COLUMN})[[LABEL_COLUMN, TEXT_COLUMN]].copy()
            normalized[LABEL_COLUMN] = normalized[LABEL_COLUMN].astype(str).str.strip().str.lower()
            normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].astype(str)
            # Scarto le righe con etichette errate
            normalized = normalized[normalized[LABEL_COLUMN].isin(VALID_LABELS)]
            total_rows += len(normalized)
            # Salvo il file in modalita append e scrivo l'header solo per il primo blocco per evitare di riscrivere piu volte l'intestazione
            normalized.to_csv(target, mode='a', index=False, header=not target.exists())
            logger.info('Normalized batch processed.', 
                        extra={'event': 'batch_processed', 
                               'extra_fields': {'step': 'normalize_dataset', 
                                                'batch_index': idx, 
                                                'batch_rows': len(normalized)}
                                                })
            
        # Creo il file json contenente i metadati dello step in modo tale da poter eventualmente riutilizzarlo nelle prossime run
        # con medesima chiave K    
        self.memoization.persist(key=key, payload=payload)
        logger.info('Normalization completed.', 
                    extra={'event': 'step_completed', 
                           'extra_fields': {'step': 'normalize_dataset', 
                                            'rows': total_rows}})
        
        return target
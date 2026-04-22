from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


## Imposto il formato JSON per essere conforme ad un eventuale upload su sistemi di consultazione di log distribuiti
## es. Kibana, Elastic search...
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'event'):
            payload['event'] = record.event
        if hasattr(record, 'extra_fields'):
            payload.update(record.extra_fields)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # La loggatura è impostata ad info per evitare di loggare informazioni di DEBUG 
    # non necessarie per questa tipologia di progetto
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

from ml.pipeline.cleaning import TextCleaningService
from ml.pipeline.memoization import MemoizationService

def test_clean_single_text_normalizes_content():
    """ Test per la corretta normalizzazione del testo presente nei messaggi SMS """
    service = TextCleaningService(MemoizationService())
    cleaned = service.clean_text('Win 100 now! Visit https://example.com')
    assert cleaned == 'win number now visit url'
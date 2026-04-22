import json
import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.main import app
from app.core.config import settings

def ensure_test_model():
    # Funzione per l'istanziazione di model di test fittizio addestrato su un piccolo insieme di dati per gli specifici unit test

    model_path = settings.models_dir / 'test_model.joblib'
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MultinomialNB())])
    pipeline.fit(['free prize now', 'win money fast', 'hello friend', 'see you tomorrow'], ['spam', 'spam', 'ham', 'ham'])
    joblib.dump(pipeline, model_path)

    manifest = {
        'model_name': 'multinomial_nb',
        'model_version': 'multinomial_nb-test',
        'trained_at': '2026-04-20T00:00:00Z',
        'artifact_path': str(model_path),
        'training_dataset_hash': 'testhash',
        'feature_config': {'max_features': None, 'ngram_range': [1, 1]},
        'metrics': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
        'memoization_formula': 'K = H(D,C,P,V[,R])',
    }

    settings.selected_model_manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

client = TestClient(app)

def test_health_endpoint():
    """ Test per l'health-check del servizio """
    ensure_test_model()
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'UP'


def test_model_metadata_endpoint():
    """ Test per la corretta stampa dei metadati """
    ensure_test_model()
    response = client.get('/model-metadata')
    assert response.status_code == 200
    assert response.json()['model_name'] == 'multinomial_nb'


def test_predict_endpoint():
    """ Test per la corretta predizione di un SMS tramite endpoint in real time """
    ensure_test_model()
    response = client.post('/predict', json={'text': 'free prize'})
    assert response.status_code == 200
    assert response.json()['prediction'] in {'spam', 'ham'}
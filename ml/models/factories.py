from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.core.config import settings


def feature_config() -> dict:
    return {'max_features': settings.max_features, 'ngram_range': (settings.ngram_min, settings.ngram_max), 'lowercase': False}


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=settings.max_features, ngram_range=(settings.ngram_min, settings.ngram_max), lowercase=False)


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        'random_forest': Pipeline([('tfidf', build_vectorizer()), ('classifier', RandomForestClassifier(n_estimators=100, random_state=settings.random_seed, n_jobs=1))]),
        'linear_svm': Pipeline([('tfidf', build_vectorizer()), ('classifier', LinearSVC(random_state=settings.random_seed))]),
        'multinomial_nb': Pipeline([('tfidf', build_vectorizer()), ('classifier', MultinomialNB())]),
    }

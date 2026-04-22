"""
Prediction Module
Loads trained models and provides sentiment prediction, VADER intensity,
and NRCLex emotion detection functions.
"""

import os
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

from ml.preprocess import clean_text


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_lr.joblib')
NB_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_nb.joblib')

# Global model cache (loaded once)
_models = {}
_vader = None


def _load_model(model_type='lr'):
    """Load a trained model pipeline from disk."""
    m_type = 'lr' if 'lr' in model_type.lower() or 'logistic' in model_type.lower() else 'nb'
    
    if m_type not in _models:
        path = LR_MODEL_PATH if m_type == 'lr' else NB_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. "
                f"Run 'python -m ml.train_model' to train first."
            )
        _models[m_type] = joblib.load(path)
    return _models[m_type]


def _get_vader():
    """Lazy-load VADER analyzer."""
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def predict_sentiment(text: str, model_type: str = 'lr') -> dict:
    """
    Predict sentiment for a single text.

    Args:
        text: Raw input text
        model_type: 'lr' for Logistic Regression, 'nb' for Naive Bayes

    Returns:
        dict with keys:
            - label: 'Positive', 'Negative', or 'Neutral'
            - confidence: float (0-100)
            - probabilities: dict of class probabilities
            - model_used: str name of model
    """
    if not text or not text.strip():
        return {
            'label': 'Neutral',
            'confidence': 0.0,
            'probabilities': {'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34},
            'model_used': 'none (empty input)',
        }

    model = _load_model(model_type)
    cleaned = clean_text(text)

    # Predict label
    label = model.predict([cleaned])[0]

    # Get probabilities
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_.tolist()
    probabilities = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}

    # Confidence = probability of predicted class
    confidence = round(float(max(proba)) * 100, 2)

    model_name = "Logistic Regression" if model_type == 'lr' else "Naive Bayes"

    return {
        'label': label,
        'confidence': confidence,
        'probabilities': probabilities,
        'model_used': model_name,
    }


def predict_batch(texts: list, model_type: str = 'lr') -> list:
    """
    Predict sentiment for a batch of texts.

    Args:
        texts: list of raw text strings
        model_type: 'lr' or 'nb'

    Returns:
        list of prediction dicts (same format as predict_sentiment)
    """
    if not texts:
        return []

    model = _load_model(model_type)
    cleaned = [clean_text(t) for t in texts]

    labels = model.predict(cleaned)
    probas = model.predict_proba(cleaned)
    classes = model.classes_.tolist()

    model_name = "Logistic Regression" if model_type == 'lr' else "Naive Bayes"

    results = []
    for i, (label, proba) in enumerate(zip(labels, probas)):
        probabilities = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
        confidence = round(float(max(proba)) * 100, 2)
        results.append({
            'text': texts[i][:200],  # truncate for display
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': model_name,
        })

    return results


def get_vader_scores(text: str) -> dict:
    """
    Get VADER sentiment intensity scores.

    Returns:
        dict with keys: compound, pos, neg, neu, intensity_pct
        - compound: float [-1, 1]
        - intensity_pct: float [0, 100] — mapped for the gauge
    """
    if not text or not text.strip():
        return {
            'compound': 0.0,
            'pos': 0.0,
            'neg': 0.0,
            'neu': 1.0,
            'intensity_pct': 50.0,
        }

    vader = _get_vader()
    scores = vader.polarity_scores(text)

    # Map compound score (-1 to 1) → intensity percentage (0 to 100)
    # -1.0 → 0% (most negative), 0.0 → 50% (neutral), 1.0 → 100% (most positive)
    intensity_pct = round((scores['compound'] + 1) * 50, 2)

    return {
        'compound': round(scores['compound'], 4),
        'pos': round(scores['pos'], 4),
        'neg': round(scores['neg'], 4),
        'neu': round(scores['neu'], 4),
        'intensity_pct': intensity_pct,
    }


def get_emotions(text: str) -> dict:
    """
    Detect emotions using NRCLex.

    Returns:
        dict with keys:
            - emotions: dict of emotion → frequency (0.0-1.0)
            - top_emotion: str name of strongest emotion
            - raw_counts: dict of emotion → count
    """
    if not text or not text.strip():
        return {
            'emotions': {},
            'top_emotion': 'none',
            'raw_counts': {},
        }

    try:
        emotion = NRCLex()
        emotion.load_raw_text(text)
        raw_counts = emotion.raw_emotion_scores
        frequencies = emotion.affect_frequencies

        # Remove 'positive' and 'negative' — we already have sentiment from ML model
        emotion_labels = ['fear', 'anger', 'anticipation', 'trust',
                          'surprise', 'sadness', 'disgust', 'joy']

        filtered = {k: round(v, 4) for k, v in frequencies.items()
                    if k in emotion_labels}
        filtered_counts = {k: v for k, v in raw_counts.items()
                          if k in emotion_labels}

        # Find top emotion
        if filtered:
            top = max(filtered, key=filtered.get)
        else:
            top = 'none'

        return {
            'emotions': filtered,
            'top_emotion': top,
            'raw_counts': filtered_counts,
        }
    except Exception:
        return {
            'emotions': {},
            'top_emotion': 'none',
            'raw_counts': {},
        }


def analyze_full(text: str, model_type: str = 'lr') -> dict:
    """
    Perform full analysis: sentiment + VADER + emotions.

    Returns combined result dict.
    """
    sentiment = predict_sentiment(text, model_type)
    vader = get_vader_scores(text)
    emotions = get_emotions(text)

    # Hybrid correction: if the ML model is unsure, use VADER's compound direction
    # to reduce obvious mistakes on negation-heavy / conversational inputs.
    try:
        conf = float(sentiment.get('confidence', 0.0))
        compound = float(vader.get('compound', 0.0))
        adjusted = False

        if conf < 60.0:
            if compound >= 0.2:
                sentiment['label'] = 'Positive'
                adjusted = True
            elif compound <= -0.2:
                sentiment['label'] = 'Negative'
                adjusted = True
            else:
                sentiment['label'] = 'Neutral'
                adjusted = True

        if adjusted:
            sentiment['model_used'] = f"Hybrid ({sentiment.get('model_used', 'ML')} + VADER)"
            sentiment['rule_adjusted'] = True
        else:
            sentiment['rule_adjusted'] = False
    except Exception:
        # Never fail the request due to hybrid post-processing
        sentiment['rule_adjusted'] = False

    return {
        'text': text[:500],
        'sentiment': sentiment,
        'vader': vader,
        'emotions': emotions,
    }

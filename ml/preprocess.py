"""
Text Preprocessing Module
Handles cleaning, tokenization, and lemmatization of text data.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Initialize once
_lemmatizer = WordNetLemmatizer()
_stop_words = None


def _get_stop_words():
    """Lazy-load stop words to avoid import-time NLTK errors."""
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
        # Keep negation words — they carry sentiment meaning
        negations = {'no', 'not', 'nor', 'neither', 'never', 'none',
                     'nobody', 'nothing', 'nowhere', 'hardly', 'barely',
                     'scarcely', "don't", "doesn't", "didn't", "isn't",
                     "aren't", "wasn't", "weren't", "won't", "wouldn't",
                     "shouldn't", "couldn't", "can't", "cannot"}
        _stop_words -= negations
    return _stop_words


def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise.
    
    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove @mentions and #hashtags symbols
        4. Remove HTML tags
        5. Remove punctuation
        6. Remove numbers
        7. Collapse multiple spaces
        8. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove @mentions (keep the word after @)
    text = re.sub(r'@(\w+)', r'\1', text)

    # Remove hashtag symbol (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove punctuation (but keep apostrophes for negation like "don't")
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenize text, remove stop words, and lemmatize each token.
    Returns a single string of processed tokens.
    """
    if not text:
        return ""

    tokens = word_tokenize(text)
    stop_words = _get_stop_words()

    processed = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            lemma = _lemmatizer.lemmatize(token, pos='v')  # verb lemmatization
            lemma = _lemmatizer.lemmatize(lemma, pos='n')  # noun lemmatization
            processed.append(lemma)

    return ' '.join(processed)


def preprocess_pipeline(text: str) -> str:
    """
    Full preprocessing pipeline: clean → tokenize → lemmatize.
    """
    cleaned = clean_text(text)
    processed = tokenize_and_lemmatize(cleaned)
    return processed


def preprocess_for_model(text: str) -> str:
    """
    Lighter preprocessing for the sklearn TfidfVectorizer.
    The vectorizer handles its own tokenization, so we just clean.
    """
    return clean_text(text)

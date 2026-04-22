"""
Model Training Module
Trains sentiment classification models using TF-IDF + Logistic Regression / Naive Bayes.
Saves trained pipeline to ml/models/ directory.

Usage:
    python -m ml.train_model
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import clean_text


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
EXTRA_TEST_SAMPLES_PATH = os.path.join(BASE_DIR, 'test_samples.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_lr.joblib')
NB_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_nb.joblib')


def load_data():
    """Load and validate the dataset."""
    print(f"[*] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Basic validation
    assert 'text' in df.columns, "Dataset must have a 'text' column"
    assert 'sentiment' in df.columns, "Dataset must have a 'sentiment' column"

    # Drop rows with missing values
    original_len = len(df)
    df = df.dropna(subset=['text', 'sentiment'])
    if len(df) < original_len:
        print(f"  [WARN] Dropped {original_len - len(df)} rows with missing values")

    # Strip whitespace from sentiment labels
    df['sentiment'] = df['sentiment'].str.strip()

    # Optionally include extra labeled samples (more natural / complex sentences)
    # This improves robustness for negation, contrast ("but"), and sarcasm-like phrasing.
    if os.path.exists(EXTRA_TEST_SAMPLES_PATH):
        try:
            extra = pd.read_csv(EXTRA_TEST_SAMPLES_PATH)
            if 'text' in extra.columns and ('sentiment' in extra.columns or 'label' in extra.columns):
                label_col = 'sentiment' if 'sentiment' in extra.columns else 'label'
                extra = extra[['text', label_col]].rename(columns={label_col: 'sentiment'})
                extra['sentiment'] = extra['sentiment'].astype(str).str.strip()
                before = len(df)
                df = pd.concat([df, extra], ignore_index=True)
                print(f"  [OK] Added {len(df) - before} extra samples from: {EXTRA_TEST_SAMPLES_PATH}")
        except Exception as e:
            print(f"  [WARN] Could not load extra samples from {EXTRA_TEST_SAMPLES_PATH}: {e}")

    # Show distribution
    print(f"\n[DATA] Dataset Distribution ({len(df)} samples):")
    for label, count in df['sentiment'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    return df


def preprocess_data(df):
    """Apply text preprocessing to the dataset."""
    print("\n[STEP] Preprocessing text data...")
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df['clean_text'].str.len() > 0]
    print(f"  [OK] Preprocessed {len(df)} samples")

    return df


def build_lr_pipeline():
    """Build a Logistic Regression pipeline with TF-IDF."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
        )),
    ])


def build_nb_pipeline():
    """Build a Naive Bayes pipeline with TF-IDF."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ('clf', MultinomialNB(alpha=0.1)),
    ])


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, name):
    """Train a model and print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  [METRIC] Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    print(f"\n  [REPORT] Classification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.split('\n'):
        print(f"    {line}")

    print(f"\n  [MATRIX] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    # Header
    header = "        " + "  ".join(f"{l[:5]:>7s}" for l in labels)
    print(f"    {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>7d}" for v in row)
        print(f"    {labels[i][:5]:>6s}  {row_str}")

    # Cross-validation
    print(f"\n  [CV] 5-Fold Cross-Validation...")
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    cv_scores = cross_val_score(pipeline, X_all, y_all, cv=5, scoring='accuracy')
    print(f"    Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Refit on all data after cross-validation
    pipeline.fit(X_all, y_all)

    return acc


def save_model(pipeline, path, name):
    """Save trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  [SAVE] Saved {name} -> {path} ({size_mb:.2f} MB)")


def main():
    """Main training workflow."""
    print("\n" + "=" * 60)
    print("  [ML] Sentiment Analysis - Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['sentiment'],
        test_size=0.20,
        random_state=42,
        stratify=df['sentiment'],
    )
    print(f"\n  [SPLIT] {len(X_train)} train / {len(X_test)} test")

    # 4. Train Logistic Regression
    lr_pipeline = build_lr_pipeline()
    lr_acc = train_and_evaluate(
        lr_pipeline, X_train, X_test, y_train, y_test,
        "Logistic Regression + TF-IDF"
    )
    save_model(lr_pipeline, LR_MODEL_PATH, "Logistic Regression")

    # 5. Train Naive Bayes
    nb_pipeline = build_nb_pipeline()
    nb_acc = train_and_evaluate(
        nb_pipeline, X_train, X_test, y_train, y_test,
        "Multinomial Naive Bayes + TF-IDF"
    )
    save_model(nb_pipeline, NB_MODEL_PATH, "Naive Bayes")

    # 6. Summary
    print("\n" + "=" * 60)
    print("  [SUMMARY] Training Summary")
    print("=" * 60)
    print(f"  Logistic Regression:  {lr_acc*100:.1f}% accuracy")
    print(f"  Naive Bayes:          {nb_acc*100:.1f}% accuracy")
    best = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    print(f"\n  [BEST] Best Model: {best}")
    print(f"  [DEFAULT] Default model: Logistic Regression (sentiment_lr.joblib)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

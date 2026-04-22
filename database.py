"""
Database Module
SQLite helper for storing and retrieving analysis history.
"""

import os
import sqlite3
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'history.db')


def _get_connection():
    """Get a database connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the analyses table if it does not exist."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT,
            vader_compound REAL,
            intensity_pct REAL,
            top_emotion TEXT,
            emotions_json TEXT,
            probabilities_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_analysis(text, sentiment, confidence, model_used='',
                  vader_compound=0.0, intensity_pct=50.0,
                  top_emotion='', emotions=None, probabilities=None):
    """Save a single analysis result to the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analyses
            (text, sentiment, confidence, model_used, vader_compound,
             intensity_pct, top_emotion, emotions_json, probabilities_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        text[:2000],  # Cap stored text length
        sentiment,
        confidence,
        model_used,
        vader_compound,
        intensity_pct,
        top_emotion,
        json.dumps(emotions or {}),
        json.dumps(probabilities or {}),
    ))
    conn.commit()
    conn.close()


def get_history(limit=50):
    """
    Retrieve analysis history, most recent first.

    Returns:
        list of dicts with all analysis fields
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM analyses
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            'id': row['id'],
            'text': row['text'],
            'sentiment': row['sentiment'],
            'confidence': row['confidence'],
            'model_used': row['model_used'],
            'vader_compound': row['vader_compound'],
            'intensity_pct': row['intensity_pct'],
            'top_emotion': row['top_emotion'],
            'emotions': json.loads(row['emotions_json'] or '{}'),
            'probabilities': json.loads(row['probabilities_json'] or '{}'),
            'created_at': row['created_at'],
        })

    return results


def get_sentiment_distribution():
    """
    Get aggregate sentiment counts for visualization.

    Returns:
        dict: {'Positive': count, 'Negative': count, 'Neutral': count}
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sentiment, COUNT(*) as count
        FROM analyses
        GROUP BY sentiment
    ''')
    rows = cursor.fetchall()
    conn.close()

    distribution = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for row in rows:
        distribution[row['sentiment']] = row['count']

    return distribution


def get_history_count():
    """Get total number of analyses stored."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM analyses')
    count = cursor.fetchone()['count']
    conn.close()
    return count


def clear_history():
    """Delete all analysis history."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analyses')
    conn.commit()
    conn.close()

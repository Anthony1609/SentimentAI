"""
Sentiment Analysis & Opinion Mining — Flask Application
Main entry point for the web server.
"""

import os
import csv
import io
from flask import (
    Flask, render_template, request, jsonify,
)
from database import init_db, save_analysis, get_history, get_sentiment_distribution, clear_history, get_history_count
from ml.predict import analyze_full


# ─── App Setup ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Initialize database
init_db()


# ─── Helper ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ─────────────────────────────────────────────────

@app.route('/')
def home():
    """Landing page."""
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page — high level metrics."""
    return render_template('dashboard.html')


@app.route('/analysis')
def analysis():
    """Analysis page — text input and file upload."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Simple health check endpoint for monitoring."""
    return jsonify({'status': 'ok'}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text submitted via AJAX."""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text is empty'}), 400

    if len(text) > 5000:
        return jsonify({'error': 'Text too long. Maximum 5000 characters.'}), 400

    model_type = data.get('model', 'lr').lower()
    if 'logistic' in model_type or model_type == 'lr':
        model_type = 'lr'
    elif 'naive' in model_type or model_type == 'nb':
        model_type = 'nb'
    else:
        model_type = 'lr'

    try:
        result = analyze_full(text, model_type)

        # Save to history
        save_analysis(
            text=text,
            sentiment=result['sentiment']['label'],
            confidence=result['sentiment']['confidence'],
            model_used=result['sentiment']['model_used'],
            vader_compound=result['vader']['compound'],
            intensity_pct=result['vader']['intensity_pct'],
            top_emotion=result['emotions']['top_emotion'],
            emotions=result['emotions']['emotions'],
            probabilities=result['sentiment']['probabilities'],
        )

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({
            'error': str(e),
            'hint': "Train models first: python -m ml.train_model",
        }), 503
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload (.txt or .csv) for batch analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .txt and .csv files are allowed'}), 400

    model_type = request.form.get('model', 'lr').lower()
    if 'logistic' in model_type or model_type == 'lr':
        model_type = 'lr'
    elif 'naive' in model_type or model_type == 'nb':
        model_type = 'nb'
    else:
        model_type = 'lr'

    try:
        content = file.read().decode('utf-8', errors='ignore')
        texts = []

        if file.filename.lower().endswith('.csv'):
            reader = csv.reader(io.StringIO(content))
            header = next(reader, None)

            # Find text column
            text_col = 0
            if header:
                for i, col in enumerate(header):
                    if col.strip().lower() in ('text', 'review', 'comment', 'message', 'content'):
                        text_col = i
                        break

            for row in reader:
                if row and len(row) > text_col:
                    t = row[text_col].strip()
                    if t:
                        texts.append(t)
        else:
            # .txt — one text per line
            for line in content.split('\n'):
                t = line.strip()
                if t:
                    texts.append(t)

        if not texts:
            return jsonify({'error': 'No text content found in file'}), 400

        # Cap at 500 texts
        if len(texts) > 500:
            texts = texts[:500]

        # Batch predict with full analysis (VADER + Emotions)
        results = []
        for text in texts:
            r = analyze_full(text, model_type)
            results.append({
                'text': r['text'],
                'label': r['sentiment']['label'],
                'confidence': r['sentiment']['confidence'],
                'model_used': r['sentiment']['model_used'],
                'vader_compound': r['vader']['compound'],
                'intensity_pct': r['vader']['intensity_pct'],
                'top_emotion': r['emotions']['top_emotion'],
                'emotions': r['emotions']['emotions'],
                'probabilities': r['sentiment']['probabilities'],
            })

        # Save each to history
        for r in results:
            save_analysis(
                text=r['text'],
                sentiment=r['label'],
                confidence=r['confidence'],
                model_used=r['model_used'],
                vader_compound=r['vader_compound'],
                intensity_pct=r['intensity_pct'],
                top_emotion=r['top_emotion'],
                emotions=r['emotions'],
                probabilities=r['probabilities'],
            )

        # Compute summary
        counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for r in results:
            counts[r['label']] = counts.get(r['label'], 0) + 1

        return jsonify({
            'total': len(results),
            'results': results[:100],  # Send first 100 for display
            'summary': counts,
        })

    except FileNotFoundError as e:
        return jsonify({
            'error': str(e),
            'hint': "Train models first: python -m ml.train_model",
        }), 503
    except Exception as e:
        return jsonify({'error': f'File processing failed: {str(e)}'}), 500


@app.route('/history')
def history():
    """History page — view past analyses."""
    return render_template('history.html')


@app.route('/api/history')
def api_history():
    """JSON API for history data."""
    limit = request.args.get('limit', 50, type=int)
    limit = min(limit, 200)

    history_data = get_history(limit)
    distribution = get_sentiment_distribution()
    total = get_history_count()

    return jsonify({
        'history': history_data,
        'distribution': distribution,
        'total': total,
    })


@app.route('/clear-history', methods=['POST'])
def clear_history_route():
    """Clear all history."""
    clear_history()
    return jsonify({'message': 'History cleared successfully'})


# ─── Error Handlers ─────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16 MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error_code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500, message="Internal server error"), 500


# ─── Run ────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  [ML] Sentiment Analysis & Opinion Mining")
    print("  [WEB] Starting server at http://localhost:5000")
    print("=" * 60 + "\n")
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=debug, host='0.0.0.0', port=port)

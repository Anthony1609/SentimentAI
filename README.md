# Sentiment Analysis & Opinion Mining (Flask)

Full-stack sentiment analysis web app (Flask) with:
- ML classifier (Logistic Regression / Naive Bayes, TF‑IDF)
- VADER intensity scoring
- NRCLex emotion detection
- SQLite history logging + dashboard UI

## Quickstart (Windows / PowerShell)

Create and activate a virtualenv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download required NLTK data (run once):

```powershell
python setup_nltk.py
```

Train models (creates `ml/models/sentiment_lr.joblib` and `ml/models/sentiment_nb.joblib`):

```powershell
python -m ml.train_model
```

Run the web app:

```powershell
python app.py
```

Open the app at `http://localhost:5000`.

## Configuration

Environment variables (optional):
- `FLASK_SECRET_KEY`: set a stable secret key (recommended).
- `FLASK_DEBUG`: set to `1` to enable debug mode locally.
- `PORT`: override the default port (5000).

Example (PowerShell):

```powershell
$env:FLASK_SECRET_KEY="change-me"
$env:FLASK_DEBUG="1"
python app.py
```

## Notes

- If you see an error about missing models, run `python -m ml.train_model`.
- History is stored in `data/history.db`.


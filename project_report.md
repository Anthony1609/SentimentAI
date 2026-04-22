# Project Report: Advanced Sentiment Analysis & Opinion Mining Platform

## 1. Executive Summary
This project is a full-stack, production-ready web application designed for high-performance sentiment analysis and opinion mining. Built with a Python/Flask backend and a modern "Data Intelligence" frontend, the system processes raw text or batch datasets to extract underlying sentiment (Positive, Negative, Neutral), emotional affect, and semantic intensity.

The application successfully bridges traditional machine learning classifiers (Logistic Regression and Naive Bayes) with lexicon-based NLP tools (VADER, NRCLex) to provide a multi-dimensional understanding of text. The entire system is wrapped in a highly polished, enterprise-grade UI designed for senior-level data analytics.

---

## 2. System Architecture

The project follows a decoupled, modular architecture:

### Backend (Python & Flask)
- **`app.py`**: The core WSGI application routing API requests, serving HTML templates, and managing file uploads.
- **`database.py`**: A lightweight SQLite wrapper (`history.db`) that persistently logs all inferences, tracking timestamps, models used, and confidence scores for historical aggregation.
- **RESTful API**: Exposes `/analyze` for single text inputs and `/upload` for batch processing via `.csv` or `.txt` files.

### Frontend (Vanilla JS, CSS, Chart.js)
- **Bento-Box Layout**: A CSS Grid-based layout that presents complex analytics in a dense, scannable "dashboard" format.
- **Asynchronous AJAX (`app.js`)**: Handles non-blocking requests to the Flask backend, rendering results seamlessly without page reloads.

---

## 3. Machine Learning & NLP Pipeline

The system's intelligence relies on a hybrid approach, combining statistical machine learning with rule-based lexical analysis.

### 3.1 Text Preprocessing (`ml/preprocess.py`)
Before inference, raw text undergoes rigorous cleaning using `nltk` and `re`:
1. Lowercasing and noise removal (URLs, HTML tags, punctuation, numbers).
2. Light normalization intended for TF‑IDF models (`clean_text`).
3. **Negation preservation**: the stopword set explicitly keeps negation terms (e.g., "not", "won't") available for downstream processing.

> Note: `ml/preprocess.py` also includes optional tokenization + lemmatization helpers, but the default sklearn pipeline uses TF‑IDF’s own tokenization and primarily relies on the cleaning step to keep features consistent between training and inference.

### 3.2 Primary Classifiers (Scikit-Learn)
The system was trained on an external dataset and exposes two distinct models:
- **Logistic Regression (LR)**: Excellent at establishing linear boundaries between sentiment classes based on word frequencies. (Achieved ~88.3% accuracy during training).
- **Naive Bayes (NB)**: A probabilistic model highly effective for text classification and spam detection.
- **Vectorization**: Both models utilize `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to evaluate word importance rather than simple counts.

### 3.3 Lexical & Emotional Analysis (`ml/predict.py`)
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Used to calculate the *intensity* of the sentiment. VADER provides a precise `Compound Score` ranging from -1 (Extremely Negative) to +1 (Extremely Positive).
- **NRCLex**: Extends basic sentiment by mapping text to 8 core human emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation). 

---

## 4. Key Features

1. **Real-Time Inference Engine**: Instantly analyzes text, providing a primary classification, a confidence percentage, and a breakdown of class probabilities.
2. **Batch Processing**: Users can drag and drop `.csv` or `.txt` files to analyze hundreds of records simultaneously. The system returns a comprehensive summary and a downloadable breakdown.
3. **Persistent Analytics History**: All queries are saved to an SQLite database. The "History" tab provides global distribution stats and an audit log of past analyses.
4. **One-Click Quick Samples**: Pre-loaded complex text samples (testing sarcasm, mixed emotion, and neutrality) to instantly demonstrate the pipeline's capabilities.

---

## 5. UI / UX Design

The user interface was rigorously engineered to meet "Senior Software Engineer" standards, utilizing an **Enterprise Data Intelligence** aesthetic.

- **Typography**: Employs `Plus Jakarta Sans` for razor-sharp headings, `Inter` for highly legible body copy, and `JetBrains Mono` (with tabular numbers) to ensure numerical data aligns perfectly.
- **Aesthetics**: An "Obsidian" dark mode utilizing deep blacks (`#030712`) layered with slightly elevated cards (`#0b0f19`).
- **Matte Texture**: A subtle CSS SVG noise overlay (`opacity: 0.025`) provides a physical, premium feel.
- **Cinematic Visualizations**: Custom Chart.js configurations strip away aggressive grid lines, favoring glowing metrics, responsive SVG gauges, and precise interactive tooltips. Emojis were replaced with crisp, uniform `Lucide` SVG icons.

---

## 6. Future Enhancements

While the system is currently robust, future iterations could include:
1. **Transformer Models**: Upgrading the core classifier from `scikit-learn` to a HuggingFace Transformer model (like `RoBERTa` or `DistilBERT`) to better understand deep context and sarcasm.
2. **Production Deployment**: Containerizing the application via Docker and replacing the built-in Flask development server with `Gunicorn` and `Nginx` for high-concurrency production environments.
3. **Data Export**: Adding the ability to export the batch processing results and historical analytics directly to PDF or Excel formats.

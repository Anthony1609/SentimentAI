"""
One-time setup script to download required NLTK data packages.
Run this once before starting the application:
    python setup_nltk.py
"""

import nltk
import ssl
import sys

def download_nltk_data():
    """Download all required NLTK datasets."""
    # Handle SSL certificate issues (common on some systems)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
    ]

    print("=" * 50)
    print("  Downloading NLTK Data Packages")
    print("=" * 50)

    success = True
    for pkg in packages:
        print(f"\n-> Downloading '{pkg}'...")
        try:
            nltk.download(pkg, quiet=False)
            print(f"  [OK] '{pkg}' downloaded successfully.")
        except Exception as e:
            print(f"  [FAIL] Failed to download '{pkg}': {e}")
            success = False

    print("\n" + "=" * 50)
    if success:
        print("  All NLTK packages downloaded successfully!")
    else:
        print("  Some packages failed. Check errors above.")
    print("=" * 50)

    return success


if __name__ == "__main__":
    ok = download_nltk_data()
    sys.exit(0 if ok else 1)

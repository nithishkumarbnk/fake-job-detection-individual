# modules/preprocessing.py

from typing import List
import re
import string
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None

def ensure_nltk_data():
    """Download required NLTK data safely (punkt, punkt_tab, stopwords, wordnet)."""
    global nltk
    if nltk is None:
        return

    # punkt
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # punkt_tab (newer NLTK)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # ignore if not available

    # corpora
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

def safe_word_tokenize(text: str) -> List[str]:
    """Try word_tokenize; gracefully fallback to simple split if NLTK resources fail."""
    if nltk is None:
        return text.split()
    try:
        return word_tokenize(text)
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            return word_tokenize(text)
        except Exception:
            return text.split()

def clean_text(text: str) -> str:
    """Basic lowercase, digit removal, punctuation removal, and whitespace normalization."""
    if text is None:
        return ""
    t = str(text).lower()
    t = re.sub(r"\d+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply cleaning, tokenization, stopword removal, and (optional) lemmatization."""
    ensure_nltk_data()

    if nltk:
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()
        try:
            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None
    else:
        sw = set()
        lemmatizer = None

    processed = []
    for text in series.astype(str).tolist():
        t = clean_text(text)
        if nltk:
            tokens = safe_word_tokenize(t)
            tokens = [w for w in tokens if w not in sw]
            if lemmatizer:
                try:
                    tokens = [lemmatizer.lemmatize(w) for w in tokens]
                except Exception:
                    pass
            processed.append(" ".join(tokens))
        else:
            processed.append(t)
    return pd.Series(processed)

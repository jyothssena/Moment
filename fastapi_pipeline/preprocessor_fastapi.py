"""
preprocessor_fastapi.py — MOMENT Preprocessing (FastAPI version)
=================================================================
Same logic as preprocessor.py but:
  - Config hardcoded (no config.yaml needed)
  - Input: DataFrames from CloudSQLLoader
  - Output: (moments, books, users) as lists of dicts
  - No file I/O
"""

import re
import hashlib
import logging
import unicodedata
from datetime import datetime
from typing import Tuple, List

import pandas as pd
import textstat
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# ── Hardcoded config ──────────────────────────────────────────────────────────

CFG = {
    "validation": {
        "min_words":         10,
        "max_words":         600,
        "min_chars":         50,
        "quality_threshold": 0.5,
    },
    "issues": {
        "profanity_ratio_threshold": 0.90,
        "caps_threshold":            0.90,
        "punct_threshold":           0.90,
        "repetitive_chars":          25,
        "repetitive_words_threshold":0.90,
    },
    "anomalies": {
        "iqr_multiplier":       1.5,
        "zscore_threshold":     2.5,
        "similarity_threshold": 0.85,
        "new_reader_ceiling":   70,
        "well_read_floor":      30,
    },
    "timestamp_format": "%Y-%m-%dT%H:%M:%S",
}

PROFANITY = {
    "damn", "hell", "crap", "ass", "bastard", "bitch",
    "shit", "fuck", "piss", "dick", "cock", "cunt",
    "whore", "slut", "fag", "retard"
}


# ── ID generation ─────────────────────────────────────────────────────────────

def _hash(text, length=8):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

def _sanitize(name):
    if not name:
        return "unknown"
    clean = name.lower()
    clean = re.sub(r"['\-]", "", clean)
    clean = re.sub(r"[^a-z0-9]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean

def make_book_id(gutenberg_id):
    return f"gutenberg_{gutenberg_id}"

def make_passage_id(book_id, passage_key):
    """Use passage_key from Cloud SQL directly."""
    safe_key = re.sub(r"[^a-z0-9_]", "_", str(passage_key).lower())
    return f"{book_id}_{safe_key}"

def make_interpretation_id(user_id, passage_id, text):
    return f"moment_{_hash(str(user_id) + passage_id + text[:100])}"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = str(text)
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2026", "...").replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                  "[EMAIL REMOVED]", text)
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_text(text):
    v = CFG["validation"]
    if not text or not text.strip():
        return {"is_valid": False, "quality_score": 0.0,
                "quality_issues": ["empty_text"], "word_count": 0, "language": "unknown"}

    issues = []
    words = text.split()
    word_count = len(words)

    if word_count < v["min_words"]:
        issues.append(f"too_short: {word_count} words")
    elif word_count > v["max_words"]:
        issues.append(f"too_long: {word_count} words")

    if len(text.replace(" ", "")) < v["min_chars"]:
        issues.append("too_few_chars")

    language = "en"
    if len(text) >= 20:
        try:
            language = detect(text)
            if language != "en":
                issues.append(f"wrong_language: {language}")
        except LangDetectException:
            language = "unknown"

    letters = [c.lower() for c in text if c.isalpha()]
    if len(letters) >= 10:
        vowel_ratio = sum(1 for c in letters if c in "aeiou") / len(letters)
        if vowel_ratio < 0.15 or vowel_ratio > 0.60:
            issues.append("gibberish_detected")

    score = 1.0
    for issue in issues:
        if "too_short" in issue or "too_long" in issue:
            score -= 0.3
        elif "wrong_language" in issue:
            score -= 0.4
        elif "gibberish" in issue:
            score -= 0.5
        elif "too_few_chars" in issue:
            score -= 0.1
    score = max(0.0, min(1.0, score))

    return {
        "is_valid": len(issues) == 0 and score >= v["quality_threshold"],
        "quality_score": round(score, 4),
        "quality_issues": issues,
        "word_count": word_count,
        "language": language,
    }


# ── Issue detection ───────────────────────────────────────────────────────────

def detect_issues(text):
    ic = CFG["issues"]
    if not text:
        return {"has_pii": False, "pii_types": [], "has_profanity": False,
                "profanity_ratio": 0.0, "is_spam": False, "spam_reasons": []}

    pii_types, spam_reasons = [], []

    if re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
        pii_types.append("email")
    if re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", text):
        pii_types.append("ssn")

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    profane = [w for w in words if w in PROFANITY]
    profanity_ratio = round(len(profane) / max(len(words), 1), 4)
    has_profanity = len(profane) > 0 and profanity_ratio >= ic["profanity_ratio_threshold"]

    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > ic["caps_threshold"]:
            spam_reasons.append(f"excessive_caps: {caps_ratio:.1%}")

    return {
        "has_pii": len(pii_types) > 0,
        "pii_types": pii_types,
        "has_profanity": has_profanity,
        "profanity_ratio": profanity_ratio,
        "is_spam": len(spam_reasons) > 0,
        "spam_reasons": spam_reasons,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(text):
    if not text or not text.strip():
        return {"word_count": 0, "char_count": 0, "sentence_count": 0,
                "avg_word_length": 0.0, "avg_sentence_length": 0.0,
                "readability_score": 0.0}

    words = text.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 3]
    if not sentences:
        sentences = [text]

    word_count      = len(words)
    char_count      = len(text.replace(" ", "").replace("\n", ""))
    sentence_count  = len(sentences)
    alpha_chars     = sum(len(re.sub(r"[^a-zA-Z]", "", w)) for w in words)
    avg_word_length = round(alpha_chars / max(word_count, 1), 2)
    avg_sent_length = round(word_count / max(sentence_count, 1), 2)

    try:
        readability = max(0.0, min(100.0, textstat.flesch_reading_ease(text)))
    except Exception:
        readability = 0.0

    return {
        "word_count": word_count, "char_count": char_count,
        "sentence_count": sentence_count, "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sent_length,
        "readability_score": round(readability, 2),
    }


# ── Main preprocessing entry point ───────────────────────────────────────────

def preprocess_all(
    interpretations_df: pd.DataFrame,
    passages_df:        pd.DataFrame,
    users_df:           pd.DataFrame,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Main entry point called by main.py.
    Returns: (moments, books, users)
    """
    ts = datetime.now().strftime(CFG["timestamp_format"])

    moments = _process_moments(interpretations_df, ts)
    books   = _process_books(passages_df, ts)
    users   = _process_users(users_df, ts)

    return moments, books, users


def _process_moments(df: pd.DataFrame, ts: str) -> List[dict]:
    records = []
    for _, row in df.iterrows():
        try:
            user_id      = row.get("user_id", "")
            passage_id_raw = str(row.get("passage_id", ""))
            book_title   = str(row.get("book", ""))
            gutenberg_id = str(row.get("gutenberg_id", "")) if "gutenberg_id" in row else ""
            passage_key  = str(row.get("passage_key", ""))
            raw_text     = str(row.get("interpretation", ""))

            # build book_id from gutenberg_id if available, else hash of title
            if gutenberg_id and gutenberg_id not in ("", "None", "nan"):
                book_id = make_book_id(gutenberg_id)
            else:
                book_id = f"book_{_hash(book_title)}"

            passage_id = make_passage_id(book_id, passage_key or passage_id_raw)

            cleaned    = clean_text(raw_text)
            validation = validate_text(cleaned)
            issues     = detect_issues(cleaned)
            metrics    = calculate_metrics(cleaned)
            interp_id  = make_interpretation_id(user_id, passage_id, cleaned)

            records.append({
                "interpretation_id":    interp_id,
                "user_id":              user_id,
                "book_id":              book_id,
                "passage_id":           passage_id,
                "book_title":           book_title,
                "book_author":          str(row.get("book_author", "")),
                "passage_key":          passage_key,
                "chapter":              str(row.get("chapter", "")),
                "page_num":             row.get("page_num", None),
                "cleaned_interpretation": cleaned,
                "original_word_count":  row.get("word_count", 0),
                "is_valid":             validation["is_valid"],
                "quality_score":        validation["quality_score"],
                "quality_issues":       str(validation["quality_issues"]),
                "detected_issues":      str(issues),
                "metrics":              str(metrics),
                "created_at":           str(row.get("created_at", "")),
                "timestamp":            ts,
            })
        except Exception as e:
            logger.error(f"Error processing moment row: {e}")
            continue

    logger.info(f"Processed {len(records)} moments")
    return records


def _process_books(df: pd.DataFrame, ts: str) -> List[dict]:
    records = []
    for _, row in df.iterrows():
        try:
            gutenberg_id = str(row.get("gutenberg_id", ""))
            title        = str(row.get("book_title", ""))

            if gutenberg_id and gutenberg_id not in ("", "None", "nan"):
                book_id = make_book_id(gutenberg_id)
            else:
                book_id = f"book_{_hash(title)}"

            records.append({
                "book_id":        book_id,
                "passage_id":     str(row.get("passage_id", "")),
                "book_title":     title,
                "book_author":    str(row.get("book_author", "")),
                "gutenberg_id":   gutenberg_id,
                "year":           row.get("year", None),
                "cover_url":      str(row.get("cover_url", "")),
                "epub_url":       str(row.get("epub_url", "")),
                "text_url":       str(row.get("text_url", "")),
                "opening_passage":str(row.get("opening_passage", "")),
                "timestamp":      ts,
            })
        except Exception as e:
            logger.error(f"Error processing book row: {e}")
            continue

    logger.info(f"Processed {len(records)} books")
    return records


def _process_users(df: pd.DataFrame, ts: str) -> List[dict]:
    records = []
    for _, row in df.iterrows():
        try:
            records.append({
                "user_id":               row.get("user_id", ""),
                "moment_count":          int(row.get("moment_count", 0)),
                "books_read":            int(row.get("books_read", 0)),
                "avg_word_count":        float(row.get("avg_word_count", 0.0) or 0.0),
                "total_interpretations": int(row.get("total_interpretations", 0)),
                "first_moment_at":       str(row.get("first_moment_at", "")),
                "last_moment_at":        str(row.get("last_moment_at", "")),
                "timestamp":             ts,
            })
        except Exception as e:
            logger.error(f"Error processing user row: {e}")
            continue

    logger.info(f"Processed {len(records)} users")
    return records
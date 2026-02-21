# ============================================================
# preprocessor.py
# MOMENT Preprocessing Pipeline
# Handles: reading, cleaning, validation, issue detection,
#          metrics, ID generation, and writing output files
# ============================================================

import re
import json
import hashlib
import logging
import os
import unicodedata
from datetime import datetime

import pandas as pd
import requests
import textstat
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# ── Profanity word list ───────────────────────────────────────────────────────
PROFANITY = {
    "damn", "hell", "crap", "ass", "bastard", "bitch",
    "shit", "fuck", "piss", "dick", "cock", "cunt",
    "whore", "slut", "fag", "retard"
}


# ============================================================
# ID GENERATION
# All IDs are deterministic: same input = same ID every time
# ============================================================

def _hash(text, length=8):
    """Return a short MD5 hash of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def _sanitize(name):
    """Convert a name to a clean lowercase underscored string."""
    if not name:
        return "unknown"
    clean = name.lower()
    clean = re.sub(r"['\-]", "", clean)           # remove apostrophes/hyphens
    clean = re.sub(r"[^a-z0-9]", "_", clean)      # non-alphanumeric → underscore
    clean = re.sub(r"_+", "_", clean).strip("_")   # collapse double underscores
    return clean


def make_book_id(gutenberg_id):
    """gutenberg_84"""
    return f"gutenberg_{gutenberg_id}"


def make_passage_id(book_id, passage_number):
    """gutenberg_84_passage_1"""
    return f"{book_id}_passage_{passage_number}"


def make_user_id(character_name):
    """user_emma_chen_a1b2c3d4"""
    return f"user_{_sanitize(character_name)}_{_hash(character_name)}"


def make_interpretation_id(character_name, passage_id, text):
    """moment_a1b2c3d4"""
    return f"moment_{_hash(character_name + passage_id + text[:100])}"


# ============================================================
# TEXT CLEANING
# Fixes formatting without changing meaning
# ============================================================

def clean_text(text):
    """
    Clean raw text:
    - Fix smart quotes and dashes
    - Normalize unicode
    - Remove emails
    - Collapse extra whitespace
    """
    if not text:
        return ""
    text = str(text)

    # fix mojibake (garbled encoding like â€™ → ')
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # smart quotes → straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')

    # em/en dashes → hyphens
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = text.replace("\u2212", "-")

    # unicode normalization
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2026", "...")   # ellipsis
    text = text.replace("\u00a0", " ")    # non-breaking space
    text = text.replace("\u200b", "")     # zero-width space

    # remove emails (PII)
    text = re.sub(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REMOVED]", text
    )

    # collapse whitespace
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# TEXT VALIDATION
# Checks if text meets quality thresholds from config
# Failed records stay in output with is_valid: false
# ============================================================

def validate_text(text, cfg, text_type="interpretation"):
    """
    Validate cleaned text against quality thresholds.

    Returns dict:
        is_valid, quality_score, quality_issues, word_count, language
    """
    if not text or not text.strip():
        return {
            "is_valid": False,
            "quality_score": 0.0,
            "quality_issues": ["empty_text"],
            "word_count": 0,
            "language": "unknown"
        }

    v = cfg["validation"]
    issues = []
    words = text.split()
    word_count = len(words)

    # word count check
    if word_count < v["min_words"]:
        issues.append(f"too_short: {word_count} words (min {v['min_words']})")
    elif word_count > v["max_words"]:
        issues.append(f"too_long: {word_count} words (max {v['max_words']})")

    # character count check
    char_count = len(text.replace(" ", ""))
    if char_count < v["min_chars"]:
        issues.append(f"too_few_chars: {char_count}")

    # language detection
    language = "en"
    if len(text) >= 20:
        try:
            language = detect(text)
            if language != "en":
                issues.append(f"wrong_language: {language}")
        except LangDetectException:
            language = "unknown"

    # gibberish check (vowel ratio)
    letters = [c.lower() for c in text if c.isalpha()]
    if len(letters) >= 10:
        vowel_ratio = sum(1 for c in letters if c in "aeiou") / len(letters)
        if vowel_ratio < 0.15 or vowel_ratio > 0.60:
            issues.append("gibberish_detected")

    # quality score: start at 1.0, deduct for issues
    score = 1.0
    for issue in issues:
        if "empty_text" in issue:
            return {"is_valid": False, "quality_score": 0.0,
                    "quality_issues": issues, "word_count": 0, "language": language}
        elif "too_short" in issue or "too_long" in issue:
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
        "language": language
    }


# ============================================================
# ISSUE DETECTION
# Scans for PII, profanity, spam
# ============================================================

def detect_issues(text, cfg):
    """
    Detect PII, profanity, and spam patterns.

    Returns dict:
        has_pii, pii_types, has_profanity, profanity_ratio,
        is_spam, spam_reasons
    """
    if not text:
        return {
            "has_pii": False, "pii_types": [],
            "has_profanity": False, "profanity_ratio": 0.0,
            "is_spam": False, "spam_reasons": []
        }

    ic = cfg["issues"]
    pii_types = []
    spam_reasons = []

    # --- PII ---
    if re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
        pii_types.append("email")

    phone_matches = re.findall(
        r"(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})", text
    )
    if any(len(re.sub(r"\D", "", "".join(m))) >= 10 for m in phone_matches):
        pii_types.append("phone_number")

    if re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", text):
        pii_types.append("ssn")

    if re.search(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", text):
        pii_types.append("credit_card")

    # --- Profanity ---
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    profane = [w for w in words if w in PROFANITY]
    profanity_ratio = round(len(profane) / max(len(words), 1), 4)
    has_profanity = (
        len(profane) > 0 and
        profanity_ratio >= ic["profanity_ratio_threshold"]
    )

    # --- Spam ---
    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > ic["caps_threshold"]:
            spam_reasons.append(f"excessive_caps: {caps_ratio:.1%}")

    if len(text) > 0:
        punct_ratio = len(re.findall(r"[^\w\s]", text)) / len(text)
        if punct_ratio > ic["punct_threshold"]:
            spam_reasons.append(f"excessive_punctuation: {punct_ratio:.1%}")

    rep_pattern = re.compile(r"(.)\1{" + str(ic["repetitive_chars"] - 1) + r",}")
    if rep_pattern.search(text):
        spam_reasons.append("repetitive_chars")

    if words:
        wc = {}
        for w in words:
            wc[w] = wc.get(w, 0) + 1
        top_ratio = max(wc.values()) / len(words)
        if top_ratio > ic["repetitive_words_threshold"]:
            spam_reasons.append(f"repetitive_words: {top_ratio:.1%}")

    return {
        "has_pii": len(pii_types) > 0,
        "pii_types": pii_types,
        "has_profanity": has_profanity,
        "profanity_ratio": profanity_ratio,
        "is_spam": len(spam_reasons) > 0,
        "spam_reasons": spam_reasons
    }


# ============================================================
# METRICS
# Calculates readability and text statistics
# ============================================================

def calculate_metrics(text):
    """
    Calculate quantitative text metrics.

    Returns dict:
        word_count, char_count, sentence_count,
        avg_word_length, avg_sentence_length, readability_score
    """
    if not text or not text.strip():
        return {
            "word_count": 0, "char_count": 0, "sentence_count": 0,
            "avg_word_length": 0.0, "avg_sentence_length": 0.0,
            "readability_score": 0.0
        }

    words = text.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 3]
    if not sentences:
        sentences = [text]

    word_count = len(words)
    char_count = len(text.replace(" ", "").replace("\n", ""))
    sentence_count = len(sentences)

    alpha_chars = sum(len(re.sub(r"[^a-zA-Z]", "", w)) for w in words)
    avg_word_length = round(alpha_chars / max(word_count, 1), 2)
    avg_sentence_length = round(word_count / max(sentence_count, 1), 2)

    try:
        readability = max(0.0, min(100.0, textstat.flesch_reading_ease(text)))
    except Exception:
        readability = 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "readability_score": round(readability, 2)
    }


# ============================================================
# GUTENBERG API LOOKUP
# Gets book metadata with caching and config fallback
# ============================================================

def lookup_books(cfg):
    """
    Look up metadata for all 3 books from Gutenberg API.
    Falls back to config if API is unavailable.

    Returns dict: { book_title: { book_id, gutenberg_id, author } }
    """
    cache = {}

    for book in cfg["books"]:
        title = book["title"]
        logger.info(f"Looking up: {title}")
        try:
            resp = requests.get(
                cfg["gutenberg_api"],
                params={"search": title},
                timeout=cfg["gutenberg_timeout"]
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            # find best match
            match = None
            for r in results:
                if title.lower() in r.get("title", "").lower():
                    match = r
                    break
            if not match and results:
                match = results[0]

            if match:
                gid = match["id"]
                authors = match.get("authors", [])
                raw_author = authors[0].get("name", "Unknown") if authors else "Unknown"
                # normalize "Last, First" → "First Last"
                if "," in raw_author:
                    parts = raw_author.split(",", 1)
                    author = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    author = raw_author

                cache[title] = {
                    "book_id": make_book_id(gid),
                    "gutenberg_id": gid,
                    "author": author
                }
                logger.info(f"  Found: {title} → {cache[title]['book_id']}, {author}")
                continue

        except Exception as e:
            logger.warning(f"  API failed for {title}: {e}. Using config fallback.")

        # fallback to config
        cache[title] = {
            "book_id": make_book_id(book["gutenberg_id"]),
            "gutenberg_id": book["gutenberg_id"],
            "author": book["author"]
        }

    return cache


# ============================================================
# READ RAW DATA
# ============================================================

def read_raw_data(cfg):
    """
    Read all 3 raw input files.

    Returns: interpretations (list), passages (list), characters (list)
    """
    paths = cfg["paths"]["raw"]

    with open(paths["interpretations"], "r", encoding="utf-8") as f:
        interpretations = json.load(f)
    logger.info(f"Loaded {len(interpretations)} interpretations.")

    passages_df = pd.read_csv(paths["passages"], keep_default_na=False)
    passages = passages_df.to_dict("records")
    # normalize passage book titles
    title_map = cfg.get("title_map", {})
    for p in passages:
        if p.get("book_title") in title_map:
            p["book_title"] = title_map[p["book_title"]]
    logger.info(f"Loaded {len(passages)} passages.")

    characters_df = pd.read_csv(paths["characters"], keep_default_na=False)
    characters = characters_df.to_dict("records")
    logger.info(f"Loaded {len(characters)} characters.")

    return interpretations, passages, characters


# ============================================================
# PROCESS BOOKS
# ============================================================

def process_books(passages, book_meta, cfg):
    """
    Clean and enrich passage records.

    Returns: list of processed book dicts
    """
    ts = datetime.now().strftime(cfg["timestamp_format"])
    processed = []

    for p in passages:
        title = p.get("book_title", "")
        meta = book_meta.get(title, {})
        if not meta:
            logger.warning(f"No metadata for book: {title}. Skipping.")
            continue

        try:
            passage_num = int(p.get("passage_id", 0))
        except (ValueError, TypeError):
            passage_num = 0

        book_id = meta["book_id"]
        passage_id = make_passage_id(book_id, passage_num)
        raw_text = str(p.get("passage_text", ""))
        cleaned = clean_text(raw_text)
        validation = validate_text(cleaned, cfg, text_type="passage")
        metrics = calculate_metrics(cleaned)

        processed.append({
            "book_id": book_id,
            "passage_id": passage_id,
            "book_title": title,
            "book_author": meta.get("author", "Unknown"),
            "chapter_number": str(p.get("chapter_number", "Unknown")),
            "passage_title": str(p.get("passage_title", "")),
            "passage_number": passage_num,
            "cleaned_passage_text": cleaned,
            "is_valid": validation["is_valid"],
            "quality_score": validation["quality_score"],
            "quality_issues": validation["quality_issues"],
            "metrics": metrics,
            "timestamp": ts
        })

    logger.info(f"Processed {len(processed)} passages.")
    return processed


# ============================================================
# PROCESS USERS
# ============================================================

def process_users(characters, interpretations, cfg):
    """
    Build enriched user profile records.

    Returns: list of processed user dicts
    """
    ts = datetime.now().strftime(cfg["timestamp_format"])

    # group interpretations by character name
    interps_by_name = {}
    for i in interpretations:
        name = i.get("character_name", "")
        interps_by_name.setdefault(name, []).append(i)

    processed = []
    for char in characters:
        name = char.get("Name", "")
        user_id = make_user_id(name)

        # reading styles from Style_1 to Style_4
        styles = [
            str(char.get(f"Style_{i}", "")).strip()
            for i in range(1, 5)
            if str(char.get(f"Style_{i}", "")).strip()
        ]

        char_interps = interps_by_name.get(name, [])
        books_interpreted = sorted(set(
            i.get("book", "") for i in char_interps if i.get("book")
        ))

        processed.append({
            "user_id": user_id,
            "character_name": name,
            "gender": char.get("Gender", ""),
            "age": int(char.get("Age", 0) or 0),
            "profession": char.get("Profession", ""),
            "distribution_category": char.get("Distribution_Category", ""),
            "personality": char.get("Personality", ""),
            "interest": char.get("Interest", ""),
            "reading_intensity": char.get("Reading_Intensity", ""),
            "reading_count": int(char.get("Reading_Count", 0) or 0),
            "experience_level": char.get("Experience_Level", ""),
            "experience_count": int(char.get("Experience_Count", 0) or 0),
            "journey": char.get("Journey", ""),
            "reading_styles": styles,
            "total_interpretations": len(char_interps),
            "books_interpreted": books_interpreted,
            "timestamp": ts
        })

    logger.info(f"Processed {len(processed)} users.")
    return processed


# ============================================================
# PROCESS MOMENTS (PASS 1)
# Individual record processing - cleaning, validation,
# issue detection, metrics, ID generation
# ============================================================

def process_moments_pass1(interpretations, book_meta, cfg):
    """
    First pass: process each interpretation individually.
    Returns partial records (anomalies added in pass 2).
    """
    ts = datetime.now().strftime(cfg["timestamp_format"])
    records = []

    for interp in interpretations:
        try:
            book_title = interp.get("book", "")
            raw_passage_id = interp.get("passage_id", "")
            character_name = interp.get("character_name", "")
            character_id = interp.get("character_id", 0)
            raw_text = interp.get("interpretation", "")
            raw_word_count = interp.get("word_count", 0)

            # parse passage number from "passage_1" → 1
            try:
                passage_num = int(str(raw_passage_id).split("_")[-1])
            except (ValueError, IndexError):
                passage_num = 0

            meta = book_meta.get(book_title, {})
            if not meta:
                logger.warning(f"No metadata for {book_title}. Skipping.")
                continue

            book_id = meta["book_id"]
            passage_id = make_passage_id(book_id, passage_num)
            user_id = make_user_id(character_name)

            cleaned = clean_text(raw_text)
            validation = validate_text(cleaned, cfg)
            issues = detect_issues(cleaned, cfg)
            metrics = calculate_metrics(cleaned)
            interpretation_id = make_interpretation_id(
                character_name, passage_id, cleaned
            )

            records.append({
                "interpretation_id": interpretation_id,
                "user_id": user_id,
                "book_id": book_id,
                "passage_id": passage_id,
                "book_title": book_title,
                "passage_number": passage_num,
                "character_id": character_id,
                "character_name": character_name,
                "cleaned_interpretation": cleaned,
                "original_word_count": raw_word_count,
                "is_valid": validation["is_valid"],
                "quality_score": validation["quality_score"],
                "quality_issues": validation["quality_issues"],
                "detected_issues": issues,
                "anomalies": {},        # filled in pass 2
                "metrics": metrics,
                "timestamp": ts
            })

        except Exception as e:
            logger.error(f"Error processing {interp.get('character_name', '?')}: {e}")
            continue

    logger.info(f"Pass 1 complete: {len(records)} interpretations processed.")
    return records


# ============================================================
# WRITE OUTPUT FILES
# ============================================================

def write_outputs(moments, books, users, cfg):
    """Write all 3 processed JSON files to data/processed/."""
    os.makedirs("data/processed", exist_ok=True)
    paths = cfg["paths"]["processed"]
    indent = cfg.get("indent", 2)

    for data, path, label in [
        (moments, paths["moments"], "moments"),
        (books,   paths["books"],   "books"),
        (users,   paths["users"],   "users"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        logger.info(f"Wrote {len(data)} {label} records → {path}")
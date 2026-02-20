# ============================================================
# anomalies.py
# MOMENT Preprocessing Pipeline - Anomaly Detection
#
# Detects statistically unusual records AFTER all 450
# interpretations have been processed individually.
# Needs all records at once to establish what is "normal".
# ============================================================

import logging
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

logger = logging.getLogger(__name__)


def detect_anomalies(records, characters, cfg):
    """
    Run all anomaly detection on the full batch of records.

    Called after process_moments_pass1() so we have all
    450 records and their metrics to compare against.

    Args:
        records:    list of moment records from pass 1
        characters: list of character dicts from characters.csv
        cfg:        full config dict

    Returns:
        records with 'anomalies' field populated for each
    """
    if not records:
        return records

    ac = cfg["anomalies"]

    # build character lookup: name → character dict
    char_lookup = {c["Name"]: c for c in characters}

    # extract metrics for baseline calculation
    word_counts = [r["metrics"].get("word_count", 0) for r in records]
    readability_scores = [r["metrics"].get("readability_score", 0) for r in records]
    texts = [r["cleaned_interpretation"] for r in records]
    ids = [r["interpretation_id"] for r in records]

    # calculate baselines
    wc_bounds = _iqr_bounds(word_counts, ac["iqr_multiplier"])
    read_stats = _mean_std(readability_scores)

    logger.info(
        f"Anomaly baselines: "
        f"word count bounds [{wc_bounds['lower']:.1f}, {wc_bounds['upper']:.1f}], "
        f"readability mean={read_stats['mean']:.2f} std={read_stats['std']:.2f}"
    )

    # build TF-IDF matrix for duplicate detection
    tfidf_matrix, vectorizer = _build_tfidf(texts)

    # detect anomalies for each record
    for i, record in enumerate(records):
        details = []
        metrics = record["metrics"]

        # 1. word count outlier (IQR)
        wc = metrics.get("word_count", 0)
        wc_outlier = _check_wc_outlier(wc, wc_bounds, details)

        # 2. readability outlier (z-score)
        rs = metrics.get("readability_score", 0)
        read_outlier = _check_read_outlier(rs, read_stats, ac["zscore_threshold"], details)

        # 3. near-duplicate detection (TF-IDF cosine similarity)
        is_dup, dup_of = _check_duplicate(
            i, texts[i], tfidf_matrix, vectorizer, ids,
            ac["similarity_threshold"], details
        )

        # 4. style vs experience mismatch
        char = char_lookup.get(record.get("character_name", ""))
        style_mismatch = _check_style_mismatch(
            rs, char, ac["new_reader_ceiling"], ac["well_read_floor"], details
        )

        record["anomalies"] = {
            "word_count_outlier": wc_outlier,
            "readability_outlier": read_outlier,
            "duplicate_risk": is_dup,
            "duplicate_of": dup_of,
            "style_mismatch": style_mismatch,
            "anomaly_details": details
        }

    # log summary
    wc_count   = sum(1 for r in records if r["anomalies"]["word_count_outlier"])
    read_count = sum(1 for r in records if r["anomalies"]["readability_outlier"])
    dup_count  = sum(1 for r in records if r["anomalies"]["duplicate_risk"])
    mis_count  = sum(1 for r in records if r["anomalies"]["style_mismatch"])

    logger.info(
        f"Anomaly detection complete: "
        f"word_count_outliers={wc_count}, "
        f"readability_outliers={read_count}, "
        f"duplicates={dup_count}, "
        f"style_mismatches={mis_count}"
    )

    return records


# ── Private helpers ───────────────────────────────────────────────────────────

def _iqr_bounds(values, multiplier):
    """
    Calculate IQR outlier bounds.
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    """
    if not values:
        return {"lower": 0, "upper": float("inf"), "q1": 0, "q3": 0, "iqr": 0}
    sv = sorted(v for v in values if v > 0)
    n = len(sv)
    q1 = sv[n // 4]
    q3 = sv[(3 * n) // 4]
    iqr = q3 - q1
    return {
        "lower": q1 - multiplier * iqr,
        "upper": q3 + multiplier * iqr,
        "q1": q1, "q3": q3, "iqr": iqr
    }


def _mean_std(values):
    """Calculate mean and standard deviation."""
    vals = [v for v in values if v > 0]
    if not vals:
        return {"mean": 0, "std": 1}
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = variance ** 0.5
    return {"mean": mean, "std": std if std > 0 else 1}


def _build_tfidf(texts):
    """
    Build TF-IDF matrix for all texts.
    Used to detect near-duplicate interpretations.
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix: {matrix.shape[0]} docs x {matrix.shape[1]} features")
        return matrix, vectorizer
    except Exception as e:
        logger.warning(f"TF-IDF build failed: {e}. Duplicate detection skipped.")
        return None, None


def _check_wc_outlier(word_count, bounds, details):
    """Flag if word count is outside IQR bounds."""
    if word_count < bounds["lower"]:
        details.append(
            f"word_count_low: {word_count} words "
            f"(below lower bound {bounds['lower']:.1f})"
        )
        return True
    if word_count > bounds["upper"]:
        details.append(
            f"word_count_high: {word_count} words "
            f"(above upper bound {bounds['upper']:.1f})"
        )
        return True
    return False


def _check_read_outlier(score, stats, threshold, details):
    """Flag if readability z-score exceeds threshold."""
    z = abs((score - stats["mean"]) / stats["std"])
    if z > threshold:
        direction = "high" if score > stats["mean"] else "low"
        details.append(
            f"readability_{direction}: score={score:.2f}, z={z:.2f}"
        )
        return True
    return False


def _check_duplicate(idx, text, matrix, vectorizer, ids, threshold, details):
    """
    Check if this text is near-duplicate of another record
    using cosine similarity on TF-IDF vectors.
    """
    if matrix is None or vectorizer is None:
        return False, None
    try:
        vec = vectorizer.transform([text])
        sims = cosine_similarity(vec, matrix).flatten()
        for i, sim in enumerate(sims):
            if i == idx:
                continue
            if sim >= threshold:
                details.append(
                    f"duplicate_risk: {sim:.2f} similarity with {ids[i]}"
                )
                return True, ids[i]
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")
    return False, None


def _check_style_mismatch(readability, character, ceiling, floor, details):
    """
    Flag if writing style does not match reader experience level.
    - NEW READER writing very complex text (low readability) = mismatch
    - Well-read reader writing very simple text (high readability) = mismatch
    """
    if not character:
        return False

    exp = character.get("Experience_Level", "").strip()
    dist = character.get("Distribution_Category", "").strip()

    if (exp == "New" or dist == "NEW READER") and readability < floor:
        details.append(
            f"style_mismatch: NEW READER with complex writing "
            f"(readability={readability:.1f}, floor={floor})"
        )
        return True

    if exp == "Well-read" and readability > ceiling:
        details.append(
            f"style_mismatch: Well-read with simple writing "
            f"(readability={readability:.1f}, ceiling={ceiling})"
        )
        return True

    return False
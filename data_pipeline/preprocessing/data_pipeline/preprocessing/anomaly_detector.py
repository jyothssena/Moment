# ============================================================
# anomaly_detector.py
# MOMENT Preprocessing Pipeline - Anomaly Detection Module
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Detects statistically unusual records in the dataset.
# This is Phase 2.5 - runs AFTER metrics_calculator.py and
# BEFORE the output is written.
#
# DIFFERENCE FROM VALIDATION:
#   - text_validator.py:  rule-based  ("word count < 10 = invalid")
#   - anomaly_detector.py: stats-based ("word count is 3 std devs
#                           below the mean = unusual")
#
# WHAT IT DETECTS:
#   1. Word count outliers    (IQR method)
#   2. Readability outliers   (Z-score method)
#   3. Near-duplicate text    (TF-IDF cosine similarity)
#   4. Style vs experience mismatch (readability vs reader profile)
#
# OUTPUT:
#   anomalies dict per record:
#   {
#       "word_count_outlier": bool,
#       "readability_outlier": bool,
#       "duplicate_risk": bool,
#       "duplicate_of": str or None,   # interpretation_id of similar record
#       "style_mismatch": bool,
#       "anomaly_details": list        # human-readable explanation
#   }
#
# NOTE: Anomalies do NOT invalidate records. They are flags
# for human review or downstream filtering. A short but
# insightful interpretation is still valuable even if it's
# a word count outlier.
# ============================================================

import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects statistical anomalies in preprocessed records.

    Requires dataset-level context to work - it compares each
    record against the full dataset, so it needs ALL records
    before it can flag outliers.

    Two-phase usage:
        1. fit(all_metrics, all_texts, all_characters)
           - establishes baselines from full dataset
        2. detect(record_metrics, text, character)
           - flags anomalies in individual records

    Usage:
        detector = AnomalyDetector(config)
        detector.fit(all_metrics_list, all_texts, all_characters)
        anomalies = detector.detect(record_metrics, text, character)
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Args:
            config: full config dict from config/config.yaml
        """
        anomaly_config = config.get("anomaly_detection", {})

        # global enable/disable
        self.enabled = anomaly_config.get("enabled", True)

        # word count outlier settings (IQR method)
        wc_config = anomaly_config.get("word_count", {})
        self.wc_method = wc_config.get("method", "iqr")
        self.iqr_multiplier = wc_config.get("iqr_multiplier", 1.5)

        # readability outlier settings (Z-score method)
        read_config = anomaly_config.get("readability", {})
        self.read_method = read_config.get("method", "zscore")
        self.zscore_threshold = read_config.get("zscore_threshold", 2.5)

        # duplicate detection settings
        dup_config = anomaly_config.get("duplicate", {})
        self.check_duplicates = dup_config.get("enabled", True)
        self.similarity_threshold = dup_config.get(
            "similarity_threshold", 0.85
        )

        # style mismatch settings
        style_config = anomaly_config.get("style_mismatch", {})
        self.check_style_mismatch = style_config.get("enabled", True)
        self.new_reader_ceiling = style_config.get(
            "new_reader_readability_ceiling", 70
        )
        self.well_read_floor = style_config.get(
            "well_read_readability_floor", 30
        )

        # baseline stats - set by fit() method
        # None until fit() is called
        self._wc_stats = None        # word count stats (q1, q3, iqr)
        self._read_stats = None      # readability stats (mean, std)
        self._tfidf_matrix = None    # TF-IDF matrix for similarity
        self._tfidf_ids = None       # record IDs matching tfidf rows
        self._fitted = False         # whether fit() has been called

        logger.debug(
            f"AnomalyDetector initialized. "
            f"enabled={self.enabled}, "
            f"iqr_multiplier={self.iqr_multiplier}, "
            f"zscore_threshold={self.zscore_threshold}, "
            f"similarity_threshold={self.similarity_threshold}"
        )

    def fit(self, all_metrics: list,
            all_texts: list,
            all_record_ids: list) -> None:
        """
        Establish baselines from the full dataset.

        MUST be called before detect(). Uses all 450 records
        to calculate:
        - Word count distribution (Q1, Q3, IQR for outlier bounds)
        - Readability distribution (mean, std for z-scores)
        - TF-IDF matrix (for duplicate detection)

        Args:
            all_metrics: list of metrics dicts from
                         MetricsCalculator.calculate()
            all_texts: list of cleaned text strings
                       (same order as all_metrics)
            all_record_ids: list of record IDs (interpretation_ids)
                            (same order as all_metrics)
        """
        if not self.enabled:
            logger.info("Anomaly detection disabled. Skipping fit().")
            return

        logger.info(
            f"Fitting anomaly detector on {len(all_metrics)} records..."
        )

        # --- Fit word count baseline (IQR method) ---
        word_counts = [
            m.get("word_count", 0) for m in all_metrics
            if m.get("word_count", 0) > 0
        ]

        if word_counts:
            sorted_wc = sorted(word_counts)
            n = len(sorted_wc)
            q1 = sorted_wc[n // 4]
            q3 = sorted_wc[(3 * n) // 4]
            iqr = q3 - q1

            # outlier bounds using IQR method
            # anything outside these bounds is flagged
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)

            self._wc_stats = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

            logger.info(
                f"Word count baseline: "
                f"Q1={q1}, Q3={q3}, IQR={iqr}, "
                f"bounds=[{lower_bound:.1f}, {upper_bound:.1f}]"
            )

        # --- Fit readability baseline (Z-score method) ---
        readability_scores = [
            m.get("readability_score", 0) for m in all_metrics
            if m.get("readability_score", 0) > 0
        ]

        if readability_scores:
            mean = sum(readability_scores) / len(readability_scores)
            variance = sum(
                (s - mean) ** 2 for s in readability_scores
            ) / len(readability_scores)
            std = variance ** 0.5

            self._read_stats = {
                "mean": mean,
                "std": std
            }

            logger.info(
                f"Readability baseline: "
                f"mean={mean:.2f}, std={std:.2f}"
            )

        # --- Fit TF-IDF matrix for duplicate detection ---
        if self.check_duplicates and all_texts:
            self._fit_tfidf(all_texts, all_record_ids)

        self._fitted = True
        logger.info("Anomaly detector fitting complete.")

    def detect(self, record_metrics: dict,
               text: str,
               record_id: str,
               character: dict = None) -> dict:
        """
        Detect anomalies in a single record.

        fit() must be called before detect().

        Args:
            record_metrics: metrics dict from MetricsCalculator
            text: cleaned interpretation text
            record_id: the interpretation_id for this record
            character: character profile dict from characters.csv
                       (used for style mismatch detection)
                       Optional - style mismatch skipped if None

        Returns:
            dict: {
                "word_count_outlier": bool,
                "readability_outlier": bool,
                "duplicate_risk": bool,
                "duplicate_of": str or None,
                "style_mismatch": bool,
                "anomaly_details": list of explanation strings
            }
        """
        # return empty anomalies if disabled or not fitted
        if not self.enabled:
            return self._empty_anomalies()

        if not self._fitted:
            logger.warning(
                "AnomalyDetector.detect() called before fit(). "
                "Returning empty anomalies. Call fit() first."
            )
            return self._empty_anomalies()

        anomaly_details = []

        # --- Check 1: word count outlier ---
        wc_outlier = False
        if self._wc_stats:
            word_count = record_metrics.get("word_count", 0)
            wc_outlier = self._is_word_count_outlier(
                word_count, anomaly_details
            )

        # --- Check 2: readability outlier ---
        read_outlier = False
        if self._read_stats:
            readability = record_metrics.get("readability_score", 0)
            read_outlier = self._is_readability_outlier(
                readability, anomaly_details
            )

        # --- Check 3: duplicate detection ---
        is_duplicate = False
        duplicate_of = None
        if self.check_duplicates and self._tfidf_matrix is not None:
            is_duplicate, duplicate_of = self._is_near_duplicate(
                text, record_id, anomaly_details
            )

        # --- Check 4: style mismatch ---
        style_mismatch = False
        if self.check_style_mismatch and character is not None:
            readability = record_metrics.get("readability_score", 0)
            style_mismatch = self._is_style_mismatch(
                readability, character, anomaly_details
            )

        result = {
            "word_count_outlier": wc_outlier,
            "readability_outlier": read_outlier,
            "duplicate_risk": is_duplicate,
            "duplicate_of": duplicate_of,
            "style_mismatch": style_mismatch,
            "anomaly_details": anomaly_details
        }

        # log if any anomalies found
        if any([wc_outlier, read_outlier, is_duplicate, style_mismatch]):
            logger.debug(
                f"Anomalies detected for {record_id}: "
                f"{anomaly_details}"
            )

        return result

    def detect_batch(self, records_metrics: list,
                     texts: list,
                     record_ids: list,
                     characters: list = None) -> list:
        """
        Detect anomalies across all records.

        Calls fit() automatically if not already fitted.

        Args:
            records_metrics: list of metrics dicts
            texts: list of cleaned texts (same order)
            record_ids: list of record IDs (same order)
            characters: list of character dicts (same order)
                        Optional

        Returns:
            list of anomaly result dicts
        """
        if not self.enabled:
            logger.info("Anomaly detection disabled.")
            return [self._empty_anomalies() for _ in records_metrics]

        # auto-fit if not already fitted
        if not self._fitted:
            self.fit(records_metrics, texts, record_ids)

        logger.info(
            f"Running anomaly detection on {len(records_metrics)} records..."
        )

        results = []
        for i, (metrics, text, rec_id) in enumerate(
            zip(records_metrics, texts, record_ids)
        ):
            character = characters[i] if characters else None
            result = self.detect(metrics, text, rec_id, character)
            results.append(result)

        # log summary
        wc_outliers = sum(1 for r in results if r["word_count_outlier"])
        read_outliers = sum(1 for r in results if r["readability_outlier"])
        duplicates = sum(1 for r in results if r["duplicate_risk"])
        mismatches = sum(1 for r in results if r["style_mismatch"])

        logger.info(
            f"Anomaly detection complete:\n"
            f"  Word count outliers:  {wc_outliers}\n"
            f"  Readability outliers: {read_outliers}\n"
            f"  Duplicate risks:      {duplicates}\n"
            f"  Style mismatches:     {mismatches}"
        )

        return results

    # --------------------------------------------------------
    # PRIVATE DETECTION METHODS
    # --------------------------------------------------------

    def _is_word_count_outlier(self, word_count: int,
                                anomaly_details: list) -> bool:
        """
        Check if word count is an outlier using IQR method.

        IQR (Interquartile Range) method:
        - Calculate Q1 (25th percentile) and Q3 (75th percentile)
        - IQR = Q3 - Q1
        - Lower bound = Q1 - 1.5 * IQR
        - Upper bound = Q3 + 1.5 * IQR
        - Anything outside bounds is an outlier

        Args:
            word_count: word count of the record
            anomaly_details: list to append explanation to

        Returns:
            bool: True if outlier
        """
        lower = self._wc_stats["lower_bound"]
        upper = self._wc_stats["upper_bound"]

        if word_count < lower:
            anomaly_details.append(
                f"word_count_low: {word_count} words "
                f"(below lower bound of {lower:.1f})"
            )
            return True

        if word_count > upper:
            anomaly_details.append(
                f"word_count_high: {word_count} words "
                f"(above upper bound of {upper:.1f})"
            )
            return True

        return False

    def _is_readability_outlier(self, readability: float,
                                 anomaly_details: list) -> bool:
        """
        Check if readability score is an outlier using Z-score.

        Z-score method:
        - Z-score = (value - mean) / std
        - If |Z-score| > threshold, it's an outlier
        - threshold = 2.5 (from config) means 2.5 std deviations

        Args:
            readability: Flesch Reading Ease score
            anomaly_details: list to append explanation to

        Returns:
            bool: True if outlier
        """
        mean = self._read_stats["mean"]
        std = self._read_stats["std"]

        # avoid division by zero if all scores are identical
        if std == 0:
            return False

        z_score = abs((readability - mean) / std)

        if z_score > self.zscore_threshold:
            direction = "high" if readability > mean else "low"
            anomaly_details.append(
                f"readability_{direction}: score={readability:.2f}, "
                f"z-score={z_score:.2f} "
                f"(threshold={self.zscore_threshold})"
            )
            return True

        return False

    def _fit_tfidf(self, texts: list, record_ids: list) -> None:
        """
        Build TF-IDF matrix for duplicate detection.

        TF-IDF (Term Frequency-Inverse Document Frequency)
        represents each text as a vector of word importance scores.
        Cosine similarity between vectors measures text similarity.

        Args:
            texts: list of all cleaned text strings
            record_ids: list of record IDs matching texts
        """
        try:
            logger.info(
                f"Building TF-IDF matrix for {len(texts)} texts..."
            )

            # initialize TF-IDF vectorizer
            # max_features=5000 keeps memory manageable
            # min_df=2 ignores terms that appear in only 1 document
            vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,           # word must appear in ≥2 docs
                stop_words="english",  # remove common words
                ngram_range=(1, 2)  # unigrams and bigrams
            )

            # fit and transform all texts to TF-IDF vectors
            self._tfidf_matrix = vectorizer.fit_transform(texts)
            self._tfidf_ids = record_ids
            self._vectorizer = vectorizer

            logger.info(
                f"TF-IDF matrix built: "
                f"{self._tfidf_matrix.shape[0]} docs x "
                f"{self._tfidf_matrix.shape[1]} features"
            )

        except Exception as e:
            logger.error(f"TF-IDF fitting failed: {e}. "
                         f"Duplicate detection will be skipped.")
            self._tfidf_matrix = None
            self._tfidf_ids = None

    def _is_near_duplicate(self, text: str,
                            record_id: str,
                            anomaly_details: list) -> tuple:
        """
        Check if text is near-duplicate of another record.

        Transforms the text to TF-IDF vector and calculates
        cosine similarity against all other records.
        If similarity > threshold, flags as duplicate risk.

        Args:
            text: text to check
            record_id: ID of this record (to exclude self-comparison)
            anomaly_details: list to append explanation to

        Returns:
            tuple: (is_duplicate: bool, duplicate_of: str or None)
        """
        try:
            # transform this text using the fitted vectorizer
            text_vector = self._vectorizer.transform([text])

            # calculate cosine similarity against all texts
            similarities = cosine_similarity(
                text_vector, self._tfidf_matrix
            ).flatten()

            # find most similar record (excluding self)
            for i, sim_score in enumerate(similarities):
                # skip self-comparison
                if self._tfidf_ids[i] == record_id:
                    continue

                if sim_score >= self.similarity_threshold:
                    anomaly_details.append(
                        f"duplicate_risk: {sim_score:.2f} similarity "
                        f"with {self._tfidf_ids[i]}"
                    )
                    return True, self._tfidf_ids[i]

            return False, None

        except Exception as e:
            logger.warning(f"Duplicate detection failed for {record_id}: {e}")
            return False, None

    def _is_style_mismatch(self, readability: float,
                            character: dict,
                            anomaly_details: list) -> bool:
        """
        Check if writing style matches reader's experience level.

        Logic:
        - NEW READER with very complex writing (low Flesch score)
          suggests the interpretation may be mislabeled or unusual
        - Well-read reader with very simple writing
          could indicate disengagement or error

        Flesch Reading Ease:
            Higher score = easier/simpler writing
            Lower score  = harder/more complex writing

        Args:
            readability: Flesch Reading Ease score
            character: character profile dict with Experience_Level
            anomaly_details: list to append explanation to

        Returns:
            bool: True if style mismatch detected
        """
        experience_level = character.get("Experience_Level", "").strip()
        distribution_category = character.get(
            "Distribution_Category", ""
        ).strip()

        # NEW READER writing very complex text
        # (low Flesch score = complex)
        if (experience_level == "New" or
                distribution_category == "NEW READER"):
            if readability < self.well_read_floor:
                anomaly_details.append(
                    f"style_mismatch: NEW READER with complex writing "
                    f"(readability={readability:.1f}, "
                    f"threshold={self.well_read_floor})"
                )
                return True

        # Well-read reader writing very simple text
        # (high Flesch score = simple)
        if experience_level == "Well-read":
            if readability > self.new_reader_ceiling:
                anomaly_details.append(
                    f"style_mismatch: Well-read reader with simple "
                    f"writing (readability={readability:.1f}, "
                    f"threshold={self.new_reader_ceiling})"
                )
                return True

        return False

    def _empty_anomalies(self) -> dict:
        """
        Return empty anomaly result when detection is disabled
        or fit() hasn't been called.

        Returns:
            dict: all anomaly flags set to False/None
        """
        return {
            "word_count_outlier": False,
            "readability_outlier": False,
            "duplicate_risk": False,
            "duplicate_of": None,
            "style_mismatch": False,
            "anomaly_details": []
        }

    def get_fit_stats(self) -> dict:
        """
        Return the baseline stats calculated during fit().

        Useful for logging and validation reports.

        Returns:
            dict: baseline stats or empty dict if not fitted
        """
        if not self._fitted:
            return {}

        return {
            "fitted": self._fitted,
            "word_count_stats": self._wc_stats,
            "readability_stats": self._read_stats,
            "tfidf_built": self._tfidf_matrix is not None,
            "tfidf_shape": (
                self._tfidf_matrix.shape
                if self._tfidf_matrix is not None else None
            )
        }


# ============================================================
# TEST BLOCK
# Run this file directly to test anomaly detection:
#   python -m data_pipeline.preprocessing.anomaly_detector
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore # for loading config in test block
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing AnomalyDetector")
    print("=" * 60)

    # load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "config", "config.yaml"
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    detector = AnomalyDetector(config)

    # mock dataset - mix of normal and anomalous records
    mock_texts = [
        # normal interpretations (varying lengths)
        "He says catastrophe before anything bad happens. "
        "The creature opened its eyes. Victor's already calling it disaster. "
        "Beautiful Great God right next to each other. His brain is breaking.",

        "Yellow eye opening by half extinguished light. Good Gothic imagery. "
        "Everything is dim and dying. He calls it catastrophe immediately. "
        "That word is interesting. Not mistake not surprise but catastrophe.",

        "The reaching destroyed me. Your whole body knows you want something "
        "before your brain admits it. The trembling is what gets me not just "
        "emotional but physical. Like you are coming apart with wanting.",

        "In vain have I struggled. Struggled implies effort to avoid. "
        "You struggled against something you did not want. "
        "So he did not want to love her. Then proposes anyway. "
        "While explaining all the reasons she is unsuitable. Obviously fails.",

        "Cold northern breeze fills me with delight. Most people fear cold. "
        "He loves it because it means he is getting closer to his goal. "
        "Perpetual splendour. Sounds beautiful. Sounds like a fairytale "
        "that will not come true. The pole is not paradise.",

        # very short (potential word count outlier)
        "Guy builds monster. Monster opens eyes. Guy runs away.",

        # near-duplicate of first record (for duplicate testing)
        "He says catastrophe before anything bad happens. "
        "The creature opened its eyes. Victor is already calling it disaster. "
        "Beautiful Great God right next to each other. His brain is breaking.",
    ]

    mock_metrics = [
        {"word_count": 45, "readability_score": 65.2},
        {"word_count": 52, "readability_score": 58.4},
        {"word_count": 61, "readability_score": 72.1},
        {"word_count": 58, "readability_score": 55.3},
        {"word_count": 55, "readability_score": 68.7},
        {"word_count": 11, "readability_score": 85.0},   # short outlier
        {"word_count": 44, "readability_score": 65.0},   # near duplicate
    ]

    mock_ids = [
        "moment_001", "moment_002", "moment_003",
        "moment_004", "moment_005", "moment_006", "moment_007"
    ]

    mock_characters = [
        {"Experience_Level": "Some classics",
         "Distribution_Category": "DELIBERATE"},
        {"Experience_Level": "Some", "Distribution_Category": "SOCIAL"},
        {"Experience_Level": "Some", "Distribution_Category": "NEW READER"},
        {"Experience_Level": "Well-read", "Distribution_Category": "HABITUAL"},
        {"Experience_Level": "New", "Distribution_Category": "NEW READER"},
        {"Experience_Level": "New", "Distribution_Category": "NEW READER"},
        {"Experience_Level": "Some classics",
         "Distribution_Category": "DELIBERATE"},
    ]

    # fit on all mock data
    print("\n--- Fitting detector ---")
    detector.fit(mock_metrics, mock_texts, mock_ids)

    # show fit stats
    stats = detector.get_fit_stats()
    print(f"  Word count bounds: "
          f"[{stats['word_count_stats']['lower_bound']:.1f}, "
          f"{stats['word_count_stats']['upper_bound']:.1f}]")
    print(f"  Readability mean: {stats['readability_stats']['mean']:.2f}")
    print(f"  TF-IDF built: {stats['tfidf_built']}")

    # detect anomalies for each record
    print("\n--- Detecting anomalies ---")
    for i, (metrics, text, rec_id, char) in enumerate(
        zip(mock_metrics, mock_texts, mock_ids, mock_characters)
    ):
        result = detector.detect(metrics, text, rec_id, char)
        has_anomaly = any([
            result["word_count_outlier"],
            result["readability_outlier"],
            result["duplicate_risk"],
            result["style_mismatch"]
        ])

        if has_anomaly:
            print(f"\n  {rec_id} - ANOMALIES FOUND:")
            for detail in result["anomaly_details"]:
                print(f"    - {detail}")
            if result["duplicate_of"]:
                print(f"    duplicate of: {result['duplicate_of']}")
        else:
            print(f"  {rec_id} - clean")

    print("\n✓ AnomalyDetector tests complete")
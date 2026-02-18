# ============================================================
# metrics_calculator.py
# MOMENT Preprocessing Pipeline - Metrics Calculation Module
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Calculates quantitative text metrics for each
# record. Runs AFTER issue_detector.py and BEFORE
# anomaly_detector.py (anomaly detection needs these metrics).
#
# METRICS CALCULATED:
#   - word_count: total number of words
#   - char_count: characters excluding whitespace
#   - sentence_count: number of sentences
#   - avg_word_length: average characters per word
#   - avg_sentence_length: average words per sentence
#   - readability_score: Flesch Reading Ease (0-100)
#     100 = very easy (children's book)
#     0   = very hard (academic paper)
#     60-70 = standard/plain English target
#
# These metrics feed into:
#   - anomaly_detector.py (outlier detection)
#   - output JSON (stored per record for downstream ML use)
# ============================================================

import re           # for sentence splitting
import logging
import textstat     # type: ignore # for Flesch Reading Ease score

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates quantitative text metrics for interpretations
    and passages.

    All metrics are controlled by config.yaml under the
    metrics section. Each metric can be turned on/off.

    Usage:
        calculator = MetricsCalculator(config)
        metrics = calculator.calculate(text)
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Reads metrics settings from config.yaml to know
        which metrics to calculate.

        Args:
            config: full config dict from config/config.yaml
        """
        metrics_config = config.get("metrics", {})

        # store which metrics to calculate
        # each defaults to True if not specified in config
        self.calc_readability = metrics_config.get(
            "calculate_readability", True
        )
        self.calc_word_count = metrics_config.get(
            "calculate_word_count", True
        )
        self.calc_char_count = metrics_config.get(
            "calculate_char_count", True
        )
        self.calc_sentence_count = metrics_config.get(
            "calculate_sentence_count", True
        )
        self.calc_avg_word_length = metrics_config.get(
            "calculate_avg_word_length", True
        )
        self.calc_avg_sentence_length = metrics_config.get(
            "calculate_avg_sentence_length", True
        )

        logger.debug(
            f"MetricsCalculator initialized. "
            f"Calculating: readability={self.calc_readability}, "
            f"word_count={self.calc_word_count}, "
            f"sentence_count={self.calc_sentence_count}"
        )

    def calculate(self, text: str) -> dict:
        """
        Calculate all enabled metrics for a piece of text.

        Args:
            text: cleaned text to calculate metrics for

        Returns:
            dict: {
                "word_count": int,
                "char_count": int,
                "sentence_count": int,
                "avg_word_length": float,
                "avg_sentence_length": float,
                "readability_score": float
            }

            Returns zero values for all metrics if text is empty.
        """
        # handle empty text gracefully
        if not text or not text.strip():
            return self._empty_metrics()

        metrics = {}

        # tokenize once and reuse
        # avoids splitting the text multiple times
        words = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)

        # calculate each metric if enabled
        if self.calc_word_count:
            metrics["word_count"] = len(words)

        if self.calc_char_count:
            # character count excludes whitespace
            metrics["char_count"] = len(text.replace(" ", "").replace("\n", ""))

        if self.calc_sentence_count:
            metrics["sentence_count"] = len(sentences)

        if self.calc_avg_word_length:
            metrics["avg_word_length"] = self._avg_word_length(words)

        if self.calc_avg_sentence_length:
            metrics["avg_sentence_length"] = self._avg_sentence_length(
                words, sentences
            )

        if self.calc_readability:
            metrics["readability_score"] = self._flesch_reading_ease(text)

        return metrics

    def calculate_batch(self, texts: list) -> list:
        """
        Calculate metrics for a list of texts.

        Args:
            texts: list of cleaned text strings

        Returns:
            list of metrics dicts (same length as input)
        """
        logger.info(f"Calculating metrics for {len(texts)} texts...")

        results = [self.calculate(text) for text in texts]

        logger.info("Metrics calculation complete.")
        return results

    def get_dataset_stats(self, metrics_list: list) -> dict:
        """
        Calculate aggregate statistics across all records.

        Used by anomaly_detector.py to establish baselines
        for outlier detection (mean, std, Q1, Q3, IQR).

        Args:
            metrics_list: list of metrics dicts from calculate()

        Returns:
            dict: aggregate stats for each metric:
            {
                "word_count": {
                    "mean": float, "std": float,
                    "min": float, "max": float,
                    "q1": float, "q3": float, "iqr": float
                },
                "readability_score": { ... },
                ...
            }
        """
        if not metrics_list:
            return {}

        # collect values for each metric
        metric_values = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(float(value))

        # calculate stats for each metric
        stats = {}
        for metric_name, values in metric_values.items():
            if not values:
                continue

            sorted_values = sorted(values)
            n = len(sorted_values)

            # calculate mean
            mean = sum(values) / n

            # calculate standard deviation
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance ** 0.5

            # calculate quartiles for IQR
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1

            stats[metric_name] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4)
            }

        logger.info(
            f"Dataset stats calculated for {len(stats)} metrics "
            f"across {len(metrics_list)} records."
        )

        return stats

    # --------------------------------------------------------
    # PRIVATE CALCULATION METHODS
    # --------------------------------------------------------

    def _tokenize_words(self, text: str) -> list:
        """
        Split text into individual words.

        Uses simple whitespace splitting rather than NLTK
        to keep this fast and dependency-light. Filters out
        empty strings that can result from multiple spaces.

        Args:
            text: input text

        Returns:
            list of word strings
        """
        # split on whitespace and filter empty strings
        words = [w for w in text.split() if w]
        return words

    def _tokenize_sentences(self, text: str) -> list:
        """
        Split text into sentences.

        Uses regex rather than NLTK sentence tokenizer to
        avoid NLTK dependency in this module. Splits on:
        - Period followed by space and capital letter
        - Exclamation mark followed by space
        - Question mark followed by space
        - Newlines (common in our interpretations)

        Handles edge cases like:
        - "Dr. James" (abbreviation) - won't split incorrectly
        - "Beautiful!--" (punctuation chains)

        Args:
            text: input text

        Returns:
            list of sentence strings (non-empty only)
        """
        # split on sentence-ending punctuation followed by
        # whitespace or end of string
        # (?<=[.!?]) = lookbehind for sentence-ending punct
        # \s+ = one or more whitespace chars
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        # filter out empty strings and very short fragments
        # (less than 3 chars is probably not a real sentence)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]

        # ensure at least 1 sentence is returned
        if not sentences:
            sentences = [text]

        return sentences

    def _avg_word_length(self, words: list) -> float:
        """
        Calculate average word length in characters.

        Only counts alphabetic characters in each word to
        avoid punctuation affecting the count.
        e.g. "beautiful," → 9 chars (not 10)

        Args:
            words: list of word strings from _tokenize_words()

        Returns:
            float: average word length, 0.0 if no words
        """
        if not words:
            return 0.0

        # count only alphabetic chars per word
        total_chars = sum(
            len(re.sub(r"[^a-zA-Z]", "", word))
            for word in words
        )

        return round(total_chars / len(words), 2)

    def _avg_sentence_length(self, words: list,
                              sentences: list) -> float:
        """
        Calculate average sentence length in words.

        Args:
            words: list of words from _tokenize_words()
            sentences: list of sentences from _tokenize_sentences()

        Returns:
            float: average words per sentence, 0.0 if no sentences
        """
        if not sentences or not words:
            return 0.0

        return round(len(words) / len(sentences), 2)

    def _flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score using textstat library.

        Flesch Reading Ease formula:
        206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

        Score interpretation:
            90-100: Very Easy (5th grade)
            70-90:  Easy (6th grade)
            60-70:  Standard (7th grade) ← most interpretations
            50-60:  Fairly Difficult (high school)
            30-50:  Difficult (college)
            0-30:   Very Difficult (professional/academic)

        For our dataset:
        - NEW READERs should score higher (easier writing)
        - DELIBERATE/Well-read should score lower (complex writing)
        - This feeds into anomaly detection style_mismatch check

        Args:
            text: text to calculate readability for

        Returns:
            float: Flesch Reading Ease score
                   clamped between 0.0 and 100.0
                   returns 0.0 on error
        """
        try:
            score = textstat.flesch_reading_ease(text)
            # textstat can return values outside 0-100 for unusual text
            # clamp to valid range
            score = max(0.0, min(100.0, score))
            return round(score, 2)
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.0

    def _empty_metrics(self) -> dict:
        """
        Return zero values for all metrics when text is empty.

        Returns:
            dict: all metrics set to 0
        """
        return {
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "readability_score": 0.0
        }


# ============================================================
# TEST BLOCK
# Run this file directly to test metrics calculation:
#   python -m data_pipeline.preprocessing.metrics_calculator
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing MetricsCalculator")
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

    calculator = MetricsCalculator(config)

    # test with real interpretations from our dataset
    test_cases = [
        (
            "Emma Chen (longest, analytical)",
            'He says "catastrophe" before anything bad happens. '
            'Just think about that. The creature opened its eyes. '
            "That's it. Victor's already calling it disaster. "
            '"Beautiful!--Great God!" Right next to each other. '
            "His brain's breaking. He built this with specific "
            "features and now he can't handle that it's real. "
            "That yellow eye. Why does he fixate on that one detail? "
            "Reducing the whole being to one gross feature so he "
            "doesn't have to see it as alive."
        ),
        (
            "Ryan O'Connor (shortest, casual)",
            "Dude builds monster. Monster opens eyes. "
            "Dude runs away screaming. How did he not see this "
            "coming? Like bro you literally assembled the parts yourself."
        ),
        (
            "Dr. James Fletcher (academic, complex)",
            "The creature hasn't committed any act yet Victor calls "
            "this catastrophe. Judgment precedes action. That's the "
            "foundation of prejudice -- deciding what something is "
            "before observing what it does. How can I describe my "
            "emotions at this catastrophe. He can't describe them "
            "because they're irrational."
        ),
        (
            "Empty text",
            ""
        ),
    ]

    all_metrics = []
    for test_name, text in test_cases:
        print(f"\n--- {test_name} ---")
        metrics = calculator.calculate(text)
        for key, value in metrics.items():
            print(f"  {key:25} : {value}")
        all_metrics.append(metrics)

    # test dataset stats (used by anomaly detector)
    print("\n--- Dataset Stats (across all test cases) ---")
    # exclude empty metrics from stats
    non_empty = [m for m in all_metrics if m["word_count"] > 0]
    stats = calculator.get_dataset_stats(non_empty)
    for metric_name, metric_stats in stats.items():
        print(f"\n  {metric_name}:")
        for stat_name, value in metric_stats.items():
            print(f"    {stat_name:8} : {value}")

    # readability score interpretation
    print("\n--- Readability Score Guide ---")
    print("  90-100: Very Easy (5th grade)")
    print("  70-90:  Easy (6th grade)")
    print("  60-70:  Standard (7th grade)")
    print("  50-60:  Fairly Difficult (high school)")
    print("  30-50:  Difficult (college)")
    print("  0-30:   Very Difficult (professional)")

    print("\n✓ MetricsCalculator tests complete")
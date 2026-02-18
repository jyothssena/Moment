# ============================================================
# text_validator.py
# MOMENT Preprocessing Pipeline - Text Validation Module
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Validates cleaned text against quality thresholds.
# This runs AFTER text_cleaner.py and BEFORE issue_detector.py
#
# WHAT IT CHECKS:
#   - Minimum/maximum word count
#   - Minimum/maximum character count
#   - Language detection (English only)
#   - Gibberish detection (vowel/consonant ratio)
#   - Character diversity (repetitive text)
#   - Overall quality score (0.0 to 1.0)
#
# IMPORTANT: Failed validation does NOT remove records.
# Records are flagged with is_valid=False and quality_issues
# listed. They stay in the output for full transparency.
# This is intentional - downstream systems decide what to
# do with invalid records.
# ============================================================

import re           # for text analysis patterns
import logging
from langdetect import detect, LangDetectException  # type: ignore # language detection

logger = logging.getLogger(__name__)


class TextValidator:
    """
    Validates cleaned text quality for interpretations and passages.

    Produces:
        - is_valid: bool (True if all checks pass)
        - quality_score: float 0.0-1.0
        - quality_issues: list of strings describing problems

    Usage:
        validator = TextValidator(config)
        result = validator.validate(text, text_type="interpretation")
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Reads validation thresholds from config.yaml under
        the validation section. Separate thresholds for
        interpretations vs passages.

        Args:
            config: full config dict from config/config.yaml
        """
        validation_config = config.get("validation", {})

        # thresholds for interpretation text
        interp_config = validation_config.get("interpretations", {})
        self.interp_min_words = interp_config.get("min_words", 10)
        self.interp_max_words = interp_config.get("max_words", 600)
        self.interp_min_chars = interp_config.get("min_chars", 50)
        self.interp_max_chars = interp_config.get("max_chars", 4000)
        self.interp_quality_threshold = interp_config.get(
            "quality_threshold", 0.5
        )

        # thresholds for passage text
        passage_config = validation_config.get("passages", {})
        self.passage_min_words = passage_config.get("min_words", 20)
        self.passage_max_words = passage_config.get("max_words", 1000)
        self.passage_min_chars = passage_config.get("min_chars", 100)
        self.passage_max_chars = passage_config.get("max_chars", 6000)
        self.passage_quality_threshold = passage_config.get(
            "quality_threshold", 0.6
        )

        logger.debug(
            f"TextValidator initialized. "
            f"Interpretation thresholds: "
            f"{self.interp_min_words}-{self.interp_max_words} words. "
            f"Passage thresholds: "
            f"{self.passage_min_words}-{self.passage_max_words} words."
        )

    def validate(self, text: str, text_type: str = "interpretation") -> dict:
        """
        Run all validation checks on a piece of text.

        Runs each check independently - a failure in one check
        doesn't prevent other checks from running. This gives
        us a complete picture of all issues in one pass.

        Args:
            text: cleaned text to validate
            text_type: "interpretation" or "passage"
                       controls which thresholds to use

        Returns:
            dict: {
                "is_valid": bool,
                "quality_score": float (0.0-1.0),
                "quality_issues": list of issue strings,
                "word_count": int,
                "char_count": int,
                "language": str
            }
        """
        # handle empty text
        if not text or not text.strip():
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "quality_issues": ["empty_text"],
                "word_count": 0,
                "char_count": 0,
                "language": "unknown"
            }

        # select thresholds based on text type
        if text_type == "passage":
            min_words = self.passage_min_words
            max_words = self.passage_max_words
            min_chars = self.passage_min_chars
            max_chars = self.passage_max_chars
            quality_threshold = self.passage_quality_threshold
        else:
            # default to interpretation thresholds
            min_words = self.interp_min_words
            max_words = self.interp_max_words
            min_chars = self.interp_min_chars
            max_chars = self.interp_max_chars
            quality_threshold = self.interp_quality_threshold

        # collect all issues found
        quality_issues = []

        # --- Run all checks ---

        # Check 1: word count
        word_count = len(text.split())
        if word_count < min_words:
            quality_issues.append(
                f"too_short: {word_count} words (min: {min_words})"
            )
        elif word_count > max_words:
            quality_issues.append(
                f"too_long: {word_count} words (max: {max_words})"
            )

        # Check 2: character count
        char_count = len(text.replace(" ", ""))  # count non-space chars
        if char_count < min_chars:
            quality_issues.append(
                f"too_few_chars: {char_count} chars (min: {min_chars})"
            )
        elif char_count > max_chars:
            quality_issues.append(
                f"too_many_chars: {char_count} chars (max: {max_chars})"
            )

        # Check 3: language detection
        language = self._detect_language(text)
        if language != "en":
            quality_issues.append(
                f"wrong_language: detected {language!r} (expected: en)"
            )

        # Check 4: gibberish detection
        if self._is_gibberish(text):
            quality_issues.append("gibberish: abnormal vowel/consonant ratio")

        # Check 5: character diversity
        if self._is_repetitive(text):
            quality_issues.append("repetitive: low character diversity")

        # --- Calculate quality score ---
        # score starts at 1.0 and decreases for each issue
        quality_score = self._calculate_quality_score(
            text, word_count, char_count,
            min_words, max_words, quality_issues
        )

        # --- Determine validity ---
        # valid if: no issues AND quality score above threshold
        is_valid = (
            len(quality_issues) == 0 and
            quality_score >= quality_threshold
        )

        result = {
            "is_valid": is_valid,
            "quality_score": round(quality_score, 4),
            "quality_issues": quality_issues,
            "word_count": word_count,
            "char_count": char_count,
            "language": language
        }

        # log invalid records at debug level
        if not is_valid:
            logger.debug(
                f"Text validation failed. "
                f"Issues: {quality_issues}. "
                f"Score: {quality_score:.4f}"
            )

        return result

    def validate_batch(self, texts: list,
                       text_type: str = "interpretation") -> list:
        """
        Validate a list of texts.

        Args:
            texts: list of cleaned text strings
            text_type: "interpretation" or "passage"

        Returns:
            list of validation result dicts
        """
        logger.info(
            f"Validating batch of {len(texts)} {text_type} texts..."
        )

        results = [self.validate(text, text_type) for text in texts]

        # log summary
        valid_count = sum(1 for r in results if r["is_valid"])
        logger.info(
            f"Batch validation complete: "
            f"{valid_count}/{len(results)} valid {text_type} texts."
        )

        return results

    # --------------------------------------------------------
    # PRIVATE VALIDATION METHODS
    # --------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of a text.

        Uses langdetect library. Returns 'en' for English.
        Returns 'unknown' if detection fails (e.g. text too short).

        Very short texts (< 20 chars) are assumed to be English
        because langdetect is unreliable on short text.

        Args:
            text: text to detect language of

        Returns:
            str: ISO language code e.g. "en", "fr", "unknown"
        """
        # langdetect is unreliable on very short text
        if len(text) < 20:
            return "en"  # assume English for very short text

        try:
            language = detect(text)
            return language
        except LangDetectException:
            # detection failed - text may be too short or ambiguous
            logger.debug(f"Language detection failed for text: {text[:50]!r}")
            return "unknown"

    def _is_gibberish(self, text: str) -> bool:
        """
        Detect if text is gibberish using vowel/consonant ratio.

        Real English text has a roughly consistent ratio of
        vowels to consonants. Gibberish (random characters,
        keyboard mashing) breaks this pattern.

        Rule: if vowels are < 15% or > 60% of total letters,
        flag as potential gibberish.

        Examples of gibberish:
            "asdfghjkl qwerty zxcvb" (too few vowels)
            "aeiouaeiou eeeee" (too many vowels)

        Args:
            text: text to check

        Returns:
            bool: True if text looks like gibberish
        """
        # get only alphabetic characters
        letters = [c.lower() for c in text if c.isalpha()]

        if len(letters) < 10:
            # too short to make a reliable judgment
            return False

        vowels = set("aeiou")
        vowel_count = sum(1 for c in letters if c in vowels)
        vowel_ratio = vowel_count / len(letters)

        # English typically has 35-50% vowels
        # we use wider bounds (15-60%) to avoid false positives
        if vowel_ratio < 0.15 or vowel_ratio > 0.60:
            logger.debug(
                f"Gibberish detected: vowel ratio = {vowel_ratio:.2f}"
            )
            return True

        return False

    def _is_repetitive(self, text: str) -> bool:
        """
        Detect if text has abnormally low character diversity.

        Repetitive text (e.g. "aaaaaaa" or "the the the the")
        indicates low quality or spam.

        Rule: if the most common character makes up > 40% of
        all characters (excluding spaces), flag as repetitive.

        Args:
            text: text to check

        Returns:
            bool: True if text appears repetitive
        """
        # remove spaces for this check
        chars = [c for c in text.lower() if not c.isspace()]

        if len(chars) < 20:
            # too short to make a reliable judgment
            return False

        # count frequency of most common character
        char_counts = {}
        for c in chars:
            char_counts[c] = char_counts.get(c, 0) + 1

        most_common_count = max(char_counts.values())
        most_common_ratio = most_common_count / len(chars)

        if most_common_ratio > 0.40:
            logger.debug(
                f"Repetitive text detected: most common char ratio = "
                f"{most_common_ratio:.2f}"
            )
            return True

        return False

    def _calculate_quality_score(self, text: str,
                                  word_count: int,
                                  char_count: int,
                                  min_words: int,
                                  max_words: int,
                                  issues: list) -> float:
        """
        Calculate an overall quality score from 0.0 to 1.0.

        Scoring approach:
        - Start at 1.0
        - Deduct points for each issue found
        - Apply bonus for ideal length range
        - Clamp final score between 0.0 and 1.0

        Deductions:
            - empty text:          -1.0 (automatic 0.0)
            - too short/long:      -0.3
            - wrong language:      -0.4
            - gibberish:           -0.5
            - repetitive:          -0.3

        Args:
            text: the text being scored
            word_count: pre-calculated word count
            char_count: pre-calculated char count
            min_words: minimum word threshold
            max_words: maximum word threshold
            issues: list of issues already found

        Returns:
            float: quality score between 0.0 and 1.0
        """
        score = 1.0

        # deduct for each type of issue
        for issue in issues:
            if "empty_text" in issue:
                return 0.0  # automatic zero for empty text
            elif "too_short" in issue or "too_long" in issue:
                score -= 0.3
            elif "too_few_chars" in issue or "too_many_chars" in issue:
                score -= 0.1  # smaller deduction - already caught by word count
            elif "wrong_language" in issue:
                score -= 0.4
            elif "gibberish" in issue:
                score -= 0.5
            elif "repetitive" in issue:
                score -= 0.3

        # apply small bonus for ideal length range
        # ideal = between min_words and max_words/2
        # (we don't want to reward extremely long interpretations)
        ideal_max = max_words // 2
        if min_words <= word_count <= ideal_max:
            score = min(1.0, score + 0.05)  # small bonus, capped at 1.0

        # clamp score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        return score


# ============================================================
# TEST BLOCK
# Run this file directly to test validation:
#   python -m data_pipeline.preprocessing.text_validator
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing TextValidator")
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

    validator = TextValidator(config)

    # test cases
    test_cases = [
        (
            "Valid interpretation (Emma Chen)",
            'He says "catastrophe" before anything bad happens. '
            'Just think about that. The creature opened its eyes. '
            'That\'s it. Victor\'s already calling it disaster. '
            '"Beautiful!--Great God!" Right next to each other. '
            'His brain\'s breaking. He built this with specific features '
            'and now he can\'t handle that it\'s real.',
            "interpretation"
        ),
        (
            "Too short (Ryan O'Connor)",
            "Dude builds monster. Monster opens eyes. Dude runs away.",
            "interpretation"
        ),
        (
            "Very short (Eric Sullivan)",
            "Creature wakes up, looks weird, Victor bolts.",
            "interpretation"
        ),
        (
            "Gibberish text",
            "asdfghjkl qwerty zxcvbnm poiuyt rewq",
            "interpretation"
        ),
        (
            "Empty text",
            "",
            "interpretation"
        ),
        (
            "Valid passage (Frankenstein)",
            'It was on a dreary night of November that I beheld the '
            'accomplishment of my toils. With an anxiety that almost '
            'amounted to agony, I collected the instruments of life '
            'around me, that I might infuse a spark of being into the '
            'lifeless thing that lay at my feet.',
            "passage"
        ),
    ]

    for test_name, text, text_type in test_cases:
        print(f"\n--- {test_name} ---")
        result = validator.validate(text, text_type)
        print(f"  is_valid:      {result['is_valid']}")
        print(f"  quality_score: {result['quality_score']}")
        print(f"  word_count:    {result['word_count']}")
        print(f"  language:      {result['language']}")
        if result['quality_issues']:
            print(f"  issues:        {result['quality_issues']}")
        else:
            print(f"  issues:        none")

    print("\n✓ TextValidator tests complete")
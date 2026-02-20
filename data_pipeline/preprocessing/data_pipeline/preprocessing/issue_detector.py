# ============================================================
# issue_detector.py
# MOMENT Preprocessing Pipeline - Issue Detection Module
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Scans cleaned text for specific quality issues
# beyond basic validation. This runs AFTER text_validator.py
# and BEFORE metrics_calculator.py.
#
# WHAT IT DETECTS:
#   - PII (Personally Identifiable Information):
#       emails, phone numbers, SSNs, credit cards
#   - Profanity: configurable word list with ratio threshold
#   - Spam patterns: excessive caps, punctuation, repetition
#
# IMPORTANT: Like validation, detected issues do NOT remove
# records. They are flagged in detected_issues field of output.
# Records with issues stay in output for full transparency.
#
# NOTE ON SYNTHESIZED DATA:
# Our current dataset is synthesized so we don't expect real
# PII or profanity. But we build this properly because in
# production, real users will submit text and these checks
# become critical.
# ============================================================

import re           # for regex pattern matching
import logging

logger = logging.getLogger(__name__)

# ============================================================
# PROFANITY WORD LIST
# Kept minimal and outside the class so it can be easily
# extended or replaced with a proper library in production.
# In production, replace this with a proper profanity library
# like 'better-profanity' or load from a config file.
# ============================================================
PROFANITY_WORDS = {
    "damn", "hell", "crap", "ass", "bastard", "bitch",
    "shit", "fuck", "piss", "dick", "cock", "cunt",
    "whore", "slut", "fag", "retard"
}

# ============================================================
# SPAM PHRASES
# Common spam/low-quality patterns to detect.
# ============================================================
SPAM_PHRASES = [
    "click here",
    "buy now",
    "free money",
    "you won",
    "congratulations you",
    "limited time offer",
    "act now",
    "call now",
    "order now",
    "visit our website"
]


class IssueDetector:
    """
    Detects specific quality issues in cleaned text.

    Produces a detected_issues dict for each record:
    {
        "has_pii": bool,
        "pii_types": list,          # e.g. ["email", "phone"]
        "has_profanity": bool,
        "profanity_ratio": float,   # ratio of profane words
        "is_spam": bool,
        "spam_reasons": list        # e.g. ["excessive_caps"]
    }

    Usage:
        detector = IssueDetector(config)
        issues = detector.detect(text)
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Reads issue_detection settings from config.yaml.

        Args:
            config: full config dict from config/config.yaml
        """
        issue_config = config.get("issue_detection", {})

        # PII detection settings
        pii_config = issue_config.get("pii", {})
        self.check_emails = pii_config.get("check_emails", True)
        self.check_phones = pii_config.get("check_phone_numbers", True)
        self.check_ssn = pii_config.get("check_ssn", True)
        self.check_credit_cards = pii_config.get("check_credit_cards", True)

        # profanity settings
        profanity_config = issue_config.get("profanity", {})
        self.check_profanity = profanity_config.get("enabled", True)
        self.profanity_ratio_threshold = profanity_config.get(
            "ratio_threshold", 0.30
        )

        # spam settings
        spam_config = issue_config.get("spam", {})
        self.check_spam = spam_config.get("enabled", True)
        self.caps_threshold = spam_config.get("caps_threshold", 0.50)
        self.punctuation_threshold = spam_config.get(
            "punctuation_threshold", 0.10
        )
        self.repetitive_chars = spam_config.get("repetitive_chars", 4)
        self.repetitive_words_threshold = spam_config.get(
            "repetitive_words_threshold", 0.30
        )

        # compile regex patterns once at init time
        # (compiling is expensive - do it once, reuse many times)
        self._compile_patterns()

        logger.debug(
            f"IssueDetector initialized. "
            f"PII checks: emails={self.check_emails}, "
            f"phones={self.check_phones}, "
            f"ssn={self.check_ssn}. "
            f"Profanity: {self.check_profanity}. "
            f"Spam: {self.check_spam}."
        )

    def _compile_patterns(self) -> None:
        """
        Compile all regex patterns once at initialization.

        Compiling regex is expensive - by doing it once here
        instead of on every detect() call, we speed up
        processing of all 450 records significantly.
        """
        # email pattern
        # matches: word@domain.tld
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        )

        # US phone number patterns
        # matches: (555) 123-4567, 555-123-4567, 5551234567, +1 555 123 4567
        self._phone_pattern = re.compile(
            r"(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})"
        )

        # SSN pattern
        # matches: 123-45-6789 or 123 45 6789
        self._ssn_pattern = re.compile(
            r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
        )

        # credit card pattern
        # matches: 4 groups of 4 digits (most card formats)
        # e.g. 1234 5678 9012 3456 or 1234-5678-9012-3456
        self._credit_card_pattern = re.compile(
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        )

        # repetitive character pattern
        # matches N or more of the same character in a row
        # pattern is built dynamically using self.repetitive_chars
        self._repetitive_char_pattern = re.compile(
            r"(.)\1{" + str(self.repetitive_chars - 1) + r",}"
        )

    def detect(self, text: str) -> dict:
        """
        Run all enabled issue detection checks on text.

        Args:
            text: cleaned text to check for issues

        Returns:
            dict: {
                "has_pii": bool,
                "pii_types": list,
                "has_profanity": bool,
                "profanity_ratio": float,
                "is_spam": bool,
                "spam_reasons": list
            }
        """
        # handle empty text
        if not text or not text.strip():
            return self._empty_result()

        # run each detection module
        pii_result = self._detect_pii(text)
        profanity_result = self._detect_profanity(text)
        spam_result = self._detect_spam(text)

        # combine results
        result = {
            # PII fields
            "has_pii": pii_result["has_pii"],
            "pii_types": pii_result["pii_types"],

            # profanity fields
            "has_profanity": profanity_result["has_profanity"],
            "profanity_ratio": profanity_result["profanity_ratio"],

            # spam fields
            "is_spam": spam_result["is_spam"],
            "spam_reasons": spam_result["spam_reasons"]
        }

        # log if any issues were found
        if result["has_pii"] or result["has_profanity"] or result["is_spam"]:
            logger.debug(
                f"Issues detected: "
                f"PII={result['has_pii']} ({result['pii_types']}), "
                f"profanity={result['has_profanity']}, "
                f"spam={result['is_spam']} ({result['spam_reasons']})"
            )

        return result

    def detect_batch(self, texts: list) -> list:
        """
        Detect issues in a list of texts.

        Args:
            texts: list of cleaned text strings

        Returns:
            list of issue result dicts
        """
        logger.info(f"Running issue detection on {len(texts)} texts...")

        results = [self.detect(text) for text in texts]

        # log summary
        pii_count = sum(1 for r in results if r["has_pii"])
        profanity_count = sum(1 for r in results if r["has_profanity"])
        spam_count = sum(1 for r in results if r["is_spam"])

        logger.info(
            f"Issue detection complete: "
            f"PII={pii_count}, "
            f"profanity={profanity_count}, "
            f"spam={spam_count} "
            f"out of {len(results)} texts."
        )

        return results

    # --------------------------------------------------------
    # PRIVATE DETECTION METHODS
    # --------------------------------------------------------

    def _detect_pii(self, text: str) -> dict:
        """
        Detect Personally Identifiable Information in text.

        Checks for: emails, phone numbers, SSNs, credit cards.
        Each check is only run if enabled in config.

        Args:
            text: text to check for PII

        Returns:
            dict: {
                "has_pii": bool,
                "pii_types": list of PII types found
            }
        """
        pii_types = []

        # check for email addresses
        if self.check_emails:
            if self._email_pattern.search(text):
                pii_types.append("email")

        # check for phone numbers
        if self.check_phones:
            # phone pattern can match many things - add extra length check
            # to reduce false positives (real phone numbers are 10+ digits)
            matches = self._phone_pattern.findall(text)
            # filter out matches that are too short to be real phone numbers
            real_phones = [
                m for m in matches
                if len(re.sub(r"\D", "", "".join(m))) >= 10
            ]
            if real_phones:
                pii_types.append("phone_number")

        # check for SSNs
        if self.check_ssn:
            if self._ssn_pattern.search(text):
                pii_types.append("ssn")

        # check for credit card numbers
        if self.check_credit_cards:
            matches = self._credit_card_pattern.findall(text)
            if matches:
                pii_types.append("credit_card")

        return {
            "has_pii": len(pii_types) > 0,
            "pii_types": pii_types
        }

    def _detect_profanity(self, text: str) -> dict:
        """
        Detect profanity in text using word list + ratio check.

        Two-stage check:
        1. Check if any profane words appear in text
        2. If found, calculate ratio of profane to total words
           Only flag if ratio exceeds threshold (default 30%)
           This prevents flagging literary text that legitimately
           contains one or two strong words.

        Args:
            text: text to check for profanity

        Returns:
            dict: {
                "has_profanity": bool,
                "profanity_ratio": float
            }
        """
        if not self.check_profanity:
            return {"has_profanity": False, "profanity_ratio": 0.0}

        # tokenize to words (lowercase, strip punctuation)
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        if not words:
            return {"has_profanity": False, "profanity_ratio": 0.0}

        # count profane words
        profane_words = [w for w in words if w in PROFANITY_WORDS]
        profanity_ratio = len(profane_words) / len(words)

        # only flag if ratio exceeds threshold
        # this prevents false positives on literary text
        has_profanity = (
            len(profane_words) > 0 and
            profanity_ratio >= self.profanity_ratio_threshold
        )

        return {
            "has_profanity": has_profanity,
            "profanity_ratio": round(profanity_ratio, 4)
        }

    def _detect_spam(self, text: str) -> dict:
        """
        Detect spam patterns in text.

        Checks for:
        1. Excessive capitalization (>50% uppercase letters)
        2. Excessive punctuation (>10% of chars are punctuation)
        3. Repetitive characters (same char 4+ times in a row)
        4. Repetitive words (any word >30% of total words)
        5. Known spam phrases

        Args:
            text: text to check for spam patterns

        Returns:
            dict: {
                "is_spam": bool,
                "spam_reasons": list of reasons
            }
        """
        if not self.check_spam:
            return {"is_spam": False, "spam_reasons": []}

        spam_reasons = []

        # Check 1: excessive capitalization
        letters = [c for c in text if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > self.caps_threshold:
                spam_reasons.append(
                    f"excessive_caps: {caps_ratio:.1%} uppercase"
                )

        # Check 2: excessive punctuation
        if len(text) > 0:
            punct_chars = re.findall(r"[^\w\s]", text)
            punct_ratio = len(punct_chars) / len(text)
            if punct_ratio > self.punctuation_threshold:
                spam_reasons.append(
                    f"excessive_punctuation: {punct_ratio:.1%} punctuation"
                )

        # Check 3: repetitive characters
        # e.g. "heeeeello" or "!!!!!!"
        if self._repetitive_char_pattern.search(text):
            spam_reasons.append(
                f"repetitive_chars: same char repeated "
                f"{self.repetitive_chars}+ times"
            )

        # Check 4: repetitive words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            most_common_word = max(word_counts, key=word_counts.get)
            most_common_ratio = word_counts[most_common_word] / len(words)

            if most_common_ratio > self.repetitive_words_threshold:
                spam_reasons.append(
                    f"repetitive_words: '{most_common_word}' appears "
                    f"{most_common_ratio:.1%} of words"
                )

        # Check 5: known spam phrases
        text_lower = text.lower()
        for phrase in SPAM_PHRASES:
            if phrase in text_lower:
                spam_reasons.append(f"spam_phrase: '{phrase}'")

        return {
            "is_spam": len(spam_reasons) > 0,
            "spam_reasons": spam_reasons
        }

    def _empty_result(self) -> dict:
        """
        Return a default result dict for empty/None text.

        Used when detect() receives empty input.

        Returns:
            dict: all-false issue result
        """
        return {
            "has_pii": False,
            "pii_types": [],
            "has_profanity": False,
            "profanity_ratio": 0.0,
            "is_spam": False,
            "spam_reasons": []
        }


# ============================================================
# TEST BLOCK
# Run this file directly to test issue detection:
#   python -m data_pipeline.preprocessing.issue_detector
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing IssueDetector")
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

    detector = IssueDetector(config)

    # test cases
    test_cases = [
        (
            "Clean interpretation (Emma Chen)",
            'He says "catastrophe" before anything bad happens. '
            'Just think about that. The creature opened its eyes.'
        ),
        (
            "PII - email address",
            "Contact me at emma.chen@example.com for my thoughts."
        ),
        (
            "PII - phone number",
            "Call me at (555) 123-4567 to discuss this passage."
        ),
        (
            "Excessive caps",
            "THIS IS AMAZING I LOVE THIS BOOK SO MUCH IT IS GREAT"
        ),
        (
            "Repetitive characters",
            "Sooooooo good. The passage is amazinggggg!!!!!!"
        ),
        (
            "Repetitive words",
            "the the the the the book the the the the passage the the"
        ),
        (
            "Real interpretation - no issues expected",
            '"Beautiful!--Great God!" Right next to each other. '
            'His brain\'s breaking. He built this with specific features '
            'and now he can\'t handle that it\'s real. The yellow eye. '
            'Why does he fixate on that one detail?'
        ),
    ]

    for test_name, text in test_cases:
        print(f"\n--- {test_name} ---")
        result = detector.detect(text)
        print(f"  has_pii:        {result['has_pii']} "
              f"{result['pii_types'] if result['pii_types'] else ''}")
        print(f"  has_profanity:  {result['has_profanity']} "
              f"(ratio: {result['profanity_ratio']})")
        print(f"  is_spam:        {result['is_spam']} "
              f"{result['spam_reasons'] if result['spam_reasons'] else ''}")

    print("\n✓ IssueDetector tests complete")
# ============================================================
# text_cleaner.py
# MOMENT Preprocessing Pipeline - Text Cleaning Module
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Cleans raw text from interpretations and passages.
# This is Phase 2 of the pipeline - runs AFTER extraction
# (reading files) and BEFORE validation.
#
# WHAT IT DOES:
#   - Fixes encoding issues (smart quotes, mojibake)
#   - Normalizes unicode characters
#   - Removes extra whitespace and newlines
#   - Optionally removes URLs and emails
#   - Does NOT change meaning - only fixes formatting
#
# IMPORTANT: This module never removes text or changes words.
# It only fixes formatting/encoding issues. The original
# text is preserved separately for reference.
# ============================================================

import re           # regular expressions for pattern matching
import unicodedata  # for unicode normalization
import logging
import chardet      # type: ignore # for detecting encoding issues

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw text for preprocessing.

    All cleaning operations are controlled by config.yaml
    under the text_cleaning section. Each operation can be
    turned on or off independently.

    Usage:
        cleaner = TextCleaner(config)
        cleaned_text = cleaner.clean(raw_text)
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Reads all text_cleaning settings from config so
        no cleaning behavior is hardcoded here.

        Args:
            config: full config dict from config/config.yaml
        """
        # extract text_cleaning settings from config
        cleaning_config = config.get("text_cleaning", {})

        # store each setting as an instance variable
        # with sensible defaults if not in config
        self.remove_extra_whitespace = cleaning_config.get(
            "remove_extra_whitespace", True
        )
        self.normalize_unicode = cleaning_config.get(
            "normalize_unicode", True
        )
        self.fix_encoding = cleaning_config.get(
            "fix_encoding", True
        )
        self.fix_smart_quotes = cleaning_config.get(
            "fix_smart_quotes", True
        )
        self.fix_dashes = cleaning_config.get(
            "fix_dashes", True
        )
        self.remove_urls = cleaning_config.get(
            "remove_urls", False   # default False - keep URLs
        )
        self.remove_emails = cleaning_config.get(
            "remove_emails", True
        )
        self.lowercase = cleaning_config.get(
            "lowercase", False     # default False - preserve casing
        )

        logger.debug(
            f"TextCleaner initialized with settings: "
            f"fix_encoding={self.fix_encoding}, "
            f"normalize_unicode={self.normalize_unicode}, "
            f"remove_urls={self.remove_urls}, "
            f"remove_emails={self.remove_emails}"
        )

    def clean(self, text: str) -> str:
        """
        Main cleaning method - applies all enabled cleaning steps.

        Runs each cleaning operation in order. Each step takes
        the output of the previous step as input, so order matters.

        Args:
            text: raw input text (interpretation or passage)

        Returns:
            str: cleaned text with all enabled operations applied

        Returns empty string if input is None or empty.
        """
        # handle None or empty input gracefully
        if not text:
            return ""

        # convert to string in case we get a non-string input
        text = str(text)

        # keep track of original for logging
        original_length = len(text)

        # apply cleaning steps in order
        # each step is only applied if enabled in config

        # Step 1: fix encoding issues first
        # (must be done before other operations)
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Step 2: fix smart quotes
        # (before unicode normalization to catch more cases)
        if self.fix_smart_quotes:
            text = self._fix_smart_quotes(text)

        # Step 3: fix dashes
        if self.fix_dashes:
            text = self._fix_dashes(text)

        # Step 4: normalize unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Step 5: remove emails (PII)
        if self.remove_emails:
            text = self._remove_emails(text)

        # Step 6: remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Step 7: remove extra whitespace (always last)
        # so previous steps don't leave gaps
        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)

        # Step 8: lowercase (if enabled)
        if self.lowercase:
            text = text.lower()

        # log if significant length change occurred
        cleaned_length = len(text)
        if original_length > 0:
            change_pct = abs(original_length - cleaned_length) / original_length * 100
            if change_pct > 20:
                # more than 20% length change is worth logging
                logger.debug(
                    f"Significant text length change after cleaning: "
                    f"{original_length} → {cleaned_length} chars "
                    f"({change_pct:.1f}% change)"
                )

        return text

    def clean_batch(self, texts: list) -> list:
        """
        Clean a list of texts.

        Convenience method for cleaning multiple texts at once.
        Used when processing all 450 interpretations.

        Args:
            texts: list of raw text strings

        Returns:
            list of cleaned text strings (same length as input)
        """
        logger.info(f"Cleaning batch of {len(texts)} texts...")

        cleaned = [self.clean(text) for text in texts]

        logger.info(f"Batch cleaning complete.")
        return cleaned

    # --------------------------------------------------------
    # PRIVATE CLEANING METHODS
    # Each method handles one specific cleaning operation.
    # All prefixed with _ to indicate they're internal.
    # --------------------------------------------------------

    def _fix_encoding(self, text: str) -> str:
        """
        Fix common encoding issues in text.

        Handles:
        - Mojibake: garbled text from wrong encoding
          e.g. â€™ → ' (right single quote)
          e.g. â€" → — (em dash)
        - Null bytes and other control characters
        - Windows-1252 characters misread as UTF-8

        Args:
            text: input text that may have encoding issues

        Returns:
            str: text with encoding issues fixed
        """
        try:
            # try to fix mojibake by encoding as latin-1 then decoding as utf-8
            # this catches the most common mojibake pattern
            fixed = text.encode("latin-1").decode("utf-8")
            return fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            # if that fails, text is probably already correct UTF-8
            pass

        # remove null bytes and other control characters
        # but keep newlines (\n), tabs (\t), and carriage returns (\r)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text

    def _fix_smart_quotes(self, text: str) -> str:
        """
        Replace typographic (smart) quotes with straight quotes.

        Smart quotes are curly quotes used in Word, Pages etc.
        They can cause issues in text processing.

        Replacements:
            " " → " "  (left/right double quotes → straight double)
            ' ' → ' '  (left/right single quotes → straight single)
            « » → " "  (French guillemets → straight double)
            „ " → " "  (German quotes → straight double)

        Args:
            text: input text with possible smart quotes

        Returns:
            str: text with smart quotes replaced
        """
        # left and right double quotation marks
        text = text.replace("\u201c", '"')   # " LEFT DOUBLE QUOTATION MARK
        text = text.replace("\u201d", '"')   # " RIGHT DOUBLE QUOTATION MARK

        # left and right single quotation marks / apostrophes
        text = text.replace("\u2018", "'")   # ' LEFT SINGLE QUOTATION MARK
        text = text.replace("\u2019", "'")   # ' RIGHT SINGLE QUOTATION MARK

        # French quotation marks (guillemets)
        text = text.replace("\u00ab", '"')   # « LEFT-POINTING DOUBLE ANGLE
        text = text.replace("\u00bb", '"')   # » RIGHT-POINTING DOUBLE ANGLE

        # German quotation marks
        text = text.replace("\u201e", '"')   # „ DOUBLE LOW-9 QUOTATION MARK

        # prime marks (used as quotes informally)
        text = text.replace("\u2032", "'")   # ′ PRIME
        text = text.replace("\u2033", '"')   # ″ DOUBLE PRIME

        return text

    def _fix_dashes(self, text: str) -> str:
        """
        Normalize dash characters to standard hyphens/dashes.

        The interpretations use various dash characters.
        We normalize to ASCII equivalents for consistency.

        Replacements:
            — (em dash)  → -- (double hyphen, readable equivalent)
            – (en dash)  → -  (single hyphen)
            ‐ (hyphen)   → -  (standard hyphen)
            ― (horizontal bar) → --

        Note: We use -- for em dashes rather than - to preserve
        the meaning of a pause/break in text.

        Args:
            text: input text with various dash characters

        Returns:
            str: text with normalized dashes
        """
        text = text.replace("\u2014", "--")  # — EM DASH → double hyphen
        text = text.replace("\u2013", "-")   # – EN DASH → single hyphen
        text = text.replace("\u2010", "-")   # ‐ HYPHEN → standard hyphen
        text = text.replace("\u2015", "--")  # ― HORIZONTAL BAR → double hyphen
        text = text.replace("\u2212", "-")   # − MINUS SIGN → hyphen

        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to their ASCII equivalents
        where possible, and remove or replace others.

        Uses NFC normalization (canonical decomposition followed
        by canonical composition) - this is the standard form
        for text storage and comparison.

        Examples:
            é → e (e + combining accent → plain e)
            ñ → n
            ü → u
            • → * (bullet point)
            … → ... (ellipsis)

        Args:
            text: input text with unicode characters

        Returns:
            str: NFC normalized text
        """
        # first apply NFC normalization
        # this composes characters like e + ́ into é
        text = unicodedata.normalize("NFC", text)

        # replace ellipsis character with three dots
        text = text.replace("\u2026", "...")  # … HORIZONTAL ELLIPSIS

        # replace bullet points with asterisk
        text = text.replace("\u2022", "*")    # • BULLET
        text = text.replace("\u2023", "*")    # ‣ TRIANGULAR BULLET
        text = text.replace("\u2043", "*")    # ⁃ HYPHEN BULLET

        # replace non-breaking space with regular space
        text = text.replace("\u00a0", " ")    # NO-BREAK SPACE

        # replace zero-width characters (invisible but cause issues)
        text = text.replace("\u200b", "")     # ZERO WIDTH SPACE
        text = text.replace("\u200c", "")     # ZERO WIDTH NON-JOINER
        text = text.replace("\u200d", "")     # ZERO WIDTH JOINER
        text = text.replace("\ufeff", "")     # ZERO WIDTH NO-BREAK SPACE (BOM)

        return text

    def _remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text (PII protection).

        Matches standard email format: user@domain.tld
        Replaces with [EMAIL REMOVED] placeholder so the
        removal is visible in output for transparency.

        Args:
            text: input text that may contain email addresses

        Returns:
            str: text with email addresses replaced
        """
        # standard email regex pattern
        # matches: word chars + special chars @ domain . tld
        email_pattern = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"

        # count matches for logging
        matches = re.findall(email_pattern, text)
        if matches:
            logger.debug(f"Removed {len(matches)} email address(es) from text.")
            text = re.sub(email_pattern, "[EMAIL REMOVED]", text)

        return text

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.

        Matches http/https URLs and www. URLs.
        Replaces with [URL REMOVED] placeholder.

        Note: remove_urls is False by default in config because
        interpretations may legitimately reference URLs.
        Only enable if needed.

        Args:
            text: input text that may contain URLs

        Returns:
            str: text with URLs replaced
        """
        # matches http/https URLs
        url_pattern = r"https?://[^\s]+"
        # matches www. URLs without http
        www_pattern = r"www\.[^\s]+"

        matches = re.findall(url_pattern, text) + re.findall(www_pattern, text)
        if matches:
            logger.debug(f"Removed {len(matches)} URL(s) from text.")
            text = re.sub(url_pattern, "[URL REMOVED]", text)
            text = re.sub(www_pattern, "[URL REMOVED]", text)

        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """
        Normalize all whitespace in text.

        Operations:
        1. Replace all tab characters with single space
        2. Replace Windows line endings (\r\n) with \n
        3. Replace multiple consecutive spaces with single space
        4. Replace 3+ consecutive newlines with double newline
           (preserves paragraph breaks but removes excess blank lines)
        5. Strip leading and trailing whitespace

        Args:
            text: input text with possible extra whitespace

        Returns:
            str: text with normalized whitespace
        """
        # replace tabs with space
        text = text.replace("\t", " ")

        # normalize Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # collapse 3+ newlines into 2 (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # strip leading and trailing whitespace
        text = text.strip()

        return text

    def get_cleaning_summary(self, original: str, cleaned: str) -> dict:
        """
        Generate a summary of what changed during cleaning.

        Useful for debugging and validation reports.

        Args:
            original: text before cleaning
            cleaned: text after cleaning

        Returns:
            dict: summary of changes made
        """
        return {
            "original_length": len(original),
            "cleaned_length": len(cleaned),
            "length_change": len(original) - len(cleaned),
            "length_change_pct": round(
                abs(len(original) - len(cleaned)) / max(len(original), 1) * 100,
                2
            ),
            "had_smart_quotes": any(c in original for c in '""'''),
            "had_em_dashes": "\u2014" in original or "\u2013" in original,
            "had_extra_whitespace": "  " in original or "\t" in original,
        }


# ============================================================
# TEST BLOCK
# Run this file directly to test cleaning operations:
#   python -m data_pipeline.preprocessing.text_cleaner
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import os

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing TextCleaner")
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

    cleaner = TextCleaner(config)

    # test cases - each tests a specific cleaning operation
    test_cases = [
        (
            "Smart quotes",
            '"Beautiful!\u2014Great God!" His yellow skin\u2026'
        ),
        (
            "Em dashes",
            "He says \u201ccatastrophe\u201d before\u2014anything bad happens."
        ),
        (
            "Extra whitespace",
            "He  worked   so   hard.\n\n\n\nAnd   then   failed."
        ),
        (
            "Mixed issues",
            "  \u2018One thought, one conception\u2019\u2014one purpose.  \n\n\n"
        ),
        (
            "Real interpretation sample",
            'He says \u201ccatastrophe\u201d before anything bad happens. '
            'Just\u2026 think about that. The creature opened its eyes. '
            'That\u2019s it. Victor\u2019s already calling it disaster.'
        ),
        (
            "Email removal",
            "Contact me at test@example.com for more info."
        ),
    ]

    for test_name, raw_text in test_cases:
        print(f"\n--- {test_name} ---")
        print(f"  Input:  {repr(raw_text[:80])}")
        cleaned = cleaner.clean(raw_text)
        print(f"  Output: {repr(cleaned[:80])}")
        summary = cleaner.get_cleaning_summary(raw_text, cleaned)
        print(f"  Summary: length {summary['original_length']} → "
              f"{summary['cleaned_length']} chars")

    # test with actual interpretation from dataset
    print("\n--- Real data test ---")
    real_text = (
        'He says \u201ccatastrophe\u201d before anything bad happens. '
        'Just\u2026 think about that. The creature opened its eyes. '
        'That\u2019s it. Victor\u2019s already calling it disaster.\n\n'
        '\u201cBeautiful!\u2014Great God!\u201d Right next to each other. '
        'His brain\u2019s breaking.'
    )
    cleaned = cleaner.clean(real_text)
    print(f"  Input length:  {len(real_text)} chars")
    print(f"  Output length: {len(cleaned)} chars")
    print(f"  Cleaned text:\n  {cleaned}")

    print("\n✓ TextCleaner tests complete")
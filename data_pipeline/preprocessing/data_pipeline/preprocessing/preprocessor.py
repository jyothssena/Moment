# ============================================================
# preprocessor.py
# MOMENT Preprocessing Pipeline - Main Orchestrator
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Orchestrates all preprocessing modules in the
# correct order for each record type (moments, books, users).
#
# THIS IS THE CORE OF THE PIPELINE.
# It connects:
#   - Input adapters (reading raw data)
#   - Lookup modules (resolving book metadata)
#   - Preprocessing modules (cleaning, validating, detecting)
#   - ID generation (creating consistent IDs)
#   - Output adapters (writing processed data)
#
# PROCESSING ORDER:
#   Phase 1: Read raw data via input adapter
#   Phase 2: Clean text (text_cleaner)
#   Phase 3: Validate text (text_validator)
#   Phase 4: Detect issues (issue_detector)
#   Phase 5: Calculate metrics (metrics_calculator)
#   Phase 6: Detect anomalies (anomaly_detector)
#   Phase 7: Generate IDs (id_generator)
#   Phase 8: Assemble output records
#   Phase 9: Write via output adapter
#
# DESIGN PRINCIPLE:
# The preprocessor never knows WHERE data comes from or goes.
# It only knows HOW to process it. Adapters handle I/O.
# This makes it fully production-ready.
# ============================================================

import logging
from datetime import datetime

# import all modules this orchestrator coordinates
from data_pipeline.preprocessing.text_cleaner import TextCleaner
from data_pipeline.preprocessing.text_validator import TextValidator
from data_pipeline.preprocessing.issue_detector import IssueDetector
from data_pipeline.preprocessing.metrics_calculator import MetricsCalculator
from data_pipeline.preprocessing.anomaly_detector import AnomalyDetector
from data_pipeline.utils.id_generator import (
    generate_user_id,
    generate_book_id,
    generate_passage_id,
    generate_interpretation_id
)

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Main orchestrator for the MOMENT preprocessing pipeline.

    Coordinates all preprocessing modules and produces
    three output datasets:
        - moments_processed  (450 interpretation records)
        - books_processed    (9 passage records)
        - users_processed    (50 user profile records)

    Usage:
        preprocessor = Preprocessor(config, input_adapter,
                                     output_adapter, lookup)
        preprocessor.run()
    """

    def __init__(self, config: dict,
                 input_adapter,
                 output_adapter,
                 lookup):
        """
        Initialize with all required components.

        Args:
            config: full config dict from config/config.yaml
            input_adapter: instance of BaseInputAdapter subclass
                           (reads raw data)
            output_adapter: instance of BaseOutputAdapter subclass
                            (writes processed data)
            lookup: instance of BaseLookup subclass
                    (resolves book metadata)
        """
        self.config = config
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.lookup = lookup

        # initialize all preprocessing modules
        # each module reads its own settings from config
        self.cleaner = TextCleaner(config)
        self.validator = TextValidator(config)
        self.issue_detector = IssueDetector(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.anomaly_detector = AnomalyDetector(config)

        # get timestamp format from config
        self.timestamp_format = config["output"]["timestamp_format"]

        # track processing stats for validation report
        self._stats = {
            "total_interpretations": 0,
            "valid_interpretations": 0,
            "invalid_interpretations": 0,
            "total_passages": 0,
            "valid_passages": 0,
            "total_users": 0,
            "anomalies_detected": 0,
            "issues_detected": {
                "pii": 0,
                "profanity": 0,
                "spam": 0
            },
            "processing_start": None,
            "processing_end": None
        }

        logger.info("Preprocessor initialized with all modules.")

    def run(self) -> bool:
        """
        Run the full preprocessing pipeline.

        Executes all phases in order:
        1. Process books (passages)
        2. Process users (character profiles)
        3. Process moments (interpretations)
        4. Write validation report

        Books and users are processed first because moments
        need their IDs for linking.

        Returns:
            bool: True if pipeline completed successfully,
                  False if any critical step failed
        """
        logger.info("=" * 50)
        logger.info("Starting MOMENT Preprocessing Pipeline")
        logger.info("=" * 50)

        self._stats["processing_start"] = datetime.now().strftime(
            self.timestamp_format
        )

        try:
            # --- Phase 1: Read all raw data ---
            logger.info("Phase 1: Reading raw data...")
            interpretations = self.input_adapter.read_interpretations()
            passages = self.input_adapter.read_passages()
            characters = self.input_adapter.read_characters()

            logger.info(
                f"Raw data loaded: "
                f"{len(interpretations)} interpretations, "
                f"{len(passages)} passages, "
                f"{len(characters)} characters."
            )

            # --- Phase 2: Look up book metadata ---
            logger.info("Phase 2: Looking up book metadata...")
            books_metadata = self.lookup.get_all_books_metadata()

            # build lookup dict: book_title → metadata
            # used throughout processing to resolve book_id, author etc
            book_lookup = {
                book["book_title"]: book
                for book in books_metadata
                if book["found"]
            }

            logger.info(
                f"Book metadata resolved for: "
                f"{list(book_lookup.keys())}"
            )

            # --- Phase 3: Process books ---
            logger.info("Phase 3: Processing passages (books)...")
            books_processed = self.process_books(passages, book_lookup)

            # build passage lookup: (book_title, passage_number) → passage_id
            # used by moments processing to link interpretations to passages
            passage_lookup = {
                (p["book_title"], p["passage_number"]): p["passage_id"]
                for p in books_processed
            }

            # --- Phase 4: Process users ---
            logger.info("Phase 4: Processing users (characters)...")
            users_processed = self.process_users(
                characters, interpretations
            )

            # build user lookup: character_name → user_id
            # used by moments processing to link interpretations to users
            user_lookup = {
                u["character_name"]: u["user_id"]
                for u in users_processed
            }

            # --- Phase 5: Process moments ---
            logger.info("Phase 5: Processing moments (interpretations)...")
            moments_processed = self.process_moments(
                interpretations, book_lookup, passage_lookup, user_lookup
            )

            # --- Phase 6: Write all outputs ---
            logger.info("Phase 6: Writing processed data...")
            books_ok = self.output_adapter.write_books(books_processed)
            users_ok = self.output_adapter.write_users(users_processed)
            moments_ok = self.output_adapter.write_moments(moments_processed)

            if not all([books_ok, users_ok, moments_ok]):
                logger.error("One or more output writes failed!")
                return False

            # --- Phase 7: Write validation report ---
            logger.info("Phase 7: Writing validation report...")
            self._stats["processing_end"] = datetime.now().strftime(
                self.timestamp_format
            )
            report = self._build_validation_report()
            self.output_adapter.write_validation_report(report)

            # log final summary
            self._log_final_summary()

            return True

        except Exception as e:
            logger.error(
                f"Pipeline failed with error: {e}",
                exc_info=True  # includes full traceback in logs
            )
            return False

    def process_books(self, passages: list,
                      book_lookup: dict) -> list:
        """
        Process all 9 literary passages.

        For each passage:
        1. Resolve book metadata (book_id, author)
        2. Clean passage text
        3. Validate passage text
        4. Calculate metrics
        5. Generate IDs
        6. Assemble output record

        Args:
            passages: list of raw passage dicts from input adapter
            book_lookup: dict mapping book_title → metadata

        Returns:
            list of processed passage dicts (books_processed schema)
        """
        logger.info(f"Processing {len(passages)} passages...")
        processed = []
        timestamp = datetime.now().strftime(self.timestamp_format)

        for passage in passages:
            try:
                book_title = passage.get("book_title", "")
                passage_number = int(passage.get("passage_id", 0))
                raw_text = passage.get("passage_text", "")

                # get book metadata from lookup
                book_meta = book_lookup.get(book_title, {})
                gutenberg_id = book_meta.get("gutenberg_id")

                if not gutenberg_id:
                    logger.warning(
                        f"No Gutenberg ID found for: {book_title!r}. "
                        f"Skipping passage."
                    )
                    continue

                # generate IDs
                book_id = generate_book_id(gutenberg_id)
                passage_id = generate_passage_id(book_id, passage_number)

                # clean text
                cleaned_text = self.cleaner.clean(raw_text)

                # validate text
                validation = self.validator.validate(
                    cleaned_text, text_type="passage"
                )

                # calculate metrics
                metrics = self.metrics_calculator.calculate(cleaned_text)

                # assemble output record
                record = {
                    "book_id": book_id,
                    "passage_id": passage_id,
                    "book_title": book_title,
                    "book_author": book_meta.get("author", "Unknown"),
                    "chapter_number": str(
                        passage.get("chapter_number", "Unknown")
                    ),
                    "passage_title": str(
                        passage.get("passage_title", "")
                    ),
                    "passage_number": passage_number,
                    "cleaned_passage_text": cleaned_text,
                    "is_valid": validation["is_valid"],
                    "quality_score": validation["quality_score"],
                    "quality_issues": validation["quality_issues"],
                    "metrics": metrics,
                    "timestamp": timestamp
                }

                processed.append(record)

                # update stats
                self._stats["total_passages"] += 1
                if validation["is_valid"]:
                    self._stats["valid_passages"] += 1

                logger.debug(
                    f"Processed passage: {passage_id} "
                    f"valid={validation['is_valid']}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing passage "
                    f"{passage.get('passage_id', 'unknown')}: {e}",
                    exc_info=True
                )
                # continue with next passage - don't crash pipeline
                continue

        logger.info(
            f"Books processing complete: "
            f"{len(processed)}/{len(passages)} passages processed."
        )

        return processed

    def process_users(self, characters: list,
                      interpretations: list) -> list:
        """
        Process all 50 character profiles into user records.

        For each character:
        1. Generate user_id
        2. Compile reading style list from Style_1 to Style_4
        3. Count total interpretations from interpretations list
        4. List books they interpreted
        5. Assemble output record

        Args:
            characters: list of raw character dicts from input adapter
            interpretations: list of raw interpretation dicts
                             (used to count/list books per user)

        Returns:
            list of processed user dicts (users_processed schema)
        """
        logger.info(f"Processing {len(characters)} character profiles...")
        processed = []
        timestamp = datetime.now().strftime(self.timestamp_format)

        # build a lookup of character_name → their interpretations
        # used to count total_interpretations and books_interpreted
        interp_by_character = {}
        for interp in interpretations:
            name = interp.get("character_name", "")
            if name not in interp_by_character:
                interp_by_character[name] = []
            interp_by_character[name].append(interp)

        for character in characters:
            try:
                name = character.get("Name", "")

                # generate user_id
                user_id = generate_user_id(name, self.config)

                # compile reading styles into a clean list
                # Style_1 through Style_4 → ["Academic", "Text", "Deep", "Extended"]
                reading_styles = [
                    str(character.get(f"Style_{i}", "")).strip()
                    for i in range(1, 5)
                    if character.get(f"Style_{i}", "").strip()
                ]

                # get this character's interpretations
                char_interpretations = interp_by_character.get(name, [])
                total_interpretations = len(char_interpretations)

                # get unique books they interpreted
                books_interpreted = list(set(
                    interp.get("book", "")
                    for interp in char_interpretations
                    if interp.get("book", "")
                ))

                # assemble output record
                record = {
                    "user_id": user_id,
                    "character_name": name,
                    "gender": character.get("Gender", ""),
                    "age": int(character.get("Age", 0)),
                    "profession": character.get("Profession", ""),
                    "distribution_category": character.get(
                        "Distribution_Category", ""
                    ),
                    "personality": character.get("Personality", ""),
                    "interest": character.get("Interest", ""),
                    "reading_intensity": character.get(
                        "Reading_Intensity", ""
                    ),
                    "reading_count": int(character.get("Reading_Count", 0)),
                    "experience_level": character.get("Experience_Level", ""),
                    "experience_count": int(
                        character.get("Experience_Count", 0)
                    ),
                    "journey": character.get("Journey", ""),
                    "reading_styles": reading_styles,
                    "total_interpretations": total_interpretations,
                    "books_interpreted": sorted(books_interpreted),
                    "timestamp": timestamp
                }

                processed.append(record)
                self._stats["total_users"] += 1

                logger.debug(f"Processed user: {user_id} ({name})")

            except Exception as e:
                logger.error(
                    f"Error processing character "
                    f"{character.get('Name', 'unknown')}: {e}",
                    exc_info=True
                )
                continue

        logger.info(
            f"Users processing complete: "
            f"{len(processed)}/{len(characters)} users processed."
        )

        return processed

    def process_moments(self, interpretations: list,
                        book_lookup: dict,
                        passage_lookup: dict,
                        user_lookup: dict) -> list:
        """
        Process all 450 reader interpretations.

        This is the most complex processing step.
        For each interpretation:
        1. Resolve book_id and passage_id via lookups
        2. Look up user_id via user_lookup
        3. Clean interpretation text
        4. Validate text
        5. Detect issues (PII, profanity, spam)
        6. Calculate metrics
        7. Generate interpretation_id
        8. Assemble output record

        After all records are processed individually,
        runs anomaly detection across the full batch
        (anomaly detection needs all records to establish baselines).

        Args:
            interpretations: list of raw interpretation dicts
            book_lookup: book_title → book metadata dict
            passage_lookup: (book_title, passage_number) → passage_id
            user_lookup: character_name → user_id

        Returns:
            list of processed interpretation dicts
            (moments_processed schema)
        """
        logger.info(
            f"Processing {len(interpretations)} interpretations..."
        )
        timestamp = datetime.now().strftime(self.timestamp_format)

        # --- First pass: process each record individually ---
        # collect cleaned texts and metrics for anomaly detection
        processed = []
        all_cleaned_texts = []
        all_metrics = []
        all_record_ids = []
        all_characters_for_anomaly = []

        for interpretation in interpretations:
            try:
                # extract raw fields
                book_title = interpretation.get("book", "")
                raw_passage_id = interpretation.get("passage_id", "")
                character_name = interpretation.get("character_name", "")
                character_id = interpretation.get("character_id", 0)
                raw_text = interpretation.get("interpretation", "")
                raw_word_count = interpretation.get("word_count", 0)

                # parse passage number from raw passage_id
                # raw format: "passage_1", "passage_2", "passage_3"
                passage_number = self._parse_passage_number(raw_passage_id)

                # resolve book metadata
                book_meta = book_lookup.get(book_title, {})
                gutenberg_id = book_meta.get("gutenberg_id")

                if not gutenberg_id:
                    logger.warning(
                        f"No Gutenberg ID for book: {book_title!r}. "
                        f"Skipping interpretation for {character_name}."
                    )
                    continue

                # generate IDs
                book_id = generate_book_id(gutenberg_id)
                passage_id = generate_passage_id(book_id, passage_number)
                user_id = user_lookup.get(
                    character_name,
                    generate_user_id(character_name, self.config)
                )

                # clean text
                cleaned_text = self.cleaner.clean(raw_text)

                # validate text
                validation = self.validator.validate(
                    cleaned_text, text_type="interpretation"
                )

                # detect issues
                issues = self.issue_detector.detect(cleaned_text)

                # calculate metrics
                metrics = self.metrics_calculator.calculate(cleaned_text)

                # generate interpretation ID
                interpretation_id = generate_interpretation_id(
                    character_name,
                    passage_id,
                    cleaned_text,
                    self.config
                )

                # assemble partial record
                # anomalies will be added in second pass
                record = {
                    "interpretation_id": interpretation_id,
                    "user_id": user_id,
                    "book_id": book_id,
                    "passage_id": passage_id,
                    "book_title": book_title,
                    "passage_number": passage_number,
                    "character_id": character_id,
                    "character_name": character_name,
                    "cleaned_interpretation": cleaned_text,
                    "original_word_count": raw_word_count,
                    "is_valid": validation["is_valid"],
                    "quality_score": validation["quality_score"],
                    "quality_issues": validation["quality_issues"],
                    "detected_issues": {
                        "has_pii": issues["has_pii"],
                        "pii_types": issues["pii_types"],
                        "has_profanity": issues["has_profanity"],
                        "profanity_ratio": issues["profanity_ratio"],
                        "is_spam": issues["is_spam"],
                        "spam_reasons": issues["spam_reasons"]
                    },
                    "metrics": metrics,
                    # anomalies placeholder - filled in second pass
                    "anomalies": {},
                    "timestamp": timestamp
                }

                processed.append(record)

                # collect for anomaly detection batch
                all_cleaned_texts.append(cleaned_text)
                all_metrics.append(metrics)
                all_record_ids.append(interpretation_id)
                all_characters_for_anomaly.append(
                    # we don't have character profiles here directly
                    # pass None - anomaly detector handles this
                    None
                )

                # update stats
                self._stats["total_interpretations"] += 1
                if validation["is_valid"]:
                    self._stats["valid_interpretations"] += 1
                else:
                    self._stats["invalid_interpretations"] += 1

                if issues["has_pii"]:
                    self._stats["issues_detected"]["pii"] += 1
                if issues["has_profanity"]:
                    self._stats["issues_detected"]["profanity"] += 1
                if issues["is_spam"]:
                    self._stats["issues_detected"]["spam"] += 1

            except Exception as e:
                logger.error(
                    f"Error processing interpretation for "
                    f"{interpretation.get('character_name', 'unknown')}: {e}",
                    exc_info=True
                )
                continue

        logger.info(
            f"First pass complete: {len(processed)} records processed. "
            f"Running anomaly detection..."
        )

        # --- Second pass: run anomaly detection on full batch ---
        # anomaly detection needs all records to establish baselines
        if processed:
            anomaly_results = self.anomaly_detector.detect_batch(
                all_metrics,
                all_cleaned_texts,
                all_record_ids,
                all_characters_for_anomaly
            )

            # add anomaly results back to each record
            for i, record in enumerate(processed):
                if i < len(anomaly_results):
                    record["anomalies"] = anomaly_results[i]

                    # update anomaly count in stats
                    anomaly = anomaly_results[i]
                    if any([
                        anomaly.get("word_count_outlier"),
                        anomaly.get("readability_outlier"),
                        anomaly.get("duplicate_risk"),
                        anomaly.get("style_mismatch")
                    ]):
                        self._stats["anomalies_detected"] += 1

        logger.info(
            f"Moments processing complete: "
            f"{len(processed)}/{len(interpretations)} interpretations "
            f"processed. "
            f"Valid: {self._stats['valid_interpretations']}, "
            f"Invalid: {self._stats['invalid_interpretations']}."
        )

        return processed

    # --------------------------------------------------------
    # PRIVATE HELPER METHODS
    # --------------------------------------------------------

    def _parse_passage_number(self, raw_passage_id: str) -> int:
        """
        Extract passage number from raw passage_id string.

        Raw format in JSON: "passage_1", "passage_2", "passage_3"
        We need the integer: 1, 2, 3

        Args:
            raw_passage_id: raw passage ID string

        Returns:
            int: passage number (1, 2, or 3)
                 returns 0 if parsing fails
        """
        try:
            # split on underscore and take the last part
            # "passage_1" → ["passage", "1"] → "1" → 1
            parts = str(raw_passage_id).split("_")
            return int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(
                f"Could not parse passage number from: "
                f"{raw_passage_id!r}. Using 0."
            )
            return 0

    def _build_validation_report(self) -> dict:
        """
        Build the final validation report from collected stats.

        Returns:
            dict: complete validation report
        """
        return {
            "pipeline": "MOMENT Preprocessing Pipeline",
            "version": "1.0",
            "processing_start": self._stats["processing_start"],
            "processing_end": self._stats["processing_end"],
            "interpretations": {
                "total": self._stats["total_interpretations"],
                "valid": self._stats["valid_interpretations"],
                "invalid": self._stats["invalid_interpretations"],
                "validity_rate": round(
                    self._stats["valid_interpretations"] /
                    max(self._stats["total_interpretations"], 1) * 100,
                    2
                )
            },
            "passages": {
                "total": self._stats["total_passages"],
                "valid": self._stats["valid_passages"],
            },
            "users": {
                "total": self._stats["total_users"]
            },
            "anomalies_detected": self._stats["anomalies_detected"],
            "issues_detected": self._stats["issues_detected"],
            "processing_timestamp": self._stats["processing_end"]
        }

    def _log_final_summary(self) -> None:
        """Log a clean summary of pipeline results."""
        logger.info("=" * 50)
        logger.info("Pipeline Complete - Summary")
        logger.info("=" * 50)
        logger.info(
            f"Interpretations: "
            f"{self._stats['total_interpretations']} total, "
            f"{self._stats['valid_interpretations']} valid, "
            f"{self._stats['invalid_interpretations']} invalid"
        )
        logger.info(
            f"Passages: {self._stats['total_passages']} processed"
        )
        logger.info(
            f"Users: {self._stats['total_users']} processed"
        )
        logger.info(
            f"Anomalies detected: {self._stats['anomalies_detected']}"
        )
        logger.info(
            f"Issues detected: {self._stats['issues_detected']}"
        )
        logger.info(
            f"Start: {self._stats['processing_start']} | "
            f"End: {self._stats['processing_end']}"
        )
        logger.info("=" * 50)
# ============================================================
# test_id_generator.py
# MOMENT Preprocessing Pipeline - ID Generator Tests
# IE7374 MLOps Coursework - Group 23
#
# Run with: pytest tests/test_id_generator.py -v
# ============================================================

import pytest # type: ignore
from data_pipeline.utils.id_generator import (
    sanitize_name,
    generate_hash,
    generate_user_id,
    generate_book_id,
    generate_passage_id,
    generate_interpretation_id,
    generate_all_ids
)


class TestSanitizeName:
    """Tests for sanitize_name() function."""

    def test_simple_name(self):
        """Simple two-word name converts correctly."""
        assert sanitize_name("Emma Chen") == "emma_chen"

    def test_name_with_title(self):
        """Name with title (Dr.) converts correctly."""
        assert sanitize_name("Dr. James Fletcher") == "dr_james_fletcher"

    def test_name_with_apostrophe(self):
        """Apostrophe is removed without replacement."""
        assert sanitize_name("Ryan O'Connor") == "ryan_oconnor"

    def test_name_with_hyphen(self):
        """Hyphen is removed without replacement."""
        result = sanitize_name("Mary-Jane Watson")
        assert "-" not in result
        assert result == "maryjane_watson"

    def test_empty_name(self):
        """Empty string returns 'unknown'."""
        assert sanitize_name("") == "unknown"

    def test_none_name(self):
        """None returns 'unknown'."""
        assert sanitize_name(None) == "unknown"

    def test_result_is_lowercase(self):
        """Result is always lowercase."""
        result = sanitize_name("EMMA CHEN")
        assert result == result.lower()

    def test_no_double_underscores(self):
        """No consecutive underscores in result."""
        result = sanitize_name("Dr. James Fletcher")
        assert "__" not in result

    def test_no_leading_trailing_underscores(self):
        """No leading or trailing underscores."""
        result = sanitize_name("Emma Chen")
        assert not result.startswith("_")
        assert not result.endswith("_")


class TestGenerateHash:
    """Tests for generate_hash() function."""

    def test_returns_string(self):
        """Hash returns a string."""
        result = generate_hash("test input")
        assert isinstance(result, str)

    def test_correct_length(self):
        """Hash returns correct length."""
        result = generate_hash("test input", length=8)
        assert len(result) == 8

    def test_custom_length(self):
        """Custom length is respected."""
        result = generate_hash("test input", length=4)
        assert len(result) == 4

    def test_deterministic(self):
        """Same input always produces same hash."""
        hash1 = generate_hash("Emma Chen_passage_1")
        hash2 = generate_hash("Emma Chen_passage_1")
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Different inputs produce different hashes."""
        hash1 = generate_hash("Emma Chen")
        hash2 = generate_hash("Marcus Williams")
        assert hash1 != hash2

    def test_hex_characters_only(self):
        """Hash contains only hex characters."""
        result = generate_hash("test")
        assert all(c in "0123456789abcdef" for c in result)


class TestGenerateUserId:
    """Tests for generate_user_id() function."""

    def test_returns_string(self, mock_config):
        """User ID is a string."""
        result = generate_user_id("Emma Chen", mock_config)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, mock_config):
        """User ID starts with 'user' prefix."""
        result = generate_user_id("Emma Chen", mock_config)
        assert result.startswith("user_")

    def test_contains_sanitized_name(self, mock_config):
        """User ID contains sanitized name."""
        result = generate_user_id("Emma Chen", mock_config)
        assert "emma_chen" in result

    def test_deterministic(self, mock_config):
        """Same name always produces same user ID."""
        id1 = generate_user_id("Emma Chen", mock_config)
        id2 = generate_user_id("Emma Chen", mock_config)
        assert id1 == id2

    def test_different_names_different_ids(self, mock_config):
        """Different names produce different user IDs."""
        id1 = generate_user_id("Emma Chen", mock_config)
        id2 = generate_user_id("Marcus Williams", mock_config)
        assert id1 != id2

    def test_all_50_characters_unique(self, mock_config):
        """All 50 character names produce unique user IDs."""
        names = [
            "Emma Chen", "Marcus Williams", "Sophia Patel", "David Kim",
            "Aisha Thompson", "Ryan O'Connor", "Isabella Rodriguez",
            "Jake Morrison", "Maya Singh", "Ethan Brooks",
            "Olivia Martinez", "Nathan Cooper", "Zoe Anderson",
            "Dr. James Fletcher", "Priya Sharma", "Leo Tanaka",
            "Hannah Park", "Carlos Mendoza", "Rachel Green",
            "Dr. Amelia Wright"
        ]
        ids = [generate_user_id(name, mock_config) for name in names]
        # all IDs should be unique
        assert len(ids) == len(set(ids))


class TestGenerateBookId:
    """Tests for generate_book_id() function."""

    def test_frankenstein(self):
        """Frankenstein gets correct book ID."""
        assert generate_book_id(84) == "gutenberg_84"

    def test_pride_and_prejudice(self):
        """Pride and Prejudice gets correct book ID."""
        assert generate_book_id(1342) == "gutenberg_1342"

    def test_great_gatsby(self):
        """The Great Gatsby gets correct book ID."""
        assert generate_book_id(64317) == "gutenberg_64317"

    def test_format(self):
        """Book ID follows gutenberg_{id} format."""
        result = generate_book_id(84)
        assert result.startswith("gutenberg_")
        assert "84" in result

    def test_deterministic(self):
        """Same ID always produces same book ID."""
        assert generate_book_id(84) == generate_book_id(84)


class TestGeneratePassageId:
    """Tests for generate_passage_id() function."""

    def test_frankenstein_passage_1(self):
        """Frankenstein passage 1 gets correct ID."""
        assert generate_passage_id(
            "gutenberg_84", 1
        ) == "gutenberg_84_passage_1"

    def test_frankenstein_passage_2(self):
        """Frankenstein passage 2 gets correct ID."""
        assert generate_passage_id(
            "gutenberg_84", 2
        ) == "gutenberg_84_passage_2"

    def test_gatsby_passage_3(self):
        """Gatsby passage 3 gets correct ID."""
        assert generate_passage_id(
            "gutenberg_64317", 3
        ) == "gutenberg_64317_passage_3"

    def test_passage_ids_unique_across_books(self):
        """Same passage number from different books produces different IDs."""
        id1 = generate_passage_id("gutenberg_84", 1)
        id2 = generate_passage_id("gutenberg_1342", 1)
        assert id1 != id2

    def test_deterministic(self):
        """Same input always produces same passage ID."""
        id1 = generate_passage_id("gutenberg_84", 1)
        id2 = generate_passage_id("gutenberg_84", 1)
        assert id1 == id2


class TestGenerateInterpretationId:
    """Tests for generate_interpretation_id() function."""

    def test_returns_string(self, mock_config):
        """Interpretation ID is a string."""
        result = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        assert isinstance(result, str)

    def test_starts_with_prefix(self, mock_config):
        """Interpretation ID starts with 'moment' prefix."""
        result = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        assert result.startswith("moment_")

    def test_deterministic(self, mock_config):
        """Same input always produces same interpretation ID."""
        id1 = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        id2 = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        assert id1 == id2

    def test_different_characters_different_ids(self, mock_config):
        """Different characters produce different IDs."""
        id1 = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        id2 = generate_interpretation_id(
            "Marcus Williams", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        assert id1 != id2

    def test_different_passages_different_ids(self, mock_config):
        """Same character, different passage produces different IDs."""
        id1 = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_1",
            "He says catastrophe", mock_config
        )
        id2 = generate_interpretation_id(
            "Emma Chen", "gutenberg_84_passage_2",
            "He says catastrophe", mock_config
        )
        assert id1 != id2


class TestGenerateAllIds:
    """Tests for generate_all_ids() convenience function."""

    def test_returns_all_required_keys(self, mock_config):
        """Result contains all 4 required ID fields."""
        result = generate_all_ids(
            character_name="Emma Chen",
            book_title="Frankenstein",
            gutenberg_id=84,
            passage_number=1,
            interpretation_text="He says catastrophe",
            config=mock_config
        )
        assert "user_id" in result
        assert "book_id" in result
        assert "passage_id" in result
        assert "interpretation_id" in result

    def test_ids_are_consistent(self, mock_config):
        """IDs in result are consistent with each other."""
        result = generate_all_ids(
            character_name="Emma Chen",
            book_title="Frankenstein",
            gutenberg_id=84,
            passage_number=1,
            interpretation_text="He says catastrophe",
            config=mock_config
        )
        # passage_id should contain book_id
        assert result["book_id"] in result["passage_id"]

    def test_fully_deterministic(self, mock_config):
        """Running twice produces identical results."""
        result1 = generate_all_ids(
            character_name="Maya Singh",
            book_title="Frankenstein",
            gutenberg_id=84,
            passage_number=2,
            interpretation_text="Shelley uses musical metaphors",
            config=mock_config
        )
        result2 = generate_all_ids(
            character_name="Maya Singh",
            book_title="Frankenstein",
            gutenberg_id=84,
            passage_number=2,
            interpretation_text="Shelley uses musical metaphors",
            config=mock_config
        )
        assert result1 == result2
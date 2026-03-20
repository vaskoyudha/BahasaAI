"""
Tests for BahasaDetector language detection.
TDD: Tests written first to define the contract.
"""

from bahasaai.core.detector import BahasaDetector
from bahasaai.core.types import Language, LanguageDetector


class TestBahasaDetectorProtocol:
    """Verify BahasaDetector satisfies the LanguageDetector protocol."""

    def test_implements_language_detector_protocol(self):
        """BahasaDetector must satisfy the LanguageDetector protocol."""
        detector = BahasaDetector()
        assert isinstance(detector, LanguageDetector)


class TestDetectIndonesian:
    """Tests for Indonesian language detection."""

    def test_detect_indonesian_pure(self):
        """Pure Indonesian text with multiple stop words → INDONESIAN."""
        detector = BahasaDetector()
        text = "Saya ingin belajar pemrograman Python untuk membuat aplikasi yang baik dan benar"
        result = detector.detect(text)
        assert result == Language.INDONESIAN

    def test_detect_single_word_indonesian(self):
        """A single Indonesian stop word → INDONESIAN."""
        detector = BahasaDetector()
        result = detector.detect("yang")
        assert result == Language.INDONESIAN

    def test_detect_case_insensitive(self):
        """Detection must be case-insensitive."""
        detector = BahasaDetector()
        result = detector.detect("YANG DAN ITU")
        assert result == Language.INDONESIAN


class TestDetectEnglish:
    """Tests for English language detection."""

    def test_detect_english_pure(self):
        """Pure English text with multiple stop words → ENGLISH."""
        detector = BahasaDetector()
        text = "I want to learn Python programming and build a great application that will work"
        result = detector.detect(text)
        assert result == Language.ENGLISH

    def test_detect_single_word_english(self):
        """A single English stop word → ENGLISH."""
        detector = BahasaDetector()
        result = detector.detect("the")
        assert result == Language.ENGLISH


class TestDetectMixed:
    """Tests for mixed language detection."""

    def test_detect_mixed(self):
        """Text with stop words from BOTH languages → MIXED."""
        detector = BahasaDetector()
        # "untuk", "dengan", "dan" → Indonesian; "the", "and", "for" → English
        text = "Saya untuk dengan dan the project and for testing"
        result = detector.detect(text)
        assert result == Language.MIXED

    def test_detect_mixed_code_switching(self):
        """Code-switching text with both language stop words → MIXED."""
        detector = BahasaDetector()
        # "mau", "kita" → Indonesian; "the", "for" → English
        text = "Saya mau implement the feature for application kita"
        result = detector.detect(text)
        assert result == Language.MIXED


class TestDetectUnknown:
    """Tests for unknown/undetectable input."""

    def test_detect_empty_string(self):
        """Empty string → UNKNOWN."""
        detector = BahasaDetector()
        result = detector.detect("")
        assert result == Language.UNKNOWN

    def test_detect_whitespace_only(self):
        """Whitespace-only string → UNKNOWN."""
        detector = BahasaDetector()
        result = detector.detect("   \t\n  ")
        assert result == Language.UNKNOWN

    def test_detect_numbers_only(self):
        """Numeric-only text → UNKNOWN."""
        detector = BahasaDetector()
        result = detector.detect("12345 67890")
        assert result == Language.UNKNOWN

    def test_detect_no_recognized_stop_words(self):
        """Text with no recognized stop words from either language → UNKNOWN."""
        detector = BahasaDetector()
        result = detector.detect("xyz abc qwerty lorem ipsum")
        assert result == Language.UNKNOWN


class TestCodeBlockStripping:
    """Tests for code block handling."""

    def test_detect_code_blocks_stripped(self):
        """Code blocks should be stripped; detect language from surrounding text."""
        detector = BahasaDetector()
        text = "Tolong jelaskan kode ini:\n```python\nprint('Hello World')\n```\nBagaimana cara kerjanya untuk"
        result = detector.detect(text)
        assert result == Language.INDONESIAN

    def test_detect_only_code(self):
        """Text that is ONLY a code block → UNKNOWN."""
        detector = BahasaDetector()
        text = "```python\nprint('hello')\n```"
        result = detector.detect(text)
        assert result == Language.UNKNOWN

    def test_detect_inline_code_stripped(self):
        """Inline code (backtick) should be stripped."""
        detector = BahasaDetector()
        text = "Gunakan `the` dan `for` untuk variabel yang baik"
        # After stripping inline code: "Gunakan  dan  untuk variabel yang baik"
        # "dan", "untuk", "yang" → Indonesian; no English stop words
        result = detector.detect(text)
        assert result == Language.INDONESIAN


class TestStopWordSets:
    """Tests for the stop word sets themselves."""

    def test_indonesian_stop_words_count(self):
        """Indonesian stop words set must have exactly 50 entries."""
        detector = BahasaDetector()
        assert len(detector.INDONESIAN_STOP_WORDS) == 50

    def test_english_stop_words_count(self):
        """English stop words set must have exactly 30 entries."""
        detector = BahasaDetector()
        assert len(detector.ENGLISH_STOP_WORDS) == 30

    def test_no_overlap_between_stop_word_sets(self):
        """Indonesian and English stop words must not overlap."""
        detector = BahasaDetector()
        overlap = detector.INDONESIAN_STOP_WORDS & detector.ENGLISH_STOP_WORDS
        assert len(overlap) == 0, f"Overlapping words: {overlap}"

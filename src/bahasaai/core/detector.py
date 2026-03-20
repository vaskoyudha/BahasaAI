"""
Language detector for BahasaAI.

Uses stop-word frequency analysis to classify text as Indonesian, English,
Mixed, or Unknown. No external NLP libraries required.
"""

from __future__ import annotations

import re

from bahasaai.core.types import Language


class BahasaDetector:
    """Stop-word based language detector for Indonesian and English.

    Implements the LanguageDetector protocol. Classifies text by counting
    recognized stop words from each language and computing ratios.
    """

    # Indonesian stop words (top 50)
    INDONESIAN_STOP_WORDS: frozenset[str] = frozenset(
        {
            "yang",
            "dan",
            "di",
            "ini",
            "itu",
            "dengan",
            "untuk",
            "pada",
            "tidak",
            "dari",
            "adalah",
            "akan",
            "ke",
            "juga",
            "sudah",
            "bisa",
            "ada",
            "atau",
            "saya",
            "mereka",
            "kami",
            "kita",
            "dia",
            "anda",
            "telah",
            "oleh",
            "masih",
            "harus",
            "karena",
            "seperti",
            "banyak",
            "sangat",
            "lalu",
            "tapi",
            "kalau",
            "mau",
            "baru",
            "lebih",
            "belum",
            "hanya",
            "semua",
            "jadi",
            "bila",
            "bukan",
            "begitu",
            "agar",
            "punya",
            "maka",
            "boleh",
            "sedang",
        }
    )

    # English stop words (top 30)
    ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
        {
            "the",
            "is",
            "at",
            "which",
            "on",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "with",
            "for",
            "this",
            "that",
            "from",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "will",
            "would",
            "could",
            "should",
            "not",
            "can",
            "do",
            "does",
        }
    )

    # Regex patterns for code block stripping
    _FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")

    def detect(self, text: str) -> Language:
        """Detect the language of the given text.

        Args:
            text: The text to analyze.

        Returns:
            The detected Language.
        """
        # 1. Empty or whitespace-only → UNKNOWN
        if not text or not text.strip():
            return Language.UNKNOWN

        # 2. Strip code blocks (fenced first, then inline)
        cleaned = self._FENCED_CODE_RE.sub(" ", text)
        cleaned = self._INLINE_CODE_RE.sub(" ", cleaned)

        # 3. Tokenize: lowercase, split on non-alphanumeric
        tokens = re.split(r"[^a-zA-Z0-9]+", cleaned.lower())

        # 4. Filter: keep only alphabetic words
        words = [t for t in tokens if t.isalpha()]

        # 5. No alphabetic words → UNKNOWN
        if not words:
            return Language.UNKNOWN

        # 6-7. Count stop word matches
        id_count = sum(1 for w in words if w in self.INDONESIAN_STOP_WORDS)
        en_count = sum(1 for w in words if w in self.ENGLISH_STOP_WORDS)

        # 8. Total matches
        total_matches = id_count + en_count

        # 9. No recognized stop words → UNKNOWN
        if total_matches == 0:
            return Language.UNKNOWN

        # 10-11. Compute ratios
        id_ratio = id_count / total_matches
        en_ratio = en_count / total_matches

        # 12-15. Classify based on ratios
        if id_ratio > 0.6:
            return Language.INDONESIAN
        if en_ratio > 0.6:
            return Language.ENGLISH

        # Both present, neither dominates → MIXED
        return Language.MIXED

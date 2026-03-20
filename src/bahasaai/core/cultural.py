"""
Cultural context detector and injector for BahasaAI.

Provides CulturalContextInjector implementing CulturalContextProvider protocol.
Detects Indonesian cultural references in text using a built-in dictionary (≤30 entries)
and generates context explanations. Unknown references are explained via LLM fallback.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import (  # pyright: ignore[reportMissingTypeStubs]
    CompletionResponse,
    Message,
    ProviderClient,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in cultural reference dictionary (≤30 entries)
# ---------------------------------------------------------------------------

CULTURAL_DICTIONARY: dict[str, str] = {
    "lebaran": "Idul Fitri, the Islamic celebration marking the end of Ramadan fasting month",
    "idul fitri": "Islamic celebration marking end of Ramadan fasting month",
    "natal": "Christmas, Christian holiday celebrating birth of Jesus",
    "nyepi": "Balinese Hindu New Year, day of silence and meditation",
    "waisak": "Buddhist holiday celebrating birth, enlightenment and death of Buddha",
    "isra miraj": "Islamic holiday commemorating Prophet Muhammad's night journey",
    "imlek": "Chinese New Year celebration widely observed in Indonesia",
    "rendang": "slow-cooked dry beef curry from West Sumatra, considered one of world's best dishes",
    "nasi goreng": "Indonesian fried rice, a national dish",
    "sate": "grilled skewered meat with peanut sauce",
    "gado-gado": "Indonesian salad with peanut sauce dressing",
    "bakso": "Indonesian meatball soup",
    "tempe": "fermented soybean cake, Indonesian superfood",
    "gotong royong": "communal cooperation and mutual assistance, core Indonesian social value",
    "musyawarah": "deliberation and consensus decision-making process",
    "silaturahmi": "maintaining family and social relationships, visiting relatives",
    "mudik": "annual homecoming migration during Lebaran, millions travel to hometown",
    "pak": "respectful title for older males, equivalent to Mr.",
    "bu": "respectful title for older females, equivalent to Mrs.",
    "mas": "Javanese informal address for younger males",
    "mbak": "Javanese informal address for younger females",
    "bps": "Badan Pusat Statistik, Indonesia's national statistics bureau",
    "ojk": "Otoritas Jasa Keuangan, Indonesia's financial services authority",
    "bi": "Bank Indonesia, the central bank of Indonesia",
    "rt": "Rukun Tetangga, smallest neighborhood administrative unit",
    "rw": "Rukun Warga, neighborhood unit above RT",
    "pkk": "Pemberdayaan Kesejahteraan Keluarga, family welfare organization",
    "arisan": "social gathering with rotating savings and credit association",
}


# ---------------------------------------------------------------------------
# Async provider cast (same pattern as enhancer.py)
# ---------------------------------------------------------------------------


class _AsyncProviderClient(Protocol):
    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> object: ...


# ---------------------------------------------------------------------------
# CulturalContextInjector
# ---------------------------------------------------------------------------


class CulturalContextInjector:
    """Detects and explains Indonesian cultural references in text.

    Implements the CulturalContextProvider protocol. Uses a built-in dictionary
    for known references and falls back to LLM for unknown ones.
    """

    def __init__(self, provider: ProviderClient, config: BahasaAIConfig) -> None:
        self._provider: _AsyncProviderClient = cast(_AsyncProviderClient, cast(object, provider))
        self._config = config
        # Sort keys longest-first so multi-word refs match before single-word substrings
        self._sorted_keys = sorted(CULTURAL_DICTIONARY.keys(), key=len, reverse=True)

    def detect_cultural_refs(self, text: str) -> list[str]:
        """Detect cultural references in text via case-insensitive dictionary scan.

        Args:
            text: The text to analyze.

        Returns:
            List of matched dictionary keys (lowercase), capped at max_entries.
        """
        if not text.strip():
            return []

        text_lower = text.lower()
        found: list[str] = []
        seen: set[str] = set()

        for key in self._sorted_keys:
            if key in text_lower and key not in seen:
                found.append(key)
                seen.add(key)

        cap = self._config.cultural_context_max_entries
        result = found[:cap]

        logger.debug(
            "cultural_detect text_len=%d refs_found=%d refs_capped=%d",
            len(text),
            len(found),
            len(result),
        )
        return result

    async def get_context(self, refs: list[str]) -> str:
        """Get cultural context explanations for references.

        Known refs use the built-in dictionary. Unknown refs are explained
        via a single LLM call.

        Args:
            refs: List of cultural reference keys.

        Returns:
            Formatted context string, or empty string if refs is empty.
        """
        if not refs:
            return ""

        cap = self._config.cultural_context_max_entries
        capped_refs = refs[:cap]

        parts: list[str] = []
        for ref in capped_refs:
            ref_lower = ref.lower()
            if ref_lower in CULTURAL_DICTIONARY:
                parts.append(f"{ref_lower} means {CULTURAL_DICTIONARY[ref_lower]}")
            else:
                # LLM fallback for unknown references
                definition = await self._explain_via_llm(ref_lower)
                parts.append(f"{ref_lower} means {definition}")

        result = "Cultural context: " + ". ".join(parts) + "."

        logger.debug(
            "cultural_context refs=%d result_len=%d",
            len(capped_refs),
            len(result),
        )
        return result

    async def _explain_via_llm(self, ref: str) -> str:
        """Ask the LLM to explain an unknown cultural reference in one sentence.

        Args:
            ref: The cultural reference to explain.

        Returns:
            A brief explanation string.
        """
        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert on Indonesian culture. "
                    "Explain the following cultural reference in one concise sentence."
                ),
            ),
            Message(
                role="user",
                content=f"What is '{ref}' in Indonesian culture?",
            ),
        ]

        response = await self._provider.complete(
            messages=messages,
            model=self._config.default_model,
            temperature=0.3,
            max_tokens=100,
        )
        # response is CompletionResponse — access .content
        content = cast(CompletionResponse, response).content
        return content.strip()

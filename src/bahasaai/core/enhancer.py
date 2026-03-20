from __future__ import annotations

import logging
import re
from typing import Protocol, cast

from bahasaai.core.types import (  # pyright: ignore[reportMissingTypeStubs]
    CompletionResponse,
    Language,
    Message,
    ProviderClient,
)

logger = logging.getLogger(__name__)


class _AsyncProviderClient(Protocol):
    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> object: ...


class AutoPromptEnhancer:
    def __init__(self, provider: ProviderClient) -> None:
        self._provider: _AsyncProviderClient = cast(_AsyncProviderClient, cast(object, provider))

    async def enhance(self, prompt: str, language: Language) -> str:
        if prompt.strip() == "":
            logger.info(
                "prompt_enhancer original=%r enhanced=%r was_enhanced=%s",
                prompt,
                "",
                False,
            )
            return ""

        score = self._vagueness_score(prompt)
        if score <= 0.6:
            logger.info(
                "prompt_enhancer original=%r enhanced=%r was_enhanced=%s",
                prompt,
                prompt,
                False,
            )
            return prompt

        system_prompt = (
            "You are a prompt engineer. Rewrite the following vague prompt into a specific, detailed "
            "prompt that will get better AI responses. Keep the same language "
            f"({language.value}). Keep the same intent. Add specificity, context, and structure. "
            "Return ONLY the enhanced prompt."
        )
        response = cast(
            CompletionResponse,
            await self._provider.complete(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=prompt),
                ],
                model="gpt-4o",
            ),
        )
        enhanced = response.content.strip()
        logger.info(
            "prompt_enhancer original=%r enhanced=%r was_enhanced=%s",
            prompt,
            enhanced,
            True,
        )
        return enhanced

    def _vagueness_score(self, prompt: str) -> float:
        text = prompt.strip()
        if text == "":
            return 0.0

        lowered = text.lower()
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        char_len = len(text)

        score = 0.5

        if char_len < 20:
            score += 0.35
        elif char_len < 40:
            score += 0.2
        elif char_len < 80:
            score += 0.05
        elif char_len > 240:
            score -= 0.15

        if word_count <= 1:
            score += 0.3
        elif word_count <= 3:
            score += 0.2
        elif word_count <= 7:
            score += 0.1
        elif word_count >= 50:
            score -= 0.3
        elif word_count >= 30:
            score -= 0.15

        has_number = bool(re.search(r"\b\d+(?:[.,]\d+)?\b", text))
        date_pattern = (
            r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|jan(?:uary|uari)?|feb(?:ruary|ruari)?|"
            + r"mar(?:ch|et)?|apr(?:il)?|mei|may|jun(?:e|i)?|jul(?:y|i)?|aug(?:ust|ustus)?|"
            + r"sep(?:tember)?|oct(?:ober)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?|des(?:ember)?)\b"
        )
        has_date = bool(
            re.search(
                date_pattern,
                lowered,
            )
        )
        has_name = bool(re.search(r"\b[A-Z][a-z]{2,}\b", text))
        structure_pattern = (
            r"\b(steps?|bullet|outline|timeline|kpi|metric|compare|analysis|langkah|"
            + r"poin|rencana|jadwal|target|metrik|analisis)\b"
        )
        has_structure = bool(
            re.search(
                structure_pattern,
                lowered,
            )
        )

        specificity_hits = sum([has_number, has_date, has_name, has_structure])
        score -= 0.12 * specificity_hits

        vague_patterns = [
            r"^\s*what\s+is\b",
            r"^\s*explain\b",
            r"^\s*tell\s+me\s+about\b",
            r"^\s*apa\s+itu\b",
            r"^\s*jelaskan\s+tentang\b",
            r"^\s*jelaskan\b",
            r"^\s*ceritakan\b",
        ]
        if any(re.search(pattern, lowered) for pattern in vague_patterns):
            score += 0.25

        if word_count >= 50 and specificity_hits >= 1:
            score = min(score, 0.35)

        return float(max(0.0, min(1.0, score)))

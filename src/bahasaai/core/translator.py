from __future__ import annotations

import logging
import re
from typing import Protocol, cast

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import CompletionResponse, Language, Message, ProviderClient

logger = logging.getLogger(__name__)


class _AsyncProviderClient(Protocol):
    async def complete(self, messages: list[Message], model: str, **kwargs: object) -> object: ...


class SemanticTranslator:
    def __init__(self, provider: ProviderClient, config: BahasaAIConfig) -> None:
        self._provider: _AsyncProviderClient = cast(_AsyncProviderClient, cast(object, provider))
        self._config: BahasaAIConfig = config

    async def translate(self, text: str, source: Language, target: Language) -> str:
        logger.info("translate start len=%d pair=%s->%s", len(text), source.value, target.value)
        if source == target:
            logger.info(
                "translate passthrough len=%d pair=%s->%s", len(text), source.value, target.value
            )
            return text

        prompt = (
            f"Translate the following {source.value} text to {target.value}. "
            "Maintain the original meaning, tone, and nuance. "
            "Do NOT translate proper nouns, technical terms, or brand names. "
            "Return ONLY the translation, no explanations.\n\n"
            f"{text}"
        )

        response = cast(
            CompletionResponse,
            await self._provider.complete(
                [Message(role="user", content=prompt)],
                model=self._config.default_model,
            ),
        )
        translated = response.content.strip()
        logger.info(
            "translate done in_len=%d out_len=%d pair=%s->%s",
            len(text),
            len(translated),
            source.value,
            target.value,
        )
        return translated

    async def translate_with_preserved_code(
        self, text: str, source: Language, target: Language
    ) -> str:
        if text == "":
            logger.info(
                "translate_with_preserved_code empty pair=%s->%s",
                source.value,
                target.value,
            )
            return ""

        logger.info(
            "translate_with_preserved_code start len=%d pair=%s->%s",
            len(text),
            source.value,
            target.value,
        )

        code_blocks: list[str] = []
        urls: list[str] = []

        def replace_code(match: re.Match[str]) -> str:
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"[CODE_BLOCK_{idx}]"

        def replace_url(match: re.Match[str]) -> str:
            idx = len(urls)
            urls.append(match.group(0))
            return f"[URL_{idx}]"

        without_fenced = re.sub(r"```[\s\S]*?```", replace_code, text)
        without_inline = re.sub(r"`[^`]+`", replace_code, without_fenced)
        placeholder_text = re.sub(r"https?://\S+", replace_url, without_inline)

        content_without_placeholders = re.sub(
            r"\[CODE_BLOCK_\d+\]|\[URL_\d+\]", "", placeholder_text
        )
        if content_without_placeholders.strip() == "":
            logger.info(
                "translate_with_preserved_code passthrough-only-preserved len=%d pair=%s->%s",
                len(text),
                source.value,
                target.value,
            )
            return text

        translated = await self.translate(placeholder_text, source, target)

        for idx, code_block in enumerate(code_blocks):
            translated = translated.replace(f"[CODE_BLOCK_{idx}]", code_block)
        for idx, url in enumerate(urls):
            translated = translated.replace(f"[URL_{idx}]", url)

        logger.info(
            "translate_with_preserved_code done in_len=%d out_len=%d pair=%s->%s",
            len(text),
            len(translated),
            source.value,
            target.value,
        )
        return translated

"""
Meta-instruction injector for English reasoning.

Forces LLMs to perform internal reasoning in English for maximum accuracy,
then deliver responses in the user's expected language. This is the core
innovation of BahasaAI.

Provides MetaInstructionInjector implementing the MetaInstructor protocol.
Pure string manipulation — no LLM calls.
"""

from __future__ import annotations

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import Message

REASONING_TEMPLATE = (
    "You are a highly capable AI assistant. "
    "IMPORTANT: Perform ALL reasoning, analysis, and thinking in English for maximum accuracy. "
    "After completing your reasoning in English, provide your final response in the language the user expects. "
    "Structure your thinking: "
    "1) Understand the request "
    "2) Reason through the problem in English "
    "3) Formulate your response "
    "4) Deliver in the expected language"
)

CULTURAL_TEMPLATE = "Consider the following cultural context: {context}"

MERGE_TEMPLATE = "{existing_content}\n\n{meta_instructions}"


class MetaInstructionInjector:
    """Injects meta-instructions for English reasoning into message lists.

    Implements the MetaInstructor protocol. Pure string manipulation,
    never mutates the original messages list or Message objects.
    """

    def __init__(self, config: BahasaAIConfig) -> None:
        """Initialize with BahasaAI configuration.

        Args:
            config: BahasaAI configuration instance.
        """
        self._config = config

    def inject(self, messages: list[Message], context: str | None = None) -> list[Message]:
        """Inject meta-instructions into a message list.

        - No system message: prepend a new system message with reasoning template.
        - Existing system message: merge meta-instructions into the first one.
        - Multiple system messages: merge into first, keep others unchanged.
        - Cultural context: append cultural template when context is provided.

        Returns a NEW list — never mutates the original.

        Args:
            messages: The original messages.
            context: Optional cultural context string.

        Returns:
            New message list with meta-instructions injected.
        """
        # Build the meta-instruction content
        meta_content = REASONING_TEMPLATE
        if context is not None:
            meta_content += "\n\n" + CULTURAL_TEMPLATE.format(context=context)

        # Find the first system message index
        first_system_idx: int | None = None
        for i, msg in enumerate(messages):
            if msg.role == "system":
                first_system_idx = i
                break

        if first_system_idx is None:
            # No system message — prepend one
            system_msg = Message(role="system", content=meta_content)
            return [system_msg, *messages]

        # System message exists — merge into the first one
        existing = messages[first_system_idx]
        merged_content = MERGE_TEMPLATE.format(
            existing_content=existing.content,
            meta_instructions=REASONING_TEMPLATE,
        )
        if context is not None:
            merged_content += "\n\n" + CULTURAL_TEMPLATE.format(context=context)

        merged_system = Message(role="system", content=merged_content)

        # Build new list: replace first system, copy everything else
        result: list[Message] = []
        for i, msg in enumerate(messages):
            if i == first_system_idx:
                result.append(merged_system)
            else:
                result.append(msg)

        return result

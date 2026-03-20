"""
Tests for MetaInstructionInjector — English reasoning injection.
TDD: Tests written first to define the contract.
"""

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.meta import (
    CULTURAL_TEMPLATE,
    MERGE_TEMPLATE,
    REASONING_TEMPLATE,
    MetaInstructionInjector,
)
from bahasaai.core.types import Message, MetaInstructor


class TestMetaInstructionInjectorProtocol:
    """Verify MetaInstructionInjector satisfies the MetaInstructor protocol."""

    def test_implements_meta_instructor_protocol(self):
        """MetaInstructionInjector must satisfy the MetaInstructor protocol."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        assert isinstance(injector, MetaInstructor)


class TestInjectPrependsSystemMessage:
    """Tests for prepending system message when none exists."""

    def test_inject_prepends_system_message(self):
        """When no system message exists, inject prepends one."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo, apa kabar?")]
        result = injector.inject(messages)
        assert result[0].role == "system"
        assert len(result) == 2
        assert result[1].role == "user"
        assert result[1].content == "Halo, apa kabar?"

    def test_inject_contains_english_instruction(self):
        """The injected system message must contain 'English'."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo")]
        result = injector.inject(messages)
        assert "English" in result[0].content


class TestImmutability:
    """Tests for immutability guarantees."""

    def test_no_mutation_of_original(self):
        """Original messages list must not be modified."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo")]
        original_len = len(messages)
        original_content = messages[0].content
        injector.inject(messages)
        assert len(messages) == original_len
        assert messages[0].content == original_content
        assert messages[0].role == "user"

    def test_returns_new_list(self):
        """Returned list must be a different object from input."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo")]
        result = injector.inject(messages)
        assert result is not messages


class TestEmptyMessages:
    """Tests for empty message list handling."""

    def test_empty_messages(self):
        """inject([]) should return a list with exactly 1 system message."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        result = injector.inject([])
        assert len(result) == 1
        assert result[0].role == "system"
        assert REASONING_TEMPLATE in result[0].content


class TestMergeExistingSystem:
    """Tests for merging with existing system messages."""

    def test_merge_existing_system(self):
        """Existing system message content is preserved and meta-instructions merged."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        original_system = "You are a helpful assistant for Indonesian users."
        messages = [
            Message(role="system", content=original_system),
            Message(role="user", content="Halo"),
        ]
        result = injector.inject(messages)
        # System message should contain BOTH original and meta-instructions
        assert original_system in result[0].content
        assert REASONING_TEMPLATE in result[0].content
        # Should use MERGE_TEMPLATE format
        expected = MERGE_TEMPLATE.format(
            existing_content=original_system,
            meta_instructions=REASONING_TEMPLATE,
        )
        assert result[0].content == expected
        # Should not add extra system messages
        assert len(result) == 2
        assert result[1].role == "user"

    def test_multiple_system_messages(self):
        """First system message is merged; second is kept as-is."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        first_system = "You are a translator."
        second_system = "Always be polite."
        messages = [
            Message(role="system", content=first_system),
            Message(role="system", content=second_system),
            Message(role="user", content="Halo"),
        ]
        result = injector.inject(messages)
        # First system merged
        assert first_system in result[0].content
        assert REASONING_TEMPLATE in result[0].content
        # Second system unchanged
        assert result[1].role == "system"
        assert result[1].content == second_system
        # User preserved
        assert result[2].role == "user"
        assert result[2].content == "Halo"
        assert len(result) == 3


class TestCulturalContext:
    """Tests for cultural context injection."""

    def test_context_injected(self):
        """When context is provided, it appears in the system message."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo")]
        context = "gotong royong = cooperation"
        result = injector.inject(messages, context=context)
        assert context in result[0].content
        assert CULTURAL_TEMPLATE.format(context=context) in result[0].content

    def test_no_context_no_cultural(self):
        """When context is None, cultural template is not in the output."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [Message(role="user", content="Halo")]
        result = injector.inject(messages, context=None)
        # The cultural template prefix should not appear
        assert "Consider the following cultural context" not in result[0].content

    def test_context_with_existing_system(self):
        """Cultural context merges correctly with existing system message."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        original_system = "You are a helpful assistant."
        messages = [
            Message(role="system", content=original_system),
            Message(role="user", content="Halo"),
        ]
        context = "gotong royong = cooperation"
        result = injector.inject(messages, context=context)
        assert original_system in result[0].content
        assert REASONING_TEMPLATE in result[0].content
        assert context in result[0].content


class TestUserMessagesPreserved:
    """Tests for user message preservation."""

    def test_user_messages_preserved(self):
        """All user messages are present in output, in order."""
        config = BahasaAIConfig()
        injector = MetaInstructionInjector(config)
        messages = [
            Message(role="user", content="Halo"),
            Message(role="assistant", content="Hai!"),
            Message(role="user", content="Apa kabar?"),
        ]
        result = injector.inject(messages)
        # System prepended
        assert result[0].role == "system"
        # All original messages preserved in order
        assert result[1].role == "user"
        assert result[1].content == "Halo"
        assert result[2].role == "assistant"
        assert result[2].content == "Hai!"
        assert result[3].role == "user"
        assert result[3].content == "Apa kabar?"
        assert len(result) == 4

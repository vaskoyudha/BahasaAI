from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from unittest.mock import AsyncMock

from bahasaai.core.enhancer import AutoPromptEnhancer
from bahasaai.core.types import CompletionResponse, Language


def _response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="gpt-4o",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class TestAutoPromptEnhancer:
    async def test_vague_prompt_enhanced(self) -> None:
        provider = AsyncMock()
        complete_mock = AsyncMock(
            return_value=_response("Jelaskan AI dalam 5 poin dengan contoh nyata.")
        )
        provider.complete = complete_mock
        enhancer = AutoPromptEnhancer(provider=provider)

        result = await enhancer.enhance("jelaskan AI", Language.INDONESIAN)

        assert result == "Jelaskan AI dalam 5 poin dengan contoh nyata."
        complete_mock.assert_awaited_once()

    async def test_specific_passthrough(self) -> None:
        provider = AsyncMock()
        complete_mock = AsyncMock(return_value=_response("ignored"))
        provider.complete = complete_mock
        enhancer = AutoPromptEnhancer(provider=provider)

        detailed_prompt = (
            "Please draft a migration plan for our warehouse analytics stack from PostgreSQL to BigQuery "
            "covering schema mapping, data validation checkpoints, rollback strategy, budget constraints, "
            "security controls, encryption at rest, IAM roles, timeline by week, owners per phase, "
            "risk register with mitigation actions, KPIs for success, monitoring dashboards, alert thresholds, "
            "backfill approach for three years of history, and a cutover checklist for production readiness "
            "with explicit acceptance criteria for data quality, latency, and cost."
        )

        result = await enhancer.enhance(detailed_prompt, Language.ENGLISH)

        assert result == detailed_prompt
        complete_mock.assert_not_called()

    async def test_empty_prompt_returns_empty(self) -> None:
        provider = AsyncMock()
        complete_mock = AsyncMock(return_value=_response("ignored"))
        provider.complete = complete_mock
        enhancer = AutoPromptEnhancer(provider=provider)

        result = await enhancer.enhance("", Language.ENGLISH)

        assert result == ""
        complete_mock.assert_not_called()

    async def test_vagueness_score_short(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        score = enhancer._vagueness_score("AI?")

        assert score > 0.6

    async def test_vagueness_score_long(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        detailed_prompt = (
            "Design a detailed onboarding curriculum for junior data analysts at a fintech startup, including "
            "weekly objectives, SQL competency milestones, dashboard design principles, stakeholder communication "
            "guidelines, compliance training for PCI and GDPR, mentoring cadence, graded assessments, capstone "
            "project scope, peer review process, and measurable outcomes tied to business reporting accuracy and "
            "decision support quality over the first ninety days."
        )
        score = enhancer._vagueness_score(detailed_prompt)

        assert score < 0.4

    async def test_score_returns_float(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        score = enhancer._vagueness_score(
            "Please summarize quarterly performance with three metrics."
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    async def test_vague_english_pattern(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        score = enhancer._vagueness_score("what is machine learning")

        assert score > 0.6

    async def test_vague_indonesian_pattern(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        score = enhancer._vagueness_score("jelaskan tentang ekonomi")

        assert score > 0.6

    async def test_single_word_vague(self) -> None:
        enhancer = AutoPromptEnhancer(provider=AsyncMock())

        score = enhancer._vagueness_score("ekonomi")

        assert score > 0.6

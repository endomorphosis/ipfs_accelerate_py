"""Offline acceptance tests for the voice-router GraphRAG boundary."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.voice_router import (  # noqa: E402
    DEFAULT_GROUNDED_FALLBACK,
    GraphRAGVoiceTemplateProvider,
    GroundedSlot,
    GroundingEvidence,
    VoiceResponsePlan,
    VoiceTurnRequest,
    process_voice_turn,
)
from ipfs_accelerate_py.voice_templates import (  # noqa: E402
    VoiceGroundingValidationError,
    VoiceTemplateValidationError,
    buildVoiceGraphRagPromptParts,
    build_voice_graphrag_prompt_parts,
    normalize_spoken_text,
)


@dataclass
class FakeSpeech:
    transcript: str = "I need food help"
    audio: bytes = b"RIFF-fake-abby"
    calls: list[tuple[str, str]] = field(default_factory=list)
    fail_tts: bool = False

    def transcribe(self, audio: object, **kwargs: object) -> str:
        self.calls.append(("transcribe", repr(audio)))
        return self.transcript

    def synthesize(self, text: str, **kwargs: object) -> bytes:
        self.calls.append(("synthesize", text))
        if self.fail_tts:
            raise TimeoutError("offline tts failure")
        return self.audio


def _evidence() -> GroundingEvidence:
    return GroundingEvidence(
        source_id="food-current",
        cid="bafy-food-current",
        uri="ipfs://bafy-food-current",
        text="Community Food Network is open today. Call 503-555-0111.",
        facts={
            "program": "Community Food Network",
            "phone": "503-555-0111",
        },
    )


def _plan() -> VoiceResponsePlan:
    return VoiceResponsePlan(
        template_id="food-help-v2",
        template=(
            "{program} can help. Call {phone}. "
            "[source](ipfs://bafy-food-current) [1]"
        ),
        slots=(
            GroundedSlot("program", "Community Food Network", ("food-current",)),
            GroundedSlot("phone", "503-555-0111", ("food-current",)),
        ),
        evidence=(_evidence(),),
        confidence=0.96,
        intent="food_assistance",
    )


@dataclass
class FakeTemplateProvider:
    plan: VoiceResponsePlan | None = field(default_factory=_plan)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider_name: str = "fake-graphrag"

    def retrieve(self, transcript: str, **kwargs: object) -> VoiceResponsePlan | None:
        self.calls.append({"transcript": transcript, **kwargs})
        if self.error:
            raise self.error
        return self.plan


def test_prompt_parts_are_canonical_and_do_not_mutate_caller_data() -> None:
    context = {"county": "Multnomah", "intent": "food_assistance"}
    grounding = {"source_cid": "bafy-food-current"}
    first = buildVoiceGraphRagPromptParts(
        "  I need food help  ",
        context=context,
        language=" en-US ",
        grounding=grounding,
        max_results=3,
    )
    second = build_voice_graphrag_prompt_parts(
        "I need food help",
        context={"intent": "food_assistance", "county": "Multnomah"},
        language="en-US",
        grounding=grounding,
        max_results=3,
    )

    assert first == second
    assert first["query"] == "I need food help"
    assert "response plan" in first["system"]
    assert context == {"county": "Multnomah", "intent": "food_assistance"}
    assert grounding == {"source_cid": "bafy-food-current"}
    json.dumps(first)


def test_spoken_normalization_removes_citations_but_keeps_human_text() -> None:
    assert normalize_spoken_text(
        "Call 211 [1]. [source](https://example.test/a) ipfs://bafy"
    ) == "Call 211."
    with pytest.raises(VoiceGroundingValidationError):
        normalize_spoken_text("[source](https://example.test/only)")


def test_full_router_turn_is_stt_retrieval_rendering_tts_and_provenance() -> None:
    speech = FakeSpeech()
    templates = FakeTemplateProvider()
    result = process_voice_turn(
        VoiceTurnRequest(
            audio=b"caller-audio",
            request_id="graphrag-turn-1",
            language="en-US",
            context={"county": "Multnomah"},
            grounding={"corpus_cid": "bafy-corpus"},
            output_format="wav",
        ),
        stt_provider=speech,
        template_provider=templates,
        tts_provider=speech,
    )

    assert result.status == "completed"
    assert result.response_text == (
        "Community Food Network can help. Call 503-555-0111."
    )
    assert result.audio == b"RIFF-fake-abby"
    assert [trace.stage for trace in result.traces] == [
        "transcription",
        "retrieval",
        "rendering",
        "synthesis",
    ]
    assert result.provenance.template_id == "food-help-v2"
    assert result.provenance.evidence[0].cid == "bafy-food-current"
    assert result.provenance.grounded_slots[1].source_ids == ("food-current",)
    assert templates.calls[0]["language"] == "en-US"
    assert speech.calls[-1] == (
        "synthesize",
        "Community Food Network can help. Call 503-555-0111.",
    )


def test_graph_rag_adapter_exposes_canonical_prompt_to_opt_in_backend() -> None:
    seen: dict[str, object] = {}

    def backend(
        transcript: str,
        *,
        prompt_parts: dict[str, object],
        context: dict[str, object],
        language: str | None,
        grounding: dict[str, object],
        max_results: int,
    ) -> dict[str, object]:
        seen.update(
            transcript=transcript,
            prompt_parts=prompt_parts,
            context=context,
            language=language,
            grounding=grounding,
            max_results=max_results,
        )
        return _plan().to_dict()

    provider = GraphRAGVoiceTemplateProvider(backend, minimum_confidence=0.9)
    plan = provider.retrieve(
        " food help ",
        context={"county": "Multnomah"},
        language="en-US",
        grounding={"source_cid": "bafy-corpus"},
        max_results=2,
    )

    assert plan is not None
    assert plan.template_id == "food-help-v2"
    assert seen["transcript"] == "food help"
    assert seen["prompt_parts"] == provider.last_prompt_parts
    assert seen["prompt_parts"]["max_results"] == 2  # type: ignore[index]


@pytest.mark.parametrize(
    "plan",
    [
        VoiceResponsePlan(
            template_id="unknown-source",
            template="Call {phone}.",
            slots=(GroundedSlot("phone", "503-555-0199", ("missing",)),),
            evidence=(_evidence(),),
        ),
        VoiceResponsePlan(
            template_id="missing-fact",
            template="Call {phone}.",
            slots=(GroundedSlot("phone", "503-555-0199", ("food-current",)),),
            evidence=(_evidence(),),
        ),
    ],
    ids=("unknown-source", "conflicting-current-fact"),
)
def test_unsafe_slots_fail_closed_to_deterministic_fallback(
    plan: VoiceResponsePlan,
) -> None:
    speech = FakeSpeech()
    result = process_voice_turn(
        VoiceTurnRequest(transcript="food help", request_id="unsafe-turn"),
        template_provider=FakeTemplateProvider(plan=plan),
        tts_provider=speech,
    )

    assert result.status == "degraded"
    assert result.response_text == DEFAULT_GROUNDED_FALLBACK
    assert "grounding_validation_failed" in result.fallback_reasons
    assert result.provenance.grounded_slots == ()
    assert speech.calls[-1][1] == DEFAULT_GROUNDED_FALLBACK


def test_retrieval_failure_is_recorded_and_fallback_is_synthesized() -> None:
    speech = FakeSpeech()
    result = process_voice_turn(
        VoiceTurnRequest(transcript="housing help", request_id="fallback-turn"),
        template_provider=FakeTemplateProvider(error=RuntimeError("backend down")),
        tts_provider=speech,
    )

    assert result.status == "degraded"
    assert result.response_text == DEFAULT_GROUNDED_FALLBACK
    assert result.fallback_reasons == ("template_retrieval_failed",)
    assert next(trace for trace in result.traces if trace.stage == "retrieval").status == "failed"


def test_tts_failure_returns_text_only_without_false_audio() -> None:
    speech = FakeSpeech(fail_tts=True)
    result = process_voice_turn(
        VoiceTurnRequest(transcript="food help", request_id="text-only-turn"),
        template_provider=FakeTemplateProvider(),
        tts_provider=speech,
    )

    assert result.status == "text_only"
    assert result.audio is None
    assert result.provenance.output_audio_sha256 is None
    assert "tts_failed" in result.fallback_reasons


def test_unsafe_template_expressions_are_rejected_before_rendering() -> None:
    with pytest.raises(VoiceTemplateValidationError):
        from ipfs_accelerate_py.voice_templates import template_fields

        template_fields("{service.name}")
    with pytest.raises(VoiceTemplateValidationError):
        from ipfs_accelerate_py.voice_templates import template_fields

        template_fields("{phone:>20}")

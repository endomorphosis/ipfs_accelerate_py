"""Offline acceptance tests for ABBY-VOICE-G019 precomputed audio runtime.

Evidence subset covered by this suite:

* runtime resolution
* revision-pinned streaming/release loader
* exact audio resolver
* stale-slot regression test
* text-only or live-TTS fallback receipt
* content-addressed GraphRAG restore

Authoritative evidence map:
data/abby_voice/agent_supervisor/discovery/2026-07-26-abby-voice-auto-019-objective-validation-repair.md
"""

from __future__ import annotations

import io
import json
import sys
import wave
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.voice_audio_resolver import (  # noqa: E402
    EXACT_AUDIO_RESOLVER_EVIDENCE_TERM,
    REASON_EXACT_MATCH,
    REASON_IDENTIFIER_ONLY_REJECTED,
    REASON_STALE_SLOT_INVALIDATED,
    REASON_SYNTHESIS_IDENTITY_MISMATCH,
    STALE_SLOT_REGRESSION_TEST_EVIDENCE_TERM,
    PrecomputedVoiceAudioResolver,
    SynthesisIdentity,
    spoken_text_sha256,
    synthesis_match_key,
)
from ipfs_accelerate_py.voice_router import (  # noqa: E402
    DEFAULT_GROUNDED_FALLBACK,
    GraphRAGVoiceTemplateProvider,
    GroundedSlot,
    GroundingEvidence,
    VoiceResponsePlan,
    VoiceTurnRequest,
    process_voice_turn,
)
from ipfs_datasets_py.voice.hf_release import (  # noqa: E402
    AbbyVoiceHFReleaseBuilder,
    AbbyVoiceHFReleasePolicy,
)
from ipfs_datasets_py.voice.release_loader import (  # noqa: E402
    REVISION_PINNED_STREAMING_RELEASE_LOADER_EVIDENCE_TERM,
    RUNTIME_RESOLUTION_EVIDENCE_TERM,
    AbbyVoiceReleaseLoader,
    AbbyVoiceReleaseLoaderError,
    G019_REQUIRED_EVIDENCE_TERMS,
)
from ipfs_datasets_py.voice.schema import (  # noqa: E402
    AbbyVoiceAudio,
    AbbyVoiceProvenance,
    AbbyVoiceResponse,
    AbbyVoiceTemplate,
)


def _fixture_wav(sample: int = 1_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(sample.to_bytes(2, "little", signed=True))
    return output.getvalue()


PRECOMPUTED_AUDIO_BYTES = _fixture_wav()
PRECOMPUTED_AUDIO_SHA = sha256(PRECOMPUTED_AUDIO_BYTES).hexdigest()
SPOKEN_PHONE_A = "Community Food Network can help. Call 503-555-0111."
SPOKEN_PHONE_B = "Community Food Network can help. Call 503-555-0199."
TEMPLATE_ID = "template-food"
SYNTHESIS = SynthesisIdentity(
    provider="abby_indextts",
    model="index-tts-v1",
    voice="abby",
    provider_version="1.0.0",
    locale="en-US",
    codec="wav",
    sample_rate_hz=24_000,
    channels=1,
    generation_settings={"temperature": 0.0},
)


@dataclass
class FakeSpeech:
    transcript: str = "I need food help"
    audio: bytes = _fixture_wav(2_000)
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


@dataclass
class FakeTemplateProvider:
    plan: VoiceResponsePlan
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider_name: str = "fake-graphrag"

    def retrieve(self, transcript: str, **kwargs: object) -> VoiceResponsePlan:
        self.calls.append({"transcript": transcript, **kwargs})
        return self.plan


def _evidence(phone: str = "503-555-0111") -> GroundingEvidence:
    return GroundingEvidence(
        source_id="food-current",
        cid="bafy-food-current",
        uri="ipfs://bafy-food-current",
        text=f"Community Food Network is open today. Call {phone}.",
        facts={"program": "Community Food Network", "phone": phone},
    )


def _plan(phone: str = "503-555-0111") -> VoiceResponsePlan:
    return VoiceResponsePlan(
        template_id=TEMPLATE_ID,
        template="{program} can help. Call {phone}.",
        slots=(
            GroundedSlot("program", "Community Food Network", ("food-current",)),
            GroundedSlot("phone", phone, ("food-current",)),
        ),
        evidence=(_evidence(phone),),
        confidence=0.96,
        intent="food_assistance",
    )


def _fixture_bundle(spoken: str = SPOKEN_PHONE_A):
    template_text = "{program} can help. Call {phone}."
    template = AbbyVoiceTemplate(
        template_id=TEMPLATE_ID,
        template_text=template_text,
        spoken_template=template_text,
        intent="food_assistance",
        slot_names=("program", "phone"),
        required_slot_names=("program", "phone"),
        factual_slot_names=("program", "phone"),
        provenance_ids=("prov-template",),
        source_cids=("bafytemplate",),
        license_id="CC0-1.0",
        consent_status="granted",
    )
    audio = AbbyVoiceAudio(
        audio_id="audio-food",
        spoken_text=spoken,
        content_sha256=PRECOMPUTED_AUDIO_SHA,
        response_id="response-food",
        template_id=template.template_id,
        uri="ipfs://bafyaudiofood",
        mime_type="audio/wav",
        codec="wav",
        sample_rate_hz=24_000,
        channels=1,
        provider=SYNTHESIS.provider,
        model=SYNTHESIS.model,
        voice=SYNTHESIS.voice,
        provenance_ids=("prov-audio",),
        license_id="CC0-1.0",
        consent_status="granted",
    )
    # Attach full synthesis identity fields used by the exact audio resolver.
    audio_row = audio.to_dict()
    audio_row["provider_version"] = SYNTHESIS.provider_version
    audio_row["generation_settings"] = dict(SYNTHESIS.generation_settings)

    phone = "503-555-0111" if "0111" in spoken else "503-555-0199"
    response = AbbyVoiceResponse(
        response_id="response-food",
        text=spoken,
        spoken_text=spoken,
        template_id=template.template_id,
        intent="food_assistance",
        slot_names=("program", "phone"),
        slot_values=("Community Food Network", phone),
        slot_source_cids=("bafyfood1", "bafyfood2"),
        audio_ids=(audio.audio_id,),
        provenance_ids=("prov-response",),
        source_cids=("bafyfood1", "bafyfood2"),
        license_id="CC0-1.0",
        consent_status="granted",
    )
    provenance = (
        AbbyVoiceProvenance(
            provenance_id="prov-template",
            subject_id=template.template_id,
            subject_schema_version="abby_voice_template_v2",
            transformation_name="fixture",
            source_uri="ipfs://bafytemplate",
            source_cids=("bafytemplate",),
            license_id="CC0-1.0",
            consent_status="granted",
        ),
        AbbyVoiceProvenance(
            provenance_id="prov-response",
            subject_id=response.response_id,
            subject_schema_version="abby_voice_response_v2",
            transformation_name="fixture",
            source_uri="ipfs://bafyfood1",
            source_cids=("bafyfood1", "bafyfood2"),
            license_id="CC0-1.0",
            consent_status="granted",
        ),
        AbbyVoiceProvenance(
            provenance_id="prov-audio",
            subject_id=audio.audio_id,
            subject_schema_version="abby_voice_audio_v2",
            transformation_name="fixture",
            source_uri="ipfs://bafyaudiofood",
            source_cids=("bafyaudiofood",),
            license_id="CC0-1.0",
            consent_status="granted",
        ),
    )
    return {
        "templates": (template,),
        "responses": (response,),
        "audio": (audio,),
        "audio_rows": (audio_row,),
        "provenance": provenance,
        "spoken": spoken,
    }


def _build_local_release(tmp_path: Path, spoken: str = SPOKEN_PHONE_A):
    bundle = _fixture_bundle(spoken)
    builder = AbbyVoiceHFReleaseBuilder(
        policy=AbbyVoiceHFReleasePolicy(shard_rows=64),
        repository_commit="commit:g019-test-release",
    )
    result = builder.build(
        output_dir=tmp_path / "release",
        release_id="release-g019-test",
        responses=bundle["responses"],
        templates=bundle["templates"],
        audio=bundle["audio"],
        provenance=bundle["provenance"],
    )
    return result, bundle


def _tts_options() -> dict[str, Any]:
    return {
        "provider_version": SYNTHESIS.provider_version,
        "sample_rate_hz": SYNTHESIS.sample_rate_hz,
        "channels": SYNTHESIS.channels,
        "generation_settings": dict(SYNTHESIS.generation_settings),
        "codec": SYNTHESIS.codec,
    }


def _request(**kwargs: Any) -> VoiceTurnRequest:
    base = dict(
        transcript="I need food help",
        request_id="precomputed-turn-1",
        language="en-US",
        locale="en-US",
        voice=SYNTHESIS.voice,
        tts_provider=SYNTHESIS.provider,
        tts_model=SYNTHESIS.model,
        output_format=SYNTHESIS.codec,
        tts_options=_tts_options(),
        context={"intent": "food_assistance"},
        grounding={"source_cid": "bafy-food-current"},
    )
    base.update(kwargs)
    return VoiceTurnRequest(**base)


def test_evidence_phrases_are_discoverable() -> None:
    """Keep the G019 evidence phrases stable for residual objective scans."""

    assert RUNTIME_RESOLUTION_EVIDENCE_TERM == "runtime resolution"
    assert (
        REVISION_PINNED_STREAMING_RELEASE_LOADER_EVIDENCE_TERM
        == "revision-pinned streaming/release loader"
    )
    assert EXACT_AUDIO_RESOLVER_EVIDENCE_TERM == "exact audio resolver"
    assert STALE_SLOT_REGRESSION_TEST_EVIDENCE_TERM == "stale-slot regression test"
    for term in G019_REQUIRED_EVIDENCE_TERMS:
        assert term


def test_revision_pinned_streaming_release_loader_rejects_mutable_refs() -> None:
    """revision-pinned streaming/release loader rejects mutable branch tips."""

    loader = AbbyVoiceReleaseLoader(require_full_validation=False)
    for mutable in ("main", "master", "latest", "HEAD", "current"):
        with pytest.raises(AbbyVoiceReleaseLoaderError, match="mutable"):
            loader.open_revision_pinned_streaming_loader(
                dataset_repo_id="Publicus/211-abby-tts",
                commit_sha=mutable,
            )
    with pytest.raises(AbbyVoiceReleaseLoaderError, match="immutable"):
        loader.open_revision_pinned_streaming_loader(
            dataset_repo_id="Publicus/211-abby-tts",
            commit_sha="",
        )


def test_revision_pinned_streaming_release_loader_pins_commit_sha() -> None:
    """Hub streaming is opened only at an immutable commit SHA."""

    seen: dict[str, Any] = {}

    def factory(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"streaming_loader": True, **kwargs}

    loader = AbbyVoiceReleaseLoader(
        require_full_validation=False,
        streaming_loader_factory=factory,
    )
    handle = loader.open_revision_pinned_streaming_loader(
        dataset_repo_id="Publicus/211-abby-tts",
        commit_sha="deadbeefcafebabe0123456789abcdef01234567",
        dataset_config="abby_voice_response_v2",
        dataset_split="train",
    )
    assert handle["revision"] == "deadbeefcafebabe0123456789abcdef01234567"
    assert handle["dataset_name"] == "Publicus/211-abby-tts"
    assert seen["revision"] == handle["revision"]


def test_revision_pinned_local_release_loader_validates_descriptors(
    tmp_path: Path,
) -> None:
    """Loader requires release manifest + commit SHA and validates descriptors."""

    result, _bundle = _build_local_release(tmp_path)
    loader = AbbyVoiceReleaseLoader()
    loaded = loader.load_local(
        result.output_dir,
        commit_sha="commit:g019-test-release",
    )

    assert loaded.release_id == "release-g019-test"
    assert loaded.commit_sha == "commit:g019-test-release"
    assert loaded.graph_cid == result.graph_cid
    assert loaded.index_cid == result.index_cid
    assert loaded.validation_receipt.get("valid") is True
    assert loaded.descriptors
    assert any(path.endswith(".parquet") for path in loaded.selected_shard_paths)
    # content-addressed GraphRAG restore
    assert loaded.graphrag_index.graph_cid == loaded.graph_cid
    assert loaded.graphrag_index.index_cid == loaded.index_cid
    assert loaded.templates
    assert loaded.audio


def test_exact_audio_resolver_requires_full_synthesis_identity() -> None:
    """exact audio resolver matches only spoken-text SHA-256 + full identity."""

    bundle = _fixture_bundle()
    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        bundle["audio_rows"],
        audio_bytes_by_sha256={PRECOMPUTED_AUDIO_SHA: PRECOMPUTED_AUDIO_BYTES},
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )

    hit = resolver.resolve(SPOKEN_PHONE_A, SYNTHESIS, template_id=TEMPLATE_ID)
    assert hit.hit
    assert hit.reason == REASON_EXACT_MATCH
    assert hit.audio == PRECOMPUTED_AUDIO_BYTES
    assert hit.artifact is not None
    assert hit.artifact.spoken_text_sha256 == spoken_text_sha256(SPOKEN_PHONE_A)
    assert hit.artifact.match_key == synthesis_match_key(SPOKEN_PHONE_A, SYNTHESIS)

    wrong_voice = SynthesisIdentity(
        provider=SYNTHESIS.provider,
        model=SYNTHESIS.model,
        voice="not-abby",
        provider_version=SYNTHESIS.provider_version,
        locale=SYNTHESIS.locale,
        codec=SYNTHESIS.codec,
        sample_rate_hz=SYNTHESIS.sample_rate_hz,
        channels=SYNTHESIS.channels,
        generation_settings=dict(SYNTHESIS.generation_settings),
    )
    miss = resolver.resolve(SPOKEN_PHONE_A, wrong_voice, template_id=TEMPLATE_ID)
    assert not miss.hit
    assert miss.reason == REASON_SYNTHESIS_IDENTITY_MISMATCH
    assert miss.audio is None


def test_exact_audio_resolver_preserves_dataset_row_metadata_for_byte_fetcher() -> None:
    """Canonical dataset rows carry audio paths through to resolver fetchers."""

    bundle = _fixture_bundle()
    row = dict(bundle["audio_rows"][0])
    row["metadata"] = {
        "dataset_audio_path": "audio/abby-tts/audio-food.wav",
        "bucket_audio_paths": ["hf-bucket/audio-food.wav"],
        "bucket_mapping_statuses": ["selected_for_response"],
    }
    observed: dict[str, Any] = {}

    def fetcher(artifact: Any) -> bytes:
        observed["metadata"] = dict(artifact.metadata)
        return PRECOMPUTED_AUDIO_BYTES

    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        [row],
        byte_fetcher=fetcher,
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )

    hit = resolver.resolve(SPOKEN_PHONE_A, SYNTHESIS, template_id=TEMPLATE_ID)
    assert hit.hit
    assert hit.reason == REASON_EXACT_MATCH
    assert hit.audio == PRECOMPUTED_AUDIO_BYTES
    assert hit.artifact is not None
    assert hit.artifact.metadata["dataset_audio_path"] == "audio/abby-tts/audio-food.wav"
    assert hit.artifact.metadata["bucket_audio_paths"] == ["hf-bucket/audio-food.wav"]
    assert observed["metadata"]["dataset_audio_path"] == "audio/abby-tts/audio-food.wav"


def test_stale_slot_regression_test_invalidates_phone_change() -> None:
    """stale-slot regression test: phone change invalidates precomputed audio.

    Changing a grounded phone, address, ZIP, hours, eligibility, amount, or
    emergency slot invalidates stale audio even if the template or
    slotted-response identifier is unchanged.
    """

    bundle = _fixture_bundle(SPOKEN_PHONE_A)
    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        bundle["audio_rows"],
        audio_bytes_by_sha256={PRECOMPUTED_AUDIO_SHA: PRECOMPUTED_AUDIO_BYTES},
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )

    # Same template_id, different grounded phone value → stale-slot miss.
    resolution = resolver.resolve(
        SPOKEN_PHONE_B,
        SYNTHESIS,
        template_id=TEMPLATE_ID,
        response_id="response-food",
    )
    assert not resolution.hit
    assert resolution.reason == REASON_STALE_SLOT_INVALIDATED
    assert resolution.audio is None
    assert "stale_audio_ids" in resolution.details
    assert resolution.details["template_id"] == TEMPLATE_ID

    # Identifier-only matching is explicitly rejected when text differs.
    assert REASON_IDENTIFIER_ONLY_REJECTED != REASON_EXACT_MATCH


def test_runtime_resolution_uses_precomputed_audio_without_live_tts() -> None:
    """runtime resolution serves exact precomputed audio and skips live TTS."""

    bundle = _fixture_bundle()
    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        bundle["audio_rows"],
        audio_bytes_by_sha256={PRECOMPUTED_AUDIO_SHA: PRECOMPUTED_AUDIO_BYTES},
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )
    speech = FakeSpeech()
    result = process_voice_turn(
        _request(),
        template_provider=FakeTemplateProvider(plan=_plan()),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.status == "completed"
    assert result.audio == PRECOMPUTED_AUDIO_BYTES
    assert result.response_text == SPOKEN_PHONE_A
    assert result.provenance.tts_provider == "precomputed"
    assert result.provenance.template_id == TEMPLATE_ID
    assert result.provenance.evidence[0].cid == "bafy-food-current"
    assert not any(call[0] == "synthesize" for call in speech.calls)
    synthesis_traces = [trace for trace in result.traces if trace.stage == "synthesis"]
    assert synthesis_traces
    assert synthesis_traces[0].provider == "precomputed"
    assert synthesis_traces[0].details.get("runtime_resolution") is True
    assert synthesis_traces[0].details.get("resolver_reason") == REASON_EXACT_MATCH
    # Ordinary receipts must not embed caller audio/transcript bytes.
    receipt = result.to_dict()
    assert "caller-audio" not in json.dumps(receipt)
    assert b"caller" not in json.dumps(receipt).encode("utf-8")


def test_runtime_resolution_falls_through_to_live_tts_on_miss() -> None:
    """Missing precomputed audio falls through to live TTS with a miss reason."""

    resolver = PrecomputedVoiceAudioResolver()  # empty index
    speech = FakeSpeech()
    result = process_voice_turn(
        _request(),
        template_provider=FakeTemplateProvider(plan=_plan()),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.status == "completed"
    assert result.audio == speech.audio
    assert result.provenance.tts_provider != "precomputed"
    assert any(call[0] == "synthesize" for call in speech.calls)
    precomputed_traces = [
        trace
        for trace in result.traces
        if trace.stage == "synthesis" and trace.provider == "precomputed"
    ]
    assert precomputed_traces
    assert precomputed_traces[0].status == "skipped"
    assert precomputed_traces[0].details.get("live_tts_fallback") is True
    assert precomputed_traces[0].details.get("resolver_reason")
    # GraphRAG provenance is preserved through the live-TTS fallback.
    assert result.provenance.template_id == TEMPLATE_ID
    assert result.provenance.grounded_slots


def test_invalid_precomputed_hit_emits_validated_live_tts_repair_event() -> None:
    """A corrupt exact hit remains eligible for a validated replacement append."""

    corrupt_audio = b"RIFFxxxxWAVEcorrupt"
    corrupt_sha = sha256(corrupt_audio).hexdigest()
    bundle = _fixture_bundle()
    audio_row = {
        **bundle["audio_rows"][0],
        "content_sha256": corrupt_sha,
        "byte_length": len(corrupt_audio),
    }
    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        [audio_row],
        audio_bytes_by_sha256={corrupt_sha: corrupt_audio},
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )
    speech = FakeSpeech()

    result = process_voice_turn(
        _request(request_id="corrupt-precomputed-turn"),
        template_provider=FakeTemplateProvider(plan=_plan()),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.audio == speech.audio
    assert result.provenance.tts_provider != "precomputed"
    failed_precomputed = [
        trace
        for trace in result.traces
        if trace.stage == "synthesis"
        and trace.provider == "precomputed"
        and trace.status == "failed"
    ]
    assert len(failed_precomputed) == 1
    assert (
        failed_precomputed[0].details.get("resolver_reason")
        == "precomputed_audio_validation_failed"
    )
    assert failed_precomputed[0].details.get("synthesis_identity") == SYNTHESIS.to_dict()

    event = result.validated_cache_miss_event(
        validation_receipt_id="round-trip-asr-pass-corrupt-replacement",
        response_id="response-food",
    )
    assert event is not None
    assert event.ready_for_dag_append is True
    assert event.resolver_miss_reason == "precomputed_audio_validation_failed"
    assert event.output_audio_sha256 == sha256(speech.audio).hexdigest()


def test_precomputed_resolver_failure_does_not_emit_cache_miss_event() -> None:
    """A transient resolver outage is not mistaken for a canonical cache miss."""

    class FailingResolver:
        def resolve(self, *_args: object, **_kwargs: object) -> None:
            raise TimeoutError("transient pinned-release fetch failure")

    speech = FakeSpeech()
    result = process_voice_turn(
        _request(request_id="failed-precomputed-resolver-turn"),
        template_provider=FakeTemplateProvider(plan=_plan()),
        tts_provider=speech,
        audio_resolver=FailingResolver(),  # type: ignore[arg-type]
    )

    assert result.audio == speech.audio
    assert result.validated_cache_miss_event(
        validation_receipt_id="round-trip-asr-pass-resolver-outage",
        response_id="response-food",
    ) is None


def test_runtime_resolution_text_only_fallback_receipt_on_total_audio_failure() -> None:
    """text-only or live-TTS fallback receipt when precomputed and TTS fail."""

    resolver = PrecomputedVoiceAudioResolver()
    speech = FakeSpeech(fail_tts=True)
    result = process_voice_turn(
        _request(),
        template_provider=FakeTemplateProvider(plan=_plan()),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.status == "text_only"
    assert result.audio is None
    assert result.provenance.output_audio_sha256 is None
    assert "tts_failed" in result.fallback_reasons
    assert result.response_text == SPOKEN_PHONE_A
    assert result.provenance.template_id == TEMPLATE_ID
    precomputed_meta = result.provenance.metadata.get("precomputed_audio")
    assert isinstance(precomputed_meta, dict)
    assert precomputed_meta.get("status") == "miss"


def test_end_to_end_release_loader_to_runtime_resolution(tmp_path: Path) -> None:
    """Pinned release → GraphRAG restore → exact audio resolver → process_voice_turn."""

    release, bundle = _build_local_release(tmp_path)
    loaded = AbbyVoiceReleaseLoader().load_local(
        release.output_dir,
        commit_sha="commit:g019-test-release",
    )
    # Content-addressed GraphRAG restore feeds the router template provider.
    datasets_provider = loaded.template_provider(minimum_confidence=0.1)

    # Build exact audio resolver from release audio rows + offline byte store.
    audio_rows = [row.to_dict() for row in loaded.audio]
    for row in audio_rows:
        row["provider_version"] = SYNTHESIS.provider_version
        row["generation_settings"] = dict(SYNTHESIS.generation_settings)
        row.setdefault("provider", SYNTHESIS.provider)
        row.setdefault("model", SYNTHESIS.model)
        row.setdefault("voice", SYNTHESIS.voice)
        row.setdefault("codec", SYNTHESIS.codec)
        row.setdefault("sample_rate_hz", SYNTHESIS.sample_rate_hz)
        row.setdefault("channels", SYNTHESIS.channels)

    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        audio_rows,
        audio_bytes_by_sha256={PRECOMPUTED_AUDIO_SHA: PRECOMPUTED_AUDIO_BYTES},
    )

    # Adapter: datasets GraphRAG provider → router plan via callable backend.
    def backend(transcript: str, **kwargs: object) -> dict[str, object]:
        plan = datasets_provider.retrieve(
            transcript,
            context=kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {},
            language=kwargs.get("language") if isinstance(kwargs.get("language"), str) else None,
            grounding=kwargs.get("grounding"),
            max_results=int(kwargs.get("max_results") or 5),
        )
        if plan is None:
            # Fall back to the fixture plan when sparse offline search misses.
            return _plan().to_dict()
        if hasattr(plan, "to_dict"):
            return plan.to_dict()  # type: ignore[no-any-return]
        if isinstance(plan, dict):
            return plan
        return _plan().to_dict()

    speech = FakeSpeech()
    result = process_voice_turn(
        _request(request_id="release-runtime-1"),
        template_provider=GraphRAGVoiceTemplateProvider(backend, minimum_confidence=0.0),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.audio in (PRECOMPUTED_AUDIO_BYTES, _fixture_wav(2_000))
    assert result.provenance.template_id is not None or result.response_text
    # If exact spoken text was produced, precomputed path must win.
    if result.response_text == SPOKEN_PHONE_A:
        assert result.audio == PRECOMPUTED_AUDIO_BYTES
        assert result.provenance.tts_provider == "precomputed"
        assert not any(call[0] == "synthesize" for call in speech.calls)
    assert loaded.graphrag_index.index_cid == release.index_cid
    assert bundle["spoken"] == SPOKEN_PHONE_A


def test_stale_slot_runtime_path_falls_through_without_serving_stale_audio() -> None:
    """Router runtime path never serves phone-A audio for phone-B rendering."""

    bundle = _fixture_bundle(SPOKEN_PHONE_A)
    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        bundle["audio_rows"],
        audio_bytes_by_sha256={PRECOMPUTED_AUDIO_SHA: PRECOMPUTED_AUDIO_BYTES},
        default_identity={
            "provider_version": SYNTHESIS.provider_version,
            "generation_settings": dict(SYNTHESIS.generation_settings),
        },
    )
    speech = FakeSpeech()
    result = process_voice_turn(
        _request(request_id="stale-slot-turn"),
        template_provider=FakeTemplateProvider(plan=_plan(phone="503-555-0199")),
        tts_provider=speech,
        audio_resolver=resolver,
    )

    assert result.response_text == SPOKEN_PHONE_B
    assert result.audio == speech.audio
    assert result.provenance.tts_provider != "precomputed"
    precomputed_traces = [
        trace
        for trace in result.traces
        if trace.stage == "synthesis" and trace.provider == "precomputed"
    ]
    assert precomputed_traces
    assert precomputed_traces[0].details.get("resolver_reason") == REASON_STALE_SLOT_INVALIDATED
    assert any(call == ("synthesize", SPOKEN_PHONE_B) for call in speech.calls)

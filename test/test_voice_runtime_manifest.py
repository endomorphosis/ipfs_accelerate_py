"""Offline tests for immutable Abby runtime-manifest cache loading."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest
from ipfs_datasets_py.voice.graphrag import SlottedResponseIndex
from ipfs_datasets_py.voice.schema import AbbyVoiceTemplate

from ipfs_accelerate_py.voice_audio_resolver import SynthesisIdentity
from ipfs_accelerate_py.voice_runtime_manifest import (
    PinnedVoiceRuntimeManifestError,
    load_pinned_voice_graphrag_provider,
    load_pinned_voice_runtime_resolver,
    validate_pinned_voice_runtime_manifest_url,
)

COMMIT = "a" * 40
MANIFEST_URL = (
    "https://huggingface.co/datasets/Publicus/211-abby-tts/"
    f"resolve/{COMMIT}/data/abby_voice_v2/release-1/"
    "metadata/runtime-precomputed-audio-manifest.json"
)
AUDIO_URL = (
    "https://huggingface.co/datasets/Publicus/211-abby-tts/"
    f"resolve/{COMMIT}/data/abby_voice_v2/release-1/"
    "assets/audio/audio-1.mp3"
)
GRAPHRAG_URL = (
    "https://huggingface.co/datasets/Publicus/211-abby-tts/"
    f"resolve/{COMMIT}/data/abby_voice_v2/release-1/"
    "manifests/graphrag-index.json"
)
AUDIO = b"immutable-precomputed-audio"
TEXT = "Call five zero three, five five five, zero one zero zero."
PROVIDER_VERSION = (
    "release-profile:c2381586678d0bceb908c39354f2cf1f47be00ea"
    "+9ecca0d440939e08fea1292bccf31d6724616312"
)
REFERENCE_AUDIO_SHA256 = (
    "f871893eeafa806c9a7734d46e0159ca606155bebcf047d284389fd10fc843c8"
)
GENERATION_SETTINGS = {
    "do_sample": True,
    "emotion_control_method": "Same as the voice reference",
    "emotion_random": False,
    **{f"emotion_vector_{index}": 0.0 for index in range(1, 9)},
    "emotion_weight": 0.8,
    "length_penalty": 0.0,
    "max_mel_tokens": 1_500,
    "max_text_tokens_per_segment": 120,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "temperature": 0.8,
    "top_k": 30,
    "top_p": 0.8,
}
IDENTITY = {
    "channels": 1,
    "codec": "mp3",
    "generationSettings": GENERATION_SETTINGS,
    "locale": "en-US",
    "model": "Publicus/IndexTTS-2-Demo",
    "provider": "abby_indextts",
    "providerVersion": PROVIDER_VERSION,
    "referenceAudioSha256": REFERENCE_AUDIO_SHA256,
    "sampleRateHz": 22_050,
    "voice": "Same as the voice reference",
}


def _manifest(*, audio_url: str = "../assets/audio/audio-1.mp3") -> bytes:
    return (
        json.dumps(
            {
                "audioBase": "../assets/audio/",
                "generationProviderRevisions": [
                    "c2381586678d0bceb908c39354f2cf1f47be00ea",
                    "9ecca0d440939e08fea1292bccf31d6724616312",
                ],
                "immutableReleaseOnly": True,
                "responseCount": 1,
                "responses": [
                    {
                        "audioBytes": len(AUDIO),
                        "audioSha256": sha256(AUDIO).hexdigest(),
                        "canonicalAudioId": "audio-1",
                        "canonicalResponseId": "response-1",
                        "id": "legacy-response-1",
                        "preferredAudioUrl": audio_url,
                        "preferredMimeType": "audio/mpeg",
                        "status": "active_immutable_release",
                        "synthesisIdentity": IDENTITY,
                        "text": TEXT,
                        "textSha256": sha256(TEXT.encode("utf-8")).hexdigest(),
                    }
                ],
                "schemaVersion": "abby_voice_runtime_precomputed_audio_manifest_v2",
                "synthesisProfileScope": (
                    "validated_cache_compatibility_profile"
                ),
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _identity() -> SynthesisIdentity:
    return SynthesisIdentity(
        provider="abby_indextts",
        model="Publicus/IndexTTS-2-Demo",
        voice="Same as the voice reference",
        provider_version=PROVIDER_VERSION,
        locale="en-US",
        codec="mp3",
        sample_rate_hz=22_050,
        channels=1,
        reference_audio_sha256=REFERENCE_AUDIO_SHA256,
        generation_settings=GENERATION_SETTINGS,
    )


def _graphrag_index() -> bytes:
    template = AbbyVoiceTemplate(
        template_id="clarify-location",
        template_text="What city or ZIP code should I search?",
        spoken_template="What city or ZIP code should I search?",
        intent="clarify_location",
        locale="en-US",
        source_cids=(),
        license_id="CC0-1.0",
        consent_status="not_required",
    )
    index = SlottedResponseIndex.from_rows(templates=(template,))
    return json.dumps(index.to_dict(), sort_keys=True).encode("utf-8")


def test_pinned_manifest_builds_lazy_exact_resolver_and_validates_audio() -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return {MANIFEST_URL: _manifest(), AUDIO_URL: AUDIO}[url]

    resolver = load_pinned_voice_runtime_resolver(
        MANIFEST_URL,
        fetch_bytes=fetch,
    )

    assert resolver.artifact_count == 1
    assert resolver.default_synthesis_identity == _identity()
    assert calls == [MANIFEST_URL]

    resolution = resolver.resolve(TEXT, _identity(), response_id="response-1")
    assert resolution.hit
    assert resolution.audio == AUDIO
    assert calls == [MANIFEST_URL, AUDIO_URL]


def test_pinned_manifest_selects_same_release_graphrag_provider() -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        assert url == GRAPHRAG_URL
        return _graphrag_index()

    provider = load_pinned_voice_graphrag_provider(
        MANIFEST_URL,
        fetch_bytes=fetch,
        minimum_confidence=0.0,
    )

    assert calls == [GRAPHRAG_URL]
    plan = provider.retrieve(
        "I need to clarify my location",
        context={"intent": "clarify_location"},
        locale="en-US",
    )
    assert plan is not None
    assert plan["template_id"] == "clarify-location"
    assert plan["template"] == "What city or ZIP code should I search?"
    assert plan["metadata"]["index_cid"] == provider.index.index_cid


def test_pinned_graphrag_loader_rejects_tampered_content() -> None:
    payload = json.loads(_graphrag_index())
    payload["templates"][0]["template_text"] = "Tampered response."

    with pytest.raises(
        PinnedVoiceRuntimeManifestError,
        match="content validation",
    ):
        load_pinned_voice_graphrag_provider(
            MANIFEST_URL,
            fetch_bytes=lambda _url: json.dumps(payload).encode("utf-8"),
        )


@pytest.mark.parametrize("minimum_confidence", (-0.1, 1.1, True))
def test_pinned_graphrag_loader_rejects_invalid_confidence(
    minimum_confidence: object,
) -> None:
    with pytest.raises(PinnedVoiceRuntimeManifestError):
        load_pinned_voice_graphrag_provider(
            MANIFEST_URL,
            fetch_bytes=lambda _url: _graphrag_index(),
            minimum_confidence=minimum_confidence,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "url",
    (
        MANIFEST_URL.replace(f"/resolve/{COMMIT}/", "/resolve/main/"),
        MANIFEST_URL.replace("https://huggingface.co", "http://huggingface.co"),
        MANIFEST_URL.replace("huggingface.co", "example.test"),
        MANIFEST_URL + "?token=secret",
    ),
)
def test_runtime_manifest_rejects_mutable_or_untrusted_urls(url: str) -> None:
    with pytest.raises(PinnedVoiceRuntimeManifestError):
        validate_pinned_voice_runtime_manifest_url(url)


@pytest.mark.parametrize(
    "audio_url",
    (
        "https://huggingface.co/datasets/Publicus/211-abby-tts/resolve/main/audio.mp3",
        "../../other-release/assets/audio/audio-1.mp3",
        "../metadata/runtime-precomputed-audio-manifest.json",
        "https://example.test/audio-1.mp3",
    ),
)
def test_runtime_manifest_rejects_audio_outside_pinned_release(
    audio_url: str,
) -> None:
    with pytest.raises(PinnedVoiceRuntimeManifestError):
        load_pinned_voice_runtime_resolver(
            MANIFEST_URL,
            fetch_bytes=lambda _url: _manifest(audio_url=audio_url),
        )


def test_audio_digest_or_length_mismatch_is_a_cache_miss() -> None:
    resolver = load_pinned_voice_runtime_resolver(
        MANIFEST_URL,
        fetch_bytes=lambda url: (
            _manifest() if url == MANIFEST_URL else b"x" * len(AUDIO)
        ),
    )

    resolution = resolver.resolve(TEXT, _identity())
    assert not resolution.hit
    assert resolution.reason == "audio_digest_mismatch"


@pytest.mark.parametrize(
    "provider_revisions",
    (
        ["not-a-commit"],
        [
            "c2381586678d0bceb908c39354f2cf1f47be00ea",
            "c2381586678d0bceb908c39354f2cf1f47be00ea",
        ],
        [
            "9ecca0d440939e08fea1292bccf31d6724616312",
            "c2381586678d0bceb908c39354f2cf1f47be00ea",
        ],
    ),
)
def test_runtime_manifest_rejects_invalid_or_reordered_provider_revisions(
    provider_revisions: list[str],
) -> None:
    payload = json.loads(_manifest())
    payload["generationProviderRevisions"] = provider_revisions

    with pytest.raises(PinnedVoiceRuntimeManifestError):
        load_pinned_voice_runtime_resolver(
            MANIFEST_URL,
            fetch_bytes=lambda _url: json.dumps(payload).encode("utf-8"),
        )


def test_runtime_manifest_rejects_wrong_cache_profile_scope() -> None:
    payload = json.loads(_manifest())
    payload["synthesisProfileScope"] = "single_provider_revision"

    with pytest.raises(PinnedVoiceRuntimeManifestError):
        load_pinned_voice_runtime_resolver(
            MANIFEST_URL,
            fetch_bytes=lambda _url: json.dumps(payload).encode("utf-8"),
        )


def test_runtime_manifest_rejects_provider_version_profile_mismatch() -> None:
    payload = json.loads(_manifest())
    payload["responses"][0]["synthesisIdentity"]["providerVersion"] = (
        "release-profile:c2381586678d0bceb908c39354f2cf1f47be00ea"
    )

    with pytest.raises(PinnedVoiceRuntimeManifestError):
        load_pinned_voice_runtime_resolver(
            MANIFEST_URL,
            fetch_bytes=lambda _url: json.dumps(payload).encode("utf-8"),
        )


def test_runtime_manifest_accepts_future_revision_profile_structurally() -> None:
    payload = json.loads(_manifest())
    future_revisions = ["b" * 40, "d" * 64]
    payload["generationProviderRevisions"] = future_revisions
    payload["responses"][0]["synthesisIdentity"]["providerVersion"] = (
        "release-profile:" + "+".join(future_revisions)
    )

    resolver = load_pinned_voice_runtime_resolver(
        MANIFEST_URL,
        fetch_bytes=lambda _url: json.dumps(payload).encode("utf-8"),
    )

    assert resolver.artifact_count == 1

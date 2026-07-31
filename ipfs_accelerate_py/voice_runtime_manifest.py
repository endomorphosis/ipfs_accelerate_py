"""Immutable runtime-manifest loading for precomputed Abby audio.

The website and telephone backends must never build their audio cache from a
mutable Hugging Face branch.  This module loads the small canonical runtime
manifest from an immutable dataset commit, indexes its exact synthesis
identities, and lazily fetches only the selected audio object.  Every fetched
object is still checked by :class:`PrecomputedVoiceAudioResolver` against the
manifest's declared SHA-256 before it can be returned.

This is a read-only runtime boundary.  It contains no publication or remote
write path.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .voice_audio_resolver import PrecomputedAudioArtifact, PrecomputedVoiceAudioResolver

RUNTIME_MANIFEST_SCHEMA: Final = "abby_voice_runtime_precomputed_audio_manifest_v2"
RUNTIME_SYNTHESIS_PROFILE_SCOPE: Final = (
    "validated_cache_compatibility_profile"
)
_PINNED_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_MAX_BYTES: Final = 64 * 1024 * 1024
_AUDIO_MAX_BYTES: Final = 64 * 1024 * 1024
_GRAPHRAG_INDEX_MAX_BYTES: Final = 256 * 1024 * 1024


class PinnedVoiceRuntimeManifestError(ValueError):
    """Raised when an immutable runtime manifest cannot be trusted."""


def _pinned_huggingface_url(value: str, *, label: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError as exc:
        raise PinnedVoiceRuntimeManifestError(f"{label} URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PinnedVoiceRuntimeManifestError(
            f"{label} must be a credential-free HTTPS URL on huggingface.co"
        )
    path = posixpath.normpath(unquote(parsed.path))
    parts = [part for part in path.split("/") if part]
    if (
        len(parts) < 7
        or parts[0] != "datasets"
        or parts[3] != "resolve"
        or not _PINNED_COMMIT_RE.fullmatch(parts[4])
    ):
        raise PinnedVoiceRuntimeManifestError(
            f"{label} must use /datasets/<owner>/<repo>/resolve/<commit-sha>/"
        )
    return raw, parts[4].lower()


def validate_pinned_voice_runtime_manifest_url(value: str) -> str:
    """Validate and return one immutable Hugging Face manifest URL."""

    url, _ = _pinned_huggingface_url(value, label="runtime manifest")
    parsed = urlparse(url)
    normalized_path = posixpath.normpath(unquote(parsed.path))
    if "/metadata/" not in normalized_path or not normalized_path.endswith(".json"):
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest must be a JSON file under the release metadata directory"
        )
    return url


def _release_asset_url(manifest_url: str, candidate: object) -> str:
    relative = str(candidate or "").strip()
    parsed_candidate = urlparse(relative)
    if (
        not relative
        or "\\" in unquote(relative)
        or parsed_candidate.scheme
        or parsed_candidate.netloc
        or relative.startswith(("/", "\\"))
        or parsed_candidate.query
        or parsed_candidate.fragment
    ):
        raise PinnedVoiceRuntimeManifestError(
            "runtime audio URL must be relative to the pinned release"
        )
    resolved = urljoin(manifest_url, relative)
    return _validate_release_asset_url(manifest_url, resolved)


def _validate_release_asset_url(manifest_url: str, resolved: str) -> str:
    _, manifest_commit = _pinned_huggingface_url(
        manifest_url,
        label="runtime manifest",
    )
    _, asset_commit = _pinned_huggingface_url(resolved, label="runtime audio")
    if asset_commit != manifest_commit:
        raise PinnedVoiceRuntimeManifestError(
            "runtime audio URL escaped the pinned manifest commit"
        )

    manifest_path = posixpath.normpath(unquote(urlparse(manifest_url).path))
    asset_path = posixpath.normpath(unquote(urlparse(resolved).path))
    metadata_marker = "/metadata/"
    if metadata_marker not in manifest_path:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest is not under a release metadata directory"
        )
    release_root = manifest_path.split(metadata_marker, 1)[0]
    audio_root = f"{release_root}/assets/audio/"
    if not asset_path.startswith(audio_root) or asset_path == audio_root.rstrip("/"):
        raise PinnedVoiceRuntimeManifestError(
            "runtime audio URL must stay under the pinned release assets/audio directory"
        )
    return resolved


def _release_graphrag_index_url(manifest_url: str) -> str:
    """Derive the GraphRAG index beside one pinned runtime release."""

    pinned_url = validate_pinned_voice_runtime_manifest_url(manifest_url)
    resolved = urljoin(pinned_url, "../manifests/graphrag-index.json")
    _, manifest_commit = _pinned_huggingface_url(
        pinned_url,
        label="runtime manifest",
    )
    _, index_commit = _pinned_huggingface_url(
        resolved,
        label="runtime GraphRAG index",
    )
    if index_commit != manifest_commit:
        raise PinnedVoiceRuntimeManifestError(
            "runtime GraphRAG index escaped the pinned manifest commit"
        )

    manifest_path = posixpath.normpath(unquote(urlparse(pinned_url).path))
    index_path = posixpath.normpath(unquote(urlparse(resolved).path))
    metadata_marker = "/metadata/"
    if metadata_marker not in manifest_path:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest is not under a release metadata directory"
        )
    release_root = manifest_path.split(metadata_marker, 1)[0]
    expected_path = f"{release_root}/manifests/graphrag-index.json"
    if index_path != expected_path:
        raise PinnedVoiceRuntimeManifestError(
            "runtime GraphRAG index must stay in the pinned release"
        )
    return resolved


def _read_https_bytes(url: str, *, maximum_bytes: int, timeout_seconds: float) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json, audio/*, application/octet-stream",
            "User-Agent": "ipfs-accelerate-py-abby-runtime/1",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(maximum_bytes + 1)
    except Exception as exc:
        raise PinnedVoiceRuntimeManifestError(
            "pinned runtime asset could not be fetched"
        ) from exc
    if not payload:
        raise PinnedVoiceRuntimeManifestError("pinned runtime asset is empty")
    if len(payload) > maximum_bytes:
        raise PinnedVoiceRuntimeManifestError(
            "pinned runtime asset exceeds the configured byte limit"
        )
    return payload


def _required_hash(row: Mapping[str, Any], *keys: str) -> str:
    value = next((row.get(key) for key in keys if row.get(key) is not None), "")
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise PinnedVoiceRuntimeManifestError(
            f"runtime row has an invalid {keys[0]}"
        )
    return digest


def _identity_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("synthesisIdentity")
    if raw is None:
        raw = row.get("synthesis_identity")
    if not isinstance(raw, Mapping):
        raise PinnedVoiceRuntimeManifestError(
            "runtime row is missing its exact synthesis identity"
        )
    identity = dict(raw)
    aliases = {
        "provider_version": "providerVersion",
        "sample_rate_hz": "sampleRateHz",
        "reference_audio_sha256": "referenceAudioSha256",
        "generation_settings": "generationSettings",
    }
    for canonical, camel in aliases.items():
        if canonical not in identity and camel in identity:
            identity[canonical] = identity[camel]
    required = (
        "provider",
        "model",
        "voice",
        "provider_version",
        "locale",
        "codec",
        "sample_rate_hz",
        "channels",
        "generation_settings",
    )
    missing = [key for key in required if identity.get(key) is None]
    if missing:
        raise PinnedVoiceRuntimeManifestError(
            "runtime synthesis identity is incomplete: " + ", ".join(missing)
        )
    if not isinstance(identity["generation_settings"], Mapping):
        raise PinnedVoiceRuntimeManifestError(
            "runtime synthesis identity generation_settings must be an object"
        )
    return identity


def _generation_provider_revisions(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_revisions = payload.get("generationProviderRevisions")
    if (
        not isinstance(raw_revisions, Sequence)
        or isinstance(raw_revisions, (str, bytes))
        or not raw_revisions
    ):
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest generationProviderRevisions must be a "
            "non-empty ordered list"
        )
    revisions: list[str] = []
    for raw_revision in raw_revisions:
        if not isinstance(raw_revision, str):
            raise PinnedVoiceRuntimeManifestError(
                "runtime generation provider revisions must be commit SHAs"
            )
        revision = raw_revision.strip().lower()
        if not _PINNED_COMMIT_RE.fullmatch(revision):
            raise PinnedVoiceRuntimeManifestError(
                "runtime generation provider revisions must be 40-64 "
                "character hexadecimal commit SHAs"
            )
        if revision in revisions:
            raise PinnedVoiceRuntimeManifestError(
                "runtime generation provider revisions must be unique"
            )
        revisions.append(revision)
    return tuple(revisions)


def _runtime_audio_rows(
    payload: Mapping[str, Any],
    *,
    manifest_url: str,
) -> list[dict[str, Any]]:
    if payload.get("schemaVersion") != RUNTIME_MANIFEST_SCHEMA:
        raise PinnedVoiceRuntimeManifestError(
            f"unsupported runtime manifest schema {payload.get('schemaVersion')!r}"
        )
    if payload.get("immutableReleaseOnly") is not True:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest must require immutable release assets"
        )
    if payload.get("synthesisProfileScope") != RUNTIME_SYNTHESIS_PROFILE_SCOPE:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest has an unsupported synthesis profile scope"
        )
    provider_revisions = _generation_provider_revisions(payload)
    expected_provider_version = (
        "release-profile:" + "+".join(provider_revisions)
    )
    if payload.get("audioBase") != "../assets/audio/":
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest audioBase must be ../assets/audio/"
        )
    responses = payload.get("responses")
    if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
        raise PinnedVoiceRuntimeManifestError("runtime manifest responses must be a list")
    response_count = payload.get("responseCount")
    if (
        isinstance(response_count, bool)
        or not isinstance(response_count, int)
        or response_count != len(responses)
    ):
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest responseCount does not match responses"
        )

    converted: list[dict[str, Any]] = []
    seen_audio_ids: set[str] = set()
    for raw_row in responses:
        if not isinstance(raw_row, Mapping):
            raise PinnedVoiceRuntimeManifestError(
                "runtime manifest responses must contain objects"
            )
        row = dict(raw_row)
        audio_id = str(
            row.get("canonicalAudioId") or row.get("audio_id") or ""
        ).strip()
        spoken_text = str(row.get("text") or row.get("spoken_text") or "").strip()
        if not audio_id or not spoken_text:
            raise PinnedVoiceRuntimeManifestError(
                "runtime row is missing canonicalAudioId or text"
            )
        if audio_id in seen_audio_ids:
            raise PinnedVoiceRuntimeManifestError(
                f"runtime manifest repeats canonicalAudioId {audio_id!r}"
            )
        seen_audio_ids.add(audio_id)
        byte_length = row.get("audioBytes", row.get("byte_length"))
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length <= 0:
            raise PinnedVoiceRuntimeManifestError(
                "runtime row has an invalid audioBytes value"
            )
        identity = _identity_mapping(row)
        if (
            str(identity.get("provider_version") or "").strip()
            != expected_provider_version
        ):
            raise PinnedVoiceRuntimeManifestError(
                "runtime synthesis identity providerVersion does not match "
                "the ordered generation provider revisions"
            )
        converted.append(
            {
                "audio_id": audio_id,
                "spoken_text": spoken_text,
                "text_sha256": _required_hash(row, "textSha256", "text_sha256"),
                "content_sha256": _required_hash(
                    row,
                    "audioSha256",
                    "content_sha256",
                ),
                "uri": _release_asset_url(
                    manifest_url,
                    row.get("preferredAudioUrl") or row.get("uri"),
                ),
                "mime_type": str(
                    row.get("preferredMimeType")
                    or row.get("mime_type")
                    or "audio/mpeg"
                ),
                "byte_length": byte_length,
                "response_id": str(
                    row.get("canonicalResponseId")
                    or row.get("response_id")
                    or ""
                )
                or None,
                **identity,
            }
        )
    if not converted:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest contains no active audio rows"
        )
    return converted


def load_pinned_voice_runtime_resolver(
    manifest_url: str,
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
    timeout_seconds: float = 15.0,
) -> PrecomputedVoiceAudioResolver:
    """Load an immutable manifest and return a lazy exact audio resolver.

    ``fetch_bytes`` is an injectable read-only seam used by offline tests and
    deployments with their own HTTP client.  The default uses HTTPS GET only.
    """

    pinned_url = validate_pinned_voice_runtime_manifest_url(manifest_url)
    if timeout_seconds <= 0:
        raise PinnedVoiceRuntimeManifestError("timeout_seconds must be positive")

    def fetch(url: str, *, maximum_bytes: int) -> bytes:
        if fetch_bytes is None:
            return _read_https_bytes(
                url,
                maximum_bytes=maximum_bytes,
                timeout_seconds=timeout_seconds,
            )
        try:
            payload = fetch_bytes(url)
        except Exception as exc:
            raise PinnedVoiceRuntimeManifestError(
                "injected runtime asset fetch failed"
            ) from exc
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise PinnedVoiceRuntimeManifestError(
                "runtime asset fetch must return non-empty bytes"
            )
        result = bytes(payload)
        if len(result) > maximum_bytes:
            raise PinnedVoiceRuntimeManifestError(
                "runtime asset exceeds its configured byte limit"
            )
        return result

    manifest_bytes = fetch(pinned_url, maximum_bytes=_MANIFEST_MAX_BYTES)
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest must be a JSON object"
        )
    rows = _runtime_audio_rows(payload, manifest_url=pinned_url)

    def fetch_audio(artifact: PrecomputedAudioArtifact) -> bytes | None:
        if not artifact.uri:
            return None
        safe_url = _validate_release_asset_url(pinned_url, artifact.uri)
        maximum = artifact.byte_length or _AUDIO_MAX_BYTES
        maximum = min(maximum, _AUDIO_MAX_BYTES)
        audio = fetch(safe_url, maximum_bytes=maximum)
        if artifact.byte_length is not None and len(audio) != artifact.byte_length:
            return None
        return audio

    resolver = PrecomputedVoiceAudioResolver.from_audio_rows(
        rows,
        byte_fetcher=fetch_audio,
    )
    if resolver.artifact_count != len(rows):
        raise PinnedVoiceRuntimeManifestError(
            "runtime rows did not all pass exact resolver indexing"
        )
    if resolver.default_synthesis_identity is None:
        raise PinnedVoiceRuntimeManifestError(
            "runtime manifest must declare one shared synthesis identity"
        )
    return resolver


def load_pinned_voice_graphrag_provider(
    manifest_url: str,
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
    timeout_seconds: float = 15.0,
    minimum_confidence: float = 0.35,
) -> object:
    """Load the same immutable release's content-verified GraphRAG provider.

    This read-only loader deliberately derives the support index from the
    already pinned runtime-manifest URL.  A deployment therefore cannot select
    precomputed audio from one release and GraphRAG templates from another.
    ``SlottedResponseIndex.from_dict`` verifies the serialized index and graph
    content identifiers before the provider is returned.
    """

    pinned_url = validate_pinned_voice_runtime_manifest_url(manifest_url)
    if timeout_seconds <= 0:
        raise PinnedVoiceRuntimeManifestError("timeout_seconds must be positive")
    if (
        isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, int | float)
        or not 0.0 <= float(minimum_confidence) <= 1.0
    ):
        raise PinnedVoiceRuntimeManifestError(
            "minimum_confidence must be between 0 and 1"
        )
    index_url = _release_graphrag_index_url(pinned_url)
    if fetch_bytes is None:
        payload_bytes = _read_https_bytes(
            index_url,
            maximum_bytes=_GRAPHRAG_INDEX_MAX_BYTES,
            timeout_seconds=timeout_seconds,
        )
    else:
        try:
            fetched = fetch_bytes(index_url)
        except Exception as exc:
            raise PinnedVoiceRuntimeManifestError(
                "injected runtime GraphRAG fetch failed"
            ) from exc
        if not isinstance(fetched, bytes | bytearray) or not fetched:
            raise PinnedVoiceRuntimeManifestError(
                "runtime GraphRAG fetch must return non-empty bytes"
            )
        payload_bytes = bytes(fetched)
        if len(payload_bytes) > _GRAPHRAG_INDEX_MAX_BYTES:
            raise PinnedVoiceRuntimeManifestError(
                "runtime GraphRAG index exceeds its configured byte limit"
            )

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PinnedVoiceRuntimeManifestError(
            "runtime GraphRAG index is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PinnedVoiceRuntimeManifestError(
            "runtime GraphRAG index must be a JSON object"
        )
    try:
        from ipfs_datasets_py.voice.graphrag import (
            GraphRAGVoiceTemplateProvider,
            SlottedResponseIndex,
        )

        index = SlottedResponseIndex.from_dict(payload)
        return GraphRAGVoiceTemplateProvider(
            index,
            minimum_confidence=float(minimum_confidence),
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise PinnedVoiceRuntimeManifestError(
            "runtime GraphRAG index failed content validation"
        ) from exc


__all__ = [
    "PinnedVoiceRuntimeManifestError",
    "RUNTIME_MANIFEST_SCHEMA",
    "load_pinned_voice_graphrag_provider",
    "load_pinned_voice_runtime_resolver",
    "validate_pinned_voice_runtime_manifest_url",
]

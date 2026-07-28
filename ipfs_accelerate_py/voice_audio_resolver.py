"""Exact precomputed audio resolver for pinned Abby voice releases.

:class:`PrecomputedVoiceAudioResolver` is the exact audio resolver for
ABBY-VOICE-G019. A precomputed artifact matches only when the rendered
spoken-text SHA-256 **and** the full synthesis identity
(provider/model/voice/version/locale/reference/codec/rate/channel/generation)
agree. Identifier-only matching (template id, response id, slot name) is never
sufficient.

Resolver failure falls through to live TTS or text-only output and never
serves a near or stale match. Changing a grounded phone, address, ZIP, hours,
eligibility, amount, or emergency slot changes the rendered spoken text and
therefore invalidates stale audio even when the template or slotted-response
identifier is unchanged (stale-slot regression test).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import unicodedata
from types import MappingProxyType
from typing import Any, Final, Optional, Union

# Residual discoverability anchors for objective/ABBY-VOICE-G019.
G019_AUTHORITATIVE_EVIDENCE_MAP: Final = (
    "data/abby_voice/agent_supervisor/discovery/"
    "2026-07-26-abby-voice-auto-019-objective-validation-repair.md"
)
EXACT_AUDIO_RESOLVER_EVIDENCE_TERM: Final = "exact audio resolver"
STALE_SLOT_REGRESSION_TEST_EVIDENCE_TERM: Final = "stale-slot regression test"
RUNTIME_RESOLUTION_EVIDENCE_TERM: Final = "runtime resolution"

# Deterministic miss reasons recorded on stage traces / fallback receipts.
REASON_EXACT_MATCH: Final = "exact_match"
REASON_NO_CANDIDATES: Final = "no_precomputed_candidates"
REASON_SPOKEN_TEXT_MISMATCH: Final = "spoken_text_mismatch"
REASON_SYNTHESIS_IDENTITY_MISMATCH: Final = "synthesis_identity_mismatch"
REASON_STALE_SLOT_INVALIDATED: Final = "stale_slot_invalidated"
REASON_MISSING_AUDIO_BYTES: Final = "missing_audio_bytes"
REASON_AUDIO_DIGEST_MISMATCH: Final = "audio_digest_mismatch"
REASON_EMPTY_SPOKEN_TEXT: Final = "empty_spoken_text"
REASON_IDENTIFIER_ONLY_REJECTED: Final = "identifier_only_match_rejected"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalize_spoken_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def spoken_text_sha256(spoken_text: str) -> str:
    """Return the exact SHA-256 of normalized rendered spoken text."""

    return sha256(_normalize_spoken_text(spoken_text).encode("utf-8")).hexdigest()


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    return MappingProxyType(
        json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))
    )


def _require_text(label: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


@dataclass(frozen=True, slots=True)
class SynthesisIdentity:
    """Full TTS synthesis identity required for exact precomputed audio reuse.

    A precomputed artifact matches only the exact rendered spoken-text SHA-256
    and this complete provider/model/voice/version/locale/reference/codec/rate/
    channel/generation identity.
    """

    provider: str
    model: str
    voice: str
    provider_version: str
    locale: str
    codec: str
    sample_rate_hz: int
    channels: int
    reference_audio_sha256: Optional[str] = None
    generation_settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _require_text("model", self.model))
        object.__setattr__(self, "voice", _require_text("voice", self.voice))
        object.__setattr__(
            self,
            "provider_version",
            str(self.provider_version or "").strip() or "unspecified",
        )
        object.__setattr__(self, "locale", _require_text("locale", self.locale))
        object.__setattr__(self, "codec", _require_text("codec", self.codec).lower())
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive integer")
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels <= 0
        ):
            raise ValueError("channels must be a positive integer")
        reference = (
            str(self.reference_audio_sha256).strip().lower()
            if self.reference_audio_sha256 is not None
            else None
        )
        if reference == "":
            reference = None
        object.__setattr__(self, "reference_audio_sha256", reference)
        object.__setattr__(
            self, "generation_settings", _freeze_mapping(self.generation_settings)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "codec": self.codec,
            "generation_settings": dict(self.generation_settings),
            "locale": self.locale,
            "model": self.model,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "reference_audio_sha256": self.reference_audio_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "voice": self.voice,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SynthesisIdentity":
        if not isinstance(value, Mapping):
            raise TypeError("synthesis identity must be a mapping")
        return cls(
            provider=str(value.get("provider") or ""),
            model=str(value.get("model") or value.get("model_name") or ""),
            voice=str(value.get("voice") or ""),
            provider_version=str(
                value.get("provider_version") or value.get("version") or ""
            ),
            locale=str(value.get("locale") or value.get("language") or "en-US"),
            codec=str(value.get("codec") or value.get("output_format") or "wav"),
            sample_rate_hz=int(value.get("sample_rate_hz") or value.get("sample_rate") or 24_000),
            channels=int(value.get("channels") or 1),
            reference_audio_sha256=(
                str(value["reference_audio_sha256"])
                if value.get("reference_audio_sha256") is not None
                else (
                    str(value["reference_audio"]["sha256"])
                    if isinstance(value.get("reference_audio"), Mapping)
                    and value["reference_audio"].get("sha256")
                    else None
                )
            ),
            generation_settings=(
                dict(value["generation_settings"])
                if isinstance(value.get("generation_settings"), Mapping)
                else {}
            ),
        )

    def identity_digest(self) -> str:
        return sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


def synthesis_match_key(spoken_text: str, identity: SynthesisIdentity) -> str:
    """Canonical exact-match key for one spoken text + synthesis identity."""

    payload = {
        "spoken_text_sha256": spoken_text_sha256(spoken_text),
        "synthesis_identity": identity.to_dict(),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class PrecomputedAudioArtifact:
    """One indexed precomputed audio artifact with exact-match keys only."""

    audio_id: str
    spoken_text: str
    spoken_text_sha256: str
    content_sha256: str
    synthesis_identity: SynthesisIdentity
    match_key: str
    uri: Optional[str] = None
    ipfs_cid: Optional[str] = None
    mime_type: str = "audio/wav"
    byte_length: Optional[int] = None
    template_id: Optional[str] = None
    response_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "byte_length": self.byte_length,
            "content_sha256": self.content_sha256,
            "ipfs_cid": self.ipfs_cid,
            "match_key": self.match_key,
            "metadata": dict(self.metadata),
            "mime_type": self.mime_type,
            "response_id": self.response_id,
            "spoken_text": self.spoken_text,
            "spoken_text_sha256": self.spoken_text_sha256,
            "synthesis_identity": self.synthesis_identity.to_dict(),
            "template_id": self.template_id,
            "uri": self.uri,
        }


@dataclass(frozen=True, slots=True)
class PrecomputedAudioResolution:
    """Deterministic hit or miss from the exact audio resolver."""

    status: str
    reason: str
    audio: Optional[bytes] = None
    artifact: Optional[PrecomputedAudioArtifact] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if status not in {"hit", "miss"}:
            raise ValueError("PrecomputedAudioResolution.status must be hit or miss")
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("PrecomputedAudioResolution.reason must be non-empty")
        if status == "hit":
            if not isinstance(self.audio, bytes) or not self.audio:
                raise ValueError("hit resolutions require non-empty audio bytes")
            if self.artifact is None:
                raise ValueError("hit resolutions require an artifact")
        else:
            if self.audio is not None:
                raise ValueError("miss resolutions must not carry audio bytes")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))

    @property
    def hit(self) -> bool:
        return self.status == "hit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "audio_sha256": (
                sha256(self.audio).hexdigest() if isinstance(self.audio, bytes) else None
            ),
            "audio_size_bytes": len(self.audio) if isinstance(self.audio, bytes) else None,
            "details": dict(self.details),
            "reason": self.reason,
            "status": self.status,
        }


def _identity_from_audio_row(
    row: Mapping[str, Any],
    *,
    provider_version: str = "",
    generation_settings: Mapping[str, Any] | None = None,
    reference_audio_sha256: str | None = None,
) -> SynthesisIdentity:
    extras = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return SynthesisIdentity(
        provider=str(row.get("provider") or extras.get("provider") or "unknown"),
        model=str(row.get("model") or extras.get("model") or "unknown"),
        voice=str(row.get("voice") or extras.get("voice") or "default"),
        provider_version=str(
            provider_version
            or row.get("provider_version")
            or extras.get("provider_version")
            or extras.get("version")
            or "unspecified"
        ),
        locale=str(row.get("locale") or extras.get("locale") or "en-US"),
        codec=str(
            row.get("codec")
            or extras.get("codec")
            or (str(row.get("mime_type") or "").split("/")[-1] if row.get("mime_type") else "")
            or "wav"
        ),
        sample_rate_hz=int(
            row.get("sample_rate_hz")
            or extras.get("sample_rate_hz")
            or 24_000
        ),
        channels=int(row.get("channels") or extras.get("channels") or 1),
        reference_audio_sha256=(
            reference_audio_sha256
            or (
                str(row.get("reference_audio_sha256"))
                if row.get("reference_audio_sha256") is not None
                else (
                    str(extras.get("reference_audio_sha256"))
                    if extras.get("reference_audio_sha256") is not None
                    else None
                )
            )
        ),
        generation_settings=(
            dict(generation_settings)
            if generation_settings is not None
            else (
                dict(extras["generation_settings"])
                if isinstance(extras.get("generation_settings"), Mapping)
                else (
                    dict(row["generation_settings"])
                    if isinstance(row.get("generation_settings"), Mapping)
                    else {}
                )
            )
        ),
    )


class PrecomputedVoiceAudioResolver:
    """Exact audio resolver: never near-match, never serve stale slots.

    Lookup is keyed solely by rendered spoken-text SHA-256 plus the full
    synthesis identity. Template / response / slot identifiers may be recorded
    for provenance but cannot alone authorize audio reuse.
    """

    provider_name = "precomputed"

    def __init__(
        self,
        artifacts: Iterable[PrecomputedAudioArtifact] = (),
        *,
        audio_bytes_by_sha256: Mapping[str, bytes] | None = None,
        audio_bytes_by_id: Mapping[str, bytes] | None = None,
        byte_fetcher: Callable[[PrecomputedAudioArtifact], bytes | None] | None = None,
    ) -> None:
        self._by_match_key: dict[str, PrecomputedAudioArtifact] = {}
        self._by_text_sha: dict[str, list[PrecomputedAudioArtifact]] = {}
        self._by_template_id: dict[str, list[PrecomputedAudioArtifact]] = {}
        self._audio_bytes_by_sha256 = {
            str(key).strip().lower(): value
            for key, value in dict(audio_bytes_by_sha256 or {}).items()
            if isinstance(value, (bytes, bytearray)) and value
        }
        self._audio_bytes_by_id = {
            str(key).strip(): bytes(value)
            for key, value in dict(audio_bytes_by_id or {}).items()
            if isinstance(value, (bytes, bytearray)) and value
        }
        self._byte_fetcher = byte_fetcher
        for artifact in artifacts:
            self.index_artifact(artifact)

    @classmethod
    def from_audio_rows(
        cls,
        rows: Iterable[Mapping[str, Any] | Any],
        *,
        audio_bytes_by_sha256: Mapping[str, bytes] | None = None,
        audio_bytes_by_id: Mapping[str, bytes] | None = None,
        byte_fetcher: Callable[[PrecomputedAudioArtifact], bytes | None] | None = None,
        default_identity: Mapping[str, Any] | None = None,
    ) -> "PrecomputedVoiceAudioResolver":
        """Index Abby voice audio rows into an exact audio resolver."""

        resolver = cls(
            audio_bytes_by_sha256=audio_bytes_by_sha256,
            audio_bytes_by_id=audio_bytes_by_id,
            byte_fetcher=byte_fetcher,
        )
        defaults = dict(default_identity or {})
        for raw in rows:
            if hasattr(raw, "to_dict") and callable(raw.to_dict):
                row = dict(raw.to_dict())
            elif isinstance(raw, Mapping):
                row = dict(raw)
            else:
                raise TypeError("audio rows must be mappings or row objects with to_dict()")
            spoken = _normalize_spoken_text(str(row.get("spoken_text") or ""))
            if not spoken:
                continue
            identity = _identity_from_audio_row(
                {**defaults, **row},
                provider_version=str(
                    row.get("provider_version")
                    or defaults.get("provider_version")
                    or ""
                ),
                generation_settings=(
                    row.get("generation_settings")
                    if isinstance(row.get("generation_settings"), Mapping)
                    else defaults.get("generation_settings")
                ),
                reference_audio_sha256=(
                    str(row["reference_audio_sha256"])
                    if row.get("reference_audio_sha256") is not None
                    else (
                        str(defaults["reference_audio_sha256"])
                        if defaults.get("reference_audio_sha256") is not None
                        else None
                    )
                ),
            )
            content_sha = str(row.get("content_sha256") or "").strip().lower()
            if not content_sha:
                continue
            text_sha = str(row.get("text_sha256") or "").strip().lower() or spoken_text_sha256(
                spoken
            )
            if text_sha != spoken_text_sha256(spoken):
                # Refuse to index rows whose declared text hash does not match
                # the spoken text — exact resolver integrity gate.
                continue
            metadata = (
                dict(row["metadata"])
                if isinstance(row.get("metadata"), Mapping)
                else {}
            )
            for key in ("segment_kind", "slot_name", "slot_value"):
                if row.get(key) is not None:
                    metadata[key] = row.get(key)
            artifact = PrecomputedAudioArtifact(
                audio_id=str(row.get("audio_id") or content_sha),
                spoken_text=spoken,
                spoken_text_sha256=text_sha,
                content_sha256=content_sha,
                synthesis_identity=identity,
                match_key=synthesis_match_key(spoken, identity),
                uri=str(row["uri"]) if row.get("uri") else None,
                ipfs_cid=str(row["ipfs_cid"]) if row.get("ipfs_cid") else None,
                mime_type=str(row.get("mime_type") or "audio/wav"),
                byte_length=(
                    int(row["byte_length"])
                    if row.get("byte_length") is not None
                    else None
                ),
                template_id=str(row["template_id"]) if row.get("template_id") else None,
                response_id=str(row["response_id"]) if row.get("response_id") else None,
                metadata=metadata,
            )
            resolver.index_artifact(artifact)
        return resolver

    def index_artifact(self, artifact: PrecomputedAudioArtifact) -> None:
        if not isinstance(artifact, PrecomputedAudioArtifact):
            raise TypeError("artifact must be a PrecomputedAudioArtifact")
        # Last writer wins for identical exact keys; never fuzzy-merge.
        self._by_match_key[artifact.match_key] = artifact
        self._by_text_sha.setdefault(artifact.spoken_text_sha256, []).append(artifact)
        if artifact.template_id:
            self._by_template_id.setdefault(artifact.template_id, []).append(artifact)

    def resolve(
        self,
        spoken_text: str,
        identity: Union[SynthesisIdentity, Mapping[str, Any]],
        *,
        template_id: Optional[str] = None,
        response_id: Optional[str] = None,
    ) -> PrecomputedAudioResolution:
        """Resolve precomputed audio by exact rendered text + synthesis identity.

        Identifier-only hints (``template_id``, ``response_id``) are never
        sufficient. When they would match a different spoken text (stale slots),
        the miss reason is ``stale_slot_invalidated``.
        """

        normalized = _normalize_spoken_text(spoken_text)
        if not normalized:
            return PrecomputedAudioResolution(
                status="miss",
                reason=REASON_EMPTY_SPOKEN_TEXT,
                details={"template_id": template_id, "response_id": response_id},
            )

        synth = (
            identity
            if isinstance(identity, SynthesisIdentity)
            else SynthesisIdentity.from_mapping(identity)
        )
        text_sha = spoken_text_sha256(normalized)
        match_key = synthesis_match_key(normalized, synth)

        artifact = self._by_match_key.get(match_key)
        if artifact is None:
            # Same spoken text but different synthesis identity is not a hit.
            text_peers = self._by_text_sha.get(text_sha, ())
            if text_peers:
                return PrecomputedAudioResolution(
                    status="miss",
                    reason=REASON_SYNTHESIS_IDENTITY_MISMATCH,
                    details={
                        "requested_spoken_text_sha256": text_sha,
                        "requested_synthesis_identity": synth.to_dict(),
                        "peer_audio_ids": [item.audio_id for item in text_peers],
                        "template_id": template_id,
                        "response_id": response_id,
                    },
                )

            # Detect stale-slot cases: same template/response id exists, but
            # the rendered spoken text no longer matches (phone/address/ZIP/…).
            id_candidates: list[PrecomputedAudioArtifact] = []
            if template_id:
                id_candidates.extend(self._by_template_id.get(str(template_id), ()))
            if response_id:
                for items in self._by_text_sha.values():
                    for item in items:
                        if item.response_id == response_id:
                            id_candidates.append(item)
            stale_candidates = list(
                {
                    item.audio_id: item
                    for item in id_candidates
                    if item.spoken_text_sha256 != text_sha
                }.values()
            )
            if stale_candidates:
                return PrecomputedAudioResolution(
                    status="miss",
                    reason=REASON_STALE_SLOT_INVALIDATED,
                    details={
                        "requested_spoken_text_sha256": text_sha,
                        "requested_synthesis_identity": synth.to_dict(),
                        "stale_audio_ids": [item.audio_id for item in stale_candidates],
                        "template_id": template_id,
                        "response_id": response_id,
                        "note": (
                            "Changing a grounded phone, address, ZIP, hours, "
                            "eligibility, amount, or emergency slot invalidates "
                            "stale audio even if the template or slotted-response "
                            "identifier is unchanged."
                        ),
                    },
                )

            # Identifier-only near matches (same template, already covered by
            # stale-slot when text differs) are explicitly rejected otherwise.
            if template_id and self._by_template_id.get(str(template_id)):
                return PrecomputedAudioResolution(
                    status="miss",
                    reason=REASON_IDENTIFIER_ONLY_REJECTED,
                    details={
                        "template_id": template_id,
                        "response_id": response_id,
                        "requested_spoken_text_sha256": text_sha,
                        "note": (
                            "identifier-only precomputed-audio matching is removed; "
                            "exact rendered text and synthesis identity are required"
                        ),
                    },
                )

            if not self._by_match_key:
                reason = REASON_NO_CANDIDATES
            else:
                reason = REASON_SPOKEN_TEXT_MISMATCH
            return PrecomputedAudioResolution(
                status="miss",
                reason=reason,
                details={
                    "requested_spoken_text_sha256": text_sha,
                    "requested_synthesis_identity": synth.to_dict(),
                    "template_id": template_id,
                    "response_id": response_id,
                },
            )

        audio_bytes = self._load_bytes(artifact)
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            return PrecomputedAudioResolution(
                status="miss",
                reason=REASON_MISSING_AUDIO_BYTES,
                details={
                    "audio_id": artifact.audio_id,
                    "content_sha256": artifact.content_sha256,
                    "uri": artifact.uri,
                },
            )
        digest = sha256(audio_bytes).hexdigest()
        if digest != artifact.content_sha256:
            return PrecomputedAudioResolution(
                status="miss",
                reason=REASON_AUDIO_DIGEST_MISMATCH,
                details={
                    "audio_id": artifact.audio_id,
                    "expected_sha256": artifact.content_sha256,
                    "actual_sha256": digest,
                },
            )
        return PrecomputedAudioResolution(
            status="hit",
            reason=REASON_EXACT_MATCH,
            audio=audio_bytes,
            artifact=artifact,
            details={
                "match_key": artifact.match_key,
                "audio_id": artifact.audio_id,
                "spoken_text_sha256": artifact.spoken_text_sha256,
                "content_sha256": artifact.content_sha256,
            },
        )

    def _load_bytes(self, artifact: PrecomputedAudioArtifact) -> bytes | None:
        by_id = self._audio_bytes_by_id.get(artifact.audio_id)
        if isinstance(by_id, bytes) and by_id:
            return by_id
        by_sha = self._audio_bytes_by_sha256.get(artifact.content_sha256)
        if isinstance(by_sha, bytes) and by_sha:
            return by_sha
        if self._byte_fetcher is not None:
            try:
                fetched = self._byte_fetcher(artifact)
            except Exception:
                return None
            if isinstance(fetched, (bytes, bytearray)) and fetched:
                return bytes(fetched)
        return None

    def resolve_or_none(
        self,
        spoken_text: str,
        identity: Union[SynthesisIdentity, Mapping[str, Any]],
        **kwargs: Any,
    ) -> Optional[bytes]:
        """Convenience wrapper returning audio bytes or ``None`` on miss."""

        resolution = self.resolve(spoken_text, identity, **kwargs)
        return resolution.audio if resolution.hit else None


__all__ = [
    "EXACT_AUDIO_RESOLVER_EVIDENCE_TERM",
    "G019_AUTHORITATIVE_EVIDENCE_MAP",
    "PrecomputedAudioArtifact",
    "PrecomputedAudioResolution",
    "PrecomputedVoiceAudioResolver",
    "REASON_AUDIO_DIGEST_MISMATCH",
    "REASON_EMPTY_SPOKEN_TEXT",
    "REASON_EXACT_MATCH",
    "REASON_IDENTIFIER_ONLY_REJECTED",
    "REASON_MISSING_AUDIO_BYTES",
    "REASON_NO_CANDIDATES",
    "REASON_SPOKEN_TEXT_MISMATCH",
    "REASON_STALE_SLOT_INVALIDATED",
    "REASON_SYNTHESIS_IDENTITY_MISMATCH",
    "RUNTIME_RESOLUTION_EVIDENCE_TERM",
    "STALE_SLOT_REGRESSION_TEST_EVIDENCE_TERM",
    "SynthesisIdentity",
    "spoken_text_sha256",
    "synthesis_match_key",
]

"""Select spoken action outcomes from the Abby library after execute/deny.

VOICE-ACTION-026 / VOICE-ACTION-G120

After an authority-plane execute or deny path yields an ``ActionReceipt``,
spoken text is chosen as follows:

1. Map the receipt status to an action-link outcome role
   (``success`` | ``denied`` | ``failed`` | ``cancelled`` | ``unknown``).
2. Prefer the matching library outcome frame for the logical action when one
   exists (from the VOICE-ACTION-024 speech-frame corpus).
3. Fall back to a safe, status-accurate generic utterance when the library has
   no frame, the frame is empty, or the frame would invent transfer success.
4. Optionally resolve precomputed audio via ``PrecomputedVoiceAudioResolver``
   when the exact spoken text + synthesis identity match.

This module never claims live-agent transfer completion unless the receipt
status is ``succeeded`` (handoff truthfulness / ``allows_spoken_success``).
Outcome speech requires a receipt: missing receipts always map to ``unknown``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .contracts import ActionReceipt, ActionStatus

# Residual discoverability anchors.
TASK_ID: Final = "VOICE-ACTION-026"
SOURCE_LABEL: Final = "voice-action-026/outcome-speech"
SCHEMA: Final = "voice-action/outcome-speech-selection@1"

# Action-link outcome roles (docs/voice_action_dag/schemas/action-link-v1.md).
OUTCOME_ROLE_SUCCESS: Final = "success"
OUTCOME_ROLE_DENIED: Final = "denied"
OUTCOME_ROLE_FAILED: Final = "failed"
OUTCOME_ROLE_CANCELLED: Final = "cancelled"
OUTCOME_ROLE_UNKNOWN: Final = "unknown"

OUTCOME_ROLES: Final = frozenset(
    {
        OUTCOME_ROLE_SUCCESS,
        OUTCOME_ROLE_DENIED,
        OUTCOME_ROLE_FAILED,
        OUTCOME_ROLE_CANCELLED,
        OUTCOME_ROLE_UNKNOWN,
    }
)

# Speech-frame corpus roles (docs/phone_dialog_generation/action_speech_frames.jsonl).
LIBRARY_ROLE_SUCCESS: Final = "success"
LIBRARY_ROLE_DENY: Final = "deny"
LIBRARY_ROLE_FAIL: Final = "fail"

# Action-link role → library corpus role (cancelled/unknown have no corpus rows).
_LINK_ROLE_TO_LIBRARY_ROLE: Final[Mapping[str, str]] = MappingProxyType(
    {
        OUTCOME_ROLE_SUCCESS: LIBRARY_ROLE_SUCCESS,
        OUTCOME_ROLE_DENIED: LIBRARY_ROLE_DENY,
        OUTCOME_ROLE_FAILED: LIBRARY_ROLE_FAIL,
    }
)

# Accept common aliases from callers / older corpora.
_LIBRARY_ROLE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "success": LIBRARY_ROLE_SUCCESS,
        "deny": LIBRARY_ROLE_DENY,
        "denied": LIBRARY_ROLE_DENY,
        "fail": LIBRARY_ROLE_FAIL,
        "failed": LIBRARY_ROLE_FAIL,
    }
)

# Logical actions that must never invent warm-transfer completion.
HANDOFF_LOGICAL_ACTIONS: Final = frozenset({"handoff_live_agent"})

DEFAULT_FRAMES_REL: Final = "docs/phone_dialog_generation/action_speech_frames.jsonl"

# Selection provenance reasons (stable for tests / receipts).
REASON_LIBRARY_HIT: Final = "library_outcome_frame"
REASON_LIBRARY_MISSING: Final = "library_frame_missing"
REASON_LIBRARY_EMPTY: Final = "library_frame_empty"
REASON_TRANSFER_SUCCESS_BLOCKED: Final = "transfer_success_blocked"
REASON_NO_RECEIPT: Final = "receipt_missing_unknown"
REASON_SAFE_FALLBACK: Final = "safe_fallback"
REASON_AUDIO_HIT: Final = "precomputed_audio_hit"
REASON_AUDIO_MISS: Final = "precomputed_audio_miss"
REASON_AUDIO_SKIPPED: Final = "precomputed_audio_skipped"

# Phrases that claim a completed live transfer (forbidden without succeeded).
_TRANSFER_SUCCESS_CLAIM_RE: Final = re.compile(
    r"(?is)\b(?:"
    r"transfer(?:\s+is|\s+was)?\s+complete"
    r"|transfer(?:\s+has\s+been)?\s+completed"
    r"|warm\s+transfer\s+(?:is\s+)?complete"
    r"|you(?:'re| are)\s+(?:now\s+)?connected"
    r"|i(?:'ve| have)\s+connected\s+you"
    r"|connected\s+you\s+to\s+(?:a\s+)?(?:live\s+)?(?:agent|specialist)"
    r"|live\s+(?:agent|specialist)\s+(?:is\s+)?(?:on\s+the\s+line|connected)"
    r"|you\s+are\s+speaking\s+with\s+(?:a\s+)?(?:live\s+)?(?:agent|specialist)"
    r")\b"
)

# Safe status-accurate fallbacks when the library has no usable frame.
_SAFE_FALLBACK_BY_ROLE: Final[Mapping[str, str]] = MappingProxyType(
    {
        OUTCOME_ROLE_SUCCESS: "That action completed successfully.",
        OUTCOME_ROLE_DENIED: "Okay, I will not take that action.",
        OUTCOME_ROLE_FAILED: (
            "I could not complete that action right now. "
            "Please try again in a moment."
        ),
        OUTCOME_ROLE_CANCELLED: "Okay, I cancelled that action.",
        OUTCOME_ROLE_UNKNOWN: (
            "I do not yet have confirmation on that request. "
            "I will not treat it as complete until a receipt confirms it."
        ),
    }
)

# Handoff-specific safe fallbacks (never invent transfer success).
_HANDOFF_SAFE_FALLBACK_BY_ROLE: Final[Mapping[str, str]] = MappingProxyType(
    {
        OUTCOME_ROLE_SUCCESS: (
            "Your request to speak with a live specialist has been confirmed "
            "by the provider."
        ),
        OUTCOME_ROLE_DENIED: (
            "Okay, I will not request a live specialist right now. "
            "We can continue here, or you can ask again later."
        ),
        OUTCOME_ROLE_FAILED: (
            "I could not complete the live specialist request just now. "
            "Please try again, or stay on the line for more options."
        ),
        OUTCOME_ROLE_CANCELLED: (
            "Okay, I cancelled the live specialist request."
        ),
        OUTCOME_ROLE_UNKNOWN: (
            "I have a live specialist request on file, but I will not treat "
            "the transfer as complete until a provider confirms it."
        ),
    }
)


def normalize_spoken_text(text: str) -> str:
    """Normalize spoken text for comparison (NFC + strip; collapse whitespace)."""

    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(value.split())


def claims_transfer_success(spoken_text: str) -> bool:
    """Return True when *spoken_text* asserts a completed live transfer."""

    text = normalize_spoken_text(spoken_text)
    if not text:
        return False
    return _TRANSFER_SUCCESS_CLAIM_RE.search(text) is not None


def allows_spoken_success(
    status_or_receipt: ActionStatus | ActionReceipt | str | None,
) -> bool:
    """Return True only when spoken transfer-success wording is authorized.

    Mirrors the handoff adapter gate: only ``succeeded`` authorizes success
    frames that claim a completed live-agent connection.
    """

    status = coerce_action_status(status_or_receipt)
    return status is ActionStatus.SUCCEEDED


def spoken_outcome_role(
    status_or_receipt: ActionStatus | ActionReceipt | str | None,
) -> str:
    """Map a receipt status to an action-link outcome frame role.

    Roles: ``success`` | ``denied`` | ``failed`` | ``cancelled`` | ``unknown``.
    Missing / indeterminate receipts always yield ``unknown`` (never invent
    success).
    """

    status = coerce_action_status(status_or_receipt)
    if status is ActionStatus.SUCCEEDED:
        return OUTCOME_ROLE_SUCCESS
    if status is ActionStatus.DENIED:
        return OUTCOME_ROLE_DENIED
    if status is ActionStatus.FAILED:
        return OUTCOME_ROLE_FAILED
    if status is ActionStatus.CANCELLED:
        return OUTCOME_ROLE_CANCELLED
    # accepted / started / unknown / timed_out / compensated / missing
    return OUTCOME_ROLE_UNKNOWN


def coerce_action_status(
    status_or_receipt: ActionStatus | ActionReceipt | str | None,
) -> ActionStatus | None:
    """Coerce receipts, enums, or strings to ``ActionStatus`` (or None)."""

    if status_or_receipt is None:
        return None
    if isinstance(status_or_receipt, ActionStatus):
        return status_or_receipt
    if isinstance(status_or_receipt, ActionReceipt):
        return status_or_receipt.status
    if isinstance(status_or_receipt, str):
        try:
            return ActionStatus(status_or_receipt.strip().lower())
        except ValueError:
            return None
    # Duck-typed receipt / request with a status attribute.
    status_attr = getattr(status_or_receipt, "status", None)
    if isinstance(status_attr, ActionStatus):
        return status_attr
    if isinstance(status_attr, str):
        try:
            return ActionStatus(status_attr.strip().lower())
        except ValueError:
            return None
    return None


def library_role_for_outcome(outcome_role: str) -> str | None:
    """Map an action-link outcome role to a speech-frame corpus role, if any."""

    role = str(outcome_role or "").strip().lower()
    return _LINK_ROLE_TO_LIBRARY_ROLE.get(role)


def outcome_frame_id(logical_action: str, library_role: str) -> str:
    """Return the canonical library outcome frame id for *logical_action*/*role*."""

    action = str(logical_action or "").strip()
    role = _LIBRARY_ROLE_ALIASES.get(str(library_role or "").strip().lower(), "")
    if not action or not role:
        raise ValueError("logical_action and library_role are required")
    suffix = {
        LIBRARY_ROLE_SUCCESS: "success",
        LIBRARY_ROLE_DENY: "denied",
        LIBRARY_ROLE_FAIL: "failed",
    }[role]
    return f"frame.action.outcome.{action}.{suffix}.v1"


def safe_fallback_spoken_text(
    *,
    logical_action: str,
    outcome_role: str,
) -> str:
    """Return a safe, status-accurate fallback utterance (never invents transfer success)."""

    role = str(outcome_role or "").strip().lower()
    if role not in OUTCOME_ROLES:
        role = OUTCOME_ROLE_UNKNOWN
    action = str(logical_action or "").strip()
    if action in HANDOFF_LOGICAL_ACTIONS:
        return _HANDOFF_SAFE_FALLBACK_BY_ROLE[role]
    return _SAFE_FALLBACK_BY_ROLE[role]


def default_action_speech_frames_path() -> Path | None:
    """Locate the pilot action speech-frame corpus when present on disk."""

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # package → ipfs_accelerate_py → repo root (common layouts)
    for parent in here.parents:
        candidates.append(parent / DEFAULT_FRAMES_REL)
        candidates.append(parent / "docs" / "phone_dialog_generation" / "action_speech_frames.jsonl")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


@dataclass(frozen=True, slots=True)
class OutcomeSpeechFrame:
    """One library-backed outcome (or confirm) speech frame."""

    frame_id: str
    logical_action: str
    role: str
    spoken_text: str
    source: str = SOURCE_LABEL
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id or "").strip()
        logical_action = str(self.logical_action or "").strip()
        role_raw = str(self.role or "").strip().lower()
        role = _LIBRARY_ROLE_ALIASES.get(role_raw, role_raw)
        spoken = normalize_spoken_text(self.spoken_text)
        if not frame_id:
            raise ValueError("frame_id is required")
        if not logical_action:
            raise ValueError("logical_action is required")
        if role not in {
            LIBRARY_ROLE_SUCCESS,
            LIBRARY_ROLE_DENY,
            LIBRARY_ROLE_FAIL,
            "confirm",
        }:
            raise ValueError(f"unsupported speech-frame role: {self.role!r}")
        if not spoken:
            raise ValueError(f"spoken_text is required for frame {frame_id}")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "logical_action", logical_action)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "spoken_text", spoken)
        object.__setattr__(self, "source", str(self.source or SOURCE_LABEL))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(k): str(v)
                    for k, v in dict(self.metadata or {}).items()
                    if isinstance(k, str)
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "logical_action": self.logical_action,
            "metadata": dict(self.metadata),
            "role": self.role,
            "source": self.source,
            "spoken_text": self.spoken_text,
        }


class OutcomeSpeechLibrary:
    """In-memory index of action speech frames keyed by (logical_action, role)."""

    def __init__(self, frames: Iterable[OutcomeSpeechFrame] = ()) -> None:
        self._by_key: dict[tuple[str, str], OutcomeSpeechFrame] = {}
        self._by_frame_id: dict[str, OutcomeSpeechFrame] = {}
        for frame in frames:
            self.index(frame)

    def index(self, frame: OutcomeSpeechFrame) -> None:
        key = (frame.logical_action, frame.role)
        self._by_key[key] = frame
        self._by_frame_id[frame.frame_id] = frame

    @property
    def frame_count(self) -> int:
        return len(self._by_key)

    def get(
        self,
        logical_action: str,
        library_role: str,
    ) -> OutcomeSpeechFrame | None:
        action = str(logical_action or "").strip()
        role = _LIBRARY_ROLE_ALIASES.get(str(library_role or "").strip().lower(), "")
        if not action or not role:
            return None
        return self._by_key.get((action, role))

    def get_by_frame_id(self, frame_id: str) -> OutcomeSpeechFrame | None:
        return self._by_frame_id.get(str(frame_id or "").strip())

    def outcome_roles_for(self, logical_action: str) -> frozenset[str]:
        action = str(logical_action or "").strip()
        return frozenset(
            role for (act, role) in self._by_key if act == action and role != "confirm"
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        outcome_roles_only: bool = True,
    ) -> "OutcomeSpeechLibrary":
        """Build a library from speech-frame JSONL row mappings."""

        frames: list[OutcomeSpeechFrame] = []
        for index, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise TypeError(f"frame record at index {index} must be a mapping")
            role_raw = str(row.get("role") or "").strip().lower()
            role = _LIBRARY_ROLE_ALIASES.get(role_raw, role_raw)
            if outcome_roles_only and role == "confirm":
                continue
            if role not in {
                LIBRARY_ROLE_SUCCESS,
                LIBRARY_ROLE_DENY,
                LIBRARY_ROLE_FAIL,
                "confirm",
            }:
                continue
            spoken = normalize_spoken_text(str(row.get("spoken_text") or ""))
            logical_action = str(row.get("logical_action") or "").strip()
            frame_id = str(row.get("frame_id") or "").strip()
            if not frame_id and logical_action and role in {
                LIBRARY_ROLE_SUCCESS,
                LIBRARY_ROLE_DENY,
                LIBRARY_ROLE_FAIL,
            }:
                frame_id = outcome_frame_id(logical_action, role)
            if not frame_id or not logical_action or not spoken:
                continue
            meta: dict[str, str] = {}
            for key in ("audio_status", "schema", "schema_version", "task_id", "source"):
                if row.get(key) is not None:
                    meta[key] = str(row.get(key))
            frames.append(
                OutcomeSpeechFrame(
                    frame_id=frame_id,
                    logical_action=logical_action,
                    role=role,
                    spoken_text=spoken,
                    source=str(row.get("source") or SOURCE_LABEL),
                    metadata=meta,
                )
            )
        return cls(frames)

    @classmethod
    def from_jsonl_path(
        cls,
        path: Path | str,
        *,
        outcome_roles_only: bool = True,
    ) -> "OutcomeSpeechLibrary":
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"action speech-frame corpus not found: {file_path}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed speech-frame JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"speech-frame row must be an object at line {line_number}"
                )
            rows.append(payload)
        return cls.from_records(rows, outcome_roles_only=outcome_roles_only)

    @classmethod
    def from_default_corpus(
        cls,
        *,
        path: Path | str | None = None,
        outcome_roles_only: bool = True,
    ) -> "OutcomeSpeechLibrary":
        """Load the pilot corpus when available; otherwise return an empty library."""

        if path is not None:
            return cls.from_jsonl_path(path, outcome_roles_only=outcome_roles_only)
        default = default_action_speech_frames_path()
        if default is None:
            return cls()
        return cls.from_jsonl_path(default, outcome_roles_only=outcome_roles_only)


@dataclass(frozen=True, slots=True)
class OutcomeSpeechSelection:
    """Deterministic spoken-outcome selection after execute/deny."""

    spoken_text: str
    outcome_role: str
    logical_action: str
    source: str
    reason: str
    frame_id: str | None = None
    library_role: str | None = None
    receipt_status: str | None = None
    spoken_success_allowed: bool = False
    audio: bytes | None = None
    audio_hit: bool = False
    audio_reason: str = REASON_AUDIO_SKIPPED
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spoken = normalize_spoken_text(self.spoken_text)
        role = str(self.outcome_role or "").strip().lower()
        if not spoken:
            raise ValueError("spoken_text is required")
        if role not in OUTCOME_ROLES:
            raise ValueError(f"invalid outcome_role: {self.outcome_role!r}")
        if self.audio_hit and not (isinstance(self.audio, bytes) and self.audio):
            raise ValueError("audio_hit requires non-empty audio bytes")
        if not self.audio_hit and self.audio is not None:
            raise ValueError("miss selections must not carry audio bytes")
        # Hard safety: never emit transfer-success claims when not authorized.
        if (
            not self.spoken_success_allowed
            and claims_transfer_success(spoken)
        ):
            raise ValueError(
                "selection invents transfer success without succeeded receipt"
            )
        object.__setattr__(self, "spoken_text", spoken)
        object.__setattr__(self, "outcome_role", role)
        object.__setattr__(self, "logical_action", str(self.logical_action or "").strip())
        object.__setattr__(self, "source", str(self.source or REASON_SAFE_FALLBACK))
        object.__setattr__(self, "reason", str(self.reason or REASON_SAFE_FALLBACK))
        object.__setattr__(
            self,
            "frame_id",
            str(self.frame_id).strip() if self.frame_id else None,
        )
        object.__setattr__(
            self,
            "library_role",
            str(self.library_role).strip() if self.library_role else None,
        )
        object.__setattr__(
            self,
            "receipt_status",
            str(self.receipt_status).strip() if self.receipt_status else None,
        )
        object.__setattr__(
            self,
            "audio_reason",
            str(self.audio_reason or REASON_AUDIO_SKIPPED),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {
                    str(k): str(v)
                    for k, v in dict(self.metadata or {}).items()
                    if isinstance(k, str)
                }
            ),
        )

    @property
    def prefers_library(self) -> bool:
        return self.source == "library"

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_hit": self.audio_hit,
            "audio_reason": self.audio_reason,
            "audio_sha256": None,  # filled by callers that hash audio if needed
            "frame_id": self.frame_id,
            "library_role": self.library_role,
            "logical_action": self.logical_action,
            "metadata": dict(self.metadata),
            "outcome_role": self.outcome_role,
            "reason": self.reason,
            "receipt_status": self.receipt_status,
            "schema": SCHEMA,
            "source": self.source,
            "spoken_success_allowed": self.spoken_success_allowed,
            "spoken_text": self.spoken_text,
            "task_id": TASK_ID,
        }


def _resolve_precomputed_audio(
    *,
    spoken_text: str,
    frame_id: str | None,
    audio_resolver: Any | None,
    synthesis_identity: Any | None,
) -> tuple[bytes | None, bool, str]:
    """Attempt exact precomputed audio resolution; miss safely on any failure."""

    if audio_resolver is None or synthesis_identity is None:
        return None, False, REASON_AUDIO_SKIPPED
    resolve = getattr(audio_resolver, "resolve", None)
    if not callable(resolve):
        return None, False, REASON_AUDIO_SKIPPED
    try:
        resolution = resolve(
            spoken_text,
            synthesis_identity,
            template_id=frame_id,
        )
    except Exception:
        return None, False, REASON_AUDIO_MISS
    hit = bool(getattr(resolution, "hit", False))
    audio = getattr(resolution, "audio", None)
    reason = str(getattr(resolution, "reason", "") or "")
    if hit and isinstance(audio, (bytes, bytearray)) and audio:
        return bytes(audio), True, reason or REASON_AUDIO_HIT
    return None, False, reason or REASON_AUDIO_MISS


def select_outcome_speech(
    *,
    logical_action: str,
    receipt: ActionStatus | ActionReceipt | str | None,
    library: OutcomeSpeechLibrary | Mapping[tuple[str, str], str] | None = None,
    outcome_frame_ids: Mapping[str, str] | None = None,
    audio_resolver: Any | None = None,
    synthesis_identity: Any | None = None,
) -> OutcomeSpeechSelection:
    """Select spoken outcome text after execute/deny.

    Preference order:
    1. Library outcome frame for (logical_action, mapped library role)
    2. Safe status-accurate fallback (handoff-aware)
    3. Never invent transfer success without a ``succeeded`` receipt
    4. Optionally attach exact-match precomputed audio when available
    """

    action = str(logical_action or "").strip()
    if not action:
        raise ValueError("logical_action is required")

    status = coerce_action_status(receipt)
    receipt_status = status.value if status is not None else None
    spoken_success_allowed = allows_spoken_success(status)
    outcome_role = spoken_outcome_role(status)

    library_role = library_role_for_outcome(outcome_role)
    frame: OutcomeSpeechFrame | None = None
    spoken_text: str
    frame_id: str | None = None
    library_role_out: str | None = None
    source = "safe_fallback"
    # Outcome speech requires a receipt: missing → unknown (doctrine).
    reason = REASON_NO_RECEIPT if status is None else REASON_SAFE_FALLBACK

    # Optional explicit frame-id map from an action-link projection.
    preferred_frame_id: str | None = None
    if outcome_frame_ids and outcome_role in outcome_frame_ids:
        preferred_frame_id = str(outcome_frame_ids[outcome_role] or "").strip() or None

    # Resolve library lookup surface.
    speech_library: OutcomeSpeechLibrary | None
    if library is None:
        speech_library = None
    elif isinstance(library, OutcomeSpeechLibrary):
        speech_library = library
    elif isinstance(library, Mapping):
        # Compact test helper: {(logical_action, library_role): spoken_text}
        built: list[OutcomeSpeechFrame] = []
        for key, text in library.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            la, lr = str(key[0]), str(key[1])
            role = _LIBRARY_ROLE_ALIASES.get(lr.strip().lower(), lr.strip().lower())
            if role not in {LIBRARY_ROLE_SUCCESS, LIBRARY_ROLE_DENY, LIBRARY_ROLE_FAIL}:
                continue
            spoken = normalize_spoken_text(str(text))
            if not spoken:
                continue
            built.append(
                OutcomeSpeechFrame(
                    frame_id=outcome_frame_id(la, role),
                    logical_action=la,
                    role=role,
                    spoken_text=spoken,
                )
            )
        speech_library = OutcomeSpeechLibrary(built)
    else:
        raise TypeError(
            "library must be OutcomeSpeechLibrary, a mapping, or None"
        )

    if speech_library is not None and library_role is not None:
        if preferred_frame_id:
            frame = speech_library.get_by_frame_id(preferred_frame_id)
            # Frame id may point at a different action/role — only accept match.
            if frame is not None and (
                frame.logical_action != action or frame.role != library_role
            ):
                frame = None
        if frame is None:
            frame = speech_library.get(action, library_role)

    if frame is not None:
        candidate = frame.spoken_text
        # Refuse library wording that invents transfer success without authority.
        if not spoken_success_allowed and claims_transfer_success(candidate):
            spoken_text = safe_fallback_spoken_text(
                logical_action=action,
                outcome_role=outcome_role,
            )
            source = "safe_fallback"
            reason = REASON_TRANSFER_SUCCESS_BLOCKED
            frame_id = None
            library_role_out = None
        elif not candidate:
            spoken_text = safe_fallback_spoken_text(
                logical_action=action,
                outcome_role=outcome_role,
            )
            source = "safe_fallback"
            reason = REASON_LIBRARY_EMPTY
            frame_id = None
            library_role_out = None
        else:
            spoken_text = candidate
            source = "library"
            reason = REASON_LIBRARY_HIT
            frame_id = frame.frame_id
            library_role_out = frame.role
    else:
        spoken_text = safe_fallback_spoken_text(
            logical_action=action,
            outcome_role=outcome_role,
        )
        source = "safe_fallback"
        if status is None:
            reason = REASON_NO_RECEIPT
        elif library_role is None:
            reason = REASON_SAFE_FALLBACK
        else:
            reason = REASON_LIBRARY_MISSING
        # Do not claim a frame_id unless library text was actually selected.
        frame_id = None
        library_role_out = None

    # Final hard gate: never emit transfer-success claims without authority.
    if not spoken_success_allowed and claims_transfer_success(spoken_text):
        demoted_role = (
            OUTCOME_ROLE_UNKNOWN
            if outcome_role == OUTCOME_ROLE_SUCCESS
            else outcome_role
        )
        spoken_text = safe_fallback_spoken_text(
            logical_action=action,
            outcome_role=demoted_role,
        )
        if outcome_role == OUTCOME_ROLE_SUCCESS:
            outcome_role = OUTCOME_ROLE_UNKNOWN
        source = "safe_fallback"
        reason = REASON_TRANSFER_SUCCESS_BLOCKED
        frame_id = None
        library_role_out = None

    audio, audio_hit, audio_reason = _resolve_precomputed_audio(
        spoken_text=spoken_text,
        frame_id=frame_id,
        audio_resolver=audio_resolver,
        synthesis_identity=synthesis_identity,
    )

    meta = {
        "task_id": TASK_ID,
        "selection_source": source,
        "is_handoff": "true" if action in HANDOFF_LOGICAL_ACTIONS else "false",
    }
    if preferred_frame_id:
        meta["preferred_frame_id"] = preferred_frame_id

    return OutcomeSpeechSelection(
        spoken_text=spoken_text,
        outcome_role=outcome_role,
        logical_action=action,
        source=source,
        reason=reason,
        frame_id=frame_id,
        library_role=library_role_out,
        receipt_status=receipt_status,
        spoken_success_allowed=spoken_success_allowed,
        audio=audio,
        audio_hit=audio_hit,
        audio_reason=audio_reason,
        metadata=meta,
    )


def select_outcome_speech_from_default_library(
    *,
    logical_action: str,
    receipt: ActionStatus | ActionReceipt | str | None,
    frames_path: Path | str | None = None,
    outcome_frame_ids: Mapping[str, str] | None = None,
    audio_resolver: Any | None = None,
    synthesis_identity: Any | None = None,
) -> OutcomeSpeechSelection:
    """Convenience wrapper that loads the default pilot speech-frame corpus."""

    library = OutcomeSpeechLibrary.from_default_corpus(path=frames_path)
    return select_outcome_speech(
        logical_action=logical_action,
        receipt=receipt,
        library=library,
        outcome_frame_ids=outcome_frame_ids,
        audio_resolver=audio_resolver,
        synthesis_identity=synthesis_identity,
    )


__all__ = [
    "DEFAULT_FRAMES_REL",
    "HANDOFF_LOGICAL_ACTIONS",
    "LIBRARY_ROLE_DENY",
    "LIBRARY_ROLE_FAIL",
    "LIBRARY_ROLE_SUCCESS",
    "OUTCOME_ROLES",
    "OUTCOME_ROLE_CANCELLED",
    "OUTCOME_ROLE_DENIED",
    "OUTCOME_ROLE_FAILED",
    "OUTCOME_ROLE_SUCCESS",
    "OUTCOME_ROLE_UNKNOWN",
    "REASON_AUDIO_HIT",
    "REASON_AUDIO_MISS",
    "REASON_AUDIO_SKIPPED",
    "REASON_LIBRARY_EMPTY",
    "REASON_LIBRARY_HIT",
    "REASON_LIBRARY_MISSING",
    "REASON_NO_RECEIPT",
    "REASON_SAFE_FALLBACK",
    "REASON_TRANSFER_SUCCESS_BLOCKED",
    "SCHEMA",
    "SOURCE_LABEL",
    "TASK_ID",
    "OutcomeSpeechFrame",
    "OutcomeSpeechLibrary",
    "OutcomeSpeechSelection",
    "allows_spoken_success",
    "claims_transfer_success",
    "coerce_action_status",
    "default_action_speech_frames_path",
    "library_role_for_outcome",
    "normalize_spoken_text",
    "outcome_frame_id",
    "safe_fallback_spoken_text",
    "select_outcome_speech",
    "select_outcome_speech_from_default_library",
    "spoken_outcome_role",
]

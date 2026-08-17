"""StateProvider@1 — shared-state provider contract for MCP++ consistency modes.

Every backend that serves a :class:`StateRef@1` implements this interface. The
provider is mode-aware: the concrete class declares exactly one consistency
mode from ADR-0004 / plan KD-8. Higher-level orchestration may select among
providers, but a single provider instance never silently cross modes.

This module owns only the contract, closed errors, and StateRef structural
helpers. Mode-specific storage (immutable CID, SQLite, Automerge, consensus)
lives in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

STATE_PROVIDER_INTERFACE = "StateProvider@1"
STATE_REF_SCHEMA = "mcp++/state/state-ref@1"

ALLOWED_CONSISTENCY_MODES = frozenset(
    {
        "immutable",
        "single_authority",
        "causal",
        "crdt",
        "consensus",
    }
)

class StateError(Exception):
    """Base error for StateProvider operations."""


class StateNotFoundError(StateError, KeyError):
    """Raised when a requested CID or logical state id is not present."""


class StateIntegrityError(StateError):
    """Raised when stored or supplied bytes do not match their declared CID."""


class StateMutationError(StateError):
    """Raised when a write would mutate bytes under an existing CID identity."""


class StateModeError(StateError):
    """Raised when a StateRef mode is missing, unknown, or provider-incompatible."""


class StateRefError(StateError):
    """Raised when a StateRef document is structurally invalid."""


@dataclass(frozen=True)
class StateWriteResult:
    """Outcome of an append-only or mode-specific write.

    Attributes:
        cid: Content-addressed identity of the written value.
        created: True when new bytes were stored; False for idempotent replay
            of identical content under the same CID.
        byte_length: Length of the canonical stored payload.
        mode: Consistency mode that accepted the write.
        provider: Backend label that performed the write.
    """

    cid: str
    created: bool
    byte_length: int
    mode: str
    provider: str


def is_portable_cid(value: object) -> bool:
    """Return True when ``value`` looks like a portable MCP++ CID string."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < 46 or len(text) > 128:
        return False
    if text.startswith("Qm"):
        # CIDv0: base58btc, fixed length 46.
        alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        return len(text) == 46 and all(ch in alphabet for ch in text)
    if text.startswith("b") and len(text) >= 59:
        # CIDv1 base32 (lowercase, no padding).
        alphabet = set("abcdefghijklmnopqrstuvwxyz234567")
        return all(ch in alphabet for ch in text[1:])
    return False


def require_portable_cid(value: object, *, field: str = "cid") -> str:
    """Validate and return a portable CID string or raise :class:`StateRefError`."""

    if not is_portable_cid(value):
        raise StateRefError(f"{field} must be a portable CIDv0/CIDv1 string")
    return str(value).strip()


def validate_state_ref(
    state_ref: Mapping[str, Any],
    *,
    require_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Structurally validate a StateRef@1 mapping and return a shallow copy.

    Fail-closed rules mirror ``state-ref-1.schema.json`` for the fields this
    layer needs: ``schema``, ``id``, ``mode``, and optional ``root_cid``.
    Full JSON Schema evaluation is left to dedicated schema tooling.
    """

    if not isinstance(state_ref, Mapping):
        raise StateRefError("state_ref must be a mapping")

    schema = state_ref.get("schema")
    if schema is not None and schema != STATE_REF_SCHEMA:
        raise StateRefError(
            f"state_ref.schema must be {STATE_REF_SCHEMA!r}, got {schema!r}"
        )

    state_id = state_ref.get("id")
    if not isinstance(state_id, str) or not state_id.strip():
        raise StateRefError("state_ref.id must be a non-empty string")

    mode = state_ref.get("mode")
    if mode is None or (isinstance(mode, str) and not mode.strip()):
        raise StateModeError("state_ref.mode is required")
    if not isinstance(mode, str):
        raise StateModeError("state_ref.mode must be a string")
    if mode not in ALLOWED_CONSISTENCY_MODES:
        raise StateModeError(
            f"state_ref.mode {mode!r} is not one of "
            f"{sorted(ALLOWED_CONSISTENCY_MODES)}"
        )
    if require_mode is not None and mode != require_mode:
        raise StateModeError(
            f"state_ref.mode is {mode!r}, provider requires {require_mode!r}"
        )

    root_cid = state_ref.get("root_cid", None)
    if root_cid is not None:
        require_portable_cid(root_cid, field="root_cid")

    schema_cid = state_ref.get("schema_cid", None)
    if schema_cid is not None:
        require_portable_cid(schema_cid, field="schema_cid")

    parents = state_ref.get("parents", None)
    if parents is not None:
        if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
            raise StateRefError("state_ref.parents must be a sequence of CIDs")
        for index, parent in enumerate(parents):
            require_portable_cid(parent, field=f"parents[{index}]")

    out = dict(state_ref)
    out["id"] = state_id.strip()
    out["mode"] = mode
    if schema is None:
        out["schema"] = STATE_REF_SCHEMA
    return out


class StateProvider(ABC):
    """Abstract StateProvider@1 contract.

    Implementations MUST:

    * declare exactly one consistency ``mode``;
    * reject writes that would violate that mode (e.g. in-place mutation under
      ``immutable``);
    * verify content against CIDs on fetch paths that return payload bytes.
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """Consistency mode this provider implements (ADR-0004 closed enum)."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable backend label (e.g. ``immutable-cid``, ``sqlite-authority``)."""

    @property
    def interface(self) -> str:
        """Wire interface label for this provider family."""

        return STATE_PROVIDER_INTERFACE

    @abstractmethod
    def write(
        self,
        payload: Mapping[str, Any] | bytes,
        *,
        expected_cid: Optional[str] = None,
        state_id: Optional[str] = None,
    ) -> StateWriteResult:
        """Persist payload under content-addressed identity.

        For ``immutable`` mode, writes are append-only: identical content is
        idempotent; differing content under an existing CID is rejected.
        """

    @abstractmethod
    def fetch(self, cid: str) -> bytes:
        """Return stored bytes for ``cid`` after verifying content address."""

    @abstractmethod
    def fetch_json(self, cid: str) -> dict[str, Any]:
        """Return a JSON object payload for ``cid`` after CID verification."""

    @abstractmethod
    def has(self, cid: str) -> bool:
        """Return True when ``cid`` is present in the local store."""

    @abstractmethod
    def bind_ref(self, state_ref: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and bind a StateRef@1 to this provider's mode.

        Returns a normalized StateRef mapping. Implementations MAY record the
        logical ``id`` for subsequent head tracking without mutating stored
        CIDs.
        """

    def get_ref(self, state_id: str) -> Optional[dict[str, Any]]:
        """Return the latest bound StateRef for ``state_id``, if tracked."""

        return None

    def stats(self) -> MutableMapping[str, Any]:
        """Return deterministic provider diagnostics."""

        return {
            "interface": self.interface,
            "provider": self.provider_id,
            "mode": self.mode,
        }


__all__ = [
    "ALLOWED_CONSISTENCY_MODES",
    "STATE_PROVIDER_INTERFACE",
    "STATE_REF_SCHEMA",
    "StateError",
    "StateIntegrityError",
    "StateModeError",
    "StateMutationError",
    "StateNotFoundError",
    "StateProvider",
    "StateRefError",
    "StateWriteResult",
    "is_portable_cid",
    "require_portable_cid",
    "validate_state_ref",
]

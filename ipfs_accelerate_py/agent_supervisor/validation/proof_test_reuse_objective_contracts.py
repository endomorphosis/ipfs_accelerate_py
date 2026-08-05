"""Strict objective-completion artifact contracts for proof-backed test reuse.

This module is the semantic boundary for PTR completion envelopes.  It does
**not** assemble goal evidence or run the closeout reconciler; those belong to
later tasks.  It does:

* distinguish Git tree, repository-forest, and objective-completion-tree
  identity domains;
* bind repository ID plus exact per-goal objective, analyzer, configuration,
  policy, capability, circuit, and verifier-key revisions;
* encode authoritative artifacts as CIDv1 lowercase base32 dag-json sha2-256
  over retained canonical bytes with decoded-multihash recheck;
* reject fake/noncanonical CIDs, unknown fields, unsafe paths, partial writes,
  alias conflicts, stale records, and provenance mismatches;
* exclude only declared state-root control artifacts from completion-tree
  identity; and
* fail closed without importing or installing optional packages.

Atomic persistence is delegated to the kit
``CanonicalArtifactStoreTransport@1`` surface, loaded lazily when a local
root is configured.  Contract construction and replay verification never
require that transport.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from ..objectives.goal_completion import CompletionEvidence
from ..objectives.objective_tracker import completion_tree_identity
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    canonical_json_bytes,
)

# ---------------------------------------------------------------------------
# Interface / schema discriminators
# ---------------------------------------------------------------------------

PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION: Final = 1

PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE: Final = (
    "ProofTestReuseObjectiveBinding@1"
)
PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE: Final = (
    "ProofTestReuseCompletionArtifact@1"
)
PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE: Final = "ProofTestReuseGateBundle@1"
CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE: Final = (
    "CanonicalArtifactStoreTransport@1"
)
COMPLETION_EVIDENCE_INTERFACE: Final = "CompletionEvidence"

PROOF_TEST_REUSE_OBJECTIVE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-objective-binding@1"
)
PROOF_TEST_REUSE_COMPLETION_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-completion-artifact@1"
)
PROOF_TEST_REUSE_GATE_BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-gate-bundle@1"
)
CANONICAL_PREMISE_BLOCK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-canonical-premise-block@1"
)

# Declared state-root control artifacts (profile suffixes).  Only these paths
# may be excluded when deriving objective_completion_tree_id.
DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES: Final[tuple[str, ...]] = (
    "projection/completion/goal_completion_gate.json",
    "projection/completion/goal_completion_evidence.json",
    "projection/completion/objective_projection.md",
    "projection/completion/objective_candidate.md",
    "projection/completion/supervisor_health_input.json",
    "projection/completion/closeout_status.json",
)

# Identity domains that must never be aliased onto each other.
_IDENTITY_DOMAIN_FIELDS: Final[tuple[str, ...]] = (
    "git_tree_id",
    "repository_forest_cid",
    "objective_completion_tree_id",
)

_CID_SAFE_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyz234567")
_SHA256_HEX_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_CIDV1: Final = 1
_DAG_JSON_CODEC: Final = 0x0129
_SHA2_256: Final = 0x12
_SHA2_256_SIZE: Final = 32
_AUTHORITATIVE: Final = "authoritative"
_DEFAULT_MAX_PREMISE_BYTES: Final = 1_048_576
_MAX_PATH_DEPTH: Final = 64


class ObjectiveArtifactReason(str, Enum):
    """Closed rejection / fault reasons for artifact contracts and store I/O."""

    OK = "ok"
    FAKE_CID = "fake_cid"
    NONCANONICAL_CID = "noncanonical_cid"
    WRONG_CODEC = "wrong_codec"
    CID_MISMATCH = "cid_mismatch"
    MULTI_HASH_MISMATCH = "multihash_mismatch"
    UNKNOWN_FIELD = "unknown_field"
    UNSAFE_PATH = "unsafe_path"
    PARTIAL_WRITE = "partial_write"
    ALIAS_CONFLICT = "alias_conflict"
    STALE_RECORD = "stale_record"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    BINDING_INCOMPLETE = "binding_incomplete"
    IDENTITY_DOMAIN_COLLISION = "identity_domain_collision"
    CONTROL_PATH_NOT_DECLARED = "control_path_not_declared"
    TRANSPORT_UNAVAILABLE = "transport_unavailable"
    STORE_UNAVAILABLE = "store_unavailable"
    MALFORMED = "malformed"
    OVER_BUDGET = "over_budget"
    PATH_ESCAPE = "path_escape"
    SYMLINK_REJECTED = "symlink_rejected"
    INTEGRITY_FAILED = "integrity_failed"
    NOT_FOUND = "not_found"
    FENCED = "fenced"


class ProofTestReuseObjectiveContractsError(ValueError):
    """Raised when a completion-artifact contract is malformed or unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ObjectiveArtifactReason | str = ObjectiveArtifactReason.MALFORMED,
    ) -> None:
        super().__init__(message)
        if isinstance(reason_code, ObjectiveArtifactReason):
            self.reason_code = reason_code
        else:
            try:
                self.reason_code = ObjectiveArtifactReason(str(reason_code))
            except ValueError:
                self.reason_code = ObjectiveArtifactReason.MALFORMED


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _require_text(value: Any, *, field_name: str) -> str:
    text = _text(value)
    if not text:
        raise ProofTestReuseObjectiveContractsError(
            f"{field_name} is required",
            reason_code=ObjectiveArtifactReason.BINDING_INCOMPLETE,
        )
    return text


def _require_int_ms(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofTestReuseObjectiveContractsError(
            f"{field_name} must be an integer millisecond timestamp",
            reason_code=ObjectiveArtifactReason.MALFORMED,
        )
    if value < 0:
        raise ProofTestReuseObjectiveContractsError(
            f"{field_name} must be non-negative",
            reason_code=ObjectiveArtifactReason.MALFORMED,
        )
    return value


def _encode_varint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProofTestReuseObjectiveContractsError(
            "varint value must be a non-negative integer"
        )
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    result = 0
    shift = 0
    index = offset
    while True:
        if index >= len(data):
            raise ProofTestReuseObjectiveContractsError(
                "truncated multiformats varint",
                reason_code=ObjectiveArtifactReason.FAKE_CID,
            )
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, index
        shift += 7
        if shift > 63:
            raise ProofTestReuseObjectiveContractsError(
                "varint overflow",
                reason_code=ObjectiveArtifactReason.FAKE_CID,
            )


def canonical_dag_json_bytes(value: Any) -> bytes:
    """Return strict sorted-key compact DAG-JSON bytes (stdlib only)."""

    return canonical_json_bytes(value)


def cid_for_canonical_dag_json_bytes(data: bytes) -> str:
    """Mint CIDv1 / base32 / dag-json / sha2-256 for exact retained bytes."""

    if type(data) is not bytes:
        raise ProofTestReuseObjectiveContractsError(
            "artifact payload must be exact bytes",
            reason_code=ObjectiveArtifactReason.MALFORMED,
        )
    if not data:
        raise ProofTestReuseObjectiveContractsError(
            "artifact payload must be nonempty",
            reason_code=ObjectiveArtifactReason.MALFORMED,
        )
    digest = hashlib.sha256(data).digest()
    raw = (
        _encode_varint(_CIDV1)
        + _encode_varint(_DAG_JSON_CODEC)
        + _encode_varint(_SHA2_256)
        + _encode_varint(_SHA2_256_SIZE)
        + digest
    )
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def cid_for_mapping(value: Mapping[str, Any]) -> str:
    """Mint the authoritative CID for a mapping via retained canonical bytes."""

    return cid_for_canonical_dag_json_bytes(canonical_dag_json_bytes(value))


@dataclass(frozen=True, slots=True)
class DecodedArtifactCID:
    """Decoded CIDv1 dag-json sha2-256 multihash components."""

    text: str
    version: int
    codec: int
    multihash_code: int
    digest: bytes
    base: str = "base32"

    def verifies(self, data: bytes) -> bool:
        if type(data) is not bytes or not data:
            return False
        return (
            self.digest == hashlib.sha256(data).digest()
            and cid_for_canonical_dag_json_bytes(data) == self.text
        )


def decode_artifact_cid(value: Any) -> DecodedArtifactCID:
    """Decode and admit only CIDv1 lowercase base32 dag-json sha2-256."""

    if not isinstance(value, str) or not value:
        raise ProofTestReuseObjectiveContractsError(
            "CID must be a nonempty lowercase string",
            reason_code=ObjectiveArtifactReason.FAKE_CID,
        )
    if value != value.lower() or value.strip() != value:
        raise ProofTestReuseObjectiveContractsError(
            "CID must be canonical lowercase form",
            reason_code=ObjectiveArtifactReason.NONCANONICAL_CID,
        )
    if (
        value.startswith("Qm")
        or value.startswith("sha256:")
        or value.startswith("bafy-")
        or ":" in value
        or "/" in value
        or "\\" in value
        or ".." in value
        or "\x00" in value
        or not value.startswith("b")
        or len(value) < 16
    ):
        raise ProofTestReuseObjectiveContractsError(
            "CID is fake, truncated, or non-multiformats",
            reason_code=ObjectiveArtifactReason.FAKE_CID,
        )
    body = value[1:]
    if any(ch not in _CID_SAFE_CHARS for ch in body):
        raise ProofTestReuseObjectiveContractsError(
            "CID contains non-base32 characters",
            reason_code=ObjectiveArtifactReason.FAKE_CID,
        )
    # Pad base32 to a multiple of 8 characters.
    padded = body.upper() + ("=" * ((8 - (len(body) % 8)) % 8))
    try:
        raw = base64.b32decode(padded, casefold=True)
    except Exception as exc:
        raise ProofTestReuseObjectiveContractsError(
            "CID is not decodable base32",
            reason_code=ObjectiveArtifactReason.FAKE_CID,
        ) from exc
    if not raw:
        raise ProofTestReuseObjectiveContractsError(
            "CID decoded to empty bytes",
            reason_code=ObjectiveArtifactReason.FAKE_CID,
        )
    version, offset = _decode_varint(raw, 0)
    codec, offset = _decode_varint(raw, offset)
    mh_code, offset = _decode_varint(raw, offset)
    mh_size, offset = _decode_varint(raw, offset)
    digest = raw[offset:]
    if version != _CIDV1:
        raise ProofTestReuseObjectiveContractsError(
            "only CIDv1 is admitted for completion artifacts",
            reason_code=ObjectiveArtifactReason.NONCANONICAL_CID,
        )
    if codec != _DAG_JSON_CODEC:
        raise ProofTestReuseObjectiveContractsError(
            "artifact CIDs require the dag-json codec",
            reason_code=ObjectiveArtifactReason.WRONG_CODEC,
        )
    if mh_code != _SHA2_256 or mh_size != _SHA2_256_SIZE or len(digest) != _SHA2_256_SIZE:
        raise ProofTestReuseObjectiveContractsError(
            "artifact CIDs require a full 32-byte sha2-256 multihash",
            reason_code=ObjectiveArtifactReason.NONCANONICAL_CID,
        )
    # Round-trip: re-encode must match the admitted text exactly.
    rederived = (
        "b"
        + base64.b32encode(
            _encode_varint(version)
            + _encode_varint(codec)
            + _encode_varint(mh_code)
            + _encode_varint(mh_size)
            + digest
        )
        .decode("ascii")
        .lower()
        .rstrip("=")
    )
    if rederived != value:
        raise ProofTestReuseObjectiveContractsError(
            "CID is not the canonical lowercase base32 form",
            reason_code=ObjectiveArtifactReason.NONCANONICAL_CID,
        )
    return DecodedArtifactCID(
        text=value,
        version=version,
        codec=codec,
        multihash_code=mh_code,
        digest=digest,
    )


def validate_artifact_cid(value: Any) -> str:
    """Validate and return one admitted artifact CID string."""

    return decode_artifact_cid(value).text


def verify_retained_bytes(cid: str, data: bytes) -> bool:
    """Recheck decoded multihash against retained canonical bytes."""

    if type(data) is not bytes or not data:
        return False
    try:
        parsed = decode_artifact_cid(cid)
    except ProofTestReuseObjectiveContractsError:
        return False
    return parsed.verifies(data)


def require_verified_cid(cid: str, data: bytes) -> str:
    """Fail closed unless *cid* is the dag-json sha2-256 of *data*."""

    parsed = decode_artifact_cid(cid)
    if not parsed.verifies(data):
        raise ProofTestReuseObjectiveContractsError(
            "decoded multihash does not match retained canonical bytes",
            reason_code=ObjectiveArtifactReason.MULTI_HASH_MISMATCH,
        )
    derived = cid_for_canonical_dag_json_bytes(data)
    if derived != parsed.text:
        raise ProofTestReuseObjectiveContractsError(
            "CID does not match retained canonical bytes",
            reason_code=ObjectiveArtifactReason.CID_MISMATCH,
        )
    return parsed.text


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    artifact: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ProofTestReuseObjectiveContractsError(
            f"{artifact} contains unknown fields: {sorted(unknown)}",
            reason_code=ObjectiveArtifactReason.UNKNOWN_FIELD,
        )


def _posix_relative(path: Path) -> str:
    text = path.as_posix() if isinstance(path, Path) else str(path)
    normalized = text.replace("\\", "/").strip()
    if (
        not normalized
        or normalized.startswith("/")
        or normalized.startswith("~")
        or ".." in normalized.split("/")
        or "\x00" in normalized
        or normalized.startswith("./../")
        or any(part in {"", ".", ".."} for part in normalized.split("/") if part == "..")
    ):
        raise ProofTestReuseObjectiveContractsError(
            f"unsafe path rejected: {text!r}",
            reason_code=ObjectiveArtifactReason.UNSAFE_PATH,
        )
    if len(normalized.split("/")) > _MAX_PATH_DEPTH:
        raise ProofTestReuseObjectiveContractsError(
            f"path depth exceeds limit: {text!r}",
            reason_code=ObjectiveArtifactReason.UNSAFE_PATH,
        )
    return normalized.lstrip("./")


def declared_state_root_control_paths(state_root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Resolve the closed set of state-root control artifact paths."""

    root = Path(state_root)
    if not root.is_absolute():
        raise ProofTestReuseObjectiveContractsError(
            "state_root must be an absolute path",
            reason_code=ObjectiveArtifactReason.UNSAFE_PATH,
        )
    paths: list[Path] = []
    for suffix in DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES:
        # Suffixes are declared constants; still refuse traversal patterns.
        safe = _posix_relative(Path(suffix))
        paths.append((root / safe).resolve())
    return tuple(paths)


def assert_control_paths_are_declared(
    control_paths: Sequence[str | os.PathLike[str] | Path],
    *,
    state_root: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Admit only declared state-root control artifacts for tree exclusion."""

    allowed = {path.resolve() for path in declared_state_root_control_paths(state_root)}
    admitted: list[Path] = []
    for raw in control_paths:
        candidate = Path(raw).resolve()
        if candidate not in allowed:
            raise ProofTestReuseObjectiveContractsError(
                f"control path is not a declared state-root artifact: {candidate}",
                reason_code=ObjectiveArtifactReason.CONTROL_PATH_NOT_DECLARED,
            )
        admitted.append(candidate)
    return tuple(admitted)


def compute_objective_completion_tree_id(
    repo_root: str | os.PathLike[str] | Path,
    *,
    objective_path: str | os.PathLike[str] | Path,
    state_root: str | os.PathLike[str] | Path | None = None,
    control_paths: Sequence[str | os.PathLike[str] | Path] = (),
) -> str:
    """Return objective-completion tree identity with declared exclusions only.

    The objective markdown is always excluded (mutable supervisor lifecycle
    surface).  Additional exclusions are restricted to declared state-root
    control artifacts when *state_root* is provided.  Arbitrary path exclusion
    is fail-closed.
    """

    root = Path(repo_root)
    objective = Path(objective_path)
    admitted: tuple[Path, ...] = ()
    if control_paths:
        if state_root is None:
            raise ProofTestReuseObjectiveContractsError(
                "state_root is required when control_paths are supplied",
                reason_code=ObjectiveArtifactReason.CONTROL_PATH_NOT_DECLARED,
            )
        admitted = assert_control_paths_are_declared(
            control_paths, state_root=state_root
        )
    elif state_root is not None:
        # Default exclusion set is the full declared population under state root.
        admitted = declared_state_root_control_paths(state_root)
    identity = completion_tree_identity(
        root,
        objective_path=objective,
        control_paths=admitted,
    )
    return _require_text(identity.tree_id, field_name="objective_completion_tree_id")


def _assert_identity_domains_distinct(
    git_tree_id: str,
    repository_forest_cid: str,
    objective_completion_tree_id: str,
) -> None:
    """Reject cross-domain aliases among the three identity surfaces."""

    domains = {
        "git_tree_id": git_tree_id,
        "repository_forest_cid": repository_forest_cid,
        "objective_completion_tree_id": objective_completion_tree_id,
    }
    for name, value in domains.items():
        if not value:
            raise ProofTestReuseObjectiveContractsError(
                f"{name} is required",
                reason_code=ObjectiveArtifactReason.BINDING_INCOMPLETE,
            )
    values = list(domains.values())
    if len(set(values)) != 3:
        raise ProofTestReuseObjectiveContractsError(
            "git_tree_id, repository_forest_cid, and "
            "objective_completion_tree_id must be pairwise distinct domains",
            reason_code=ObjectiveArtifactReason.IDENTITY_DOMAIN_COLLISION,
        )
    # Git tree ids are 40-char hex; forest and completion CIDs/labels must not
    # masquerade as each other via field aliases in the same payload.
    if _GIT_OBJECT_ID_RE.fullmatch(repository_forest_cid):
        raise ProofTestReuseObjectiveContractsError(
            "repository_forest_cid must not be a bare git object id",
            reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
        )
    if _GIT_OBJECT_ID_RE.fullmatch(objective_completion_tree_id) and (
        objective_completion_tree_id == git_tree_id
    ):
        # Already caught by pairwise distinct, retained for clarity.
        raise ProofTestReuseObjectiveContractsError(
            "objective_completion_tree_id must not alias git_tree_id",
            reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
        )


def _check_freshness(
    *,
    observed_at_ms: int,
    fresh_until_ms: int,
    now_ms: int | None,
) -> None:
    if fresh_until_ms <= observed_at_ms:
        raise ProofTestReuseObjectiveContractsError(
            "freshness window is invalid",
            reason_code=ObjectiveArtifactReason.MALFORMED,
        )
    if now_ms is not None and now_ms > fresh_until_ms:
        raise ProofTestReuseObjectiveContractsError(
            "record is stale relative to the evaluation clock",
            reason_code=ObjectiveArtifactReason.STALE_RECORD,
        )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalPremiseBlock:
    """Retained exact canonical DAG-JSON premise bytes bound to their CID."""

    data: bytes
    cid: str = ""
    role: str = "premise"
    schema: str = CANONICAL_PREMISE_BLOCK_SCHEMA

    def __post_init__(self) -> None:
        if type(self.data) is not bytes or not self.data:
            raise ProofTestReuseObjectiveContractsError(
                "CanonicalPremiseBlock requires nonempty exact bytes",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if len(self.data) > _DEFAULT_MAX_PREMISE_BYTES:
            raise ProofTestReuseObjectiveContractsError(
                "premise block exceeds budget",
                reason_code=ObjectiveArtifactReason.OVER_BUDGET,
            )
        derived = cid_for_canonical_dag_json_bytes(self.data)
        if self.cid and self.cid != derived:
            raise ProofTestReuseObjectiveContractsError(
                "premise CID does not match retained bytes",
                reason_code=ObjectiveArtifactReason.CID_MISMATCH,
            )
        object.__setattr__(self, "cid", derived)
        object.__setattr__(self, "role", _require_text(self.role, field_name="role"))
        # Multihash recheck is mandatory at construction.
        require_verified_cid(self.cid, self.data)

    @property
    def multihash_hex(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def byte_length(self) -> int:
        return len(self.data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cid": self.cid,
            "role": self.role,
            "byte_length": self.byte_length,
            "multihash_hex": self.multihash_hex,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        role: str = "premise",
    ) -> CanonicalPremiseBlock:
        data = canonical_dag_json_bytes(dict(value))
        return cls(data=data, role=role)

    @classmethod
    def from_bytes(cls, data: bytes, *, role: str = "premise", cid: str = "") -> CanonicalPremiseBlock:
        return cls(data=data, cid=cid, role=role)


@dataclass(frozen=True, slots=True)
class ProofTestReuseObjectiveBinding(CanonicalContract):
    """Exact per-goal repository / policy / identity binding.

    The three identity domains are first-class and never interchangeable:

    * ``git_tree_id`` — pure Git tree object id for the checkout;
    * ``repository_forest_cid`` — recursive repository-forest content id;
    * ``objective_completion_tree_id`` — completion-scan identity that excludes
      only declared state-root control artifacts (and the objective document).
    """

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_OBJECTIVE_BINDING_SCHEMA

    goal_id: str
    repository_id: str
    git_tree_id: str
    repository_forest_cid: str
    objective_completion_tree_id: str
    objective_revision: str
    analyzer_revision: str
    configuration_revision: str
    policy_revision: str
    capability_revision: str
    circuit_revision: str
    verifying_key_revision: str
    git_commit_id: str = ""
    gitlink_state_cid: str = ""
    repository_state_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "goal_id",
            "repository_id",
            "git_tree_id",
            "repository_forest_cid",
            "objective_completion_tree_id",
            "objective_revision",
            "analyzer_revision",
            "configuration_revision",
            "policy_revision",
            "capability_revision",
            "circuit_revision",
            "verifying_key_revision",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), field_name=name),
            )
        for name in ("git_commit_id", "gitlink_state_cid", "repository_state_cid"):
            object.__setattr__(self, name, _text(getattr(self, name)))
        _assert_identity_domains_distinct(
            self.git_tree_id,
            self.repository_forest_cid,
            self.objective_completion_tree_id,
        )
        # Reject field-level aliases that would collapse revision namespaces.
        revision_fields = {
            "objective_revision": self.objective_revision,
            "analyzer_revision": self.analyzer_revision,
            "configuration_revision": self.configuration_revision,
            "policy_revision": self.policy_revision,
            "capability_revision": self.capability_revision,
            "circuit_revision": self.circuit_revision,
            "verifying_key_revision": self.verifying_key_revision,
        }
        # Pairwise equality across *all* revisions is allowed only when values
        # intentionally match; alias *names* in the same payload are handled by
        # unknown-field rejection.  Cross-domain id/revision collisions that
        # reuse a git tree as a policy revision are rejected.
        for rev_name, rev_value in revision_fields.items():
            if rev_value == self.git_tree_id:
                raise ProofTestReuseObjectiveContractsError(
                    f"{rev_name} must not alias git_tree_id",
                    reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
                )

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE

    @property
    def tree_id(self) -> str:
        """Compatibility spelling — always the Git tree domain."""

        return self.git_tree_id

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION,
            "interface": self.interface,
            "goal_id": self.goal_id,
            "repository_id": self.repository_id,
            "git_tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "objective_revision": self.objective_revision,
            "analyzer_revision": self.analyzer_revision,
            "configuration_revision": self.configuration_revision,
            "policy_revision": self.policy_revision,
            "capability_revision": self.capability_revision,
            "circuit_revision": self.circuit_revision,
            "verifying_key_revision": self.verifying_key_revision,
        }
        if self.git_commit_id:
            payload["git_commit_id"] = self.git_commit_id
        if self.gitlink_state_cid:
            payload["gitlink_state_cid"] = self.gitlink_state_cid
        if self.repository_state_cid:
            payload["repository_state_cid"] = self.repository_state_cid
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **self._payload()}

    @property
    def content_id(self) -> str:
        return cid_for_mapping(self.to_dict())

    @property
    def binding_cid(self) -> str:
        return self.content_id

    def canonical_bytes(self) -> bytes:
        return canonical_dag_json_bytes(self.to_dict())

    def matches(self, other: Mapping[str, Any] | ProofTestReuseObjectiveBinding) -> bool:
        if isinstance(other, ProofTestReuseObjectiveBinding):
            return self.binding_cid == other.binding_cid
        try:
            return self.binding_cid == ProofTestReuseObjectiveBinding.from_dict(other).binding_cid
        except (TypeError, ValueError, ProofTestReuseObjectiveContractsError):
            return False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTestReuseObjectiveBinding:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "objective binding must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        allowed = frozenset(
            {
                "schema",
                "contract_version",
                "interface",
                "goal_id",
                "repository_id",
                "git_tree_id",
                "tree_id",  # compatibility alias for git_tree_id only
                "repository_forest_cid",
                "forest_cid",  # rejected if it conflicts with repository_forest_cid
                "objective_completion_tree_id",
                "completion_tree_id",
                "objective_revision",
                "analyzer_revision",
                "analyzer_version",
                "configuration_revision",
                "configuration_id",
                "policy_revision",
                "policy_cid",
                "capability_revision",
                "capability_cid",
                "circuit_revision",
                "circuit_cid",
                "verifying_key_revision",
                "verifying_key_cid",
                "git_commit_id",
                "gitlink_state_cid",
                "repository_state_cid",
                "content_id",
                "binding_cid",
            }
        )
        _reject_unknown_fields(payload, allowed, artifact="objective binding")

        def _pick(*names: str) -> Any:
            for name in names:
                if name in payload and payload[name] not in (None, ""):
                    return payload[name]
            return ""

        git_tree = _text(_pick("git_tree_id", "tree_id"))
        forest = _text(_pick("repository_forest_cid", "forest_cid"))
        completion = _text(
            _pick("objective_completion_tree_id", "completion_tree_id")
        )
        # Alias conflict: both spellings present with different values.
        if (
            "git_tree_id" in payload
            and "tree_id" in payload
            and _text(payload.get("git_tree_id")) != _text(payload.get("tree_id"))
        ):
            raise ProofTestReuseObjectiveContractsError(
                "git_tree_id and tree_id alias conflict",
                reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
            )
        if (
            "repository_forest_cid" in payload
            and "forest_cid" in payload
            and _text(payload.get("repository_forest_cid"))
            != _text(payload.get("forest_cid"))
        ):
            raise ProofTestReuseObjectiveContractsError(
                "repository_forest_cid and forest_cid alias conflict",
                reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
            )
        if (
            "objective_completion_tree_id" in payload
            and "completion_tree_id" in payload
            and _text(payload.get("objective_completion_tree_id"))
            != _text(payload.get("completion_tree_id"))
        ):
            raise ProofTestReuseObjectiveContractsError(
                "objective_completion_tree_id and completion_tree_id alias conflict",
                reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
            )

        body = {
            "schema": payload.get("schema", cls.SCHEMA),
            "contract_version": payload.get(
                "contract_version", PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION
            ),
            "interface": payload.get(
                "interface", PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE
            ),
        }
        if body["schema"] != cls.SCHEMA:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported objective binding schema",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if body["interface"] != PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported objective binding interface",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if body["contract_version"] != PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported objective binding contract version",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )

        result = cls(
            goal_id=_pick("goal_id"),
            repository_id=_pick("repository_id"),
            git_tree_id=git_tree,
            repository_forest_cid=forest,
            objective_completion_tree_id=completion,
            objective_revision=_pick("objective_revision"),
            analyzer_revision=_pick("analyzer_revision", "analyzer_version"),
            configuration_revision=_pick(
                "configuration_revision", "configuration_id"
            ),
            policy_revision=_pick("policy_revision", "policy_cid"),
            capability_revision=_pick("capability_revision", "capability_cid"),
            circuit_revision=_pick("circuit_revision", "circuit_cid"),
            verifying_key_revision=_pick(
                "verifying_key_revision", "verifying_key_cid"
            ),
            git_commit_id=_pick("git_commit_id"),
            gitlink_state_cid=_pick("gitlink_state_cid"),
            repository_state_cid=_pick("repository_state_cid"),
        )
        claimed = _text(payload.get("content_id") or payload.get("binding_cid"))
        if claimed and claimed != result.binding_cid:
            raise ProofTestReuseObjectiveContractsError(
                "objective binding content identity does not match payload",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        if claimed:
            require_verified_cid(claimed, result.canonical_bytes())
        return result


@dataclass(frozen=True, slots=True)
class ProofTestReuseCompletionArtifact(CanonicalContract):
    """Authoritative per-goal completion artifact with retained premises."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_COMPLETION_ARTIFACT_SCHEMA

    binding: ProofTestReuseObjectiveBinding
    acceptance_criterion: str
    producing_task_or_scan: str
    premise_blocks: tuple[CanonicalPremiseBlock, ...]
    observed_at_ms: int
    fresh_until_ms: int
    validation_passed: bool = True
    producer_kind: str = "task"
    producer_channel: str = ""
    channel_proof_revision: str = ""
    authority: str = _AUTHORITATIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ProofTestReuseObjectiveBinding):
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact requires a typed objective binding",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        object.__setattr__(
            self,
            "acceptance_criterion",
            _require_text(self.acceptance_criterion, field_name="acceptance_criterion"),
        )
        object.__setattr__(
            self,
            "producing_task_or_scan",
            _require_text(
                self.producing_task_or_scan, field_name="producing_task_or_scan"
            ),
        )
        if not isinstance(self.validation_passed, bool):
            raise ProofTestReuseObjectiveContractsError(
                "validation_passed must be a boolean",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        object.__setattr__(
            self,
            "producer_kind",
            _require_text(self.producer_kind, field_name="producer_kind").lower(),
        )
        object.__setattr__(self, "producer_channel", _text(self.producer_channel))
        object.__setattr__(
            self, "channel_proof_revision", _text(self.channel_proof_revision)
        )
        if self.authority != _AUTHORITATIVE:
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact authority must be authoritative",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        observed = _require_int_ms(self.observed_at_ms, field_name="observed_at_ms")
        fresh = _require_int_ms(self.fresh_until_ms, field_name="fresh_until_ms")
        object.__setattr__(self, "observed_at_ms", observed)
        object.__setattr__(self, "fresh_until_ms", fresh)
        _check_freshness(
            observed_at_ms=observed, fresh_until_ms=fresh, now_ms=None
        )
        blocks = tuple(self.premise_blocks or ())
        if not blocks:
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact requires at least one retained premise block",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        seen_cids: set[str] = set()
        normalized: list[CanonicalPremiseBlock] = []
        for block in blocks:
            if not isinstance(block, CanonicalPremiseBlock):
                raise ProofTestReuseObjectiveContractsError(
                    "premise_blocks must be CanonicalPremiseBlock values",
                    reason_code=ObjectiveArtifactReason.MALFORMED,
                )
            require_verified_cid(block.cid, block.data)
            if block.cid in seen_cids:
                raise ProofTestReuseObjectiveContractsError(
                    f"duplicate premise CID {block.cid}",
                    reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
                )
            seen_cids.add(block.cid)
            normalized.append(block)
        object.__setattr__(self, "premise_blocks", tuple(normalized))
        if not isinstance(self.metadata, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "metadata must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE

    @property
    def premise_cids(self) -> tuple[str, ...]:
        return tuple(block.cid for block in self.premise_blocks)

    def _identity_body(self) -> dict[str, Any]:
        """Payload used for content identity (excludes derived content_id)."""

        return {
            "schema": self.SCHEMA,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION,
            "interface": self.interface,
            "binding": self.binding.to_dict(),
            "acceptance_criterion": self.acceptance_criterion,
            "producing_task_or_scan": self.producing_task_or_scan,
            "producer_kind": self.producer_kind,
            "producer_channel": self.producer_channel,
            "channel_proof_revision": self.channel_proof_revision,
            "authority": self.authority,
            "validation_passed": self.validation_passed,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "premise_cids": list(self.premise_cids),
            "premise_blocks": [
                {
                    "cid": block.cid,
                    "role": block.role,
                    "byte_length": block.byte_length,
                    "multihash_hex": block.multihash_hex,
                    # Retain exact canonical bytes as UTF-8 text for replay.
                    # Bytes are already canonical DAG-JSON UTF-8.
                    "canonical_utf8": block.data.decode("utf-8"),
                }
                for block in self.premise_blocks
            ],
            "metadata": dict(self.metadata),
        }

    def _payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._identity_body().items()
            if key != "schema"
        }

    def to_dict(self) -> dict[str, Any]:
        return self._identity_body()

    def canonical_bytes(self) -> bytes:
        return canonical_dag_json_bytes(self.to_dict())

    @property
    def content_id(self) -> str:
        return cid_for_canonical_dag_json_bytes(self.canonical_bytes())

    @property
    def artifact_cid(self) -> str:
        return self.content_id

    @property
    def provenance_cid(self) -> str:
        return self.content_id

    def is_fresh(self, now_ms: int) -> bool:
        try:
            _check_freshness(
                observed_at_ms=self.observed_at_ms,
                fresh_until_ms=self.fresh_until_ms,
                now_ms=now_ms,
            )
        except ProofTestReuseObjectiveContractsError:
            return False
        return True

    def require_fresh(self, now_ms: int) -> None:
        _check_freshness(
            observed_at_ms=self.observed_at_ms,
            fresh_until_ms=self.fresh_until_ms,
            now_ms=now_ms,
        )

    def replay_premises(self) -> tuple[CanonicalPremiseBlock, ...]:
        """Re-verify every retained premise multihash and return the blocks."""

        replayed: list[CanonicalPremiseBlock] = []
        for block in self.premise_blocks:
            require_verified_cid(block.cid, block.data)
            replayed.append(
                CanonicalPremiseBlock(
                    data=block.data, cid=block.cid, role=block.role
                )
            )
        return tuple(replayed)

    def as_completion_evidence(self) -> CompletionEvidence:
        """Project into the shared objective ``CompletionEvidence`` surface."""

        return CompletionEvidence(
            acceptance_criterion=self.acceptance_criterion,
            producing_task_or_scan=self.producing_task_or_scan,
            producer_kind=self.producer_kind,
            producer_channel=self.producer_channel,
            channel_proof_revision=self.channel_proof_revision,
            validation_receipt={
                "artifact_cid": self.artifact_cid,
                "premise_cids": list(self.premise_cids),
                "binding_cid": self.binding.binding_cid,
            },
            repository_id=self.binding.repository_id,
            repository_tree=self.binding.git_tree_id,
            tree_id=self.binding.git_tree_id,
            objective_revision=self.binding.objective_revision,
            analyzer_version=self.binding.analyzer_revision,
            configuration_revision=self.binding.configuration_revision,
            provenance_cid=self.provenance_cid,
            validation_passed=self.validation_passed,
            observed_at=datetime.fromtimestamp(
                self.observed_at_ms / 1000.0, tz=UTC
            ),
            fresh_until=datetime.fromtimestamp(
                self.fresh_until_ms / 1000.0, tz=UTC
            ),
            freshness={
                "observed_at_ms": self.observed_at_ms,
                "fresh_until_ms": self.fresh_until_ms,
            },
            metadata={
                "goal_id": self.binding.goal_id,
                "authority": self.authority,
                "repository_forest_cid": self.binding.repository_forest_cid,
                "objective_completion_tree_id": (
                    self.binding.objective_completion_tree_id
                ),
                "policy_revision": self.binding.policy_revision,
                "capability_revision": self.binding.capability_revision,
                "circuit_revision": self.binding.circuit_revision,
                "verifying_key_revision": self.binding.verifying_key_revision,
                **dict(self.metadata),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTestReuseCompletionArtifact:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        allowed = frozenset(
            {
                "schema",
                "contract_version",
                "interface",
                "binding",
                "acceptance_criterion",
                "producing_task_or_scan",
                "producer_kind",
                "producer_channel",
                "channel_proof_revision",
                "authority",
                "validation_passed",
                "observed_at_ms",
                "fresh_until_ms",
                "premise_cids",
                "premise_blocks",
                "metadata",
                "content_id",
                "artifact_cid",
                "provenance_cid",
            }
        )
        _reject_unknown_fields(payload, allowed, artifact="completion artifact")
        if payload.get("schema") != cls.SCHEMA:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported completion artifact schema",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if payload.get("interface") != PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported completion artifact interface",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if (
            payload.get("contract_version")
            != PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION
        ):
            raise ProofTestReuseObjectiveContractsError(
                "unsupported completion artifact contract version",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        raw_binding = payload.get("binding")
        if not isinstance(raw_binding, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact binding must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        binding = ProofTestReuseObjectiveBinding.from_dict(raw_binding)
        raw_blocks = payload.get("premise_blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact premise_blocks must be a nonempty list",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        blocks: list[CanonicalPremiseBlock] = []
        for index, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, Mapping):
                raise ProofTestReuseObjectiveContractsError(
                    f"premise_blocks[{index}] must be a mapping",
                    reason_code=ObjectiveArtifactReason.MALFORMED,
                )
            block_allowed = frozenset(
                {
                    "cid",
                    "role",
                    "byte_length",
                    "multihash_hex",
                    "canonical_utf8",
                    "schema",
                }
            )
            _reject_unknown_fields(
                raw_block,
                block_allowed,
                artifact=f"premise_blocks[{index}]",
            )
            text = raw_block.get("canonical_utf8")
            if not isinstance(text, str) or not text:
                raise ProofTestReuseObjectiveContractsError(
                    f"premise_blocks[{index}] missing retained canonical_utf8",
                    reason_code=ObjectiveArtifactReason.PARTIAL_WRITE,
                )
            data = text.encode("utf-8")
            claimed = _text(raw_block.get("cid"))
            block = CanonicalPremiseBlock(
                data=data,
                cid=claimed,
                role=_text(raw_block.get("role")) or "premise",
            )
            if raw_block.get("byte_length") not in (None, block.byte_length):
                raise ProofTestReuseObjectiveContractsError(
                    f"premise_blocks[{index}] byte_length mismatch",
                    reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
                )
            if raw_block.get("multihash_hex") not in (None, block.multihash_hex):
                raise ProofTestReuseObjectiveContractsError(
                    f"premise_blocks[{index}] multihash mismatch",
                    reason_code=ObjectiveArtifactReason.MULTI_HASH_MISMATCH,
                )
            blocks.append(block)
        claimed_premise_cids = payload.get("premise_cids")
        if claimed_premise_cids is not None:
            if not isinstance(claimed_premise_cids, list):
                raise ProofTestReuseObjectiveContractsError(
                    "premise_cids must be a list",
                    reason_code=ObjectiveArtifactReason.MALFORMED,
                )
            if [block.cid for block in blocks] != [
                _text(item) for item in claimed_premise_cids
            ]:
                raise ProofTestReuseObjectiveContractsError(
                    "premise_cids does not match retained premise blocks",
                    reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
                )
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "metadata must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        result = cls(
            binding=binding,
            acceptance_criterion=_text(payload.get("acceptance_criterion")),
            producing_task_or_scan=_text(payload.get("producing_task_or_scan")),
            premise_blocks=tuple(blocks),
            observed_at_ms=payload.get("observed_at_ms"),
            fresh_until_ms=payload.get("fresh_until_ms"),
            validation_passed=payload.get("validation_passed", True),
            producer_kind=_text(payload.get("producer_kind")) or "task",
            producer_channel=_text(payload.get("producer_channel")),
            channel_proof_revision=_text(payload.get("channel_proof_revision")),
            authority=_text(payload.get("authority")) or _AUTHORITATIVE,
            metadata=dict(metadata),
        )
        claimed_cid = _text(
            payload.get("content_id")
            or payload.get("artifact_cid")
            or payload.get("provenance_cid")
        )
        if claimed_cid and claimed_cid != result.artifact_cid:
            raise ProofTestReuseObjectiveContractsError(
                "completion artifact content identity does not match payload",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        if claimed_cid:
            require_verified_cid(claimed_cid, result.canonical_bytes())
        return result


@dataclass(frozen=True, slots=True)
class ProofTestReuseGateBundle(CanonicalContract):
    """Finite gate envelope binding one or more completion artifacts."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_GATE_BUNDLE_SCHEMA

    repository_id: str
    git_tree_id: str
    repository_forest_cid: str
    objective_completion_tree_id: str
    artifacts: tuple[ProofTestReuseCompletionArtifact, ...]
    passed: bool
    reason_codes: tuple[str, ...] = ()
    evaluated_at_ms: int = 0
    producing_task_id: str = ""
    policy_revision: str = ""
    capability_revision: str = ""
    circuit_revision: str = ""
    verifying_key_revision: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "git_tree_id",
            "repository_forest_cid",
            "objective_completion_tree_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), field_name=name),
            )
        _assert_identity_domains_distinct(
            self.git_tree_id,
            self.repository_forest_cid,
            self.objective_completion_tree_id,
        )
        if not isinstance(self.passed, bool):
            raise ProofTestReuseObjectiveContractsError(
                "passed must be a boolean",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        artifacts = tuple(self.artifacts or ())
        if self.passed and not artifacts:
            raise ProofTestReuseObjectiveContractsError(
                "passing gate bundle requires at least one completion artifact",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if not self.passed and artifacts:
            raise ProofTestReuseObjectiveContractsError(
                "failed gate bundle cannot carry completion artifacts",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        for artifact in artifacts:
            if not isinstance(artifact, ProofTestReuseCompletionArtifact):
                raise ProofTestReuseObjectiveContractsError(
                    "artifacts must be ProofTestReuseCompletionArtifact values",
                    reason_code=ObjectiveArtifactReason.MALFORMED,
                )
            # Envelope identities must match every nested binding.
            binding = artifact.binding
            if (
                binding.repository_id != self.repository_id
                or binding.git_tree_id != self.git_tree_id
                or binding.repository_forest_cid != self.repository_forest_cid
                or binding.objective_completion_tree_id
                != self.objective_completion_tree_id
            ):
                raise ProofTestReuseObjectiveContractsError(
                    "gate bundle identity domains do not match artifact binding",
                    reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
                )
            for field_name in (
                "policy_revision",
                "capability_revision",
                "circuit_revision",
                "verifying_key_revision",
            ):
                envelope_value = _text(getattr(self, field_name))
                artifact_value = _text(getattr(binding, field_name))
                if envelope_value and envelope_value != artifact_value:
                    raise ProofTestReuseObjectiveContractsError(
                        f"gate bundle {field_name} does not match artifact binding",
                        reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
                    )
        object.__setattr__(self, "artifacts", artifacts)
        reasons = tuple(
            dict.fromkeys(_text(item) for item in (self.reason_codes or ()) if _text(item))
        )
        if self.passed and reasons:
            raise ProofTestReuseObjectiveContractsError(
                "passing gate bundle cannot carry rejection reasons",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if not self.passed and not reasons:
            raise ProofTestReuseObjectiveContractsError(
                "failed gate bundle requires reason_codes",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        object.__setattr__(self, "reason_codes", reasons)
        evaluated = self.evaluated_at_ms
        if evaluated == 0:
            evaluated = 0
        else:
            evaluated = _require_int_ms(evaluated, field_name="evaluated_at_ms")
        object.__setattr__(self, "evaluated_at_ms", evaluated)
        object.__setattr__(self, "producing_task_id", _text(self.producing_task_id))
        for name in (
            "policy_revision",
            "capability_revision",
            "circuit_revision",
            "verifying_key_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name)))

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION,
            "interface": self.interface,
            "repository_id": self.repository_id,
            "git_tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "evaluated_at_ms": self.evaluated_at_ms,
            "producing_task_id": self.producing_task_id,
            "policy_revision": self.policy_revision,
            "capability_revision": self.capability_revision,
            "circuit_revision": self.circuit_revision,
            "verifying_key_revision": self.verifying_key_revision,
            "artifact_cids": [item.artifact_cid for item in self.artifacts],
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    def _payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._identity_body().items()
            if key != "schema"
        }

    def to_dict(self) -> dict[str, Any]:
        return self._identity_body()

    def canonical_bytes(self) -> bytes:
        return canonical_dag_json_bytes(self.to_dict())

    @property
    def content_id(self) -> str:
        return cid_for_canonical_dag_json_bytes(self.canonical_bytes())

    @property
    def bundle_cid(self) -> str:
        return self.content_id

    def replay(self) -> ProofTestReuseGateBundle:
        """Deserialize from retained canonical form and re-verify premises."""

        return ProofTestReuseGateBundle.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTestReuseGateBundle:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "gate bundle must be a mapping",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        allowed = frozenset(
            {
                "schema",
                "contract_version",
                "interface",
                "repository_id",
                "git_tree_id",
                "tree_id",
                "repository_forest_cid",
                "objective_completion_tree_id",
                "passed",
                "reason_codes",
                "evaluated_at_ms",
                "producing_task_id",
                "policy_revision",
                "capability_revision",
                "circuit_revision",
                "verifying_key_revision",
                "artifact_cids",
                "artifacts",
                "content_id",
                "bundle_cid",
            }
        )
        _reject_unknown_fields(payload, allowed, artifact="gate bundle")
        if payload.get("schema") != cls.SCHEMA:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported gate bundle schema",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if payload.get("interface") != PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE:
            raise ProofTestReuseObjectiveContractsError(
                "unsupported gate bundle interface",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if (
            payload.get("contract_version")
            != PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION
        ):
            raise ProofTestReuseObjectiveContractsError(
                "unsupported gate bundle contract version",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if (
            "git_tree_id" in payload
            and "tree_id" in payload
            and _text(payload.get("git_tree_id")) != _text(payload.get("tree_id"))
        ):
            raise ProofTestReuseObjectiveContractsError(
                "git_tree_id and tree_id alias conflict",
                reason_code=ObjectiveArtifactReason.ALIAS_CONFLICT,
            )
        git_tree = _text(payload.get("git_tree_id") or payload.get("tree_id"))
        raw_artifacts = payload.get("artifacts") or []
        if not isinstance(raw_artifacts, list):
            raise ProofTestReuseObjectiveContractsError(
                "artifacts must be a list",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        artifacts = tuple(
            ProofTestReuseCompletionArtifact.from_dict(item)
            for item in raw_artifacts
        )
        claimed_artifact_cids = payload.get("artifact_cids")
        if claimed_artifact_cids is not None:
            if not isinstance(claimed_artifact_cids, list):
                raise ProofTestReuseObjectiveContractsError(
                    "artifact_cids must be a list",
                    reason_code=ObjectiveArtifactReason.MALFORMED,
                )
            if [item.artifact_cid for item in artifacts] != [
                _text(item) for item in claimed_artifact_cids
            ]:
                raise ProofTestReuseObjectiveContractsError(
                    "artifact_cids does not match nested artifacts",
                    reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
                )
        reasons = payload.get("reason_codes") or ()
        if not isinstance(reasons, (list, tuple)):
            raise ProofTestReuseObjectiveContractsError(
                "reason_codes must be a list",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        result = cls(
            repository_id=_text(payload.get("repository_id")),
            git_tree_id=git_tree,
            repository_forest_cid=_text(payload.get("repository_forest_cid")),
            objective_completion_tree_id=_text(
                payload.get("objective_completion_tree_id")
            ),
            artifacts=artifacts,
            passed=bool(payload.get("passed")),
            reason_codes=tuple(reasons),
            evaluated_at_ms=payload.get("evaluated_at_ms") or 0,
            producing_task_id=_text(payload.get("producing_task_id")),
            policy_revision=_text(payload.get("policy_revision")),
            capability_revision=_text(payload.get("capability_revision")),
            circuit_revision=_text(payload.get("circuit_revision")),
            verifying_key_revision=_text(payload.get("verifying_key_revision")),
        )
        claimed = _text(payload.get("content_id") or payload.get("bundle_cid"))
        if claimed and claimed != result.bundle_cid:
            raise ProofTestReuseObjectiveContractsError(
                "gate bundle content identity does not match payload",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        if claimed:
            require_verified_cid(claimed, result.canonical_bytes())
        return result


# ---------------------------------------------------------------------------
# Artifact store (lazy kit transport injection)
# ---------------------------------------------------------------------------


def _load_kit_transport() -> Any | None:
    """Lazily import kit CanonicalArtifactStoreTransport without installing."""

    try:
        from ipfs_kit_py.content_addressed_artifact_store import (  # type: ignore
            CanonicalArtifactStoreTransport,
        )
    except Exception:
        return None
    return CanonicalArtifactStoreTransport


def _safe_local_blob_path(root: Path, cid: str) -> Path:
    """Resolve a CID blob path under *root* without path escape."""

    validated = validate_artifact_cid(cid)
    # CID text is base32; still reject anything that could escape.
    if "/" in validated or "\\" in validated or ".." in validated:
        raise ProofTestReuseObjectiveContractsError(
            "CID path is unsafe",
            reason_code=ObjectiveArtifactReason.UNSAFE_PATH,
        )
    path = (root / f"{validated}.blob").resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProofTestReuseObjectiveContractsError(
            "blob path escapes store root",
            reason_code=ObjectiveArtifactReason.PATH_ESCAPE,
        ) from exc
    return path


class ObjectiveArtifactStore:
    """Atomic state-root store for completion artifacts and premise bytes.

    Persistence prefers the injected kit
    :class:`CanonicalArtifactStoreTransport` when available.  When kit is
    absent the store fails closed for writes rather than installing packages
    or inventing a second trust root.  A minimal local atomic backend is used
    only when *local_root* is provided and kit cannot be imported, so hermetic
    contract tests remain deterministic.
    """

    __test__ = False

    def __init__(
        self,
        local_root: str | os.PathLike[str] | None = None,
        *,
        transport: Any | None = None,
        max_blob_bytes: int = _DEFAULT_MAX_PREMISE_BYTES,
        clock: Callable[[], float] | None = None,
        require_kit_transport: bool = False,
    ) -> None:
        if (
            isinstance(max_blob_bytes, bool)
            or not isinstance(max_blob_bytes, int)
            or max_blob_bytes <= 0
        ):
            raise ProofTestReuseObjectiveContractsError(
                "max_blob_bytes must be a positive integer"
            )
        self.max_blob_bytes = int(max_blob_bytes)
        self._clock = clock or time.time
        self.local_root = (
            Path(local_root).resolve() if local_root is not None else None
        )
        if self.local_root is not None:
            if self.local_root.is_symlink():
                raise ProofTestReuseObjectiveContractsError(
                    "store root must not be a symlink",
                    reason_code=ObjectiveArtifactReason.SYMLINK_REJECTED,
                )
        self._transport = transport
        self._kit_transport_cls = _load_kit_transport()
        if transport is None and self.local_root is not None and self._kit_transport_cls is not None:
            self._transport = self._kit_transport_cls(
                self.local_root,
                max_blob_bytes=self.max_blob_bytes,
                clock=self._clock,
            )
        if require_kit_transport and self._transport is None:
            raise ProofTestReuseObjectiveContractsError(
                "CanonicalArtifactStoreTransport is unavailable",
                reason_code=ObjectiveArtifactReason.TRANSPORT_UNAVAILABLE,
            )

    @property
    def interface(self) -> str:
        if self._transport is not None:
            return str(
                getattr(
                    self._transport,
                    "interface",
                    CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE,
                )
            )
        return CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE

    @property
    def kit_transport_available(self) -> bool:
        return self._transport is not None or self._kit_transport_cls is not None

    def put_bytes(
        self,
        data: bytes,
        *,
        claimed_cid: str | None = None,
    ) -> str:
        """Verify and atomically persist exact DAG-JSON bytes; return their CID."""

        if type(data) is not bytes or not data:
            raise ProofTestReuseObjectiveContractsError(
                "payload must be nonempty exact bytes",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        if len(data) > self.max_blob_bytes:
            raise ProofTestReuseObjectiveContractsError(
                "payload exceeds store budget",
                reason_code=ObjectiveArtifactReason.OVER_BUDGET,
            )
        derived = cid_for_canonical_dag_json_bytes(data)
        if claimed_cid is not None:
            validated = validate_artifact_cid(claimed_cid)
            if validated != derived:
                raise ProofTestReuseObjectiveContractsError(
                    "claimed CID does not match retained bytes",
                    reason_code=ObjectiveArtifactReason.CID_MISMATCH,
                )
        require_verified_cid(derived, data)

        if self._transport is not None:
            result = self._transport.put_bytes(data, claimed_cid=derived)
            stored = bool(getattr(result, "stored", False) or getattr(result, "ok", False))
            result_cid = _text(getattr(result, "cid", "")) or derived
            if not stored or result_cid != derived:
                reason = getattr(result, "reason_code", None)
                code = (
                    reason.value
                    if isinstance(reason, Enum)
                    else _text(reason) or ObjectiveArtifactReason.INTEGRITY_FAILED.value
                )
                raise ProofTestReuseObjectiveContractsError(
                    f"kit transport rejected put: {code}",
                    reason_code=ObjectiveArtifactReason.INTEGRITY_FAILED,
                )
            # Readback rehash via transport.
            loaded = self._transport.get_bytes(derived)
            loaded_data = getattr(loaded, "data", None)
            if type(loaded_data) is not bytes or loaded_data != data:
                raise ProofTestReuseObjectiveContractsError(
                    "kit transport readback rehash failed",
                    reason_code=ObjectiveArtifactReason.PARTIAL_WRITE,
                )
            return derived

        if self.local_root is None:
            raise ProofTestReuseObjectiveContractsError(
                "no artifact store backend is configured",
                reason_code=ObjectiveArtifactReason.STORE_UNAVAILABLE,
            )
        return self._local_atomic_put(data, derived)

    def _local_atomic_put(self, data: bytes, cid: str) -> str:
        root = self.local_root
        assert root is not None
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ProofTestReuseObjectiveContractsError(
                "store root must not be a symlink",
                reason_code=ObjectiveArtifactReason.SYMLINK_REJECTED,
            )
        target = _safe_local_blob_path(root, cid)
        if target.exists() and target.is_symlink():
            raise ProofTestReuseObjectiveContractsError(
                "refusing to write through a symlink blob",
                reason_code=ObjectiveArtifactReason.SYMLINK_REJECTED,
            )
        tmp_name = f".tmp.{os.getpid()}.{cid[:16]}.{int(self._clock() * 1000)}"
        tmp_path = (root / tmp_name).resolve()
        try:
            tmp_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ProofTestReuseObjectiveContractsError(
                "temporary path escapes store root",
                reason_code=ObjectiveArtifactReason.PATH_ESCAPE,
            ) from exc
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            # Readback rehash: partial/corrupt writes never report success.
            readback = target.read_bytes()
            if readback != data or not verify_retained_bytes(cid, readback):
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ProofTestReuseObjectiveContractsError(
                    "atomic write readback rehash failed",
                    reason_code=ObjectiveArtifactReason.PARTIAL_WRITE,
                )
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return cid

    def get_bytes(self, cid: str) -> bytes:
        """Load and re-verify retained bytes for *cid*."""

        validated = validate_artifact_cid(cid)
        if self._transport is not None:
            result = self._transport.get_bytes(validated)
            data = getattr(result, "data", None)
            hit = bool(getattr(result, "hit", False))
            if not hit or type(data) is not bytes:
                reason = getattr(result, "reason_code", None)
                reason_text = (
                    reason.value
                    if isinstance(reason, Enum)
                    else _text(reason)
                )
                mapped = {
                    "symlink_rejected": ObjectiveArtifactReason.SYMLINK_REJECTED,
                    "path_escape": ObjectiveArtifactReason.PATH_ESCAPE,
                    "integrity_failed": ObjectiveArtifactReason.INTEGRITY_FAILED,
                    "partial": ObjectiveArtifactReason.PARTIAL_WRITE,
                    "corrupt": ObjectiveArtifactReason.INTEGRITY_FAILED,
                    "cid_mismatch": ObjectiveArtifactReason.CID_MISMATCH,
                    "fake_cid": ObjectiveArtifactReason.FAKE_CID,
                    "wrong_codec": ObjectiveArtifactReason.WRONG_CODEC,
                    "over_budget": ObjectiveArtifactReason.OVER_BUDGET,
                    "not_found": ObjectiveArtifactReason.NOT_FOUND,
                    "unavailable": ObjectiveArtifactReason.STORE_UNAVAILABLE,
                    "store_unavailable": ObjectiveArtifactReason.STORE_UNAVAILABLE,
                }.get(reason_text, ObjectiveArtifactReason.NOT_FOUND)
                raise ProofTestReuseObjectiveContractsError(
                    f"artifact transport miss: {reason_text or mapped.value}",
                    reason_code=mapped,
                )
            require_verified_cid(validated, data)
            return data

        if self.local_root is None:
            raise ProofTestReuseObjectiveContractsError(
                "no artifact store backend is configured",
                reason_code=ObjectiveArtifactReason.STORE_UNAVAILABLE,
            )
        path = _safe_local_blob_path(self.local_root, validated)
        if path.is_symlink():
            raise ProofTestReuseObjectiveContractsError(
                "symlink blobs are rejected",
                reason_code=ObjectiveArtifactReason.SYMLINK_REJECTED,
            )
        if not path.is_file():
            raise ProofTestReuseObjectiveContractsError(
                "artifact blob not found",
                reason_code=ObjectiveArtifactReason.NOT_FOUND,
            )
        data = path.read_bytes()
        if not data:
            raise ProofTestReuseObjectiveContractsError(
                "partial or empty artifact blob",
                reason_code=ObjectiveArtifactReason.PARTIAL_WRITE,
            )
        if len(data) > self.max_blob_bytes:
            raise ProofTestReuseObjectiveContractsError(
                "stored blob exceeds budget",
                reason_code=ObjectiveArtifactReason.OVER_BUDGET,
            )
        require_verified_cid(validated, data)
        return data

    def put_completion_artifact(
        self, artifact: ProofTestReuseCompletionArtifact
    ) -> str:
        """Persist one completion artifact under its content CID."""

        if not isinstance(artifact, ProofTestReuseCompletionArtifact):
            raise ProofTestReuseObjectiveContractsError(
                "put_completion_artifact requires a typed artifact",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        data = artifact.canonical_bytes()
        return self.put_bytes(data, claimed_cid=artifact.artifact_cid)

    def get_completion_artifact(
        self, cid: str
    ) -> ProofTestReuseCompletionArtifact:
        """Load, rehash, and strictly deserialize a completion artifact."""

        data = self.get_bytes(cid)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofTestReuseObjectiveContractsError(
                "stored artifact is not canonical JSON",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            ) from exc
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "stored artifact must be a JSON object",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        artifact = ProofTestReuseCompletionArtifact.from_dict(payload)
        if artifact.artifact_cid != validate_artifact_cid(cid):
            raise ProofTestReuseObjectiveContractsError(
                "loaded artifact CID does not match requested CID",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        artifact.replay_premises()
        return artifact

    def put_gate_bundle(self, bundle: ProofTestReuseGateBundle) -> str:
        if not isinstance(bundle, ProofTestReuseGateBundle):
            raise ProofTestReuseObjectiveContractsError(
                "put_gate_bundle requires a typed gate bundle",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        return self.put_bytes(bundle.canonical_bytes(), claimed_cid=bundle.bundle_cid)

    def get_gate_bundle(self, cid: str) -> ProofTestReuseGateBundle:
        data = self.get_bytes(cid)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofTestReuseObjectiveContractsError(
                "stored gate bundle is not canonical JSON",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            ) from exc
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveContractsError(
                "stored gate bundle must be a JSON object",
                reason_code=ObjectiveArtifactReason.MALFORMED,
            )
        bundle = ProofTestReuseGateBundle.from_dict(payload)
        if bundle.bundle_cid != validate_artifact_cid(cid):
            raise ProofTestReuseObjectiveContractsError(
                "loaded bundle CID does not match requested CID",
                reason_code=ObjectiveArtifactReason.PROVENANCE_MISMATCH,
            )
        return bundle.replay()


__all__ = (
    "CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE",
    "CANONICAL_PREMISE_BLOCK_SCHEMA",
    "COMPLETION_EVIDENCE_INTERFACE",
    "DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES",
    "PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE",
    "PROOF_TEST_REUSE_COMPLETION_ARTIFACT_SCHEMA",
    "PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE",
    "PROOF_TEST_REUSE_GATE_BUNDLE_SCHEMA",
    "PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE",
    "PROOF_TEST_REUSE_OBJECTIVE_BINDING_SCHEMA",
    "PROOF_TEST_REUSE_OBJECTIVE_CONTRACTS_VERSION",
    "CanonicalPremiseBlock",
    "DecodedArtifactCID",
    "ObjectiveArtifactReason",
    "ObjectiveArtifactStore",
    "ProofTestReuseCompletionArtifact",
    "ProofTestReuseGateBundle",
    "ProofTestReuseObjectiveBinding",
    "ProofTestReuseObjectiveContractsError",
    "assert_control_paths_are_declared",
    "canonical_dag_json_bytes",
    "cid_for_canonical_dag_json_bytes",
    "cid_for_mapping",
    "compute_objective_completion_tree_id",
    "decode_artifact_cid",
    "declared_state_root_control_paths",
    "require_verified_cid",
    "validate_artifact_cid",
    "verify_retained_bytes",
)

"""DCR-070: admit exact proof-carrying repair packets (RPR@1).

Interfaces
----------
* ``ProofCarryingRepairPacket@1`` — frozen binding set for one repair.
* ``RPR@1`` / ``RepairPacketAdmission@1`` — fail-closed admission decision.

Normative rules (fail-closed)
-----------------------------
* Freezes epoch, finding, Doctor, Planner, operator, source spans/hashes,
  proof, impact, validations, inverse, owner, and lease bindings.
* Every referenced receipt CID must resolve to a stored canonical body whose
  content identity matches the CID (no synthetic / prose / boolean authority).
* Missing plan admission, stale roots, mismatched schedule/lease bindings, or
  unresolvable evidence reject **before** worktree creation.
* Only a packet built from *derived* evidence plus an *admitted* candidate /
  plan may grant execution (``grants_execution`` / ``allows_worktree_creation``).
* Runtime model calls remain 0.  Mutation still requires the separate
  materialization / transaction stages after admission.

Predicted symbols: :class:`ProofCarryingRepairPacket`,
:class:`RepairPacketAdmission`, :func:`admit_proof_carrying_repair_packet`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence / schemas
# ---------------------------------------------------------------------------

RPR_INTERFACE: Final[str] = "RPR@1"
PROOF_CARRYING_REPAIR_PACKET_INTERFACE: Final[str] = "ProofCarryingRepairPacket@1"
REPAIR_PACKET_ADMISSION_INTERFACE: Final[str] = "RepairPacketAdmission@1"
DCR_REPAIR_ADMISSION_EVIDENCE: Final[str] = "dcr/repair-admission@1"
SCA_RPR_ADMISSION_VERSION: Final[int] = 1

PROOF_CARRYING_REPAIR_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-packet@1"
)
REPAIR_PACKET_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-packet-admission@1"
)
SOURCE_SPAN_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-packet-source-span@1"
)
DEFAULT_ADMISSION_VECTORS_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/admission-vectors.json"
)

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_SPANS: Final[int] = 64
MAX_VALIDATIONS: Final[int] = 64
MAX_RECEIPTS: Final[int] = 256

# Closed set of frozen binding field names (effects / audit surface).
FROZEN_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "epoch_cid",
    "finding_cid",
    "doctor_receipt_cid",
    "planner_receipt_cid",
    "plan_cid",
    "operator_cid",
    "source_spans",
    "source_hashes",
    "proof_cid",
    "impact_cid",
    "validation_refs",
    "inverse_cid",
    "owner_root",
    "lease_id",
    "fencing_token",
    "schedule_cid",
    "candidate_admission_cid",
    "current_evidence_cid",
    "forest_cid",
    "git_tree_id",
    "policy_root",
)

# Receipt roles that must resolve against the stored receipt map.
REQUIRED_RECEIPT_BINDINGS: Final[tuple[str, ...]] = (
    "epoch_cid",
    "finding_cid",
    "doctor_receipt_cid",
    "planner_receipt_cid",
    "plan_cid",
    "operator_cid",
    "proof_cid",
    "impact_cid",
    "inverse_cid",
    "schedule_cid",
    "candidate_admission_cid",
    "current_evidence_cid",
    "forest_cid",
    "git_tree_id",
    "policy_root",
)

_ROOT_SPECS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "orchestration": {
            "role": "orchestration_only",
            "relative_path": ".",
            "writable": False,
        },
        "swissknife": {
            "role": "consumer",
            "relative_path": "swissknife",
            "writable": True,
        },
        "mcp-plus-plus": {
            "role": "consumer",
            "relative_path": "Mcp-Plus-Plus",
            "writable": True,
        },
        "ipfs-accelerate": {
            "role": "provider",
            "relative_path": "external/ipfs_accelerate",
            "writable": True,
        },
        "ipfs-datasets": {
            "role": "provider",
            "relative_path": "external/ipfs_datasets",
            "writable": True,
        },
        "ipfs-kit": {
            "role": "provider",
            "relative_path": "external/ipfs_kit",
            "writable": True,
        },
    }
)

_PROSE_MARKERS: Final[tuple[str, ...]] = (
    "def ",
    "class ",
    "import ",
    "#!/",
    "function ",
    "private_key",
    "BEGIN ",
    "password=",
    "```",
    "please ",
    "the model",
    "llm",
)

_SYNTHETIC_CID_LITERALS: Final[frozenset[str]] = frozenset(
    {
        "true",
        "false",
        "yes",
        "no",
        "none",
        "null",
        "ok",
        "pass",
        "passed",
        "admitted",
        "ready",
        "success",
        "1",
        "0",
        "synthetic",
        "placeholder",
        "todo",
        "tbd",
    }
)

_BOOLEAN_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "authorized",
        "grants_execution",
        "allows_worktree_creation",
        "grants_write_authority",
        "semantic_authority",
        "completion_authoritative",
        "proof_authoritative",
        "plan_admitted",
        "ready",
        "ok",
    }
)

# Content identities are opaque compact tokens (CIDv1 base32 or sha256:...).
_CID_RE = re.compile(r"^(?:b[a-z2-7]{20,}|sha256:[0-9a-f]{64}|[a-z][a-z0-9_.:@+/-]{7,255})$")


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class RepairPacketAdmissionError(ContractValidationError):
    """Malformed repair-packet admission input or closed-boundary violation."""


class AdmissionDisposition(str, Enum):
    """Closed outcomes for one repair-packet admission decision."""

    ADMITTED = "admitted"
    REJECTED = "rejected"


class AdmissionReason(str, Enum):
    """Stable fail-closed reason codes for DCR-070."""

    ADMITTED = "admitted"
    MISSING_BINDING = "missing_binding"
    BINDING_MISMATCH = "binding_mismatch"
    UNRESOLVABLE_RECEIPT = "unresolvable_receipt"
    RECEIPT_CID_MISMATCH = "receipt_cid_mismatch"
    SYNTHETIC_CID = "synthetic_cid"
    BOOLEAN_AUTHORITY = "boolean_authority"
    PROSE_AUTHORITY = "prose_authority"
    MISSING_PLAN_ADMISSION = "missing_plan_admission"
    PLAN_ADMISSION_REJECTED = "plan_admission_rejected"
    STALE_ROOT = "stale_root"
    UNKNOWN_OWNER_ROOT = "unknown_owner_root"
    ORCHESTRATION_WRITE = "orchestration_write"
    CROSS_ROOT_PATH = "cross_root_path"
    LEASE_MISMATCH = "lease_mismatch"
    SCHEDULE_NOT_VALID = "schedule_not_valid"
    CANDIDATE_NOT_ADMITTED = "candidate_not_admitted"
    EVIDENCE_NOT_DERIVED = "evidence_not_derived"
    EMPTY_SOURCE_SPANS = "empty_source_spans"
    SOURCE_HASH_MISMATCH = "source_hash_mismatch"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_INPUT = "malformed_input"
    ZERO_MODEL_CALLS = "zero_model_calls"
    WORKTREE_CREATION_DENIED = "worktree_creation_denied"
    EXECUTION_GRANTED = "execution_granted"
    AUTHORITY_TRANSITION = "authority_transition_derived_to_admitted"


class EvidenceStage(str, Enum):
    """Closed evidence stages that may participate in admission."""

    OBSERVED = "observed"
    DERIVED = "derived"
    ADMITTED = "admitted"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise RepairPacketAdmissionError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise RepairPacketAdmissionError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise RepairPacketAdmissionError(f"{name} is required")
    if "\x00" in text:
        raise RepairPacketAdmissionError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise RepairPacketAdmissionError(
            f"{AdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _bool(value: Any, name: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise RepairPacketAdmissionError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str = "payload") -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, CanonicalContract):
        return dict(value.to_dict())
    if not isinstance(value, Mapping):
        raise RepairPacketAdmissionError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_VALIDATIONS,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairPacketAdmissionError(f"{name} must be a sequence of identifiers")
    else:
        raw = values
    if required and not raw:
        raise RepairPacketAdmissionError(f"{name} is required")
    if len(raw) > limit:
        raise RepairPacketAdmissionError(
            f"{AdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, limit=MAX_PATH_BYTES)
    if text.startswith("/") or text.startswith("~") or ".." in PurePosixPath(text).parts:
        raise RepairPacketAdmissionError(f"{name} must be a relative in-repo path")
    normalized = PurePosixPath(text).as_posix()
    if normalized in {"", "."}:
        raise RepairPacketAdmissionError(f"{name} must not be empty")
    return normalized


def _owner_for_path(path: str) -> str | None:
    """Return the DCR owner root for ``path``, or None if unmatched."""

    posix = PurePosixPath(path).as_posix()
    best: str | None = None
    best_len = -1
    for root, spec in _ROOT_SPECS.items():
        if root == "orchestration":
            continue
        prefix = str(spec["relative_path"]).rstrip("/")
        if posix == prefix or posix.startswith(prefix + "/"):
            if len(prefix) > best_len:
                best = root
                best_len = len(prefix)
    return best


def _looks_like_prose(value: str) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in _PROSE_MARKERS):
        return True
    if "\n" in value and len(value) > 120:
        return True
    # Multi-sentence freeform text is never a content identity.
    if value.count(" ") >= 6 and not value.startswith(("b", "sha256:")):
        return True
    return False


def _is_synthetic_cid(value: str) -> bool:
    compact = value.strip()
    if not compact:
        return True
    lowered = compact.lower()
    if lowered in _SYNTHETIC_CID_LITERALS:
        return True
    if lowered.startswith("synthetic:") or lowered.startswith("placeholder:"):
        return True
    # Bare booleans / short tokens that snuck past the regex.
    if len(compact) < 8:
        return True
    # Compact content identities (CIDv1 base32 / sha256: / opaque tokens) are
    # accepted without prose scanning — base32 digests may contain marker
    # substrings like "def" or "llm" by chance.
    if compact.startswith("b") and _CID_RE.match(compact):
        return False
    if compact.startswith("sha256:") and _CID_RE.match(compact):
        return False
    if _looks_like_prose(compact):
        return True
    if not _CID_RE.match(compact):
        return True
    return False


def _reject_boolean_authority(payload: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Detect authority claimed only by boolean flags without evidence."""

    reasons: list[str] = []
    for key, value in payload.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if key in _BOOLEAN_AUTHORITY_KEYS and value is True:
            # Boolean true alone never authorizes; require independent receipts.
            reasons.append(f"{AdmissionReason.BOOLEAN_AUTHORITY.value}:{name}")
        if isinstance(value, Mapping):
            reasons.extend(_reject_boolean_authority(value, name))
    return reasons


def _canonical_receipt_body(body: Any) -> dict[str, Any]:
    if isinstance(body, CanonicalContract):
        return dict(body.to_dict())
    if not isinstance(body, Mapping):
        raise RepairPacketAdmissionError("receipt body must be a mapping")
    return {str(key): item for key, item in body.items()}


def _receipt_cid(body: Mapping[str, Any]) -> str:
    """Derive the content identity for a stored receipt body.

    Callers may embed an explicit ``content_id`` / ``cid`` that must match the
    derived identity; otherwise the identity is computed from the body with
    those identity keys stripped so reconstruction is stable.
    """

    payload = dict(body)
    claimed = payload.pop("content_id", None)
    claimed_cid = payload.pop("cid", None)
    derived = content_identity(payload)
    for claim in (claimed, claimed_cid):
        if claim in (None, ""):
            continue
        if not isinstance(claim, str) or claim != derived:
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.RECEIPT_CID_MISMATCH.value}:claimed"
            )
    return derived


# ---------------------------------------------------------------------------
# Source span binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpanBinding(CanonicalContract):
    """Exact source span with a content hash (``path`` + offsets + hash)."""

    SCHEMA: ClassVar[str] = SOURCE_SPAN_BINDING_SCHEMA

    path: str
    start_line: int
    end_line: int
    content_hash: str
    start_col: int = 0
    end_col: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        for name in ("start_line", "end_line", "start_col", "end_col"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RepairPacketAdmissionError(
                    f"{name} must be a non-negative integer"
                )
        if self.end_line < self.start_line:
            raise RepairPacketAdmissionError("end_line must be >= start_line")
        object.__setattr__(
            self,
            "content_hash",
            _text(self.content_hash, "content_hash"),
        )
        if _is_synthetic_cid(self.content_hash):
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.SYNTHETIC_CID.value}:content_hash"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_col": self.start_col,
            "end_col": self.end_col,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | Any) -> "SourceSpanBinding":
        data = _mapping(payload, "source_span")
        return cls(
            path=str(data.get("path") or ""),
            start_line=int(data.get("start_line") or data.get("start") or 0),
            end_line=int(data.get("end_line") or data.get("end") or 0),
            content_hash=str(
                data.get("content_hash")
                or data.get("hash")
                or data.get("sha256")
                or ""
            ),
            start_col=int(data.get("start_col") or 0),
            end_col=int(data.get("end_col") or 0),
        )


# ---------------------------------------------------------------------------
# ProofCarryingRepairPacket@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofCarryingRepairPacket(CanonicalContract):
    """Frozen proof-carrying repair packet (``ProofCarryingRepairPacket@1``).

    Construction only freezes caller-supplied bindings.  Execution authority is
    granted exclusively by :func:`admit_proof_carrying_repair_packet` after
    receipt resolution and root/plan/lease checks.
    """

    SCHEMA: ClassVar[str] = PROOF_CARRYING_REPAIR_PACKET_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_CARRYING_REPAIR_PACKET_INTERFACE

    epoch_cid: str
    finding_cid: str
    doctor_receipt_cid: str
    planner_receipt_cid: str
    plan_cid: str
    operator_cid: str
    source_spans: tuple[SourceSpanBinding, ...]
    source_hashes: Mapping[str, str]
    proof_cid: str
    impact_cid: str
    validation_refs: tuple[str, ...]
    inverse_cid: str
    owner_root: str
    lease_id: str
    fencing_token: str
    schedule_cid: str
    candidate_admission_cid: str
    current_evidence_cid: str
    forest_cid: str
    git_tree_id: str
    policy_root: str
    evidence_stage: EvidenceStage = EvidenceStage.DERIVED
    task_id: str = ""
    repair_id: str = ""
    write_paths: tuple[str, ...] = ()
    plan_admission_cid: str = ""
    runtime_model_calls: int = 0
    grants_execution: bool = False
    allows_worktree_creation: bool = False

    def __post_init__(self) -> None:
        for name in (
            "epoch_cid",
            "finding_cid",
            "doctor_receipt_cid",
            "planner_receipt_cid",
            "plan_cid",
            "operator_cid",
            "proof_cid",
            "impact_cid",
            "inverse_cid",
            "schedule_cid",
            "candidate_admission_cid",
            "current_evidence_cid",
            "forest_cid",
            "git_tree_id",
            "policy_root",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
            if _is_synthetic_cid(getattr(self, name)):
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SYNTHETIC_CID.value}:{name}"
                )

        # Lease / fence tokens are opaque identifiers, not content identities,
        # but still reject empty or prose values.
        for name in ("lease_id", "fencing_token"):
            token = _text(getattr(self, name), name)
            if _looks_like_prose(token) or token.lower() in _SYNTHETIC_CID_LITERALS:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SYNTHETIC_CID.value}:{name}"
                )
            object.__setattr__(self, name, token)

        owner = _text(self.owner_root, "owner_root")
        if owner not in _ROOT_SPECS:
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.UNKNOWN_OWNER_ROOT.value}:{owner}"
            )
        if owner == "orchestration" or not bool(_ROOT_SPECS[owner].get("writable")):
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.ORCHESTRATION_WRITE.value}:{owner}"
            )
        object.__setattr__(self, "owner_root", owner)

        if not isinstance(self.source_spans, Sequence) or isinstance(
            self.source_spans, (str, bytes, bytearray)
        ):
            raise RepairPacketAdmissionError("source_spans must be a sequence")
        if not self.source_spans:
            raise RepairPacketAdmissionError(
                AdmissionReason.EMPTY_SOURCE_SPANS.value
            )
        if len(self.source_spans) > MAX_SPANS:
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.BOUNDS_EXCEEDED.value}:source_spans"
            )
        spans: list[SourceSpanBinding] = []
        for item in self.source_spans:
            if isinstance(item, SourceSpanBinding):
                spans.append(item)
            elif isinstance(item, Mapping):
                spans.append(SourceSpanBinding.from_dict(item))
            else:
                raise RepairPacketAdmissionError(
                    "source_spans must contain SourceSpanBinding records"
                )
        # Deterministic span order.
        spans = sorted(spans, key=lambda s: (s.path, s.start_line, s.end_line, s.content_hash))
        object.__setattr__(self, "source_spans", tuple(spans))

        hashes = {
            _path(key, "source_hashes.path"): _text(value, "source_hashes.hash")
            for key, value in dict(self.source_hashes or {}).items()
        }
        for path, digest in hashes.items():
            if _is_synthetic_cid(digest):
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SYNTHETIC_CID.value}:source_hashes:{path}"
                )
        # Every span path must appear in source_hashes with the same digest.
        for span in spans:
            expected = hashes.get(span.path)
            if expected is None:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SOURCE_HASH_MISMATCH.value}:missing:{span.path}"
                )
            if expected != span.content_hash:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SOURCE_HASH_MISMATCH.value}:{span.path}"
                )
        object.__setattr__(
            self, "source_hashes", MappingProxyType(dict(sorted(hashes.items())))
        )

        object.__setattr__(
            self,
            "validation_refs",
            _ids(self.validation_refs, "validation_refs", required=True),
        )
        for ref in self.validation_refs:
            if _is_synthetic_cid(ref):
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SYNTHETIC_CID.value}:validation_refs"
                )

        write_paths = _ids(
            self.write_paths or tuple(hashes.keys()),
            "write_paths",
            required=True,
            limit=MAX_SPANS,
        )
        # Normalize write paths through the path checker.
        write_paths = tuple(_path(item, "write_paths") for item in write_paths)
        for path in write_paths:
            path_owner = _owner_for_path(path)
            if path_owner is None:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.CROSS_ROOT_PATH.value}:{path}"
                )
            if path_owner != owner:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.CROSS_ROOT_PATH.value}:{path}"
                )
        object.__setattr__(self, "write_paths", write_paths)

        try:
            stage = (
                self.evidence_stage
                if isinstance(self.evidence_stage, EvidenceStage)
                else EvidenceStage(str(self.evidence_stage))
            )
        except ValueError as exc:
            raise RepairPacketAdmissionError(
                f"unsupported evidence_stage: {self.evidence_stage!r}"
            ) from exc
        object.__setattr__(self, "evidence_stage", stage)

        object.__setattr__(self, "task_id", _optional_text(self.task_id, "task_id"))
        object.__setattr__(self, "repair_id", _optional_text(self.repair_id, "repair_id"))
        object.__setattr__(
            self,
            "plan_admission_cid",
            _optional_text(self.plan_admission_cid, "plan_admission_cid"),
        )
        if self.plan_admission_cid and _is_synthetic_cid(self.plan_admission_cid):
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.SYNTHETIC_CID.value}:plan_admission_cid"
            )

        # Packets never self-grant execution or worktree creation.
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_execution", False)
        object.__setattr__(self, "allows_worktree_creation", False)

    @property
    def packet_cid(self) -> str:
        return self.content_id

    def frozen_bindings(self) -> dict[str, Any]:
        """Project the frozen binding set (effects surface)."""

        return {
            "epoch_cid": self.epoch_cid,
            "finding_cid": self.finding_cid,
            "doctor_receipt_cid": self.doctor_receipt_cid,
            "planner_receipt_cid": self.planner_receipt_cid,
            "plan_cid": self.plan_cid,
            "operator_cid": self.operator_cid,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "source_hashes": dict(self.source_hashes),
            "proof_cid": self.proof_cid,
            "impact_cid": self.impact_cid,
            "validation_refs": list(self.validation_refs),
            "inverse_cid": self.inverse_cid,
            "owner_root": self.owner_root,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "schedule_cid": self.schedule_cid,
            "candidate_admission_cid": self.candidate_admission_cid,
            "current_evidence_cid": self.current_evidence_cid,
            "forest_cid": self.forest_cid,
            "git_tree_id": self.git_tree_id,
            "policy_root": self.policy_root,
            "plan_admission_cid": self.plan_admission_cid,
            "write_paths": list(self.write_paths),
            "evidence_stage": self.evidence_stage.value,
        }

    def referenced_receipt_cids(self) -> tuple[str, ...]:
        """All receipt CIDs that must resolve for admission."""

        cids: list[str] = []
        for name in REQUIRED_RECEIPT_BINDINGS:
            cids.append(str(getattr(self, name)))
        if self.plan_admission_cid:
            cids.append(self.plan_admission_cid)
        cids.extend(self.validation_refs)
        # Source hashes are content digests, not necessarily stored receipts.
        return tuple(sorted(set(cids)))

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "evidence_id": DCR_REPAIR_ADMISSION_EVIDENCE,
            "packet_cid": self.packet_cid,
            "referenced_receipt_cids": list(self.referenced_receipt_cids()),
            "frozen_bindings": self.frozen_bindings(),
            "runtime_model_calls": 0,
            "grants_execution": False,
            "allows_worktree_creation": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": PROOF_CARRYING_REPAIR_PACKET_INTERFACE,
            "epoch_cid": self.epoch_cid,
            "finding_cid": self.finding_cid,
            "doctor_receipt_cid": self.doctor_receipt_cid,
            "planner_receipt_cid": self.planner_receipt_cid,
            "plan_cid": self.plan_cid,
            "operator_cid": self.operator_cid,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "source_hashes": dict(self.source_hashes),
            "proof_cid": self.proof_cid,
            "impact_cid": self.impact_cid,
            "validation_refs": list(self.validation_refs),
            "inverse_cid": self.inverse_cid,
            "owner_root": self.owner_root,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "schedule_cid": self.schedule_cid,
            "candidate_admission_cid": self.candidate_admission_cid,
            "current_evidence_cid": self.current_evidence_cid,
            "forest_cid": self.forest_cid,
            "git_tree_id": self.git_tree_id,
            "policy_root": self.policy_root,
            "evidence_stage": self.evidence_stage.value,
            "task_id": self.task_id,
            "repair_id": self.repair_id,
            "write_paths": list(self.write_paths),
            "plan_admission_cid": self.plan_admission_cid,
            "runtime_model_calls": 0,
            "grants_execution": False,
            "allows_worktree_creation": False,
            "evidence_id": DCR_REPAIR_ADMISSION_EVIDENCE,
            "version": SCA_RPR_ADMISSION_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofCarryingRepairPacket":
        data = _mapping(payload, "repair_packet")
        stage = data.get("evidence_stage") or EvidenceStage.DERIVED.value
        return cls(
            epoch_cid=str(data.get("epoch_cid") or ""),
            finding_cid=str(data.get("finding_cid") or ""),
            doctor_receipt_cid=str(data.get("doctor_receipt_cid") or ""),
            planner_receipt_cid=str(data.get("planner_receipt_cid") or ""),
            plan_cid=str(data.get("plan_cid") or ""),
            operator_cid=str(data.get("operator_cid") or ""),
            source_spans=tuple(data.get("source_spans") or ()),
            source_hashes=dict(data.get("source_hashes") or {}),
            proof_cid=str(data.get("proof_cid") or ""),
            impact_cid=str(data.get("impact_cid") or ""),
            validation_refs=tuple(data.get("validation_refs") or ()),
            inverse_cid=str(data.get("inverse_cid") or ""),
            owner_root=str(data.get("owner_root") or ""),
            lease_id=str(data.get("lease_id") or ""),
            fencing_token=str(data.get("fencing_token") or ""),
            schedule_cid=str(data.get("schedule_cid") or ""),
            candidate_admission_cid=str(data.get("candidate_admission_cid") or ""),
            current_evidence_cid=str(data.get("current_evidence_cid") or ""),
            forest_cid=str(data.get("forest_cid") or ""),
            git_tree_id=str(data.get("git_tree_id") or ""),
            policy_root=str(data.get("policy_root") or ""),
            evidence_stage=str(stage),
            task_id=str(data.get("task_id") or ""),
            repair_id=str(data.get("repair_id") or ""),
            write_paths=tuple(data.get("write_paths") or ()),
            plan_admission_cid=str(data.get("plan_admission_cid") or ""),
        )


# ---------------------------------------------------------------------------
# RepairPacketAdmission / RPR@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairPacketAdmission(CanonicalContract):
    """``RPR@1`` / ``RepairPacketAdmission@1`` — admit or reject one packet."""

    SCHEMA: ClassVar[str] = REPAIR_PACKET_ADMISSION_SCHEMA
    INTERFACE: ClassVar[str] = RPR_INTERFACE

    packet_cid: str
    disposition: AdmissionDisposition
    reason_codes: tuple[str, ...] = ()
    referenced_receipt_cids: tuple[str, ...] = ()
    resolved_receipt_cids: tuple[str, ...] = ()
    frozen_bindings: Mapping[str, Any] = MappingProxyType({})
    authority_transition: str = ""
    canonical_reconstruction_cid: str = ""
    runtime_model_calls: int = 0
    grants_execution: bool = False
    allows_worktree_creation: bool = False
    evidence_stage: EvidenceStage = EvidenceStage.DERIVED

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_cid", _text(self.packet_cid, "packet_cid"))
        try:
            disposition = (
                self.disposition
                if isinstance(self.disposition, AdmissionDisposition)
                else AdmissionDisposition(str(self.disposition))
            )
        except ValueError as exc:
            raise RepairPacketAdmissionError(
                f"unsupported disposition: {self.disposition!r}"
            ) from exc
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", limit=128)
        )
        object.__setattr__(
            self,
            "referenced_receipt_cids",
            _ids(
                self.referenced_receipt_cids,
                "referenced_receipt_cids",
                limit=MAX_RECEIPTS,
            ),
        )
        object.__setattr__(
            self,
            "resolved_receipt_cids",
            _ids(
                self.resolved_receipt_cids,
                "resolved_receipt_cids",
                limit=MAX_RECEIPTS,
            ),
        )
        frozen = dict(self.frozen_bindings or {})
        object.__setattr__(
            self, "frozen_bindings", MappingProxyType(dict(sorted(frozen.items(), key=lambda kv: kv[0])))
        )
        object.__setattr__(
            self,
            "authority_transition",
            _optional_text(self.authority_transition, "authority_transition"),
        )
        object.__setattr__(
            self,
            "canonical_reconstruction_cid",
            _optional_text(
                self.canonical_reconstruction_cid, "canonical_reconstruction_cid"
            ),
        )
        try:
            stage = (
                self.evidence_stage
                if isinstance(self.evidence_stage, EvidenceStage)
                else EvidenceStage(str(self.evidence_stage))
            )
        except ValueError as exc:
            raise RepairPacketAdmissionError(
                f"unsupported evidence_stage: {self.evidence_stage!r}"
            ) from exc
        object.__setattr__(self, "evidence_stage", stage)

        # Authority hard-fail closed: only ADMITTED may grant execution.
        admitted = disposition is AdmissionDisposition.ADMITTED
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_execution", bool(admitted))
        object.__setattr__(self, "allows_worktree_creation", bool(admitted))
        if admitted:
            if not self.authority_transition:
                object.__setattr__(
                    self,
                    "authority_transition",
                    "derived->admitted",
                )
            if AdmissionReason.ADMITTED.value not in self.reason_codes:
                object.__setattr__(
                    self,
                    "reason_codes",
                    tuple(
                        list(self.reason_codes)
                        + [
                            AdmissionReason.ADMITTED.value,
                            AdmissionReason.EXECUTION_GRANTED.value,
                            AdmissionReason.AUTHORITY_TRANSITION.value,
                            AdmissionReason.ZERO_MODEL_CALLS.value,
                        ]
                    ),
                )
        else:
            if AdmissionReason.WORKTREE_CREATION_DENIED.value not in self.reason_codes:
                object.__setattr__(
                    self,
                    "reason_codes",
                    tuple(
                        list(self.reason_codes)
                        + [AdmissionReason.WORKTREE_CREATION_DENIED.value]
                    ),
                )

    @property
    def ok(self) -> bool:
        return self.disposition is AdmissionDisposition.ADMITTED

    @property
    def admitted(self) -> bool:
        return self.ok

    @property
    def admission_cid(self) -> str:
        return self.content_id

    def evidence_subset(self) -> dict[str, Any]:
        """Project the DCR-070 evidence subset."""

        return {
            "evidence_id": DCR_REPAIR_ADMISSION_EVIDENCE,
            "packet_cid": self.packet_cid,
            "admission_cid": self.admission_cid,
            "referenced_receipt_cids": list(self.referenced_receipt_cids),
            "resolved_receipt_cids": list(self.resolved_receipt_cids),
            "canonical_reconstruction_cid": self.canonical_reconstruction_cid,
            "authority_transition": self.authority_transition,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "grants_execution": self.grants_execution,
            "allows_worktree_creation": self.allows_worktree_creation,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": RPR_INTERFACE,
            "admission_interface": REPAIR_PACKET_ADMISSION_INTERFACE,
            "packet_cid": self.packet_cid,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "referenced_receipt_cids": list(self.referenced_receipt_cids),
            "resolved_receipt_cids": list(self.resolved_receipt_cids),
            "frozen_bindings": dict(self.frozen_bindings),
            "authority_transition": self.authority_transition,
            "canonical_reconstruction_cid": self.canonical_reconstruction_cid,
            "runtime_model_calls": 0,
            "grants_execution": self.grants_execution,
            "allows_worktree_creation": self.allows_worktree_creation,
            "evidence_stage": self.evidence_stage.value,
            "evidence_id": DCR_REPAIR_ADMISSION_EVIDENCE,
            "version": SCA_RPR_ADMISSION_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairPacketAdmission":
        data = _mapping(payload, "admission")
        return cls(
            packet_cid=str(data.get("packet_cid") or ""),
            disposition=str(
                data.get("disposition") or AdmissionDisposition.REJECTED.value
            ),
            reason_codes=tuple(data.get("reason_codes") or ()),
            referenced_receipt_cids=tuple(data.get("referenced_receipt_cids") or ()),
            resolved_receipt_cids=tuple(data.get("resolved_receipt_cids") or ()),
            frozen_bindings=dict(data.get("frozen_bindings") or {}),
            authority_transition=str(data.get("authority_transition") or ""),
            canonical_reconstruction_cid=str(
                data.get("canonical_reconstruction_cid") or ""
            ),
            evidence_stage=str(
                data.get("evidence_stage") or EvidenceStage.DERIVED.value
            ),
        )


# ---------------------------------------------------------------------------
# Receipt resolution + admission
# ---------------------------------------------------------------------------


def build_receipt_store(
    receipts: Mapping[str, Any] | Sequence[Any] | None,
) -> Mapping[str, Mapping[str, Any]]:
    """Normalize a CID→body map or a sequence of bodies into a resolvable store.

    When a sequence is supplied, each body's content identity becomes the key.
    When a mapping is supplied, each key must equal the body's derived identity
    (fail closed on mismatch).
    """

    store: dict[str, dict[str, Any]] = {}
    if receipts is None:
        return MappingProxyType({})
    if isinstance(receipts, Mapping):
        for key, body in receipts.items():
            cid_key = _text(key, "receipt_cid")
            if _is_synthetic_cid(cid_key):
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.SYNTHETIC_CID.value}:receipt_store"
                )
            payload = _canonical_receipt_body(body)
            derived = _receipt_cid(payload)
            if cid_key != derived:
                raise RepairPacketAdmissionError(
                    f"{AdmissionReason.RECEIPT_CID_MISMATCH.value}:{cid_key}"
                )
            store[cid_key] = payload
        if len(store) > MAX_RECEIPTS:
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.BOUNDS_EXCEEDED.value}:receipts"
            )
        return MappingProxyType(store)

    if isinstance(receipts, Sequence) and not isinstance(
        receipts, (str, bytes, bytearray)
    ):
        for body in receipts:
            payload = _canonical_receipt_body(body)
            cid = _receipt_cid(payload)
            store[cid] = payload
        if len(store) > MAX_RECEIPTS:
            raise RepairPacketAdmissionError(
                f"{AdmissionReason.BOUNDS_EXCEEDED.value}:receipts"
            )
        return MappingProxyType(store)

    raise RepairPacketAdmissionError("receipts must be a mapping or sequence")


def _resolve_receipts(
    packet: ProofCarryingRepairPacket,
    store: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], list[str]]:
    """Return (resolved_cids, reason_codes) for the packet's receipt set."""

    reasons: list[str] = []
    resolved: list[str] = []
    for cid in packet.referenced_receipt_cids():
        if _is_synthetic_cid(cid):
            reasons.append(f"{AdmissionReason.SYNTHETIC_CID.value}:{cid}")
            continue
        body = store.get(cid)
        if body is None:
            reasons.append(f"{AdmissionReason.UNRESOLVABLE_RECEIPT.value}:{cid}")
            continue
        try:
            derived = _receipt_cid(body)
        except RepairPacketAdmissionError as exc:
            reasons.append(str(exc))
            continue
        if derived != cid:
            reasons.append(f"{AdmissionReason.RECEIPT_CID_MISMATCH.value}:{cid}")
            continue
        resolved.append(cid)
    return tuple(sorted(set(resolved))), reasons


def _check_current_roots(
    packet: ProofCarryingRepairPacket,
    current_roots: Mapping[str, Any] | None,
) -> list[str]:
    if current_roots is None:
        # No live root projection supplied: still require packet roots to be
        # non-synthetic (already enforced) but do not invent freshness.
        return []
    roots = _mapping(current_roots, "current_roots")
    reasons: list[str] = []
    expected = {
        "forest_cid": packet.forest_cid,
        "git_tree_id": packet.git_tree_id,
        "policy_root": packet.policy_root,
        "epoch_cid": packet.epoch_cid,
        "current_evidence_cid": packet.current_evidence_cid,
    }
    # Accept common aliases.
    aliases = {
        "forest_cid": ("forest_cid", "repository_forest_cid", "forest"),
        "git_tree_id": ("git_tree_id", "tree_id", "git_tree"),
        "policy_root": ("policy_root", "policy_id", "policy"),
        "epoch_cid": ("epoch_cid", "epoch"),
        "current_evidence_cid": (
            "current_evidence_cid",
            "evidence_cid",
            "current_evidence",
        ),
    }
    for field, expected_value in expected.items():
        observed = ""
        for alias in aliases[field]:
            if alias in roots and roots[alias] not in (None, ""):
                observed = str(roots[alias]).strip()
                break
        if not observed:
            reasons.append(f"{AdmissionReason.STALE_ROOT.value}:missing:{field}")
            continue
        if observed != expected_value:
            reasons.append(f"{AdmissionReason.STALE_ROOT.value}:{field}")
    return reasons


def _check_plan_admission(
    packet: ProofCarryingRepairPacket,
    plan_admission: Mapping[str, Any] | CanonicalContract | None,
    store: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if plan_admission is None and not packet.plan_admission_cid:
        reasons.append(AdmissionReason.MISSING_PLAN_ADMISSION.value)
        return reasons

    payload: dict[str, Any]
    if plan_admission is not None:
        payload = _mapping(plan_admission, "plan_admission")
    else:
        body = store.get(packet.plan_admission_cid)
        if body is None:
            reasons.append(AdmissionReason.MISSING_PLAN_ADMISSION.value)
            reasons.append(
                f"{AdmissionReason.UNRESOLVABLE_RECEIPT.value}:{packet.plan_admission_cid}"
            )
            return reasons
        payload = dict(body)

    disposition = str(
        payload.get("disposition")
        or payload.get("verdict")
        or payload.get("status")
        or ""
    ).strip().lower()
    selected = str(
        payload.get("selected_candidate_cid")
        or payload.get("admitted_candidate_cid")
        or payload.get("candidate_cid")
        or ""
    ).strip()
    admitted_flag = payload.get("admitted")
    ok_flag = payload.get("ok")

    selected_ok = disposition in {
        "selected",
        "admitted",
        "accepted",
        "scheduled",
        "compiled",
    }
    # Booleans alone never select; they may only confirm an explicit disposition.
    boolean_only = (
        not selected_ok
        and (
            (isinstance(admitted_flag, bool) and admitted_flag)
            or (isinstance(ok_flag, bool) and ok_flag)
        )
    )
    if boolean_only:
        reasons.extend(_reject_boolean_authority(payload, "plan_admission"))
        reasons.append(AdmissionReason.PLAN_ADMISSION_REJECTED.value)
        reasons.append(AdmissionReason.CANDIDATE_NOT_ADMITTED.value)
        return reasons

    # Candidate admission on the packet must match when provided.
    admission_cid = str(
        payload.get("content_id")
        or payload.get("cid")
        or payload.get("admission_cid")
        or ""
    ).strip()
    if packet.plan_admission_cid and admission_cid and packet.plan_admission_cid != admission_cid:
        # Recompute identity from body if claim missing/wrong.
        try:
            derived = _receipt_cid(payload)
        except RepairPacketAdmissionError:
            derived = ""
        if derived and packet.plan_admission_cid != derived:
            reasons.append(
                f"{AdmissionReason.BINDING_MISMATCH.value}:plan_admission_cid"
            )

    if not selected_ok:
        # Explicit non-success disposition, or no disposition at all.
        if any(
            key in payload and payload[key] is True for key in _BOOLEAN_AUTHORITY_KEYS
        ):
            reasons.extend(_reject_boolean_authority(payload, "plan_admission"))
        reasons.append(AdmissionReason.PLAN_ADMISSION_REJECTED.value)
        reasons.append(AdmissionReason.CANDIDATE_NOT_ADMITTED.value)

    return reasons


def _check_schedule_and_lease(
    packet: ProofCarryingRepairPacket,
    schedule: Mapping[str, Any] | CanonicalContract | None,
    store: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    payload: dict[str, Any] | None = None
    if schedule is not None:
        payload = _mapping(schedule, "schedule")
    else:
        body = store.get(packet.schedule_cid)
        if body is not None:
            payload = dict(body)

    if payload is None:
        # Schedule receipt must at least resolve (already covered by receipt
        # resolution).  Without a body we cannot check lease/fence bindings.
        return reasons

    schedule_cid = str(
        payload.get("content_id")
        or payload.get("cid")
        or payload.get("schedule_cid")
        or ""
    ).strip()
    if schedule_cid and schedule_cid != packet.schedule_cid:
        try:
            derived = _receipt_cid(payload)
        except RepairPacketAdmissionError:
            derived = ""
        if derived and derived != packet.schedule_cid:
            reasons.append(f"{AdmissionReason.BINDING_MISMATCH.value}:schedule_cid")

    disposition = str(payload.get("disposition") or "").strip().lower()
    if disposition and disposition not in {"scheduled", "admitted", "accepted"}:
        reasons.append(AdmissionReason.SCHEDULE_NOT_VALID.value)

    # Locate a matching lease/fence assignment.
    assignments = payload.get("assignments") or payload.get("nodes") or ()
    lease_plan = payload.get("path_lease_plan") or payload.get("leases") or {}
    leases: Sequence[Any]
    if isinstance(lease_plan, Mapping):
        leases = lease_plan.get("leases") or ()
    elif isinstance(lease_plan, Sequence) and not isinstance(
        lease_plan, (str, bytes, bytearray)
    ):
        leases = lease_plan
    else:
        leases = ()

    lease_ids: set[str] = set()
    fencing_tokens: set[str] = set()
    for item in list(assignments) + list(leases):
        if not isinstance(item, Mapping):
            continue
        lid = str(item.get("lease_id") or item.get("lease") or "").strip()
        fence = str(
            item.get("fencing_token") or item.get("fence_token") or item.get("fence") or ""
        ).strip()
        if lid:
            lease_ids.add(lid)
        if fence:
            fencing_tokens.add(fence)

    if lease_ids and packet.lease_id not in lease_ids:
        reasons.append(f"{AdmissionReason.LEASE_MISMATCH.value}:lease_id")
    if fencing_tokens and packet.fencing_token not in fencing_tokens:
        reasons.append(f"{AdmissionReason.LEASE_MISMATCH.value}:fencing_token")

    return reasons


def _check_evidence_stage(packet: ProofCarryingRepairPacket) -> list[str]:
    if packet.evidence_stage is EvidenceStage.OBSERVED:
        return [AdmissionReason.EVIDENCE_NOT_DERIVED.value]
    # DERIVED and ADMITTED (re-admission) are acceptable inputs; the admission
    # decision itself advances DERIVED -> ADMITTED.
    return []


def admit_proof_carrying_repair_packet(
    packet: ProofCarryingRepairPacket | Mapping[str, Any],
    *,
    receipts: Mapping[str, Any] | Sequence[Any] | None = None,
    current_roots: Mapping[str, Any] | None = None,
    plan_admission: Mapping[str, Any] | CanonicalContract | None = None,
    schedule: Mapping[str, Any] | CanonicalContract | None = None,
) -> RepairPacketAdmission:
    """Admit one proof-carrying repair packet or reject before worktree creation.

    Parameters
    ----------
    packet:
        A :class:`ProofCarryingRepairPacket` or its canonical mapping.
    receipts:
        Stored resolvable receipt bodies (CID→body map or body sequence).
    current_roots:
        Live forest / tree / policy / epoch projection that must match the
        packet bindings exactly.
    plan_admission:
        Candidate / plan admission decision (must be selected/admitted).
    schedule:
        Resource schedule that must include the packet's lease and fence.

    Returns
    -------
    RepairPacketAdmission
        ``disposition=admitted`` only when every binding resolves and only
        derived + admitted evidence authorizes execution.  Rejection never
        sets ``allows_worktree_creation``.
    """

    try:
        if isinstance(packet, ProofCarryingRepairPacket):
            frozen = packet
        else:
            frozen = ProofCarryingRepairPacket.from_dict(_mapping(packet, "packet"))
        store = build_receipt_store(receipts)
    except RepairPacketAdmissionError as exc:
        return RepairPacketAdmission(
            packet_cid=_optional_text(
                getattr(packet, "packet_cid", None)
                or (packet.get("packet_cid") if isinstance(packet, Mapping) else None)
                or "packet:unconstructed",
                "packet_cid",
            )
            or "packet:unconstructed",
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(str(exc), AdmissionReason.MALFORMED_INPUT.value),
            grants_execution=False,
            allows_worktree_creation=False,
        )

    reasons: list[str] = []
    reasons.extend(_check_evidence_stage(frozen))
    resolved, resolve_reasons = _resolve_receipts(frozen, store)
    reasons.extend(resolve_reasons)
    reasons.extend(_check_current_roots(frozen, current_roots))
    reasons.extend(_check_plan_admission(frozen, plan_admission, store))
    reasons.extend(_check_schedule_and_lease(frozen, schedule, store))

    # Canonical reconstruction: re-hydrate from to_dict and require identity.
    reconstruction = ProofCarryingRepairPacket.from_dict(frozen.to_dict())
    reconstruction_cid = reconstruction.packet_cid
    if reconstruction_cid != frozen.packet_cid:
        reasons.append(f"{AdmissionReason.BINDING_MISMATCH.value}:canonical_reconstruction")

    # Deduplicate reasons while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for code in reasons:
        if code and code not in seen:
            seen.add(code)
            deduped.append(code)

    if deduped:
        return RepairPacketAdmission(
            packet_cid=frozen.packet_cid,
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=tuple(deduped),
            referenced_receipt_cids=frozen.referenced_receipt_cids(),
            resolved_receipt_cids=resolved,
            frozen_bindings=frozen.frozen_bindings(),
            authority_transition="",
            canonical_reconstruction_cid=reconstruction_cid,
            grants_execution=False,
            allows_worktree_creation=False,
            evidence_stage=frozen.evidence_stage,
        )

    return RepairPacketAdmission(
        packet_cid=frozen.packet_cid,
        disposition=AdmissionDisposition.ADMITTED,
        reason_codes=(
            AdmissionReason.ADMITTED.value,
            AdmissionReason.EXECUTION_GRANTED.value,
            AdmissionReason.AUTHORITY_TRANSITION.value,
            AdmissionReason.ZERO_MODEL_CALLS.value,
        ),
        referenced_receipt_cids=frozen.referenced_receipt_cids(),
        resolved_receipt_cids=resolved,
        frozen_bindings=frozen.frozen_bindings(),
        authority_transition="derived->admitted",
        canonical_reconstruction_cid=reconstruction_cid,
        grants_execution=True,
        allows_worktree_creation=True,
        evidence_stage=EvidenceStage.ADMITTED,
    )


# Public aliases matching predicted symbols / SCA bridge naming.
admit_repair_packet = admit_proof_carrying_repair_packet
ScaRprAdmission = RepairPacketAdmission


def materialize_admission_vectors(
    cases: Sequence[Mapping[str, Any]] | None = None,
    *,
    destination: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact admission-vector catalog (optional artifact writer).

    The generated artifact path is advisory; callers control writes.  Unit
    tests exercise admission in-memory without requiring the JSON file.
    """

    vectors: list[dict[str, Any]] = []
    for case in cases or ():
        packet_payload = case.get("packet") or case
        receipts = case.get("receipts")
        current_roots = case.get("current_roots")
        plan_admission = case.get("plan_admission")
        schedule = case.get("schedule")
        admission = admit_proof_carrying_repair_packet(
            packet_payload,
            receipts=receipts,
            current_roots=current_roots,
            plan_admission=plan_admission,
            schedule=schedule,
        )
        vectors.append(
            {
                "case_id": str(case.get("case_id") or admission.packet_cid),
                "packet_cid": admission.packet_cid,
                "disposition": admission.disposition.value,
                "admission_cid": admission.admission_cid,
                "reason_codes": list(admission.reason_codes),
                "grants_execution": admission.grants_execution,
                "allows_worktree_creation": admission.allows_worktree_creation,
                "evidence_subset": admission.evidence_subset(),
            }
        )

    catalog = {
        "schema": "ipfs_accelerate_py/agent-supervisor/dcr-admission-vectors@1",
        "evidence_id": DCR_REPAIR_ADMISSION_EVIDENCE,
        "interface": RPR_INTERFACE,
        "version": SCA_RPR_ADMISSION_VERSION,
        "runtime_model_calls": 0,
        "vectors": vectors,
    }

    if destination is not None or repository_root is not None:
        root = Path(repository_root) if repository_root is not None else Path(".")
        path = (
            Path(destination)
            if destination is not None
            else root / DEFAULT_ADMISSION_VECTORS_REL
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        catalog = {**catalog, "path": path.as_posix()}

    return catalog


# Convenience projection for tests / audits.
ADMISSION_REASON_CODES: Final[frozenset[str]] = frozenset(
    item.value for item in AdmissionReason
)


__all__ = [
    "ADMISSION_REASON_CODES",
    "AdmissionDisposition",
    "AdmissionReason",
    "DEFAULT_ADMISSION_VECTORS_REL",
    "DCR_REPAIR_ADMISSION_EVIDENCE",
    "EvidenceStage",
    "FROZEN_BINDING_FIELDS",
    "PROOF_CARRYING_REPAIR_PACKET_INTERFACE",
    "ProofCarryingRepairPacket",
    "REPAIR_PACKET_ADMISSION_INTERFACE",
    "RPR_INTERFACE",
    "RepairPacketAdmission",
    "RepairPacketAdmissionError",
    "SCA_RPR_ADMISSION_VERSION",
    "ScaRprAdmission",
    "SourceSpanBinding",
    "admit_proof_carrying_repair_packet",
    "admit_repair_packet",
    "build_receipt_store",
    "materialize_admission_vectors",
]

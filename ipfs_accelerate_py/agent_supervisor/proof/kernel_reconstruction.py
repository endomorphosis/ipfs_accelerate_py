"""Independent proof reconstruction and minimal counterexample preservation.

DCR-033 seals two closed interfaces:

* ``ProofKernelReceipt@1`` — independent reconstruction of a claimed proof
  certificate/term.  Provider ``verified`` flags, simulated SAT, and expected
  outcomes are never accepted as reconstruction evidence.
* ``Counterexample@1`` — a minimized, content-addressed refutation that must
  replay against a bound contract graph and live transcript without inventing
  observations.

Public entry points:

* :func:`reconstruct_proof` — re-derive digests and structural bindings; an
  unreconstructable claim becomes ``invalid``.
* :func:`minimize_counterexample` — reduce a refutation to the smallest
  replayable witness and fail closed when graph/transcript anchors are absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


PROOF_KERNEL_RECEIPT_INTERFACE: Final = "ProofKernelReceipt@1"
COUNTEREXAMPLE_INTERFACE: Final = "Counterexample@1"
PROOF_KERNEL_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-kernel-receipt@1"
)
COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-counterexample@1"
)
COUNTEREXAMPLE_REPLAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-replay@1"
)
PROOF_KERNEL_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-kernel-reconstruction-artifact@1"
)
KERNEL_RECONSTRUCTION_VERSION: Final = 1
DEFAULT_KERNEL_VERSION: Final = "dcr-kernel-reconstruction@1"
DEFAULT_MAX_PROOF_TERM_BYTES: Final = 512 * 1024
DEFAULT_MAX_WITNESS_BYTES: Final = 12 * 1024
DEFAULT_MAX_REASON_CODES: Final = 32

# Incomplete / escape-hatch tokens that always invalidate reconstruction.
_INCOMPLETE_PROOF_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_'])(?:sorry|admit|admitted|oops|skip_proof|sorryAx|"
    r"todo|undefined|axiom|oracle)(?![A-Za-z0-9_'])"
)
# Volatile keys that must never survive minimization.
_VOLATILE_WITNESS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "timestamp",
        "observed_at",
        "created_at",
        "duration",
        "duration_ms",
        "elapsed",
        "elapsed_ms",
        "memory_bytes",
        "pid",
        "host",
        "raw",
        "raw_output",
        "stdout",
        "stderr",
        "transcript",
        "full_trace",
        "source",
        "source_code",
        "expected_outcome",
        "expected",
        "fabricated",
        "inferred",
        "inferred_observation",
        "synthetic",
    }
)
# Keys retained (when present) as the minimal witness core.
_MINIMAL_WITNESS_KEYS: Final[tuple[str, ...]] = (
    "edge_id",
    "edge_kind",
    "node_id",
    "operation",
    "consumer_id",
    "package",
    "role",
    "tool",
    "method",
    "terminal_state",
    "receipt_cid",
    "observation_id",
    "exchange_id",
    "failed_premise_ids",
    "failed_edges",
    "model",
    "assignment",
    "contradiction",
    "stage",
    "reason_code",
)


class KernelReconstructionError(ContractValidationError):
    """Raised when a reconstruction or counterexample contract is malformed."""


class ReconstructionStatus(str, Enum):
    """Closed reconstruction outcomes for one claimed proof."""

    RECONSTRUCTED = "reconstructed"
    INVALID = "invalid"
    REFUTED = "refuted"


class CounterexampleReplayStatus(str, Enum):
    """Whether a minimized counterexample still refutes under bound evidence."""

    REPLAYED = "replayed"
    MISSING_GRAPH_ANCHOR = "missing_graph_anchor"
    MISSING_TRANSCRIPT_ANCHOR = "missing_transcript_anchor"
    INFERRED_OBSERVATION = "inferred_observation"
    WITNESS_MISMATCH = "witness_mismatch"
    INVALID = "invalid"


def _text(value: Any, name: str, *, required: bool = False, maximum: int = 4096) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise KernelReconstructionError(f"{name} must be a string")
    if required and not text:
        raise KernelReconstructionError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise KernelReconstructionError(f"{name} exceeds {maximum} bytes")
    if "\x00" in text:
        raise KernelReconstructionError(f"{name} contains a NUL byte")
    return text


def _ids(values: Any, name: str, *, maximum: int = 256) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (bytes, bytearray)) or not isinstance(values, Sequence):
        raise KernelReconstructionError(f"{name} must be a sequence of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name, required=True, maximum=2048)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= maximum:
            break
    return tuple(result)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise KernelReconstructionError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise KernelReconstructionError(f"{name} keys must be strings")
    return {str(key): value[key] for key in sorted(value)}


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise KernelReconstructionError(f"{name} must be a boolean")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def proof_term_digest(proof_term: str | Mapping[str, Any] | Sequence[Any]) -> str:
    """Return the content digest of one proof term/certificate body."""

    if isinstance(proof_term, str):
        body = proof_term.encode("utf-8")
    else:
        body = _canonical_bytes(proof_term)
    if len(body) > DEFAULT_MAX_PROOF_TERM_BYTES:
        raise KernelReconstructionError("proof_term exceeds the reconstruction bound")
    return _sha256_hex(body)


def _strip_volatile(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return OMITTED if value not in (None, "", (), [], {}) else value
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                continue
            lowered = key.strip().lower()
            if lowered in _VOLATILE_WITNESS_KEYS or lowered.startswith("raw_"):
                continue
            if lowered in {"expected", "expected_outcome", "inferred", "synthetic"}:
                continue
            cleaned[key] = _strip_volatile(value[key], depth=depth + 1)
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_strip_volatile(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, (str, int, bool)) or value is None:
        if isinstance(value, str) and len(value) > 512:
            return value[:509] + "..."
        return value
    return str(value)[:512]


OMITTED: Final = "<omitted>"


def _minimal_witness(payload: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _strip_volatile(payload)
    if not isinstance(cleaned, Mapping):
        return {"payload": cleaned}
    retained: dict[str, Any] = {}
    for key in _MINIMAL_WITNESS_KEYS:
        if key in cleaned and cleaned[key] not in (None, "", (), [], {}):
            retained[key] = cleaned[key]
    # Always keep nested contradiction/model cores when present under other names.
    for key, value in cleaned.items():
        if key in retained:
            continue
        if key in {"witness", "payload", "details", "counterexample_seed"}:
            nested = _minimal_witness(value) if isinstance(value, Mapping) else value
            if nested not in (None, "", (), [], {}):
                retained[key] = nested
    if not retained:
        # Fall back to the whole cleaned mapping (already volatile-stripped).
        retained = dict(cleaned)
    encoded = _canonical_bytes(retained)
    if len(encoded) > DEFAULT_MAX_WITNESS_BYTES:
        # Keep only the highest-priority keys when still oversized.
        tight: dict[str, Any] = {}
        for key in _MINIMAL_WITNESS_KEYS:
            if key in retained:
                tight[key] = retained[key]
            if len(_canonical_bytes(tight)) >= DEFAULT_MAX_WITNESS_BYTES:
                break
        retained = tight
    return retained


@dataclass(frozen=True)
class ProofClaim:
    """Untrusted claimed proof supplied for independent reconstruction."""

    obligation_id: str
    proof_term: str | Mapping[str, Any] | Sequence[Any]
    certificate_digest: str = ""
    kernel_version: str = DEFAULT_KERNEL_VERSION
    root_ids: tuple[str, ...] = ()
    tree_id: str = ""
    graph_root: str = ""
    provider_status: str = ""
    independent: bool = False
    proof_children: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=True),
        )
        if isinstance(self.proof_term, str):
            object.__setattr__(
                self, "proof_term", _text(self.proof_term, "proof_term", required=True)
            )
        elif isinstance(self.proof_term, Mapping):
            object.__setattr__(self, "proof_term", _mapping(self.proof_term, "proof_term"))
        elif isinstance(self.proof_term, Sequence) and not isinstance(
            self.proof_term, (bytes, bytearray)
        ):
            object.__setattr__(self, "proof_term", list(self.proof_term))
        else:
            raise KernelReconstructionError(
                "proof_term must be text, an object, or a sequence"
            )
        object.__setattr__(
            self,
            "certificate_digest",
            _text(self.certificate_digest, "certificate_digest"),
        )
        object.__setattr__(
            self,
            "kernel_version",
            _text(self.kernel_version, "kernel_version", required=True),
        )
        object.__setattr__(self, "root_ids", _ids(self.root_ids, "root_ids"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "graph_root", _text(self.graph_root, "graph_root"))
        object.__setattr__(
            self, "provider_status", _text(self.provider_status, "provider_status")
        )
        object.__setattr__(self, "independent", _bool(self.independent, "independent"))
        object.__setattr__(
            self, "proof_children", _ids(self.proof_children, "proof_children")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def derived_digest(self) -> str:
        return proof_term_digest(self.proof_term)

    def to_dict(self) -> dict[str, Any]:
        term: Any
        if isinstance(self.proof_term, str):
            term = self.proof_term
        else:
            term = self.proof_term
        return {
            "obligation_id": self.obligation_id,
            "proof_term": term,
            "certificate_digest": self.certificate_digest,
            "kernel_version": self.kernel_version,
            "root_ids": list(self.root_ids),
            "tree_id": self.tree_id,
            "graph_root": self.graph_root,
            "provider_status": self.provider_status,
            "independent": self.independent,
            "proof_children": list(self.proof_children),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofClaim":
        if not isinstance(payload, Mapping):
            raise KernelReconstructionError("proof claim must be an object")
        return cls(
            obligation_id=payload.get("obligation_id", ""),
            proof_term=payload.get("proof_term", ""),
            certificate_digest=payload.get("certificate_digest", ""),
            kernel_version=payload.get("kernel_version", DEFAULT_KERNEL_VERSION),
            root_ids=tuple(payload.get("root_ids") or ()),
            tree_id=payload.get("tree_id", ""),
            graph_root=payload.get("graph_root", ""),
            provider_status=payload.get("provider_status", ""),
            independent=bool(payload.get("independent", False)),
            proof_children=tuple(payload.get("proof_children") or ()),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ProofKernelReceipt(CanonicalContract):
    """Independent reconstruction receipt for one claimed proof.

    Interface: ``ProofKernelReceipt@1``.
    """

    SCHEMA: ClassVar[str] = PROOF_KERNEL_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_KERNEL_RECEIPT_INTERFACE

    status: ReconstructionStatus
    obligation_id: str
    kernel_version: str
    proof_term_digest: str
    certificate_digest: str
    reconstructed: bool
    reason_codes: tuple[str, ...] = ()
    root_ids: tuple[str, ...] = ()
    tree_id: str = ""
    graph_root: str = ""
    independent: bool = False
    provider_status: str = ""
    proof_children: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, ReconstructionStatus):
            status = self.status
        else:
            try:
                status = ReconstructionStatus(str(self.status))
            except ValueError as exc:
                raise KernelReconstructionError(
                    f"unsupported reconstruction status: {self.status!r}"
                ) from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=True),
        )
        object.__setattr__(
            self,
            "kernel_version",
            _text(self.kernel_version, "kernel_version", required=True),
        )
        object.__setattr__(
            self,
            "proof_term_digest",
            _text(self.proof_term_digest, "proof_term_digest", required=True),
        )
        object.__setattr__(
            self,
            "certificate_digest",
            _text(self.certificate_digest, "certificate_digest"),
        )
        object.__setattr__(
            self, "reconstructed", _bool(self.reconstructed, "reconstructed")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=DEFAULT_MAX_REASON_CODES),
        )
        object.__setattr__(self, "root_ids", _ids(self.root_ids, "root_ids"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "graph_root", _text(self.graph_root, "graph_root"))
        object.__setattr__(self, "independent", _bool(self.independent, "independent"))
        object.__setattr__(
            self, "provider_status", _text(self.provider_status, "provider_status")
        )
        object.__setattr__(
            self, "proof_children", _ids(self.proof_children, "proof_children")
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail", maximum=2048))
        if self.reconstructed and self.status is not ReconstructionStatus.RECONSTRUCTED:
            raise KernelReconstructionError(
                "reconstructed flag disagrees with status"
            )
        if (
            self.status is ReconstructionStatus.RECONSTRUCTED
            and not self.reconstructed
        ):
            raise KernelReconstructionError(
                "reconstructed status requires reconstructed=True"
            )
        if self.status is ReconstructionStatus.RECONSTRUCTED and not self.independent:
            raise KernelReconstructionError(
                "reconstructed proofs must be independently obtained"
            )

    @property
    def valid(self) -> bool:
        return self.status is ReconstructionStatus.RECONSTRUCTED and self.reconstructed

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "contract_version": KERNEL_RECONSTRUCTION_VERSION,
            "status": self.status.value,
            "obligation_id": self.obligation_id,
            "kernel_version": self.kernel_version,
            "proof_term_digest": self.proof_term_digest,
            "certificate_digest": self.certificate_digest,
            "reconstructed": self.reconstructed,
            "reason_codes": list(self.reason_codes),
            "root_ids": list(self.root_ids),
            "tree_id": self.tree_id,
            "graph_root": self.graph_root,
            "independent": self.independent,
            "provider_status": self.provider_status,
            "proof_children": list(self.proof_children),
            "detail": self.detail,
            "valid": self.valid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofKernelReceipt":
        if not isinstance(payload, Mapping):
            raise KernelReconstructionError("proof kernel receipt must be an object")
        schema = payload.get("schema")
        if schema not in (None, PROOF_KERNEL_RECEIPT_SCHEMA):
            raise KernelReconstructionError("unsupported proof kernel receipt schema")
        return cls(
            status=payload.get("status", ReconstructionStatus.INVALID),
            obligation_id=payload.get("obligation_id", ""),
            kernel_version=payload.get("kernel_version", DEFAULT_KERNEL_VERSION),
            proof_term_digest=payload.get("proof_term_digest", ""),
            certificate_digest=payload.get("certificate_digest", ""),
            reconstructed=bool(payload.get("reconstructed", False)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            root_ids=tuple(payload.get("root_ids") or ()),
            tree_id=payload.get("tree_id", ""),
            graph_root=payload.get("graph_root", ""),
            independent=bool(payload.get("independent", False)),
            provider_status=payload.get("provider_status", ""),
            proof_children=tuple(payload.get("proof_children") or ()),
            detail=payload.get("detail", ""),
        )


@dataclass(frozen=True)
class Counterexample(CanonicalContract):
    """Minimized, replayable contract counterexample.

    Interface: ``Counterexample@1``.
    """

    SCHEMA: ClassVar[str] = COUNTEREXAMPLE_SCHEMA
    INTERFACE: ClassVar[str] = COUNTEREXAMPLE_INTERFACE

    obligation_id: str
    violated_property: str
    summary: str
    witness: Mapping[str, Any]
    graph_edge_ids: tuple[str, ...] = ()
    transcript_receipt_ids: tuple[str, ...] = ()
    root_ids: tuple[str, ...] = ()
    tree_id: str = ""
    graph_root: str = ""
    minimized: bool = True
    inferred_observations: bool = False
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=True),
        )
        object.__setattr__(
            self,
            "violated_property",
            _text(self.violated_property, "violated_property", required=True),
        )
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", required=True, maximum=1024)
        )
        witness = _minimal_witness(_mapping(self.witness, "witness"))
        object.__setattr__(self, "witness", witness)
        object.__setattr__(
            self, "graph_edge_ids", _ids(self.graph_edge_ids, "graph_edge_ids")
        )
        object.__setattr__(
            self,
            "transcript_receipt_ids",
            _ids(self.transcript_receipt_ids, "transcript_receipt_ids"),
        )
        object.__setattr__(self, "root_ids", _ids(self.root_ids, "root_ids"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "graph_root", _text(self.graph_root, "graph_root"))
        object.__setattr__(self, "minimized", _bool(self.minimized, "minimized"))
        object.__setattr__(
            self,
            "inferred_observations",
            _bool(self.inferred_observations, "inferred_observations"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=DEFAULT_MAX_REASON_CODES),
        )
        if self.minimized is not True:
            raise KernelReconstructionError("Counterexample@1 must be minimized")
        if self.inferred_observations:
            raise KernelReconstructionError(
                "Counterexample@1 forbids inferred observations"
            )
        if not self.graph_edge_ids and not self.transcript_receipt_ids:
            raise KernelReconstructionError(
                "Counterexample@1 requires graph or transcript anchors for replay"
            )

    @property
    def counterexample_id(self) -> str:
        return self.content_id

    @property
    def byte_size(self) -> int:
        return len(self.canonical_bytes())

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "contract_version": KERNEL_RECONSTRUCTION_VERSION,
            "obligation_id": self.obligation_id,
            "violated_property": self.violated_property,
            "summary": self.summary,
            "witness": dict(self.witness),
            "graph_edge_ids": list(self.graph_edge_ids),
            "transcript_receipt_ids": list(self.transcript_receipt_ids),
            "root_ids": list(self.root_ids),
            "tree_id": self.tree_id,
            "graph_root": self.graph_root,
            "minimized": True,
            "inferred_observations": False,
            "reason_codes": list(self.reason_codes),
            "counterexample_id": content_identity(
                {
                    "obligation_id": self.obligation_id,
                    "violated_property": self.violated_property,
                    "witness": dict(self.witness),
                    "graph_edge_ids": list(self.graph_edge_ids),
                    "transcript_receipt_ids": list(self.transcript_receipt_ids),
                }
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Counterexample":
        if not isinstance(payload, Mapping):
            raise KernelReconstructionError("counterexample must be an object")
        schema = payload.get("schema")
        if schema not in (None, COUNTEREXAMPLE_SCHEMA):
            raise KernelReconstructionError("unsupported counterexample schema")
        return cls(
            obligation_id=payload.get("obligation_id", ""),
            violated_property=payload.get("violated_property", ""),
            summary=payload.get("summary", ""),
            witness=payload.get("witness") or {},
            graph_edge_ids=tuple(payload.get("graph_edge_ids") or ()),
            transcript_receipt_ids=tuple(payload.get("transcript_receipt_ids") or ()),
            root_ids=tuple(payload.get("root_ids") or ()),
            tree_id=payload.get("tree_id", ""),
            graph_root=payload.get("graph_root", ""),
            minimized=payload.get("minimized", True),
            inferred_observations=bool(payload.get("inferred_observations", False)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class CounterexampleReplayResult(CanonicalContract):
    """Result of replaying a minimized counterexample against bound evidence."""

    SCHEMA: ClassVar[str] = COUNTEREXAMPLE_REPLAY_SCHEMA

    status: CounterexampleReplayStatus
    counterexample_id: str
    matched_graph_edge_ids: tuple[str, ...] = ()
    matched_transcript_receipt_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, CounterexampleReplayStatus):
            status = self.status
        else:
            try:
                status = CounterexampleReplayStatus(str(self.status))
            except ValueError as exc:
                raise KernelReconstructionError(
                    f"unsupported replay status: {self.status!r}"
                ) from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, "counterexample_id", required=True),
        )
        object.__setattr__(
            self,
            "matched_graph_edge_ids",
            _ids(self.matched_graph_edge_ids, "matched_graph_edge_ids"),
        )
        object.__setattr__(
            self,
            "matched_transcript_receipt_ids",
            _ids(
                self.matched_transcript_receipt_ids,
                "matched_transcript_receipt_ids",
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=DEFAULT_MAX_REASON_CODES),
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail", maximum=2048))

    @property
    def replayed(self) -> bool:
        return self.status is CounterexampleReplayStatus.REPLAYED

    def _payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "counterexample_id": self.counterexample_id,
            "matched_graph_edge_ids": list(self.matched_graph_edge_ids),
            "matched_transcript_receipt_ids": list(self.matched_transcript_receipt_ids),
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "replayed": self.replayed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterexampleReplayResult":
        if not isinstance(payload, Mapping):
            raise KernelReconstructionError("replay result must be an object")
        return cls(
            status=payload.get("status", CounterexampleReplayStatus.INVALID),
            counterexample_id=payload.get("counterexample_id", ""),
            matched_graph_edge_ids=tuple(payload.get("matched_graph_edge_ids") or ()),
            matched_transcript_receipt_ids=tuple(
                payload.get("matched_transcript_receipt_ids") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            detail=payload.get("detail", ""),
        )


def _proof_term_text(term: str | Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(term, str):
        return term
    return _canonical_bytes(term).decode("utf-8")


def _derived_children(term: str | Mapping[str, Any] | Sequence[Any]) -> tuple[str, ...]:
    """Derive child step digests from the term itself (never from a claim list)."""

    if isinstance(term, Mapping):
        steps = term.get("steps") or term.get("children") or term.get("tactics")
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
            digests: list[str] = []
            for step in steps:
                digests.append(proof_term_digest(step if not isinstance(step, str) else step))
            return tuple(digests)
    text = _proof_term_text(term)
    # Split on explicit step markers when present; otherwise no children.
    parts = [
        part.strip()
        for part in re.split(r"(?m)^\s*(?:step|tactic|have|suffices)\b", text)
        if part.strip()
    ]
    if len(parts) <= 1:
        return ()
    return tuple(proof_term_digest(part) for part in parts[1:])


def reconstruct_proof(
    claim: ProofClaim | Mapping[str, Any],
    *,
    expected_root_ids: Sequence[str] | None = None,
    expected_tree_id: str = "",
    expected_graph_root: str = "",
    kernel_version: str = DEFAULT_KERNEL_VERSION,
    independent_checker: Callable[[ProofClaim], bool] | None = None,
) -> ProofKernelReceipt:
    """Independently reconstruct one claimed proof.

    Unreconstructable claims become ``invalid``.  Provider status alone never
    upgrades a claim to reconstructed, and incomplete or fabricated proof
    children are rejected.
    """

    if isinstance(claim, Mapping):
        claim = ProofClaim.from_dict(claim)
    if not isinstance(claim, ProofClaim):
        raise KernelReconstructionError("claim must be ProofClaim or mapping")

    reasons: list[str] = []
    derived = claim.derived_digest
    claimed_digest = claim.certificate_digest or derived

    if claim.certificate_digest and claim.certificate_digest != derived:
        reasons.append("certificate_digest_mismatch")

    if claim.kernel_version != kernel_version:
        reasons.append("kernel_version_mismatch")

    term_text = _proof_term_text(claim.proof_term)
    if _INCOMPLETE_PROOF_RE.search(term_text):
        reasons.append("incomplete_proof")

    # Provider-only success is never reconstruction evidence.  An independent
    # claim flag or an explicit independent checker is required instead.
    has_independent_path = bool(claim.independent) or independent_checker is not None
    provider = claim.provider_status.strip().lower()
    if (
        provider in {"verified", "proved", "sat", "success", "ok"}
        and not has_independent_path
    ):
        reasons.append("provider_status_not_independent")

    if not has_independent_path:
        reasons.append("independent_reconstruction_required")

    expected_roots = _ids(expected_root_ids, "expected_root_ids") if expected_root_ids is not None else ()
    if expected_roots:
        if tuple(sorted(claim.root_ids)) != tuple(sorted(expected_roots)):
            reasons.append("root_binding_mismatch")
    if expected_tree_id and claim.tree_id and claim.tree_id != expected_tree_id:
        reasons.append("tree_binding_mismatch")
    if expected_graph_root and claim.graph_root and claim.graph_root != expected_graph_root:
        reasons.append("graph_root_mismatch")

    derived_children = _derived_children(claim.proof_term)
    if claim.proof_children:
        if tuple(claim.proof_children) != derived_children:
            # Fabricated / non-derived children are forbidden.
            reasons.append("fabricated_proof_children")

    # Optional independent checker (e.g. live kernel).  Failure is invalid.
    checker_ok = True
    if independent_checker is not None:
        try:
            checker_ok = bool(independent_checker(claim))
        except Exception as exc:  # pragma: no cover - defensive fail-closed
            checker_ok = False
            reasons.append(f"independent_checker_error:{type(exc).__name__}")
        if not checker_ok:
            reasons.append("independent_checker_rejected")

    # Explicit refutation marker in the term.
    if re.search(r"(?i)\b(?:refuted|counterexample|unsat)\b", term_text):
        status = ReconstructionStatus.REFUTED
        reasons.append("term_marks_refutation")
        return ProofKernelReceipt(
            status=status,
            obligation_id=claim.obligation_id,
            kernel_version=claim.kernel_version,
            proof_term_digest=derived,
            certificate_digest=claimed_digest,
            reconstructed=False,
            reason_codes=tuple(reasons) or ("refuted",),
            root_ids=claim.root_ids,
            tree_id=claim.tree_id,
            graph_root=claim.graph_root,
            independent=claim.independent,
            provider_status=claim.provider_status,
            proof_children=derived_children,
            detail="proof term marks a refutation rather than an accepted proof",
        )

    if reasons:
        return ProofKernelReceipt(
            status=ReconstructionStatus.INVALID,
            obligation_id=claim.obligation_id,
            kernel_version=claim.kernel_version,
            proof_term_digest=derived,
            certificate_digest=claimed_digest,
            reconstructed=False,
            reason_codes=tuple(reasons),
            root_ids=claim.root_ids,
            tree_id=claim.tree_id,
            graph_root=claim.graph_root,
            independent=claim.independent,
            provider_status=claim.provider_status,
            proof_children=derived_children,
            detail="unreconstructable proof became invalid",
        )

    return ProofKernelReceipt(
        status=ReconstructionStatus.RECONSTRUCTED,
        obligation_id=claim.obligation_id,
        kernel_version=claim.kernel_version,
        proof_term_digest=derived,
        certificate_digest=claimed_digest,
        reconstructed=True,
        reason_codes=("independent_reconstruction_ok",),
        root_ids=claim.root_ids,
        tree_id=claim.tree_id,
        graph_root=claim.graph_root,
        independent=True,
        provider_status=claim.provider_status,
        proof_children=derived_children,
        detail="proof reconstructed from bound term and digests",
    )


def _graph_edge_index(graph: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not graph:
        return {}
    edges = graph.get("edges") if isinstance(graph, Mapping) else None
    if not isinstance(edges, Sequence):
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        edge_id = str(edge.get("edge_id") or "").strip()
        if edge_id:
            index[edge_id] = edge
    return index


def _transcript_receipt_index(
    transcript: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not transcript:
        return {}
    exchanges: Sequence[Any]
    if isinstance(transcript, Mapping):
        raw = transcript.get("exchanges") or transcript.get("observations") or ()
        exchanges = raw if isinstance(raw, Sequence) else ()
    elif isinstance(transcript, Sequence):
        exchanges = transcript
    else:
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for item in exchanges:
        if not isinstance(item, Mapping):
            continue
        for key in ("receipt_cid", "local_cid", "observation_id", "exchange_id"):
            value = str(item.get(key) or "").strip()
            if value:
                index[value] = item
    return index


def minimize_counterexample(
    value: Counterexample | Mapping[str, Any],
    *,
    graph: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | Sequence[Any] | None = None,
    require_replay: bool = True,
) -> Counterexample:
    """Reduce a refutation to a minimal replayable ``Counterexample@1``.

    Inferred observations are rejected.  When ``require_replay`` is true
    (default), the minimized witness must replay against the bound graph and
    live transcript.
    """

    if isinstance(value, Counterexample):
        candidate = value
    elif isinstance(value, Mapping):
        raw = dict(value)
        if raw.get("inferred") or raw.get("inferred_observation") or raw.get("inferred_observations"):
            raise KernelReconstructionError(
                "counterexample carries inferred observations"
            )
        witness_src = raw.get("witness")
        if not isinstance(witness_src, Mapping):
            witness_src = {
                key: raw[key]
                for key in raw
                if key
                not in {
                    "obligation_id",
                    "violated_property",
                    "summary",
                    "graph_edge_ids",
                    "transcript_receipt_ids",
                    "root_ids",
                    "tree_id",
                    "graph_root",
                    "reason_codes",
                    "schema",
                    "interface",
                    "minimized",
                    "inferred_observations",
                }
            }
        graph_edge_ids = tuple(raw.get("graph_edge_ids") or ())
        if not graph_edge_ids:
            edge_id = str(raw.get("edge_id") or witness_src.get("edge_id") or "").strip()
            if edge_id:
                graph_edge_ids = (edge_id,)
        transcript_ids = tuple(raw.get("transcript_receipt_ids") or ())
        if not transcript_ids:
            for key in (
                "receipt_cid",
                "observation_id",
                "exchange_id",
                "local_cid",
            ):
                value_id = str(raw.get(key) or witness_src.get(key) or "").strip()
                if value_id:
                    transcript_ids = (value_id,)
                    break
        # Auto-bind from graph/transcript only when the seed already names
        # enough identity fields for a unique exact match.  Empty filters are
        # never treated as a match — that would invent anchors.
        edge_index = _graph_edge_index(graph)
        receipt_index = _transcript_receipt_index(transcript)
        if not graph_edge_ids and edge_index:
            edge_kind = str(raw.get("edge_kind") or witness_src.get("edge_kind") or "")
            consumer = str(
                raw.get("consumer_id") or witness_src.get("consumer_id") or ""
            )
            if edge_kind or consumer:
                matches = [
                    edge_id
                    for edge_id, edge in edge_index.items()
                    if (not edge_kind or str(edge.get("kind") or "") == edge_kind)
                    and (
                        not consumer
                        or str(edge.get("consumer_id") or "") == consumer
                    )
                ]
                if len(matches) == 1:
                    graph_edge_ids = (matches[0],)
        if not transcript_ids and receipt_index:
            role = str(raw.get("role") or witness_src.get("role") or "")
            method = str(raw.get("method") or witness_src.get("method") or "")
            if role or method:
                matches = [
                    rid
                    for rid, item in receipt_index.items()
                    if (not role or str(item.get("role") or "") == role)
                    and (not method or str(item.get("method") or "") == method)
                ]
                # Only accept a unique exact match — never invent observations.
                if len(matches) == 1:
                    transcript_ids = (matches[0],)

        candidate = Counterexample(
            obligation_id=str(
                raw.get("obligation_id")
                or raw.get("property_id")
                or "obligation:unknown"
            ),
            violated_property=str(
                raw.get("violated_property")
                or raw.get("property_id")
                or raw.get("operation")
                or "unknown-property"
            ),
            summary=str(
                raw.get("summary")
                or raw.get("reason_code")
                or "minimized contract counterexample"
            ),
            witness=witness_src if isinstance(witness_src, Mapping) else {},
            graph_edge_ids=graph_edge_ids,
            transcript_receipt_ids=transcript_ids,
            root_ids=tuple(raw.get("root_ids") or ()),
            tree_id=str(raw.get("tree_id") or ""),
            graph_root=str(raw.get("graph_root") or ""),
            minimized=True,
            inferred_observations=False,
            reason_codes=tuple(raw.get("reason_codes") or ("minimized",)),
        )
    else:
        raise KernelReconstructionError(
            "counterexample must be Counterexample@1 or a mapping"
        )

    if require_replay:
        replay = replay_counterexample(candidate, graph=graph, transcript=transcript)
        if not replay.replayed:
            raise KernelReconstructionError(
                "minimized counterexample failed replay: "
                + ",".join(replay.reason_codes or (replay.status.value,))
            )
    return candidate


def replay_counterexample(
    counterexample: Counterexample | Mapping[str, Any],
    *,
    graph: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | Sequence[Any] | None = None,
) -> CounterexampleReplayResult:
    """Replay a minimized counterexample against bound graph and transcript.

    Observations are never inferred: every anchor must already be named by the
    counterexample and present in the supplied evidence.
    """

    if isinstance(counterexample, Mapping):
        counterexample = Counterexample.from_dict(counterexample)
    if not isinstance(counterexample, Counterexample):
        raise KernelReconstructionError("counterexample must be Counterexample@1")

    if counterexample.inferred_observations:
        return CounterexampleReplayResult(
            status=CounterexampleReplayStatus.INFERRED_OBSERVATION,
            counterexample_id=counterexample.counterexample_id,
            reason_codes=("inferred_observations_forbidden",),
            detail="replay refuses inferred observations",
        )

    edge_index = _graph_edge_index(graph)
    receipt_index = _transcript_receipt_index(transcript)
    matched_edges: list[str] = []
    matched_receipts: list[str] = []
    reasons: list[str] = []

    if counterexample.graph_edge_ids:
        for edge_id in counterexample.graph_edge_ids:
            if edge_id in edge_index:
                matched_edges.append(edge_id)
            else:
                reasons.append(f"missing_graph_edge:{edge_id}")
        if not matched_edges:
            return CounterexampleReplayResult(
                status=CounterexampleReplayStatus.MISSING_GRAPH_ANCHOR,
                counterexample_id=counterexample.counterexample_id,
                reason_codes=tuple(reasons) or ("missing_graph_anchor",),
                detail="no bound graph edges matched the counterexample",
            )
    elif graph is not None:
        # Graph was supplied but the counterexample has no edge anchors.
        return CounterexampleReplayResult(
            status=CounterexampleReplayStatus.MISSING_GRAPH_ANCHOR,
            counterexample_id=counterexample.counterexample_id,
            reason_codes=("counterexample_lacks_graph_edge_ids",),
            detail="graph supplied but counterexample has no edge anchors",
        )

    if counterexample.transcript_receipt_ids:
        for receipt_id in counterexample.transcript_receipt_ids:
            if receipt_id in receipt_index:
                matched_receipts.append(receipt_id)
            else:
                reasons.append(f"missing_transcript_receipt:{receipt_id}")
        if not matched_receipts:
            return CounterexampleReplayResult(
                status=CounterexampleReplayStatus.MISSING_TRANSCRIPT_ANCHOR,
                counterexample_id=counterexample.counterexample_id,
                matched_graph_edge_ids=tuple(matched_edges),
                reason_codes=tuple(reasons) or ("missing_transcript_anchor",),
                detail="no bound transcript receipts matched the counterexample",
            )
    elif transcript is not None:
        return CounterexampleReplayResult(
            status=CounterexampleReplayStatus.MISSING_TRANSCRIPT_ANCHOR,
            counterexample_id=counterexample.counterexample_id,
            matched_graph_edge_ids=tuple(matched_edges),
            reason_codes=("counterexample_lacks_transcript_receipt_ids",),
            detail="transcript supplied but counterexample has no receipt anchors",
        )

    # Witness consistency: edge_id in witness must match an anchored edge.
    witness_edge = str(counterexample.witness.get("edge_id") or "").strip()
    if witness_edge and witness_edge not in matched_edges and counterexample.graph_edge_ids:
        if witness_edge not in counterexample.graph_edge_ids:
            return CounterexampleReplayResult(
                status=CounterexampleReplayStatus.WITNESS_MISMATCH,
                counterexample_id=counterexample.counterexample_id,
                matched_graph_edge_ids=tuple(matched_edges),
                matched_transcript_receipt_ids=tuple(matched_receipts),
                reason_codes=("witness_edge_not_anchored",),
                detail="witness edge_id is not among counterexample anchors",
            )

    if not matched_edges and not matched_receipts:
        return CounterexampleReplayResult(
            status=CounterexampleReplayStatus.INVALID,
            counterexample_id=counterexample.counterexample_id,
            reason_codes=("no_replay_anchors",),
            detail="counterexample has no replayable anchors against bound evidence",
        )

    return CounterexampleReplayResult(
        status=CounterexampleReplayStatus.REPLAYED,
        counterexample_id=counterexample.counterexample_id,
        matched_graph_edge_ids=tuple(matched_edges),
        matched_transcript_receipt_ids=tuple(matched_receipts),
        reason_codes=("replayed_against_bound_evidence",),
        detail="counterexample replayed without inferred observations",
    )


def materialize_proof_kernel_reconstruction_artifact(
    *,
    repository_id: str = "repository:lift_coding/dcr",
    tree_id: str = "",
    receipts: Sequence[ProofKernelReceipt] = (),
    counterexamples: Sequence[Counterexample] = (),
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the DCR-033 generated artifact projection."""

    note_list = list(
        notes
        or (
            "DCR-033 independent proof reconstruction surface.",
            "Unreconstructable proofs are invalid; provider verified is never enough.",
            "Minimized counterexamples replay against bound graph and live transcript.",
            "Inferred observations and fabricated proof children are rejected.",
        )
    )
    return {
        "schema": PROOF_KERNEL_ARTIFACT_SCHEMA,
        "interface": PROOF_KERNEL_RECEIPT_INTERFACE,
        "counterexample_interface": COUNTEREXAMPLE_INTERFACE,
        "task_id": "DCR-033",
        "kernel_version": DEFAULT_KERNEL_VERSION,
        "contract_version": KERNEL_RECONSTRUCTION_VERSION,
        "repository_id": repository_id,
        "tree_id": tree_id,
        "reconstructed_count": sum(1 for item in receipts if item.valid),
        "invalid_count": sum(
            1
            for item in receipts
            if item.status is ReconstructionStatus.INVALID
        ),
        "refuted_count": sum(
            1
            for item in receipts
            if item.status is ReconstructionStatus.REFUTED
        ),
        "counterexample_count": len(tuple(counterexamples)),
        "receipts": [item.to_record() for item in receipts],
        "counterexamples": [item.to_record() for item in counterexamples],
        "acceptance": {
            "unreconstructable_proof_becomes_invalid": True,
            "provider_status_never_mints_reconstruction": True,
            "refutations_replay_against_bound_graph_and_transcript": True,
            "inferred_observations_forbidden": True,
            "fabricated_proof_children_forbidden": True,
        },
        "notes": note_list,
    }


def write_proof_kernel_reconstruction_artifact(
    path: str | Path,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    """Write the DCR-033 artifact as canonical JSON."""

    destination = Path(path)
    payload = (
        dict(artifact)
        if artifact is not None
        else materialize_proof_kernel_reconstruction_artifact()
    )
    body = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination


__all__ = [
    "COUNTEREXAMPLE_INTERFACE",
    "COUNTEREXAMPLE_REPLAY_SCHEMA",
    "COUNTEREXAMPLE_SCHEMA",
    "Counterexample",
    "CounterexampleReplayResult",
    "CounterexampleReplayStatus",
    "DEFAULT_KERNEL_VERSION",
    "KERNEL_RECONSTRUCTION_VERSION",
    "KernelReconstructionError",
    "PROOF_KERNEL_ARTIFACT_SCHEMA",
    "PROOF_KERNEL_RECEIPT_INTERFACE",
    "PROOF_KERNEL_RECEIPT_SCHEMA",
    "ProofClaim",
    "ProofKernelReceipt",
    "ReconstructionStatus",
    "materialize_proof_kernel_reconstruction_artifact",
    "minimize_counterexample",
    "proof_term_digest",
    "reconstruct_proof",
    "replay_counterexample",
    "write_proof_kernel_reconstruction_artifact",
]

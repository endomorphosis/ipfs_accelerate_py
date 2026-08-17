"""Pure contracts for the prompt-v3 protected acceptance transition.

This module is intentionally free of filesystem, Git, profile, and validator
imports.  It defines the closed vocabulary exchanged by the merge-layer
builder and the highest-layer entrypoint composition facade.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

Q_INVENTORY_SCHEMA = "ipfs_accelerate_py.agent_supervisor.prompt-v3-q-inventory@1"
PRODUCT_PROVENANCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-product-generation@1"
)
PHASE_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-phase-authority@1"
)
PHASE_RECEIPT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.prompt-v3-phase-receipt@2"
EVIDENCE_HANDLE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-bounded-evidence@1"
)
RUNTIME_LAUNCH_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-runtime-launch-authority@1"
)
P031_ATTEMPT_LEDGER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-p031-attempt-ledger@1"
)
PRE_Q_PRODUCT_TASKS = frozenset(
    {"ASE3-019", "ASE3-030", "ASE3-031", "ASE3-032", "ASE3-023", "ASE3-027"}
)
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_RECORDS = 10_000

_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_TASK_RE = re.compile(r"ASE3-[0-9]{3}\Z")


class ProtectedAcceptanceError(ValueError):
    """Base class for fail-closed contract errors."""


class ProtectedAcceptanceDenied(ProtectedAcceptanceError):
    """The requested transition is outside the protected policy."""


class PromptV3Phase(str, Enum):
    Q = "Q"
    R = "R"
    P019 = "P019"
    A019 = "A019"
    A030 = "A030"
    P031 = "P031"
    A031 = "A031"
    A032 = "A032"
    A023_027 = "A023/027"
    L = "L"
    BIRTH = "birth"


PROMPT_V3_PHASE_ORDER: tuple[PromptV3Phase, ...] = (
    PromptV3Phase.Q,
    PromptV3Phase.R,
    PromptV3Phase.P019,
    PromptV3Phase.A019,
    PromptV3Phase.A030,
    PromptV3Phase.P031,
    PromptV3Phase.A031,
    PromptV3Phase.A032,
    PromptV3Phase.A023_027,
    PromptV3Phase.L,
    PromptV3Phase.BIRTH,
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the sole canonical JSON encoding used by these contracts."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON input must be a mapping")
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtectedAcceptanceError(
            "artifact is not canonical-JSON encodable"
        ) from exc


def content_id(data: bytes) -> str:
    if type(data) is not bytes:
        raise TypeError("content identity input must be bytes")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def phase_authority_content_id(
    *,
    phase: PromptV3Phase,
    nonce: str,
    parent_commit: str,
    identity_did: str,
    issued_at_ns: int,
    expires_at_ns: int,
) -> str:
    """Bind a phase-authority ID to its entire immutable unsigned body."""

    if not isinstance(phase, PromptV3Phase):
        raise TypeError("phase authority content requires PromptV3Phase")
    return content_id(
        canonical_json_bytes(
            {
                "schema": PHASE_AUTHORITY_SCHEMA,
                "phase": phase.value,
                "nonce": nonce,
                "parent_commit": parent_commit,
                "identity_did": identity_did,
                "issued_at_ns": issued_at_ns,
                "expires_at_ns": expires_at_ns,
            }
        )
    )


def _strict_keys(
    value: Mapping[str, Any], expected: Sequence[str], context: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ProtectedAcceptanceDenied(f"{context} has unsupported or missing fields")


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not value
        or len(value.encode("utf-8")) > maximum
        or any(character in value for character in ("\0", "\r", "\n"))
    ):
        raise ProtectedAcceptanceError(f"{name} is invalid")
    return value


def _object_id(value: Any, name: str) -> str:
    text = _text(value, name, maximum=64)
    if not _OBJECT_RE.fullmatch(text):
        raise ProtectedAcceptanceError(f"{name} must be a full lowercase Git object ID")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name, maximum=71)
    if not _SHA256_RE.fullmatch(text):
        raise ProtectedAcceptanceError(f"{name} must be a sha256 content ID")
    return text


def _positive_int(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ProtectedAcceptanceError(f"{name} is outside its closed bound")
    return value


def protected_git_path(value: Any) -> str:
    text = _text(value, "protected Git path", maximum=1024)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
        or text.startswith("-")
        or "\\" in text
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ProtectedAcceptanceDenied("protected Git path is non-canonical")
    return text


def immediate_parent_phase(phase: PromptV3Phase) -> PromptV3Phase | None:
    if not isinstance(phase, PromptV3Phase):
        raise TypeError("phase must be PromptV3Phase")
    position = PROMPT_V3_PHASE_ORDER.index(phase)
    return None if position == 0 else PROMPT_V3_PHASE_ORDER[position - 1]


@dataclass(frozen=True)
class EvidenceHandle:
    """A bounded public evidence reference; never an artifact filesystem path."""

    kind: str
    content_id: str
    byte_length: int
    record_count: int = 1
    schema: str = EVIDENCE_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_HANDLE_SCHEMA:
            raise ProtectedAcceptanceError("unsupported evidence-handle schema")
        if type(self.kind) is not str or not _TOKEN_RE.fullmatch(self.kind):
            raise ProtectedAcceptanceError("evidence kind is invalid")
        _digest(self.content_id, "evidence content_id")
        _positive_int(self.byte_length, "evidence byte_length", MAX_EVIDENCE_BYTES)
        _positive_int(self.record_count, "evidence record_count", MAX_EVIDENCE_RECORDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "content_id": self.content_id,
            "byte_length": self.byte_length,
            "record_count": self.record_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvidenceHandle:
        _strict_keys(
            value,
            ("schema", "kind", "content_id", "byte_length", "record_count"),
            "evidence handle",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class GitFileIdentity:
    path: str
    mode: str
    blob_id: str
    raw_content_id: str
    byte_length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", protected_git_path(self.path))
        if self.mode not in {"100644", "100755"}:
            raise ProtectedAcceptanceDenied("reviewed product Git mode is unsupported")
        _object_id(self.blob_id, "blob_id")
        _digest(self.raw_content_id, "raw_content_id")
        _positive_int(self.byte_length, "file byte_length", MAX_ARTIFACT_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "blob_id": self.blob_id,
            "raw_content_id": self.raw_content_id,
            "byte_length": self.byte_length,
        }


@dataclass(frozen=True)
class ProductGenerationRecord:
    role: str
    commit: str
    parent: str
    tree: str
    files: tuple[GitFileIdentity, ...]
    test_evidence: tuple[EvidenceHandle, ...]
    canonical_patch_content_id: str

    def __post_init__(self) -> None:
        if self.role not in {"source", "replay", "integrated"}:
            raise ProtectedAcceptanceError("product role is invalid")
        _object_id(self.commit, f"{self.role} commit")
        _object_id(self.parent, f"{self.role} parent")
        _object_id(self.tree, f"{self.role} tree")
        if type(self.files) is not tuple or not self.files:
            raise TypeError("product files must be a non-empty tuple")
        if any(not isinstance(item, GitFileIdentity) for item in self.files):
            raise TypeError("product files must contain GitFileIdentity values")
        if len({item.path for item in self.files}) != len(self.files):
            raise ProtectedAcceptanceError("product files contain duplicate paths")
        if type(self.test_evidence) is not tuple or not self.test_evidence:
            raise TypeError("product test evidence must be a non-empty tuple")
        if any(not isinstance(item, EvidenceHandle) for item in self.test_evidence):
            raise TypeError("product test evidence must contain EvidenceHandle values")
        _digest(
            self.canonical_patch_content_id, f"{self.role} canonical patch content_id"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "commit": self.commit,
            "parent": self.parent,
            "tree": self.tree,
            "files": [item.to_dict() for item in self.files],
            "test_evidence": [item.to_dict() for item in self.test_evidence],
            "canonical_patch_content_id": self.canonical_patch_content_id,
        }


@dataclass(frozen=True)
class ProductProvenance:
    task_id: str
    source: ProductGenerationRecord
    replay: ProductGenerationRecord
    integrated: ProductGenerationRecord
    canonical_diff_content_id: str
    schema: str = PRODUCT_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCT_PROVENANCE_SCHEMA:
            raise ProtectedAcceptanceError("unsupported product provenance schema")
        if self.task_id not in PRE_Q_PRODUCT_TASKS:
            raise ProtectedAcceptanceDenied(
                "Q provenance names a non-product or self task"
            )
        if (self.source.role, self.replay.role, self.integrated.role) != (
            "source",
            "replay",
            "integrated",
        ):
            raise ProtectedAcceptanceError(
                "product records are assigned to the wrong roles"
            )
        _digest(self.canonical_diff_content_id, "canonical diff content_id")

        def inventory(record: ProductGenerationRecord) -> tuple[tuple[Any, ...], ...]:
            return tuple(
                (
                    item.path,
                    item.mode,
                    item.blob_id,
                    item.raw_content_id,
                    item.byte_length,
                )
                for item in record.files
            )

        if not (
            inventory(self.source)
            == inventory(self.replay)
            == inventory(self.integrated)
        ):
            raise ProtectedAcceptanceDenied(
                "source, clean replay, and integrated product inventories differ"
            )
        if len({self.source.commit, self.replay.commit, self.integrated.commit}) != 3:
            raise ProtectedAcceptanceDenied(
                "source, replay, and integrated commits must be independent"
            )
        if not (
            self.source.canonical_patch_content_id
            == self.replay.canonical_patch_content_id
            == self.integrated.canonical_patch_content_id
        ):
            raise ProtectedAcceptanceDenied(
                "source, replay, and integrated canonical full-index patches differ"
            )
        if self.canonical_diff_content_id != self.source.canonical_patch_content_id:
            raise ProtectedAcceptanceDenied(
                "top-level canonical diff binding is inconsistent"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "source": self.source.to_dict(),
            "replay": self.replay.to_dict(),
            "integrated": self.integrated.to_dict(),
            "canonical_diff_content_id": self.canonical_diff_content_id,
        }


@dataclass(frozen=True)
class ProductProvenanceRequest:
    """Typed inspection request for one independently replayed pre-Q product."""

    task_id: str
    source_commit: str
    replay_commit: str
    integrated_commit: str
    product_paths: tuple[str, ...]
    source_test_evidence: tuple[EvidenceHandle, ...]
    replay_test_evidence: tuple[EvidenceHandle, ...]
    integrated_test_evidence: tuple[EvidenceHandle, ...]

    def __post_init__(self) -> None:
        if self.task_id not in PRE_Q_PRODUCT_TASKS:
            raise ProtectedAcceptanceDenied(
                "provenance request is not one of the six pre-Q products"
            )
        _object_id(self.source_commit, "source commit")
        _object_id(self.replay_commit, "replay commit")
        _object_id(self.integrated_commit, "integrated commit")
        if len({self.source_commit, self.replay_commit, self.integrated_commit}) != 3:
            raise ProtectedAcceptanceDenied(
                "source, replay, and integrated commits must be distinct"
            )
        if type(self.product_paths) is not tuple or not self.product_paths:
            raise TypeError("product_paths must be a non-empty tuple")
        paths = tuple(protected_git_path(item) for item in self.product_paths)
        if len(set(paths)) != len(paths):
            raise ProtectedAcceptanceError("product provenance paths are duplicated")
        object.__setattr__(self, "product_paths", paths)
        for name in (
            "source_test_evidence",
            "replay_test_evidence",
            "integrated_test_evidence",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(not isinstance(item, EvidenceHandle) for item in values)
            ):
                raise TypeError(f"{name} must be a non-empty tuple of EvidenceHandle")


@dataclass(frozen=True)
class StableQPolicy:
    policy_id: str
    phases: tuple[str, ...] = tuple(item.value for item in PROMPT_V3_PHASE_ORDER)
    maximum_p031_attempts: int = 3

    def __post_init__(self) -> None:
        _digest(self.policy_id, "Q policy_id")
        if type(self.phases) is not tuple or self.phases != tuple(
            item.value for item in PROMPT_V3_PHASE_ORDER
        ):
            raise ProtectedAcceptanceDenied(
                "Q policy phase order is not the stable closed order"
            )
        if (
            type(self.maximum_p031_attempts) is not int
            or self.maximum_p031_attempts != 3
        ):
            raise ProtectedAcceptanceDenied(
                "Q policy must retain the three-attempt P031 bound"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "phases": list(self.phases),
            "maximum_p031_attempts": self.maximum_p031_attempts,
        }


@dataclass(frozen=True)
class PromptV3QInventory:
    lifecycle_root_identity_did: str
    stable_policy: StableQPolicy
    product_provenance: tuple[ProductProvenance, ...]
    schema: str = Q_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != Q_INVENTORY_SCHEMA:
            raise ProtectedAcceptanceError("unsupported Q inventory schema")
        root = _text(
            self.lifecycle_root_identity_did, "lifecycle root DID", maximum=256
        )
        if not root.startswith("did:key:z"):
            raise ProtectedAcceptanceError("lifecycle root identity must be a did:key")
        if not isinstance(self.stable_policy, StableQPolicy):
            raise TypeError("stable_policy must be StableQPolicy")
        if type(self.product_provenance) is not tuple or any(
            not isinstance(item, ProductProvenance) for item in self.product_provenance
        ):
            raise TypeError("product_provenance must be a tuple of ProductProvenance")
        task_ids = tuple(item.task_id for item in self.product_provenance)
        if len(set(task_ids)) != len(task_ids) or set(task_ids) != PRE_Q_PRODUCT_TASKS:
            raise ProtectedAcceptanceDenied(
                "Q requires exactly the six independent pre-Q products"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lifecycle_root_identity_did": self.lifecycle_root_identity_did,
            "stable_policy": self.stable_policy.to_dict(),
            "product_provenance": [item.to_dict() for item in self.product_provenance],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PromptV3QInventory:
        # The exact-key check is the future-pin barrier.  Reviewer identity,
        # profile, anchor, signature, timestamps, observations, generation,
        # capsule output, and all other future authority are rejected here.
        _strict_keys(
            value,
            (
                "schema",
                "lifecycle_root_identity_did",
                "stable_policy",
                "product_provenance",
            ),
            "Q inventory",
        )
        policy = value["stable_policy"]
        _strict_keys(
            policy, ("policy_id", "phases", "maximum_p031_attempts"), "Q stable policy"
        )
        products = value["product_provenance"]
        if type(products) is not list:
            raise TypeError("Q product_provenance must be a list")
        parsed_products = tuple(
            product_provenance_from_mapping(item) for item in products
        )
        phases = policy["phases"]
        if type(phases) is not list or any(type(item) is not str for item in phases):
            raise TypeError("Q stable-policy phases must be a list of strings")
        return cls(
            schema=value["schema"],
            lifecycle_root_identity_did=value["lifecycle_root_identity_did"],
            stable_policy=StableQPolicy(
                policy_id=policy["policy_id"],
                phases=tuple(phases),
                maximum_p031_attempts=policy["maximum_p031_attempts"],
            ),
            product_provenance=parsed_products,
        )


def _generation_from_mapping(
    value: Mapping[str, Any], expected_role: str
) -> ProductGenerationRecord:
    _strict_keys(
        value,
        (
            "role",
            "commit",
            "parent",
            "tree",
            "files",
            "test_evidence",
            "canonical_patch_content_id",
        ),
        "product record",
    )
    if value["role"] != expected_role:
        raise ProtectedAcceptanceError("product record role mismatch")
    files = value["files"]
    tests = value["test_evidence"]
    if type(files) is not list or type(tests) is not list:
        raise TypeError("product files and test evidence must be lists")
    parsed_files = []
    for item in files:
        _strict_keys(
            item,
            ("path", "mode", "blob_id", "raw_content_id", "byte_length"),
            "file identity",
        )
        parsed_files.append(GitFileIdentity(**dict(item)))
    return ProductGenerationRecord(
        role=value["role"],
        commit=value["commit"],
        parent=value["parent"],
        tree=value["tree"],
        files=tuple(parsed_files),
        test_evidence=tuple(EvidenceHandle.from_mapping(item) for item in tests),
        canonical_patch_content_id=value["canonical_patch_content_id"],
    )


def product_provenance_from_mapping(value: Mapping[str, Any]) -> ProductProvenance:
    _strict_keys(
        value,
        (
            "schema",
            "task_id",
            "source",
            "replay",
            "integrated",
            "canonical_diff_content_id",
        ),
        "product provenance",
    )
    return ProductProvenance(
        schema=value["schema"],
        task_id=value["task_id"],
        source=_generation_from_mapping(value["source"], "source"),
        replay=_generation_from_mapping(value["replay"], "replay"),
        integrated=_generation_from_mapping(value["integrated"], "integrated"),
        canonical_diff_content_id=value["canonical_diff_content_id"],
    )


@dataclass(frozen=True)
class PhaseAuthority:
    phase: PromptV3Phase
    authority_id: str
    nonce: str
    parent_commit: str
    identity_did: str
    issued_at_ns: int
    expires_at_ns: int
    schema: str = PHASE_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PHASE_AUTHORITY_SCHEMA or not isinstance(
            self.phase, PromptV3Phase
        ):
            raise ProtectedAcceptanceError("phase authority schema or phase is invalid")
        _digest(self.authority_id, "phase authority_id")
        if type(self.nonce) is not str or not re.fullmatch(
            r"[A-Za-z0-9_-]{22,128}", self.nonce
        ):
            raise ProtectedAcceptanceError("phase authority nonce is invalid")
        _object_id(self.parent_commit, "phase authority parent")
        if not self.identity_did.startswith("did:key:z"):
            raise ProtectedAcceptanceError("phase authority identity must be did:key")
        if (
            type(self.issued_at_ns) is not int
            or type(self.expires_at_ns) is not int
            or self.issued_at_ns <= 0
            or self.expires_at_ns <= self.issued_at_ns
        ):
            raise ProtectedAcceptanceError("phase authority time bounds are invalid")
        expected_id = phase_authority_content_id(
            phase=self.phase,
            nonce=self.nonce,
            parent_commit=self.parent_commit,
            identity_did=self.identity_did,
            issued_at_ns=self.issued_at_ns,
            expires_at_ns=self.expires_at_ns,
        )
        if self.authority_id != expected_id:
            raise ProtectedAcceptanceDenied(
                "phase authority content ID is inconsistent"
            )


@dataclass(frozen=True)
class PhasePolicy:
    phase: PromptV3Phase
    expected_parent_phase: PromptV3Phase | None
    allowed_paths: tuple[str, ...]
    required_evidence_kinds: tuple[str, ...]
    validator_ids: tuple[str, ...]
    maximum_total_bytes: int = MAX_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.phase, PromptV3Phase):
            raise TypeError("phase policy phase must be PromptV3Phase")
        if self.expected_parent_phase != immediate_parent_phase(self.phase):
            raise ProtectedAcceptanceDenied(
                "phase policy skips or rewrites the phase order"
            )
        if type(self.allowed_paths) is not tuple or not self.allowed_paths:
            raise TypeError("allowed_paths must be a non-empty tuple")
        paths = tuple(protected_git_path(item) for item in self.allowed_paths)
        if len(set(paths)) != len(paths):
            raise ProtectedAcceptanceError("phase policy contains duplicate paths")
        if (
            type(self.required_evidence_kinds) is not tuple
            or not self.required_evidence_kinds
            or any(
                type(item) is not str or not _TOKEN_RE.fullmatch(item)
                for item in self.required_evidence_kinds
            )
        ):
            raise TypeError(
                "required evidence kinds must be a non-empty tuple of bounded tokens"
            )
        if (
            type(self.validator_ids) is not tuple
            or not self.validator_ids
            or any(
                type(item) is not str or not _TOKEN_RE.fullmatch(item)
                for item in self.validator_ids
            )
        ):
            raise TypeError("validator_ids must be a non-empty tuple of bounded tokens")
        _positive_int(
            self.maximum_total_bytes, "maximum_total_bytes", MAX_ARTIFACT_BYTES
        )


@dataclass(frozen=True)
class ArtifactBytes:
    path: str
    data: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", protected_git_path(self.path))
        if (
            type(self.data) is not bytes
            or not self.data
            or len(self.data) > MAX_ARTIFACT_BYTES
        ):
            raise ProtectedAcceptanceError(
                "artifact bytes are empty, untyped, or oversized"
            )


@dataclass(frozen=True)
class RepositoryBinding:
    root: str
    target_ref: str
    # Kept as a literal here to preserve the core package's no-upward-import
    # rule.  It is the canonical merge.checkout_lock namespace.
    lease_name: str = "implementation-main-merge.lock"

    def __post_init__(self) -> None:
        if (
            type(self.root) is not str
            or not self.root.startswith("/")
            or "\0" in self.root
        ):
            raise ProtectedAcceptanceError(
                "repository root must be an absolute typed binding"
            )
        if (
            type(self.target_ref) is not str
            or not re.fullmatch(
                r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{0,240}", self.target_ref
            )
            or ".." in self.target_ref
            or self.target_ref.endswith("/")
        ):
            raise ProtectedAcceptanceError(
                "target ref is not a canonical protected branch ref"
            )
        if type(self.lease_name) is not str or not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]{0,95}\.lock", self.lease_name
        ):
            raise ProtectedAcceptanceError("lease name is invalid")


@dataclass(frozen=True)
class PhaseCandidateRequest:
    repository: RepositoryBinding
    policy: PhasePolicy
    parent_commit: str
    parent_phase: PromptV3Phase | None
    authority: PhaseAuthority
    artifacts: tuple[ArtifactBytes, ...]
    evidence_handles: tuple[EvidenceHandle, ...]
    commit_message: str
    commit_timestamp: str
    observed_at_ns: int
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository, RepositoryBinding) or not isinstance(
            self.policy, PhasePolicy
        ):
            raise TypeError("candidate repository and policy must be typed bindings")
        _object_id(self.parent_commit, "candidate parent_commit")
        if self.parent_phase != self.policy.expected_parent_phase:
            raise ProtectedAcceptanceDenied(
                "candidate parent phase does not immediately precede target"
            )
        if not isinstance(self.authority, PhaseAuthority) or (
            self.authority.phase != self.policy.phase
            or self.authority.parent_commit != self.parent_commit
        ):
            raise ProtectedAcceptanceDenied("candidate authority is not phase-local")
        if (
            type(self.artifacts) is not tuple
            or not self.artifacts
            or any(not isinstance(item, ArtifactBytes) for item in self.artifacts)
        ):
            raise TypeError(
                "candidate artifacts must be a non-empty tuple of ArtifactBytes"
            )
        paths = tuple(item.path for item in self.artifacts)
        if len(set(paths)) != len(paths) or set(paths) != set(
            self.policy.allowed_paths
        ):
            raise ProtectedAcceptanceDenied(
                "candidate artifact inventory differs from phase policy"
            )
        if (
            sum(len(item.data) for item in self.artifacts)
            > self.policy.maximum_total_bytes
        ):
            raise ProtectedAcceptanceDenied(
                "candidate artifacts exceed phase byte bound"
            )
        if type(self.evidence_handles) is not tuple or any(
            not isinstance(item, EvidenceHandle) for item in self.evidence_handles
        ):
            raise TypeError(
                "candidate evidence must contain typed EvidenceHandle values"
            )
        kinds = {item.kind for item in self.evidence_handles}
        if not set(self.policy.required_evidence_kinds).issubset(kinds):
            raise ProtectedAcceptanceDenied("candidate lacks required bounded evidence")
        _text(self.commit_message, "commit message", maximum=512)
        if type(self.commit_timestamp) is not str or not re.fullmatch(
            r"[0-9]{10,20} [+-][0-9]{4}", self.commit_timestamp
        ):
            raise ProtectedAcceptanceError(
                "commit_timestamp must be a canonical Git timestamp"
            )
        if (
            type(self.observed_at_ns) is not int
            or not self.authority.issued_at_ns
            <= self.observed_at_ns
            < self.authority.expires_at_ns
        ):
            raise ProtectedAcceptanceDenied("phase authority is not currently fresh")
        if type(self.dry_run) is not bool:
            raise TypeError("dry_run must be bool")


@dataclass(frozen=True)
class CandidatePlan:
    request: PhaseCandidateRequest
    tree_id: str
    commit_id: str
    rescue_ref: str
    file_identities: tuple[GitFileIdentity, ...]
    lease_id: str
    lease_device: int
    lease_inode: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, PhaseCandidateRequest):
            raise TypeError("candidate plan request must be PhaseCandidateRequest")
        _object_id(self.tree_id, "candidate tree")
        _object_id(self.commit_id, "candidate commit")
        if type(self.rescue_ref) is not str or not self.rescue_ref.startswith(
            "refs/agent-supervisor/protected-acceptance-rescue/"
        ):
            raise ProtectedAcceptanceError("candidate rescue ref is invalid")
        if type(self.file_identities) is not tuple or any(
            not isinstance(item, GitFileIdentity) for item in self.file_identities
        ):
            raise TypeError("candidate file identities are invalid")
        if any(item.mode != "100644" for item in self.file_identities):
            raise ProtectedAcceptanceDenied(
                "new protected artifacts must be mode 100644"
            )
        _digest(self.lease_id, "checkout lease_id")
        _positive_int(self.lease_device, "lease device", 2**63 - 1)
        _positive_int(self.lease_inode, "lease inode", 2**63 - 1)


@dataclass(frozen=True)
class PhaseEvidenceResult:
    candidate: CandidatePlan
    handles: tuple[EvidenceHandle, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate, CandidatePlan)
            or type(self.handles) is not tuple
        ):
            raise TypeError("phase evidence result is not strictly typed")
        if any(not isinstance(item, EvidenceHandle) for item in self.handles):
            raise TypeError("phase evidence result contains an untyped handle")
        required = set(self.candidate.request.policy.required_evidence_kinds)
        if not required.issubset({item.kind for item in self.handles}):
            raise ProtectedAcceptanceDenied("phase evidence result is incomplete")


@dataclass(frozen=True)
class ValidatedCandidate:
    evidence: PhaseEvidenceResult
    validation_handles: tuple[EvidenceHandle, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evidence, PhaseEvidenceResult)
            or type(self.validation_handles) is not tuple
        ):
            raise TypeError("validated candidate is not strictly typed")
        if any(
            not isinstance(item, EvidenceHandle) for item in self.validation_handles
        ):
            raise TypeError("validation handles must be EvidenceHandle values")
        validators = set(self.evidence.candidate.request.policy.validator_ids)
        if not validators.issubset({item.kind for item in self.validation_handles}):
            raise ProtectedAcceptanceDenied(
                "not every phase validator produced evidence"
            )


@dataclass(frozen=True)
class PublicationResult:
    candidate: CandidatePlan
    old_commit: str
    new_commit: str
    published: bool
    dry_run: bool
    settlement_pending: bool = False


@dataclass(frozen=True)
class RejectionResult:
    candidate: CandidatePlan
    target_rolled_back: bool
    rescue_ref_deleted: bool
    lease_released: bool


@dataclass(frozen=True)
class QuiescenceRequest:
    generation: int
    required_lane_ids: tuple[str, ...]
    evidence_handles: tuple[EvidenceHandle, ...]

    def __post_init__(self) -> None:
        _positive_int(self.generation, "quiescence generation", 2**63 - 1)
        if (
            type(self.required_lane_ids) is not tuple
            or not self.required_lane_ids
            or any(
                type(item) is not str or not _TOKEN_RE.fullmatch(item)
                for item in self.required_lane_ids
            )
        ):
            raise TypeError("quiescence lane IDs must be a non-empty tuple of tokens")
        if len(set(self.required_lane_ids)) != len(self.required_lane_ids):
            raise ProtectedAcceptanceError("quiescence lane IDs must be distinct")
        if type(self.evidence_handles) is not tuple or any(
            not isinstance(item, EvidenceHandle) for item in self.evidence_handles
        ):
            raise TypeError("quiescence evidence handles are invalid")


@dataclass(frozen=True)
class QuiescenceObservation:
    generation: int
    terminal_lane_ids: tuple[str, ...]
    fenced: bool
    evidence: EvidenceHandle

    def __post_init__(self) -> None:
        _positive_int(self.generation, "observed generation", 2**63 - 1)
        if type(self.terminal_lane_ids) is not tuple or len(
            set(self.terminal_lane_ids)
        ) != len(self.terminal_lane_ids):
            raise TypeError("terminal lane IDs must be a distinct tuple")
        if type(self.fenced) is not bool or not isinstance(
            self.evidence, EvidenceHandle
        ):
            raise TypeError("quiescence observation is not strictly typed")


@dataclass(frozen=True)
class SignedArtifactRequest:
    phase: PromptV3Phase
    authority: PhaseAuthority
    body: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, PromptV3Phase) or not isinstance(
            self.authority, PhaseAuthority
        ):
            raise TypeError("signed artifact request requires typed phase authority")
        if self.authority.phase != self.phase:
            raise ProtectedAcceptanceDenied("artifact authority is not phase-local")
        if not isinstance(self.body, Mapping):
            raise TypeError("artifact body must be a mapping")
        canonical_json_bytes(self.body)


@dataclass(frozen=True)
class P031FailedAttempt:
    attempt: int
    authorization_id: str
    nonce: str
    failure_evidence: EvidenceHandle
    signed_failure_content_id: str

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt not in {1, 2, 3}:
            raise ProtectedAcceptanceDenied(
                "P031 attempt is outside the three-attempt bound"
            )
        _digest(self.authorization_id, "P031 authorization_id")
        if type(self.nonce) is not str or not re.fullmatch(
            r"[A-Za-z0-9_-]{22,128}", self.nonce
        ):
            raise ProtectedAcceptanceError("P031 nonce is invalid")
        if not isinstance(self.failure_evidence, EvidenceHandle):
            raise TypeError("P031 failure evidence must be an EvidenceHandle")
        _digest(self.signed_failure_content_id, "signed P031 failure content_id")


@dataclass(frozen=True)
class P031AttemptLedger:
    attempts: tuple[P031FailedAttempt, ...]
    schema: str = P031_ATTEMPT_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != P031_ATTEMPT_LEDGER_SCHEMA
            or type(self.attempts) is not tuple
        ):
            raise TypeError("P031 attempt ledger is not strictly typed")
        if len(self.attempts) > 3 or any(
            not isinstance(item, P031FailedAttempt) for item in self.attempts
        ):
            raise ProtectedAcceptanceDenied("P031 ledger exceeds its closed bound")
        if tuple(item.attempt for item in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ProtectedAcceptanceDenied(
                "P031 ledger must be append-only and sequential"
            )
        if len({item.authorization_id for item in self.attempts}) != len(self.attempts):
            raise ProtectedAcceptanceDenied(
                "P031 authorization cannot be retried or reused"
            )
        if len({item.nonce for item in self.attempts}) != len(self.attempts):
            raise ProtectedAcceptanceDenied(
                "P031 reauthorization must use a fresh nonce"
            )

    def append(self, failed: P031FailedAttempt) -> P031AttemptLedger:
        if not isinstance(failed, P031FailedAttempt):
            raise TypeError("P031 append requires P031FailedAttempt")
        return P031AttemptLedger(attempts=(*self.attempts, failed))


@dataclass(frozen=True)
class RuntimeLaunchAuthorityRequest:
    l_commit: str
    expected_l_tree: str
    expected_l_raw_content_id: str
    target_generation: int
    evidence_handles: tuple[EvidenceHandle, ...]

    def __post_init__(self) -> None:
        _object_id(self.l_commit, "L commit")
        _object_id(self.expected_l_tree, "L tree")
        _digest(self.expected_l_raw_content_id, "L raw content_id")
        _positive_int(self.target_generation, "target generation", 2**63 - 1)
        if type(self.evidence_handles) is not tuple or any(
            not isinstance(item, EvidenceHandle) for item in self.evidence_handles
        ):
            raise TypeError("runtime authority evidence must be typed")


@dataclass(frozen=True)
class VerifiedRuntimeLaunchAuthority:
    l_commit: str
    l_tree: str
    l_raw_content_id: str
    runtime_native_authorization_id: str
    target_generation: int
    accepted_a031_id: str
    accepted_a032_id: str
    accepted_a023_027_id: str
    pin_binding_id: str
    config_binding_id: str
    control_plane_binding_id: str
    schema: str = RUNTIME_LAUNCH_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_LAUNCH_AUTHORITY_SCHEMA:
            raise ProtectedAcceptanceError("runtime authority schema is invalid")
        _object_id(self.l_commit, "verified L commit")
        _object_id(self.l_tree, "verified L tree")
        _digest(self.l_raw_content_id, "verified L raw content_id")
        _positive_int(self.target_generation, "verified target generation", 2**63 - 1)
        for name in (
            "runtime_native_authorization_id",
            "accepted_a031_id",
            "accepted_a032_id",
            "accepted_a023_027_id",
            "pin_binding_id",
            "config_binding_id",
            "control_plane_binding_id",
        ):
            _digest(getattr(self, name), name)
        if self.runtime_native_authorization_id in {
            self.accepted_a031_id,
            self.accepted_a032_id,
            self.accepted_a023_027_id,
        }:
            raise ProtectedAcceptanceDenied(
                "runtime native authority must be fresh and non-reused"
            )


class ProductProvenanceInspector(Protocol):
    def __call__(self, request: ProductProvenanceRequest) -> ProductProvenance: ...


class ArtifactSigner(Protocol):
    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, str]: ...


class PhaseEvidenceRunner(Protocol):
    def __call__(self, candidate: CandidatePlan) -> tuple[EvidenceHandle, ...]: ...


class PhaseCandidateValidator(Protocol):
    def __call__(
        self, candidate: CandidatePlan, evidence: PhaseEvidenceResult
    ) -> tuple[EvidenceHandle, ...]: ...


class PreCASValidator(Protocol):
    def __call__(self, candidate: ValidatedCandidate) -> bool: ...


class QuiescenceObserver(Protocol):
    def __call__(self, request: QuiescenceRequest) -> QuiescenceObservation: ...


class RuntimeAuthorityLoader(Protocol):
    def __call__(self, request: RuntimeLaunchAuthorityRequest) -> Mapping[str, Any]: ...


class RuntimeAuthorityValidator(Protocol):
    def __call__(
        self, request: RuntimeLaunchAuthorityRequest, loaded: Mapping[str, Any]
    ) -> VerifiedRuntimeLaunchAuthority: ...


__all__ = (
    "EVIDENCE_HANDLE_SCHEMA",
    "P031_ATTEMPT_LEDGER_SCHEMA",
    "PHASE_AUTHORITY_SCHEMA",
    "PHASE_RECEIPT_SCHEMA",
    "PRE_Q_PRODUCT_TASKS",
    "PRODUCT_PROVENANCE_SCHEMA",
    "PROMPT_V3_PHASE_ORDER",
    "Q_INVENTORY_SCHEMA",
    "RUNTIME_LAUNCH_AUTHORITY_SCHEMA",
    "ArtifactBytes",
    "ArtifactSigner",
    "CandidatePlan",
    "EvidenceHandle",
    "GitFileIdentity",
    "P031AttemptLedger",
    "P031FailedAttempt",
    "PhaseAuthority",
    "PhaseCandidateRequest",
    "PhaseCandidateValidator",
    "PhaseEvidenceResult",
    "PhaseEvidenceRunner",
    "PhasePolicy",
    "PreCASValidator",
    "ProductGenerationRecord",
    "ProductProvenance",
    "ProductProvenanceInspector",
    "ProductProvenanceRequest",
    "PromptV3Phase",
    "PromptV3QInventory",
    "ProtectedAcceptanceDenied",
    "ProtectedAcceptanceError",
    "PublicationResult",
    "QuiescenceObservation",
    "QuiescenceObserver",
    "QuiescenceRequest",
    "RejectionResult",
    "RepositoryBinding",
    "RuntimeAuthorityLoader",
    "RuntimeAuthorityValidator",
    "RuntimeLaunchAuthorityRequest",
    "SignedArtifactRequest",
    "StableQPolicy",
    "ValidatedCandidate",
    "VerifiedRuntimeLaunchAuthority",
    "canonical_json_bytes",
    "content_id",
    "immediate_parent_phase",
    "phase_authority_content_id",
    "product_provenance_from_mapping",
    "protected_git_path",
)

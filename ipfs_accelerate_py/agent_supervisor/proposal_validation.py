"""Strict, content-addressed validation of implementation proposals.

The proposal gate is deliberately cheaper than test, semantic, or proof
validation.  It normalizes the candidate diff, checks its authority and path
envelope, proves that it has an observable effect, and emits a tamper-evident
receipt.  The receipt is not completion or proof authority; it is an admission
token consumed by :mod:`validation_scheduler`.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .code_proof_obligations import CandidateDiffEntry, DiffChangeKind


NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID = (
    "314133036252270790078901745919131980427"
)
PROPOSAL_VALIDATION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-policy@1"
)
PROPOSAL_VALIDATION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-request@1"
)
PROPOSAL_VALIDATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-receipt@1"
)
PROPOSAL_REJECTION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-rejection-evidence@1"
)


class ProposalValidationError(ValueError):
    """Raised when a persisted proposal-validation record is inconsistent."""


class ProposalGate(str, Enum):
    SCHEMA = "schema"
    AUTHORITY = "authority"
    PATCH = "patch"
    PATH = "path"
    AST_INTERFACE = "ast_interface"


ORDERED_PROPOSAL_GATES = tuple(ProposalGate)


class ProposalFindingCode(str, Enum):
    INVALID_SCHEMA = "invalid_schema"
    AUTHORITY_MISMATCH = "authority_mismatch"
    STALE_BASELINE = "stale_baseline"
    EMPTY_PATCH = "empty_patch"
    NO_SEMANTIC_CHANGE = "no_semantic_change"
    PATCH_TOO_LARGE = "patch_too_large"
    UNSAFE_PATH = "unsafe_path"
    PATH_OUTSIDE_SCOPE = "path_outside_scope"
    DECLARED_PATH_MISMATCH = "declared_path_mismatch"
    BINARY_CHANGE_FORBIDDEN = "binary_change_forbidden"
    GENERATED_CHANGE_FORBIDDEN = "generated_change_forbidden"
    TEST_DELETION_FORBIDDEN = "test_deletion_forbidden"
    PYTHON_SYNTAX_ERROR = "python_syntax_error"


QUALIFYING_FAIL_FAST_CODES = frozenset(
    {
        ProposalFindingCode.EMPTY_PATCH,
        ProposalFindingCode.NO_SEMANTIC_CHANGE,
        ProposalFindingCode.PATH_OUTSIDE_SCOPE,
        ProposalFindingCode.UNSAFE_PATH,
    }
)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProposalValidationError("non-finite values are not canonical")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProposalValidationError("record keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_canonical_json)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    raise ProposalValidationError(
        f"unsupported canonical value: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _path_matches(path: str, pattern: str) -> bool:
    pattern = str(pattern).strip().replace("\\", "/").lstrip("./")
    if not pattern:
        return False
    if any(character in pattern for character in "*?["):
        return fnmatch.fnmatchcase(path, pattern)
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern or path.startswith(pattern.rstrip("/") + "/")


def _entry_has_effect(entry: CandidateDiffEntry) -> bool:
    if entry.change_kind in {DiffChangeKind.ADD, DiffChangeKind.COPY}:
        return bool(entry.after_source is not None or entry.after_blob_id)
    if entry.change_kind is DiffChangeKind.DELETE:
        return bool(entry.before_source is not None or entry.before_blob_id)
    if entry.old_path != entry.new_path:
        return True
    if entry.before_source is not None and entry.after_source is not None:
        return entry.before_source != entry.after_source
    if entry.before_blob_id and entry.after_blob_id:
        return entry.before_blob_id != entry.after_blob_id
    # A modification with only one materialized side is observable, although
    # the later AST/scope compiler may conservatively reject it.
    return bool(
        entry.before_source is not None
        or entry.after_source is not None
        or entry.before_blob_id
        or entry.after_blob_id
    )


@dataclass(frozen=True)
class ProposalValidationFinding:
    code: ProposalFindingCode
    gate: ProposalGate
    message: str
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ProposalFindingCode(self.code))
        object.__setattr__(self, "gate", ProposalGate(self.gate))
        object.__setattr__(self, "message", " ".join(str(self.message).split()))
        object.__setattr__(self, "path", str(self.path or "").strip())
        if not self.message:
            raise ProposalValidationError("proposal finding message is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "gate": self.gate.value,
            "path": self.path,
            "message": self.message,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProposalValidationFinding":
        return cls(
            code=payload.get("code", ""),
            gate=payload.get("gate", ""),
            path=str(payload.get("path") or ""),
            message=str(payload.get("message") or ""),
        )


@dataclass(frozen=True)
class ProposalValidationPolicy:
    allowed_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...] = (
        ".git/",
        ".env",
        ".ssh/",
    )
    expected_task_id: str = ""
    expected_plan_id: str = ""
    expected_repository_id: str = ""
    expected_repository_tree_id: str = ""
    expected_objective_id: str = ""
    allow_binary: bool = False
    allow_generated: bool = False
    allow_test_deletion: bool = False
    require_declared_paths: bool = True
    require_python_syntax: bool = True
    max_diff_entries: int = 256
    max_patch_bytes: int = 2_000_000
    max_findings: int = 32
    policy_version: str = "strict-proposal-v1"
    policy_id: str = ""

    def __post_init__(self) -> None:
        allowed = _strings(self.allowed_paths)
        forbidden = _strings(self.forbidden_paths)
        if not allowed:
            raise ProposalValidationError("allowed_paths must not be empty")
        object.__setattr__(self, "allowed_paths", allowed)
        object.__setattr__(self, "forbidden_paths", forbidden)
        for name in (
            "expected_task_id",
            "expected_plan_id",
            "expected_repository_id",
            "expected_repository_tree_id",
            "expected_objective_id",
            "policy_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in ("max_diff_entries", "max_patch_bytes", "max_findings"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ProposalValidationError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        claimed = str(self.policy_id or "").strip()
        object.__setattr__(self, "policy_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal policy identity mismatch")
        object.__setattr__(self, "policy_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_POLICY_SCHEMA,
            "allowed_paths": self.allowed_paths,
            "forbidden_paths": self.forbidden_paths,
            "expected_task_id": self.expected_task_id,
            "expected_plan_id": self.expected_plan_id,
            "expected_repository_id": self.expected_repository_id,
            "expected_repository_tree_id": self.expected_repository_tree_id,
            "expected_objective_id": self.expected_objective_id,
            "allow_binary": self.allow_binary,
            "allow_generated": self.allow_generated,
            "allow_test_deletion": self.allow_test_deletion,
            "require_declared_paths": self.require_declared_paths,
            "require_python_syntax": self.require_python_syntax,
            "max_diff_entries": self.max_diff_entries,
            "max_patch_bytes": self.max_patch_bytes,
            "max_findings": self.max_findings,
            "policy_version": self.policy_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationPolicy":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_POLICY_SCHEMA)
        if schema != PROPOSAL_VALIDATION_POLICY_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal policy schema: {schema}")
        return cls(
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            forbidden_paths=tuple(payload.get("forbidden_paths") or ()),
            expected_task_id=str(payload.get("expected_task_id") or ""),
            expected_plan_id=str(payload.get("expected_plan_id") or ""),
            expected_repository_id=str(payload.get("expected_repository_id") or ""),
            expected_repository_tree_id=str(
                payload.get("expected_repository_tree_id") or ""
            ),
            expected_objective_id=str(payload.get("expected_objective_id") or ""),
            allow_binary=payload.get("allow_binary", False),
            allow_generated=payload.get("allow_generated", False),
            allow_test_deletion=payload.get("allow_test_deletion", False),
            require_declared_paths=payload.get("require_declared_paths", True),
            require_python_syntax=payload.get("require_python_syntax", True),
            max_diff_entries=int(payload.get("max_diff_entries", 256)),
            max_patch_bytes=int(payload.get("max_patch_bytes", 2_000_000)),
            max_findings=int(payload.get("max_findings", 32)),
            policy_version=str(payload.get("policy_version") or "strict-proposal-v1"),
            policy_id=str(payload.get("policy_id") or ""),
        )


@dataclass(frozen=True)
class ImplementationProposal:
    task_id: str
    accepted_plan_id: str
    repository_id: str
    repository_tree_id: str
    objective_id: str
    baseline_id: str
    candidate_diff: tuple[CandidateDiffEntry, ...]
    declared_paths: tuple[str, ...]
    context_id: str = ""
    proposal_version: str = "1"
    proposal_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "accepted_plan_id",
            "repository_id",
            "repository_tree_id",
            "objective_id",
            "baseline_id",
            "context_id",
            "proposal_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "task_id",
            "accepted_plan_id",
            "repository_tree_id",
            "objective_id",
            "baseline_id",
        ):
            if not getattr(self, name):
                raise ProposalValidationError(f"{name} is required")
        entries = tuple(
            item
            if isinstance(item, CandidateDiffEntry)
            else CandidateDiffEntry.from_mapping(item)
            for item in self.candidate_diff
        )
        object.__setattr__(self, "candidate_diff", entries)
        object.__setattr__(self, "declared_paths", _strings(self.declared_paths))
        claimed = str(self.proposal_id or "").strip()
        object.__setattr__(self, "proposal_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal identity mismatch")
        object.__setattr__(self, "proposal_id", actual)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        paths: set[str] = set()
        for entry in self.candidate_diff:
            paths.update(path for path in (entry.old_path, entry.new_path) if path)
        return tuple(sorted(paths))

    @property
    def effective_entries(self) -> tuple[CandidateDiffEntry, ...]:
        return tuple(entry for entry in self.candidate_diff if _entry_has_effect(entry))

    @property
    def diff_digest(self) -> str:
        return _identity(
            [entry.to_dict(include_sources=True) for entry in self.candidate_diff]
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_REQUEST_SCHEMA,
            "proposal_version": self.proposal_version,
            "task_id": self.task_id,
            "accepted_plan_id": self.accepted_plan_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "baseline_id": self.baseline_id,
            "context_id": self.context_id,
            "declared_paths": self.declared_paths,
            "candidate_diff": [
                entry.to_dict(include_sources=True) for entry in self.candidate_diff
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "changed_paths": self.changed_paths,
            "diff_digest": self.diff_digest,
            "proposal_id": self.proposal_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationProposal":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_REQUEST_SCHEMA)
        if schema != PROPOSAL_VALIDATION_REQUEST_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal schema: {schema}")
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            accepted_plan_id=str(
                payload.get("accepted_plan_id") or payload.get("plan_id") or ""
            ),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(
                payload.get("repository_tree_id") or payload.get("tree_id") or ""
            ),
            objective_id=str(
                payload.get("objective_id") or payload.get("goal_id") or ""
            ),
            baseline_id=str(payload.get("baseline_id") or ""),
            context_id=str(payload.get("context_id") or ""),
            declared_paths=tuple(payload.get("declared_paths") or ()),
            candidate_diff=tuple(
                CandidateDiffEntry.from_mapping(item)
                for item in payload.get("candidate_diff") or ()
            ),
            proposal_version=str(payload.get("proposal_version") or "1"),
            proposal_id=str(payload.get("proposal_id") or ""),
        )
        if payload.get("diff_digest") and payload["diff_digest"] != result.diff_digest:
            raise ProposalValidationError("proposal diff digest mismatch")
        if payload.get("changed_paths") and tuple(payload["changed_paths"]) != result.changed_paths:
            raise ProposalValidationError("proposal changed paths mismatch")
        return result


ProposalValidationRequest = ImplementationProposal


@dataclass(frozen=True)
class ProposalRejectionEvidence:
    requirement_id: str
    proposal_id: str
    receipt_id: str
    repository_tree_id: str
    objective_id: str
    policy_id: str
    rejection_codes: tuple[str, ...]
    expensive_node_ids: tuple[str, ...]
    expensive_checks_started: int = 0
    evidence_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "proposal_id",
            "receipt_id",
            "repository_tree_id",
            "objective_id",
            "policy_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if self.requirement_id != NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID:
            raise ProposalValidationError("unsupported fail-fast requirement")
        if not all(
            (
                self.proposal_id,
                self.receipt_id,
                self.repository_tree_id,
                self.objective_id,
                self.policy_id,
            )
        ):
            raise ProposalValidationError("rejection evidence binding is incomplete")
        codes = _strings(self.rejection_codes)
        if not set(codes).intersection(code.value for code in QUALIFYING_FAIL_FAST_CODES):
            raise ProposalValidationError(
                "rejection evidence requires a no-op or out-of-scope code"
            )
        object.__setattr__(self, "rejection_codes", codes)
        object.__setattr__(self, "expensive_node_ids", _strings(self.expensive_node_ids))
        if isinstance(self.expensive_checks_started, bool):
            raise ProposalValidationError("expensive_checks_started must be an integer")
        if int(self.expensive_checks_started) != 0:
            raise ProposalValidationError(
                "fail-fast evidence requires closed expensive dispatch"
            )
        object.__setattr__(self, "expensive_checks_started", 0)
        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("rejection evidence identity mismatch")
        object.__setattr__(self, "evidence_id", actual)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_REJECTION_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "proposal_id": self.proposal_id,
            "receipt_id": self.receipt_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "rejection_codes": self.rejection_codes,
            "expensive_node_ids": self.expensive_node_ids,
            "expensive_checks_started": self.expensive_checks_started,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalRejectionEvidence":
        schema = str(payload.get("schema") or PROPOSAL_REJECTION_EVIDENCE_SCHEMA)
        if schema != PROPOSAL_REJECTION_EVIDENCE_SCHEMA:
            raise ProposalValidationError(
                f"unsupported rejection evidence schema: {schema}"
            )
        return cls(
            requirement_id=str(payload.get("requirement_id") or ""),
            proposal_id=str(payload.get("proposal_id") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            rejection_codes=tuple(payload.get("rejection_codes") or ()),
            expensive_node_ids=tuple(payload.get("expensive_node_ids") or ()),
            expensive_checks_started=payload.get("expensive_checks_started", -1),
            evidence_id=str(payload.get("evidence_id") or ""),
        )


@dataclass(frozen=True)
class ProposalValidationReceipt:
    proposal_id: str
    policy_id: str
    repository_tree_id: str
    objective_id: str
    diff_digest: str
    allowed_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    accepted: bool
    findings: tuple[ProposalValidationFinding, ...]
    gate_trace: tuple[ProposalGate, ...]
    expensive_node_ids: tuple[str, ...] = ()
    expensive_checks_started: int = 0
    rejection_evidence: ProposalRejectionEvidence | None = None
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "policy_id",
            "repository_tree_id",
            "objective_id",
            "diff_digest",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
            if not getattr(self, name):
                raise ProposalValidationError(f"{name} is required")
        object.__setattr__(self, "allowed_paths", _strings(self.allowed_paths))
        object.__setattr__(self, "changed_paths", _strings(self.changed_paths))
        findings = tuple(
            item
            if isinstance(item, ProposalValidationFinding)
            else ProposalValidationFinding.from_dict(item)
            for item in self.findings
        )
        object.__setattr__(self, "findings", findings)
        trace = tuple(ProposalGate(item) for item in self.gate_trace)
        expected_prefix = ORDERED_PROPOSAL_GATES[: len(trace)]
        if not trace or trace != expected_prefix:
            raise ProposalValidationError("proposal gate trace is incomplete or unordered")
        object.__setattr__(self, "gate_trace", trace)
        if bool(self.accepted) == bool(findings):
            raise ProposalValidationError(
                "accepted proposals have no findings; rejected proposals require findings"
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "expensive_node_ids", _strings(self.expensive_node_ids))
        if isinstance(self.expensive_checks_started, bool) or int(
            self.expensive_checks_started
        ) < 0:
            raise ProposalValidationError(
                "expensive_checks_started must be a non-negative integer"
            )
        object.__setattr__(
            self, "expensive_checks_started", int(self.expensive_checks_started)
        )
        evidence = self.rejection_evidence
        if evidence is not None and not isinstance(evidence, ProposalRejectionEvidence):
            evidence = ProposalRejectionEvidence.from_dict(evidence)
        object.__setattr__(self, "rejection_evidence", None)
        claimed = str(self.receipt_id or "").strip()
        object.__setattr__(self, "receipt_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal receipt identity mismatch")
        object.__setattr__(self, "receipt_id", actual)
        if evidence is not None:
            if (
                self.accepted
                or evidence.receipt_id != actual
                or evidence.proposal_id != self.proposal_id
                or evidence.repository_tree_id != self.repository_tree_id
                or evidence.objective_id != self.objective_id
                or evidence.policy_id != self.policy_id
                or evidence.expensive_node_ids != self.expensive_node_ids
                or evidence.expensive_checks_started != self.expensive_checks_started
                or not set(evidence.rejection_codes).issubset(
                    finding.code.value for finding in findings
                )
            ):
                raise ProposalValidationError(
                    "rejection evidence is detached from proposal receipt"
                )
            object.__setattr__(self, "rejection_evidence", evidence)

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(finding.code.value for finding in self.findings)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (
            self.rejection_evidence.proved_requirement_ids
            if self.rejection_evidence is not None
            else ()
        )

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        """Proposal admission never proves the resulting implementation."""

        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_RECEIPT_SCHEMA,
            "proposal_id": self.proposal_id,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "diff_digest": self.diff_digest,
            "allowed_paths": self.allowed_paths,
            "changed_paths": self.changed_paths,
            "accepted": self.accepted,
            "findings": [finding.to_dict() for finding in self.findings],
            "gate_trace": [gate.value for gate in self.gate_trace],
            "expensive_node_ids": self.expensive_node_ids,
            "expensive_checks_started": self.expensive_checks_started,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "rejection_evidence": (
                self.rejection_evidence.to_dict()
                if self.rejection_evidence is not None
                else None
            ),
            "proved_requirement_ids": self.proved_requirement_ids,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationReceipt":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_RECEIPT_SCHEMA)
        if schema != PROPOSAL_VALIDATION_RECEIPT_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal receipt schema: {schema}")
        for field_name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
        ):
            if payload.get(field_name) not in (None, False):
                raise ProposalValidationError(
                    f"proposal receipt cannot claim {field_name}"
                )
        receipt = cls(
            proposal_id=str(payload.get("proposal_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            diff_digest=str(payload.get("diff_digest") or ""),
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            accepted=payload.get("accepted", False),
            findings=tuple(
                ProposalValidationFinding.from_dict(item)
                for item in payload.get("findings") or ()
            ),
            gate_trace=tuple(payload.get("gate_trace") or ()),
            expensive_node_ids=tuple(payload.get("expensive_node_ids") or ()),
            expensive_checks_started=payload.get("expensive_checks_started", 0),
            receipt_id=str(payload.get("receipt_id") or ""),
        )
        evidence_payload = payload.get("rejection_evidence")
        if evidence_payload:
            evidence = ProposalRejectionEvidence.from_dict(evidence_payload)
            receipt = cls(
                proposal_id=receipt.proposal_id,
                policy_id=receipt.policy_id,
                repository_tree_id=receipt.repository_tree_id,
                objective_id=receipt.objective_id,
                diff_digest=receipt.diff_digest,
                allowed_paths=receipt.allowed_paths,
                changed_paths=receipt.changed_paths,
                accepted=receipt.accepted,
                findings=receipt.findings,
                gate_trace=receipt.gate_trace,
                expensive_node_ids=receipt.expensive_node_ids,
                expensive_checks_started=receipt.expensive_checks_started,
                rejection_evidence=evidence,
                receipt_id=receipt.receipt_id,
            )
        claimed_requirements = tuple(payload.get("proved_requirement_ids") or ())
        if claimed_requirements and claimed_requirements != receipt.proved_requirement_ids:
            raise ProposalValidationError("proposal requirement claims mismatch")
        return receipt

    def with_dispatch_outcome(
        self,
        *,
        expensive_node_ids: Iterable[str],
        expensive_checks_started: int,
    ) -> "ProposalValidationReceipt":
        """Bind the scheduler-owned dispatch outcome and derive evidence."""

        # Dispatch closure is the evidence needed for a rejected proposal.
        # Accepted proposal identity stays stable when handed to downstream
        # semantic and completion gates.
        if self.accepted:
            return self
        node_ids = _strings(expensive_node_ids)
        started = int(expensive_checks_started)
        base = ProposalValidationReceipt(
            proposal_id=self.proposal_id,
            policy_id=self.policy_id,
            repository_tree_id=self.repository_tree_id,
            objective_id=self.objective_id,
            diff_digest=self.diff_digest,
            allowed_paths=self.allowed_paths,
            changed_paths=self.changed_paths,
            accepted=self.accepted,
            findings=self.findings,
            gate_trace=self.gate_trace,
            expensive_node_ids=node_ids,
            expensive_checks_started=started,
        )
        qualifying = (
            not base.accepted
            and started == 0
            and set(base.rejection_codes).intersection(
                code.value for code in QUALIFYING_FAIL_FAST_CODES
            )
        )
        if not qualifying:
            return base
        evidence = ProposalRejectionEvidence(
            requirement_id=NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
            proposal_id=base.proposal_id,
            receipt_id=base.receipt_id,
            repository_tree_id=base.repository_tree_id,
            objective_id=base.objective_id,
            policy_id=base.policy_id,
            rejection_codes=base.rejection_codes,
            expensive_node_ids=base.expensive_node_ids,
            expensive_checks_started=0,
        )
        return ProposalValidationReceipt(
            proposal_id=base.proposal_id,
            policy_id=base.policy_id,
            repository_tree_id=base.repository_tree_id,
            objective_id=base.objective_id,
            diff_digest=base.diff_digest,
            allowed_paths=base.allowed_paths,
            changed_paths=base.changed_paths,
            accepted=base.accepted,
            findings=base.findings,
            gate_trace=base.gate_trace,
            expensive_node_ids=base.expensive_node_ids,
            expensive_checks_started=0,
            rejection_evidence=evidence,
            receipt_id=base.receipt_id,
        )


@dataclass(frozen=True)
class ProposalValidationResult:
    proposal: ImplementationProposal
    policy: ProposalValidationPolicy
    receipt: ProposalValidationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, ImplementationProposal):
            object.__setattr__(
                self, "proposal", ImplementationProposal.from_dict(self.proposal)
            )
        if not isinstance(self.policy, ProposalValidationPolicy):
            object.__setattr__(
                self, "policy", ProposalValidationPolicy.from_dict(self.policy)
            )
        if not isinstance(self.receipt, ProposalValidationReceipt):
            object.__setattr__(
                self, "receipt", ProposalValidationReceipt.from_dict(self.receipt)
            )
        if (
            self.receipt.proposal_id != self.proposal.proposal_id
            or self.receipt.policy_id != self.policy.policy_id
            or self.receipt.repository_tree_id != self.proposal.repository_tree_id
            or self.receipt.objective_id != self.proposal.objective_id
            or self.receipt.diff_digest != self.proposal.diff_digest
            or self.receipt.allowed_paths != self.policy.allowed_paths
            or self.receipt.changed_paths != self.proposal.changed_paths
        ):
            raise ProposalValidationError("proposal result binding mismatch")

    @property
    def accepted(self) -> bool:
        return self.receipt.accepted

    @property
    def findings(self) -> tuple[ProposalValidationFinding, ...]:
        return self.receipt.findings

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "policy": self.policy.to_dict(),
            "receipt": self.receipt.to_dict(),
            "accepted": self.accepted,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationResult":
        for field_name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
        ):
            if payload.get(field_name) not in (None, False):
                raise ProposalValidationError(
                    f"proposal result cannot claim {field_name}"
                )
        result = cls(
            proposal=ImplementationProposal.from_dict(payload.get("proposal") or {}),
            policy=ProposalValidationPolicy.from_dict(payload.get("policy") or {}),
            receipt=ProposalValidationReceipt.from_dict(payload.get("receipt") or {}),
        )
        if "accepted" in payload and bool(payload["accepted"]) != result.accepted:
            raise ProposalValidationError("proposal result verdict mismatch")
        return result

    def with_dispatch_outcome(
        self,
        *,
        expensive_node_ids: Iterable[str],
        expensive_checks_started: int,
    ) -> "ProposalValidationResult":
        return ProposalValidationResult(
            proposal=self.proposal,
            policy=self.policy,
            receipt=self.receipt.with_dispatch_outcome(
                expensive_node_ids=expensive_node_ids,
                expensive_checks_started=expensive_checks_started,
            ),
        )


class ProposalValidator:
    """Deterministic evaluator for the strict proposal envelope."""

    def __init__(self, policy: ProposalValidationPolicy) -> None:
        self.policy = policy

    def validate(
        self, proposal: ImplementationProposal | Mapping[str, Any]
    ) -> ProposalValidationResult:
        if not isinstance(proposal, ImplementationProposal):
            proposal = ImplementationProposal.from_dict(proposal)
        policy = self.policy
        findings: list[ProposalValidationFinding] = []

        def add(
            code: ProposalFindingCode,
            gate: ProposalGate,
            message: str,
            path: str = "",
        ) -> None:
            if len(findings) < policy.max_findings:
                findings.append(ProposalValidationFinding(code, gate, message, path))

        # Authority is exact and non-compensable.
        expected = (
            ("task_id", policy.expected_task_id),
            ("accepted_plan_id", policy.expected_plan_id),
            ("repository_id", policy.expected_repository_id),
            ("repository_tree_id", policy.expected_repository_tree_id),
            ("objective_id", policy.expected_objective_id),
        )
        for name, required in expected:
            if required and getattr(proposal, name) != required:
                add(
                    ProposalFindingCode.STALE_BASELINE
                    if name == "repository_tree_id"
                    else ProposalFindingCode.AUTHORITY_MISMATCH,
                    ProposalGate.AUTHORITY,
                    f"{name} does not match the frozen proposal authority",
                )

        entries = proposal.candidate_diff
        if len(entries) > policy.max_diff_entries:
            add(
                ProposalFindingCode.PATCH_TOO_LARGE,
                ProposalGate.PATCH,
                "candidate diff exceeds the entry bound",
            )
        patch_bytes = sum(
            len((entry.before_source or "").encode("utf-8", errors="surrogatepass"))
            + len((entry.after_source or "").encode("utf-8", errors="surrogatepass"))
            for entry in entries
        )
        if patch_bytes > policy.max_patch_bytes:
            add(
                ProposalFindingCode.PATCH_TOO_LARGE,
                ProposalGate.PATCH,
                "candidate diff exceeds the byte bound",
            )
        if not entries:
            add(
                ProposalFindingCode.EMPTY_PATCH,
                ProposalGate.PATCH,
                "candidate diff contains no file changes",
            )
        elif not proposal.effective_entries:
            add(
                ProposalFindingCode.NO_SEMANTIC_CHANGE,
                ProposalGate.PATCH,
                "candidate diff has no observable content or path change",
            )

        actual_paths = proposal.changed_paths
        if policy.require_declared_paths and proposal.declared_paths != actual_paths:
            add(
                ProposalFindingCode.DECLARED_PATH_MISMATCH,
                ProposalGate.PATH,
                "declared paths do not exactly match the normalized candidate diff",
            )
        for entry in entries:
            for path in (entry.old_path, entry.new_path):
                if not path:
                    continue
                if path.startswith("/") or path == ".git" or path.startswith(".git/"):
                    add(
                        ProposalFindingCode.UNSAFE_PATH,
                        ProposalGate.PATH,
                        "candidate path crosses a protected repository boundary",
                        path,
                    )
                    continue
                if any(_path_matches(path, denied) for denied in policy.forbidden_paths):
                    add(
                        ProposalFindingCode.UNSAFE_PATH,
                        ProposalGate.PATH,
                        "candidate path is forbidden by repository policy",
                        path,
                    )
                if not any(_path_matches(path, allowed) for allowed in policy.allowed_paths):
                    add(
                        ProposalFindingCode.PATH_OUTSIDE_SCOPE,
                        ProposalGate.PATH,
                        "candidate path is outside the task-owned scope",
                        path,
                    )
            if entry.binary and not policy.allow_binary:
                add(
                    ProposalFindingCode.BINARY_CHANGE_FORBIDDEN,
                    ProposalGate.PATH,
                    "binary changes require explicit policy authority",
                    entry.path,
                )
            if entry.generated is True and not policy.allow_generated:
                add(
                    ProposalFindingCode.GENERATED_CHANGE_FORBIDDEN,
                    ProposalGate.PATH,
                    "generated-file changes require explicit policy authority",
                    entry.path,
                )
            if (
                entry.change_kind is DiffChangeKind.DELETE
                and not policy.allow_test_deletion
                and (
                    entry.path.startswith(("test/", "tests/"))
                    or entry.path.rsplit("/", 1)[-1].startswith("test_")
                )
            ):
                add(
                    ProposalFindingCode.TEST_DELETION_FORBIDDEN,
                    ProposalGate.PATH,
                    "test deletion requires explicit task authority",
                    entry.path,
                )
            if (
                policy.require_python_syntax
                and entry.is_python
                and entry.change_kind is not DiffChangeKind.DELETE
                and entry.after_source is not None
            ):
                try:
                    ast.parse(entry.after_source, filename=entry.path)
                except (SyntaxError, ValueError) as exc:
                    add(
                        ProposalFindingCode.PYTHON_SYNTAX_ERROR,
                        ProposalGate.AST_INTERFACE,
                        f"candidate Python does not parse: {exc.msg if isinstance(exc, SyntaxError) else exc}",
                        entry.path,
                    )

        # Gate trace is complete even after a failure because all proposal
        # checks are bounded and cheap.  This yields better repair diagnostics
        # without admitting any expensive descendant.
        findings.sort(
            key=lambda item: (
                ORDERED_PROPOSAL_GATES.index(item.gate),
                item.path,
                item.code.value,
                item.message,
            )
        )
        receipt = ProposalValidationReceipt(
            proposal_id=proposal.proposal_id,
            policy_id=policy.policy_id,
            repository_tree_id=proposal.repository_tree_id,
            objective_id=proposal.objective_id,
            diff_digest=proposal.diff_digest,
            allowed_paths=policy.allowed_paths,
            changed_paths=proposal.changed_paths,
            accepted=not findings,
            findings=tuple(findings),
            gate_trace=ORDERED_PROPOSAL_GATES,
        )
        return ProposalValidationResult(proposal, policy, receipt)


def validate_implementation_proposal(
    proposal: ImplementationProposal | Mapping[str, Any],
    *,
    policy: ProposalValidationPolicy | Mapping[str, Any],
) -> ProposalValidationResult:
    """Validate one proposal against a frozen strict policy."""

    if not isinstance(policy, ProposalValidationPolicy):
        policy = ProposalValidationPolicy.from_dict(policy)
    return ProposalValidator(policy).validate(proposal)


validate_proposal = validate_implementation_proposal
StrictProposalValidator = ProposalValidator


__all__ = [
    "ImplementationProposal",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID",
    "ORDERED_PROPOSAL_GATES",
    "PROPOSAL_REJECTION_EVIDENCE_SCHEMA",
    "PROPOSAL_VALIDATION_POLICY_SCHEMA",
    "PROPOSAL_VALIDATION_RECEIPT_SCHEMA",
    "PROPOSAL_VALIDATION_REQUEST_SCHEMA",
    "ProposalFindingCode",
    "ProposalGate",
    "ProposalRejectionEvidence",
    "ProposalValidationError",
    "ProposalValidationFinding",
    "ProposalValidationPolicy",
    "ProposalValidationReceipt",
    "ProposalValidationRequest",
    "ProposalValidationResult",
    "ProposalValidator",
    "StrictProposalValidator",
    "validate_implementation_proposal",
    "validate_proposal",
]

"""Isolated mutant rescan and semantic admission guardrails (AAE-024).

Interface surface:

* ``admit_mutation`` — validate a caller-supplied owned disposable worktree,
  rescan declared changes, block verifier/policy/key/oracle edits, parse and
  structurally validate, reject trivial invalidity, estimate equivalence,
  predict detection, and commit immutable mutant identity.

This module **does not** create or destroy worktrees. Worktree lifecycle
ownership remains with ``WorktreeLifecycleStore`` / AAE-041. Optional lineage
recording reuses ``MutationLedger`` without treating it as the campaign engine.

Cold import is side-effect free: no Git, ledger, process, network, or
filesystem operations run at import time.
"""

from __future__ import annotations

import ast
import re
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.analysis.mutation_ledger import (
    MutationContext,
    MutationFileSpec,
    MutationLedger,
    MutationRecordResult,
    content_digest_of,
    open_mutation_ledger,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorkspaceLifecycleRecord,
    WorkspaceLifecycleState,
    WorktreeLifecycleStore,
    normalize_workspace_path,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceArtifactHeader,
    AssuranceBaseError,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    DetectionAssuranceManifest,
    DetectionPredictionError,
    predict_detection_set,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    EquivalenceAssessmentStatus,
    EquivalenceMethod,
    ExpectedDetectionSet,
    MutationEquivalenceAssessment,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    MutationCandidate,
    MutationContractError,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

ADMIT_MUTATION_INTERFACE: Final[str] = "admit_mutation@1"
MUTATION_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-admission@1"
)
MUTATION_ADMISSION_INTERFACE: Final[str] = "MutationAdmissionResult@1"
CHANGED_PATH_SCAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-changed-path-scan@1"
)
STRUCTURAL_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-structural-validation@1"
)

GENERATOR_ID: Final[str] = "mutation_admission"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_CHANGED_PATHS: Final[int] = 256
MAX_FILE_BYTES: Final[int] = 1_000_000
MAX_PATCH_BYTES: Final[int] = 2_000_000
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_TEXT_CHARS: Final[int] = 16_384
GIT_TIMEOUT_SECONDS: Final[int] = 60

_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Authority surfaces that mutants must not edit unless this is an explicit
# verifier/policy fixture campaign (allow_authority_fixture=True).
_BLOCKED_AUTHORITY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "verification",
        "verifier",
        "verifiers",
        "policy",
        "policies",
        "oracle",
        "oracles",
        "trusted_keys",
        "secrets",
        "credentials",
        "private_keys",
        "signing_keys",
    }
)
_BLOCKED_AUTHORITY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)
_BLOCKED_AUTHORITY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "authorized_keys",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "policy.json",
        "policy.yaml",
        "policy.yml",
        "oracle.json",
        "benchmark_oracle.json",
        "golden.json",
    }
)
_BLOCKED_AUTHORITY_PREFIXES: Final[tuple[str, ...]] = (
    ".ssh/",
    ".aws/",
    ".gnupg/",
    "config/",
    "secrets/",
    "credentials/",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/validation/",
)

# Paths never admitted as mutation targets (always forbidden).
_ALWAYS_FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    ".git/",
    ".git",
    ".agent_supervisor/",
    "data/agent_supervisor/",
    "__pycache__/",
    ".env",
)

_SUPPORTED_PARSE_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi"})


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class AdmissionError(ValueError):
    """Raised when admission inputs are malformed before a sealed result."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "malformed_input")


class AdmissionDisposition(str, Enum):
    """Closed terminal disposition for one admission attempt."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    INVALID = "invalid"
    EQUIVALENT = "equivalent"


class AdmissionReasonCode(str, Enum):
    """Closed reason codes for admission success or fail-closed rejection."""

    OK = "ok"
    WORKTREE_MISSING = "worktree_missing"
    WORKTREE_NOT_OWNED = "worktree_not_owned"
    WORKTREE_TERMINAL = "worktree_terminal"
    WORKTREE_OWNERSHIP_MISMATCH = "worktree_ownership_mismatch"
    WORKTREE_NOT_DISPOSABLE = "worktree_not_disposable"
    WORKTREE_PRODUCTION_ROOT = "worktree_production_root"
    LIFECYCLE_STORE_REQUIRED = "lifecycle_store_required"
    UNDECLARED_PATH_CHANGE = "undeclared_path_change"
    UNDECLARED_SYMBOL_CHANGE = "undeclared_symbol_change"
    NO_DECLARED_CHANGES = "no_declared_changes"
    EMPTY_DIFF = "empty_diff"
    AUTHORITY_PATH_BLOCKED = "authority_path_blocked"
    FORBIDDEN_PATH = "forbidden_path"
    PARSE_FAILURE = "parse_failure"
    STRUCTURAL_INVALID = "structural_invalid"
    TRIVIAL_INVALIDITY = "trivial_invalidity"
    NO_OP_MUTATION = "no_op_mutation"
    FILE_TOO_LARGE = "file_too_large"
    TOO_MANY_PATHS = "too_many_paths"
    PATCH_TOO_LARGE = "patch_too_large"
    CANDIDATE_INVALID = "candidate_invalid"
    MANIFEST_REQUIRED = "manifest_required"
    DETECTION_PREDICTION_FAILED = "detection_prediction_failed"
    BASE_COMMIT_MISMATCH = "base_commit_mismatch"
    GIT_SCAN_FAILED = "git_scan_failed"
    EQUIVALENCE_ESTIMATE = "equivalence_estimate"
    LEDGER_RECORD_FAILED = "ledger_record_failed"


class StructuralOutcome(str, Enum):
    """Per-path structural validation outcome."""

    OK = "ok"
    PARSE_FAILED = "parse_failed"
    STRUCTURAL_INVALID = "structural_invalid"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    MISSING = "missing"
    TOO_LARGE = "too_large"


class EquivalenceEstimate(str, Enum):
    """Bounded pre-AAE-025 equivalence estimate (never invents proof)."""

    NOT_EQUIVALENT = "not_equivalent"
    PROBABLY_EQUIVALENT = "probably_equivalent"
    EQUIVALENT = "equivalent"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AdmissionError(f"{name} must be a string", reason_code="malformed_input")
    text = unicodedata.normalize("NFC", value)
    if not empty and not text.strip():
        raise AdmissionError(f"{name} must not be empty", reason_code="malformed_input")
    if len(text) > MAX_TEXT_CHARS:
        raise AdmissionError(f"{name} exceeds maximum length", reason_code="malformed_input")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True) or None


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise AdmissionError(f"{name} must be a boolean", reason_code="malformed_input")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise AdmissionError(
            f"{name} must be a non-negative integer", reason_code="malformed_input"
        )
    return value


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.fullmatch(text):
        raise AdmissionError(f"{name} is not a valid token", reason_code="malformed_input")
    return text


def _normalize_repo_path(value: Any, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError(f"{name} must be a non-empty path", reason_code="malformed_input")
    raw = value.replace("\\", "/").strip()
    if raw.startswith("/") or raw.startswith("~"):
        raise AdmissionError(f"{name} must be repository-relative", reason_code="malformed_input")
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise AdmissionError(f"{name} must not contain . or .. segments", reason_code="malformed_input")
    if len(raw.encode("utf-8")) > 4_096:
        raise AdmissionError(f"{name} exceeds path byte limit", reason_code="malformed_input")
    return str(PurePosixPath(raw))


def _path_under_prefix(path: str, prefix: str) -> bool:
    path_n = path.replace("\\", "/").strip("/")
    pref = prefix.replace("\\", "/").strip()
    if not pref:
        return False
    if pref.endswith("/"):
        return path_n == pref[:-1] or path_n.startswith(pref)
    return path_n == pref or path_n.startswith(pref + "/")


def _path_under_any(path: str, prefixes: Sequence[str]) -> bool:
    return any(_path_under_prefix(path, prefix) for prefix in prefixes)


def blocked_authority_path(path: str) -> bool:
    """Return True when *path* is a verifier/policy/key/oracle authority surface."""

    try:
        normalized = _normalize_repo_path(path, name="path")
    except AdmissionError:
        return True
    lower = normalized.lower()
    if _path_under_any(lower, _BLOCKED_AUTHORITY_PREFIXES):
        return True
    if PurePosixPath(lower).name in _BLOCKED_AUTHORITY_BASENAMES:
        return True
    if any(lower.endswith(suffix) for suffix in _BLOCKED_AUTHORITY_SUFFIXES):
        return True
    for segment in PurePosixPath(lower).parts:
        if segment in _BLOCKED_AUTHORITY_SEGMENTS:
            return True
    return False


def _always_forbidden(path: str) -> bool:
    try:
        normalized = _normalize_repo_path(path, name="path")
    except AdmissionError:
        return True
    return _path_under_any(normalized, _ALWAYS_FORBIDDEN_PREFIXES)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise AdmissionError(f"{name} must be a mapping", reason_code="malformed_input")
    # Reject private / model-authority keys at the admission boundary.
    for key in value:
        key_text = str(key)
        lowered = key_text.lower()
        if any(
            marker in lowered
            for marker in (
                "private_key",
                "api_key",
                "secret",
                "password",
                "model_authority",
                "llm_authority",
                "self_authorized",
                "promotion_authority",
            )
        ):
            raise AdmissionError(
                f"{name} contains forbidden authority key {key_text!r}",
                reason_code="malformed_input",
            )
    return MappingProxyType(dict(value))


def _normalize_candidate(value: Any) -> MutationCandidate:
    if isinstance(value, MutationCandidate):
        return value
    if isinstance(value, Mapping):
        try:
            return MutationCandidate.from_dict(value)
        except (MutationContractError, AssuranceBaseError, TypeError, ValueError) as exc:
            raise AdmissionError(
                f"candidate is invalid: {exc}",
                reason_code=AdmissionReasonCode.CANDIDATE_INVALID.value,
            ) from exc
    raise AdmissionError(
        "candidate must be MutationCandidate or mapping",
        reason_code=AdmissionReasonCode.CANDIDATE_INVALID.value,
    )


def _normalize_manifest(value: Any) -> DetectionAssuranceManifest:
    if isinstance(value, DetectionAssuranceManifest):
        return value
    if isinstance(value, Mapping):
        try:
            return DetectionAssuranceManifest.normalize(value)
        except (DetectionPredictionError, AssuranceBaseError, TypeError, ValueError) as exc:
            raise AdmissionError(
                f"assurance_manifest is invalid: {exc}",
                reason_code=AdmissionReasonCode.MANIFEST_REQUIRED.value,
            ) from exc
    raise AdmissionError(
        "assurance_manifest must be DetectionAssuranceManifest or mapping",
        reason_code=AdmissionReasonCode.MANIFEST_REQUIRED.value,
    )


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise AdmissionError(
            "git executable is required for mutant rescan",
            reason_code=AdmissionReasonCode.GIT_SCAN_FAILED.value,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AdmissionError(
            f"git timed out: {' '.join(command)}",
            reason_code=AdmissionReasonCode.GIT_SCAN_FAILED.value,
        ) from exc


def _git_ok(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> str:
    completed = _run_git(args, cwd=cwd, timeout=timeout)
    if completed.returncode != 0:
        raise AdmissionError(
            _clip(
                f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
            ),
            reason_code=AdmissionReasonCode.GIT_SCAN_FAILED.value,
        )
    return completed.stdout


def _rev_parse(repo: Path, rev: str) -> str:
    oid = _git_ok(["rev-parse", "--verify", rev], cwd=repo).strip()
    if not _OID_RE.fullmatch(oid):
        raise AdmissionError(
            f"invalid git object id for {rev!r}",
            reason_code=AdmissionReasonCode.GIT_SCAN_FAILED.value,
        )
    return oid


def _is_git_worktree(path: Path) -> bool:
    if not path.is_dir():
        return False
    completed = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path)
    return completed.returncode == 0 and completed.stdout.strip() == "true"


# ---------------------------------------------------------------------------
# Scan / structural models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChangedPathRecord:
    """One rescanned path relative to the disposable worktree root."""

    path: str
    change_kind: str  # modified | added | deleted | renamed | untracked
    before_digest: str
    after_digest: str
    before_bytes: int
    after_bytes: int
    structural_outcome: str
    symbols_touched: tuple[str, ...] = ()
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "change_kind": self.change_kind,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "before_bytes": int(self.before_bytes),
            "after_bytes": int(self.after_bytes),
            "structural_outcome": self.structural_outcome,
            "symbols_touched": list(self.symbols_touched),
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class PathScanResult:
    """Bounded rescan of declared-versus-observed worktree changes."""

    base_commit: str
    head_commit: str
    changed_paths: tuple[ChangedPathRecord, ...]
    undeclared_paths: tuple[str, ...]
    declared_paths: tuple[str, ...]
    total_after_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANGED_PATH_SCAN_SCHEMA,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "changed_paths": [item.to_dict() for item in self.changed_paths],
            "undeclared_paths": list(self.undeclared_paths),
            "declared_paths": list(self.declared_paths),
            "total_after_bytes": int(self.total_after_bytes),
        }


@dataclass(frozen=True, slots=True)
class MutationAdmissionResult:
    """Sealed outcome of ``admit_mutation@1``.

    Successful admissions bind candidate identity, rescanned changes,
    equivalence estimate, expected detection set, and a content-addressed
    admission identity. Failures are sealed with disposition and reason codes
    and never invent kill authority.
    """

    disposition: str
    reason_codes: tuple[str, ...]
    candidate_id: str
    candidate_cid: str
    worktree_path: str
    lease_id: str
    fence: int
    admitted: bool
    identity_cid: str
    scan: Mapping[str, Any] | None = None
    equivalence_status: str = EquivalenceEstimate.UNKNOWN.value
    equivalence_methods: tuple[str, ...] = ()
    equivalence_assessment_cid: str | None = None
    detection_set_cid: str | None = None
    detection_set_id: str | None = None
    predicted_detector_ids: tuple[str, ...] = ()
    ledger_mutation_id: str | None = None
    lifecycle_record_id: str | None = None
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = MUTATION_ADMISSION_SCHEMA
    interface_id: str = MUTATION_ADMISSION_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            AdmissionDisposition(self.disposition).value,
        )
        codes = tuple(
            AdmissionReasonCode(code).value
            if code in {item.value for item in AdmissionReasonCode}
            else str(code)
            for code in self.reason_codes
        )
        if not codes:
            raise AdmissionError("reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "candidate_id", _token(self.candidate_id, "candidate_id"))
        try:
            object.__setattr__(
                self, "candidate_cid", validate_cid(self.candidate_cid)
            )
        except Exception as exc:  # noqa: BLE001 — fail closed on bad CID
            raise AdmissionError(
                f"candidate_cid is invalid: {exc}",
                reason_code="malformed_input",
            ) from exc
        object.__setattr__(
            self, "worktree_path", _text(self.worktree_path, "worktree_path")
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id", empty=True))
        object.__setattr__(self, "fence", _nonneg_int(self.fence, "fence"))
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        if self.admitted and self.disposition != AdmissionDisposition.ADMITTED.value:
            # Equivalent disposition may carry identity but is not execution-admitted.
            if self.disposition != AdmissionDisposition.EQUIVALENT.value:
                raise AdmissionError(
                    "admitted=True requires admitted or equivalent disposition"
                )
        if self.disposition == AdmissionDisposition.ADMITTED.value and not self.admitted:
            raise AdmissionError("disposition admitted requires admitted=True")
        object.__setattr__(
            self, "identity_cid", _text(self.identity_cid, "identity_cid")
        )
        object.__setattr__(
            self,
            "equivalence_status",
            EquivalenceEstimate(self.equivalence_status).value,
        )
        methods = tuple(str(item) for item in self.equivalence_methods)
        object.__setattr__(self, "equivalence_methods", methods)
        object.__setattr__(
            self,
            "predicted_detector_ids",
            tuple(str(item) for item in self.predicted_detector_ids),
        )
        object.__setattr__(self, "diagnostic", _clip(str(self.diagnostic or "")))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.scan is not None and not isinstance(self.scan, Mapping):
            raise AdmissionError("scan must be a mapping or None")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "worktree_path": self.worktree_path,
            "lease_id": self.lease_id,
            "fence": int(self.fence),
            "admitted": bool(self.admitted),
            "equivalence_status": self.equivalence_status,
            "equivalence_methods": list(self.equivalence_methods),
            "equivalence_assessment_cid": self.equivalence_assessment_cid,
            "detection_set_cid": self.detection_set_cid,
            "detection_set_id": self.detection_set_id,
            "predicted_detector_ids": list(self.predicted_detector_ids),
            "ledger_mutation_id": self.ledger_mutation_id,
            "lifecycle_record_id": self.lifecycle_record_id,
            "scan": dict(self.scan) if self.scan is not None else None,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    @property
    def admission_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["identity_cid"] = self.identity_cid
        payload["admission_cid"] = self.admission_cid
        return payload


def admission_reason_codes() -> tuple[str, ...]:
    """Return the closed admission reason-code vocabulary."""

    return tuple(item.value for item in AdmissionReasonCode)


def admission_dispositions() -> tuple[str, ...]:
    """Return the closed admission disposition vocabulary."""

    return tuple(item.value for item in AdmissionDisposition)


# ---------------------------------------------------------------------------
# Worktree ownership validation (no create/destroy)
# ---------------------------------------------------------------------------


def _validate_owned_disposable_worktree(
    *,
    worktree_path: Path,
    repo_root: Path | None,
    lifecycle_store: WorktreeLifecycleStore | None,
    lease_id: str,
    fence: int,
    require_lifecycle: bool,
) -> tuple[WorkspaceLifecycleRecord | None, tuple[str, ...], str]:
    """Validate caller-supplied owned disposable worktree. Never mutates it."""

    if not worktree_path.exists() or not worktree_path.is_dir():
        return None, (AdmissionReasonCode.WORKTREE_MISSING.value,), "worktree path missing"

    if not _is_git_worktree(worktree_path):
        return (
            None,
            (AdmissionReasonCode.WORKTREE_MISSING.value,),
            "path is not a git worktree",
        )

    resolved_wt = worktree_path.resolve()
    if repo_root is not None:
        resolved_root = repo_root.resolve()
        if resolved_wt == resolved_root:
            return (
                None,
                (AdmissionReasonCode.WORKTREE_PRODUCTION_ROOT.value,),
                "mutations must not target the production repository root",
            )

    if lifecycle_store is None:
        if require_lifecycle:
            return (
                None,
                (AdmissionReasonCode.LIFECYCLE_STORE_REQUIRED.value,),
                "WorktreeLifecycleStore is required to prove disposable ownership",
            )
        return None, (), ""

    record = lifecycle_store.load_workspace(worktree_path)
    if record is None:
        return (
            None,
            (AdmissionReasonCode.WORKTREE_NOT_OWNED.value,),
            "no lifecycle ownership record for worktree",
        )
    if record.is_terminal or record.state is WorkspaceLifecycleState.TERMINAL:
        return (
            record,
            (AdmissionReasonCode.WORKTREE_TERMINAL.value,),
            "worktree lifecycle is terminal",
        )
    if str(record.lease_id) != str(lease_id) or int(record.fence) != int(fence):
        return (
            record,
            (AdmissionReasonCode.WORKTREE_OWNERSHIP_MISMATCH.value,),
            "caller lease/fence does not match lifecycle owner",
        )

    # Disposable ownership: record must point at this workspace and not the
    # production root when repo_root is known.
    try:
        record_path = Path(record.workspace_path).resolve()
    except OSError:
        record_path = Path(normalize_workspace_path(record.workspace_path))
    if record_path != resolved_wt and normalize_workspace_path(
        record.workspace_path
    ) != normalize_workspace_path(worktree_path):
        return (
            record,
            (AdmissionReasonCode.WORKTREE_NOT_OWNED.value,),
            "lifecycle workspace_path does not match supplied worktree",
        )
    if record.repo_root:
        try:
            record_root = Path(record.repo_root).resolve()
            if record_root == resolved_wt:
                return (
                    record,
                    (AdmissionReasonCode.WORKTREE_NOT_DISPOSABLE.value,),
                    "lifecycle repo_root equals worktree; not disposable",
                )
        except OSError:
            pass

    return record, (), ""


# ---------------------------------------------------------------------------
# Declared-change rescan
# ---------------------------------------------------------------------------


def _declared_paths_for_candidate(
    candidate: MutationCandidate,
    declared_paths: Sequence[str] | None,
) -> tuple[str, ...]:
    if declared_paths is not None:
        if isinstance(declared_paths, (str, bytes, bytearray)):
            raise AdmissionError(
                "declared_paths must be a sequence of paths",
                reason_code="malformed_input",
            )
        paths = tuple(
            dict.fromkeys(_normalize_repo_path(item, name="declared_paths") for item in declared_paths)
        )
        return paths
    scope = tuple(candidate.scope_paths or ())
    if not scope:
        return ()
    return tuple(
        dict.fromkeys(_normalize_repo_path(item, name="scope_paths") for item in scope)
    )


def _path_is_declared(path: str, declared: Sequence[str]) -> bool:
    if not declared:
        # Empty declared set means "no path authority" — fail closed later.
        return False
    if path in declared:
        return True
    return any(_path_under_prefix(path, prefix) for prefix in declared)


def _list_changed_paths(worktree: Path, *, base_commit: str) -> list[tuple[str, str]]:
    """Return (path, change_kind) pairs vs base, including untracked files."""

    # Committed + staged + unstaged tracked changes.
    name_status = _git_ok(
        ["diff", "--name-status", "--find-renames", f"{base_commit}"],
        cwd=worktree,
    )
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in name_status.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            path = parts[2]
            kind = "renamed"
        elif status.startswith("A"):
            path = parts[1]
            kind = "added"
        elif status.startswith("D"):
            path = parts[1]
            kind = "deleted"
        else:
            path = parts[-1]
            kind = "modified"
        path = path.replace("\\", "/").strip()
        if path and path not in seen:
            seen.add(path)
            results.append((path, kind))

    # Untracked files (applied mutant may leave worktree dirty without commit).
    untracked = _git_ok(
        ["ls-files", "--others", "--exclude-standard"],
        cwd=worktree,
    )
    for line in untracked.splitlines():
        path = line.replace("\\", "/").strip()
        if path and path not in seen:
            seen.add(path)
            results.append((path, "untracked"))

    # Also include staged-only via diff against HEAD if base == HEAD and dirty index.
    # Already covered by diff base_commit when base is HEAD^ or similar.

    results.sort(key=lambda item: item[0])
    return results


def _read_worktree_bytes(worktree: Path, rel_path: str) -> bytes | None:
    target = worktree / rel_path
    if not target.is_file():
        return None
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise AdmissionError(
            f"failed to read {rel_path}: {exc}",
            reason_code=AdmissionReasonCode.GIT_SCAN_FAILED.value,
        ) from exc
    return data


def _read_base_bytes(worktree: Path, *, base_commit: str, rel_path: str) -> bytes | None:
    completed = _run_git(
        ["show", f"{base_commit}:{rel_path}"],
        cwd=worktree,
    )
    if completed.returncode != 0:
        return None
    # git show returns text mode; re-run as binary for exact bytes.
    try:
        raw = subprocess.run(
            ["git", "show", f"{base_commit}:{rel_path}"],
            cwd=str(worktree),
            check=False,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return completed.stdout.encode("utf-8")
    if raw.returncode != 0:
        return None
    return raw.stdout


def _python_symbols(source: bytes) -> tuple[str, ...]:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        return ()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return tuple(sorted(names))


def _structural_validate_python(after: bytes | None) -> tuple[str, str, tuple[str, ...]]:
    if after is None:
        return StructuralOutcome.MISSING.value, "after content missing", ()
    if len(after) > MAX_FILE_BYTES:
        return StructuralOutcome.TOO_LARGE.value, "file exceeds byte budget", ()
    try:
        text = after.decode("utf-8")
    except UnicodeDecodeError:
        return StructuralOutcome.STRUCTURAL_INVALID.value, "not valid utf-8", ()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return (
            StructuralOutcome.PARSE_FAILED.value,
            _clip(f"syntax error: {exc}"),
            (),
        )
    # Basic structural check: module body must not be empty for non-deleted files
    # when the mutation claims a semantic change — emptiness is handled separately.
    symbols = _python_symbols(after)
    # Walk for obviously broken constructs already caught by parse.
    _ = tree
    return StructuralOutcome.OK.value, "", symbols


def _ast_normalized_dump(source: bytes | None) -> str | None:
    if source is None:
        return None
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _estimate_equivalence(
    *,
    candidate: MutationCandidate,
    records: Sequence[ChangedPathRecord],
    before_by_path: Mapping[str, bytes | None],
    after_by_path: Mapping[str, bytes | None],
) -> tuple[str, tuple[str, ...], str]:
    """Bounded equivalence estimate. Unknown never becomes equivalent."""

    methods: list[str] = []
    # No-op / byte-identical on all paths → equivalent (trivial).
    if records and all(
        item.before_digest == item.after_digest and item.change_kind not in {"added", "deleted", "untracked", "renamed"}
        for item in records
    ):
        methods.append(EquivalenceMethod.AST_COMPARISON.value)
        return EquivalenceEstimate.EQUIVALENT.value, tuple(methods), "byte-identical after rescan"

    # AST-normalized comparison for pure Python sets.
    python_paths = [
        item.path
        for item in records
        if PurePosixPath(item.path).suffix in _SUPPORTED_PARSE_SUFFIXES
    ]
    if python_paths and len(python_paths) == len(records):
        methods.append(EquivalenceMethod.AST_COMPARISON.value)
        all_ast_equal = True
        any_comparable = False
        for path in python_paths:
            before_dump = _ast_normalized_dump(before_by_path.get(path))
            after_dump = _ast_normalized_dump(after_by_path.get(path))
            if before_dump is None or after_dump is None:
                all_ast_equal = False
                break
            any_comparable = True
            if before_dump != after_dump:
                all_ast_equal = False
                break
        if any_comparable and all_ast_equal:
            methods.append(EquivalenceMethod.NORMALIZED_IR.value)
            return (
                EquivalenceEstimate.EQUIVALENT.value,
                tuple(dict.fromkeys(methods)),
                "AST-normalized form unchanged",
            )
        if any_comparable and not all_ast_equal:
            return (
                EquivalenceEstimate.NOT_EQUIVALENT.value,
                tuple(dict.fromkeys(methods)),
                "AST differs after mutation",
            )

    if candidate.likely_equivalent:
        methods.append(EquivalenceMethod.HUMAN_REVIEW.value)
        # Difficulty/likely flag alone never proves equivalence.
        return (
            EquivalenceEstimate.PROBABLY_EQUIVALENT.value,
            tuple(dict.fromkeys(methods)),
            "candidate.likely_equivalent requests human review; not auto-equivalent",
        )

    if not methods:
        methods.append(EquivalenceMethod.AST_COMPARISON.value)
    return (
        EquivalenceEstimate.UNKNOWN.value,
        tuple(dict.fromkeys(methods)),
        "bounded estimate inconclusive",
    )


def _rescan_declared_changes(
    *,
    worktree: Path,
    base_commit: str,
    declared_paths: Sequence[str],
    scope_symbol_ids: Sequence[str],
) -> tuple[PathScanResult | None, tuple[str, ...], str, dict[str, bytes | None], dict[str, bytes | None]]:
    try:
        head = _rev_parse(worktree, "HEAD")
        base = _rev_parse(worktree, base_commit)
    except AdmissionError as exc:
        return None, (exc.reason_code,), str(exc), {}, {}

    try:
        changed = _list_changed_paths(worktree, base_commit=base)
    except AdmissionError as exc:
        return None, (exc.reason_code,), str(exc), {}, {}

    if len(changed) > MAX_CHANGED_PATHS:
        return (
            None,
            (AdmissionReasonCode.TOO_MANY_PATHS.value,),
            f"changed path count exceeds {MAX_CHANGED_PATHS}",
            {},
            {},
        )

    records: list[ChangedPathRecord] = []
    undeclared: list[str] = []
    before_by_path: dict[str, bytes | None] = {}
    after_by_path: dict[str, bytes | None] = {}
    total_after = 0

    for rel_path, kind in changed:
        try:
            path = _normalize_repo_path(rel_path, name="changed_path")
        except AdmissionError:
            undeclared.append(rel_path)
            continue

        if _always_forbidden(path):
            undeclared.append(path)
            records.append(
                ChangedPathRecord(
                    path=path,
                    change_kind=kind,
                    before_digest="",
                    after_digest="",
                    before_bytes=0,
                    after_bytes=0,
                    structural_outcome=StructuralOutcome.STRUCTURAL_INVALID.value,
                    diagnostic="forbidden path",
                )
            )
            continue

        if not _path_is_declared(path, declared_paths):
            undeclared.append(path)

        before = None if kind in {"added", "untracked"} else _read_base_bytes(
            worktree, base_commit=base, rel_path=path
        )
        after = None if kind == "deleted" else _read_worktree_bytes(worktree, path)
        before_by_path[path] = before
        after_by_path[path] = after
        before_len = len(before or b"")
        after_len = len(after or b"")
        total_after += after_len
        if after_len > MAX_FILE_BYTES:
            records.append(
                ChangedPathRecord(
                    path=path,
                    change_kind=kind,
                    before_digest=content_digest_of(before) if before is not None else "",
                    after_digest=content_digest_of(after) if after is not None else "",
                    before_bytes=before_len,
                    after_bytes=after_len,
                    structural_outcome=StructuralOutcome.TOO_LARGE.value,
                    diagnostic="file exceeds single-file budget",
                )
            )
            continue

        symbols: tuple[str, ...] = ()
        if PurePosixPath(path).suffix in _SUPPORTED_PARSE_SUFFIXES and after is not None:
            outcome, diagnostic, symbols = _structural_validate_python(after)
        elif PurePosixPath(path).suffix in _SUPPORTED_PARSE_SUFFIXES and after is None:
            outcome, diagnostic, symbols = StructuralOutcome.OK.value, "", ()
        else:
            outcome, diagnostic, symbols = (
                StructuralOutcome.SKIPPED_UNSUPPORTED.value,
                "",
                (),
            )

        records.append(
            ChangedPathRecord(
                path=path,
                change_kind=kind,
                before_digest=content_digest_of(before) if before is not None else "",
                after_digest=content_digest_of(after) if after is not None else "",
                before_bytes=before_len,
                after_bytes=after_len,
                structural_outcome=outcome,
                symbols_touched=symbols,
                diagnostic=diagnostic,
            )
        )

    if total_after > MAX_PATCH_BYTES:
        return (
            None,
            (AdmissionReasonCode.PATCH_TOO_LARGE.value,),
            f"aggregate after bytes exceed {MAX_PATCH_BYTES}",
            before_by_path,
            after_by_path,
        )

    # Symbol guard: when scope symbols are declared and Python files changed,
    # require that at least one declared symbol is present among touched symbols
    # OR the change is a pure deletion of a path that previously held them.
    symbol_reason = ""
    if scope_symbol_ids and records and not undeclared:
        declared_syms = {str(item) for item in scope_symbol_ids}
        # Accept short names and dotted forms (mod.fn → fn).
        short = {item.split(".")[-1] for item in declared_syms}
        ok_symbol = False
        for item in records:
            if item.structural_outcome == StructuralOutcome.SKIPPED_UNSUPPORTED.value:
                ok_symbol = True
                break
            touched = set(item.symbols_touched)
            if touched & declared_syms or touched & short:
                ok_symbol = True
                break
            # Path-level declaration alone is insufficient when symbols are named;
            # still allow if file path basename matches module of symbol.
            for sym in declared_syms:
                module_hint = sym.split(".")[0] if "." in sym else ""
                if module_hint and PurePosixPath(item.path).stem == module_hint:
                    ok_symbol = True
                    break
            if ok_symbol:
                break
        if not ok_symbol and any(
            PurePosixPath(item.path).suffix in _SUPPORTED_PARSE_SUFFIXES
            for item in records
        ):
            symbol_reason = AdmissionReasonCode.UNDECLARED_SYMBOL_CHANGE.value

    scan = PathScanResult(
        base_commit=base,
        head_commit=head,
        changed_paths=tuple(records),
        undeclared_paths=tuple(sorted(set(undeclared))),
        declared_paths=tuple(declared_paths),
        total_after_bytes=total_after,
    )
    reasons: list[str] = []
    if symbol_reason:
        reasons.append(symbol_reason)
    return scan, tuple(reasons), "", before_by_path, after_by_path


# ---------------------------------------------------------------------------
# Identity commit + result assembly
# ---------------------------------------------------------------------------


def _stable_identity_cid(
    *,
    candidate: MutationCandidate,
    disposition: str,
    reason_codes: Sequence[str],
    scan: PathScanResult | None,
    equivalence_status: str,
    detection_set_cid: str | None,
    worktree_path: str,
    lease_id: str,
    fence: int,
) -> str:
    payload = {
        "interface": ADMIT_MUTATION_INTERFACE,
        "schema": MUTATION_ADMISSION_SCHEMA,
        "candidate_id": candidate.candidate_id,
        "candidate_cid": candidate.candidate_cid,
        "source_root_cid": candidate.source_root_cid,
        "repository_state_cid": candidate.repository_state_cid,
        "disposition": disposition,
        "reason_codes": list(reason_codes),
        "equivalence_status": equivalence_status,
        "detection_set_cid": detection_set_cid,
        "worktree_path": normalize_workspace_path(worktree_path),
        "lease_id": lease_id,
        "fence": int(fence),
        "scan_digest": content_identity(scan.to_dict()) if scan is not None else "",
        "changed_digests": (
            [
                {
                    "path": item.path,
                    "before": item.before_digest,
                    "after": item.after_digest,
                    "kind": item.change_kind,
                }
                for item in scan.changed_paths
            ]
            if scan is not None
            else []
        ),
    }
    return content_identity(payload)


def _build_equivalence_assessment(
    *,
    candidate: MutationCandidate,
    status: str,
    methods: Sequence[str],
    notes: str,
) -> MutationEquivalenceAssessment | None:
    """Best-effort sealed assessment; returns None if header construction fails."""

    try:
        header_payload = candidate.header.to_dict()
        header_payload = dict(header_payload)
        header_payload["artifact_kind"] = "mutation_equivalence_assessment"
        # Generator identity for admission estimate.
        versions = dict(header_payload.get("versions") or {})
        generator = dict(versions.get("generator") or {})
        generator["generator_id"] = GENERATOR_ID
        generator["generator_version"] = GENERATOR_VERSION
        generator["interface_id"] = ADMIT_MUTATION_INTERFACE
        versions["generator"] = generator
        header_payload["versions"] = versions
        header = AssuranceArtifactHeader.from_dict(header_payload)
        assessment_id = f"eq_{candidate.candidate_id}"[:128]
        # assessment_id must be a token
        assessment_id = re.sub(r"[^a-z0-9_.:/+-]", "_", assessment_id.lower())
        if not assessment_id[0].isalpha():
            assessment_id = "eq_" + assessment_id
        mapped_status = {
            EquivalenceEstimate.EQUIVALENT.value: EquivalenceAssessmentStatus.EQUIVALENT.value,
            EquivalenceEstimate.PROBABLY_EQUIVALENT.value: (
                EquivalenceAssessmentStatus.PROBABLY_EQUIVALENT.value
            ),
            EquivalenceEstimate.NOT_EQUIVALENT.value: (
                EquivalenceAssessmentStatus.NOT_EQUIVALENT.value
            ),
            EquivalenceEstimate.UNKNOWN.value: EquivalenceAssessmentStatus.UNKNOWN.value,
        }[status]
        method_enums = list(methods) or [EquivalenceMethod.AST_COMPARISON.value]
        return MutationEquivalenceAssessment(
            header=header,
            assessment_id=assessment_id,
            candidate_id=candidate.candidate_id,
            candidate_cid=candidate.candidate_cid,
            assessment_status=mapped_status,
            methods=method_enums,
            evidence_cids=(),
            difficulty_to_kill_not_evidence=True,
            notes=_clip(notes, limit=512) or None,
            metadata={"estimator": GENERATOR_ID, "estimator_version": GENERATOR_VERSION},
        )
    except (AssuranceBaseError, MutationContractError, TypeError, ValueError):
        return None


def _maybe_record_ledger(
    *,
    ledger: MutationLedger | None,
    candidate: MutationCandidate,
    worktree_path: str,
    lease_id: str,
    fence: int,
    scan: PathScanResult,
    before_by_path: Mapping[str, bytes | None],
    after_by_path: Mapping[str, bytes | None],
    base_commit: str,
) -> tuple[str | None, str | None]:
    """Optionally commit lineage to MutationLedger. Returns (mutation_id, error)."""

    if ledger is None:
        return None, None
    try:
        if not ledger.is_open:
            ledger.open()
        worktree_id = content_identity(
            {
                "kind": "aae-admission-worktree",
                "path": normalize_workspace_path(worktree_path),
            }
        )
        fence_token = f"{lease_id}:{fence}:{base_commit}"
        mutation_fence = ledger.register_fence(
            worktree_id=worktree_id,
            token=fence_token,
            generation=max(1, int(fence)),
            lease_id=lease_id,
            before_snapshot_id=base_commit,
            before_tree_id=scan.base_commit,
            supersede_active=True,
        )
        files: list[MutationFileSpec] = []
        for item in scan.changed_paths:
            files.append(
                MutationFileSpec(
                    path=item.path,
                    before_content=before_by_path.get(item.path),
                    after_content=after_by_path.get(item.path),
                )
            )
        context = MutationContext(
            task_id=f"aae-admit:{candidate.candidate_id}",
            worktree_id=worktree_id,
            fence_id=mutation_fence.fence_id,
            before_snapshot_id=base_commit,
            attempt_id=f"fence:{fence}",
            operator_id=candidate.operator_id,
            lease_id=lease_id,
            repository_id=candidate.header.repository_id,
            before_tree_id=scan.base_commit,
            after_tree_id=scan.head_commit,
            validation_outcome="admitted",
        )
        result: MutationRecordResult = ledger.record_mutation(context, files)
        mutation_id = getattr(result, "mutation_id", None) or getattr(
            result, "mutation_set_id", None
        )
        return (str(mutation_id) if mutation_id else None), None
    except Exception as exc:  # noqa: BLE001 — ledger is optional; surface diagnostic
        return None, _clip(f"ledger record failed: {exc}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def admit_mutation(
    candidate: MutationCandidate | Mapping[str, Any],
    *,
    worktree_path: str | Path,
    lease_id: str,
    fence: int,
    lifecycle_store: WorktreeLifecycleStore | None = None,
    repo_root: str | Path | None = None,
    base_commit: str | None = None,
    declared_paths: Sequence[str] | None = None,
    assurance_manifest: DetectionAssuranceManifest | Mapping[str, Any] | None = None,
    allow_authority_fixture: bool = False,
    require_lifecycle: bool = True,
    mutation_ledger: MutationLedger | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MutationAdmissionResult:
    """Admit one mutant from a caller-supplied owned disposable worktree.

    Interface: ``admit_mutation@1``

    Steps (fail-closed, no worktree create/destroy):

    1. Validate lifecycle ownership of the disposable worktree.
    2. Rescan observed changes vs ``base_commit`` (default ``HEAD`` clean base
       is the merge-base supplied by the caller; when omitted, uses
       ``HEAD`` only for identity and requires a dirty tree vs index/worktree
       relative to ``HEAD`` by diffing ``HEAD``).
    3. Prove only declared paths (and optionally symbols) changed.
    4. Block verifier/policy/key/oracle authority path edits unless
       ``allow_authority_fixture`` is true.
    5. Parse and structurally validate changed artifacts.
    6. Reject trivial invalidity (empty diff, parse failure, no-op).
    7. Estimate equivalence (bounded; unknown never auto-promoted to
       equivalent).
    8. Predict detection via ``predict_detection_set``.
    9. Commit content-addressed admission identity (and optional ledger row).

    Returns a sealed :class:`MutationAdmissionResult` for both success and
    rejection paths. Does not raise for policy rejections; raises
    :class:`AdmissionError` only for malformed API inputs that prevent sealing.
    """

    sealed_candidate = _normalize_candidate(candidate)
    wt = Path(worktree_path)
    root = Path(repo_root) if repo_root is not None else None
    lease = _text(lease_id, "lease_id", empty=True)
    fence_i = _nonneg_int(fence, "fence")
    if fence_i < 1:
        raise AdmissionError("fence must be a positive integer", reason_code="malformed_input")
    allow_auth = _bool(allow_authority_fixture, "allow_authority_fixture")
    require_lc = _bool(require_lifecycle, "require_lifecycle")
    note_text = _optional_text(notes, "notes")
    meta = dict(_mapping(metadata, "metadata"))
    if note_text:
        meta.setdefault("notes", note_text)

    def _result(
        *,
        disposition: AdmissionDisposition,
        reason_codes: Sequence[str],
        admitted: bool,
        scan: PathScanResult | None = None,
        equivalence_status: str = EquivalenceEstimate.UNKNOWN.value,
        equivalence_methods: Sequence[str] = (),
        equivalence_assessment_cid: str | None = None,
        detection_set: ExpectedDetectionSet | None = None,
        ledger_mutation_id: str | None = None,
        lifecycle_record_id: str | None = None,
        diagnostic: str = "",
    ) -> MutationAdmissionResult:
        detection_set_cid = detection_set.detection_set_cid if detection_set else None
        detection_set_id = detection_set.detection_set_id if detection_set else None
        predicted_ids = (
            tuple(detection_set.predicted_detector_ids) if detection_set else ()
        )
        identity = _stable_identity_cid(
            candidate=sealed_candidate,
            disposition=disposition.value,
            reason_codes=reason_codes,
            scan=scan,
            equivalence_status=equivalence_status,
            detection_set_cid=detection_set_cid,
            worktree_path=str(wt),
            lease_id=lease,
            fence=fence_i,
        )
        return MutationAdmissionResult(
            disposition=disposition.value,
            reason_codes=tuple(reason_codes),
            candidate_id=sealed_candidate.candidate_id,
            candidate_cid=sealed_candidate.candidate_cid,
            worktree_path=normalize_workspace_path(wt),
            lease_id=lease,
            fence=fence_i,
            admitted=admitted,
            identity_cid=identity,
            scan=scan.to_dict() if scan is not None else None,
            equivalence_status=equivalence_status,
            equivalence_methods=tuple(equivalence_methods),
            equivalence_assessment_cid=equivalence_assessment_cid,
            detection_set_cid=detection_set_cid,
            detection_set_id=detection_set_id,
            predicted_detector_ids=predicted_ids,
            ledger_mutation_id=ledger_mutation_id,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=_clip(diagnostic),
            metadata=meta,
        )

    # --- 1. owned disposable worktree ---
    record, wt_reasons, wt_diag = _validate_owned_disposable_worktree(
        worktree_path=wt,
        repo_root=root,
        lifecycle_store=lifecycle_store,
        lease_id=lease,
        fence=fence_i,
        require_lifecycle=require_lc,
    )
    lifecycle_record_id = record.record_id if record is not None else None
    if wt_reasons:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=wt_reasons,
            admitted=False,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=wt_diag,
        )

    # --- 2/3. rescan declared changes ---
    try:
        declared = _declared_paths_for_candidate(sealed_candidate, declared_paths)
    except AdmissionError as exc:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(exc.reason_code,),
            admitted=False,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=str(exc),
        )

    if not declared:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(AdmissionReasonCode.NO_DECLARED_CHANGES.value,),
            admitted=False,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic="candidate scope_paths and declared_paths are empty",
        )

    base = base_commit or "HEAD"
    scan, scan_extra_reasons, scan_diag, before_by_path, after_by_path = (
        _rescan_declared_changes(
            worktree=wt,
            base_commit=base,
            declared_paths=declared,
            scope_symbol_ids=tuple(sealed_candidate.scope_symbol_ids),
        )
    )
    if scan is None:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=scan_extra_reasons
            or (AdmissionReasonCode.GIT_SCAN_FAILED.value,),
            admitted=False,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=scan_diag or "rescan failed",
        )

    if not scan.changed_paths:
        return _result(
            disposition=AdmissionDisposition.INVALID,
            reason_codes=(AdmissionReasonCode.EMPTY_DIFF.value,),
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic="no changes observed in disposable worktree",
        )

    if scan.undeclared_paths:
        # Distinguish forbidden vs mere undeclared.
        forbidden = [p for p in scan.undeclared_paths if _always_forbidden(p)]
        if forbidden:
            return _result(
                disposition=AdmissionDisposition.REJECTED,
                reason_codes=(AdmissionReasonCode.FORBIDDEN_PATH.value,),
                admitted=False,
                scan=scan,
                lifecycle_record_id=lifecycle_record_id,
                diagnostic=_clip(f"forbidden paths: {forbidden[:8]}"),
            )
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(AdmissionReasonCode.UNDECLARED_PATH_CHANGE.value,),
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=_clip(
                f"undeclared path changes: {list(scan.undeclared_paths)[:8]}"
            ),
        )

    if scan_extra_reasons:
        # Symbol guard failures after declared paths validated.
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=scan_extra_reasons,
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic="declared symbol scope not evidenced by rescanned changes",
        )

    # --- 4. block authority surfaces ---
    if not allow_auth:
        blocked = [
            item.path
            for item in scan.changed_paths
            if blocked_authority_path(item.path)
        ]
        if blocked:
            return _result(
                disposition=AdmissionDisposition.REJECTED,
                reason_codes=(AdmissionReasonCode.AUTHORITY_PATH_BLOCKED.value,),
                admitted=False,
                scan=scan,
                lifecycle_record_id=lifecycle_record_id,
                diagnostic=_clip(
                    f"verifier/policy/key/oracle path edits blocked: {blocked[:8]}"
                ),
            )

    # --- 5/6. structural validation + trivial invalidity ---
    parse_failures = [
        item
        for item in scan.changed_paths
        if item.structural_outcome == StructuralOutcome.PARSE_FAILED.value
    ]
    if parse_failures:
        return _result(
            disposition=AdmissionDisposition.INVALID,
            reason_codes=(AdmissionReasonCode.PARSE_FAILURE.value,),
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=_clip(
                "; ".join(
                    f"{item.path}: {item.diagnostic}" for item in parse_failures[:4]
                )
            ),
        )

    structural_failures = [
        item
        for item in scan.changed_paths
        if item.structural_outcome
        in {
            StructuralOutcome.STRUCTURAL_INVALID.value,
            StructuralOutcome.TOO_LARGE.value,
        }
    ]
    if structural_failures:
        codes = []
        for item in structural_failures:
            if item.structural_outcome == StructuralOutcome.TOO_LARGE.value:
                codes.append(AdmissionReasonCode.FILE_TOO_LARGE.value)
            else:
                codes.append(AdmissionReasonCode.STRUCTURAL_INVALID.value)
        return _result(
            disposition=AdmissionDisposition.INVALID,
            reason_codes=tuple(dict.fromkeys(codes)),
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=_clip(
                "; ".join(
                    f"{item.path}: {item.diagnostic or item.structural_outcome}"
                    for item in structural_failures[:4]
                )
            ),
        )

    # Trivial no-op: all digests identical.
    if all(
        item.before_digest == item.after_digest
        and item.change_kind in {"modified"}
        for item in scan.changed_paths
    ):
        return _result(
            disposition=AdmissionDisposition.INVALID,
            reason_codes=(AdmissionReasonCode.NO_OP_MUTATION.value,),
            admitted=False,
            scan=scan,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic="mutation is a byte-level no-op",
        )

    # --- 7. equivalence estimate ---
    eq_status, eq_methods, eq_notes = _estimate_equivalence(
        candidate=sealed_candidate,
        records=scan.changed_paths,
        before_by_path=before_by_path,
        after_by_path=after_by_path,
    )
    assessment = _build_equivalence_assessment(
        candidate=sealed_candidate,
        status=eq_status,
        methods=eq_methods,
        notes=eq_notes,
    )
    assessment_cid = assessment.assessment_cid if assessment is not None else None

    if eq_status == EquivalenceEstimate.EQUIVALENT.value:
        # Commit identity as equivalent — not execution-admitted for kill scoring.
        ledger_id, ledger_err = _maybe_record_ledger(
            ledger=mutation_ledger,
            candidate=sealed_candidate,
            worktree_path=str(wt),
            lease_id=lease,
            fence=fence_i,
            scan=scan,
            before_by_path=before_by_path,
            after_by_path=after_by_path,
            base_commit=scan.base_commit,
        )
        diag = eq_notes
        if ledger_err:
            diag = f"{diag}; {ledger_err}"
        return _result(
            disposition=AdmissionDisposition.EQUIVALENT,
            reason_codes=(AdmissionReasonCode.EQUIVALENCE_ESTIMATE.value,),
            admitted=False,
            scan=scan,
            equivalence_status=eq_status,
            equivalence_methods=eq_methods,
            equivalence_assessment_cid=assessment_cid,
            ledger_mutation_id=ledger_id,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=diag,
        )

    # --- 8. predict detection ---
    if assurance_manifest is None:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(AdmissionReasonCode.MANIFEST_REQUIRED.value,),
            admitted=False,
            scan=scan,
            equivalence_status=eq_status,
            equivalence_methods=eq_methods,
            equivalence_assessment_cid=assessment_cid,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic="assurance_manifest is required to predict detection",
        )

    try:
        sealed_manifest = _normalize_manifest(assurance_manifest)
        detection_set = predict_detection_set(sealed_candidate, sealed_manifest)
    except AdmissionError as exc:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(exc.reason_code,),
            admitted=False,
            scan=scan,
            equivalence_status=eq_status,
            equivalence_methods=eq_methods,
            equivalence_assessment_cid=assessment_cid,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=str(exc),
        )
    except (DetectionPredictionError, AssuranceBaseError, TypeError, ValueError) as exc:
        return _result(
            disposition=AdmissionDisposition.REJECTED,
            reason_codes=(AdmissionReasonCode.DETECTION_PREDICTION_FAILED.value,),
            admitted=False,
            scan=scan,
            equivalence_status=eq_status,
            equivalence_methods=eq_methods,
            equivalence_assessment_cid=assessment_cid,
            lifecycle_record_id=lifecycle_record_id,
            diagnostic=_clip(f"detection prediction failed: {exc}"),
        )

    # --- 9. commit identity (+ optional ledger) ---
    ledger_id, ledger_err = _maybe_record_ledger(
        ledger=mutation_ledger,
        candidate=sealed_candidate,
        worktree_path=str(wt),
        lease_id=lease,
        fence=fence_i,
        scan=scan,
        before_by_path=before_by_path,
        after_by_path=after_by_path,
        base_commit=scan.base_commit,
    )
    reason_codes: list[str] = [AdmissionReasonCode.OK.value]
    diagnostic = "mutation admitted"
    if ledger_err:
        reason_codes.append(AdmissionReasonCode.LEDGER_RECORD_FAILED.value)
        diagnostic = f"{diagnostic}; {ledger_err}"

    return _result(
        disposition=AdmissionDisposition.ADMITTED,
        reason_codes=reason_codes,
        admitted=True,
        scan=scan,
        equivalence_status=eq_status,
        equivalence_methods=eq_methods,
        equivalence_assessment_cid=assessment_cid,
        detection_set=detection_set,
        ledger_mutation_id=ledger_id,
        lifecycle_record_id=lifecycle_record_id,
        diagnostic=diagnostic,
    )


__all__ = [
    "ADMIT_MUTATION_INTERFACE",
    "MUTATION_ADMISSION_INTERFACE",
    "MUTATION_ADMISSION_SCHEMA",
    "AdmissionDisposition",
    "AdmissionError",
    "AdmissionReasonCode",
    "ChangedPathRecord",
    "EquivalenceEstimate",
    "MutationAdmissionResult",
    "PathScanResult",
    "StructuralOutcome",
    "admit_mutation",
    "admission_dispositions",
    "admission_reason_codes",
    "blocked_authority_path",
    "open_mutation_ledger",
]

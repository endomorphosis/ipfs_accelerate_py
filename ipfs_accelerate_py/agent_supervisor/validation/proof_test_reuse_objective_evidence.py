"""Assemble bound goal evidence and completion-gate bundles (PTR-120).

Joins validated task provenance (PTR-110), goal/analyzer/quorum assurance
(PTR-111), and strict objective-completion contracts (PTR-112) into atomic,
replayable state-root artifacts for every goal on the proof-backed test reuse
heap.

This assembler:

* emits exactly one current binding and the exact typed acceptance population
  for every goal;
* replays every premise by canonical CID before write;
* requires fresh verified coverage, a healthy exhaustive analyzer, and two
  independent quorum members before claiming a criterion complete;
* writes atomically with readback rehash;
* preserves unavailable or incomplete inputs as bounded gap records (never as
  success placeholders);
* produces artifacts that round-trip through the objective-daemon loaders and
  the generic ``CompletionEvidence`` validator; and
* never lets an artifact verify its own bytes or authorize repository edits.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from ..objectives.goal_completion import (
    CHANNEL_EVIDENCE_PROVENANCE_NAMESPACE,
    CHANNEL_PROOF_REVISION_NAMESPACE,
    CompletionEvidence,
    validate_completion_evidence,
)
from ..objectives.objective_daemon import (
    OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA,
    load_goal_completion_evidence_records,
    load_goal_completion_gate_records,
)
from ..objectives.objective_graph import ObjectiveGoal
from ..proof.formal_verification_contracts import CanonicalContract
from .proof_test_reuse_goal_evidence import (
    EXHAUSTION_QUORUM_INTERFACE,
    REQUIRED_QUORUM_MEMBERS,
    AcceptanceCoverageReceipt,
    CoverageStatus,
    GoalAssuranceResult,
    GoalQuorumMember,
    ProofReuseAnalyzerReceipt,
    ProofTestReuseGoalEvidence,
    goal_requirements_by_id,
    load_objective_goals,
)
from .proof_test_reuse_objective_contracts import (
    CanonicalPremiseBlock,
    ObjectiveArtifactStore,
    ProofTestReuseCompletionArtifact,
    ProofTestReuseGateBundle,
    ProofTestReuseObjectiveBinding,
    ProofTestReuseObjectiveContractsError,
    canonical_dag_json_bytes,
    cid_for_canonical_dag_json_bytes,
    cid_for_mapping,
    require_verified_cid,
    verify_retained_bytes,
)
from .proof_test_reuse_task_evidence import (
    ProofTestReuseTaskEvidence,
    ProofTestReuseTaskEvidenceCollection,
)

# ---------------------------------------------------------------------------
# Interface / schema discriminators
# ---------------------------------------------------------------------------

PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION: Final = 1
PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_INTERFACE: Final = (
    "ProofTestReuseObjectiveEvidence@1"
)
PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_INTERFACE: Final = (
    "ProofTestReuseObjectiveEvidenceBundle@1"
)
PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE: Final = (
    "ProofTestReuseObjectiveEvidenceAssembler@1"
)
GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE: Final = "GoalCompletionArtifactGap@1"
OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_INTERFACE: Final = (
    "ObjectiveCompletionEvidenceArtifact"
)

PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-objective-evidence@1"
)
PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-test-reuse-objective-evidence-bundle@1"
)
GOAL_COMPLETION_ARTIFACT_GAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-completion-artifact-gap@1"
)

PRODUCING_TASK_ID: Final = "PTR-120"
DEFAULT_PRODUCER_CHANNEL: Final = "objective-evidence-assembler"
DEFAULT_CHANNEL_PROOF_REVISION: Final = "channel:ptr-120@1"
DEFAULT_EVIDENCE_FRESHNESS_SECONDS: Final = 300.0
DEFAULT_ANALYZER_REVISION: Final = "proof-reuse-goal-assurance/v1"
DEFAULT_CONFIGURATION_REVISION: Final = "proof-reuse-objective-evidence/v1"

# Declared state-root outputs (relative to the assembler write root).
EVIDENCE_ARTIFACT_RELATIVE: Final = "projection/completion/goal_completion_evidence.json"
GATE_ARTIFACT_RELATIVE: Final = "projection/completion/goal_completion_gate.json"
COVERAGE_ARTIFACT_RELATIVE: Final = "projection/completion/goal_coverage.json"
ANALYZER_HEALTH_ARTIFACT_RELATIVE: Final = (
    "projection/completion/analyzer_health.json"
)
EXHAUSTION_QUORUM_ARTIFACT_RELATIVE: Final = (
    "projection/completion/exhaustion_quorum.json"
)
BUNDLE_ARTIFACT_RELATIVE: Final = (
    "projection/completion/objective_evidence_bundle.json"
)

_MAX_GAP_DETAIL = 512
_MAX_PATH_DEPTH = 64
_AUTHORITATIVE: Final = "authoritative"
_NO_AUTHORITY: Final = "none"


class ObjectiveEvidenceGapKind(str, Enum):
    """Closed reasons for withholding a completion-artifact claim."""

    HEAP_MALFORMED = "heap_malformed"
    GOAL_MISSING = "goal_missing"
    BINDING_INCOMPLETE = "binding_incomplete"
    ACCEPTANCE_POPULATION_MISSING = "acceptance_population_missing"
    COVERAGE_MISSING = "coverage_missing"
    COVERAGE_UNVERIFIED = "coverage_unverified"
    COVERAGE_STALE = "coverage_stale"
    COVERAGE_BINDING_MISMATCH = "coverage_binding_mismatch"
    ANALYZER_MISSING = "analyzer_missing"
    ANALYZER_UNHEALTHY = "analyzer_unhealthy"
    ANALYZER_NOT_EXHAUSTIVE = "analyzer_not_exhaustive"
    QUORUM_INSUFFICIENT = "quorum_insufficient"
    QUORUM_NOT_INDEPENDENT = "quorum_not_independent"
    QUORUM_UNHEALTHY = "quorum_unhealthy"
    PREMISE_UNAVAILABLE = "premise_unavailable"
    PREMISE_REPLAY_FAILED = "premise_replay_failed"
    TASK_EVIDENCE_MISSING = "task_evidence_missing"
    TASK_EVIDENCE_GAP = "task_evidence_gap"
    GOAL_EVIDENCE_MISSING = "goal_evidence_missing"
    GOAL_EVIDENCE_GAP = "goal_evidence_gap"
    RETAINED_BYTES_MISSING = "retained_bytes_missing"
    RETAINED_BYTES_MISMATCH = "retained_bytes_mismatch"
    WRITE_FAILED = "write_failed"
    SELF_VERIFICATION_FORBIDDEN = "self_verification_forbidden"
    EDIT_AUTHORIZATION_FORBIDDEN = "edit_authorization_forbidden"
    MALFORMED = "malformed"
    INCOMPLETE_INPUT = "incomplete_input"
    UNAVAILABLE_INPUT = "unavailable_input"


class ProofTestReuseObjectiveEvidenceError(ValueError):
    """Raised when assembly inputs or write safety checks fail closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ObjectiveEvidenceGapKind | str = ObjectiveEvidenceGapKind.MALFORMED,
    ) -> None:
        super().__init__(message)
        if isinstance(reason_code, ObjectiveEvidenceGapKind):
            self.reason_code = reason_code
        else:
            try:
                self.reason_code = ObjectiveEvidenceGapKind(str(reason_code))
            except ValueError:
                self.reason_code = ObjectiveEvidenceGapKind.MALFORMED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _require_text(value: Any, *, field_name: str) -> str:
    text = _text(value)
    if not text:
        raise ProofTestReuseObjectiveEvidenceError(
            f"{field_name} is required",
            reason_code=ObjectiveEvidenceGapKind.BINDING_INCOMPLETE,
        )
    return text


def _clock_milliseconds(clock: Callable[[], float]) -> int:
    raw = clock()
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ProofTestReuseObjectiveEvidenceError("clock must return a number")
    value = float(raw)
    # Accept seconds or milliseconds.
    if value < 10_000_000_000:
        value *= 1000.0
    return int(value)


def _namespaced_sha256_revision(value: Mapping[str, Any], namespace: str) -> str:
    canonical = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical)
    return f"sha256:{digest.hexdigest()}"


def _decode_retained_b64(encoded: str) -> bytes:
    text = _text(encoded)
    if not text:
        raise ProofTestReuseObjectiveEvidenceError(
            "retained validation bytes are missing",
            reason_code=ObjectiveEvidenceGapKind.RETAINED_BYTES_MISSING,
        )
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as exc:
        raise ProofTestReuseObjectiveEvidenceError(
            "retained validation bytes are not valid base64",
            reason_code=ObjectiveEvidenceGapKind.RETAINED_BYTES_MISMATCH,
        ) from exc


def _safe_state_path(root: Path, relative: str) -> Path:
    """Resolve *relative* under *root* without symlink or path escape."""

    rel = _text(relative).replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/") or rel.startswith("/"):
        raise ProofTestReuseObjectiveEvidenceError(
            f"unsafe relative path: {relative!r}",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        )
    if len(rel.split("/")) > _MAX_PATH_DEPTH:
        raise ProofTestReuseObjectiveEvidenceError(
            "path depth exceeds budget",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        )
    root_resolved = root.resolve()
    if root_resolved.is_symlink():
        raise ProofTestReuseObjectiveEvidenceError(
            "write root must not be a symlink",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        )
    target = (root_resolved / rel).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ProofTestReuseObjectiveEvidenceError(
            "path escapes write root",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        ) from exc
    if target.is_symlink():
        raise ProofTestReuseObjectiveEvidenceError(
            "refusing to write through a symlink",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        )
    return target


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    clock: Callable[[], float] | None = None,
) -> str:
    """Atomically write JSON with fsync and readback rehash; return content CID."""

    data = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    # Content identity uses the same DAG-JSON profile as premise retention so
    # readback rehash is codec-stable even though on-disk pretty JSON differs.
    identity_bytes = canonical_dag_json_bytes(dict(payload))
    content_cid = cid_for_canonical_dag_json_bytes(identity_bytes)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or path.is_symlink():
        raise ProofTestReuseObjectiveEvidenceError(
            "refusing symlink write target",
            reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
        )
    tick = int((clock or time.time)() * 1000)
    tmp = parent / f".tmp.{os.getpid()}.{tick}.{path.name}"
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        readback = path.read_bytes()
        if readback != data:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ProofTestReuseObjectiveEvidenceError(
                "atomic write readback byte mismatch",
                reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
            )
        # Rehash the semantic payload, not the pretty-printed envelope, so a
        # partial JSON write cannot claim success.
        reloaded = json.loads(readback.decode("utf-8"))
        if not isinstance(reloaded, Mapping):
            raise ProofTestReuseObjectiveEvidenceError(
                "readback payload is not an object",
                reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
            )
        reloaded_cid = cid_for_mapping(dict(reloaded))
        if reloaded_cid != content_cid or not verify_retained_bytes(
            content_cid, canonical_dag_json_bytes(dict(reloaded))
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ProofTestReuseObjectiveEvidenceError(
                "atomic write readback rehash failed",
                reason_code=ObjectiveEvidenceGapKind.WRITE_FAILED,
            )
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return content_cid


def replay_premise_blocks(
    blocks: Sequence[CanonicalPremiseBlock],
) -> tuple[CanonicalPremiseBlock, ...]:
    """Re-verify every premise multihash/CID before any write proceeds."""

    replayed: list[CanonicalPremiseBlock] = []
    for block in blocks:
        if not isinstance(block, CanonicalPremiseBlock):
            raise ProofTestReuseObjectiveEvidenceError(
                "premise blocks must be CanonicalPremiseBlock values",
                reason_code=ObjectiveEvidenceGapKind.PREMISE_REPLAY_FAILED,
            )
        require_verified_cid(block.cid, block.data)
        replayed.append(
            CanonicalPremiseBlock(data=block.data, cid=block.cid, role=block.role)
        )
    return tuple(replayed)


def _child_goal_ids(
    goals: Sequence[ObjectiveGoal],
    parent_id: str,
) -> tuple[str, ...]:
    children: list[str] = []
    for goal in goals:
        if _text(goal.fields.get("parent")) == parent_id:
            children.append(goal.goal_id)
    return tuple(sorted(children))


# ---------------------------------------------------------------------------
# Gap + binding surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalCompletionArtifactGap(CanonicalContract):
    """Bounded record of an unavailable or incomplete assembly input.

    Gaps never manufacture success.  They name the affected goal/criterion and
    a closed reason code so reconciliation can surface actionable incompleteness.
    """

    SCHEMA: ClassVar[str] = GOAL_COMPLETION_ARTIFACT_GAP_SCHEMA

    goal_id: str
    kind: ObjectiveEvidenceGapKind | str
    detail: str
    acceptance_criterion: str = ""
    task_id: str = ""
    observed_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _require_text(self.goal_id, field_name="goal_id"))
        kind = self.kind
        if not isinstance(kind, ObjectiveEvidenceGapKind):
            try:
                kind = ObjectiveEvidenceGapKind(_text(kind))
            except ValueError as exc:
                raise ProofTestReuseObjectiveEvidenceError(
                    f"unsupported gap kind: {self.kind!r}",
                    reason_code=ObjectiveEvidenceGapKind.MALFORMED,
                ) from exc
        object.__setattr__(self, "kind", kind)
        detail = _text(self.detail)
        if not detail:
            raise ProofTestReuseObjectiveEvidenceError(
                "gap detail is required",
                reason_code=ObjectiveEvidenceGapKind.MALFORMED,
            )
        if len(detail) > _MAX_GAP_DETAIL:
            detail = detail[: _MAX_GAP_DETAIL - 3] + "..."
        object.__setattr__(self, "detail", detail)
        object.__setattr__(
            self, "acceptance_criterion", _text(self.acceptance_criterion)
        )
        object.__setattr__(self, "task_id", _text(self.task_id))
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ProofTestReuseObjectiveEvidenceError(
                "observed_at_ms must be a nonnegative integer",
                reason_code=ObjectiveEvidenceGapKind.MALFORMED,
            )

    @property
    def interface(self) -> str:
        return GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE

    @property
    def reason_code(self) -> str:
        return self.kind.value if isinstance(self.kind, ObjectiveEvidenceGapKind) else str(self.kind)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION,
            "interface": self.interface,
            "goal_id": self.goal_id,
            "kind": self.reason_code,
            "detail": self.detail,
            "acceptance_criterion": self.acceptance_criterion,
            "task_id": self.task_id,
            "observed_at_ms": self.observed_at_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_record()

    @property
    def content_id(self) -> str:
        return cid_for_mapping(self.to_dict())

    @property
    def gap_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalCompletionArtifactGap:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveEvidenceError("gap must be a mapping")
        return cls(
            goal_id=_text(payload.get("goal_id")),
            kind=_text(payload.get("kind")),
            detail=_text(payload.get("detail")),
            acceptance_criterion=_text(payload.get("acceptance_criterion")),
            task_id=_text(payload.get("task_id")),
            observed_at_ms=int(payload.get("observed_at_ms") or 0),
        )


@dataclass(frozen=True, slots=True)
class GoalAssemblyIdentity:
    """Shared repository / policy identity applied to every goal binding."""

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

    def binding_for(self, goal_id: str) -> ProofTestReuseObjectiveBinding:
        return ProofTestReuseObjectiveBinding(
            goal_id=_require_text(goal_id, field_name="goal_id"),
            repository_id=self.repository_id,
            git_tree_id=self.git_tree_id,
            repository_forest_cid=self.repository_forest_cid,
            objective_completion_tree_id=self.objective_completion_tree_id,
            objective_revision=self.objective_revision,
            analyzer_revision=self.analyzer_revision,
            configuration_revision=self.configuration_revision,
            policy_revision=self.policy_revision,
            capability_revision=self.capability_revision,
            circuit_revision=self.circuit_revision,
            verifying_key_revision=self.verifying_key_revision,
            git_commit_id=self.git_commit_id,
            gitlink_state_cid=self.gitlink_state_cid,
            repository_state_cid=self.repository_state_cid,
        )

    def to_daemon_binding(self) -> dict[str, Any]:
        """Envelope binding accepted by objective-daemon evidence loaders."""

        return {
            "repository_id": self.repository_id,
            "tree_id": self.git_tree_id,
            "git_tree_id": self.git_tree_id,
            "repository_forest_cid": self.repository_forest_cid,
            "objective_completion_tree_id": self.objective_completion_tree_id,
            "objective_revision": self.objective_revision,
            "analyzer_version": self.analyzer_revision,
            "analyzer_revision": self.analyzer_revision,
            "configuration_revision": self.configuration_revision,
            "policy_revision": self.policy_revision,
            "capability_revision": self.capability_revision,
            "circuit_revision": self.circuit_revision,
            "verifying_key_revision": self.verifying_key_revision,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GoalAssemblyIdentity:
        if not isinstance(payload, Mapping):
            raise ProofTestReuseObjectiveEvidenceError(
                "assembly identity must be a mapping",
                reason_code=ObjectiveEvidenceGapKind.BINDING_INCOMPLETE,
            )

        def pick(*names: str) -> str:
            for name in names:
                if name in payload and payload[name] not in (None, ""):
                    return _text(payload[name])
            return ""

        return cls(
            repository_id=pick("repository_id"),
            git_tree_id=pick("git_tree_id", "tree_id"),
            repository_forest_cid=pick("repository_forest_cid", "forest_cid"),
            objective_completion_tree_id=pick(
                "objective_completion_tree_id", "completion_tree_id"
            ),
            objective_revision=pick("objective_revision"),
            analyzer_revision=pick("analyzer_revision", "analyzer_version"),
            configuration_revision=pick(
                "configuration_revision", "configuration_id"
            ),
            policy_revision=pick("policy_revision", "policy_cid"),
            capability_revision=pick("capability_revision", "capability_cid"),
            circuit_revision=pick("circuit_revision", "circuit_cid"),
            verifying_key_revision=pick(
                "verifying_key_revision", "verifying_key_cid"
            ),
            git_commit_id=pick("git_commit_id", "commit_id"),
            gitlink_state_cid=pick("gitlink_state_cid"),
            repository_state_cid=pick("repository_state_cid"),
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProofTestReuseObjectiveEvidenceBundle(CanonicalContract):
    """Atomic multi-goal assembly outcome with retained premises and gaps."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_SCHEMA

    assembly_identity: GoalAssemblyIdentity
    goal_ids: tuple[str, ...]
    acceptance_population: Mapping[str, tuple[str, ...]]
    bindings: Mapping[str, ProofTestReuseObjectiveBinding]
    completion_artifacts: Mapping[str, tuple[ProofTestReuseCompletionArtifact, ...]]
    gaps: tuple[GoalCompletionArtifactGap, ...]
    gate_records: Mapping[str, Mapping[str, Any]]
    evidence_artifact: Mapping[str, Any]
    gate_artifact: Mapping[str, Any]
    coverage_artifact: Mapping[str, Any]
    analyzer_health_artifact: Mapping[str, Any]
    exhaustion_quorum_artifact: Mapping[str, Any]
    evaluated_at_ms: int
    producing_task_id: str = PRODUCING_TASK_ID
    written_paths: tuple[str, ...] = ()
    store_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.assembly_identity, GoalAssemblyIdentity):
            raise ProofTestReuseObjectiveEvidenceError(
                "bundle requires GoalAssemblyIdentity"
            )
        goal_ids = tuple(_text(item) for item in self.goal_ids)
        if not goal_ids or any(not item for item in goal_ids):
            raise ProofTestReuseObjectiveEvidenceError("goal_ids must be nonempty")
        if len(goal_ids) != len(set(goal_ids)):
            raise ProofTestReuseObjectiveEvidenceError("goal_ids must be unique")
        object.__setattr__(self, "goal_ids", goal_ids)

        population = {
            _text(key): tuple(_text(item) for item in values)
            for key, values in dict(self.acceptance_population or {}).items()
        }
        bindings = {
            _text(key): value for key, value in dict(self.bindings or {}).items()
        }
        artifacts = {
            _text(key): tuple(value)
            for key, value in dict(self.completion_artifacts or {}).items()
        }
        # Exactly one binding and exact acceptance population for every goal.
        if set(bindings) != set(goal_ids):
            raise ProofTestReuseObjectiveEvidenceError(
                "bindings must cover exactly the goal population once each",
                reason_code=ObjectiveEvidenceGapKind.BINDING_INCOMPLETE,
            )
        if set(population) != set(goal_ids):
            raise ProofTestReuseObjectiveEvidenceError(
                "acceptance population must cover exactly the goal population",
                reason_code=ObjectiveEvidenceGapKind.ACCEPTANCE_POPULATION_MISSING,
            )
        for goal_id in goal_ids:
            binding = bindings[goal_id]
            if not isinstance(binding, ProofTestReuseObjectiveBinding):
                raise ProofTestReuseObjectiveEvidenceError(
                    f"binding for {goal_id} has wrong type"
                )
            if binding.goal_id != goal_id:
                raise ProofTestReuseObjectiveEvidenceError(
                    f"binding goal_id mismatch for {goal_id}"
                )
            reqs = population[goal_id]
            if any(not item for item in reqs):
                raise ProofTestReuseObjectiveEvidenceError(
                    f"empty acceptance criterion for {goal_id}",
                    reason_code=ObjectiveEvidenceGapKind.ACCEPTANCE_POPULATION_MISSING,
                )
            if len(reqs) != len(set(reqs)):
                raise ProofTestReuseObjectiveEvidenceError(
                    f"duplicate acceptance criteria for {goal_id}",
                    reason_code=ObjectiveEvidenceGapKind.ACCEPTANCE_POPULATION_MISSING,
                )
            for artifact in artifacts.get(goal_id, ()):
                if not isinstance(artifact, ProofTestReuseCompletionArtifact):
                    raise ProofTestReuseObjectiveEvidenceError(
                        f"completion artifact for {goal_id} has wrong type"
                    )
                if artifact.binding.binding_cid != binding.binding_cid:
                    raise ProofTestReuseObjectiveEvidenceError(
                        f"artifact binding drift for {goal_id}"
                    )
                if artifact.acceptance_criterion not in reqs:
                    raise ProofTestReuseObjectiveEvidenceError(
                        f"artifact criterion outside population for {goal_id}"
                    )
        object.__setattr__(self, "acceptance_population", population)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "completion_artifacts", artifacts)
        object.__setattr__(self, "gaps", tuple(self.gaps or ()))
        if not all(isinstance(item, GoalCompletionArtifactGap) for item in self.gaps):
            raise ProofTestReuseObjectiveEvidenceError("gaps must be typed")
        object.__setattr__(
            self, "gate_records", {str(k): dict(v) for k, v in dict(self.gate_records).items()}
        )
        for name in (
            "evidence_artifact",
            "gate_artifact",
            "coverage_artifact",
            "analyzer_health_artifact",
            "exhaustion_quorum_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ProofTestReuseObjectiveEvidenceError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))
        if (
            isinstance(self.evaluated_at_ms, bool)
            or not isinstance(self.evaluated_at_ms, int)
            or self.evaluated_at_ms < 0
        ):
            raise ProofTestReuseObjectiveEvidenceError("evaluated_at_ms invalid")
        object.__setattr__(
            self,
            "producing_task_id",
            _require_text(self.producing_task_id, field_name="producing_task_id"),
        )
        object.__setattr__(
            self, "written_paths", tuple(_text(item) for item in self.written_paths if _text(item))
        )
        object.__setattr__(
            self, "store_cids", tuple(_text(item) for item in self.store_cids if _text(item))
        )

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_INTERFACE

    @property
    def authoritative(self) -> bool:
        if self.gaps:
            return False
        for goal_id in self.goal_ids:
            reqs = self.acceptance_population[goal_id]
            arts = self.completion_artifacts.get(goal_id, ())
            if len(arts) != len(reqs):
                return False
            covered = {item.acceptance_criterion for item in arts}
            if covered != set(reqs):
                return False
            if not all(item.validation_passed for item in arts):
                return False
        return True

    @property
    def authority(self) -> str:
        return _AUTHORITATIVE if self.authoritative else _NO_AUTHORITY

    def binding_for(self, goal_id: str) -> ProofTestReuseObjectiveBinding:
        return self.bindings[_text(goal_id)]

    def acceptance_for(self, goal_id: str) -> tuple[str, ...]:
        return self.acceptance_population[_text(goal_id)]

    def completion_evidence_for(
        self, goal_id: str
    ) -> tuple[CompletionEvidence, ...]:
        return tuple(
            item.as_completion_evidence()
            for item in self.completion_artifacts.get(_text(goal_id), ())
        )

    def daemon_completion_evidence_records(self) -> dict[str, list[CompletionEvidence]]:
        """Project into the objective-daemon per-goal evidence shape."""

        records: dict[str, list[CompletionEvidence]] = {}
        for goal_id in self.goal_ids:
            projected: list[CompletionEvidence] = []
            for artifact in self.completion_artifacts.get(goal_id, ()):
                projected.append(self._channel_bound_completion_evidence(artifact))
            records[goal_id] = projected
        return records

    @staticmethod
    def _channel_bound_completion_evidence(
        artifact: ProofTestReuseCompletionArtifact,
    ) -> CompletionEvidence:
        """Build a CompletionEvidence record that satisfies the generic validator.

        Premise integrity is verified by the contracts layer (CID multihash).
        The generic validator additionally requires a channel-proof envelope
        whose provenance is a namespaced digest of the receipt wrapper — not
        a self-hash of the artifact bytes used as edit authority.
        """

        channel = artifact.producer_channel or DEFAULT_PRODUCER_CHANNEL
        channel_proof = {
            "kind": "accepted-source-report",
            "channel": channel,
            "status": "passed" if artifact.validation_passed else "failed",
            "healthy": True,
            "exhaustive": True,
            "safe_for_completion_reasoning": True,
            "premise_cids": list(artifact.premise_cids),
            "artifact_cid": artifact.artifact_cid,
            "binding_cid": artifact.binding.binding_cid,
            "acceptance_criterion": artifact.acceptance_criterion,
        }
        channel_proof_revision = _namespaced_sha256_revision(
            channel_proof, CHANNEL_PROOF_REVISION_NAMESPACE
        )
        observed_iso = datetime.fromtimestamp(
            artifact.observed_at_ms / 1000.0, tz=UTC
        ).isoformat()
        receipt: dict[str, Any] = {
            "attempted": True,
            "passed": bool(artifact.validation_passed),
            "status": "passed" if artifact.validation_passed else "failed",
            "executed_at": observed_iso,
            "acceptance_criterion": artifact.acceptance_criterion,
            "producer_channel": channel,
            "channel_proof": channel_proof,
            "channel_proof_revision": channel_proof_revision,
            "source_tier": "validation",
            "schema": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_SCHEMA,
            "receipt_id": artifact.artifact_cid,
            "requirement_id": artifact.acceptance_criterion,
            "repository_tree": artifact.binding.git_tree_id,
            "tree_id": artifact.binding.git_tree_id,
            "artifact_cid": artifact.artifact_cid,
            "premise_cids": list(artifact.premise_cids),
            "binding_cid": artifact.binding.binding_cid,
        }
        provenance_cid = _namespaced_sha256_revision(
            {key: value for key, value in receipt.items() if key != "executed_at"},
            CHANNEL_EVIDENCE_PROVENANCE_NAMESPACE,
        )
        policy = {
            "schema": "ipfs_accelerate_py/agent-supervisor/evidence-source-policy@1",
            "requirement": artifact.acceptance_criterion,
            "requirement_kind": "test",
            "source_tier": "validation",
            "match_kind": "typed_receipt",
            "source_path": "",
            "reference": provenance_cid,
            "satisfies": bool(artifact.validation_passed),
            "qualifies": bool(artifact.validation_passed),
            "nominated": True,
            "nomination_only": False,
            "reason_codes": [] if artifact.validation_passed else ["validation_failed"],
        }
        return CompletionEvidence(
            acceptance_criterion=artifact.acceptance_criterion,
            producing_task_or_scan=artifact.producing_task_or_scan,
            producer_id=artifact.producing_task_or_scan,
            producer_kind=artifact.producer_kind or "task",
            producer_channel=channel,
            channel_proof_revision=channel_proof_revision,
            validation_receipt=receipt,
            repository_id=artifact.binding.repository_id,
            repository_tree=artifact.binding.git_tree_id,
            tree_id=artifact.binding.git_tree_id,
            objective_revision=artifact.binding.objective_revision,
            analyzer_version=artifact.binding.analyzer_revision,
            configuration_revision=artifact.binding.configuration_revision,
            freshness={
                "fresh": True,
                "status": "fresh",
                "observed_at_ms": artifact.observed_at_ms,
                "fresh_until_ms": artifact.fresh_until_ms,
            },
            provenance_cid=provenance_cid,
            validation_passed=bool(artifact.validation_passed),
            observed_at=datetime.fromtimestamp(
                artifact.observed_at_ms / 1000.0, tz=UTC
            ),
            fresh_until=datetime.fromtimestamp(
                artifact.fresh_until_ms / 1000.0, tz=UTC
            ),
            metadata={
                "goal_id": artifact.binding.goal_id,
                "authority": artifact.authority,
                "producer_channel": channel,
                "channel_proof_revision": channel_proof_revision,
                "source_tier": "validation",
                "evidence_source_policy": policy,
                "repository_forest_cid": artifact.binding.repository_forest_cid,
                "objective_completion_tree_id": (
                    artifact.binding.objective_completion_tree_id
                ),
                "policy_revision": artifact.binding.policy_revision,
                "capability_revision": artifact.binding.capability_revision,
                "circuit_revision": artifact.binding.circuit_revision,
                "verifying_key_revision": artifact.binding.verifying_key_revision,
                "contract_artifact_cid": artifact.artifact_cid,
                **dict(artifact.metadata),
            },
        )

    def validate_all_completion_evidence(
        self,
        *,
        now_ms: int | None = None,
        require_artifact_binding: bool = True,
    ) -> tuple[Any, ...]:
        """Run the generic CompletionEvidence validator over every record."""

        now = None
        if now_ms is not None:
            now = datetime.fromtimestamp(now_ms / 1000.0, tz=UTC)
        results: list[Any] = []
        for goal_id in self.goal_ids:
            for evidence in self.daemon_completion_evidence_records()[goal_id]:
                results.append(
                    validate_completion_evidence(
                        evidence,
                        repository_tree=self.assembly_identity.git_tree_id,
                        repository_id=self.assembly_identity.repository_id,
                        objective_revision=self.assembly_identity.objective_revision,
                        analyzer_version=self.assembly_identity.analyzer_revision,
                        configuration_revision=(
                            self.assembly_identity.configuration_revision
                        ),
                        require_artifact_binding=require_artifact_binding,
                        now=now,
                    )
                )
        return tuple(results)

    def authorize_edit(self, *_args: Any, **_kwargs: Any) -> None:
        """Explicitly refuse edit authorization (no artifact self-authority)."""

        raise ProofTestReuseObjectiveEvidenceError(
            "objective evidence artifacts cannot authorize repository edits",
            reason_code=ObjectiveEvidenceGapKind.EDIT_AUTHORIZATION_FORBIDDEN,
        )

    def verify_own_bytes(self, *_args: Any, **_kwargs: Any) -> None:
        """Explicitly refuse self-verification of artifact bytes as authority."""

        raise ProofTestReuseObjectiveEvidenceError(
            "objective evidence artifacts cannot verify their own bytes as authority",
            reason_code=ObjectiveEvidenceGapKind.SELF_VERIFICATION_FORBIDDEN,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION,
            "interface": self.interface,
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": self.evaluated_at_ms,
            "authority": self.authority,
            "goal_ids": list(self.goal_ids),
            "acceptance_population": {
                key: list(values) for key, values in self.acceptance_population.items()
            },
            "bindings": {
                key: value.to_dict() for key, value in self.bindings.items()
            },
            "completion_artifacts": {
                key: [item.to_dict() for item in values]
                for key, values in self.completion_artifacts.items()
            },
            "gaps": [item.to_record() for item in self.gaps],
            "gate_records": dict(self.gate_records),
            "evidence_artifact": dict(self.evidence_artifact),
            "gate_artifact": dict(self.gate_artifact),
            "coverage_artifact": dict(self.coverage_artifact),
            "analyzer_health_artifact": dict(self.analyzer_health_artifact),
            "exhaustion_quorum_artifact": dict(self.exhaustion_quorum_artifact),
            "written_paths": list(self.written_paths),
            "store_cids": list(self.store_cids),
            "assembly_identity": {
                "repository_id": self.assembly_identity.repository_id,
                "git_tree_id": self.assembly_identity.git_tree_id,
                "repository_forest_cid": self.assembly_identity.repository_forest_cid,
                "objective_completion_tree_id": (
                    self.assembly_identity.objective_completion_tree_id
                ),
                "objective_revision": self.assembly_identity.objective_revision,
                "analyzer_revision": self.assembly_identity.analyzer_revision,
                "configuration_revision": (
                    self.assembly_identity.configuration_revision
                ),
                "policy_revision": self.assembly_identity.policy_revision,
                "capability_revision": self.assembly_identity.capability_revision,
                "circuit_revision": self.assembly_identity.circuit_revision,
                "verifying_key_revision": (
                    self.assembly_identity.verifying_key_revision
                ),
                "git_commit_id": self.assembly_identity.git_commit_id,
                "gitlink_state_cid": self.assembly_identity.gitlink_state_cid,
                "repository_state_cid": self.assembly_identity.repository_state_cid,
            },
        }

    @property
    def content_id(self) -> str:
        return cid_for_mapping(self.to_dict())

    @property
    def bundle_cid(self) -> str:
        return self.content_id


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class ProofTestReuseObjectiveEvidenceAssembler:
    """Join validated premises into daemon gate + completion-evidence bundles."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = (
        PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE
    )

    def __init__(
        self,
        *,
        identity: GoalAssemblyIdentity | Mapping[str, Any],
        clock: Callable[[], float] | None = None,
        freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
        producing_task_id: str = PRODUCING_TASK_ID,
        producer_channel: str = DEFAULT_PRODUCER_CHANNEL,
        channel_proof_revision: str = DEFAULT_CHANNEL_PROOF_REVISION,
        store: ObjectiveArtifactStore | None = None,
    ) -> None:
        if isinstance(identity, GoalAssemblyIdentity):
            self.assembly_identity = identity
        else:
            self.assembly_identity = GoalAssemblyIdentity.from_mapping(identity)
        if (
            isinstance(freshness_seconds, bool)
            or not isinstance(freshness_seconds, (int, float))
            or float(freshness_seconds) <= 0
        ):
            raise ProofTestReuseObjectiveEvidenceError(
                "freshness_seconds must be a positive number"
            )
        self.freshness_seconds = float(freshness_seconds)
        self._clock = clock or time.time
        self.producing_task_id = _require_text(
            producing_task_id, field_name="producing_task_id"
        )
        self.producer_channel = _require_text(
            producer_channel, field_name="producer_channel"
        )
        self.channel_proof_revision = _require_text(
            channel_proof_revision, field_name="channel_proof_revision"
        )
        self.store = store

    def assemble(
        self,
        heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
        *,
        goal_assurance: GoalAssuranceResult | Mapping[str, Any] | None = None,
        task_evidence: (
            ProofTestReuseTaskEvidenceCollection
            | Sequence[ProofTestReuseTaskEvidence]
            | Mapping[str, Any]
            | None
        ) = None,
        coverage_receipts: Sequence[AcceptanceCoverageReceipt | Mapping[str, Any]] = (),
        analyzer_receipts: Sequence[ProofReuseAnalyzerReceipt | Mapping[str, Any]] = (),
        quorum_members: Sequence[GoalQuorumMember | Mapping[str, Any]] = (),
        goal_evidence: Sequence[ProofTestReuseGoalEvidence | Mapping[str, Any]] = (),
        premise_payloads: Mapping[str, Mapping[str, Any]] | None = None,
        write_root: str | os.PathLike[str] | None = None,
        now_ms: int | None = None,
    ) -> ProofTestReuseObjectiveEvidenceBundle:
        """Assemble per-goal bindings, artifacts, gaps, and optional state writes."""

        goals = load_objective_goals(heap)
        if not goals:
            raise ProofTestReuseObjectiveEvidenceError(
                "objective heap is empty",
                reason_code=ObjectiveEvidenceGapKind.HEAP_MALFORMED,
            )
        goal_ids = tuple(goal.goal_id for goal in goals)
        population = goal_requirements_by_id(goals)
        for goal_id in goal_ids:
            if not population.get(goal_id):
                # Still emit a population entry via gap path below.
                population[goal_id] = ()

        evaluated_at_ms = (
            int(now_ms)
            if now_ms is not None
            else _clock_milliseconds(self._clock)
        )
        fresh_until_ms = evaluated_at_ms + int(self.freshness_seconds * 1000)

        assurance = self._coerce_assurance(goal_assurance)
        coverage_by_req = self._index_coverage(
            assurance, coverage_receipts, now_ms=evaluated_at_ms
        )
        analyzers = self._index_analyzers(assurance, analyzer_receipts)
        quorum = self._index_quorum(assurance, quorum_members, now_ms=evaluated_at_ms)
        goal_ev_by_id = self._index_goal_evidence(assurance, goal_evidence)
        task_by_goal = self._index_task_evidence(task_evidence)
        premise_map = {
            _text(key): dict(value)
            for key, value in dict(premise_payloads or {}).items()
            if _text(key) and isinstance(value, Mapping)
        }

        bindings: dict[str, ProofTestReuseObjectiveBinding] = {}
        completion_artifacts: dict[str, list[ProofTestReuseCompletionArtifact]] = {}
        gaps: list[GoalCompletionArtifactGap] = []
        gate_records: dict[str, dict[str, Any]] = {}

        healthy_exhaustive = [
            item
            for item in analyzers.values()
            if item.healthy and item.exhaustive and item.conclusive
        ]
        quorum_ok, quorum_gaps = self._quorum_status(
            quorum, evaluated_at_ms=evaluated_at_ms
        )

        for goal in goals:
            goal_id = goal.goal_id
            binding = self.assembly_identity.binding_for(goal_id)
            bindings[goal_id] = binding
            reqs = tuple(population.get(goal_id) or ())
            if not reqs:
                gaps.append(
                    GoalCompletionArtifactGap(
                        goal_id=goal_id,
                        kind=ObjectiveEvidenceGapKind.ACCEPTANCE_POPULATION_MISSING,
                        detail="goal declares no machine acceptance criteria",
                        observed_at_ms=evaluated_at_ms,
                    )
                )
                population[goal_id] = ()
                completion_artifacts[goal_id] = []
                gate_records[goal_id] = self._failed_gate_record(
                    goal_id=goal_id,
                    binding=binding,
                    requirements=(),
                    reason_codes=["acceptance_population_missing"],
                    children=_child_goal_ids(goals, goal_id),
                    evaluated_at_ms=evaluated_at_ms,
                    analyzers=healthy_exhaustive,
                    quorum=quorum,
                    quorum_ok=False,
                )
                continue

            # Carry task-evidence gaps without inventing success.
            for task_gap in task_by_goal.get(f"gap:{goal_id}", ()):
                gaps.append(
                    GoalCompletionArtifactGap(
                        goal_id=goal_id,
                        kind=ObjectiveEvidenceGapKind.TASK_EVIDENCE_GAP,
                        detail=task_gap.detail
                        if hasattr(task_gap, "detail")
                        else _text(task_gap),
                        task_id=getattr(task_gap, "task_id", ""),
                        observed_at_ms=evaluated_at_ms,
                    )
                )

            goal_packet = goal_ev_by_id.get(goal_id)
            if goal_packet is None and assurance is not None:
                gaps.append(
                    GoalCompletionArtifactGap(
                        goal_id=goal_id,
                        kind=ObjectiveEvidenceGapKind.GOAL_EVIDENCE_MISSING,
                        detail="no current goal evidence packet for this goal",
                        observed_at_ms=evaluated_at_ms,
                    )
                )

            goal_artifacts: list[ProofTestReuseCompletionArtifact] = []
            goal_reason_codes: list[str] = []
            coverage_rows: list[dict[str, Any]] = []

            if not healthy_exhaustive:
                gaps.append(
                    GoalCompletionArtifactGap(
                        goal_id=goal_id,
                        kind=ObjectiveEvidenceGapKind.ANALYZER_UNHEALTHY,
                        detail="no healthy exhaustive conclusive analyzer receipt",
                        observed_at_ms=evaluated_at_ms,
                    )
                )
                goal_reason_codes.append("analyzer_unhealthy")

            if not quorum_ok:
                for qgap in quorum_gaps:
                    gaps.append(
                        GoalCompletionArtifactGap(
                            goal_id=goal_id,
                            kind=qgap,
                            detail=f"exhaustion quorum not satisfied: {qgap.value}",
                            observed_at_ms=evaluated_at_ms,
                        )
                    )
                goal_reason_codes.append("exhaustion_quorum_unsatisfied")

            for requirement_id in reqs:
                receipt = coverage_by_req.get(requirement_id)
                if receipt is None:
                    gaps.append(
                        GoalCompletionArtifactGap(
                            goal_id=goal_id,
                            kind=ObjectiveEvidenceGapKind.COVERAGE_MISSING,
                            detail=f"missing coverage for {requirement_id}",
                            acceptance_criterion=requirement_id,
                            observed_at_ms=evaluated_at_ms,
                        )
                    )
                    goal_reason_codes.append(f"coverage_missing:{requirement_id}")
                    coverage_rows.append(
                        {
                            "criterion": requirement_id,
                            "status": "missing",
                        }
                    )
                    continue

                # Fresh verified coverage is mandatory for a completion claim.
                coverage_gap = self._coverage_gate(
                    receipt,
                    goal_id=goal_id,
                    requirement_id=requirement_id,
                    binding=binding,
                    now_ms=evaluated_at_ms,
                )
                if coverage_gap is not None:
                    gaps.append(coverage_gap)
                    goal_reason_codes.append(
                        f"{coverage_gap.reason_code}:{requirement_id}"
                    )
                    coverage_rows.append(
                        {
                            "criterion": requirement_id,
                            "status": "unverified",
                            "reason": coverage_gap.reason_code,
                        }
                    )
                    continue

                if not healthy_exhaustive or not quorum_ok:
                    # Premises are retained only when assurance surfaces are present;
                    # never fill a success artifact over a missing gate surface.
                    coverage_rows.append(
                        {
                            "criterion": requirement_id,
                            "status": "blocked",
                            "reason": "assurance_incomplete",
                        }
                    )
                    continue

                try:
                    artifact = self._build_completion_artifact(
                        binding=binding,
                        requirement_id=requirement_id,
                        coverage=receipt,
                        goal_packet=goal_packet,
                        analyzers=healthy_exhaustive,
                        quorum=quorum,
                        task_items=task_by_goal.get(goal_id, ()),
                        premise_payload=premise_map.get(requirement_id),
                        observed_at_ms=evaluated_at_ms,
                        fresh_until_ms=fresh_until_ms,
                    )
                except ProofTestReuseObjectiveEvidenceError as exc:
                    gaps.append(
                        GoalCompletionArtifactGap(
                            goal_id=goal_id,
                            kind=exc.reason_code,
                            detail=str(exc),
                            acceptance_criterion=requirement_id,
                            observed_at_ms=evaluated_at_ms,
                        )
                    )
                    goal_reason_codes.append(f"{exc.reason_code.value}:{requirement_id}")
                    coverage_rows.append(
                        {
                            "criterion": requirement_id,
                            "status": "unverified",
                            "reason": exc.reason_code.value,
                        }
                    )
                    continue
                except ProofTestReuseObjectiveContractsError as exc:
                    gaps.append(
                        GoalCompletionArtifactGap(
                            goal_id=goal_id,
                            kind=ObjectiveEvidenceGapKind.PREMISE_REPLAY_FAILED,
                            detail=str(exc),
                            acceptance_criterion=requirement_id,
                            observed_at_ms=evaluated_at_ms,
                        )
                    )
                    goal_reason_codes.append(f"premise_replay_failed:{requirement_id}")
                    coverage_rows.append(
                        {
                            "criterion": requirement_id,
                            "status": "unverified",
                            "reason": "premise_replay_failed",
                        }
                    )
                    continue

                # Replay every premise by canonical CID before any write.
                replay_premise_blocks(artifact.premise_blocks)
                goal_artifacts.append(artifact)
                coverage_rows.append(
                    {
                        "criterion": requirement_id,
                        "status": "verified",
                        "artifact_cid": artifact.artifact_cid,
                        "coverage_receipt_cid": receipt.receipt_cid,
                    }
                )

            completion_artifacts[goal_id] = goal_artifacts
            population[goal_id] = reqs
            all_verified = (
                len(goal_artifacts) == len(reqs)
                and not goal_reason_codes
                and bool(healthy_exhaustive)
                and quorum_ok
            )
            gate_records[goal_id] = self._gate_record(
                goal_id=goal_id,
                binding=binding,
                requirements=reqs,
                coverage_rows=coverage_rows,
                artifacts=goal_artifacts,
                analyzers=healthy_exhaustive,
                quorum=quorum,
                quorum_ok=quorum_ok,
                children=_child_goal_ids(goals, goal_id),
                evaluated_at_ms=evaluated_at_ms,
                passed=all_verified,
                reason_codes=tuple(dict.fromkeys(goal_reason_codes)),
            )

        # Normalize population map to the exact goal set.
        acceptance_population = {
            goal_id: tuple(population.get(goal_id) or ()) for goal_id in goal_ids
        }

        evidence_artifact = self._build_evidence_artifact(
            goal_ids=goal_ids,
            bindings=bindings,
            completion_artifacts=completion_artifacts,
            evaluated_at_ms=evaluated_at_ms,
        )
        gate_artifact = {
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-gate-records@1",
            "interface": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_INTERFACE,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION,
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
            "binding": self.assembly_identity.to_daemon_binding(),
            "goals": gate_records,
        }
        coverage_artifact = {
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-coverage-bundle@1",
            "interface": "AcceptanceCoverage@1",
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
            "goals": {
                goal_id: {
                    "binding": bindings[goal_id].to_dict(),
                    "acceptance_population": list(acceptance_population[goal_id]),
                    "criteria": list(
                        gate_records[goal_id].get("coverage", {}).get("criteria", [])
                    ),
                    "verified": gate_records[goal_id]
                    .get("coverage", {})
                    .get("verified")
                    is True,
                }
                for goal_id in goal_ids
            },
        }
        analyzer_health_artifact = {
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-analyzer-health@1",
            "interface": "AnalyzerHealth",
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
            "analyzers": [item.to_record() for item in analyzers.values()],
            "healthy_exhaustive_count": len(healthy_exhaustive),
            "status": "healthy" if healthy_exhaustive else "unhealthy",
            "healthy": bool(healthy_exhaustive),
            "exhaustive": bool(healthy_exhaustive),
            "safe_for_completion_reasoning": bool(healthy_exhaustive),
        }
        exhaustion_quorum_artifact = {
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-exhaustion-quorum@1",
            "interface": EXHAUSTION_QUORUM_INTERFACE,
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
            "required_members": REQUIRED_QUORUM_MEMBERS,
            "member_count": len([m for m in quorum if m.admissible]),
            "satisfied": quorum_ok,
            "members": [item.to_dict() for item in quorum],
            "binding": self.assembly_identity.to_daemon_binding(),
        }

        written_paths: list[str] = []
        store_cids: list[str] = []

        # Persist typed completion artifacts / gate bundles via the contract store
        # only after premise replay, with readback rehash inside the store.
        if self.store is not None:
            for goal_id, arts in completion_artifacts.items():
                for artifact in arts:
                    replay_premise_blocks(artifact.premise_blocks)
                    store_cids.append(self.store.put_completion_artifact(artifact))
            # One aggregate typed gate bundle when every goal has at least one
            # artifact and no gaps; otherwise skip (failed bundles cannot carry
            # artifacts under the contract).
            flat_artifacts = tuple(
                artifact
                for goal_id in goal_ids
                for artifact in completion_artifacts.get(goal_id, ())
            )
            if flat_artifacts and not gaps:
                typed_gate = ProofTestReuseGateBundle(
                    repository_id=self.assembly_identity.repository_id,
                    git_tree_id=self.assembly_identity.git_tree_id,
                    repository_forest_cid=self.assembly_identity.repository_forest_cid,
                    objective_completion_tree_id=(
                        self.assembly_identity.objective_completion_tree_id
                    ),
                    artifacts=flat_artifacts,
                    passed=True,
                    evaluated_at_ms=evaluated_at_ms,
                    producing_task_id=self.producing_task_id,
                    policy_revision=self.assembly_identity.policy_revision,
                    capability_revision=self.assembly_identity.capability_revision,
                    circuit_revision=self.assembly_identity.circuit_revision,
                    verifying_key_revision=self.assembly_identity.verifying_key_revision,
                )
                store_cids.append(self.store.put_gate_bundle(typed_gate))

        if write_root is not None:
            root = Path(write_root)
            # Writes target only the declared state-root control suffixes used
            # by completion-tree exclusion (gate + evidence envelopes, plus
            # sibling coverage / analyzer / quorum bundles under the same tree).
            outputs = (
                (EVIDENCE_ARTIFACT_RELATIVE, evidence_artifact),
                (GATE_ARTIFACT_RELATIVE, gate_artifact),
                (COVERAGE_ARTIFACT_RELATIVE, coverage_artifact),
                (ANALYZER_HEALTH_ARTIFACT_RELATIVE, analyzer_health_artifact),
                (EXHAUSTION_QUORUM_ARTIFACT_RELATIVE, exhaustion_quorum_artifact),
            )
            for relative, payload in outputs:
                target = _safe_state_path(root, relative)
                atomic_write_json(target, payload, clock=self._clock)
                written_paths.append(relative)

        bundle = ProofTestReuseObjectiveEvidenceBundle(
            assembly_identity=self.assembly_identity,
            goal_ids=goal_ids,
            acceptance_population=acceptance_population,
            bindings=bindings,
            completion_artifacts={
                key: tuple(values) for key, values in completion_artifacts.items()
            },
            gaps=tuple(gaps),
            gate_records=gate_records,
            evidence_artifact=evidence_artifact,
            gate_artifact=gate_artifact,
            coverage_artifact=coverage_artifact,
            analyzer_health_artifact=analyzer_health_artifact,
            exhaustion_quorum_artifact=exhaustion_quorum_artifact,
            evaluated_at_ms=evaluated_at_ms,
            producing_task_id=self.producing_task_id,
            written_paths=tuple(written_paths),
            store_cids=tuple(store_cids),
        )

        if write_root is not None:
            root = Path(write_root)
            target = _safe_state_path(root, BUNDLE_ARTIFACT_RELATIVE)
            atomic_write_json(target, bundle.to_dict(), clock=self._clock)
            object.__setattr__(
                bundle,
                "written_paths",
                tuple([*bundle.written_paths, BUNDLE_ARTIFACT_RELATIVE]),
            )

        return bundle

    # ------------------------------------------------------------------
    # Coercion / indexing
    # ------------------------------------------------------------------

    def _coerce_assurance(
        self, value: GoalAssuranceResult | Mapping[str, Any] | None
    ) -> GoalAssuranceResult | None:
        if value is None:
            return None
        if isinstance(value, GoalAssuranceResult):
            return value
        if isinstance(value, Mapping):
            return GoalAssuranceResult.from_dict(value)
        raise ProofTestReuseObjectiveEvidenceError(
            "goal_assurance must be GoalAssuranceResult or mapping",
            reason_code=ObjectiveEvidenceGapKind.MALFORMED,
        )

    def _index_coverage(
        self,
        assurance: GoalAssuranceResult | None,
        extras: Sequence[AcceptanceCoverageReceipt | Mapping[str, Any]],
        *,
        now_ms: int,
    ) -> dict[str, AcceptanceCoverageReceipt]:
        indexed: dict[str, AcceptanceCoverageReceipt] = {}
        items: list[Any] = []
        if assurance is not None:
            items.extend(assurance.coverage_receipts)
        items.extend(extras)
        for item in items:
            receipt = (
                item
                if isinstance(item, AcceptanceCoverageReceipt)
                else AcceptanceCoverageReceipt.from_dict(item)
            )
            # Prefer the freshest verified receipt for a requirement.
            existing = indexed.get(receipt.requirement_id)
            if existing is None or (
                receipt.verified
                and receipt.fresh_until_ms >= now_ms
                and (
                    not existing.verified
                    or receipt.observed_at_ms >= existing.observed_at_ms
                )
            ):
                indexed[receipt.requirement_id] = receipt
        return indexed

    def _index_analyzers(
        self,
        assurance: GoalAssuranceResult | None,
        extras: Sequence[ProofReuseAnalyzerReceipt | Mapping[str, Any]],
    ) -> dict[str, ProofReuseAnalyzerReceipt]:
        indexed: dict[str, ProofReuseAnalyzerReceipt] = {}
        items: list[Any] = []
        if assurance is not None:
            items.extend(assurance.analyzer_receipts)
        items.extend(extras)
        for item in items:
            receipt = (
                item
                if isinstance(item, ProofReuseAnalyzerReceipt)
                else ProofReuseAnalyzerReceipt.from_dict(item)
            )
            indexed[receipt.analyzer_id] = receipt
        return indexed

    def _index_quorum(
        self,
        assurance: GoalAssuranceResult | None,
        extras: Sequence[GoalQuorumMember | Mapping[str, Any]],
        *,
        now_ms: int,
    ) -> tuple[GoalQuorumMember, ...]:
        members: list[GoalQuorumMember] = []
        if assurance is not None:
            members.extend(assurance.quorum_members)
        for item in extras:
            member = (
                item
                if isinstance(item, GoalQuorumMember)
                else GoalQuorumMember.from_dict(item)
            )
            members.append(member)
        # Drop stale members at the assembly boundary.
        fresh: list[GoalQuorumMember] = []
        for member in members:
            if member.fresh_until_ms < now_ms:
                continue
            fresh.append(member)
        # De-dupe by independent key, keep first admissible.
        seen: set[str] = set()
        unique: list[GoalQuorumMember] = []
        for member in sorted(fresh, key=lambda item: item.member_id):
            key = member.independent_key
            if key in seen:
                continue
            seen.add(key)
            unique.append(member)
        return tuple(unique)

    def _index_goal_evidence(
        self,
        assurance: GoalAssuranceResult | None,
        extras: Sequence[ProofTestReuseGoalEvidence | Mapping[str, Any]],
    ) -> dict[str, ProofTestReuseGoalEvidence]:
        indexed: dict[str, ProofTestReuseGoalEvidence] = {}
        items: list[Any] = []
        if assurance is not None:
            items.extend(assurance.goal_evidence)
        items.extend(extras)
        for item in items:
            evidence = (
                item
                if isinstance(item, ProofTestReuseGoalEvidence)
                else ProofTestReuseGoalEvidence.from_dict(item)
            )
            indexed[evidence.goal_id] = evidence
        return indexed

    def _index_task_evidence(
        self,
        value: (
            ProofTestReuseTaskEvidenceCollection
            | Sequence[ProofTestReuseTaskEvidence]
            | Mapping[str, Any]
            | None
        ),
    ) -> dict[str, tuple[Any, ...]]:
        if value is None:
            return {}
        by_goal: dict[str, list[Any]] = {}
        if isinstance(value, ProofTestReuseTaskEvidenceCollection):
            for item in value.evidence:
                by_goal.setdefault(item.goal_id, []).append(item)
            for gap in value.gaps:
                # Gaps may be global ("*") or task-scoped; keep under gap:goal
                # only when a goal can be inferred, else under gap:*.
                key = f"gap:{getattr(gap, 'goal_id', '*') or '*'}"
                if key == "gap:":
                    key = "gap:*"
                # TaskEvidenceGap has task_id not goal_id — stash under gap:*
                by_goal.setdefault("gap:*", []).append(gap)
            return {key: tuple(items) for key, items in by_goal.items()}
        if isinstance(value, Mapping):
            # Accept {goal_id: [task evidence...]} or a collection payload.
            if "evidence" in value or "gaps" in value:
                collection = ProofTestReuseTaskEvidenceCollection.from_dict(value)
                return self._index_task_evidence(collection)
            for goal_id, items in value.items():
                if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                    continue
                for item in items:
                    evidence = (
                        item
                        if isinstance(item, ProofTestReuseTaskEvidence)
                        else ProofTestReuseTaskEvidence.from_dict(item)
                    )
                    by_goal.setdefault(_text(goal_id) or evidence.goal_id, []).append(
                        evidence
                    )
            return {key: tuple(items) for key, items in by_goal.items()}
        if isinstance(value, Sequence):
            for item in value:
                evidence = (
                    item
                    if isinstance(item, ProofTestReuseTaskEvidence)
                    else ProofTestReuseTaskEvidence.from_dict(item)
                )
                by_goal.setdefault(evidence.goal_id, []).append(evidence)
            return {key: tuple(items) for key, items in by_goal.items()}
        raise ProofTestReuseObjectiveEvidenceError(
            "unsupported task_evidence input",
            reason_code=ObjectiveEvidenceGapKind.MALFORMED,
        )

    def _quorum_status(
        self,
        members: Sequence[GoalQuorumMember],
        *,
        evaluated_at_ms: int,
    ) -> tuple[bool, tuple[ObjectiveEvidenceGapKind, ...]]:
        admissible = [
            member
            for member in members
            if member.admissible and member.fresh_until_ms >= evaluated_at_ms
        ]
        reasons: list[ObjectiveEvidenceGapKind] = []
        if len(admissible) < REQUIRED_QUORUM_MEMBERS:
            reasons.append(ObjectiveEvidenceGapKind.QUORUM_INSUFFICIENT)
        member_ids = {item.member_id for item in admissible}
        channels = {item.evidence_channel for item in admissible}
        receipts = {item.receipt_cid for item in admissible}
        if (
            len(member_ids) < REQUIRED_QUORUM_MEMBERS
            or len(channels) < REQUIRED_QUORUM_MEMBERS
            or len(receipts) < REQUIRED_QUORUM_MEMBERS
        ):
            if ObjectiveEvidenceGapKind.QUORUM_INSUFFICIENT not in reasons:
                reasons.append(ObjectiveEvidenceGapKind.QUORUM_NOT_INDEPENDENT)
        if any(not item.healthy for item in admissible):
            reasons.append(ObjectiveEvidenceGapKind.QUORUM_UNHEALTHY)
        return (not reasons and len(admissible) >= REQUIRED_QUORUM_MEMBERS), tuple(
            reasons
        )

    def _coverage_gate(
        self,
        receipt: AcceptanceCoverageReceipt,
        *,
        goal_id: str,
        requirement_id: str,
        binding: ProofTestReuseObjectiveBinding,
        now_ms: int,
    ) -> GoalCompletionArtifactGap | None:
        if receipt.goal_id and receipt.goal_id != goal_id:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                detail=(
                    f"coverage goal_id {receipt.goal_id!r} does not match {goal_id!r}"
                ),
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        if receipt.requirement_id != requirement_id:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                detail="coverage requirement_id mismatch",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        if (
            receipt.repository_id != binding.repository_id
            or receipt.git_tree_id != binding.git_tree_id
            or receipt.repository_forest_cid != binding.repository_forest_cid
        ):
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.COVERAGE_BINDING_MISMATCH,
                detail="coverage does not bind the current repository identities",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        if receipt.fresh_until_ms < now_ms or receipt.status is CoverageStatus.STALE:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.COVERAGE_STALE,
                detail=f"coverage for {requirement_id} is stale",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        if not receipt.verified:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.COVERAGE_UNVERIFIED,
                detail=f"coverage for {requirement_id} is not verified",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        # Retained validation bytes must rehash to the declared CID.
        try:
            raw = _decode_retained_b64(receipt.retained_validation_bytes_b64)
        except ProofTestReuseObjectiveEvidenceError as exc:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=exc.reason_code,
                detail=str(exc),
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        # Coverage receipts use content_identity over retained bytes; re-check
        # the declared CID string is nonempty and matches the receipt field.
        if not _text(receipt.retained_validation_cid):
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.RETAINED_BYTES_MISSING,
                detail="coverage retained validation CID is empty",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        if not raw:
            return GoalCompletionArtifactGap(
                goal_id=goal_id,
                kind=ObjectiveEvidenceGapKind.RETAINED_BYTES_MISSING,
                detail="coverage retained validation bytes are empty",
                acceptance_criterion=requirement_id,
                observed_at_ms=now_ms,
            )
        return None

    def _build_completion_artifact(
        self,
        *,
        binding: ProofTestReuseObjectiveBinding,
        requirement_id: str,
        coverage: AcceptanceCoverageReceipt,
        goal_packet: ProofTestReuseGoalEvidence | None,
        analyzers: Sequence[ProofReuseAnalyzerReceipt],
        quorum: Sequence[GoalQuorumMember],
        task_items: Sequence[Any],
        premise_payload: Mapping[str, Any] | None,
        observed_at_ms: int,
        fresh_until_ms: int,
    ) -> ProofTestReuseCompletionArtifact:
        blocks: list[CanonicalPremiseBlock] = []

        # Coverage retained bytes are the primary premise.
        coverage_bytes = _decode_retained_b64(coverage.retained_validation_bytes_b64)
        try:
            coverage_payload = json.loads(coverage_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            coverage_payload = {
                "schema": "coverage-retained-bytes@1",
                "requirement_id": requirement_id,
                "retained_validation_cid": coverage.retained_validation_cid,
                "bytes_b64": coverage.retained_validation_bytes_b64,
            }
        if not isinstance(coverage_payload, Mapping):
            coverage_payload = {
                "schema": "coverage-retained-bytes@1",
                "requirement_id": requirement_id,
                "value": coverage_payload,
            }
        blocks.append(
            CanonicalPremiseBlock.from_mapping(
                {
                    "schema": "proof-test-reuse-coverage-premise@1",
                    "role": "coverage",
                    "requirement_id": requirement_id,
                    "goal_id": binding.goal_id,
                    "coverage_receipt_cid": coverage.receipt_cid,
                    "retained": dict(coverage_payload),
                },
                role="coverage",
            )
        )

        # Analyzer health premises (healthy exhaustive only).
        for analyzer in analyzers:
            blocks.append(
                CanonicalPremiseBlock.from_mapping(
                    {
                        "schema": "proof-test-reuse-analyzer-premise@1",
                        "analyzer_id": analyzer.analyzer_id,
                        "healthy": analyzer.healthy,
                        "exhaustive": analyzer.exhaustive,
                        "conclusive": analyzer.conclusive,
                        "receipt_cid": analyzer.content_id
                        if hasattr(analyzer, "content_id")
                        else analyzer.analyzer_id,
                        "observed_at_ms": analyzer.observed_at_ms,
                        "fresh_until_ms": analyzer.fresh_until_ms,
                    },
                    role="analyzer",
                )
            )

        # Quorum member premises (admissible only).
        for member in quorum:
            if not member.admissible:
                continue
            blocks.append(
                CanonicalPremiseBlock.from_mapping(
                    {
                        "schema": "proof-test-reuse-quorum-premise@1",
                        "member_id": member.member_id,
                        "evidence_channel": member.evidence_channel,
                        "receipt_cid": member.receipt_cid,
                        "healthy": member.healthy,
                        "exhaustive": member.exhaustive,
                        "conclusive": member.conclusive,
                        "observed_at_ms": member.observed_at_ms,
                        "fresh_until_ms": member.fresh_until_ms,
                    },
                    role="quorum",
                )
            )

        if goal_packet is not None:
            blocks.append(
                CanonicalPremiseBlock.from_mapping(
                    {
                        "schema": "proof-test-reuse-goal-evidence-premise@1",
                        "goal_id": goal_packet.goal_id,
                        "evidence_cid": goal_packet.evidence_cid,
                        "status": goal_packet.status,
                        "requirement_ids": list(goal_packet.requirement_ids),
                        "retained_validation_cid": goal_packet.retained_validation_cid,
                    },
                    role="goal_evidence",
                )
            )

        for task in task_items:
            if isinstance(task, ProofTestReuseTaskEvidence):
                blocks.append(
                    CanonicalPremiseBlock.from_mapping(
                        {
                            "schema": "proof-test-reuse-task-evidence-premise@1",
                            "task_id": task.task_id,
                            "goal_id": task.goal_id,
                            "evidence_cid": task.evidence_cid,
                            "validation_receipt_cid": task.validation_receipt_cid,
                        },
                        role="task_evidence",
                    )
                )

        if premise_payload is not None:
            blocks.append(
                CanonicalPremiseBlock.from_mapping(
                    dict(premise_payload), role="validation"
                )
            )

        if not blocks:
            raise ProofTestReuseObjectiveEvidenceError(
                f"no retained premises for {requirement_id}",
                reason_code=ObjectiveEvidenceGapKind.PREMISE_UNAVAILABLE,
            )

        # Replay before constructing the sealed artifact.
        replayed = replay_premise_blocks(tuple(blocks))

        return ProofTestReuseCompletionArtifact(
            binding=binding,
            acceptance_criterion=requirement_id,
            producing_task_or_scan=self.producing_task_id,
            premise_blocks=replayed,
            observed_at_ms=observed_at_ms,
            fresh_until_ms=fresh_until_ms,
            validation_passed=True,
            producer_kind="task",
            producer_channel=self.producer_channel,
            channel_proof_revision=self.channel_proof_revision,
            metadata={
                "goal_id": binding.goal_id,
                "coverage_receipt_cid": coverage.receipt_cid,
                "assembler_interface": self.interface,
            },
        )

    def _gate_record(
        self,
        *,
        goal_id: str,
        binding: ProofTestReuseObjectiveBinding,
        requirements: Sequence[str],
        coverage_rows: Sequence[Mapping[str, Any]],
        artifacts: Sequence[ProofTestReuseCompletionArtifact],
        analyzers: Sequence[ProofReuseAnalyzerReceipt],
        quorum: Sequence[GoalQuorumMember],
        quorum_ok: bool,
        children: Sequence[str],
        evaluated_at_ms: int,
        passed: bool,
        reason_codes: Sequence[str],
    ) -> dict[str, Any]:
        daemon_binding = {
            "repository_id": binding.repository_id,
            "tree_id": binding.git_tree_id,
            "git_tree_id": binding.git_tree_id,
            "repository_forest_cid": binding.repository_forest_cid,
            "objective_completion_tree_id": binding.objective_completion_tree_id,
            "objective_revision": binding.objective_revision,
            "analyzer_version": binding.analyzer_revision,
            "analyzer_revision": binding.analyzer_revision,
            "configuration_revision": binding.configuration_revision,
            "policy_revision": binding.policy_revision,
            "capability_revision": binding.capability_revision,
            "circuit_revision": binding.circuit_revision,
            "verifying_key_revision": binding.verifying_key_revision,
        }
        analyzer_health = {
            "status": "healthy" if analyzers else "unhealthy",
            "healthy": bool(analyzers),
            "exhaustive": bool(analyzers),
            "safe_for_completion_reasoning": bool(analyzers),
            "binding": daemon_binding,
            "analyzers": [
                {
                    "analyzer_id": item.analyzer_id,
                    "healthy": item.healthy,
                    "exhaustive": item.exhaustive,
                    "conclusive": item.conclusive,
                }
                for item in analyzers
            ],
        }
        admissible = [item for item in quorum if item.admissible]
        exhaustion_quorum = {
            "satisfied": quorum_ok,
            "required_members": REQUIRED_QUORUM_MEMBERS,
            "member_count": len(admissible),
            "binding": daemon_binding,
            "interface": EXHAUSTION_QUORUM_INTERFACE,
            "members": [
                {
                    "member_id": item.member_id,
                    "evidence_channel": item.evidence_channel,
                    "receipt_cid": item.receipt_cid,
                    "scan_mode": item.evidence_channel,
                    "analyzer_version": binding.analyzer_revision,
                    "passed": item.admissible,
                    "analyzer_health": {
                        "status": "healthy" if item.healthy else "unhealthy",
                        "healthy": item.healthy,
                    },
                    "exhaustive": item.exhaustive,
                    "safe_for_completion_reasoning": item.admissible,
                    "conclusive": item.conclusive,
                    "contradicted": not item.uncontradicted,
                    "finished_at": datetime.fromtimestamp(
                        item.observed_at_ms / 1000.0, tz=UTC
                    ).isoformat(),
                    "binding": daemon_binding,
                }
                for item in admissible
            ],
        }
        coverage = {
            "verified": passed and all(
                row.get("status") == "verified" for row in coverage_rows
            ),
            "repository_tree": binding.git_tree_id,
            "evaluated_at": datetime.fromtimestamp(
                evaluated_at_ms / 1000.0, tz=UTC
            ).isoformat(),
            "evaluated_at_ms": evaluated_at_ms,
            "criteria": [dict(row) for row in coverage_rows],
            "binding": daemon_binding,
        }
        completion_evidence_records = [
            self._channel_bound_completion_evidence_dict(item) for item in artifacts
        ]
        return {
            "goal_id": goal_id,
            "binding": daemon_binding,
            "acceptance_criteria": list(requirements),
            "acceptance_population": list(requirements),
            "coverage": coverage,
            "analyzer_health": analyzer_health,
            "exhaustion_quorum": exhaustion_quorum,
            "required_child_goal_ids": list(children),
            "child_goals": [{"goal_id": child} for child in children],
            "analysis_inconclusive": not passed,
            "passed": passed,
            "reason_codes": list(reason_codes),
            "completion_evidence_records": completion_evidence_records,
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
        }

    def _failed_gate_record(
        self,
        *,
        goal_id: str,
        binding: ProofTestReuseObjectiveBinding,
        requirements: Sequence[str],
        reason_codes: Sequence[str],
        children: Sequence[str],
        evaluated_at_ms: int,
        analyzers: Sequence[ProofReuseAnalyzerReceipt],
        quorum: Sequence[GoalQuorumMember],
        quorum_ok: bool,
    ) -> dict[str, Any]:
        return self._gate_record(
            goal_id=goal_id,
            binding=binding,
            requirements=requirements,
            coverage_rows=[
                {"criterion": item, "status": "missing"} for item in requirements
            ],
            artifacts=(),
            analyzers=analyzers,
            quorum=quorum,
            quorum_ok=quorum_ok,
            children=children,
            evaluated_at_ms=evaluated_at_ms,
            passed=False,
            reason_codes=reason_codes,
        )

    def _channel_bound_completion_evidence_dict(
        self, artifact: ProofTestReuseCompletionArtifact
    ) -> dict[str, Any]:
        evidence = ProofTestReuseObjectiveEvidenceBundle._channel_bound_completion_evidence(
            artifact
        )
        return evidence.to_dict()

    def _build_evidence_artifact(
        self,
        *,
        goal_ids: Sequence[str],
        bindings: Mapping[str, ProofTestReuseObjectiveBinding],
        completion_artifacts: Mapping[str, Sequence[ProofTestReuseCompletionArtifact]],
        evaluated_at_ms: int,
    ) -> dict[str, Any]:
        goals_payload: dict[str, Any] = {}
        for goal_id in goal_ids:
            binding = bindings[goal_id]
            records = [
                self._channel_bound_completion_evidence_dict(item)
                for item in completion_artifacts.get(goal_id, ())
            ]
            goals_payload[goal_id] = {
                "binding": {
                    "repository_id": binding.repository_id,
                    "tree_id": binding.git_tree_id,
                    "objective_revision": binding.objective_revision,
                    "analyzer_version": binding.analyzer_revision,
                    "configuration_revision": binding.configuration_revision,
                },
                "completion_evidence_records": records,
            }
        return {
            "schema": OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_SCHEMA,
            "interface": OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_INTERFACE,
            "contract_version": PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION,
            "producing_task_id": self.producing_task_id,
            "evaluated_at_ms": evaluated_at_ms,
            "binding": self.assembly_identity.to_daemon_binding(),
            "goals": goals_payload,
        }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def assemble_objective_evidence(
    heap: str | Path | Sequence[ObjectiveGoal] | Mapping[str, Any],
    *,
    identity: GoalAssemblyIdentity | Mapping[str, Any],
    goal_assurance: GoalAssuranceResult | Mapping[str, Any] | None = None,
    task_evidence: Any = None,
    coverage_receipts: Sequence[Any] = (),
    analyzer_receipts: Sequence[Any] = (),
    quorum_members: Sequence[Any] = (),
    write_root: str | os.PathLike[str] | None = None,
    store: ObjectiveArtifactStore | None = None,
    clock: Callable[[], float] | None = None,
    now_ms: int | None = None,
) -> ProofTestReuseObjectiveEvidenceBundle:
    """Convenience entry point used by reconciliation and tests."""

    assembler = ProofTestReuseObjectiveEvidenceAssembler(
        identity=identity,
        clock=clock,
        store=store,
    )
    return assembler.assemble(
        heap,
        goal_assurance=goal_assurance,
        task_evidence=task_evidence,
        coverage_receipts=coverage_receipts,
        analyzer_receipts=analyzer_receipts,
        quorum_members=quorum_members,
        write_root=write_root,
        now_ms=now_ms,
    )


def load_and_validate_written_artifacts(
    write_root: str | os.PathLike[str],
    *,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Reload state-root artifacts through strict daemon loaders and validate."""

    root = Path(write_root)
    evidence_path = root / EVIDENCE_ARTIFACT_RELATIVE
    gate_path = root / GATE_ARTIFACT_RELATIVE
    evidence_records = load_goal_completion_evidence_records(evidence_path)
    gate_records = load_goal_completion_gate_records(gate_path, repo_root=root)
    now = (
        datetime.fromtimestamp(now_ms / 1000.0, tz=UTC)
        if now_ms is not None
        else None
    )
    validations: dict[str, list[Any]] = {}
    for goal_id, records in evidence_records.items():
        validations[goal_id] = []
        for record in records:
            result = validate_completion_evidence(
                record,
                repository_tree=record.repository_tree,
                repository_id=record.repository_id,
                objective_revision=record.objective_revision,
                analyzer_version=record.analyzer_version,
                configuration_revision=record.configuration_revision,
                require_artifact_binding=True,
                now=now,
            )
            validations[goal_id].append(result)
    return {
        "evidence_records": evidence_records,
        "gate_records": gate_records,
        "validations": validations,
    }


__all__ = (
    "ANALYZER_HEALTH_ARTIFACT_RELATIVE",
    "BUNDLE_ARTIFACT_RELATIVE",
    "COVERAGE_ARTIFACT_RELATIVE",
    "DEFAULT_CHANNEL_PROOF_REVISION",
    "DEFAULT_PRODUCER_CHANNEL",
    "EVIDENCE_ARTIFACT_RELATIVE",
    "EXHAUSTION_QUORUM_ARTIFACT_RELATIVE",
    "GATE_ARTIFACT_RELATIVE",
    "GOAL_COMPLETION_ARTIFACT_GAP_INTERFACE",
    "GOAL_COMPLETION_ARTIFACT_GAP_SCHEMA",
    "OBJECTIVE_COMPLETION_EVIDENCE_ARTIFACT_INTERFACE",
    "PRODUCING_TASK_ID",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_ASSEMBLER_INTERFACE",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_INTERFACE",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_BUNDLE_SCHEMA",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_INTERFACE",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_SCHEMA",
    "PROOF_TEST_REUSE_OBJECTIVE_EVIDENCE_VERSION",
    "GoalAssemblyIdentity",
    "GoalCompletionArtifactGap",
    "ObjectiveEvidenceGapKind",
    "ProofTestReuseObjectiveEvidenceAssembler",
    "ProofTestReuseObjectiveEvidenceBundle",
    "ProofTestReuseObjectiveEvidenceError",
    "assemble_objective_evidence",
    "atomic_write_json",
    "load_and_validate_written_artifacts",
    "replay_premise_blocks",
)

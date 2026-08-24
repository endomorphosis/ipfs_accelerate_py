"""Evidence-only consumer of the frozen proof-context runtime.

The harness is intentionally not a scheduler, scorer, approver, or release
qualification authority.  It makes one bounded observation per frozen task,
preserves the observed status/failure, and emits canonical JSON-ready records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.proof_context import EngineIdentities, RuntimeOptions, open_runtime
from ipfs_accelerate_py.proof_context.bootstrap import RUNTIME_CID
from ipfs_accelerate_py.proof_context.errors import ProofContextError, from_provider_error

from .experiment import EVIDENCE_KINDS, ExperimentPlan, SelfHostingTask, canonical_json, stable_id

EVIDENCE_SCHEMA: Final[str] = "ipfs-accelerate.self-hosting.v0.1/evidence"
ATTEMPT_SCHEMA: Final[str] = "ipfs-accelerate.self-hosting.v0.1/attempt"
FAILURE_SCHEMA: Final[str] = "ipfs-accelerate.self-hosting.v0.1/typed-failure"


@dataclass(frozen=True)
class TypedFailure:
    code: str
    status: str
    message: str

    def to_mapping(self) -> dict[str, str]:
        return {"schema": FAILURE_SCHEMA, "code": self.code, "status": self.status, "message": self.message[:240]}


@dataclass(frozen=True)
class AttemptEvidence:
    attempt_id: str
    task_id: str
    status: str
    provenance: str
    identities: Mapping[str, str]
    engine_record: Mapping[str, Any] | None
    failure: TypedFailure | None
    worktree: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": ATTEMPT_SCHEMA, "attempt_id": self.attempt_id, "task_id": self.task_id, "status": self.status, "provenance": self.provenance, "identities": dict(self.identities), "engine_record": dict(self.engine_record) if self.engine_record else None, "failure": self.failure.to_mapping() if self.failure else None, "worktree": dict(self.worktree)}
        result["evidence_id"] = stable_id("self-hosting-attempt", result)
        return result


class SelfHostingQualificationHarness:
    """Run frozen tasks through the public runtime without decision authority."""

    def __init__(self, plan: ExperimentPlan, repository: str | Path | None = None, *, worktree_parent: str | Path | None = None) -> None:
        self.plan = plan
        if repository is None and plan.evidence_kind != "replayed":
            raise ValueError("repository is required for live and simulated plans")
        self.repository = Path(repository) if repository is not None else None
        self.worktree_parent = Path(worktree_parent) if worktree_parent is not None else None
        if plan.engine_id != RUNTIME_CID:
            raise ValueError("engine_id must bind the installed proof-context runtime")

    def run(self) -> dict[str, Any]:
        attempts = [self._run_task(task).to_mapping() for task in self.plan.tasks]
        # The explicit kind lets aggregation consumers reject replay/simulation
        # without inferring provenance from individual records.  This surface
        # deliberately reports observations only; it never derives a result.
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "evidence_kind": self.plan.evidence_kind,
            "plan": self.plan.to_mapping(),
            "attempts": attempts,
            "qualification": None,
            "not_a_qualification": True,
            "authority": {
                "execution": False,
                "qualification": False,
                "self_approval": False,
            },
        }
        evidence["evidence_id"] = stable_id("self-hosting-evidence", evidence)
        return evidence

    def _run_task(self, task: SelfHostingTask) -> AttemptEvidence:
        engine_identities = self._engine_identities(task)
        identities = self._evidence_identities(task, engine_identities)
        attempt_id = stable_id("self-hosting-attempt-key", {"plan_id": self.plan.plan_id, "task_id": task.task_id})
        if self.plan.evidence_kind == "replayed":
            record = dict(task.replay_record or {})
            return AttemptEvidence(
                attempt_id,
                task.task_id,
                str(record.get("status", "unavailable")),
                "replayed",
                identities,
                record,
                self._record_failure(record),
                {"disposable": False, "reason": "replay-observation"},
            )
        options = RuntimeOptions(worktree_parent=self.worktree_parent)
        mode = "supervised"
        if self.plan.evidence_kind == "simulated":
            mode = "simulation"
            options = RuntimeOptions(worktree_parent=self.worktree_parent, fail_provenance="simulated")
        try:
            # ``repository`` is guaranteed above for all modes that execute.
            bundle = open_runtime(self.repository, identities=EngineIdentities(**engine_identities), mode=mode, options=options)
            record = bundle.engine.run(dict(task.proposal))
            record_mapping = dict(record.to_mapping())
            worktree = dict(record_mapping.get("payload", {}).get("worktree", {}))
            # Discard is a runtime-owned cleanup operation; it never accepts a patch.
            cleanup = bundle.session.worktree.discard(bundle.session.lifecycle_identities, bundle.session.repository)
            worktree["cleanup"] = dict(cleanup)
            return AttemptEvidence(
                attempt_id,
                task.task_id,
                record.status,
                self.plan.evidence_kind,
                identities,
                record_mapping,
                self._record_failure(record_mapping),
                worktree,
            )
        except Exception as exc:
            typed = self._typed_failure(exc)
            return AttemptEvidence(attempt_id, task.task_id, typed.status, self.plan.evidence_kind, identities, None, typed, {"disposable": True, "cleanup": {"discarded": False, "reason": "runtime-not-opened"}})

    def _engine_identities(self, task: SelfHostingTask) -> dict[str, str]:
        base = {"plan_id": self.plan.plan_id, "task_id": task.task_id, "repository_state_cid": self.plan.repository_state_cid, "configuration_cid": self.plan.configuration_cid}
        return {"repository_id": self.plan.repository_id, "repository_state_cid": self.plan.repository_state_cid, "task_id": task.task_id, "run_id": stable_id("self-hosting-run", base), "trace_id": stable_id("self-hosting-trace", base)}

    def _evidence_identities(
        self, task: SelfHostingTask, engine_identities: Mapping[str, str]
    ) -> dict[str, str]:
        """Carry every input binding on every individual observation."""
        return {
            **dict(engine_identities),
            "engine_id": self.plan.engine_id,
            "package_id": self.plan.package_id,
            "package_identity": self.plan.package_identity,
            "configuration_id": self.plan.configuration_id,
            "configuration_cid": self.plan.configuration_cid,
            "task_specification_cid": task.task_specification_cid,
            "plan_id": self.plan.plan_id,
        }

    @staticmethod
    def _typed_failure(exc: Exception) -> TypedFailure:
        if isinstance(exc, ProofContextError):
            return TypedFailure(str(getattr(exc, "code", "infrastructure_failure")), str(getattr(exc, "status", "rejected")), str(exc))
        converted = from_provider_error(exc)
        return TypedFailure(str(converted.code), str(converted.status), str(converted))

    @staticmethod
    def _record_failure(record: Mapping[str, Any]) -> TypedFailure | None:
        """Preserve a lifecycle-declared non-success as typed attempt evidence."""
        status = str(record.get("status", "invalid"))
        if status == "succeeded":
            return None
        payload = record.get("payload")
        details = payload if isinstance(payload, Mapping) else {}
        return TypedFailure(
            str(details.get("error") or status),
            status,
            str(details.get("reason") or f"governed lifecycle returned {status}"),
        )


def canonical_evidence_json(evidence: Mapping[str, Any]) -> str:
    """Serialize an evidence envelope deterministically for JSONL consumers."""
    return canonical_json(evidence)


def is_evidence_envelope(value: Any) -> bool:
    """Validate envelope shape without interpreting it as a qualification.

    This is intentionally structural: downstream qualification owners retain
    all policy decisions, including whether a live observation is sufficient.
    """
    if not isinstance(value, Mapping):
        return False
    kind = value.get("evidence_kind")
    attempts = value.get("attempts")
    authority = value.get("authority")
    if (
        value.get("schema") != EVIDENCE_SCHEMA
        or kind not in EVIDENCE_KINDS
        or value.get("qualification") is not None
        or value.get("not_a_qualification") is not True
        or not isinstance(attempts, list)
        or not attempts
        or not isinstance(authority, Mapping)
    ):
        return False
    if any(authority.get(name) is not False for name in ("execution", "qualification", "self_approval")):
        return False
    return all(
        isinstance(attempt, Mapping)
        and attempt.get("schema") == ATTEMPT_SCHEMA
        and attempt.get("provenance") == kind
        and isinstance(attempt.get("identities"), Mapping)
        and isinstance(attempt.get("attempt_id"), str)
        and isinstance(attempt.get("task_id"), str)
        for attempt in attempts
    )

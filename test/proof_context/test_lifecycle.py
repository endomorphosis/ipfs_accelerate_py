"""PCCE-021: governed patch lifecycle coordinator."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import pytest

from ipfs_accelerate_py.proof_context.compatibility import CompatibilityError
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    ProofContextError,
)
from ipfs_accelerate_py.proof_context.lifecycle import (
    APPLY_STAGE,
    CHECKPOINT_SCHEMA,
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_VERSION,
    DISPOSITION_STAGE,
    LIFECYCLE_CID,
    LIFECYCLE_DESCRIPTOR,
    LIFECYCLE_RECORD_SCHEMA,
    LIFECYCLE_SCHEMA,
    MODES,
    PCCE_006_CONTENT_ID,
    POLICY_CID,
    PROTECTED_REFS,
    PROVIDER_BOUND,
    PUBLICATION_REQUIREMENTS,
    RESULT_STATE_CID,
    SCHEMA,
    SEAL_STAGE,
    SIBLING_LAYOUT_REQUIRED,
    SOLE_AUTHORITY,
    STAGE_ARTIFACT_SCHEMA,
    STAGE_CONTRACTS,
    STAGES,
    STATUSES,
    STOP_PUBLICATION_STATUSES,
    VERIFY_STAGE,
    GovernanceReceipts,
    LifecycleIdentities,
    LifecyclePorts,
    PatchLifecycle,
    StageArtifact,
    admit_stage,
    frozen_lifecycle,
    lifecycle_cid,
    lifecycle_descriptor,
    mint_lifecycle_cid,
)
from ipfs_accelerate_py.proof_context.policy import POLICY_CID as BOUND_POLICY_CID
from ipfs_accelerate_py.proof_context.results import (
    RESULT_STATE_CID as BOUND_RESULT_STATE_CID,
    TERMINAL_STATUSES,
)

VALID_CID = "bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4"
CANONICAL_HEAD = "b" + "d" * 58


def _cid(label: str) -> str:
    body = "".join(ch if ch in "abcdefghijklmnopqrstuvwxyz234567" else "a" for ch in label)
    return "b" + (body + "a" * 58)[:58]


def _identities(**overrides: Any) -> LifecycleIdentities:
    values = {
        "operator_id": "operator-pcce-021",
        "repository_id": "example/ordinary-python-repo",
        "repository_state_cid": VALID_CID,
        "task_id": "PCCE-021",
        "run_id": "run-pcce-021",
        "trace_id": "trace-pcce-021",
        "contract_version": CONTRACT_VERSION,
        "patch_id": None,
        "artifact_id": None,
        "evidence_cid": None,
        "lease_id": None,
        "fence_id": None,
        "worktree_id": None,
    }
    values.update(overrides)
    return LifecycleIdentities(**values)


def _artifact(
    stage: str,
    identities: LifecycleIdentities,
    *,
    status: str = "succeeded",
    provenance: str = "live",
    payload: Mapping[str, Any] | None = None,
    inbound_cid: str | None = None,
    error: str | None = None,
) -> StageArtifact:
    return StageArtifact(
        schema=STAGE_ARTIFACT_SCHEMA,
        stage=stage,
        status=status,
        identities=identities,
        artifact_cid=_cid(stage),
        provenance=provenance,
        payload=payload or {},
        inbound_cid=inbound_cid,
        error=error,
    )


def _ordinary_python_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    package = root / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "main").write_text("deadbeef" * 5 + "\n", encoding="utf-8")
    return root


@dataclass
class FakeStagePort:
    identities: LifecycleIdentities
    stage: str
    calls: list[str]
    fail_at: str | None = None
    fail_status: str = "rejected"
    fail_error: str | None = None
    extra_payload: Mapping[str, Any] = field(default_factory=dict)
    inbound: str | None = None

    def _emit(self, stage: str, identities: LifecycleIdentities) -> StageArtifact:
        self.calls.append(stage)
        status = "succeeded"
        error = None
        payload: dict[str, Any] = dict(self.extra_payload)
        bound = identities
        if stage == "proposal":
            bound = LifecycleIdentities.from_mapping(
                {**identities.to_mapping(), "patch_id": identities.patch_id or _cid("patch")}
            )
            payload.setdefault("declared_files", ["src/demo/__init__.py"])
            payload.setdefault("adapter_id", "adapter-a")
            payload.setdefault("approver_id", "coordinator")
        if stage == APPLY_STAGE:
            payload.setdefault("disposable", True)
            payload.setdefault("canonical_mutated", False)
            payload.setdefault("canonical_head", CANONICAL_HEAD)
            payload.setdefault("worktree_id", _cid("worktree"))
            payload.setdefault("target_ref", "pcce-disposable")
        if stage == VERIFY_STAGE:
            payload.setdefault("planner_authority", "canonical")
            payload.setdefault("selected_independently", False)
        if stage == SEAL_STAGE:
            payload.setdefault("seal_cid", _cid("sealcid"))
            payload.setdefault("sealed", True)
        if stage == DISPOSITION_STAGE:
            payload.setdefault("seal_cid", _cid("sealcid"))
            payload.setdefault("sealed", True)
        if self.fail_at == stage:
            status = self.fail_status
            error = self.fail_error
            if status == "succeeded":
                error = None
        return _artifact(
            stage,
            bound,
            status=status,
            error=error,
            payload=payload,
            inbound_cid=self.inbound,
        )


@dataclass
class FakeOperatorPort(FakeStagePort):
    def identify(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeRepositoryPort(FakeStagePort):
    def resolve(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeSemanticPort:
    identities: LifecycleIdentities
    calls: list[str]
    fail_at: str | None = None
    fail_status: str = "rejected"
    fail_error: str | None = None

    def _port(self, stage: str) -> FakeStagePort:
        return FakeStagePort(
            self.identities,
            stage,
            self.calls,
            fail_at=self.fail_at,
            fail_status=self.fail_status,
            fail_error=self.fail_error,
        )

    def scan(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("scan-semantic")._emit("scan-semantic", identities)

    def invalidate(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("invalidate")._emit("invalidate", identities)

    def context_pack(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("context-pack")._emit("context-pack", identities)

    def sufficiency(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("sufficiency")._emit("sufficiency", identities)

    def impact(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("impact")._emit("impact", identities)

    def escalate(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._port("escalate")._emit("escalate", identities)


@dataclass
class FakeRoutePort(FakeStagePort):
    def route(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeProposalPort(FakeStagePort):
    def propose(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeScopePort(FakeStagePort):
    def check(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeWorktreePort(FakeStagePort):
    discarded: bool = True
    discard_calls: list[str] = field(default_factory=list)

    def apply(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._emit(self.stage, identities)

    def discard(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.discard_calls.append("discard")
        return {"discarded": self.discarded, "worktree_id": identities.worktree_id}


@dataclass
class FakeVerificationPort(FakeStagePort):
    def verify(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeAssurancePort(FakeStagePort):
    def assure(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeSealingPort(FakeStagePort):
    def seal(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeDispositionPort(FakeStagePort):
    def decide(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._emit(self.stage, identities)


@dataclass
class FakeGovernancePort:
    identities: LifecycleIdentities
    calls: list[str]
    halt_status: str | None = None
    halt_before: str | None = None
    lease_valid: bool = True
    fence_valid: bool = True
    schedule_admitted: bool = True
    _checked: int = 0

    def acquire_lease(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.calls.append("lease")
        return {
            "lease_id": _cid("lease"),
            "valid": self.lease_valid,
            "receipt_cid": _cid("leasercpt"),
        }

    def acquire_fence(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.calls.append("fence")
        return {
            "fence_id": _cid("fence"),
            "valid": self.fence_valid,
            "receipt_cid": _cid("fencercpt"),
        }

    def admit_schedule(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.calls.append("schedule")
        return {
            "admitted": self.schedule_admitted,
            "status": "succeeded" if self.schedule_admitted else "unavailable",
            "receipt_cid": _cid("sched"),
        }

    def check_cancellation(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.calls.append("cancel-check")
        self._checked += 1
        if self.halt_status and (self.halt_before is None or self._checked >= 1):
            # Halt on the first stage check when halt_before is None; otherwise
            # the coordinator maps halt_before via caller-provided fail_at.
            if self.halt_before is None:
                return {"status": self.halt_status, "error": self.halt_status}
        return {"status": "succeeded"}


@dataclass
class FakePersistencePort:
    identities: LifecycleIdentities
    records: list[Mapping[str, Any]] = field(default_factory=list)

    def persist(
        self,
        artifact: StageArtifact | Mapping[str, Any],
        *,
        published: bool,
    ) -> Mapping[str, Any]:
        payload = (
            artifact.to_mapping()
            if hasattr(artifact, "to_mapping")
            else dict(artifact)
        )
        evidence_cid = _cid(f"evidence{len(self.records)}")
        record = {
            "published": published,
            "evidence_cid": evidence_cid,
            "payload": payload,
        }
        self.records.append(record)
        return {"evidence_cid": evidence_cid, "published": published}


@dataclass
class HaltGovernance(FakeGovernancePort):
    halt_stage: str | None = None
    _stage_index: int = 0

    def check_cancellation(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.calls.append("cancel-check")
        current = STAGES[self._stage_index] if self._stage_index < len(STAGES) else STAGES[-1]
        self._stage_index += 1
        if self.halt_status and current == self.halt_stage:
            error = self.halt_status if self.halt_status in {
                "timeout",
                "cancelled",
                "unavailable_capability",
            } else self.halt_status
            if self.halt_status == "unavailable":
                error = "unavailable_capability"
            return {"status": self.halt_status, "error": error}
        return {"status": "succeeded"}


def _ports(
    identities: LifecycleIdentities,
    *,
    fail_at: str | None = None,
    fail_status: str = "rejected",
    fail_error: str | None = None,
    discarded: bool = True,
    extra_payload: Mapping[str, str | bool] | None = None,
    halt_status: str | None = None,
    halt_stage: str | None = None,
    lease_valid: bool = True,
    fence_valid: bool = True,
    **overrides: Any,
) -> LifecyclePorts:
    calls: list[str] = []
    payload = dict(extra_payload or {})
    values = {
        "operator": FakeOperatorPort(
            identities, "identify-operator", calls, fail_at, fail_status, fail_error, payload
        ),
        "repository": FakeRepositoryPort(
            identities, "resolve-repository", calls, fail_at, fail_status, fail_error, payload
        ),
        "semantic": FakeSemanticPort(identities, calls, fail_at, fail_status, fail_error),
        "route": FakeRoutePort(
            identities, "route", calls, fail_at, fail_status, fail_error, payload
        ),
        "proposal": FakeProposalPort(
            identities, "proposal", calls, fail_at, fail_status, fail_error, payload
        ),
        "scope": FakeScopePort(
            identities, "scope-check", calls, fail_at, fail_status, fail_error, payload
        ),
        "worktree": FakeWorktreePort(
            identities,
            APPLY_STAGE,
            calls,
            fail_at,
            fail_status,
            fail_error,
            payload,
            discarded=discarded,
        ),
        "verification": FakeVerificationPort(
            identities, VERIFY_STAGE, calls, fail_at, fail_status, fail_error, payload
        ),
        "assurance": FakeAssurancePort(
            identities, "assurance", calls, fail_at, fail_status, fail_error, payload
        ),
        "sealing": FakeSealingPort(
            identities, SEAL_STAGE, calls, fail_at, fail_status, fail_error, payload
        ),
        "disposition": FakeDispositionPort(
            identities, DISPOSITION_STAGE, calls, fail_at, fail_status, fail_error, payload
        ),
        "governance": HaltGovernance(
            identities,
            calls,
            halt_status=halt_status,
            halt_stage=halt_stage,
            lease_valid=lease_valid,
            fence_valid=fence_valid,
        ),
        "persistence": FakePersistencePort(identities),
    }
    values.update(overrides)
    return LifecyclePorts(**values)


def _open(
    tmp_path: Path,
    *,
    mode: str = "production",
    identities: LifecycleIdentities | None = None,
    **port_kwargs: Any,
) -> tuple[PatchLifecycle, LifecyclePorts, Path]:
    repo = _ordinary_python_repo(tmp_path)
    bound = identities or _identities()
    ports = _ports(bound, **port_kwargs)
    engine = PatchLifecycle.open(repo, ports=ports, identities=bound, mode=mode)
    return engine, ports, repo


def test_cold_import_creates_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    imported = importlib.import_module("ipfs_accelerate_py.proof_context.lifecycle")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert imported.SCHEMA == SCHEMA
    assert imported.SOLE_AUTHORITY is True
    assert imported.PROVIDER_BOUND is False
    assert imported.SIBLING_LAYOUT_REQUIRED is False


def test_frozen_lifecycle_is_exact_ordered_sequence() -> None:
    assert STAGES == (
        "identify-operator",
        "resolve-repository",
        "scan-semantic",
        "invalidate",
        "context-pack",
        "sufficiency",
        "route",
        "proposal",
        "scope-check",
        "isolated-apply",
        "impact",
        "incremental-verify",
        "escalate",
        "assurance",
        "seal",
        "disposition",
    )
    assert len(STAGES) == 16
    assert len(STAGES) == len(set(STAGES))
    assert tuple(STAGE_CONTRACTS) == STAGES
    for stage, contract in STAGE_CONTRACTS.items():
        assert contract.startswith("pcce/proof-context/v0.1/")
        assert admit_stage(stage) == stage
    with pytest.raises(TypeError):
        STAGES[0] = "skip-seal"  # type: ignore[index]
    with pytest.raises(ProofContextError):
        admit_stage("skip-seal")
    assert SOLE_AUTHORITY is True
    assert "main" in PROTECTED_REFS
    assert "unsealed" not in PUBLICATION_REQUIREMENTS or "sealed" in PUBLICATION_REQUIREMENTS
    assert tuple(STOP_PUBLICATION_STATUSES) == tuple(
        status for status in STATUSES if status != "succeeded"
    )


def test_success_path_traverses_every_stage_and_publishes_only_after_seal(
    tmp_path: Path,
) -> None:
    engine, ports, repo = _open(tmp_path)
    before = (repo / ".git" / "HEAD").read_text(encoding="utf-8")
    init_text = (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8")
    record = engine.run({"declared_files": ["src/demo/__init__.py"]})
    assert record.stages == STAGES
    assert [artifact.stage for artifact in record.artifacts] == list(STAGES)
    assert record.status == "succeeded"
    assert record.published is True
    assert record.sealed is True
    assert record.accepted is True
    assert record.mode == "production"
    assert record.provenance == "live"
    assert record.identities.operator_id == "operator-pcce-021"
    assert record.identities.task_id == "PCCE-021"
    assert record.identities.run_id == "run-pcce-021"
    assert record.identities.trace_id == "trace-pcce-021"
    assert record.identities.repository_id == "example/ordinary-python-repo"
    assert record.identities.patch_id is not None
    assert record.identities.evidence_cid is not None
    assert record.identities.lease_id is not None
    assert record.identities.fence_id is not None
    assert record.governance.lease["valid"] is True
    assert record.governance.fence["valid"] is True
    assert record.governance.worktree["disposable"] is True
    assert record.governance.worktree["canonical_mutated"] is False
    inbound = None
    for artifact in record.artifacts:
        assert artifact.identities.operator_id == record.identities.operator_id
        assert artifact.identities.run_id == record.identities.run_id
        assert artifact.identities.trace_id == record.identities.trace_id
        assert artifact.inbound_cid == inbound
        assert artifact.artifact_cid.startswith("b")
        inbound = artifact.artifact_cid
    assert record.to_mapping()["lifecycle_cid"] == LIFECYCLE_CID
    assert (repo / ".git" / "HEAD").read_text(encoding="utf-8") == before
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == init_text
    persisted_published = [item for item in ports.persistence.records if item["published"] is True]
    assert persisted_published
    assert ports.worktree.discard_calls == []


def test_supervised_mode_is_live_and_identity_bound(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path, mode="supervised")
    record = engine.run()
    assert record.mode == "supervised"
    assert record.published is True
    assert record.accepted is True
    assert record.identities.operator_id == "operator-pcce-021"


@pytest.mark.parametrize(
    "status,stage,error",
    [
        ("rejected", "scope-check", "boundary_violation"),
        ("verification_failed", "incremental-verify", "verification_failed"),
        ("proof_failed", "incremental-verify", "proof_failed"),
        ("assurance_failed", "assurance", "assurance_failed"),
        ("context_insufficient", "sufficiency", "context_insufficient"),
        ("model_escalation_required", "escalate", "context_insufficient"),
        ("human_review_required", "disposition", "human_review_required"),
        ("invalid", "identify-operator", "malformed"),
        ("stale", "resolve-repository", "stale_root"),
        ("infrastructure_failure", "route", "infrastructure_failure"),
        ("repair_required", "impact", "repair_required"),
    ],
)
def test_non_success_stops_publication_and_persists_evidence(
    tmp_path: Path,
    status: str,
    stage: str,
    error: str,
) -> None:
    engine, ports, _repo = _open(
        tmp_path,
        fail_at=stage,
        fail_status=status,
        fail_error=error,
    )
    record = engine.run()
    assert record.published is False
    assert record.accepted is False
    assert record.status == status
    assert record.error is not None
    assert stage in record.stages
    assert DISPOSITION_STAGE not in record.stages or stage == DISPOSITION_STAGE
    assert any(item["published"] is False for item in ports.persistence.records)
    assert all(item["published"] is False for item in ports.persistence.records)
    mapping = record.to_mapping()
    assert mapping["published"] is False
    assert mapping["trace"]


@pytest.mark.parametrize("status", ("timeout", "cancelled", "unavailable"))
def test_timeout_cancellation_and_unavailable_stop_publication(
    tmp_path: Path,
    status: str,
) -> None:
    engine, ports, _repo = _open(
        tmp_path,
        halt_status=status,
        halt_stage="scan-semantic",
    )
    record = engine.run()
    assert record.published is False
    assert record.accepted is False
    assert record.status == status
    assert "scan-semantic" in record.stages or record.stages[-1] == "scan-semantic"
    assert any(item["published"] is False for item in ports.persistence.records)
    assert "scan-semantic" not in ports.semantic.calls


def test_concealed_partial_effect_flag_is_rejected(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.worktree.apply

    def _conceal(
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        artifact = original(identities, repository, proposal)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={**dict(artifact.payload), "concealed_partial_effect": True},
            inbound_cid=artifact.inbound_cid,
        )

    ports.worktree.apply = _conceal  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "boundary_violation"


def test_partial_effect_is_not_concealed_and_is_not_published(tmp_path: Path) -> None:
    engine, ports, _repo = _open(
        tmp_path,
        fail_at="assurance",
        fail_status="assurance_failed",
        fail_error="assurance_failed",
        discarded=False,
    )
    record = engine.run()
    assert record.published is False
    assert record.status == "partial_effect"
    assert record.error == "partial_effect"
    assert APPLY_STAGE in record.stages
    assert "assurance" in record.stages
    assert ports.worktree.discard_calls == ["discard"]
    assert any(item["published"] is False for item in ports.persistence.records)
    assert record.payload["applied"] is True


def test_bypass_skip_and_start_at_are_rejected(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path)
    with pytest.raises(BoundaryViolationError):
        engine.run({"skip_stages": ["assurance", "seal"]})
    with pytest.raises(BoundaryViolationError):
        engine.run({"start_at": "seal"})
    with pytest.raises(BoundaryViolationError):
        engine.run({"bypass": True})
    with pytest.raises(BoundaryViolationError):
        engine.run({"self_approved": True})
    methods = {
        name
        for name, _value in inspect.getmembers(PatchLifecycle, predicate=inspect.isfunction)
    }
    assert "skip" not in methods
    assert "run_from" not in methods
    assert "bypass" not in methods


def test_checkpoint_gap_is_rejected_as_bypass(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path)
    identities = _identities()
    with pytest.raises(BoundaryViolationError):
        engine.resume(
            {
                "schema": CHECKPOINT_SCHEMA,
                "mode": "production",
                "identities": dict(identities.to_mapping()),
                "completed": [
                    dict(_artifact("seal", identities).to_mapping()),
                ],
            }
        )


def test_incomplete_checkpoint_cannot_claim_publication(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path)
    identities = _identities()
    with pytest.raises(BoundaryViolationError):
        engine.resume(
            {
                "schema": CHECKPOINT_SCHEMA,
                "mode": "production",
                "identities": dict(identities.to_mapping()),
                "published": True,
                "completed": [
                    dict(_artifact("identify-operator", identities).to_mapping()),
                ],
            }
        )


def test_resume_reuses_completed_prefix_and_does_not_reinvoke(
    tmp_path: Path,
) -> None:
    engine, ports, _repo = _open(tmp_path)
    first = engine.run()
    completed = [dict(artifact.to_mapping()) for artifact in first.artifacts[:6]]
    engine2, ports2, _repo2 = _open(tmp_path / "resume")
    resumed = engine2.resume(
        {
            "schema": CHECKPOINT_SCHEMA,
            "mode": "production",
            "identities": dict(first.identities.to_mapping()),
            "governance": dict(first.governance.to_mapping()),
            "completed": completed,
        }
    )
    assert resumed.published is True
    assert resumed.stages == STAGES
    assert "identify-operator" not in ports2.operator.calls
    assert "scan-semantic" not in ports2.semantic.calls
    assert "route" in ports2.route.calls
    assert first.identities.trace_id == resumed.identities.trace_id


def test_adapter_self_approval_is_rejected(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.proposal.propose

    def _self_approve(
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        artifact = original(identities, repository, proposal)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={**dict(artifact.payload), "adapter_id": "adapter-a", "approver_id": "adapter-a"},
            inbound_cid=artifact.inbound_cid,
        )

    ports.proposal.propose = _self_approve  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.accepted is False
    assert record.status in {"rejected", "invalid"} or record.error == "boundary_violation"


def test_canonical_branch_mutation_is_rejected(tmp_path: Path) -> None:
    engine, ports, repo = _open(tmp_path)
    before = (repo / ".git" / "HEAD").read_text(encoding="utf-8")
    original = ports.worktree.apply

    def _mutate(
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        artifact = original(identities, repository, proposal)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={
                **dict(artifact.payload),
                "canonical_mutated": True,
                "target_ref": "main",
                "disposable": False,
            },
            inbound_cid=artifact.inbound_cid,
        )

    ports.worktree.apply = _mutate  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "boundary_violation"
    assert (repo / ".git" / "HEAD").read_text(encoding="utf-8") == before


def test_independent_test_selection_is_rejected(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.verification.verify

    def _independent(
        identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact:
        artifact = original(identities, repository)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={
                **dict(artifact.payload),
                "selected_independently": True,
                "planner_authority": "adapter",
            },
            inbound_cid=artifact.inbound_cid,
        )

    ports.verification.verify = _independent  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "boundary_violation"


def test_unsealed_production_patch_is_not_published(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.sealing.seal

    def _unsealed(
        identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact:
        artifact = original(identities, repository)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={"sealed": False, "seal_cid": None, "unsealed": True},
            inbound_cid=artifact.inbound_cid,
        )

    ports.sealing.seal = _unsealed  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.accepted is False


def test_production_rejects_simulated_stage_provenance(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.route.route

    def _simulated(
        identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact:
        artifact = original(identities, repository)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance="simulated",
            payload=artifact.payload,
            inbound_cid=artifact.inbound_cid,
        )

    ports.route.route = _simulated  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "simulated_promoted"


def test_simulated_run_cannot_be_published(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path, mode="simulation")
    record = engine.run()
    assert record.published is False
    assert record.accepted is False
    assert record.status == "simulated"
    assert record.provenance == "simulated"
    assert record.mode == "simulation"


def test_evaluation_completes_without_production_publication(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path, mode="evaluation")
    record = engine.run()
    assert record.mode == "evaluation"
    assert record.published is False
    assert record.accepted is False
    assert record.stages == STAGES
    assert record.sealed is True


def test_identity_drift_is_rejected(tmp_path: Path) -> None:
    identities = _identities()
    engine, ports, _repo = _open(tmp_path, identities=identities)
    original = ports.operator.identify

    def _drift(bound: LifecycleIdentities, repository: Path) -> StageArtifact:
        artifact = original(bound, repository)
        drifted = _identities(task_id="OTHER-TASK")
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status=artifact.status,
            identities=drifted,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload=artifact.payload,
        )

    ports.operator.identify = _drift  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "identity_inconsistent"


def test_production_rejects_mock_ports(tmp_path: Path) -> None:
    identities = _identities()
    mock_ports = LifecyclePorts(
        operator=Mock(),
        repository=Mock(),
        semantic=Mock(),
        route=Mock(),
        proposal=Mock(),
        scope=Mock(),
        worktree=Mock(),
        verification=Mock(),
        assurance=Mock(),
        sealing=Mock(),
        disposition=Mock(),
        governance=Mock(),
        persistence=Mock(),
    )
    with pytest.raises((CompatibilityError, ProofContextError)):
        PatchLifecycle.open(
            tmp_path,
            ports=mock_ports,
            identities=identities,
            mode="production",
        )


def test_pseudo_cid_identities_are_rejected() -> None:
    with pytest.raises(ProofContextError):
        _identities(repository_state_cid="sha256:deadbeef")


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    identities = _identities()
    with pytest.raises(ProofContextError):
        PatchLifecycle.open(
            tmp_path,
            ports=_ports(identities),
            identities=identities,
            mode="shadow",
        )


def test_lease_and_fence_receipts_are_required(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path, lease_valid=False)
    record = engine.run()
    assert record.published is False
    assert record.accepted is False
    assert record.status == "unavailable"
    assert record.error == "unavailable_capability"
    assert any(item["published"] is False for item in ports.persistence.records)


def test_stage_transition_trace_is_complete_and_ordered(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path)
    record = engine.run()
    mapping = record.to_mapping()
    trace = mapping["trace"]
    assert [item["stage"] for item in trace] == list(STAGES)
    for item in trace:
        assert item["schema"] == STAGE_ARTIFACT_SCHEMA
        assert item["contract"] == STAGE_CONTRACTS[item["stage"]]
        for field in (
            "operator_id",
            "repository_id",
            "task_id",
            "run_id",
            "trace_id",
        ):
            assert item["identities"][field]
    checkpoint = record.to_checkpoint()
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["published"] is True
    assert [item["stage"] for item in checkpoint["completed"]] == list(STAGES)


def test_every_terminal_status_is_exercised(tmp_path: Path) -> None:
    seen: set[str] = set()
    success, _ports, _repo = _open(tmp_path / "ok")
    seen.add(success.run().status)
    mapping = {
        "rejected": ("scope-check", "boundary_violation"),
        "verification_failed": ("incremental-verify", "verification_failed"),
        "proof_failed": ("incremental-verify", "proof_failed"),
        "assurance_failed": ("assurance", "assurance_failed"),
        "cancelled": None,
        "invalid": ("identify-operator", "malformed"),
        "stale": ("resolve-repository", "stale_root"),
        "simulated": None,
    }
    for status, spec in mapping.items():
        if status == "cancelled":
            engine, _p, _r = _open(
                tmp_path / status,
                halt_status="cancelled",
                halt_stage="route",
            )
            seen.add(engine.run().status)
            continue
        if status == "simulated":
            engine, _p, _r = _open(tmp_path / status, mode="simulation")
            seen.add(engine.run().status)
            continue
        assert spec is not None
        engine, _p, _r = _open(
            tmp_path / status,
            fail_at=spec[0],
            fail_status=status,
            fail_error=spec[1],
        )
        seen.add(engine.run().status)
    required = {
        "succeeded",
        "rejected",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "cancelled",
        "invalid",
        "stale",
        "simulated",
    }
    assert required <= seen
    assert set(TERMINAL_STATUSES) <= seen


def test_descriptor_cid_is_stable_and_bound() -> None:
    descriptor = lifecycle_descriptor()
    assert descriptor is LIFECYCLE_DESCRIPTOR
    assert descriptor["schema"] == LIFECYCLE_SCHEMA
    assert descriptor["cid"] == LIFECYCLE_CID
    assert lifecycle_cid() == LIFECYCLE_CID
    assert LIFECYCLE_CID.startswith("b")
    body = {key: value for key, value in descriptor.items() if key != "cid"}
    assert mint_lifecycle_cid(body) == LIFECYCLE_CID
    frozen = frozen_lifecycle()
    assert frozen["stages"] == STAGES
    assert frozen["sole_authority"] is True
    assert frozen["pcce_006_content_id"] == PCCE_006_CONTENT_ID
    assert frozen["policy_cid"] == POLICY_CID == BOUND_POLICY_CID
    assert frozen["result_state_cid"] == RESULT_STATE_CID == BOUND_RESULT_STATE_CID
    assert COMPATIBILITY_MATRIX_CONTENT_ID.endswith("e920")
    digest = hashlib.sha256(LIFECYCLE_CID.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    with pytest.raises(TypeError):
        descriptor["stages"] = ()  # type: ignore[index]


def test_provider_neutral_ast_and_no_sibling_imports() -> None:
    source = Path(inspect.getfile(PatchLifecycle)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "openai" not in imported
    assert "anthropic" not in imported
    assert "ipfs_datasets_py" not in imported
    assert "ipfs_kit_py" not in imported
    assert PROVIDER_BOUND is False
    assert SIBLING_LAYOUT_REQUIRED is False


def test_route_credentials_are_rejected(tmp_path: Path) -> None:
    engine, ports, _repo = _open(tmp_path)
    original = ports.route.route

    def _creds(identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        artifact = original(identities, repository)
        return StageArtifact(
            schema=artifact.schema,
            stage=artifact.stage,
            status="succeeded",
            identities=artifact.identities,
            artifact_cid=artifact.artifact_cid,
            provenance=artifact.provenance,
            payload={"tier": "small_local_model", "credentials": {"token": "secret"}},
            inbound_cid=artifact.inbound_cid,
        )

    ports.route.route = _creds  # type: ignore[method-assign]
    record = engine.run()
    assert record.published is False
    assert record.error == "boundary_violation"


def test_lifecycle_record_mapping_is_immutable(tmp_path: Path) -> None:
    engine, _ports, _repo = _open(tmp_path)
    record = engine.run()
    payload = record.to_mapping()
    assert isinstance(payload, MappingProxyType)
    with pytest.raises(TypeError):
        payload["published"] = False  # type: ignore[index]
    assert record.schema == LIFECYCLE_RECORD_SCHEMA


def test_malformed_ports_and_missing_repository_fail_closed(tmp_path: Path) -> None:
    identities = _identities()
    with pytest.raises(ProofContextError):
        PatchLifecycle.open(
            tmp_path / "missing",
            ports=_ports(identities),
            identities=identities,
        )
    with pytest.raises(ProofContextError):
        PatchLifecycle.open(
            tmp_path,
            ports=object(),  # type: ignore[arg-type]
            identities=identities,
        )


def test_context_insufficient_does_not_apply_or_publish(tmp_path: Path) -> None:
    engine, ports, _repo = _open(
        tmp_path,
        fail_at="sufficiency",
        fail_status="context_insufficient",
        fail_error="context_insufficient",
    )
    record = engine.run()
    assert record.status == "context_insufficient"
    assert record.published is False
    assert APPLY_STAGE not in record.stages
    assert APPLY_STAGE not in ports.worktree.calls


def test_verify_context_insufficient_reaches_escalate_then_stops_if_unresolved(
    tmp_path: Path,
) -> None:
    engine, _ports, _repo = _open(
        tmp_path,
        fail_at="incremental-verify",
        fail_status="context_insufficient",
        fail_error="context_insufficient",
    )
    record = engine.run()
    assert record.published is False
    assert VERIFY_STAGE in record.stages
    assert "escalate" in record.stages


def test_governance_receipts_round_trip() -> None:
    receipts = GovernanceReceipts(
        lease={"lease_id": _cid("lease"), "valid": True},
        fence={"fence_id": _cid("fence"), "valid": True},
        worktree={"worktree_id": _cid("worktree"), "disposable": True},
        schedule={"admitted": True},
    )
    mapping = receipts.to_mapping()
    assert mapping["lease"]["valid"] is True
    assert mapping["fence"]["valid"] is True
    with pytest.raises(TypeError):
        mapping["lease"] = {}  # type: ignore[index]


def test_import_does_not_read_promotion_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PCCE_MODE", "simulation")
    monkeypatch.chdir(tmp_path)
    imported = importlib.reload(
        importlib.import_module("ipfs_accelerate_py.proof_context.lifecycle")
    )
    assert imported.MODES == MODES
    assert "simulation" in imported.MODES
    assert os.environ.get("PCCE_MODE") == "simulation"
    engine, _ports, _repo = _open(tmp_path / "env")
    record = engine.run()
    assert record.mode == "production"
    assert record.published is True

"""PCCE-024: interruption recovery, idempotent resume, and fenced publication."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.errors import BoundaryViolationError, ProofContextError
from ipfs_accelerate_py.proof_context.lifecycle import (
    APPLY_STAGE,
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    DISPOSITION_STAGE,
    LIFECYCLE_CID,
    STAGE_ARTIFACT_SCHEMA,
    STAGES,
    LifecycleIdentities,
    LifecyclePorts,
    PatchLifecycle,
    StageArtifact,
)
from ipfs_accelerate_py.proof_context.policy import POLICY_CID
from ipfs_accelerate_py.proof_context.recovery import (
    BOUNDARIES,
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CRASH_MATRIX,
    CRASH_POSITIONS,
    EFFECTFUL_STAGES,
    FENCE_BOUNDARY,
    INFER_SUCCESS_FROM_PROCESS_EXIT,
    LEASE_BOUNDARY,
    PCCE_006_CONTENT_ID,
    PERSISTENCE_AUTHORITY,
    PROVIDER_BOUND,
    PUBLISH_BOUNDARY,
    RECOVERY_CID,
    RECOVERY_DESCRIPTOR,
    RECOVERY_RECORD_SCHEMA,
    RECOVERY_SCHEMA,
    RESTART_AWARE,
    RESULT_STATE_CID,
    SCHEMA,
    SEAL_STAGE,
    SECOND_WAL,
    SIBLING_LAYOUT_REQUIRED,
    SOLE_LIFECYCLE_AUTHORITY,
    TERMINAL_ADAPTER_STAGES,
    VALID_TERMINAL_STATUSES,
    VERIFY_STAGE,
    WAL_IMPLEMENTATION,
    AttemptIdentity,
    CrashInterrupt,
    FencedCheckpointStore,
    RecoveryCoordinator,
    StaleWriterError,
    admit_boundary,
    admit_position,
    crash_matrix,
    mint_idempotency_key,
    mint_recovery_cid,
    recovery_cid,
    recovery_descriptor,
    replay_trace,
)
from ipfs_accelerate_py.proof_context.results import RESULT_STATE_CID as BOUND_RESULT_STATE_CID

VALID_CID = "bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4"
CANONICAL_HEAD = "b" + "d" * 58


def _live_crash_interrupt() -> type[BaseException]:
    """Resolve CrashInterrupt after importlib.reload of recovery."""

    return importlib.import_module(
        "ipfs_accelerate_py.proof_context.recovery"
    ).CrashInterrupt


def _cid(label: str) -> str:
    body = "".join(ch if ch in "abcdefghijklmnopqrstuvwxyz234567" else "a" for ch in label)
    return "b" + (body + "a" * 58)[:58]


def _identities(**overrides: Any) -> LifecycleIdentities:
    values = {
        "operator_id": "operator-pcce-024",
        "repository_id": "example/ordinary-python-repo",
        "repository_state_cid": VALID_CID,
        "task_id": "PCCE-024",
        "run_id": "run-pcce-024",
        "trace_id": "trace-pcce-024",
        "contract_version": CONTRACT_VERSION,
        "patch_id": None,
        "artifact_id": None,
        "evidence_cid": None,
        "lease_id": _cid("lease"),
        "fence_id": _cid("fence"),
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
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir(exist_ok=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads").mkdir(parents=True, exist_ok=True)
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
    lease_valid: bool = True
    fence_valid: bool = True
    schedule_admitted: bool = True

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
        payload = artifact.to_mapping() if hasattr(artifact, "to_mapping") else dict(artifact)
        evidence_cid = _cid(f"evidence{len(self.records)}")
        self.records.append(
            {"published": published, "evidence_cid": evidence_cid, "payload": payload}
        )
        return {"evidence_cid": evidence_cid, "published": published}


class ManualClock:
    def __init__(self, now: int = 0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _ports(identities: LifecycleIdentities, **overrides: Any) -> LifecyclePorts:
    calls: list[str] = []
    values = {
        "operator": FakeOperatorPort(identities, "identify-operator", calls),
        "repository": FakeRepositoryPort(identities, "resolve-repository", calls),
        "semantic": FakeSemanticPort(identities, calls),
        "route": FakeRoutePort(identities, "route", calls),
        "proposal": FakeProposalPort(identities, "proposal", calls),
        "scope": FakeScopePort(identities, "scope-check", calls),
        "worktree": FakeWorktreePort(identities, APPLY_STAGE, calls),
        "verification": FakeVerificationPort(identities, VERIFY_STAGE, calls),
        "assurance": FakeAssurancePort(identities, "assurance", calls),
        "sealing": FakeSealingPort(identities, SEAL_STAGE, calls),
        "disposition": FakeDispositionPort(identities, DISPOSITION_STAGE, calls),
        "governance": FakeGovernancePort(identities, calls),
        "persistence": FakePersistencePort(identities),
    }
    values.update(overrides)
    return LifecyclePorts(**values)


def _attempt(
    identities: LifecycleIdentities | None = None,
    **overrides: Any,
) -> AttemptIdentity:
    bound = identities or _identities()
    values = {
        "attempt_id": "attempt-pcce-024",
        "writer_id": "writer-a",
        "writer_generation": 1,
        "fence_token": _cid("ftoken"),
        "lease_id": bound.lease_id or _cid("lease"),
        "fence_id": bound.fence_id or _cid("fence"),
        "identities": bound,
        "lease_expires_at": 2147483647,
    }
    values.update(overrides)
    return AttemptIdentity(**values)


def _open(
    tmp_path: Path,
    *,
    identities: LifecycleIdentities | None = None,
    attempt: AttemptIdentity | None = None,
    store: FencedCheckpointStore | None = None,
    clock: ManualClock | None = None,
    repo: Path | None = None,
    mode: str = "production",
    **port_kwargs: Any,
) -> tuple[RecoveryCoordinator, LifecyclePorts, FencedCheckpointStore, AttemptIdentity, Path]:
    root = repo if repo is not None else _ordinary_python_repo(tmp_path / "repo")
    bound = identities or (attempt.identities if attempt is not None else _identities())
    ports = _ports(bound, **port_kwargs)
    store = store or FencedCheckpointStore()
    attempt = attempt or _attempt(bound)
    engine = RecoveryCoordinator.open(
        root,
        ports=ports,
        identities=bound,
        attempt=attempt,
        store=store,
        mode=mode,
        clock=clock,
    )
    return engine, ports, store, attempt, root


def test_cold_import_creates_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    imported = importlib.import_module("ipfs_accelerate_py.proof_context.recovery")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert imported.SCHEMA == SCHEMA
    assert imported.SECOND_WAL is False
    assert imported.WAL_IMPLEMENTATION is None
    assert imported.INFER_SUCCESS_FROM_PROCESS_EXIT is False
    assert imported.PROVIDER_BOUND is False
    assert imported.SIBLING_LAYOUT_REQUIRED is False
    assert imported.PERSISTENCE_AUTHORITY == "injected-kit-checkpoint-store"


def test_every_lifecycle_boundary_has_idempotency_key_and_durable_checkpoint(
    tmp_path: Path,
) -> None:
    engine, _ports, store, attempt, _repo = _open(tmp_path)
    record = engine.run()
    assert record.published is True
    assert record.status == "succeeded"
    assert record.sealed is True
    history = store.history(attempt.attempt_id)
    after = {
        item["stage"]: item
        for item in history
        if item.get("position") == "after" and item.get("stage") in BOUNDARIES
    }
    for boundary in BOUNDARIES:
        assert boundary in after, f"missing durable checkpoint for {boundary}"
        key = after[boundary]["idempotency_key"]
        assert isinstance(key, str) and key.startswith("b") and len(key) >= 59
        assert after[boundary]["checkpoint_cid"].startswith("b")
        assert record.idempotency_keys[boundary] == key
    assert set(record.idempotency_keys) == set(BOUNDARIES)
    for stage in STAGES:
        assert store.invocation_count(attempt.attempt_id, stage) == 1


def test_success_resume_is_idempotent_and_does_not_reinvoke(tmp_path: Path) -> None:
    engine, _ports, store, attempt, repo = _open(tmp_path)
    first = engine.run()
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "restart",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    second = engine2.resume()
    assert second.status == first.status == "succeeded"
    assert second.published is True
    assert second.evidence_cid == first.evidence_cid
    assert second.settled is True
    assert ports2.worktree.calls == []
    assert ports2.sealing.calls == []
    assert store.invocation_count(attempt.attempt_id, APPLY_STAGE) == 1
    assert store.invocation_count(attempt.attempt_id, SEAL_STAGE) == 1


@pytest.mark.parametrize(("stage", "position"), list(CRASH_MATRIX))
def test_crash_matrix_converges_to_one_valid_state(
    tmp_path: Path,
    stage: str,
    position: str,
) -> None:
    engine, ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(stage, position)
    with pytest.raises(_live_crash_interrupt()) as crashed:
        engine.run()
    assert crashed.value.stage == stage
    assert crashed.value.position == position
    latest = store.latest(attempt.attempt_id)
    assert latest is not None
    assert latest.get("published") is not True
    assert latest.get("status") != "succeeded" or latest.get("position") != "after" or latest.get(
        "stage"
    ) != PUBLISH_BOUNDARY
    first_invocations = store.invocation_count(attempt.attempt_id, stage)
    if position == "before":
        assert first_invocations == 0
        assert stage not in ports.worktree.calls + ports.verification.calls + ports.sealing.calls
    else:
        assert first_invocations == 1

    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "resume",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    recovered = engine2.resume()
    assert recovered.status in VALID_TERMINAL_STATUSES
    assert recovered.published is False or recovered.status == "succeeded"
    if position == "during":
        assert recovered.status == "repair_required"
        assert recovered.published is False
        assert recovered.error == "repair_required"
        assert recovered.repair_receipt is not None
        assert recovered.repair_receipt["ambiguous"] is True
        assert recovered.repair_receipt["stage"] == stage
        assert recovered.repair_receipt["infer_success_from_process_exit"] is False
        assert stage not in ports2.worktree.calls
        assert stage not in ports2.verification.calls
        assert stage not in ports2.sealing.calls
        assert store.invocation_count(attempt.attempt_id, stage) == 1
        if stage == SEAL_STAGE:
            assert store.invocation_count(attempt.attempt_id, SEAL_STAGE) == 1
            assert SEAL_STAGE not in ports2.sealing.calls
    else:
        assert recovered.status == "succeeded"
        assert recovered.published is True
        assert recovered.sealed is True
        assert store.invocation_count(attempt.attempt_id, stage) == 1
        if position == "after":
            assert stage not in ports2.worktree.calls
            assert stage not in ports2.verification.calls
            assert stage not in ports2.sealing.calls
        if stage == SEAL_STAGE:
            # Crash-before never invoked the terminal adapter; resume must
            # invoke it once. Crash-after already completed it.
            if position == "before":
                assert SEAL_STAGE in ports2.sealing.calls
            else:
                assert SEAL_STAGE not in ports2.sealing.calls
            assert store.invocation_count(attempt.attempt_id, SEAL_STAGE) == 1

    engine3, ports3, store, attempt, _repo = _open(
        tmp_path / "resume2",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    again = engine3.resume()
    assert again.status == recovered.status
    assert again.published == recovered.published
    assert again.evidence_cid == recovered.evidence_cid
    assert store.invocation_count(attempt.attempt_id, stage) == 1
    assert ports3.sealing.calls == []


def test_process_exit_is_never_success(tmp_path: Path) -> None:
    assert INFER_SUCCESS_FROM_PROCESS_EXIT is False
    engine, _ports, store, attempt, _repo = _open(tmp_path)
    engine.inject_crash(APPLY_STAGE, "during")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    latest = store.latest(attempt.attempt_id)
    assert latest is not None
    assert latest["position"] == "during"
    assert latest["in_flight"] is True
    assert latest["published"] is not True
    assert latest.get("settled") is not True


def test_stale_writer_cannot_publish(tmp_path: Path) -> None:
    engine, _ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(APPLY_STAGE, "after")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    store.reclaim(
        attempt.attempt_id,
        writer_id="writer-b",
        fence_token=_cid("newtoken"),
    )
    with pytest.raises(StaleWriterError):
        store.put(
            {
                "attempt_id": attempt.attempt_id,
                "idempotency_key": mint_idempotency_key(
                    attempt_id=attempt.attempt_id,
                    run_id=attempt.identities.run_id,
                    trace_id=attempt.identities.trace_id,
                    stage=PUBLISH_BOUNDARY,
                    position="after",
                    inbound_cid=None,
                    generation=attempt.writer_generation,
                ),
                "stage": PUBLISH_BOUNDARY,
                "position": "after",
                "published": True,
            },
            writer_id=attempt.writer_id,
            generation=attempt.writer_generation,
            fence_token=attempt.fence_token,
        )
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "stale",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    record = engine2.resume()
    assert record.published is False
    assert record.status in {"stale", "repair_required"}
    assert record.error in {"stale_root", "repair_required"}
    assert APPLY_STAGE not in ports2.worktree.calls
    published = [item for item in store.history(attempt.attempt_id) if item.get("published") is True]
    assert published == []


def test_expired_lease_cannot_publish(tmp_path: Path) -> None:
    clock = ManualClock(now=10)
    identities = _identities()
    attempt = _attempt(identities, lease_expires_at=50)
    engine, _ports, store, attempt, repo = _open(
        tmp_path,
        identities=identities,
        attempt=attempt,
        clock=clock,
    )
    engine.inject_crash(VERIFY_STAGE, "after")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    clock.now = 80
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "expired-lease",
        store=store,
        attempt=attempt,
        identities=identities,
        repo=repo,
        clock=clock,
    )
    record = engine2.resume()
    assert record.published is False
    assert record.status == "stale"
    assert record.error == "stale_root"
    assert VERIFY_STAGE not in ports2.verification.calls
    assert SEAL_STAGE not in ports2.sealing.calls


def test_expired_fence_cannot_publish(tmp_path: Path) -> None:
    engine, _ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(SEAL_STAGE, "after")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    store.invalidate_fence(attempt.attempt_id)
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "expired-fence",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    record = engine2.resume()
    assert record.published is False
    assert record.status == "stale"
    assert record.error == "stale_root"
    assert SEAL_STAGE not in ports2.sealing.calls
    assert DISPOSITION_STAGE not in ports2.disposition.calls


def test_terminal_adapter_is_never_reinvoked_after_seal_checkpoint(
    tmp_path: Path,
) -> None:
    engine, _ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(SEAL_STAGE, "after")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    assert store.invocation_count(attempt.attempt_id, SEAL_STAGE) == 1
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "seal-resume",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    record = engine2.resume()
    assert record.status == "succeeded"
    assert record.published is True
    assert SEAL_STAGE not in ports2.sealing.calls
    assert store.invocation_count(attempt.attempt_id, SEAL_STAGE) == 1
    assert DISPOSITION_STAGE in ports2.disposition.calls


def test_replay_trace_is_ordered_and_durable(tmp_path: Path) -> None:
    engine, _ports, store, attempt, _repo = _open(tmp_path)
    record = engine.run()
    trace = replay_trace(store, attempt.attempt_id)
    assert trace
    assert record.replay_trace
    stages_after = [item["stage"] for item in trace if item["position"] == "after"]
    for boundary in (LEASE_BOUNDARY, FENCE_BOUNDARY, *STAGES, PUBLISH_BOUNDARY):
        assert boundary in stages_after
    assert stages_after[0] == LEASE_BOUNDARY
    assert PUBLISH_BOUNDARY in stages_after


def test_partial_effect_repair_receipt_is_auditable(tmp_path: Path) -> None:
    engine, ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(APPLY_STAGE, "during")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    engine2, _ports2, store, attempt, _repo = _open(
        tmp_path / "repair",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    record = engine2.resume()
    assert record.status == "repair_required"
    assert record.repair_receipt is not None
    assert record.repair_receipt["schema"].endswith("repair-receipt")
    assert record.repair_receipt["ambiguous"] is True
    assert record.repair_receipt["published"] is False
    assert "discarded" in record.repair_receipt
    assert record.repair_receipt["discarded"] in {True, False}
    completed = list(record.payload["completed"])
    assert APPLY_STAGE not in completed


def test_bypass_is_rejected(tmp_path: Path) -> None:
    engine, _ports, _store, _attempt, _repo = _open(tmp_path)
    with pytest.raises(BoundaryViolationError):
        engine.run({"skip_stages": ["seal"]})
    with pytest.raises(BoundaryViolationError):
        engine.resume({"bypass": True, "schema": CHECKPOINT_SCHEMA})


def test_no_second_wal_and_descriptor_is_bound() -> None:
    assert SECOND_WAL is False
    assert WAL_IMPLEMENTATION is None
    assert FencedCheckpointStore.second_wal is False
    assert SOLE_LIFECYCLE_AUTHORITY is True
    assert RESTART_AWARE is True
    descriptor = recovery_descriptor()
    assert descriptor is RECOVERY_DESCRIPTOR
    assert descriptor["schema"] == RECOVERY_SCHEMA
    assert descriptor["cid"] == RECOVERY_CID == recovery_cid()
    assert descriptor["second_wal"] is False
    assert descriptor["infer_success_from_process_exit"] is False
    assert descriptor["crash_matrix"] == CRASH_MATRIX == crash_matrix()
    assert descriptor["boundaries"] == BOUNDARIES
    assert descriptor["effectful_stages"] == EFFECTFUL_STAGES
    assert descriptor["terminal_adapter_stages"] == TERMINAL_ADAPTER_STAGES
    assert descriptor["lifecycle_cid"] == LIFECYCLE_CID
    assert descriptor["policy_cid"] == POLICY_CID
    assert descriptor["result_state_cid"] == RESULT_STATE_CID == BOUND_RESULT_STATE_CID
    assert descriptor["pcce_006_content_id"] == PCCE_006_CONTENT_ID
    body = {key: value for key, value in descriptor.items() if key != "cid"}
    assert mint_recovery_cid(body) == RECOVERY_CID
    assert COMPATIBILITY_MATRIX_CONTENT_ID.endswith("e920")
    digest = hashlib.sha256(RECOVERY_CID.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    with pytest.raises(TypeError):
        descriptor["boundaries"] = ()  # type: ignore[index]


def test_provider_neutral_ast_and_no_wal_class() -> None:
    source = Path(inspect.getfile(RecoveryCoordinator)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    class_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)
    assert "openai" not in imported
    assert "anthropic" not in imported
    assert "ipfs_datasets_py" not in imported
    assert "ipfs_kit_py" not in imported
    assert not any("WAL" in name for name in class_names)
    assert PROVIDER_BOUND is False
    assert SIBLING_LAYOUT_REQUIRED is False
    assert not any(isinstance(node, ast.Pass) for node in ast.walk(tree))


def test_idempotency_key_is_stable_and_admitted() -> None:
    first = mint_idempotency_key(
        attempt_id="attempt-pcce-024",
        run_id="run-pcce-024",
        trace_id="trace-pcce-024",
        stage=APPLY_STAGE,
        position="after",
        inbound_cid=None,
        generation=1,
    )
    second = mint_idempotency_key(
        attempt_id="attempt-pcce-024",
        run_id="run-pcce-024",
        trace_id="trace-pcce-024",
        stage=APPLY_STAGE,
        position="after",
        inbound_cid=None,
        generation=1,
    )
    assert first == second
    assert first.startswith("b") and len(first) >= 59
    other = mint_idempotency_key(
        attempt_id="attempt-pcce-024",
        run_id="run-pcce-024",
        trace_id="trace-pcce-024",
        stage=APPLY_STAGE,
        position="before",
        inbound_cid=None,
        generation=1,
    )
    assert other != first
    assert admit_position("during") == "during"
    assert admit_boundary(LEASE_BOUNDARY) == LEASE_BOUNDARY
    with pytest.raises(ProofContextError):
        admit_position("sideways")
    with pytest.raises(ProofContextError):
        admit_boundary("skip-seal")


def test_fenced_store_rejects_stale_generation() -> None:
    store = FencedCheckpointStore()
    key = mint_idempotency_key(
        attempt_id="attempt-pcce-024",
        run_id="run-pcce-024",
        trace_id="trace-pcce-024",
        stage=LEASE_BOUNDARY,
        position="after",
        inbound_cid=None,
        generation=1,
    )
    store.put(
        {
            "attempt_id": "attempt-pcce-024",
            "idempotency_key": key,
            "stage": LEASE_BOUNDARY,
            "position": "after",
        },
        writer_id="writer-a",
        generation=1,
        fence_token=_cid("ftoken"),
    )
    store.reclaim("attempt-pcce-024", writer_id="writer-b", fence_token=_cid("newtoken"))
    with pytest.raises(StaleWriterError):
        store.put(
            {
                "attempt_id": "attempt-pcce-024",
                "idempotency_key": mint_idempotency_key(
                    attempt_id="attempt-pcce-024",
                    run_id="run-pcce-024",
                    trace_id="trace-pcce-024",
                    stage=PUBLISH_BOUNDARY,
                    position="after",
                    inbound_cid=None,
                    generation=1,
                ),
                "stage": PUBLISH_BOUNDARY,
                "published": True,
            },
            writer_id="writer-a",
            generation=1,
            fence_token=_cid("ftoken"),
        )


def test_crash_positions_and_effectful_stages_are_closed() -> None:
    assert CRASH_POSITIONS == ("before", "during", "after")
    assert EFFECTFUL_STAGES == (APPLY_STAGE, VERIFY_STAGE, SEAL_STAGE)
    assert len(CRASH_MATRIX) == 9
    assert len(BOUNDARIES) == len(STAGES) + 3
    assert RecoveryCoordinator.second_wal is False
    assert RecoveryCoordinator.infer_success_from_process_exit is False
    assert PatchLifecycle.sole_authority is True


def test_import_does_not_read_promotion_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PCCE_MODE", "simulation")
    monkeypatch.setenv("PROOF_CONTEXT_MODE", "simulation")
    imported = importlib.reload(
        importlib.import_module("ipfs_accelerate_py.proof_context.recovery")
    )
    assert imported.RECOVERY_DESCRIPTOR["modes"][0] == "production"


def test_recovery_record_mapping_is_immutable(tmp_path: Path) -> None:
    engine, _ports, _store, _attempt, _repo = _open(tmp_path)
    record = engine.run()
    mapping = record.to_mapping()
    assert mapping["schema"] == RECOVERY_RECORD_SCHEMA
    with pytest.raises(TypeError):
        mapping["published"] = False  # type: ignore[index]
    assert record.accepted is True


def test_resume_reuses_completed_prefix_after_apply_checkpoint(tmp_path: Path) -> None:
    engine, ports, store, attempt, repo = _open(tmp_path)
    engine.inject_crash(APPLY_STAGE, "after")
    with pytest.raises(_live_crash_interrupt()):
        engine.run()
    assert APPLY_STAGE in ports.worktree.calls
    engine2, ports2, store, attempt, _repo = _open(
        tmp_path / "prefix",
        store=store,
        attempt=attempt,
        identities=attempt.identities,
        repo=repo,
    )
    record = engine2.resume()
    assert record.published is True
    assert APPLY_STAGE not in ports2.worktree.calls
    assert VERIFY_STAGE in ports2.verification.calls
    assert SEAL_STAGE in ports2.sealing.calls
    assert store.invocation_count(attempt.attempt_id, APPLY_STAGE) == 1

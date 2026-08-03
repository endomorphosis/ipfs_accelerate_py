"""Live Git integration tests for DeterministicDoctorTransaction@2."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE,
    DeterministicDoctorTransaction,
    DoctorCheckoutLock,
    DoctorSandboxPolicy,
    DoctorStepApplyRequest,
    DoctorStepApplyResult,
    DoctorStepDisposition,
    DoctorTransactionDisposition,
    DoctorTransactionReason,
    DoctorWriterLease,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import (
    DoctorExactEdit,
    DoctorWorktreeAdapter,
    DoctorWorktreeTamperError,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "doctor-test@example.invalid")
    _git(root, "config", "user.name", "Doctor Test")
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _hash(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:live",
        forest_id="forest:live",
        tree_id="tree:live",
        overlay_id="overlay:live",
        file_root_id="file-root:live",
        ast_root_id="ast:live",
        graph_id="graph:live",
        corpus_id="corpus:live",
        index_id="index:live",
        model_id="model:live",
        cache_id="cache:live",
        operator_registry_id="operators:live",
        translator_id="translator:live",
        solver_id="solver:live",
        kernel_id="kernel:live",
        toolchain_id="toolchain:live",
        policy_id="policy:live",
        sandbox_id="sandbox:live",
        environment_id="environment:live",
        lease_id="lease:live",
    )


def _plan(
    before: dict[str, bytes],
    *,
    one_scc: bool = True,
) -> DeterministicDoctorPlan:
    roots = _roots()
    consumers = tuple(
        DoctorConsumerDisposition(
            roots=roots,
            consumer_id=f"consumer:{index}",
            disposition=DoctorRepairDisposition.SUPPORTED,
            reason_codes=("supported",),
        )
        for index, _path in enumerate(before)
    )
    sites = tuple(
        DoctorEditSite(
            path=path,
            before_hash=_hash(body),
            span_start=0,
            span_end=len(body),
            artifact_id=f"blob:{index}",
        )
        for index, (path, body) in enumerate(before.items())
    )
    scc = ("scc:impact",) if one_scc else ()
    steps = tuple(
        DoctorPlanStep(
            step_id=f"step:{index}",
            kind="analytical",
            operator_id="operator:exact",
            consumer_ids=(f"consumer:{index}",),
            edit_site_refs=(sites[index].content_id,),
            write_paths=(path,),
            dependency_step_ids=((f"step:{index - 1}",) if index else ()),
            validation_refs=scc,
        )
        for index, path in enumerate(before)
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:live",
        snapshot_id="snapshot:live",
        finding_ids=("finding:live",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=consumers,
        impact_closure_id="impact:live",
        steps=steps,
        edit_sites=sites,
        operator_ids=("operator:exact",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:exact",
        scc_refs=scc,
        permitted_read_paths=tuple(before),
        permitted_write_paths=tuple(before),
        lease_id="lease:live",
        checkpoint_ref="checkpoint:durable",
        rollback_ref="rollback:exact",
        proof_refs=("proof:live",),
        invalidation_refs=("tree:live",),
    )


def _adapter(
    root: Path,
    state: Path,
    paths: tuple[str, ...],
    *,
    fault=None,
) -> DoctorWorktreeAdapter:
    return DoctorWorktreeAdapter(
        root,
        state,
        paths,
        permitted_refs=("refs/heads/main",),
        fault_injector=fault,
    )


def _legacy_inputs(plan: DeterministicDoctorPlan):
    paths = tuple(plan.permitted_write_paths)
    sandbox = DoctorSandboxPolicy(
        sandbox_id="sandbox:live",
        worktree_root_ref="worktree:fake",
        permitted_paths=paths,
    )
    lock = DoctorCheckoutLock(
        lock_id="lock:fake",
        holder_id="holder:fake",
        worktree_root_ref="worktree:fake",
        base_tree_cid="tree:base",
    )
    lease = DoctorWriterLease(
        lease_id="lease:live",
        fence_id="fence:fake",
        holder_id="holder:fake",
        permitted_write_paths=paths,
        permitted_read_paths=paths,
    )
    hashes = tuple(
        PathBeforeHash(path=site.path, before_hash=site.before_hash)
        for site in plan.edit_sites
    )
    return sandbox, lock, lease, hashes


def test_v2_default_and_noop_fake_applicators_cannot_report_committed() -> None:
    before = {"pkg/a.py": b"value = 1\n"}
    plan = _plan(before)
    sandbox, lock, lease, hashes = _legacy_inputs(plan)
    default = DeterministicDoctorTransaction().execute(
        plan,
        sandbox_policy=sandbox,
        checkout_lock=lock,
        lease=lease,
        path_before_hashes=hashes,
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:claimed",
    )
    assert DETERMINISTIC_DOCTOR_TRANSACTION_INTERFACE == (
        "DeterministicDoctorTransaction@2"
    )
    assert not default.committed
    assert default.disposition is DoctorTransactionDisposition.QUARANTINED
    assert (
        DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value
        in default.reason_codes
    )

    def fake(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            written_paths=request.step.write_paths,
            observed_before_hashes=hashes,
        )

    no_op = DeterministicDoctorTransaction(step_applicator=fake).execute(
        plan,
        sandbox_policy=sandbox,
        checkout_lock=lock,
        lease=lease,
        path_before_hashes=hashes,
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:claimed",
    )
    assert not no_op.committed
    assert (
        DoctorTransactionReason.EFFECT_EVIDENCE_MISSING.value
        in no_op.reason_codes
    )


def test_live_transaction_owns_checkpoint_complete_scc_and_ref_cas(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    after = {"pkg/a.py": b"a = 2\n", "pkg/b.py": b"b = 2\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    plan = _plan(before)
    edits = tuple(
        DoctorExactEdit(path, _hash(body), after[path], step_id=f"step:{index}")
        for index, (path, body) in enumerate(before.items())
    )
    report = DeterministicDoctorTransaction().execute_live(
        plan,
        worktree_adapter=_adapter(
            root, tmp_path / "state", tuple(before)
        ),
        edits=edits,
        target_ref="refs/heads/main",
        transaction_id="txn:integration",
    )
    assert report.committed
    assert report.disposition is DoctorTransactionDisposition.COMMITTED
    assert report.merge_cas is not None
    assert report.merge_cas.expected_ref == base
    assert report.merge_cas.desired_ref == _git(
        root, "rev-parse", "refs/heads/main"
    )
    assert len(report.group_receipts) == 1
    assert {item.step_id for item in report.group_receipts[0].step_receipts} == {
        "step:0",
        "step:1",
    }
    assert report.candidate_tree is not None
    assert set(report.candidate_tree.written_paths) == set(before)
    assert len(report.candidate_tree.changed_blob_cids) == 2
    assert report.candidate_tree.observed_tree_cid != (
        report.candidate_tree.base_tree_cid
    )
    assert report.candidate_tree.observed_forest_cid
    assert report.candidate_tree.durable_effect_refs
    for path, body in after.items():
        assert _git(root, "show", f"refs/heads/main:{path}") == body.decode().strip()


def test_incomplete_scc_edit_set_is_rejected_before_worktree_creation(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    root = _repo(tmp_path, before)
    plan = _plan(before)
    adapter = _adapter(root, tmp_path / "state", tuple(before))
    with pytest.raises(Exception, match="cover.*complete|complete exact"):
        DeterministicDoctorTransaction().execute_live(
            plan,
            worktree_adapter=adapter,
            edits=(
                DoctorExactEdit(
                    "pkg/a.py", _hash(before["pkg/a.py"]), b"a = 2\n", step_id="step:0"
                ),
            ),
            target_ref="refs/heads/main",
        )
    assert not list((tmp_path / "state/sessions").iterdir())


def test_failure_inside_atomic_scc_restores_all_candidate_bytes_and_ref(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    plan = _plan(before)
    adapter = _adapter(root, tmp_path / "state", tuple(before))
    edits = (
        DoctorExactEdit(
            "pkg/a.py", _hash(before["pkg/a.py"]), b"a = 2\n", step_id="step:0"
        ),
        DoctorExactEdit(
            "pkg/b.py", _hash(b"stale\n"), b"b = 2\n", step_id="step:1"
        ),
    )
    with pytest.raises(DoctorWorktreeTamperError, match="before_hash"):
        DeterministicDoctorTransaction().execute_live(
            plan,
            worktree_adapter=adapter,
            edits=edits,
            target_ref="refs/heads/main",
            transaction_id="txn:scc-fail",
        )
    assert _git(root, "rev-parse", "refs/heads/main") == base
    journal = next((tmp_path / "state/sessions").glob("*/intent.json"))
    assert '"state":"rolled_back"' in journal.read_text(encoding="utf-8")


def test_crash_after_ref_cas_restores_exact_ref_and_bytes(tmp_path: Path) -> None:
    before = {"pkg/a.py": b"a = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    boundaries: list[str] = []

    def crash(boundary: str) -> None:
        boundaries.append(boundary)
        if boundary == "after_cas_fsync":
            raise RuntimeError("simulated crash")

    plan = _plan(before)
    report = DeterministicDoctorTransaction().execute_live(
        plan,
        worktree_adapter=_adapter(
            root, tmp_path / "state", tuple(before), fault=crash
        ),
        edits=(
            DoctorExactEdit(
                "pkg/a.py", _hash(before["pkg/a.py"]), b"a = 2\n", step_id="step:0"
            ),
        ),
        target_ref="refs/heads/main",
        transaction_id="txn:crash-after-cas",
    )
    assert "after_cas_fsync" in boundaries
    assert not report.committed
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert report.rollback is not None and report.rollback.restored
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "a = 1"


def test_tamper_after_durable_group_is_detected_and_restored(tmp_path: Path) -> None:
    before = {"pkg/a.py": b"a = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    adapter_holder: dict[str, DoctorWorktreeAdapter] = {}

    def tamper(boundary: str) -> None:
        if boundary == "after_group_effect_fsync":
            session_dir = next(
                (adapter_holder["adapter"].state_root / "sessions").iterdir()
            )
            (session_dir / "worktree/pkg/a.py").write_bytes(b"tampered\n")

    adapter = _adapter(
        root, tmp_path / "state", tuple(before), fault=tamper
    )
    adapter_holder["adapter"] = adapter
    report = DeterministicDoctorTransaction().execute_live(
        _plan(before),
        worktree_adapter=adapter,
        edits=(
            DoctorExactEdit(
                "pkg/a.py", _hash(before["pkg/a.py"]), b"a = 2\n", step_id="step:0"
            ),
        ),
        target_ref="refs/heads/main",
        transaction_id="txn:tamper-after-group",
    )
    assert not report.committed
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert DoctorTransactionReason.DRIFT.value in report.reason_codes
    assert _git(root, "rev-parse", "refs/heads/main") == base


def test_default_restore_does_not_trust_boolean_and_quarantines_without_context(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n"}
    plan = _plan(before)
    sandbox, lock, lease, hashes = _legacy_inputs(plan)

    def fail(_request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.FAILED,
            reason_codes=(DoctorTransactionReason.STEP_FAILURE.value,),
        )

    report = DeterministicDoctorTransaction(step_applicator=fail).execute(
        plan,
        sandbox_policy=sandbox,
        checkout_lock=lock,
        lease=lease,
        path_before_hashes=hashes,
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert report.disposition is DoctorTransactionDisposition.QUARANTINED
    assert report.rollback is not None
    assert not report.rollback.restored
    assert report.rollback.quarantined
    assert DoctorTransactionReason.RESTORE_FAILED.value in report.reason_codes

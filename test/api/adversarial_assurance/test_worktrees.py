"""Tests for disposable mutation worktree executor and admission pipeline (AAE-041)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.admission import (
    ADMIT_MUTATION_INTERFACE,
    AdmissionDisposition,
    AdmissionReasonCode,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.worktrees import (
    AAE_ISOLATED_EXECUTOR_EVIDENCE,
    ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE,
    IsolatedMutationWorktree,
    IsolatedMutationWorktreeExecutor,
    MutationWorktreeError,
    MutationWorktreeFenceError,
    MutationWorktreePhase,
    MutationWriteScope,
    apply_file_replacements,
    create_mutation_worktree,
    isolated_mutation_worktree_executor_descriptor,
    recover_mutation_worktree,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    CleanupDisposition,
    WorktreeLifecycleStore,
    normalize_workspace_path,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    ClaimBinding,
    DependencyRelation,
    DetectionAssuranceManifest,
    DetectorCatalogEntry,
    SemanticDependencyEdge,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    DetectorKind,
    DetectorStrength,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    MutationCandidate,
    MutationRiskClass,
    PropertyClass,
    SeedConfigBinding,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae041"
REPO_STATE = _cid("repo-state-aae041")


def _git(
    cwd: Path, *args: str, check: bool = True, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
        )
    return completed


def _init_repo(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init")
    _git(root, "config", "user.email", "aae041@example.com")
    _git(root, "config", "user.name", "AAE041")
    _git(root, "checkout", "-b", "main")
    (root / "mod.py").write_text(MOD_SOURCE, encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("X = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "baseline")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    return head, tree


def _lifecycle_store(repo: Path, tmp_path: Path) -> WorktreeLifecycleStore:
    return WorktreeLifecycleStore(
        repo_root=repo,
        store_dir=tmp_path / "lifecycle",
        lease_seconds=300.0,
    )


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "mutation_campaign",
        "generator_version": "1.0.0",
        "interface_id": "generate_mutation_candidates@1",
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _versions(**overrides: object) -> VersionBinding:
    fields = {
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "campaign_policy_id": "default_campaign",
        "campaign_policy_version": "1.0.0",
        "generator": _generator(),
    }
    fields.update(overrides)
    return VersionBinding(**fields)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> ArtifactProvenance:
    fields = {
        "producer_id": "adversarial_assurance",
        "producer_version": "1",
        "execution_mode": ExecutionMode.LIVE,
        "authority_source": AuthoritySource.DETERMINISTIC,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("mutator.v1",),
        "policy_cid": _cid("policy"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str, **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": _cid("environment"),
        "dependency_lock_cid": _cid("dependency-lock"),
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (_cid("receipt-a"),),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {"risk_class": "local_bug"},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


def _seed_config(**overrides: object) -> SeedConfigBinding:
    fields = {
        "seed": 41,
        "config": {"max_depth": 2, "operator_budget": 4},
    }
    fields.update(overrides)
    return SeedConfigBinding(**fields)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> MutationCandidate:
    fields = {
        "header": _header("mutation_candidate"),
        "candidate_id": "cand_control_flow_invert_0",
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "operator_cid": _cid("operator-control-flow"),
        "target_id": "mod_fn",
        "target_cid": _cid("target-mod-fn"),
        "seed_config": _seed_config(),
        "source_root_cid": _cid("source-root"),
        "repository_state_cid": REPO_STATE,
        "transformation_summary": "invert if-test at mod.fn:12",
        "expected_violated_property_classes": (PropertyClass.CONTROL_INVARIANT,),
        "risk_class": MutationRiskClass.LOCAL_BUG,
        "likely_equivalent": False,
        "scope_symbol_ids": ("mod.fn",),
        "scope_paths": ("mod.py",),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationCandidate(**fields)  # type: ignore[arg-type]


def _unit_detector(**overrides: object) -> DetectorCatalogEntry:
    fields = {
        "detector_id": "unit.test_branch",
        "detector_revision": "3.2.1",
        "detector_kind": DetectorKind.UNIT_TEST,
        "covered_property_classes": (PropertyClass.CONTROL_INVARIANT,),
        "anchor_ids": ("tests.test_branch",),
        "default_strength": DetectorStrength.REQUIRED,
        "expected_terminal_status": AssuranceTerminalStatus.COMPLETE,
        "observation_template": "unit test asserts inverted branch is rejected",
        "claim_ids": ("claim.control_branch",),
        "notes": "selected unit detector",
        "metadata": {},
    }
    fields.update(overrides)
    return DetectorCatalogEntry(**fields)  # type: ignore[arg-type]


def _edge(
    from_id: str,
    to_id: str,
    relation: DependencyRelation | str = DependencyRelation.TESTED_BY,
) -> SemanticDependencyEdge:
    return SemanticDependencyEdge(
        from_id=from_id,
        to_id=to_id,
        relation=relation,
        notes=None,
    )


def _claim(**overrides: object) -> ClaimBinding:
    fields = {
        "claim_id": "claim.control_branch",
        "property_class": PropertyClass.CONTROL_INVARIANT,
        "statement": "branch predicate must preserve control invariant",
        "symbol_ids": ("mod.fn",),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ClaimBinding(**fields)  # type: ignore[arg-type]


def _manifest(**overrides: object) -> DetectionAssuranceManifest:
    fields = {
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "detectors": (_unit_detector(),),
        "dependency_edges": (
            _edge("mod.fn", "tests.test_branch", DependencyRelation.TESTED_BY),
        ),
        "claims": (_claim(),),
        "enable_type_check_fallback": True,
        "enable_full_suite_fallback": True,
        "enable_incremental_seal_fallback": True,
        "enable_human_review_fallback": True,
        "observation_complete": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return DetectionAssuranceManifest(**fields)  # type: ignore[arg-type]


MOD_SOURCE = textwrap.dedent(
    '''\
    """Sample module for AAE-041 mutation worktree tests."""


    def fn(flag: bool) -> int:
        if flag:
            return 1
        return 0
    '''
)

MOD_MUTATED = textwrap.dedent(
    '''\
    """Sample module for AAE-041 mutation worktree tests."""


    def fn(flag: bool) -> int:
        if not flag:
            return 1
        return 0
    '''
)

MOD_SYNTAX_ERROR = textwrap.dedent(
    '''\
    def fn(flag: bool) -> int:
        if flag
            return 1
        return 0
    '''
)


def _scope(**overrides: object) -> MutationWriteScope:
    payload: dict[str, object] = {
        "allowed_paths": ("mod.py", "pkg/"),
        "effect_paths": ("mod.py",),
        "task_owned_paths": ("mod.py", "pkg/"),
    }
    payload.update(overrides)
    return MutationWriteScope.from_dict(payload)


# ---------------------------------------------------------------------------
# Cold import / vocabulary
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.worktrees"
    )
    assert (
        module.ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE
        == "IsolatedMutationWorktreeExecutor@1"
    )
    descriptor = isolated_mutation_worktree_executor_descriptor()
    assert descriptor["interface"] == ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE
    assert "IsolatedMutationWorktreeExecutor" in descriptor["symbols"]
    assert "create_mutation_worktree" in descriptor["symbols"]
    assert descriptor["evidence"] == AAE_ISOLATED_EXECUTOR_EVIDENCE
    assert descriptor["admit_mutation_interface"] == ADMIT_MUTATION_INTERFACE
    assert "sole_mutation_worktree_lifecycle_owner" in descriptor["invariants"]


def test_write_scope_round_trip_and_authority_block() -> None:
    scope = _scope(max_files=32)
    restored = MutationWriteScope.from_dict(scope.to_dict())
    assert restored.allowed_paths == scope.allowed_paths
    assert restored.effect_paths == scope.effect_paths
    assert restored.max_files == 32

    ok, reason = scope.admits("mod.py")
    assert ok is True
    assert reason == ""

    ok, reason = scope.admits("README.md")
    assert ok is False
    assert reason in {"outside_allowlist", "outside_effect_scope", "outside_task_owned"}

    ok, reason = scope.admits("config/policy.json")
    assert ok is False
    assert reason in {"authority_path_blocked", "forbidden_path", "protected_path"}

    ok, reason = scope.admits("../escape.py")
    assert ok is False
    assert reason == "unsafe_path"


def test_scope_from_candidate() -> None:
    scope = MutationWriteScope.from_candidate(_candidate())
    assert "mod.py" in scope.allowed_paths
    assert scope.effect_paths == ("mod.py",)


# ---------------------------------------------------------------------------
# Create / apply / production isolation
# ---------------------------------------------------------------------------


def test_create_detached_worktree_does_not_touch_production(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    wt_path = parent / "mutant-1"

    branches_before = set(
        line.strip()
        for line in _git(repo, "branch", "--format=%(refname:short)").stdout.splitlines()
        if line.strip()
    )
    root_mod_before = (repo / "mod.py").read_text(encoding="utf-8")

    with create_mutation_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-create",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        assert isolated.phase is MutationWorktreePhase.READY
        assert isolated.worktree_path.is_dir()
        assert (isolated.worktree_path / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE
        assert _git(isolated.worktree_path, "rev-parse", "HEAD").stdout.strip() == head
        # Linked disposable worktree (not production root).
        assert (isolated.worktree_path / ".git").is_file()
        assert (repo / ".git").is_dir()

        applied = isolated.apply_replacements(
            {"mod.py": MOD_MUTATED},
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
        )
        assert applied.applied is True
        assert applied.pre_tree == tree
        assert applied.post_tree != applied.pre_tree
        assert (isolated.worktree_path / "mod.py").read_text(encoding="utf-8") == MOD_MUTATED

        # Production untouched.
        assert (repo / "mod.py").read_text(encoding="utf-8") == root_mod_before
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head
        branches_after = set(
            line.strip()
            for line in _git(repo, "branch", "--format=%(refname:short)").stdout.splitlines()
            if line.strip()
        )
        # Detached worktree add must not create production branches.
        assert branches_after == branches_before

    assert not wt_path.exists()
    record = store.load_workspace(wt_path)
    assert record is not None
    assert record.is_terminal


def test_apply_rejects_production_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    result = apply_file_replacements(
        {"mod.py": MOD_MUTATED},
        worktree_root=repo,
        scope=_scope(),
        expected_base_commit=head,
        expected_base_tree=tree,
        require_linked_worktree=True,
    )
    assert result.applied is False
    assert "production_root" in result.reason_codes
    assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE


def test_apply_rejects_path_escape_and_authority(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    with create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "escape",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-escape",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        with pytest.raises(MutationWorktreeError) as excinfo:
            MutationWriteScope.from_dict({"allowed_paths": ("../secret",)})
        assert excinfo.value.reason_code == "unsafe_path"

        blocked = isolated.apply_replacements(
            {"config/policy.json": "{}\n"},
            MutationWriteScope(
                allowed_paths=("config/", "mod.py"),
                effect_paths=("config/policy.json",),
                task_owned_paths=("config/", "mod.py"),
            ),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
        )
        assert blocked.applied is False
        assert "authority_path_blocked" in blocked.reason_codes

        out_of_scope = isolated.apply_replacements(
            {"README.md": "# changed\n"},
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
        )
        assert out_of_scope.applied is False
        assert any(
            code in out_of_scope.reason_codes
            for code in (
                "outside_allowlist",
                "outside_effect_scope",
                "outside_task_owned",
            )
        )


def test_create_rejects_mismatched_base_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, _tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    with pytest.raises(MutationWorktreeError) as excinfo:
        create_mutation_worktree(
            repo_root=repo,
            worktree_path=tmp_path / "owned" / "bad-tree",
            worktree_parent=tmp_path / "owned",
            base_commit=head,
            base_tree="0" * 40,
            task_id="AAE-041-badtree",
            attempt=1,
            lifecycle_store=store,
        )
    assert excinfo.value.reason_code == "stale_base"


def test_create_rejects_worktree_equal_to_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    with pytest.raises(MutationWorktreeError) as excinfo:
        create_mutation_worktree(
            repo_root=repo,
            worktree_path=repo,
            worktree_parent=tmp_path / "owned",
            base_commit=head,
            base_tree=tree,
            task_id="AAE-041-prodroot",
            attempt=1,
            lifecycle_store=store,
        )
    assert excinfo.value.reason_code == "production_root"


# ---------------------------------------------------------------------------
# Full admission pipeline
# ---------------------------------------------------------------------------


def test_execute_and_admit_happy_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    executor = IsolatedMutationWorktreeExecutor.create(
        repo_root=repo,
        worktree_parent=tmp_path / "owned-wts",
        lifecycle_store=store,
    )
    candidate = _candidate()
    manifest = _manifest()
    root_before = (repo / "mod.py").read_text(encoding="utf-8")

    result = executor.execute_and_admit(
        candidate,
        file_replacements={"mod.py": MOD_MUTATED},
        task_id="AAE-041-exec",
        attempt=1,
        base_commit=head,
        base_tree=tree,
        assurance_manifest=manifest,
        cleanup=True,
    )

    assert result.executed is True
    assert result.disposition == AdmissionDisposition.ADMITTED.value
    assert result.admitted is True
    assert AdmissionReasonCode.OK.value in result.reason_codes
    assert result.identity_cid
    assert result.cleaned is True
    assert result.apply is not None
    assert result.apply["applied"] is True
    assert result.admission is not None
    assert result.admission["admitted"] is True
    assert "unit.test_branch" in result.admission["predicted_detector_ids"]
    assert result.root_head == head
    # Production never mutated.
    assert (repo / "mod.py").read_text(encoding="utf-8") == root_before
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head
    # Disposable worktree cleaned.
    if result.worktree_path:
        assert not Path(result.worktree_path).exists()

    payload = result.to_dict()
    assert payload["schema"]
    assert payload["interface"] == ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE
    assert payload["evidence"] == AAE_ISOLATED_EXECUTOR_EVIDENCE
    json.dumps(payload)  # serializable


def test_execute_and_admit_rejects_invalid_syntax(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    executor = IsolatedMutationWorktreeExecutor.create(
        repo_root=repo,
        worktree_parent=tmp_path / "owned-wts",
        lifecycle_store=store,
    )
    result = executor.execute_and_admit(
        _candidate(),
        file_replacements={"mod.py": MOD_SYNTAX_ERROR},
        task_id="AAE-041-syntax",
        attempt=1,
        base_commit=head,
        base_tree=tree,
        assurance_manifest=_manifest(),
        cleanup=True,
    )
    assert result.executed is True  # apply succeeded; admission failed closed
    assert result.disposition in {
        AdmissionDisposition.INVALID.value,
        AdmissionDisposition.REJECTED.value,
    }
    assert result.admitted is False
    assert any(
        code in result.reason_codes
        for code in (
            AdmissionReasonCode.PARSE_FAILURE.value,
            AdmissionReasonCode.STRUCTURAL_INVALID.value,
            AdmissionReasonCode.TRIVIAL_INVALIDITY.value,
        )
    )
    assert result.cleaned is True
    assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE


def test_manual_create_apply_admit_pipeline(tmp_path: Path) -> None:
    """AAE-024 only validates caller-supplied worktrees; lifecycle stays here."""

    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    wt = parent / "manual"

    isolated = create_mutation_worktree(
        repo_root=repo,
        worktree_path=wt,
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-manual",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        applied = isolated.apply_replacements(
            {"mod.py": MOD_MUTATED},
            MutationWriteScope.from_candidate(_candidate()),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
        )
        assert applied.applied is True
        admission = isolated.admit(
            _candidate(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            assurance_manifest=_manifest(),
        )
        assert admission.admitted is True
        assert admission.disposition == AdmissionDisposition.ADMITTED.value
        assert admission.worktree_path == normalize_workspace_path(wt)
        assert admission.lease_id == isolated.lease_id
        assert int(admission.fence) == int(isolated.fence)
    finally:
        isolated.cleanup(
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            reason="test_done",
        )
    assert not wt.exists()


# ---------------------------------------------------------------------------
# Fencing / recovery
# ---------------------------------------------------------------------------


def test_stale_owner_cannot_apply_or_clean(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    isolated = create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "peer",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-peer",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        live_lease = isolated.lease_id
        live_fence = isolated.fence
        stale_fence = live_fence - 1 if live_fence > 1 else 0

        with pytest.raises(MutationWorktreeFenceError):
            isolated.apply_replacements(
                {"mod.py": MOD_MUTATED},
                _scope(),
                lease_id=live_lease,
                fence=stale_fence,
            )

        with pytest.raises(MutationWorktreeFenceError):
            isolated.cleanup(
                lease_id="not-the-owner",
                fence=live_fence,
            )

        decision = store.evaluate_cleanup(workspace_path=isolated.worktree_path)
        assert decision.allowed is False
        assert decision.disposition is CleanupDisposition.DENY

        # Live owner can still clean.
        cleaned = isolated.cleanup(
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            reason="test_done",
        )
        assert cleaned["cleaned"] is True
        assert not isolated.worktree_path.exists()
    finally:
        if not isolated._closed:
            try:
                isolated.cleanup(
                    lease_id=isolated.lease_id,
                    fence=isolated.fence,
                    reason="finally",
                )
            except Exception:
                recover_mutation_worktree(
                    lifecycle_store=store,
                    worktree_path=isolated.worktree_path,
                    repo_root=repo,
                    worktree_parent=parent,
                )


def test_interrupted_prepare_recovers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    workspace = tmp_path / "owned-wts" / "interrupted-prep"
    record = store.begin_preparing(
        task_id="AAE-041-irecover",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="aae-mutant/irecover-a1",
        merge_target="HEAD",
    )
    import hashlib

    digest = hashlib.sha256(
        normalize_workspace_path(workspace).encode("utf-8")
    ).hexdigest()[:16]
    journal_path = Path(store.store_dir) / f"aae-attempt-{digest}.json"  # type: ignore[arg-type]
    journal_path.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worktree-attempt@1",
                "phase": "preparing",
                "lease_id": record.lease_id,
                "fence": record.fence,
                "repo_root": str(repo),
                "worktree_path": str(workspace),
                "worktree_parent": str(tmp_path / "owned-wts"),
                "base_commit": head,
                "base_tree": tree,
                "task_id": "AAE-041-irecover",
                "attempt": 1,
            }
        ),
        encoding="utf-8",
    )
    recovery = recover_mutation_worktree(
        lifecycle_store=store,
        worktree_path=workspace,
        repo_root=repo,
        worktree_parent=tmp_path / "owned-wts",
        caller_lease_id=record.lease_id,
    )
    assert recovery["recovered"] is True
    assert "marked_terminal" in recovery["actions"]
    loaded = store.load_workspace(workspace)
    assert loaded is not None and loaded.is_terminal
    decision = store.evaluate_cleanup(workspace_path=workspace)
    assert decision.allowed is True


def test_interrupted_apply_recovers_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    wt_path = parent / "interrupted-apply"
    isolated = create_mutation_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-iapply",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        (isolated.worktree_path / "mod.py").write_text(
            "VALUE = dirty\n", encoding="utf-8"
        )
        isolated.phase = MutationWorktreePhase.APPLYING
        isolated._write_journal()
        recovery = recover_mutation_worktree(
            lifecycle_store=store,
            worktree_path=wt_path,
            repo_root=repo,
            worktree_parent=parent,
            caller_lease_id=isolated.lease_id,
        )
        assert recovery["recovered"] is True
        assert "reset_base" in recovery["actions"] or "removed" in recovery["actions"]
        if wt_path.exists():
            content = (wt_path / "mod.py").read_text(encoding="utf-8")
            assert content == MOD_SOURCE
        assert (repo / "mod.py").read_text(encoding="utf-8") == MOD_SOURCE
    finally:
        recover_mutation_worktree(
            lifecycle_store=store,
            worktree_path=wt_path,
            repo_root=repo,
            worktree_parent=parent,
            caller_lease_id=isolated.lease_id,
        )


def test_context_manager_cleans_up(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    wt_path = parent / "cm"
    with create_mutation_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-cm",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        assert isolated.worktree_path.is_dir()
        assert isolated.phase is MutationWorktreePhase.READY
        assert isinstance(isolated, IsolatedMutationWorktree)
    assert not wt_path.exists()
    record = store.load_workspace(wt_path)
    assert record is not None
    assert record.is_terminal


@pytest.mark.skipif(os.name != "posix", reason="lifecycle fencing assumes POSIX")
def test_duplicate_attempt_claim_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    first = create_mutation_worktree(
        repo_root=repo,
        worktree_path=parent / "dup-1",
        worktree_parent=parent,
        base_commit=head,
        base_tree=tree,
        task_id="AAE-041-dup",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        with pytest.raises(MutationWorktreeFenceError) as excinfo:
            create_mutation_worktree(
                repo_root=repo,
                worktree_path=parent / "dup-2",
                worktree_parent=parent,
                base_commit=head,
                base_tree=tree,
                task_id="AAE-041-dup",
                attempt=1,
                lifecycle_store=store,
            )
        assert excinfo.value.reason_code == "duplicate_attempt"
    finally:
        first.cleanup(lease_id=first.lease_id, fence=first.fence)


def test_network_git_subcommand_rejected(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import worktrees as wt

    repo = tmp_path / "repo"
    _init_repo(repo)
    with pytest.raises(MutationWorktreeError) as excinfo:
        wt._run_git(["fetch", "origin"], cwd=repo)
    assert excinfo.value.reason_code == "network_denied"


def test_executor_to_dict_and_descriptor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    executor = IsolatedMutationWorktreeExecutor.create(
        repo_root=repo,
        worktree_parent=tmp_path / "owned-wts",
        lifecycle_store=store,
    )
    payload = executor.to_dict()
    assert payload["interface"] == ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE
    assert payload["evidence"] == AAE_ISOLATED_EXECUTOR_EVIDENCE
    assert payload["admit_mutation_interface"] == ADMIT_MUTATION_INTERFACE
    json.dumps(payload)


def test_recovery_refuses_production_root_removal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    # Publish a terminal lifecycle claim for the production root path (misconfig).
    record = store.begin_preparing(
        task_id="AAE-041-prod-clean",
        attempt=1,
        lane_id="lane",
        workspace_path=repo,
        branch="aae-mutant/prod-a1",
        merge_target="HEAD",
    )
    store.mark_active(repo, lease_id=record.lease_id, expected_fence=record.fence)
    record = store.load_workspace(repo)
    assert record is not None
    store.mark_terminal(
        repo,
        lease_id=record.lease_id,
        expected_fence=record.fence,
        reason="test",
    )
    recovery = recover_mutation_worktree(
        lifecycle_store=store,
        worktree_path=repo,
        repo_root=repo,
        worktree_parent=tmp_path / "owned-wts",
        caller_lease_id=record.lease_id,
    )
    assert recovery["recovered"] is True
    assert "production_root_denied" in recovery["actions"]
    assert repo.is_dir()
    assert (repo / "mod.py").exists()


def test_apply_is_deterministic_post_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    parent = tmp_path / "owned-wts"
    trees: list[str] = []
    for attempt in (1, 2):
        with create_mutation_worktree(
            repo_root=repo,
            worktree_path=parent / f"det-{attempt}",
            worktree_parent=parent,
            base_commit=head,
            base_tree=tree,
            task_id="AAE-041-det",
            attempt=attempt,
            lifecycle_store=store,
        ) as isolated:
            result = apply_file_replacements(
                {"mod.py": MOD_MUTATED},
                worktree_root=isolated.worktree_path,
                scope=_scope(),
                expected_base_commit=head,
                expected_base_tree=tree,
            )
            assert result.applied
            trees.append(result.post_tree)
    assert trees[0] == trees[1]
    assert trees[0] != tree

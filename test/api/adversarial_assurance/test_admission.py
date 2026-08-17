"""Tests for isolated mutant rescan and semantic admission (AAE-024)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.admission import (
    ADMIT_MUTATION_INTERFACE,
    MUTATION_ADMISSION_SCHEMA,
    AdmissionDisposition,
    AdmissionError,
    AdmissionReasonCode,
    EquivalenceEstimate,
    MutationAdmissionResult,
    admit_mutation,
    admission_dispositions,
    admission_reason_codes,
    blocked_authority_path,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
    current_process_birth,
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


REPO_ID = "repository:sha256:test-repo-identity-aae024"
REPO_STATE = _cid("repo-state-aae024")


def _run(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _run(["git", "init"], cwd=root)
    _run(["git", "config", "user.email", "aae024@example.com"], cwd=root)
    _run(["git", "config", "user.name", "AAE024 Test"], cwd=root)
    # Deterministic default branch.
    _run(["git", "checkout", "-b", "main"], cwd=root)


def _commit_file(root: Path, rel: str, content: str, *, message: str = "commit") -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _run(["git", "add", "--", rel], cwd=root)
    _run(["git", "commit", "-m", message], cwd=root)
    oid = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return oid


def _add_worktree(repo: Path, worktree: Path, *, branch: str) -> None:
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "worktree", "add", "-b", branch, str(worktree), "HEAD"],
        cwd=repo,
    )


def _lifecycle(
    repo: Path,
    worktree: Path,
    *,
    lease_id: str = "lease-aae024",
    fence: int = 1,
    task_id: str = "AAE-024",
) -> WorktreeLifecycleStore:
    store = WorktreeLifecycleStore(repo_root=repo)
    store.begin_preparing(
        task_id=task_id,
        canonical_task_cid=_cid(f"task:{task_id}"),
        attempt=1,
        lane_id="lane-0",
        workspace_path=worktree,
        branch=f"mutant/{task_id}",
        merge_target="main",
        lease_id=lease_id,
        owner=current_process_birth(),
    )
    store.mark_active(worktree, lease_id=lease_id, expected_fence=fence)
    # Reload to get actual fence after mark_active may bump it.
    return store


def _active_lease_fence(
    store: WorktreeLifecycleStore, worktree: Path
) -> tuple[str, int]:
    record = store.load_workspace(worktree)
    assert record is not None
    return record.lease_id, int(record.fence)


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
        "seed": 42,
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
    """Sample module for AAE-024 admission tests."""


    def fn(flag: bool) -> int:
        if flag:
            return 1
        return 0
    '''
)

MOD_MUTATED = textwrap.dedent(
    '''\
    """Sample module for AAE-024 admission tests."""


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


@pytest.fixture
def mutant_env(tmp_path: Path) -> dict[str, object]:
    """Fresh repo + owned disposable worktree with a clean mod.py baseline."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _commit_file(repo, "mod.py", MOD_SOURCE, message="baseline mod.py")
    worktree = tmp_path / "worktrees" / "mutant-1"
    _add_worktree(repo, worktree, branch="mutant/AAE-024-1")
    store = _lifecycle(repo, worktree, lease_id="lease-aae024", fence=1)
    lease_id, fence = _active_lease_fence(store, worktree)
    return {
        "repo": repo,
        "worktree": worktree,
        "store": store,
        "lease_id": lease_id,
        "fence": fence,
        "base": base,
        "candidate": _candidate(),
        "manifest": _manifest(),
    }


# ---------------------------------------------------------------------------
# Vocabulary / helpers
# ---------------------------------------------------------------------------


def test_admission_vocabularies_are_closed() -> None:
    codes = admission_reason_codes()
    assert AdmissionReasonCode.OK.value in codes
    assert AdmissionReasonCode.AUTHORITY_PATH_BLOCKED.value in codes
    assert AdmissionReasonCode.WORKTREE_PRODUCTION_ROOT.value in codes
    dispositions = admission_dispositions()
    assert set(dispositions) == {
        "admitted",
        "rejected",
        "invalid",
        "equivalent",
    }
    assert ADMIT_MUTATION_INTERFACE == "admit_mutation@1"
    assert "mutation-admission" in MUTATION_ADMISSION_SCHEMA


def test_blocked_authority_path_detects_verifier_policy_key_oracle() -> None:
    assert blocked_authority_path("ipfs_accelerate_py/agent_supervisor/verification/plan.py")
    assert blocked_authority_path("config/policy.json")
    assert blocked_authority_path("secrets/api.pem")
    assert blocked_authority_path("lib/oracle/judge.py")
    assert blocked_authority_path("pkg/policy/engine.py")
    assert not blocked_authority_path("mod.py")
    assert not blocked_authority_path("pkg/utils/helpers.py")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_admit_mutation_happy_path_rescans_predicts_and_commits_identity(
    mutant_env: dict[str, object],
) -> None:
    worktree = mutant_env["worktree"]  # type: ignore[assignment]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")

    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )

    assert isinstance(result, MutationAdmissionResult)
    assert result.disposition == AdmissionDisposition.ADMITTED.value
    assert result.admitted is True
    assert AdmissionReasonCode.OK.value in result.reason_codes
    assert result.identity_cid
    assert result.candidate_id == "cand_control_flow_invert_0"
    assert result.detection_set_cid
    assert result.detection_set_id
    assert "unit.test_branch" in result.predicted_detector_ids
    assert result.scan is not None
    assert result.scan["changed_paths"]
    assert result.scan["changed_paths"][0]["path"] == "mod.py"
    assert result.equivalence_status == EquivalenceEstimate.NOT_EQUIVALENT.value
    assert result.lifecycle_record_id

    # Deterministic identity for identical inputs.
    again = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert again.identity_cid == result.identity_cid
    assert again.admission_cid == result.admission_cid
    assert again.to_dict()["schema"] == MUTATION_ADMISSION_SCHEMA


def test_admit_mutation_does_not_create_or_destroy_worktrees(
    mutant_env: dict[str, object],
) -> None:
    worktree = mutant_env["worktree"]  # type: ignore[assignment]
    assert isinstance(worktree, Path)
    repo = mutant_env["repo"]
    assert isinstance(repo, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")

    before_worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    before_exists = worktree.is_dir()

    admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=repo,
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )

    after_worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before_exists and worktree.is_dir()
    assert before_worktrees == after_worktrees


# ---------------------------------------------------------------------------
# Worktree ownership guardrails
# ---------------------------------------------------------------------------


def test_reject_missing_worktree(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = admit_mutation(
        _candidate(),
        worktree_path=missing,
        lease_id="lease",
        fence=1,
        lifecycle_store=None,
        require_lifecycle=False,
        assurance_manifest=_manifest(),
    )
    assert result.admitted is False
    assert result.disposition == AdmissionDisposition.REJECTED.value
    assert AdmissionReasonCode.WORKTREE_MISSING.value in result.reason_codes


def test_reject_production_root_as_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "mod.py", MOD_SOURCE)
    store = WorktreeLifecycleStore(repo_root=repo)
    # Even with a lifecycle record, targeting repo root is forbidden.
    store.begin_preparing(
        task_id="t",
        attempt=1,
        lane_id="l",
        workspace_path=repo,
        branch="mutant/bad",
        merge_target="main",
        lease_id="lease-root",
        owner=current_process_birth(),
    )
    record = store.load_workspace(repo)
    assert record is not None
    result = admit_mutation(
        _candidate(),
        worktree_path=repo,
        lease_id=record.lease_id,
        fence=record.fence,
        lifecycle_store=store,
        repo_root=repo,
        assurance_manifest=_manifest(),
    )
    assert result.admitted is False
    assert AdmissionReasonCode.WORKTREE_PRODUCTION_ROOT.value in result.reason_codes


def test_reject_unowned_worktree(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id="wrong-lease",
        fence=999,
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert (
        AdmissionReasonCode.WORKTREE_OWNERSHIP_MISMATCH.value in result.reason_codes
    )


def test_reject_without_lifecycle_when_required(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=None,
        require_lifecycle=True,
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert AdmissionReasonCode.LIFECYCLE_STORE_REQUIRED.value in result.reason_codes


# ---------------------------------------------------------------------------
# Declared-change rescan / authority blocks
# ---------------------------------------------------------------------------


def test_reject_undeclared_path_change(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    (worktree / "extra.py").write_text("x = 1\n", encoding="utf-8")
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert AdmissionReasonCode.UNDECLARED_PATH_CHANGE.value in result.reason_codes
    assert result.scan is not None
    assert "extra.py" in result.scan["undeclared_paths"]


def test_reject_authority_path_edits(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    # Declare the authority path so undeclared-path check is not the first failure.
    auth_rel = "config/policy.json"
    (worktree / "config").mkdir(parents=True, exist_ok=True)
    (worktree / auth_rel).write_text('{"allow": true}\n', encoding="utf-8")
    candidate = _candidate(scope_paths=(auth_rel,), scope_symbol_ids=("policy",))
    result = admit_mutation(
        candidate,
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=_manifest(),
        declared_paths=(auth_rel,),
    )
    assert result.admitted is False
    assert AdmissionReasonCode.AUTHORITY_PATH_BLOCKED.value in result.reason_codes


def test_allow_authority_fixture_override(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    auth_rel = "pkg/oracle/judge.py"
    (worktree / "pkg" / "oracle").mkdir(parents=True, exist_ok=True)
    (worktree / auth_rel).write_text(
        "def judge(x):\n    return not x\n",
        encoding="utf-8",
    )
    candidate = _candidate(
        scope_paths=(auth_rel,),
        scope_symbol_ids=("judge",),
        expected_violated_property_classes=(PropertyClass.CONTROL_INVARIANT,),
    )
    manifest = _manifest(
        detectors=(
            DetectorCatalogEntry(
                detector_id="unit.oracle_judge",
                detector_revision="1.0.0",
                detector_kind=DetectorKind.UNIT_TEST,
                covered_property_classes=(PropertyClass.CONTROL_INVARIANT,),
                anchor_ids=("tests.oracle_judge",),
                default_strength=DetectorStrength.REQUIRED,
                expected_terminal_status=AssuranceTerminalStatus.COMPLETE,
                observation_template="unit test observes oracle mutation",
                claim_ids=("claim.oracle",),
                notes=None,
                metadata={},
            ),
        ),
        dependency_edges=(
            SemanticDependencyEdge(
                from_id="judge",
                to_id="tests.oracle_judge",
                relation=DependencyRelation.TESTED_BY,
                notes=None,
            ),
        ),
        claims=(
            ClaimBinding(
                claim_id="claim.oracle",
                property_class=PropertyClass.CONTROL_INVARIANT,
                statement="oracle judge control",
                symbol_ids=("judge",),
                notes=None,
                metadata={},
            ),
        ),
    )

    blocked = admit_mutation(
        candidate,
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=manifest,
        declared_paths=(auth_rel,),
        allow_authority_fixture=False,
    )
    assert blocked.admitted is False
    assert AdmissionReasonCode.AUTHORITY_PATH_BLOCKED.value in blocked.reason_codes

    allowed = admit_mutation(
        candidate,
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=manifest,
        declared_paths=(auth_rel,),
        allow_authority_fixture=True,
    )
    assert allowed.admitted is True
    assert allowed.disposition == AdmissionDisposition.ADMITTED.value


# ---------------------------------------------------------------------------
# Parse / trivial invalidity / empty diff
# ---------------------------------------------------------------------------


def test_reject_parse_failure(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_SYNTAX_ERROR, encoding="utf-8")
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert result.disposition == AdmissionDisposition.INVALID.value
    assert AdmissionReasonCode.PARSE_FAILURE.value in result.reason_codes


def test_reject_empty_diff(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    # No edits relative to base.
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert result.disposition == AdmissionDisposition.INVALID.value
    assert AdmissionReasonCode.EMPTY_DIFF.value in result.reason_codes


def test_reject_missing_manifest(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=None,
    )
    assert result.admitted is False
    assert AdmissionReasonCode.MANIFEST_REQUIRED.value in result.reason_codes


def test_reject_empty_declared_paths(mutant_env: dict[str, object]) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    candidate = _candidate(scope_paths=())
    result = admit_mutation(
        candidate,
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        declared_paths=(),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.admitted is False
    assert AdmissionReasonCode.NO_DECLARED_CHANGES.value in result.reason_codes


# ---------------------------------------------------------------------------
# Equivalence estimate
# ---------------------------------------------------------------------------


def test_equivalent_estimate_for_ast_identical_formatting(
    mutant_env: dict[str, object],
) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    # Semantically identical AST with different whitespace/formatting.
    formatted = textwrap.dedent(
        '''\
        """Sample module for AAE-024 admission tests."""


        def fn(flag: bool) -> int:
            if flag:
                return 1
            return 0
        '''
    )
    # Ensure it differs by bytes from baseline but AST-dumps equal.
    baseline = (worktree / "mod.py").read_text(encoding="utf-8")
    # Add trailing spaces / blank lines without changing AST.
    mutated = baseline.rstrip() + "\n\n\n"
    if mutated == baseline:
        mutated = baseline.replace("return 1", "return 1  ")
    (worktree / "mod.py").write_text(mutated, encoding="utf-8")
    # If still AST-identical after parse...
    import ast

    assert ast.dump(ast.parse(baseline)) == ast.dump(ast.parse(mutated))

    result = admit_mutation(
        mutant_env["candidate"],  # type: ignore[arg-type]
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    assert result.disposition == AdmissionDisposition.EQUIVALENT.value
    assert result.admitted is False
    assert result.equivalence_status == EquivalenceEstimate.EQUIVALENT.value
    assert AdmissionReasonCode.EQUIVALENCE_ESTIMATE.value in result.reason_codes
    assert result.identity_cid  # identity still committed


def test_likely_equivalent_never_auto_promotes_to_equivalent(
    mutant_env: dict[str, object],
) -> None:
    worktree = mutant_env["worktree"]
    assert isinstance(worktree, Path)
    (worktree / "mod.py").write_text(MOD_MUTATED, encoding="utf-8")
    candidate = _candidate(likely_equivalent=True)
    result = admit_mutation(
        candidate,
        worktree_path=worktree,
        lease_id=str(mutant_env["lease_id"]),
        fence=int(mutant_env["fence"]),  # type: ignore[arg-type]
        lifecycle_store=mutant_env["store"],  # type: ignore[arg-type]
        repo_root=mutant_env["repo"],  # type: ignore[arg-type]
        base_commit=str(mutant_env["base"]),
        assurance_manifest=mutant_env["manifest"],  # type: ignore[arg-type]
    )
    # AST differs, so not_equivalent wins over likely_equivalent flag.
    assert result.admitted is True
    assert result.equivalence_status == EquivalenceEstimate.NOT_EQUIVALENT.value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_malformed_fence_raises() -> None:
    with pytest.raises(AdmissionError, match="positive"):
        admit_mutation(
            _candidate(),
            worktree_path="/tmp/x",
            lease_id="lease",
            fence=0,
            require_lifecycle=False,
        )


def test_invalid_candidate_mapping_raises() -> None:
    with pytest.raises(AdmissionError):
        admit_mutation(
            {"not": "a candidate"},
            worktree_path="/tmp/x",
            lease_id="lease",
            fence=1,
            require_lifecycle=False,
        )


def test_admission_module_public_surface() -> None:
    """Declared surface is the admission module (package __init__ is out of scope)."""
    from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import admission as mod

    assert callable(mod.admit_mutation)
    assert mod.ADMIT_MUTATION_INTERFACE == ADMIT_MUTATION_INTERFACE
    assert mod.ADMIT_MUTATION_INTERFACE == "admit_mutation@1"
    assert "admit_mutation" in mod.__all__

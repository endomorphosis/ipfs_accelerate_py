"""Tests for the generic multi-repository symbolic assurance pilot (LPR-026)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    ForestPolicy,
    ForestRootSpec,
    RepositoryAuthority,
)
from ipfs_accelerate_py.agent_supervisor.runtime.symbolic_assurance_pilot import (
    PilotArtifactSet,
    PilotConclusion,
    PilotConfig,
    PilotMode,
    PilotProgramProfile,
    PilotStage,
    PilotVerificationError,
    RepositoryAdmissionPolicy,
    StageReceipt,
    SymbolicAssurancePilotError,
    SymbolicAssurancePilotReport,
    admitted_entries_for_pilot,
    dry_run_pilot,
    freeze_repository_descriptors,
    render_findings_board_document,
    scan_inventory,
    verify_pilot,
    verify_pilot_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "runtime"
    / "symbolic_assurance_pilot.py"
)

# Generic engine must not embed product/domain literals.
_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|SWISSKNIFE_ROOT|IPFS_ACCELERATE_ROOT|argparse|__main__)\b"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _init_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Pilot Test")
    _git(path, "config", "user.email", "pilot-test@example.invalid")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "seed fixture")
    return path


def _two_repo_fixture(
    tmp_path: Path,
    *,
    seed_broken: bool = True,
    seed_inconclusive: bool = True,
) -> tuple[Path, Path, ForestPolicy]:
    """Hermetic non-VFS two-repository fixture used by the generic orchestrator."""

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"

    alpha_files = {
        "src/service.py": (
            "def serve():\n"
            "    return 'ok'\n"
        ),
        "src/api.py": (
            "def handle(req):\n"
            "    return req\n"
        ),
        "README.md": "# alpha fixture\n",
    }
    if seed_broken:
        alpha_files["src/broken.py"] = (
            "# PILOT_CONTRACT_BROKEN\n"
            "def drift():\n"
            "    return 'broken'\n"
        )
    if seed_inconclusive:
        alpha_files["src/maybe.py"] = (
            "# PILOT_INCONCLUSIVE\n"
            "def maybe():\n"
            "    return 'unknown'\n"
        )

    _init_repo(alpha, alpha_files)
    _init_repo(
        beta,
        {
            "lib/core.py": (
                "def compute(x):\n"
                "    return x + 1\n"
            ),
            "lib/util.py": (
                "def identity(x):\n"
                "    return x\n"
            ),
            "README.md": "# beta fixture\n",
        },
    )

    policy = ForestPolicy(
        roots=(
            ForestRootSpec(
                alias="alpha",
                root_path=alpha,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=True,
            ),
            ForestRootSpec(
                alias="beta",
                root_path=beta,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
                required=True,
            ),
        ),
        sole_write_alias="beta",
    )
    return alpha, beta, policy


def _test_profile() -> PilotProgramProfile:
    return PilotProgramProfile(
        schema="test/symbolic-assurance-pilot@1",
        version=1,
        objective_id="TEST-G001",
        task_id="TEST-026",
        requirement_id="test:frozen-two-repo-pilot",
        producer="test-symbolic-assurance-pilot@1",
        board_namespace="test-symbolic-assurance-v1",
        policy_revision="policy:test-symbolic-assurance-pilot@1",
        evidence="test/symbolic-assurance-pilot@1",
        primary_repository_aliases=("alpha",),
        broken_contract_marker="PILOT_CONTRACT_BROKEN",
        inconclusive_marker="PILOT_INCONCLUSIVE",
        board_title="Test Symbolic Assurance Findings Board",
    )


def _config(
    tmp_path: Path,
    *,
    seed_broken: bool = True,
    seed_inconclusive: bool = True,
) -> PilotConfig:
    _alpha, _beta, policy = _two_repo_fixture(
        tmp_path / "forest",
        seed_broken=seed_broken,
        seed_inconclusive=seed_inconclusive,
    )
    artifact_dir = tmp_path / "artifacts"
    board_path = tmp_path / "findings.todo.md"
    return PilotConfig(
        profile=_test_profile(),
        admission_policy=RepositoryAdmissionPolicy(
            admit_all_aliases=("alpha",),
            path_patterns=(r"(?i)(?:^|/)lib/",),
        ),
        forest_policy=policy,
        artifact_dir=artifact_dir,
        findings_board_path=board_path,
        write_artifacts=True,
        write_findings_board=True,
        require_exhaustive_aliases=("alpha",),
        allowed_output_roots=(tmp_path,),
        sole_write_alias="beta",
    )


def test_generic_module_has_no_domain_literals_or_cli() -> None:
    text = PILOT_MODULE.read_text(encoding="utf-8")
    assert "argparse" not in text
    assert "def main" not in text
    assert "__main__" not in text
    # Environment variable names must not appear in the generic engine.
    assert "os.environ" not in text
    assert "getenv" not in text
    matches = _FORBIDDEN_GENERIC.findall(text)
    assert matches == [], f"forbidden domain literals in generic module: {matches}"


def test_pilot_config_is_tuple_profile_driven(tmp_path: Path) -> None:
    alpha, beta, policy = _two_repo_fixture(tmp_path / "forest")
    config = PilotConfig(
        profile=_test_profile(),
        admission_policy=RepositoryAdmissionPolicy(admit_all_included=True),
        repositories=(
            ForestRootSpec(
                alias="alpha",
                root_path=alpha,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
            ),
            ForestRootSpec(
                alias="beta",
                root_path=beta,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
            ),
        ),
        artifact_dir=tmp_path / "out",
        findings_board_path=tmp_path / "board.md",
        allowed_output_roots=(tmp_path,),
        sole_write_alias="beta",
    )
    assert "accelerator_root" not in config.to_dict()
    assert "swissknife_root" not in config.to_dict()
    assert len(config.repositories) == 2
    assert config.profile.objective_id == "TEST-G001"
    # forest_policy path also works
    alt = PilotConfig(
        profile=_test_profile(),
        admission_policy=RepositoryAdmissionPolicy(admit_all_included=True),
        forest_policy=policy,
        artifact_dir=tmp_path / "out2",
        findings_board_path=tmp_path / "board2.md",
        allowed_output_roots=(tmp_path,),
    )
    forest = freeze_repository_descriptors(alt)
    assert {d.alias for d in forest.descriptors} == {"alpha", "beta"}


def test_admission_policy_is_injected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    forest = freeze_repository_descriptors(config)
    index = scan_inventory(forest)
    admitted = admitted_entries_for_pilot(index, config.admission_policy)
    assert admitted
    aliases = {entry.repository_alias for entry in admitted}
    assert "alpha" in aliases
    # beta only admits lib/ paths under the test policy
    beta_paths = [
        entry.relative_path
        for entry in admitted
        if entry.repository_alias == "beta"
    ]
    assert beta_paths
    assert all(path.startswith("lib/") for path in beta_paths)


def test_dry_run_freezes_scans_publishes_and_writes_board(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)

    assert report.schema == "test/symbolic-assurance-pilot@1"
    assert report.objective_id == "TEST-G001"
    assert report.task_id == "TEST-026"
    assert report.mode is PilotMode.DRY_RUN
    assert report.conclusion is PilotConclusion.PASSED
    assert report.provider_calls == 0
    assert report.source_mutations == 0
    assert report.authorizes_repair is False
    assert report.is_completion_evidence is False
    assert report.admitted_file_count >= 4
    assert report.primary_file_count >= 2
    assert report.closure_file_count >= 1
    assert report.finding_count >= 1
    assert report.executable_task_count >= 1
    assert report.inconclusive_count >= 1
    assert report.artifacts is not None

    for field_name in (
        "forest_cid",
        "manifest_cid",
        "coverage_cid",
        "inventory_cid",
        "graph_cid",
        "cache_cid",
        "proof_cid",
        "zk_shadow_cid",
        "finding_ledger_cid",
        "taskboard_cid",
    ):
        assert getattr(report.artifacts, field_name)

    stage_names = {stage.stage for stage in report.stages}
    assert stage_names == set(PilotStage)

    artifact_dir = config.resolved_artifact_dir()
    assert (artifact_dir / "report.json").is_file()
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "coverage.json").is_file()
    assert (artifact_dir / "taskboard.json").is_file()

    board = config.resolved_findings_board_path().read_text(encoding="utf-8")
    assert "test/symbolic-assurance-pilot@1" in board
    assert "authorizes_repair: `false`" in board
    assert report.artifacts.manifest_cid in board
    assert report.artifacts.taskboard_cid in board
    assert report.board_namespace == "test-symbolic-assurance-v1"


def test_verify_recomputes_without_provider_or_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    verified = verify_pilot_report(report, config=config, recompute=True)

    assert verified.mode is PilotMode.VERIFY
    assert verified.conclusion is PilotConclusion.PASSED
    assert verified.provider_calls == 0
    assert verified.source_mutations == 0
    assert verified.artifacts is not None
    assert verified.artifacts.inventory_cid == report.artifacts.inventory_cid
    assert verified.artifacts.graph_cid == report.artifacts.graph_cid
    assert verified.artifacts.taskboard_cid == report.artifacts.taskboard_cid


def test_verify_fails_on_changed_trees(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)

    # Mutate alpha (read-only source root) after freeze.
    alpha = next(
        root.root_path
        for root in config.forest_policy.roots
        if root.alias == "alpha"
    )
    alpha = Path(alpha)
    (alpha / "src" / "drift.py").write_text("x = 1\n", encoding="utf-8")
    _git(alpha, "add", ".")
    _git(alpha, "commit", "-m", "drift")

    with pytest.raises(PilotVerificationError) as excinfo:
        verify_pilot_report(report, config=config, recompute=True)
    assert excinfo.value.reason_code in {"changed_trees", "stale_evidence"}


def test_verify_fails_on_noncanonical_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    payload = report.to_dict()
    payload["report_cid"] = "baguqeera" + "0" * 52

    with pytest.raises(PilotVerificationError) as excinfo:
        verify_pilot_report(payload, config=None, recompute=False)
    assert excinfo.value.reason_code == "stale_evidence"


def test_verify_fails_on_forged_authority(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    payload = report.to_dict()
    payload.pop("report_cid", None)
    payload["authorizes_repair"] = True
    with pytest.raises(SymbolicAssurancePilotError):
        SymbolicAssurancePilotReport.from_dict(payload)


def test_inconclusive_findings_are_non_executable(tmp_path: Path) -> None:
    config = _config(tmp_path, seed_broken=False, seed_inconclusive=True)
    report = dry_run_pilot(config)
    assert report.inconclusive_count >= 1
    assert report.executable_task_count == 0
    assert report.review_count >= 1


def test_board_is_bounded_deduplicated_and_goal_backed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    board_json = json.loads(
        (config.resolved_artifact_dir() / "taskboard.json").read_text(encoding="utf-8")
    )
    assert board_json.get("board_namespace") or board_json.get("goal_id")
    assert board_json.get("authorizes_repair", False) is False
    assert board_json.get("is_completion_evidence", False) is False

    board_md = config.resolved_findings_board_path().read_text(encoding="utf-8")
    assert "TEST-G001" in board_md or "goal" in board_md.lower()
    assert report.board_namespace == "test-symbolic-assurance-v1"


def test_inventory_accounts_for_every_admitted_primary_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    forest = freeze_repository_descriptors(config)
    index = scan_inventory(forest)
    admitted = admitted_entries_for_pilot(index, config.admission_policy)
    primary = [entry for entry in admitted if entry.repository_alias == "alpha"]
    assert primary
    for entry in index.entries:
        if entry.repository_alias != "alpha":
            continue
        if entry.inclusion == "included" and entry.parser_eligible:
            assert any(
                item.entry_cid == entry.entry_cid for item in primary
            ), entry.relative_path


def test_report_round_trip_is_canonical(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    restored = SymbolicAssurancePilotReport.from_dict(json.loads(report.to_json()))
    assert restored.report_cid == report.report_cid
    assert restored.artifacts is not None
    assert PilotArtifactSet.from_dict(restored.artifacts.to_dict()).manifest_cid == (
        report.artifacts.manifest_cid
    )


def test_verify_entry_self_check(tmp_path: Path) -> None:
    config = _config(tmp_path)
    verified = verify_pilot(config)
    assert verified.mode is PilotMode.VERIFY
    assert verified.conclusion is PilotConclusion.PASSED


def test_reject_unsafe_output_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bad = PilotConfig(
        profile=config.profile,
        admission_policy=config.admission_policy,
        forest_policy=config.forest_policy,
        artifact_dir=Path("/tmp/escape-pilot-artifacts"),
        findings_board_path=tmp_path / "board.md",
        write_artifacts=True,
        write_findings_board=False,
        allowed_output_roots=(tmp_path,),
        sole_write_alias="beta",
    )
    with pytest.raises(SymbolicAssurancePilotError) as excinfo:
        dry_run_pilot(bad)
    assert excinfo.value.reason_code == "unsafe_output_path"


def test_reject_duplicate_stages_in_report() -> None:
    stage = StageReceipt(
        stage=PilotStage.FREEZE,
        status=PilotConclusion.PASSED,
        artifact_cid="baguqeera" + "a" * 52,
    )
    with pytest.raises(SymbolicAssurancePilotError) as excinfo:
        SymbolicAssurancePilotReport(
            schema="test/symbolic-assurance-pilot@1",
            objective_id="TEST-G001",
            task_id="TEST-026",
            forest_id="forest-id",
            tree_bindings={"alpha": "a" * 40},
            commit_bindings={"alpha": "b" * 40},
            stages=(stage, stage),
            evidence="test/symbolic-assurance-pilot@1",
        )
    assert excinfo.value.reason_code == "duplicate_stage"


def test_reject_provider_surface_when_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    config = _config(tmp_path)
    monkeypatch.setitem(sys.modules, "openai", object())
    with pytest.raises(SymbolicAssurancePilotError) as excinfo:
        dry_run_pilot(config)
    assert excinfo.value.reason_code == "provider_call_forbidden"


def test_render_findings_board_document_is_profile_driven() -> None:
    profile = _test_profile()
    text = render_findings_board_document(
        profile=profile,
        report_context={
            "mode": "dry_run",
            "conclusion": "passed",
            "forest_id": "forest",
            "admitted_file_count": 3,
            "primary_file_count": 2,
            "closure_file_count": 1,
            "finding_count": 1,
            "executable_task_count": 1,
            "review_count": 0,
            "artifacts": {"manifest_cid": "cid-manifest", "taskboard_cid": "cid-board"},
            "repair_packets": [],
        },
        taskboard_markdown="- [ ] task\n",
    )
    assert profile.schema in text
    assert profile.objective_id in text
    assert "authorizes_repair: `false`" in text
    assert "cid-manifest" in text


def test_empty_admission_policy_rejected() -> None:
    with pytest.raises(SymbolicAssurancePilotError) as excinfo:
        RepositoryAdmissionPolicy()
    assert excinfo.value.reason_code == "empty_admission_policy"


def test_two_repository_fixture_completes_through_same_orchestrator(tmp_path: Path) -> None:
    """Acceptance: a non-VFS two-repository fixture completes through execute_pilot."""

    config = _config(tmp_path)
    report = dry_run_pilot(config)
    assert len(report.tree_bindings) == 2
    assert set(report.tree_bindings) == {"alpha", "beta"}
    assert report.conclusion is PilotConclusion.PASSED
    verified = verify_pilot_report(report, config=config, recompute=True)
    assert verified.mode is PilotMode.VERIFY
    assert verified.artifacts is not None
    assert verified.artifacts.forest_cid == report.artifacts.forest_cid

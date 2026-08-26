"""Focused fail-closed tests for the VRIF-030 recovery materializer."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    sha256_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/materialize_vrif_frozen_benchmark.py"


def _load_materializer() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        f"vrif_frozen_benchmark_materializer_test_{id(object())}", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write(repo: Path, relative: str, payload: bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_git_probe(
    tmp_path: Path,
    *,
    name: str,
    sentinel: Path,
    passthrough: bool = False,
    exit_code: int = 1,
) -> Path:
    probe = tmp_path / name
    probe.write_text(
        "#!/bin/sh\n"
        f'printf "invoked\\n" > {shlex.quote(str(sentinel))}\n'
        + ("/bin/cat\n" if passthrough else "")
        + f"exit {exit_code}\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    return probe


def _fixture_repository(
    tmp_path: Path,
    *,
    admission_identity_valid: bool = True,
) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "vrif-materializer@example.invalid")
    _git(repo, "config", "user.name", "VRIF materializer fixture")

    split_root = sha256_identity({"fixture": "split"})
    admission_body = {
        "schema": "vrif-materializer-fixture-admission@1",
        "disposition": "training_unavailable",
        "corpus_root": sha256_identity({"fixture": "corpus"}),
        "source_rights_root": sha256_identity({"fixture": "rights"}),
        "split_root": split_root,
    }
    admission_id = content_identity(admission_body)
    if not admission_identity_valid:
        admission_id = "baguq" + "a" * 60

    files = {
        "docs/architecture/agent_supervisor_residual_intelligence.objectives.md": (
            b"# Synthetic objectives\n"
        ),
        "docs/architecture/agent_supervisor_residual_intelligence.todo.md": (
            b"# Synthetic task board\n"
        ),
        "ipfs_accelerate_py/agent_supervisor/control/control_plane.py": (
            b"# Synthetic operation catalog\n"
        ),
        "config/agent_supervisor_residual_intelligence_scheduler.json": (
            _json_bytes({"provider": "synthetic"})
        ),
        (
            "docs/architecture/residual_intelligence_inventory/"
            "residual_model_call_inventory.json"
        ): _json_bytes({"model_calls": []}),
        (
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_training_admission.json"
        ): _json_bytes({**admission_body, "admission_id": admission_id}),
        (
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_split_manifest.json"
        ): _json_bytes({"split_root": split_root}),
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json": (
            b'{"placeholder":true}\n'
        ),
        "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl": (
            b'{"placeholder":true}\n'
        ),
        "test/api/residual_intelligence/test_benchmark.py": (
            b"from __future__ import annotations\n\n"
            b"# BEGIN VRIF-030 PORTAL BASELINE (materializer-owned)\n"
            b'VRIF_PORTAL_BASELINE_COMMIT = "0000000000000000000000000000000000000000"\n'
            b'VRIF_PORTAL_BASELINE_TREE = "1111111111111111111111111111111111111111"\n'
            b"# END VRIF-030 PORTAL BASELINE (materializer-owned)\n\n"
            b"def test_fixture() -> None:\n"
            b"    assert True\n"
        ),
    }
    for relative, payload in files.items():
        _write(repo, relative, payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "synthetic VRIF baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    return repo, commit, tree


def test_dry_run_write_and_check_are_deterministic_and_canonical(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materializer = _load_materializer()
    repo, baseline, tree = _fixture_repository(tmp_path)
    tracked = tuple(sorted(materializer._MUTABLE_PATHS))
    before = {relative: (repo / relative).read_bytes() for relative in tracked}

    assert (
        materializer.main(
            ["--repo-root", str(repo), "--baseline-commit", baseline, "--dry-run"]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert set(dry_run["changed_paths"]) == set(tracked)
    assert dry_run["baseline_commit"] == baseline
    assert dry_run["baseline_tree"] == tree
    assert dry_run["case_count"] == 96
    assert {relative: (repo / relative).read_bytes() for relative in tracked} == before

    assert (
        materializer.main(
            ["--repo-root", str(repo), "--baseline-commit", baseline, "--check"]
        )
        == 1
    )
    capsys.readouterr()
    assert (
        materializer.main(
            ["--repo-root", str(repo), "--baseline-commit", baseline, "--write"]
        )
        == 0
    )
    write_summary = json.loads(capsys.readouterr().out)
    assert set(write_summary["changed_paths"]) == set(tracked)
    assert set(_git(repo, "diff", "--name-only").splitlines()) == set(tracked)

    test_bytes = (repo / materializer.TEST_PATH).read_bytes()
    assert f'VRIF_PORTAL_BASELINE_COMMIT = "{baseline}"'.encode() in test_bytes
    assert f'VRIF_PORTAL_BASELINE_TREE = "{tree}"'.encode() in test_bytes
    assert test_bytes == materializer._rewrite_test_marker(
        before[materializer.TEST_PATH.as_posix()], commit=baseline, tree=tree
    )

    manifest_bytes = (repo / materializer.MANIFEST_PATH).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == _json_bytes(manifest)
    case_lines = (repo / materializer.CASES_PATH).read_bytes().splitlines(keepends=True)
    assert len(case_lines) == 96
    assert all(line == _json_bytes(json.loads(line)) for line in case_lines)

    assert (
        materializer.main(
            ["--repo-root", str(repo), "--baseline-commit", baseline, "--check"]
        )
        == 0
    )
    checked = json.loads(capsys.readouterr().out)
    assert checked["changed_paths"] == []
    stable = {relative: (repo / relative).read_bytes() for relative in tracked}
    assert (
        materializer.main(
            ["--repo-root", str(repo), "--baseline-commit", baseline, "--write"]
        )
        == 0
    )
    capsys.readouterr()
    assert {relative: (repo / relative).read_bytes() for relative in tracked} == stable


def test_materializer_hashes_final_test_bytes_with_operator_exact_nested_argv(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    result = materializer.build_materialization(repo, baseline_commit=baseline)
    expected_test = result["expected"][materializer.TEST_PATH.as_posix()]
    manifest = json.loads(result["expected"][materializer.MANIFEST_PATH.as_posix()])
    expected_validation_policy = sha256_identity(
        {
            "argv": [[materializer.VALIDATION_COMMAND]],
            "test_blob_identity": sha256_identity(expected_test),
        }
    )
    assert (
        manifest["benchmark_freeze"]["bindings"]["validation_policy"]
        == expected_validation_policy
    )
    assert manifest["benchmark_freeze"]["source"]["commit"] == baseline


def test_materialized_commit_passes_independent_portal_semantic_acceptance(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, tree = _fixture_repository(tmp_path)
    result = materializer.build_materialization(repo, baseline_commit=baseline)
    materializer.write_materialization(result)
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "materialize owner-exact VRIF benchmark")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    record = SimpleNamespace(
        task_alias="VRIF-030",
        outputs=tuple({"path": path} for path in sorted(materializer._MUTABLE_PATHS)),
        validations=({"argv": [materializer.VALIDATION_COMMAND]},),
        body={},
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
        repository_root=repo,
    )

    bridge._verify_vrif_benchmark_acceptance(
        record=record,
        baseline_commit=baseline,
        baseline_tree=tree,
        implementation_commit=implementation_commit,
    )


@pytest.mark.parametrize("defect", ["missing", "duplicate", "malformed"])
def test_materializer_rejects_missing_duplicate_or_malformed_marker(
    tmp_path: Path,
    defect: str,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    test_path = repo / materializer.TEST_PATH
    marker = test_path.read_bytes()
    if defect == "missing":
        marker = b"def test_fixture() -> None:\n    assert True\n"
    elif defect == "duplicate":
        marker = marker + b"\n" + marker
    else:
        marker = marker.replace(
            b'VRIF_PORTAL_BASELINE_TREE = "1111111111111111111111111111111111111111"',
            b'VRIF_PORTAL_BASELINE_TREE = "not-a-git-tree"',
        )
    test_path.write_bytes(marker)

    with pytest.raises(materializer.MaterializationError, match="baseline marker"):
        materializer.build_materialization(repo, baseline_commit=baseline)


@pytest.mark.parametrize("committed", [False, True])
def test_materializer_rejects_changes_outside_exact_output_scope(
    tmp_path: Path,
    committed: bool,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    objective = (
        repo / "docs/architecture/agent_supervisor_residual_intelligence.objectives.md"
    )
    objective.write_text(
        "# Unauthorized candidate objective change\n", encoding="utf-8"
    )
    if committed:
        _git(repo, "add", objective.relative_to(repo).as_posix())
        _git(repo, "commit", "-qm", "out-of-scope candidate change")

    with pytest.raises(materializer.MaterializationError, match="dirty outside"):
        materializer.build_materialization(repo, baseline_commit=baseline)


def test_materializer_rejects_invalid_baseline_and_admission_identity(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    with pytest.raises(materializer.MaterializationError, match="40-hex"):
        materializer.build_materialization(repo, baseline_commit=baseline[:-1])

    invalid_repo, invalid_baseline, _ = _fixture_repository(
        tmp_path / "invalid", admission_identity_valid=False
    )
    with pytest.raises(
        materializer.MaterializationError, match="identity does not verify"
    ):
        materializer.build_materialization(
            invalid_repo,
            baseline_commit=invalid_baseline,
        )


@pytest.mark.parametrize("defect", ["symlink", "untracked"])
def test_materializer_requires_regular_tracked_output_files(
    tmp_path: Path,
    defect: str,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    manifest = repo / materializer.MANIFEST_PATH
    if defect == "symlink":
        manifest.unlink()
        manifest.symlink_to("cases.jsonl")
    else:
        _git(repo, "rm", "--cached", materializer.MANIFEST_PATH.as_posix())

    with pytest.raises(materializer.MaterializationError, match="regular"):
        materializer.build_materialization(repo, baseline_commit=baseline)


def test_materializer_rejects_symlinked_input_parent_component(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, _, _ = _fixture_repository(tmp_path)
    original_parent = repo / "config"
    external_parent = tmp_path / "external-config"
    original_parent.rename(external_parent)
    original_parent.symlink_to(external_parent, target_is_directory=True)

    with pytest.raises(materializer.MaterializationError):
        materializer._read_regular_tracked_blob(repo, materializer.PROVIDER_PATH)


def test_materializer_rejects_output_parent_symlink_swap_without_external_write(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    result = materializer.build_materialization(repo, baseline_commit=baseline)
    output_parent = (repo / materializer.TEST_PATH).parent
    external_parent = tmp_path / "external-test-parent"
    output_parent.rename(external_parent)
    output_parent.symlink_to(external_parent, target_is_directory=True)
    external_test = external_parent / materializer.TEST_PATH.name
    external_before = external_test.read_bytes()

    with pytest.raises(materializer.MaterializationError):
        materializer.write_materialization(result)

    assert external_test.read_bytes() == external_before


def test_materializer_disables_repository_local_fsmonitor_hook(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    sentinel = tmp_path / "fsmonitor-invoked"
    probe = _write_git_probe(
        tmp_path,
        name="hostile-fsmonitor-probe",
        sentinel=sentinel,
    )
    _git(repo, "config", "core.fsmonitor", str(probe))

    _git(repo, "status", "--short")
    assert sentinel.is_file(), "fixture must prove the hostile fsmonitor is armed"
    sentinel.unlink()

    try:
        materializer.build_materialization(repo, baseline_commit=baseline)
    except materializer.MaterializationError:
        pass

    assert not sentinel.exists()


def test_materializer_disables_repository_local_external_diff_command(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    sentinel = tmp_path / "external-diff-invoked"
    probe = _write_git_probe(
        tmp_path,
        name="hostile-external-diff",
        sentinel=sentinel,
        exit_code=0,
    )
    manifest_relative = materializer.MANIFEST_PATH.as_posix()
    (repo / materializer.MANIFEST_PATH).write_bytes(b'{"changed":true}\n')
    _git(repo, "config", "diff.external", str(probe))

    _git(repo, "diff", baseline, "--", manifest_relative)
    assert sentinel.is_file(), "fixture must prove the hostile diff command is armed"
    sentinel.unlink()

    try:
        materializer.build_materialization(repo, baseline_commit=baseline)
    except materializer.MaterializationError:
        pass

    assert not sentinel.exists()


def test_materializer_disables_repository_local_post_index_hook(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    sentinel = tmp_path / "post-index-change-invoked"
    hooks = tmp_path / "hostile-hooks"
    hooks.mkdir()
    _write_git_probe(
        hooks,
        name="post-index-change",
        sentinel=sentinel,
        exit_code=0,
    )
    _git(repo, "config", "core.hooksPath", str(hooks))
    provider = repo / materializer.PROVIDER_PATH
    provider.write_bytes(provider.read_bytes())

    _git(repo, "status", "--short")
    assert sentinel.is_file(), "fixture must prove the hostile Git hook is armed"
    sentinel.unlink()
    provider.write_bytes(provider.read_bytes())

    try:
        materializer.build_materialization(repo, baseline_commit=baseline)
    except materializer.MaterializationError:
        pass

    assert not sentinel.exists()


def test_materializer_never_executes_repository_local_clean_filter(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, baseline, _ = _fixture_repository(tmp_path)
    sentinel = tmp_path / "clean-filter-invoked"
    probe = _write_git_probe(
        tmp_path,
        name="hostile-clean-filter",
        sentinel=sentinel,
        passthrough=True,
        exit_code=0,
    )
    manifest_relative = materializer.MANIFEST_PATH.as_posix()
    info_attributes = repo / ".git/info/attributes"
    info_attributes.write_text(
        f"{manifest_relative} filter=hostile\n",
        encoding="utf-8",
    )
    _git(repo, "config", "filter.hostile.clean", str(probe))
    (repo / materializer.MANIFEST_PATH).write_bytes(b'{"changed":true}\n')

    _git(repo, "diff", "--name-only", baseline, "--", manifest_relative)
    assert sentinel.is_file(), "fixture must prove the hostile clean filter is armed"
    sentinel.unlink()

    try:
        materializer.build_materialization(repo, baseline_commit=baseline)
    except materializer.MaterializationError:
        pass

    assert not sentinel.exists()


def test_materializer_rejects_duplicate_json_keys_before_construction(
    tmp_path: Path,
) -> None:
    materializer = _load_materializer()
    repo, _, _ = _fixture_repository(tmp_path)
    admission = repo / materializer.ADMISSION_PATH
    admission.write_bytes(b'{"admission_id":"first","admission_id":"second"}\n')
    _git(repo, "add", admission.relative_to(repo).as_posix())
    _git(repo, "commit", "-qm", "malformed admission input")
    malformed_baseline = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(materializer.MaterializationError, match="duplicate key"):
        materializer.build_materialization(
            repo,
            baseline_commit=malformed_baseline,
        )

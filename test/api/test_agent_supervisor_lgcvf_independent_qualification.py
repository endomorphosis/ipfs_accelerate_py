"""Independent-judge tests for LGCVF revision-2 qualification."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualify_logic_governed_compositional_verification_fabric.py"


def _load() -> ModuleType:
    name = "lgcvf_independent_qualification_tested"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _result(module: ModuleType) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": module.SCHEMA,
        "plan_cid": module.PLAN_CID,
        "predecessor_plan_cid": module.PREDECESSOR_PLAN_CID,
        "cohort": "hermetic_local_execution",
        "candidate_suites_are_self_authority": False,
        "independent_fixed_manifest_executed": True,
        "checkout_fingerprint_cid": module.content_identity({"checkout": "unchanged"}),
        "checkout_unchanged": True,
        "passed": True,
        "totals": {
            "collected": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
            "error_count": 0,
        },
        "suites": [],
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
        "limitations": ["hermetic only"],
    }
    value["result_cid"] = module.content_identity(value)
    return value


def test_missing_candidate_suite_fails_closed() -> None:
    module = _load()
    candidate = module.Suite(
        "missing_candidate",
        ".",
        ("test/api/definitely_missing_lgcvf_candidate.py",),
        True,
    )
    assert candidate.candidate_authored is True
    with pytest.raises(module.QualificationError, match="not a file"):
        module._suite_manifest(candidate)


def test_self_consistent_result_cannot_omit_required_suites() -> None:
    module = _load()
    value = _result(module)
    with pytest.raises(module.QualificationError, match="suite population differs"):
        module.validate_result(value)


def test_self_consistent_result_cannot_raise_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    monkeypatch.setattr(module, "SUITES", ())
    value = _result(module)
    module.validate_result(value)

    value["production_authorized"] = True
    value["result_cid"] = module.content_identity(
        {key: item for key, item in value.items() if key != "result_cid"}
    )
    with pytest.raises(module.QualificationError, match="raises production_authorized"):
        module.validate_result(value)


def test_check_reconstructs_instead_of_trusting_stored_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    monkeypatch.setattr(module, "SUITES", ())
    stored = _result(module)
    output = tmp_path / "qualification.json"
    output.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(module, "OUTPUT", output)

    reconstructed = _result(module)
    reconstructed["limitations"] = ["different hermetic evidence"]
    reconstructed["result_cid"] = module.content_identity(
        {key: item for key, item in reconstructed.items() if key != "result_cid"}
    )
    monkeypatch.setattr(module, "build_result", lambda: reconstructed)

    assert module.main(("--check",)) == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_candidate_sandbox_denies_checkout_mutation_and_network(tmp_path: Path) -> None:
    protected = tmp_path / "protected.txt"
    protected.write_text("original", encoding="utf-8")
    writable = tmp_path / "writable"
    writable.mkdir()
    probe = """
import importlib.util
import json
import socket
import sys
from pathlib import Path

script, protected, writable = map(Path, sys.argv[1:])
specification = importlib.util.spec_from_file_location("lgcvf_sandbox_probe", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
result = {"sandbox": module._install_candidate_sandbox(writable)}
try:
    protected.write_text("forged", encoding="utf-8")
except OSError as exc:
    result["write_errno"] = exc.errno
try:
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except OSError as exc:
    result["socket_errno"] = exc.errno
allowed = writable / "allowed.txt"
allowed.write_text("ok", encoding="utf-8")
result["allowed_write"] = allowed.read_text(encoding="utf-8")
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        (sys.executable, "-c", probe, str(SCRIPT), str(protected), str(writable)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "error" not in result
    assert result["write_errno"] in {1, 13}
    assert result["socket_errno"] == 1
    assert result["allowed_write"] == "ok"
    assert protected.read_text(encoding="utf-8") == "original"
    assert result["sandbox"]["checkout_write_permitted"] is False
    assert result["sandbox"]["network_permitted"] is False


def test_execution_projection_copies_judged_input_without_copying_authority(
    tmp_path: Path,
) -> None:
    module = _load()
    source = tmp_path / "source"
    destination = tmp_path / "projection"
    candidate = source / "test/api/candidate.py"
    authority = source / "scripts/protected_judge.py"
    candidate.parent.mkdir(parents=True)
    authority.parent.mkdir(parents=True)
    candidate.write_text("def test_candidate():\n    assert True\n", encoding="utf-8")
    authority.write_text("AUTHORITY = True\n", encoding="utf-8")
    suite = module.Suite("candidate", ".", ("test/api/candidate.py",), True)

    projection = module._prepare_execution_checkout(source, destination, (suite,))

    projected_candidate = destination / "test/api/candidate.py"
    projected_authority = destination / "scripts/protected_judge.py"
    assert projected_candidate.is_file() and not projected_candidate.is_symlink()
    assert projected_candidate.read_bytes() == candidate.read_bytes()
    assert projected_authority.parent.is_symlink()
    assert projection["original_checkout_writable"] is False


def test_protected_input_projection_survives_only_declared_evidence_commits(
    tmp_path: Path,
) -> None:
    module = _load()
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    (repository / "judge.py").write_text("JUDGE = 1\n", encoding="utf-8")
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "control.json").write_text('{"revision":1}\n', encoding="utf-8")
    _git(repository, "add", "judge.py", "source.py", "control.json")
    _git(repository, "commit", "-qm", "protected baseline")

    exclusions = (
        "qualification.json",
        "benchmark.json",
        "release-report.md",
    )
    projection_kwargs = {
        "root": repository,
        "excluded_paths": exclusions,
        "authority_paths": ("judge.py", "control.json"),
    }
    baseline = module._protected_input_projection([], **projection_kwargs)

    # The producer's own result is inert while untracked, staged, and committed.
    (repository / "qualification.json").write_text("{}\n", encoding="utf-8")
    assert module._protected_input_projection([], **projection_kwargs) == baseline
    _git(repository, "add", "qualification.json")
    assert module._protected_input_projection([], **projection_kwargs) == baseline
    _git(repository, "commit", "-qm", "qualification evidence")
    assert module._protected_input_projection([], **projection_kwargs) == baseline

    # Authorized downstream evidence/report commits are equally inert.
    (repository / "benchmark.json").write_text("{}\n", encoding="utf-8")
    (repository / "release-report.md").write_text("# No-go\n", encoding="utf-8")
    _git(repository, "add", "benchmark.json", "release-report.md")
    _git(repository, "commit", "-qm", "downstream evidence")
    assert module._protected_input_projection([], **projection_kwargs) == baseline

    # A protected judge remains semantic input whether dirty or committed.
    (repository / "judge.py").write_text("JUDGE = 2\n", encoding="utf-8")
    drifted = module._protected_input_projection([], **projection_kwargs)
    assert drifted["fingerprint_cid"] != baseline["fingerprint_cid"]
    _git(repository, "add", "judge.py")
    _git(repository, "commit", "-qm", "forged judge")
    assert module._protected_input_projection([], **projection_kwargs) != baseline


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_real_protected_suite_runs_with_writable_log_sink_in_sandbox() -> None:
    module = _load()
    suite = module.SANDBOX_SMOKE_SUITE
    manifest = module._suite_manifest(suite)

    observation = module._run_suite(suite, expected_manifest=manifest)

    assert observation["passed"] is True
    assert observation["collected"] == 1
    assert observation["passed_count"] == 1
    assert module._sandbox_evidence_is_valid(observation["isolation"])

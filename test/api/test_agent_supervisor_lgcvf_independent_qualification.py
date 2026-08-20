"""Independent-judge tests for LGCVF revision-2 qualification."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualify_logic_governed_compositional_verification_fabric.py"
_GITLINK_IMPORT_CANDIDATES = (
    "implementation.py",
    "deep/implementation.py",
    "deep/__pycache__/implementation.py",
    "deep/__pycache__/implementation.pyc",
    "deep/implementation.pyo",
    "deep/native.so",
    "deep/native.pyd",
    "deep/native.dylib",
    "deep/native" + importlib.machinery.EXTENSION_SUFFIXES[0],
)


def _load() -> ModuleType:
    name = "lgcvf_independent_qualification_tested"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module._require_isolated_recovery_runtime = lambda: None
    return module


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _create_sealed_datasets_repository(repository: Path) -> Path:
    """Create the minimal clean nested Git repository required by recovery."""

    nested = repository / "ipfs_datasets_py"
    package = nested / "ipfs_datasets_py"
    tests = nested / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tests / ".gitkeep").write_text("", encoding="utf-8")
    (nested / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\n", encoding="utf-8"
    )
    _git(nested, "init", "-q")
    _git(nested, "config", "user.email", "fixture@example.invalid")
    _git(nested, "config", "user.name", "LGCVF Fixture")
    _git(nested, "add", ".")
    _git(nested, "commit", "-qm", "sealed datasets source")
    return nested


def _add_sealed_stage_zero_gitlinks(
    repository: Path,
    relatives: tuple[str, ...],
) -> str:
    """Commit exact Gitlinks and leave only empty uninitialized placeholders."""

    source = repository.parent / f"{repository.name}-gitlink-source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "fixture@example.invalid")
    _git(source, "config", "user.name", "LGCVF Fixture")
    _git(source, "commit", "--allow-empty", "-qm", "sealed Gitlink target")
    object_id = _git_output(source, "rev-parse", "HEAD")
    for relative in relatives:
        _git(
            repository,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{object_id},{relative}",
        )
    _git(repository, "commit", "-qm", "bind stage-zero Gitlinks")
    for relative in relatives:
        (repository / relative).mkdir(parents=True)
    return object_id


def _sealed_qualifier_script(tmp_path: Path) -> Path:
    """Build a minimal clean tracked checkout for exact isolated subprocesses."""

    repository = tmp_path / "sealed-qualifier-source"
    script = repository / "scripts" / SCRIPT.name
    contract = (
        repository
        / "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py"
    )
    script.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    shutil.copy2(
        ROOT
        / "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py",
        contract,
    )
    for relative in (
        "ipfs_accelerate_py/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/proof/__init__.py",
    ):
        path = repository / relative
        path.write_text("", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\n", encoding="utf-8"
    )
    (repository / "test").mkdir()
    (repository / "test/.gitkeep").write_text("", encoding="utf-8")
    _create_sealed_datasets_repository(repository)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "sealed qualifier source")
    return script


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


def _record_identity(payload: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode(
        "ascii"
    ).rstrip("=")
    return f"sha256={digest}", str(len(payload))


def _development_blob_entries(
    root: Path,
    paths: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    """Supply current-worktree blobs only to source-dirty execution tests."""

    result: dict[str, tuple[str, str]] = {}
    for relative in paths:
        path = root.joinpath(*Path(relative).parts)
        payload = path.read_bytes()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        digest = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii"))
        digest.update(payload)
        result[relative] = (mode, digest.hexdigest())
    return result


def _fake_duckdb_site(
    module: ModuleType,
    root: Path,
    *,
    versions: tuple[str, ...] = ("1.5.5",),
    extra_native: bool = False,
) -> None:
    extension = str(module.sysconfig.get_config_var("EXT_SUFFIX"))
    runtime = {
        relative: f"# fixture {relative}\n".encode()
        for relative in module._DUCKDB_RUNTIME_SOURCE_PATHS
    }
    runtime[f"_duckdb{extension}"] = b"fixture-native-extension"
    if extra_native:
        runtime[f"_duckdb_extra{extension}"] = b"second-native-extension"
    for relative, payload in runtime.items():
        path = root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    pycache = root / "duckdb/__pycache__/__init__.cpython-312.pyc"
    pycache.parent.mkdir(parents=True, exist_ok=True)
    pycache.write_bytes(b"ambient-bytecode-is-not-projected")
    for version in versions:
        distribution_name = f"duckdb-{version}.dist-info"
        distribution = root / distribution_name
        distribution.mkdir()
        metadata = f"Metadata-Version: 2.4\nName: DuckDB\nVersion: {version}\n".encode()
        (distribution / "METADATA").write_bytes(metadata)
        rows = [
            (relative, *_record_identity(payload))
            for relative, payload in sorted(runtime.items())
        ]
        rows.extend(
            [
                (
                    f"{distribution_name}/METADATA",
                    *_record_identity(metadata),
                ),
                (
                    "duckdb/__pycache__/__init__.cpython-312.pyc",
                    *_record_identity(pycache.read_bytes()),
                ),
            ]
        )
        (distribution / "RECORD").write_text(
            "".join(f"{path},{digest},{size}\n" for path, digest, size in rows),
            encoding="utf-8",
        )


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


@pytest.mark.parametrize(
    "encoded",
    (
        '{"schema":"first","schema":"second"}',
        '{"schema":"receipt","suites":[{"task_id":"first","task_id":"second"}]}',
    ),
)
def test_authority_json_loader_rejects_duplicate_keys_at_every_depth(
    encoded: str,
) -> None:
    module = _load()

    with pytest.raises(
        module.QualificationError,
        match="contains duplicate JSON object key",
    ):
        module._strict_json_loads(encoded, noun="authority receipt")


def test_authority_json_loader_accepts_canonical_recovery_receipt() -> None:
    module = _load()
    receipt = module._recovery_unavailable(
        module.QualificationError("sandbox unavailable")
    )

    decoded = module._strict_json_loads(
        json.dumps(receipt, sort_keys=True),
        noun="recovery receipt",
    )

    assert decoded == receipt
    assert (
        module.verify_preregistered_recovery_qualification(
            decoded,
            require_passed=False,
        )
        == receipt
    )


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


def test_seccomp_resolution_fails_closed_when_network_rules_are_unavailable() -> None:
    module = _load()

    def resolver(name: bytes) -> int:
        return -1 if name.decode("ascii") in {"socket", "connect"} else 1

    with pytest.raises(
        module.QualificationError,
        match="cannot resolve required network syscalls: socket, connect",
    ):
        module._resolve_seccomp_rules(resolver)


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
    external = tmp_path / "external-marker.py"
    external.write_text("EXTERNAL_MARKER = True\n", encoding="utf-8")
    omitted = source / "test/run_dashboard_tests.py"
    omitted.symlink_to(external)
    suite = module.Suite("candidate", ".", ("test/api/candidate.py",), True)

    projection = module._prepare_execution_checkout(source, destination, (suite,))

    projected_candidate = destination / "test/api/candidate.py"
    projected_authority = destination / "scripts/protected_judge.py"
    assert projected_candidate.is_file() and not projected_candidate.is_symlink()
    assert projected_candidate.read_bytes() == candidate.read_bytes()
    assert projected_authority.parent.is_symlink()
    assert projection["original_checkout_writable"] is False
    assert projection["schema"] == "lgcvf-readonly-test-projection@2"
    assert projection["omitted_source_symlinks"] == [
        {
            "path": "test/run_dashboard_tests.py",
            "git_target": str(external),
            "disposition": "omitted_source_symlink",
        }
    ]
    assert os.path.lexists(destination / "test/run_dashboard_tests.py") is False
    assert projection["projection_cid"] == module.content_identity(
        {key: item for key, item in projection.items() if key != "projection_cid"}
    )


def test_recovery_execution_projection_is_a_closed_copy_without_source_links(
    tmp_path: Path,
) -> None:
    module = _load()
    source = tmp_path / "source"
    destination = tmp_path / "projection"
    for relative in (
        "scripts",
        "ipfs_accelerate_py",
        "ipfs_datasets_py/ipfs_datasets_py",
        "test/api",
        "test/common",
    ):
        (source / relative).mkdir(parents=True)
    files = {
        "scripts/qualify_logic_governed_compositional_verification_fabric.py": (
            "VALUE = 'copied judge'\n"
        ),
        "ipfs_accelerate_py/__init__.py": "",
        "ipfs_datasets_py/ipfs_datasets_py/__init__.py": "",
        "test/__init__.py": "",
        "test/conftest.py": "from test.common.fixtures import *\n",
        "test/api/__init__.py": "",
        "test/api/conftest.py": "",
        "test/api/candidate.py": "def test_candidate():\n    assert True\n",
        "test/common/__init__.py": "",
        "test/common/fixtures.py": "FIXTURE = True\n",
    }
    for relative, contents in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    external = tmp_path / "external.py"
    external.write_text("RAISED = True\n", encoding="utf-8")
    (source / "test/run_dashboard_tests.py").symlink_to(external)
    nested = source / "ipfs_datasets_py"
    for repository in (nested, source):
        _git(repository, "init", "-q")
        _git(repository, "config", "user.email", "fixture@example.invalid")
        _git(repository, "config", "user.name", "LGCVF Fixture")
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "sealed projection source")
    suite = module.Suite("recovery_fixture", ".", ("test/api/candidate.py",), True)

    receipt = module._prepare_recovery_execution_checkout(
        source, destination, (suite,)
    )

    assert receipt["schema"] == "lgcvf-closed-recovery-test-projection@1"
    assert receipt["contains_live_source_links"] is False
    assert receipt["original_checkout_writable"] is False
    assert all(not path.is_symlink() for path in destination.rglob("*"))
    assert (destination / "scripts" / Path(module.__file__).name).is_file()
    assert (destination / "test/common/fixtures.py").read_text(encoding="utf-8") == (
        "FIXTURE = True\n"
    )
    assert os.path.lexists(destination / "test/run_dashboard_tests.py") is False
    assert receipt["omitted_source_symlinks"] == [
        {
            "path": "test/run_dashboard_tests.py",
            "git_target": str(external),
            "disposition": "omitted_source_symlink",
        }
    ]
    assert receipt["projection_cid"] == module.content_identity(
        {key: item for key, item in receipt.items() if key != "projection_cid"}
    )

    candidate = source / "test/api/candidate.py"
    candidate.unlink()
    candidate.symlink_to(external)
    with pytest.raises(module.QualificationError, match="source is unavailable"):
        module._prepare_recovery_execution_checkout(
            source, tmp_path / "rejected-projection", (suite,)
        )


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


def test_recovery_population_is_closed_and_task_spec_bound() -> None:
    module = _load()
    expected = (
        (
            "LGCVF-051",
            "baguqeera5akr2w56xcy6mus4td2ghwaita5k52lnqkh54tgooqq35fai72nq",
            "ipfs_datasets_py",
            "tests/unit/logic/test_compositional_verification_public_api.py",
            "baguqeeracrex4xlp2f2mtqtc3drgce5kjlqpdqlrxw6wvf22uoe4fm3b6paa",
        ),
        (
            "LGCVF-060",
            "baguqeeratlqqozbenktvzhzzk36ewsti3mzegge2ynft24s3mznewzgswfoq",
            "ipfs_datasets_py",
            "tests/unit/logic/backends/test_interpolation.py",
            "baguqeerah6qhkb44do4u4k6x6gtsvlw3cppoprwdn42xqeufa6pnrjcxsqpa",
        ),
        (
            "LGCVF-061",
            "baguqeeraqopotj43fgxcfptvziv3g3kna4wcwhtzarcobv2obvs32njoggjq",
            "ipfs_datasets_py",
            "tests/unit/logic/software_verification/test_cegar.py",
            "baguqeerafssejhozbsz47tp73e24wjymj4fvsp5zoly6pmpjiyzghgxjbmaa",
        ),
        (
            "LGCVF-070",
            "baguqeeraej2zz7zlrd2l5p6adjnzinnitzuqmxx4agfhfzekixdqm2mnqyda",
            "ipfs_datasets_py",
            "tests/unit/logic/formalization/test_translation_receipts.py",
            "baguqeeranxa5irflxuqxbuh4ojreywhgd33xobmv7yfulnmiaszi7rd7yzzq",
        ),
        (
            "LGCVF-071",
            "baguqeerar3vmbqw7f2qk6mjyhsx3hq7gpbnqcydt7cecm6og5xejbd2vz6cq",
            "ipfs_datasets_py",
            "tests/unit/logic/software_verification/test_obligation_slicing.py",
            "baguqeeraymjjn45qmczidqx4yhgaheb6izz7ts5y3kxbmpnay4k2xpvfg7ka",
        ),
        (
            "LGCVF-080",
            "baguqeera22uu4o4ux6kzp4fgv5gxqupas3nhdjtbrtt73x2kc6mhtxkdbtwq",
            ".",
            "test/api/test_agent_supervisor_program_repair_egraph.py",
            "baguqeeradva3nkqggtxukgy7nizn7nol7gqaeszvxahcyofllmc4lrok4seq",
        ),
    )
    actual = tuple(
        (
            item.task_id,
            item.task_cid,
            item.owner_root,
            item.path,
            item.validation_spec()["validation_spec_cid"],
        )
        for item in module.RECOVERY_VALIDATIONS
    )
    assert actual == expected
    assert all(item.suite.candidate_authored for item in module.RECOVERY_VALIDATIONS)
    assert list(inspect.signature(module.run_preregistered_recovery_qualification).parameters) == [
        "root"
    ]
    assert "runner" not in inspect.signature(
        module.verify_preregistered_recovery_qualification
    ).parameters


def test_recovery_provider_import_and_provider_module_process_are_denied() -> None:
    probe = r"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_provider_guard_probe", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
result = {}
import_guard = module._RecoveryProviderGuard()
try:
    with import_guard:
        __import__("ipfs_accelerate_py.agent_supervisor.provider_fallback_runner")
except module.QualificationError as exc:
    result["import_denied"] = type(exc).__name__
result["import_attempts"] = import_guard.import_attempts
result["module_loaded"] = (
    "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner" in sys.modules
)
process_guard = module._RecoveryProviderGuard()
try:
    with process_guard:
        subprocess.run(
            [sys.executable, "-m", "openai"],
            check=False,
            capture_output=True,
        )
except module.QualificationError as exc:
    result["process_denied"] = type(exc).__name__
result["process_attempts"] = process_guard.process_attempts
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        (sys.executable, "-c", probe, str(SCRIPT)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["import_denied"] == "QualificationError"
    assert result["module_loaded"] is False
    assert result["import_attempts"] == [
        "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner"
    ]
    assert result["process_denied"] == "QualificationError"
    assert result["process_attempts"] == ["openai"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_recovery_observation_is_sealed_bounded_and_rejects_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    monkeypatch.setattr(
        module,
        "_recovery_projection_git_entries",
        _development_blob_entries,
    )
    validation = module.RECOVERY_VALIDATIONS[-1]
    manifest_before = module._suite_manifest(validation.suite)
    qualifier_before = SCRIPT.read_bytes()

    observation = module._run_suite(
        validation.suite,
        expected_manifest=manifest_before,
    )

    module._validate_recovery_observation(observation, validation)
    assert module._suite_manifest(validation.suite) == manifest_before
    assert SCRIPT.read_bytes() == qualifier_before
    assert observation["provider_imports_observed"] == []
    assert observation["provider_import_attempts"] == []
    assert observation["provider_process_attempts"] == []
    assert observation["cache_reused"] is False
    assert observation["candidate_authored"] is True
    assert observation["self_authority"] is False
    assert observation["completion_authoritative"] is False
    assert observation["isolation"]["network_permitted"] is False
    assert observation["isolation"]["resource_limits"]["processes"] == 4_096
    projection = observation["readonly_projection"]
    assert projection["schema"] == "lgcvf-closed-recovery-test-projection@1"
    assert projection["contains_live_source_links"] is False
    assert projection["original_checkout_writable"] is False
    assert projection["copied_source_count"] > 0
    assert projection["copied_source_bytes"] > 0
    assert projection["projection_cid"] == module.content_identity(
        {key: item for key, item in projection.items() if key != "projection_cid"}
    )
    assert observation["isolation"]["worker_pycache"] == {
        "schema": "lgcvf-recovery-worker-pycache-isolation@1",
        "write_root_relative_path": "python-pycache",
        "environment_variable": "PYTHONPYCACHEPREFIX",
        "python_prefix_active": True,
        "dont_write_bytecode": True,
        "owner_matches_worker": True,
        "mode_octal": "0700",
        "empty_before": True,
        "empty_after": True,
    }
    policy = observation["isolation"]["seccomp_policy"]
    assert policy["required_network_syscalls"] == list(
        module._DENIED_NETWORK_SYSCALLS
    )
    assert set(module._DENIED_NETWORK_SYSCALLS).issubset(
        policy["installed_syscalls"]
    )
    assert policy["policy_cid"] == module.content_identity(
        {key: item for key, item in policy.items() if key != "policy_cid"}
    )
    assert module._sandbox_evidence_is_valid(
        observation["isolation"], require_recovery_policy=True
    )
    assert len(json.dumps(observation).encode("utf-8")) < module._MAX_WORKER_RECEIPT_BYTES

    forged = copy.deepcopy(observation)
    forged["task_cid"] = "baguqeeraforged"
    worker_body = {
        key: item
        for key, item in forged.items()
        if key
        not in {
            "worker_observation_cid",
            "raw_stdout_size_bytes",
            "raw_stdout_sha256",
            "raw_stderr_size_bytes",
            "raw_stderr_sha256",
            "observation_cid",
        }
    }
    forged["worker_observation_cid"] = module.content_identity(worker_body)
    forged["observation_cid"] = module.content_identity(
        {key: item for key, item in forged.items() if key != "observation_cid"}
    )
    with pytest.raises(module.QualificationError, match="authority differs"):
        module._validate_recovery_observation(forged, validation)

    weakened = copy.deepcopy(observation)
    weakened_policy = weakened["isolation"]["seccomp_policy"]
    weakened_policy["installed_syscalls"].remove("socket")
    weakened_policy["unavailable_optional_syscalls"].append("socket")
    weakened_policy["policy_cid"] = module.content_identity(
        {
            key: item
            for key, item in weakened_policy.items()
            if key != "policy_cid"
        }
    )
    weakened["isolation"]["seccomp_denied_syscall_count"] -= 1
    worker_body = {
        key: item
        for key, item in weakened.items()
        if key
        not in {
            "worker_observation_cid",
            "raw_stdout_size_bytes",
            "raw_stdout_sha256",
            "raw_stderr_size_bytes",
            "raw_stderr_sha256",
            "observation_cid",
        }
    }
    weakened["worker_observation_cid"] = module.content_identity(worker_body)
    weakened["observation_cid"] = module.content_identity(
        {key: item for key, item in weakened.items() if key != "observation_cid"}
    )
    with pytest.raises(module.QualificationError, match="did not pass exactly"):
        module._validate_recovery_observation(weakened, validation)

    forged_pycache = copy.deepcopy(observation)
    forged_pycache["isolation"]["worker_pycache"]["empty_after"] = False
    worker_body = {
        key: item
        for key, item in forged_pycache.items()
        if key
        not in {
            "worker_observation_cid",
            "raw_stdout_size_bytes",
            "raw_stdout_sha256",
            "raw_stderr_size_bytes",
            "raw_stderr_sha256",
            "observation_cid",
        }
    }
    forged_pycache["worker_observation_cid"] = module.content_identity(worker_body)
    forged_pycache["observation_cid"] = module.content_identity(
        {
            key: item
            for key, item in forged_pycache.items()
            if key != "observation_cid"
        }
    )
    with pytest.raises(module.QualificationError, match="did not pass exactly"):
        module._validate_recovery_observation(forged_pycache, validation)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT required")
def test_recovery_output_flood_is_bounded_while_worker_runs(tmp_path: Path) -> None:
    module = _load()
    output = tmp_path / "worker.stdout"
    writable = tmp_path / "writable"
    writable.mkdir()
    probe = r"""
import importlib.util
import os
import resource
import sys
from pathlib import Path

script, writable = map(Path, sys.argv[1:])
specification = importlib.util.spec_from_file_location("lgcvf_output_probe", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
module._install_candidate_sandbox(writable)
module._lower_resource_limit(resource.RLIMIT_FSIZE, module._MAX_WORKER_TRANSCRIPT_BYTES)
os.write(1, b"x" * (module._MAX_WORKER_TRANSCRIPT_BYTES + 1))
"""
    with output.open("wb") as stream:
        completed = subprocess.run(
            (sys.executable, "-c", probe, str(SCRIPT), str(writable)),
            cwd=ROOT,
            check=False,
            stdout=stream,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    assert output.stat().st_size <= 4 * 1024 * 1024
    assert completed.returncode != 0 or output.stat().st_size == 4 * 1024 * 1024
    with pytest.raises(module.QualificationError, match="reached its hard bound"):
        module._bounded_worker_stream(output, suite_id="flood", label="stdout")
    budget = [8]
    capture = module._BoundedTextCapture(budget)
    capture.write("12345678")
    with pytest.raises(module.QualificationError, match="exceeded"):
        capture.write("x")


def test_recovery_receipt_pipe_is_concurrently_drained_and_bounded() -> None:
    module = _load()
    receipt_read, receipt_write = os.pipe()
    drain = module._BoundedReceiptPipeDrain(receipt_read, maximum_bytes=1024)
    drain.start()
    writer_errors: list[BaseException] = []

    def flood_pipe() -> None:
        view = memoryview(b"x" * (256 * 1024))
        try:
            while view:
                written = os.write(receipt_write, view)
                view = view[written:]
        except BaseException as exc:  # asserted in the parent test thread
            writer_errors.append(exc)
        finally:
            os.close(receipt_write)

    writer = Thread(target=flood_pipe, daemon=True)
    writer.start()
    writer.join(5)
    assert writer.is_alive() is False
    assert writer_errors == []
    with pytest.raises(module.QualificationError, match="exceeds its pipe bound"):
        drain.finish(timeout_seconds=5)


def test_recovery_source_binding_requires_clean_exact_gitlink(tmp_path: Path) -> None:
    module = _load()
    isolated_script = _sealed_qualifier_script(tmp_path)
    repository = tmp_path / "accelerator"
    datasets = repository / "ipfs_datasets_py"
    scripts = repository / "scripts"
    datasets.mkdir(parents=True)
    scripts.mkdir()
    _git(datasets, "init", "-q")
    _git(datasets, "config", "user.email", "fixture@example.invalid")
    _git(datasets, "config", "user.name", "LGCVF Fixture")
    (datasets / "semantic.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(datasets, "add", "semantic.py")
    _git(datasets, "commit", "-qm", "datasets")
    datasets_head = _git_output(datasets, "rev-parse", "HEAD")

    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    (repository / ".gitmodules").write_text(
        '[submodule "ipfs_datasets_py"]\n'
        "\tpath = ipfs_datasets_py\n"
        "\turl = ./ipfs_datasets_py\n",
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    _git(repository, "add", ".gitmodules", "scripts")
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{datasets_head},ipfs_datasets_py",
    )
    _git(repository, "commit", "-qm", "accelerator")

    binding = module._recovery_source_binding(root=repository)
    assert binding["schema"] == "lgcvf-recovery-source-binding@2"
    assert binding["repository_topology"] == "accelerator_with_datasets_gitlink"
    assert binding["datasets_gitlink"] == datasets_head
    assert binding["datasets_head"] == datasets_head
    assert binding["accelerator_clean"] is True
    assert binding["datasets_clean"] is True
    assert binding["toolchain"]["duckdb_runtime"] == (
        module.bound_duckdb_runtime_evidence()
    )

    probe = """
import importlib.util
import json
import sys
from pathlib import Path

script, repository = map(Path, sys.argv[1:])
specification = importlib.util.spec_from_file_location("lgcvf_isolated_binding", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
before = "pytest" in sys.modules
binding = module._recovery_source_binding(root=repository)
print(json.dumps({
    "before": before,
    "after": "pytest" in sys.modules,
    "binding": binding,
}, sort_keys=True))
"""
    isolated = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            probe,
            str(isolated_script),
            str(repository),
        ),
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert isolated.returncode == 0, isolated.stderr
    isolated_result = json.loads(isolated.stdout)
    assert isolated_result["before"] is False
    assert isolated_result["after"] is False
    assert isolated_result["binding"] == binding

    (datasets / "semantic.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(module.QualificationError, match="clean .* overlays"):
        module._recovery_source_binding(root=repository)


def test_bound_duckdb_runtime_isolated_sequential_and_fail_closed(
    tmp_path: Path,
) -> None:
    isolated_script = _sealed_qualifier_script(tmp_path)
    marker = tmp_path / "startup-marker"
    hostile = tmp_path / "hostile-site"
    hostile.mkdir()
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('sitecustomize')\n",
        encoding="utf-8",
    )
    (hostile / "hostile.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('pth')\n",
        encoding="utf-8",
    )
    probe = r'''
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_duckdb_capsule", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
evidence = module.bound_duckdb_runtime_evidence()
cid = evidence["runtime_cid"]
original_path = list(sys.path)
original_meta_path = list(sys.meta_path)
original_dont_write = sys.dont_write_bytecode
wrong_rejected = False
try:
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid="baguqeera-wrong"):
        raise AssertionError("wrong runtime admitted")
except module.QualificationError:
    wrong_rejected = True

with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
    import duckdb
    import _duckdb
    first = duckdb.connect(":memory:").execute("select 40 + 2").fetchone()[0]
    package_origin = duckdb.__file__
    native_origin = _duckdb.__file__
    projection = str(module._ACTIVE_DUCKDB_RUNTIME_PROJECTION)
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
        nested = duckdb.connect(":memory:").execute("select 6 * 7").fetchone()[0]
restored_after_first = (
    sys.path == original_path
    and sys.meta_path == original_meta_path
    and sys.dont_write_bytecode == original_dont_write
)

body_failure = False
try:
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
        raise RuntimeError("candidate operation failed")
except RuntimeError:
    body_failure = True
injected_rejected = False
sys.modules["duckdb.injected"] = ModuleType("duckdb.injected")
try:
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
        raise AssertionError("originless injected module was admitted")
except module.QualificationError:
    injected_rejected = True
finally:
    sys.modules.pop("duckdb.injected", None)
identity_rejected = False
original_version_module = sys.modules["duckdb._version"]
replacement_version_module = ModuleType("duckdb._version")
replacement_version_module.__file__ = original_version_module.__file__
sys.modules["duckdb._version"] = replacement_version_module
try:
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
        raise AssertionError("replaced DuckDB module was admitted")
except module.QualificationError:
    identity_rejected = True
finally:
    sys.modules["duckdb._version"] = original_version_module
with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
    second = duckdb.connect(":memory:").execute("select 84 / 2").fetchone()[0]
    same_origins = duckdb.__file__ == package_origin and _duckdb.__file__ == native_origin
restored_after_second = (
    sys.path == original_path
    and sys.meta_path == original_meta_path
    and sys.dont_write_bytecode == original_dont_write
)
projection_path = Path(projection)
inventory = sorted(
    path.relative_to(projection_path).as_posix()
    for path in projection_path.rglob("*")
)
all_origins_bound = all(
    bool(getattr(value, "__file__", None))
    and Path(value.__file__).resolve().is_relative_to(projection_path.resolve())
    for name, value in sys.modules.items()
    if name == "duckdb"
    or name.startswith("duckdb.")
    or name == "_duckdb"
    or name.startswith("_duckdb.")
)
tamper_path = projection_path / "duckdb/__init__.py"
tamper_path.chmod(0o600)
tamper_rejected = False
try:
    with module.isolated_bound_duckdb_runtime(expected_runtime_cid=cid):
        raise AssertionError("tampered projection admitted")
except module.QualificationError:
    tamper_rejected = True
print(json.dumps({
    "body_failure": body_failure,
    "cid": cid,
    "first": first,
    "identity_rejected": identity_rejected,
    "injected_rejected": injected_rejected,
    "nested": nested,
    "second": second,
    "same_origins": same_origins,
    "all_origins_bound": all_origins_bound,
    "restored_after_first": restored_after_first,
    "restored_after_second": restored_after_second,
    "wrong_rejected": wrong_rejected,
    "tamper_rejected": tamper_rejected,
    "inventory": inventory,
    "schema": evidence["schema"],
    "soabi": evidence["python_soabi"],
    "extension_suffix": evidence["native_extension_suffix"],
}, sort_keys=True))
'''
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", probe, str(isolated_script)),
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(hostile),
            "PYTHONUSERBASE": str(hostile),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["schema"] == "lgcvf-bound-duckdb-runtime@1"
    assert result["cid"].startswith("baguqeera")
    assert result["soabi"]
    assert result["extension_suffix"].startswith(".")
    assert result["first"] == result["nested"] == result["second"] == 42
    assert result["identity_rejected"] is True
    assert result["injected_rejected"] is True
    assert result["same_origins"] is True
    assert result["all_origins_bound"] is True
    assert result["restored_after_first"] is True
    assert result["restored_after_second"] is True
    assert result["body_failure"] is True
    assert result["wrong_rejected"] is True
    assert result["tamper_rejected"] is True
    assert not any("__pycache__" in path or path.endswith(".pth") for path in result["inventory"])
    assert marker.exists() is False


def test_direct_isolated_recovery_ignores_dependency_bytecode_cache(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    qualifier_script = scripts / SCRIPT.name
    materializer_script = (
        scripts
        / "materialize_logic_governed_compositional_verification_fabric_control_plane.py"
    )
    shutil.copy2(SCRIPT, qualifier_script)
    shutil.copy2(
        ROOT
        / "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py",
        materializer_script,
    )
    config = (
        repository
        / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
    )
    config.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT
        / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json",
        config,
    )

    package = repository / "ipfs_accelerate_py/agent_supervisor"
    for relative in (
        "../__init__.py",
        "__init__.py",
        "merge/__init__.py",
        "planning/__init__.py",
        "proof/__init__.py",
        "task_sources/__init__.py",
    ):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (package / "merge/database_coordination.py").write_text(
        "def read_coordination_history_projection(*args, **kwargs): return {}\n"
        "def read_coordination_registry_projection(*args, **kwargs): return {}\n",
        encoding="utf-8",
    )
    (package / "planning/formal_planning_contracts.py").write_text(
        "FormalWorkPlan = object\n", encoding="utf-8"
    )
    revision_names = (
        "CompletionAuthority",
        "DeltaEffectClass",
        "LifecycleState",
        "MergeStrategyKind",
        "PlanAuthorityRoots",
        "PlanCompletionRule",
        "PlanConflictContract",
        "PlanDelta",
        "PlanDeltaItem",
        "PlanDeltaOperation",
        "PlanLeaseContract",
        "PlanMergeStrategy",
        "PlanOrigin",
        "PlanPopulationDigest",
        "PlanProviderContract",
        "PlanResourceContract",
        "PlanRetryContract",
        "PlanRevision",
        "PlanWorktreeContract",
        "PopulationKind",
    )
    (package / "planning/plan_revision_contracts.py").write_text(
        "\n".join(f"{name} = object" for name in revision_names) + "\n",
        encoding="utf-8",
    )
    (package / "task_sources/intent_repository.py").write_text(
        "def task_authority_spec_cid(*args, **kwargs): return 'fixture'\n"
        "def task_projection_spec_cid(*args, **kwargs): return 'fixture'\n",
        encoding="utf-8",
    )
    (package / "task_sources/todo_vector_index.py").write_text(
        "def parse_todo_blocks(*args, **kwargs): return []\n"
        "def split_csv(*args, **kwargs): return []\n",
        encoding="utf-8",
    )

    marker = tmp_path / "malicious-pyc-executed"
    dependency = package / "proof/formal_verification_contracts.py"
    malicious = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "def content_identity(value): return 'fixture'\n"
    ).encode()
    benign = b"def content_identity(value): return 'fixture'\n"
    source_size = max(len(malicious), len(benign)) + 64

    def padded(value: bytes) -> bytes:
        return value + b"#" + (b"x" * (source_size - len(value) - 2)) + b"\n"

    timestamp = 1_700_000_000
    dependency.write_bytes(padded(malicious))
    os.utime(dependency, (timestamp, timestamp))
    bytecode = Path(importlib.util.cache_from_source(str(dependency)))
    bytecode.parent.mkdir(parents=True)
    py_compile.compile(
        str(dependency),
        cfile=str(bytecode),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    dependency.write_bytes(padded(benign))
    os.utime(dependency, (timestamp, timestamp))
    (repository / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\nignored_shadow.py\n*.so\n",
        encoding="utf-8",
    )
    (repository / "test").mkdir()
    (repository / "test/.gitkeep").write_text("", encoding="utf-8")
    datasets_repository = _create_sealed_datasets_repository(repository)
    (datasets_repository / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\nignored_shadow.py\n*.so\n",
        encoding="utf-8",
    )
    _git(datasets_repository, "add", ".gitignore")
    _git(datasets_repository, "commit", "-qm", "bind ignored import shadows")
    (repository / "conftest.py").write_text("", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "sealed bytecode fixture")

    python = Path("/usr/bin/python3.12")
    environment = {"HOME": str(tmp_path), "LANG": "C", "PATH": "/usr/bin:/bin"}
    vulnerable_control = subprocess.run(
        (
            str(python),
            "-I",
            "-S",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts",
            str(repository),
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert vulnerable_control.returncode == 0, vulnerable_control.stderr
    assert marker.read_text(encoding="utf-8") == "executed"
    marker.unlink()

    qualifier = subprocess.run(
        (str(python), "-I", "-S", "-B", str(qualifier_script), "--recovery"),
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert qualifier.returncode == 1, qualifier.stderr
    assert isinstance(json.loads(qualifier.stdout), dict)
    assert qualifier.stderr == ""
    assert marker.exists() is False


    materializer = subprocess.run(
        (
            str(python),
            "-I",
            "-S",
            "-B",
            str(materializer_script),
            "recovery-preview",
        ),
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert materializer.returncode == 2, materializer.stderr
    assert json.loads(materializer.stdout)["valid"] is False
    assert marker.exists() is False

    def assert_import_boundary_rejects() -> None:
        rejected = subprocess.run(
            (
                str(python),
                "-I",
                "-S",
                "-B",
                str(qualifier_script),
                "--recovery",
            ),
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert rejected.returncode != 0
        assert "protected recovery" in rejected.stderr
        assert marker.exists() is False

    adjacent_bytecode = dependency.with_suffix(".pyc")
    native_shadow = dependency.parent / "untracked_shadow.so"
    python_shadow = dependency.parent / "untracked_shadow.py"
    for shadow, payload in (
        (adjacent_bytecode, bytecode.read_bytes()),
        (native_shadow, b"not-a-native-extension"),
        (python_shadow, b"raise RuntimeError('untracked source executed')\n"),
    ):
        shadow.write_bytes(payload)
        assert_import_boundary_rejects()
        shadow.unlink()

    clean_dependency = dependency.read_bytes()
    dependency.write_bytes(clean_dependency + b"# dirty tracked import\n")
    assert_import_boundary_rejects()
    dependency.write_bytes(clean_dependency)

    dependency_relative = dependency.relative_to(repository).as_posix()
    _git(repository, "update-index", "--assume-unchanged", dependency_relative)
    assert_import_boundary_rejects()
    _git(repository, "update-index", "--no-assume-unchanged", dependency_relative)
    _git(repository, "update-index", "--skip-worktree", dependency_relative)
    assert_import_boundary_rejects()
    _git(repository, "update-index", "--no-skip-worktree", dependency_relative)

    dependency.chmod(0o666)
    assert_import_boundary_rejects()
    dependency.chmod(0o644)
    proof_directory = dependency.parent
    proof_directory.chmod(0o777)
    assert_import_boundary_rejects()
    proof_directory.chmod(0o755)

    hardlink_peer = tmp_path / "dependency-hardlink-peer"
    hardlink_peer.write_bytes(clean_dependency)
    dependency.unlink()
    os.link(hardlink_peer, dependency)
    assert_import_boundary_rejects()
    dependency.unlink()
    dependency.write_bytes(clean_dependency)
    dependency.chmod(0o644)

    for ignored_shadow in (
        repository / "test/ignored_shadow.py",
        repository / "ignored_shadow.py",
        datasets_repository / "tests/ignored_shadow.py",
        datasets_repository / "tests/ignored_shadow.so",
    ):
        ignored_shadow.write_bytes(b"raise RuntimeError('ignored shadow executed')\n")
        assert_import_boundary_rejects()
        ignored_shadow.unlink()

    root_conftest = repository / "conftest.py"
    root_conftest.write_text("raise RuntimeError('dirty conftest')\n", encoding="utf-8")
    assert_import_boundary_rejects()
    root_conftest.write_text("", encoding="utf-8")

    nested_source = datasets_repository / "ipfs_datasets_py/__init__.py"
    nested_source.write_text("raise RuntimeError('dirty nested source')\n", encoding="utf-8")
    assert_import_boundary_rejects()
    nested_source.write_text("", encoding="utf-8")
    nested_relative = nested_source.relative_to(datasets_repository).as_posix()
    _git(datasets_repository, "update-index", "--assume-unchanged", nested_relative)
    assert_import_boundary_rejects()
    _git(
        datasets_repository,
        "update-index",
        "--no-assume-unchanged",
        nested_relative,
    )


@pytest.mark.parametrize("repository_scope", ("accelerator", "datasets"))
@pytest.mark.parametrize("substitution", ("grafts", "replace"))
def test_recovery_qualifier_rejects_git_object_substitution(
    tmp_path: Path,
    repository_scope: str,
    substitution: str,
) -> None:
    script = _sealed_qualifier_script(tmp_path)
    repository = script.parents[1]
    selected = (
        repository
        if repository_scope == "accelerator"
        else repository / "ipfs_datasets_py"
    )
    common = Path(
        subprocess.run(
            (
                "/usr/bin/git",
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if substitution == "grafts":
        head = _git_output(selected, "rev-parse", "HEAD")
        grafts = common / "info/grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(head + "\n", encoding="utf-8")
    else:
        marker = selected / "replacement-parent-fixture"
        marker.write_text("replacement parent\n", encoding="utf-8")
        _git(selected, "add", marker.name)
        _git(selected, "commit", "-qm", "replacement parent fixture")
        head = _git_output(selected, "rev-parse", "HEAD")
        parent = _git_output(selected, "rev-parse", "HEAD^")
        if repository_scope == "datasets":
            _git(repository, "add", "ipfs_datasets_py")
            _git(repository, "commit", "-qm", "bind nested replacement fixture")
        _git(selected, "update-ref", f"refs/replace/{head}", parent)
    module = _load()
    with pytest.raises(RuntimeError, match="(?:object substitution|replacement refs)"):
        module._git_object_substitution_state(selected)

    completed = subprocess.run(
        (
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            str(script),
            "--recovery",
        ),
        cwd=repository,
        env={"HOME": str(repository), "LANG": "C", "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "protected recovery" in completed.stderr


@pytest.mark.parametrize("candidate", _GITLINK_IMPORT_CANDIDATES)
def test_recovery_qualifier_gitlink_rejects_initialized_nested_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    repository = tmp_path / "outer"
    package = repository / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "sealed outer source")
    object_id = _add_sealed_stage_zero_gitlinks(repository, ("package/gitlink",))

    module = _load()
    monkeypatch.setattr(module, "_ISOLATED_RECOVERY_PYCACHE_DIRECTORY", object())
    entries = module._tracked_recovery_import_entries(
        repository, pathspecs=("package",)
    )
    assert entries["package/gitlink"] == ("160000", object_id)
    module._scan_isolated_recovery_import_roots(
        repository,
        roots=("package",),
        tracked_pathspecs=("package",),
        root_import_candidates=False,
    )

    nested = repository / "package/gitlink"
    candidate_path = nested / candidate
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(b"untrusted nested import candidate\n")
    _git(nested, "init", "-q")
    _git(nested, "config", "user.email", "fixture@example.invalid")
    _git(nested, "config", "user.name", "LGCVF Fixture")
    _git(nested, "add", ".")
    _git(nested, "commit", "-qm", "initialized nested code")
    with pytest.raises(RuntimeError, match="Gitlink contains an import candidate"):
        module._scan_isolated_recovery_import_roots(
            repository,
            roots=("package",),
            tracked_pathspecs=("package",),
            root_import_candidates=False,
        )


def test_recovery_qualifier_gitlink_rejects_non_directory_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _sealed_qualifier_script(tmp_path)
    repository = script.parents[1]
    relative = "ipfs_accelerate_py/mcplusplus"
    _add_sealed_stage_zero_gitlinks(repository, (relative,))
    placeholder = repository / relative
    placeholder.rmdir()
    placeholder.write_bytes(b"not a Gitlink directory\n")
    module = _load()
    monkeypatch.setattr(module, "_ISOLATED_RECOVERY_PYCACHE_DIRECTORY", object())
    with pytest.raises(RuntimeError, match="Gitlink placeholder is unsafe"):
        module._scan_isolated_recovery_import_roots(
            repository,
            roots=("ipfs_accelerate_py",),
            tracked_pathspecs=("ipfs_accelerate_py",),
            root_import_candidates=False,
        )


@pytest.mark.parametrize("drift", ("mode", "object-id"))
def test_recovery_qualifier_gitlink_rejects_index_identity_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    repository = tmp_path / "outer"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    _git(repository, "commit", "--allow-empty", "-qm", "sealed outer source")
    original = _add_sealed_stage_zero_gitlinks(repository, ("package/gitlink",))
    if drift == "object-id":
        replacement_source = tmp_path / "replacement"
        replacement_source.mkdir()
        _git(replacement_source, "init", "-q")
        _git(replacement_source, "config", "user.email", "fixture@example.invalid")
        _git(replacement_source, "config", "user.name", "LGCVF Fixture")
        _git(
            replacement_source,
            "commit",
            "--allow-empty",
            "-qm",
            "replacement Gitlink target",
        )
        replacement = _git_output(replacement_source, "rev-parse", "HEAD")
        replacement_mode = "160000"
    else:
        forged = tmp_path / "forged-ordinary-blob"
        forged.write_text("forged ordinary blob\n", encoding="utf-8")
        replacement = _git_output(
            repository, "hash-object", "-w", str(forged)
        )
        replacement_mode = "100644"
    assert replacement != original
    _git(
        repository,
        "update-index",
        "--cacheinfo",
        f"{replacement_mode},{replacement},package/gitlink",
    )
    module = _load()
    with pytest.raises(RuntimeError, match="ordinary index differs"):
        module._tracked_recovery_import_entries(
            repository, pathspecs=("package",)
        )


@pytest.mark.parametrize("repository_scope", ("accelerator", "datasets"))
@pytest.mark.parametrize("attack", ("fsmonitor", "filter", "same-size-mtime"))
def test_recovery_qualifier_git_observations_are_raw_head_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_scope: str,
    attack: str,
) -> None:
    outer = tmp_path / "accelerator"
    outer_source = outer / "package/source.py"
    outer_source.parent.mkdir(parents=True)
    outer_source.write_bytes(b"VALUE = 'AAAA'\n")
    nested = outer / "ipfs_datasets_py"
    nested_source = nested / "package/source.py"
    nested_source.parent.mkdir(parents=True)
    nested_source.write_bytes(b"VALUE = 'AAAA'\n")
    for repository in (nested, outer):
        _git(repository, "init", "-q")
        _git(repository, "config", "user.email", "fixture@example.invalid")
        _git(repository, "config", "user.name", "LGCVF Fixture")
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "sealed raw Git fixture")

    module = _load()
    monkeypatch.setattr(module, "_ISOLATED_RECOVERY_PYCACHE_DIRECTORY", object())
    selected = outer if repository_scope == "accelerator" else nested
    source = outer_source if repository_scope == "accelerator" else nested_source
    original = source.stat()
    source.write_bytes(b"VALUE = 'BBBB'\n")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    if attack == "same-size-mtime":
        with pytest.raises(RuntimeError, match="differs from HEAD"):
            module._scan_isolated_recovery_import_roots(
                selected,
                roots=("package",),
                tracked_pathspecs=("package",),
                root_import_candidates=False,
            )
        return

    common = Path(_git_output(selected, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    marker = tmp_path / f"{repository_scope}-{attack}-executed"
    hook = common / f"{attack}-hook"
    hook.write_text(
        "#!/bin/sh\n"
        f"touch {str(marker)!r}\n"
        "cat\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    if attack == "fsmonitor":
        _git(selected, "config", "core.fsmonitor", str(hook))
        with pytest.raises(RuntimeError, match="source is not clean"):
            module._clean_recovery_import_source(selected)
    else:
        _git(selected, "config", "filter.lgcvf-evil.clean", str(hook))
        attributes = common / "info/attributes"
        attributes.parent.mkdir(parents=True, exist_ok=True)
        attributes.write_text("*.py filter=lgcvf-evil\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="(?:substitution|filter drivers)"):
            module._git_object_substitution_state(selected)
    assert marker.exists() is False


@pytest.mark.parametrize("repository_scope", ("accelerator", "datasets"))
def test_recovery_projection_rejects_raw_nonpython_blob_drift(
    tmp_path: Path,
    repository_scope: str,
) -> None:
    outer = tmp_path / "accelerator"
    outer.mkdir()
    outer_config = outer / "pytest.ini"
    outer_config.write_bytes(b"[pytest]\naddopts=-q\n")
    nested = outer / "ipfs_datasets_py"
    nested.mkdir()
    nested_config = nested / "pyproject.toml"
    nested_config.write_bytes(b"[tool.pytest.ini_options]\naddopts='-q'\n")
    for repository in (nested, outer):
        _git(repository, "init", "-q")
        _git(repository, "config", "user.email", "fixture@example.invalid")
        _git(repository, "config", "user.name", "LGCVF Fixture")
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "sealed projection fixture")
    module = _load()
    if repository_scope == "accelerator":
        public_path = "pytest.ini"
        selected = outer_config
    else:
        public_path = "ipfs_datasets_py/pyproject.toml"
        selected = nested_config
    module._recovery_projection_manifest(outer, (public_path,))
    original = selected.stat()
    payload = bytearray(selected.read_bytes())
    payload[-2] = ord("x") if payload[-2] != ord("x") else ord("y")
    selected.write_bytes(payload)
    os.utime(selected, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(module.QualificationError, match="index and HEAD"):
        module._recovery_projection_manifest(outer, (public_path,))


@pytest.mark.parametrize(
    ("flags", "accepted"),
    (
        ((), False),
        (("-S", "-B"), False),
        (("-I", "-B"), False),
        (("-I", "-S"), False),
        (("-I", "-S", "-B"), True),
    ),
)
def test_protected_recovery_requires_exact_isolated_python_flags(
    flags: tuple[str, ...],
    accepted: bool,
    tmp_path: Path,
) -> None:
    isolated_script = _sealed_qualifier_script(tmp_path)
    probe = r'''
import importlib.util
import sys
from pathlib import Path

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_flag_guard", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
try:
    module._require_isolated_recovery_runtime()
except module.QualificationError:
    print("rejected")
else:
    print("accepted")
'''
    completed = subprocess.run(
        (sys.executable, *flags, "-c", probe, str(isolated_script)),
        cwd=ROOT,
        env={"HOME": str(ROOT), "LANG": "C", "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ("accepted" if accepted else "rejected")


def test_protected_recovery_rejects_mutable_flag_spoofs(tmp_path: Path) -> None:
    isolated_script = _sealed_qualifier_script(tmp_path)
    probe = r'''
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_flag_spoof", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
original_flags = sys.flags
original_dont_write_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    if sys.argv[2] == "all-python-flags":
        sys.flags = SimpleNamespace(
            isolated=1,
            ignore_environment=1,
            no_site=1,
            safe_path=True,
            dont_write_bytecode=1,
        )
    try:
        module._require_isolated_recovery_runtime()
    except module.QualificationError:
        outcome = "rejected"
    else:
        outcome = "accepted"
finally:
    sys.flags = original_flags
    sys.dont_write_bytecode = original_dont_write_bytecode
print(outcome)
'''
    environment = {"HOME": str(ROOT), "LANG": "C", "PATH": "/usr/bin:/bin"}
    missing_b = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(isolated_script),
            "runtime-only",
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert missing_b.returncode == 0, missing_b.stderr
    assert missing_b.stdout.strip() == "rejected"

    replaced_flags = subprocess.run(
        (
            sys.executable,
            "-c",
            probe,
            str(isolated_script),
            "all-python-flags",
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replaced_flags.returncode == 0, replaced_flags.stderr
    assert replaced_flags.stdout.strip() == "rejected"


def test_recovery_public_apis_and_cli_reject_nonisolated_python(tmp_path: Path) -> None:
    isolated_script = _sealed_qualifier_script(tmp_path)
    probe = r'''
import importlib.util
import json
import sys
from pathlib import Path

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_public_guard", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
if len(sys.argv) > 2 and sys.argv[2] == "spoof-write-bytecode":
    sys.dont_write_bytecode = True
unavailable = module._recovery_unavailable(module.QualificationError("fixture"))
operations = (
    module.run_preregistered_recovery_qualification,
    lambda: module.verify_preregistered_recovery_qualification(
        unavailable,
        require_passed=False,
    ),
    lambda: module.isolated_bound_duckdb_runtime(
        expected_runtime_cid="baguqeera-untrusted"
    ).__enter__(),
)
errors = []
for operation in operations:
    try:
        operation()
    except module.QualificationError as exc:
        errors.append(str(exc))
    else:
        raise SystemExit("nonisolated recovery API was admitted")
print(json.dumps(errors))
'''
    environment = {"HOME": str(ROOT), "LANG": "C", "PATH": "/usr/bin:/bin"}
    direct = subprocess.run(
        (sys.executable, "-c", probe, str(isolated_script)),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert direct.returncode == 0, direct.stderr
    assert json.loads(direct.stdout) == [
        "protected recovery requires python -I -S -B"
    ] * 3

    missing_b_spoof = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(isolated_script),
            "spoof-write-bytecode",
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert missing_b_spoof.returncode == 0, missing_b_spoof.stderr
    assert json.loads(missing_b_spoof.stdout) == [
        "protected recovery requires python -I -S -B"
    ] * 3

    cli = subprocess.run(
        (sys.executable, str(isolated_script), "--recovery"),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode == 1, cli.stderr
    result = json.loads(cli.stdout)
    assert result == {
        "valid": False,
        "error": "QualificationError",
        "reason": "protected recovery requires python -I -S -B",
    }


def test_bound_duckdb_runtime_rejects_ambient_preload(tmp_path: Path) -> None:
    isolated_script = _sealed_qualifier_script(tmp_path)
    probe = r'''
import importlib
import importlib.util
import sys
from pathlib import Path

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_duckdb_preload", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
evidence = module.bound_duckdb_runtime_evidence()
site_root = str(module._pytest_site_roots()[0])
sys.path.insert(0, site_root)
try:
    importlib.import_module("duckdb")
finally:
    sys.path.remove(site_root)
try:
    with module.isolated_bound_duckdb_runtime(
        expected_runtime_cid=evidence["runtime_cid"]
    ):
        raise AssertionError("preloaded runtime admitted")
except module.QualificationError:
    raise SystemExit(0)
raise SystemExit(2)
'''
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", probe, str(isolated_script)),
        cwd=ROOT,
        env={"HOME": str(ROOT), "LANG": "C", "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_bound_duckdb_runtime_record_and_projection_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    site_root = tmp_path / "site"
    site_root.mkdir()
    _fake_duckdb_site(module, site_root)
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (site_root,))

    evidence, projected = module._resolve_bound_duckdb_runtime()
    assert evidence["schema"] == "lgcvf-bound-duckdb-runtime@1"
    assert evidence["runtime_cid"] == module.content_identity(
        {key: value for key, value in evidence.items() if key != "runtime_cid"}
    )
    assert set(projected) == {
        *module._DUCKDB_RUNTIME_SOURCE_PATHS,
        evidence["native_extension_path"],
        evidence["metadata"]["path"],
        evidence["record"]["path"],
    }
    assert not any("__pycache__" in path for path in projected)

    duplicate_record = b"duckdb/__init__.py,,\nduckdb/__init__.py,,\n"
    with pytest.raises(module.QualificationError, match="duplicate RECORD path"):
        module._duckdb_record_rows(duplicate_record, noun="duplicate DuckDB RECORD")
    traversal_record = b"duckdb/../duckdb/__init__.py,,\n"
    with pytest.raises(module.QualificationError, match="unsafe RECORD path"):
        module._duckdb_record_rows(traversal_record, noun="traversal DuckDB RECORD")

    distribution = site_root / "duckdb-1.5.5.dist-info"
    metadata = distribution / "METADATA"
    metadata.write_bytes(metadata.read_bytes() + b"Summary: tampered\n")
    with pytest.raises(module.QualificationError, match="METADATA identity differs"):
        module._resolve_bound_duckdb_runtime()

    missing_root = tmp_path / "missing-metadata-row"
    missing_root.mkdir()
    _fake_duckdb_site(module, missing_root)
    missing_record = missing_root / "duckdb-1.5.5.dist-info/RECORD"
    missing_record.write_text(
        "".join(
            line
            for line in missing_record.read_text(encoding="utf-8").splitlines(True)
            if not line.startswith("duckdb-1.5.5.dist-info/METADATA,")
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (missing_root,))
    with pytest.raises(module.QualificationError, match="does not bind its exact METADATA"):
        module._resolve_bound_duckdb_runtime()

    duplicate_root = tmp_path / "duplicate-metadata-row"
    duplicate_root.mkdir()
    _fake_duckdb_site(module, duplicate_root)
    duplicate_record_path = duplicate_root / "duckdb-1.5.5.dist-info/RECORD"
    duplicate_record_lines = duplicate_record_path.read_text(
        encoding="utf-8"
    ).splitlines(True)
    metadata_row = next(
        line
        for line in duplicate_record_lines
        if line.startswith("duckdb-1.5.5.dist-info/METADATA,")
    )
    duplicate_record_path.write_text(
        "".join((*duplicate_record_lines, metadata_row)),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (duplicate_root,))
    with pytest.raises(module.QualificationError, match="duplicate RECORD path"):
        module._resolve_bound_duckdb_runtime()


def test_bound_duckdb_runtime_rejects_ambiguous_native_and_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    extra_native_root = tmp_path / "extra-native"
    extra_native_root.mkdir()
    _fake_duckdb_site(module, extra_native_root, extra_native=True)
    monkeypatch.setattr(
        module,
        "_pytest_site_roots",
        lambda: (extra_native_root,),
    )
    with pytest.raises(module.QualificationError, match="one native runtime"):
        module._resolve_bound_duckdb_runtime()

    duplicate_root = tmp_path / "duplicates"
    duplicate_root.mkdir()
    _fake_duckdb_site(module, duplicate_root, versions=("1.5.5", "1.5.6"))
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (duplicate_root,))
    with pytest.raises(
        module.QualificationError,
        match="one exact RECORD-matched distribution",
    ):
        module._resolve_bound_duckdb_runtime()

    bounded_root = tmp_path / "bounded"
    bounded_root.mkdir()
    for ordinal in range(module._MAX_DUCKDB_DISTRIBUTION_CANDIDATES + 1):
        (bounded_root / f"duckdb-1.5.{ordinal}.dist-info").mkdir()
    descriptor = os.open(bounded_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(module.QualificationError, match="population exceeds"):
            module._duckdb_distribution_candidate_names(descriptor)
    finally:
        os.close(descriptor)


def test_bound_duckdb_runtime_rejects_symlink_and_hardlink_files(
    tmp_path: Path,
) -> None:
    module = _load()
    root = tmp_path / "site"
    package = root / "duckdb"
    package.mkdir(parents=True)
    target = root / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "symlink.py").symlink_to(target)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(module.QualificationError, match="unavailable"):
            module._read_owned_regular_relative(
                descriptor,
                "duckdb/symlink.py",
                noun="symlinked DuckDB runtime",
                limit=1024,
            )
        os.link(target, package / "hardlink.py")
        with pytest.raises(module.QualificationError, match="not a bounded owned"):
            module._read_owned_regular_relative(
                descriptor,
                "duckdb/hardlink.py",
                noun="hardlinked DuckDB runtime",
                limit=1024,
            )
    finally:
        os.close(descriptor)


def test_pytest_distribution_resolution_uses_exact_record_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    package = tmp_path / "pytest"
    package.mkdir()
    package_bytes = b'__version__ = "9.1.1"\n'
    (package / "__init__.py").write_bytes(package_bytes)

    def add_distribution(version: str, recorded: bytes) -> None:
        distribution = tmp_path / f"pytest-{version}.dist-info"
        distribution.mkdir()
        metadata_bytes = (
            "Metadata-Version: 2.1\n"
            "Name: pytest\n"
            f"Version: {version}\n\n"
        ).encode()
        (distribution / "METADATA").write_bytes(metadata_bytes)
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(recorded).digest()
        ).decode("ascii").rstrip("=")
        metadata_digest = base64.urlsafe_b64encode(
            hashlib.sha256(metadata_bytes).digest()
        ).decode("ascii").rstrip("=")
        (distribution / "RECORD").write_text(
            f"../../../bin/py.test,sha256={digest},{len(recorded)}\n"
            f"../../../bin/pytest,sha256={digest},{len(recorded)}\n"
            f"pytest/__init__.py,sha256={digest},{len(recorded)}\n"
            f"pytest-{version}.dist-info/METADATA,"
            f"sha256={metadata_digest},{len(metadata_bytes)}\n"
            f"pytest-{version}.dist-info/RECORD,,\n",
            encoding="utf-8",
        )

    add_distribution("9.0.3", b"stale package bytes")
    add_distribution("9.1.1", package_bytes)
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (tmp_path,))

    assert module._pytest_distribution_version() == "9.1.1"

    add_distribution("9.1.2", package_bytes)
    with pytest.raises(
        module.QualificationError,
        match="one exact RECORD-matched installation",
    ):
        module._pytest_distribution_version()


@pytest.mark.parametrize(
    "alias",
    (
        "pytest/../pytest/__init__.py",
        "other/../pytest/__init__.py",
        r"pytest\__init__.py",
        "/pytest/__init__.py",
        "pytest/\x00/__init__.py",
        "C:/pytest/__init__.py",
        "pytest//__init__.py",
        "pytest/./__init__.py",
    ),
)
def test_pytest_record_rejects_noncanonical_aliases(alias: str) -> None:
    module = _load()
    metadata_bytes = b"Metadata-Version: 2.1\nName: pytest\nVersion: 1\n\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(b"package").digest()).decode(
        "ascii"
    ).rstrip("=")
    metadata_digest = base64.urlsafe_b64encode(
        hashlib.sha256(metadata_bytes).digest()
    ).decode("ascii").rstrip("=")
    record = (
        f"pytest/__init__.py,sha256={digest},7\n"
        "pytest-1.dist-info/METADATA,"
        f"sha256={metadata_digest},{len(metadata_bytes)}\n"
        f"{alias},sha256={digest},7\n"
    ).encode()

    with pytest.raises(module.QualificationError, match="unsafe RECORD path"):
        module._pytest_record_identity(
            record,
            noun="pytest alias RECORD",
            distribution_name="pytest-1.dist-info",
            metadata_bytes=metadata_bytes,
        )


@pytest.mark.parametrize("mutation", ("missing", "tampered", "duplicate"))
def test_pytest_record_requires_exact_metadata_identity(mutation: str) -> None:
    module = _load()
    metadata_bytes = b"Metadata-Version: 2.1\nName: pytest\nVersion: 1\n\n"
    package_digest = base64.urlsafe_b64encode(
        hashlib.sha256(b"package").digest()
    ).decode("ascii").rstrip("=")
    recorded_metadata = metadata_bytes if mutation != "tampered" else b"tampered"
    metadata_digest = base64.urlsafe_b64encode(
        hashlib.sha256(recorded_metadata).digest()
    ).decode("ascii").rstrip("=")
    metadata_row = (
        "pytest-1.dist-info/METADATA,"
        f"sha256={metadata_digest},{len(recorded_metadata)}\n"
    )
    if mutation == "missing":
        metadata_rows = ""
    elif mutation == "duplicate":
        metadata_rows = metadata_row + metadata_row
    else:
        metadata_rows = metadata_row
    record = (
        f"pytest/__init__.py,sha256={package_digest},7\n" + metadata_rows
    ).encode()

    with pytest.raises(module.QualificationError, match="METADATA|duplicate"):
        module._pytest_record_identity(
            record,
            noun="pytest metadata RECORD",
            distribution_name="pytest-1.dist-info",
            metadata_bytes=metadata_bytes,
        )


def test_pytest_distribution_candidate_enumeration_is_bounded_and_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    for ordinal in range(module._MAX_PYTEST_DISTRIBUTION_CANDIDATES + 1):
        (tmp_path / f"pytest-{ordinal}.dist-info").mkdir()
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (tmp_path,))

    with pytest.raises(module.QualificationError, match="population exceeds"):
        module._pytest_distribution_version()


def test_pytest_distribution_resolution_rejects_unsafe_distribution_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    package = tmp_path / "pytest"
    package.mkdir()
    package_bytes = b'__version__ = "1"\n'
    (package / "__init__.py").write_bytes(package_bytes)
    distribution = tmp_path / "pytest-1.dist-info"
    distribution.mkdir()
    outside = tmp_path / "outside-metadata"
    outside.write_text(
        "Metadata-Version: 2.1\nName: pytest\nVersion: 1\n\n",
        encoding="utf-8",
    )
    (distribution / "METADATA").symlink_to(outside)
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(package_bytes).digest()
    ).decode("ascii").rstrip("=")
    (distribution / "RECORD").write_text(
        f"pytest/__init__.py,sha256={digest},{len(package_bytes)}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_pytest_site_roots", lambda: (tmp_path,))

    with pytest.raises(module.QualificationError, match="METADATA is unavailable"):
        module._pytest_distribution_version()


def test_recovery_unavailable_is_typed_and_cannot_be_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()

    def unavailable(*, root: Path = module.ROOT) -> dict[str, object]:
        del root
        raise module.QualificationError("sandbox unavailable")

    monkeypatch.setattr(module, "_recovery_source_binding", unavailable)
    result = module.run_preregistered_recovery_qualification()
    assert result["schema"] == module.RECOVERY_UNAVAILABLE_SCHEMA
    assert result["disposition"] == "unavailable"
    assert result["completion_authoritative"] is False
    assert (
        module.verify_preregistered_recovery_qualification(
            result, require_passed=False
        )
        == result
    )
    with pytest.raises(module.QualificationError, match="unavailable"):
        module.verify_preregistered_recovery_qualification(result)

    forged = dict(result)
    forged["reason"] = "forged pass"
    with pytest.raises(module.QualificationError, match="differs"):
        module.verify_preregistered_recovery_qualification(
            forged, require_passed=False
        )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_public_recovery_runner_executes_exact_six_without_completion_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    # The shared development worktree is intentionally dirty while this judge
    # and its consumer are edited.  Source-binding behavior has a real Git
    # fixture above; only that precondition is substituted here so the public
    # runner still executes all six real closed suites through the OS sandbox.
    source = {
        "schema": "lgcvf-recovery-source-binding-test-fixture@1",
        "source_binding_cid": module.content_identity({"fixture": "unchanged"}),
        "accelerator_head": "1" * 40,
        "accelerator_tree": "2" * 40,
        "datasets_gitlink": "3" * 40,
        "datasets_tree": "4" * 40,
    }
    monkeypatch.setattr(
        module,
        "_recovery_source_binding",
        lambda *, root=module.ROOT: dict(source),
    )
    # Raw index/HEAD closure is exercised against real temporary repositories
    # above.  This expensive exact-six regression runs from the intentionally
    # dirty shared development worktree, so provide only the equivalent blob
    # identities while retaining the production no-follow raw-byte checks.
    monkeypatch.setattr(
        module,
        "_recovery_projection_git_entries",
        _development_blob_entries,
    )

    receipt = module.run_preregistered_recovery_qualification()

    verified = module.verify_preregistered_recovery_qualification(receipt)
    assert verified == receipt
    assert receipt["disposition"] == "passed"
    assert receipt["passed"] is True
    assert receipt["totals"]["collected"] == 126
    assert receipt["totals"]["passed_count"] == 126
    assert [item["task_id"] for item in receipt["suites"]] == [
        item.task_id for item in module.RECOVERY_VALIDATIONS
    ]
    assert all(item["candidate_authored"] is True for item in receipt["suites"])
    assert all(item["self_authority"] is False for item in receipt["suites"])
    assert receipt["completion_authoritative"] is False
    assert receipt["test_qualification_complete"] is False
    assert receipt["release_qualified"] is False
    assert receipt["production_authorized"] is False
    omission = receipt["validation_projection_omission_commitment"]
    assert omission["schema"] == (
        "lgcvf-recovery-validation-projection-omission@1"
    )
    assert omission["accelerator_head"] == source["accelerator_head"]
    assert omission["accelerator_tree"] == source["accelerator_tree"]
    assert omission["datasets_gitlink"] == source["datasets_gitlink"]
    assert omission["datasets_tree"] == source["datasets_tree"]
    assert omission["commitment_cid"] == module.content_identity(
        {key: item for key, item in omission.items() if key != "commitment_cid"}
    )
    assert receipt["validation_projection_omission_root"] == omission[
        "commitment_cid"
    ]
    projection_evidence = receipt["validation_projection_evidence_commitment"]
    assert projection_evidence["schema"] == (
        "lgcvf-recovery-validation-projection-evidence@1"
    )
    assert projection_evidence["omission_root"] == omission["commitment_cid"]
    assert [item["task_id"] for item in projection_evidence["ordered_suites"]] == [
        item.task_id for item in module.RECOVERY_VALIDATIONS
    ]
    assert projection_evidence["commitment_cid"] == module.content_identity(
        {
            key: item
            for key, item in projection_evidence.items()
            if key != "commitment_cid"
        }
    )
    assert receipt["validation_projection_evidence_root"] == projection_evidence[
        "commitment_cid"
    ]

    forged = copy.deepcopy(receipt)
    forged["limitations"] = ["trust caveats removed"]
    forged["receipt_cid"] = module.content_identity(
        {key: item for key, item in forged.items() if key != "receipt_cid"}
    )
    with pytest.raises(module.QualificationError, match="mandatory caveats"):
        module.verify_preregistered_recovery_qualification(forged)

    forged_omission = copy.deepcopy(receipt)
    forged_omission["validation_projection_omission_commitment"][
        "omitted_source_symlinks"
    ] = []
    forged_omission["validation_projection_omission_commitment"][
        "commitment_cid"
    ] = module.content_identity(
        {
            key: item
            for key, item in forged_omission[
                "validation_projection_omission_commitment"
            ].items()
            if key != "commitment_cid"
        }
    )
    forged_omission["validation_projection_omission_root"] = forged_omission[
        "validation_projection_omission_commitment"
    ]["commitment_cid"]
    forged_omission["receipt_cid"] = module.content_identity(
        {
            key: item
            for key, item in forged_omission.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(module.QualificationError, match="omission commitment"):
        module.verify_preregistered_recovery_qualification(forged_omission)

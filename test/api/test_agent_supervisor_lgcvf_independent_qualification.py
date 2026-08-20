"""Independent-judge tests for LGCVF revision-2 qualification."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
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


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
def test_recovery_observation_is_sealed_bounded_and_rejects_forgery() -> None:
    module = _load()
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
    assert binding["repository_topology"] == "accelerator_with_datasets_gitlink"
    assert binding["datasets_gitlink"] == datasets_head
    assert binding["datasets_head"] == datasets_head
    assert binding["accelerator_clean"] is True
    assert binding["datasets_clean"] is True

    (datasets / "semantic.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(module.QualificationError, match="clean .* overlays"):
        module._recovery_source_binding(root=repository)


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
    }
    monkeypatch.setattr(
        module,
        "_recovery_source_binding",
        lambda *, root=module.ROOT: dict(source),
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

    forged = copy.deepcopy(receipt)
    forged["limitations"] = ["trust caveats removed"]
    forged["receipt_cid"] = module.content_identity(
        {key: item for key, item in forged.items() if key != "receipt_cid"}
    )
    with pytest.raises(module.QualificationError, match="mandatory caveats"):
        module.verify_preregistered_recovery_qualification(forged)

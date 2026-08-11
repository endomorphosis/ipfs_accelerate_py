"""Control-plane tests for the deliberately unresolved SCH-000 seal."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SEAL_PATH = REPO_ROOT / "config/semantic_state_dependencies.seal.json"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_semantic_state_dependencies.py"
PLAN_PATH = REPO_ROOT / "docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md"
TODO_PATH = REPO_ROOT / "docs/architecture/semantic_compression_harness.todo.md"


def _load_validator():
    spec = importlib.util.spec_from_file_location("sch_dependency_seal_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_authority(
    tmp_path: Path,
    *,
    command: list[str] | None = None,
    timeout: int = 10,
) -> tuple[Path, dict[str, Any]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.email", "seal@example.invalid")
    _git(repository, "config", "user.name", "Seal Test")
    (repository / "README.md").write_text("sealed\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-q", "-m", "sealed")
    _git(
        repository,
        "remote",
        "add",
        "origin",
        "https://github.com/endomorphosis/ipfs_accelerate_py",
    )
    authority = {
        "role": "accelerate_harness",
        "origin": "https://github.com/endomorphosis/ipfs_accelerate_py",
        "commit": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "required_blobs": [
            {
                "path": "README.md",
                "oid": _git(repository, "rev-parse", "HEAD:README.md"),
            }
        ],
        "required_test_commands": [command or ["python3.12", "-c", "raise SystemExit(0)"]],
        "test_timeout_seconds": timeout,
    }
    return repository, authority


def test_checked_in_seal_is_intentionally_unresolved_and_fails_closed(
    capsys,
) -> None:
    validator = _load_validator()
    seal = validator.load_seal(SEAL_PATH)
    authorities = {item["role"]: item for item in seal["authorities"]}

    assert seal["status"] == "unresolved"
    assert authorities["accelerate_harness"]["commit"] == ("UNRESOLVED_REPAIRED_ACCELERATE_COMMIT")
    assert authorities["accelerate_harness"]["tree"] == ("UNRESOLVED_REPAIRED_ACCELERATE_TREE")
    assert authorities["accelerate_harness"]["interface_fingerprint"] == (
        "UNRESOLVED_REPAIRED_ACCELERATE_INTERFACE_FINGERPRINT"
    )
    assert authorities["incremental_semantic_index"]["commit"] == ("UNRESOLVED_FINAL_ISI_COMMIT")
    assert authorities["semantic_state_contracts"]["commit"] == ("UNRESOLVED_FINAL_DSS_COMMIT")
    assert authorities["kit_state_roots"]["commit"] == ("05ba9375923cd5fb52e2c9c18b98b530d57d077f")
    assert authorities["mcp_plus_plus"]["commit"] == ("dc3164653a48d059ae9812078359daeafb451c07")
    assert seal["unresolved_authority_reasons"]["accelerate_harness"] == [
        "live_owner_without_heartbeat_can_split_brain_and_swallow_lost_fence",
        "stale_owner_can_overwrite_newer_active_task_index",
        "empty_or_unavailable_process_snapshot_fails_open",
        "whitespace_validation_omits_untracked_and_submodule_outputs",
        "fast_zombie_birth_capture_can_leak_lease",
    ]

    errors = validator.validate_seal(SEAL_PATH)
    assert "seal: status must be 'sealed'" in errors
    assert "seal: unresolved placeholder present" in errors
    assert any(error.startswith("checkout bindings missing:") for error in errors)

    assert validator.main(["--check", str(SEAL_PATH)]) == 1
    assert "ERROR: seal: unresolved placeholder present" in capsys.readouterr().err


def test_resolved_authority_fingerprints_bind_complete_contracts() -> None:
    validator = _load_validator()
    seal = validator.load_seal(SEAL_PATH)
    authorities = {item["role"]: item for item in seal["authorities"]}

    for role in ("kit_state_roots", "mcp_plus_plus"):
        authority = authorities[role]
        assert authority["interface_fingerprint"] == (validator.authority_fingerprint(authority))


def test_document_contract_rejects_unknown_fields_even_while_unresolved() -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    seal["unexpected_relaxation"] = True

    assert "seal: unknown fields: unexpected_relaxation" in validator.validate_document(seal)


def test_accelerator_audit_reasons_cannot_be_partially_cleared() -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    seal["unresolved_authority_reasons"]["accelerate_harness"].pop()

    assert (
        "seal: unresolved authority reasons do not equal the operator audit"
        in validator.validate_document(seal)
    )


def test_loader_rejects_duplicate_json_members(tmp_path: Path) -> None:
    validator = _load_validator()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")

    try:
        validator.load_seal(duplicate)
    except validator.DuplicateKeyError as exc:
        assert "duplicate JSON member 'schema'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate JSON member was accepted")


def test_wire_boundary_and_consumer_references_are_exact() -> None:
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    assert seal["wire_contract"] == {
        "authority_role": "mcp_plus_plus",
        "profiles": ["A", "B", "F"],
        "payload_role": "accelerate_application_payload_only",
        "generic_envelope_types_owned_externally": True,
        "local_envelope_hasher_forbidden": True,
    }
    contracts = {
        item["role"]: item["interface_contract"]["consumer_api_requirements"]
        for item in seal["authorities"]
    }
    accelerate = contracts["accelerate_harness"]
    datasets = contracts["semantic_state_contracts"]
    assert any(value.startswith("SemanticCapsuleRef(") for value in accelerate)
    assert any(value.startswith("TestSelectionRef(") for value in accelerate)
    assert (
        "SemanticStateProvider.open_semantic_state(root_cid:str,"
        "get_block:Callable[[str],bytes])->SemanticStateView"
    ) in accelerate
    assert (
        "open_semantic_state(root_cid:str,get_block:Callable[[str],bytes])->SemanticStateView"
    ) in datasets
    assert any(value.startswith("SemanticStateView.") for value in datasets)


def test_role_manifest_and_test_commands_cannot_be_replaced_with_noops() -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    authority = seal["authorities"][0]
    authority["required_blobs"] = [{"path": "README.md", "oid": "a" * 40}]
    authority["required_test_commands"] = [["python3.12", "-c", "pass"]]
    authority["interface_fingerprint"] = validator.authority_fingerprint(authority)

    errors = validator.validate_document(seal)
    assert any("required_blobs paths do not equal" in error for error in errors)
    assert any("required_test_commands do not equal" in error for error in errors)


def test_operator_owned_pin_and_interface_cannot_be_self_resealed() -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    kit = seal["authorities"][3]
    kit["commit"] = "a" * 40
    kit["interface_contract"]["consumer_api_requirements"] = ["SyntheticBypass()"]
    kit["interface_fingerprint"] = validator.authority_fingerprint(kit)

    errors = validator.validate_document(seal)
    assert any("commit does not equal the operator-owned pin" in error for error in errors)
    assert any("interface_contract must equal" in error for error in errors)


def test_checkout_binding_must_name_canonical_worktree_root(tmp_path: Path) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(tmp_path)
    (repository / "nested").mkdir()

    assert validator.validate_checkout(authority, repository) == []
    assert validator.validate_checkout(authority, repository / "nested") == [
        "checkout[accelerate_harness]: path must be the canonical Git worktree root"
    ]


def test_every_role_requires_a_separate_checkout(tmp_path: Path) -> None:
    validator = _load_validator()
    shared = tmp_path.resolve()
    errors = validator._distinct_checkout_errors(
        {
            "incremental_semantic_index": shared,
            "semantic_state_contracts": shared,
        }
    )

    assert errors == [
        "checkout bindings must be separate; shared path for roles: "
        "incremental_semantic_index, semantic_state_contracts"
    ]


def test_ast_audit_detects_duplicate_authorities_and_alias_bypasses(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    package = tmp_path / "ipfs_accelerate_py/agent_supervisor/semantic_state"
    package.mkdir(parents=True)
    (package / "bad.py").write_text(
        "import ast as syntax\n"
        "from producer import Contract as SemanticStateView\n"
        "ExecutionEnvelope = dict\n"
        "class TestSelection:\n"
        "    pass\n"
        "def envelope_cid(value):\n"
        "    return value\n",
        encoding="utf-8",
    )

    violations = validator._forbidden_duplicate_authorities(tmp_path)
    assert any("ast" in item for item in violations)
    assert any("SemanticStateView" in item for item in violations)
    assert any("ExecutionEnvelope" in item for item in violations)
    assert any("TestSelection" in item for item in violations)
    assert any("envelope_cid" in item for item in violations)


def test_ast_audit_allows_only_the_sealed_provider_delegation_methods(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    package = tmp_path / "ipfs_accelerate_py/agent_supervisor/semantic_state"
    package.mkdir(parents=True)
    (package / "adapter.py").write_text(
        "class IpfsDatasetsSemanticStateProvider:\n"
        "    def open_semantic_state(self, root_cid, get_block):\n"
        "        return self._api.open_semantic_state(root_cid, get_block)\n"
        "    def scan_repository(self, repo_path, previous_state=None):\n"
        "        return self._api.scan_repository(repo_path, previous_state)\n",
        encoding="utf-8",
    )

    assert validator._forbidden_duplicate_authorities(tmp_path) == []


def test_sealed_validation_cannot_skip_required_test_execution(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    seal["status"] = "sealed"
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(seal), encoding="utf-8")

    errors = validator.validate_seal(path, run_tests=False)
    assert "seal: sealed validation requires --run-tests" in errors


def test_test_environment_replaces_ambient_import_overrides(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    repository = tmp_path.resolve()
    monkeypatch.setenv("PYTHONPATH", "/ambient/editable")
    monkeypatch.setenv("PYTHONHOME", "/ambient/home")
    monkeypatch.setenv("VIRTUAL_ENV", "/ambient/venv")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=required-tests")

    environment = validator._test_environment(repository)
    assert environment["PYTHONPATH"] == os.fspath(repository)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert "PYTEST_ADDOPTS" not in environment


def test_required_test_timeout_is_bounded(tmp_path: Path) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(
        tmp_path,
        command=["python3.12", "-c", "import time; time.sleep(5)"],
        timeout=1,
    )

    errors = validator._run_required_tests(authority, repository)
    assert len(errors) == 1
    assert "timed out after 1s" in errors[0]


def test_post_test_checkout_mutation_fails_closed(tmp_path: Path) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(tmp_path)
    mutation = repository / "mutation.txt"
    authority["required_test_commands"] = [
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(mutation)!r}).write_text('x')",
        ]
    ]

    errors = validator._run_required_tests(authority, repository)
    assert "checkout[accelerate_harness]: checkout is dirty" in errors


def test_loader_rejects_non_finite_and_recursive_non_json_values(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"nested":[{"value":NaN}]}', encoding="utf-8")

    try:
        validator.load_seal(invalid)
    except ValueError as exc:
        assert "non-finite JSON number" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("NaN was accepted")
    assert validator._validate_json_value({"nested": [float("inf")]}) == [
        "seal.nested[0]: floating-point values are forbidden"
    ]


def test_source_extraction_is_recomputed_from_pinned_blob(tmp_path: Path) -> None:
    validator = _load_validator()
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.email", "seal@example.invalid")
    _git(repository, "config", "user.name", "Seal Test")
    (repository / "api.py").write_text(
        "def open_state(root: str, *, strict: bool = True) -> bytes:\n    return root.encode()\n",
        encoding="utf-8",
    )
    _git(repository, "add", "api.py")
    _git(repository, "commit", "-q", "-m", "api")
    extraction = {
        "kind": "python_signature",
        "path": "api.py",
        "selector": "open_state",
        "value": "def open_state(root: str, *, strict: bool=True) -> bytes",
    }
    authority = {
        "role": "synthetic",
        "commit": _git(repository, "rev-parse", "HEAD"),
        "interface_contract": {"source_extractions": [extraction]},
    }

    assert validator._verify_source_extractions(authority, repository) == []
    extraction["value"] = "def open_state(root: str) -> bytes"
    assert (
        "source extraction mismatch"
        in validator._verify_source_extractions(authority, repository)[0]
    )


def test_checkout_rejects_hidden_index_flags_and_verifies_working_bytes(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(tmp_path)
    _git(repository, "update-index", "--assume-unchanged", "README.md")
    (repository / "README.md").write_text("hidden mutation\n", encoding="utf-8")

    errors = validator.validate_checkout(authority, repository)
    assert any("skip-worktree/assume-unchanged" in error for error in errors)
    assert any("working bytes differ from HEAD" in error for error in errors)


def test_checkout_rejects_skip_worktree_even_without_a_visible_diff(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(tmp_path)
    _git(repository, "update-index", "--skip-worktree", "README.md")

    errors = validator.validate_checkout(authority, repository)

    assert any("skip-worktree/assume-unchanged" in error for error in errors)


def test_toolchain_binds_exact_executable_pytest_and_distribution_digest() -> None:
    validator = _load_validator()
    seal = validator.load_seal(SEAL_PATH)
    toolchain = copy.deepcopy(seal["toolchain"])
    executable = Path(toolchain["python_executable"])

    assert validator.validate_toolchain(toolchain, executable) == []
    toolchain["pytest_sha256"] = "sha256:" + "0" * 64
    assert any(
        "live Python/pytest projection differs from seal" in error
        for error in validator.validate_toolchain(toolchain, executable)
    )


def test_private_materialization_reconstructs_exact_git_objects(tmp_path: Path) -> None:
    validator = _load_validator()
    repository, authority = _repository_authority(tmp_path)
    materialization = tmp_path / "private"
    materialization.mkdir(mode=0o700)

    validator._materialize_commit(repository, authority["commit"], materialization)

    assert materialization != repository
    assert materialization.stat().st_mode & 0o077 == 0
    assert validator._verify_materialization(repository, authority["commit"], materialization) == []


def test_process_group_fence_detects_and_kills_descendants(tmp_path: Path) -> None:
    validator = _load_validator()
    code = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )

    exit_code, _stdout, _stderr, timed_out, leaked = validator._run_fenced_command(
        [sys.executable, "-c", code], tmp_path, {}, 5
    )

    assert exit_code == 0
    assert timed_out is False
    assert leaked is True


def test_process_group_fence_kills_descendants_that_ignore_sigterm(tmp_path: Path) -> None:
    validator = _load_validator()
    pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time;from pathlib import Path;"
        f"p=Path({str(pid_path)!r});"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "[(time.sleep(0.01)) for _ in range(200) if not p.exists()]"
    )

    exit_code, _stdout, _stderr, timed_out, leaked = validator._run_fenced_command(
        [sys.executable, "-c", parent_code], tmp_path, {}, 5
    )

    assert exit_code == 0
    assert timed_out is False
    assert leaked is True
    assert pid_path.is_file()
    descendant = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while Path(f"/proc/{descendant}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{descendant}").exists()


def test_producer_receipts_are_closed_content_addressed_records(tmp_path: Path) -> None:
    validator = _load_validator()
    body = {field: "bound" for field in validator.RECEIPT_FIELDS}
    roots = {
        role: {"commit": f"{index + 1:x}" * 40, "tree": f"{index + 6:x}" * 40}
        for index, role in enumerate(validator.EXPECTED_ROLES)
    }
    own_root = roots["accelerate_harness"]
    body.update(
        {
            "schema": validator.PRODUCER_RECEIPT_SCHEMA,
            "role": "accelerate_harness",
            "repository": validator.EXPECTED_REPOSITORIES["accelerate_harness"],
            "commit": own_root["commit"],
            "tree": own_root["tree"],
            "command_index": 0,
            "argv": [sys.executable, "-m", "pytest"],
            "python_executable": sys.executable,
            "python_sha256": "sha256:" + "1" * 64,
            "pytest_sha256": "sha256:" + "2" * 64,
            "environment_policy_sha256": "sha256:" + "3" * 64,
            "materialization_tree": own_root["tree"],
            "closure_tree": own_root["tree"],
            "stdout_sha256": "sha256:" + "4" * 64,
            "stderr_sha256": "sha256:" + "5" * 64,
            "exit_code": 0,
            "timed_out": False,
            "descendant_leak_detected": False,
            "pre_roots": roots,
            "post_roots": copy.deepcopy(roots),
        }
    )
    receipt = validator.seal_producer_receipt(body)

    assert validator.verify_producer_receipt(receipt) is True
    validator._write_receipt(tmp_path, receipt)
    address = receipt["receipt_sha256"].removeprefix("sha256:")
    assert (tmp_path / f"{address}.json").stat().st_mode & 0o077 == 0
    assert validator._receipt_population_errors(tmp_path, [receipt]) == []
    receipt["exit_code"] = 1
    assert validator.verify_producer_receipt(receipt) is False
    receipt = validator.seal_producer_receipt(receipt)
    receipt["unexpected"] = True
    assert validator.verify_producer_receipt(receipt) is False
    malformed = validator.seal_producer_receipt({**body, "exit_code": "0"})
    assert validator.verify_producer_receipt(malformed) is False


def test_all_five_roots_are_revalidated_around_every_command(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    calls: list[tuple[str, ...]] = []
    materializations: list[str] = []
    authorities = {
        role: {
            "role": role,
            "repository": validator.EXPECTED_REPOSITORIES[role],
            "commit": f"{index + 1:x}" * 40,
            "tree": f"{index + 6:x}" * 40,
            "required_test_commands": [["/sealed/python", "-c", "pass"]],
            "test_timeout_seconds": 1,
        }
        for index, role in enumerate(validator.EXPECTED_ROLES)
    }
    repositories = {
        role: tmp_path / f"repo-{index}" for index, role in enumerate(validator.EXPECTED_ROLES)
    }

    def revalidate(_authorities, _repositories):
        calls.append(tuple(validator.EXPECTED_ROLES))
        return []

    monkeypatch.setattr(validator, "_revalidate_all_roots", revalidate)
    monkeypatch.setattr(validator, "_materialize_commit", lambda *_args: None)
    monkeypatch.setattr(validator, "_verify_materialization", lambda *_args: [])
    monkeypatch.setattr(validator, "_test_environment", lambda _path: {})

    def run(_command, cwd, _environment, _timeout):
        materializations.append(str(cwd))
        return 0, b"", b"", False, False

    monkeypatch.setattr(validator, "_run_fenced_command", run)
    toolchain = {
        "python_executable": "/sealed/python",
        "python_sha256": "sha256:" + "a" * 64,
        "pytest_sha256": "sha256:" + "b" * 64,
        "environment_policy": validator.EXPECTED_ENVIRONMENT_POLICY,
    }
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()

    errors, receipts = validator._run_all_required_tests(
        authorities, repositories, toolchain, receipt_dir
    )

    assert errors == []
    assert len(receipts) == len(validator.EXPECTED_ROLES)
    assert len(calls) == 2 * len(validator.EXPECTED_ROLES)
    assert len(set(materializations)) == len(validator.EXPECTED_ROLES)
    assert all(not Path(path).exists() for path in materializations)


def test_ast_audit_rejects_dynamic_identity_and_forged_provider_bodies(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    package = tmp_path / "ipfs_accelerate_py/agent_supervisor/semantic_state"
    package.mkdir(parents=True)
    (package / "bad.py").write_text(
        "import importlib\n"
        "import json as codec\n"
        "from unsealed_wire import ExecutionReceipt as ForeignReceipt\n"
        "WireArtifactEnvelope = dict\n"
        "def WireExecutionEnvelope(value):\n"
        "    return codec.dumps(value, sort_keys=True, separators=(',', ':'))\n"
        "dynamic = importlib.import_module('hashlib')\n"
        "indirect = getattr(importlib, 'import_module')\n"
        "class IpfsDatasetsSemanticStateProvider:\n"
        "    def open_semantic_state(self, root_cid, get_block):\n"
        "        return self._api.open_semantic_state(root_cid, get_block, forged=True)\n",
        encoding="utf-8",
    )

    violations = validator._forbidden_duplicate_authorities(tmp_path)
    assert any("dynamic import" in item for item in violations)
    assert any("local canonicalizer" in item for item in violations)
    assert any("WireExecutionEnvelope" in item for item in violations)
    assert any("WireArtifactEnvelope" in item for item in violations)
    assert any("unsealed module" in item for item in violations)
    assert any("dynamic code/import indirection" in item for item in violations)
    assert any("not direct delegation" in item for item in violations)


def test_required_accelerator_surface_and_fixed_kit_mcp_sources() -> None:
    validator = _load_validator()
    seal = validator.load_seal(SEAL_PATH)
    authorities = {item["role"]: item for item in seal["authorities"]}
    paths = {
        role: {item["path"] for item in authority["required_blobs"]}
        for role, authority in authorities.items()
    }

    assert {
        "ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py",
        "ipfs_accelerate_py/mcp_server/mcplusplus/kubo_cid.py",
    } <= paths["accelerate_harness"]
    assert "ipfs_kit_py/mcp_server/mcplusplus/artifacts.py" in paths["kit_state_roots"]
    mcp_selectors = {
        item["selector"]
        for item in authorities["mcp_plus_plus"]["interface_contract"]["source_extractions"]
    }
    assert "MCPIDLValidator.REQUIRED_DESCRIPTOR_FIELDS" in mcp_selectors
    assert "CIDExecutionValidator.REQUIRED_ENVELOPE_FIELDS" in mcp_selectors
    assert "EventDAGValidator.validate_event.required_fields" in mcp_selectors
    fixed_checkouts = {
        "kit_state_roots": REPO_ROOT.parent / "ipfs-kit-semantic-state-roots",
        "mcp_plus_plus": REPO_ROOT.parents[1] / "Mcp-Plus-Plus",
    }
    for role, checkout in fixed_checkouts.items():
        assert validator._verify_source_extractions(authorities[role], checkout) == []


def test_control_documents_keep_manual_gate_open_and_state_reachability_honestly() -> None:
    plan = PLAN_PATH.read_text(encoding="utf-8")
    todo = TODO_PATH.read_text(encoding="utf-8")
    assert "exact_clean_head" in plan
    assert "does not claim" in plan
    assert "SCH_ISI_CHECKOUT" in plan
    assert "SCH_DSS_CHECKOUT" in plan
    assert "UNRESOLVED_REPAIRED_ACCELERATE_COMMIT" in plan
    assert "live owner without heartbeat" in plan
    assert "newer active task index" in plan
    assert "empty or unavailable process snapshot fails open" in plan
    assert "untracked and submodule outputs" in plan
    assert "fast-zombie birth capture can leak a lease" in plan
    block = todo.split("## SCH-000", 1)[1].split("## SCH-001", 1)[0]
    assert "- Status: todo" in block
    assert "- Completion: manual" in block
    assert "locks/rechecks task-index publication" in block
    assert "isolated index or equivalent materialized proposal" in block
    assert "fails closed on empty/unavailable process snapshots" in block
    assert "cleans up fast-zombie birth-capture leases" in block

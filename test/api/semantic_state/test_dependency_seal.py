"""Control-plane tests for the deliberately unresolved SCH-000 seal."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SEAL_PATH = REPO_ROOT / "config/semantic_state_dependencies.seal.json"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_semantic_state_dependencies.py"
PLAN_PATH = REPO_ROOT / "docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md"
TODO_PATH = REPO_ROOT / "docs/architecture/semantic_compression_harness.todo.md"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "sch_dependency_seal_validator", VALIDATOR_PATH
    )
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
        "required_test_commands": [
            command or ["python3.12", "-c", "raise SystemExit(0)"]
        ],
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
    assert authorities["accelerate_harness"]["commit"] == (
        "ea11293bb996f052d620eae989f5377a956764b1"
    )
    assert authorities["incremental_semantic_index"]["commit"] == (
        "UNRESOLVED_FINAL_ISI_COMMIT"
    )
    assert authorities["semantic_state_contracts"]["commit"] == (
        "UNRESOLVED_FINAL_DSS_COMMIT"
    )
    assert authorities["kit_state_roots"]["commit"] == (
        "05ba9375923cd5fb52e2c9c18b98b530d57d077f"
    )
    assert authorities["mcp_plus_plus"]["commit"] == (
        "dc3164653a48d059ae9812078359daeafb451c07"
    )

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

    for role in ("accelerate_harness", "kit_state_roots", "mcp_plus_plus"):
        authority = authorities[role]
        assert authority["interface_fingerprint"] == (
            validator.authority_fingerprint(authority)
        )


def test_document_contract_rejects_unknown_fields_even_while_unresolved() -> None:
    validator = _load_validator()
    seal = copy.deepcopy(validator.load_seal(SEAL_PATH))
    seal["unexpected_relaxation"] = True

    assert (
        "seal: unknown fields: unexpected_relaxation"
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
        item["role"]: item["interface_contract"]["public_api"]
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
        "open_semantic_state(root_cid:str,"
        "get_block:Callable[[str],bytes])->SemanticStateView"
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
    kit["interface_contract"]["public_api"] = ["SyntheticBypass()"]
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


def test_test_environment_replaces_ambient_import_overrides(
    tmp_path: Path, monkeypatch
) -> None:
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
    repository, authority = _repository_authority(
        tmp_path,
        command=[
            "python3.12",
            "-c",
            "from pathlib import Path; Path('mutation.txt').write_text('x')",
        ],
    )

    errors = validator._run_required_tests(authority, repository)
    assert "checkout[accelerate_harness]: checkout is dirty" in errors


def test_control_documents_keep_manual_gate_open_and_state_reachability_honestly() -> None:
    plan = PLAN_PATH.read_text(encoding="utf-8")
    todo = TODO_PATH.read_text(encoding="utf-8")
    assert "exact_clean_head" in plan
    assert "does not claim" in plan
    assert "SCH_ISI_CHECKOUT" in plan
    assert "SCH_DSS_CHECKOUT" in plan
    block = todo.split("## SCH-000", 1)[1].split("## SCH-001", 1)[0]
    assert "- Status: todo" in block
    assert "- Completion: manual" in block

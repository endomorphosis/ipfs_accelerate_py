"""EAAEF-044: qualify onboarding against supported, unsupported and malicious fixtures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.project_adapters.assessment import (
    assess_repository,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.base import (
    GenericProjectAdapter,
    SupportOutcome,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.python import (
    PythonProjectAdapter,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.security.repository_policy import (
    RepositoryPolicyError,
    admit_repository,
)

MUTATION_CANDIDATE_OUTCOMES = frozenset(
    {"supported_inventory", "mutation_not_admitted"}
)
UNSUPPORTED_OUTCOMES = frozenset(
    {
        "unsupported_language",
        "preview_only",
        "unsupported_build_system",
    }
)
RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "project_onboarding.json"
)
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_agent_project_onboarding.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "evidence_mode",
    "fixture_contracts",
    "live_mutation_invoked",
    "live_runtime_invoked",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "repository_hooks_executed",
    "schema",
    "task_completion_claimed",
    "task_id",
}


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-044"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_project_onboarding_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_mutation_invoked"] is False
    assert payload["repository_hooks_executed"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    sealed = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    sealed["artifact_cid"] = content_identity(sealed)
    _validate_receipt(sealed)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def _python_with_tests(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.1'\n\n"
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        encoding="utf-8",
    )
    (root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_demo.py").write_text(
        "def test_value() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    return root


def _python_without_tests(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'bare'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    (root / "src" / "bare.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def test_supported_python_fixture_is_inventory_not_live_mutation(tmp_path: Path) -> None:
    root = _python_with_tests(tmp_path / "python-ok")
    assessment = assess_repository(root)
    assert assessment["outcome"] == "supported_inventory"
    assert "python" in assessment["languages"]
    assert assessment["mutation_admitted"] is False
    support = GenericProjectAdapter().inspect(root)
    assert support.outcome is SupportOutcome.SUPPORTED_INVENTORY
    assert support.mutation_admitted is False
    adapter = PythonProjectAdapter()
    python_support = adapter.inspect(root)
    assert "python" in python_support.languages
    assert adapter.mutation_admitted(python_support) is False
    missing_tests = assess_repository(_python_without_tests(tmp_path / "python-bare"))
    assert missing_tests["outcome"] in {
        "supported_inventory",
        "mutation_not_admitted",
        "insufficient_validation",
    }
    assert missing_tests["mutation_admitted"] is False


def test_unsupported_language_is_not_mutation_admitted(tmp_path: Path) -> None:
    root = tmp_path / "rust-only"
    root.mkdir()
    (root / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    assessment = assess_repository(root)
    assert assessment["outcome"] in UNSUPPORTED_OUTCOMES
    assert assessment["mutation_admitted"] is False
    support = GenericProjectAdapter().inspect(root)
    assert support.outcome.value in UNSUPPORTED_OUTCOMES
    assert support.mutation_admitted is False
    assert PythonProjectAdapter().mutation_admitted(support) is False


def test_malicious_repository_is_unsafe_and_not_mutation_admitted(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks-root"
    hooks.mkdir()
    (hooks / ".git" / "hooks").mkdir(parents=True)
    (hooks / ".git" / "hooks" / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(RepositoryPolicyError, match="hooks"):
        admit_repository(hooks)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("nope\n", encoding="utf-8")
    escape = tmp_path / "escape-root"
    escape.mkdir()
    os.symlink(outside / "secret", escape / "link")
    with pytest.raises(RepositoryPolicyError, match="symlink"):
        admit_repository(escape)
    escaped = assess_repository(escape)
    assert escaped["outcome"] == "unsafe_repository"
    assert escaped["mutation_admitted"] is False

    secrets = tmp_path / "secrets-root"
    secrets.mkdir()
    (secrets / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (secrets / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    with pytest.raises(RepositoryPolicyError):
        admit_repository(secrets)
    secret_support = GenericProjectAdapter().inspect(secrets)
    assert secret_support.mutation_admitted is False
    assert PythonProjectAdapter().mutation_admitted(secret_support) is False


def test_write_offline_project_onboarding_receipt(tmp_path: Path) -> None:
    supported = assess_repository(_python_with_tests(tmp_path / "supported"))
    assert supported["outcome"] == "supported_inventory"
    assert supported["mutation_admitted"] is False

    rust = tmp_path / "unsupported"
    rust.mkdir()
    (rust / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    unsupported = assess_repository(rust)
    assert unsupported["outcome"] in UNSUPPORTED_OUTCOMES
    assert unsupported["mutation_admitted"] is False

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / ".git" / "hooks").mkdir(parents=True)
    (unsafe / ".git" / "hooks" / "pre-commit").write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )
    with pytest.raises(RepositoryPolicyError, match="hooks"):
        admit_repository(unsafe)

    receipt = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-044",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_project_onboarding_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_mutation_invoked": False,
            "repository_hooks_executed": False,
            "fixture_contracts": {
                "malicious_repository_mutation_allowed": False,
                "supported_python_inventory_only": True,
                "unsupported_language_mutation_allowed": False,
            },
        }
    )
    _validate_receipt(receipt)

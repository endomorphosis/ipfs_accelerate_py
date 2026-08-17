#!/usr/bin/env python3
"""Deterministic LGSWF-003 datasets semantic-producer readiness writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

BINDINGS_REL = (
    "ipfs_datasets_py/logic/software_contracts/semantic_state/contract_bindings.py"
)
TEST_REL = "tests/unit/logic/software_contracts/test_lgswf_semantic_producer.py"

BINDINGS_SOURCE = '''"""Canonical callable/program contract bindings for the datasets producer.

This module is a datasets-owned metadata surface over the existing bindings
owner.  It does not add operational fields, a new index, or accelerator
control-plane types.
"""

from __future__ import annotations

from ipfs_datasets_py.logic.software_contracts.semantic_state.bindings import (
    BindingsError,
    relevant_binding_projection_for_symbol,
)

__all__ = (
    "BindingsError",
    "relevant_binding_projection_for_symbol",
    "OPERATIONAL_FIELDS_FORBIDDEN",
)

OPERATIONAL_FIELDS_FORBIDDEN = (
    "lease_id",
    "fencing_token",
    "claim_id",
    "duckdb_path",
    "quack_endpoint",
    "attempt_id",
)
'''

TEST_SOURCE = '''"""LGSWF-003 producer readiness: scan, contract bindings, no operational fields."""

from __future__ import annotations

from pathlib import Path

from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import (
    scan_repository_state,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state import capsules
from ipfs_datasets_py.logic.software_contracts.semantic_state import contract_bindings


def test_cold_and_incremental_roots_match(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "def answer(value: int) -> int:\\n    return value\\n",
        encoding="utf-8",
    )
    first = scan_repository_state(tmp_path, repository_id="repo:lgswf-003")
    second = scan_repository_state(
        tmp_path, repository_id="repo:lgswf-003", previous_state=first
    )
    assert first.state_cid == second.state_cid


def test_syntax_and_empty_namespace_fail_safely(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\\n", encoding="utf-8")
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    state = scan_repository_state(tmp_path, repository_id="repo:lgswf-003")
    kinds = {item.kind for item in state.artifacts}
    assert "python-analysis" in kinds


def test_contract_bindings_are_datasets_owned_and_non_operational() -> None:
    assert contract_bindings.OPERATIONAL_FIELDS_FORBIDDEN
    assert "lease_id" in contract_bindings.OPERATIONAL_FIELDS_FORBIDDEN
    assert hasattr(contract_bindings, "relevant_binding_projection_for_symbol")
    assert hasattr(capsules, "SEMANTIC_CAPSULE_COMPILER_INTERFACE")
    assert capsules.SEMANTIC_CAPSULE_COMPILER_INTERFACE == "SemanticCapsuleCompiler@1"


def test_adversarial_assurance_stays_unavailable() -> None:
    root = Path(__file__).resolve().parents[4]
    missing = (
        root
        / "ipfs_datasets_py"
        / "logic"
        / "software_contracts"
        / "adversarial_assurance"
    )
    assert not missing.exists()
'''


def write_lgswf_003_producer(workspace: Path) -> dict[str, Any]:
    """Write the missing contract-bindings surface and producer test."""

    root = Path(workspace)
    datasets = root / "ipfs_datasets_py"
    if not datasets.is_dir():
        raise RuntimeError("ipfs_datasets_py worktree is missing")
    bindings = datasets / BINDINGS_REL
    test_path = datasets / TEST_REL
    bindings.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text(BINDINGS_SOURCE, encoding="utf-8")
    test_path.write_text(TEST_SOURCE, encoding="utf-8")
    add = subprocess.run(
        [
            "git",
            "--literal-pathspecs",
            "add",
            "--force",
            "--",
            BINDINGS_REL,
            TEST_REL,
        ],
        cwd=datasets,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "datasets_root": "ipfs_datasets_py",
        "bindings": BINDINGS_REL,
        "test": TEST_REL,
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(write_lgswf_003_producer(Path.cwd()), indent=2, sort_keys=True))

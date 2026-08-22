"""End-to-end trust tests for the hermetic compositional-verification slice."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.compositional_verification_vertical import (
    CompositionalVerificationArtifact,
    VerticalSliceError,
    _analyze_components,
    _build_contract_graph,
    _copy_fixture,
    _fixture_default,
    _run_pytest,
    _test_receipt_id,
    verify_compositional_artifact,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.index import (
    scan_repository,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
    build_semantic_state,
    verify_semantic_state_bundle,
)
from ipfs_datasets_py.logic.verification_api import get_verification_api

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "ipfs_datasets_py"


def _pythonpath() -> str:
    return os.pathsep.join(
        part
        for part in (
            str(REPO_ROOT),
            str(DATASETS_ROOT),
            os.environ.get("PYTHONPATH", ""),
        )
        if part
    )


def test_full_vertical_route_is_deterministic_model_free_and_independently_checked(
    tmp_path: Path,
) -> None:
    """Execute the public route in a fresh interpreter and inspect its receipts."""

    output = tmp_path / "vertical-result.json"
    script = textwrap.dedent(
        """
        import json
        import sys
        from importlib import import_module
        from pathlib import Path

        markers = ("anthropic", "llm_router", "model_provider", "openai")
        initial_modules = set(sys.modules)
        vertical_module = import_module(
            "ipfs_accelerate_py.agent_supervisor.validation."
            "compositional_verification_vertical"
        )
        run_compositional_verification_vertical_slice = (
            vertical_module.run_compositional_verification_vertical_slice
        )

        imported_by_adapter = sorted(
            name
            for name in set(sys.modules) - initial_modules
            if any(marker in name.casefold() for marker in markers)
        )
        assert imported_by_adapter == [], imported_by_adapter
        result = run_compositional_verification_vertical_slice(
            output_path=Path(sys.argv[1])
        )
        imported_by_route = sorted(
            name
            for name in set(sys.modules) - initial_modules
            if any(marker in name.casefold() for marker in markers)
        )
        assert imported_by_route == [], imported_by_route
        assert result["model_invocation_count"] == 0
        assert result["provider_modules_imported_during_route"] == []
        print(json.dumps({"imported": imported_by_route, "status": result["status"]}))
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = _pythonpath()
    completed = subprocess.run(
        (sys.executable, "-c", script, str(output)),
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    guard = json.loads(completed.stdout.strip().splitlines()[-1])
    assert guard == {"imported": [], "status": "completed_hermetic_local"}

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "completed_hermetic_local"
    assert result["model_invocation_count"] == 0
    assert result["provider_modules_imported_during_route"] == []
    assert result["release_qualified"] is False
    assert result["production_authorized"] is False

    assert result["baseline"]["discharge"]["disposition"] == "proved"
    assert result["fault"]["discharge"]["disposition"] == "disproved"
    assert result["fault"]["counterexamples"]
    localization = result["fault"]["localization"]
    assert localization["incremental_solver"]["status"] == "unsat"
    assert localization["incremental_solver"]["core_validated"] is True
    assert localization["incremental_solver"]["unsat_core"] == [
        "consumer-upper-bound",
        "producer-lower-bound",
    ]
    assert localization["interpolant"]["status"] == "validated"

    decisions = {
        item["binding_id"]: item["disposition"]
        for item in result["fault"]["incremental_plan"]["evidence_decisions"]
    }
    assert decisions == {
        "abstract:A": "invalidated",
        "proof:unaffected": "reused",
    }

    synthesis = result["repair"]["repair_synthesis"]
    assert synthesis["disposition"] == "supported"
    assert synthesis["deterministic_zero_model_calls"] is True
    assert synthesis["llm_invocation_count"] == 0
    assert synthesis["model_provider_call_count"] == 0
    assert synthesis["provider_invoked"] is False
    assert synthesis["selected_candidate"]["candidate_id"] == "candidate:constant:10"
    assert result["repair"]["transaction"]["disposition"] == "committed"
    assert result["repair"]["fixed_point"]["complete"] is True

    assert result["final"]["discharge"]["disposition"] == "proved"
    assert result["final"]["selected_tests"]["status"] == "passed"
    assert result["final"]["full_tests"]["status"] == "passed"
    assert result["artifact_verification"]["valid"] is True
    assert result["artifact_verification"]["issues"] == []
    assert result["artifact_verification"]["checks"]["selected_tests_replayed"] is True
    assert result["benchmark"]["challenger"]["abstract_state_reused"] == 1
    assert result["benchmark"]["challenger"]["proof_test_reuse_bps"] == 10_000
    assert result["benchmark"]["comparison"]["safety_floor_violations"] == 0


def test_self_consistent_forged_artifact_is_rejected_by_independent_replay(
    tmp_path: Path,
) -> None:
    """A fresh self-hash cannot turn a false semantic-root claim into evidence."""

    repository = tmp_path / "fixture"
    _copy_fixture(_fixture_default(), repository)
    state = scan_repository(repository)
    semantic_root = verify_semantic_state_bundle(build_semantic_state(state))
    api = get_verification_api(reset=True)
    analyses = _analyze_components(repository, api)
    graph, _contracts = _build_contract_graph(repository, state, analyses, api=api)
    discharge = api.discharge_assume_guarantee(
        graph,
        expected_semantic_state_root=state.state_cid,
        expected_contract_root=graph.contract_root,
    )
    selected_test_receipt = _run_pytest(repository, ("tests/test_selected.py",))
    assert selected_test_receipt["returncode"] == 0
    payload = {
        "final_discharge_receipt_cid": discharge.receipt_cid,
        "selected_test_receipt_cid": _test_receipt_id(selected_test_receipt),
        "semantic_state_root_cid": semantic_root.root_cid,
        "trust_scope": "hermetic_local_fixture_only",
    }

    artifact = CompositionalVerificationArtifact(payload)
    replay = verify_compositional_artifact(
        artifact,
        worktree=repository,
        expected_state=state,
        expected_graph=graph,
    )
    assert replay.valid

    with pytest.raises(VerticalSliceError, match="content identity mismatch"):
        CompositionalVerificationArtifact(payload, artifact_cid="forged:producer-claim")

    forged_payload = dict(payload)
    forged_payload["semantic_state_root_cid"] = "sha256:" + "0" * 64
    self_consistent_forgery = CompositionalVerificationArtifact(forged_payload)
    rejected = verify_compositional_artifact(
        self_consistent_forgery,
        worktree=repository,
        expected_state=state,
        expected_graph=graph,
    )
    assert not rejected.valid
    assert rejected.disposition == "rejected"
    assert "semantic_state_root_mismatch" in rejected.issues


def test_compositional_public_api_preserves_hammer_import_isolation_contract() -> None:
    """Regression: additive facade imports keep the existing Hammer loader lazy."""

    script = textwrap.dedent(
        """
        import os
        import sys

        original_home = os.environ.get("HOME")
        original_prefix = sys.prefix
        import ipfs_datasets_py.logic.verification_api
        from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
            HAMMER_IMPORT_ISOLATION,
            HAMMER_IMPORT_ISOLATION_HARDENED,
            get_isolated_hammer_loader,
        )

        loader = get_isolated_hammer_loader()
        report = loader.isolation_report()
        assert HAMMER_IMPORT_ISOLATION == HAMMER_IMPORT_ISOLATION_HARDENED
        assert report["import_isolation"] == HAMMER_IMPORT_ISOLATION_HARDENED
        assert report["concurrency_safe"] is True
        assert report["mutates_home"] is False
        assert report["mutates_sys_prefix"] is False
        assert report["process_global"] is False
        assert "ipfs_datasets_py.logic.hammers" not in sys.modules
        assert os.environ.get("HOME") == original_home
        assert sys.prefix == original_prefix
        print("ok")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = _pythonpath()
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "ok"

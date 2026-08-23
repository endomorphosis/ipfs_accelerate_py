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
    REQUIRED_VERTICAL_STAGES,
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
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]
    guard = json.loads(completed.stdout.strip().splitlines()[-1])
    assert guard == {"imported": [], "status": "completed_hermetic_local"}

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "completed_hermetic_local"
    assert result["model_invocation_count"] == 0
    assert result["provider_modules_imported_during_route"] == []
    assert result["release_qualified"] is False
    assert result["production_authorized"] is False
    assert tuple(REQUIRED_VERTICAL_STAGES) == (
        "identity",
        "scan",
        "abstract_states",
        "contracts",
        "initial_discharge",
        "incremental_smt",
        "counterexample",
        "unsat_core",
        "interpolant",
        "capsules",
        "context",
        "mutation",
        "exact_invalidation",
        "unaffected_reuse",
        "deterministic_repair",
        "affected_only_replay",
        "live_fixed_point",
        "independently_verified_artifact",
        "final_context",
        "zero_model_calls",
        "token_metrics",
        "work_reuse_metrics",
    )
    assert len(REQUIRED_VERTICAL_STAGES) == 22
    assert result["required_stages"] == list(REQUIRED_VERTICAL_STAGES)
    assert set(result["stages"]) == set(REQUIRED_VERTICAL_STAGES)
    for stage_id in REQUIRED_VERTICAL_STAGES:
        stage = result["stages"][stage_id]
        assert stage["stage_id"] == stage_id
        assert stage["status"] == "completed", stage_id

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
    assert result["rollback"]["restored_fault_bytes"] is True
    assert result["rollback"]["original_repository_unmutated"] is True
    assert result["rollback"]["receipt_cid"]
    assert result["stages"]["mutation"]["after_hash"] != result["stages"]["mutation"]["before_hash"]
    assert result["stages"]["deterministic_repair"]["rollback_receipt_cid"] == (
        result["rollback"]["receipt_cid"]
    )

    assert result["final"]["discharge"]["disposition"] == "proved"
    assert result["final"]["selected_tests"]["status"] == "passed"
    assert result["final"]["full_tests"]["status"] == "passed"
    assert result["final"]["context"]["residual_disposition"] == "deterministic_closed"
    assert result["final"]["context"]["input_tokens"] > 0
    assert result["artifact_verification"]["valid"] is True
    assert result["artifact_verification"]["issues"] == []
    assert result["artifact_verification"]["checks"]["selected_tests_replayed"] is True
    assert result["benchmark"]["challenger"]["abstract_state_reused"] == 1
    assert result["benchmark"]["challenger"]["proof_test_reuse_bps"] == 10_000
    assert result["benchmark"]["challenger"]["context_tokens"] == result["final"]["context"]["input_tokens"]
    assert result["benchmark"]["comparison"]["safety_floor_violations"] == 0
    assert result["stages"]["token_metrics"]["context_tokens"] == (
        result["final"]["context"]["input_tokens"]
    )
    assert result["stages"]["work_reuse_metrics"]["proof_test_reuse_bps"] == 10_000
    assert result["stages"]["zero_model_calls"]["model_invocation_count"] == 0


def test_fixture_recipe_is_compact_and_matches_source_bytes() -> None:
    """The hermetic fixture is a recipe, not a golden envelope dump."""

    root = _fixture_default()
    recipe_path = root / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["schema"] == "lgcvf-compositional-verification-fixture-recipe@1"
    producer = (root / "pkg/module_a.py").read_text(encoding="utf-8")
    consumer = (root / "pkg/module_b.py").read_text(encoding="utf-8")
    present = (root / "pkg/module_c.py").read_text(encoding="utf-8")
    unaffected = (root / "pkg/unaffected.py").read_text(encoding="utf-8")
    assert f"return {recipe['baseline']['producer_return']}" in producer
    assert "MAX_PRODUCED_VALUE" in consumer
    assert "from .module_b import consume" in present
    assert 'return "unaffected"' in unaffected
    selected = (root / "tests/test_selected.py").read_text(encoding="utf-8")
    unselected = (root / "tests/test_unselected.py").read_text(encoding="utf-8")
    assert "test_selected_contract_path" in selected
    assert "stable_label" in unselected
    assert recipe["fault"]["path"] == "pkg/module_a.py"
    assert recipe["repair"]["candidate_id"] == "candidate:constant:10"


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
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-400:]
    assert completed.stdout.strip() == "ok"

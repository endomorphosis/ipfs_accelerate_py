"""Semantic certification of finite-trace Runtime MTL (FVT-039 / FVT-G103).

Exercises ``tools/logic/certification/runtime_mtl.py`` and the Runtime MTL
corpus fixture.

Acceptance covered:

* live satisfied and violated traces;
* interval and event mutations change the verdict;
* shortest violating-prefix discovery and deterministic replay;
* closed vs open timestamp boundaries;
* malformed traces fail closed;
* clean finite prefixes never become theorems;
* receipts bind formula, trace, clock policy, bounds, implementation, and
  source tree;
* Python/TypeScript golden parity (when the TS package is available);
* resulting authority is finite-trace only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl.py"
MANIFEST_PATH = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "formal_verification"
    / "toolchains"
    / "runtime_mtl"
    / "manifest.json"
)
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
CENTRAL_CERTIFIER = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)

INTERFACE = "RuntimeMTLSemanticCertification@1"
SCHEMA_VERSION = "runtime-mtl-semantic-certification/v1"
MANIFEST_SCHEMA = "runtime-mtl-semantic-corpus/v1"
GOAL_ID = "FVT-G103"
TASK_ID = "FVT-039"
LANE_ID = "runtime_mtl"
HANDLER_ID = "runtime_mtl_semantic_certification@1"
TOOL_ID = "runtime-mtl"
AUTHORITY_CEILING = "finite_trace"

REQUIRED_CATEGORIES = {
    "satisfied",
    "violated",
    "interval_mutation",
    "event_mutation",
    "shortest_violating_prefix",
    "timestamp_boundary",
    "malformed",
    "clean_prefix",
    "parity",
}
REQUIRED_MUTATIONS = {"interval", "event"}


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_module(CERTIFIER_PATH, "tools_logic_certification_runtime_mtl")


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing corpus manifest: {MANIFEST_PATH}"
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def certificate(certifier, manifest) -> dict[str, Any]:
    return certifier.certify_runtime_mtl_semantics(
        manifest=manifest,
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Expected outputs / fixture contract
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    assert Path(__file__).is_file()


def test_certifier_interface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.LANE_ID == LANE_ID
    assert certifier.HANDLER_ID == HANDLER_ID
    assert certifier.TOOL_ID == TOOL_ID
    assert certifier.AUTHORITY_CEILING == AUTHORITY_CEILING
    assert certifier.AUTHORITY_SCOPE == "finite_trace_monitor_only"
    assert certifier.CERTIFICATION_SURFACE == "tools.logic.certification.runtime_mtl"


def test_manifest_schema_and_recipes(manifest: dict[str, Any]) -> None:
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["tool_id"] == TOOL_ID
    assert manifest["lane_id"] == LANE_ID
    assert manifest["handler_id"] == HANDLER_ID
    assert manifest["authority_ceiling"] == AUTHORITY_CEILING
    assert manifest["forbids_theorem_authority"] is True
    assert set(manifest["required_categories"]) == REQUIRED_CATEGORIES
    assert set(manifest["required_mutation_kinds"]) == REQUIRED_MUTATIONS

    policy = manifest["policy"]
    assert policy["in_process_only"] is True
    assert policy["no_external_parity_install"] is True
    assert policy["no_central_certificate_edit"] is True
    assert policy["finite_trace_authority_only"] is True
    assert policy["clean_prefix_never_theorem"] is True
    assert policy["shortest_violating_prefix_replay"] is True
    assert policy["python_typescript_golden_parity"] is True

    recipes = manifest["case_recipes"]
    assert isinstance(recipes, list) and recipes
    categories = {item["category"] for item in recipes}
    assert REQUIRED_CATEGORIES <= categories
    mutation_kinds = {
        item["mutation_kind"]
        for item in recipes
        if item.get("mutation_kind")
    }
    assert REQUIRED_MUTATIONS <= mutation_kinds
    # Compact recipes: no bulk golden dumps.
    for item in recipes:
        assert "formula" not in item
        assert "trace" not in item
        assert "events" not in item
        assert item["recipe"]


# ---------------------------------------------------------------------------
# Full semantic certification
# ---------------------------------------------------------------------------


def test_runtime_mtl_is_semantically_certified(certificate: dict[str, Any]) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["lane_id"] == LANE_ID
    assert certificate["tool_id"] == TOOL_ID
    assert certificate["certified"] is True
    assert certificate["production_certified"] is True
    assert certificate["authority_ceiling"] == AUTHORITY_CEILING
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    assert certificate["policy"]["grants_finite_trace_authority"] is True
    assert certificate["block_reasons"] == []
    assert certificate["checks"]
    assert all(
        check["status"] in {"passed", "skipped"} for check in certificate["checks"]
    )
    assert all(check["authorizes_global_proof"] is False for check in certificate["checks"])


def test_required_categories_exercised(certificate: dict[str, Any]) -> None:
    categories = set(certificate["categories_exercised"])
    assert REQUIRED_CATEGORIES <= categories
    case_ids = {item["case_id"] for item in certificate["case_results"]}
    for category in REQUIRED_CATEGORIES:
        assert any(
            category in case_id
            or case_id.endswith(category)
            or f".{category}" in case_id
            or f":{category}" in case_id
            or any(r.get("category") == category for r in certificate["case_results"])
            for case_id in case_ids
        ), (category, sorted(case_ids))


def test_satisfied_and_violated_live_traces(certificate: dict[str, Any]) -> None:
    statuses = {item["status"] for item in certificate["case_results"]}
    assert "satisfied" in statuses
    assert "violated" in statuses
    for item in certificate["case_results"]:
        if item["status"] in {"satisfied", "violated", "unknown", "malformed"}:
            assert item["authority"] == "monitor"
            assert item["authorizes_global_proof"] is False


def test_interval_and_event_mutations_change_verdict(certifier, certificate) -> None:
    mutation_checks = [
        check
        for check in certificate["checks"]
        if check["kind"] == "mutation"
    ]
    assert len(mutation_checks) >= 2
    assert all(check["status"] == "passed" for check in mutation_checks)

    for mutation_kind in REQUIRED_MUTATIONS:
        specs = [
            spec
            for spec in certifier.default_case_specs()
            if spec.mutation_kind == mutation_kind
        ]
        assert specs, mutation_kind
        for spec in specs:
            base = certifier._golden_by_id(spec.base_fixture_id)
            baseline = certifier.run_case(
                {
                    "case_id": f"{spec.case_id}:baseline",
                    "category": "baseline",
                    "formula": base["formula"],
                    "trace": base["trace"],
                    "position": base.get("position", 0),
                }
            )
            case = certifier.materialize_case(spec)
            mutated = certifier.run_case(case)
            assert (
                mutated.status != baseline.status
                or mutated.verdict != baseline.verdict
            ), (mutation_kind, baseline.status, mutated.status)
            assert mutated.status == spec.expected_status
            assert mutated.verdict == spec.expected_verdict
            assert (
                mutated.formula_digest != baseline.formula_digest
                or mutated.trace_digest != baseline.trace_digest
            )
            assert mutated.authorizes_global_proof is False


def test_shortest_violating_prefix_replay(certifier) -> None:
    base = certifier._golden_by_id("prefix-always-violation")
    prefix, length, record = certifier.shortest_violating_prefix(
        base["formula"],
        base["trace"],
        position=int(base.get("position", 0)),
    )
    assert prefix is not None
    assert length is not None and length >= 1
    assert record is not None
    assert record.status == "violated"
    assert record.verdict == "false"
    assert record.shortest_prefix_length == length

    replay = certifier.run_case(
        {
            "case_id": "shortest:replay",
            "category": "shortest_violating_prefix",
            "formula": base["formula"],
            "trace": prefix,
            "position": base.get("position", 0),
        }
    )
    assert replay.status == record.status
    assert replay.verdict == record.verdict
    assert replay.result_digest == record.result_digest


def test_timestamp_boundaries_closed_vs_open(certifier) -> None:
    closed = certifier._golden_by_id("mtl-closed-interval-includes-boundary")
    open_upper = certifier._golden_by_id("mtl-open-upper-excludes-boundary")
    closed_result = certifier.run_case(
        {
            "case_id": "boundary:closed",
            "category": "timestamp_boundary",
            "formula": closed["formula"],
            "trace": closed["trace"],
        }
    )
    open_result = certifier.run_case(
        {
            "case_id": "boundary:open",
            "category": "timestamp_boundary",
            "formula": open_upper["formula"],
            "trace": open_upper["trace"],
        }
    )
    assert closed_result.status == "satisfied"
    assert closed_result.verdict == "true"
    assert open_result.status == "violated"
    assert open_result.verdict == "false"
    # Same timed word; only interval openness differs.
    assert closed_result.trace_digest == open_result.trace_digest
    assert closed_result.formula_digest != open_result.formula_digest


def test_malformed_traces_fail_closed(certifier) -> None:
    late = certifier._golden_by_id("late-event-malformed")
    result = certifier.run_case(
        {
            "case_id": "malformed:late",
            "category": "malformed",
            "formula": late["formula"],
            "trace": late["trace"],
        }
    )
    assert result.status == "malformed"
    assert result.late_events is True
    assert result.authority == "monitor"
    assert result.authorizes_global_proof is False
    assert result.verdict == "inconclusive"


def test_clean_prefix_never_becomes_theorem(certifier) -> None:
    case = certifier._golden_by_id("prefix-always-inconclusive")
    result = certifier.run_case(
        {
            "case_id": "clean:prefix",
            "category": "clean_prefix",
            "formula": case["formula"],
            "trace": case["trace"],
        }
    )
    assert result.status == "unknown"
    assert result.verdict == "inconclusive"
    assert result.authority == "monitor"
    assert result.authorizes_global_proof is False
    # Direct portable evaluation also refuses theorem elevation.
    evaluation = certifier.evaluate_portable(case["formula"], case["trace"])
    assert evaluation.authorizes_global_proof is False
    assert evaluation.status.value == "unknown"


def test_inconclusive_prefix_stays_unknown(certifier) -> None:
    for fixture_id in (
        "prefix-always-inconclusive",
        "mtl-prefix-before-horizon-inconclusive",
        "explicit-missing-atom-inconclusive",
    ):
        case = certifier._golden_by_id(fixture_id)
        result = certifier.run_case(
            {
                "case_id": f"unknown:{fixture_id}",
                "category": "clean_prefix",
                "formula": case["formula"],
                "trace": case["trace"],
            }
        )
        assert result.status == "unknown", fixture_id
        assert result.verdict == "inconclusive", fixture_id
        assert result.authorizes_global_proof is False


def test_receipts_bind_formula_trace_clock_bounds_implementation(
    certificate: dict[str, Any],
) -> None:
    bindings = certificate["bindings"]
    assert bindings["formula"] is True
    assert bindings["trace"] is True
    assert bindings["clock_policy"] is True
    assert bindings["bounds"] is True
    assert bindings["implementation"] is True
    assert bindings["source_tree"] is True

    impl = certificate["implementation"]
    assert impl["exists"] is True
    assert impl["content_sha256"]
    assert len(impl["content_sha256"]) == 64
    assert impl["module"]
    assert impl["interface"] == "RuntimeMTLMonitor@1"

    source_tree = certificate["source_tree"]
    assert source_tree["tree_digest_sha256"]
    assert len(source_tree["tree_digest_sha256"]) == 64

    for record in certificate["case_results"]:
        if record["category"] == "parity":
            continue
        assert record["formula_digest"]
        assert record["trace_digest"]
        assert len(record["formula_digest"]) == 64
        assert len(record["trace_digest"]) == 64
        if record["status"] != "malformed":
            assert record["clock_policy_digest"]
            assert record["bounds_digest"]
            assert record["result_digest"]
        assert record["authority"] == "monitor"
        assert record["authorizes_global_proof"] is False


def test_python_typescript_golden_parity(certifier) -> None:
    check, detail = certifier.run_python_typescript_parity(repo_root=REPO_ROOT)
    assert check.kind == "parity"
    assert check.status in {"passed", "skipped"}
    if check.status == "passed":
        assert detail["compared_cases"] > 0
        assert detail["mismatches"] == []


def test_golden_fixtures_self_check(certifier) -> None:
    from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (
        evaluate_case,
        golden_fixtures,
    )

    for case in golden_fixtures():
        result = evaluate_case(
            {
                "formula": case["formula"],
                "trace": case["trace"],
                "position": case.get("position", 0),
            }
        )
        expected = case["expected"]
        for key, value in expected.items():
            assert result[key] == value, (case["case_id"], key, result.get(key), value)
        assert result["authority"] == "monitor"
        assert result["authorizes_global_proof"] is False


def test_certificate_digest_is_stable(certifier) -> None:
    first = certifier.certify_runtime_mtl_semantics(
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )
    second = certifier.certify_runtime_mtl_semantics(
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )
    assert first["certificate_digest_sha256"] == second["certificate_digest_sha256"]
    assert first["certified"] is True
    assert second["certified"] is True
    assert len(first["certificate_digest_sha256"]) == 64


def test_lane_handler_reports_certified(certifier) -> None:
    result = certifier.runtime_mtl_lane_handler(repo_root=REPO_ROOT)
    assert result["lane_id"] == LANE_ID
    assert result["handler_id"] == HANDLER_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["authority_ceiling"] == AUTHORITY_CEILING
    assert result["grants_theorem_authority"] is False
    assert result["grants_finite_trace_authority"] is True
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64

    alias = certifier.lane_handler(repo_root=REPO_ROOT)
    assert alias["certified"] is True
    assert alias["handler_id"] == HANDLER_ID


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    certifier,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface missing")

    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    bound = certifier.bind_runtime_mtl_lane(policy, replace=True)
    handler = bound.get_lane_handler(LANE_ID)
    assert handler is not None
    result = handler(repo_root=REPO_ROOT)
    assert result["certified"] is True
    assert result["handler_id"] == HANDLER_ID
    assert result["grants_theorem_authority"] is False

    # Central multi-prover certifier must remain untouched by this lane.
    assert CENTRAL_CERTIFIER.is_file()
    text = CENTRAL_CERTIFIER.read_text(encoding="utf-8")
    # The central certifier may *reference* runtime-mtl as usable, but this
    # semantic lane must not rewrite it. Guard: file is readable and still a
    # multi-prover orchestrator (not replaced by the Runtime MTL certifier).
    assert "build_certificate" in text or "FormalVerification" in text
    assert "RuntimeMTLSemanticCertification@1" not in text


def test_policy_forbids_external_install_and_central_certificate_edit(
    certificate: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    for policy in (certificate["policy"], manifest["policy"]):
        assert policy["no_external_parity_install"] is True
        assert policy["no_central_certificate_edit"] is True
        assert policy["in_process_only"] is True
        assert policy.get("grants_theorem_authority", False) is False

#!/usr/bin/env python3
"""Build the controlled semantic-capsule selection-evaluation corpus.

Recipes stay compact. Re-run this script after editing RECIPES to refresh
``corpus_manifest.json``.

This corpus is measurement evidence for IVP-015 only. Seeded outcomes never
grant production verification authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "incremental-verification-selection-corpus@1"
)
CORPUS_ID = "ivp-semantic-capsule-controlled-v1"
DESCRIPTION = (
    "Hermetic controlled semantic-capsule fixture recipes for "
    "selected-versus-full-suite differential evaluation. Ground-truth "
    "affected tests and full-suite failure oracles are reviewed seeds; "
    "observations never assert target success or production acceptance."
)

# Shared node ids for the mini catalog.
TEST_A = "tests/test_mod.py::test_fn"
TEST_B = "tests/test_mod.py::test_helper"
TEST_C = "tests/test_other.py::test_unrelated"
TEST_D = "tests/test_config.py::test_configured"
TEST_E = "tests/test_fixture.py::test_with_fixture"
TEST_FAIL = "tests/test_mod.py::test_deliberately_fails"

ALL_TESTS = [TEST_A, TEST_B, TEST_C, TEST_D, TEST_E, TEST_FAIL]

POLICY_ID = "policy:ivp-selection-eval@1"
ENV_ID = "env:ivp-hermetic@1"
LOCK_ID = "lock:ivp-requirements@1"
REPO_ID = "repository:ivp-controlled-fixture@1"
TREE_ID = "tree:ivp-controlled-semantic@1"


def _snapshot(fixture_id: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "tree_id": TREE_ID,
        "environment_id": ENV_ID,
        "lock_id": LOCK_ID,
        "fixture_id": fixture_id,
        "policy_id": POLICY_ID,
        "repository_id": REPO_ID,
    }
    payload.update(overrides)
    return payload


def _edge(
    source: str,
    target: str,
    kind: str,
    *,
    disposition: str = "exact",
    truncated: bool = False,
    opaque: bool = False,
    uncovered: bool = False,
    critical: bool = True,
    edge_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source,
        "target": target,
        "kind": kind,
        "disposition": disposition,
        "truncated": truncated,
        "opaque": opaque,
        "uncovered": uncovered,
        "critical": critical,
    }
    if edge_id is not None:
        payload["edge_id"] = edge_id
    return payload


def _catalog() -> dict[str, Any]:
    return {
        "tests": list(ALL_TESTS),
        "static_checks": [],
        "type_checks": [],
        "proof_obligations": [],
    }


def _policy(**kwargs: Any) -> dict[str, Any]:
    base = {
        "policy_id": POLICY_ID,
        "critical_uncertainty_requires_full_suite": False,
    }
    base.update(kwargs)
    return base


def _obs(
    mode: str,
    fixture_id: str,
    outcomes: dict[str, str],
    *,
    suite_status: str = "completed",
    duration_ms: int = 10,
    reason_codes: list[str] | None = None,
    order: list[str] | None = None,
    snapshot_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snap = _snapshot(fixture_id, **(snapshot_overrides or {}))
    return {
        "mode": mode,
        "snapshot": snap,
        "suite_status": suite_status,
        "test_outcomes": dict(outcomes),
        "test_order": list(order or sorted(outcomes)),
        "duration_ms": duration_ms,
        "wall_time_ms": duration_ms,
        "reason_codes": list(reason_codes or []),
        "selector_identity": f"selector:{fixture_id}:{mode}",
    }


def _pass_all(nodes: list[str]) -> dict[str, str]:
    return {node: "passed" for node in nodes}


def _recipe(
    fixture_id: str,
    change_kind: str,
    *,
    ground_truth: list[str],
    changed_symbols: list[str] | None = None,
    changed_paths: list[str] | None = None,
    edges: list[dict[str, Any]] | None = None,
    selected_outcomes: dict[str, str] | None = None,
    full_outcomes: dict[str, str] | None = None,
    selected_status: str = "completed",
    full_status: str = "completed",
    forced_selected: list[str] | None = None,
    equivalence_label: str = "",
    corpus_present: bool = True,
    validation_selection: dict[str, Any] | None = None,
    uncovered_symbols: list[str] | None = None,
    truncated: bool = False,
    requires_broader: bool = False,
    policy: dict[str, Any] | None = None,
    description: str = "",
    selected_reasons: list[str] | None = None,
    full_reasons: list[str] | None = None,
    selected_duration_ms: int = 5,
    full_duration_ms: int = 20,
    selected_order: list[str] | None = None,
    full_order: list[str] | None = None,
) -> dict[str, Any]:
    gt = list(ground_truth)
    selected_nodes = list(
        forced_selected
        if forced_selected is not None
        else gt
    )
    sel_out = selected_outcomes
    if sel_out is None:
        sel_out = _pass_all(selected_nodes)
    full_out = full_outcomes
    if full_out is None:
        full_out = _pass_all(ALL_TESTS)
        # Mark ground-truth and selected as present.
        for node in selected_nodes:
            full_out.setdefault(node, "passed")

    payload: dict[str, Any] = {
        "fixture_id": fixture_id,
        "change_kind": change_kind,
        "snapshot": _snapshot(fixture_id),
        "ground_truth_affected_tests": gt,
        "all_tests": list(ALL_TESTS),
        "changed_symbols": list(changed_symbols or []),
        "changed_paths": list(changed_paths or []),
        "edges": list(edges or []),
        "catalog": _catalog(),
        "policy": policy if policy is not None else _policy(),
        "validation_selection": validation_selection,
        "uncovered_symbols": list(uncovered_symbols or []),
        "uncovered_paths": [],
        "truncated": truncated,
        "requires_broader_selection": requires_broader,
        "equivalence_label": equivalence_label,
        "corpus_id": CORPUS_ID,
        "corpus_present": corpus_present,
        "description": description,
        "forced_selected_tests": forced_selected,
        "selected_observation": _obs(
            "selected",
            fixture_id,
            sel_out,
            suite_status=selected_status,
            duration_ms=selected_duration_ms,
            reason_codes=selected_reasons,
            order=selected_order,
        ),
        "full_suite_observation": _obs(
            "full_suite",
            fixture_id,
            full_out,
            suite_status=full_status,
            duration_ms=full_duration_ms,
            reason_codes=full_reasons,
            order=full_order,
        ),
    }
    return payload


def _build_recipes() -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []

    # 1. Direct symbol change — exact selection, zero FN/FP.
    recipes.append(
        _recipe(
            "direct-symbol-fn",
            "direct_symbol",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            full_outcomes={
                **_pass_all(ALL_TESTS),
            },
            description="Direct symbol change selects the exact tested_by edge.",
        )
    )

    # 2. Transitive dependency change.
    recipes.append(
        _recipe(
            "transitive-helper",
            "transitive",
            ground_truth=[TEST_A, TEST_B],
            changed_symbols=["pkg.mod.Helper"],
            edges=[
                _edge("pkg.mod.fn", "pkg.mod.Helper", "depends_on"),
                _edge("pkg.mod.fn", TEST_A, "tested_by"),
                _edge("pkg.mod.Helper", TEST_B, "tested_by"),
            ],
            description="Transitive depends_on expands the cone to both tests.",
        )
    )

    # 3. Unrelated edit — empty exact selection.
    recipes.append(
        _recipe(
            "unrelated-edit",
            "unrelated",
            ground_truth=[],
            changed_symbols=["pkg.other.g"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            forced_selected=[],
            description="Unrelated symbol does not expand selection.",
        )
    )

    # 4. Fixture edge change.
    recipes.append(
        _recipe(
            "fixture-edge-change",
            "fixture_edge",
            ground_truth=[TEST_E],
            changed_symbols=["fixtures.sample_data"],
            edges=[
                _edge("fixtures.sample_data", TEST_E, "fixtures"),
            ],
            description="Fixture edge selects tests depending on the fixture.",
        )
    )

    # 5. Config edge change.
    recipes.append(
        _recipe(
            "config-edge-change",
            "config_edge",
            ground_truth=[TEST_D],
            changed_paths=["config/settings.toml"],
            edges=[
                _edge("config/settings.toml", TEST_D, "configures"),
            ],
            description="Config edge selects configured tests.",
        )
    )

    # 6. Environment identity change (still measured under fresh equal snaps).
    recipes.append(
        _recipe(
            "environment-identity",
            "environment",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            description=(
                "Environment-bound snapshot; selected and full share identity."
            ),
        )
    )

    # 7. Lock identity change.
    recipes.append(
        _recipe(
            "lock-identity",
            "lock",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            description="Lock-bound snapshot; selected and full share identity.",
        )
    )

    # 8. Opaque critical edge — broader required.
    recipes.append(
        _recipe(
            "opaque-critical",
            "opaque",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[
                _edge("pkg.mod.fn", TEST_A, "tested_by"),
                _edge(
                    "pkg.mod.fn",
                    "dynamic.target",
                    "opaque",
                    opaque=True,
                    disposition="opaque",
                ),
            ],
            requires_broader=True,
            description="Opaque critical edge forces broader suite before acceptance.",
        )
    )

    # 9. Dynamic critical edge.
    recipes.append(
        _recipe(
            "dynamic-critical",
            "dynamic",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[
                _edge("pkg.mod.fn", TEST_A, "tested_by"),
                _edge(
                    "pkg.mod.fn",
                    "runtime.target",
                    "dynamic",
                    disposition="conservative",
                ),
            ],
            requires_broader=True,
            description="Dynamic critical edge forces broader suite before acceptance.",
        )
    )

    # 10. Deliberately failing test — selected observes the failure.
    recipes.append(
        _recipe(
            "deliberately-failing-observed",
            "deliberately_failing",
            ground_truth=[TEST_FAIL],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_FAIL, "tested_by")],
            selected_outcomes={TEST_FAIL: "failed"},
            full_outcomes={
                **_pass_all([TEST_A, TEST_B, TEST_C, TEST_D, TEST_E]),
                TEST_FAIL: "failed",
            },
            description=(
                "Mutation-caused failure is selected and observed; not a FN."
            ),
        )
    )

    # 11. Seeded false negative — GT affected omitted from selection.
    recipes.append(
        _recipe(
            "seeded-false-negative",
            "false_negative_seed",
            ground_truth=[TEST_A, TEST_FAIL],
            changed_symbols=["pkg.mod.fn"],
            edges=[
                _edge("pkg.mod.fn", TEST_A, "tested_by"),
                _edge("pkg.mod.fn", TEST_FAIL, "tested_by"),
            ],
            forced_selected=[TEST_A],
            selected_outcomes={TEST_A: "passed"},
            full_outcomes={
                **_pass_all([TEST_A, TEST_B, TEST_C, TEST_D, TEST_E]),
                TEST_FAIL: "failed",
            },
            description=(
                "Ground-truth affected TEST_FAIL omitted; full suite fails it."
            ),
        )
    )

    # 12. Seeded false positive — selected outside ground truth.
    recipes.append(
        _recipe(
            "seeded-false-positive",
            "false_positive_seed",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            forced_selected=[TEST_A, TEST_C],
            selected_outcomes={TEST_A: "passed", TEST_C: "passed"},
            full_outcomes=_pass_all(ALL_TESTS),
            description=(
                "Selected unrelated TEST_C is a false positive; passing is not."
            ),
        )
    )

    # 13. Flaky outcome discrepancy — inconclusive.
    recipes.append(
        _recipe(
            "flaky-outcome",
            "flaky",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            selected_outcomes={TEST_A: "flaky"},
            full_outcomes={**_pass_all(ALL_TESTS), TEST_A: "flaky"},
            description="Flaky selected/full outcomes are inconclusive.",
        )
    )

    # 14. Order-dependent outcome — inconclusive.
    recipes.append(
        _recipe(
            "order-dependent",
            "order_dependent",
            ground_truth=[TEST_A, TEST_B],
            changed_symbols=["pkg.mod.fn"],
            edges=[
                _edge("pkg.mod.fn", TEST_A, "tested_by"),
                _edge("pkg.mod.fn", TEST_B, "tested_by"),
            ],
            forced_selected=[TEST_A, TEST_B],
            selected_outcomes={TEST_A: "passed", TEST_B: "order_dependent"},
            full_outcomes={
                **_pass_all(ALL_TESTS),
                TEST_B: "order_dependent",
            },
            selected_order=[TEST_A, TEST_B],
            full_order=[TEST_B, TEST_A],
            selected_reasons=["order_dependent"],
            full_reasons=["order_dependent"],
            description="Order-dependent outcomes are inconclusive.",
        )
    )

    # 15. Full-suite timeout — not_measured.
    recipes.append(
        _recipe(
            "full-suite-timeout",
            "full_suite_timeout",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            full_status="timeout",
            full_outcomes={},
            description="Full-suite timeout is not_measured, never zero FN.",
        )
    )

    # 16. Full-suite unavailable — not_measured.
    recipes.append(
        _recipe(
            "full-suite-unavailable",
            "full_suite_unavailable",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            full_status="unavailable",
            full_outcomes={},
            description="Full-suite unavailable is not_measured, never zero FN.",
        )
    )

    # 17. Missing validation-ID mapping — broader required.
    recipes.append(
        _recipe(
            "validation-mapping-incomplete",
            "validation_mapping",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            validation_selection={
                "validation_ids": ["val:a", "val:missing"],
                "mapped_pytest_node_ids": [TEST_A],
                "unmapped_validation_ids": ["val:missing"],
                "requires_broader_selection": True,
            },
            description=(
                "Missing validation-ID-to-node-ID mapping requires broader suite."
            ),
        )
    )

    # 18/19. Equivalent controlled fixtures with distinct labels.
    recipes.append(
        _recipe(
            "equivalent-controlled-a",
            "equivalent_controlled",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            equivalence_label="equivalent-pair:mod-fn@1",
            description="Equivalent controlled fixture A (distinct label).",
        )
    )
    recipes.append(
        _recipe(
            "equivalent-controlled-b",
            "equivalent_controlled",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            equivalence_label="equivalent-pair:mod-fn@1",
            description="Equivalent controlled fixture B (same pair, distinct id).",
        )
    )

    # 20. Outcome discrepancy (selected pass / full fail) — inconclusive.
    recipes.append(
        _recipe(
            "outcome-discrepancy",
            "deliberately_failing",
            ground_truth=[TEST_A],
            changed_symbols=["pkg.mod.fn"],
            edges=[_edge("pkg.mod.fn", TEST_A, "tested_by")],
            selected_outcomes={TEST_A: "passed"},
            full_outcomes={**_pass_all(ALL_TESTS), TEST_A: "failed"},
            description=(
                "Selected pass with full-suite fail is outcome discrepancy "
                "(inconclusive), not automatic FN without observation gap."
            ),
        )
    )

    return recipes


def build_manifest() -> dict[str, Any]:
    cases = _build_recipes()
    return {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "description": DESCRIPTION,
        "authoritative": False,
        "target_success_asserted": False,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = build_manifest()
    path = root / "corpus_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path} with {manifest['case_count']} cases")


if __name__ == "__main__":
    main()

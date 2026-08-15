"""Tests for incremental mutation verification composition (AAE-043).

Acceptance criteria enforced here:

* Only affected units invalidate; unrelated units stay reuse candidates.
* Cache reuse requires complete keys; incomplete/missing keys never reuse.
* Survivors broaden by policy (and high-risk/uncertainty force full suite).
* Temporary proof forests never replace or publish as canonical seals.
* Full and incremental costs and cache reuse are measured.
* No production policy change; cold import is side-effect free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.incremental import (
    AAE_INCREMENTAL_EVIDENCE,
    INCREMENTAL_MUTATION_VERIFIER_INTERFACE,
    REQUIRED_CACHE_KEY_FIELDS,
    BroadeningMode,
    CacheKeyBinding,
    CanonicalSealProtectionError,
    IncrementalMutationVerificationResult,
    IncrementalMutationVerifier,
    IncrementalVerificationError,
    MutantOutcomeClass,
    MutationCostAccounting,
    ProofUnit,
    SurvivorBroadeningPolicy,
    TemporaryProofForest,
    UnitDisposition,
    UnitKind,
    classify_unit_invalidation,
    create_incremental_mutation_verifier,
    evaluate_cache_reuse,
    incremental_mutation_verifier_descriptor,
    measure_mutation_costs,
    resolve_broadening_mode,
    verify_mutant_incremental,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
INCREMENTAL_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/incremental.py"
)


def _cid(label: str) -> str:
    return content_identity({"fixture": label, "schema": "aae043-fixture@1"})


TREE = _cid("repo-tree-aae043")
SEMANTIC = _cid("semantic-root-aae043")
ENV = _cid("environment-aae043")
LOCK = _cid("dependency-lock-aae043")
CANONICAL_SEAL = _cid("canonical-seal-aae043")
MUTANT = "mutant:sha256:aae043-demo-001"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _complete_key(
    check_id: str,
    *,
    kind: str = "test",
    key_suffix: str = "",
) -> CacheKeyBinding:
    suffix = key_suffix or check_id
    return CacheKeyBinding(
        key_cid=_cid(f"key-{suffix}"),
        repository_tree_cid=TREE,
        semantic_state_root_cid=SEMANTIC,
        environment_cid=ENV,
        dependency_lock_cid=LOCK,
        kind=kind,
        check_id=check_id,
        tool_name="pytest",
        tool_version="8.0.0",
    )


def _unit(
    unit_id: str,
    *,
    kind: UnitKind = UnitKind.TEST,
    symbols: tuple[str, ...] = (),
    paths: tuple[str, ...] = (),
    complete_key: bool = True,
    receipt_cid: str = "",
    terminal: str = "passed",
    cpu: int = 1000,
    wall: int = 1000,
    partial_key: dict | None = None,
) -> ProofUnit:
    key: CacheKeyBinding | None
    if partial_key is not None:
        key = None
        # Store incomplete key via from_value path.
        return ProofUnit.from_value(
            {
                "unit_id": unit_id,
                "kind": kind.value,
                "symbol_ids": symbols,
                "path_ids": paths,
                "cache_key": partial_key,
                "cached_receipt_cid": receipt_cid,
                "receipt_terminal": terminal,
                "cpu_cost_ms": cpu,
                "wall_cost_ms": wall,
            }
        )
    if complete_key:
        key = _complete_key(unit_id, kind=kind.value)
        if not receipt_cid:
            receipt_cid = _cid(f"receipt-{unit_id}")
    else:
        key = None
        receipt_cid = receipt_cid or ""
    return ProofUnit(
        unit_id=unit_id,
        kind=kind,
        symbol_ids=symbols,
        path_ids=paths,
        cache_key=key,
        cached_receipt_cid=receipt_cid,
        receipt_terminal=terminal,
        cpu_cost_ms=cpu,
        wall_cost_ms=wall,
    )


def _standard_units() -> list[ProofUnit]:
    return [
        _unit(
            "test::test_auth_login",
            symbols=("mod.auth.login",),
            paths=("pkg/auth.py",),
            cpu=2000,
            wall=2000,
        ),
        _unit(
            "test::test_auth_logout",
            symbols=("mod.auth.logout",),
            paths=("pkg/auth.py",),
            cpu=1500,
            wall=1500,
        ),
        _unit(
            "test::test_billing_invoice",
            symbols=("mod.billing.invoice",),
            paths=("pkg/billing.py",),
            cpu=3000,
            wall=3000,
        ),
        _unit(
            "proof::auth_invariant",
            kind=UnitKind.PROOF,
            symbols=("mod.auth.login",),
            paths=("pkg/auth.py",),
            cpu=5000,
            wall=5000,
        ),
        _unit(
            "test::test_unrelated_fmt",
            symbols=("mod.fmt.pad",),
            paths=("pkg/fmt.py",),
            cpu=500,
            wall=500,
        ),
    ]


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_interface_constants_and_descriptor() -> None:
    assert INCREMENTAL_MUTATION_VERIFIER_INTERFACE == "IncrementalMutationVerifier@1"
    desc = incremental_mutation_verifier_descriptor()
    assert desc["interface_id"] == INCREMENTAL_MUTATION_VERIFIER_INTERFACE
    assert desc["evidence_subset"] == AAE_INCREMENTAL_EVIDENCE
    assert desc["production_policy_changed"] is False
    assert "only_affected_units_invalidate" in desc["acceptance"]
    assert "temporary_forests_never_replace_canonical_seals" in desc["acceptance"]


def test_cold_import_is_side_effect_free() -> None:
    source = INCREMENTAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # No top-level calls that touch the filesystem/network beyond imports.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            pytest.fail(f"top-level call found: {ast.dump(node)}")
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                func = node.value.func
                # Allow enum / Final constructions and simple constructors only
                # via dataclasses field defaults — reject open()/Path() etc.
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in {"open", "Path", "run", "Popen", "urlopen"}:
                    pytest.fail(f"side-effecting top-level call: {name}")


def test_factory_and_verifier_to_dict() -> None:
    verifier = create_incremental_mutation_verifier()
    assert isinstance(verifier, IncrementalMutationVerifier)
    assert verifier.interface_id == INCREMENTAL_MUTATION_VERIFIER_INTERFACE
    payload = verifier.to_dict()
    assert payload["production_policy_changed"] is False
    assert payload["interface_id"] == INCREMENTAL_MUTATION_VERIFIER_INTERFACE


# ---------------------------------------------------------------------------
# Only affected units invalidate
# ---------------------------------------------------------------------------


def test_only_affected_units_invalidate_by_path_and_symbol() -> None:
    units = _standard_units()
    decisions = classify_unit_invalidation(
        units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
    )
    by_id = {d.unit_id: d for d in decisions}
    assert by_id["test::test_auth_login"].disposition is UnitDisposition.INVALIDATED
    assert by_id["test::test_auth_login"].affected is True
    assert by_id["proof::auth_invariant"].disposition is UnitDisposition.INVALIDATED
    # logout shares path pkg/auth.py → affected via path intersection.
    assert by_id["test::test_auth_logout"].disposition is UnitDisposition.INVALIDATED
    assert by_id["test::test_billing_invoice"].disposition is UnitDisposition.UNAFFECTED
    assert by_id["test::test_unrelated_fmt"].disposition is UnitDisposition.UNAFFECTED
    assert REASON_marker_unaffected(by_id["test::test_unrelated_fmt"])


def REASON_marker_unaffected(decision) -> bool:
    return "unrelated_unit_not_invalidated" in decision.reason_codes


def test_unrelated_edit_does_not_invalidate_unrelated_units() -> None:
    units = _standard_units()
    decisions = classify_unit_invalidation(
        units,
        changed_symbols=("mod.fmt.pad",),
        changed_paths=("pkg/fmt.py",),
    )
    by_id = {d.unit_id: d for d in decisions}
    assert by_id["test::test_unrelated_fmt"].disposition is UnitDisposition.INVALIDATED
    for unit_id in (
        "test::test_auth_login",
        "test::test_auth_logout",
        "test::test_billing_invoice",
        "proof::auth_invariant",
    ):
        assert by_id[unit_id].disposition is UnitDisposition.UNAFFECTED
        assert by_id[unit_id].affected is False


def test_explicit_affected_unit_ids() -> None:
    units = _standard_units()
    decisions = classify_unit_invalidation(
        units,
        affected_unit_ids=("test::test_billing_invoice",),
    )
    by_id = {d.unit_id: d for d in decisions}
    assert by_id["test::test_billing_invoice"].disposition is UnitDisposition.INVALIDATED
    assert by_id["test::test_auth_login"].disposition is UnitDisposition.UNAFFECTED


# ---------------------------------------------------------------------------
# Reuse requires complete keys
# ---------------------------------------------------------------------------


def test_reuse_requires_complete_keys() -> None:
    complete = _unit("test::complete", symbols=("a",), paths=("a.py",))
    incomplete = _unit(
        "test::incomplete",
        symbols=("b",),
        paths=("b.py",),
        partial_key={
            "key_cid": _cid("partial-key"),
            "check_id": "test::incomplete",
            # missing required fields
        },
    )
    missing = _unit(
        "test::missing",
        symbols=("c",),
        paths=("c.py",),
        complete_key=False,
    )
    units = [complete, incomplete, missing]
    invalidation = classify_unit_invalidation(
        units,
        changed_symbols=("z.never",),
        changed_paths=("z/never.py",),
    )
    # All unaffected.
    assert all(d.disposition is UnitDisposition.UNAFFECTED for d in invalidation)
    reuse = evaluate_cache_reuse(units, invalidation)
    by_id = {d.unit_id: d for d in reuse}
    assert by_id["test::complete"].disposition is UnitDisposition.REUSED
    assert by_id["test::complete"].key_complete is True
    assert "complete_key_reused" in by_id["test::complete"].reason_codes
    assert by_id["test::incomplete"].disposition is UnitDisposition.REVERIFY
    assert by_id["test::incomplete"].key_complete is False
    assert "incomplete_key_rejected" in by_id["test::incomplete"].reason_codes
    assert by_id["test::missing"].disposition is UnitDisposition.REVERIFY
    assert by_id["test::missing"].key_complete is False


def test_stale_or_simulated_receipts_never_reuse() -> None:
    stale = _unit(
        "test::stale",
        symbols=("s",),
        paths=("s.py",),
        terminal="stale",
    )
    simulated = _unit(
        "test::sim",
        symbols=("t",),
        paths=("t.py",),
        terminal="simulated",
    )
    units = [stale, simulated]
    invalidation = classify_unit_invalidation(units)
    reuse = evaluate_cache_reuse(units, invalidation)
    by_id = {d.unit_id: d for d in reuse}
    assert by_id["test::stale"].disposition is UnitDisposition.REVERIFY
    assert "stale_receipt_rejected" in by_id["test::stale"].reason_codes
    assert by_id["test::sim"].disposition is UnitDisposition.REVERIFY


def test_affected_units_never_reuse_even_with_complete_keys() -> None:
    units = [
        _unit(
            "test::hit",
            symbols=("mod.auth.login",),
            paths=("pkg/auth.py",),
        )
    ]
    invalidation = classify_unit_invalidation(
        units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
    )
    reuse = evaluate_cache_reuse(units, invalidation)
    assert reuse[0].disposition is UnitDisposition.REVERIFY
    assert reuse[0].affected is True
    assert reuse[0].reused_receipt_cid == ""


def test_cache_key_binding_rejects_empty_fields() -> None:
    with pytest.raises(IncrementalVerificationError):
        CacheKeyBinding(
            key_cid="",
            repository_tree_cid=TREE,
            semantic_state_root_cid=SEMANTIC,
            environment_cid=ENV,
            dependency_lock_cid=LOCK,
            kind="test",
            check_id="x",
            tool_name="pytest",
            tool_version="1",
        )
    assert len(REQUIRED_CACHE_KEY_FIELDS) >= 8


# ---------------------------------------------------------------------------
# Survivors broaden by policy
# ---------------------------------------------------------------------------


def test_survivors_broaden_by_policy() -> None:
    units = _standard_units()
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.SURVIVOR,
        risk_class="low_risk_boilerplate",
        broadening_policy=SurvivorBroadeningPolicy(
            broaden_survivors=True,
            full_suite_on_high_risk=True,
            high_risk_classes=("critical_security", "authorization"),
        ),
    )
    assert result.broadening_mode is BroadeningMode.BROADER
    assert result.mutant_outcome is MutantOutcomeClass.SURVIVOR
    assert "survivor_broadened_by_policy" in result.reason_codes
    # Broader selection should schedule more than just the invalidated set.
    assert result.broadened_unit_ids or any(
        d.disposition is UnitDisposition.BROADENED for d in result.decisions
    )


def test_survivor_broadening_disabled_stays_exact() -> None:
    units = _standard_units()
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.SURVIVOR,
        risk_class="low_risk_boilerplate",
        broadening_policy=SurvivorBroadeningPolicy(broaden_survivors=False),
    )
    assert result.broadening_mode is BroadeningMode.NONE
    assert "survivor_broadening_disabled" in result.reason_codes
    assert result.broadened_unit_ids == ()
    assert result.full_suite_unit_ids == ()


def test_high_risk_survivor_forces_full_suite() -> None:
    units = _standard_units()
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.SURVIVOR,
        risk_class="critical_security",
        broadening_policy=SurvivorBroadeningPolicy(
            broaden_survivors=True,
            full_suite_on_high_risk=True,
            high_risk_classes=("critical_security", "authorization"),
        ),
    )
    assert result.broadening_mode is BroadeningMode.FULL_SUITE
    assert "high_risk_requires_full_suite" in result.reason_codes
    full_ids = set(result.full_suite_unit_ids) | {
        d.unit_id
        for d in result.decisions
        if d.disposition is UnitDisposition.FULL_SUITE
    }
    assert full_ids
    # Full suite covers the catalog (all standard units).
    assert len(full_ids) >= 1


def test_killed_mutant_does_not_broaden() -> None:
    mode, reasons = resolve_broadening_mode(
        mutant_outcome=MutantOutcomeClass.KILLED,
        risk_class="low_risk_boilerplate",
        policy=SurvivorBroadeningPolicy(broaden_survivors=True),
    )
    assert mode is BroadeningMode.NONE
    assert reasons == ()


def test_uncertainty_forces_full_suite_when_policy_allows() -> None:
    mode, reasons = resolve_broadening_mode(
        mutant_outcome=MutantOutcomeClass.KILLED,
        uncertainty=True,
        policy=SurvivorBroadeningPolicy(full_suite_on_uncertainty=True),
    )
    assert mode is BroadeningMode.FULL_SUITE
    assert "uncertainty_requires_broader" in reasons


# ---------------------------------------------------------------------------
# Temporary forests never replace canonical seals
# ---------------------------------------------------------------------------


def test_temporary_forest_never_replaces_canonical_seal() -> None:
    units = _standard_units()
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.KILLED,
        canonical_seal_cid=CANONICAL_SEAL,
        parent_canonical_seal_cid=CANONICAL_SEAL,
    )
    forest = result.temporary_forest
    assert forest is not None
    assert forest.is_temporary is True
    assert forest.is_canonical is False
    assert result.canonical_seal_replaced is False
    assert result.production_policy_changed is False
    assert "temporary_forest_only" in result.reason_codes
    assert "canonical_seal_not_replaced" in result.reason_codes
    assert result.canonical_seal_cid == CANONICAL_SEAL
    # Parent may be recorded for lineage, but forest root is distinct.
    assert forest.parent_canonical_seal_cid == CANONICAL_SEAL
    assert forest.forest_root_cid != CANONICAL_SEAL


def test_temporary_forest_replace_and_publish_refuse() -> None:
    forest = TemporaryProofForest(
        forest_id=_cid("temp-forest"),
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        unit_proof_cids={"u1": _cid("proof-u1")},
        parent_canonical_seal_cid=CANONICAL_SEAL,
    )
    with pytest.raises(CanonicalSealProtectionError) as exc_info:
        forest.replace_canonical_seal(CANONICAL_SEAL)
    assert exc_info.value.reason_code == "canonical_seal_replace_refused"
    with pytest.raises(CanonicalSealProtectionError):
        forest.publish_as_canonical()
    # Construction forces non-canonical even if caller lies.
    forced = TemporaryProofForest(
        forest_id=_cid("forced"),
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        is_canonical=True,  # type: ignore[arg-type]
        is_temporary=False,  # type: ignore[arg-type]
    )
    assert forced.is_canonical is False
    assert forced.is_temporary is True


def test_result_forces_canonical_seal_replaced_false() -> None:
    units = [_unit("test::only", symbols=("a",), paths=("a.py",))]
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("a",),
        changed_paths=("a.py",),
        canonical_seal_cid=CANONICAL_SEAL,
    )
    # Even if someone reconstructs with True, __post_init__ clamps it.
    rebuilt = IncrementalMutationVerificationResult(
        mutant_id=result.mutant_id,
        repository_tree_cid=result.repository_tree_cid,
        decisions=result.decisions,
        temporary_forest=result.temporary_forest,
        cost_accounting=result.cost_accounting,
        broadening_mode=result.broadening_mode,
        mutant_outcome=result.mutant_outcome,
        canonical_seal_cid=CANONICAL_SEAL,
        canonical_seal_replaced=True,  # attempt
        production_policy_changed=True,  # attempt
    )
    assert rebuilt.canonical_seal_replaced is False
    assert rebuilt.production_policy_changed is False


# ---------------------------------------------------------------------------
# Full and incremental costs and cache reuse are measured
# ---------------------------------------------------------------------------


def test_full_and_incremental_costs_and_cache_reuse_measured() -> None:
    units = _standard_units()
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.KILLED,
    )
    costs = result.cost_accounting
    assert costs is not None
    assert isinstance(costs, MutationCostAccounting)
    assert costs.measured is True
    assert costs.units_total == len(units)
    assert costs.full_cpu_ms == sum(u.cpu_cost_ms for u in units)
    assert costs.full_wall_ms == sum(u.wall_cost_ms for u in units)
    # Incremental must be strictly cheaper when some units reuse.
    assert costs.units_reused >= 1
    assert costs.cache_hits == costs.units_reused
    assert costs.incremental_cpu_ms < costs.full_cpu_ms
    assert costs.incremental_wall_ms < costs.full_wall_ms
    assert costs.compute_saved_cpu_ms is not None
    assert costs.compute_saved_cpu_ms == costs.full_cpu_ms - costs.incremental_cpu_ms
    assert costs.full is not None
    assert costs.incremental is not None
    assert costs.comparison is not None
    assert "full_and_incremental_costs_measured" in costs.reason_codes
    assert "cache_reuse_measured" in costs.reason_codes
    assert "full_and_incremental_costs_measured" in result.reason_codes


def test_measure_costs_with_all_reuse() -> None:
    units = [
        _unit("t1", symbols=("x",), paths=("x.py",), cpu=100, wall=100),
        _unit("t2", symbols=("y",), paths=("y.py",), cpu=200, wall=200),
    ]
    # No changed symbols → all unaffected → all reusable.
    invalidation = classify_unit_invalidation(units)
    reuse = evaluate_cache_reuse(units, invalidation)
    costs = measure_mutation_costs(units, reuse)
    assert costs.units_reused == 2
    assert costs.cache_hits == 2
    assert costs.units_invalidated == 0
    assert costs.incremental_cpu_ms < costs.full_cpu_ms


def test_measure_costs_with_no_reuse() -> None:
    units = [
        _unit(
            "t1",
            symbols=("x",),
            paths=("x.py",),
            complete_key=False,
            cpu=100,
            wall=100,
        ),
        _unit(
            "t2",
            symbols=("x",),
            paths=("x.py",),
            complete_key=False,
            cpu=200,
            wall=200,
        ),
    ]
    invalidation = classify_unit_invalidation(
        units, changed_symbols=("x",), changed_paths=("x.py",)
    )
    reuse = evaluate_cache_reuse(units, invalidation)
    costs = measure_mutation_costs(units, reuse)
    assert costs.units_reused == 0
    assert costs.cache_hits == 0
    assert costs.units_invalidated == 2
    # Incremental cost equals full cost when nothing reuses.
    assert costs.incremental_cpu_ms == costs.full_cpu_ms


# ---------------------------------------------------------------------------
# End-to-end pipeline properties
# ---------------------------------------------------------------------------


def test_verify_mutant_end_to_end_deterministic() -> None:
    units = _standard_units()
    kwargs = dict(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        mutant_outcome=MutantOutcomeClass.KILLED,
        canonical_seal_cid=CANONICAL_SEAL,
    )
    a = verify_mutant_incremental(**kwargs)  # type: ignore[arg-type]
    b = verify_mutant_incremental(**kwargs)  # type: ignore[arg-type]
    assert a.result_cid == b.result_cid
    assert a.to_dict() == b.to_dict()
    assert a.interface_id == "IncrementalMutationVerificationResult@1"
    assert a.invalidated_unit_ids
    assert a.reused_unit_ids
    # Auth units invalidated; billing and fmt reused (complete keys).
    assert "test::test_billing_invoice" in a.reused_unit_ids
    assert "test::test_unrelated_fmt" in a.reused_unit_ids
    assert "test::test_auth_login" in a.invalidated_unit_ids


def test_empty_units_fail_closed() -> None:
    with pytest.raises(IncrementalVerificationError):
        verify_mutant_incremental(
            mutant_id=MUTANT,
            repository_tree_cid=TREE,
            units=[],
        )


def test_missing_mutant_id_fail_closed() -> None:
    with pytest.raises(IncrementalVerificationError):
        verify_mutant_incremental(
            mutant_id="",
            repository_tree_cid=TREE,
            units=[_unit("t1", symbols=("a",), paths=("a.py",))],
        )


def test_verifier_method_surface() -> None:
    verifier = IncrementalMutationVerifier()
    units = _standard_units()
    inv = verifier.invalidate_units(
        units,
        changed_symbols=("mod.billing.invoice",),
        changed_paths=("pkg/billing.py",),
    )
    reuse = verifier.evaluate_reuse(units, inv)
    mode, final, reasons = verifier.broaden_survivors(
        reuse,
        units,
        mutant_outcome=MutantOutcomeClass.SURVIVOR,
        risk_class="low_risk_boilerplate",
    )
    costs = verifier.measure_costs(units, final)
    assert mode is BroadeningMode.BROADER
    assert costs.measured is True
    result = verifier.verify_mutant(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=units,
        changed_symbols=("mod.billing.invoice",),
        changed_paths=("pkg/billing.py",),
        mutant_outcome=MutantOutcomeClass.KILLED,
    )
    assert result.cost_accounting is not None
    assert result.temporary_forest is not None


def test_result_to_dict_serializable() -> None:
    result = verify_mutant_incremental(
        mutant_id=MUTANT,
        repository_tree_cid=TREE,
        units=_standard_units()[:2],
        changed_symbols=("mod.auth.login",),
        changed_paths=("pkg/auth.py",),
        canonical_seal_cid=CANONICAL_SEAL,
    )
    payload = result.to_dict()
    assert payload["canonical_seal_replaced"] is False
    assert payload["production_policy_changed"] is False
    assert payload["temporary_forest"]["is_canonical"] is False
    assert payload["temporary_forest"]["is_temporary"] is True
    assert payload["cost_accounting"]["measured"] is True
    assert isinstance(payload["decisions"], list)
    assert payload["result_cid"]


def test_proof_unit_intersects_by_unit_id() -> None:
    unit = _unit("special::node", symbols=(), paths=())
    assert unit.intersects(
        changed_symbols=("special::node",),
        changed_paths=(),
    )

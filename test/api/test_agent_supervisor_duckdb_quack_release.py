"""Tests for DuckDBControlPlaneReleaseReceipt@1 (DQP-039)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
)
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
    SAFETY_FLOOR_KEYS,
)
from ipfs_accelerate_py.agent_supervisor.validation import duckdb_quack_release as release
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release import (
    BOARD_NAMESPACE,
    DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE,
    DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE,
    EVIDENCE,
    GOAL_ID,
    NON_CLAIMS,
    RELEASE_CONTRACT_VERSION,
    REQUIRED_EVIDENCE_ROOTS,
    TASK_ID,
    DenialReason,
    DuckDBControlPlaneReleasePolicy,
    DuckDBControlPlaneReleaseReceipt,
    DuckDBControlPlaneReleaseVerifier,
    EvidenceClass,
    ReleaseEvidence,
    ReleaseEvidenceItem,
    ReleaseVerdict,
    classify_evidence_disposition,
    hermetic_passing_evidence,
    issue_release_receipt,
    replay_release_receipt,
    validate_duckdb_quack_release,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "duckdb_quack_release.py"
)
_DOC_PATH = (
    _REPO_ROOT / "docs" / "architecture" / "AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md"
)

REQUIRED_AST_SYMBOLS = {
    "DuckDBControlPlaneReleasePolicy",
    "DuckDBControlPlaneReleaseReceipt",
    "DuckDBControlPlaneReleaseVerifier",
    "ReleaseEvidence",
    "ReleaseEvidenceItem",
    "classify_evidence_disposition",
    "hermetic_passing_evidence",
    "issue_release_receipt",
    "replay_release_receipt",
    "validate_duckdb_quack_release",
}


# ---------------------------------------------------------------------------
# Identity / surface
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert (
        DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE
        == "DuckDBControlPlaneReleaseReceipt@1"
    )
    assert (
        DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE
        == "DuckDBControlPlaneReleaseVerifier@1"
    )
    assert DuckDBControlPlaneReleaseReceipt.INTERFACE == (
        DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE
    )
    assert DuckDBControlPlaneReleaseVerifier.INTERFACE == (
        DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE
    )
    assert TASK_ID == "DQP-039"
    assert GOAL_ID == "DQP-G090"
    assert BOARD_NAMESPACE == "agent-supervisor-duckdb-quack-control-plane-v1"
    assert EVIDENCE == "dqp/duckdb-quack-release@1"
    assert RELEASE_CONTRACT_VERSION == 1


def test_module_exports_required_ast_symbols() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = REQUIRED_AST_SYMBOLS - names
    assert not missing, f"missing symbols: {sorted(missing)}"


def test_required_evidence_roots_cover_joined_surfaces() -> None:
    expected = {
        "schema",
        "quack",
        "import_export",
        "intent",
        "runtime",
        "worktree",
        "ast_mutation",
        "symbolic_proof",
        "context_churn",
        "control",
        "watchdog",
        "backup",
        "chaos",
        "canary",
        "shadow",
        "cutover",
        "rollback",
    }
    assert set(REQUIRED_EVIDENCE_ROOTS) == expected
    assert set(release.RELEASE_SAFETY_FLOOR_KEYS) == set(SAFETY_FLOOR_KEYS)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_hermetic_release_passes_and_records_experimental_scope() -> None:
    receipt = issue_release_receipt(hermetic_passing_evidence())
    assert receipt.verdict is ReleaseVerdict.PASS
    assert receipt.passed is True
    assert receipt.reason_codes == ()
    assert set(receipt.admissible_roots) == set(REQUIRED_EVIDENCE_ROOTS)
    assert receipt.modules_missing == ()
    assert receipt.experimental_scope is True
    assert receipt.production_ha_claimed is False
    assert receipt.duckdb_2_0_compatibility_claimed is False
    assert receipt.promotion_allowed is False
    assert receipt.completion_authoritative is False
    assert receipt.mutation_authorized is False
    assert tuple(receipt.non_claims) == NON_CLAIMS
    assert "not_production_ha" in receipt.non_claims
    assert "not_duckdb_2_0_compatible_until_separately_tested" in receipt.non_claims
    for limitation in DEFAULT_QUACK_BETA_LIMITATIONS:
        assert limitation in receipt.beta_limitations
    assert all(value == 0 for value in receipt.safety_floors.values())

    payload = receipt.to_dict()
    assert payload["interface"] == DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE
    assert payload["passed"] is True
    assert payload["task_id"] == "DQP-039"
    assert payload["production_ha_claimed"] is False
    assert payload["duckdb_2_0_compatibility_claimed"] is False
    assert payload["promotion_allowed"] is False
    assert payload["experimental_scope"] is True
    assert "identity_id" in payload
    assert payload["identity_id"].startswith("sha256:")


def test_validate_alias_and_replay_identity() -> None:
    evidence = hermetic_passing_evidence()
    first = validate_duckdb_quack_release(evidence)
    second = issue_release_receipt(evidence)
    assert first.identity_id == second.identity_id
    assert first.evidence_identity == evidence.identity_id
    replay = replay_release_receipt(first, evidence)
    assert replay["identity_ok"] is True
    assert replay["production_ha_claimed"] is False
    assert replay["duckdb_2_0_compatibility_claimed"] is False
    assert replay["experimental_scope"] is True


def test_receipt_hard_seals_non_claims_even_if_constructed_wrong() -> None:
    # Direct construction still seals honest scope.
    receipt = DuckDBControlPlaneReleaseReceipt(
        verdict=ReleaseVerdict.PASS,
        policy_identity="policy:x",
        evidence_identity="evidence:x",
        tree_id="tree:x",
        database_identity="db:x",
        schema_checksum="sha256:" + ("cc" * 32),
        extension_fingerprint="sha256:" + ("dd" * 32),
        duckdb_version="1.5.2",
        quack_profile="profile:p",
        production_ha_claimed=True,
        duckdb_2_0_compatibility_claimed=True,
        promotion_allowed=True,
        completion_authoritative=True,
        mutation_authorized=True,
        experimental_scope=False,
    )
    assert receipt.production_ha_claimed is False
    assert receipt.duckdb_2_0_compatibility_claimed is False
    assert receipt.promotion_allowed is False
    assert receipt.completion_authoritative is False
    assert receipt.mutation_authorized is False
    assert receipt.experimental_scope is True


# ---------------------------------------------------------------------------
# Evidence class rejections
# ---------------------------------------------------------------------------


def _replace_root(
    evidence: ReleaseEvidence,
    root: str,
    **kwargs: Any,
) -> ReleaseEvidence:
    items = []
    for item in evidence.items:
        if item.root == root:
            items.append(replace(item, **kwargs))
        else:
            items.append(item)
    return replace(evidence, items=tuple(items))


@pytest.mark.parametrize(
    "evidence_class,denial",
    [
        (EvidenceClass.STALE, DenialReason.STALE_EVIDENCE.value),
        (EvidenceClass.SYNTHETIC, DenialReason.SYNTHETIC_EVIDENCE.value),
        (EvidenceClass.SKIPPED, DenialReason.SKIPPED_EVIDENCE.value),
        (EvidenceClass.MISSING, DenialReason.MISSING_EVIDENCE.value),
        (EvidenceClass.FORGED, DenialReason.FORGED_EVIDENCE.value),
    ],
)
def test_release_fails_on_bad_evidence_class(
    evidence_class: EvidenceClass, denial: str
) -> None:
    evidence = hermetic_passing_evidence()
    evidence = _replace_root(
        evidence,
        "canary",
        evidence_class=evidence_class,
        passed=evidence_class in {EvidenceClass.MEASURED, EvidenceClass.CURRENT},
    )
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert receipt.passed is False
    assert any(code.startswith(f"{denial}:canary") for code in receipt.reason_codes)


def test_release_fails_on_missing_evidence_root() -> None:
    evidence = hermetic_passing_evidence()
    items = tuple(item for item in evidence.items if item.root != "chaos")
    evidence = replace(evidence, items=items)
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert "missing_evidence:chaos" in receipt.reason_codes
    assert "chaos" in receipt.missing_roots


def test_release_fails_on_stale_age() -> None:
    evidence = hermetic_passing_evidence(age_seconds=10**9)
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert any(
        code.startswith(DenialReason.STALE_EVIDENCE.value)
        for code in receipt.reason_codes
    )


def test_release_fails_on_tree_mismatch() -> None:
    evidence = hermetic_passing_evidence()
    evidence = _replace_root(evidence, "schema", tree_id="tree:other")
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert any(
        code.startswith(DenialReason.TREE_MISMATCH.value)
        for code in receipt.reason_codes
    )


def test_classify_evidence_disposition_map() -> None:
    item = ReleaseEvidenceItem(
        root="canary",
        identity="e:canary",
        evidence_class=EvidenceClass.MEASURED,
        age_seconds=1,
        passed=True,
    )
    assert classify_evidence_disposition(item) == "admissible"
    bad = replace(item, evidence_class=EvidenceClass.SYNTHETIC)
    assert classify_evidence_disposition(bad) == DenialReason.SYNTHETIC_EVIDENCE.value


# ---------------------------------------------------------------------------
# Authority / safety / lineage rejections
# ---------------------------------------------------------------------------


def test_release_fails_on_legacy_file_decision_read() -> None:
    evidence = replace(
        hermetic_passing_evidence(), legacy_file_decision_reads=1
    )
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert DenialReason.LEGACY_FILE_DECISION_READ.value in receipt.reason_codes


def test_release_fails_when_database_not_sole_authority() -> None:
    evidence = replace(
        hermetic_passing_evidence(),
        database_sole_decision_authority=False,
    )
    receipt = issue_release_receipt(evidence)
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert DenialReason.DATABASE_NOT_SOLE_AUTHORITY.value in receipt.reason_codes


def test_release_fails_on_unauthorized_sql() -> None:
    evidence = replace(hermetic_passing_evidence(), unauthorized_sql_count=1)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.UNAUTHORIZED_SQL.value in receipt.reason_codes
    assert receipt.passed is False


def test_release_fails_on_stale_lease_write() -> None:
    evidence = replace(hermetic_passing_evidence(), stale_lease_write_count=2)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.STALE_LEASE_WRITE.value in receipt.reason_codes


def test_release_fails_on_false_completion() -> None:
    evidence = replace(hermetic_passing_evidence(), false_completion_count=1)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.FALSE_COMPLETION.value in receipt.reason_codes


def test_release_fails_on_accepted_state_loss() -> None:
    evidence = replace(hermetic_passing_evidence(), accepted_state_loss_count=1)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.ACCEPTED_STATE_LOSS.value in receipt.reason_codes


def test_release_fails_on_incomplete_mutation_lineage() -> None:
    evidence = replace(
        hermetic_passing_evidence(), mutation_lineage_complete=False
    )
    receipt = issue_release_receipt(evidence)
    assert DenialReason.INCOMPLETE_MUTATION_LINEAGE.value in receipt.reason_codes


def test_release_fails_on_projection_divergence() -> None:
    evidence = replace(hermetic_passing_evidence(), projection_divergent=True)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.PROJECTION_DIVERGENCE.value in receipt.reason_codes


def test_release_fails_on_safety_and_quality_regression() -> None:
    evidence = replace(
        hermetic_passing_evidence(),
        safety_regression=True,
        quality_regression=True,
    )
    receipt = issue_release_receipt(evidence)
    assert DenialReason.SAFETY_REGRESSION.value in receipt.reason_codes
    assert DenialReason.QUALITY_REGRESSION.value in receipt.reason_codes


def test_release_fails_on_absent_rollback() -> None:
    evidence = replace(
        hermetic_passing_evidence(),
        rollback_present=False,
        rollback_identity="",
    )
    receipt = issue_release_receipt(evidence)
    assert DenialReason.ABSENT_ROLLBACK.value in receipt.reason_codes


def test_release_fails_on_nonzero_safety_floor() -> None:
    floors = {key: 0 for key in SAFETY_FLOOR_KEYS}
    floors["unauthorized_sql"] = 1
    evidence = replace(hermetic_passing_evidence(), safety_floors=floors)
    receipt = issue_release_receipt(evidence)
    assert receipt.passed is False
    assert any(
        code.startswith(DenialReason.SAFETY_FLOOR_NONZERO.value)
        for code in receipt.reason_codes
    )


# ---------------------------------------------------------------------------
# Scope honesty
# ---------------------------------------------------------------------------


def test_release_fails_on_production_ha_claim() -> None:
    evidence = replace(hermetic_passing_evidence(), production_ha_claimed=True)
    receipt = issue_release_receipt(evidence)
    assert DenialReason.PRODUCTION_HA_CLAIM.value in receipt.reason_codes
    assert receipt.passed is False
    # Receipt still hard-seals non-claim on the output surface.
    assert receipt.production_ha_claimed is False


def test_release_fails_on_duckdb_2_0_compatibility_claim() -> None:
    evidence = replace(
        hermetic_passing_evidence(), duckdb_2_0_compatibility_claimed=True
    )
    receipt = issue_release_receipt(evidence)
    assert (
        DenialReason.DUCKDB_2_0_COMPATIBILITY_CLAIM.value in receipt.reason_codes
    )
    assert receipt.duckdb_2_0_compatibility_claimed is False


def test_release_fails_when_beta_scope_unrecorded() -> None:
    evidence = replace(
        hermetic_passing_evidence(),
        experimental_scope=False,
        beta_limitations=(),
    )
    receipt = issue_release_receipt(evidence)
    assert DenialReason.BETA_SCOPE_UNRECORDED.value in receipt.reason_codes


def test_release_fails_on_completion_or_mutation_authority_claim() -> None:
    evidence = replace(
        hermetic_passing_evidence(),
        completion_authoritative=True,
        mutation_authorized=True,
    )
    receipt = issue_release_receipt(evidence)
    assert DenialReason.COMPLETION_AUTHORITY_CLAIM.value in receipt.reason_codes
    assert DenialReason.MUTATION_AUTHORITY_CLAIM.value in receipt.reason_codes


def test_release_fails_on_missing_required_module() -> None:
    policy = DuckDBControlPlaneReleasePolicy(
        required_modules=(
            "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release",
            "ipfs_accelerate_py.agent_supervisor.does_not_exist_module_dqp039",
        )
    )
    receipt = issue_release_receipt(hermetic_passing_evidence(), policy=policy)
    assert receipt.passed is False
    assert any(
        "does_not_exist_module_dqp039" in code for code in receipt.reason_codes
    )


# ---------------------------------------------------------------------------
# Documentation + content identity
# ---------------------------------------------------------------------------


def test_release_doc_records_experimental_scope_and_non_claims() -> None:
    assert _DOC_PATH.is_file()
    text = _DOC_PATH.read_text(encoding="utf-8")
    for phrase in (
        "DuckDBControlPlaneReleaseReceipt@1",
        "DQP-039",
        "experimental",
        "beta",
        "not production HA",
        "DuckDB 2.0",
        "fail-closed",
        "legacy file",
        "unauthorized SQL",
        "rollback",
        "mutation lineage",
    ):
        assert phrase.lower() in text.lower(), phrase


def test_content_identity_stable_for_same_evidence() -> None:
    a = hermetic_passing_evidence()
    b = hermetic_passing_evidence()
    assert a.identity_id == b.identity_id
    r1 = issue_release_receipt(a)
    r2 = issue_release_receipt(b)
    assert r1.identity_id == r2.identity_id


def test_verifier_facade() -> None:
    verifier = DuckDBControlPlaneReleaseVerifier()
    receipt = verifier.evaluate(hermetic_passing_evidence())
    assert receipt.passed is True
    assert receipt.task_id == "DQP-039"

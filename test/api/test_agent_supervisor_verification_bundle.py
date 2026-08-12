"""Tests for verification bundles, compact summaries, and commitments.

Normative non-claims covered by this module and its implementation docs:

* a verification commitment is **not** a zero-knowledge proof;
* signed receipts do **not** prove execution unless the issuer is trusted;
* structural validation is **not** cryptographic validation of tool execution.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.verification.bundle import (
    COMMITMENT_EMPTY_DOMAIN,
    COMMITMENT_HASH_ALGORITHM,
    COMMITMENT_IS_ZERO_KNOWLEDGE_PROOF,
    COMMITMENT_LEAF_CODEC,
    COMMITMENT_LEAF_DOMAIN,
    COMMITMENT_NODE_DOMAIN,
    VERIFICATION_COMMITMENT_EVIDENCE,
    VERIFICATION_SUMMARY_EVIDENCE,
    BundleReceiptClassification,
    VerificationBundleBuildError,
    build_verification_bundle,
    build_verification_commitment,
    build_verification_summary,
    classify_bundle_receipts,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDecision,
    CacheReuseDisposition,
    CounterexampleReceipt,
    ModelRoute,
    ModelRouteDecision,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationCommitment,
    VerificationIdentityError,
    VerificationReceiptKind,
    VerificationSummary,
    aggregate_terminal_status,
)
from test.api.test_agent_supervisor_verification_contracts import (
    ENVIRONMENT_SCHEMA,
    _artifact,
    _compiler_kwargs,
    _expected_environment,
    _key,
    _observation,
    _plan,
    _route,
    _structured_cid,
)


# ---------------------------------------------------------------------------
# Helpers local to this suite
# ---------------------------------------------------------------------------


def _counterexample(
    key,
    failed_receipt,
    *,
    label: str = "cx",
) -> CounterexampleReceipt:
    return CounterexampleReceipt(
        failed_key_cid=key.key_id,
        failed_receipt_cid=failed_receipt.receipt_id,
        failed_selector=key.selector_cid,
        failure_identity_cid=_artifact(f"{label}-failure-identity"),
        relevant_symbol_version_cids=key.affected_symbol_version_cids,
        minimized_traceback=("example.py:12: expected str, observed int",),
        relevant_assertion="result must be a string",
        relevant_input={"state": "present", "value": {"argument_type": "int"}},
        expected_output={"state": "present", "value": "str"},
        observed_output={"state": "present", "value": "int"},
        source_spans=(),
        environment_cid=key.environment_cid,
        dependency_lock_cid=key.dependency_lock_cid,
        reproduction_argv=failed_receipt.execution.command_argv,
        artifact_cids=(_artifact(f"{label}-diagnostic"),),
        minimized=True,
        reason_codes=("deterministic_slice_preserved_failure",),
    )


def _passing_type(key, *, label: str = "pass") -> TypeCheckReceipt:
    return TypeCheckReceipt(key, _observation(key, TerminalStatus.PASSED, label=label))


def _failed_type(key, *, label: str = "fail") -> TypeCheckReceipt:
    return TypeCheckReceipt(key, _observation(key, TerminalStatus.FAILED, label=label))


def _manual_merkle_root(leaves: list[dict[str, object]]) -> str:
    """Independent recomputation of the sealed IVP@1 Merkle construction."""

    if not leaves:
        digest = hashlib.sha256(b"IVP-EMPTY@1\x00").digest()
        return "sha256:" + digest.hex()
    ordered = sorted(leaves, key=lambda item: (item["key_cid"], item["receipt_cid"]))
    level = [
        hashlib.sha256(b"IVP-LEAF@1\x00" + canonical_json_bytes(item)).digest()
        for item in ordered
    ]
    while len(level) > 1:
        next_level: list[bytes] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                # Odd-node promotion: unpaired digest advances unchanged.
                next_level.append(level[index])
            else:
                next_level.append(
                    hashlib.sha256(
                        b"IVP-NODE@1\x00" + level[index] + level[index + 1]
                    ).digest()
                )
        level = next_level
    return "sha256:" + level[0].hex()


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def test_build_bundle_distinguishes_required_admitted_reused_executed_unresolved() -> (
    None
):
    first = _key()
    second = _key(receipt_schema_version=2)
    reused_receipt = _passing_type(first, label="reused")
    executed_receipt = _passing_type(second, label="executed")
    reuse_plan = replace(
        _plan(first, second),
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=first.key_id,
                disposition=CacheReuseDisposition.REUSED,
                reason_codes=("exact_current_production_receipt",),
                candidate_receipt=reused_receipt,
            ),
            CacheReuseDecision(
                key_cid=second.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
        ),
    )

    bundle = build_verification_bundle(
        reuse_plan,
        reused_receipts=(reused_receipt,),
        executed_receipts=(executed_receipt,),
    )
    classification = classify_bundle_receipts(bundle)

    assert isinstance(classification, BundleReceiptClassification)
    assert set(classification.required_key_cids) == {first.key_id, second.key_id}
    assert set(classification.admitted_receipt_cids) == {
        reused_receipt.receipt_id,
        executed_receipt.receipt_id,
    }
    assert classification.reused_receipt_cids == (reused_receipt.receipt_id,)
    assert classification.executed_receipt_cids == (executed_receipt.receipt_id,)
    assert classification.unresolved_requirement_ids == ()
    assert bundle.structurally_complete

    partial = build_verification_bundle(
        _plan(first, second),
        executed_receipts=(executed_receipt,),
    )
    partial_view = classify_bundle_receipts(partial)
    assert partial_view.unresolved_requirement_ids == (first.key_id,)
    assert set(partial_view.admitted_receipt_cids) == {executed_receipt.receipt_id}
    assert partial_view.reused_receipt_cids == ()
    assert not partial.structurally_complete


def test_build_bundle_rejects_mixed_tree_or_environment_leaves() -> None:
    key = _key()
    receipt = _passing_type(key, label="base")
    env_values = _compiler_kwargs()
    environment = {
        **env_values["observed_environment"],  # type: ignore[dict-item]
        "platform": {
            "schema": "verification-platform@1",
            "os": "linux",
            "architecture": "aarch64",
            "libc": "glibc-2.39",
        },
    }
    mixed_key = _key(
        observed_environment=environment,
        claimed_environment_cid=_structured_cid(
            ENVIRONMENT_SCHEMA,
            _expected_environment(
                {**env_values, "observed_environment": environment}
            ),
        ),
    )
    mixed_receipt = TypeCheckReceipt(
        mixed_key,
        _observation(mixed_key, label="mixed-env"),
    )
    # Plan construction itself rejects heterogeneous tree/environment keys.
    with pytest.raises(VerificationIdentityError, match="plan identities"):
        _plan(key, mixed_key)
    # A mixed-environment leaf is a distinct key and is therefore outside the
    # sealed required set; the builder also re-checks tree/environment
    # homogeneity for any admitted leaf that survives membership.
    with pytest.raises(
        VerificationIdentityError,
        match="required check set|mixed repository tree or environment",
    ):
        build_verification_bundle(
            _plan(key),
            executed_receipts=(receipt, mixed_receipt),
        )

    # Defensive homogeneity check: plan identity mutation against sealed keys
    # fails closed before any mixed leaf can be admitted.
    valid = build_verification_bundle(_plan(key), executed_receipts=(receipt,))
    with pytest.raises(VerificationIdentityError, match="plan identities"):
        replace(
            valid.verification_plan,
            repository_tree_cid=_artifact("foreign-tree"),
        )

    # Direct leaf homogeneity guard used by the builder/commitment path.
    import ipfs_accelerate_py.agent_supervisor.verification.bundle as bundle_mod

    with pytest.raises(
        VerificationIdentityError,
        match="mixed repository tree or environment",
    ):
        bundle_mod._reject_mixed_tree_or_environment(_plan(key), (mixed_receipt,))


def test_build_bundle_rejects_outside_required_set_and_duplicate_keys() -> None:
    key = _key()
    foreign = _key(tool_version="other")
    foreign_receipt = _passing_type(foreign, label="foreign")
    with pytest.raises(VerificationIdentityError, match="required check set"):
        build_verification_bundle(
            _plan(key),
            executed_receipts=(foreign_receipt,),
        )

    first = _passing_type(key, label="a")
    second = _passing_type(key, label="b")
    with pytest.raises(VerificationBundleBuildError, match="more than one result"):
        build_verification_bundle(
            _plan(key),
            executed_receipts=(first, second),
        )


def test_build_bundle_cannot_downgrade_plan_human_review() -> None:
    key = _key()
    receipt = _passing_type(key, label="review")
    review_plan = replace(
        _plan(key),
        human_review_required=True,
        human_review_reason_codes=("policy_authority_unresolved",),
    )
    with pytest.raises(VerificationBundleBuildError, match="human review"):
        build_verification_bundle(
            review_plan,
            executed_receipts=(receipt,),
            human_review_required=False,
        )
    bundle = build_verification_bundle(
        review_plan,
        executed_receipts=(receipt,),
    )
    assert bundle.human_review_required is True


def test_build_bundle_classifies_missing_required_as_unresolved() -> None:
    key = _key()
    bundle = build_verification_bundle(_plan(key))
    assert bundle.receipts == ()
    assert bundle.unresolved_requirement_ids == (key.key_id,)
    assert classify_bundle_receipts(bundle).unresolved_requirement_ids == (key.key_id,)


def test_build_bundle_binds_counterexamples_to_failed_executed_receipts() -> None:
    key = _key()
    failed = _failed_type(key, label="cx-fail")
    cx = _counterexample(key, failed)
    bundle = build_verification_bundle(
        _plan(key),
        executed_receipts=(failed,),
        counterexamples=(cx,),
    )
    assert bundle.counterexamples == (cx,)
    assert not bundle.structurally_complete


# ---------------------------------------------------------------------------
# Compact summary
# ---------------------------------------------------------------------------


def test_summary_includes_cone_checks_failures_counterexamples_fallback_timing_route() -> (
    None
):
    # All required keys must share tree/environment; full-suite keys must be
    # test receipts.  Use one environment-homogeneous TEST plan.
    first = _key(VerificationReceiptKind.TEST)
    second = _key(VerificationReceiptKind.TEST, receipt_schema_version=2)
    full_suite_key = _key(VerificationReceiptKind.TEST, receipt_schema_version=3)
    assert first.environment_cid == second.environment_cid == full_suite_key.environment_cid

    reused = TestReceipt(first, _observation(first, TerminalStatus.PASSED, label="sum-reused"))
    failed = TestReceipt(second, _observation(second, TerminalStatus.FAILED, label="sum-fail"))
    cx = _counterexample(second, failed, label="sum-cx")
    pending_plan = replace(
        _plan(first, second, full_suite_key),
        cache_reuse_decisions=(
            CacheReuseDecision(
                key_cid=first.key_id,
                disposition=CacheReuseDisposition.REUSED,
                reason_codes=("exact_current_production_receipt",),
                candidate_receipt=reused,
            ),
            CacheReuseDecision(
                key_cid=second.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
            CacheReuseDecision(
                key_cid=full_suite_key.key_id,
                disposition=CacheReuseDisposition.MISSING,
                reason_codes=("cache_miss",),
            ),
        ),
        affected_tests=("test/test_example.py::test_calculate",),
        full_suite_required=True,
        full_suite_receipt_key_cids=(full_suite_key.key_id,),
        full_suite_reason_codes=("selector_uncertain",),
        required_type_checks=(),
        required_static_checks=(),
        expected_proof_slots=0,
    )
    bundle = build_verification_bundle(
        pending_plan,
        reused_receipts=(reused,),
        executed_receipts=(failed,),
        counterexamples=(cx,),
    )
    assert bundle.mandatory_fallback_pending is True

    route = ModelRouteDecision(
        route=ModelRoute.SMALL_LOCAL_MODEL,
        considered_routes=(ModelRoute.SMALL_LOCAL_MODEL,),
        decisive_reason_codes=("localized_exact_counterexample",),
        required_capabilities=("exact_contracts", "bounded_context"),
        context_token_estimate=1_024,
        policy_cid=_artifact("summary-route-policy"),
    )
    summary = build_verification_summary(
        bundle,
        route,
        dependency_cone_symbols=("example.calculate", "example.helper"),
        selected_tests=("test/test_example.py::test_calculate",),
        verification_wall_time_ms=250,
        reused_time_saved_ms=125,
    )

    assert isinstance(summary, VerificationSummary)
    assert summary.repository_tree_cid == bundle.repository_tree_cid
    assert summary.environment_cid == bundle.environment_cid
    assert first.affected_symbol_version_cids[0] in summary.changed_symbol_version_cids
    assert summary.dependency_cone_symbols == (
        "example.calculate",
        "example.helper",
    )
    assert summary.selected_tests == ("test/test_example.py::test_calculate",)
    assert summary.reused_check_key_cids == (first.key_id,)
    assert summary.executed_check_key_cids == (second.key_id,)
    assert summary.failure_receipt_cids == (failed.receipt_id,)
    assert summary.counterexample_cids == (cx.counterexample_id,)
    assert summary.full_suite_pending is True
    assert summary.human_review_required is False
    assert summary.verification_wall_time_ms == 250
    assert summary.reused_time_saved_ms == 125
    assert summary.counterexample_context_tokens > 0
    assert summary.aggregate_terminal_status is TerminalStatus.FAILED
    assert summary.model_route_decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert summary.policy_cid == bundle.policy_cid
    assert len(summary.canonical_bytes()) <= 262_144
    assert VerificationSummary.from_dict(summary.to_record()) == summary


def test_summary_defaults_timing_from_receipt_durations_and_rejects_route_downgrade() -> (
    None
):
    key = _key()
    receipt = _passing_type(key, label="timed")
    review_plan = replace(
        _plan(key),
        human_review_required=True,
        human_review_reason_codes=("policy_authority_unresolved",),
    )
    bundle = build_verification_bundle(
        review_plan,
        executed_receipts=(receipt,),
    )
    human_route = _route(human=True)
    summary = build_verification_summary(bundle, human_route)
    assert summary.verification_wall_time_ms == receipt.execution.duration_ms
    assert summary.reused_time_saved_ms == 0
    assert summary.human_review_required is True
    assert summary.model_route_decision.requires_human_review is True

    small_route = _route(human=False)
    with pytest.raises(VerificationBundleBuildError, match="human review"):
        build_verification_summary(bundle, small_route)


def test_summary_token_estimate_is_compact_and_excludes_raw_logs() -> None:
    key = _key()
    failed = _failed_type(key, label="token-fail")
    cx = _counterexample(key, failed, label="token-cx")
    bundle = build_verification_bundle(
        _plan(key),
        executed_receipts=(failed,),
        counterexamples=(cx,),
    )
    summary = build_verification_summary(bundle, _route())
    # Token estimate is derived from compact counterexample canonical bytes.
    expected = (
        len(cx.canonical_bytes()) + 3
    ) // 4
    assert summary.counterexample_context_tokens == expected
    # Public summary payload never embeds raw log text.
    payload = summary.to_record()
    blob = str(payload)
    assert "stdout" not in blob
    assert "secret" not in blob


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------


def test_commitment_binds_schema_hash_codec_domains_and_public_statement() -> None:
    key = _key()
    receipt = _passing_type(key, label="commit-pass")
    bundle = build_verification_bundle(_plan(key), executed_receipts=(receipt,))
    commitment = build_verification_commitment(bundle)

    assert commitment.HASH_ALGORITHM == COMMITMENT_HASH_ALGORITHM == "sha2-256"
    assert commitment.LEAF_CODEC == COMMITMENT_LEAF_CODEC == "canonical-dag-json@1"
    assert commitment.LEAF_DOMAIN == COMMITMENT_LEAF_DOMAIN == "IVP-LEAF@1"
    assert commitment.NODE_DOMAIN == COMMITMENT_NODE_DOMAIN == "IVP-NODE@1"
    assert commitment.EMPTY_DOMAIN == COMMITMENT_EMPTY_DOMAIN == "IVP-EMPTY@1"
    assert COMMITMENT_IS_ZERO_KNOWLEDGE_PROOF is False
    assert commitment.IS_ZERO_KNOWLEDGE_PROOF is False
    assert commitment.repository_tree_cid == bundle.repository_tree_cid
    assert commitment.environment_cid == bundle.environment_cid
    assert commitment.required_check_key_cids == bundle.required_check_key_cids
    assert commitment.unresolved_obligation_count == 0
    assert commitment.aggregate_terminal_status is TerminalStatus.PASSED
    statement = commitment.public_statement
    assert statement["repository_tree_cid"] == bundle.repository_tree_cid
    assert statement["environment_cid"] == bundle.environment_cid
    assert statement["required_check_set_cid"] == commitment.required_check_set_cid
    assert statement["unresolved_obligation_count"] == 0
    assert statement["aggregate_terminal_status"] == TerminalStatus.PASSED.value
    assert commitment.merkle_root == _manual_merkle_root(
        [
            {
                "key_cid": receipt.key.key_id,
                "receipt_cid": receipt.receipt_id,
                "receipt_kind": receipt.key.receipt_kind.value,
                "status": receipt.status.value,
            }
        ]
    )
    assert VerificationCommitment.from_dict(commitment.to_record()) == commitment


def test_commitment_odd_node_promotion_and_empty_domain() -> None:
    keys = (
        _key(),
        _key(receipt_schema_version=2),
        _key(receipt_schema_version=3),
    )
    receipts = tuple(
        _passing_type(key, label=f"odd-{index}") for index, key in enumerate(keys)
    )
    bundle = build_verification_bundle(
        _plan(*keys),
        executed_receipts=receipts,
    )
    commitment = build_verification_commitment(bundle)
    leaves = [
        {
            "key_cid": item.key.key_id,
            "receipt_cid": item.receipt_id,
            "receipt_kind": item.key.receipt_kind.value,
            "status": item.status.value,
        }
        for item in receipts
    ]
    assert commitment.merkle_root == _manual_merkle_root(leaves)

    empty = build_verification_bundle(_plan(keys[0]))
    empty_commitment = build_verification_commitment(empty)
    assert empty_commitment.admitted_leaves == ()
    assert empty_commitment.merkle_root == _manual_merkle_root([])
    assert empty_commitment.aggregate_terminal_status is TerminalStatus.UNKNOWN


def test_commitment_permutation_invariant_membership_and_content_sensitive() -> None:
    first = _key()
    second = _key(receipt_schema_version=2)
    first_receipt = _passing_type(first, label="perm-a")
    second_receipt = _passing_type(second, label="perm-b")

    forward = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt, second_receipt),
        )
    )
    reverse = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(second_receipt, first_receipt),
        )
    )
    assert forward.merkle_root == reverse.merkle_root
    assert forward.commitment_id == reverse.commitment_id
    assert forward.required_check_set_cid == reverse.required_check_set_cid

    # Content change of a required admitted leaf changes the commitment.
    failed_second = _failed_type(second, label="perm-fail")
    changed = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt, failed_second),
        )
    )
    assert changed.merkle_root != forward.merkle_root
    assert changed.commitment_id != forward.commitment_id
    assert changed.required_check_set_cid == forward.required_check_set_cid

    # Membership change (narrower required set) changes required-check-set and root.
    narrower = build_verification_commitment(
        build_verification_bundle(
            _plan(first),
            executed_receipts=(first_receipt,),
        )
    )
    assert narrower.required_check_set_cid != forward.required_check_set_cid
    assert narrower.merkle_root != forward.merkle_root

    # Removing an admitted required leaf (unresolved) changes commitment root.
    unresolved = build_verification_commitment(
        build_verification_bundle(
            _plan(first, second),
            executed_receipts=(first_receipt,),
        )
    )
    assert unresolved.merkle_root != forward.merkle_root
    assert unresolved.aggregate_terminal_status is TerminalStatus.UNKNOWN


def test_commitment_fail_closed_aggregate_lattice_cannot_upgrade_leaves() -> None:
    key = _key()
    for adverse in (
        TerminalStatus.INVALID,
        TerminalStatus.STALE,
        TerminalStatus.SIMULATED,
        TerminalStatus.CANCELLED,
        TerminalStatus.TIMEOUT,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.UNKNOWN,
        TerminalStatus.NOT_MODELED,
        TerminalStatus.FAILED,
    ):
        receipt = TypeCheckReceipt(
            key,
            _observation(key, adverse, label=f"agg-{adverse.value}"),
        )
        commitment = build_verification_commitment(
            build_verification_bundle(_plan(key), executed_receipts=(receipt,))
        )
        assert commitment.aggregate_terminal_status is adverse

    # Mixed success + adverse collapses to the worse fail-closed status.
    assert (
        aggregate_terminal_status((TerminalStatus.PASSED, TerminalStatus.TIMEOUT))
        is TerminalStatus.TIMEOUT
    )
    assert (
        aggregate_terminal_status((TerminalStatus.PROVED, TerminalStatus.PASSED))
        is TerminalStatus.PASSED
    )
    assert aggregate_terminal_status((TerminalStatus.PROVED,)) is TerminalStatus.PROVED


def test_commitment_and_docs_state_non_claims_explicitly() -> None:
    import ipfs_accelerate_py.agent_supervisor.verification.bundle as bundle_module
    import re

    module_doc = bundle_module.__doc__ or ""
    # Markdown emphasis may wrap "not" as **not**; match either form.
    assert re.search(
        r"(\*\*)?not(\*\*)?\s+a zero-knowledge proof", module_doc, re.I
    )
    assert "trusted" in module_doc.lower()
    assert "structural validation" in module_doc.lower()
    assert re.search(
        r"(\*\*)?not(\*\*)?\s+cryptographic validation", module_doc, re.I
    )

    commitment_doc = inspect.getdoc(build_verification_commitment) or ""
    assert re.search(r"not a ZK proof|not a zero-knowledge", commitment_doc, re.I)
    assert "trusted" in commitment_doc.lower()
    assert "structural validation" in commitment_doc.lower()

    # Class-level non-claim on the sealed contract.
    class_doc = inspect.getdoc(VerificationCommitment) or ""
    assert "not a zero-knowledge proof" in class_doc.lower()
    assert "trusted" in class_doc.lower()
    assert re.search(
        r"structural validation\s+is not cryptographic validation",
        class_doc,
        re.I,
    )

    assert VERIFICATION_SUMMARY_EVIDENCE == "ivp/verification-summary@1"
    assert VERIFICATION_COMMITMENT_EVIDENCE == "ivp/verification-commitment@1"
    assert VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF is False


def test_build_commitment_rejects_non_bundle_input() -> None:
    with pytest.raises(VerificationBundleBuildError, match="VerificationBundle"):
        build_verification_commitment({})  # type: ignore[arg-type]


def test_end_to_end_bundle_summary_commitment_pipeline() -> None:
    key = _key()
    receipt = _passing_type(key, label="e2e")
    bundle = build_verification_bundle(_plan(key), executed_receipts=(receipt,))
    summary = build_verification_summary(bundle, _route())
    commitment = build_verification_commitment(bundle)

    assert summary.aggregate_terminal_status is commitment.aggregate_terminal_status
    assert summary.repository_tree_cid == commitment.repository_tree_cid
    assert summary.executed_check_key_cids == (key.key_id,)
    assert classify_bundle_receipts(bundle).executed_receipt_cids == (
        receipt.receipt_id,
    )
    assert bundle.bundle_id
    assert summary.summary_id
    assert commitment.commitment_id
    assert commitment.merkle_root.startswith("sha256:")

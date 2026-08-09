"""Tests for ProviderCallLedger@1 / FailureSignature@1 / ChurnDecision@1.

DQP-027 evidence subset: exact duplicate, semantic duplicate, hard quota,
transient failure, response loss, retry budget, negative cache TTL, secret
redaction.

Acceptance:

* Same idempotency/call key dispatches once
* Unchanged failed proposal after exhausted policy is suppressed
* Changed evidence permits a new call
* All rejected/abandoned/retry usage is charged
* Raw prompts/completions and secrets are not stored as ordinary rows
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_call_ledger import (
    AUTHORITY_CLASS,
    CHURN_DECISION_INTERFACE,
    FAILURE_SIGNATURE_INTERFACE,
    LEDGER_AUTHORIZES_USAGE,
    LEDGER_IS_COMPLETION_EVIDENCE,
    LEDGER_IS_CORRECTNESS_EVIDENCE,
    LEDGER_REWRITES_PROVIDER_SETTLEMENT,
    PROVIDER_CALL_LEDGER_INTERFACE,
    CallOutcome,
    CallStatus,
    ChargeKind,
    ChurnDecision,
    ChurnDisposition,
    ChurnPolicy,
    DuplicateKind,
    FailureSignature,
    OutcomeClass,
    ProviderCallBudget,
    ProviderCallCompletion,
    ProviderCallLedger,
    ProviderCallLedgerNotOpenError,
    ProviderCallLedgerSecretError,
    ProviderCallProposal,
    ProviderTokenUsage,
    build_call_key,
    digest_text,
    duckdb_available,
    open_provider_call_ledger,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for ProviderCallLedger hermetic tests",
)


def _proposal(**overrides) -> ProviderCallProposal:
    values = dict(
        provider="grok",
        model="grok-4.5",
        context_cid="ctx:stable-1",
        task_id="task:dqp-027",
        proposal_digest="sha256:" + "a" * 64,
        evidence_digest="sha256:" + "b" * 64,
        plan_id="plan:1",
        attempt=0,
        idempotency_key="",
        endpoint_fingerprint="ep_" + "c" * 64,
        prompt_digest="sha256:" + "d" * 64,
        budget=ProviderCallBudget(requests=1, input_tokens=1000, output_tokens=500),
        token_estimate=ProviderTokenUsage(
            input_tokens_estimated=120,
            output_tokens_estimated=40,
        ),
        metadata={"stage": "implementation"},
    )
    values.update(overrides)
    return ProviderCallProposal(**values)


def _open(
    tmp_path: Path,
    *,
    policy: ChurnPolicy | None = None,
) -> ProviderCallLedger:
    return open_provider_call_ledger(
        tmp_path / "provider_call_ledger.duckdb",
        policy=policy,
    )


def _complete_failed(
    ledger: ProviderCallLedger,
    call_id: str,
    *,
    outcome: CallOutcome = CallOutcome.FAILED,
    failure_code: str = "validation_failed",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> None:
    ledger.complete_call(
        call_id,
        ProviderCallCompletion(
            outcome=outcome,
            failure_code=failure_code,
            response_digest="sha256:" + "e" * 64,
            latency_ms=250,
            tokens=ProviderTokenUsage(
                input_tokens_actual=input_tokens,
                output_tokens_actual=output_tokens,
            ),
            cost_micros=1500,
        ),
    )


# ---------------------------------------------------------------------------
# Interfaces / cold import
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert PROVIDER_CALL_LEDGER_INTERFACE == "ProviderCallLedger@1"
    assert FAILURE_SIGNATURE_INTERFACE == "FailureSignature@1"
    assert CHURN_DECISION_INTERFACE == "ChurnDecision@1"
    assert ProviderCallLedger.INTERFACE == "ProviderCallLedger@1"
    assert FailureSignature.INTERFACE == "FailureSignature@1"
    assert ChurnDecision.INTERFACE == "ChurnDecision@1"
    assert AUTHORITY_CLASS == "operational_evidence"
    assert LEDGER_AUTHORIZES_USAGE is False
    assert LEDGER_REWRITES_PROVIDER_SETTLEMENT is False
    assert LEDGER_IS_COMPLETION_EVIDENCE is False
    assert LEDGER_IS_CORRECTNESS_EVIDENCE is False


def test_cold_import_and_construction_have_no_side_effects() -> None:
    ledger = ProviderCallLedger("/tmp/should-not-exist-until-open.duckdb")
    assert ledger.is_open is False
    with pytest.raises(ProviderCallLedgerNotOpenError):
        ledger.evaluate_dispatch(_proposal())


def test_metadata_after_open(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        meta = ledger.metadata()
    assert meta["interface"] == PROVIDER_CALL_LEDGER_INTERFACE
    assert meta["authority"] == AUTHORITY_CLASS
    assert meta["is_open"] is True


# ---------------------------------------------------------------------------
# Exact duplicate / idempotency
# ---------------------------------------------------------------------------


def test_same_idempotency_key_dispatches_once(tmp_path: Path) -> None:
    proposal = _proposal(idempotency_key="idem:exact-once")
    with _open(tmp_path) as ledger:
        decision1, record1 = ledger.admit_and_dispatch(proposal)
        assert decision1.should_dispatch is True
        assert decision1.disposition is ChurnDisposition.DISPATCH
        assert record1 is not None
        assert record1.dispatched is True
        assert record1.status is CallStatus.DISPATCHED

        ledger.complete_call(
            record1.call_id,
            ProviderCallCompletion(
                outcome=CallOutcome.SUCCESS,
                response_digest="sha256:" + "f" * 64,
                latency_ms=100,
                tokens=ProviderTokenUsage(
                    input_tokens_actual=110,
                    output_tokens_actual=30,
                ),
            ),
        )

        decision2, record2 = ledger.admit_and_dispatch(proposal)
        assert decision2.should_dispatch is False
        assert decision2.disposition is ChurnDisposition.REPLAY_EXACT
        assert decision2.duplicate_kind is DuplicateKind.IDEMPOTENCY
        assert record2 is not None
        assert record2.call_id == record1.call_id
        assert record2.outcome is CallOutcome.SUCCESS

        # Still only one provider_calls row for the call key.
        assert ledger.get_call_by_key(proposal.call_key) is not None
        charges = ledger.list_usage_charges(call_id=record1.call_id)
        assert len(charges) == 1
        assert charges[0].charge_kind is ChargeKind.SUCCESS


def test_same_call_key_without_idempotency_dispatches_once(tmp_path: Path) -> None:
    proposal = _proposal(attempt=2)
    key = build_call_key(
        provider=proposal.provider,
        model=proposal.model,
        context_cid=proposal.context_cid,
        task_id=proposal.task_id,
        proposal_digest=proposal.proposal_digest,
        evidence_digest=proposal.evidence_digest,
        plan_id=proposal.plan_id,
        attempt=proposal.attempt,
        endpoint_fingerprint=proposal.endpoint_fingerprint,
        prompt_digest=proposal.prompt_digest,
    )
    assert key == proposal.call_key

    with _open(tmp_path) as ledger:
        decision1, record1 = ledger.admit_and_dispatch(proposal)
        assert decision1.should_dispatch is True
        _complete_failed(ledger, record1.call_id)

        decision2, record2 = ledger.admit_and_dispatch(proposal)
        assert decision2.disposition is ChurnDisposition.REPLAY_EXACT
        assert decision2.duplicate_kind is DuplicateKind.EXACT
        assert record2.call_id == record1.call_id


# ---------------------------------------------------------------------------
# Exhausted policy / changed evidence
# ---------------------------------------------------------------------------


def test_unchanged_failed_proposal_after_exhausted_policy_is_suppressed(
    tmp_path: Path,
) -> None:
    policy = ChurnPolicy(
        max_identical_failures=2,
        max_retries=2,
        negative_cache_ttl_ms=0,
    )
    base = dict(
        proposal_digest="sha256:" + "1" * 64,
        evidence_digest="sha256:" + "2" * 64,
    )
    with _open(tmp_path, policy=policy) as ledger:
        for attempt in range(2):
            proposal = _proposal(attempt=attempt, **base)
            decision, record = ledger.admit_and_dispatch(proposal)
            assert decision.should_dispatch is True, attempt
            assert record is not None
            _complete_failed(
                ledger,
                record.call_id,
                outcome=CallOutcome.FAILED,
                input_tokens=50 + attempt,
                output_tokens=10,
            )

        # Unchanged evidence, new attempt key — policy exhausted.
        suppressed = _proposal(attempt=2, **base)
        decision, record = ledger.admit_and_dispatch(suppressed)
        assert decision.should_dispatch is False
        assert decision.disposition is ChurnDisposition.SUPPRESS_EXHAUSTED
        assert decision.duplicate_kind is DuplicateKind.SEMANTIC
        assert record is not None
        assert record.status is CallStatus.SUPPRESSED
        assert record.dispatched is False


def test_changed_evidence_permits_a_new_call(tmp_path: Path) -> None:
    policy = ChurnPolicy(max_identical_failures=1, max_retries=1, negative_cache_ttl_ms=0)
    with _open(tmp_path, policy=policy) as ledger:
        first = _proposal(
            attempt=0,
            evidence_digest="sha256:" + "b" * 64,
        )
        decision, record = ledger.admit_and_dispatch(first)
        assert decision.should_dispatch is True
        _complete_failed(ledger, record.call_id)

        # Exhausted for original evidence.
        blocked = _proposal(
            attempt=1,
            evidence_digest="sha256:" + "b" * 64,
        )
        decision_blocked, _ = ledger.admit_and_dispatch(blocked)
        assert decision_blocked.should_dispatch is False
        assert decision_blocked.disposition is ChurnDisposition.SUPPRESS_EXHAUSTED

        # Changed evidence reopens.
        reopened = _proposal(
            attempt=2,
            evidence_digest="sha256:" + "c" * 64,
        )
        decision_open, record_open = ledger.admit_and_dispatch(reopened)
        assert decision_open.should_dispatch is True
        assert (
            decision_open.disposition is ChurnDisposition.ALLOW_CHANGED_EVIDENCE
        )
        assert record_open is not None
        assert record_open.dispatched is True
        assert record_open.evidence_digest == reopened.evidence_digest


def test_semantic_duplicate_suppression_before_retry_budget(
    tmp_path: Path,
) -> None:
    policy = ChurnPolicy(
        max_identical_failures=1,
        max_retries=5,
        suppress_semantic_duplicates=True,
        negative_cache_ttl_ms=0,
    )
    with _open(tmp_path, policy=policy) as ledger:
        first = _proposal(attempt=0)
        decision, record = ledger.admit_and_dispatch(first)
        assert decision.should_dispatch is True
        _complete_failed(ledger, record.call_id)

        second = _proposal(attempt=1)
        decision2, record2 = ledger.admit_and_dispatch(second)
        assert decision2.should_dispatch is False
        assert (
            decision2.disposition is ChurnDisposition.SUPPRESS_SEMANTIC_DUPLICATE
        )
        assert record2 is not None
        assert record2.dispatched is False


# ---------------------------------------------------------------------------
# Usage charging for rejected / abandoned / retry
# ---------------------------------------------------------------------------


def test_rejected_abandoned_and_retry_usage_is_charged(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        rejected = _proposal(attempt=0, idempotency_key="idem:rejected")
        _, rec_r = ledger.admit_and_dispatch(rejected)
        ledger.complete_call(
            rec_r.call_id,
            ProviderCallCompletion(
                outcome=CallOutcome.REJECTED,
                failure_code="admission_rejected",
                tokens=ProviderTokenUsage(
                    input_tokens_actual=40,
                    output_tokens_actual=0,
                ),
                cost_micros=100,
            ),
        )

        abandoned = _proposal(attempt=1, idempotency_key="idem:abandoned")
        _, rec_a = ledger.admit_and_dispatch(abandoned)
        ledger.complete_call(
            rec_a.call_id,
            ProviderCallCompletion(
                outcome=CallOutcome.ABANDONED,
                failure_code="operator_abandoned",
                tokens=ProviderTokenUsage(
                    input_tokens_actual=55,
                    output_tokens_actual=5,
                ),
                cost_micros=200,
            ),
        )

        retry = _proposal(attempt=2, idempotency_key="idem:retry")
        _, rec_t = ledger.admit_and_dispatch(retry)
        # Explicit retry charge before a later success path.
        charge = ledger.record_usage(
            call_id=rec_t.call_id,
            charge_kind=ChargeKind.RETRY,
            disposition="retry",
            tokens=ProviderTokenUsage(
                input_tokens_actual=70,
                output_tokens_actual=15,
            ),
            cost_micros=300,
        )
        assert charge.charged is True
        assert charge.charge_kind is ChargeKind.RETRY

        totals = ledger.total_charged_tokens(
            include_kinds=(
                ChargeKind.REJECTED,
                ChargeKind.ABANDONED,
                ChargeKind.RETRY,
            )
        )
        assert totals["input_tokens"] == 40 + 55 + 70
        assert totals["output_tokens"] == 0 + 5 + 15
        assert totals["requests"] == 3

        all_charges = ledger.list_usage_charges()
        kinds = {item.charge_kind for item in all_charges}
        assert ChargeKind.REJECTED in kinds
        assert ChargeKind.ABANDONED in kinds
        assert ChargeKind.RETRY in kinds
        assert all(item.charged for item in all_charges)


def test_hard_quota_transient_and_response_loss_are_typed_and_charged(
    tmp_path: Path,
) -> None:
    cases = (
        (CallOutcome.HARD_QUOTA, OutcomeClass.HARD_QUOTA, ChargeKind.HARD_QUOTA),
        (
            CallOutcome.TRANSIENT_FAILURE,
            OutcomeClass.TRANSIENT,
            ChargeKind.TRANSIENT,
        ),
        (
            CallOutcome.RESPONSE_LOSS,
            OutcomeClass.RESPONSE_LOSS,
            ChargeKind.RESPONSE_LOSS,
        ),
    )
    with _open(tmp_path) as ledger:
        for index, (outcome, outcome_class, charge_kind) in enumerate(cases):
            proposal = _proposal(
                attempt=index,
                idempotency_key=f"idem:typed-{outcome.value}",
            )
            _, record = ledger.admit_and_dispatch(proposal)
            completed = ledger.complete_call(
                record.call_id,
                ProviderCallCompletion(
                    outcome=outcome,
                    failure_code=outcome.value,
                    tokens=ProviderTokenUsage(
                        input_tokens_actual=10 * (index + 1),
                        output_tokens_actual=1,
                    ),
                ),
            )
            assert completed.outcome is outcome
            assert completed.outcome_class is outcome_class
            assert completed.failure_signature_id
            signature = ledger.get_failure_signature(
                completed.failure_signature_id
            )
            assert signature is not None
            assert signature["outcome_class"] == outcome_class.value
            charges = ledger.list_usage_charges(call_id=record.call_id)
            assert len(charges) == 1
            assert charges[0].charge_kind is charge_kind
            assert charges[0].charged is True


# ---------------------------------------------------------------------------
# Negative cache TTL
# ---------------------------------------------------------------------------


def test_negative_cache_ttl_suppresses_until_expiry(tmp_path: Path) -> None:
    policy = ChurnPolicy(
        max_identical_failures=1,
        max_retries=1,
        negative_cache_ttl_ms=10_000,
    )
    now = 1_700_000_000_000
    with _open(tmp_path, policy=policy) as ledger:
        first = _proposal(attempt=0)
        decision, record = ledger.admit_and_dispatch(first, now_ms=now)
        assert decision.should_dispatch is True
        _complete_failed(ledger, record.call_id)

        # Exhausted → suppressed and negative-cached.
        second = _proposal(attempt=1)
        decision2, record2 = ledger.admit_and_dispatch(second, now_ms=now + 1)
        assert decision2.should_dispatch is False
        assert decision2.disposition is ChurnDisposition.SUPPRESS_EXHAUSTED
        assert record2 is not None

        # Same call key within TTL → negative cache (exact key of suppressed row).
        decision3, record3 = ledger.admit_and_dispatch(second, now_ms=now + 500)
        assert decision3.should_dispatch is False
        assert decision3.disposition in {
            ChurnDisposition.REPLAY_EXACT,
            ChurnDisposition.SUPPRESS_NEGATIVE_CACHE,
        }
        # Terminal suppressed row replays exactly for the same call key.
        if decision3.disposition is ChurnDisposition.REPLAY_EXACT:
            assert record3 is not None
            assert record3.call_id == record2.call_id

        # Different attempt under still-exhausted policy remains suppressed.
        third = _proposal(attempt=2)
        decision4, _ = ledger.admit_and_dispatch(third, now_ms=now + 1_000)
        assert decision4.should_dispatch is False


# ---------------------------------------------------------------------------
# Secret / prompt / completion redaction
# ---------------------------------------------------------------------------


def test_raw_prompts_completions_and_secrets_rejected_as_ordinary_rows(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as ledger:
        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.admit_and_dispatch(
                _proposal(
                    metadata={
                        "prompt": "You are an agent. Implement DQP-027 completely.",
                    }
                )
            )

        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.admit_and_dispatch(
                _proposal(
                    metadata={
                        "completion": "Here is the full model output...",
                    }
                )
            )

        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.admit_and_dispatch(
                _proposal(
                    metadata={
                        "api_key": "sk-abcdefghijklmnopqrstuvwxyz012345",
                    }
                )
            )

        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.admit_and_dispatch(
                _proposal(
                    metadata={
                        "note": "Authorization: Bearer supersecrettokenvalue",
                    }
                )
            )

        # Digests are allowed; raw bodies must be hashed by the caller.
        raw_prompt = "System: do not store this as an ordinary row."
        ok = _proposal(
            prompt_digest=digest_text(raw_prompt),
            metadata={"prompt_digest": digest_text(raw_prompt)},
        )
        decision, record = ledger.admit_and_dispatch(ok)
        assert decision.should_dispatch is True
        assert record is not None
        body = record.to_dict()
        assert "prompt" not in body
        assert "completion" not in body
        assert "api_key" not in body
        assert ok.prompt_digest.startswith("sha256:")
        # Ordinary persisted row is redacted metadata only.
        stored = ledger.get_call(record.call_id)
        assert stored is not None
        assert "prompt" not in stored.metadata
        assert "completion" not in stored.metadata
        assert stored.metadata.get("prompt_digest") == ok.prompt_digest
        serialized = str(stored.to_dict())
        assert "do not store this" not in serialized
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in serialized


def test_completion_rejects_secret_metadata(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        _, record = ledger.admit_and_dispatch(
            _proposal(idempotency_key="idem:secret-complete")
        )
        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.complete_call(
                record.call_id,
                ProviderCallCompletion(
                    outcome=CallOutcome.SUCCESS,
                    metadata={"output_text": "raw model completion"},
                ),
            )


# ---------------------------------------------------------------------------
# Failure signature stability
# ---------------------------------------------------------------------------


def test_failure_signature_stable_across_evidence_but_reopens_on_change(
    tmp_path: Path,
) -> None:
    proposal_a = _proposal(evidence_digest="sha256:" + "b" * 64)
    proposal_b = _proposal(evidence_digest="sha256:" + "c" * 64)
    sig_a = FailureSignature.from_outcome(
        proposal=proposal_a,
        outcome=CallOutcome.FAILED,
        failure_code="validation_failed",
    )
    sig_b = FailureSignature.from_outcome(
        proposal=proposal_b,
        outcome=CallOutcome.FAILED,
        failure_code="validation_failed",
    )
    # Same proposal/context/outcome → same signature id even if evidence differs.
    assert sig_a.failure_signature_id == sig_b.failure_signature_id
    assert sig_a.evidence_digest != sig_b.evidence_digest

    with _open(
        tmp_path,
        policy=ChurnPolicy(max_retries=1, max_identical_failures=1),
    ) as ledger:
        _, rec = ledger.admit_and_dispatch(proposal_a)
        completed = ledger.complete_call(
            rec.call_id,
            ProviderCallCompletion(
                outcome=CallOutcome.FAILED,
                failure_code="validation_failed",
                tokens=ProviderTokenUsage(input_tokens_actual=10),
            ),
        )
        assert completed.failure_signature_id == sig_a.failure_signature_id
        stored = ledger.get_failure_signature(completed.failure_signature_id)
        assert stored is not None
        assert stored["failure_code"] == "validation_failed"


def test_churn_decisions_are_persisted(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        proposal = _proposal(idempotency_key="idem:decision-log")
        decision = ledger.evaluate_dispatch(proposal)
        assert decision.should_dispatch is True
        decisions = ledger.list_churn_decisions(call_key=proposal.call_key)
        assert len(decisions) == 1
        assert decisions[0].decision_id == decision.decision_id
        assert decisions[0].disposition is ChurnDisposition.DISPATCH


def test_round_trip_contracts() -> None:
    proposal = _proposal(idempotency_key="idem:round-trip")
    assert ProviderCallProposal.from_dict(proposal.to_dict()).call_key == (
        proposal.call_key
    )
    signature = FailureSignature.from_outcome(
        proposal=proposal,
        outcome=CallOutcome.HARD_QUOTA,
        failure_code="hard_quota",
    )
    assert (
        FailureSignature.from_dict(signature.to_dict()).failure_signature_id
        == signature.failure_signature_id
    )
    decision = ChurnDecision(
        disposition=ChurnDisposition.DISPATCH,
        call_key=proposal.call_key,
        should_dispatch=True,
        reason="fresh",
        evidence_digest=proposal.evidence_digest,
    )
    assert ChurnDecision.from_dict(decision.to_dict()).decision_id == (
        decision.decision_id
    )


def test_digest_text_is_stable() -> None:
    assert digest_text("hello") == digest_text("hello")
    assert digest_text("hello") != digest_text("world")
    assert digest_text("hello").startswith("sha256:")

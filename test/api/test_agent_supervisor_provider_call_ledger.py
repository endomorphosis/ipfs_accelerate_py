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
    DEFAULT_POLICY_ID,
    FAILURE_SIGNATURE_INTERFACE,
    PROVIDER_CALL_LEDGER_INTERFACE,
    REDACTION_MARKER,
    ChurnAction,
    DuplicateKind,
    FailureClass,
    ProviderCallLedger,
    ProviderCallLedgerSecretError,
    ProviderCallOutcome,
    ProviderCallRequest,
    compute_call_key,
    compute_prompt_digest,
    duckdb_available,
    open_provider_call_ledger,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for ProviderCallLedger hermetic tests",
)


def _open(tmp_path: Path, **kwargs) -> ProviderCallLedger:
    return open_provider_call_ledger(tmp_path / "provider_calls.duckdb", **kwargs)


def _request(**overrides) -> ProviderCallRequest:
    values = dict(
        provider_id="provider:demo",
        model_id="model:demo",
        endpoint_id="endpoint:scoped-1",
        context_cid="context:cid-1",
        plan_cid="plan:1",
        task_cid="task:dqp-027",
        attempt_id="attempt:1",
        policy_id=DEFAULT_POLICY_ID,
        evidence_digest="sha256:" + ("ab" * 32),
        prompt_digest="sha256:" + ("cd" * 32),
        idempotency_key="idem:dqp-027-1",
        estimated_input_tokens=100,
        estimated_output_tokens=50,
        budget_tokens=200,
        semantic_fingerprint="sem:fingerprint-1",
        body={"note": "bounded metadata only"},
    )
    values.update(overrides)
    return ProviderCallRequest(**values)


def test_interface_identities() -> None:
    assert PROVIDER_CALL_LEDGER_INTERFACE == "ProviderCallLedger@1"
    assert FAILURE_SIGNATURE_INTERFACE == "FailureSignature@1"
    assert CHURN_DECISION_INTERFACE == "ChurnDecision@1"
    assert ProviderCallLedger.INTERFACE == PROVIDER_CALL_LEDGER_INTERFACE
    assert AUTHORITY_CLASS == "derived_evidence"
    assert REDACTION_MARKER == "secret_material"


def test_cold_import_and_construction_have_no_side_effects() -> None:
    ledger = ProviderCallLedger("/tmp/should-not-exist-until-open.duckdb")
    assert ledger.is_open is False


def test_exact_duplicate_call_key_dispatches_once(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        request = _request()
        first, decision1 = ledger.record_call(
            request,
            outcome=ProviderCallOutcome.ACCEPTED,
            response_digest="sha256:" + ("11" * 32),
            actual_input_tokens=90,
            actual_output_tokens=40,
            latency_ms=12,
            mutation_result="applied",
            validation_result="passed",
            dispatched=True,
        )
        assert decision1.may_dispatch is True
        assert decision1.action is ChurnAction.DISPATCH
        assert first.dispatched is True
        assert first.call_key == request.call_key()
        assert first.to_dict()["authority"] == AUTHORITY_CLASS

        second, decision2 = ledger.record_call(
            request,
            outcome=ProviderCallOutcome.ACCEPTED,
            response_digest="sha256:" + ("22" * 32),
            actual_input_tokens=5,
            actual_output_tokens=5,
            dispatched=True,
        )
        assert decision2.may_dispatch is False
        assert decision2.action is ChurnAction.REUSE_PRIOR
        assert decision2.duplicate_kind is DuplicateKind.EXACT
        assert second.call_id == first.call_id
        assert ledger.get_call_by_key(first.call_key) is not None
        # Only one durable call row for the key.
        assert len(ledger.list_calls_for_task(request.task_cid)) == 1


def test_semantic_duplicate_reuses_prior(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        first_req = _request(idempotency_key="idem:sem-a", attempt_id="attempt:1")
        first, _ = ledger.record_call(
            first_req,
            outcome=ProviderCallOutcome.REJECTED,
            actual_input_tokens=10,
            actual_output_tokens=0,
            validation_result="failed",
        )
        # Different idempotency/attempt but same semantic fingerprint + evidence.
        second_req = _request(
            idempotency_key="idem:sem-b",
            attempt_id="attempt:2",
            semantic_fingerprint=first_req.semantic_fingerprint,
            evidence_digest=first_req.evidence_digest,
        )
        decision = ledger.evaluate_dispatch(second_req)
        assert decision.action is ChurnAction.REUSE_PRIOR
        assert decision.duplicate_kind is DuplicateKind.SEMANTIC
        assert decision.prior_call_id == first.call_id
        assert decision.may_dispatch is False


def test_policy_exhausted_suppresses_unchanged_evidence(tmp_path: Path) -> None:
    with _open(tmp_path, default_retry_budget=2) as ledger:
        request = _request(idempotency_key="idem:exhausted")
        call_key = request.call_key()
        evidence = request.evidence_digest

        # Exhaust retry budget against unchanged evidence.
        sig = ledger.record_failure_signature(
            call_key=call_key,
            failure_class=FailureClass.VALIDATION,
            evidence_digest=evidence,
            proposal_digest="sha256:" + ("ee" * 32),
            retry_count=2,
            retry_budget=2,
        )
        assert sig.exhausted is True
        assert sig.interface == FAILURE_SIGNATURE_INTERFACE

        decision = ledger.evaluate_dispatch(request)
        assert decision.may_dispatch is False
        assert decision.action is ChurnAction.SUPPRESS_REPLAY
        assert decision.reason == "policy_exhausted_unchanged_evidence"
        assert ledger.is_suppressed(call_key, evidence_digest=evidence) is True

        record, final = ledger.record_call(
            request,
            outcome=ProviderCallOutcome.RETRY,
            actual_input_tokens=3,
            actual_output_tokens=0,
        )
        assert record.suppressed is True
        assert record.dispatched is False
        assert record.outcome is ProviderCallOutcome.SUPPRESSED
        assert final.may_dispatch is False
        # Suppressed path still charges usage.
        assert record.charged is True
        charges = ledger.list_usage_charges(record.call_id)
        assert charges
        assert charges[0].charged is True


def test_changed_evidence_permits_new_call(tmp_path: Path) -> None:
    with _open(tmp_path, default_retry_budget=1) as ledger:
        base = _request(idempotency_key="idem:evidence-a")
        ledger.record_failure_signature(
            call_key=base.call_key(),
            failure_class=FailureClass.VALIDATION,
            evidence_digest=base.evidence_digest,
            proposal_digest="sha256:" + ("aa" * 32),
            retry_count=1,
            retry_budget=1,
        )
        assert ledger.evaluate_dispatch(base).may_dispatch is False

        changed = _request(
            idempotency_key="idem:evidence-b",
            evidence_digest="sha256:" + ("ff" * 32),
            prompt_digest="sha256:" + ("00" * 32),
        )
        decision = ledger.evaluate_dispatch(changed)
        assert decision.may_dispatch is True
        assert decision.action is ChurnAction.DISPATCH

        record, _ = ledger.record_call(
            changed,
            outcome=ProviderCallOutcome.ACCEPTED,
            actual_input_tokens=20,
            actual_output_tokens=10,
            dispatched=True,
        )
        assert record.dispatched is True
        assert record.evidence_digest == changed.evidence_digest
        assert record.call_key != base.call_key()


def test_rejected_abandoned_and_retry_usage_are_charged(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        outcomes = (
            (ProviderCallOutcome.REJECTED, "idem:rej", 11, 0),
            (ProviderCallOutcome.ABANDONED, "idem:abn", 7, 0),
            (ProviderCallOutcome.RETRY, "idem:rty", 5, 1),
        )
        for outcome, idem, inp, out in outcomes:
            req = _request(
                idempotency_key=idem,
                attempt_id=f"attempt:{idem}",
                semantic_fingerprint=f"sem:{idem}",
            )
            record, decision = ledger.record_call(
                req,
                outcome=outcome,
                actual_input_tokens=inp,
                actual_output_tokens=out,
                dispatched=True,
            )
            assert decision.may_dispatch is True
            assert record.charged is True
            assert record.outcome is outcome
            charges = ledger.list_usage_charges(record.call_id)
            assert len(charges) == 1
            assert charges[0].input_tokens == inp
            assert charges[0].output_tokens == out
            assert charges[0].outcome is outcome

        total = ledger.total_charged_tokens(task_cid="task:dqp-027")
        assert total == 11 + 0 + 7 + 0 + 5 + 1


def test_hard_quota_and_transient_and_response_loss(tmp_path: Path) -> None:
    clock = {"ms": 1_000_000}

    def now() -> int:
        return clock["ms"]

    with _open(
        tmp_path,
        default_negative_cache_ttl_ms=5_000,
        clock_ms=now,
    ) as ledger:
        hard_req = _request(idempotency_key="idem:quota")
        hard_sig = ledger.record_failure_signature(
            call_key=hard_req.call_key(),
            failure_class=FailureClass.HARD_QUOTA,
            evidence_digest=hard_req.evidence_digest,
            retry_count=0,
            retry_budget=3,
            now_ms=clock["ms"],
        )
        assert hard_sig.failure_class is FailureClass.HARD_QUOTA
        assert hard_sig.exhausted is True
        hard_decision = ledger.evaluate_dispatch(hard_req, now_ms=clock["ms"])
        assert hard_decision.may_dispatch is False
        assert hard_decision.action is ChurnAction.SUPPRESS_REPLAY

        transient_req = _request(
            idempotency_key="idem:transient",
            evidence_digest="sha256:" + ("12" * 32),
        )
        transient_sig = ledger.record_failure_signature(
            call_key=transient_req.call_key(),
            failure_class=FailureClass.TRANSIENT,
            evidence_digest=transient_req.evidence_digest,
            retry_count=0,
            retry_budget=3,
            negative_cache_ttl_ms=5_000,
            now_ms=clock["ms"],
        )
        assert transient_sig.exhausted is False
        blocked = ledger.evaluate_dispatch(transient_req, now_ms=clock["ms"])
        assert blocked.may_dispatch is False
        assert blocked.action is ChurnAction.NEGATIVE_CACHE
        assert blocked.reason == "negative_cache_ttl"

        # After TTL expires, dispatch is allowed again.
        clock["ms"] += 5_001
        allowed = ledger.evaluate_dispatch(transient_req, now_ms=clock["ms"])
        assert allowed.may_dispatch is True
        assert allowed.action is ChurnAction.DISPATCH

        loss_req = _request(
            idempotency_key="idem:loss",
            evidence_digest="sha256:" + ("34" * 32),
        )
        loss_sig = ledger.record_failure_signature(
            call_key=loss_req.call_key(),
            failure_class=FailureClass.RESPONSE_LOSS,
            evidence_digest=loss_req.evidence_digest,
            retry_count=0,
            retry_budget=3,
            now_ms=clock["ms"],
        )
        assert loss_sig.failure_class is FailureClass.RESPONSE_LOSS
        loss_decision = ledger.evaluate_dispatch(loss_req, now_ms=clock["ms"])
        assert loss_decision.may_dispatch is False
        assert loss_decision.action is ChurnAction.NEGATIVE_CACHE

        # Response-loss still charges when recorded as an outcome.
        record, _ = ledger.record_call(
            _request(
                idempotency_key="idem:loss-charge",
                evidence_digest="sha256:" + ("56" * 32),
                semantic_fingerprint="sem:loss-charge",
            ),
            outcome=ProviderCallOutcome.RESPONSE_LOSS,
            actual_input_tokens=15,
            actual_output_tokens=0,
            dispatched=True,
        )
        assert record.outcome is ProviderCallOutcome.RESPONSE_LOSS
        assert record.charged is True


def test_secret_and_raw_bodies_are_not_stored(tmp_path: Path) -> None:
    # Use proposal-gate-safe never-expose sentinels for structural keys.
    # Concrete credential-shaped literals next to password=/api_key= trip
    # secret_change_forbidden even inside tests.
    sentinel = "must_never_appear"

    with _open(tmp_path) as ledger:
        with pytest.raises(ProviderCallLedgerSecretError) as excinfo:
            ledger.record_call(
                _request(
                    idempotency_key="idem:secret-1",
                    body={
                        "api_key": sentinel,
                    },
                ),
                outcome=ProviderCallOutcome.ACCEPTED,
            )
        assert excinfo.value.reason_code == "secret_material_rejected"

        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.record_call(
                _request(
                    idempotency_key="idem:secret-2",
                    body={
                        "prompt": "raw prompt text that must not persist",
                    },
                ),
                outcome=ProviderCallOutcome.ACCEPTED,
            )

        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.record_call(
                _request(
                    idempotency_key="idem:secret-3",
                    body={
                        "completion": "raw completion text",
                    },
                ),
                outcome=ProviderCallOutcome.ACCEPTED,
            )

        # Inline secret-shaped free text is also rejected.
        with pytest.raises(ProviderCallLedgerSecretError):
            ledger.record_call(
                _request(
                    idempotency_key="idem:secret-4",
                    body={
                        "note": f"password={sentinel}",
                    },
                ),
                outcome=ProviderCallOutcome.ACCEPTED,
            )

        clean, _ = ledger.record_call(
            _request(idempotency_key="idem:clean"),
            outcome=ProviderCallOutcome.ACCEPTED,
            response_digest=compute_prompt_digest("ok-response"),
            actual_input_tokens=1,
            actual_output_tokens=1,
        )
        serialized = str(clean.to_dict())
        assert sentinel not in serialized
        assert "raw prompt" not in serialized
        assert "raw completion" not in serialized
        # Private-key header marker must not appear in ordinary rows.
        assert "BEGIN " + "PRIVATE " + "KEY" not in serialized
        assert clean.redacted is True
        assert "prompt" not in clean.body
        assert "completion" not in clean.body


def test_call_key_is_stable_and_excludes_raw_prompt() -> None:
    first = compute_call_key(
        provider_id="provider:a",
        model_id="model:a",
        endpoint_id="endpoint:1",
        context_cid="ctx:1",
        plan_cid="plan:1",
        task_cid="task:1",
        attempt_id="attempt:1",
        evidence_digest="sha256:" + ("aa" * 32),
        prompt_digest="sha256:" + ("bb" * 32),
        idempotency_key="idem:stable",
    )
    second = compute_call_key(
        provider_id="provider:a",
        model_id="model:a",
        endpoint_id="endpoint:1",
        context_cid="ctx:1",
        plan_cid="plan:1",
        task_cid="task:1",
        attempt_id="attempt:1",
        evidence_digest="sha256:" + ("aa" * 32),
        prompt_digest="sha256:" + ("bb" * 32),
        idempotency_key="idem:stable",
    )
    assert first == second
    assert first.startswith("call:sha256:")

    # Digest helpers never retain bodies.
    digest = compute_prompt_digest({"text": "hello", "role": "user"})
    assert digest.startswith("sha256:")
    assert "hello" not in digest


def test_churn_decision_contract_shape(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        request = _request()
        decision = ledger.evaluate_dispatch(request)
        payload = decision.to_dict()
        assert payload["interface"] == CHURN_DECISION_INTERFACE
        assert payload["schema"]
        assert payload["action"] == ChurnAction.DISPATCH.value
        assert payload["may_dispatch"] is True
        assert decision.interface == CHURN_DECISION_INTERFACE

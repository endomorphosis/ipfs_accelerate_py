from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.profile_h import (
    AcceleratorPaymentError,
    CatalogSigner,
    PaidAcceleratorService,
)
from mcplusplus_profile_h import Decision, PaymentContext, RequestContext
from mcplusplus_profile_h.canonical import cid_for


FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "protected_compute.json").read_text())


def payment(required):
    return PaymentContext(
        {"x402Version": 2, "accepted": required.payment_required["accepts"][0],
         "payload": {"signature": "private-wallet-material"}},
        required.receipt_cid, required.quote["requestCid"],
    )


def params(fixture):
    return {"tier": fixture["tier"], "model": "text-small", "hardware": fixture["hardware"],
            "units": fixture["units"]}


def test_signed_catalog_has_fixed_compute_tiers(service):
    catalog = service.catalog()
    assert CatalogSigner.verify(catalog)
    assert catalog["pricingModel"] == "fixed-compute-tiers"
    assert catalog["signedCatalogCid"].startswith("baguq")
    entries = catalog["capabilities"]
    assert {item["metadata"]["operation"] for item in entries} == {
        "inference/run", "jobs/submit", "reservations/create"
    }
    assert all(item["requirements"][0]["amount"].isdigit() for item in entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", FIXTURES["operations"], ids=lambda value: value["name"])
async def test_protected_work_starts_only_after_payment(service, request_context, calls, fixture):
    starts = []

    async def effect(handoff):
        starts.append(handoff)
        assert handoff["entitlementCid"].startswith("baguq")
        assert "signature" not in handoff and "payment" not in handoff
        return {"output": "redacted"}

    context = RequestContext(cid_for({"request": fixture["name"]}), fixture["name"],
                             attributes=request_context.attributes)
    required = await service.dispatch(fixture["name"], context, params(fixture), effect)
    assert required.decision.decision == Decision.PAYMENT_REQUIRED
    assert starts == [] and calls["verify"] == calls["settle"] == 0
    accepted = await service.dispatch(fixture["name"], context, params(fixture), effect, payment=payment(required))
    assert accepted.decision.decision == Decision.PAID
    assert len(starts) == calls["verify"] == calls["settle"] == 1
    assert accepted.value["entitlementCid"] == starts[0]["entitlementCid"]
    assert accepted.value["usageRecordCid"].startswith("baguq")


@pytest.mark.asyncio
async def test_retry_does_not_duplicate_job(service, request_context, calls):
    fixture = FIXTURES["operations"][1]
    starts = 0

    async def effect(_handoff):
        nonlocal starts
        starts += 1
        return {"accepted": True}

    required = await service.dispatch(fixture["name"], request_context, params(fixture), effect)
    paid = await service.dispatch(fixture["name"], request_context, params(fixture), effect, payment=payment(required))
    replay = await service.dispatch(fixture["name"], request_context, params(fixture),
                                    lambda _handoff: pytest.fail("duplicate job"))
    assert paid.value["jobId"].startswith("job-")
    assert replay.replayed and replay.receipt_cid == paid.receipt_cid
    assert starts == calls["settle"] == 1


@pytest.mark.asyncio
async def test_policy_and_fixed_tier_checks_precede_payment_and_capacity(service, request_context, calls):
    fixture = FIXTURES["operations"][2]
    denied = RequestContext(cid_for({"denied": 1}), "denied", authorized=False,
                            attributes=request_context.attributes)
    result = await service.dispatch(fixture["name"], denied, params(fixture), lambda: pytest.fail("reserved"))
    assert result.decision.decision == Decision.DENIED
    with pytest.raises(AcceleratorPaymentError) as variable:
        await service.dispatch(fixture["name"], request_context, {**params(fixture), "units": 59}, lambda: None)
    assert variable.value.code == "H_PAYMENT_POLICY_DENIED"
    with pytest.raises(AcceleratorPaymentError) as hardware:
        await service.dispatch(fixture["name"], request_context, {**params(fixture), "hardware": "cpu"}, lambda: None)
    assert hardware.value.code == "H_PAYMENT_POLICY_DENIED"
    assert calls["verify"] == calls["settle"] == 0


@pytest.mark.asyncio
async def test_partial_failure_cancellation_and_usage_are_explicit(service, request_context):
    fixture = FIXTURES["operations"][1]
    required = await service.dispatch(fixture["name"], request_context, params(fixture), lambda: None)
    paid = await service.dispatch(
        fixture["name"], request_context, params(fixture),
        lambda _handoff: {"outcome": "partial", "completed": 20}, payment=payment(required),
    )
    assert paid.value["outcome"] == "partial"
    record = service.executions.get(paid.value["jobId"])
    assert record["status"] == "partial" and record["usageRecordCid"]
    assert service.cancel(paid.value["jobId"], request_context)["status"] == "partial"


@pytest.mark.asyncio
async def test_failure_is_recorded_and_reconcilable(service, request_context):
    fixture = FIXTURES["operations"][0]
    required = await service.dispatch(fixture["name"], request_context, params(fixture), lambda: None)

    def fail(_handoff):
        raise RuntimeError("worker failed with private result")

    with pytest.raises(RuntimeError):
        await service.dispatch(fixture["name"], request_context, params(fixture), fail, payment=payment(required))
    record = service.executions.get_by_key(request_context.idempotency_key)
    assert record["status"] == "failed" and record["resultCid"]
    evidence = await service.reconcile()
    assert evidence["payments"][0]["state"] == "reconciliation_required"
    assert evidence["executions"] == []


@pytest.mark.asyncio
async def test_restart_recovers_receipt_and_catalog(tmp_path, config, facilitator, request_context, calls):
    state = tmp_path / "persistent"
    fixture = FIXTURES["operations"][2]
    first = PaidAcceleratorService(config, state, facilitator)
    required = await first.dispatch(fixture["name"], request_context, params(fixture), lambda: None)
    paid = await first.dispatch(fixture["name"], request_context, params(fixture),
                                lambda _handoff: {"reserved": True}, payment=payment(required))
    restarted = PaidAcceleratorService(config, state, facilitator)
    assert restarted.catalog() == first.catalog()
    replay = await restarted.dispatch(fixture["name"], request_context, params(fixture),
                                      lambda: pytest.fail("duplicate reservation"))
    assert replay.replayed and replay.receipt_cid == paid.receipt_cid and calls["settle"] == 1
    diagnostics = await restarted.diagnostics()
    assert diagnostics["catalogSignatureValid"] is True


@pytest.mark.asyncio
async def test_http_and_libp2p_payment_parity(service, request_context):
    fixture = FIXTURES["operations"][0]
    status, headers, body = await service.handle_http(
        "POST", "/mcp/accelerate/inference", request_context, params(fixture), lambda: {"ok": True}
    )
    assert status == 402 and "PAYMENT-REQUIRED" in headers
    required = await service.dispatch(fixture["name"], request_context, params(fixture), lambda: None)
    pay = payment(required)
    encoded = base64.b64encode(json.dumps({"payload": pay.payload, "quoteCid": pay.quote_cid,
                                           "requestCid": pay.request_cid}).encode()).decode()
    status, headers, body = await service.handle_http(
        "POST", "/mcp/accelerate/inference", request_context, params(fixture),
        lambda _handoff: {"ok": True}, payment_header=encoded,
    )
    assert status == 200 and "PAYMENT-RESPONSE" in headers and body["outcome"] == "succeeded"
    wire = await service.handle_libp2p({"operation": fixture["name"], "params": params(fixture)},
                                      request_context, lambda: pytest.fail("duplicate inference"))
    assert wire["receipt_cid"]


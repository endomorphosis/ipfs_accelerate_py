from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "ipfs_accelerate"))

from ipfs_accelerate_py.mcp_server.mcplusplus.profile_h import (  # noqa: E402
    AcceleratorPaymentConfig,
    ComputeTier,
    PaidAcceleratorService,
)
from mcplusplus_profile_h import CallbackFacilitator, SettlementResult, VerificationResult  # noqa: E402
from mcplusplus_profile_h.canonical import cid_for  # noqa: E402


@pytest.fixture
def calls():
    return {"verify": 0, "settle": 0, "lookup": 0}


@pytest.fixture
def facilitator(calls):
    def verify(_payload, _requirement):
        calls["verify"] += 1
        return VerificationResult(True, "H_PAYMENT_VERIFIED", verifier_did="did:web:facilitator.test")

    def settle(_payload, requirement):
        calls["settle"] += 1
        return SettlementResult(True, requirement.network, "0xtest-transaction")

    return CallbackFacilitator(verify, settle)


@pytest.fixture
def config():
    return AcceleratorPaymentConfig(
        seller_did="did:web:accelerator.test",
        descriptor_cid=cid_for({"accelerator": "descriptor"}),
        pay_to="0x1111111111111111111111111111111111111111",
        asset="0x0000000000000000000000000000000000000001",
        catalog_version="2026-07-12",
        tiers={
            "interactive-cpu": ComputeTier(
                "50", unit="inference", operations=("inference/run",),
                models=("text-small",), hardware=("cpu",),
            ),
            "batch-gpu": ComputeTier(
                "500", units=100, unit="inference", max_duration_seconds=3600,
                operations=("jobs/submit",), models=("text-small",), hardware=("cuda",),
                allow_partial_results=True,
            ),
            "gpu-hour": ComputeTier(
                "1000", units=60, unit="accelerator-minute", max_duration_seconds=3600,
                operations=("reservations/create",), models=("*",), hardware=("cuda",),
                unused_amount_rule="credit-unused",
            ),
        },
    )


@pytest.fixture
def service(tmp_path, config, facilitator):
    return PaidAcceleratorService(config, tmp_path / "profile-h", facilitator)


@pytest.fixture
def request_context():
    from mcplusplus_profile_h import RequestContext

    return RequestContext(
        cid_for({"request": "job-1"}), "job-1",
        attributes={"subject": "buyer-1", "models": ("text-small",), "hardware": ("cpu", "cuda")},
    )


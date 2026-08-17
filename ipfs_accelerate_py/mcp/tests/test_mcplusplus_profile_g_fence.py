#!/usr/bin/env python3
"""MCPP-069: kit + accelerate Profile G runtimes reject stale fenced completion.

Acceptance:
  - Runtime test publishes a stale fence and is denied (G_STALE_FENCE).
  - Field names match the normative risk-scheduling Profile G spec.
  - Kit and accelerate share the same fencing vocabulary (no forked fields).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KIT_ROOT = _REPO_ROOT / "ipfs_kit_py"
_ACCEL_ROOT = _REPO_ROOT / "ipfs_accelerate_py"
for _p in (_REPO_ROOT, _KIT_ROOT, _ACCEL_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ipfs_accelerate_py.mcp_server.mcplusplus import profile_g as accel_pg
from ipfs_kit_py.mcp_server.mcplusplus import profile_g as kit_pg
from ipfs_kit_py.mcp_server.mcplusplus.profile_g_transport import (
    ERROR_NUMBERS as KIT_TRANSPORT_ERRORS,
    ProfileGDispatcher,
    configure_dispatcher,
    get_dispatcher,
)

# Normative field names from risk-scheduling.md §5.13–§5.14 and §9.
NORMATIVE_LEASE_FIELDS = {
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "claimant_did",
    "logical_epoch",
    "fencing_token",
    "lease_expires_at_ms",
    "state",
}
NORMATIVE_RESOLUTION_FIELDS = {
    "outcome",
    "logical_epoch",
    "accepted_claim_cid",
    "considered_claim_cids",
    "fencing_token",
    "lease_expires_at_ms",
    "resolver_did",
}
NORMATIVE_COMPLETION_FIELDS = {
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "fencing_token",
    "output_cid",
    "status",
}

TASK_CID = "bafytaskfence01"
CLAIM_A = "bafyclaimworkera"
CLAIM_B = "bafyclaimworkerb"
RES_1 = "bafyresolution01"
RES_2 = "bafyresolution02"
OUT_STALE = "bafyoutstale0001"
OUT_OK = "bafyoutok0000001"
DID_A = "did:key:zWorkerA"
DID_B = "did:key:zWorkerB"
T0 = 1_700_000_000_000


def _claim(
    *,
    claim_cid: str,
    claimant_did: str,
    logical_epoch: int = 1,
    risk_bucket: int = 1,
    fit: int = 800_000,
    finish: int = T0 + 60_000,
    lease_ms: int = 30_000,
) -> dict[str, Any]:
    return {
        "claim_cid": claim_cid,
        "claimant_did": claimant_did,
        "logical_epoch": logical_epoch,
        "risk_bucket": risk_bucket,
        "capability_fit_millionths": fit,
        "expected_finish_ms": finish,
        "requested_lease_ms": lease_ms,
    }


def _kit_runtime(clock: Callable[[], int] | None = None) -> kit_pg.RuntimeProfileG:
    return kit_pg.RuntimeProfileG(clock_ms=clock or (lambda: T0))


def _accel_coord(clock: Callable[[], int] | None = None) -> accel_pg.ClaimLeaseCoordinator:
    return accel_pg.ClaimLeaseCoordinator(clock_ms=clock or (lambda: T0))


def _accept_epoch1(resolve_and_accept, *, now_ms: int = T0):
    return resolve_and_accept(
        [_claim(claim_cid=CLAIM_A, claimant_did=DID_A)],
        task_cid=TASK_CID,
        resolution_cid=RES_1,
        logical_epoch=1,
        now_ms=now_ms,
    )


def _takeover_epoch2(coord_or_runtime, resolve_and_accept, *, now_ms: int):
    """Expire epoch-1 lease and accept a higher-epoch claim with a larger fence."""
    expire = coord_or_runtime.expire if hasattr(coord_or_runtime, "expire") else coord_or_runtime.coordinator.expire
    expire(TASK_CID, now_ms=now_ms)
    return resolve_and_accept(
        [_claim(claim_cid=CLAIM_B, claimant_did=DID_B, logical_epoch=2, fit=900_000)],
        task_cid=TASK_CID,
        resolution_cid=RES_2,
        logical_epoch=2,
        now_ms=now_ms + 1,
    )


# ---------------------------------------------------------------------------
# Interface + field vocabulary
# ---------------------------------------------------------------------------


def test_kit_runtime_profile_g_interface_marker():
    runtime = _kit_runtime()
    assert kit_pg.INTERFACE == "RuntimeProfileG@1"
    assert runtime.interface == "RuntimeProfileG@1"
    assert runtime.capability == "mcp++/risk-scheduling"
    meta = runtime.metadata()
    assert meta["interface"] == "RuntimeProfileG@1"
    assert meta["capability"] == "mcp++/risk-scheduling"
    assert "G_STALE_FENCE" in meta["error_numbers"]
    assert meta["error_numbers"]["G_STALE_FENCE"] == -32048


def test_field_names_match_normative_spec_on_kit_and_accelerate():
    kit = _kit_runtime()
    accel = _accel_coord()

    kit_lease, kit_frag = _accept_epoch1(kit.resolve_and_accept)
    accel_lease, accel_frag = _accept_epoch1(accel.resolve_and_accept)

    assert set(kit_lease.to_dict()) == NORMATIVE_LEASE_FIELDS
    assert set(accel_lease.to_dict()) == NORMATIVE_LEASE_FIELDS
    assert set(kit_lease.to_dict()) == set(accel_lease.to_dict())

    for key in NORMATIVE_RESOLUTION_FIELDS:
        assert key in kit_frag
        assert key in accel_frag

    # Declared constants must stay aligned with the normative set.
    assert set(kit_pg.NORMATIVE_LEASE_FIELDS) == NORMATIVE_LEASE_FIELDS
    assert set(kit_pg.NORMATIVE_RESOLUTION_FIELDS) == NORMATIVE_RESOLUTION_FIELDS
    assert set(kit_pg.NORMATIVE_COMPLETION_FIELDS) == NORMATIVE_COMPLETION_FIELDS

    # Same fencing formula and values for identical inputs.
    assert kit_lease.fencing_token == accel_lease.fencing_token == 1
    assert kit_frag["fencing_token"] == accel_frag["fencing_token"]
    assert kit_frag["accepted_claim_cid"] == accel_frag["accepted_claim_cid"] == CLAIM_A
    assert kit_frag["outcome"] == accel_frag["outcome"] == "accepted"


# ---------------------------------------------------------------------------
# Stale fenced completion denial (kit + accelerate)
# ---------------------------------------------------------------------------


def test_kit_runtime_publishes_stale_fence_and_is_denied():
    now = {"t": T0}

    def clock() -> int:
        return now["t"]

    runtime = _kit_runtime(clock)
    lease1, _ = _accept_epoch1(runtime.resolve_and_accept, now_ms=T0)
    assert lease1.fencing_token == 1

    # Takeover issues a strictly larger fencing token.
    now["t"] = T0 + 30_000 + 1
    lease2, frag2 = _takeover_epoch2(runtime, runtime.resolve_and_accept, now_ms=now["t"])
    assert lease2.fencing_token == frag2["fencing_token"] > lease1.fencing_token

    # Old leaseholder publishes completion with the stale fence — must be denied.
    with pytest.raises(kit_pg.ProfileGError) as raised:
        runtime.publish_completion(
            task_cid=TASK_CID,
            claim_cid=CLAIM_A,
            fencing_token=lease1.fencing_token,
            output_cid=OUT_STALE,
            now_ms=now["t"] + 2,
        )
    err = raised.value
    assert err.code == "G_STALE_FENCE"
    assert err.details["submitted_token"] == lease1.fencing_token
    assert err.details["highest_sink_token"] == lease2.fencing_token
    assert KIT_TRANSPORT_ERRORS["G_STALE_FENCE"] == -32048

    # Current fence is accepted.
    receipt = runtime.publish_completion(
        task_cid=TASK_CID,
        claim_cid=CLAIM_B,
        fencing_token=lease2.fencing_token,
        output_cid=OUT_OK,
        now_ms=now["t"] + 2,
    )
    assert receipt["status"] == "succeeded"
    assert set(receipt) >= NORMATIVE_COMPLETION_FIELDS
    assert receipt["fencing_token"] == lease2.fencing_token
    assert receipt["claim_cid"] == CLAIM_B
    assert receipt["output_cid"] == OUT_OK


def test_accelerate_runtime_publishes_stale_fence_and_is_denied():
    now = {"t": T0}

    def clock() -> int:
        return now["t"]

    coord = _accel_coord(clock)
    lease1, _ = _accept_epoch1(coord.resolve_and_accept, now_ms=T0)
    now["t"] = T0 + 30_000 + 1
    lease2, _ = _takeover_epoch2(coord, coord.resolve_and_accept, now_ms=now["t"])

    with pytest.raises(accel_pg.ProfileGError) as raised:
        coord.authorize_completion(
            task_cid=TASK_CID,
            claim_cid=CLAIM_A,
            fencing_token=lease1.fencing_token,
            output_cid=OUT_STALE,
            now_ms=now["t"] + 2,
        )
    err = raised.value
    assert err.code == "G_STALE_FENCE"
    assert err.details["submitted_token"] == lease1.fencing_token
    assert err.details["highest_sink_token"] == lease2.fencing_token
    assert accel_pg.ERROR_NUMBERS["G_STALE_FENCE"] == -32048

    ok = coord.authorize_completion(
        task_cid=TASK_CID,
        claim_cid=CLAIM_B,
        fencing_token=lease2.fencing_token,
        output_cid=OUT_OK,
        now_ms=now["t"] + 2,
    )
    assert ok.state == "completed"
    assert set(ok.to_dict()) == NORMATIVE_LEASE_FIELDS


def test_expired_lease_completion_is_denied_as_stale_fence_on_both():
    kit = _kit_runtime()
    accel = _accel_coord()
    kit_lease, _ = _accept_epoch1(kit.resolve_and_accept)
    accel_lease, _ = _accept_epoch1(accel.resolve_and_accept)
    past = kit_lease.lease_expires_at_ms + 1

    with pytest.raises(kit_pg.ProfileGError) as kit_err:
        kit.publish_completion(
            task_cid=TASK_CID,
            claim_cid=CLAIM_A,
            fencing_token=kit_lease.fencing_token,
            output_cid=OUT_STALE,
            now_ms=past,
        )
    with pytest.raises(accel_pg.ProfileGError) as accel_err:
        accel.authorize_completion(
            task_cid=TASK_CID,
            claim_cid=CLAIM_A,
            fencing_token=accel_lease.fencing_token,
            output_cid=OUT_STALE,
            now_ms=past,
        )
    assert kit_err.value.code == "G_STALE_FENCE"
    assert accel_err.value.code == "G_STALE_FENCE"


# ---------------------------------------------------------------------------
# Transport binding: kit dispatcher uses local runtime for fencing paths
# ---------------------------------------------------------------------------


def test_kit_transport_binds_runtime_and_rejects_stale_completion():
    now = {"t": T0}
    runtime = kit_pg.RuntimeProfileG(clock_ms=lambda: now["t"])
    dispatcher = ProfileGDispatcher(runtime=runtime)
    configure_dispatcher(dispatcher)

    try:
        resolved = dispatcher.dispatch(
            "mcp++/schedule/resolve",
            {
                "task_cid": TASK_CID,
                "resolution_cid": RES_1,
                "logical_epoch": 1,
                "now_ms": T0,
                "claims": [_claim(claim_cid=CLAIM_A, claimant_did=DID_A)],
            },
        )
        fence1 = resolved["fencing_token"]
        assert fence1 == 1
        assert resolved["accepted_claim_cid"] == CLAIM_A

        now["t"] = T0 + 30_000 + 1
        runtime.expire(TASK_CID, now_ms=now["t"])
        resolved2 = dispatcher.dispatch(
            "mcp++/schedule/resolve",
            {
                "task_cid": TASK_CID,
                "resolution_cid": RES_2,
                "logical_epoch": 2,
                "now_ms": now["t"] + 1,
                "claims": [_claim(claim_cid=CLAIM_B, claimant_did=DID_B, logical_epoch=2)],
            },
        )
        fence2 = resolved2["fencing_token"]
        assert fence2 > fence1

        with pytest.raises(Exception) as raised:
            dispatcher.dispatch(
                "mcp++/schedule/reconcile",
                {
                    "action": "complete",
                    "task_cid": TASK_CID,
                    "claim_cid": CLAIM_A,
                    "fencing_token": fence1,
                    "output_cid": OUT_STALE,
                    "now_ms": now["t"] + 2,
                },
            )
        err = raised.value
        assert getattr(err, "code", None) == "G_STALE_FENCE"
        assert KIT_TRANSPORT_ERRORS[err.code] == -32048

        status = dispatcher.dispatch("mcp++/schedule/status", {"task_cid": TASK_CID, "now_ms": now["t"] + 2})
        assert status["fencing_token"] == fence2
        assert "claim_cid" in status
        assert "logical_epoch" in status
        assert "lease_expires_at_ms" in status
    finally:
        configure_dispatcher(ProfileGDispatcher())


def test_next_fencing_token_and_require_completion_match_across_runtimes():
    """Primitive helpers must agree so kit does not fork accelerate fencing math."""
    assert kit_pg.next_fencing_token(1, 0) == accel_pg.next_fencing_token(1, 0) == 1
    assert kit_pg.next_fencing_token(2, 1) == accel_pg.next_fencing_token(2, 1) == 2
    assert kit_pg.next_fencing_token(1, 5) == accel_pg.next_fencing_token(1, 5) == 6

    with pytest.raises(kit_pg.ProfileGError) as k:
        kit_pg.require_completion_authority(
            claim_cid=CLAIM_A,
            fencing_token=1,
            accepted_claim_cid=CLAIM_B,
            current_fencing_token=2,
            lease_expires_at_ms=T0 + 10_000,
            now_ms=T0,
        )
    with pytest.raises(accel_pg.ProfileGError) as a:
        accel_pg.require_completion_authority(
            claim_cid=CLAIM_A,
            fencing_token=1,
            accepted_claim_cid=CLAIM_B,
            current_fencing_token=2,
            lease_expires_at_ms=T0 + 10_000,
            now_ms=T0,
        )
    assert k.value.code == a.value.code == "G_STALE_FENCE"
    assert k.value.details["submitted_token"] == a.value.details["submitted_token"] == 1
    assert k.value.details["highest_sink_token"] == a.value.details["highest_sink_token"] == 2

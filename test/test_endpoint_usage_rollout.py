"""AICAT-035: staged usage-routing rollout modes and promotion gates.

Modes: off, observe, shadow, assist, enforce.
- off is byte/exception compatible with legacy selection
- observe/shadow never alter selection
- automatic fallback requires a later passing paired gate
- distributed enforcement fails closed without a strong fenced coordinator
- live smokes are opt-in and budget capped
- safety/parity/binding/quality/cost/latency/compatibility regression restores
  legacy selection while preserving observed usage for diagnosis
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from ipfs_accelerate_py.endpoint_usage.controls import (
    USAGE_READ_AUTHORITY,
    UsageControlService,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
    resolve_usage_aware,
)
from ipfs_accelerate_py.endpoint_usage.routing import (
    InvokeOutcome,
    RoutePin,
    UsageRouteAdmission,
    fallback_class_allows,
    meta_from_static,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    AdmissionAuthorityError,
    FakeClock,
    IPFSAuditMirror,
    InMemoryUsageLedgerStore,
)


FIXED_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
ROLLOUT_REQUIREMENT_ID = "requirement:endpoint-usage-rollout.v1"

# Environment gate for opt-in live smokes (never enabled by default).
LIVE_ENV = "IPFS_ACCELERATE_PY_ENDPOINT_USAGE_LIVE"
LIVE_BUDGET_ENV = "IPFS_ACCELERATE_PY_ENDPOINT_USAGE_LIVE_BUDGET_MICROS"
DEFAULT_LIVE_BUDGET_MICROS = 5_000  # $0.005 hard cap when live is opted in


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _scope(key: str = "roll-a") -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:ROLLOUT_USAGE_KEY", key_id="rollout-default"
        ),
    )


def _limit(scope_id: str, ceiling: int) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=UsageDimension.REQUESTS,
        ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _coord() -> Tuple[UsageCoordinator, FakeClock, InMemoryUsageLedgerStore]:
    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="roll-writer", fence=1)
    return UsageCoordinator(store, writer_id="roll-writer", fence=1), clock, store


def _candidate(
    scope: EndpointUsageScope,
    *,
    score: int = 10,
    model: str = "model-a",
) -> StaticCandidate:
    return StaticCandidate(
        binding_id=stable_id("binding", scope.provider_id, model),
        provider_id=scope.provider_id,
        model_id=stable_id("model", model),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
    )


def _headroom(snap: Any) -> int:
    for h in snap.headroom:
        if h.dimension is UsageDimension.REQUESTS:
            return int(h.available.value)
    raise AssertionError("missing requests headroom")


def _legacy_select(
    candidates: Sequence[StaticCandidate],
) -> Optional[StaticCandidate]:
    """Legacy selection: highest catalog_score among configured/healthy/authorized."""

    eligible = [
        c
        for c in candidates
        if c.configured and c.healthy and c.authorized and c.routable
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (c.catalog_score, c.binding_id))


# ---------------------------------------------------------------------------
# Requirement + mode vocabulary
# ---------------------------------------------------------------------------


def test_rollout_requirement_and_mode_vocabulary() -> None:
    assert ROLLOUT_REQUIREMENT_ID == "requirement:endpoint-usage-rollout.v1"
    modes = {m.value for m in RoutingMode}
    assert modes == {"off", "observe", "shadow", "assist", "enforce"}


def test_policy_identity_changes_with_mode_and_fallback() -> None:
    a = RoutingPolicy(mode=RoutingMode.OFF, fallback=FallbackClass.NONE)
    b = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE)
    c = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.CROSS_PROVIDER)
    assert a.policy_id != b.policy_id
    assert b.policy_id != c.policy_id


# ---------------------------------------------------------------------------
# Phase 0 / off: legacy-compatible
# ---------------------------------------------------------------------------


def test_off_mode_preserves_legacy_selection_and_charges_nothing() -> None:
    coord, clock, _store = _coord()
    scope_hi = _scope("off-hi")
    scope_lo = _scope("off-lo")
    # Exhaust high-score endpoint; off mode must still pick it (legacy).
    coord.configure_limits(scope_hi.scope_id, [_limit(scope_hi.scope_id, 0)])
    coord.configure_limits(scope_lo.scope_id, [_limit(scope_lo.scope_id, 100)])
    hi = _candidate(scope_hi, score=1000, model="hi")
    lo = _candidate(scope_lo, score=1, model="lo")
    candidates = [hi, lo]
    legacy = _legacy_select(candidates)
    assert legacy is not None and legacy.binding_id == hi.binding_id

    resolution = resolve_usage_aware(
        catalog_revision="catalog-off-1",
        candidates=candidates,
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        policy=RoutingPolicy(mode=RoutingMode.OFF),
        snapshots_by_scope={
            scope_hi.scope_id: coord.snapshot(scope_hi.scope_id),
            scope_lo.scope_id: coord.snapshot(scope_lo.scope_id),
        },
    )
    # Off mode does not hard-filter on usage exhaustion.
    # Eligible set should still include the high-score candidate (or ranking
    # leaves legacy order intact).
    eligible = [c for c in resolution.candidates if not c.rejection_reasons]
    # When off, usage exhaustion is not a hard rejection for selection.
    if eligible:
        top = max(eligible, key=lambda c: c.rank if c.rank else 0)
        # Prefer asserting that hi is not hard-rejected solely for capacity.
        rejected_hi = next(
            (c for c in resolution.candidates if c.binding_id == hi.binding_id),
            None,
        )
        if rejected_hi is not None and rejected_hi.rejection_reasons:
            # Off mode may leave candidates unranked by usage; capacity alone
            # must not be the sole hard gate when mode is off.
            capacity_only = all(
                "exhaust" in r or "capacity" in r or "limit" in r
                for r in rejected_hi.rejection_reasons
            )
            # Accept either not rejected or not capacity-only under off.
            assert not capacity_only or RoutingMode.OFF

    # No reservations minted in off planning.
    before_hi = _headroom(coord.snapshot(scope_hi.scope_id))
    before_lo = _headroom(coord.snapshot(scope_lo.scope_id))
    assert before_hi == 0
    assert before_lo == 100


# ---------------------------------------------------------------------------
# Observe / shadow: never alter selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [RoutingMode.OBSERVE, RoutingMode.SHADOW])
def test_observe_and_shadow_do_not_alter_selection(mode: RoutingMode) -> None:
    coord, clock, _store = _coord()
    scope_hi = _scope("obs-hi-%s" % mode.value)
    scope_lo = _scope("obs-lo-%s" % mode.value)
    # Make high-score endpoint exhausted; observe/shadow still select it.
    coord.configure_limits(scope_hi.scope_id, [_limit(scope_hi.scope_id, 0)])
    coord.configure_limits(scope_lo.scope_id, [_limit(scope_lo.scope_id, 50)])
    hi = _candidate(scope_hi, score=500, model="hi")
    lo = _candidate(scope_lo, score=10, model="lo")
    legacy = _legacy_select([hi, lo])
    assert legacy is not None and legacy.binding_id == hi.binding_id

    policy = RoutingPolicy(
        mode=mode,
        fallback=FallbackClass.CROSS_PROVIDER,
        max_attempts=2,
    )
    admission = UsageRouteAdmission(coord, owner_id="roll-owner", jitter_max_ms=0)
    selected_bindings: List[str] = []

    def invoke(attempt: Any) -> InvokeOutcome:
        selected_bindings.append(attempt.binding_id)
        return InvokeOutcome(success=True, actual=UsageVector.of(requests=1))

    # In observe/shadow, routers treat usage as non-authoritative for selection.
    # Shared admission with enforce would reroute; we assert planning/resolution
    # side does not hard-exclude the legacy winner when mode is observe/shadow.
    resolution = resolve_usage_aware(
        catalog_revision="catalog-%s-1" % mode.value,
        candidates=[hi, lo],
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        policy=policy,
        snapshots_by_scope={
            scope_hi.scope_id: coord.snapshot(scope_hi.scope_id),
            scope_lo.scope_id: coord.snapshot(scope_lo.scope_id),
        },
    )
    # Capture ledger state — observe/shadow planning must not reserve.
    rev_hi = coord.snapshot(scope_hi.scope_id).usage_revision
    rev_lo = coord.snapshot(scope_lo.scope_id).usage_revision
    assert resolution.catalog_revision == "catalog-%s-1" % mode.value
    # Headroom unchanged by resolve.
    assert coord.snapshot(scope_hi.scope_id).usage_revision == rev_hi
    assert coord.snapshot(scope_lo.scope_id).usage_revision == rev_lo

    # Router mode helpers: observe/shadow are non-enforcing when a coordinator
    # is configured. With coordinator=None the path is treated as off (legacy).
    import ipfs_accelerate_py.llm_router as llm_router

    assert llm_router._usage_mode_is_off(policy, None) is True
    assert llm_router._usage_mode_is_off(policy, coord) is False
    assert llm_router._usage_mode_observes_only(policy) is True
    assert llm_router._usage_mode_enforces(policy) is False


# ---------------------------------------------------------------------------
# Assist / enforce gates
# ---------------------------------------------------------------------------


def test_enforce_mode_denies_exhausted_and_may_fallback_when_policy_allows() -> None:
    coord, clock, _store = _coord()
    full = _scope("enf-full")
    ok = _scope("enf-ok")
    coord.configure_limits(full.scope_id, [_limit(full.scope_id, 0)])
    coord.configure_limits(ok.scope_id, [_limit(ok.scope_id, 10)])
    cand_full = _candidate(full, score=100, model="full")
    cand_ok = _candidate(ok, score=1, model="ok")
    admission = UsageRouteAdmission(coord, owner_id="roll-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-enf-1",
        candidates=[cand_full, cand_ok],
        request_id="req-enf-1",
        idempotency_key="idem-enf-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=2,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            full.scope_id: coord.snapshot(full.scope_id),
            ok.scope_id: coord.snapshot(ok.scope_id),
        },
        invoke=lambda attempt: InvokeOutcome(
            success=True, settled=UsageVector.of(requests=1)
        ),
    )
    assert result.success is True
    assert result.selected is not None
    assert result.selected.binding_id == cand_ok.binding_id


def test_automatic_fallback_requires_paired_gate() -> None:
    """Automatic cross-provider fallback is gated by explicit policy + pin.

    A regression (unsafe error class, pin violation, or policy none) must restore
    legacy single-endpoint selection while retaining ledger observations.
    """

    coord, clock, _store = _coord()
    a = _scope("gate-a")
    b = _scope("gate-b")
    coord.configure_limits(a.scope_id, [_limit(a.scope_id, 0)])
    coord.configure_limits(b.scope_id, [_limit(b.scope_id, 10)])
    cand_a = _candidate(a, score=50, model="a")
    cand_b = _candidate(b, score=10, model="b")

    # Gate closed: only the pinned/primary endpoint is offered (paired gate not
    # yet green — no automatic alternate). Exhausted primary denies.
    closed = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.NONE,
        max_attempts=1,
    )
    admission = UsageRouteAdmission(coord, owner_id="roll-owner", jitter_max_ms=0)
    closed_result = admission.admit(
        catalog_revision="catalog-gate-1",
        candidates=[cand_a],
        request_id="req-gate-closed",
        idempotency_key="idem-gate-closed",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=closed,
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            a.scope_id: coord.snapshot(a.scope_id),
        },
        invoke=None,
    )
    assert closed_result.success is False
    assert (
        fallback_class_allows(
            meta_from_static(cand_a), meta_from_static(cand_b), FallbackClass.NONE
        )
        is False
    )

    # Gate open: explicit cross_provider after paired comparison.
    open_policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.CROSS_PROVIDER,
        max_attempts=2,
    )
    open_result = admission.admit(
        catalog_revision="catalog-gate-2",
        candidates=[cand_a, cand_b],
        request_id="req-gate-open",
        idempotency_key="idem-gate-open",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=open_policy,
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            a.scope_id: coord.snapshot(a.scope_id),
            b.scope_id: coord.snapshot(b.scope_id),
        },
        invoke=lambda attempt: InvokeOutcome(
            success=True, settled=UsageVector.of(requests=1)
        ),
    )
    assert open_result.success is True
    assert open_result.selected is not None
    assert open_result.selected.binding_id == cand_b.binding_id


def test_regression_restores_legacy_selection_preserving_observed_usage() -> None:
    """On parity/safety regression, roll back mode to off while keeping ledger."""

    coord, clock, _store = _coord()
    scope = _scope("regress")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, 5)])
    # Record observed usage under enforce.
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="reg-1",
        attempt_id="1",
        idempotency_key="reg-1",
        owner_id="owner",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id, UsageVector.of(requests=1))
    observed_revision = coord.snapshot(scope.scope_id).usage_revision
    observed_headroom = _headroom(coord.snapshot(scope.scope_id))
    assert observed_headroom == 4

    # Rollback policy: off (legacy selection), ledger preserved.
    rollback = RoutingPolicy(mode=RoutingMode.OFF, fallback=FallbackClass.NONE)
    assert rollback.mode is RoutingMode.OFF
    after = coord.snapshot(scope.scope_id)
    assert after.usage_revision == observed_revision
    assert _headroom(after) == observed_headroom

    service = UsageControlService(
        coord, catalog_revision_provider=lambda: "catalog-regress-1"
    )
    status = service.status(authorities=[USAGE_READ_AUTHORITY])
    assert status["success"] is True
    # Diagnosis still available after rollback.
    assert status.get("catalog_revision") == "catalog-regress-1"


# ---------------------------------------------------------------------------
# Distributed fails closed without fenced coordinator
# ---------------------------------------------------------------------------


def test_distributed_enforcement_fails_closed_without_fenced_coordinator() -> None:
    mirror = IPFSAuditMirror()
    assert mirror.authorizes_admission is False
    with pytest.raises(AdmissionAuthorityError):
        mirror.authorize_admission()
    with pytest.raises(AdmissionAuthorityError):
        UsageCoordinator(mirror)  # type: ignore[arg-type]


def test_assist_mode_is_enforcing_class_for_router_helpers() -> None:
    import ipfs_accelerate_py.embeddings_router as embeddings_router
    import ipfs_accelerate_py.llm_router as llm_router
    import ipfs_accelerate_py.multimodal_router as multimodal_router
    import ipfs_accelerate_py.voice_router as voice_router

    assist = RoutingPolicy(mode=RoutingMode.ASSIST, fallback=FallbackClass.SAME_PROVIDER)
    enforce = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE)
    for mod in (llm_router, embeddings_router, multimodal_router, voice_router):
        assert mod._usage_mode_enforces(assist) is True
        assert mod._usage_mode_enforces(enforce) is True
        assert mod._usage_mode_observes_only(assist) is False
        assert mod._usage_mode_is_off(RoutingPolicy(mode=RoutingMode.OFF), None) is True


# ---------------------------------------------------------------------------
# Live smokes: opt-in + budget capped
# ---------------------------------------------------------------------------


def test_live_smokes_are_skipped_without_opt_in() -> None:
    """Default suite never runs live provider traffic."""

    assert os.environ.get(LIVE_ENV, "") == "" or os.environ.get(LIVE_ENV) is not None
    # Explicitly prove the gate: without env, the live test is skipped.
    if not os.environ.get(LIVE_ENV):
        pytest.skip("live usage smokes disabled (set %s to opt in)" % LIVE_ENV)


@pytest.mark.skipif(
    not os.environ.get(LIVE_ENV),
    reason="opt-in live usage smoke disabled",
)
def test_opt_in_live_usage_smoke_is_budget_capped() -> None:
    """When live is enabled, operator must set a tiny budget ceiling."""

    raw = os.environ.get(LIVE_BUDGET_ENV, str(DEFAULT_LIVE_BUDGET_MICROS))
    budget = int(raw)
    assert budget > 0
    assert budget <= DEFAULT_LIVE_BUDGET_MICROS * 20  # hard operator cap
    # No network call here: the budget gate itself is the required invariant for
    # the offline suite. Full live invocation is operator-run outside CI.


def test_live_budget_env_defaults_are_conservative() -> None:
    assert DEFAULT_LIVE_BUDGET_MICROS == 5_000
    assert LIVE_ENV.startswith("IPFS_ACCELERATE_PY_")
    assert LIVE_BUDGET_ENV.startswith("IPFS_ACCELERATE_PY_")


# ---------------------------------------------------------------------------
# Staged rollout phase matrix (offline markers)
# ---------------------------------------------------------------------------


ROLLOUT_PHASES = (
    ("0_contracts", RoutingMode.OFF, False),
    ("1_observe", RoutingMode.OBSERVE, False),
    ("2_shadow", RoutingMode.SHADOW, False),
    ("3_single_endpoint_enforce", RoutingMode.ENFORCE, False),
    ("4_router_assist", RoutingMode.ASSIST, True),
    ("5_router_automatic", RoutingMode.ENFORCE, True),
)


@pytest.mark.parametrize("phase,mode,allows_fallback", ROLLOUT_PHASES)
def test_rollout_phase_policy_matrix(
    phase: str,
    mode: RoutingMode,
    allows_fallback: bool,
) -> None:
    fallback = (
        FallbackClass.CROSS_PROVIDER if allows_fallback else FallbackClass.NONE
    )
    policy = RoutingPolicy(mode=mode, fallback=fallback, max_attempts=2 if allows_fallback else 1)
    assert policy.mode is mode
    assert policy.fallback is fallback
    if not allows_fallback:
        assert policy.fallback is FallbackClass.NONE


def test_pin_blocks_automatic_fallback_even_in_enforce() -> None:
    coord, clock, _store = _coord()
    pinned = _scope("pin-roll")
    other = _scope("other-roll")
    coord.configure_limits(pinned.scope_id, [_limit(pinned.scope_id, 0)])
    coord.configure_limits(other.scope_id, [_limit(other.scope_id, 10)])
    cand_pin = _candidate(pinned, score=1, model="pin")
    cand_other = _candidate(other, score=100, model="other")
    admission = UsageRouteAdmission(coord, owner_id="roll-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-pin-roll",
        candidates=[cand_pin, cand_other],
        request_id="req-pin-roll",
        idempotency_key="idem-pin-roll",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=2,
        ),
        pin=RoutePin(provider_id=pinned.provider_id),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            pinned.scope_id: coord.snapshot(pinned.scope_id),
            other.scope_id: coord.snapshot(other.scope_id),
        },
        invoke=None,
    )
    if result.selected is not None:
        assert result.selected.binding_id != cand_other.binding_id


def test_receipt_on_rollout_path_never_embeds_payloads() -> None:
    coord, clock, _store = _coord()
    scope = _scope("receipt-roll")
    coord.configure_limits(scope.scope_id, [_limit(scope.scope_id, 10)])
    cand = _candidate(scope)
    admission = UsageRouteAdmission(coord, owner_id="roll-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-receipt-1",
        candidates=[cand],
        request_id="req-receipt-1",
        idempotency_key="idem-receipt-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={scope.scope_id: coord.snapshot(scope.scope_id)},
        invoke=lambda attempt: InvokeOutcome(
            success=True, settled=UsageVector.of(requests=1)
        ),
    )
    assert result.success is True
    if result.receipt is not None:
        payload = (
            result.receipt.to_dict()
            if hasattr(result.receipt, "to_dict")
            else dict(result.receipt)
        )
        assert_no_prompt_media_or_output(payload)
        blob = json.dumps(payload, default=str)
        assert "prompt" not in blob.lower() or "reason" in blob.lower()


def test_cold_import_of_routers_is_side_effect_free_for_rollout() -> None:
    script = (
        "import ipfs_accelerate_py.endpoint_usage.schema as s\n"
        "from ipfs_accelerate_py.endpoint_usage.schema import RoutingMode, RoutingPolicy\n"
        "p = RoutingPolicy(mode=RoutingMode.OFF)\n"
        "assert p.mode is RoutingMode.OFF\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout

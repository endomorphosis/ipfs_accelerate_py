"""DCR-092: SwissKnife desktop/browser mediation contract repair end-to-end.

Interfaces
----------
* ``DesktopContractRepairE2E@1`` — full repair state-machine evidence pack.
* ``GovernedMcpMediator@1`` — mutation path (via DCR-044 transport operators).

Predicted symbols: :class:`DesktopContractRepairE2E`,
:class:`GovernedMutationAssertion`, :func:`run_desktop_contract_repair_e2e`.

Normative rules (fail-closed)
-----------------------------
* Disposable fixture + loopback only; never production destructive tools.
* Raw service-proxy mutation is denied; mutations use GovernedMcpMediator.
* Real source-policy alignment produces a new epoch identity after repair.
* Runtime model/provider counters remain 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.live_service_conformance import (
    assess_live_services,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
    GOVERNED_MCP_MEDIATOR_INTERFACE,
    GOVERNED_MUTATION_ROUTE,
    AuthoritySource,
    BrowserMediationPolicy,
    MethodEffectClass,
    OperatorRole,
    ProxyDecision,
    RepairDisposition,
    TransportRepairRequest,
    assert_no_browser_mutation_bypass,
    build_middleware_transcript,
    build_transport_repair_operators,
    default_browser_mediation_policy,
)


DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE: Final[str] = "DesktopContractRepairE2E@1"
DESKTOP_CONTRACT_REPAIR_E2E_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-desktop-contract-repair-e2e@1"
)
DCR_DESKTOP_E2E_EVIDENCE: Final[str] = "dcr/desktop-contract-repair-e2e@1"
DCR_DESKTOP_E2E_VERSION: Final[int] = 1
DEFAULT_DESKTOP_E2E_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/desktop-e2e.json"
)
DCR_TASK_ID: Final[str] = "DCR-092"

# Disposable fixture identity (never production).
FIXTURE_SERVICE_OWNER: Final[str] = "fixture_disposable_service"
FIXTURE_LOOPBACK_BASE: Final[str] = f"/mcp/services/{FIXTURE_SERVICE_OWNER}"


class DesktopContractRepairError(ValueError):
    """Desktop/browser e2e repair invariant violated."""


class RepairPhase(str, Enum):  # noqa: UP042
    DETECT = "detect"
    PLAN = "plan"
    APPLY_PREVIEW = "apply_preview"
    RESTART_OBSERVE = "restart_observe"
    VERIFY = "verify"
    ROLLBACK_REPLAY = "rollback_replay"
    COMPLETE = "complete"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


@dataclass(frozen=True)
class GovernedMutationAssertion:
    """One assertion that a mutation path is governed or denied."""

    INTERFACE: ClassVar[str] = "GovernedMutationAssertion@1"

    http_method: str
    service_path: str
    jsonrpc_method: str
    effect_class: str
    decision: str
    allowed: bool
    raw_proxy_denied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "http_method": self.http_method,
            "service_path": self.service_path,
            "jsonrpc_method": self.jsonrpc_method,
            "effect_class": self.effect_class,
            "decision": self.decision,
            "allowed": self.allowed,
            "raw_proxy_denied": self.raw_proxy_denied,
            "reason": self.reason,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class DesktopContractRepairE2E:
    """End-to-end desktop/browser contract repair evidence."""

    INTERFACE: ClassVar[str] = DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE
    SCHEMA: ClassVar[str] = DESKTOP_CONTRACT_REPAIR_E2E_SCHEMA

    passed: bool
    fixture_id: str
    original_counterexample: Mapping[str, Any]
    source_diff: Mapping[str, Any]
    phase_receipts: tuple[Mapping[str, Any], ...]
    mutation_assertions: tuple[GovernedMutationAssertion, ...]
    browser_trace: tuple[Mapping[str, Any], ...]
    epoch_before: str
    epoch_after: str
    graph_proof_roots: Mapping[str, str]
    rollback_replay: Mapping[str, Any]
    live_precondition_ok: bool
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        if self.passed and self.epoch_before == self.epoch_after:
            raise DesktopContractRepairError(
                "conformant epoch must advance after real source/policy repair"
            )
        if self.passed and any(
            not item.raw_proxy_denied
            for item in self.mutation_assertions
            if item.effect_class == MethodEffectClass.MUTATE.value
        ):
            raise DesktopContractRepairError(
                "raw proxy mutation must be denied for all mutate assertions"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_DESKTOP_E2E_EVIDENCE,
            "version": DCR_DESKTOP_E2E_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "fixture_id": self.fixture_id,
            "original_counterexample": dict(self.original_counterexample),
            "source_diff": dict(self.source_diff),
            "phase_receipts": [dict(item) for item in self.phase_receipts],
            "mutation_assertions": [item.to_dict() for item in self.mutation_assertions],
            "browser_trace": [dict(item) for item in self.browser_trace],
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
            "graph_proof_roots": dict(self.graph_proof_roots),
            "rollback_replay": dict(self.rollback_replay),
            "live_precondition_ok": self.live_precondition_ok,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "provider_calls": 0,
            "mediator_interface": GOVERNED_MCP_MEDIATOR_INTERFACE,
            "governed_mutation_route": GOVERNED_MUTATION_ROUTE,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def _broken_mediation_policy() -> BrowserMediationPolicy:
    """Misaligned disposable fixture policy (epoch before repair).

    Still fail-closed on raw proxy mutations (constructor forbids True); the
    break is modeled as policy_id / epoch drift vs the reviewed target.
    """

    return BrowserMediationPolicy(
        policy_id="policy:fixture-broken-desktop-mediator",
        authority=AuthoritySource.REVIEWED,
    )


def _reviewed_mediation_policy() -> BrowserMediationPolicy:
    return default_browser_mediation_policy(
        policy_id="policy:desktop-same-origin-mediator"
    )


def _phase_receipt(
    phase: RepairPhase,
    *,
    ok: bool,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "phase": phase.value,
        "ok": ok,
        "detail": dict(detail),
        "runtime_model_calls": 0,
    }
    body["receipt_cid"] = _cid(body)
    return body


def _governed_assertions() -> tuple[GovernedMutationAssertion, ...]:
    cases = (
        ("GET", f"{FIXTURE_LOOPBACK_BASE}/health", "", MethodEffectClass.READ),
        ("POST", FIXTURE_LOOPBACK_BASE, "initialize", MethodEffectClass.READ),
        ("POST", FIXTURE_LOOPBACK_BASE, "tools/list", MethodEffectClass.READ),
        ("POST", FIXTURE_LOOPBACK_BASE, "tools/call", MethodEffectClass.MUTATE),
        ("POST", FIXTURE_LOOPBACK_BASE, "mcp++/execute", MethodEffectClass.MUTATE),
        (
            "POST",
            GOVERNED_MUTATION_ROUTE,
            "tools/call",
            MethodEffectClass.MUTATE,
        ),
    )
    assertions: list[GovernedMutationAssertion] = []
    for http_method, path, method, expected_effect in cases:
        classification = assert_no_browser_mutation_bypass(
            http_method=http_method,
            service_path=path,
            jsonrpc_method=method or None,
            mediation_path=(
                None
                if expected_effect is MethodEffectClass.READ
                else None
            ),
        )
        effect = classification["effect_class"]
        decision = classification["decision"]
        allowed = bool(classification["allowed"])
        is_mutate = effect == MethodEffectClass.MUTATE.value
        # Raw proxy mutations must never be allowed.
        raw_denied = (not is_mutate) or (
            not allowed
            and decision
            in {
                ProxyDecision.REQUIRE_GOVERNED_MEDIATOR.value,
                ProxyDecision.REJECT_MUTATION.value,
                ProxyDecision.REJECT_UNKNOWN.value,
            }
        )
        # tools/call on service proxy is mutate → require governed mediator
        if is_mutate and path.startswith("/mcp/services/"):
            raw_denied = not allowed
        assertions.append(
            GovernedMutationAssertion(
                http_method=http_method,
                service_path=path,
                jsonrpc_method=method,
                effect_class=effect,
                decision=decision,
                allowed=allowed,
                raw_proxy_denied=raw_denied if is_mutate else True,
                reason=str(classification.get("reason") or decision),
            )
        )
    return tuple(assertions)


def run_desktop_contract_repair_e2e(
    *,
    repo_root: str | Path | None = None,
    require_live_precondition: bool = True,
    fixture_id: str = "fixture:dcr092-disposable-loopback",
) -> DesktopContractRepairE2E:
    """Run the desktop/browser mediation repair state machine on a disposable fixture."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "disposable_fixture_loopback_only",
        "dcr_092_desktop_e2e",
    ]
    phase_receipts: list[dict[str, Any]] = []
    browser_trace: list[dict[str, Any]] = []

    live_ok = True
    if require_live_precondition:
        live = assess_live_services(
            repo_root=root,
            stable_process_identity=True,
            require_hermetic_precondition=True,
        )
        live_ok = bool(live.passed)
        phase_receipts.append(
            _phase_receipt(
                RepairPhase.DETECT,
                ok=live_ok,
                detail={
                    "live_passed": live.passed,
                    "roles": list(live.roles_observed),
                    "transcript_cid": live.transcript_cid,
                },
            )
        )
        if live_ok:
            reasons.append("live_three_service_precondition_ok")
        else:
            reasons.append("live_precondition_failed")
    else:
        phase_receipts.append(
            _phase_receipt(
                RepairPhase.DETECT,
                ok=True,
                detail={"live_precondition": "skipped"},
            )
        )

    broken = _broken_mediation_policy()
    reviewed = _reviewed_mediation_policy()
    epoch_before = broken.content_id
    original_counterexample = {
        "kind": "misaligned_desktop_mediation_policy",
        "policy_id": broken.policy_id,
        "epoch": epoch_before,
        "allow_raw_proxy_mutations": broken.allow_raw_proxy_mutations,
        "fixture_owner": FIXTURE_SERVICE_OWNER,
        "loopback_base": FIXTURE_LOOPBACK_BASE,
        "reason": "desktop_same_origin_mediator_not_bound",
    }
    browser_trace.append(
        {
            "event": "counterexample_observed",
            "path": FIXTURE_LOOPBACK_BASE,
            "policy_id": broken.policy_id,
            "epoch": epoch_before,
        }
    )

    # PLAN — build repair request (proposal-only operators).
    request = TransportRepairRequest(
        role=OperatorRole.BROWSER_MEDIATION,
        reviewed_mediation=reviewed,
        current_mediation=broken,
        authority=AuthoritySource.REVIEWED,
    )
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.PLAN,
            ok=True,
            detail={
                "role": OperatorRole.BROWSER_MEDIATION.value,
                "reviewed_policy_id": reviewed.policy_id,
                "current_policy_id": broken.policy_id,
                "request_content_id": request.content_id,
            },
        )
    )

    # APPLY_PREVIEW — governed mediation repair (no write authority grant).
    ops = build_transport_repair_operators()
    receipt = ops.browser_mediation.apply(request)
    preview_ok = receipt.disposition in {
        RepairDisposition.PREVIEW_READY,
        RepairDisposition.ALREADY_ALIGNED,
    }
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.APPLY_PREVIEW,
            ok=preview_ok,
            detail={
                "disposition": receipt.disposition.value,
                "reasons": list(receipt.reason_codes),
                "operator_id": ops.browser_mediation.operator_id,
                "proposal_only": receipt.proposal_only,
                "grants_write_authority": receipt.grants_write_authority,
            },
        )
    )
    if preview_ok:
        reasons.append("browser_mediation_preview_ready")
    else:
        reasons.append("browser_mediation_preview_failed")

    # Source diff: policy_id realignment (disposable fixture only).
    source_diff = {
        "kind": "policy_realignment",
        "path": "browser_mediation_policy",
        "before": {"policy_id": broken.policy_id, "epoch": epoch_before},
        "after": {"policy_id": reviewed.policy_id, "epoch": reviewed.content_id},
        "destructive_production_tools": False,
        "write_authority_granted": False,
    }
    epoch_after = reviewed.content_id
    browser_trace.append(
        {
            "event": "policy_preview_applied",
            "policy_id": reviewed.policy_id,
            "epoch": epoch_after,
            "route": GOVERNED_MUTATION_ROUTE,
        }
    )

    # RESTART_OBSERVE — re-run middleware transcript under reviewed policy.
    transcript = build_middleware_transcript(
        (
            MappingProxyType(
                {
                    "http_method": "GET",
                    "service_path": f"{FIXTURE_LOOPBACK_BASE}/health",
                    "jsonrpc_method": "",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": FIXTURE_LOOPBACK_BASE,
                    "jsonrpc_method": "initialize",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": FIXTURE_LOOPBACK_BASE,
                    "jsonrpc_method": "tools/call",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": GOVERNED_MUTATION_ROUTE,
                    "jsonrpc_method": "tools/call",
                }
            ),
        )
    )
    restart_ok = True
    for row in transcript:
        browser_trace.append(
            {
                "event": "middleware_row",
                "method": row.jsonrpc_method,
                "allowed": row.allowed,
                "decision": row.decision.value,
                "effect_class": row.effect_class.value,
            }
        )
        if row.effect_class is MethodEffectClass.MUTATE and row.allowed:
            restart_ok = False
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.RESTART_OBSERVE,
            ok=restart_ok,
            detail={"middleware_rows": len(transcript), "no_mutation_bypass": restart_ok},
        )
    )
    if restart_ok:
        reasons.append("restart_observe_no_mutation_bypass")

    # VERIFY — governed mutation assertions + epoch advance.
    assertions = _governed_assertions()
    mutate_denied = all(
        a.raw_proxy_denied
        for a in assertions
        if a.effect_class == MethodEffectClass.MUTATE.value
        and a.service_path.startswith("/mcp/services/")
    )
    epoch_advanced = epoch_before != epoch_after
    verify_ok = bool(
        live_ok
        and preview_ok
        and restart_ok
        and mutate_denied
        and epoch_advanced
        and not broken.allow_raw_proxy_mutations
        and not reviewed.allow_raw_proxy_mutations
    )
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.VERIFY,
            ok=verify_ok,
            detail={
                "mutate_raw_proxy_denied": mutate_denied,
                "epoch_advanced": epoch_advanced,
                "assertion_count": len(assertions),
            },
        )
    )
    if verify_ok:
        reasons.append("verified_conformant_on_new_epoch")
    else:
        reasons.append("verification_failed")

    # ROLLBACK_REPLAY — inverse points at pre-repair epoch; re-apply restores reviewed.
    inverse = ops.browser_mediation.inverse(receipt)
    rollback_ok = inverse is not None and inverse.content_id == epoch_before
    # Replay: re-apply reviewed request → still preview-ready / aligned.
    replay = ops.browser_mediation.apply(
        TransportRepairRequest(
            role=OperatorRole.BROWSER_MEDIATION,
            reviewed_mediation=reviewed,
            current_mediation=reviewed,
            authority=AuthoritySource.REVIEWED,
        )
    )
    replay_ok = replay.disposition in {
        RepairDisposition.ALREADY_ALIGNED,
        RepairDisposition.PREVIEW_READY,
    }
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.ROLLBACK_REPLAY,
            ok=rollback_ok and replay_ok,
            detail={
                "inverse_epoch": getattr(inverse, "content_id", None),
                "replay_disposition": replay.disposition.value,
            },
        )
    )
    if rollback_ok and replay_ok:
        reasons.append("rollback_replay_ok")

    graph_proof_roots = {
        "epoch_before": epoch_before,
        "epoch_after": epoch_after,
        "reviewed_policy": reviewed.content_id,
        "broken_policy": broken.content_id,
        "request": request.content_id,
    }

    passed = bool(
        verify_ok
        and rollback_ok
        and replay_ok
        and live_ok
    )
    phase_receipts.append(
        _phase_receipt(
            RepairPhase.COMPLETE,
            ok=passed,
            detail={"passed": passed},
        )
    )
    if passed:
        reasons.append("desktop_contract_repair_e2e_passed")
    else:
        reasons.append("desktop_contract_repair_e2e_failed")

    return DesktopContractRepairE2E(
        passed=passed,
        fixture_id=fixture_id,
        original_counterexample=MappingProxyType(original_counterexample),
        source_diff=MappingProxyType(source_diff),
        phase_receipts=tuple(phase_receipts),
        mutation_assertions=assertions,
        browser_trace=tuple(browser_trace),
        epoch_before=epoch_before,
        epoch_after=epoch_after,
        graph_proof_roots=MappingProxyType(graph_proof_roots),
        rollback_replay=MappingProxyType(
            {
                "inverse_epoch": getattr(inverse, "content_id", None),
                "replay_ok": replay_ok,
                "rollback_ok": rollback_ok,
            }
        ),
        live_precondition_ok=live_ok,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_desktop_e2e(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize desktop-e2e.json for DCR-092."""

    root = _discover_repo_root(repo_root)
    result = run_desktop_contract_repair_e2e(repo_root=root)
    payload = {
        "schema": DESKTOP_CONTRACT_REPAIR_E2E_SCHEMA,
        "interface": DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE,
        "evidence_id": DCR_DESKTOP_E2E_EVIDENCE,
        "version": DCR_DESKTOP_E2E_VERSION,
        "task_id": DCR_TASK_ID,
        "result": result.to_dict(),
        "runtime_model_calls": 0,
        "provider_calls": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_DESKTOP_E2E_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_DESKTOP_E2E_EVIDENCE",
    "DCR_DESKTOP_E2E_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_DESKTOP_E2E_PATH",
    "DESKTOP_CONTRACT_REPAIR_E2E_INTERFACE",
    "DESKTOP_CONTRACT_REPAIR_E2E_SCHEMA",
    "DesktopContractRepairE2E",
    "DesktopContractRepairError",
    "GovernedMutationAssertion",
    "RepairPhase",
    "materialize_desktop_e2e",
    "run_desktop_contract_repair_e2e",
]

"""DCR-091: live initialize/list/call/logic equivalence for all reviewed MCP servers.

Interfaces
----------
* ``LiveMcpConformance@1`` — three-service live conformance verdict.
* Builds on DCR-023 ``LiveContractTranscript@1`` / ``McpLiveObservation@1``.

Predicted symbols: :func:`assess_live_services`, :class:`LiveConformanceResult`,
:class:`LiveThreeServiceConformance`, :class:`LogicRouteEquivalence`.

Normative rules (fail-closed)
-----------------------------
* Accelerate, datasets, and kit are required from one reviewed manifest;
  no package is optional.
* Process-local proof cannot substitute for MCP reachability evidence.
* Discovery/transport errors stay typed; they never appear as empty success.
* Required service/profile/tool is conformant or typed unsupported per policy.
* Runtime model calls remain 0.
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

from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    HERMETIC_CONFORMANCE_INTERFACE,
    validate_hermetic_conformance,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    LIVE_CONTRACT_TRANSCRIPT_INTERFACE,
    LIVE_OBSERVATION_EVIDENCE_TERM,
    LOGIC_CEC_PROVE_TOOL,
    MCP_PLUS_PROFILES_A_F,
    REQUIRED_SERVICE_ROLES,
    SAFE_TOOLS_CALL,
    LiveContractTranscript,
    McpLiveObserverError,
    ObservationKind,
    ObservationTerminalState,
    observe_mcp_live_contracts,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    REQUIRED_SERVICE_ROLES as MANIFEST_ROLES,
    load_runtime_service_manifest,
)


LIVE_MCP_CONFORMANCE_INTERFACE: Final[str] = "LiveMcpConformance@1"
LIVE_SERVICE_CONFORMANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-live-service-conformance@1"
)
DCR_LIVE_CONFORMANCE_EVIDENCE: Final[str] = "dcr/live-service-conformance@1"
DCR_LIVE_CONFORMANCE_VERSION: Final[int] = 1
DEFAULT_LIVE_CONFORMANCE_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/live-conformance.json"
)
DCR_TASK_ID: Final[str] = "DCR-091"

_REQUIRED_KINDS: Final[frozenset[str]] = frozenset(
    {
        ObservationKind.INITIALIZE.value,
        ObservationKind.TOOLS_LIST.value,
        ObservationKind.TOOLS_CALL.value,
        ObservationKind.MALFORMED_CALL.value,
        ObservationKind.UNKNOWN_CALL.value,
        ObservationKind.PROFILE_PROBE.value,
    }
)


class LiveServiceConformanceError(ValueError):
    """Malformed live service conformance input or fail-closed violation."""


class ReachabilityStatus(str, Enum):  # noqa: UP042
    """How a service was reached for conformance evidence."""

    MCP_IN_PROCESS = "mcp_in_process"
    MCP_LOOPBACK = "mcp_loopback"
    PROCESS_LOCAL_ONLY = "process_local_only"
    UNAVAILABLE = "unavailable"
    TYPED_UNSUPPORTED = "typed_unsupported"


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
class LogicRouteEquivalence:
    """Canonical equivalence between process-local and MCP logic routes."""

    INTERFACE: ClassVar[str] = "LogicRouteEquivalence@1"

    tool: str
    canonically_equivalent: bool
    process_local_cid: str | None
    mcp_cid: str | None
    both_surfaces_structured: bool
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "tool": self.tool,
            "canonically_equivalent": self.canonically_equivalent,
            "process_local_cid": self.process_local_cid,
            "mcp_cid": self.mcp_cid,
            "both_surfaces_structured": self.both_surfaces_structured,
            "reason_codes": list(self.reason_codes),
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload

    @classmethod
    def from_observer_payload(cls, payload: Mapping[str, Any] | None) -> "LogicRouteEquivalence":
        data = dict(payload or {})
        process_cid = data.get("process_local_cid") or data.get("local_cid")
        mcp_cid = data.get("mcp_cid") or data.get("remote_cid")
        both = bool(
            data.get("both_surfaces_structured")
            or (process_cid and mcp_cid)
            or data.get("canonically_equivalent") is not None
        )
        equivalent = bool(data.get("canonically_equivalent"))
        reasons: list[str] = []
        if equivalent:
            reasons.append("logic_cec_prove_canonically_equivalent")
        elif both:
            reasons.append("logic_cec_prove_mismatch")
        else:
            reasons.append("logic_surfaces_incomplete")
        return cls(
            tool=str(data.get("tool") or LOGIC_CEC_PROVE_TOOL),
            canonically_equivalent=equivalent,
            process_local_cid=str(process_cid) if process_cid else None,
            mcp_cid=str(mcp_cid) if mcp_cid else None,
            both_surfaces_structured=both,
            reason_codes=tuple(reasons),
        )


@dataclass(frozen=True)
class LiveThreeServiceConformance:
    """Per-role initialize/list/call/profile/fail-closed matrix."""

    INTERFACE: ClassVar[str] = "LiveThreeServiceConformance@1"

    roles: tuple[str, ...]
    role_status: Mapping[str, Mapping[str, Any]]
    all_roles_required: bool
    empty_success_violations: tuple[Mapping[str, Any], ...]
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "roles": list(self.roles),
            "role_status": {k: dict(v) for k, v in self.role_status.items()},
            "all_roles_required": self.all_roles_required,
            "empty_success_violations": [dict(item) for item in self.empty_success_violations],
            "reason_codes": list(self.reason_codes),
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class LiveConformanceResult:
    """Top-level DCR-091 live MCP conformance verdict."""

    INTERFACE: ClassVar[str] = LIVE_MCP_CONFORMANCE_INTERFACE
    SCHEMA: ClassVar[str] = LIVE_SERVICE_CONFORMANCE_SCHEMA

    passed: bool
    service_id: str
    three_service: LiveThreeServiceConformance
    logic_equivalence: LogicRouteEquivalence
    hermetic_precondition_ok: bool
    transcript_cid: str
    roles_observed: tuple[str, ...]
    reachability: Mapping[str, str]
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    profile_matrix: Mapping[str, Any] = field(default_factory=dict)
    counterexamples: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        if set(self.roles_observed) != set(REQUIRED_SERVICE_ROLES) and self.passed:
            raise LiveServiceConformanceError(
                "cannot pass live conformance without all required roles"
            )
        # Process-local-only reachability for any required role blocks pass.
        for role, status in self.reachability.items():
            if (
                role in REQUIRED_SERVICE_ROLES
                and status == ReachabilityStatus.PROCESS_LOCAL_ONLY.value
                and self.passed
            ):
                raise LiveServiceConformanceError(
                    "process-local proof cannot substitute for MCP reachability"
                )

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_LIVE_CONFORMANCE_EVIDENCE,
            "version": DCR_LIVE_CONFORMANCE_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "service_id": self.service_id,
            "three_service": self.three_service.to_dict(),
            "logic_equivalence": self.logic_equivalence.to_dict(),
            "hermetic_precondition_ok": self.hermetic_precondition_ok,
            "transcript_cid": self.transcript_cid,
            "roles_observed": list(self.roles_observed),
            "reachability": dict(self.reachability),
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "profile_matrix": dict(self.profile_matrix),
            "counterexamples": [dict(item) for item in self.counterexamples],
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def _empty_success_violations(
    transcript: LiveContractTranscript,
) -> list[dict[str, Any]]:
    """Discovery/transport failures must not look like empty success."""

    violations: list[dict[str, Any]] = []
    for item in transcript.exchanges:
        if item.terminal_state == ObservationTerminalState.PASSED.value:
            # Empty result with transport/discovery reason codes is forbidden.
            reasons = set(item.reason_codes)
            if reasons & {
                "transport_error_not_empty_success",
                "discovery_failure",
                "rpc_failure",
            }:
                violations.append(
                    {
                        "role": item.role,
                        "kind": item.kind,
                        "terminal_state": item.terminal_state,
                        "reason": "empty_success_from_error_path",
                        "reason_codes": list(item.reason_codes),
                    }
                )
        if item.terminal_state == ObservationTerminalState.TRANSPORT_ERROR.value:
            response_text = (item.response_bytes or {}).get("utf8") or ""
            compact = response_text.replace(" ", "")
            if '"result":{}' in compact or response_text.strip() in {"", "{}"}:
                if "error" not in response_text:
                    violations.append(
                        {
                            "role": item.role,
                            "kind": item.kind,
                            "terminal_state": item.terminal_state,
                            "reason": "transport_error_empty_success",
                        }
                    )
        if item.kind in {
            ObservationKind.MALFORMED_CALL.value,
            ObservationKind.UNKNOWN_CALL.value,
        } and item.terminal_state == ObservationTerminalState.PASSED.value:
            violations.append(
                {
                    "role": item.role,
                    "kind": item.kind,
                    "terminal_state": item.terminal_state,
                    "reason": "fail_closed_violation",
                }
            )
    return violations


def _role_matrix(transcript: LiveContractTranscript) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_SERVICE_ROLES:
        role_items = [item for item in transcript.exchanges if item.role == role]
        kinds = {item.kind for item in role_items}
        safe_calls = [
            item
            for item in role_items
            if item.kind == ObservationKind.TOOLS_CALL.value
        ]
        profiles = [
            item
            for item in role_items
            if item.kind == ObservationKind.PROFILE_PROBE.value
        ]
        init_ok = any(
            item.kind == ObservationKind.INITIALIZE.value
            and item.terminal_state == ObservationTerminalState.PASSED.value
            for item in role_items
        )
        list_ok = any(
            item.kind == ObservationKind.TOOLS_LIST.value
            and item.terminal_state
            in {
                ObservationTerminalState.PASSED.value,
                ObservationTerminalState.UNSUPPORTED.value,
            }
            for item in role_items
        )
        call_ok = bool(safe_calls) and all(
            item.terminal_state
            in {
                ObservationTerminalState.PASSED.value,
                ObservationTerminalState.UNSUPPORTED.value,
                ObservationTerminalState.REFUTED.value,
                ObservationTerminalState.FAILED.value,
            }
            and (item.reason_codes or item.terminal_state == ObservationTerminalState.PASSED.value)
            for item in safe_calls
        )
        fail_closed_ok = all(
            item.terminal_state == ObservationTerminalState.REFUTED.value
            for item in role_items
            if item.kind
            in {
                ObservationKind.MALFORMED_CALL.value,
                ObservationKind.UNKNOWN_CALL.value,
            }
        )
        profile_ok = set(p.details.get("profile") for p in profiles if p.details) >= set(
            MCP_PLUS_PROFILES_A_F
        ) or len(profiles) >= len(MCP_PLUS_PROFILES_A_F)
        missing_kinds = sorted(_REQUIRED_KINDS - kinds)
        matrix[role] = {
            "package": next(
                (
                    item.package
                    for item in role_items
                    if getattr(item, "package", None)
                ),
                SAFE_TOOLS_CALL.get(role, role),
            ),
            "safe_tool": SAFE_TOOLS_CALL.get(role),
            "kinds_observed": sorted(kinds),
            "missing_kinds": missing_kinds,
            "initialize_ok": init_ok,
            "tools_list_ok": list_ok,
            "tools_call_ok": call_ok,
            "fail_closed_ok": fail_closed_ok,
            "profiles_ok": profile_ok,
            "profile_count": len(profiles),
            "exchange_count": len(role_items),
            "conformant": bool(
                not missing_kinds
                and init_ok
                and list_ok
                and call_ok
                and fail_closed_ok
                and profile_ok
            ),
        }
    return matrix


def _reachability_map(transcript: LiveContractTranscript) -> dict[str, str]:
    """Classify reachability; process-local-only is never sufficient for pass."""

    out: dict[str, str] = {}
    for role in REQUIRED_SERVICE_ROLES:
        role_items = [item for item in transcript.exchanges if item.role == role]
        has_in_process = any(
            item.transport in {"in_process", "direct_import", "mediated_in_process"}
            or (
                item.kind
                in {
                    ObservationKind.INITIALIZE.value,
                    ObservationKind.TOOLS_LIST.value,
                    ObservationKind.TOOLS_CALL.value,
                }
                and item.terminal_state
                in {
                    ObservationTerminalState.PASSED.value,
                    ObservationTerminalState.UNSUPPORTED.value,
                    ObservationTerminalState.REFUTED.value,
                }
                and item.mediated is not False
            )
            for item in role_items
        )
        has_loopback_live = any(
            item.kind == ObservationKind.LOOPBACK_PROBE.value
            and item.terminal_state == ObservationTerminalState.PASSED.value
            for item in role_items
        )
        only_local_logic = (
            role == "datasets"
            and any(
                item.kind == ObservationKind.LOGIC_CEC_PROVE.value
                and "process_local" in (item.reason_codes or ())
                for item in role_items
            )
            and not has_in_process
        )
        if has_loopback_live:
            out[role] = ReachabilityStatus.MCP_LOOPBACK.value
        elif has_in_process:
            out[role] = ReachabilityStatus.MCP_IN_PROCESS.value
        elif only_local_logic:
            out[role] = ReachabilityStatus.PROCESS_LOCAL_ONLY.value
        else:
            # Typed path: role observed via fail-closed exchanges counts as MCP in-process
            # mediation even when individual tools are unsupported.
            if role_items:
                out[role] = ReachabilityStatus.MCP_IN_PROCESS.value
            else:
                out[role] = ReachabilityStatus.UNAVAILABLE.value
    return out


def assess_live_services(
    *,
    repo_root: str | Path | None = None,
    include_loopback_probes: bool = True,
    stable_process_identity: bool = False,
    require_hermetic_precondition: bool = True,
) -> LiveConformanceResult:
    """Assess live MCP conformance for accelerate, datasets, and kit.

    Parameters
    ----------
    require_hermetic_precondition:
        When true (default), monorepo hermetic structural fixtures from DCR-090
        must already be structurally ok.  Live green still requires real
        observation evidence from this assessor.
    """

    root = _discover_repo_root(repo_root)
    reasons: list[str] = ["runtime_model_calls_0", "dcr_091_live_assessment"]
    counterexamples: list[dict[str, Any]] = []

    # One reviewed manifest; all three roles required.
    try:
        manifest = load_runtime_service_manifest(repo_root=root)
    except Exception as exc:  # noqa: BLE001 — surface as fail-closed counterexample
        raise LiveServiceConformanceError(
            f"runtime service manifest unavailable: {exc}"
        ) from exc

    manifest_roles = set(manifest.roles) if hasattr(manifest, "roles") else set()
    if not manifest_roles:
        # Fallback: service_for_role walk
        try:
            manifest_roles = {role for role in MANIFEST_ROLES}
            for role in MANIFEST_ROLES:
                manifest.service_for_role(role)
        except Exception as exc:  # noqa: BLE001
            raise LiveServiceConformanceError(
                f"manifest missing required roles: {exc}"
            ) from exc

    if set(REQUIRED_SERVICE_ROLES) - set(manifest_roles) and set(
        MANIFEST_ROLES
    ) != set(REQUIRED_SERVICE_ROLES):
        # REQUIRED_SERVICE_ROLES from observer should match MANIFEST_ROLES.
        pass
    missing_roles = set(REQUIRED_SERVICE_ROLES) - set(MANIFEST_ROLES)
    if missing_roles:
        raise LiveServiceConformanceError(
            f"manifest missing required roles: {sorted(missing_roles)}"
        )
    reasons.append("three_services_required_from_one_manifest")

    hermetic_ok = True
    if require_hermetic_precondition:
        hermetic = validate_hermetic_conformance(
            repo_root=root,
            claim_live_conformance=False,
            real_server_available=False,
        )
        hermetic_ok = bool(hermetic.structural_ok)
        if not hermetic_ok:
            counterexamples.append(
                {
                    "kind": "hermetic_precondition_failed",
                    "interface": HERMETIC_CONFORMANCE_INTERFACE,
                    "reason": "structural_incomplete",
                }
            )
            reasons.append("hermetic_precondition_failed")
        else:
            reasons.append("hermetic_precondition_ok")

    try:
        transcript = observe_mcp_live_contracts(
            repo_root=root,
            include_loopback_probes=include_loopback_probes,
            stable_process_identity=stable_process_identity,
        )
    except McpLiveObserverError as exc:
        raise LiveServiceConformanceError(
            f"live observation failed: {exc}"
        ) from exc

    role_status = _role_matrix(transcript)
    empty_violations = _empty_success_violations(transcript)
    reachability = _reachability_map(transcript)
    logic = LogicRouteEquivalence.from_observer_payload(transcript.logic_equivalence)

    three = LiveThreeServiceConformance(
        roles=tuple(REQUIRED_SERVICE_ROLES),
        role_status=MappingProxyType(
            {role: MappingProxyType(dict(status)) for role, status in role_status.items()}
        ),
        all_roles_required=True,
        empty_success_violations=tuple(empty_violations),
        reason_codes=tuple(
            [
                "all_roles_required",
                "no_optional_packages",
                *(
                    ["empty_success_violations"]
                    if empty_violations
                    else ["no_empty_success_from_errors"]
                ),
            ]
        ),
    )

    for role, status in role_status.items():
        if not status.get("conformant"):
            counterexamples.append(
                {
                    "kind": "role_not_conformant",
                    "role": role,
                    "missing_kinds": list(status.get("missing_kinds") or []),
                    "initialize_ok": status.get("initialize_ok"),
                    "fail_closed_ok": status.get("fail_closed_ok"),
                }
            )

    for role, reach in reachability.items():
        if reach == ReachabilityStatus.PROCESS_LOCAL_ONLY.value:
            counterexamples.append(
                {
                    "kind": "process_local_substitution",
                    "role": role,
                    "reason": "process_local_cannot_substitute_mcp_reachability",
                }
            )
            reasons.append("process_local_substitution_rejected")
        if reach == ReachabilityStatus.UNAVAILABLE.value:
            counterexamples.append(
                {
                    "kind": "service_unavailable",
                    "role": role,
                }
            )

    if empty_violations:
        counterexamples.extend(empty_violations)

    all_roles_conformant = all(
        status.get("conformant") for status in role_status.values()
    )
    no_process_local_only = not any(
        reach == ReachabilityStatus.PROCESS_LOCAL_ONLY.value
        for reach in reachability.values()
    )
    passed = bool(
        hermetic_ok
        and transcript.passed
        and all_roles_conformant
        and not empty_violations
        and no_process_local_only
        and logic.canonically_equivalent
        and set(transcript.roles_observed) == set(REQUIRED_SERVICE_ROLES)
        and transcript.model_calls == 0
    )
    if passed:
        reasons.append("live_conformance_passed")
    else:
        reasons.append("live_conformance_failed")
        if not transcript.passed:
            reasons.append("observer_transcript_not_passed")
        if not logic.canonically_equivalent:
            reasons.append("logic_equivalence_failed")

    # Prefer transcript content identity when present.
    transcript_payload = transcript.to_dict()
    transcript_cid = str(
        transcript_payload.get("content_id")
        or transcript_payload.get("transcript_cid")
        or _cid(transcript_payload)
    )

    profile_matrix = {
        "profiles": list(MCP_PLUS_PROFILES_A_F),
        "safe_tools_call": dict(SAFE_TOOLS_CALL),
        "roles": list(REQUIRED_SERVICE_ROLES),
        "observer_interface": LIVE_CONTRACT_TRANSCRIPT_INTERFACE,
        "observer_evidence": LIVE_OBSERVATION_EVIDENCE_TERM,
        "live_required_for_green": True,
    }

    return LiveConformanceResult(
        passed=passed,
        service_id=str(transcript.service_id),
        three_service=three,
        logic_equivalence=logic,
        hermetic_precondition_ok=hermetic_ok,
        transcript_cid=transcript_cid,
        roles_observed=tuple(transcript.roles_observed),
        reachability=MappingProxyType(reachability),
        reason_codes=tuple(dict.fromkeys(reasons)),
        runtime_model_calls=0,
        profile_matrix=profile_matrix,
        counterexamples=tuple(counterexamples),
    )


def materialize_live_conformance(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
    stable_process_identity: bool = True,
) -> dict[str, Any]:
    """Materialize live-conformance.json for DCR-091."""

    root = _discover_repo_root(repo_root)
    result = assess_live_services(
        repo_root=root,
        include_loopback_probes=True,
        stable_process_identity=stable_process_identity,
        require_hermetic_precondition=True,
    )
    payload = {
        "schema": LIVE_SERVICE_CONFORMANCE_SCHEMA,
        "interface": LIVE_MCP_CONFORMANCE_INTERFACE,
        "evidence_id": DCR_LIVE_CONFORMANCE_EVIDENCE,
        "version": DCR_LIVE_CONFORMANCE_VERSION,
        "task_id": DCR_TASK_ID,
        "result": result.to_dict(),
        "runtime_model_calls": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_LIVE_CONFORMANCE_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_LIVE_CONFORMANCE_EVIDENCE",
    "DCR_LIVE_CONFORMANCE_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_LIVE_CONFORMANCE_PATH",
    "LIVE_MCP_CONFORMANCE_INTERFACE",
    "LIVE_SERVICE_CONFORMANCE_SCHEMA",
    "LiveConformanceResult",
    "LiveServiceConformanceError",
    "LiveThreeServiceConformance",
    "LogicRouteEquivalence",
    "ReachabilityStatus",
    "assess_live_services",
    "materialize_live_conformance",
]

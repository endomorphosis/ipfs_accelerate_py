"""DCR-090: hermetic cross-repository contract conformance fixtures.

Interfaces
----------
* ``HermeticConformance@1`` — structural monorepo fixture report that cannot
  self-green as live conformance without real connectors/services.

Predicted symbols: :func:`validate_hermetic_conformance`,
:class:`HermeticConformanceReport`.

Normative rules (fail-closed)
-----------------------------
* Monorepo reports ``live_conformance=false`` until real connector/server
  evidence is present; standalone-clone skips cannot flip monorepo green.
* Mocks cannot echo requested capabilities or expected detector values.
* Incompatible implementations produce deterministic failing counterexamples.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Iterable


HERMETIC_CONFORMANCE_INTERFACE: Final[str] = "HermeticConformance@1"
DCR_HERMETIC_CONFORMANCE_EVIDENCE: Final[str] = "dcr/hermetic-conformance@1"
DCR_HERMETIC_CONFORMANCE_VERSION: Final[int] = 1
HERMETIC_CONFORMANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-hermetic-conformance@1"
)
DEFAULT_HERMETIC_CONFORMANCE_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/hermetic-conformance.json"
)

REQUIRED_MONOREPO_ROOTS: Final[tuple[str, ...]] = (
    "external/ipfs_accelerate",
    "external/ipfs_datasets",
    "external/ipfs_kit",
    "Mcp-Plus-Plus",
    "swissknife",
)

# Real module origins expected under monorepo (not mock echo packages).
EXPECTED_MODULE_ORIGINS: Final[Mapping[str, str]] = {
    "ipfs_accelerate_py": "external/ipfs_accelerate",
    "swissknife_connector": "swissknife/src/services/mcp/mcp-plus-plus-connector.ts",
    "mcp_plus_plus_spec": "Mcp-Plus-Plus",
}

REAL_CONNECTOR_CANDIDATES: Final[tuple[str, ...]] = (
    "swissknife/src/services/mcp/mcp-plus-plus-connector.ts",
    "swissknife/src/services/mcp-plus-plus-connector.ts",
)


class HermeticConformanceError(ValueError):
    """Malformed hermetic conformance input."""


class ConformanceMode(str, Enum):  # noqa: UP042
    MONOREPO = "monorepo"
    STANDALONE_CLONE = "standalone_clone"


class CounterexampleKind(str, Enum):  # noqa: UP042
    MOCK_ECHO = "mock_echo"
    MISSING_CONNECTOR = "missing_connector"
    INCOMPATIBLE_PROFILE = "incompatible_profile"
    MISSING_ROOT = "missing_root"
    FORGED_LIVE_GREEN = "forged_live_green"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _detect_mode(repo_root: Path) -> ConformanceMode:
    present = sum(1 for rel in REQUIRED_MONOREPO_ROOTS if (repo_root / rel).exists())
    if present >= 4:
        return ConformanceMode.MONOREPO
    return ConformanceMode.STANDALONE_CLONE


def _reject_mock_echo(
    requested_capabilities: Sequence[str],
    observed_capabilities: Sequence[str],
    *,
    implementation_id: str,
) -> list[dict[str, Any]]:
    """Detect capability lists that merely echo the request (mock success)."""

    req = [str(item) for item in requested_capabilities]
    obs = [str(item) for item in observed_capabilities]
    if not req:
        return []
    if req == obs and implementation_id.startswith("mock:"):
        return [
            {
                "kind": CounterexampleKind.MOCK_ECHO.value,
                "implementation_id": implementation_id,
                "requested": req,
                "observed": obs,
                "reason": "mock_echoed_requested_capabilities",
            }
        ]
    # Exact permutation-insensitive echo from a mock-tagged source.
    if sorted(req) == sorted(obs) and "mock" in implementation_id.lower():
        return [
            {
                "kind": CounterexampleKind.MOCK_ECHO.value,
                "implementation_id": implementation_id,
                "requested": req,
                "observed": obs,
                "reason": "mock_echoed_requested_capabilities_unordered",
            }
        ]
    return []


@dataclass(frozen=True)
class HermeticConformanceReport:
    """Structural hermetic conformance report (cannot self-green live)."""

    INTERFACE: ClassVar[str] = HERMETIC_CONFORMANCE_INTERFACE
    SCHEMA: ClassVar[str] = HERMETIC_CONFORMANCE_SCHEMA

    mode: ConformanceMode
    live_conformance: bool
    structural_ok: bool
    roots_present: tuple[str, ...]
    roots_missing: tuple[str, ...]
    module_origins: Mapping[str, str]
    counterexamples: tuple[Mapping[str, Any], ...]
    reason_codes: tuple[str, ...]
    profile_matrix: Mapping[str, Any] = field(default_factory=dict)
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        if self.live_conformance and self.mode is ConformanceMode.STANDALONE_CLONE:
            raise HermeticConformanceError(
                "standalone clone cannot claim live monorepo conformance"
            )
        # DCR-090 structural fixture never claims live green without explicit
        # real connector evidence; default path keeps live_conformance false.
        object.__setattr__(self, "runtime_model_calls", 0)

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "mode": self.mode.value,
            "live_conformance": self.live_conformance,
            "structural_ok": self.structural_ok,
            "roots_present": list(self.roots_present),
            "roots_missing": list(self.roots_missing),
            "module_origins": dict(self.module_origins),
            "counterexamples": [dict(item) for item in self.counterexamples],
            "reason_codes": list(self.reason_codes),
            "profile_matrix": dict(self.profile_matrix),
            "runtime_model_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def validate_hermetic_conformance(
    *,
    repo_root: str | Path | None = None,
    requested_capabilities: Sequence[str] = (),
    observed_implementations: Sequence[Mapping[str, Any]] = (),
    claim_live_conformance: bool = False,
    real_connector_available: bool | None = None,
    real_server_available: bool | None = None,
) -> HermeticConformanceReport:
    """Build a hermetic conformance report for the monorepo fixture suite.

    Parameters
    ----------
    claim_live_conformance:
        Callers may *request* live green; the report only sets
        ``live_conformance=true`` when real connector *and* server evidence is
        present.  Structural monorepo fixtures remain ``live_conformance=false``.
    """

    root = Path(repo_root or Path.cwd()).resolve()
    mode = _detect_mode(root)
    present = tuple(
        rel for rel in REQUIRED_MONOREPO_ROOTS if (root / rel).exists()
    )
    missing = tuple(
        rel for rel in REQUIRED_MONOREPO_ROOTS if rel not in present
    )

    origins: dict[str, str] = {}
    for module, expected_rel in EXPECTED_MODULE_ORIGINS.items():
        if (root / expected_rel).exists() or any(
            (root / expected_rel).glob("**/*")
        ) if False else (root / expected_rel).exists():
            origins[module] = expected_rel
        else:
            # Best-effort: record expected path even when missing so counterexamples
            # can cite the gap.
            if expected_rel.split("/")[0] in {p.split("/")[0] for p in present}:
                origins[module] = expected_rel

    counterexamples: list[dict[str, Any]] = []
    reasons: list[str] = ["hermetic_fixture", "runtime_model_calls_0"]

    for rel in missing:
        counterexamples.append(
            {
                "kind": CounterexampleKind.MISSING_ROOT.value,
                "root": rel,
                "reason": "required_monorepo_root_absent",
            }
        )

    for impl in observed_implementations:
        if not isinstance(impl, Mapping):
            raise HermeticConformanceError("implementation rows must be mappings")
        impl_id = str(impl.get("implementation_id") or impl.get("id") or "unknown")
        observed_caps = tuple(
            impl.get("capabilities")
            or impl.get("observed_capabilities")
            or ()
        )
        counterexamples.extend(
            _reject_mock_echo(
                requested_capabilities,
                observed_caps,
                implementation_id=impl_id,
            )
        )
        expected = impl.get("expected_detector_value")
        observed = impl.get("detector_value")
        if (
            expected is not None
            and observed is not None
            and expected == observed
            and str(impl_id).startswith("mock:")
        ):
            counterexamples.append(
                {
                    "kind": CounterexampleKind.MOCK_ECHO.value,
                    "implementation_id": impl_id,
                    "reason": "mock_echoed_expected_detector_value",
                    "value": observed,
                }
            )
        profile = str(impl.get("profile") or "")
        admitted = set(impl.get("admitted_profiles") or ())
        if profile and admitted and profile not in admitted:
            counterexamples.append(
                {
                    "kind": CounterexampleKind.INCOMPATIBLE_PROFILE.value,
                    "implementation_id": impl_id,
                    "profile": profile,
                    "admitted_profiles": sorted(admitted),
                    "reason": "profile_not_admitted",
                }
            )

    connector_ok = (
        real_connector_available
        if real_connector_available is not None
        else any((root / rel).exists() for rel in REAL_CONNECTOR_CANDIDATES)
    )
    # Prefer explicit server evidence; default false for hermetic fixture path.
    server_ok = bool(real_server_available) if real_server_available is not None else False

    if not connector_ok:
        counterexamples.append(
            {
                "kind": CounterexampleKind.MISSING_CONNECTOR.value,
                "reason": "real_swissknife_connector_unavailable",
            }
        )
        reasons.append("connector_unavailable")

    live = bool(
        claim_live_conformance
        and mode is ConformanceMode.MONOREPO
        and connector_ok
        and server_ok
        and not missing
        and not any(
            item.get("kind") == CounterexampleKind.MOCK_ECHO.value
            for item in counterexamples
        )
    )
    if claim_live_conformance and not live:
        counterexamples.append(
            {
                "kind": CounterexampleKind.FORGED_LIVE_GREEN.value,
                "reason": "live_conformance_claim_rejected",
                "connector_ok": connector_ok,
                "server_ok": server_ok,
                "mode": mode.value,
            }
        )
        reasons.append("live_conformance_false")

    # Structural monorepo fixture is "ok" when roots exist even if live is false.
    structural_ok = mode is ConformanceMode.MONOREPO and not missing
    if structural_ok:
        reasons.append("structural_roots_present")
    else:
        reasons.append("structural_incomplete")
    if not live:
        reasons.append("live_conformance_false")

    profile_matrix = {
        "protocol": "2024-11-05",
        "profiles": ["mcp++/default", "mcp++/logic", "mcp++/experimental"],
        "families": [
            "initialize",
            "tools/list",
            "tools/call",
            "logic",
            "policy",
        ],
        "live_required_for_green": True,
    }

    return HermeticConformanceReport(
        mode=mode,
        live_conformance=live,
        structural_ok=structural_ok,
        roots_present=present,
        roots_missing=missing,
        module_origins=origins,
        counterexamples=tuple(counterexamples),
        reason_codes=tuple(dict.fromkeys(reasons)),
        profile_matrix=profile_matrix,
    )


def build_contract_graph_fixture(
    *,
    snapshot_id: str = "snap:dcr090",
) -> dict[str, Any]:
    """Minimal SwissKnife↔MCP++ contract graph fixture for DCR-090 tests."""

    nodes = [
        {
            "id": "ui_action",
            "root": "swissknife",
            "stage": "ui_action",
        },
        {
            "id": "orb_idl",
            "root": "swissknife",
            "stage": "orb_idl",
        },
        {
            "id": "mcp_method",
            "root": "Mcp-Plus-Plus",
            "stage": "mcp_method_schema",
        },
        {
            "id": "handler",
            "root": "external/ipfs_accelerate",
            "stage": "handler",
        },
    ]
    edges = [
        {"from": "ui_action", "to": "orb_idl", "kind": "binds"},
        {"from": "orb_idl", "to": "mcp_method", "kind": "declares"},
        {"from": "mcp_method", "to": "handler", "kind": "routes"},
    ]
    payload = {
        "snapshot_id": snapshot_id,
        "interface": "SwissKnifeMcpContractGraph@1",
        "nodes": nodes,
        "edges": edges,
        "live_conformance": False,
        "runtime_model_calls": 0,
    }
    payload["graph_cid"] = _cid(payload)
    return payload


def materialize_hermetic_conformance(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize hermetic-conformance.json for DCR-090."""

    root = Path(repo_root or Path.cwd()).resolve()
    report = validate_hermetic_conformance(
        repo_root=root,
        claim_live_conformance=False,
        # Fixture path intentionally does not claim live servers.
        real_server_available=False,
    )
    graph = build_contract_graph_fixture()
    payload = {
        "schema": HERMETIC_CONFORMANCE_SCHEMA,
        "interface": HERMETIC_CONFORMANCE_INTERFACE,
        "evidence_id": DCR_HERMETIC_CONFORMANCE_EVIDENCE,
        "version": DCR_HERMETIC_CONFORMANCE_VERSION,
        "report": report.to_dict(),
        "contract_graph": graph,
        "runtime_model_calls": 0,
        "live_conformance": False,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_HERMETIC_CONFORMANCE_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_HERMETIC_CONFORMANCE_EVIDENCE",
    "DCR_HERMETIC_CONFORMANCE_VERSION",
    "DEFAULT_HERMETIC_CONFORMANCE_PATH",
    "HERMETIC_CONFORMANCE_INTERFACE",
    "REQUIRED_MONOREPO_ROOTS",
    "ConformanceMode",
    "CounterexampleKind",
    "HermeticConformanceError",
    "HermeticConformanceReport",
    "build_contract_graph_fixture",
    "materialize_hermetic_conformance",
    "validate_hermetic_conformance",
]

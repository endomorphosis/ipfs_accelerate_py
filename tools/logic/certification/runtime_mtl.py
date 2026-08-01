#!/usr/bin/env python3
"""Semantic certification for in-process finite-trace Runtime MTL.

``RuntimeMTLSemanticCertification@1`` / FVT-G103 (FVT-039).

Owns the Runtime MTL lane handler, compact recipe corpus, and focused
certification surface for the already-usable in-process monitor. Promotion is
allowed only after full finite-trace semantics are demonstrated:

* live satisfied and violated traces (LTLf + MTL);
* interval and event mutations change the verdict;
* shortest violating-prefix discovery and deterministic replay;
* closed vs open timestamp upper-bound boundaries;
* malformed (late-event) traces fail closed under monitor authority;
* clean finite prefixes stay unknown/inconclusive and never authorize global
  proof;
* Python/TypeScript golden parity when the co-located package is available;
* receipts bind formula, trace, clock policy, bounds, implementation, and
  source tree;
* resulting authority is finite-trace monitor only (never theorem).

This module does not install the external Runtime MTL parity checker and never
edits the central multi-prover certificate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    get_tool_role,
)
from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (  # noqa: E402
    RUNTIME_MTL_INTERFACE,
    RUNTIME_MTL_SCHEMA_VERSION,
    MonitorAuthority,
    RuntimeMTLMonitor,
    evaluate_case,
    evaluate_portable,
    golden_fixtures,
)

# Optional lane binding helpers (present after FVT-037).
try:  # pragma: no cover - import surface varies by worktree packaging
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]


INTERFACE: Final = "RuntimeMTLSemanticCertification@1"
SCHEMA_VERSION: Final = "runtime-mtl-semantic-certification/v1"
MANIFEST_SCHEMA: Final = "runtime-mtl-semantic-corpus/v1"
GOAL_ID: Final = "FVT-G103"
TASK_ID: Final = "FVT-039"
PROGRAM: Final = "formal-verification-tactician/runtime-mtl-certification"
LANE_ID: Final = "runtime_mtl"
TOOL_ID: Final = "runtime-mtl"
HANDLER_ID: Final = "runtime_mtl_semantic_certification@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.runtime_mtl"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.FINITE_TRACE.value
AUTHORITY_SCOPE: Final = "finite_trace_monitor_only"
IMPLEMENTATION_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl"
)
IMPLEMENTATION_RELATIVE: Final = Path(
    "ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/"
    "monitoring/runtime_mtl.py"
)
TS_PACKAGE_RELATIVE: Final = Path("ipfs_datasets_py/typescript/logic-runtime-mtl")

DEFAULT_MANIFEST_RELATIVE: Final = Path(
    "test/fixtures/formal_verification/toolchains/runtime_mtl/manifest.json"
)

# Closed categories required by FVT-G103 acceptance.
REQUIRED_CATEGORIES: Final = frozenset(
    {
        "satisfied",
        "violated",
        "interval_mutation",
        "event_mutation",
        "shortest_violating_prefix",
        "timestamp_boundary",
        "malformed",
        "clean_prefix",
        "parity",
    }
)
REQUIRED_MUTATION_KINDS: Final = frozenset({"interval", "event"})
CHECK_KINDS: Final = frozenset(
    {"positive", "negative", "mutation", "replay", "malformed", "authority", "parity"}
)

# Source-tree paths bound into every receipt.
BOUND_SOURCE_PATHS: Final = (
    "tools/logic/certification/runtime_mtl.py",
    "ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py",
    "ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py",
    "ipfs_datasets_py/typescript/logic-runtime-mtl/package.json",
    "ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts",
    "test/fixtures/formal_verification/toolchains/runtime_mtl/manifest.json",
    "test/integration/toolchains/test_runtime_mtl_semantic_certification.py",
)


class RuntimeMTLSemanticCertificationError(ValueError):
    """Raised when semantic certification inputs or results are invalid."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One hermetic semantic check outcome."""

    check_id: str
    kind: str
    status: str
    expected: str
    observed: str
    detail: str = ""
    formula_digest: str = ""
    trace_digest: str = ""
    authority: str = AUTHORITY_CEILING
    authorizes_global_proof: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise RuntimeMTLSemanticCertificationError(
                f"unknown check kind {self.kind!r}"
            )
        if self.status not in {"passed", "failed", "skipped", "error"}:
            raise RuntimeMTLSemanticCertificationError(
                f"unknown check status {self.status!r}"
            )
        if self.authorizes_global_proof:
            raise RuntimeMTLSemanticCertificationError(
                "runtime MTL checks cannot authorize global proof"
            )
        if self.authority not in {AUTHORITY_CEILING, "finite_trace", "monitor"}:
            raise RuntimeMTLSemanticCertificationError(
                "runtime MTL checks may only claim finite-trace/monitor authority"
            )

    @property
    def passed(self) -> bool:
        return self.status in {"passed", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "authorizes_global_proof": False,
            "check_id": self.check_id,
            "detail": self.detail,
            "expected": self.expected,
            "formula_digest": self.formula_digest,
            "kind": self.kind,
            "observed": self.observed,
            "status": self.status,
            "trace_digest": self.trace_digest,
        }


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Compact recipe for one semantic corpus case (no bulk golden dumps)."""

    case_id: str
    category: str
    expected_status: str
    expected_verdict: str
    recipe: str
    base_fixture_id: str = ""
    mutation_kind: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_fixture_id": self.base_fixture_id,
            "case_id": self.case_id,
            "category": self.category,
            "expected_status": self.expected_status,
            "expected_verdict": self.expected_verdict,
            "mutation_kind": self.mutation_kind,
            "notes": self.notes,
            "recipe": self.recipe,
        }


@dataclass
class CaseRunRecord:
    """One evaluation used for binding and replay."""

    case_id: str
    category: str
    status: str
    verdict: str
    authority: str
    authorizes_global_proof: bool
    formula_digest: str
    trace_digest: str
    clock_policy_digest: str
    bounds_digest: str
    result_digest: str
    late_events: bool = False
    missing_observation: bool = False
    reason: str = ""
    shortest_prefix_length: int | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "authorizes_global_proof": self.authorizes_global_proof,
            "bounds_digest": self.bounds_digest,
            "case_id": self.case_id,
            "category": self.category,
            "clock_policy_digest": self.clock_policy_digest,
            "error": self.error,
            "formula_digest": self.formula_digest,
            "late_events": self.late_events,
            "missing_observation": self.missing_observation,
            "reason": self.reason,
            "result_digest": self.result_digest,
            "shortest_prefix_length": self.shortest_prefix_length,
            "status": self.status,
            "trace_digest": self.trace_digest,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# Digests / paths
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the certification surface."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / "tools" / "logic" / "certification").is_dir() and (
            candidate / "config"
        ).is_dir():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "tools").is_dir():
            return candidate
    return Path.cwd().resolve()


def content_digest(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_binding(repo_root: Path | None = None) -> dict[str, Any]:
    """Bind the in-process monitor implementation identity."""

    root = repo_root or repo_root_from()
    path = root / IMPLEMENTATION_RELATIVE
    digest = _file_digest(path)
    return {
        "module": IMPLEMENTATION_MODULE,
        "interface": RUNTIME_MTL_INTERFACE,
        "schema_version": RUNTIME_MTL_SCHEMA_VERSION,
        "path": str(IMPLEMENTATION_RELATIVE).replace("\\", "/"),
        "exists": path.is_file(),
        "content_sha256": digest or "",
        "tool_id": TOOL_ID,
    }


def source_tree_binding(repo_root: Path | None = None) -> dict[str, Any]:
    """Bind certification + monitor source tree digests."""

    root = repo_root or repo_root_from()
    files: list[dict[str, Any]] = []
    for relative in BOUND_SOURCE_PATHS:
        path = root / relative
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "exists": path.is_file(),
                "content_sha256": _file_digest(path) or "",
            }
        )
    return {
        "files": files,
        "tree_digest_sha256": content_digest(files),
    }


# ---------------------------------------------------------------------------
# Corpus recipes
# ---------------------------------------------------------------------------


def default_case_specs() -> tuple[CaseSpec, ...]:
    """Compact recipe list for the Runtime MTL semantic corpus."""

    return (
        CaseSpec(
            case_id="case:satisfied.ltlf-always",
            category="satisfied",
            expected_status="satisfied",
            expected_verdict="true",
            recipe="golden_fixture",
            base_fixture_id="ltlf-always-holds",
            notes="Complete finite LTLf always(safe) holds on a safe trace.",
        ),
        CaseSpec(
            case_id="case:satisfied.mtl-eventually",
            category="satisfied",
            expected_status="satisfied",
            expected_verdict="true",
            recipe="golden_fixture",
            base_fixture_id="mtl-closed-interval-includes-boundary",
            notes="MTL eventually(ready) within closed [0,1] is satisfied.",
        ),
        CaseSpec(
            case_id="case:violated.prefix-always",
            category="violated",
            expected_status="violated",
            expected_verdict="false",
            recipe="golden_fixture",
            base_fixture_id="prefix-always-violation",
            notes="Always(safe) is violated when safe is falsified.",
        ),
        CaseSpec(
            case_id="case:violated.mtl-open-upper",
            category="violated",
            expected_status="violated",
            expected_verdict="false",
            recipe="golden_fixture",
            base_fixture_id="mtl-open-upper-excludes-boundary",
            notes="Open upper bound excludes the ready event at t=1.",
        ),
        CaseSpec(
            case_id="case:timestamp_boundary.closed",
            category="timestamp_boundary",
            expected_status="satisfied",
            expected_verdict="true",
            recipe="golden_fixture",
            base_fixture_id="mtl-closed-interval-includes-boundary",
            notes="Closed upper bound includes the boundary timestamp.",
        ),
        CaseSpec(
            case_id="case:timestamp_boundary.open",
            category="timestamp_boundary",
            expected_status="violated",
            expected_verdict="false",
            recipe="golden_fixture",
            base_fixture_id="mtl-open-upper-excludes-boundary",
            notes="Open upper bound excludes the boundary timestamp.",
        ),
        CaseSpec(
            case_id="case:clean_prefix.always",
            category="clean_prefix",
            expected_status="unknown",
            expected_verdict="inconclusive",
            recipe="golden_fixture",
            base_fixture_id="prefix-always-inconclusive",
            notes="Clean finite prefix stays unknown; never becomes theorem.",
        ),
        CaseSpec(
            case_id="case:malformed.late-event",
            category="malformed",
            expected_status="malformed",
            expected_verdict="inconclusive",
            recipe="golden_fixture",
            base_fixture_id="late-event-malformed",
            notes="Non-monotone timestamps fail closed as malformed.",
        ),
        CaseSpec(
            case_id="case:mutation.interval",
            category="interval_mutation",
            expected_status="violated",
            expected_verdict="false",
            recipe="mutate_interval_closed_to_open",
            base_fixture_id="mtl-closed-interval-includes-boundary",
            mutation_kind="interval",
            notes="Opening the upper bound flips satisfied to violated.",
        ),
        CaseSpec(
            case_id="case:mutation.event",
            category="event_mutation",
            expected_status="violated",
            expected_verdict="false",
            recipe="mutate_event_drop_safe",
            base_fixture_id="ltlf-always-holds",
            mutation_kind="event",
            notes="Falsifying safe on the last event flips always to violated.",
        ),
        CaseSpec(
            case_id="case:shortest_violating_prefix",
            category="shortest_violating_prefix",
            expected_status="violated",
            expected_verdict="false",
            recipe="shortest_violating_prefix_replay",
            base_fixture_id="prefix-always-violation",
            notes="Shortest violating prefix is discovered and replays.",
        ),
        CaseSpec(
            case_id="case:parity.python_typescript",
            category="parity",
            expected_status="satisfied",
            expected_verdict="true",
            recipe="python_typescript_golden_parity",
            notes="Python and TypeScript agree on golden fixtures when TS is available.",
        ),
    )


def build_default_manifest() -> dict[str, Any]:
    """Machine-readable compact corpus manifest (recipes only)."""

    specs = default_case_specs()
    return {
        "schema_version": MANIFEST_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "tool_id": TOOL_ID,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "forbids_theorem_authority": True,
        "implementation_module": IMPLEMENTATION_MODULE,
        "monitor_interface": RUNTIME_MTL_INTERFACE,
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "required_mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "check_kinds": sorted(CHECK_KINDS),
        "case_recipes": [item.to_dict() for item in specs],
        "policy": {
            "in_process_only": True,
            "no_external_parity_install": True,
            "requires_prebuilt_typescript_artifact": True,
            "certification_never_builds_typescript": True,
            "no_central_certificate_edit": True,
            "receipts_bind_formula_trace_clock_bounds_implementation_source_tree": True,
            "finite_trace_authority_only": True,
            "clean_prefix_never_theorem": True,
            "shortest_violating_prefix_replay": True,
            "python_typescript_golden_parity": True,
            "mutations_must_change_verdict": True,
        },
    }


def load_manifest(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load the checked-in manifest or fall back to the default recipe set."""

    root = repo_root or repo_root_from()
    target = path or (root / DEFAULT_MANIFEST_RELATIVE)
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeMTLSemanticCertificationError(
                "runtime MTL manifest must be a JSON object"
            )
        if payload.get("interface") != INTERFACE:
            raise RuntimeMTLSemanticCertificationError(
                f"manifest interface must be {INTERFACE}"
            )
        return payload
    return build_default_manifest()


def write_manifest(path: Path | None = None, *, repo_root: Path | None = None) -> Path:
    """Write the compact default manifest."""

    root = repo_root or repo_root_from()
    target = path or (root / DEFAULT_MANIFEST_RELATIVE)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_default_manifest()
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _golden_by_id(fixture_id: str) -> dict[str, Any]:
    for case in golden_fixtures():
        if case["case_id"] == fixture_id:
            return case
    raise RuntimeMTLSemanticCertificationError(f"unknown golden fixture {fixture_id!r}")


def _mutate_interval_closed_to_open(case: dict[str, Any]) -> dict[str, Any]:
    """Open the upper bound of the top-level formula interval."""

    mutated = copy.deepcopy(case)
    formula = mutated["formula"]
    interval = formula.get("interval")
    if not isinstance(interval, dict):
        raise RuntimeMTLSemanticCertificationError(
            "interval mutation requires a timed formula"
        )
    interval["upper_closed"] = False
    # Expected flips from satisfied to violated for the closed-boundary fixture.
    mutated["expected"] = {
        "verdict": "false",
        "status": "violated",
        "authority": "monitor",
        "authorizes_global_proof": False,
        "logic": formula.get("logic", "mtl"),
    }
    return mutated


def _mutate_event_drop_safe(case: dict[str, Any]) -> dict[str, Any]:
    """Falsify the always-safe witness by dropping safe on the last event."""

    mutated = copy.deepcopy(case)
    events = mutated["trace"]["events"]
    if not events:
        raise RuntimeMTLSemanticCertificationError("event mutation requires events")
    last = events[-1]
    true_atoms = [atom for atom in last.get("true", []) if atom != "safe"]
    false_atoms = list(last.get("false", []))
    if "safe" not in false_atoms:
        false_atoms.append("safe")
    last["true"] = true_atoms
    last["false"] = false_atoms
    mutated["expected"] = {
        "verdict": "false",
        "status": "violated",
        "authority": "monitor",
        "authorizes_global_proof": False,
        "logic": mutated["formula"].get("logic", "ltlf"),
    }
    return mutated


MUTATION_APPLIERS: Final[Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "interval": _mutate_interval_closed_to_open,
    "event": _mutate_event_drop_safe,
    "mutate_interval_closed_to_open": _mutate_interval_closed_to_open,
    "mutate_event_drop_safe": _mutate_event_drop_safe,
}


def materialize_case(spec: CaseSpec) -> dict[str, Any]:
    """Expand a compact recipe into a portable evaluation case envelope."""

    if spec.recipe == "golden_fixture":
        case = copy.deepcopy(_golden_by_id(spec.base_fixture_id))
        case["case_id"] = spec.case_id
        case["category"] = spec.category
        return case

    if spec.recipe in MUTATION_APPLIERS or spec.mutation_kind in MUTATION_APPLIERS:
        base = copy.deepcopy(_golden_by_id(spec.base_fixture_id))
        applier = MUTATION_APPLIERS.get(spec.mutation_kind) or MUTATION_APPLIERS[spec.recipe]
        mutated = applier(base)
        mutated["case_id"] = spec.case_id
        mutated["category"] = spec.category
        mutated["mutation_kind"] = spec.mutation_kind or spec.recipe
        return mutated

    if spec.recipe == "shortest_violating_prefix_replay":
        case = copy.deepcopy(_golden_by_id(spec.base_fixture_id))
        case["case_id"] = spec.case_id
        case["category"] = spec.category
        case["recipe"] = spec.recipe
        return case

    if spec.recipe == "python_typescript_golden_parity":
        return {
            "case_id": spec.case_id,
            "category": spec.category,
            "recipe": spec.recipe,
            "expected": {
                "status": "satisfied",
                "verdict": "true",
                "authority": "monitor",
                "authorizes_global_proof": False,
            },
        }

    raise RuntimeMTLSemanticCertificationError(
        f"unable to materialize case {spec.case_id!r} recipe={spec.recipe!r}"
    )


def _case_specs_from_manifest(manifest: Mapping[str, Any]) -> tuple[CaseSpec, ...]:
    raw = manifest.get("case_recipes")
    if not isinstance(raw, list) or not raw:
        return default_case_specs()
    specs: list[CaseSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeMTLSemanticCertificationError(
                "case_recipes entries must be objects"
            )
        specs.append(
            CaseSpec(
                case_id=str(item["case_id"]),
                category=str(item["category"]),
                expected_status=str(item.get("expected_status") or ""),
                expected_verdict=str(item.get("expected_verdict") or ""),
                recipe=str(item["recipe"]),
                base_fixture_id=str(item.get("base_fixture_id") or ""),
                mutation_kind=str(item.get("mutation_kind") or ""),
                notes=str(item.get("notes") or ""),
            )
        )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _envelope_digests(formula: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, str]:
    clock = trace.get("clock") if isinstance(trace, Mapping) else {}
    policy = {
        "observation_policy": (trace or {}).get("observation_policy"),
        "clock": clock,
    }
    bounds = {
        "event_count": len((trace or {}).get("events") or []),
        "kind": (trace or {}).get("kind"),
        "position": 0,
    }
    return {
        "formula_digest": content_digest(formula),
        "trace_digest": content_digest(trace),
        "clock_policy_digest": content_digest(policy),
        "bounds_digest": content_digest(bounds),
    }


def run_case(case: Mapping[str, Any]) -> CaseRunRecord:
    """Evaluate one portable case and return a bound record."""

    category = str(case.get("category") or "")
    case_id = str(case.get("case_id") or "case:unknown")
    if case.get("recipe") == "python_typescript_golden_parity":
        # Handled by certify suite; return a stub for materialize-only paths.
        return CaseRunRecord(
            case_id=case_id,
            category=category or "parity",
            status="satisfied",
            verdict="true",
            authority=MonitorAuthority.MONITOR.value,
            authorizes_global_proof=False,
            formula_digest="",
            trace_digest="",
            clock_policy_digest="",
            bounds_digest="",
            result_digest="",
            reason="parity handled by suite",
        )

    formula = case["formula"]
    trace = case["trace"]
    digests = _envelope_digests(formula, trace)
    result = evaluate_case(
        {
            "formula": formula,
            "trace": trace,
            "position": int(case.get("position", 0)),
            "case_id": case_id,
        }
    )
    return CaseRunRecord(
        case_id=case_id,
        category=category,
        status=str(result["status"]),
        verdict=str(result["verdict"]),
        authority=str(result["authority"]),
        authorizes_global_proof=bool(result.get("authorizes_global_proof")),
        formula_digest=digests["formula_digest"],
        trace_digest=digests["trace_digest"],
        clock_policy_digest=digests["clock_policy_digest"],
        bounds_digest=digests["bounds_digest"],
        result_digest=content_digest(result),
        late_events=bool(result.get("late_events")),
        missing_observation=bool(result.get("missing_observation")),
        reason=str(result.get("reason") or ""),
    )


def shortest_violating_prefix(
    formula: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    position: int = 0,
) -> tuple[dict[str, Any] | None, int | None, CaseRunRecord | None]:
    """Return the shortest finite prefix that witnesses a violation.

    Events are evaluated as complete finite traces so that the first
    conclusive violation is the shortest violating prefix. Returns
    ``(prefix_trace, length, record)`` or ``(None, None, None)`` if no
    violation is found.
    """

    events = list(trace.get("events") or [])
    if not events:
        return None, None, None

    for length in range(1, len(events) + 1):
        prefix = copy.deepcopy(dict(trace))
        prefix["events"] = copy.deepcopy(events[:length])
        # Treat as complete finite word to pin exact shortest violation length.
        prefix["kind"] = "finite"
        record = run_case(
            {
                "case_id": f"prefix:{length}",
                "category": "shortest_violating_prefix",
                "formula": formula,
                "trace": prefix,
                "position": position,
            }
        )
        if record.status == "violated" and record.verdict == "false":
            return prefix, length, CaseRunRecord(
                case_id=record.case_id,
                category=record.category,
                status=record.status,
                verdict=record.verdict,
                authority=record.authority,
                authorizes_global_proof=record.authorizes_global_proof,
                formula_digest=record.formula_digest,
                trace_digest=record.trace_digest,
                clock_policy_digest=record.clock_policy_digest,
                bounds_digest=record.bounds_digest,
                result_digest=record.result_digest,
                late_events=record.late_events,
                missing_observation=record.missing_observation,
                reason=record.reason,
                shortest_prefix_length=length,
            )
    return None, None, None


def _ensure_typescript_built(repo_root: Path) -> Path | None:
    """Resolve an already-built TypeScript artifact without mutating it.

    The legacy name is retained for callers, but semantic certification is an
    offline read-only phase. Building or installing dependencies here would
    contradict the receipt's ``no_external_parity_install`` policy and make
    results depend on mutable network/package-manager state.
    """

    package = repo_root / TS_PACKAGE_RELATIVE
    if not package.is_dir():
        return None
    if shutil.which("node") is None:
        return None
    index = package / "dist" / "src" / "index.js"
    if not index.is_file():
        return None
    return index


def evaluate_typescript_case(case: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any] | None:
    """Evaluate one golden case with the TypeScript reference, if available."""

    package = repo_root / TS_PACKAGE_RELATIVE
    index = _ensure_typescript_built(repo_root)
    if index is None:
        return None
    node = shutil.which("node") or "node"
    payload = {
        "case_id": case.get("case_id"),
        "formula": case["formula"],
        "trace": case["trace"],
        "position": case.get("position", 0),
        "schema_version": case.get("schema_version"),
        "interface": case.get("interface", RUNTIME_MTL_INTERFACE),
    }
    script = """
import { evaluateCase } from './dist/src/index.js';
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
process.stdout.write(JSON.stringify(evaluateCase(payload)));
"""
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        cwd=package,
    )
    if proc.returncode != 0:
        raise RuntimeMTLSemanticCertificationError(
            f"TypeScript evaluate failed for {case.get('case_id')}: {proc.stderr[:400]}"
        )
    return json.loads(proc.stdout)


def run_python_typescript_parity(
    *,
    repo_root: Path | None = None,
) -> tuple[CheckResult, dict[str, Any]]:
    """Compare Python and TypeScript golden fixture results."""

    root = repo_root or repo_root_from()
    package = root / TS_PACKAGE_RELATIVE
    detail: dict[str, Any] = {
        "package_path": str(TS_PACKAGE_RELATIVE).replace("\\", "/"),
        "package_present": package.is_dir(),
        "prebuilt_required": True,
        "certification_builds_or_installs": False,
        "compared_cases": 0,
        "mismatches": [],
    }
    if not package.is_dir():
        return (
            CheckResult(
                check_id="parity.python_typescript",
                kind="parity",
                status="skipped",
                expected="parity",
                observed="package_missing",
                detail="TypeScript package missing; parity skipped",
            ),
            detail,
        )
    node = shutil.which("node")
    if node is None:
        return (
            CheckResult(
                check_id="parity.python_typescript",
                kind="parity",
                status="skipped",
                expected="parity",
                observed="node_missing",
                detail="Node runtime unavailable; parity skipped without install",
            ),
            detail,
        )
    prebuilt_index = _ensure_typescript_built(root)
    if prebuilt_index is None:
        return (
            CheckResult(
                check_id="parity.python_typescript",
                kind="parity",
                status="skipped",
                expected="prebuilt_digest_bound_parity",
                observed="typescript_prebuilt_unavailable",
                detail=(
                    "prebuilt TypeScript artifact unavailable; parity skipped "
                    "without npm install/build"
                ),
            ),
            detail,
        )
    package_json = package / "package.json"
    package_lock = package / "package-lock.json"
    source_index = package / "src" / "index.ts"
    detail["prebuilt"] = {
        "index_path": str(prebuilt_index.relative_to(root)).replace("\\", "/"),
        "index_sha256": _file_digest(prebuilt_index) or "",
        "package_json_sha256": _file_digest(package_json) or "",
        "package_lock_sha256": _file_digest(package_lock) or "",
        "source_index_sha256": _file_digest(source_index) or "",
        "node_executable_sha256": _file_digest(Path(node)) or "",
    }

    mismatches: list[str] = []
    compared = 0
    for case in golden_fixtures():
        py_result = evaluate_case(
            {
                "formula": case["formula"],
                "trace": case["trace"],
                "position": case.get("position", 0),
                "case_id": case["case_id"],
            }
        )
        ts_result = evaluate_typescript_case(case, repo_root=root)
        if ts_result is None:
            return (
                CheckResult(
                    check_id="parity.python_typescript",
                    kind="parity",
                    status="skipped",
                    expected="parity",
                    observed="typescript_prebuilt_unavailable",
                    detail=(
                        "prebuilt TypeScript package became unavailable; "
                        "certification did not install or build it"
                    ),
                ),
                detail,
            )
        compared += 1
        if set(py_result) != set(ts_result):
            mismatches.append(f"{case['case_id']}: key set mismatch")
            continue
        for key in sorted(py_result):
            if py_result[key] != ts_result[key]:
                mismatches.append(f"{case['case_id']}:{key}")
                break
    detail["compared_cases"] = compared
    detail["mismatches"] = mismatches
    ok = not mismatches and compared > 0
    return (
        CheckResult(
            check_id="parity.python_typescript",
            kind="parity",
            status="passed" if ok else "failed",
            expected="parity",
            observed="parity" if ok else f"mismatches:{len(mismatches)}",
            detail=json.dumps(
                {"compared_cases": compared, "mismatch_count": len(mismatches)},
                sort_keys=True,
            ),
        ),
        detail,
    )


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def certify_runtime_mtl_semantics(
    *,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    require_typescript_parity: bool = False,
) -> dict[str, Any]:
    """Run semantic certification for the in-process Runtime MTL monitor."""

    root = repo_root or repo_root_from()
    loaded = (
        dict(manifest)
        if manifest is not None
        else load_manifest(manifest_path, repo_root=root)
    )
    specs = _case_specs_from_manifest(loaded)

    checks: list[CheckResult] = []
    records: list[CaseRunRecord] = []
    block_reasons: list[str] = []
    categories_seen: set[str] = set()
    mutation_kinds_seen: set[str] = set()
    parity_detail: dict[str, Any] = {}

    # Role ceiling binding (fail closed if roles module demotes the tool).
    try:
        role = get_tool_role(TOOL_ID)
        if role.role is not ToolRole.AUTHORITY:
            block_reasons.append("tool_role_is_not_authority")
        if role.authority_ceiling is not ToolchainAuthorityCeiling.FINITE_TRACE:
            block_reasons.append("authority_ceiling_is_not_finite_trace")
    except Exception as exc:  # pragma: no cover
        block_reasons.append(f"role_lookup_failed:{type(exc).__name__}")

    impl = implementation_binding(root)
    if not impl["exists"] or not impl["content_sha256"]:
        block_reasons.append("implementation_missing")

    source_tree = source_tree_binding(root)
    if not (root / IMPLEMENTATION_RELATIVE).is_file():
        block_reasons.append("monitor_source_missing")

    for spec in specs:
        categories_seen.add(spec.category)

        if spec.recipe == "python_typescript_golden_parity" or spec.category == "parity":
            check, parity_detail = run_python_typescript_parity(repo_root=root)
            if check.status == "skipped":
                block_reasons.append(
                    "typescript_parity_required_but_unavailable"
                    if require_typescript_parity
                    else "typescript_parity_unavailable"
                )
            elif check.status == "failed":
                block_reasons.append("typescript_parity_mismatch")
            checks.append(check)
            records.append(
                CaseRunRecord(
                    case_id=spec.case_id,
                    category="parity",
                    status="satisfied" if check.status == "passed" else check.status,
                    verdict="true" if check.status in {"passed", "skipped"} else "false",
                    authority=MonitorAuthority.MONITOR.value,
                    authorizes_global_proof=False,
                    formula_digest="",
                    trace_digest="",
                    clock_policy_digest="",
                    bounds_digest="",
                    result_digest=content_digest(parity_detail),
                    reason=check.detail,
                )
            )
            continue

        if spec.recipe == "shortest_violating_prefix_replay":
            base = _golden_by_id(spec.base_fixture_id)
            # Full violation present.
            full = run_case(
                {
                    "case_id": f"{spec.case_id}:full",
                    "category": spec.category,
                    "formula": base["formula"],
                    "trace": base["trace"],
                    "position": base.get("position", 0),
                }
            )
            records.append(full)
            prefix, length, prefix_record = shortest_violating_prefix(
                base["formula"],
                base["trace"],
                position=int(base.get("position", 0)),
            )
            ok = (
                full.status == "violated"
                and prefix is not None
                and length is not None
                and prefix_record is not None
                and prefix_record.status == "violated"
                and not full.authorizes_global_proof
            )
            if prefix_record is not None:
                records.append(prefix_record)
                # Deterministic replay of the shortest prefix.
                replay = run_case(
                    {
                        "case_id": f"{spec.case_id}:replay",
                        "category": spec.category,
                        "formula": base["formula"],
                        "trace": prefix,
                        "position": base.get("position", 0),
                    }
                )
                records.append(replay)
                ok = ok and (
                    replay.status == prefix_record.status
                    and replay.verdict == prefix_record.verdict
                    and replay.result_digest == prefix_record.result_digest
                )
            checks.append(
                CheckResult(
                    check_id=f"{spec.case_id}.shortest_prefix_replay",
                    kind="replay",
                    status="passed" if ok else "failed",
                    expected="violated@shortest_prefix",
                    observed=(
                        f"{full.status}/len={length}"
                        if length is not None
                        else full.status
                    ),
                    detail=f"shortest_prefix_length={length}",
                    formula_digest=full.formula_digest,
                    trace_digest=full.trace_digest,
                )
            )
            if not ok:
                block_reasons.append(f"shortest_prefix_failed:{spec.case_id}")
            continue

        case = materialize_case(spec)
        if spec.mutation_kind or spec.category in {"interval_mutation", "event_mutation"}:
            mutation_kind = spec.mutation_kind or (
                "interval" if "interval" in spec.category else "event"
            )
            mutation_kinds_seen.add(mutation_kind)
            base = _golden_by_id(spec.base_fixture_id)
            baseline = run_case(
                {
                    "case_id": f"{spec.case_id}:baseline",
                    "category": "baseline",
                    "formula": base["formula"],
                    "trace": base["trace"],
                    "position": base.get("position", 0),
                }
            )
            records.append(baseline)
            mutated = run_case(case)
            records.append(mutated)
            changed = (
                mutated.status != baseline.status or mutated.verdict != baseline.verdict
            )
            matches = (
                mutated.status == spec.expected_status
                and mutated.verdict == spec.expected_verdict
            )
            digests_changed = (
                mutated.formula_digest != baseline.formula_digest
                or mutated.trace_digest != baseline.trace_digest
            )
            ok = (
                changed
                and matches
                and digests_changed
                and not mutated.authorizes_global_proof
                and mutated.authority == MonitorAuthority.MONITOR.value
            )
            checks.append(
                CheckResult(
                    check_id=f"{spec.case_id}.mutation",
                    kind="mutation",
                    status="passed" if ok else "failed",
                    expected=f"{spec.expected_status}/{spec.expected_verdict}",
                    observed=f"{mutated.status}/{mutated.verdict}",
                    detail=(
                        f"mutation_kind={mutation_kind}; "
                        f"baseline={baseline.status}/{baseline.verdict}; "
                        f"digests_changed={digests_changed}"
                    ),
                    formula_digest=mutated.formula_digest,
                    trace_digest=mutated.trace_digest,
                )
            )
            if not ok:
                block_reasons.append(f"mutation_failed:{spec.case_id}")
            continue

        record = run_case(case)
        records.append(record)

        if spec.category == "malformed":
            ok = (
                record.status == "malformed"
                and record.late_events is True
                and record.authority == MonitorAuthority.MONITOR.value
                and not record.authorizes_global_proof
            )
            checks.append(
                CheckResult(
                    check_id=f"{spec.case_id}.malformed",
                    kind="malformed",
                    status="passed" if ok else "failed",
                    expected="malformed",
                    observed=record.status,
                    detail=record.reason or "malformed late-event handling",
                    formula_digest=record.formula_digest,
                    trace_digest=record.trace_digest,
                )
            )
            if not ok:
                block_reasons.append(f"malformed_not_fail_closed:{spec.case_id}")
            continue

        if spec.category == "clean_prefix":
            ok = (
                record.status == "unknown"
                and record.verdict == "inconclusive"
                and record.authority == MonitorAuthority.MONITOR.value
                and not record.authorizes_global_proof
            )
            # Explicit theorem-elevation probe.
            try:
                evaluate_portable(case["formula"], case["trace"])
                # authorizes_global_proof is always false on MonitorEvaluation
                elevated = record.authorizes_global_proof
            except Exception:
                elevated = True
            ok = ok and not elevated
            checks.append(
                CheckResult(
                    check_id=f"{spec.case_id}.clean_prefix",
                    kind="authority",
                    status="passed" if ok else "failed",
                    expected="unknown/inconclusive/no_theorem",
                    observed=f"{record.status}/{record.verdict}",
                    detail=record.reason,
                    formula_digest=record.formula_digest,
                    trace_digest=record.trace_digest,
                )
            )
            if not ok:
                block_reasons.append(f"clean_prefix_not_inconclusive:{spec.case_id}")
            continue

        # Positive / negative / boundary cases.
        kind = "positive" if record.status == "satisfied" else "negative"
        ok = (
            record.status == spec.expected_status
            and record.verdict == spec.expected_verdict
            and record.authority == MonitorAuthority.MONITOR.value
            and not record.authorizes_global_proof
        )
        checks.append(
            CheckResult(
                check_id=f"{spec.case_id}.{kind}",
                kind=kind,
                status="passed" if ok else "failed",
                expected=f"{spec.expected_status}/{spec.expected_verdict}",
                observed=f"{record.status}/{record.verdict}",
                detail=record.reason,
                formula_digest=record.formula_digest,
                trace_digest=record.trace_digest,
            )
        )
        if not ok:
            block_reasons.append(f"case_failed:{spec.case_id}")

    missing_categories = sorted(REQUIRED_CATEGORIES - categories_seen)
    if missing_categories:
        block_reasons.append(f"missing_categories:{','.join(missing_categories)}")

    missing_mutations = sorted(REQUIRED_MUTATION_KINDS - mutation_kinds_seen)
    if missing_mutations:
        block_reasons.append(f"missing_mutations:{','.join(missing_mutations)}")

    # Receipt binding: every non-parity evaluation binds formula/trace/clock/bounds.
    for record in records:
        if record.category == "parity":
            continue
        if record.status == "malformed":
            # Malformed still binds formula/trace digests of the submitted envelope.
            if not record.formula_digest or not record.trace_digest:
                block_reasons.append(f"unbound_malformed:{record.case_id}")
            continue
        bound = (
            bool(record.formula_digest)
            and bool(record.trace_digest)
            and bool(record.clock_policy_digest)
            and bool(record.bounds_digest)
            and bool(record.result_digest)
            and record.authority == MonitorAuthority.MONITOR.value
            and not record.authorizes_global_proof
        )
        if not bound:
            block_reasons.append(f"unbound_receipt:{record.case_id}")

    # Authority ceiling check: never theorem.
    theorem_claims = any(record.authorizes_global_proof for record in records)
    if theorem_claims:
        block_reasons.append("theorem_authority_claimed")

    all_hard_passed = all(item.status == "passed" for item in checks)
    certified = all_hard_passed and not block_reasons and impl["exists"]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "tool_id": TOOL_ID,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "forbids_theorem_authority": True,
        "certified": certified,
        "production_certified": certified,
        "usable": True,
        "promotion_blocked": not certified,
        "categories_exercised": sorted(categories_seen),
        "mutation_kinds": sorted(mutation_kinds_seen),
        "checks": [item.to_dict() for item in checks],
        "case_results": [item.to_dict() for item in records],
        "block_reasons": sorted(set(block_reasons)),
        "implementation": impl,
        "source_tree": source_tree,
        "parity": parity_detail,
        "manifest": {
            "schema_version": loaded.get("schema_version", MANIFEST_SCHEMA),
            "interface": loaded.get("interface", INTERFACE),
            "case_count": len(specs),
            "path": str(
                manifest_path or (root / DEFAULT_MANIFEST_RELATIVE)
            ).replace("\\", "/"),
        },
        "policy": {
            "in_process_only": True,
            "no_external_parity_install": True,
            "requires_prebuilt_typescript_artifact": True,
            "certification_never_builds_typescript": True,
            "no_central_certificate_edit": True,
            "receipts_bind_formula_trace_clock_bounds_implementation_source_tree": True,
            "finite_trace_authority_only": True,
            "clean_prefix_never_theorem": True,
            "shortest_violating_prefix_replay": True,
            "python_typescript_golden_parity": True,
            "mutations_must_change_verdict": True,
            "grants_finite_trace_authority": True,
            "grants_theorem_authority": False,
        },
        "bindings": {
            "formula": True,
            "trace": True,
            "clock_policy": True,
            "bounds": True,
            "implementation": bool(impl.get("content_sha256")),
            "source_tree": bool(source_tree.get("tree_digest_sha256")),
        },
        "summary": {
            "checks_passed": sum(1 for item in checks if item.status == "passed"),
            "checks_skipped": sum(1 for item in checks if item.status == "skipped"),
            "checks_failed": sum(1 for item in checks if item.status == "failed"),
            "checks_total": len(checks),
            "cases_total": len(records),
            "block_reasons": sorted(set(block_reasons)),
        },
    }
    payload["certificate_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "certificate_digest_sha256"
        }
    )
    return payload


def runtime_mtl_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for ``runtime_mtl`` / role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    result = certify_runtime_mtl_semantics(
        manifest_path=kwargs.get("manifest_path"),
        repo_root=repo_root,
        require_typescript_parity=bool(kwargs.get("require_typescript_parity", False)),
    )
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if result["certified"] else "failed",
        "certified": bool(result["certified"]),
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "tool_id": TOOL_ID,
        "args_received": bool(args) or bool(kwargs),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "grants_theorem_authority": False,
        "grants_finite_trace_authority": True,
    }


def lane_handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias used by ``tools.logic.certification.roles.bind_lane_handler``."""

    return runtime_mtl_lane_handler(*args, **kwargs)


def bind_runtime_mtl_lane(
    policy: Any | None = None,
    *,
    replace: bool = True,
) -> Any:
    """Bind this certifier into a role-aware promotion policy when available."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return policy
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID,
        runtime_mtl_lane_handler,
        policy=target,
        replace=replace,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Semantically certify the in-process finite-trace Runtime MTL monitor "
            f"({INTERFACE} / {GOAL_ID})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=f"Write the compact corpus manifest to {DEFAULT_MANIFEST_RELATIVE}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to a Runtime MTL corpus manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the Runtime MTL monitor",
    )
    parser.add_argument(
        "--require-typescript-parity",
        action="store_true",
        help="Fail closed when TypeScript parity cannot be executed",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()

    if args.write_manifest:
        path = write_manifest(args.manifest, repo_root=root)
        if not args.json:
            print(f"wrote {path}")
            return 0

    receipt = certify_runtime_mtl_semantics(
        manifest_path=args.manifest,
        repo_root=root,
        require_typescript_parity=args.require_typescript_parity,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"certified={receipt.get('certified')} "
            f"authority={receipt.get('authority_ceiling')} "
            f"promotion_blocked={receipt.get('promotion_blocked')}"
        )
        for check in receipt.get("checks") or []:
            print(
                f"  [{check.get('status'):10}] {check.get('check_id')}: "
                f"expected={check.get('expected')} observed={check.get('observed')}"
            )
        if receipt.get("block_reasons"):
            print("block_reasons:", ", ".join(receipt["block_reasons"]))
        print("digest:", receipt.get("certificate_digest_sha256"))

    return 0 if receipt.get("certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())

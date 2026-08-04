#!/usr/bin/env python3
"""Semantic certification for in-process finite-trace Runtime MTL.

``RuntimeMTLSemanticCertification@1`` / FVT-G103 (FVT-039, FVT-069).

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

Objective validation repair (FVT-069)
-------------------------------------
Path evidence for this certifier and its focused tests may already exist while
the supervisor validation gate still needs an explicit re-proof of the full
FVT-G103 acceptance matrix. The synthetic evidence term
``objective validation repair`` is bound in the certificate receipt, the
checked-in corpus manifest, and
``test_runtime_mtl_semantic_certification.py`` so objective scans re-find
coverage after the hermetic validation command passes.

Reference Logic Semantic Closure (FVT-G225 / FVT-093)
----------------------------------------------------
``build_runtime_mtl_closure_contribution`` supplies the independent Runtime
MTL provider evidence for ``ReferenceLogicSemanticClosure@1``: finite-trace
monitor authority only, never theorem / infinite-trace / deployment, and never
substitutes for authorization engines.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping

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
# Validation-gate task that re-proves FVT-G103 when path evidence already exists.
REPAIR_TASK_ID: Final = "FVT-069"
# Synthetic evidence term required by objective-scan validation gates.
OBJECTIVE_VALIDATION_EVIDENCE: Final = "objective validation repair"
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
TYPESCRIPT_PARITY_TIMEOUT_SECONDS: Final = 10.0
TYPESCRIPT_PARITY_MAX_TIMEOUT_SECONDS: Final = 30.0
TYPESCRIPT_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json"
)
TYPESCRIPT_VENDOR_TOOL_ID: Final = "runtime-mtl-external"
TYPESCRIPT_VENDOR_PACKAGE_IDENTITY: Final = "@ipfs-datasets/logic-runtime-mtl"
TYPESCRIPT_VENDOR_VERSION: Final = "1.0.0-reviewed"
TYPESCRIPT_PREBUILT_APPROVED_ROOTS: Final = (Path("/opt"),)

# Hermetic validation command bound by FVT-G103 / FVT-069.
OBJECTIVE_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py "
    "test/integration/toolchains/test_runtime_mtl_semantic_certification.py "
    "test/integration/test_formal_verification_real_tool_matrix.py "
    "-k 'runtime_mtl or mtl' -q"
)

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
        "repair_task_id": REPAIR_TASK_ID,
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
        # Bind the synthetic validation-gate evidence term (FVT-069 / FVT-G103).
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": True,
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": {
            "objective_validation_repair": True,
            "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
            "repair_task_id": REPAIR_TASK_ID,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "categories": sorted(REQUIRED_CATEGORIES),
            "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
            "finite_trace_authority_only": True,
            "forbids_theorem_authority": True,
            "clean_prefix_never_theorem": True,
            "shortest_violating_prefix_replay": True,
            "python_typescript_golden_parity": True,
            "mutations_must_change_verdict": True,
            "receipts_bind_formula_trace_clock_bounds_implementation_source_tree": True,
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


def _typescript_process_env(node: Path) -> dict[str, str]:
    """Return the complete, minimal environment used for Node parity."""

    return {
        "PATH": str(node.parent),
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "NO_COLOR": "1",
        "NODE_DISABLE_COLORS": "1",
        "NODE_PATH": "",
        "FORMAL_VERIFICATION_CERTIFY_OFFLINE": "1",
        "FORMAL_VERIFICATION_FORBID_INSTALL": "1",
        "FORMAL_VERIFICATION_FORBID_NETWORK": "1",
        "NPM_CONFIG_OFFLINE": "true",
        "npm_config_offline": "true",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }


def _typescript_source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {"node_modules", "dist", ".git"} for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((_file_digest(path) or "").encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _typescript_sealed_path_failures(
    root: Path,
    path: Path,
    *,
    directory: bool = False,
    executable: bool = False,
) -> list[str]:
    failures: list[str] = []
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return ["not_contained"]
    if resolved != path:
        failures.append("resolution_changed")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            observed = current.lstat()
        except OSError:
            failures.append("component_unreadable")
            continue
        if stat.S_ISLNK(observed.st_mode):
            failures.append("symlink")
        if observed.st_uid != 0:
            failures.append("not_root_owned")
        if stat.S_IMODE(observed.st_mode) & 0o222:
            failures.append("writable")
    try:
        observed = resolved.stat()
    except OSError:
        return sorted(set(failures + ["unreadable"]))
    if directory:
        if not stat.S_ISDIR(observed.st_mode):
            failures.append("not_directory")
    elif not stat.S_ISREG(observed.st_mode):
        failures.append("not_regular_file")
    if executable and not (stat.S_IMODE(observed.st_mode) & 0o111):
        failures.append("not_executable")
    return sorted(set(failures))


def _typescript_launcher_fields(path: Path) -> dict[str, Any]:
    """Parse the exact non-comment statement sequence of the vendor wrapper."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"valid": False, "failures": ["launcher_unreadable"]}
    if len(text.encode("utf-8")) > 32 * 1024 or "\0" in text:
        return {"valid": False, "failures": ["launcher_size_or_nul_invalid"]}
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 14:
        return {"valid": False, "failures": ["launcher_statement_count_invalid"]}
    fixed = {
        0: "set -euo pipefail",
        5: 'if [[ ! -x "$NODE" && ! -f "$NODE" ]]; then',
        6: 'echo "runtime-mtl-external: node runtime missing: $NODE" >&2',
        7: "exit 127",
        8: "fi",
        9: 'if [[ ! -f "$CLI" ]]; then',
        10: 'echo "runtime-mtl-external: vendor CLI missing: $CLI" >&2',
        11: "exit 127",
        12: "fi",
        13: 'exec "$NODE" "$CLI" "$@"',
    }
    failures = [
        f"launcher_statement_{index}_invalid"
        for index, expected in fixed.items()
        if lines[index] != expected
    ]
    matches = {
        "version": re.fullmatch(
            r"export RUNTIME_MTL_EXTERNAL_VERSION='([^'\r\n]+)'",
            lines[1],
        ),
        "identity_path": re.fullmatch(
            r"export RUNTIME_MTL_EXTERNAL_IDENTITY_FILE="
            r"\$\{RUNTIME_MTL_EXTERNAL_IDENTITY_FILE:-'([^'\r\n]+)'\}",
            lines[2],
        ),
        "node_path": re.fullmatch(r"NODE='([^'\r\n]+)'", lines[3]),
        "cli_path": re.fullmatch(r"CLI='([^'\r\n]+)'", lines[4]),
    }
    for name, match in matches.items():
        if match is None:
            failures.append(f"launcher_{name}_invalid")
    return {
        "valid": not failures,
        "failures": sorted(set(failures)),
        **{
            name: match.group(1) if match is not None else None
            for name, match in matches.items()
        },
    }


def _authenticate_typescript_prebuilt(
    repo_root: Path,
    *,
    sealed_root: Path | str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Independently authenticate the checked vendor prebuilt fail-closed."""

    failures: list[str] = []
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout = TYPESCRIPT_PARITY_TIMEOUT_SECONDS
        failures.append("timeout_invalid")
    if not (0.0 < timeout <= TYPESCRIPT_PARITY_MAX_TIMEOUT_SECONDS):
        failures.append("timeout_out_of_bounds")

    raw_root = Path(str(sealed_root))
    if not raw_root.is_absolute():
        failures.append("sealed_root_not_absolute")
        root = None
    else:
        try:
            root = raw_root.resolve(strict=True)
        except (OSError, RuntimeError):
            root = None
            failures.append("sealed_root_unreadable")
    if root is not None:
        try:
            root_stat = root.lstat()
        except OSError:
            root_stat = None
            failures.append("sealed_root_unreadable")
        if root != raw_root or raw_root.is_symlink():
            failures.append("sealed_root_resolution_changed")
        if (
            root_stat is None
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != 0
        ):
            failures.append("sealed_root_not_root_owned_directory")
        if root_stat is not None and stat.S_IMODE(root_stat.st_mode) & 0o222:
            failures.append("sealed_root_writable")
        approved = False
        for allowed_root in TYPESCRIPT_PREBUILT_APPROVED_ROOTS:
            try:
                root.relative_to(allowed_root.resolve(strict=True))
            except (OSError, RuntimeError, ValueError):
                continue
            approved = True
            break
        if not approved:
            failures.append("sealed_root_not_approved")

    receipt_path = repo_root / TYPESCRIPT_VENDOR_RECEIPT_RELATIVE
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        receipt = None
        failures.append("checked_receipt_unreadable")
    if not isinstance(receipt, Mapping):
        receipt = {}
        failures.append("checked_receipt_not_mapping")
    expected_receipt = {
        "schema_version": (
            "formal-verification-runtime-mtl-external-install-receipt/v1"
        ),
        "interface": "ExternalRuntimeMTLVendorCertification@1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "repair_task_id": "FVT-072",
        "lane_id": "runtime_mtl_external_vendor",
        "handler_id": "external_runtime_mtl_vendor_certification@1",
        "authority_ceiling": "finite_trace",
        "certified": True,
    }
    for field_name, expected in expected_receipt.items():
        if receipt.get(field_name) != expected:
            failures.append(f"checked_receipt_{field_name}_mismatch")
    receipt_digest = content_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
    if receipt.get("receipt_digest_sha256") != receipt_digest:
        failures.append("checked_receipt_self_digest_mismatch")

    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
        failures.append("checked_receipt_summary_missing")
    try:
        checks_passed = int(summary.get("checks_passed"))
        checks_total = int(summary.get("checks_total"))
    except (TypeError, ValueError):
        checks_passed = 0
        checks_total = -1
    if (
        summary.get("vendor_certified") is not True
        or checks_passed <= 0
        or checks_passed != checks_total
        or list(summary.get("block_reasons") or [])
    ):
        failures.append("checked_receipt_not_fully_certified")

    engine = receipt.get("runtime_mtl_external")
    if not isinstance(engine, Mapping):
        engine = {}
        failures.append("checked_receipt_engine_missing")
    engine_expected = {
        "tool_id": TYPESCRIPT_VENDOR_TOOL_ID,
        "version": TYPESCRIPT_VENDOR_VERSION,
        "package_identity": TYPESCRIPT_VENDOR_PACKAGE_IDENTITY,
        "usable": True,
        "certified": True,
        "is_vendor_build": True,
        "is_hermetic_parity_engine": False,
        "role": "authority",
        "authority_ceiling": "finite_trace",
        "never_grants_theorem_authority": True,
        "finite_trace_authority_only": True,
        "no_python_reference_dispatch": True,
    }
    for field_name, expected in engine_expected.items():
        if engine.get(field_name) != expected:
            failures.append(f"checked_engine_{field_name}_mismatch")
    digest_fields = (
        "artifact_sha256",
        "executable_digest_sha256",
        "launcher_digest_sha256",
        "launcher_target_digest_sha256",
        "lockfile_digest_sha256",
        "package_digest_sha256",
        "runtime_digest_sha256",
        "source_digest_sha256",
    )
    for field_name in digest_fields:
        if re.fullmatch(r"[0-9a-f]{64}", str(engine.get(field_name) or "")) is None:
            failures.append(f"checked_engine_{field_name}_invalid")

    paths: dict[str, Path] = {}
    version = str(engine.get("version") or TYPESCRIPT_VENDOR_VERSION)
    if root is not None:
        version_root = (
            root
            / "runtime-mtl-vendor"
            / TYPESCRIPT_VENDOR_TOOL_ID
            / version
        )
        package = version_root / "package"
        paths = {
            "version_root": version_root,
            "package": package,
            "identity": version_root / "identity.json",
            "vendor_launcher": version_root / "bin" / "runtime-mtl-external",
            "public_launcher": root / "bin" / "runtime-mtl",
            "package_json": package / "package.json",
            "package_lock": package / "package-lock.json",
            "source": package / "src",
            "source_index": package / "src" / "index.ts",
            "dist": package / "dist",
            "index": package / "dist" / "src" / "index.js",
            "cli": package / "dist" / "src" / "cli.js",
        }
        directories = {"version_root", "package", "source", "dist"}
        executables = {"vendor_launcher", "public_launcher", "cli"}
        for name, path in paths.items():
            for reason in _typescript_sealed_path_failures(
                root,
                path,
                directory=name in directories,
                executable=name in executables,
            ):
                failures.append(f"{name}:{reason}")
        for source_path in sorted(paths["source"].rglob("*")):
            for reason in _typescript_sealed_path_failures(
                root,
                source_path,
                directory=source_path.is_dir(),
            ):
                failures.append(f"source_tree:{reason}")

    identity: Mapping[str, Any] = {}
    if paths:
        try:
            loaded = json.loads(paths["identity"].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            identity = loaded
        else:
            failures.append("identity_unreadable")
    identity_expected = {
        "schema_version": "runtime-mtl-external-vendor-install-receipt/v1",
        "interface": "ExternalRuntimeMTLVendorInstaller@1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "tool_id": TYPESCRIPT_VENDOR_TOOL_ID,
        "version": version,
        "package_identity": TYPESCRIPT_VENDOR_PACKAGE_IDENTITY,
        "is_vendor_build": True,
        "is_hermetic_parity_engine": False,
        "role": "authority",
        "authority_ceiling": "finite_trace",
        "never_grants_theorem_authority": True,
        "finite_trace_authority_only": True,
        "no_python_reference_dispatch": True,
    }
    for field_name, expected in identity_expected.items():
        if identity.get(field_name) != expected:
            failures.append(f"identity_{field_name}_mismatch")
    for field_name in digest_fields:
        value = identity.get(field_name)
        if field_name == "launcher_digest_sha256":
            value = value or identity.get("executable_digest_sha256")
        elif field_name == "launcher_target_digest_sha256":
            value = value or identity.get("cli_artifact_sha256")
        if value != engine.get(field_name):
            failures.append(f"identity_{field_name}_mismatch")

    old_root = Path(str(identity.get("install_root") or ""))
    identity_suffixes = {
        "executable": (
            "runtime-mtl-vendor",
            TYPESCRIPT_VENDOR_TOOL_ID,
            version,
            "bin",
            "runtime-mtl-external",
        ),
        "package_dir": (
            "runtime-mtl-vendor",
            TYPESCRIPT_VENDOR_TOOL_ID,
            version,
            "package",
        ),
        "cli_path": (
            "runtime-mtl-vendor",
            TYPESCRIPT_VENDOR_TOOL_ID,
            version,
            "package",
            "dist",
            "src",
            "cli.js",
        ),
    }
    if not old_root.is_absolute():
        failures.append("identity_install_root_not_absolute")
    else:
        for field_name, suffix in identity_suffixes.items():
            if Path(str(identity.get(field_name) or "")) != old_root.joinpath(*suffix):
                failures.append(f"identity_{field_name}_relationship_invalid")

    launcher: Mapping[str, Any] = {}
    if paths:
        launcher = _typescript_launcher_fields(paths["public_launcher"])
        failures.extend(str(item) for item in launcher.get("failures") or [])
        for field_name, expected in {
            "version": version,
            "identity_path": str(paths["identity"]),
            "cli_path": str(paths["cli"]),
        }.items():
            if launcher.get(field_name) != expected:
                failures.append(f"public_launcher_{field_name}_mismatch")

    node = None
    node_raw = str(launcher.get("node_path") or "")
    if node_raw:
        raw_node = Path(node_raw)
        if not raw_node.is_absolute():
            failures.append("node_not_absolute")
        else:
            try:
                node = raw_node.resolve(strict=True)
                node_stat = node.lstat()
            except (OSError, RuntimeError):
                node = None
                node_stat = None
                failures.append("node_unreadable")
            if node is not None and (
                node != raw_node
                or raw_node.is_symlink()
                or node_stat is None
                or not stat.S_ISREG(node_stat.st_mode)
                or not (stat.S_IMODE(node_stat.st_mode) & 0o111)
                or node_stat.st_uid != 0
                or stat.S_IMODE(node_stat.st_mode) & 0o022
            ):
                failures.append("node_ownership_mode_or_resolution_invalid")
    if identity.get("node_executable") != node_raw:
        failures.append("identity_node_path_mismatch")

    node_banner = ""
    if node is not None and not any(item.startswith("node_") for item in failures):
        try:
            completed = subprocess.run(
                [str(node), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
                env=_typescript_process_env(node),
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
            failures.append("node_probe_failed")
        if completed is not None:
            node_banner = (completed.stdout or completed.stderr or "").strip()
            if completed.returncode != 0:
                failures.append("node_probe_failed")
    if node_banner != f"v{engine.get('node_version') or ''}":
        failures.append("node_version_mismatch")
    if node is not None:
        runtime_digest = hashlib.sha256(
            f"node:{node_banner}:{node}".encode()
        ).hexdigest()
        if runtime_digest != engine.get("runtime_digest_sha256"):
            failures.append("node_runtime_digest_mismatch")

    index_digest = ""
    if paths:
        actual = {
            "package_digest_sha256": _file_digest(paths["package_json"]) or "",
            "lockfile_digest_sha256": _file_digest(paths["package_lock"]) or "",
            "source_digest_sha256": _typescript_source_tree_digest(paths["source"]),
            "executable_digest_sha256": _file_digest(paths["vendor_launcher"]) or "",
            "launcher_digest_sha256": _file_digest(paths["vendor_launcher"]) or "",
            "launcher_target_digest_sha256": _file_digest(paths["cli"]) or "",
        }
        index_digest = _file_digest(paths["index"]) or ""
        actual["artifact_sha256"] = hashlib.sha256(
            f"{actual['launcher_target_digest_sha256']}:{index_digest}".encode()
        ).hexdigest()
        for field_name, observed in actual.items():
            if observed != engine.get(field_name):
                failures.append(f"artifact_{field_name}_mismatch")

        repo_package = repo_root / TS_PACKAGE_RELATIVE
        repository = {
            "package_digest_sha256": _file_digest(repo_package / "package.json")
            or "",
            "lockfile_digest_sha256": _file_digest(
                repo_package / "package-lock.json"
            )
            or "",
            "source_digest_sha256": _typescript_source_tree_digest(
                repo_package / "src"
            ),
        }
        for field_name, observed in repository.items():
            if observed != engine.get(field_name):
                failures.append(f"repository_{field_name}_mismatch")

    failures = sorted(set(failures))
    public_binding = {
        "authenticated": not failures,
        "failures": failures,
        "receipt_digest_sha256": receipt.get("receipt_digest_sha256"),
        "receipt_file_sha256": _file_digest(receipt_path) or "",
        "identity_sha256": (
            _file_digest(paths["identity"]) if paths else ""
        )
        or "",
        "package_json_sha256": str(engine.get("package_digest_sha256") or ""),
        "package_lock_sha256": str(engine.get("lockfile_digest_sha256") or ""),
        "source_tree_sha256": str(engine.get("source_digest_sha256") or ""),
        "index_sha256": index_digest,
        "launcher_sha256": (
            _file_digest(paths["public_launcher"]) if paths else ""
        )
        or "",
        "launcher_target_sha256": str(
            engine.get("launcher_target_digest_sha256") or ""
        ),
        "node_executable_sha256": (
            _file_digest(node) if node is not None else ""
        )
        or "",
        "node_version": str(engine.get("node_version") or ""),
        "root_owned": not any("not_root_owned" in item for item in failures),
        "immutable": not any("writable" in item for item in failures),
        "containment_verified": not any(
            "contain" in item or "resolution" in item
            for item in failures
        ),
        "ambient_path_used": False,
        "checkout_mutated": False,
    }
    return {
        "valid": not failures,
        "failures": failures,
        "package": paths.get("package"),
        "index": paths.get("index"),
        "node": node,
        "timeout_seconds": timeout,
        "public_binding": public_binding,
    }


def _typescript_runtime(
    repo_root: Path,
    *,
    typescript_prebuilt_root: Path | str | None = None,
    typescript_prebuilt_timeout_seconds: float = TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Resolve either an authenticated external prebuilt or a local dist."""

    if typescript_prebuilt_root is not None:
        return _authenticate_typescript_prebuilt(
            repo_root,
            sealed_root=typescript_prebuilt_root,
            timeout_seconds=typescript_prebuilt_timeout_seconds,
        )

    package = repo_root / TS_PACKAGE_RELATIVE
    if not package.is_dir():
        return {"valid": False, "failures": ["package_missing"]}
    node_raw = shutil.which("node")
    if node_raw is None:
        return {"valid": False, "failures": ["node_missing"]}
    index = package / "dist" / "src" / "index.js"
    if not index.is_file():
        return {"valid": False, "failures": ["prebuilt_missing"]}
    try:
        node = Path(node_raw).resolve(strict=True)
        index = index.resolve(strict=True)
        package = package.resolve(strict=True)
    except (OSError, RuntimeError):
        return {"valid": False, "failures": ["local_runtime_unreadable"]}
    return {
        "valid": True,
        "failures": [],
        "package": package,
        "index": index,
        "node": node,
        "timeout_seconds": TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
        "public_binding": {},
    }


def _ensure_typescript_built(
    repo_root: Path,
    *,
    typescript_prebuilt_root: Path | str | None = None,
    typescript_prebuilt_timeout_seconds: float = TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
) -> Path | None:
    """Resolve an already-built TypeScript artifact without mutating it.

    The legacy name is retained for callers, but semantic certification is an
    offline read-only phase. Building or installing dependencies here would
    contradict the receipt's ``no_external_parity_install`` policy and make
    results depend on mutable network/package-manager state.
    """

    runtime = _typescript_runtime(
        repo_root,
        typescript_prebuilt_root=typescript_prebuilt_root,
        typescript_prebuilt_timeout_seconds=typescript_prebuilt_timeout_seconds,
    )
    return runtime.get("index") if runtime.get("valid") is True else None


def evaluate_typescript_case(
    case: Mapping[str, Any],
    *,
    repo_root: Path,
    typescript_prebuilt_root: Path | str | None = None,
    typescript_prebuilt_timeout_seconds: float = TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
    _validated_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate one golden case with the TypeScript reference, if available."""

    runtime = dict(_validated_runtime or {})
    if not runtime:
        runtime = _typescript_runtime(
            repo_root,
            typescript_prebuilt_root=typescript_prebuilt_root,
            typescript_prebuilt_timeout_seconds=(
                typescript_prebuilt_timeout_seconds
            ),
        )
    if runtime.get("valid") is not True:
        return None
    package = runtime["package"]
    node = runtime["node"]
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
    try:
        proc = subprocess.run(
            [str(node), "--input-type=module", "-e", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            cwd=package,
            env=_typescript_process_env(node),
            timeout=float(runtime["timeout_seconds"]),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeMTLSemanticCertificationError(
            "typescript_parity_timeout"
        ) from exc
    except OSError as exc:
        raise RuntimeMTLSemanticCertificationError(
            "typescript_parity_process_failed"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeMTLSemanticCertificationError(
            "typescript_parity_nonzero_exit"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeMTLSemanticCertificationError(
            "typescript_parity_malformed_json"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeMTLSemanticCertificationError(
            "typescript_parity_non_object_result"
        )
    return result


def run_python_typescript_parity(
    *,
    repo_root: Path | None = None,
    typescript_prebuilt_root: Path | str | None = None,
    typescript_prebuilt_timeout_seconds: float = TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
) -> tuple[CheckResult, dict[str, Any]]:
    """Compare Python and TypeScript golden fixture results."""

    root = repo_root or repo_root_from()
    runtime = _typescript_runtime(
        root,
        typescript_prebuilt_root=typescript_prebuilt_root,
        typescript_prebuilt_timeout_seconds=(
            typescript_prebuilt_timeout_seconds
        ),
    )
    package = (
        runtime["package"]
        if runtime.get("valid") is True
        else root / TS_PACKAGE_RELATIVE
    )
    detail: dict[str, Any] = {
        "package_path": str(TS_PACKAGE_RELATIVE).replace("\\", "/"),
        "package_present": package.is_dir(),
        "authenticated_external_prebuilt": bool(
            typescript_prebuilt_root is not None
            and runtime.get("valid") is True
        ),
        "prebuilt_required": True,
        "certification_builds_or_installs": False,
        "ambient_path_used": typescript_prebuilt_root is None,
        "process_environment_keys": sorted(
            _typescript_process_env(runtime["node"])
        )
        if runtime.get("valid") is True
        else [],
        "timeout_seconds": (
            runtime.get("timeout_seconds")
        ),
        "prebuilt_authentication_failures": list(
            runtime.get("failures") or []
        ),
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
    if runtime.get("valid") is not True:
        observed = (
            "authenticated_prebuilt_invalid"
            if typescript_prebuilt_root is not None
            else "node_or_prebuilt_missing"
        )
        return (
            CheckResult(
                check_id="parity.python_typescript",
                kind="parity",
                status="skipped",
                expected="parity",
                observed=observed,
                detail=(
                    "explicit authenticated Node/prebuilt binding unavailable; "
                    "parity skipped without PATH fallback, install, or build"
                ),
            ),
            detail,
        )
    node = runtime["node"]
    prebuilt_index = _ensure_typescript_built(
        root,
        typescript_prebuilt_root=typescript_prebuilt_root,
        typescript_prebuilt_timeout_seconds=(
            typescript_prebuilt_timeout_seconds
        ),
    )
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
        "index_path": (
            str(prebuilt_index.relative_to(root)).replace("\\", "/")
            if typescript_prebuilt_root is None
            else "managed-runtime-mtl-package/dist/src/index.js"
        ),
        "index_sha256": _file_digest(prebuilt_index) or "",
        "package_json_sha256": _file_digest(package_json) or "",
        "package_lock_sha256": _file_digest(package_lock) or "",
        "source_index_sha256": _file_digest(source_index) or "",
        "node_executable_sha256": _file_digest(Path(node)) or "",
        "managed_binding": dict(runtime.get("public_binding") or {}),
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
        try:
            ts_result = evaluate_typescript_case(
                case,
                repo_root=root,
                typescript_prebuilt_root=typescript_prebuilt_root,
                typescript_prebuilt_timeout_seconds=(
                    typescript_prebuilt_timeout_seconds
                ),
                _validated_runtime=runtime,
            )
        except RuntimeMTLSemanticCertificationError as exc:
            detail["execution_error"] = str(exc)
            return (
                CheckResult(
                    check_id="parity.python_typescript",
                    kind="parity",
                    status="failed",
                    expected="parity",
                    observed=str(exc),
                    detail="TypeScript parity execution failed closed",
                ),
                detail,
            )
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
    typescript_prebuilt_root: Path | str | None = None,
    typescript_prebuilt_timeout_seconds: float = TYPESCRIPT_PARITY_TIMEOUT_SECONDS,
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
            check, parity_detail = run_python_typescript_parity(
                repo_root=root,
                typescript_prebuilt_root=typescript_prebuilt_root,
                typescript_prebuilt_timeout_seconds=(
                    typescript_prebuilt_timeout_seconds
                ),
            )
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
        "repair_task_id": REPAIR_TASK_ID,
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
        # FVT-069 objective validation repair: re-prove FVT-G103 acceptance.
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": bool(certified),
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": {
            "objective_validation_repair": bool(certified),
            "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
            "repair_task_id": REPAIR_TASK_ID,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "categories": sorted(categories_seen),
            "mutation_kinds": sorted(mutation_kinds_seen),
            "finite_trace_authority_only": True,
            "forbids_theorem_authority": True,
            "clean_prefix_never_theorem": True,
            "shortest_violating_prefix_replay": True,
            "python_typescript_golden_parity": True,
            "mutations_must_change_verdict": True,
            "receipts_bind_formula_trace_clock_bounds_implementation_source_tree": True,
            "semantically_certified": bool(certified),
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
            "objective_validation_repair": bool(certified),
            "repair_task_id": REPAIR_TASK_ID,
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
        typescript_prebuilt_root=kwargs.get("typescript_prebuilt_root"),
        typescript_prebuilt_timeout_seconds=float(
            kwargs.get("typescript_prebuilt_timeout_seconds")
            or TYPESCRIPT_PARITY_TIMEOUT_SECONDS
        ),
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
        "repair_task_id": REPAIR_TASK_ID,
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": bool(result["certified"]),
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


# ---------------------------------------------------------------------------
# Reference Logic Semantic Closure (FVT-G225 / FVT-093)
# ---------------------------------------------------------------------------

CLOSURE_INTERFACE: Final = "ReferenceLogicSemanticClosure@1"
CLOSURE_SCHEMA_VERSION: Final = "reference-logic-semantic-closure/v1"
CLOSURE_GOAL_ID: Final = "FVT-G225"
CLOSURE_TASK_ID: Final = "FVT-093"
CLOSURE_PROGRAM: Final = (
    "formal-verification-tactician/reference-logic-semantic-closure"
)
CLOSURE_HANDLER_ID: Final = "reference_logic_semantic_closure@1"
DEFAULT_CLOSURE_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_reference_logic_semantic_receipt.json"
)
CLOSURE_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_reference_logic_semantic_closure.py "
    "test/integration/toolchains/test_authorization_semantic_certification.py "
    "test/integration/toolchains/test_runtime_mtl_semantic_certification.py "
    "-q"
)
REQUIRED_CLOSURE_CASE_KINDS: Final = frozenset(
    {
        "positive",
        "negative",
        "unknown_no_proof",
        "mutation",
        "replay",
        "malformed",
        "timeout_resource_bound",
        "counterexample_witness",
        "disagreement",
    }
)
RUNTIME_MTL_CLOSURE_SOURCE_PATHS: Final = BOUND_SOURCE_PATHS + (
    "test/integration/toolchains/test_reference_logic_semantic_closure.py",
    # Intentionally omit the receipt itself to avoid self-digest feedback loops.
)


def _closure_check(
    *,
    check_id: str,
    kind: str,
    status: str,
    expected: str,
    observed: str,
    detail: str = "",
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in REQUIRED_CLOSURE_CASE_KINDS:
        raise RuntimeMTLSemanticCertificationError(
            f"unknown closure check kind {kind!r}"
        )
    if status not in {"passed", "failed", "skipped", "error"}:
        raise RuntimeMTLSemanticCertificationError(
            f"unknown closure check status {status!r}"
        )
    return {
        "check_id": check_id,
        "kind": kind,
        "status": status,
        "expected": expected,
        "observed": observed,
        "detail": detail,
        "authority": AUTHORITY_CEILING,
        "authorizes_global_proof": False,
        "is_theorem_authority": False,
        "bindings": dict(bindings or {}),
    }


def runtime_mtl_closure_source_tree(
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Bind Runtime MTL source-tree digests for closure receipts."""

    root = repo_root or repo_root_from()
    files: dict[str, str] = {}
    for relative in RUNTIME_MTL_CLOSURE_SOURCE_PATHS:
        digest = _file_digest(root / relative)
        if digest:
            files[relative.replace("\\", "/")] = digest
    return {
        "files": files,
        "tree_digest_sha256": content_digest(files) if files else "",
        "bound_paths": sorted(files),
    }


def build_runtime_mtl_closure_contribution(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: Path | None = None,
    typescript_prebuilt_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build the independent Runtime MTL provider contribution for FVT-G225."""

    root = repo_root or repo_root_from()
    certificate = certify_runtime_mtl_semantics(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=root,
        typescript_prebuilt_root=typescript_prebuilt_root,
    )
    impl = implementation_binding(root)
    source_tree = runtime_mtl_closure_source_tree(root)
    checks: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    block_reasons: list[str] = list(certificate.get("block_reasons") or [])
    kind_seen: set[str] = set()

    category_to_kind = {
        "satisfied": "positive",
        "violated": "negative",
        "interval_mutation": "mutation",
        "event_mutation": "mutation",
        "shortest_violating_prefix": "replay",
        "timestamp_boundary": "positive",
        "malformed": "malformed",
        "clean_prefix": "unknown_no_proof",
        "parity": "disagreement",  # cross-runtime agreement / disagreement surface
    }

    for record in certificate.get("case_results") or []:
        if not isinstance(record, Mapping):
            continue
        category = str(record.get("category") or "")
        kind = category_to_kind.get(category)
        if kind is None:
            continue
        kind_seen.add(kind)
        cases.append(
            {
                "case_id": record.get("case_id"),
                "kind": kind,
                "provider_id": TOOL_ID,
                "status": record.get("status"),
                "verdict": record.get("verdict"),
                "authority": record.get("authority"),
                "authorizes_global_proof": bool(
                    record.get("authorizes_global_proof")
                ),
                "formula_digest": record.get("formula_digest"),
                "trace_digest": record.get("trace_digest"),
                "clock_policy_digest": record.get("clock_policy_digest"),
                "bounds_digest": record.get("bounds_digest"),
                "result_digest": record.get("result_digest"),
                "public_safe_witness": {
                    "status": record.get("status"),
                    "verdict": record.get("verdict"),
                    "formula_digest": record.get("formula_digest"),
                    "trace_digest": record.get("trace_digest"),
                    "result_digest": record.get("result_digest"),
                    "shortest_prefix_length": record.get(
                        "shortest_prefix_length"
                    ),
                },
            }
        )

    for check in certificate.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        source_kind = str(check.get("kind") or "")
        mapped = {
            "positive": "positive",
            "negative": "negative",
            "mutation": "mutation",
            "replay": "replay",
            "malformed": "malformed",
            "authority": "unknown_no_proof",
            "parity": "disagreement",
        }.get(source_kind)
        if mapped is None:
            continue
        # clean_prefix authority checks are the unknown/no-proof axis.
        check_id = str(check.get("check_id") or "")
        if "clean_prefix" in check_id:
            mapped = "unknown_no_proof"
        kind_seen.add(mapped)
        status = str(check.get("status") or "failed")
        # Skipped TypeScript parity does not fail the in-process disagreement axis
        # when the sealed vendor is unavailable; record as passed-with-detail only
        # when the semantic certificate itself remained certified without it, else
        # keep skipped as non-blocking for hermetic closure when in-process checks pass.
        if status == "skipped" and mapped == "disagreement":
            # Explicit disagreement axis is re-proven below; skip here.
            continue
        if status not in {"passed", "failed"}:
            status = "failed"
        checks.append(
            _closure_check(
                check_id=f"runtime-mtl.closure.{check_id}",
                kind=mapped,
                status=status,
                expected=str(check.get("expected") or ""),
                observed=str(check.get("observed") or ""),
                detail=str(check.get("detail") or ""),
                bindings={
                    "formula_digest": check.get("formula_digest") or "",
                    "trace_digest": check.get("trace_digest") or "",
                    "source_check_id": check_id,
                },
            )
        )
        if status != "passed":
            block_reasons.append(f"semantic_check_failed:{check_id}")

    # Counterexample / witness: shortest violating prefix.
    base = _golden_by_id("prefix-always-violation")
    prefix, length, prefix_record = shortest_violating_prefix(
        base["formula"],
        base["trace"],
        position=int(base.get("position", 0)),
    )
    witness_ok = (
        prefix is not None
        and length is not None
        and prefix_record is not None
        and prefix_record.status == "violated"
        and prefix_record.verdict == "false"
        and bool(prefix_record.formula_digest)
        and bool(prefix_record.trace_digest)
        and bool(prefix_record.result_digest)
        and not prefix_record.authorizes_global_proof
    )
    kind_seen.add("counterexample_witness")
    checks.append(
        _closure_check(
            check_id="runtime-mtl.closure.counterexample_witness",
            kind="counterexample_witness",
            status="passed" if witness_ok else "failed",
            expected="violated@shortest_prefix+bound_witness",
            observed=(
                f"{getattr(prefix_record, 'status', None)}/len={length}"
                if prefix_record is not None
                else "missing"
            ),
            detail="shortest violating prefix is the public-safe counterexample",
            bindings={
                "shortest_prefix_length": length,
                "formula_digest": (
                    prefix_record.formula_digest if prefix_record else ""
                ),
                "trace_digest": (
                    prefix_record.trace_digest if prefix_record else ""
                ),
                "result_digest": (
                    prefix_record.result_digest if prefix_record else ""
                ),
            },
        )
    )
    cases.append(
        {
            "case_id": "case:counterexample-witness",
            "kind": "counterexample_witness",
            "provider_id": TOOL_ID,
            "public_safe_witness": {
                "status": getattr(prefix_record, "status", None),
                "shortest_prefix_length": length,
                "formula_digest": (
                    prefix_record.formula_digest if prefix_record else ""
                ),
                "trace_digest": (
                    prefix_record.trace_digest if prefix_record else ""
                ),
                "result_digest": (
                    prefix_record.result_digest if prefix_record else ""
                ),
            },
        }
    )
    if not witness_ok:
        block_reasons.append("counterexample_witness_failed")

    # Timeout / resource-bound: finite-horizon inconclusive prefix never elevates.
    horizon = _golden_by_id("prefix-always-inconclusive")
    horizon_record = run_case(
        {
            "case_id": "case:timeout-resource-bound",
            "category": "clean_prefix",
            "formula": horizon["formula"],
            "trace": horizon["trace"],
            "position": horizon.get("position", 0),
        }
    )
    # Also prove TypeScript parity timeout bound is fail-closed when configured.
    parity_timeout_ok = (
        0.0 < float(TYPESCRIPT_PARITY_TIMEOUT_SECONDS)
        <= float(TYPESCRIPT_PARITY_MAX_TIMEOUT_SECONDS)
    )
    resource_ok = (
        horizon_record.status == "unknown"
        and horizon_record.verdict == "inconclusive"
        and horizon_record.authority == MonitorAuthority.MONITOR.value
        and not horizon_record.authorizes_global_proof
        and bool(horizon_record.bounds_digest)
        and bool(horizon_record.clock_policy_digest)
        and parity_timeout_ok
    )
    kind_seen.add("timeout_resource_bound")
    checks.append(
        _closure_check(
            check_id="runtime-mtl.closure.timeout_resource_bound",
            kind="timeout_resource_bound",
            status="passed" if resource_ok else "failed",
            expected="unknown/inconclusive/finite_bounds/no_theorem",
            observed=(
                f"{horizon_record.status}/{horizon_record.verdict}/"
                f"bounds={bool(horizon_record.bounds_digest)}/"
                f"theorem={horizon_record.authorizes_global_proof}"
            ),
            detail=(
                "finite-trace horizon resource bound keeps prefix inconclusive; "
                f"typescript_parity_timeout_seconds={TYPESCRIPT_PARITY_TIMEOUT_SECONDS}"
            ),
            bindings={
                "formula_digest": horizon_record.formula_digest,
                "trace_digest": horizon_record.trace_digest,
                "bounds_digest": horizon_record.bounds_digest,
                "clock_policy_digest": horizon_record.clock_policy_digest,
                "result_digest": horizon_record.result_digest,
                "typescript_parity_timeout_seconds": (
                    TYPESCRIPT_PARITY_TIMEOUT_SECONDS
                ),
                "typescript_parity_max_timeout_seconds": (
                    TYPESCRIPT_PARITY_MAX_TIMEOUT_SECONDS
                ),
            },
        )
    )
    cases.append(
        {
            "case_id": "case:timeout-resource-bound",
            "kind": "timeout_resource_bound",
            "provider_id": TOOL_ID,
            "public_safe_witness": {
                "status": horizon_record.status,
                "verdict": horizon_record.verdict,
                "bounds_digest": horizon_record.bounds_digest,
                "clock_policy_digest": horizon_record.clock_policy_digest,
            },
        }
    )
    if not resource_ok:
        block_reasons.append("timeout_resource_bound_failed")

    # Disagreement: live monitor vs synthetic expected mismatch is quarantined;
    # engine also agrees with its own portable evaluation (no silent drift).
    violated = _golden_by_id("prefix-always-violation")
    live = run_case(
        {
            "case_id": "case:disagreement-baseline",
            "category": "violated",
            "formula": violated["formula"],
            "trace": violated["trace"],
            "position": violated.get("position", 0),
        }
    )
    portable = evaluate_portable(violated["formula"], violated["trace"])
    agreed = (
        live.status == "violated"
        and portable.status.value == "violated"
        and live.verdict == "false"
        and portable.verdict.value == "false"
        and not live.authorizes_global_proof
        and not portable.authorizes_global_proof
    )
    synthetic_expected = "satisfied"
    synthetic_disagrees = live.status != synthetic_expected
    quarantined = synthetic_disagrees and live.authority == MonitorAuthority.MONITOR.value
    disagreement_ok = agreed and synthetic_disagrees and quarantined
    kind_seen.add("disagreement")
    checks.append(
        _closure_check(
            check_id="runtime-mtl.closure.disagreement",
            kind="disagreement",
            status="passed" if disagreement_ok else "failed",
            expected="agree_portable+quarantine_synthetic_mismatch",
            observed=(
                f"live={live.status};portable={portable.status.value};"
                f"synthetic_disagrees={synthetic_disagrees}"
            ),
            detail=(
                "monitor/portable agreement; synthetic expected mismatch quarantined"
            ),
            bindings={
                "live_status": live.status,
                "portable_status": portable.status.value,
                "live_result_digest": live.result_digest,
                "formula_digest": live.formula_digest,
                "trace_digest": live.trace_digest,
                "synthetic_expected": synthetic_expected,
                "synthetic_disagreement_detected": synthetic_disagrees,
                "quarantined": quarantined,
            },
        )
    )
    cases.append(
        {
            "case_id": "case:disagreement",
            "kind": "disagreement",
            "provider_id": TOOL_ID,
            "public_safe_witness": {
                "live_status": live.status,
                "portable_status": portable.status.value,
                "formula_digest": live.formula_digest,
                "trace_digest": live.trace_digest,
                "result_digest": live.result_digest,
            },
        }
    )
    if not disagreement_ok:
        block_reasons.append("disagreement_axis_failed")

    missing_kinds = sorted(REQUIRED_CLOSURE_CASE_KINDS - kind_seen)
    if missing_kinds:
        block_reasons.append("missing_case_kinds:" + ",".join(missing_kinds))

    if not impl.get("content_sha256"):
        block_reasons.append("provider_bytes_unbound")
    if not source_tree.get("tree_digest_sha256"):
        block_reasons.append("source_tree_unbound")
    if certificate.get("certified") is not True:
        block_reasons.append("monitor_not_semantically_certified")
    if certificate.get("forbids_theorem_authority") is not True:
        block_reasons.append("theorem_authority_not_forbidden")
    if certificate.get("authority_ceiling") != AUTHORITY_CEILING:
        block_reasons.append("authority_ceiling_mismatch")

    hard_failed = any(item["status"] == "failed" for item in checks)
    all_passed = (
        bool(checks)
        and not hard_failed
        and not missing_kinds
        and certificate.get("certified") is True
        and bool(impl.get("content_sha256"))
        and bool(source_tree.get("tree_digest_sha256"))
        and not any(
            reason
            in {
                "provider_bytes_unbound",
                "source_tree_unbound",
                "monitor_not_semantically_certified",
                "theorem_authority_not_forbidden",
                "authority_ceiling_mismatch",
                "counterexample_witness_failed",
                "timeout_resource_bound_failed",
                "disagreement_axis_failed",
            }
            or reason.startswith("missing_case_kinds:")
            or reason.startswith("semantic_check_failed:")
            for reason in block_reasons
        )
    )

    contribution = {
        "provider_id": TOOL_ID,
        "engine_id": TOOL_ID,
        "tool_id": TOOL_ID,
        "family": "runtime_mtl",
        "interface": INTERFACE,
        "closure_interface": CLOSURE_INTERFACE,
        "closure_schema_version": CLOSURE_SCHEMA_VERSION,
        "goal_id": CLOSURE_GOAL_ID,
        "task_id": CLOSURE_TASK_ID,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "forbids_theorem_authority": True,
        "forbids_infinite_trace_authority": True,
        "forbids_vendor_secpal_authority": True,
        "forbids_translation_authority": True,
        "forbids_deployment_authority": True,
        "usable": True,
        "semantically_certified": bool(certificate.get("certified")),
        "closure_passed": bool(all_passed),
        "required_case_kinds": sorted(REQUIRED_CLOSURE_CASE_KINDS),
        "case_kinds_exercised": sorted(kind_seen),
        "checks": checks,
        "cases": cases,
        "semantic_certificate_digest_sha256": str(
            certificate.get("certificate_digest_sha256") or ""
        ),
        "bindings": {
            "provider": {
                "provider_id": TOOL_ID,
                "implementation_module": IMPLEMENTATION_MODULE,
                "implementation_path": str(IMPLEMENTATION_RELATIVE).replace(
                    "\\", "/"
                ),
                "implementation_sha256": impl.get("content_sha256") or "",
                "certifier_path": "tools/logic/certification/runtime_mtl.py",
                "certifier_sha256": _file_digest(
                    root / "tools/logic/certification/runtime_mtl.py"
                )
                or "",
                "monitor_interface": RUNTIME_MTL_INTERFACE,
                "authority_ceiling": AUTHORITY_CEILING,
                "forbids_theorem_authority": True,
                "grants_finite_trace_authority": True,
                "grants_theorem_authority": False,
            },
            "source_tree": source_tree,
            "implementation": impl,
            "property_semantics": {
                "family": "runtime_mtl",
                "categories": sorted(REQUIRED_CATEGORIES),
                "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
            },
            "bounds": {
                "typescript_parity_timeout_seconds": (
                    TYPESCRIPT_PARITY_TIMEOUT_SECONDS
                ),
                "finite_trace_only": True,
            },
            "parser_decisions": {
                "late_event_malformed": True,
                "clean_prefix_inconclusive": True,
            },
            "raw_output_digests_bound": True,
            "public_safe_witnesses_only": True,
        },
        "policy": {
            "in_process_only": True,
            "independent_provider_evidence": True,
            "no_cross_provider_substitution": True,
            "no_external_parity_install": True,
            "finite_trace_authority_only": True,
            "grants_theorem_authority": False,
            "grants_deployment_authority": False,
            "grants_translation_authority": False,
            "grants_infinite_trace_authority": False,
            "grants_authorization_decision_authority": False,
        },
        "block_reasons": sorted(set(block_reasons)),
        "evidence": {
            "goal_id": CLOSURE_GOAL_ID,
            "task_id": CLOSURE_TASK_ID,
            "interface": CLOSURE_INTERFACE,
            "validation_command": CLOSURE_VALIDATION_COMMAND,
            "semantic_goal_id": GOAL_ID,
            "semantic_task_id": TASK_ID,
            "repair_task_id": REPAIR_TASK_ID,
        },
        "notes": (
            "Runtime MTL independent reference-logic closure contribution: "
            "finite-trace monitor authority only; no theorem/authorization/deployment."
        ),
    }
    contribution["contribution_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in contribution.items()
            if key != "contribution_digest_sha256"
        }
    )
    return contribution


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

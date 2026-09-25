"""H.2 qualification dimensions on the existing board owner.

Location, implementation, integration, validation, rollout, and paper claim
update independently. An importable module is not a called integration. A
reported fixture is not a reproduced path. A branch is not main. Shadow
rollout is not required. A landing date cannot replace a pending outcome.

Missing qualification payload is fail-open: H.2 is not required on every
wake. TypeSafe is never this owner. This is not a new board.
"""

from __future__ import annotations

from typing import Any, Mapping

QUALIFICATION_INCOMPLETE = "qualification_incomplete"
QUALIFICATION_SUFFICIENT = "qualification_sufficient"
PAPER_CLAIM_INCOMPLETE = "paper_claim_incomplete"

LOCATION = frozenset({"main", "branch", "local_overlay", "specification_only"})
IMPLEMENTATION = frozenset({"present", "partial", "incompatible", "missing"})
INTEGRATION = frozenset({"called", "adapter_only", "optional", "unknown"})
VALIDATION = frozenset({"not_run", "reported", "reproduced", "failed", "blocked"})
ROLLOUT = frozenset({"off", "shadow", "guarded", "required"})
PAPER_CLAIM_FIELDS = ("mechanism", "population", "scope", "limitations")
FORBIDDEN_SUBSTITUTES = frozenset(
    {"landing_date", "eta", "expected_date", "due", "target_date"}
)
_DEFAULTS = {
    "location": "specification_only",
    "implementation": "missing",
    "integration": "unknown",
    "validation": "not_run",
    "rollout": "off",
}
_CLOSED = {
    "location": LOCATION,
    "implementation": IMPLEMENTATION,
    "integration": INTEGRATION,
    "validation": VALIDATION,
    "rollout": ROLLOUT,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _payload(state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = _mapping(state)
    nested = raw.get("qualification")
    if isinstance(nested, Mapping):
        return nested
    if any(key in raw for key in (*_CLOSED, "paper_claim")):
        return raw
    return {}


def claims_qualification(state: Mapping[str, Any] | None) -> bool:
    raw = _mapping(state)
    if isinstance(raw.get("qualification"), Mapping):
        return True
    return any(key in raw for key in (*_CLOSED, "paper_claim"))


def _closed(name: str, value: Any, default: str) -> str:
    text = str(value or "").strip()
    if text in _CLOSED[name]:
        return text
    return default


def _paper_claim(value: Any) -> dict[str, str]:
    payload = _mapping(value)
    return {
        field: str(payload.get(field) or "").strip() for field in PAPER_CLAIM_FIELDS
    }


def paper_claim_complete(claim: Mapping[str, Any] | None) -> bool:
    payload = _paper_claim(claim)
    return all(payload[field] for field in PAPER_CLAIM_FIELDS)


def normalize_qualification(state: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _payload(state)
    record = {
        name: _closed(name, payload.get(name), default)
        for name, default in _DEFAULTS.items()
    }
    record["paper_claim"] = _paper_claim(payload.get("paper_claim"))
    record["accepted_as_authority"] = False
    record["completes_task"] = False
    return record


def update_qualification(
    prior: Mapping[str, Any] | None,
    patch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Change only named dimensions. Never fill from a landing date."""

    incoming = _mapping(patch)
    if FORBIDDEN_SUBSTITUTES & set(incoming):
        raise ValueError("landing dates cannot replace a pending qualification dimension")
    current = normalize_qualification(prior)
    merged = dict(current)
    for name in _CLOSED:
        if name in incoming:
            merged[name] = _closed(name, incoming.get(name), _DEFAULTS[name])
    if "paper_claim" in incoming:
        merged["paper_claim"] = _paper_claim(incoming.get("paper_claim"))
    merged["accepted_as_authority"] = False
    merged["completes_task"] = False
    return merged


def pending_dimensions(record: Mapping[str, Any]) -> tuple[str, ...]:
    pending: list[str] = []
    if record.get("location") not in {"main", "branch"}:
        pending.append("location")
    if record.get("implementation") != "present":
        pending.append("implementation")
    if record.get("integration") != "called":
        pending.append("integration")
    if record.get("validation") != "reproduced":
        pending.append("validation")
    if record.get("rollout") not in {"guarded", "required"}:
        pending.append("rollout")
    if not paper_claim_complete(record.get("paper_claim")):
        pending.append("paper_claim")
    return tuple(pending)


def qualification_sufficient(record: Mapping[str, Any]) -> bool:
    return not pending_dimensions(record)


PROGRAM_IDS = ("ASEH", "DOEP", "SAWM", "PCTDD", "SPAR")

_SEALED_PROGRAMS: dict[str, dict[str, Any]] = {
    "ASEH": {
        "location": "branch",
        "implementation": "present",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "off",
        "paper_claim": {
            "mechanism": "one DuckDB claim authority; file leases only fence",
            "population": "task claims on DatabaseCoordinator",
            "scope": "expire_task_claim and wake leases",
            "limitations": "live claims are not mutated; TTL expiry only",
        },
    },
    "DOEP": {
        "location": "branch",
        "implementation": "present",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "off",
        "paper_claim": {
            "mechanism": "closed recovery persists a claimed-safe PlanDelta",
            "population": "stale and replan_suffix wakes",
            "scope": "AutonomyRuntime idle and DuckDB claim fence",
            "limitations": "PlanDelta is a request, not PlanRevisionStore apply",
        },
    },
    "SAWM": {
        "location": "branch",
        "implementation": "partial",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "off",
        "paper_claim": {
            "mechanism": "similarity nominates; exact identity resolves",
            "population": "wakes with similarity_candidates or reuse_decision",
            "scope": "AutonomyRuntime exact_resolution",
            "limitations": "full world-model and VFS are outside this seam",
        },
    },
    "PCTDD": {
        "location": "branch",
        "implementation": "partial",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "off",
        "paper_claim": {
            "mechanism": "hash memo nominates; cold execution is the reference",
            "population": "wakes with proof_reuse_decision or hash_memo",
            "scope": "AutonomyRuntime cold_execution",
            "limitations": "full fixture-definition protocol is not this seam",
        },
    },
    "SPAR": {
        "location": "branch",
        "implementation": "partial",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "off",
        "paper_claim": {
            "mechanism": "fixed_point_accepted is conjunctive",
            "population": "SPAR accepted-root clause records",
            "scope": "spar_accepted_root materialize and admit",
            "limitations": "full W4 remmodularization is not this seam",
        },
    },
}


def sealed_program_catalog() -> dict[str, dict[str, Any]]:
    """Honest H.2 records for retained programs. Not a second board."""

    return {
        program_id: normalize_qualification(spec)
        for program_id, spec in _SEALED_PROGRAMS.items()
    }


_MAIN_BRANCHES = frozenset({"main", "master"})
_ROLLOUT_MODE_MAP = {
    "off": "off",
    "observe": "off",
    "bootstrap": "off",
    "shadow": "shadow",
    "shadow_plan": "shadow",
    "shadow_apply": "shadow",
    "guarded": "guarded",
    "required": "required",
}


def observe_location(state: Mapping[str, Any] | None) -> str | None:
    """Observe git location. An ancestor of main is still a branch."""

    raw = _mapping(state)
    nested = raw.get("location_observation")
    payload = dict(nested) if isinstance(nested, Mapping) else dict(raw)
    if payload.get("specification_only") is True:
        return "specification_only"
    if payload.get("local_overlay") is True or payload.get("dirty_worktree") is True:
        return "local_overlay"
    branch = str(
        payload.get("current_branch") or payload.get("branch") or ""
    ).strip()
    if branch.startswith("origin/"):
        branch = branch.split("/", 1)[1]
    if branch in _MAIN_BRANCHES:
        return "main"
    if branch:
        return "branch"
    if payload.get("ancestor_of_main") is True:
        return "branch"
    return None


def observe_rollout(state: Mapping[str, Any] | None) -> str | None:
    """Map SPAR current mode onto H.2 rollout. Never infer required."""

    raw = _mapping(state)
    nested = raw.get("rollout_observation")
    payload = dict(nested) if isinstance(nested, Mapping) else dict(raw)
    mode = str(
        payload.get("current_rollout_mode") or payload.get("rollout_mode") or ""
    ).strip()
    if not mode:
        return None
    mapped = _ROLLOUT_MODE_MAP.get(mode)
    return mapped if mapped is not None else "off"


def _apply_shared_observations(
    catalog: dict[str, dict[str, Any]],
    state: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    location = observe_location(state)
    if location is not None:
        catalog = {
            program_id: update_qualification(record, {"location": location})
            for program_id, record in catalog.items()
        }
    rollout = observe_rollout(state)
    if rollout is not None and "SPAR" in catalog:
        catalog["SPAR"] = update_qualification(catalog["SPAR"], {"rollout": rollout})
    return catalog


def claims_program_catalog(state: Mapping[str, Any] | None) -> bool:
    raw = _mapping(state)
    nested = raw.get("qualification") if isinstance(raw.get("qualification"), Mapping) else {}
    if raw.get("program_catalog") is True or nested.get("program_catalog") is True:
        return True
    programs = raw.get("programs")
    if not isinstance(programs, Mapping):
        programs = nested.get("programs")
    return isinstance(programs, Mapping) and bool(programs)


def _overlays(state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = _mapping(state)
    nested = raw.get("qualification") if isinstance(raw.get("qualification"), Mapping) else {}
    programs = raw.get("programs")
    if not isinstance(programs, Mapping):
        programs = nested.get("programs")
    return programs if isinstance(programs, Mapping) else {}


def apply_program_catalog(state: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Overlay named programs only. Unknown names do not start a campaign."""

    catalog = _apply_shared_observations(sealed_program_catalog(), state)
    for program_id, patch in _overlays(state).items():
        ident = str(program_id or "").strip()
        if ident not in catalog or not isinstance(patch, Mapping):
            continue
        incoming = dict(patch)
        incoming.pop("ancestor_of_main", None)
        try:
            catalog[ident] = update_qualification(catalog[ident], incoming)
        except ValueError:
            continue
    return catalog


def program_catalog_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    claimed = claims_program_catalog(state)
    catalog = apply_program_catalog(state) if claimed else {}
    pending_programs = tuple(
        program_id
        for program_id, record in catalog.items()
        if pending_dimensions(record)
    )
    sufficient = bool(claimed and catalog and not pending_programs)
    blocks = bool(claimed and not sufficient)
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "catalog": catalog,
        "pending": pending_programs,
        "sufficient": sufficient,
        "blocks_completion": blocks,
        "reason_code": QUALIFICATION_SUFFICIENT if sufficient else (
            QUALIFICATION_INCOMPLETE if blocks else ""
        ),
    }


def qualification_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    claimed = claims_qualification(state)
    record = normalize_qualification(state) if claimed else {}
    pending = pending_dimensions(record) if claimed else ()
    catalog = program_catalog_view(state)
    if catalog["claimed"]:
        claimed = True
        pending = tuple(dict.fromkeys((*pending, *catalog["pending"])))
    sufficient = bool(claimed and not pending)
    blocks = bool(claimed and not sufficient)
    reason = ""
    if sufficient:
        reason = QUALIFICATION_SUFFICIENT
    elif blocks:
        reason = (
            PAPER_CLAIM_INCOMPLETE
            if pending == ("paper_claim",)
            else QUALIFICATION_INCOMPLETE
        )
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "record": record,
        "catalog": catalog.get("catalog") or {},
        "pending": pending,
        "sufficient": sufficient,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def qualification_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(qualification_view(state)["blocks_completion"])


__all__ = [
    "FORBIDDEN_SUBSTITUTES",
    "IMPLEMENTATION",
    "INTEGRATION",
    "LOCATION",
    "PAPER_CLAIM_FIELDS",
    "PAPER_CLAIM_INCOMPLETE",
    "PROGRAM_IDS",
    "QUALIFICATION_INCOMPLETE",
    "QUALIFICATION_SUFFICIENT",
    "ROLLOUT",
    "VALIDATION",
    "apply_program_catalog",
    "claims_program_catalog",
    "claims_qualification",
    "observe_location",
    "observe_rollout",
    "normalize_qualification",
    "paper_claim_complete",
    "pending_dimensions",
    "program_catalog_view",
    "qualification_blocks_completion",
    "qualification_sufficient",
    "qualification_view",
    "sealed_program_catalog",
    "update_qualification",
]

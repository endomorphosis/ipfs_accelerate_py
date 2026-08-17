"""DistributedStorageCrashAssuranceCampaign@1 compact fixtures and probes.

AAE-054: fourteen plan §11 distributed-state, storage-durability, and crash
boundary mutations. Expands compact recipes into sealed assurance fixtures
(matching AssuranceFixtureCorpus@1 identity rules), evaluates controlled
mutation observations against declared kill mechanisms / crash recovery
invariants, and emits a campaign receipt. Never mutates production policy or
opens network services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    RequirementProvenance,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent
RECIPES_PATH: Final[Path] = FIXTURE_DIR / "recipes.json"
PROBES_PATH: Final[Path] = FIXTURE_DIR / "probes.json"
PARENT_CORPUS_MANIFEST: Final[Path] = FIXTURE_DIR.parent / "manifest.json"
PARENT_SCHEMAS_DIR: Final[Path] = FIXTURE_DIR.parent / "schemas"
PARENT_RECIPE_CATALOG: Final[Path] = PARENT_SCHEMAS_DIR / "recipes.json"

INTERFACE: Final[str] = "DistributedStorageCrashAssuranceCampaign@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/distributed-storage-crash-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
EVIDENCE_ID: Final[str] = "aae/distributed-storage-crash-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-distributed-storage-crash-v1"
TASK_ID: Final[str] = "AAE-054"
BUNDLE: Final[str] = "distributed_storage_crash"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 1
MUTATION_INDEX_END: Final[int] = 14

# Plan §11 / acceptance order is normative for this campaign:
# transitions, CAS, fencing, owners, leases, idempotency, compensation,
# durable acknowledgement/read-back, and every required injected crash boundary.
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "illegal_state_transition",
    "cas_ignore_expected_old",
    "accept_stale_fencing_token",
    "mutate_without_ownership",
    "ignore_lease_expiry",
    "drop_idempotency_key",
    "incomplete_distributed_compensation",
    "ack_before_durable_commit",
    "skip_read_back_verification",
    "crash_after_mutant_create",
    "crash_during_worktree_setup",
    "crash_after_receipt_persist",
    "crash_before_policy_cas",
    "crash_after_cas_before_cleanup",
)

REQUIRED_CRASH_BOUNDARIES: Final[tuple[str, ...]] = (
    "crash.boundary.mutant_create",
    "crash.boundary.worktree_setup",
    "crash.boundary.receipt_persist",
    "crash.boundary.before_policy_cas",
    "crash.boundary.after_cas_before_cleanup",
)

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "sd_illegal_state_transition",
        "sd_cas_ignore_expected_old",
        "sd_accept_stale_fencing_token",
        "sd_mutate_without_ownership",
        "sd_ignore_lease_expiry",
        "sd_drop_idempotency_key",
        "sd_incomplete_distributed_compensation",
        "st_ack_before_durable_commit",
        "st_skip_read_back_verification",
        "st_claim_commit_without_sync",
        "st_claim_commit_without_read_back",
        "sd_cas_accept_stale_head",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "state.transition.legal",
        "state.cas.expected_old",
        "state.fencing.current",
        "state.ownership.required",
        "state.lease.valid",
        "state.idempotency.key_present",
        "state.compensation.complete",
        "storage.ack.after_durable",
        "storage.read_back.verified",
        "crash.boundary.mutant_create",
        "crash.boundary.worktree_setup",
        "crash.boundary.receipt_persist",
        "crash.boundary.before_policy_cas",
        "crash.boundary.after_cas_before_cleanup",
    }
)

SCENARIO_TO_ACCEPTANCE: Final[dict[str, str]] = {
    "illegal_state_transition": "transitions",
    "cas_ignore_expected_old": "CAS",
    "accept_stale_fencing_token": "fencing",
    "mutate_without_ownership": "owners",
    "ignore_lease_expiry": "leases",
    "drop_idempotency_key": "idempotency",
    "incomplete_distributed_compensation": "compensation",
    "ack_before_durable_commit": "durable acknowledgement",
    "skip_read_back_verification": "read-back",
    "crash_after_mutant_create": "crash boundary mutant_create",
    "crash_during_worktree_setup": "crash boundary worktree_setup",
    "crash_after_receipt_persist": "crash boundary receipt_persist",
    "crash_before_policy_cas": "crash boundary before_policy_cas",
    "crash_after_cas_before_cleanup": "crash boundary after_cas_before_cleanup",
}

EXPECTED_AUTHORITIES: Final[dict[str, str]] = {
    "illegal_state_transition": "state.transition.legal",
    "cas_ignore_expected_old": "state.cas.expected_old",
    "accept_stale_fencing_token": "state.fencing.current",
    "mutate_without_ownership": "state.ownership.required",
    "ignore_lease_expiry": "state.lease.valid",
    "drop_idempotency_key": "state.idempotency.key_present",
    "incomplete_distributed_compensation": "state.compensation.complete",
    "ack_before_durable_commit": "storage.ack.after_durable",
    "skip_read_back_verification": "storage.read_back.verified",
    "crash_after_mutant_create": "crash.boundary.mutant_create",
    "crash_during_worktree_setup": "crash.boundary.worktree_setup",
    "crash_after_receipt_persist": "crash.boundary.receipt_persist",
    "crash_before_policy_cas": "crash.boundary.before_policy_cas",
    "crash_after_cas_before_cleanup": "crash.boundary.after_cas_before_cleanup",
}

CRASH_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "crash_after_mutant_create",
        "crash_during_worktree_setup",
        "crash_after_receipt_persist",
        "crash_before_policy_cas",
        "crash_after_cas_before_cleanup",
    }
)


class DistributedStorageCrashCampaignError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise DistributedStorageCrashCampaignError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise DistributedStorageCrashCampaignError(
            "recipes.interface must be DistributedStorageCrashAssuranceCampaign@1"
        )
    if doc.get("task_id") != TASK_ID:
        raise DistributedStorageCrashCampaignError("recipes.task_id must be AAE-054")
    if doc.get("bundle") != BUNDLE:
        raise DistributedStorageCrashCampaignError(
            "recipes.bundle must be distributed_storage_crash"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise DistributedStorageCrashCampaignError(
            "production policy change is forbidden for fixture campaigns"
        )
    recipes = list(doc.get("recipes") or [])
    expected = MUTATION_INDEX_END - MUTATION_INDEX_START + 1
    if len(recipes) != expected:
        raise DistributedStorageCrashCampaignError(
            f"expected {expected} distributed_storage_crash recipes, "
            f"found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise DistributedStorageCrashCampaignError(
            "probes.interface must be DistributedStorageCrashAssuranceCampaign@1"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise DistributedStorageCrashCampaignError(
            "probes must forbid production policy change"
        )
    probes = list(doc.get("probes") or [])
    if len(probes) != 14:
        raise DistributedStorageCrashCampaignError(
            f"expected 14 probes, found {len(probes)}"
        )
    return doc


def expand_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic expansion of a compact recipe into a sealed fixture."""
    required = {
        "requirement_id",
        "intended_behavior",
        "source_id",
        "source_path",
        "operator_id",
        "operator_version",
        "operator_class",
        "detector_id",
        "detector_revision",
        "detector_kind",
        "detector_strength",
        "expected_terminal_status",
        "violated_claim",
        "observation_rationale",
        "expected_outcome",
        "kill_mechanisms",
        "max_steps",
        "max_depth",
        "timeout_ms",
        "observation_points",
        "fail_closed",
        "fixture_id",
        "campaign",
        "campaign_bundle",
        "scenario",
        "partition",
        "risk",
        "relatedness_key",
        "target_id",
        "used_for_candidate_generation",
        "critical",
    }
    missing = required - set(recipe)
    if missing:
        raise DistributedStorageCrashCampaignError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise DistributedStorageCrashCampaignError(
            f"{recipe['fixture_id']}: campaign_bundle must be "
            "distributed_storage_crash"
        )
    if recipe["campaign"] != "distributed_storage_crash":
        raise DistributedStorageCrashCampaignError(
            f"{recipe['fixture_id']}: campaign must be distributed_storage_crash"
        )
    if recipe["critical"] is not True:
        raise DistributedStorageCrashCampaignError(
            f"{recipe['fixture_id']}: all distributed/crash fixtures are critical"
        )
    if recipe["operator_class"] not in {
        "state_distributed",
        "storage_durability",
    }:
        raise DistributedStorageCrashCampaignError(
            f"{recipe['fixture_id']}: operator_class must be state_distributed "
            "or storage_durability"
        )

    notes = recipe.get("notes")
    prov = RequirementProvenance(
        requirement_id=recipe["requirement_id"],
        intended_behavior=recipe["intended_behavior"],
        source_id=recipe["source_id"],
        requirement_cid=None,
        source_path=recipe["source_path"],
        notes=notes,
    )
    operator = {
        "operator_id": recipe["operator_id"],
        "operator_version": recipe["operator_version"],
        "operator_class": recipe["operator_class"],
    }
    expected_detector = {
        "schema": DETECTOR_SCHEMA,
        "detector_id": recipe["detector_id"],
        "detector_revision": recipe["detector_revision"],
        "detector_kind": recipe["detector_kind"],
        "strength": recipe["detector_strength"],
        "expected_terminal_status": recipe["expected_terminal_status"],
        "violated_claim": recipe["violated_claim"],
        "observation_rationale": recipe["observation_rationale"],
    }
    expected_detector["detector_cid"] = cid_for_structured(
        {k: v for k, v in expected_detector.items() if k != "detector_cid"}
    )
    oracle_body = {
        "schema": ORACLE_SCHEMA,
        "oracle_id": f"oracle.{recipe['fixture_id']}",
        "expected_outcome": recipe["expected_outcome"],
        "kill_mechanisms": list(recipe["kill_mechanisms"]),
        "bounds": {
            "max_steps": recipe["max_steps"],
            "max_depth": recipe["max_depth"],
            "timeout_ms": recipe["timeout_ms"],
        },
        "observation_points": list(recipe["observation_points"]),
        "fail_closed": recipe["fail_closed"],
    }
    oracle_body["oracle_cid"] = cid_for_structured(
        {k: v for k, v in oracle_body.items() if k != "oracle_cid"}
    )
    recipe_identity = {
        key: value for key, value in recipe.items() if key != "mutation_index"
    }
    identity_payload = {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": recipe["fixture_id"],
        "campaign": recipe["campaign"],
        "campaign_bundle": recipe["campaign_bundle"],
        "scenario": recipe["scenario"],
        "partition": recipe["partition"],
        "risk": recipe["risk"],
        "operator": operator,
        "requirement_provenance": prov.to_dict(),
        "expected_detector": expected_detector,
        "bounded_oracle": oracle_body,
        "relatedness_key": recipe["relatedness_key"],
        "target_id": recipe["target_id"],
        "used_for_candidate_generation": recipe["used_for_candidate_generation"],
        "critical": recipe["critical"],
        "notes": notes,
    }
    fixture = dict(identity_payload)
    fixture["fixture_cid"] = cid_for_structured(identity_payload)
    fixture["recipe_cid"] = cid_for_structured(
        {"schema": RECIPE_SCHEMA, **recipe_identity}
    )
    if "mutation_index" in recipe:
        fixture["mutation_index"] = int(recipe["mutation_index"])
    return fixture


def expand_all_fixtures(
    recipes_doc: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    doc = recipes_doc if recipes_doc is not None else load_recipes()
    fixtures = [expand_recipe(r) for r in doc["recipes"]]
    fixtures.sort(
        key=lambda item: (item.get("mutation_index") or 0, item["fixture_id"])
    )
    return fixtures


def parent_corpus_distributed_storage_crash_fixtures() -> list[dict[str, Any]]:
    """Load sealed distributed_storage_crash fixtures from the parent corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise DistributedStorageCrashCampaignError(
            f"parent corpus missing: {PARENT_CORPUS_MANIFEST}"
        )
    manifest = _load_json(PARENT_CORPUS_MANIFEST)
    fixtures = [
        f
        for f in manifest.get("fixtures") or []
        if f.get("campaign_bundle") == BUNDLE
    ]
    fixtures.sort(key=lambda item: item["fixture_id"])
    if len(fixtures) != 14:
        raise DistributedStorageCrashCampaignError(
            "parent corpus distributed_storage_crash count mismatch: "
            f"{len(fixtures)}"
        )
    return fixtures


def _fixture_identity_without_mutation_index(
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in fixture.items()
        if key not in {"fixture_cid", "recipe_cid", "mutation_index"}
    }


def assert_fixtures_match_parent_corpus(
    fixtures: Sequence[Mapping[str, Any]],
) -> None:
    """Expanded campaign fixtures must seal-equal parent corpus records."""
    parent = {
        f["fixture_id"]: f
        for f in parent_corpus_distributed_storage_crash_fixtures()
    }
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise DistributedStorageCrashCampaignError(
            "fixture_id set diverges from parent corpus "
            "distributed_storage_crash bundle: "
            f"local_only={sorted(set(local) - set(parent))} "
            f"parent_only={sorted(set(parent) - set(local))}"
        )
    for fixture_id, expanded in local.items():
        sealed = parent[fixture_id]
        left = _fixture_identity_without_mutation_index(expanded)
        right = {
            k: v
            for k, v in sealed.items()
            if k not in {"fixture_cid", "recipe_cid"}
        }
        if left != right:
            raise DistributedStorageCrashCampaignError(
                f"{fixture_id}: expanded fixture identity diverges from "
                "parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise DistributedStorageCrashCampaignError(
                f"{fixture_id}: fixture_cid mismatch with parent corpus"
            )
        if expanded["recipe_cid"] != sealed["recipe_cid"]:
            raise DistributedStorageCrashCampaignError(
                f"{fixture_id}: recipe_cid mismatch with parent corpus"
            )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of evaluating one controlled distributed/crash mutation probe."""

    fixture_id: str
    scenario: str
    mutation_index: int
    killed: bool
    kill_mechanism: str
    reason: str
    terminal_status: str
    authority: str
    details: Mapping[str, Any]
    crash_boundary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "scenario": self.scenario,
            "mutation_index": self.mutation_index,
            "killed": self.killed,
            "kill_mechanism": self.kill_mechanism,
            "reason": self.reason,
            "terminal_status": self.terminal_status,
            "authority": self.authority,
            "details": dict(self.details),
            "crash_boundary": self.crash_boundary,
        }


def _policy_changed(observation: Mapping[str, Any]) -> bool:
    return bool(observation.get("production_policy_changed"))


def _decision_accept(observation: Mapping[str, Any]) -> bool:
    decision = str(observation.get("decision") or "").lower()
    if decision in {"allow", "allowed", "accept", "accepted", "commit", "committed"}:
        return True
    return bool(
        observation.get("write_accepted")
        or observation.get("mutation_accepted")
        or observation.get("commit_claimed")
        or observation.get("success_claimed")
        or observation.get("presented_as_valid")
    )


# ---------------------------------------------------------------------------
# State / distributed evaluators
# ---------------------------------------------------------------------------


def _evaluate_illegal_state_transition(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    from_state = observation.get("from_state")
    to_state = observation.get("to_state")
    legal = {
        (edge[0], edge[1])
        for edge in (observation.get("legal_transitions") or [])
        if isinstance(edge, (list, tuple)) and len(edge) >= 2
    }
    accepted = bool(observation.get("transition_accepted")) or _decision_accept(
        observation
    )
    edge = (from_state, to_state)
    illegal = (
        from_state is not None
        and to_state is not None
        and (not legal or edge not in legal)
    )
    legal_list = sorted(([a, b] for a, b in legal), key=lambda item: (str(item[0]), str(item[1])))
    if accepted and illegal:
        return True, "illegal_state_transition", {
            "from_state": from_state,
            "to_state": to_state,
            "legal_transitions": legal_list,
        }
    return False, "transition_legal", {
        "from_state": from_state,
        "to_state": to_state,
        "illegal": illegal,
    }


def _evaluate_cas_ignore_expected_old(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    expected_old = observation.get("expected_old_revision")
    observed_old = observation.get("observed_old_revision")
    cas_enforced = observation.get("cas_expected_old_enforced")
    accepted = bool(observation.get("write_accepted")) or _decision_accept(
        observation
    )
    mismatch = (
        expected_old is not None
        and observed_old is not None
        and expected_old != observed_old
    )
    ignored = cas_enforced is False or (
        accepted and mismatch and observation.get("cas_checked") is False
    )
    if accepted and (mismatch or cas_enforced is False) and ignored:
        return True, "cas_ignore_expected_old", {
            "expected_old_revision": expected_old,
            "observed_old_revision": observed_old,
            "cas_expected_old_enforced": cas_enforced,
        }
    if accepted and mismatch:
        return True, "cas_ignore_expected_old", {
            "expected_old_revision": expected_old,
            "observed_old_revision": observed_old,
        }
    return False, "cas_expected_old_enforced", {
        "expected_old_revision": expected_old,
        "observed_old_revision": observed_old,
        "mismatch": mismatch,
    }


def _evaluate_accept_stale_fencing_token(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    presented = observation.get("presented_fencing_token")
    current = observation.get("current_fencing_token")
    accepted = bool(observation.get("write_accepted")) or _decision_accept(
        observation
    )
    stale = (
        presented is not None
        and current is not None
        and int(presented) < int(current)
    )
    if accepted and stale:
        return True, "stale_fencing_token_accepted", {
            "presented_fencing_token": presented,
            "current_fencing_token": current,
        }
    return False, "fencing_token_current", {
        "presented_fencing_token": presented,
        "current_fencing_token": current,
        "stale": stale,
    }


def _evaluate_mutate_without_ownership(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    owner = observation.get("current_owner")
    actor = observation.get("actor_id")
    has_ownership = bool(observation.get("ownership_present"))
    accepted = bool(observation.get("mutation_accepted")) or _decision_accept(
        observation
    )
    missing_owner = not has_ownership or (
        owner is not None and actor is not None and owner != actor
    )
    if accepted and missing_owner:
        return True, "mutate_without_ownership", {
            "current_owner": owner,
            "actor_id": actor,
            "ownership_present": has_ownership,
        }
    return False, "ownership_required", {
        "current_owner": owner,
        "actor_id": actor,
        "missing_owner": missing_owner,
    }


def _evaluate_ignore_lease_expiry(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    expires_at = observation.get("lease_expires_at_ms")
    now_ms = observation.get("now_ms")
    exclusive = bool(observation.get("exclusive_mutation_accepted")) or bool(
        observation.get("write_accepted")
    )
    expired = (
        expires_at is not None
        and now_ms is not None
        and int(now_ms) > int(expires_at)
    )
    if exclusive and expired:
        return True, "lease_expiry_ignored", {
            "lease_expires_at_ms": expires_at,
            "now_ms": now_ms,
        }
    return False, "lease_not_expired", {
        "lease_expires_at_ms": expires_at,
        "now_ms": now_ms,
        "expired": expired,
    }


def _evaluate_drop_idempotency_key(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    key_present = bool(observation.get("idempotency_key_present"))
    retry_safe = bool(observation.get("retry_safe_claimed", True))
    accepted = bool(observation.get("mutation_accepted")) or _decision_accept(
        observation
    )
    if accepted and retry_safe and not key_present:
        return True, "idempotency_key_dropped", {
            "idempotency_key_present": key_present,
            "retry_safe_claimed": retry_safe,
        }
    return False, "idempotency_key_present", {
        "idempotency_key_present": key_present,
        "retry_safe_claimed": retry_safe,
    }


def _evaluate_incomplete_distributed_compensation(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    partial = bool(observation.get("partial_failure"))
    required = list(observation.get("required_compensations") or [])
    completed = list(observation.get("completed_compensations") or [])
    terminal_ok = bool(observation.get("terminal_success_claimed", True))
    missing = sorted(set(required) - set(completed))
    if partial and terminal_ok and missing:
        return True, "incomplete_distributed_compensation", {
            "missing_compensations": missing,
            "partial_failure": partial,
        }
    return False, "compensation_complete", {
        "missing_compensations": missing,
        "partial_failure": partial,
    }


# ---------------------------------------------------------------------------
# Storage / durability evaluators
# ---------------------------------------------------------------------------


def _evaluate_ack_before_durable_commit(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    ack_sent = bool(observation.get("ack_sent"))
    durable = bool(observation.get("durable_commit_observed"))
    success = bool(observation.get("success_claimed")) or _decision_accept(
        observation
    )
    if (ack_sent or success) and not durable:
        return True, "ack_before_durable_commit", {
            "ack_sent": ack_sent,
            "durable_commit_observed": durable,
            "success_claimed": success,
        }
    return False, "ack_after_durable", {
        "ack_sent": ack_sent,
        "durable_commit_observed": durable,
    }


def _evaluate_skip_read_back_verification(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    read_back = bool(observation.get("read_back_verified"))
    success = bool(observation.get("success_claimed")) or _decision_accept(
        observation
    )
    write_performed = bool(observation.get("write_performed", True))
    if write_performed and success and not read_back:
        return True, "read_back_verification_skipped", {
            "read_back_verified": read_back,
            "success_claimed": success,
        }
    return False, "read_back_verified", {
        "read_back_verified": read_back,
        "success_claimed": success,
    }


# ---------------------------------------------------------------------------
# Injected crash boundary evaluators
# ---------------------------------------------------------------------------


def _evaluate_crash_after_mutant_create(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    injected = bool(observation.get("crash_injected"))
    boundary = str(observation.get("crash_boundary") or "")
    partial_promotion = bool(observation.get("partial_promotion"))
    ambiguous = bool(observation.get("ambiguous_success_claimed"))
    recovered_clean = bool(observation.get("recovered_without_partial_promotion"))
    # Kill when crash at mutant-create boundary leaves partial promotion or
    # ambiguous success, or recovery fails to reject partial state.
    violation = injected and boundary in {
        "mutant_create",
        "crash.boundary.mutant_create",
    } and (partial_promotion or ambiguous or not recovered_clean)
    if violation:
        return True, "crash_after_mutant_create_unrecovered", {
            "crash_boundary": boundary,
            "partial_promotion": partial_promotion,
            "ambiguous_success_claimed": ambiguous,
            "recovered_without_partial_promotion": recovered_clean,
        }
    return False, "recover_after_mutant_create", {
        "crash_boundary": boundary,
        "injected": injected,
    }


def _evaluate_crash_during_worktree_setup(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    injected = bool(observation.get("crash_injected"))
    boundary = str(observation.get("crash_boundary") or "")
    foreign = list(observation.get("foreign_paths_touched") or [])
    only_owned = bool(observation.get("only_owned_disposable_state", True))
    cleanup_safe = bool(observation.get("cleanup_safe", True))
    violation = injected and boundary in {
        "worktree_setup",
        "crash.boundary.worktree_setup",
    } and (foreign or not only_owned or not cleanup_safe)
    if violation:
        return True, "crash_during_worktree_setup_unsafe", {
            "crash_boundary": boundary,
            "foreign_paths_touched": foreign,
            "only_owned_disposable_state": only_owned,
            "cleanup_safe": cleanup_safe,
        }
    return False, "recover_worktree_setup", {
        "crash_boundary": boundary,
        "injected": injected,
    }


def _evaluate_crash_after_receipt_persist(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    injected = bool(observation.get("crash_injected"))
    boundary = str(observation.get("crash_boundary") or "")
    receipt_present = bool(observation.get("receipt_persisted"))
    immutable = bool(observation.get("immutable_artifacts_preserved", True))
    ambiguous = bool(observation.get("ambiguous_success_claimed"))
    # Kill when crash after receipt persist loses immutability or claims
    # ambiguous success without a completed receipt.
    violation = injected and boundary in {
        "receipt_persist",
        "crash.boundary.receipt_persist",
    } and (not immutable or ambiguous or not receipt_present)
    if violation:
        return True, "crash_after_receipt_persist_ambiguous", {
            "crash_boundary": boundary,
            "receipt_persisted": receipt_present,
            "immutable_artifacts_preserved": immutable,
            "ambiguous_success_claimed": ambiguous,
        }
    return False, "recover_after_receipt_persist", {
        "crash_boundary": boundary,
        "injected": injected,
    }


def _evaluate_crash_before_policy_cas(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    injected = bool(observation.get("crash_injected"))
    boundary = str(observation.get("crash_boundary") or "")
    policy_promoted = bool(observation.get("policy_promoted"))
    pre_cas_revision = observation.get("pre_crash_policy_revision")
    post_revision = observation.get("post_recovery_policy_revision")
    changed = (
        pre_cas_revision is not None
        and post_revision is not None
        and pre_cas_revision != post_revision
    )
    violation = injected and boundary in {
        "before_policy_cas",
        "crash.boundary.before_policy_cas",
    } and (policy_promoted or changed)
    if violation:
        return True, "crash_before_policy_cas_partial_promotion", {
            "crash_boundary": boundary,
            "policy_promoted": policy_promoted,
            "pre_crash_policy_revision": pre_cas_revision,
            "post_recovery_policy_revision": post_revision,
        }
    return False, "no_partial_promotion", {
        "crash_boundary": boundary,
        "injected": injected,
    }


def _evaluate_crash_after_cas_before_cleanup(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    injected = bool(observation.get("crash_injected"))
    boundary = str(observation.get("crash_boundary") or "")
    cas_committed = bool(observation.get("cas_committed"))
    promotion_preserved = bool(observation.get("promotion_preserved", True))
    cleanup_recovered = bool(observation.get("cleanup_recovered", True))
    # Kill when CAS succeeded but restart loses promotion or fails cleanup.
    violation = injected and boundary in {
        "after_cas_before_cleanup",
        "crash.boundary.after_cas_before_cleanup",
    } and cas_committed and (not promotion_preserved or not cleanup_recovered)
    if violation:
        return True, "crash_after_cas_before_cleanup_unrecovered", {
            "crash_boundary": boundary,
            "cas_committed": cas_committed,
            "promotion_preserved": promotion_preserved,
            "cleanup_recovered": cleanup_recovered,
        }
    return False, "cas_then_cleanup_recoverable", {
        "crash_boundary": boundary,
        "injected": injected,
    }


_SCENARIO_EVALUATORS = {
    "illegal_state_transition": _evaluate_illegal_state_transition,
    "cas_ignore_expected_old": _evaluate_cas_ignore_expected_old,
    "accept_stale_fencing_token": _evaluate_accept_stale_fencing_token,
    "mutate_without_ownership": _evaluate_mutate_without_ownership,
    "ignore_lease_expiry": _evaluate_ignore_lease_expiry,
    "drop_idempotency_key": _evaluate_drop_idempotency_key,
    "incomplete_distributed_compensation": (
        _evaluate_incomplete_distributed_compensation
    ),
    "ack_before_durable_commit": _evaluate_ack_before_durable_commit,
    "skip_read_back_verification": _evaluate_skip_read_back_verification,
    "crash_after_mutant_create": _evaluate_crash_after_mutant_create,
    "crash_during_worktree_setup": _evaluate_crash_during_worktree_setup,
    "crash_after_receipt_persist": _evaluate_crash_after_receipt_persist,
    "crash_before_policy_cas": _evaluate_crash_before_policy_cas,
    "crash_after_cas_before_cleanup": _evaluate_crash_after_cas_before_cleanup,
}


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory for this campaign."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise DistributedStorageCrashCampaignError(
            f"no evaluator for scenario {scenario!r}"
        )
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise DistributedStorageCrashCampaignError(
            f"{scenario}: observation must be a mapping"
        )
    if _policy_changed(observation):
        raise DistributedStorageCrashCampaignError(
            f"{probe.get('fixture_id')}: production policy change observed; "
            "forbidden"
        )
    killed, reason, details = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = (
        authority
        if authority in kill_mechanisms
        else (kill_mechanisms[0] if kill_mechanisms else authority)
    )
    if "runtime.invariant" not in kill_mechanisms and not any(
        m.startswith("state.")
        or m.startswith("storage.")
        or m.startswith("crash.")
        for m in kill_mechanisms
    ):
        raise DistributedStorageCrashCampaignError(
            f"{probe.get('fixture_id')}: probe must declare runtime/state/"
            "storage/crash kill mechanisms"
        )
    expected_kill = probe.get("expected_kill", True)
    if expected_kill and not killed:
        raise DistributedStorageCrashCampaignError(
            f"{probe.get('fixture_id')}: expected kill but probe survived "
            f"(reason={reason})"
        )
    crash_boundary = None
    if scenario in CRASH_SCENARIOS:
        crash_boundary = authority if authority.startswith("crash.boundary.") else (
            str(observation.get("crash_boundary") or authority)
        )
    return ProbeResult(
        fixture_id=str(probe["fixture_id"]),
        scenario=scenario,
        mutation_index=int(probe.get("mutation_index") or 0),
        killed=bool(killed),
        kill_mechanism=kill_mechanism,
        reason=reason,
        terminal_status="rejected" if killed else "accepted",
        authority=authority,
        details=details,
        crash_boundary=crash_boundary,
    )


def evaluate_all_probes(
    probes_doc: Mapping[str, Any] | None = None,
) -> list[ProbeResult]:
    doc = probes_doc if probes_doc is not None else load_probes()
    results = [evaluate_probe(p) for p in doc["probes"]]
    results.sort(key=lambda item: item.mutation_index)
    return results


def build_campaign(
    *,
    recipes_doc: Mapping[str, Any] | None = None,
    probes_doc: Mapping[str, Any] | None = None,
    verify_parent: bool = True,
    run_probes: bool = True,
) -> dict[str, Any]:
    """Build the sealed DistributedStorageCrashAssuranceCampaign@1 document."""
    recipes = recipes_doc if recipes_doc is not None else load_recipes()
    probes = probes_doc if probes_doc is not None else load_probes()
    fixtures = expand_all_fixtures(recipes)
    if verify_parent:
        assert_fixtures_match_parent_corpus(fixtures)

    scenarios = [f["scenario"] for f in fixtures]
    if tuple(scenarios) != REQUIRED_SCENARIOS:
        by_index = sorted(
            fixtures,
            key=lambda item: int(item.get("mutation_index") or 0),
        )
        scenarios = [f["scenario"] for f in by_index]
        if tuple(scenarios) != REQUIRED_SCENARIOS:
            raise DistributedStorageCrashCampaignError(
                f"scenario order mismatch: {scenarios}"
            )
        fixtures = by_index

    probe_results: list[dict[str, Any]] = []
    if run_probes:
        results = evaluate_all_probes(probes)
        probe_results = [r.to_dict() for r in results]
        if not all(item["killed"] for item in probe_results):
            survivors = [
                item["fixture_id"]
                for item in probe_results
                if not item["killed"]
            ]
            raise DistributedStorageCrashCampaignError(
                f"controlled mutations survived detectors: {survivors}"
            )

    membership = {
        "diagnosis": sorted(
            f["fixture_id"] for f in fixtures if f["partition"] == "diagnosis"
        ),
        "development": sorted(
            f["fixture_id"] for f in fixtures if f["partition"] == "development"
        ),
        "held_out": sorted(
            f["fixture_id"] for f in fixtures if f["partition"] == "held_out"
        ),
    }

    crash_boundaries_covered = sorted(
        {
            f["expected_detector"]["detector_id"]
            for f in fixtures
            if f["expected_detector"]["detector_id"].startswith("crash.boundary.")
        }
    )
    if tuple(crash_boundaries_covered) != tuple(sorted(REQUIRED_CRASH_BOUNDARIES)):
        missing = set(REQUIRED_CRASH_BOUNDARIES) - set(crash_boundaries_covered)
        if missing:
            raise DistributedStorageCrashCampaignError(
                f"missing required crash boundaries: {sorted(missing)}"
            )

    body: dict[str, Any] = {
        "interface": INTERFACE,
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "evidence_id": EVIDENCE_ID,
        "parent_corpus_id": PARENT_CORPUS_ID,
        "mutation_index_start": MUTATION_INDEX_START,
        "mutation_index_end": MUTATION_INDEX_END,
        "fixture_count": len(fixtures),
        "scenarios": list(REQUIRED_SCENARIOS),
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "required_operator_ids": sorted(REQUIRED_OPERATOR_IDS),
        "required_detector_ids": sorted(REQUIRED_DETECTOR_IDS),
        "required_crash_boundaries": list(REQUIRED_CRASH_BOUNDARIES),
        "crash_boundaries_covered": crash_boundaries_covered,
        "partition_membership": membership,
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "fixtures": [
            {
                "fixture_id": f["fixture_id"],
                "fixture_cid": f["fixture_cid"],
                "scenario": f["scenario"],
                "mutation_index": f.get("mutation_index"),
                "partition": f["partition"],
                "operator_id": f["operator"]["operator_id"],
                "operator_class": f["operator"]["operator_class"],
                "detector_id": f["expected_detector"]["detector_id"],
                "kill_mechanisms": list(f["bounded_oracle"]["kill_mechanisms"]),
                "expected_outcome": f["bounded_oracle"]["expected_outcome"],
                "fail_closed": f["bounded_oracle"]["fail_closed"],
                "critical": f["critical"],
                "acceptance_class": SCENARIO_TO_ACCEPTANCE[f["scenario"]],
            }
            for f in fixtures
        ],
        "probe_results": probe_results,
        "notes": (
            "Controlled distributed-state, storage-durability, and crash "
            "boundary mutations (14). Covers transitions, CAS, fencing, owners, "
            "leases, idempotency, compensation, durable acknowledgement/"
            "read-back, and every required injected crash boundary. Campaign "
            "does not change production policy."
        ),
    }
    campaign_cid = cid_for_structured(body)
    body["campaign_cid"] = campaign_cid
    return body


def write_campaign_snapshot(path: Path | None = None) -> dict[str, Any]:
    """Materialize a sealed campaign snapshot (optional durable artifact)."""
    campaign = build_campaign(verify_parent=True, run_probes=True)
    target = path or (FIXTURE_DIR / "campaign.json")
    target.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return campaign


def materialize_recipes_from_parent_catalog(
    path: Path | None = None,
) -> dict[str, Any]:
    """Build campaign recipes.json from the AAE-049 parent recipe catalog."""
    catalog = _load_json(PARENT_RECIPE_CATALOG)
    by_scenario = {
        r["scenario"]: r
        for r in catalog.get("recipes") or []
        if r.get("campaign_bundle") == BUNDLE
    }
    recipes: list[dict[str, Any]] = []
    for index, scenario in enumerate(REQUIRED_SCENARIOS, start=1):
        if scenario not in by_scenario:
            raise DistributedStorageCrashCampaignError(
                f"parent recipe catalog missing scenario {scenario!r}"
            )
        recipe = dict(by_scenario[scenario])
        recipe["mutation_index"] = index
        recipes.append(recipe)
    doc: dict[str, Any] = {
        "interface": INTERFACE,
        "schema": RECIPE_SCHEMA,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "campaign_id": CAMPAIGN_ID,
        "evidence_id": EVIDENCE_ID,
        "parent_corpus_id": PARENT_CORPUS_ID,
        "mutation_index_start": MUTATION_INDEX_START,
        "mutation_index_end": MUTATION_INDEX_END,
        "recipe_count": len(recipes),
        "production_policy_change_allowed": False,
        "scenarios": list(REQUIRED_SCENARIOS),
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "required_crash_boundaries": list(REQUIRED_CRASH_BOUNDARIES),
        "recipes": recipes,
        "notes": (
            "Compact distributed-storage-crash recipes expanded by catalog.py; "
            "identity matches AssuranceFixtureCorpus@1."
        ),
    }
    target = path or RECIPES_PATH
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def default_probes_document() -> dict[str, Any]:
    """Hermetic observations that distributed/durability/crash detectors kill."""
    probes = [
        {
            "mutation_index": 1,
            "fixture_id": "dist.illegal_state_transition",
            "scenario": "illegal_state_transition",
            "operator_id": "sd_illegal_state_transition",
            "detector_id": "state.transition.legal",
            "authority": "state.transition.legal",
            "kill_mechanisms": ["runtime.invariant", "state.transition.legal"],
            "expected_kill": True,
            "observation": {
                "from_state": "pending",
                "to_state": "promoted",
                "legal_transitions": [
                    ["pending", "running"],
                    ["running", "completed"],
                    ["running", "failed"],
                ],
                "transition_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 2,
            "fixture_id": "dist.cas_ignore_expected_old",
            "scenario": "cas_ignore_expected_old",
            "operator_id": "sd_cas_ignore_expected_old",
            "detector_id": "state.cas.expected_old",
            "authority": "state.cas.expected_old",
            "kill_mechanisms": ["runtime.invariant", "state.cas.expected_old"],
            "expected_kill": True,
            "observation": {
                "expected_old_revision": "rev-10",
                "observed_old_revision": "rev-12",
                "cas_checked": False,
                "cas_expected_old_enforced": False,
                "write_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 3,
            "fixture_id": "dist.stale_fencing",
            "scenario": "accept_stale_fencing_token",
            "operator_id": "sd_accept_stale_fencing_token",
            "detector_id": "state.fencing.current",
            "authority": "state.fencing.current",
            "kill_mechanisms": ["runtime.invariant", "state.fencing.current"],
            "expected_kill": True,
            "observation": {
                "presented_fencing_token": 3,
                "current_fencing_token": 7,
                "write_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 4,
            "fixture_id": "dist.mutate_without_ownership",
            "scenario": "mutate_without_ownership",
            "operator_id": "sd_mutate_without_ownership",
            "detector_id": "state.ownership.required",
            "authority": "state.ownership.required",
            "kill_mechanisms": [
                "runtime.invariant",
                "state.ownership.required",
            ],
            "expected_kill": True,
            "observation": {
                "current_owner": "worker-a",
                "actor_id": "worker-b",
                "ownership_present": False,
                "mutation_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 5,
            "fixture_id": "dist.ignore_lease_expiry",
            "scenario": "ignore_lease_expiry",
            "operator_id": "sd_ignore_lease_expiry",
            "detector_id": "state.lease.valid",
            "authority": "state.lease.valid",
            "kill_mechanisms": ["runtime.invariant", "state.lease.valid"],
            "expected_kill": True,
            "observation": {
                "lease_expires_at_ms": 1_000,
                "now_ms": 5_000,
                "exclusive_mutation_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 6,
            "fixture_id": "dist.drop_idempotency_key",
            "scenario": "drop_idempotency_key",
            "operator_id": "sd_drop_idempotency_key",
            "detector_id": "state.idempotency.key_present",
            "authority": "state.idempotency.key_present",
            "kill_mechanisms": [
                "runtime.invariant",
                "state.idempotency.key_present",
            ],
            "expected_kill": True,
            "observation": {
                "idempotency_key_present": False,
                "retry_safe_claimed": True,
                "mutation_accepted": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 7,
            "fixture_id": "dist.incomplete_compensation",
            "scenario": "incomplete_distributed_compensation",
            "operator_id": "sd_incomplete_distributed_compensation",
            "detector_id": "state.compensation.complete",
            "authority": "state.compensation.complete",
            "kill_mechanisms": [
                "runtime.invariant",
                "state.compensation.complete",
            ],
            "expected_kill": True,
            "observation": {
                "partial_failure": True,
                "required_compensations": [
                    "undo_replica_a",
                    "undo_replica_b",
                    "release_lease",
                ],
                "completed_compensations": ["undo_replica_a"],
                "terminal_success_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 8,
            "fixture_id": "dist.ack_before_durable",
            "scenario": "ack_before_durable_commit",
            "operator_id": "st_ack_before_durable_commit",
            "detector_id": "storage.ack.after_durable",
            "authority": "storage.ack.after_durable",
            "kill_mechanisms": [
                "runtime.invariant",
                "storage.ack.after_durable",
            ],
            "expected_kill": True,
            "observation": {
                "ack_sent": True,
                "durable_commit_observed": False,
                "success_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 9,
            "fixture_id": "dist.skip_read_back",
            "scenario": "skip_read_back_verification",
            "operator_id": "st_skip_read_back_verification",
            "detector_id": "storage.read_back.verified",
            "authority": "storage.read_back.verified",
            "kill_mechanisms": [
                "runtime.invariant",
                "storage.read_back.verified",
            ],
            "expected_kill": True,
            "observation": {
                "write_performed": True,
                "read_back_verified": False,
                "success_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 10,
            "fixture_id": "dist.crash_after_mutant_create",
            "scenario": "crash_after_mutant_create",
            "operator_id": "st_claim_commit_without_sync",
            "detector_id": "crash.boundary.mutant_create",
            "authority": "crash.boundary.mutant_create",
            "kill_mechanisms": [
                "crash.boundary.mutant_create",
                "runtime.invariant",
            ],
            "expected_kill": True,
            "observation": {
                "crash_injected": True,
                "crash_boundary": "crash.boundary.mutant_create",
                "partial_promotion": True,
                "ambiguous_success_claimed": True,
                "recovered_without_partial_promotion": False,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 11,
            "fixture_id": "dist.crash_during_worktree_setup",
            "scenario": "crash_during_worktree_setup",
            "operator_id": "st_claim_commit_without_sync",
            "detector_id": "crash.boundary.worktree_setup",
            "authority": "crash.boundary.worktree_setup",
            "kill_mechanisms": [
                "crash.boundary.worktree_setup",
                "runtime.invariant",
            ],
            "expected_kill": True,
            "observation": {
                "crash_injected": True,
                "crash_boundary": "crash.boundary.worktree_setup",
                "foreign_paths_touched": ["/tmp/shared-policy"],
                "only_owned_disposable_state": False,
                "cleanup_safe": False,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 12,
            "fixture_id": "dist.crash_after_receipt_persist",
            "scenario": "crash_after_receipt_persist",
            "operator_id": "st_claim_commit_without_read_back",
            "detector_id": "crash.boundary.receipt_persist",
            "authority": "crash.boundary.receipt_persist",
            "kill_mechanisms": [
                "crash.boundary.receipt_persist",
                "runtime.invariant",
            ],
            "expected_kill": True,
            "observation": {
                "crash_injected": True,
                "crash_boundary": "crash.boundary.receipt_persist",
                "receipt_persisted": True,
                "immutable_artifacts_preserved": False,
                "ambiguous_success_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 13,
            "fixture_id": "dist.crash_before_policy_cas",
            "scenario": "crash_before_policy_cas",
            "operator_id": "sd_cas_accept_stale_head",
            "detector_id": "crash.boundary.before_policy_cas",
            "authority": "crash.boundary.before_policy_cas",
            "kill_mechanisms": [
                "crash.boundary.before_policy_cas",
                "runtime.invariant",
            ],
            "expected_kill": True,
            "observation": {
                "crash_injected": True,
                "crash_boundary": "crash.boundary.before_policy_cas",
                "policy_promoted": True,
                "pre_crash_policy_revision": "pol-1",
                "post_recovery_policy_revision": "pol-2",
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 14,
            "fixture_id": "dist.crash_after_cas_before_cleanup",
            "scenario": "crash_after_cas_before_cleanup",
            "operator_id": "sd_cas_ignore_expected_old",
            "detector_id": "crash.boundary.after_cas_before_cleanup",
            "authority": "crash.boundary.after_cas_before_cleanup",
            "kill_mechanisms": [
                "crash.boundary.after_cas_before_cleanup",
                "runtime.invariant",
            ],
            "expected_kill": True,
            "observation": {
                "crash_injected": True,
                "crash_boundary": "crash.boundary.after_cas_before_cleanup",
                "cas_committed": True,
                "promotion_preserved": False,
                "cleanup_recovered": False,
                "production_policy_changed": False,
            },
        },
    ]
    return {
        "interface": INTERFACE,
        "schema": "aae/distributed-storage-crash-probes@1",
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "campaign_id": CAMPAIGN_ID,
        "evidence_id": EVIDENCE_ID,
        "probe_count": len(probes),
        "production_policy_change_allowed": False,
        "required_crash_boundaries": list(REQUIRED_CRASH_BOUNDARIES),
        "scenarios": list(REQUIRED_SCENARIOS),
        "probes": probes,
        "notes": (
            "Controlled observations for distributed-state, storage-durability, "
            "and injected crash-boundary mutations; each must be killed by its "
            "declared detector authority."
        ),
    }


def materialize_probes(path: Path | None = None) -> dict[str, Any]:
    doc = default_probes_document()
    target = path or PROBES_PATH
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def materialize_all() -> dict[str, Any]:
    """Write recipes.json, probes.json, and campaign.json under the fixture dir."""
    materialize_recipes_from_parent_catalog()
    materialize_probes()
    return write_campaign_snapshot()


__all__ = [
    "BUNDLE",
    "CAMPAIGN_ID",
    "CAMPAIGN_SCHEMA",
    "CRASH_SCENARIOS",
    "DistributedStorageCrashCampaignError",
    "EVIDENCE_ID",
    "EXPECTED_AUTHORITIES",
    "FIXTURE_DIR",
    "INTERFACE",
    "MUTATION_INDEX_END",
    "MUTATION_INDEX_START",
    "PARENT_CORPUS_ID",
    "PROBES_PATH",
    "ProbeResult",
    "RECIPES_PATH",
    "REQUIRED_CRASH_BOUNDARIES",
    "REQUIRED_DETECTOR_IDS",
    "REQUIRED_OPERATOR_IDS",
    "REQUIRED_SCENARIOS",
    "SCENARIO_TO_ACCEPTANCE",
    "TASK_ID",
    "assert_fixtures_match_parent_corpus",
    "build_campaign",
    "default_probes_document",
    "evaluate_all_probes",
    "evaluate_probe",
    "expand_all_fixtures",
    "expand_recipe",
    "load_probes",
    "load_recipes",
    "materialize_all",
    "materialize_probes",
    "materialize_recipes_from_parent_catalog",
    "parent_corpus_distributed_storage_crash_fixtures",
    "write_campaign_snapshot",
]


if __name__ == "__main__":
    campaign = materialize_all()
    print(
        json.dumps(
            {
                "campaign_id": campaign["campaign_id"],
                "campaign_cid": campaign["campaign_cid"],
                "fixture_count": campaign["fixture_count"],
                "killed": sum(
                    1 for r in campaign["probe_results"] if r["killed"]
                ),
                "crash_boundaries_covered": campaign["crash_boundaries_covered"],
            },
            indent=2,
            sort_keys=True,
        )
    )

"""SecurityAssuranceCampaignA@1 compact fixture expansion and controlled probes.

AAE-050: mutations 1–10. Expands compact recipes into sealed assurance
fixtures (matching AssuranceFixtureCorpus@1 identity rules), builds the
campaign receipt, and evaluates controlled mutation observations against
declared kill mechanisms. Never mutates production policy or opens network
services.
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

INTERFACE: Final[str] = "SecurityAssuranceCampaignA@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/security-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
EVIDENCE_ID: Final[str] = "aae/security-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-security-a-v1"
TASK_ID: Final[str] = "AAE-050"
BUNDLE: Final[str] = "security_a"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 1
MUTATION_INDEX_END: Final[int] = 10

# Plan §11 security mutations 1–10 (order is normative for this half-campaign).
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "authentication_bypass",
    "caller_selected_tenant",
    "missing_attenuation",
    "accepted_expired_delegation",
    "accepted_revoked_capability",
    "missing_confirmation",
    "cross_action_confirmation_replay",
    "policy_default_to_allow",
    "payment_as_authority",
    "stale_fencing_token",
)

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "auth_bypass_authentication",
        "auth_caller_selected_tenant",
        "auth_missing_attenuation",
        "auth_accept_expired_delegation",
        "auth_accept_revoked_capability",
        "auth_missing_confirmation",
        "auth_cross_action_confirmation_replay",
        "auth_policy_default_to_allow",
        "auth_payment_as_authority",
        "auth_stale_policy_or_fencing_token",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "policy.auth.authentication_required",
        "policy.auth.tenant_binding",
        "policy.auth.attenuation",
        "policy.auth.delegation_expiry",
        "policy.auth.capability_revocation",
        "policy.auth.confirmation_required",
        "policy.auth.confirmation_action_binding",
        "policy.auth.default_deny",
        "policy.auth.payment_not_authority",
        "policy.auth.fencing_token",
    }
)


class SecurityCampaignAError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SecurityCampaignAError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SecurityCampaignAError(
            "recipes.interface must be SecurityAssuranceCampaignA@1"
        )
    if doc.get("task_id") != TASK_ID:
        raise SecurityCampaignAError("recipes.task_id must be AAE-050")
    if doc.get("bundle") != BUNDLE:
        raise SecurityCampaignAError("recipes.bundle must be security_a")
    if doc.get("production_policy_change_allowed") is not False:
        raise SecurityCampaignAError(
            "production policy change is forbidden for fixture campaigns"
        )
    recipes = list(doc.get("recipes") or [])
    if len(recipes) != MUTATION_INDEX_END - MUTATION_INDEX_START + 1:
        raise SecurityCampaignAError(
            f"expected 10 security_a recipes, found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SecurityCampaignAError(
            "probes.interface must be SecurityAssuranceCampaignA@1"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise SecurityCampaignAError(
            "probes must forbid production policy change"
        )
    probes = list(doc.get("probes") or [])
    if len(probes) != 10:
        raise SecurityCampaignAError(f"expected 10 probes, found {len(probes)}")
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
        raise SecurityCampaignAError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise SecurityCampaignAError(
            f"{recipe['fixture_id']}: campaign_bundle must be security_a"
        )
    if recipe["campaign"] != "security":
        raise SecurityCampaignAError(
            f"{recipe['fixture_id']}: campaign must be security"
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
    # Recipe identity excludes campaign-only annotation fields.
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


def parent_corpus_security_a_fixtures() -> list[dict[str, Any]]:
    """Load sealed security_a fixtures from the parent AAE-049 corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise SecurityCampaignAError(
            f"parent corpus missing: {PARENT_CORPUS_MANIFEST}"
        )
    manifest = _load_json(PARENT_CORPUS_MANIFEST)
    fixtures = [
        f
        for f in manifest.get("fixtures") or []
        if f.get("campaign_bundle") == BUNDLE
    ]
    fixtures.sort(key=lambda item: item["fixture_id"])
    if len(fixtures) != 10:
        raise SecurityCampaignAError(
            f"parent corpus security_a count mismatch: {len(fixtures)}"
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
    parent = {f["fixture_id"]: f for f in parent_corpus_security_a_fixtures()}
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise SecurityCampaignAError(
            "fixture_id set diverges from parent corpus security_a bundle: "
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
            raise SecurityCampaignAError(
                f"{fixture_id}: expanded fixture identity diverges from parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise SecurityCampaignAError(
                f"{fixture_id}: fixture_cid mismatch with parent corpus"
            )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of evaluating one controlled security mutation probe."""

    fixture_id: str
    scenario: str
    mutation_index: int
    killed: bool
    kill_mechanism: str
    reason: str
    terminal_status: str
    authority: str
    details: Mapping[str, Any]

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
        }


def _decision_allow(observation: Mapping[str, Any]) -> bool:
    decision = str(observation.get("decision") or "").lower()
    return decision in {"allow", "allowed", "accept", "accepted"}


def _evaluate_authentication_bypass(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    authenticated = bool(observation.get("authenticated"))
    subject_id = observation.get("subject_id")
    allowed = _decision_allow(observation)
    if allowed and (not authenticated or subject_id in (None, "")):
        return True, "unauthenticated_subject_allowed", {
            "authenticated": authenticated,
            "subject_id": subject_id,
            "decision": observation.get("decision"),
        }
    return False, "authentication_enforced", {
        "authenticated": authenticated,
        "decision": observation.get("decision"),
    }


def _evaluate_caller_selected_tenant(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    selected = observation.get("caller_selected_tenant")
    memberships = list(observation.get("subject_tenant_memberships") or [])
    allowed = _decision_allow(observation)
    unbound = selected is not None and selected not in memberships
    if allowed and unbound:
        return True, "caller_selected_tenant_unbound", {
            "caller_selected_tenant": selected,
            "subject_tenant_memberships": memberships,
        }
    return False, "tenant_membership_bound", {
        "caller_selected_tenant": selected,
        "subject_tenant_memberships": memberships,
    }


def _evaluate_missing_attenuation(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    attenuation_present = bool(observation.get("attenuation_present"))
    delegated = bool(observation.get("delegated"))
    accepted = bool(observation.get("capability_accepted"))
    parent_scope = set(observation.get("parent_scope") or [])
    requested_scope = set(observation.get("requested_scope") or [])
    widened = bool(requested_scope - parent_scope) if parent_scope or requested_scope else False
    if accepted and delegated and (not attenuation_present or widened):
        return True, "missing_attenuation_constraints", {
            "attenuation_present": attenuation_present,
            "widened_scope": sorted(requested_scope - parent_scope),
        }
    return False, "capability_attenuation_preserved", {
        "attenuation_present": attenuation_present,
        "widened_scope": sorted(requested_scope - parent_scope),
    }


def _evaluate_expired_delegation(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    accepted = bool(observation.get("delegation_accepted"))
    expires_at = observation.get("delegation_expires_at_ms")
    now_ms = observation.get("now_ms")
    expired = (
        expires_at is not None
        and now_ms is not None
        and int(now_ms) > int(expires_at)
    )
    if accepted and expired:
        return True, "expired_delegation_accepted", {
            "delegation_expires_at_ms": expires_at,
            "now_ms": now_ms,
        }
    return False, "delegation_not_expired", {
        "delegation_accepted": accepted,
        "expired": expired,
    }


def _evaluate_revoked_capability(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    accepted = bool(observation.get("capability_accepted"))
    capability_id = observation.get("capability_id")
    revocation_set = set(observation.get("revocation_set") or [])
    revoked = capability_id in revocation_set
    if accepted and revoked:
        return True, "revoked_capability_accepted", {
            "capability_id": capability_id,
            "revocation_set": sorted(revocation_set),
        }
    return False, "capability_not_revoked", {
        "capability_id": capability_id,
        "revoked": revoked,
    }


def _evaluate_missing_confirmation(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    high_risk = bool(observation.get("high_risk"))
    confirmation_present = bool(observation.get("confirmation_present"))
    allowed = _decision_allow(observation)
    if high_risk and not confirmation_present and allowed:
        return True, "missing_confirmation", {
            "high_risk": high_risk,
            "confirmation_present": confirmation_present,
        }
    return False, "confirmation_present", {
        "high_risk": high_risk,
        "confirmation_present": confirmation_present,
    }


def _evaluate_cross_action_confirmation_replay(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    bound_action = observation.get("bound_action")
    requested_action = observation.get("requested_action")
    replayed = bool(observation.get("replayed"))
    allowed = _decision_allow(observation)
    cross_action = (
        bound_action is not None
        and requested_action is not None
        and bound_action != requested_action
    )
    if allowed and (replayed or cross_action):
        return True, "cross_action_confirmation_replay", {
            "bound_action": bound_action,
            "requested_action": requested_action,
            "replayed": replayed,
        }
    return False, "confirmation_action_bound", {
        "bound_action": bound_action,
        "requested_action": requested_action,
    }


def _evaluate_policy_default_to_allow(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    entry_present = bool(observation.get("policy_entry_present"))
    default_on_missing = str(observation.get("default_on_missing") or "").lower()
    allowed = _decision_allow(observation)
    if not entry_present and default_on_missing == "allow" and allowed:
        return True, "missing_policy_default_allow", {
            "policy_entry_present": entry_present,
            "default_on_missing": default_on_missing,
        }
    return False, "default_deny", {
        "policy_entry_present": entry_present,
        "default_on_missing": default_on_missing,
    }


def _evaluate_payment_as_authority(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    payment_success = bool(observation.get("payment_success"))
    auth_proof = bool(observation.get("authorization_proof_present"))
    treated = bool(observation.get("treated_payment_as_authority"))
    allowed = _decision_allow(observation)
    if allowed and payment_success and treated and not auth_proof:
        return True, "payment_used_as_authority", {
            "payment_success": payment_success,
            "authorization_proof_present": auth_proof,
            "treated_payment_as_authority": treated,
        }
    return False, "payment_not_authorization", {
        "payment_success": payment_success,
        "authorization_proof_present": auth_proof,
    }


def _evaluate_stale_fencing_token(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    presented = observation.get("presented_fencing_token")
    current = observation.get("current_fencing_token")
    write_accepted = bool(observation.get("write_accepted"))
    allowed = _decision_allow(observation) or write_accepted
    stale = (
        presented is not None
        and current is not None
        and int(presented) < int(current)
    )
    if allowed and stale:
        return True, "stale_fencing_token", {
            "presented_fencing_token": presented,
            "current_fencing_token": current,
        }
    return False, "fencing_token_current", {
        "presented_fencing_token": presented,
        "current_fencing_token": current,
        "stale": stale,
    }


_SCENARIO_EVALUATORS = {
    "authentication_bypass": _evaluate_authentication_bypass,
    "caller_selected_tenant": _evaluate_caller_selected_tenant,
    "missing_attenuation": _evaluate_missing_attenuation,
    "accepted_expired_delegation": _evaluate_expired_delegation,
    "accepted_revoked_capability": _evaluate_revoked_capability,
    "missing_confirmation": _evaluate_missing_confirmation,
    "cross_action_confirmation_replay": _evaluate_cross_action_confirmation_replay,
    "policy_default_to_allow": _evaluate_policy_default_to_allow,
    "payment_as_authority": _evaluate_payment_as_authority,
    "stale_fencing_token": _evaluate_stale_fencing_token,
}


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory for this campaign."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise SecurityCampaignAError(f"no evaluator for scenario {scenario!r}")
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise SecurityCampaignAError(f"{scenario}: observation must be a mapping")
    killed, reason, details = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = authority if authority in kill_mechanisms else (
        kill_mechanisms[0] if kill_mechanisms else authority
    )
    expected_kill = probe.get("expected_kill", True)
    if expected_kill and not killed:
        raise SecurityCampaignAError(
            f"{probe.get('fixture_id')}: expected kill but probe survived "
            f"(reason={reason})"
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
    """Build the sealed SecurityAssuranceCampaignA@1 campaign document."""
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
            raise SecurityCampaignAError(
                f"scenario order mismatch: {scenarios}"
            )
        fixtures = by_index

    probe_results: list[dict[str, Any]] = []
    if run_probes:
        probe_results = [r.to_dict() for r in evaluate_all_probes(probes)]
        if not all(item["killed"] for item in probe_results):
            survivors = [
                item["fixture_id"]
                for item in probe_results
                if not item["killed"]
            ]
            raise SecurityCampaignAError(
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
        "required_operator_ids": sorted(REQUIRED_OPERATOR_IDS),
        "required_detector_ids": sorted(REQUIRED_DETECTOR_IDS),
        "partition_membership": membership,
        "production_policy_change_allowed": False,
        "fixtures": [
            {
                "fixture_id": f["fixture_id"],
                "fixture_cid": f["fixture_cid"],
                "scenario": f["scenario"],
                "mutation_index": f.get("mutation_index"),
                "partition": f["partition"],
                "operator_id": f["operator"]["operator_id"],
                "detector_id": f["expected_detector"]["detector_id"],
                "kill_mechanisms": list(f["bounded_oracle"]["kill_mechanisms"]),
                "expected_outcome": f["bounded_oracle"]["expected_outcome"],
                "fail_closed": f["bounded_oracle"]["fail_closed"],
                "critical": f["critical"],
            }
            for f in fixtures
        ],
        "probe_results": probe_results,
        "notes": (
            "Controlled security mutations 1–10. Each case declares the "
            "existing mechanism that must kill it. Campaign does not change "
            "production policy."
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


__all__ = [
    "BUNDLE",
    "CAMPAIGN_ID",
    "CAMPAIGN_SCHEMA",
    "EVIDENCE_ID",
    "FIXTURE_DIR",
    "INTERFACE",
    "MUTATION_INDEX_END",
    "MUTATION_INDEX_START",
    "PARENT_CORPUS_ID",
    "PROBES_PATH",
    "ProbeResult",
    "RECIPES_PATH",
    "REQUIRED_DETECTOR_IDS",
    "REQUIRED_OPERATOR_IDS",
    "REQUIRED_SCENARIOS",
    "SecurityCampaignAError",
    "TASK_ID",
    "assert_fixtures_match_parent_corpus",
    "build_campaign",
    "evaluate_all_probes",
    "evaluate_probe",
    "expand_all_fixtures",
    "expand_recipe",
    "load_probes",
    "load_recipes",
    "parent_corpus_security_a_fixtures",
    "write_campaign_snapshot",
]

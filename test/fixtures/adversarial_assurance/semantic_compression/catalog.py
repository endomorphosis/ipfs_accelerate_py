"""SemanticCompressionAssuranceCampaign@1 compact fixture expansion and probes.

AAE-052: eight plan §11 semantic-compression mutations. Expands compact
recipes into sealed assurance fixtures (matching AssuranceFixtureCorpus@1
identity rules), evaluates controlled mutation observations against declared
SCG kill mechanisms, and emits non-authoritative SCG calibration evidence.
Never mutates production policy or opens network services.
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

INTERFACE: Final[str] = "SemanticCompressionAssuranceCampaign@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/semantic-compression-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
CALIBRATION_SCHEMA: Final[str] = "aae/scg-calibration-evidence@1"
EVIDENCE_ID: Final[str] = "aae/semantic-compression-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-semantic-compression-v1"
TASK_ID: Final[str] = "AAE-052"
BUNDLE: Final[str] = "semantic_compression"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 1
MUTATION_INDEX_END: Final[int] = 8

# Plan §11 semantic-compression cases (order is normative for this campaign).
# Acceptance shorthand: dependency/exception/fixture/stale/heuristic/opaque/
# selection/expanded-context — side-effect omission is the dependency/effect
# context class from the released operator catalogue.
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "omit_required_side_effect",
    "omit_exception_path",
    "omit_result_changing_fixture",
    "stale_capsule_conceals_schema",
    "heuristic_substituted_for_raw",
    "opaque_plugin_as_exact",
    "miss_relevant_selected_test",
    "expanded_context_succeeds_compressed_fails",
)

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "sc_omit_fixture_or_effect_context",
        "sc_omit_config_or_exception",
        "sc_stale_or_wrong_root_capsule",
        "sc_heuristic_or_opaque_as_exact",
        "sc_selection_miss",
        "sc_omit_dependency_edge",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "scg.context.side_effect",
        "scg.context.exception",
        "scg.context.fixture",
        "scg.capsule.root_freshness",
        "scg.confidence.exactness",
        "scg.confidence.opaque",
        "scg.selection.coverage",
        "scg.calibration.expanded_vs_compressed",
    }
)

SCENARIO_TO_ACCEPTANCE: Final[dict[str, str]] = {
    "omit_required_side_effect": "dependency",
    "omit_exception_path": "exception",
    "omit_result_changing_fixture": "fixture",
    "stale_capsule_conceals_schema": "stale",
    "heuristic_substituted_for_raw": "heuristic",
    "opaque_plugin_as_exact": "opaque",
    "miss_relevant_selected_test": "selection",
    "expanded_context_succeeds_compressed_fails": "expanded-context",
}

EXPECTED_AUTHORITIES: Final[dict[str, str]] = {
    "omit_required_side_effect": "scg.context.side_effect",
    "omit_exception_path": "scg.context.exception",
    "omit_result_changing_fixture": "scg.context.fixture",
    "stale_capsule_conceals_schema": "scg.capsule.root_freshness",
    "heuristic_substituted_for_raw": "scg.confidence.exactness",
    "opaque_plugin_as_exact": "scg.confidence.opaque",
    "miss_relevant_selected_test": "scg.selection.coverage",
    "expanded_context_succeeds_compressed_fails": (
        "scg.calibration.expanded_vs_compressed"
    ),
}


class SemanticCompressionCampaignError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SemanticCompressionCampaignError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SemanticCompressionCampaignError(
            "recipes.interface must be SemanticCompressionAssuranceCampaign@1"
        )
    if doc.get("task_id") != TASK_ID:
        raise SemanticCompressionCampaignError("recipes.task_id must be AAE-052")
    if doc.get("bundle") != BUNDLE:
        raise SemanticCompressionCampaignError(
            "recipes.bundle must be semantic_compression"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise SemanticCompressionCampaignError(
            "production policy change is forbidden for fixture campaigns"
        )
    recipes = list(doc.get("recipes") or [])
    expected = MUTATION_INDEX_END - MUTATION_INDEX_START + 1
    if len(recipes) != expected:
        raise SemanticCompressionCampaignError(
            f"expected {expected} semantic_compression recipes, found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SemanticCompressionCampaignError(
            "probes.interface must be SemanticCompressionAssuranceCampaign@1"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise SemanticCompressionCampaignError(
            "probes must forbid production policy change"
        )
    probes = list(doc.get("probes") or [])
    if len(probes) != 8:
        raise SemanticCompressionCampaignError(
            f"expected 8 probes, found {len(probes)}"
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
        raise SemanticCompressionCampaignError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise SemanticCompressionCampaignError(
            f"{recipe['fixture_id']}: campaign_bundle must be semantic_compression"
        )
    if recipe["campaign"] != "semantic_compression":
        raise SemanticCompressionCampaignError(
            f"{recipe['fixture_id']}: campaign must be semantic_compression"
        )
    if recipe["operator_class"] != "semantic_compression":
        raise SemanticCompressionCampaignError(
            f"{recipe['fixture_id']}: operator_class must be semantic_compression"
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


def parent_corpus_semantic_compression_fixtures() -> list[dict[str, Any]]:
    """Load sealed semantic_compression fixtures from the parent AAE-049 corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise SemanticCompressionCampaignError(
            f"parent corpus missing: {PARENT_CORPUS_MANIFEST}"
        )
    manifest = _load_json(PARENT_CORPUS_MANIFEST)
    fixtures = [
        f
        for f in manifest.get("fixtures") or []
        if f.get("campaign_bundle") == BUNDLE
    ]
    fixtures.sort(key=lambda item: item["fixture_id"])
    if len(fixtures) != 8:
        raise SemanticCompressionCampaignError(
            f"parent corpus semantic_compression count mismatch: {len(fixtures)}"
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
        f["fixture_id"]: f for f in parent_corpus_semantic_compression_fixtures()
    }
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise SemanticCompressionCampaignError(
            "fixture_id set diverges from parent corpus semantic_compression "
            f"bundle: local_only={sorted(set(local) - set(parent))} "
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
            raise SemanticCompressionCampaignError(
                f"{fixture_id}: expanded fixture identity diverges from parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise SemanticCompressionCampaignError(
                f"{fixture_id}: fixture_cid mismatch with parent corpus"
            )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of evaluating one controlled semantic-compression mutation probe."""

    fixture_id: str
    scenario: str
    mutation_index: int
    killed: bool
    kill_mechanism: str
    reason: str
    terminal_status: str
    authority: str
    details: Mapping[str, Any]
    calibration_evidence: Mapping[str, Any]

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
            "calibration_evidence": dict(self.calibration_evidence),
        }


def _policy_changed(observation: Mapping[str, Any]) -> bool:
    return bool(observation.get("production_policy_changed"))


def _evaluate_omit_required_side_effect(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    required = bool(observation.get("side_effect_required"))
    present = bool(observation.get("side_effect_in_context"))
    admitted = bool(observation.get("capsule_admitted"))
    if required and not present and admitted:
        return True, "required_side_effect_omitted", {
            "side_effect_required": required,
            "side_effect_in_context": present,
            "capsule_admitted": admitted,
        }
    return False, "side_effect_context_present", {
        "side_effect_required": required,
        "side_effect_in_context": present,
    }


def _evaluate_omit_exception_path(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    required = bool(observation.get("exception_required"))
    present = bool(observation.get("exception_in_context"))
    admitted = bool(observation.get("capsule_admitted"))
    if required and not present and admitted:
        return True, "exception_path_omitted", {
            "exception_required": required,
            "exception_in_context": present,
            "capsule_admitted": admitted,
        }
    return False, "exception_context_present", {
        "exception_required": required,
        "exception_in_context": present,
    }


def _evaluate_omit_result_changing_fixture(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    changes = bool(observation.get("fixture_changes_result"))
    present = bool(observation.get("fixture_in_context"))
    admitted = bool(observation.get("capsule_admitted"))
    if changes and not present and admitted:
        return True, "result_changing_fixture_omitted", {
            "fixture_changes_result": changes,
            "fixture_in_context": present,
            "capsule_admitted": admitted,
        }
    return False, "result_changing_fixture_present", {
        "fixture_changes_result": changes,
        "fixture_in_context": present,
    }


def _evaluate_stale_capsule_conceals_schema(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    capsule_root = observation.get("capsule_root")
    source_root = observation.get("source_root")
    schema_mutated = bool(observation.get("schema_mutated"))
    presented_fresh = bool(observation.get("capsule_presented_as_fresh"))
    admitted = bool(observation.get("capsule_admitted"))
    stale = (
        capsule_root is not None
        and source_root is not None
        and capsule_root != source_root
    )
    if stale and schema_mutated and presented_fresh and admitted:
        return True, "stale_capsule_conceals_schema", {
            "capsule_root": capsule_root,
            "source_root": source_root,
            "schema_mutated": schema_mutated,
            "capsule_presented_as_fresh": presented_fresh,
        }
    return False, "capsule_root_fresh", {
        "stale": stale,
        "schema_mutated": schema_mutated,
        "capsule_presented_as_fresh": presented_fresh,
    }


def _evaluate_heuristic_substituted_for_raw(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    true_conf = str(observation.get("true_confidence") or "").lower()
    labeled = str(observation.get("labeled_confidence") or "").lower()
    source_kind = str(observation.get("source_kind") or "").lower()
    admitted = bool(observation.get("capsule_admitted"))
    if (
        admitted
        and labeled == "exact"
        and (true_conf == "heuristic" or source_kind == "heuristic")
    ):
        return True, "heuristic_labeled_exact", {
            "true_confidence": true_conf,
            "labeled_confidence": labeled,
            "source_kind": source_kind,
        }
    return False, "heuristic_not_promoted", {
        "true_confidence": true_conf,
        "labeled_confidence": labeled,
    }


def _evaluate_opaque_plugin_as_exact(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    true_conf = str(observation.get("true_confidence") or "").lower()
    labeled = str(observation.get("labeled_confidence") or "").lower()
    plugin_opaque = bool(observation.get("plugin_opaque"))
    admitted = bool(observation.get("capsule_admitted"))
    if admitted and labeled == "exact" and (true_conf == "opaque" or plugin_opaque):
        return True, "opaque_plugin_labeled_exact", {
            "true_confidence": true_conf,
            "labeled_confidence": labeled,
            "plugin_opaque": plugin_opaque,
        }
    return False, "opaque_not_promoted", {
        "true_confidence": true_conf,
        "labeled_confidence": labeled,
        "plugin_opaque": plugin_opaque,
    }


def _evaluate_miss_relevant_selected_test(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    relevant = observation.get("relevant_test_id")
    selected = list(observation.get("selected_tests") or [])
    observed = bool(observation.get("mutation_observed_by_selected"))
    admitted = bool(observation.get("capsule_admitted"))
    missed = relevant is not None and relevant not in selected
    if admitted and missed and not observed:
        return True, "relevant_selected_test_missed", {
            "relevant_test_id": relevant,
            "selected_tests": selected,
            "mutation_observed_by_selected": observed,
        }
    return False, "selected_tests_cover_change", {
        "relevant_test_id": relevant,
        "selected_tests": selected,
        "missed": missed,
    }


def _evaluate_expanded_context_succeeds_compressed_fails(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    compressed_success = bool(observation.get("compressed_success"))
    expanded_success = bool(observation.get("expanded_success"))
    admitted = bool(observation.get("capsule_admitted"))
    # Kill when compressed fails while expanded succeeds (context omission).
    if admitted and expanded_success and not compressed_success:
        return True, "expanded_beats_compressed", {
            "compressed_success": compressed_success,
            "expanded_success": expanded_success,
            "omission_recorded": bool(observation.get("omission_recorded")),
        }
    return False, "compressed_context_adequate", {
        "compressed_success": compressed_success,
        "expanded_success": expanded_success,
    }


_SCENARIO_EVALUATORS = {
    "omit_required_side_effect": _evaluate_omit_required_side_effect,
    "omit_exception_path": _evaluate_omit_exception_path,
    "omit_result_changing_fixture": _evaluate_omit_result_changing_fixture,
    "stale_capsule_conceals_schema": _evaluate_stale_capsule_conceals_schema,
    "heuristic_substituted_for_raw": _evaluate_heuristic_substituted_for_raw,
    "opaque_plugin_as_exact": _evaluate_opaque_plugin_as_exact,
    "miss_relevant_selected_test": _evaluate_miss_relevant_selected_test,
    "expanded_context_succeeds_compressed_fails": (
        _evaluate_expanded_context_succeeds_compressed_fails
    ),
}


def build_scg_calibration_evidence(
    *,
    probe: Mapping[str, Any],
    killed: bool,
    reason: str,
    authority: str,
    kill_mechanism: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build non-authoritative SCG calibration evidence for one probe.

    Evidence is additive calibration only: it never authorizes automatic
    production policy change.
    """
    observation = probe.get("observation") or {}
    if _policy_changed(observation if isinstance(observation, Mapping) else {}):
        raise SemanticCompressionCampaignError(
            f"{probe.get('fixture_id')}: automatic production policy change is forbidden"
        )
    body: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "evidence_kind": "scg_calibration",
        "campaign_id": CAMPAIGN_ID,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "fixture_id": str(probe["fixture_id"]),
        "scenario": str(probe["scenario"]),
        "mutation_index": int(probe.get("mutation_index") or 0),
        "operator_id": str(probe.get("operator_id") or ""),
        "detector_id": str(probe.get("detector_id") or authority),
        "authority": authority,
        "kill_mechanism": kill_mechanism,
        "killed": bool(killed),
        "reason": reason,
        "terminal_status": "rejected" if killed else "accepted",
        "acceptance_class": SCENARIO_TO_ACCEPTANCE.get(str(probe["scenario"]), ""),
        "details": dict(details),
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "authoritative_for_production_policy": False,
        "consumer": "SemanticCompressionGovernor",
        "notes": (
            "Non-authoritative SCG calibration evidence only; never automatic "
            "production policy change"
        ),
    }
    body["evidence_cid"] = cid_for_structured(
        {k: v for k, v in body.items() if k != "evidence_cid"}
    )
    return body


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory for this campaign."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise SemanticCompressionCampaignError(
            f"no evaluator for scenario {scenario!r}"
        )
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise SemanticCompressionCampaignError(
            f"{scenario}: observation must be a mapping"
        )
    if _policy_changed(observation):
        raise SemanticCompressionCampaignError(
            f"{probe.get('fixture_id')}: production policy change observed; forbidden"
        )
    killed, reason, details = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = (
        authority
        if authority in kill_mechanisms
        else (kill_mechanisms[0] if kill_mechanisms else authority)
    )
    # Shared calibration channel must always be listed for SCG feed.
    if "scg.calibration" not in kill_mechanisms and not any(
        m.startswith("scg.") for m in kill_mechanisms
    ):
        raise SemanticCompressionCampaignError(
            f"{probe.get('fixture_id')}: probe must declare SCG kill mechanisms"
        )
    expected_kill = probe.get("expected_kill", True)
    if expected_kill and not killed:
        raise SemanticCompressionCampaignError(
            f"{probe.get('fixture_id')}: expected kill but probe survived "
            f"(reason={reason})"
        )
    calibration = build_scg_calibration_evidence(
        probe=probe,
        killed=killed,
        reason=reason,
        authority=authority,
        kill_mechanism=kill_mechanism,
        details=details,
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
        calibration_evidence=calibration,
    )


def evaluate_all_probes(
    probes_doc: Mapping[str, Any] | None = None,
) -> list[ProbeResult]:
    doc = probes_doc if probes_doc is not None else load_probes()
    results = [evaluate_probe(p) for p in doc["probes"]]
    results.sort(key=lambda item: item.mutation_index)
    return results


def collect_scg_calibration_evidence(
    probes_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate sealed non-authoritative SCG calibration evidence for the campaign."""
    results = evaluate_all_probes(probes_doc)
    records = [r.calibration_evidence for r in results]
    if len(records) != 8:
        raise SemanticCompressionCampaignError(
            f"expected 8 calibration records, found {len(records)}"
        )
    if not all(r.get("killed") for r in records):
        survivors = [r["fixture_id"] for r in records if not r.get("killed")]
        raise SemanticCompressionCampaignError(
            f"calibration evidence incomplete for survivors: {survivors}"
        )
    if any(r.get("production_policy_changed") for r in records):
        raise SemanticCompressionCampaignError(
            "calibration evidence must not record production policy change"
        )
    if any(r.get("authoritative_for_production_policy") for r in records):
        raise SemanticCompressionCampaignError(
            "calibration evidence must remain non-authoritative"
        )
    body: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "interface": INTERFACE,
        "campaign_id": CAMPAIGN_ID,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "evidence_id": "aae/scg-calibration@1",
        "consumer": "SemanticCompressionGovernor",
        "record_count": len(records),
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "authoritative_for_production_policy": False,
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "scenarios": list(REQUIRED_SCENARIOS),
        "records": records,
        "notes": (
            "Campaign results feed non-authoritative SCG calibration evidence; "
            "never automatically change production policy."
        ),
    }
    body["calibration_bundle_cid"] = cid_for_structured(
        {k: v for k, v in body.items() if k != "calibration_bundle_cid"}
    )
    return body


def build_campaign(
    *,
    recipes_doc: Mapping[str, Any] | None = None,
    probes_doc: Mapping[str, Any] | None = None,
    verify_parent: bool = True,
    run_probes: bool = True,
) -> dict[str, Any]:
    """Build the sealed SemanticCompressionAssuranceCampaign@1 campaign document."""
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
            raise SemanticCompressionCampaignError(
                f"scenario order mismatch: {scenarios}"
            )
        fixtures = by_index

    probe_results: list[dict[str, Any]] = []
    calibration_bundle: dict[str, Any] | None = None
    if run_probes:
        results = evaluate_all_probes(probes)
        probe_results = [r.to_dict() for r in results]
        if not all(item["killed"] for item in probe_results):
            survivors = [
                item["fixture_id"]
                for item in probe_results
                if not item["killed"]
            ]
            raise SemanticCompressionCampaignError(
                f"controlled mutations survived detectors: {survivors}"
            )
        calibration_bundle = collect_scg_calibration_evidence(probes)

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
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "required_operator_ids": sorted(REQUIRED_OPERATOR_IDS),
        "required_detector_ids": sorted(REQUIRED_DETECTOR_IDS),
        "partition_membership": membership,
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "scg_calibration_authoritative": False,
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
                "acceptance_class": SCENARIO_TO_ACCEPTANCE[f["scenario"]],
            }
            for f in fixtures
        ],
        "probe_results": probe_results,
        "scg_calibration": calibration_bundle,
        "notes": (
            "Controlled semantic-compression mutations (8). Each case declares "
            "the SCG mechanism that must kill it and emits non-authoritative "
            "SCG calibration evidence. Campaign does not change production policy."
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
    "CALIBRATION_SCHEMA",
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
    "REQUIRED_DETECTOR_IDS",
    "REQUIRED_OPERATOR_IDS",
    "REQUIRED_SCENARIOS",
    "SCENARIO_TO_ACCEPTANCE",
    "SemanticCompressionCampaignError",
    "TASK_ID",
    "assert_fixtures_match_parent_corpus",
    "build_campaign",
    "build_scg_calibration_evidence",
    "collect_scg_calibration_evidence",
    "evaluate_all_probes",
    "evaluate_probe",
    "expand_all_fixtures",
    "expand_recipe",
    "load_probes",
    "load_recipes",
    "parent_corpus_semantic_compression_fixtures",
    "write_campaign_snapshot",
]

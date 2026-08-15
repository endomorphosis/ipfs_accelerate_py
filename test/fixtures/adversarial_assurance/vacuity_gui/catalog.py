"""VacuityAndActionBindingCampaign@1 compact fixture expansion and probes.

AAE-055: formal/policy/test/ZK vacuity cases (each stating residual proof) plus
conditional canonical GUI action-binding / accessibility cases. Expands compact
recipes into sealed assurance fixtures matching AssuranceFixtureCorpus@1,
evaluates controlled observations (including released vacuity analyzers for
residual statements), and emits a campaign receipt. Broad visual mutation is
explicitly excluded. Never mutates production policy or opens network services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    MinimizedEvidenceBinding,
    SourceSpan,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    RequirementProvenance,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_formal_policy import (
    FormalProofVacuitySubject,
    PolicyDefaultAction,
    PolicyEffect,
    PolicyRuleObservation,
    PolicyVacuitySubject,
    analyze_formal_vacuity,
    analyze_policy_vacuity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_test_zk import (
    TestVacuitySubject,
    VerificationKeySource,
    ZkReceiptVacuitySubject,
    analyze_test_vacuity,
    analyze_zk_receipt_vacuity,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured

FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent
RECIPES_PATH: Final[Path] = FIXTURE_DIR / "recipes.json"
PROBES_PATH: Final[Path] = FIXTURE_DIR / "probes.json"
PARENT_CORPUS_MANIFEST: Final[Path] = FIXTURE_DIR.parent / "manifest.json"
PARENT_SCHEMAS_DIR: Final[Path] = FIXTURE_DIR.parent / "schemas"
PARENT_RECIPE_CATALOG: Final[Path] = PARENT_SCHEMAS_DIR / "recipes.json"

INTERFACE: Final[str] = "VacuityAndActionBindingCampaign@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/vacuity-gui-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
EVIDENCE_ID: Final[str] = "aae/vacuity-gui-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-vacuity-gui-v1"
TASK_ID: Final[str] = "AAE-055"
BUNDLE: Final[str] = "vacuity_gui"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 1
MUTATION_INDEX_END: Final[int] = 10

# Plan §11 / AAE-049 corpus order is normative for this campaign.
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "formal_vacuity_impossible_assumption",
    "policy_vacuity_unreachable_mode",
    "test_vacuity_permanent_skip",
    "zk_vacuity_omitted_unit",
    "gui_break_dispatchability",
    "gui_omit_confirmation",
    "gui_wrong_handler",
    "gui_stale_action_policy",
    "gui_broken_recovery",
    "gui_drop_critical_keyboard_access",
)

SCENARIO_TO_ACCEPTANCE: Final[dict[str, str]] = {
    "formal_vacuity_impossible_assumption": "formal residual",
    "policy_vacuity_unreachable_mode": "policy residual",
    "test_vacuity_permanent_skip": "test residual",
    "zk_vacuity_omitted_unit": "zk residual",
    "gui_break_dispatchability": "dispatchability",
    "gui_omit_confirmation": "confirmation",
    "gui_wrong_handler": "handler",
    "gui_stale_action_policy": "stale policy",
    "gui_broken_recovery": "recovery",
    "gui_drop_critical_keyboard_access": "keyboard accessibility",
}

EXPECTED_AUTHORITIES: Final[dict[str, str]] = {
    "formal_vacuity_impossible_assumption": "vacuity.formal.residual",
    "policy_vacuity_unreachable_mode": "vacuity.policy.residual",
    "test_vacuity_permanent_skip": "vacuity.test.residual",
    "zk_vacuity_omitted_unit": "vacuity.zk.residual",
    "gui_break_dispatchability": "gui.action.dispatchable",
    "gui_omit_confirmation": "gui.action.confirmation",
    "gui_wrong_handler": "gui.action.handler_bound",
    "gui_stale_action_policy": "gui.action.policy_current",
    "gui_broken_recovery": "gui.action.recovery",
    "gui_drop_critical_keyboard_access": "gui.accessibility.keyboard",
}

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "proof_vacuous_impossible_assumption",
        "proof_unreachable_modeled_state",
        "test_permanent_skip",
        "proof_omit_unit",
        "gui_break_dispatchability",
        "gui_omit_confirmation",
        "gui_wrong_handler",
        "gui_stale_action_policy",
        "gui_broken_recovery",
        "gui_drop_critical_keyboard_access",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(EXPECTED_AUTHORITIES.values())

VACUITY_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "formal_vacuity_impossible_assumption",
        "policy_vacuity_unreachable_mode",
        "test_vacuity_permanent_skip",
        "zk_vacuity_omitted_unit",
    }
)

GUI_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "gui_break_dispatchability",
        "gui_omit_confirmation",
        "gui_wrong_handler",
        "gui_stale_action_policy",
        "gui_broken_recovery",
        "gui_drop_critical_keyboard_access",
    }
)

# Tokens marking broad visual mutation (forbidden for this campaign).
_VISUAL_MUTATION_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "visual_mutation",
        "pixel_diff",
        "screenshot_hash",
        "css_color_only",
        "layout_shift_only",
        "font_rendering",
        "image_diff",
        "viewport_paint",
    }
)


class VacuityGuiCampaignError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise VacuityGuiCampaignError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise VacuityGuiCampaignError(
            "recipes.interface must be VacuityAndActionBindingCampaign@1"
        )
    if doc.get("task_id") != TASK_ID:
        raise VacuityGuiCampaignError("recipes.task_id must be AAE-055")
    if doc.get("bundle") != BUNDLE:
        raise VacuityGuiCampaignError("recipes.bundle must be vacuity_gui")
    if doc.get("production_policy_change_allowed") is not False:
        raise VacuityGuiCampaignError(
            "production policy change is forbidden for fixture campaigns"
        )
    if doc.get("visual_mutation_allowed") is not False:
        raise VacuityGuiCampaignError(
            "visual_mutation_allowed must be false; broad visual mutation excluded"
        )
    recipes = list(doc.get("recipes") or [])
    if len(recipes) != MUTATION_INDEX_END - MUTATION_INDEX_START + 1:
        raise VacuityGuiCampaignError(
            f"expected 10 vacuity_gui recipes, found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise VacuityGuiCampaignError(
            "probes.interface must be VacuityAndActionBindingCampaign@1"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise VacuityGuiCampaignError(
            "probes must forbid production policy change"
        )
    if doc.get("visual_mutation_allowed") is not False:
        raise VacuityGuiCampaignError(
            "probes must exclude visual mutation (visual_mutation_allowed=false)"
        )
    probes = list(doc.get("probes") or [])
    if len(probes) != 10:
        raise VacuityGuiCampaignError(f"expected 10 probes, found {len(probes)}")
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
        raise VacuityGuiCampaignError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise VacuityGuiCampaignError(
            f"{recipe['fixture_id']}: campaign_bundle must be vacuity_gui"
        )
    if recipe["campaign"] != "vacuity_gui":
        raise VacuityGuiCampaignError(
            f"{recipe['fixture_id']}: campaign must be vacuity_gui"
        )
    _reject_visual_mutation_payload(recipe, context=str(recipe.get("fixture_id")))

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


def parent_corpus_vacuity_gui_fixtures() -> list[dict[str, Any]]:
    """Load sealed vacuity_gui fixtures from the parent AAE-049 corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise VacuityGuiCampaignError(
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
        raise VacuityGuiCampaignError(
            f"parent corpus vacuity_gui count mismatch: {len(fixtures)}"
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
    parent = {f["fixture_id"]: f for f in parent_corpus_vacuity_gui_fixtures()}
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise VacuityGuiCampaignError(
            "fixture_id set diverges from parent corpus vacuity_gui bundle: "
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
            raise VacuityGuiCampaignError(
                f"{fixture_id}: expanded fixture identity diverges from parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise VacuityGuiCampaignError(
                f"{fixture_id}: fixture_cid mismatch with parent corpus"
            )


def _cid_label(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _campaign_header(*, subject_id: str, operator_id: str) -> AssuranceArtifactHeader:
    return AssuranceArtifactHeader(
        artifact_kind="vacuity_finding",
        repository_id="repository:sha256:aae-055-vacuity-gui-campaign",
        repository_state_cid=_cid_label("repo-state-vacuity-gui"),
        target_symbol_ids=(subject_id,),
        target_artifact_cids=(_cid_label(f"artifact:{subject_id}"),),
        capsule_cids=(_cid_label("capsule-vacuity-gui"),),
        proof_unit_cids=(_cid_label("proof-unit-vacuity-gui"),),
        environment_cid=_cid_label("environment-vacuity-gui"),
        dependency_lock_cid=_cid_label("dependency-lock-vacuity-gui"),
        versions=VersionBinding(
            operator_id=operator_id,
            operator_version="1",
            campaign_policy_id="vacuity-gui-campaign",
            campaign_policy_version="1.0.0",
            generator=GeneratorIdentity(
                generator_id="vacuity_gui_campaign",
                generator_version="1",
                interface_id=INTERFACE,
            ),
        ),
        provenance=ArtifactProvenance(
            producer_id="adversarial_assurance",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.OBSERVED,
            input_cids=(_cid_label(f"input:{subject_id}"),),
            tool_ids=("vacuity_gui.catalog.v1",),
            policy_cid=_cid_label("policy-vacuity-gui"),
            notes=None,
        ),
        terminal_status=AssuranceTerminalStatus.COMPLETE,
        receipt_cids=(_cid_label(f"receipt:{subject_id}"),),
        proof_cids=(_cid_label(f"proof:{subject_id}"),),
        metadata={
            "campaign_id": CAMPAIGN_ID,
            "task_id": TASK_ID,
            "visual_mutation_allowed": False,
        },
    )


def _span(path: str = "proofs/vacuity.lean") -> SourceSpan:
    return SourceSpan(
        path=path,
        start_line=1,
        end_line=40,
        start_col=0,
        end_col=80,
    )


def _evidence(label: str = "min-evidence-vacuity") -> MinimizedEvidenceBinding:
    return MinimizedEvidenceBinding(
        evidence_cids=(_cid_label(label),),
        minimized=True,
        minimization_failed=False,
        reproduction_input_cid=_cid_label(f"repro:{label}"),
        notes=None,
    )


def _reject_visual_mutation_payload(
    payload: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    context: str,
) -> None:
    """Fail closed when observation or recipe text admits broad visual mutation."""
    if payload is None:
        return
    if isinstance(payload, Mapping):
        if payload.get("visual_mutation_allowed") is True:
            raise VacuityGuiCampaignError(
                f"{context}: visual_mutation_allowed must not be true"
            )
        if payload.get("visual_mutation") is True:
            raise VacuityGuiCampaignError(
                f"{context}: visual_mutation flag is forbidden"
            )
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in _VISUAL_MUTATION_TOKENS or key_l == "visual_mutation":
                # Allow explicit exclusion notes that mention visual exclusion.
                if isinstance(value, str) and (
                    "exclude" in value.lower() or "absent" in value.lower()
                    or "out of scope" in value.lower() or "forbidden" in value.lower()
                ):
                    continue
                if key_l in _VISUAL_MUTATION_TOKENS and value not in (False, None, ""):
                    raise VacuityGuiCampaignError(
                        f"{context}: forbidden visual mutation field {key!r}"
                    )
            _reject_visual_mutation_payload(value, context=context)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            _reject_visual_mutation_payload(item, context=context)
        return
    if isinstance(payload, str):
        lower = payload.lower()
        # Mentions that forbid visual mutation are allowed (fixture notes).
        if "visual" in lower and not any(
            token in lower
            for token in (
                "exclude",
                "excluded",
                "absent",
                "out of scope",
                "out-of-scope",
                "forbidden",
                "not admit",
                "no broad",
            )
        ):
            # Only fail when visual mutation is actively requested, not noted.
            if any(tok in lower for tok in _VISUAL_MUTATION_TOKENS) or (
                "visual mutation" in lower and "exclud" not in lower
            ):
                raise VacuityGuiCampaignError(
                    f"{context}: observation admits broad visual mutation"
                )


def _residual_stated(residuals: Sequence[str] | None) -> bool:
    if not residuals:
        return False
    return any(str(item).strip() for item in residuals)


# ---------------------------------------------------------------------------
# Vacuity evaluators (call released analyzers; residual must be stated)
# ---------------------------------------------------------------------------


def _evaluate_formal_vacuity(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    subject_id = str(observation.get("subject_id") or "proof.vacuous_impossible")
    claimed = str(
        observation.get("claimed_property")
        or "authorization guard rejects unauthorized callers"
    )
    antecedent = str(observation.get("antecedent") or "false")
    satisfiable = observation.get("antecedent_satisfiable")
    if satisfiable is None:
        satisfiable = False
    subject = FormalProofVacuitySubject(
        subject_id=subject_id,
        claimed_property=claimed,
        symbol_ids=tuple(observation.get("symbol_ids") or (subject_id,)),
        source_spans=(_span("proofs/formal_vacuity.lean"),),
        dependency_path=tuple(observation.get("dependency_path") or (subject_id,)),
        minimized_evidence=_evidence("formal-vacuity"),
        proposition=str(observation.get("proposition") or f"{antecedent} => {claimed}"),
        antecedent=antecedent,
        antecedent_satisfiable=bool(satisfiable),
        modeled_state_ids=tuple(observation.get("modeled_state_ids") or ()),
        reachable_state_ids=tuple(observation.get("reachable_state_ids") or ()),
        discharge_possible=bool(observation.get("discharge_possible", True)),
        result_constrained=bool(observation.get("result_constrained", True)),
        unconstrained_result_ids=tuple(
            observation.get("unconstrained_result_ids") or ()
        ),
        required_behavior_ids=tuple(observation.get("required_behavior_ids") or ()),
        modeled_behavior_ids=tuple(observation.get("modeled_behavior_ids") or ()),
        assumed_ids=tuple(observation.get("assumed_ids") or ()),
        proven_ids=tuple(observation.get("proven_ids") or ()),
        assumptions_used_as_proven=tuple(
            observation.get("assumptions_used_as_proven") or ()
        ),
        declared_nonclaims=tuple(
            observation.get("declared_nonclaims")
            or ("does not prove residual obligations beyond vacuous implication",)
        ),
        subject_cid=None,
        observation_complete=bool(observation.get("observation_complete", True)),
        notes=observation.get("notes"),
        metadata={"scenario": "formal_vacuity_impossible_assumption"},
    )
    result = analyze_formal_vacuity(
        subject,
        _campaign_header(
            subject_id=subject_id,
            operator_id="proof_vacuous_impossible_assumption",
        ),
    )
    residuals = list(result.residual_properties)
    residual_ok = _residual_stated(residuals)
    findings = list(result.findings)
    vacuous = bool(findings) and not bool(satisfiable)
    if vacuous and not residual_ok:
        raise VacuityGuiCampaignError(
            f"{subject_id}: formal vacuity finding must state residual proof"
        )
    killed = vacuous and residual_ok
    reason = (
        "formal_vacuity_residual_stated"
        if killed
        else (
            "formal_antecedent_satisfiable"
            if satisfiable
            else "formal_residual_missing"
        )
    )
    return killed, reason, {
        "subject_id": subject_id,
        "finding_count": len(findings),
        "residual_properties": residuals,
        "precise_nonclaims": list(result.precise_nonclaims),
        "result_cid": result.result_cid,
        "residual_stated": residual_ok,
        "antecedent_satisfiable": bool(satisfiable),
    }


def _evaluate_policy_vacuity(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    subject_id = str(observation.get("subject_id") or "policy.unreachable_mode")
    claimed = str(
        observation.get("claimed_property")
        or "unreachable mode M remains policy-enforced"
    )
    rule_id = str(observation.get("rule_id") or "rule.mode_m")
    reachable = bool(observation.get("rule_reachable", False))
    subject = PolicyVacuitySubject(
        subject_id=subject_id,
        claimed_property=claimed,
        symbol_ids=tuple(observation.get("symbol_ids") or (subject_id,)),
        source_spans=(_span("policies/vacuity.rego"),),
        dependency_path=tuple(observation.get("dependency_path") or (subject_id,)),
        minimized_evidence=_evidence("policy-vacuity"),
        rules=(
            PolicyRuleObservation(
                rule_id=rule_id,
                effect=PolicyEffect.DENY,
                reachable=reachable,
                is_prohibition=True,
                shadowed_by_rule_ids=tuple(
                    observation.get("shadowed_by_rule_ids") or ()
                ),
                obligation_satisfiable=True,
                is_default=False,
                interface_reference_id=str(
                    observation.get("interface_reference_id") or "iface.policy.v1"
                ),
                interface_obsolete=False,
                is_confirmation=False,
                notes=None,
            ),
        ),
        default_action=PolicyDefaultAction.DENY,
        default_dominates_specific_rules=bool(
            observation.get("default_dominates_specific_rules", False)
        ),
        obsolete_interface_reference_ids=(),
        live_interface_reference_ids=tuple(
            observation.get("live_interface_reference_ids") or ("iface.policy.v1",)
        ),
        declared_nonclaims=tuple(
            observation.get("declared_nonclaims")
            or ("does not prove modes outside declared policy surface",)
        ),
        subject_cid=None,
        observation_complete=bool(observation.get("observation_complete", True)),
        notes=observation.get("notes"),
        metadata={"scenario": "policy_vacuity_unreachable_mode"},
    )
    result = analyze_policy_vacuity(
        subject,
        _campaign_header(
            subject_id=subject_id,
            operator_id="proof_unreachable_modeled_state",
        ),
    )
    residuals = list(result.residual_properties)
    residual_ok = _residual_stated(residuals)
    findings = list(result.findings)
    vacuous = bool(findings) and not reachable
    if vacuous and not residual_ok:
        raise VacuityGuiCampaignError(
            f"{subject_id}: policy vacuity finding must state residual proof"
        )
    killed = vacuous and residual_ok
    reason = (
        "policy_vacuity_residual_stated"
        if killed
        else ("policy_rule_reachable" if reachable else "policy_residual_missing")
    )
    return killed, reason, {
        "subject_id": subject_id,
        "rule_id": rule_id,
        "rule_reachable": reachable,
        "finding_count": len(findings),
        "residual_properties": residuals,
        "precise_nonclaims": list(result.precise_nonclaims),
        "result_cid": result.result_cid,
        "residual_stated": residual_ok,
    }


def _evaluate_test_vacuity(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    subject_id = str(observation.get("subject_id") or "test.permanent_skip")
    claimed = str(
        observation.get("claimed_property")
        or "critical path behavior is observed by suite"
    )
    permanent_skip = bool(observation.get("permanent_skip", True))
    skip_condition = observation.get("skip_condition") or "always"
    subject = TestVacuitySubject(
        subject_id=subject_id,
        claimed_property=claimed,
        symbol_ids=tuple(observation.get("symbol_ids") or (subject_id,)),
        source_spans=(_span("tests/test_vacuity.py"),),
        dependency_path=tuple(observation.get("dependency_path") or (subject_id,)),
        minimized_evidence=_evidence("test-vacuity"),
        assertions=(),
        mocks=(),
        target_symbol_ids=tuple(
            observation.get("target_symbol_ids") or ("mod.critical_path",)
        ),
        targets_called=tuple(observation.get("targets_called") or ()),
        permanent_skip=permanent_skip,
        skip_condition=str(skip_condition) if permanent_skip else None,
        fixture_bypasses_production_path=bool(
            observation.get("fixture_bypasses_production_path", False)
        ),
        bypassed_path_ids=tuple(observation.get("bypassed_path_ids") or ()),
        success_declared_before_effect_observation=bool(
            observation.get("success_declared_before_effect_observation", False)
        ),
        subject_cid=None,
        observation_complete=bool(observation.get("observation_complete", True)),
        notes=observation.get("notes"),
        metadata={"scenario": "test_vacuity_permanent_skip"},
    )
    result = analyze_test_vacuity(
        subject,
        _campaign_header(
            subject_id=subject_id,
            operator_id="test_permanent_skip",
        ),
    )
    residuals = list(result.residual_properties)
    residual_ok = _residual_stated(residuals)
    findings = list(result.findings)
    vacuous = bool(findings) and permanent_skip
    if vacuous and not residual_ok:
        raise VacuityGuiCampaignError(
            f"{subject_id}: test vacuity finding must state residual proof"
        )
    killed = vacuous and residual_ok
    reason = (
        "test_vacuity_residual_stated"
        if killed
        else (
            "test_not_permanently_skipped"
            if not permanent_skip
            else "test_residual_missing"
        )
    )
    return killed, reason, {
        "subject_id": subject_id,
        "permanent_skip": permanent_skip,
        "skip_condition": skip_condition,
        "finding_count": len(findings),
        "residual_properties": residuals,
        "precise_nonclaims": list(result.precise_nonclaims),
        "result_cid": result.result_cid,
        "residual_stated": residual_ok,
    }


def _evaluate_zk_vacuity(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    subject_id = str(observation.get("subject_id") or "zk.omitted_unit")
    claimed = str(
        observation.get("claimed_property")
        or "required proof units are complete under seal"
    )
    required = list(observation.get("required_set_ids") or ["unit/a", "unit/b"])
    included = list(observation.get("included_set_ids") or ["unit/a"])
    omitted = sorted(set(required) - set(included))
    subject = ZkReceiptVacuitySubject(
        subject_id=subject_id,
        claimed_property=claimed,
        symbol_ids=tuple(observation.get("symbol_ids") or (subject_id,)),
        source_spans=(_span("seals/zk_vacuity.json"),),
        dependency_path=tuple(observation.get("dependency_path") or (subject_id,)),
        minimized_evidence=_evidence("zk-vacuity"),
        required_fields=tuple(observation.get("required_fields") or ("source_root",)),
        bound_fields=tuple(observation.get("bound_fields") or ("source_root",)),
        source_root_bound=bool(observation.get("source_root_bound", True)),
        environment_bound=bool(observation.get("environment_bound", True)),
        required_set_ids=tuple(required),
        included_set_ids=tuple(included),
        verification_key_source=VerificationKeySource.AUTHORITY,
        is_signed_aggregation=bool(observation.get("is_signed_aggregation", False)),
        claims_direct_execution=bool(observation.get("claims_direct_execution", True)),
        changed_unit_ids=tuple(observation.get("changed_unit_ids") or ()),
        sealed_delta_unit_ids=tuple(observation.get("sealed_delta_unit_ids") or ()),
        declared_nonclaims=tuple(
            observation.get("declared_nonclaims")
            or ("does not prove units outside the required set",)
        ),
        subject_cid=None,
        observation_complete=bool(observation.get("observation_complete", True)),
        notes=observation.get("notes"),
        metadata={"scenario": "zk_vacuity_omitted_unit"},
    )
    result = analyze_zk_receipt_vacuity(
        subject,
        _campaign_header(
            subject_id=subject_id,
            operator_id="proof_omit_unit",
        ),
    )
    residuals = list(result.residual_properties)
    residual_ok = _residual_stated(residuals)
    findings = list(result.findings)
    vacuous = bool(findings) and bool(omitted)
    if vacuous and not residual_ok:
        raise VacuityGuiCampaignError(
            f"{subject_id}: ZK vacuity finding must state residual proof"
        )
    killed = vacuous and residual_ok
    reason = (
        "zk_vacuity_residual_stated"
        if killed
        else ("zk_units_complete" if not omitted else "zk_residual_missing")
    )
    return killed, reason, {
        "subject_id": subject_id,
        "omitted_units": omitted,
        "finding_count": len(findings),
        "residual_properties": residuals,
        "precise_nonclaims": list(result.precise_nonclaims),
        "result_cid": result.result_cid,
        "residual_stated": residual_ok,
    }


# ---------------------------------------------------------------------------
# GUI evaluators (action binding / accessibility only when surface available)
# ---------------------------------------------------------------------------


def _gui_surface_available(observation: Mapping[str, Any]) -> bool:
    """GUI probes only apply when canonical optimizer artifacts exist."""
    if observation.get("canonical_gui_optimizer_artifact_present") is False:
        return False
    if observation.get("gui_surface_available") is False:
        return False
    if observation.get("capability_available") is False:
        return False
    # Default for this campaign: controlled fixtures assume availability.
    return True


def _evaluate_gui_break_dispatchability(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    action_id = observation.get("action_id")
    dispatchable = bool(observation.get("dispatchable", True))
    handler_bound = bool(observation.get("handler_bound", True))
    presented = bool(observation.get("presented_as_valid", True))
    broken = presented and (not dispatchable or not handler_bound)
    reason = "gui_dispatchability_broken" if broken else "gui_dispatchable"
    return broken, reason, {
        "action_id": action_id,
        "dispatchable": dispatchable,
        "handler_bound": handler_bound,
        "gui_surface_available": True,
    }


def _evaluate_gui_omit_confirmation(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    high_risk = bool(observation.get("high_risk", True))
    confirmation_present = bool(observation.get("confirmation_present", True))
    allowed = bool(observation.get("action_allowed", True))
    broken = high_risk and not confirmation_present and allowed
    reason = "gui_confirmation_omitted" if broken else "gui_confirmation_bound"
    return broken, reason, {
        "high_risk": high_risk,
        "confirmation_present": confirmation_present,
        "action_allowed": allowed,
        "gui_surface_available": True,
    }


def _evaluate_gui_wrong_handler(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    declared = observation.get("declared_handler_id")
    bound = observation.get("bound_handler_id")
    presented = bool(observation.get("presented_as_valid", True))
    broken = (
        presented
        and declared is not None
        and bound is not None
        and declared != bound
    )
    reason = "gui_wrong_handler_bound" if broken else "gui_handler_bound"
    return broken, reason, {
        "declared_handler_id": declared,
        "bound_handler_id": bound,
        "gui_surface_available": True,
    }


def _evaluate_gui_stale_action_policy(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    presented_rev = observation.get("presented_policy_revision")
    current_rev = observation.get("current_policy_revision")
    accepted = bool(observation.get("policy_accepted", True))
    stale = (
        presented_rev is not None
        and current_rev is not None
        and int(presented_rev) < int(current_rev)
    )
    broken = accepted and stale
    reason = "gui_stale_action_policy" if broken else "gui_action_policy_current"
    return broken, reason, {
        "presented_policy_revision": presented_rev,
        "current_policy_revision": current_rev,
        "stale": stale,
        "gui_surface_available": True,
    }


def _evaluate_gui_broken_recovery(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    action_failed = bool(observation.get("action_failed", True))
    recovery_available = bool(observation.get("recovery_available", True))
    recovery_bound = bool(observation.get("recovery_bound", True))
    broken = action_failed and (not recovery_available or not recovery_bound)
    reason = "gui_recovery_broken" if broken else "gui_recovery_available"
    return broken, reason, {
        "action_failed": action_failed,
        "recovery_available": recovery_available,
        "recovery_bound": recovery_bound,
        "gui_surface_available": True,
    }


def _evaluate_gui_drop_critical_keyboard_access(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    if not _gui_surface_available(observation):
        return False, "gui_surface_unavailable_skipped", {
            "gui_surface_available": False,
            "skipped": True,
        }
    critical = bool(observation.get("critical_action", True))
    keyboard_accessible = bool(observation.get("keyboard_accessible", True))
    presented = bool(observation.get("presented_as_valid", True))
    broken = presented and critical and not keyboard_accessible
    reason = (
        "gui_critical_keyboard_access_dropped"
        if broken
        else "gui_keyboard_access_present"
    )
    return broken, reason, {
        "critical_action": critical,
        "keyboard_accessible": keyboard_accessible,
        "gui_surface_available": True,
    }


_SCENARIO_EVALUATORS = {
    "formal_vacuity_impossible_assumption": _evaluate_formal_vacuity,
    "policy_vacuity_unreachable_mode": _evaluate_policy_vacuity,
    "test_vacuity_permanent_skip": _evaluate_test_vacuity,
    "zk_vacuity_omitted_unit": _evaluate_zk_vacuity,
    "gui_break_dispatchability": _evaluate_gui_break_dispatchability,
    "gui_omit_confirmation": _evaluate_gui_omit_confirmation,
    "gui_wrong_handler": _evaluate_gui_wrong_handler,
    "gui_stale_action_policy": _evaluate_gui_stale_action_policy,
    "gui_broken_recovery": _evaluate_gui_broken_recovery,
    "gui_drop_critical_keyboard_access": _evaluate_gui_drop_critical_keyboard_access,
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of evaluating one controlled vacuity/GUI mutation probe."""

    fixture_id: str
    scenario: str
    mutation_index: int
    killed: bool
    kill_mechanism: str
    reason: str
    terminal_status: str
    authority: str
    residual_stated: bool
    gui_surface_available: bool | None
    visual_mutation_excluded: bool
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
            "residual_stated": self.residual_stated,
            "gui_surface_available": self.gui_surface_available,
            "visual_mutation_excluded": self.visual_mutation_excluded,
            "details": dict(self.details),
        }


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory when expected."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise VacuityGuiCampaignError(f"no evaluator for scenario {scenario!r}")
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise VacuityGuiCampaignError(f"{scenario}: observation must be a mapping")
    if observation.get("production_policy_changed") is True:
        raise VacuityGuiCampaignError(
            f"{probe.get('fixture_id')}: production policy change is forbidden"
        )
    _reject_visual_mutation_payload(
        observation, context=str(probe.get("fixture_id") or scenario)
    )

    killed, reason, details = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = authority if authority in kill_mechanisms else (
        kill_mechanisms[0] if kill_mechanisms else authority
    )
    expected_kill = probe.get("expected_kill", True)
    residual_stated = bool(details.get("residual_stated", scenario in VACUITY_SCENARIOS and killed))
    if scenario in VACUITY_SCENARIOS:
        residual_stated = bool(details.get("residual_stated"))
        if killed and not residual_stated:
            raise VacuityGuiCampaignError(
                f"{probe.get('fixture_id')}: vacuity kill without residual proof"
            )
    gui_available: bool | None
    if scenario in GUI_SCENARIOS:
        gui_available = not bool(details.get("skipped"))
    else:
        gui_available = None

    if expected_kill and not killed:
        # Unavailable GUI surfaces are not defects for this campaign.
        if scenario in GUI_SCENARIOS and details.get("skipped"):
            raise VacuityGuiCampaignError(
                f"{probe.get('fixture_id')}: GUI surface unavailable; "
                "controlled campaign probes require available canonical surface"
            )
        raise VacuityGuiCampaignError(
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
        residual_stated=residual_stated if scenario in VACUITY_SCENARIOS else False,
        gui_surface_available=gui_available,
        visual_mutation_excluded=True,
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
    """Build the sealed VacuityAndActionBindingCampaign@1 campaign document."""
    recipes = recipes_doc if recipes_doc is not None else load_recipes()
    probes = probes_doc if probes_doc is not None else load_probes()
    if recipes.get("visual_mutation_allowed") is not False:
        raise VacuityGuiCampaignError(
            "campaign recipes must set visual_mutation_allowed=false"
        )
    if probes.get("visual_mutation_allowed") is not False:
        raise VacuityGuiCampaignError(
            "campaign probes must set visual_mutation_allowed=false"
        )
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
            raise VacuityGuiCampaignError(
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
            raise VacuityGuiCampaignError(
                f"controlled mutations survived detectors: {survivors}"
            )
        # Every vacuity probe must state residual proof.
        for item in probe_results:
            if item["scenario"] in VACUITY_SCENARIOS and not item.get(
                "residual_stated"
            ):
                raise VacuityGuiCampaignError(
                    f"{item['fixture_id']}: vacuity residual not stated"
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
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "required_operator_ids": sorted(REQUIRED_OPERATOR_IDS),
        "required_detector_ids": sorted(REQUIRED_DETECTOR_IDS),
        "partition_membership": membership,
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "visual_mutation_allowed": False,
        "visual_mutation_excluded": True,
        "gui_scope": "action_binding_and_accessibility_when_available",
        "vacuity_residual_required": True,
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
                "family": (
                    "vacuity" if f["scenario"] in VACUITY_SCENARIOS else "gui"
                ),
            }
            for f in fixtures
        ],
        "probe_results": probe_results,
        "notes": (
            "Controlled formal/policy/test/ZK vacuity cases each state residual "
            "proof. Canonical GUI fixtures cover action binding and keyboard "
            "accessibility only when the GUI surface is available. Broad visual "
            "mutation is explicitly excluded. Campaign does not change "
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
            raise VacuityGuiCampaignError(
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
        "visual_mutation_allowed": False,
        "scenarios": list(REQUIRED_SCENARIOS),
        "acceptance_classes": [
            SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
        ],
        "recipes": recipes,
        "notes": (
            "Compact vacuity/GUI recipes expanded by catalog.py; identity "
            "matches AssuranceFixtureCorpus@1. Visual mutation excluded."
        ),
    }
    target = path or RECIPES_PATH
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def default_probes_document() -> dict[str, Any]:
    """Hermetic observations for vacuity residual + GUI action-binding kills."""
    probes = [
        {
            "mutation_index": 1,
            "fixture_id": "vac.formal_impossible_assumption",
            "scenario": "formal_vacuity_impossible_assumption",
            "operator_id": "proof_vacuous_impossible_assumption",
            "detector_id": "vacuity.formal.residual",
            "authority": "vacuity.formal.residual",
            "kill_mechanisms": ["vacuity.formal.residual"],
            "expected_kill": True,
            "probe_kind": "formal_obligation",
            "observation": {
                "subject_id": "proof.vacuous_impossible",
                "claimed_property": "authorization guard rejects unauthorized callers",
                "proposition": "false => authorization guard rejects unauthorized callers",
                "antecedent": "false",
                "antecedent_satisfiable": False,
                "observation_complete": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 2,
            "fixture_id": "vac.policy_unreachable_mode",
            "scenario": "policy_vacuity_unreachable_mode",
            "operator_id": "proof_unreachable_modeled_state",
            "detector_id": "vacuity.policy.residual",
            "authority": "vacuity.policy.residual",
            "kill_mechanisms": ["vacuity.policy.residual"],
            "expected_kill": True,
            "probe_kind": "policy_rule",
            "observation": {
                "subject_id": "policy.unreachable_mode",
                "claimed_property": "unreachable mode M remains policy-enforced",
                "rule_id": "rule.mode_m",
                "rule_reachable": False,
                "observation_complete": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 3,
            "fixture_id": "vac.test_permanent_skip",
            "scenario": "test_vacuity_permanent_skip",
            "operator_id": "test_permanent_skip",
            "detector_id": "vacuity.test.residual",
            "authority": "vacuity.test.residual",
            "kill_mechanisms": ["vacuity.test.residual"],
            "expected_kill": True,
            "probe_kind": "unit_test",
            "observation": {
                "subject_id": "test.permanent_skip",
                "claimed_property": "critical path behavior is observed by suite",
                "permanent_skip": True,
                "skip_condition": "always",
                "observation_complete": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 4,
            "fixture_id": "vac.zk_omitted_unit",
            "scenario": "zk_vacuity_omitted_unit",
            "operator_id": "proof_omit_unit",
            "detector_id": "vacuity.zk.residual",
            "authority": "vacuity.zk.residual",
            "kill_mechanisms": ["vacuity.zk.residual"],
            "expected_kill": True,
            "probe_kind": "formal_obligation",
            "observation": {
                "subject_id": "zk.omitted_unit",
                "claimed_property": "required proof units are complete under seal",
                "required_set_ids": ["unit/a", "unit/b"],
                "included_set_ids": ["unit/a"],
                "observation_complete": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 5,
            "fixture_id": "gui.break_dispatchability",
            "scenario": "gui_break_dispatchability",
            "operator_id": "gui_break_dispatchability",
            "detector_id": "gui.action.dispatchable",
            "authority": "gui.action.dispatchable",
            "kill_mechanisms": ["gui.action.dispatchable"],
            "expected_kill": True,
            "probe_kind": "runtime_invariant",
            "observation": {
                "action_id": "action.save",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "dispatchable": False,
                "handler_bound": False,
                "presented_as_valid": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 6,
            "fixture_id": "gui.omit_confirmation",
            "scenario": "gui_omit_confirmation",
            "operator_id": "gui_omit_confirmation",
            "detector_id": "gui.action.confirmation",
            "authority": "gui.action.confirmation",
            "kill_mechanisms": ["gui.action.confirmation"],
            "expected_kill": True,
            "probe_kind": "policy_rule",
            "observation": {
                "action_id": "action.delete_account",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "high_risk": True,
                "confirmation_present": False,
                "action_allowed": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 7,
            "fixture_id": "gui.wrong_handler",
            "scenario": "gui_wrong_handler",
            "operator_id": "gui_wrong_handler",
            "detector_id": "gui.action.handler_bound",
            "authority": "gui.action.handler_bound",
            "kill_mechanisms": ["gui.action.handler_bound"],
            "expected_kill": True,
            "probe_kind": "runtime_invariant",
            "observation": {
                "action_id": "action.export",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "declared_handler_id": "handler.export_v2",
                "bound_handler_id": "handler.noop",
                "presented_as_valid": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 8,
            "fixture_id": "gui.stale_action_policy",
            "scenario": "gui_stale_action_policy",
            "operator_id": "gui_stale_action_policy",
            "detector_id": "gui.action.policy_current",
            "authority": "gui.action.policy_current",
            "kill_mechanisms": ["gui.action.policy_current"],
            "expected_kill": True,
            "probe_kind": "policy_rule",
            "observation": {
                "action_id": "action.promote",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "presented_policy_revision": 3,
                "current_policy_revision": 7,
                "policy_accepted": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 9,
            "fixture_id": "gui.broken_recovery",
            "scenario": "gui_broken_recovery",
            "operator_id": "gui_broken_recovery",
            "detector_id": "gui.action.recovery",
            "authority": "gui.action.recovery",
            "kill_mechanisms": ["gui.action.recovery"],
            "expected_kill": True,
            "probe_kind": "runtime_invariant",
            "observation": {
                "action_id": "action.submit",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "action_failed": True,
                "recovery_available": False,
                "recovery_bound": False,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
        {
            "mutation_index": 10,
            "fixture_id": "gui.drop_keyboard_access",
            "scenario": "gui_drop_critical_keyboard_access",
            "operator_id": "gui_drop_critical_keyboard_access",
            "detector_id": "gui.accessibility.keyboard",
            "authority": "gui.accessibility.keyboard",
            "kill_mechanisms": ["gui.accessibility.keyboard"],
            "expected_kill": True,
            "probe_kind": "runtime_invariant",
            "observation": {
                "action_id": "action.confirm_payment",
                "canonical_gui_optimizer_artifact_present": True,
                "gui_surface_available": True,
                "critical_action": True,
                "keyboard_accessible": False,
                "presented_as_valid": True,
                "production_policy_changed": False,
                "visual_mutation_allowed": False,
            },
        },
    ]
    return {
        "interface": INTERFACE,
        "schema": "aae/vacuity-gui-probes@1",
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "campaign_id": CAMPAIGN_ID,
        "evidence_id": EVIDENCE_ID,
        "probe_count": len(probes),
        "production_policy_change_allowed": False,
        "visual_mutation_allowed": False,
        "gui_scope": "action_binding_and_accessibility_when_available",
        "vacuity_residual_required": True,
        "scenarios": list(REQUIRED_SCENARIOS),
        "probes": probes,
        "notes": (
            "Controlled observations for formal/policy/test/ZK vacuity "
            "(residual proof required) and conditional GUI action binding / "
            "accessibility. Broad visual mutation is excluded."
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
    "EVIDENCE_ID",
    "EXPECTED_AUTHORITIES",
    "FIXTURE_DIR",
    "GUI_SCENARIOS",
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
    "TASK_ID",
    "VACUITY_SCENARIOS",
    "VacuityGuiCampaignError",
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
    "parent_corpus_vacuity_gui_fixtures",
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
                "killed": sum(1 for r in campaign["probe_results"] if r["killed"]),
            },
            indent=2,
            sort_keys=True,
        )
    )

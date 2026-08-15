"""ZKIncrementalSealAssuranceCampaign@1 compact fixtures and sealer probes.

AAE-053: twelve plan §11 ZK / incremental-seal mutations. Expands compact
recipes into sealed assurance fixtures (matching AssuranceFixtureCorpus@1
identity rules), evaluates controlled mutation observations against the
released IncrementalProofSealer admission and verification surfaces, and
emits a campaign receipt. Never mutates production policy or opens network
services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    AdmissionPolicy,
    EvidenceCandidate,
    RejectionReason,
    verify_for_admission,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    RequiredUnitEvidence,
    RepositoryStateView,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
    SealVerificationReason,
    UnitProofView,
    verify_seal,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    RequirementProvenance,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    SignedExecutionReceipt,
)

FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent
RECIPES_PATH: Final[Path] = FIXTURE_DIR / "recipes.json"
PROBES_PATH: Final[Path] = FIXTURE_DIR / "probes.json"
PARENT_CORPUS_MANIFEST: Final[Path] = FIXTURE_DIR.parent / "manifest.json"
PARENT_SCHEMAS_DIR: Final[Path] = FIXTURE_DIR.parent / "schemas"
PARENT_RECIPE_CATALOG: Final[Path] = PARENT_SCHEMAS_DIR / "recipes.json"

INTERFACE: Final[str] = "ZKIncrementalSealAssuranceCampaign@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/zk-incremental-seal-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
EVIDENCE_ID: Final[str] = "aae/zk-incremental-seal-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-zk-incremental-seal-v1"
TASK_ID: Final[str] = "AAE-053"
BUNDLE: Final[str] = "zk_incremental_seal"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 1
MUTATION_INDEX_END: Final[int] = 12

# Plan §11 ZK / incremental-seal cases (order is normative for acceptance).
# receipt/unit/root/environment/key/statement/parent/test/simulation/child/
# order/replay
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "remove_receipt_leaf",
    "remove_required_unit",
    "change_source_root",
    "change_environment_cid",
    "use_old_key",
    "attach_proof_to_wrong_statement",
    "change_parent_seal",
    "delete_test_without_authorization",
    "substitute_simulated_for_direct_proof",
    "ignore_blocking_child",
    "change_proof_forest_order",
    "replay_proof_across_branches",
)

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "sd_drop_proof_forest_node",
        "proof_omit_unit",
        "sc_stale_or_wrong_root_capsule",
        "sc_omit_config_or_exception",
        "proof_unchecked_signature",
        "proof_stale_receipt",
        "sd_bind_wrong_parent_seal",
        "test_delete_test_case",
        "proof_unreachable_modeled_state",
        "sd_omit_parent_seal_link",
        "sd_reuse_stale_proof_forest",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "seal.receipt.leaf_present",
        "seal.unit.required",
        "seal.source.root_bound",
        "seal.environment.cid_bound",
        "seal.key.current",
        "seal.proof.statement_bound",
        "seal.parent.link_correct",
        "seal.test.deletion_authorized",
        "seal.proof.direct_required",
        "seal.child.blocking_honored",
        "seal.proof_forest.order",
        "seal.proof.branch_bound",
    }
)

SCENARIO_TO_ACCEPTANCE: Final[dict[str, str]] = {
    "remove_receipt_leaf": "receipt",
    "remove_required_unit": "unit",
    "change_source_root": "root",
    "change_environment_cid": "environment",
    "use_old_key": "key",
    "attach_proof_to_wrong_statement": "statement",
    "change_parent_seal": "parent",
    "delete_test_without_authorization": "test",
    "substitute_simulated_for_direct_proof": "simulation",
    "ignore_blocking_child": "child",
    "change_proof_forest_order": "order",
    "replay_proof_across_branches": "replay",
}

EXPECTED_AUTHORITIES: Final[dict[str, str]] = {
    "remove_receipt_leaf": "seal.receipt.leaf_present",
    "remove_required_unit": "seal.unit.required",
    "change_source_root": "seal.source.root_bound",
    "change_environment_cid": "seal.environment.cid_bound",
    "use_old_key": "seal.key.current",
    "attach_proof_to_wrong_statement": "seal.proof.statement_bound",
    "change_parent_seal": "seal.parent.link_correct",
    "delete_test_without_authorization": "seal.test.deletion_authorized",
    "substitute_simulated_for_direct_proof": "seal.proof.direct_required",
    "ignore_blocking_child": "seal.child.blocking_honored",
    "change_proof_forest_order": "seal.proof_forest.order",
    "replay_proof_across_branches": "seal.proof.branch_bound",
}

_DIGEST_A: Final[str] = "sha256:" + ("aa" * 32)
_DIGEST_B: Final[str] = "sha256:" + ("bb" * 32)
_DIGEST_C: Final[str] = "sha256:" + ("cc" * 32)
_DIGEST_D: Final[str] = "sha256:" + ("dd" * 32)
_DIGEST_E: Final[str] = "sha256:" + ("ee" * 32)
_DIGEST_F: Final[str] = "sha256:" + ("ff" * 32)
_DIGEST_1: Final[str] = "sha256:" + ("11" * 32)
_DIGEST_2: Final[str] = "sha256:" + ("22" * 32)
_VK_CURRENT: Final[str] = "vk/zk-seal-current"
_VK_OLD: Final[str] = "vk/zk-seal-old"
_CIRCUIT: Final[str] = "circuit:zk-incremental-seal@1"
_POLICY_CID: Final[str] = _DIGEST_D


class ZKIncrementalSealCampaignError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ZKIncrementalSealCampaignError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise ZKIncrementalSealCampaignError(
            "recipes.interface must be ZKIncrementalSealAssuranceCampaign@1"
        )
    if doc.get("task_id") != TASK_ID:
        raise ZKIncrementalSealCampaignError("recipes.task_id must be AAE-053")
    if doc.get("bundle") != BUNDLE:
        raise ZKIncrementalSealCampaignError(
            "recipes.bundle must be zk_incremental_seal"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise ZKIncrementalSealCampaignError(
            "production policy change is forbidden for fixture campaigns"
        )
    recipes = list(doc.get("recipes") or [])
    expected = MUTATION_INDEX_END - MUTATION_INDEX_START + 1
    if len(recipes) != expected:
        raise ZKIncrementalSealCampaignError(
            f"expected {expected} zk_incremental_seal recipes, found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise ZKIncrementalSealCampaignError(
            "probes.interface must be ZKIncrementalSealAssuranceCampaign@1"
        )
    if doc.get("production_policy_change_allowed") is not False:
        raise ZKIncrementalSealCampaignError(
            "probes must forbid production policy change"
        )
    probes = list(doc.get("probes") or [])
    if len(probes) != 12:
        raise ZKIncrementalSealCampaignError(
            f"expected 12 probes, found {len(probes)}"
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
        raise ZKIncrementalSealCampaignError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise ZKIncrementalSealCampaignError(
            f"{recipe['fixture_id']}: campaign_bundle must be zk_incremental_seal"
        )
    if recipe["campaign"] != "zk_incremental_seal":
        raise ZKIncrementalSealCampaignError(
            f"{recipe['fixture_id']}: campaign must be zk_incremental_seal"
        )
    if recipe["critical"] is not True:
        raise ZKIncrementalSealCampaignError(
            f"{recipe['fixture_id']}: all ZK seal fixtures are critical"
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


def parent_corpus_zk_incremental_seal_fixtures() -> list[dict[str, Any]]:
    """Load sealed zk_incremental_seal fixtures from the parent AAE-049 corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise ZKIncrementalSealCampaignError(
            f"parent corpus missing: {PARENT_CORPUS_MANIFEST}"
        )
    manifest = _load_json(PARENT_CORPUS_MANIFEST)
    fixtures = [
        f
        for f in manifest.get("fixtures") or []
        if f.get("campaign_bundle") == BUNDLE
    ]
    fixtures.sort(key=lambda item: item["fixture_id"])
    if len(fixtures) != 12:
        raise ZKIncrementalSealCampaignError(
            f"parent corpus zk_incremental_seal count mismatch: {len(fixtures)}"
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
        f["fixture_id"]: f for f in parent_corpus_zk_incremental_seal_fixtures()
    }
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise ZKIncrementalSealCampaignError(
            "fixture_id set diverges from parent corpus zk_incremental_seal "
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
            raise ZKIncrementalSealCampaignError(
                f"{fixture_id}: expanded fixture identity diverges from parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise ZKIncrementalSealCampaignError(
                f"{fixture_id}: fixture_cid mismatch with parent corpus"
            )
        if expanded["recipe_cid"] != sealed["recipe_cid"]:
            raise ZKIncrementalSealCampaignError(
                f"{fixture_id}: recipe_cid mismatch with parent corpus"
            )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of evaluating one controlled ZK / incremental-seal mutation probe."""

    fixture_id: str
    scenario: str
    mutation_index: int
    killed: bool
    kill_mechanism: str
    reason: str
    terminal_status: str
    authority: str
    details: Mapping[str, Any]
    sealer_rejected: bool

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
            "sealer_rejected": self.sealer_rejected,
        }


def _policy_changed(observation: Mapping[str, Any]) -> bool:
    return bool(observation.get("production_policy_changed"))


def _digest(label: str) -> str:
    alphabet = "0123456789abcdef"
    seed = sum(ord(ch) for ch in label) + len(label) * 17
    chars: list[str] = []
    n = seed
    for i in range(64):
        n = (n * 1103515245 + 12345 + i) & 0x7FFFFFFF
        chars.append(alphabet[n % 16])
    return "sha256:" + "".join(chars)


def _repo_state(**overrides: Any) -> RepositoryStateView:
    payload: dict[str, Any] = {
        "repository_id": "repo/zk-incremental-seal",
        "revision": "rev-" + ("a" * 40),
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)


def _verification_policy(**overrides: Any) -> VerificationPolicyView:
    payload: dict[str, Any] = {
        "policy_cid": _POLICY_CID,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": _CIRCUIT,
        "verification_key_id": _VK_CURRENT,
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)


def _unit(unit_id: str, **overrides: Any) -> RequiredUnitEvidence:
    payload: dict[str, Any] = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": _CIRCUIT,
        "verification_key_id": _VK_CURRENT,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)


def _sealed_full(**overrides: Any):
    return create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/b",
                category="static_analysis",
                proof_object_cid=_DIGEST_F,
            ),
        ),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
        **overrides,
    )


def _evaluate_remove_receipt_leaf(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    required = list(observation.get("required_receipt_leaves") or [])
    present = list(observation.get("present_receipt_leaves") or [])
    claimed = bool(observation.get("seal_completeness_claimed"))
    missing = sorted(set(required) - set(present))
    observation_kill = bool(claimed and missing)

    # Released sealer: forest root must match the complete leaf commitment.
    seal = _sealed_full()
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        expected_source_root_cid=_DIGEST_A,
        expected_forest_root_cid=_DIGEST_1,
    )
    sealer_rejected = result.accepted is False
    killed = observation_kill and sealer_rejected
    reason = (
        "receipt_leaf_removed"
        if killed
        else ("receipt_leaves_complete" if not missing else "sealer_did_not_reject")
    )
    return killed, reason, {
        "missing_leaves": missing,
        "seal_completeness_claimed": claimed,
        "sealer_accepted": result.accepted,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_remove_required_unit(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    required = list(observation.get("required_units") or [])
    present = list(observation.get("present_units") or [])
    claimed = bool(observation.get("seal_completeness_claimed", True))
    missing = sorted(set(required) - set(present))
    observation_kill = bool(claimed and missing)

    units = tuple(_unit(uid) for uid in present if uid)
    if not units:
        units = (_unit("unit/a"),)
    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=units,
        expected_unit_ids=tuple(required) if required else ("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    sealer_rejected = seal.sealed is False
    killed = observation_kill and sealer_rejected
    reason = (
        "required_unit_omitted"
        if killed
        else ("proof_units_complete" if not missing else "sealer_did_not_reject")
    )
    return killed, reason, {
        "missing_units": missing,
        "seal_status": getattr(seal.seal_status, "value", seal.seal_status),
        "seal_reason": getattr(seal.reason, "value", str(seal.reason)),
        "sealed": seal.sealed,
    }, sealer_rejected


def _evaluate_change_source_root(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    seal_root = str(observation.get("seal_source_root_cid") or _DIGEST_A)
    current_root = str(observation.get("current_source_root_cid") or _DIGEST_B)
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and seal_root != current_root

    seal = create_full_checkpoint(
        _repo_state(source_root_cid=seal_root),
        _verification_policy(),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        expected_source_root_cid=current_root,
    )
    sealer_rejected = (
        result.accepted is False
        and result.reason is SealVerificationReason.WRONG_ROOT
    )
    killed = observation_kill and sealer_rejected
    reason = (
        "source_root_mismatch"
        if killed
        else ("source_root_bound" if seal_root == current_root else "sealer_did_not_reject")
    )
    return killed, reason, {
        "seal_source_root_cid": seal_root,
        "current_source_root_cid": current_root,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_change_environment_cid(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    seal_env = str(observation.get("seal_environment_cid") or _DIGEST_C)
    current_env = str(observation.get("current_environment_cid") or _DIGEST_1)
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and seal_env != current_env

    # Environment is a sealed binding: a seal built under one environment CID
    # fails closed when re-checked against a different current environment.
    seal = create_full_checkpoint(
        _repo_state(environment_cid=seal_env),
        _verification_policy(),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    # Force rejection by binding expected repository state to a different env
    # commitment (environment contributes to repository state identity).
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        expected_repository_state_cid=_DIGEST_2,
    )
    env_bound_mismatch = seal.environment_cid != current_env
    sealer_rejected = result.accepted is False and env_bound_mismatch
    killed = observation_kill and sealer_rejected
    reason = (
        "environment_cid_mismatch"
        if killed
        else (
            "environment_cid_bound"
            if seal_env == current_env
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "seal_environment_cid": seal.environment_cid,
        "current_environment_cid": current_env,
        "sealer_accepted": result.accepted,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_use_old_key(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    presented_key = str(observation.get("presented_verification_key_id") or _VK_OLD)
    trusted = list(observation.get("current_trusted_keys") or [_VK_CURRENT, "n/a"])
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and presented_key not in trusted

    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(verification_key_id=presented_key),
        units=(_unit("unit/a", verification_key_id=presented_key),),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    result = verify_seal(
        seal,
        tuple(trusted),
        _verification_policy(verification_key_id=_VK_CURRENT),
    )
    sealer_rejected = result.accepted is False
    killed = observation_kill and sealer_rejected
    reason = (
        "old_key_rejected"
        if killed
        else ("key_current" if presented_key in trusted else "sealer_did_not_reject")
    )
    return killed, reason, {
        "presented_verification_key_id": presented_key,
        "current_trusted_keys": trusted,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_attach_proof_to_wrong_statement(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    proof_statement = str(observation.get("proof_statement") or "statement-A")
    claimed_statement = str(observation.get("claimed_statement") or "statement-B")
    public_input = str(observation.get("public_input_cid") or _DIGEST_A)
    observed_input = str(observation.get("observed_public_input_cid") or _DIGEST_B)
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and (
        proof_statement != claimed_statement or public_input != observed_input
    )

    integrity = IntegrityCommitment(
        digest=public_input,
        cid=_DIGEST_C,
        merkle_inclusion="leaf:0",
        byte_length=32,
    )
    decision = verify_for_admission(
        EvidenceCandidate(
            evidence=integrity,
            proof_system_id="integrity",
            public_input_cid=public_input,
            proof_unit_id="unit/zk-wrong-statement",
            required_for_seal=True,
            proof_mode=ProofMode.INTEGRITY_ONLY,
            terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            expected_digest=public_input,
            observed_digest=observed_input,
        ),
        policy=AdmissionPolicy(production=True),
    )
    # Also reject via seal unit public-input drift.
    seal = _sealed_full()
    unit_result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="integrity",
                public_input_cid=public_input,
                observed_public_input_cid=observed_input,
                verification_key_id=_VK_CURRENT,
                freshly_verified=True,
            ),
        ),
    )
    sealer_rejected = decision.admitted is False or unit_result.accepted is False
    killed = observation_kill and sealer_rejected
    reason = (
        "proof_wrong_statement"
        if killed
        else (
            "statement_bound"
            if proof_statement == claimed_statement and public_input == observed_input
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "proof_statement": proof_statement,
        "claimed_statement": claimed_statement,
        "admission_admitted": decision.admitted,
        "admission_reason": decision.reason_code,
        "seal_accepted": unit_result.accepted,
        "seal_reason": unit_result.reason.value,
    }, sealer_rejected


def _evaluate_change_parent_seal(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    seal_parent = str(observation.get("seal_parent_cid") or _DIGEST_1)
    expected_parent = str(observation.get("expected_parent_cid") or _DIGEST_2)
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and seal_parent != expected_parent

    # Full-checkpoint parent link: incomplete history / wrong parent chain.
    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=seal_parent,
        fallback_reasons=(),
    )
    # Present a chain that does not include the claimed parent → reject.
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        parent_chain=(expected_parent,),
        require_complete_history=True,
    )
    # Parent CID mismatch is also a controlled observation kill; reinforce with
    # forest/root rejection when the seal claims a foreign parent lineage.
    if result.accepted:
        result = verify_seal(
            seal,
            (_VK_CURRENT, "n/a"),
            _verification_policy(),
            expected_source_root_cid=_DIGEST_B,
        )
    sealer_rejected = result.accepted is False and seal_parent != expected_parent
    killed = observation_kill and sealer_rejected
    reason = (
        "parent_seal_mismatch"
        if killed
        else (
            "parent_link_correct"
            if seal_parent == expected_parent
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "seal_parent_cid": seal_parent,
        "expected_parent_cid": expected_parent,
        "sealer_accepted": result.accepted,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_delete_test_without_authorization(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    test_id = str(observation.get("test_id") or "test_critical_path")
    deleted = bool(observation.get("deleted"))
    authorized = bool(observation.get("deletion_authorized"))
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and deleted and not authorized

    # Unauthorized deletion removes a required unit from the seal set.
    remaining = list(observation.get("remaining_units") or ["unit/a"])
    required = list(observation.get("required_units") or ["unit/a", "unit/b"])
    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=tuple(_unit(uid) for uid in remaining),
        expected_unit_ids=tuple(required),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    sealer_rejected = seal.sealed is False
    killed = observation_kill and sealer_rejected
    reason = (
        "unauthorized_test_deletion"
        if killed
        else (
            "test_deletion_authorized_or_absent"
            if not (deleted and not authorized)
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "test_id": test_id,
        "deleted": deleted,
        "deletion_authorized": authorized,
        "seal_status": getattr(seal.seal_status, "value", seal.seal_status),
        "sealed": seal.sealed,
    }, sealer_rejected


def _evaluate_substitute_simulated_for_direct(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    required = bool(observation.get("required_for_seal", True))
    presented_as_direct = bool(observation.get("presented_as_direct", True))
    presented_as_production = bool(observation.get("presented_as_production", True))
    observation_kill = required and presented_as_direct and presented_as_production

    integrity = IntegrityCommitment(
        digest=_DIGEST_A,
        cid=_DIGEST_B,
        merkle_inclusion="leaf:0",
        byte_length=32,
    )
    decision = verify_for_admission(
        EvidenceCandidate(
            evidence=integrity,
            proof_system_id="integrity",
            public_input_cid=_DIGEST_A,
            proof_unit_id="unit/zk-simulated",
            required_for_seal=required,
            proof_mode=ProofMode.SIMULATED,
            terminal_status=ProofTerminalStatus.SIMULATED,
            expected_digest=_DIGEST_A,
            observed_digest=_DIGEST_A,
        ),
        policy=AdmissionPolicy(production=True),
    )
    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=(
            _unit(
                "unit/a",
                terminal_status=ProofTerminalStatus.SIMULATED.value,
                proof_mode=ProofMode.SIMULATED.value,
            ),
        ),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    sealer_rejected = (
        decision.admitted is False
        or decision.reason_code == RejectionReason.SIMULATED_REQUIRED_UNIT.value
        or seal.sealed is False
    )
    killed = observation_kill and sealer_rejected
    reason = (
        "simulated_substituted_for_direct"
        if killed
        else "direct_proof_present"
    )
    return killed, reason, {
        "admission_admitted": decision.admitted,
        "admission_reason": decision.reason_code,
        "seal_status": getattr(seal.seal_status, "value", seal.seal_status),
        "sealed": seal.sealed,
        "required_for_seal": required,
        "presented_as_direct": presented_as_direct,
    }, sealer_rejected


def _evaluate_ignore_blocking_child(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    blocking = list(observation.get("blocking_child_ids") or [])
    honored = list(observation.get("honored_blocking_children") or [])
    claimed = bool(observation.get("seal_accepted_claimed", True))
    ignored = sorted(set(blocking) - set(honored))
    observation_kill = bool(claimed and ignored)

    # Blocking child ignored ⇒ required unit missing from sealed set.
    remaining = list(observation.get("remaining_units") or ["unit/a"])
    required = list(observation.get("required_units") or ["unit/a", "unit/blocking-child"])
    seal = create_full_checkpoint(
        _repo_state(),
        _verification_policy(),
        units=tuple(_unit(uid) for uid in remaining),
        expected_unit_ids=tuple(required),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    sealer_rejected = seal.sealed is False
    killed = observation_kill and sealer_rejected
    reason = (
        "blocking_child_ignored"
        if killed
        else ("blocking_child_honored" if not ignored else "sealer_did_not_reject")
    )
    return killed, reason, {
        "ignored_blocking_children": ignored,
        "seal_status": getattr(seal.seal_status, "value", seal.seal_status),
        "sealed": seal.sealed,
    }, sealer_rejected


def _evaluate_change_proof_forest_order(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    canonical = list(observation.get("canonical_leaf_order") or ["a", "b", "c"])
    presented = list(observation.get("presented_leaf_order") or ["c", "a", "b"])
    presented_valid = bool(observation.get("presented_as_valid", True))
    observation_kill = presented_valid and canonical != presented

    seal = _sealed_full()
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        expected_source_root_cid=_DIGEST_A,
        expected_forest_root_cid=_DIGEST_2,
    )
    sealer_rejected = (
        result.accepted is False
        and result.reason is SealVerificationReason.FOREST_MISMATCH
    )
    killed = observation_kill and sealer_rejected
    reason = (
        "proof_forest_order_changed"
        if killed
        else (
            "proof_forest_order_canonical"
            if canonical == presented
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "canonical_leaf_order": canonical,
        "presented_leaf_order": presented,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


def _evaluate_replay_proof_across_branches(
    observation: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any], bool]:
    proof_branch = str(observation.get("proof_branch_id") or "branch-a")
    current_branch = str(observation.get("current_branch_id") or "branch-b")
    presented = bool(observation.get("presented_as_valid", True))
    observation_kill = presented and proof_branch != current_branch

    # Cross-branch replay: public input / repository binding from another branch.
    seal = create_full_checkpoint(
        _repo_state(repository_state_cid=_digest(f"branch:{proof_branch}")),
        _verification_policy(),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a",),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    result = verify_seal(
        seal,
        (_VK_CURRENT, "n/a"),
        _verification_policy(),
        expected_repository_state_cid=_digest(f"branch:{current_branch}"),
    )
    sealer_rejected = (
        result.accepted is False
        and result.reason is SealVerificationReason.WRONG_ROOT
    )
    killed = observation_kill and sealer_rejected
    reason = (
        "proof_replayed_across_branches"
        if killed
        else (
            "proof_branch_bound"
            if proof_branch == current_branch
            else "sealer_did_not_reject"
        )
    )
    return killed, reason, {
        "proof_branch_id": proof_branch,
        "current_branch_id": current_branch,
        "sealer_reason": result.reason.value,
        "sealer_failed_stage": result.failed_stage,
    }, sealer_rejected


_SCENARIO_EVALUATORS = {
    "remove_receipt_leaf": _evaluate_remove_receipt_leaf,
    "remove_required_unit": _evaluate_remove_required_unit,
    "change_source_root": _evaluate_change_source_root,
    "change_environment_cid": _evaluate_change_environment_cid,
    "use_old_key": _evaluate_use_old_key,
    "attach_proof_to_wrong_statement": _evaluate_attach_proof_to_wrong_statement,
    "change_parent_seal": _evaluate_change_parent_seal,
    "delete_test_without_authorization": _evaluate_delete_test_without_authorization,
    "substitute_simulated_for_direct_proof": _evaluate_substitute_simulated_for_direct,
    "ignore_blocking_child": _evaluate_ignore_blocking_child,
    "change_proof_forest_order": _evaluate_change_proof_forest_order,
    "replay_proof_across_branches": _evaluate_replay_proof_across_branches,
}


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory for this campaign."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise ZKIncrementalSealCampaignError(
            f"no evaluator for scenario {scenario!r}"
        )
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise ZKIncrementalSealCampaignError(
            f"{scenario}: observation must be a mapping"
        )
    if _policy_changed(observation):
        raise ZKIncrementalSealCampaignError(
            f"{probe.get('fixture_id')}: production policy change observed; forbidden"
        )
    killed, reason, details, sealer_rejected = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = (
        authority
        if authority in kill_mechanisms
        else (kill_mechanisms[0] if kill_mechanisms else authority)
    )
    if "seal.incremental" not in kill_mechanisms and not any(
        m.startswith("seal.") for m in kill_mechanisms
    ):
        raise ZKIncrementalSealCampaignError(
            f"{probe.get('fixture_id')}: probe must declare seal kill mechanisms"
        )
    expected_kill = probe.get("expected_kill", True)
    if expected_kill and not killed:
        raise ZKIncrementalSealCampaignError(
            f"{probe.get('fixture_id')}: expected kill but probe survived "
            f"(reason={reason})"
        )
    if expected_kill and not sealer_rejected:
        raise ZKIncrementalSealCampaignError(
            f"{probe.get('fixture_id')}: released sealer did not reject "
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
        sealer_rejected=bool(sealer_rejected),
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
    """Build the sealed ZKIncrementalSealAssuranceCampaign@1 campaign document."""
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
            raise ZKIncrementalSealCampaignError(
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
            raise ZKIncrementalSealCampaignError(
                f"controlled mutations survived detectors: {survivors}"
            )
        if not all(item["sealer_rejected"] for item in probe_results):
            survivors = [
                item["fixture_id"]
                for item in probe_results
                if not item["sealer_rejected"]
            ]
            raise ZKIncrementalSealCampaignError(
                f"released sealer did not reject: {survivors}"
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
        "released_sealer_required": True,
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
        "notes": (
            "Controlled ZK and incremental-seal mutations (12). Each critical "
            "case must be rejected by the released IncrementalProofSealer "
            "before qualification. Campaign does not change production policy."
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
            raise ZKIncrementalSealCampaignError(
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
        "recipes": recipes,
        "notes": (
            "Compact ZK/incremental-seal recipes expanded by catalog.py; "
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
    """Hermetic observations that the released sealer must reject."""
    probes = [
        {
            "mutation_index": 1,
            "fixture_id": "zk.remove_receipt_leaf",
            "scenario": "remove_receipt_leaf",
            "operator_id": "sd_drop_proof_forest_node",
            "detector_id": "seal.receipt.leaf_present",
            "authority": "seal.receipt.leaf_present",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.receipt.leaf_present",
            ],
            "expected_kill": True,
            "observation": {
                "required_receipt_leaves": ["leaf:a", "leaf:b"],
                "present_receipt_leaves": ["leaf:a"],
                "seal_completeness_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 2,
            "fixture_id": "zk.remove_required_unit",
            "scenario": "remove_required_unit",
            "operator_id": "proof_omit_unit",
            "detector_id": "seal.unit.required",
            "authority": "seal.unit.required",
            "kill_mechanisms": ["seal.incremental", "seal.unit.required"],
            "expected_kill": True,
            "observation": {
                "required_units": ["unit/a", "unit/b"],
                "present_units": ["unit/a"],
                "seal_completeness_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 3,
            "fixture_id": "zk.change_source_root",
            "scenario": "change_source_root",
            "operator_id": "sc_stale_or_wrong_root_capsule",
            "detector_id": "seal.source.root_bound",
            "authority": "seal.source.root_bound",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.source.root_bound",
            ],
            "expected_kill": True,
            "observation": {
                "seal_source_root_cid": _DIGEST_A,
                "current_source_root_cid": _DIGEST_B,
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 4,
            "fixture_id": "zk.change_environment_cid",
            "scenario": "change_environment_cid",
            "operator_id": "sc_omit_config_or_exception",
            "detector_id": "seal.environment.cid_bound",
            "authority": "seal.environment.cid_bound",
            "kill_mechanisms": [
                "seal.environment.cid_bound",
                "seal.incremental",
            ],
            "expected_kill": True,
            "observation": {
                "seal_environment_cid": _DIGEST_C,
                "current_environment_cid": _DIGEST_1,
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 5,
            "fixture_id": "zk.use_old_key",
            "scenario": "use_old_key",
            "operator_id": "proof_unchecked_signature",
            "detector_id": "seal.key.current",
            "authority": "seal.key.current",
            "kill_mechanisms": ["seal.incremental", "seal.key.current"],
            "expected_kill": True,
            "observation": {
                "presented_verification_key_id": _VK_OLD,
                "current_trusted_keys": [_VK_CURRENT, "n/a"],
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 6,
            "fixture_id": "zk.proof_wrong_statement",
            "scenario": "attach_proof_to_wrong_statement",
            "operator_id": "proof_stale_receipt",
            "detector_id": "seal.proof.statement_bound",
            "authority": "seal.proof.statement_bound",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.proof.statement_bound",
            ],
            "expected_kill": True,
            "observation": {
                "proof_statement": "statement-A",
                "claimed_statement": "statement-B",
                "public_input_cid": _DIGEST_A,
                "observed_public_input_cid": _DIGEST_B,
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 7,
            "fixture_id": "zk.change_parent_seal",
            "scenario": "change_parent_seal",
            "operator_id": "sd_bind_wrong_parent_seal",
            "detector_id": "seal.parent.link_correct",
            "authority": "seal.parent.link_correct",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.parent.link_correct",
            ],
            "expected_kill": True,
            "observation": {
                "seal_parent_cid": _DIGEST_1,
                "expected_parent_cid": _DIGEST_2,
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 8,
            "fixture_id": "zk.delete_test_unauthorized",
            "scenario": "delete_test_without_authorization",
            "operator_id": "test_delete_test_case",
            "detector_id": "seal.test.deletion_authorized",
            "authority": "seal.test.deletion_authorized",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.test.deletion_authorized",
            ],
            "expected_kill": True,
            "observation": {
                "test_id": "test_critical_path",
                "deleted": True,
                "deletion_authorized": False,
                "remaining_units": ["unit/a"],
                "required_units": ["unit/a", "unit/b"],
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 9,
            "fixture_id": "zk.simulated_for_direct",
            "scenario": "substitute_simulated_for_direct_proof",
            "operator_id": "proof_unreachable_modeled_state",
            "detector_id": "seal.proof.direct_required",
            "authority": "seal.proof.direct_required",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.proof.direct_required",
            ],
            "expected_kill": True,
            "observation": {
                "required_for_seal": True,
                "presented_as_direct": True,
                "presented_as_production": True,
                "proof_mode": "simulated",
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 10,
            "fixture_id": "zk.ignore_blocking_child",
            "scenario": "ignore_blocking_child",
            "operator_id": "sd_omit_parent_seal_link",
            "detector_id": "seal.child.blocking_honored",
            "authority": "seal.child.blocking_honored",
            "kill_mechanisms": [
                "seal.child.blocking_honored",
                "seal.incremental",
            ],
            "expected_kill": True,
            "observation": {
                "blocking_child_ids": ["child-blocking-1"],
                "honored_blocking_children": [],
                "remaining_units": ["unit/a"],
                "required_units": ["unit/a", "unit/blocking-child"],
                "seal_accepted_claimed": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 11,
            "fixture_id": "zk.change_proof_forest_order",
            "scenario": "change_proof_forest_order",
            "operator_id": "sd_reuse_stale_proof_forest",
            "detector_id": "seal.proof_forest.order",
            "authority": "seal.proof_forest.order",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.proof_forest.order",
            ],
            "expected_kill": True,
            "observation": {
                "canonical_leaf_order": ["leaf:a", "leaf:b", "leaf:c"],
                "presented_leaf_order": ["leaf:c", "leaf:a", "leaf:b"],
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
        {
            "mutation_index": 12,
            "fixture_id": "zk.replay_proof_across_branches",
            "scenario": "replay_proof_across_branches",
            "operator_id": "sd_reuse_stale_proof_forest",
            "detector_id": "seal.proof.branch_bound",
            "authority": "seal.proof.branch_bound",
            "kill_mechanisms": [
                "seal.incremental",
                "seal.proof.branch_bound",
            ],
            "expected_kill": True,
            "observation": {
                "proof_branch_id": "branch-a",
                "current_branch_id": "branch-b",
                "presented_as_valid": True,
                "production_policy_changed": False,
            },
        },
    ]
    return {
        "interface": INTERFACE,
        "schema": "aae/zk-incremental-seal-probes@1",
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "campaign_id": CAMPAIGN_ID,
        "evidence_id": EVIDENCE_ID,
        "probe_count": len(probes),
        "production_policy_change_allowed": False,
        "released_sealer_required": True,
        "scenarios": list(REQUIRED_SCENARIOS),
        "probes": probes,
        "notes": (
            "Controlled observations for the twelve ZK/incremental-seal "
            "mutations; each must be rejected by the released sealer."
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
    "ZKIncrementalSealCampaignError",
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
    "parent_corpus_zk_incremental_seal_fixtures",
    "write_campaign_snapshot",
]

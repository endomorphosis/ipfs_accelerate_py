"""SecurityAssuranceCampaignB@1 compact fixture expansion and controlled probes.

AAE-051: mutations 11–20. Expands compact recipes into sealed assurance
fixtures (matching AssuranceFixtureCorpus@1 identity rules), builds the
campaign receipt, and evaluates controlled mutation observations against
declared kill mechanisms using existing fail-closed authorities where
available. Never mutates production policy or opens network services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    TestExecutionIdentityError,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    AdmissionPolicy,
    EvidenceCandidate,
    RejectionReason,
    verify_for_admission,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.provers import (
    IncrementalProofBackendAdapter,
    ProverInvocation,
    ProverReasonCode,
    ProverStatus,
    ProvingKeyHandle,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import SetupOrigin
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

INTERFACE: Final[str] = "SecurityAssuranceCampaignB@1"
CAMPAIGN_SCHEMA: Final[str] = "aae/security-campaign@1"
RECIPE_SCHEMA: Final[str] = "aae/fixture-recipe@1"
FIXTURE_SCHEMA: Final[str] = "aae/assurance-fixture@1"
ORACLE_SCHEMA: Final[str] = "aae/bounded-oracle@1"
DETECTOR_SCHEMA: Final[str] = "aae/expected-detector@1"
EVIDENCE_ID: Final[str] = "aae/security-campaign@1"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-security-b-v1"
TASK_ID: Final[str] = "AAE-051"
BUNDLE: Final[str] = "security_b"
PARENT_CORPUS_ID: Final[str] = "adversarial-assurance-fixture-corpus-v1"
MUTATION_INDEX_START: Final[int] = 11
MUTATION_INDEX_END: Final[int] = 20

# Plan §11 security mutations 11–20 (order is normative for this half-campaign).
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "retry_double_execution",
    "uncompensated_partial_mutation",
    "provider_ack_as_verified_storage",
    "receipt_before_observed_effect",
    "invalid_signature",
    "pseudo_cid",
    "stale_proof_receipt",
    "omitted_proof_unit",
    "unknown_prover_as_passed",
    "simulated_production_evidence",
)

REQUIRED_OPERATOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "se_retry_without_dedup",
        "se_skip_compensation_on_partial_failure",
        "st_trust_provider_ack_without_verify",
        "se_ack_before_effect_complete",
        "proof_unchecked_signature",
        "proof_stale_receipt",
        "proof_omit_unit",
        "proof_vacuous_impossible_assumption",
        "proof_unreachable_modeled_state",
    }
)

REQUIRED_DETECTOR_IDS: Final[frozenset[str]] = frozenset(
    {
        "runtime.retry.idempotency",
        "runtime.compensation.required",
        "storage.durability.provider_ack",
        "receipt.effect.order",
        "receipt.signature.verify",
        "content.cid.authentic",
        "proof.receipt.freshness",
        "proof.unit.completeness",
        "proof.prover.known",
        "evidence.mode.declared",
    }
)


class SecurityCampaignBError(ValueError):
    """Fail-closed campaign catalog error."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SecurityCampaignBError(f"missing fixture file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes() -> dict[str, Any]:
    doc = _load_json(RECIPES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SecurityCampaignBError("recipes.interface must be SecurityAssuranceCampaignB@1")
    if doc.get("task_id") != TASK_ID:
        raise SecurityCampaignBError("recipes.task_id must be AAE-051")
    if doc.get("bundle") != BUNDLE:
        raise SecurityCampaignBError("recipes.bundle must be security_b")
    if doc.get("production_policy_change_allowed") is not False:
        raise SecurityCampaignBError("production policy change is forbidden for fixture campaigns")
    recipes = list(doc.get("recipes") or [])
    if len(recipes) != MUTATION_INDEX_END - MUTATION_INDEX_START + 1:
        raise SecurityCampaignBError(
            f"expected 10 security_b recipes, found {len(recipes)}"
        )
    return doc


def load_probes() -> dict[str, Any]:
    doc = _load_json(PROBES_PATH)
    if doc.get("interface") != INTERFACE:
        raise SecurityCampaignBError("probes.interface must be SecurityAssuranceCampaignB@1")
    if doc.get("production_policy_change_allowed") is not False:
        raise SecurityCampaignBError("probes must forbid production policy change")
    probes = list(doc.get("probes") or [])
    if len(probes) != 10:
        raise SecurityCampaignBError(f"expected 10 probes, found {len(probes)}")
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
        raise SecurityCampaignBError(
            f"recipe missing fields: {sorted(missing)}"
        )
    if recipe["campaign_bundle"] != BUNDLE:
        raise SecurityCampaignBError(
            f"{recipe['fixture_id']}: campaign_bundle must be security_b"
        )
    if recipe["campaign"] != "security":
        raise SecurityCampaignBError(
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
        key: value
        for key, value in recipe.items()
        if key != "mutation_index"
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


def expand_all_fixtures(recipes_doc: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    doc = recipes_doc if recipes_doc is not None else load_recipes()
    fixtures = [expand_recipe(r) for r in doc["recipes"]]
    fixtures.sort(key=lambda item: (item.get("mutation_index") or 0, item["fixture_id"]))
    return fixtures


def parent_corpus_security_b_fixtures() -> list[dict[str, Any]]:
    """Load sealed security_b fixtures from the parent AAE-049 corpus."""
    if not PARENT_CORPUS_MANIFEST.is_file():
        raise SecurityCampaignBError(
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
        raise SecurityCampaignBError(
            f"parent corpus security_b count mismatch: {len(fixtures)}"
        )
    return fixtures


def _fixture_identity_without_mutation_index(fixture: Mapping[str, Any]) -> dict[str, Any]:
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
        f["fixture_id"]: f for f in parent_corpus_security_b_fixtures()
    }
    local = {f["fixture_id"]: f for f in fixtures}
    if set(local) != set(parent):
        raise SecurityCampaignBError(
            "fixture_id set diverges from parent corpus security_b bundle: "
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
            raise SecurityCampaignBError(
                f"{fixture_id}: expanded fixture identity diverges from parent corpus"
            )
        if expanded["fixture_cid"] != sealed["fixture_cid"]:
            raise SecurityCampaignBError(
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


def _digest(label: str) -> str:
    """Stable synthetic digest label for hermetic probe evidence."""
    # Match existing IPS test style: sha256: + 64 hex chars.
    alphabet = "0123456789abcdef"
    seed = sum(ord(ch) for ch in label) + len(label) * 17
    chars = []
    n = seed
    for i in range(64):
        n = (n * 1103515245 + 12345 + i) & 0x7FFFFFFF
        chars.append(alphabet[n % 16])
    return "sha256:" + "".join(chars)


def _evaluate_retry(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    executions = int(observation.get("effect_executions") or 0)
    idem_key = observation.get("idempotency_key")
    effect_idempotent = bool(observation.get("effect_idempotent"))
    retry_attempted = bool(observation.get("retry_attempted"))
    double_without_dedup = (
        executions >= 2
        and not idem_key
        and not effect_idempotent
        and retry_attempted
    )
    if double_without_dedup:
        return True, "double_execution_without_idempotency", {
            "effect_executions": executions,
            "idempotency_key": idem_key,
        }
    return False, "retry_idempotent_or_single_execution", {"effect_executions": executions}


def _evaluate_compensation(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    applied = int(observation.get("steps_applied") or 0)
    failed = int(observation.get("steps_failed") or 0)
    compensation_ran = bool(observation.get("compensation_ran"))
    success_claimed = bool(observation.get("success_claimed"))
    partial = applied > 0 and failed > 0
    if partial and not compensation_ran and success_claimed:
        return True, "partial_mutation_without_compensation", {
            "steps_applied": applied,
            "steps_failed": failed,
            "compensation_ran": compensation_ran,
        }
    return False, "compensated_or_complete", {
        "steps_applied": applied,
        "compensation_ran": compensation_ran,
    }


def _evaluate_provider_ack(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    ack = bool(observation.get("provider_ack_received"))
    durable = bool(observation.get("durable_commit_observed"))
    read_back = bool(observation.get("read_back_verified"))
    treated = bool(observation.get("treated_as_verified_storage"))
    if ack and treated and not (durable and read_back):
        return True, "provider_ack_treated_as_durable", {
            "provider_ack_received": ack,
            "durable_commit_observed": durable,
            "read_back_verified": read_back,
        }
    return False, "provider_ack_not_elevated", {"treated_as_verified_storage": treated}


def _evaluate_receipt_order(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    receipt_issued = bool(observation.get("receipt_issued"))
    effect_done = bool(observation.get("effect_observed_complete"))
    effect_ts = observation.get("effect_complete_timestamp_ms")
    if receipt_issued and not effect_done:
        return True, "receipt_before_effect", {
            "receipt_issued": receipt_issued,
            "effect_observed_complete": effect_done,
            "effect_complete_timestamp_ms": effect_ts,
        }
    receipt_ts = observation.get("receipt_timestamp_ms")
    if (
        receipt_issued
        and effect_done
        and receipt_ts is not None
        and effect_ts is not None
        and int(receipt_ts) < int(effect_ts)
    ):
        return True, "receipt_timestamp_before_effect", {
            "receipt_timestamp_ms": receipt_ts,
            "effect_complete_timestamp_ms": effect_ts,
        }
    return False, "receipt_after_effect", {"receipt_issued": receipt_issued}


def _evaluate_stale_receipt(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    presented = bool(observation.get("presented_as_current"))
    source_mismatch = observation.get("receipt_source_root") != observation.get(
        "current_source_root"
    )
    env_mismatch = observation.get("receipt_environment_cid") != observation.get(
        "current_environment_cid"
    )
    if presented and (source_mismatch or env_mismatch):
        return True, "stale_proof_receipt", {
            "source_mismatch": source_mismatch,
            "environment_mismatch": env_mismatch,
        }
    return False, "proof_receipt_fresh", {
        "source_mismatch": source_mismatch,
        "environment_mismatch": env_mismatch,
    }


def _evaluate_omitted_unit(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    required = list(observation.get("required_units") or [])
    present = list(observation.get("present_units") or [])
    claimed = bool(observation.get("seal_completeness_claimed"))
    missing = sorted(set(required) - set(present))
    if claimed and missing:
        return True, "required_unit_omitted", {"missing_units": missing}
    return False, "proof_units_complete", {"missing_units": missing}


def _evaluate_invalid_signature(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Exercise signature verification fail-closed path via admission verifier hook."""
    signature = str(observation.get("signature") or "")
    allowed = str(observation.get("allowed_signature") or "")
    signer = str(observation.get("signer") or "allowlist/security-b-operator")
    digest = _digest("security-b-invalid-sig-digest")

    def _verifier(evidence: Any, _meta: Mapping[str, Any]) -> bool:
        sig = getattr(evidence, "signature", None)
        if sig is None and isinstance(evidence, Mapping):
            sig = evidence.get("signature")
        return sig == allowed

    receipt = SignedExecutionReceipt(
        signer_id=signer,
        receipt_digest=digest,
        signature=signature,
        statement="security-b controlled invalid signature probe",
    )
    policy = AdmissionPolicy(
        allowed_signers=frozenset({signer}),
        production=True,
        require_signature_for_signed_receipts=True,
    )
    decision = verify_for_admission(
        EvidenceCandidate(
            evidence=receipt,
            proof_system_id="signed_receipt",
            public_input_cid=digest,
            proof_unit_id="unit/security-b/invalid-signature",
            required_for_seal=True,
            proof_mode=ProofMode.SIGNED_RECEIPT,
            terminal_status=ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
        ),
        policy=policy,
        verifier=_verifier,
    )
    killed = decision.admitted is False
    reason = decision.reason_code or "signature_check"
    return killed, str(reason), {
        "admitted": decision.admitted,
        "reason_code": decision.reason_code,
        "signature_matches_allowed": signature == allowed,
    }


def _evaluate_pseudo_cid(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    presented = str(observation.get("presented_cid") or "")
    field_name = str(observation.get("field_name") or "cid")
    try:
        reject_pseudo_cid(presented, field_name=field_name)
    except TestExecutionIdentityError as exc:
        return True, "pseudo_cid_rejected", {"error": str(exc), "presented_cid": presented}
    return False, "cid_accepted", {"presented_cid": presented}


def _evaluate_unknown_prover(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    backend_id = str(observation.get("backend_id") or "exotic-unknown-prover")
    presented_as_passed = bool(observation.get("presented_as_passed"))
    circuit_id = "circuit:security-b-unknown@1"
    handle = ProvingKeyHandle(
        key_id="pk/security-b-unknown",
        key_cid="bafybeigprovingkeysecuritybunknown0000000000001",
        circuit_ids=frozenset({circuit_id}),
        setup_origin=SetupOrigin.OPERATOR_REVIEWED,
        test_only=False,
        paired_verification_key_id="vk/security-b-unknown",
        epoch=1,
    )
    invocation = ProverInvocation(
        program_id="program:security-b-unknown@1",
        circuit_id=circuit_id,
        public_input=b"security-b-unknown-prover",
        witness=b"witness-security-b",
        proving_key_handle=handle,
        verification_key_id="vk/security-b-unknown",
        verification_key_cid="bafybeigverificationkeysecuritybunknown00000001",
        backend_id=backend_id,
        proof_unit_id="unit/security-b/unknown-prover",
        production=True,
    )
    outcome = IncrementalProofBackendAdapter().prove(invocation)
    proved = bool(getattr(outcome, "proved", False))
    status = getattr(outcome, "status", None)
    reason_code = getattr(outcome, "reason_code", None)
    unknown = (
        status is ProverStatus.UNKNOWN
        or reason_code == ProverReasonCode.UNKNOWN_BACKEND.value
    )
    # Detector kills the mutation that would treat an unknown prover as passed.
    if presented_as_passed and (unknown or not proved):
        killed = True
        reason = "unknown_backend" if unknown else str(reason_code or "prover_failed")
    elif proved and presented_as_passed:
        killed = False
        reason = "prover_passed_unexpectedly"
    else:
        killed = unknown or not proved
        reason = str(reason_code or "prover_not_passed")
    return killed, reason, {
        "proved": proved,
        "status": getattr(status, "value", status),
        "reason_code": reason_code,
        "backend_id": backend_id,
        "presented_as_passed": presented_as_passed,
    }


def _evaluate_simulated(observation: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    required = bool(observation.get("required_for_seal", True))
    presented_as_production = bool(observation.get("presented_as_production", True))
    digest = _digest("security-b-simulated-digest")
    integrity = IntegrityCommitment(
        digest=digest,
        cid=_digest("security-b-simulated-cid"),
        merkle_inclusion="leaf:0",
        byte_length=32,
    )
    decision = verify_for_admission(
        EvidenceCandidate(
            evidence=integrity,
            proof_system_id="integrity",
            public_input_cid=digest,
            proof_unit_id="unit/security-b/simulated",
            required_for_seal=required,
            proof_mode=ProofMode.SIMULATED,
            terminal_status=ProofTerminalStatus.SIMULATED,
            expected_digest=digest,
            observed_digest=digest,
        ),
        policy=AdmissionPolicy(production=True),
    )
    killed = decision.admitted is False and presented_as_production
    reason = decision.reason_code or "simulated_rejected"
    if decision.reason_code == RejectionReason.SIMULATED_REQUIRED_UNIT.value:
        reason = "simulated_required_unit"
        killed = True
    return killed, str(reason), {
        "admitted": decision.admitted,
        "reason_code": decision.reason_code,
        "required_for_seal": required,
        "presented_as_production": presented_as_production,
    }


_SCENARIO_EVALUATORS = {
    "retry_double_execution": _evaluate_retry,
    "uncompensated_partial_mutation": _evaluate_compensation,
    "provider_ack_as_verified_storage": _evaluate_provider_ack,
    "receipt_before_observed_effect": _evaluate_receipt_order,
    "invalid_signature": _evaluate_invalid_signature,
    "pseudo_cid": _evaluate_pseudo_cid,
    "stale_proof_receipt": _evaluate_stale_receipt,
    "omitted_proof_unit": _evaluate_omitted_unit,
    "unknown_prover_as_passed": _evaluate_unknown_prover,
    "simulated_production_evidence": _evaluate_simulated,
}


def evaluate_probe(probe: Mapping[str, Any]) -> ProbeResult:
    """Evaluate one controlled mutation probe; kill is mandatory for this campaign."""
    scenario = str(probe["scenario"])
    evaluator = _SCENARIO_EVALUATORS.get(scenario)
    if evaluator is None:
        raise SecurityCampaignBError(f"no evaluator for scenario {scenario!r}")
    observation = probe.get("observation") or {}
    if not isinstance(observation, Mapping):
        raise SecurityCampaignBError(f"{scenario}: observation must be a mapping")
    killed, reason, details = evaluator(observation)
    authority = str(probe.get("authority") or probe.get("detector_id") or "")
    kill_mechanisms = list(probe.get("kill_mechanisms") or [])
    kill_mechanism = authority if authority in kill_mechanisms else (
        kill_mechanisms[0] if kill_mechanisms else authority
    )
    expected_kill = probe.get("expected_kill", True)
    if expected_kill and not killed:
        raise SecurityCampaignBError(
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


def evaluate_all_probes(probes_doc: Mapping[str, Any] | None = None) -> list[ProbeResult]:
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
    """Build the sealed SecurityAssuranceCampaignB@1 campaign document."""
    recipes = recipes_doc if recipes_doc is not None else load_recipes()
    probes = probes_doc if probes_doc is not None else load_probes()
    fixtures = expand_all_fixtures(recipes)
    if verify_parent:
        assert_fixtures_match_parent_corpus(fixtures)

    scenarios = [f["scenario"] for f in fixtures]
    if tuple(scenarios) != REQUIRED_SCENARIOS:
        # Allow recipe order by mutation_index even if fixture_id sort differs.
        by_index = sorted(
            fixtures,
            key=lambda item: int(item.get("mutation_index") or 0),
        )
        scenarios = [f["scenario"] for f in by_index]
        if tuple(scenarios) != REQUIRED_SCENARIOS:
            raise SecurityCampaignBError(
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
            raise SecurityCampaignBError(
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
            "Controlled security mutations 11–20. Each case declares the "
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
    "SecurityCampaignBError",
    "TASK_ID",
    "assert_fixtures_match_parent_corpus",
    "build_campaign",
    "evaluate_all_probes",
    "evaluate_probe",
    "expand_all_fixtures",
    "expand_recipe",
    "load_probes",
    "load_recipes",
    "parent_corpus_security_b_fixtures",
    "write_campaign_snapshot",
]

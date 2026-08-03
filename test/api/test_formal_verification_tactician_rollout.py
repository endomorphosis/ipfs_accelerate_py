"""Executable contract for FormalVerificationTacticianRollout@1 (FVT-035 / FVT-G080).

Validates that
``docs/architecture/formal_verification_tactician_rollout.md`` defines
property-specific rollout, promotion, and rollback for the goal-directed
formal verification tactician:

* stages ``off``, ``shadow``, ``assist``, ``auto_safe``, and ``enforced``;
* adjacent promotion and automatic quarantine/rollback;
* gates that consume actual conformance, benchmark, and toolchain receipts;
* auto-safe admission limited to allowlisted independently validated steps;
* hard-zero signals (false proof/closure, leakage, binding mismatch, authority
  escalation, unresolved disagreement);
* disclosure of unsupported and unavailable lanes;
* no global enforcement from aggregate portfolio success.

Pure policy helpers in this module re-encode the documented rules so the suite
is more than string matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_DOC = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_rollout.md"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)
BENCHMARK_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_benchmark.json"
)
TOOLCHAIN_CERT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_toolchain_certificate.json"
)
BASELINE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_readiness_baseline.json"
)
LFV_ROLLOUT_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "docs"
    / "logic"
    / "software_verification_rollout.md"
)

INTERFACE = "FormalVerificationTacticianRollout@1"
GOAL_ID = "FVT-G080"
TASK_ID = "FVT-035"
SCHEMA = "formal-verification-tactician-rollout/v1"

STAGES: Final[tuple[str, ...]] = (
    "off",
    "shadow",
    "assist",
    "auto_safe",
    "enforced",
)

HARD_ZERO_SIGNALS: Final[tuple[str, ...]] = (
    "false proof",
    "false_closure",
    "leakage",
    "binding mismatch",
    "authority",
    "disagreement",
)

REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## 1. Program invariants",
    "## 2. Stage ladder",
    "## 3. Promotion rules",
    "## 4. Auto-safe admission allowlist",
    "## 5. Hard-zero quarantine and rollback",
    "## 6. Property- and provider-specific policy",
    "## 8. Validation checklist",
    "## 10. Acceptance mapping",
)

REQUIRED_EVIDENCE_PATHS: Final[tuple[str, ...]] = (
    "docs/architecture/formal_verification_tactician_benchmark.json",
    "docs/architecture/formal_verification_toolchain_certificate.json",
    "docs/architecture/formal_verification_readiness_baseline.json",
    "ipfs_datasets_py/docs/logic/software_verification_rollout.md",
)

DEFAULT_ALLOWLIST_STEPS: Final[frozenset[str]] = frozenset(
    {
        "admit_validated_lemma",
        "admit_validated_invariant",
        "admit_validated_contract",
        "admit_plan_step_with_receipt",
        "close_counterexample_with_verifier_receipt",
        "apply_deterministic_minimization",
        "replay_confirmed_counterexample",
    }
)

FORBIDDEN_STEPS: Final[frozenset[str]] = frozenset(
    {
        "promote_proof_authority",
        "admit_goal",
        "force_complete",
        "close_plan",
        "lease_steal",
    }
)


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------


def _read_rollout() -> str:
    assert ROLLOUT_DOC.is_file(), f"missing rollout policy: {ROLLOUT_DOC}"
    text = ROLLOUT_DOC.read_text(encoding="utf-8")
    assert text.strip(), "rollout policy is empty"
    assert ROLLOUT_DOC.stat().st_size > 2000, "rollout policy unexpectedly small"
    return text


def _objective_section() -> str:
    assert OBJECTIVES_PATH.is_file(), f"missing objectives heap: {OBJECTIVES_PATH}"
    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert f"## {GOAL_ID}" in objectives, f"{GOAL_ID} missing from objectives heap"
    return objectives.split(f"## {GOAL_ID}", 1)[1].split("\n## ", 1)[0]


# ---------------------------------------------------------------------------
# Pure policy model (mirrors the documented contract)
# ---------------------------------------------------------------------------


class TacticianRolloutStage(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTO_SAFE = "auto_safe"
    ENFORCED = "enforced"


_PROMOTION_ORDER: Final[tuple[TacticianRolloutStage, ...]] = (
    TacticianRolloutStage.OFF,
    TacticianRolloutStage.SHADOW,
    TacticianRolloutStage.ASSIST,
    TacticianRolloutStage.AUTO_SAFE,
    TacticianRolloutStage.ENFORCED,
)


@dataclass(frozen=True)
class PairIdentity:
    property_kind: str
    provider_id: str
    authority_class: str = "bounded_solver_outcome"


@dataclass(frozen=True)
class GateEvidence:
    """Receipt-backed inputs required by §3.2."""

    conformance_receipt_ids: tuple[str, ...] = ()
    benchmark_receipt_id: str = ""
    toolchain_certificate_id: str = ""
    hard_correctness_pass: bool = False
    hard_privacy_pass: bool = False
    hard_authority_pass: bool = False
    allow_auto_safe_promotion: bool = False
    allowlist_steps: frozenset[str] = field(default_factory=frozenset)
    independent_validation_receipt_ids: tuple[str, ...] = ()
    tools_usable: bool = False
    disclosed_unavailable: bool = False
    disclosed_unsupported: bool = False


@dataclass(frozen=True)
class HardZeroCounters:
    false_proof_count: int = 0
    false_closure_count: int = 0
    secret_or_witness_leakage_count: int = 0
    binding_mismatch_count: int = 0
    authority_boundary_violations: int = 0
    unresolved_cross_provider_disagreement_count: int = 0
    fabricated_readiness_count: int = 0

    def positive_signals(self) -> tuple[str, ...]:
        mapping = {
            "false_proof_observed": self.false_proof_count,
            "false_closure_observed": self.false_closure_count,
            "secret_or_witness_leakage": self.secret_or_witness_leakage_count,
            "binding_mismatch": self.binding_mismatch_count,
            "authority_boundary_violation": self.authority_boundary_violations,
            "unresolved_disagreement": (
                self.unresolved_cross_provider_disagreement_count
            ),
            "fabricated_readiness": self.fabricated_readiness_count,
        }
        return tuple(sorted(name for name, count in mapping.items() if count > 0))


@dataclass(frozen=True)
class PromotionDecision:
    pair: PairIdentity
    from_stage: TacticianRolloutStage
    to_stage: TacticianRolloutStage
    allowed: bool
    reason_codes: tuple[str, ...]
    demotion_target: TacticianRolloutStage | None = None

    def __post_init__(self) -> None:
        if self.allowed == bool(self.reason_codes):
            raise ValueError("allowed must be true exactly when reason_codes is empty")


def _as_stage(value: TacticianRolloutStage | str) -> TacticianRolloutStage:
    if isinstance(value, TacticianRolloutStage):
        return value
    return TacticianRolloutStage(str(value))


def evaluate_tactician_promotion(
    pair: PairIdentity,
    *,
    from_stage: TacticianRolloutStage | str,
    to_stage: TacticianRolloutStage | str,
    evidence: GateEvidence,
    hard_zero: HardZeroCounters | None = None,
) -> PromotionDecision:
    """Evaluate an adjacent promotion using the documented fail-closed rules."""

    source = _as_stage(from_stage)
    target = _as_stage(to_stage)
    counters = hard_zero or HardZeroCounters()
    reasons: list[str] = []

    if source not in _PROMOTION_ORDER or target not in _PROMOTION_ORDER:
        reasons.append("unsupported_promotion_mode")
    elif _PROMOTION_ORDER.index(target) != _PROMOTION_ORDER.index(source) + 1:
        reasons.append("promotion_must_be_adjacent")

    signals = counters.positive_signals()
    if signals:
        reasons.extend(signals)

    # Receipt-backed gates: require actual evidence identities, not empty claims.
    if target is not TacticianRolloutStage.OFF:
        if not evidence.conformance_receipt_ids:
            reasons.append("conformance_receipts_missing")
        if not evidence.benchmark_receipt_id:
            reasons.append("benchmark_receipt_missing")
        if not evidence.toolchain_certificate_id:
            reasons.append("toolchain_certificate_missing")

    if target in {
        TacticianRolloutStage.ASSIST,
        TacticianRolloutStage.AUTO_SAFE,
        TacticianRolloutStage.ENFORCED,
    }:
        if not (
            evidence.hard_correctness_pass
            and evidence.hard_privacy_pass
            and evidence.hard_authority_pass
        ):
            reasons.append("benchmark_hard_gates_not_pass")

    if target is TacticianRolloutStage.AUTO_SAFE:
        if not evidence.allow_auto_safe_promotion:
            reasons.append("auto_safe_promotion_not_explicitly_authorized")
        if not evidence.allowlist_steps:
            reasons.append("auto_safe_allowlist_empty")
        elif not evidence.allowlist_steps.issubset(DEFAULT_ALLOWLIST_STEPS):
            reasons.append("auto_safe_allowlist_contains_unknown_steps")
        if not evidence.independent_validation_receipt_ids:
            reasons.append("independent_validation_receipts_missing")

    if target is TacticianRolloutStage.ENFORCED:
        if not evidence.tools_usable and not evidence.disclosed_unavailable:
            reasons.append("tools_neither_usable_nor_disclosed_unavailable")
        if not evidence.tools_usable:
            # Explicit unavailable disclosure blocks enforcement claims.
            reasons.append("enforced_requires_usable_tools")
        if not evidence.allowlist_steps:
            reasons.append("enforced_requires_prior_allowlist")
        if not evidence.allow_auto_safe_promotion:
            reasons.append("enforced_requires_prior_auto_safe_authorization")

    # Aggregate portfolio success is intentionally not an input; only pair-local
    # evidence is considered (conflict policy).
    normalized = tuple(sorted(set(reasons)))
    return PromotionDecision(
        pair=pair,
        from_stage=source,
        to_stage=target,
        allowed=not normalized,
        reason_codes=normalized,
    )


def evaluate_auto_safe_admission(
    *,
    stage: TacticianRolloutStage | str,
    step_kind: str,
    allowlist: Sequence[str] | frozenset[str],
    independent_validation_receipt_ids: Sequence[str],
    hard_zero: HardZeroCounters | None = None,
    authority_exceeds_ceiling: bool = False,
    counterexample_closed_without_verifier: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed auto-safe admission per §4."""

    live = _as_stage(stage)
    reasons: list[str] = []
    if live not in {TacticianRolloutStage.AUTO_SAFE, TacticianRolloutStage.ENFORCED}:
        reasons.append("stage_cannot_admit")
    allowed_set = frozenset(allowlist)
    if step_kind in FORBIDDEN_STEPS:
        reasons.append("forbidden_step_kind")
    if step_kind not in allowed_set:
        reasons.append("step_not_on_allowlist")
    if step_kind not in DEFAULT_ALLOWLIST_STEPS and step_kind not in FORBIDDEN_STEPS:
        reasons.append("step_outside_default_closed_set")
    if not independent_validation_receipt_ids:
        reasons.append("independent_validation_receipts_missing")
    signals = (hard_zero or HardZeroCounters()).positive_signals()
    if signals:
        reasons.append("pair_under_quarantine")
    if authority_exceeds_ceiling:
        reasons.append("authority_ceiling_exceeded")
    if (
        step_kind == "close_counterexample_with_verifier_receipt"
        and counterexample_closed_without_verifier
    ):
        reasons.append("counterexample_requires_fresh_verifier_receipt")
    normalized = tuple(sorted(set(reasons)))
    return (not normalized, normalized)


def quarantine_demotion_target(
    signal: str,
) -> TacticianRolloutStage:
    """Map hard-zero severity to demotion target (§5.2)."""

    severe = {
        "false_proof_observed",
        "false_closure_observed",
        "secret_or_witness_leakage",
        "authority_boundary_violation",
        "fabricated_readiness",
    }
    if signal in severe:
        return TacticianRolloutStage.OFF
    return TacticianRolloutStage.SHADOW


def aggregate_portfolio_must_not_promote(
    pair_decisions: Mapping[str, bool],
) -> bool:
    """Return True when policy correctly refuses global promotion from aggregates.

    Aggregate success is never an input to evaluate_tactician_promotion; this
    helper documents and locks that invariant for the suite.
    """

    # Even if every *other* pair is green, a red pair stays unpromoted when its
    # own decision is False. Global AND-of-others is irrelevant.
    return all(
        (name == "target" and allowed is False) or name != "target"
        for name, allowed in pair_decisions.items()
    ) and pair_decisions.get("target") is False


# ---------------------------------------------------------------------------
# Document presence and identity
# ---------------------------------------------------------------------------


def test_rollout_document_exists_and_declares_interface() -> None:
    text = _read_rollout()
    assert INTERFACE in text
    assert GOAL_ID in text
    assert TASK_ID in text
    assert SCHEMA in text
    assert "Formal Verification Tactician Rollout Policy" in text


def test_objective_heap_lists_rollout_evidence() -> None:
    section = _objective_section()
    assert "docs/architecture/formal_verification_tactician_rollout.md" in section
    assert "test/api/test_formal_verification_tactician_rollout.py" in section
    assert INTERFACE in section or "FormalVerificationTacticianRollout" in section
    assert "test_formal_verification_tactician_rollout.py" in section


def test_objective_heap_acceptance_matches_policy() -> None:
    section = _objective_section().lower()
    for phrase in (
        "conformance",
        "benchmark",
        "toolchain",
        "auto-safe",
        "allowlist",
        "quarantine",
        "rollback",
        "unsupported",
        "unavailable",
    ):
        assert phrase in section, phrase


# ---------------------------------------------------------------------------
# Structural contract of the markdown policy
# ---------------------------------------------------------------------------


def test_required_sections_present() -> None:
    text = _read_rollout()
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    assert not missing, missing


def test_stage_ladder_documents_all_modes() -> None:
    text = _read_rollout()
    for stage in STAGES:
        # Stages appear as tokens in tables and prose.
        assert f"`{stage}`" in text, stage
    # Adjacent ladder spelling.
    assert "off → shadow → assist → auto_safe → enforced" in text
    assert "promotion_must_be_adjacent" in text


def test_gates_consume_actual_receipts() -> None:
    text = _read_rollout()
    for path in REQUIRED_EVIDENCE_PATHS:
        assert path in text, path
    assert "never" in text.lower()
    assert "synthetic" in text.lower() or "hardcoded" in text.lower()
    assert "conformance" in text.lower()
    assert "benchmark" in text.lower()
    assert "toolchain" in text.lower()
    # Timing is observational / not correctness by default.
    assert "timing" in text.lower()
    assert "correctness" in text.lower()


def test_auto_safe_allowlist_and_independent_validation() -> None:
    text = _read_rollout()
    assert "auto_safe" in text
    assert "allowlist" in text.lower() or "allow-listed" in text.lower()
    assert "independently validated" in text.lower() or "independent validation" in text.lower()
    assert "allow_auto_safe_promotion" in text
    assert "auto_safe_promotion_not_explicitly_authorized" in text
    for step in DEFAULT_ALLOWLIST_STEPS:
        assert step in text, step
    for forbidden in ("promote_proof_authority", "force_complete", "lease_steal"):
        assert forbidden in text, forbidden
    assert "only auto_safe mode can produce an admission receipt" in text or (
        "admission receipt" in text.lower() and "auto_safe" in text
    )


def test_hard_zero_quarantine_and_rollback_language() -> None:
    text = _read_rollout().lower()
    collapsed = re.sub(r"\s+", " ", text)
    for signal in HARD_ZERO_SIGNALS:
        assert signal in text, signal
    assert "quarantine" in text
    assert "rollback" in text
    assert "hard-zero" in text or "hard zero" in text
    assert "historical receipts remain immutable" in collapsed
    assert "false_proof" in text or "false proof" in text
    assert "authority_boundary" in text or "authority escalation" in text


def test_property_provider_specific_not_global() -> None:
    text = _read_rollout().lower()
    assert "property-specific" in text or "property/provider" in text
    assert "aggregate" in text
    assert "never" in text
    assert "global" in text
    # Explicit conflict policy echo.
    assert "do not globally enforce" in text or "no global" in text


def test_unsupported_unavailable_disclosed() -> None:
    text = _read_rollout().lower()
    assert "unsupported" in text
    assert "unavailable" in text
    assert "disclosed" in text or "disclosure" in text
    assert "never fabricate" in text or "never" in text and "fabricate" in text


def test_capability_coverage_for_goal() -> None:
    """Goal text: formalization, proof-gap, plans, counterexample-guided repair."""

    text = _read_rollout().lower()
    assert "formalization" in text
    assert "proof-gap" in text or "proof gap" in text or "hole" in text
    assert "plan" in text
    assert "counterexample" in text
    assert "cegis" in text or "repair" in text


def test_cross_links_and_companions() -> None:
    text = _read_rollout()
    assert "software_verification_rollout.md" in text
    assert "formal_verification_tactician.md" in text
    assert "formal_verification_tactician_runbook.md" in text
    assert "GoalDevelopmentMode" in text or "goal_development_contracts" in text
    assert "evaluate_goal_rollout_promotion" in text
    # Companion files that exist in-tree when present should be referenced.
    if BENCHMARK_PATH.is_file():
        assert "formal_verification_tactician_benchmark.json" in text
    if TOOLCHAIN_CERT_PATH.is_file():
        assert "formal_verification_toolchain_certificate.json" in text
    if BASELINE_PATH.is_file():
        assert "formal_verification_readiness_baseline.json" in text
    if LFV_ROLLOUT_PATH.is_file():
        assert "software_verification_rollout.md" in text


def test_acceptance_mapping_section_covers_criteria() -> None:
    text = _read_rollout()
    section = text.split("## 10. Acceptance mapping", 1)[1]
    lower = section.lower()
    for needle in (
        "conformance",
        "benchmark",
        "toolchain",
        "allowlist",
        "quarantine",
        "unsupported",
        "unavailable",
        "aggregate",
    ):
        assert needle in lower, needle


# ---------------------------------------------------------------------------
# Executable policy helpers
# ---------------------------------------------------------------------------


def _green_evidence(**overrides: object) -> GateEvidence:
    base = dict(
        conformance_receipt_ids=("conformance:fixture-corpus-v1",),
        benchmark_receipt_id="benchmark:goal-tactician-v1",
        toolchain_certificate_id="toolchain:cert-v1",
        hard_correctness_pass=True,
        hard_privacy_pass=True,
        hard_authority_pass=True,
        allow_auto_safe_promotion=True,
        allowlist_steps=frozenset({"admit_validated_lemma", "admit_plan_step_with_receipt"}),
        independent_validation_receipt_ids=("validation:lemma-1",),
        tools_usable=True,
        disclosed_unavailable=False,
        disclosed_unsupported=False,
    )
    base.update(overrides)
    return GateEvidence(**base)  # type: ignore[arg-type]


def test_adjacent_promotion_off_to_shadow_with_receipts() -> None:
    pair = PairIdentity("contract", "z3")
    decision = evaluate_tactician_promotion(
        pair,
        from_stage="off",
        to_stage="shadow",
        evidence=_green_evidence(),
    )
    assert decision.allowed
    assert decision.reason_codes == ()


def test_non_adjacent_promotion_rejected() -> None:
    pair = PairIdentity("contract", "z3")
    decision = evaluate_tactician_promotion(
        pair,
        from_stage="off",
        to_stage="assist",
        evidence=_green_evidence(),
    )
    assert not decision.allowed
    assert "promotion_must_be_adjacent" in decision.reason_codes


def test_missing_receipts_block_promotion() -> None:
    pair = PairIdentity("invariant", "cvc5")
    decision = evaluate_tactician_promotion(
        pair,
        from_stage="off",
        to_stage="shadow",
        evidence=GateEvidence(),
    )
    assert not decision.allowed
    assert "conformance_receipts_missing" in decision.reason_codes
    assert "benchmark_receipt_missing" in decision.reason_codes
    assert "toolchain_certificate_missing" in decision.reason_codes


def test_hard_zero_blocks_and_demotes() -> None:
    pair = PairIdentity("contract", "z3")
    decision = evaluate_tactician_promotion(
        pair,
        from_stage="assist",
        to_stage="auto_safe",
        evidence=_green_evidence(),
        hard_zero=HardZeroCounters(false_proof_count=1, authority_boundary_violations=1),
    )
    assert not decision.allowed
    assert "false_proof_observed" in decision.reason_codes
    assert "authority_boundary_violation" in decision.reason_codes
    assert quarantine_demotion_target("false_proof_observed") is TacticianRolloutStage.OFF
    assert (
        quarantine_demotion_target("binding_mismatch") is TacticianRolloutStage.SHADOW
    )


def test_auto_safe_requires_opt_in_and_allowlist() -> None:
    pair = PairIdentity("contract", "z3")
    denied = evaluate_tactician_promotion(
        pair,
        from_stage="assist",
        to_stage="auto_safe",
        evidence=_green_evidence(allow_auto_safe_promotion=False),
    )
    assert not denied.allowed
    assert "auto_safe_promotion_not_explicitly_authorized" in denied.reason_codes

    empty = evaluate_tactician_promotion(
        pair,
        from_stage="assist",
        to_stage="auto_safe",
        evidence=_green_evidence(allowlist_steps=frozenset()),
    )
    assert not empty.allowed
    assert "auto_safe_allowlist_empty" in empty.reason_codes

    ok = evaluate_tactician_promotion(
        pair,
        from_stage="assist",
        to_stage="auto_safe",
        evidence=_green_evidence(),
    )
    assert ok.allowed


def test_enforced_requires_usable_tools_not_mere_disclosure() -> None:
    pair = PairIdentity("theorem", "lean")
    disclosed_only = evaluate_tactician_promotion(
        pair,
        from_stage="auto_safe",
        to_stage="enforced",
        evidence=_green_evidence(tools_usable=False, disclosed_unavailable=True),
    )
    assert not disclosed_only.allowed
    assert "enforced_requires_usable_tools" in disclosed_only.reason_codes

    usable = evaluate_tactician_promotion(
        pair,
        from_stage="auto_safe",
        to_stage="enforced",
        evidence=_green_evidence(tools_usable=True),
    )
    assert usable.allowed


def test_auto_safe_admission_allowlist_enforced() -> None:
    allowlist = ["admit_validated_lemma", "close_counterexample_with_verifier_receipt"]
    ok, reasons = evaluate_auto_safe_admission(
        stage="auto_safe",
        step_kind="admit_validated_lemma",
        allowlist=allowlist,
        independent_validation_receipt_ids=("v:1",),
    )
    assert ok and reasons == ()

    blocked, codes = evaluate_auto_safe_admission(
        stage="assist",
        step_kind="admit_validated_lemma",
        allowlist=allowlist,
        independent_validation_receipt_ids=("v:1",),
    )
    assert not blocked
    assert "stage_cannot_admit" in codes

    forbidden, fcodes = evaluate_auto_safe_admission(
        stage="auto_safe",
        step_kind="promote_proof_authority",
        allowlist=allowlist,
        independent_validation_receipt_ids=("v:1",),
    )
    assert not forbidden
    assert "forbidden_step_kind" in fcodes

    no_verifier, ncodes = evaluate_auto_safe_admission(
        stage="auto_safe",
        step_kind="close_counterexample_with_verifier_receipt",
        allowlist=allowlist,
        independent_validation_receipt_ids=("v:1",),
        counterexample_closed_without_verifier=True,
    )
    assert not no_verifier
    assert "counterexample_requires_fresh_verifier_receipt" in ncodes

    quarantined, qcodes = evaluate_auto_safe_admission(
        stage="auto_safe",
        step_kind="admit_validated_lemma",
        allowlist=allowlist,
        independent_validation_receipt_ids=("v:1",),
        hard_zero=HardZeroCounters(secret_or_witness_leakage_count=1),
    )
    assert not quarantined
    assert "pair_under_quarantine" in qcodes


def test_aggregate_success_does_not_promote_target_pair() -> None:
    # Three pairs green, target red — policy helper must not flip target.
    decisions = {
        "contract/z3": True,
        "invariant/cvc5": True,
        "liveness/tlc": True,
        "target": False,
    }
    assert aggregate_portfolio_must_not_promote(decisions)
    # evaluate_tactician_promotion never accepts a portfolio score argument.
    source = evaluate_tactician_promotion.__code__.co_varnames[
        : evaluate_tactician_promotion.__code__.co_argcount
    ]
    assert "portfolio" not in source
    assert "aggregate" not in source


def test_full_ladder_green_path() -> None:
    pair = PairIdentity("contract", "z3")
    evidence = _green_evidence()
    stage = TacticianRolloutStage.OFF
    for target in _PROMOTION_ORDER[1:]:
        decision = evaluate_tactician_promotion(
            pair,
            from_stage=stage,
            to_stage=target,
            evidence=evidence,
        )
        assert decision.allowed, (stage, target, decision.reason_codes)
        stage = target
    assert stage is TacticianRolloutStage.ENFORCED


def test_hard_zero_counter_names_match_document() -> None:
    text = _read_rollout()
    for name in (
        "authority_boundary_violations",
        "false_proof_count",
        "false_completion_count",
        "secret_or_witness_leakage_count",
        "unresolved_cross_provider_disagreement_count",
    ):
        assert name in text, name


def test_stage_enum_matches_goal_development_modes_subset() -> None:
    """Documented stages include the GoalDevelopmentMode ladder tokens."""

    from ipfs_accelerate_py.agent_supervisor.objectives.goal_development_contracts import (
        GoalDevelopmentMode,
    )

    mode_values = {mode.value for mode in GoalDevelopmentMode}
    for required in ("off", "shadow", "assist", "auto_safe"):
        assert required in mode_values
        assert required in STAGES
    # enforced is tactician pair-local (LFV-aligned), not a GoalDevelopmentMode.
    assert "enforced" in STAGES
    assert "enforced" not in mode_values


def test_document_mentions_validation_command() -> None:
    text = _read_rollout()
    assert "test/api/test_formal_verification_tactician_rollout.py" in text
    assert "pytest" in text


def test_no_placeholder_stubs_in_policy() -> None:
    text = _read_rollout().lower()
    for banned in ("todo: implement", "tbd", "placeholder", "coming soon"):
        assert banned not in text, banned
    # Ensure substantial normative content.
    assert len(re.findall(r"^## ", text, flags=re.MULTILINE)) >= 8

#!/usr/bin/env python3
"""Bounded hyperproperty toolchain certification.

``HyperpropertyToolchainCertification@1`` / FVT-G170 (FVT-046).

Explicit strict installation selects reviewed HyperLTL, AutoHyper, and MCHyper
artifacts.  The certification corpus covers:

* quantifier order and observation projections preserved through translation;
* satisfaction under the pin-bound engines;
* violating multi-trace tuples with observation-map replay;
* semantic mutations (observation projection / quantifier signature);
* deterministic replay;
* malformed output, disagreement, and timeout fail-closed / quarantine;
* exact self-composition bounds disclosure;
* results retain declared **bounded** hyperproperty authority and never
  authorize universal claims beyond bounds.

This lane never edits the central multi-prover certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for _candidate in (_REPO_ROOT, _DATASETS_ROOT):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ipfs_datasets_py.logic.backends.hyperproperties.adapters import (  # noqa: E402
    AutoHyperBackend,
    HyperCheckOutcomeStatus,
    HyperEngine,
    HyperEvidencePath,
    HyperLTLBackend,
    HyperpropertyBackend,
    MCHyperBackend,
    ObservationMap,
    QuantifierOrder,
    parse_hyper_counterexample,
    render_hyperltl_formula,
    replay_hyper_counterexample,
)
from ipfs_datasets_py.logic.backends.installers import hyperproperty as hyper_installer  # noqa: E402
from ipfs_datasets_py.logic.backends.process import (  # noqa: E402
    BoundedToolRunner,
    ToolRunLimits,
    ToolRunRequest,
    ToolRuntime,
)
from ipfs_datasets_py.logic.backends.results import ResultAuthority, ResultStatus  # noqa: E402
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolRole,
    ToolchainAuthorityCeiling,
    get_tool_role,
)
from ipfs_datasets_py.logic.families.models import EvidenceAuthority  # noqa: E402
from ipfs_datasets_py.logic.software_verification.hyperproperties import (  # noqa: E402
    HyperpropertyFormula,
    HyperpropertyIR,
    HyperpropertyKind,
    InformationFlowPolicy,
    ObservationKind,
    ObservationSpec,
    QuantifierBinding,
    SecurityLabel,
    SecurityLevel,
    SelfCompositionBound,
    TraceQuantifier,
    TraceVariable,
)

INTERFACE: Final = "HyperpropertyToolchainCertification@1"
SCHEMA_VERSION: Final = "hyperproperty-toolchain-certification/v1"
GOAL_ID: Final = "FVT-G170"
TASK_ID: Final = "FVT-046"
PROGRAM: Final = "formal-verification-tactician/hyperproperty-toolchains"
LANE_ID: Final = "hyperltl"
HANDLER_ID: Final = "hyperproperty_toolchain_certification@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.hyperproperty"

AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.BOUNDED.value
AUTHORITY_ROLE: Final = ToolRole.AUTHORITY.value

TOOL_HYPERLTL: Final = hyper_installer.TOOL_HYPERLTL
TOOL_AUTOHYPER: Final = hyper_installer.TOOL_AUTOHYPER
TOOL_MCHYPER: Final = hyper_installer.TOOL_MCHYPER
EXTERNAL_ENGINES: Final = hyper_installer.EXTERNAL_TOOLS

REQUIRED_CATEGORIES: Final = frozenset(
    {
        "satisfaction",
        "violation",
        "mutation",
        "replay",
        "malformed",
        "disagreement",
        "timeout",
        "bounds",
    }
)
REQUIRED_MUTATION_KINDS: Final = frozenset({"observation", "quantifier"})
CHECK_KINDS: Final = frozenset(
    {
        "positive",
        "violation",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "disagreement_quarantine",
        "translation",
        "bounds",
        "authority",
        "install",
        "role",
    }
)

_BACKEND_TYPES: Final[Mapping[str, type[HyperpropertyBackend]]] = {
    TOOL_HYPERLTL: HyperLTLBackend,
    TOOL_AUTOHYPER: AutoHyperBackend,
    TOOL_MCHYPER: MCHyperBackend,
}


class HyperpropertyCertificationError(ValueError):
    """Raised when hyperproperty toolchain certification fails closed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One hermetic hyperproperty check outcome."""

    check_id: str
    kind: str
    status: str
    expected: str
    observed: str
    detail: str = ""
    engine_id: str = ""
    authority: str = AUTHORITY_CEILING
    is_theorem_authority: bool = False
    authorizes_universal_proof: bool = False
    quarantined: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise HyperpropertyCertificationError(f"unknown check kind {self.kind!r}")
        if self.status not in {
            "passed",
            "failed",
            "quarantined",
            "error",
            "skipped",
        }:
            raise HyperpropertyCertificationError(
                f"unknown check status {self.status!r}"
            )
        if self.is_theorem_authority:
            raise HyperpropertyCertificationError(
                "hyperproperty checks cannot claim theorem authority"
            )
        if self.authorizes_universal_proof:
            raise HyperpropertyCertificationError(
                "hyperproperty checks cannot authorize universal proof"
            )
        if self.authority not in {AUTHORITY_CEILING, "bounded", "hyperproperty"}:
            raise HyperpropertyCertificationError(
                "hyperproperty checks must report bounded authority"
            )

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority or AUTHORITY_CEILING,
            "authorizes_universal_proof": False,
            "check_id": self.check_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "expected": self.expected,
            "is_theorem_authority": False,
            "kind": self.kind,
            "observed": self.observed,
            "quarantined": self.quarantined,
            "status": self.status,
        }


@dataclass
class EngineRunRecord:
    """One engine evaluation for certification comparison."""

    engine_id: str
    case_id: str
    outcome: str
    status: str
    expected: str
    agreed: bool
    timed_out: bool = False
    malformed: bool = False
    detail: str = ""
    executable: str = ""
    engine_version: str = ""
    document_digest: str = ""
    quantifier_signature: tuple[str, ...] = ()
    observation_fields: tuple[str, ...] = ()
    authority: str = AUTHORITY_CEILING
    is_theorem_authority: bool = False
    authorizes_universal_proof: bool = False
    quarantined: bool = False
    counterexample_traces: int = 0
    translation_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreed": self.agreed,
            "authority": self.authority,
            "authorizes_universal_proof": False,
            "case_id": self.case_id,
            "counterexample_traces": self.counterexample_traces,
            "detail": self.detail,
            "document_digest": self.document_digest,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "executable": self.executable,
            "expected": self.expected,
            "is_theorem_authority": False,
            "malformed": self.malformed,
            "observation_fields": list(self.observation_fields),
            "outcome": self.outcome,
            "quantifier_signature": list(self.quantifier_signature),
            "quarantined": self.quarantined,
            "status": self.status,
            "timed_out": self.timed_out,
            "translation_preserved": self.translation_preserved,
        }


@dataclass
class EngineCertification:
    """Per-engine bounded hyperproperty certification summary."""

    engine_id: str
    version: str
    executable: str
    usable: bool
    certified: bool
    role: str
    authority_ceiling: str
    checks: list[CheckResult] = field(default_factory=list)
    case_results: list[EngineRunRecord] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)
    install_status: str = ""
    authorizes_universal_proof: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "authorizes_universal_proof": False,
            "block_reasons": list(self.block_reasons),
            "case_results": [item.to_dict() for item in self.case_results],
            "certified": self.certified,
            "checks": [item.to_dict() for item in self.checks],
            "engine_id": self.engine_id,
            "executable": self.executable,
            "install_status": self.install_status,
            "role": self.role,
            "usable": self.usable,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Compact corpus recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One compact certification recipe (not a bulk golden dump)."""

    case_id: str
    category: str
    expected: str
    mutation_kind: str = ""
    notes: str = ""
    force_verdict: str = ""
    observations: tuple[str, ...] = ("status", "public_token")
    quantifier_signature: tuple[str, ...] = ("forall", "forall")
    max_traces: int = 8
    max_pairs: int = 16
    base_case_id: str = ""


def default_case_specs() -> tuple[CaseSpec, ...]:
    """Compact recipes covering FVT-G170 acceptance categories."""

    return (
        CaseSpec(
            case_id="case:ni_holds",
            category="satisfaction",
            expected="satisfied",
            notes="Two-trace noninterference holds under hermetic model",
        ),
        CaseSpec(
            case_id="case:ni_violated",
            category="violation",
            expected="violated",
            force_verdict="violated",
            notes="Violating multi-trace tuple with observation projection",
        ),
        CaseSpec(
            case_id="case:mutation_observation",
            category="mutation",
            expected="satisfied",
            mutation_kind="observation",
            observations=("status",),
            base_case_id="case:ni_holds",
            notes="Observation projection mutation preserves quantifiers",
        ),
        CaseSpec(
            case_id="case:mutation_quantifier",
            category="mutation",
            expected="satisfied",
            mutation_kind="quantifier",
            quantifier_signature=("forall", "exists"),
            base_case_id="case:ni_holds",
            notes="Quantifier signature mutation is preserved in translation",
        ),
        CaseSpec(
            case_id="case:replay_holds",
            category="replay",
            expected="satisfied",
            notes="Deterministic satisfaction replay",
        ),
        CaseSpec(
            case_id="case:bounds_exact",
            category="bounds",
            expected="satisfied",
            max_traces=4,
            max_pairs=8,
            notes="Exact self-composition bounds retained; no universal claim",
        ),
    )


def _policy(
    *,
    observations: tuple[str, ...] = ("status", "public_token"),
) -> InformationFlowPolicy:
    labels: list[SecurityLabel] = [
        SecurityLabel("label:user", "user_id", SecurityLevel.LOW, ObservationKind.INPUT),
        SecurityLabel(
            "label:secret", "secret", SecurityLevel.HIGH, ObservationKind.INPUT
        ),
    ]
    for field_name in observations:
        labels.append(
            SecurityLabel(
                f"label:{field_name}",
                field_name,
                SecurityLevel.LOW,
                ObservationKind.OUTPUT,
            )
        )
    observation_specs = tuple(
        ObservationSpec(
            f"obs:{field}",
            field,
            ObservationKind.OUTPUT,
            SecurityLevel.LOW,
        )
        for field in observations
    )
    return InformationFlowPolicy(
        policy_id="policy:ni-cert-v1",
        low_input_fields=("user_id",),
        high_input_fields=("secret",),
        observation_fields=observations,
        labels=tuple(labels),
        observations=observation_specs,
        subject_fields=("task_id",),
        description="Certification noninterference policy",
    )


def _bound(*, max_traces: int = 8, max_pairs: int = 16) -> SelfCompositionBound:
    return SelfCompositionBound(
        "bound:cert-finite",
        max_traces=max_traces,
        max_pairs=max_pairs,
        max_steps=64,
        description="Certification finite self-composition envelope",
    )


def materialize_document(spec: CaseSpec) -> HyperpropertyIR:
    """Build a HyperpropertyIR document from a compact recipe."""

    if spec.quantifier_signature == ("forall", "forall") and not spec.mutation_kind:
        return HyperpropertyIR.noninterference_document(
            policy=_policy(observations=spec.observations),
            bound=_bound(max_traces=spec.max_traces, max_pairs=spec.max_pairs),
            metadata={
                "case_id": spec.case_id,
                "category": spec.category,
                "expected": spec.expected,
            },
        )

    variables = tuple(
        TraceVariable(f"var:pi{index + 1}", f"pi{index + 1}")
        for index in range(len(spec.quantifier_signature))
    )
    quantifiers = tuple(
        TraceQuantifier(token) for token in spec.quantifier_signature
    )
    prefix = tuple(
        QuantifierBinding(
            f"bind:{index}",
            quantifiers[index],
            variables[index].variable_id,
            index,
        )
        for index in range(len(variables))
    )
    matrix = " ".join(
        f"{item.quantifier.value} {variables[index].name}."
        for index, item in enumerate(prefix)
    ) + " true"
    formula = HyperpropertyFormula(
        formula_id=f"formula:{spec.case_id}",
        kind=HyperpropertyKind.GENERAL,
        variables=variables,
        quantifier_prefix=prefix,
        matrix_statement=matrix,
    )
    return HyperpropertyIR(
        formula=formula,
        information_flow_policy=_policy(observations=spec.observations),
        self_composition_bound=_bound(
            max_traces=spec.max_traces, max_pairs=spec.max_pairs
        ),
        metadata={
            "case_id": spec.case_id,
            "category": spec.category,
            "expected": spec.expected,
            "mutation_kind": spec.mutation_kind,
        },
    )


def backend_for(
    engine_id: str,
    *,
    executable: str,
    runner: BoundedToolRunner | None = None,
) -> HyperpropertyBackend:
    cls = _BACKEND_TYPES.get(engine_id)
    if cls is None:
        raise HyperpropertyCertificationError(f"unknown engine {engine_id!r}")
    return cls(executable=executable, runner=runner or BoundedToolRunner())


def _runner_with_env(extra: Mapping[str, str] | None = None) -> BoundedToolRunner:
    """BoundedToolRunner that forwards hermetic certification env controls.

    The default runner only inherits a tiny allowlist (PATH/LANG/…).  Hermetic
    force/disagree/malformed/timeout probes therefore must be injected via
    ``base_environment``.
    """

    base: dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "HOME", "TMPDIR"):
        value = os.environ.get(key)
        if value is not None:
            base[key] = value
    if extra:
        base.update({str(k): str(v) for k, v in extra.items()})
    return BoundedToolRunner(base_environment=base)


def run_engine_case(
    engine_id: str,
    case_id: str,
    document: HyperpropertyIR | None,
    *,
    executable: str,
    engine_version: str = "",
    expected: str = "satisfied",
    force_verdict: str = "",
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 5.0,
    expect_error: bool = False,
    runner: BoundedToolRunner | None = None,
) -> EngineRunRecord:
    """Run one certification case on one pin-bound engine."""

    run_env: dict[str, str] = {}
    if force_verdict:
        run_env[hyper_installer.ENV_FORCE_VERDICT] = force_verdict
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})
    run_env.setdefault(hyper_installer.ENV_CASE_ID, case_id)

    if expect_error or document is None:
        tool_runner = runner or _runner_with_env(run_env)
        with tempfile.TemporaryDirectory(prefix="hyper-malformed-") as tmp:
            bad = Path(tmp) / "property.hltl"
            bad.write_text("{not a valid hyperltl formula@@@@\n", encoding="utf-8")
            request_env = dict(run_env)
            request_env.setdefault(hyper_installer.ENV_MALFORMED, "1")
            request = ToolRunRequest(
                argv=(executable, str(bad)),
                runtime=ToolRuntime.NATIVE,
                limits=ToolRunLimits(
                    timeout_seconds=timeout_seconds,
                    cpu_seconds=timeout_seconds,
                    memory_bytes=64 * 1024 * 1024,
                    max_output_bytes=64 * 1024,
                    max_input_bytes=16 * 1024,
                    max_workspace_bytes=64 * 1024,
                ),
                environment=request_env,
            )
            try:
                result = tool_runner.run(request)
            except Exception as exc:
                return EngineRunRecord(
                    engine_id=engine_id,
                    case_id=case_id,
                    outcome="error",
                    status="error",
                    expected="error",
                    agreed=True,
                    malformed=True,
                    detail=str(exc)[:240],
                    executable=executable,
                    engine_version=engine_version,
                    quarantined=True,
                )
            combined = f"{result.stdout or ''}\n{result.stderr or ''}"
            folded = combined.casefold()
            looks_satisfied = any(
                marker in folded
                for marker in ("holds", "verified", "satisfied", "sat", "true")
            ) and "violat" not in folded
            malformed = bool(
                result.timed_out is False
                and (
                    "%%%" in combined
                    or not looks_satisfied
                    or "malformed" in folded
                )
            )
            if looks_satisfied and "%%%" not in combined:
                return EngineRunRecord(
                    engine_id=engine_id,
                    case_id=case_id,
                    outcome="satisfied",
                    status="unexpected_success",
                    expected="error",
                    agreed=False,
                    malformed=True,
                    detail="malformed input produced satisfaction",
                    executable=executable,
                    engine_version=engine_version,
                    quarantined=True,
                )
            return EngineRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                outcome="error" if malformed else "unknown",
                status="error",
                expected="error",
                agreed=True,
                malformed=True,
                detail="malformed input fail-closed",
                executable=executable,
                engine_version=engine_version,
                quarantined=True,
            )

    tool_runner = runner or _runner_with_env(run_env)
    backend = backend_for(engine_id, executable=executable, runner=tool_runner)
    translation = backend.translate(document)
    quantifier_ok = translation.quantifier_order.matches_document(document)
    obs_ok = (
        translation.observation_map.observation_fields
        == document.information_flow_policy.observation_fields
    )
    translation_preserved = quantifier_ok and obs_ok

    if hyper_installer.ENV_SLEEP_SECONDS in run_env:
        from ipfs_datasets_py.logic.ir_core.protocols import (
            BackendRequest,
            ExecutionBounds,
            QueryKind,
        )
        from ipfs_datasets_py.logic.ir_core.claims import FrozenMap

        request = BackendRequest(
            request_id=f"request:{case_id}",
            claim_id=f"claim:{case_id}",
            declaration_id=f"declaration:{case_id}",
            claim_digest="a" * 64,
            obligation_id=f"obligation:{case_id}",
            obligation_digest="b" * 64,
            assumption_ids=("assumption:reviewed",),
            logic_family="hyperproperty",
            query_kind=QueryKind.SATISFIABILITY,
            bounds=ExecutionBounds(
                timeout_ms=max(1, int(timeout_seconds * 1000)),
                max_steps=20,
            ),
            payload=FrozenMap({"document": document.to_dict()}),
            requested_backend_id=engine_id,
        )
        outcome = backend.run(request)
    else:
        outcome = backend.check(document)

    receipt = outcome.receipt
    status_token = receipt.status.value
    if receipt.status is HyperCheckOutcomeStatus.TIMEOUT:
        return EngineRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            outcome="timeout",
            status="timeout",
            expected=expected,
            agreed=False,
            timed_out=True,
            detail=receipt.reason,
            executable=executable,
            engine_version=engine_version or receipt.tool_version,
            document_digest=translation.document_digest,
            quantifier_signature=translation.quantifier_order.signature,
            observation_fields=translation.observation_map.observation_fields,
            quarantined=True,
            translation_preserved=translation_preserved,
        )

    if receipt.status is HyperCheckOutcomeStatus.MALFORMED:
        return EngineRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            outcome="error",
            status="error",
            expected=expected,
            agreed=False,
            malformed=True,
            detail=receipt.reason,
            executable=executable,
            engine_version=engine_version or receipt.tool_version,
            document_digest=translation.document_digest,
            quantifier_signature=translation.quantifier_order.signature,
            observation_fields=translation.observation_map.observation_fields,
            quarantined=True,
            translation_preserved=translation_preserved,
        )

    # Unknown without recognized markers under force-malformed is treated malformed.
    if (
        receipt.status is HyperCheckOutcomeStatus.UNKNOWN
        and "%%%" in (receipt.stdout or "")
    ):
        return EngineRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            outcome="error",
            status="error",
            expected=expected,
            agreed=False,
            malformed=True,
            detail="unrecognized malformed output",
            executable=executable,
            engine_version=engine_version or receipt.tool_version,
            document_digest=translation.document_digest,
            quantifier_signature=translation.quantifier_order.signature,
            observation_fields=translation.observation_map.observation_fields,
            quarantined=True,
            translation_preserved=translation_preserved,
        )

    # Map satisfied/violated to expected tokens.
    if receipt.status is HyperCheckOutcomeStatus.SATISFIED:
        observed = "satisfied"
    elif receipt.status is HyperCheckOutcomeStatus.VIOLATED:
        observed = "violated"
    elif receipt.status is HyperCheckOutcomeStatus.UNSUPPORTED:
        observed = "unsupported"
    else:
        observed = status_token

    agreed = observed == expected
    cex_traces = (
        len(receipt.counterexample.traces)
        if receipt.counterexample is not None
        else 0
    )
    if receipt.authorizes_universal_proof:
        agreed = False
    if outcome.result.authority is not ResultAuthority.HYPERPROPERTY:
        agreed = False
    if outcome.result.translation_ceiling is not EvidenceAuthority.BOUNDED:
        agreed = False

    return EngineRunRecord(
        engine_id=engine_id,
        case_id=case_id,
        outcome=observed,
        status="agreed" if agreed else "disagreement",
        expected=expected,
        agreed=agreed,
        detail="" if agreed else f"expected {expected}, got {observed}",
        executable=executable,
        engine_version=engine_version or receipt.tool_version,
        document_digest=translation.document_digest,
        quantifier_signature=translation.quantifier_order.signature,
        observation_fields=translation.observation_map.observation_fields,
        quarantined=not agreed,
        counterexample_traces=cex_traces,
        translation_preserved=translation_preserved,
        authorizes_universal_proof=False,
    )


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def _install_engines(
    *,
    install_root: Path | str | None,
    force: bool = False,
) -> hyper_installer.HyperpropertyInstallBundle:
    return hyper_installer.ensure_hyperproperty(
        yes=True,
        strict=True,
        force=force,
        install_root=install_root,
        hermetic_engine=True,
        checksum_verified=True,
    )


def certify_engine(
    engine_id: str,
    *,
    identity: hyper_installer.EngineIdentity,
    install_status: str = "installed",
    specs: Sequence[CaseSpec] | None = None,
) -> EngineCertification:
    """Run the full bounded hyperproperty matrix for one pin-bound engine."""

    selected = tuple(specs or default_case_specs())
    checks: list[CheckResult] = []
    records: list[EngineRunRecord] = []
    block_reasons: list[str] = []

    try:
        role = get_tool_role(engine_id)
        role_ok = (
            role.role is ToolRole.AUTHORITY
            and role.authority_ceiling is ToolchainAuthorityCeiling.BOUNDED
        )
    except Exception as exc:
        role_ok = False
        block_reasons.append(f"role_lookup_failed:{type(exc).__name__}")
        role = None  # type: ignore[assignment]

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.role.authority_bounded",
            kind="role",
            status="passed" if role_ok else "failed",
            expected="authority/bounded",
            observed=(
                f"{role.role.value}/{role.authority_ceiling.value}"
                if role is not None
                else "unavailable"
            ),
            detail="hyperproperty engines retain bounded authority",
            engine_id=engine_id,
        )
    )
    if not role_ok:
        block_reasons.append("role_not_authority_bounded")

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.install.strict_pin",
            kind="install",
            status="passed" if identity.version else "failed",
            expected=identity.version,
            observed=identity.version,
            detail=f"executable={identity.executable}",
            engine_id=engine_id,
        )
    )

    usable = Path(identity.executable).is_file()
    if not usable:
        block_reasons.append("executable_missing")

    category_seen: set[str] = set()
    mutation_seen: set[str] = set()

    # ---- translation preservation (quantifiers + observation projections)
    base_doc = materialize_document(
        CaseSpec(
            case_id="case:translation",
            category="satisfaction",
            expected="satisfied",
        )
    )
    backend = backend_for(engine_id, executable=identity.executable)
    translation = backend.translate(base_doc)
    order = translation.quantifier_order
    obs = translation.observation_map
    formula_text = translation.formula_text
    translation_ok = (
        order.matches_document(base_doc)
        and obs.observation_fields == base_doc.information_flow_policy.observation_fields
        and "forall pi1." in formula_text
        and "forall pi2." in formula_text
        and formula_text.index("forall pi1.") < formula_text.index("forall pi2.")
        and "status" in formula_text
        and "public_token" in formula_text
        and "secret" not in formula_text
    )
    # Round-trip auxiliary JSON packages.
    restored_order = QuantifierOrder.from_dict(
        json.loads(translation.auxiliary_files["quantifier_order.json"])
    )
    restored_obs = ObservationMap.from_dict(
        json.loads(translation.auxiliary_files["observation_map.json"])
    )
    translation_ok = translation_ok and restored_order.to_dict() == order.to_dict()
    translation_ok = translation_ok and restored_obs.to_dict() == obs.to_dict()
    if engine_id == TOOL_AUTOHYPER:
        translation_ok = translation_ok and "system.explicit" in translation.auxiliary_files

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.translation.quantifiers_observations",
            kind="translation",
            status="passed" if translation_ok else "failed",
            expected="preserved quantifier order + observation projection",
            observed="preserved" if translation_ok else "drift",
            detail="quantifiers and observation projections survive translation",
            engine_id=engine_id,
        )
    )
    if not translation_ok:
        block_reasons.append("translation_not_preserved")

    # ---- positive / violation / replay / bounds corpus
    for spec in selected:
        if spec.category not in {
            "satisfaction",
            "violation",
            "replay",
            "bounds",
        }:
            continue
        document = materialize_document(spec)
        record = run_engine_case(
            engine_id,
            spec.case_id,
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected=spec.expected,
            force_verdict=spec.force_verdict,
        )
        records.append(record)
        category_seen.add(spec.category)

        kind = (
            "positive"
            if spec.category in {"satisfaction", "replay", "bounds"}
            else "violation"
        )
        ok = (
            record.agreed
            and record.outcome == spec.expected
            and not record.is_theorem_authority
            and not record.authorizes_universal_proof
            and record.translation_preserved
        )
        if spec.category == "violation":
            ok = ok and record.counterexample_traces >= 2
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.{kind}",
                kind=kind,
                status="passed" if ok else "failed",
                expected=spec.expected,
                observed=record.outcome,
                detail=spec.notes or record.detail,
                engine_id=engine_id,
                quarantined=record.quarantined,
            )
        )
        if not ok:
            block_reasons.append(f"{kind}_failed:{spec.case_id}")

        if spec.category == "bounds":
            bound = document.self_composition_bound
            bounds_ok = (
                bound.max_traces == spec.max_traces
                and bound.max_pairs == spec.max_pairs
                and not record.authorizes_universal_proof
            )
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.bounds",
                    kind="bounds",
                    status="passed" if bounds_ok else "failed",
                    expected=f"max_traces={spec.max_traces},max_pairs={spec.max_pairs}",
                    observed=(
                        f"max_traces={bound.max_traces},max_pairs={bound.max_pairs}"
                    ),
                    detail="exact bounds retained; no universal claim",
                    engine_id=engine_id,
                )
            )
            if not bounds_ok:
                block_reasons.append(f"bounds_failed:{spec.case_id}")
            category_seen.add("bounds")

        if spec.category in {"satisfaction", "replay"}:
            # Deterministic replay of the same document.
            replay = run_engine_case(
                engine_id,
                f"{spec.case_id}:replay",
                document,
                executable=identity.executable,
                engine_version=identity.version,
                expected=spec.expected,
                force_verdict=spec.force_verdict,
            )
            records.append(replay)
            replay_ok = (
                replay.outcome == record.outcome
                and replay.document_digest == record.document_digest
                and replay.agreed == record.agreed
            )
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.replay",
                    kind="replay",
                    status="passed" if replay_ok else "failed",
                    expected=record.outcome,
                    observed=replay.outcome,
                    detail="engine replay must be deterministic",
                    engine_id=engine_id,
                )
            )
            if not replay_ok:
                block_reasons.append(f"replay_unstable:{spec.case_id}")
            category_seen.add("replay")

    # ---- semantic mutations
    base_holds = materialize_document(
        CaseSpec(case_id="case:ni_holds", category="satisfaction", expected="satisfied")
    )
    base_record = run_engine_case(
        engine_id,
        "case:ni_holds:baseline",
        base_holds,
        executable=identity.executable,
        engine_version=identity.version,
        expected="satisfied",
    )
    records.append(base_record)

    for spec in selected:
        if spec.category != "mutation":
            continue
        if spec.mutation_kind not in REQUIRED_MUTATION_KINDS:
            continue
        document = materialize_document(spec)
        # Translation of mutation must reflect the changed projection/signature.
        mut_backend = backend_for(engine_id, executable=identity.executable)
        mut_translation = mut_backend.translate(document)
        supported, unsupported_reason = mut_backend.supports_prefix(document)
        # Engines that reject the mutated quantifier fragment must report
        # unsupported rather than invent a satisfaction.
        run_expected = spec.expected if supported else "unsupported"
        mutated = run_engine_case(
            engine_id,
            spec.case_id,
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected=run_expected,
            force_verdict=spec.force_verdict if supported else "",
        )
        records.append(mutated)
        mutation_seen.add(spec.mutation_kind)
        category_seen.add("mutation")

        if spec.mutation_kind == "observation":
            structure_changed = (
                mut_translation.observation_map.observation_fields
                != ObservationMap.from_document(base_holds).observation_fields
            )
            structure_ok = structure_changed and mut_translation.quantifier_order.matches_document(
                document
            )
        else:
            structure_changed = (
                mut_translation.quantifier_order.signature
                != QuantifierOrder.from_document(base_holds).signature
            )
            structure_ok = structure_changed and mut_translation.quantifier_order.matches_document(
                document
            )
            if not supported:
                # Rejecting the mutated fragment is a valid mutation signal.
                structure_ok = structure_changed

        ok = (
            structure_ok
            and mutated.translation_preserved
            and not mutated.authorizes_universal_proof
            and mutated.agreed
            and mutated.outcome in {"satisfied", "violated", "unsupported"}
        )
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.mutation",
                kind="mutation",
                status="passed" if ok else "failed",
                expected=f"mutation_kind={spec.mutation_kind};outcome={run_expected}",
                observed=mutated.outcome,
                detail=(
                    spec.notes
                    or f"structure_changed={structure_changed}; supported={supported}; "
                    f"{unsupported_reason}"
                ),
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"mutation_failed:{spec.case_id}")

    missing_mutations = sorted(REQUIRED_MUTATION_KINDS - mutation_seen)
    if missing_mutations:
        block_reasons.append(f"missing_mutations:{','.join(missing_mutations)}")

    # ---- malformed output fail-closed
    malformed = run_engine_case(
        engine_id,
        "case:malformed",
        None,
        executable=identity.executable,
        engine_version=identity.version,
        expected="error",
        expect_error=True,
    )
    records.append(malformed)
    category_seen.add("malformed")
    malformed_ok = (
        malformed.outcome != "satisfied"
        and malformed.malformed
        and malformed.quarantined
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:malformed.malformed",
            kind="malformed",
            status="passed" if malformed_ok else "failed",
            expected="error|quarantine (never satisfied)",
            observed=malformed.outcome,
            detail=malformed.detail,
            engine_id=engine_id,
            quarantined=malformed.quarantined,
        )
    )
    if not malformed_ok:
        block_reasons.append("malformed_not_fail_closed")

    # ---- timeout probe
    timed = run_engine_case(
        engine_id,
        "case:timeout",
        base_holds,
        executable=identity.executable,
        engine_version=identity.version,
        expected="satisfied",
        timeout_seconds=0.25,
        env={hyper_installer.ENV_SLEEP_SECONDS: "2.0"},
    )
    records.append(timed)
    category_seen.add("timeout")
    timeout_ok = timed.timed_out and timed.quarantined
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:timeout.timeout",
            kind="timeout",
            status="passed" if timeout_ok else "failed",
            expected="timeout+quarantine",
            observed=timed.outcome,
            detail=timed.detail or "bounded timeout must fire",
            engine_id=engine_id,
            quarantined=timed.quarantined,
        )
    )
    if not timeout_ok:
        block_reasons.append("timeout_not_enforced")

    # ---- deliberate disagreement must quarantine promotion
    disagree = run_engine_case(
        engine_id,
        "case:disagreement",
        base_holds,
        executable=identity.executable,
        engine_version=identity.version,
        expected="satisfied",
        env={hyper_installer.ENV_DISAGREE: "1"},
    )
    records.append(disagree)
    category_seen.add("disagreement")
    disagree_ok = (
        not disagree.agreed
        and disagree.quarantined
        and disagree.outcome != disagree.expected
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:disagreement.disagreement_quarantine",
            kind="disagreement_quarantine",
            status="passed" if disagree_ok else "failed",
            expected="disagreement+quarantine",
            observed=f"{disagree.outcome} vs {disagree.expected}",
            detail="any disagreement quarantines promotion",
            engine_id=engine_id,
            quarantined=disagree.quarantined,
        )
    )
    if not disagree_ok:
        block_reasons.append("disagreement_not_quarantined")

    # Authority: bounded only, never theorem / universal.
    authority_ok = (
        identity.role == AUTHORITY_ROLE
        and identity.authority_ceiling == AUTHORITY_CEILING
        and identity.authorizes_universal_proof is False
        and all(
            not record.is_theorem_authority and not record.authorizes_universal_proof
            for record in records
        )
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.authority.bounded_only",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="authority/bounded; no universal claims",
            observed=f"{identity.role}/{identity.authority_ceiling}",
            detail="results retain bounded hyperproperty authority",
            engine_id=engine_id,
        )
    )
    if not authority_ok:
        block_reasons.append("authority_breach")

    missing_categories = sorted(
        {"satisfaction", "violation", "mutation", "replay", "malformed", "disagreement", "timeout", "bounds"}
        - category_seen
    )
    if missing_categories:
        block_reasons.append(f"missing_categories:{','.join(missing_categories)}")

    all_passed = all(item.passed for item in checks) and not block_reasons and usable
    return EngineCertification(
        engine_id=engine_id,
        version=identity.version,
        executable=identity.executable,
        usable=usable,
        certified=all_passed,
        role=AUTHORITY_ROLE,
        authority_ceiling=AUTHORITY_CEILING,
        checks=checks,
        case_results=records,
        block_reasons=sorted(set(block_reasons)),
        install_status=install_status,
        authorizes_universal_proof=False,
    )


def certify_hyperproperty_toolchains(
    *,
    install_root: Path | str | None = None,
    engines: Sequence[str] | None = None,
    force_install: bool = False,
    skip_install: bool = False,
    identities: Mapping[str, hyper_installer.EngineIdentity] | None = None,
) -> dict[str, Any]:
    """Run full hyperproperty toolchain certification for FVT-G170."""

    selected = tuple(engines or EXTERNAL_ENGINES)
    install_bundle: hyper_installer.HyperpropertyInstallBundle | None = None
    resolved_identities: dict[str, hyper_installer.EngineIdentity] = {}
    install_statuses: dict[str, str] = {}

    if identities:
        resolved_identities = dict(identities)
        for tool_id in selected:
            install_statuses[tool_id] = "provided"
    elif skip_install:
        root = hyper_installer._expand_install_root(install_root)
        for tool_id in selected:
            pin = hyper_installer.pin_for_tool(tool_id)
            identity = hyper_installer._identity_from_disk(tool_id, root, pin)
            if identity is None:
                raise HyperpropertyCertificationError(
                    f"skip_install requested but {tool_id} is not installed under {root}"
                )
            resolved_identities[tool_id] = identity
            install_statuses[tool_id] = "already_present"
    else:
        install_bundle = _install_engines(
            install_root=install_root,
            force=force_install,
        )
        if not install_bundle.ok:
            raise HyperpropertyCertificationError(
                "strict installation failed: "
                + "; ".join(
                    f"{r.tool_id}:{r.status}:{r.detail}" for r in install_bundle.receipts
                )
            )
        for receipt in install_bundle.receipts:
            if receipt.identity is None:
                continue
            resolved_identities[receipt.tool_id] = receipt.identity
            install_statuses[receipt.tool_id] = receipt.status

    engine_results: list[EngineCertification] = []
    for engine_id in selected:
        identity = resolved_identities.get(engine_id)
        if identity is None:
            raise HyperpropertyCertificationError(
                f"no installed identity for {engine_id!r}"
            )
        pin = hyper_installer.pin_for_tool(engine_id)
        if identity.version != pin["version"]:
            raise HyperpropertyCertificationError(
                f"strict pin mismatch for {engine_id}: "
                f"{identity.version!r} != {pin['version']!r}"
            )
        engine_results.append(
            certify_engine(
                engine_id,
                identity=identity,
                install_status=install_statuses.get(engine_id, "installed"),
            )
        )

    all_certified = bool(engine_results) and all(item.certified for item in engine_results)
    # Deliberate disagreement / timeout / malformed probes are expected to
    # quarantine; they must not demote an otherwise certified corpus.
    corpus_disagreement = any(
        (not record.agreed)
        and not record.timed_out
        and not record.malformed
        and record.case_id
        not in {
            "case:disagreement",
            "case:timeout",
            "case:malformed",
        }
        and not record.case_id.endswith(":replay")
        and not record.case_id.endswith(":baseline")
        for engine in engine_results
        for record in engine.case_results
    )
    if corpus_disagreement:
        all_certified = False

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_universal_claims_beyond_bounds": True,
        "certified": all_certified,
        "engines": [item.to_dict() for item in engine_results],
        "engine_ids": [item.engine_id for item in engine_results],
        "external_engines": list(EXTERNAL_ENGINES),
        "categories_exercised": sorted(REQUIRED_CATEGORIES),
        "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "install": None if install_bundle is None else install_bundle.to_dict(),
        "policy": {
            "strict_installation_selects_reviewed_pins": True,
            "quantifiers_and_observation_projections_preserved": True,
            "disagreement_quarantines_promotion": True,
            "never_grants_theorem_authority": True,
            "never_authorizes_universal_proof": True,
            "cannot_make_universal_claims_beyond_bounds": True,
            "authority_ceiling": AUTHORITY_CEILING,
            "no_central_certificate_edit": True,
            "grants_theorem_authority": False,
            "authorizes_universal_proof": False,
        },
        "digest_sha256": "",
    }
    digest_body = {
        key: value
        for key, value in payload.items()
        if key != "digest_sha256"
    }
    payload["digest_sha256"] = hashlib.sha256(
        json.dumps(digest_body, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return payload


def hyperproperty_lane_handler(
    *,
    install_root: Path | str | None = None,
    skip_install: bool = False,
    force_install: bool = False,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for ``hyperproperty_toolchain_certification@1``."""

    certificate = certify_hyperproperty_toolchains(
        install_root=install_root,
        skip_install=skip_install,
        force_install=force_install,
    )
    return {
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "status": "certified" if certificate["certified"] else "failed",
        "certified": certificate["certified"],
        "authority_ceiling": AUTHORITY_CEILING,
        "grants_theorem_authority": False,
        "authorizes_universal_proof": False,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "engine_ids": certificate["engine_ids"],
        "certificate": certificate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Certify HyperLTL / AutoHyper / MCHyper toolchains (FVT-G170)"
    )
    parser.add_argument(
        "--install-root",
        default=None,
        help="User-local install root for pin-bound hermetic engines",
    )
    parser.add_argument(
        "--force-install",
        action="store_true",
        help="Force re-materialization of hermetic engines",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse engines already present under install-root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the certification receipt as JSON on stdout",
    )
    args = parser.parse_args(argv)
    try:
        receipt = certify_hyperproperty_toolchains(
            install_root=args.install_root,
            force_install=args.force_install,
            skip_install=args.skip_install,
        )
    except Exception as exc:
        print(f"hyperproperty certification failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        status = "CERTIFIED" if receipt["certified"] else "FAILED"
        print(
            f"[{status}] {INTERFACE} engines={receipt['engine_ids']} "
            f"authority={receipt['authority_ceiling']}"
        )
    return 0 if receipt["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "HANDLER_ID",
    "CERTIFICATION_SURFACE",
    "AUTHORITY_CEILING",
    "EXTERNAL_ENGINES",
    "REQUIRED_CATEGORIES",
    "REQUIRED_MUTATION_KINDS",
    "CaseSpec",
    "CheckResult",
    "EngineCertification",
    "EngineRunRecord",
    "HyperpropertyCertificationError",
    "backend_for",
    "certify_engine",
    "certify_hyperproperty_toolchains",
    "default_case_specs",
    "hyperproperty_lane_handler",
    "main",
    "materialize_document",
    "run_engine_case",
]

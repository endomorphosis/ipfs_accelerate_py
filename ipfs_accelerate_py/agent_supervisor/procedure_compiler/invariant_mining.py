"""Invariant candidate mining and independent non-vacuity validation.

``InvariantMiner`` proposes invariant-kind properties only.  It reuses
specification mining's closed source schema, provenance/tier retention, and
fail-closed conflict rule.  Mining never certifies or promotes candidates.

``NonVacuityValidator`` and ``InvariantValidator`` are the later independent
obligations: they run bounded hermetic vacuity campaigns through
``AssuranceApiAdapter`` (the existing ``AssuranceCampaignApi@1``), reject
vacuous candidates, persist counterexamples, and leave survivors in candidate
state with current bindings and adversarial receipts.  Survivors cannot claim
completeness beyond the tested obligations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final

from .contracts import (
    MAX_ITEMS,
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    ExecutionTrajectory,
    ProcedureSpec,
    TaskFamily,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _strings,
    content_identity,
)
from .contracts import (
    InvariantCandidate as InvariantCandidateArtifact,
)
from .contracts import (
    InvariantValidationReceipt as InvariantValidationReceiptArtifact,
)
from .contracts import (
    NonVacuityReceipt as NonVacuityReceiptArtifact,
)
from .specification_mining import (
    AdmittedSource,
    CandidateStatus,
    EvidenceTier,
    MiningSource,
    PropertyKind,
    PropertyNomination,
    SourceKind,
    SpecificationCandidate,
    SpecificationCounterexample,
    SpecificationMiner,
    SpecificationMiningError,
    SpecificationMiningResult,
)

INVARIANT_MINER_REVISION: Final[str] = "invariant-miner@1"
NON_VACUITY_VALIDATOR_REVISION: Final[str] = "non-vacuity-validator@1"
INVARIANT_VALIDATOR_REVISION: Final[str] = "invariant-validator@1"
ASSURANCE_API_ADAPTER_REVISION: Final[str] = "assurance-api-adapter@1"
ASSURANCE_CAMPAIGN_API_INTERFACE_PIN: Final[str] = "AssuranceCampaignApi@1"
ANALYZE_VACUITY_COMMAND: Final[str] = "analyze_vacuity"
VACUITY_TEST_FAMILY: Final[str] = "test"


class InvariantMiningError(SpecificationMiningError):
    """Invariant candidates could not be mined or non-vacuity-validated."""


@dataclass(frozen=True)
class InvariantMiningResult:
    """Invariant-only mining output with candidate-status wire artifacts."""

    bindings: ArtifactBindings
    candidates: tuple[SpecificationCandidate, ...]
    refused: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    invariant_artifacts: tuple[InvariantCandidateArtifact, ...]
    retained_source_kinds: tuple[SourceKind, ...]
    retained_evidence_tiers: tuple[EvidenceTier, ...]

    @property
    def upgraded_count(self) -> int:
        return 0


def _invariant_nomination(
    *,
    property_id: str,
    binding: str,
    operator: ConditionOperator,
    operand: object,
    evidence_cid: str,
) -> PropertyNomination:
    return PropertyNomination(
        property_kind=PropertyKind.INVARIANT,
        property_id=property_id,
        binding=binding,
        operator=operator,
        operand=operand,
        evidence_cid=evidence_cid,
        required=True,
    )


def project_invariant_sources(sources: Sequence[MiningSource]) -> tuple[AdmittedSource, ...]:
    """Project extra invariant nominations that hold across a source artifact."""

    extra: list[AdmittedSource] = []
    for item in sources:
        if isinstance(item, AdmittedSource):
            continue
        if isinstance(item, ProcedureSpec):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-runtime.{item.name}",
                    source_kind=SourceKind.RUNTIME_CHECK,
                    evidence_tier=EvidenceTier.RUNTIME_OBSERVATION,
                    provenance_cid=provenance,
                    artifact_cid=provenance,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.scope-respected",
                            binding="procedure.scope_paths",
                            operator=ConditionOperator.SUBSET_OF,
                            operand=item.scope_paths,
                            evidence_cid=provenance,
                        ),
                        _invariant_nomination(
                            property_id="invariant.tree-current",
                            binding="bindings.tree_id",
                            operator=ConditionOperator.CURRENT,
                            operand=item.bindings.tree_id,
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
            continue
        if isinstance(item, ExecutionTrajectory):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-trace.{item.source_episode_cid}",
                    source_kind=SourceKind.RUNTIME_CHECK,
                    evidence_tier=EvidenceTier.RUNTIME_OBSERVATION,
                    provenance_cid=provenance,
                    artifact_cid=item.source_episode_cid,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.contiguous-state-chain",
                            binding="trajectory.steps",
                            operator=ConditionOperator.EQUALS,
                            operand=tuple(
                                step.terminal_state_cid == nxt.initial_state_cid
                                for step, nxt in zip(
                                    item.steps, item.steps[1:], strict=False
                                )
                            ),
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
            continue
        if isinstance(item, TaskFamily):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-family.{item.name}",
                    source_kind=SourceKind.TYPE,
                    evidence_tier=EvidenceTier.TYPE_DECLARATION,
                    provenance_cid=provenance,
                    artifact_cid=provenance,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.effect-ceiling",
                            binding="task_family.boundary.permitted_effect_classes",
                            operator=ConditionOperator.SUBSET_OF,
                            operand=tuple(
                                effect.value for effect in item.boundary.permitted_effect_classes
                            ),
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
    return tuple(extra)


class InvariantMiner:
    """Propose bounded invariant candidates; never certify or promote them."""

    def __init__(
        self,
        *,
        miner_revision: str = INVARIANT_MINER_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.miner_revision = _identifier(miner_revision, "miner_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self._specification_miner = SpecificationMiner(
            miner_revision=self.miner_revision,
            emitted_at_ms=self.emitted_at_ms,
        )

    def mine(self, sources: Sequence[MiningSource]) -> InvariantMiningResult:
        extra = project_invariant_sources(sources)
        result = self._specification_miner.mine((*sources, *extra))
        candidates = tuple(
            item for item in result.candidates if item.property_kind is PropertyKind.INVARIANT
        )
        refused = tuple(
            item for item in result.refused if item.property_kind is PropertyKind.INVARIANT
        )
        counterexamples = tuple(
            item
            for item in result.counterexamples
            if item.property_kind is PropertyKind.INVARIANT
        )
        artifacts = tuple(
            item.to_invariant_artifact(result.bindings, emitted_at_ms=self.emitted_at_ms)
            for item in candidates
        )
        if any(item.state is not ArtifactState.CANDIDATE for item in artifacts):
            raise InvariantMiningError("invariant candidates cannot leave candidate state")
        return InvariantMiningResult(
            bindings=result.bindings,
            candidates=candidates,
            refused=refused,
            counterexamples=counterexamples,
            invariant_artifacts=artifacts,
            retained_source_kinds=result.retained_source_kinds,
            retained_evidence_tiers=result.retained_evidence_tiers,
        )


class VacuityClass(str, Enum):
    """Closed adversarial vacuity vocabulary for specification candidates."""

    IMPOSSIBLE_PRECONDITION = "impossible_precondition"
    UNREACHABLE_BRANCH = "unreachable_branch"
    EMPTY_OUTPUT = "empty_output"
    NO_INVOCATION = "no_invocation"
    MOCK_EFFECT = "mock_effect"
    FIXTURE_SHORTCUT = "fixture_shortcut"
    CONSTANT_RESTATEMENT = "constant_restatement"
    INVARIANT_COUNTEREXAMPLE = "invariant_counterexample"


REQUIRED_VACUITY_CLASSES: Final[tuple[VacuityClass, ...]] = (
    VacuityClass.IMPOSSIBLE_PRECONDITION,
    VacuityClass.UNREACHABLE_BRANCH,
    VacuityClass.EMPTY_OUTPUT,
    VacuityClass.NO_INVOCATION,
    VacuityClass.MOCK_EFFECT,
    VacuityClass.FIXTURE_SHORTCUT,
    VacuityClass.CONSTANT_RESTATEMENT,
    VacuityClass.INVARIANT_COUNTEREXAMPLE,
)

_MOCK_MARKERS: Final[tuple[str, ...]] = ("mock", "stub", "fake-effect", "synthetic-effect")
_SHORTCUT_MARKERS: Final[tuple[str, ...]] = ("shortcut", "fixture-only", "fixture_only")
_CONSTANT_BINDINGS: Final[frozenset[str]] = frozenset(
    {"constant", "tautology", "true", "literal"}
)


class AssuranceApiStatus(str, Enum):
    AVAILABLE = "available"
    TYPED_UNAVAILABLE = "typed_unavailable"
    INCOMPATIBLE = "incompatible"


def _vacuity_class(value: Any) -> VacuityClass:
    return _enum(value, VacuityClass, "vacuity_class")


def _assurance_status(value: Any) -> AssuranceApiStatus:
    return _enum(value, AssuranceApiStatus, "assurance_api_status")


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise InvariantMiningError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _unique_identifiers(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    return _strings(tuple(values), field_name, identifiers=True, preserve_order=True)


def _empty_domain(operand: Any) -> bool:
    if operand is None or operand is False:
        return True
    if isinstance(operand, (str, bytes, bytearray, memoryview)):
        return len(operand) == 0
    if isinstance(operand, Mapping):
        return len(operand) == 0
    if isinstance(operand, Sequence):
        return len(operand) == 0
    return False


def _contains_marker(value: Any, markers: Sequence[str]) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in markers)


def _plan_cid(payload: Mapping[str, Any]) -> str:
    return content_identity(
        {"schema": "procedure-compiler/non-vacuity-campaign@1", **dict(payload)}
    )


def _reachable_step_ids(procedure: ProcedureSpec) -> frozenset[str]:
    successors: dict[str, list[str]] = {}
    for step in procedure.steps:
        successors.setdefault(step.step_id, [])
        if step.next_step_id:
            successors[step.step_id].append(step.next_step_id)
    for branch in procedure.branches:
        successors.setdefault(branch.branch_id, [])
        successors[branch.branch_id].extend((branch.true_step_id, branch.false_step_id))
        for step in procedure.steps:
            if branch.observation_id in step.evidence_outputs:
                successors.setdefault(step.step_id, []).append(branch.branch_id)
    for loop in procedure.loops:
        successors.setdefault(loop.loop_id, [])
        successors[loop.loop_id].extend((loop.body_step_id, loop.exit_step_id))
    seen: set[str] = set()
    stack = [procedure.entry_step_id]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(successors.get(node, ()))
    return frozenset(seen)


def _contradicts(left: SpecificationCandidate, right: SpecificationCandidate) -> bool:
    if left.binding != right.binding:
        return False
    if left.operator is ConditionOperator.EXISTS and right.operator is ConditionOperator.NOT_EXISTS:
        return left.operand == right.operand
    if left.operator is ConditionOperator.NOT_EXISTS and right.operator is ConditionOperator.EXISTS:
        return left.operand == right.operand
    if left.operator is ConditionOperator.EQUALS and right.operator is ConditionOperator.NOT_EQUALS:
        return left.operand == right.operand
    if left.operator is ConditionOperator.NOT_EQUALS and right.operator is ConditionOperator.EQUALS:
        return left.operand == right.operand
    if left.operator is ConditionOperator.EQUALS and right.operator is ConditionOperator.EQUALS:
        return left.operand != right.operand
    if left.operator is ConditionOperator.CURRENT and right.operator is ConditionOperator.CURRENT:
        return left.operand != right.operand
    return False


@dataclass(frozen=True)
class AdversarialFixture:
    """One bounded hermetic attack against a candidate obligation."""

    fixture_id: str
    vacuity_class: VacuityClass
    target_property_id: str
    evidence_cid: str
    witness: Mapping[str, Any] = field(default_factory=dict)
    expected_vacuous: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        object.__setattr__(self, "vacuity_class", _vacuity_class(self.vacuity_class))
        object.__setattr__(
            self,
            "target_property_id",
            _identifier(self.target_property_id, "target_property_id"),
        )
        object.__setattr__(self, "evidence_cid", _identifier(self.evidence_cid, "evidence_cid"))
        object.__setattr__(self, "witness", _freeze(self.witness, "witness"))
        if not isinstance(self.witness, Mapping):
            raise InvariantMiningError("witness must be a mapping")
        object.__setattr__(
            self, "expected_vacuous", _bool(self.expected_vacuous, "expected_vacuous")
        )

    def applies_to(self, candidate: SpecificationCandidate) -> bool:
        return self.target_property_id == candidate.property_id

    def to_subject(self) -> dict[str, Any]:
        return {
            "vacuity_family": VACUITY_TEST_FAMILY,
            "subject": {
                "fixture_id": self.fixture_id,
                "vacuity_class": self.vacuity_class.value,
                "property_id": self.target_property_id,
                "evidence_cid": self.evidence_cid,
                "expected_vacuous": self.expected_vacuous,
                "witness": dict(self.witness),
            },
        }


@dataclass(frozen=True)
class AssuranceApiProbe:
    """Fail-closed probe of the existing campaign API."""

    interface_id: str
    command: str
    status: AssuranceApiStatus
    available: bool
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface_id", _identifier(self.interface_id, "interface_id"))
        object.__setattr__(self, "command", _identifier(self.command, "command"))
        object.__setattr__(self, "status", _assurance_status(self.status))
        object.__setattr__(self, "available", _bool(self.available, "available"))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        if self.available and self.status is not AssuranceApiStatus.AVAILABLE:
            raise InvariantMiningError("available probes must report available status")


@dataclass(frozen=True)
class AssuranceCampaignObservation:
    """Receipt-bearing observation from AssuranceCampaignApi, or typed unavailability."""

    status: AssuranceApiStatus
    interface_id: str
    reason_code: str
    plan_cid: str
    receipt_cids: tuple[str, ...] = ()
    finding_property_ids: tuple[str, ...] = ()
    residual_properties: tuple[str, ...] = ()
    precise_nonclaims: tuple[str, ...] = ()
    findings: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _assurance_status(self.status))
        object.__setattr__(self, "interface_id", _identifier(self.interface_id, "interface_id"))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(
            self, "plan_cid", _identifier(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self,
            "receipt_cids",
            _strings(self.receipt_cids, "receipt_cids", identifiers=True),
        )
        object.__setattr__(
            self,
            "finding_property_ids",
            _strings(self.finding_property_ids, "finding_property_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "residual_properties",
            _strings(self.residual_properties, "residual_properties"),
        )
        object.__setattr__(
            self,
            "precise_nonclaims",
            _strings(self.precise_nonclaims, "precise_nonclaims"),
        )
        frozen_findings = _freeze(self.findings, "findings")
        if not isinstance(frozen_findings, tuple):
            raise InvariantMiningError("findings must be a sequence")
        object.__setattr__(self, "findings", frozen_findings)


def _load_campaign_api() -> Any | None:
    try:
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
            create_assurance_campaign_api,
        )
    except Exception:
        return None
    try:
        return create_assurance_campaign_api()
    except Exception:
        return None


class AssuranceApiAdapter:
    """Narrow adapter over existing ``AssuranceCampaignApi@1``.

    This is not an assurance engine.  It probes and invokes the canonical
    campaign API, then records receipts or typed unavailability.  Missing or
    incompatible leaves never become simulated success.
    """

    def __init__(
        self,
        campaign_api: Any | None = None,
        *,
        adapter_revision: str = ASSURANCE_API_ADAPTER_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.adapter_revision = _identifier(adapter_revision, "adapter_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self._injected = campaign_api is not None
        self._campaign_api = campaign_api

    @property
    def interface_id(self) -> str:
        api = self._campaign_api
        pinned = getattr(api, "interface_id", None) if api is not None else None
        if isinstance(pinned, str) and pinned:
            return pinned
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def campaign_api(self) -> Any | None:
        if self._campaign_api is not None or self._injected:
            return self._campaign_api
        self._campaign_api = _load_campaign_api()
        return self._campaign_api

    def probe(self, command: str = ANALYZE_VACUITY_COMMAND) -> AssuranceApiProbe:
        command = _identifier(command, "command")
        api = self.campaign_api()
        if api is None:
            return AssuranceApiProbe(
                interface_id=self.interface_id,
                command=command,
                status=AssuranceApiStatus.TYPED_UNAVAILABLE,
                available=False,
                reason_code="assurance_api_unavailable",
            )
        probe_fn = getattr(api, "probe_api", None)
        if callable(probe_fn):
            try:
                payload = probe_fn(command)
            except Exception:
                return AssuranceApiProbe(
                    interface_id=self.interface_id,
                    command=command,
                    status=AssuranceApiStatus.INCOMPATIBLE,
                    available=False,
                    reason_code="assurance_api_probe_failed",
                )
            if isinstance(payload, Mapping):
                available = bool(payload.get("available"))
                status_text = str(payload.get("status") or "")
                if available and status_text in {"", AssuranceApiStatus.AVAILABLE.value}:
                    status = AssuranceApiStatus.AVAILABLE
                    reason = "assurance_api_available"
                elif available:
                    status = AssuranceApiStatus.INCOMPATIBLE
                    reason = "assurance_api_incompatible"
                    available = False
                else:
                    status = AssuranceApiStatus.TYPED_UNAVAILABLE
                    reason = str(payload.get("reason_code") or "assurance_api_unavailable")
                    if not _identifier_safe(reason):
                        reason = "assurance_api_unavailable"
                return AssuranceApiProbe(
                    interface_id=self.interface_id,
                    command=command,
                    status=status,
                    available=available,
                    reason_code=reason,
                )
        if callable(getattr(api, command, None)):
            return AssuranceApiProbe(
                interface_id=self.interface_id,
                command=command,
                status=AssuranceApiStatus.AVAILABLE,
                available=True,
                reason_code="assurance_api_available",
            )
        return AssuranceApiProbe(
            interface_id=self.interface_id,
            command=command,
            status=AssuranceApiStatus.TYPED_UNAVAILABLE,
            available=False,
            reason_code="assurance_api_missing_command",
        )

    def analyze_vacuity(
        self,
        *,
        bindings: ArtifactBindings,
        fixtures: Sequence[AdversarialFixture],
        candidate_ids: Sequence[str],
        tested_obligation_ids: Sequence[str],
    ) -> AssuranceCampaignObservation:
        bindings = _bindings(bindings)
        if len(fixtures) > MAX_ITEMS:
            raise InvariantMiningError("adversarial fixtures exceed their item bound")
        fixture_ids = _unique_identifiers(
            tuple(item.fixture_id for item in fixtures), "fixture_ids"
        )
        plan_payload = {
            "adapter_revision": self.adapter_revision,
            "interface_id": self.interface_id,
            "tree_id": bindings.tree_id,
            "repository_commit": bindings.repository_commit,
            "candidate_ids": list(candidate_ids),
            "fixture_ids": list(fixture_ids),
            "tested_obligation_ids": list(tested_obligation_ids),
            "vacuity_classes": [item.value for item in REQUIRED_VACUITY_CLASSES],
            "completeness_claimed": False,
        }
        plan_cid = _plan_cid(plan_payload)
        probe = self.probe(ANALYZE_VACUITY_COMMAND)
        unavailable = AssuranceCampaignObservation(
            status=probe.status,
            interface_id=self.interface_id,
            reason_code=probe.reason_code,
            plan_cid=plan_cid,
            precise_nonclaims=(
                "completeness-beyond-tested-obligations",
                "untested-obligations-remain-unclaimed",
            ),
        )
        if not probe.available:
            return unavailable
        api = self.campaign_api()
        analyze = getattr(api, ANALYZE_VACUITY_COMMAND, None)
        if not callable(analyze):
            return replace(
                unavailable,
                status=AssuranceApiStatus.TYPED_UNAVAILABLE,
                reason_code="assurance_api_missing_command",
            )
        manifest = {
            "manifest_cid": plan_cid,
            "adapter_revision": self.adapter_revision,
            "candidate_ids": list(candidate_ids),
            "fixture_ids": list(fixture_ids),
            "tested_obligation_ids": list(tested_obligation_ids),
            "completeness_claimed": False,
        }
        repository_state = {
            "repository_state_cid": bindings.tree_id,
            "tree_id": bindings.tree_id,
            "repository_id": bindings.repository_id,
            "repository_commit": bindings.repository_commit,
        }
        subjects = tuple(item.to_subject() for item in fixtures)
        try:
            raw = analyze(
                manifest,
                repository_state,
                subjects=subjects,
                notes="non-vacuity-campaign",
            )
        except Exception:
            return replace(
                unavailable,
                status=AssuranceApiStatus.INCOMPATIBLE,
                reason_code="assurance_api_invocation_failed",
            )
        return _observation_from_api_result(
            raw,
            interface_id=self.interface_id,
            plan_cid=plan_cid,
        )


def _identifier_safe(value: str) -> bool:
    try:
        _identifier(value, "reason_code")
    except Exception:
        return False
    return True


def _observation_from_api_result(
    raw: Any,
    *,
    interface_id: str,
    plan_cid: str,
) -> AssuranceCampaignObservation:
    payload: Mapping[str, Any]
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        converted = raw.to_dict()
        if not isinstance(converted, Mapping):
            return AssuranceCampaignObservation(
                status=AssuranceApiStatus.INCOMPATIBLE,
                interface_id=interface_id,
                reason_code="assurance_api_invalid_result",
                plan_cid=plan_cid,
            )
        payload = converted
    elif isinstance(raw, Mapping):
        payload = raw
    else:
        payload = {
            "result_cid": getattr(raw, "result_cid", ""),
            "finding_cids": getattr(raw, "finding_cids", ()),
            "findings": getattr(raw, "findings", ()),
            "residual_properties": getattr(raw, "residual_properties", ()),
            "precise_nonclaims": getattr(raw, "precise_nonclaims", ()),
        }
    receipt_cids: list[str] = []
    for key in ("result_cid", "assurance_manifest_cid"):
        value = payload.get(key)
        if isinstance(value, str) and value and _identifier_safe(value):
            receipt_cids.append(value)
    for item in payload.get("finding_cids") or ():
        if isinstance(item, str) and item and _identifier_safe(item):
            receipt_cids.append(item)
    findings_raw = payload.get("findings") or ()
    if not isinstance(findings_raw, Sequence) or isinstance(findings_raw, (str, bytes)):
        findings_raw = ()
    finding_property_ids: list[str] = []
    findings: list[Mapping[str, Any]] = []
    for item in findings_raw:
        if not isinstance(item, Mapping):
            continue
        findings.append(dict(item))
        for key in ("property_id", "target_property_id", "subject_cid"):
            value = item.get(key)
            if isinstance(value, str) and value and _identifier_safe(value):
                finding_property_ids.append(value)
                break
    residual = tuple(
        str(item) for item in (payload.get("residual_properties") or ()) if isinstance(item, str)
    )
    nonclaims = [
        str(item)
        for item in (payload.get("precise_nonclaims") or ())
        if isinstance(item, str)
    ]
    if "completeness-beyond-tested-obligations" not in nonclaims:
        nonclaims.append("completeness-beyond-tested-obligations")
    return AssuranceCampaignObservation(
        status=AssuranceApiStatus.AVAILABLE,
        interface_id=interface_id,
        reason_code="assurance_api_available",
        plan_cid=plan_cid,
        receipt_cids=_unique_identifiers(receipt_cids, "receipt_cids")
        if receipt_cids
        else (),
        finding_property_ids=_unique_identifiers(finding_property_ids, "finding_property_ids")
        if finding_property_ids
        else (),
        residual_properties=residual,
        precise_nonclaims=tuple(nonclaims),
        findings=tuple(findings),
    )


def _current_binding_value(bindings: ArtifactBindings, binding: str) -> str | None:
    field_name = binding.rsplit(".", 1)[-1] if "." in binding else binding
    if field_name.startswith("binding:"):
        field_name = field_name.split(":", 1)[1]
    if hasattr(bindings, field_name):
        value = getattr(bindings, field_name)
        return value if isinstance(value, str) else None
    return None


def _referenced_step_id(candidate: SpecificationCandidate) -> str | None:
    for raw in (candidate.binding, candidate.property_id):
        if raw.startswith("step:") or raw.startswith("procedure.steps."):
            return raw.rsplit(".", 1)[-1] if "." in raw else raw.split(":", 1)[-1]
        if raw.startswith("branch:") or raw.startswith("procedure.branches."):
            return raw.rsplit(".", 1)[-1] if "." in raw else raw.split(":", 1)[-1]
    return None


def _is_constant_restatement(candidate: SpecificationCandidate) -> bool:
    binding = candidate.binding.rsplit(".", 1)[-1]
    if binding in _CONSTANT_BINDINGS:
        return True
    if candidate.operator in {ConditionOperator.EQUALS, ConditionOperator.EXISTS}:
        if candidate.operand is True:
            return True
    if candidate.operator is ConditionOperator.EQUALS and candidate.operand == candidate.property_id:
        return True
    return False


def _is_mock_effect(candidate: SpecificationCandidate) -> bool:
    if candidate.property_kind is PropertyKind.EFFECT:
        if any(_contains_marker(cid, _MOCK_MARKERS) for cid in candidate.evidence_cids):
            return True
        if any(
            _contains_marker(item.provenance_cid, _MOCK_MARKERS)
            or _contains_marker(item.artifact_cid, _MOCK_MARKERS)
            for item in candidate.source_provenances
        ):
            return True
        kinds = candidate.source_kinds
        if kinds == (SourceKind.MUTANT,) or kinds == (SourceKind.AUTHORITATIVE_DOCUMENTATION,):
            return True
    return any(_contains_marker(cid, _MOCK_MARKERS) for cid in candidate.evidence_cids)


def _is_fixture_shortcut(candidate: SpecificationCandidate) -> bool:
    kinds = set(candidate.source_kinds)
    if candidate.passing_test_count > 0 and kinds <= {SourceKind.TEST}:
        return True
    if any(_contains_marker(cid, _SHORTCUT_MARKERS) for cid in candidate.evidence_cids):
        return True
    return False


def _is_no_invocation(
    candidate: SpecificationCandidate,
    *,
    procedure: ProcedureSpec | None,
    trajectory: ExecutionTrajectory | None,
) -> bool:
    if any(
        _contains_marker(cid, ("no-invocation", "uninvoiced", "not-invoked"))
        for cid in candidate.evidence_cids
    ):
        return True
    test_only = set(candidate.source_kinds) <= {SourceKind.TEST}
    if not test_only:
        return False
    invoked: set[str] = set()
    if procedure is not None:
        for step in procedure.steps:
            invoked.add(step.step_id)
            invoked.update(step.evidence_outputs)
            invoked.update(step.declared_effect_ids)
    if trajectory is not None:
        for step in trajectory.steps:
            invoked.add(step.operation.value)
            invoked.update(step.effect_ids)
            invoked.update(step.observation_cids)
    if procedure is None and trajectory is None:
        return False
    if not invoked:
        return True
    token = candidate.property_id.rsplit(".", 1)[-1]
    return token not in invoked and candidate.binding not in invoked


def _is_impossible_precondition(
    candidate: SpecificationCandidate,
    *,
    current_bindings: ArtifactBindings,
    peers: Sequence[SpecificationCandidate],
) -> bool:
    if candidate.property_kind is not PropertyKind.PRECONDITION:
        return False
    if candidate.required and candidate.operator in {
        ConditionOperator.EXISTS,
        ConditionOperator.IN_CLOSED_SET,
        ConditionOperator.SUBSET_OF,
    } and _empty_domain(candidate.operand):
        return True
    if candidate.operator is ConditionOperator.EQUALS and candidate.operand is False:
        return True
    if candidate.operator is ConditionOperator.CURRENT:
        current = _current_binding_value(current_bindings, candidate.binding)
        if current is not None and candidate.operand not in {None, current}:
            return True
    return any(
        peer is not candidate
        and peer.property_kind is PropertyKind.PRECONDITION
        and peer.property_id == candidate.property_id
        and _contradicts(candidate, peer)
        for peer in peers
    )


def _is_empty_output(candidate: SpecificationCandidate) -> bool:
    if candidate.property_kind not in {PropertyKind.POSTCONDITION, PropertyKind.INVARIANT}:
        return False
    if candidate.operator in {
        ConditionOperator.EXISTS,
        ConditionOperator.IN_CLOSED_SET,
        ConditionOperator.SUBSET_OF,
        ConditionOperator.EQUALS,
    } and _empty_domain(candidate.operand):
        return True
    return False


def _is_unreachable(
    candidate: SpecificationCandidate,
    *,
    procedure: ProcedureSpec | None,
) -> bool:
    if procedure is None:
        return False
    reachable = _reachable_step_ids(procedure)
    step_id = _referenced_step_id(candidate)
    if step_id is not None and step_id not in reachable:
        return True
    if candidate.binding in reachable or candidate.property_id in reachable:
        return False
    if candidate.binding in {step.step_id for step in procedure.steps}:
        return candidate.binding not in reachable
    return False


def _invariant_counterexample_hit(
    candidate: SpecificationCandidate,
    *,
    current_bindings: ArtifactBindings,
    trajectory: ExecutionTrajectory | None,
) -> bool:
    if candidate.property_kind is not PropertyKind.INVARIANT:
        return False
    if candidate.operator is ConditionOperator.CURRENT:
        current = _current_binding_value(current_bindings, candidate.binding)
        if current is not None and candidate.operand not in {None, current}:
            return True
    if candidate.operator is ConditionOperator.SUBSET_OF and trajectory is not None:
        allowed = candidate.operand
        if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)):
            for step in trajectory.steps:
                for effect_id in step.effect_ids:
                    if effect_id.startswith("escape:") or effect_id.startswith("out-of-scope"):
                        return True
    if candidate.operator is ConditionOperator.NOT_EXISTS and candidate.operand:
        if trajectory is not None:
            observed = []
            for step in trajectory.steps:
                observed.extend(step.effect_ids)
                observed.extend(step.observation_cids)
            if candidate.operand in observed:
                return True
    return False


def _fixture_flags(fixture: AdversarialFixture) -> Mapping[str, Any]:
    return fixture.witness


def evaluate_vacuity_classes(
    candidate: SpecificationCandidate,
    *,
    current_bindings: ArtifactBindings,
    fixtures: Sequence[AdversarialFixture],
    peers: Sequence[SpecificationCandidate],
    procedure: ProcedureSpec | None = None,
    trajectory: ExecutionTrajectory | None = None,
) -> tuple[VacuityClass, ...]:
    """Return the vacuity classes that fire against one candidate."""

    hits: list[VacuityClass] = []

    def add(item: VacuityClass) -> None:
        if item not in hits:
            hits.append(item)

    matching = tuple(item for item in fixtures if item.applies_to(candidate))
    for fixture in matching:
        if not fixture.expected_vacuous:
            continue
        add(fixture.vacuity_class)
        witness = _fixture_flags(fixture)
        if witness.get("reachable") is False:
            add(VacuityClass.UNREACHABLE_BRANCH)
        if witness.get("invoked") is False:
            add(VacuityClass.NO_INVOCATION)
        if witness.get("mock_substitution") is True:
            add(VacuityClass.MOCK_EFFECT)
        if witness.get("fixture_shortcut") is True:
            add(VacuityClass.FIXTURE_SHORTCUT)
        if witness.get("constant_restatement") is True:
            add(VacuityClass.CONSTANT_RESTATEMENT)
        if witness.get("empty_domain") is True:
            add(VacuityClass.EMPTY_OUTPUT)
        if witness.get("contradictory") is True:
            add(VacuityClass.IMPOSSIBLE_PRECONDITION)
        if witness.get("observed_effect") is False:
            add(VacuityClass.MOCK_EFFECT)
        if witness.get("violating_observation_cid"):
            add(VacuityClass.INVARIANT_COUNTEREXAMPLE)

    if _is_impossible_precondition(
        candidate, current_bindings=current_bindings, peers=peers
    ):
        add(VacuityClass.IMPOSSIBLE_PRECONDITION)
    if _is_unreachable(candidate, procedure=procedure):
        add(VacuityClass.UNREACHABLE_BRANCH)
    if _is_empty_output(candidate):
        add(VacuityClass.EMPTY_OUTPUT)
    if _is_no_invocation(candidate, procedure=procedure, trajectory=trajectory):
        add(VacuityClass.NO_INVOCATION)
    if _is_mock_effect(candidate):
        add(VacuityClass.MOCK_EFFECT)
    if _is_fixture_shortcut(candidate):
        add(VacuityClass.FIXTURE_SHORTCUT)
    if _is_constant_restatement(candidate):
        add(VacuityClass.CONSTANT_RESTATEMENT)
    if _invariant_counterexample_hit(
        candidate, current_bindings=current_bindings, trajectory=trajectory
    ):
        add(VacuityClass.INVARIANT_COUNTEREXAMPLE)
    return tuple(hits)


def _refuse(
    candidate: SpecificationCandidate,
    extra_cids: Sequence[str] = (),
) -> SpecificationCandidate:
    evidence = _unique_identifiers(
        (*candidate.evidence_cids, *extra_cids), "evidence_cids"
    )
    return replace(candidate, status=CandidateStatus.REFUSED, evidence_cids=evidence)


def _retain_receipts(
    candidate: SpecificationCandidate,
    extra_cids: Sequence[str],
) -> SpecificationCandidate:
    if not extra_cids:
        return candidate
    evidence = _unique_identifiers(
        (*candidate.evidence_cids, *extra_cids), "evidence_cids"
    )
    retained = replace(candidate, evidence_cids=evidence)
    if retained.status is not CandidateStatus.CANDIDATE:
        raise InvariantMiningError("surviving candidates must remain in candidate status")
    return retained


def _vacuity_counterexample(
    candidate: SpecificationCandidate,
    vacuity_class: VacuityClass,
    *,
    evidence_cid: str,
) -> SpecificationCounterexample:
    left_evidence = candidate.evidence_cids[0]
    left_provenance = candidate.source_provenances[0].provenance_cid
    return SpecificationCounterexample(
        property_kind=candidate.property_kind,
        property_id=candidate.property_id,
        conflict_class=vacuity_class.value,
        left_claim=candidate.claim,
        right_claim=("vacuity", vacuity_class.value, candidate.property_id),
        left_evidence_cid=left_evidence,
        right_evidence_cid=evidence_cid,
        left_provenance_cid=left_provenance,
        right_provenance_cid=evidence_cid,
    )


def _primary_evidence_cid(
    candidate: SpecificationCandidate,
    hits: Sequence[VacuityClass],
    fixtures: Sequence[AdversarialFixture],
) -> str:
    for fixture in fixtures:
        if fixture.applies_to(candidate) and fixture.vacuity_class in hits:
            return fixture.evidence_cid
    return f"vacuity.{hits[0].value}.{candidate.property_id}"


@dataclass(frozen=True)
class NonVacuityValidationResult:
    """Bounded non-vacuity campaign output.  Completeness is never claimed."""

    bindings: ArtifactBindings
    surviving: tuple[SpecificationCandidate, ...]
    refused: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    fixtures: tuple[AdversarialFixture, ...]
    tested_obligation_ids: tuple[str, ...]
    vacuity_classes_tested: tuple[VacuityClass, ...]
    adversarial_receipt_cids: tuple[str, ...]
    assurance_api_status: AssuranceApiStatus
    campaign: AssuranceCampaignObservation
    receipt: NonVacuityReceiptArtifact
    completeness_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        if self.completeness_claimed:
            raise InvariantMiningError("non-vacuity cannot claim completeness")
        object.__setattr__(
            self, "assurance_api_status", _assurance_status(self.assurance_api_status)
        )
        object.__setattr__(
            self,
            "tested_obligation_ids",
            _strings(self.tested_obligation_ids, "tested_obligation_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "adversarial_receipt_cids",
            _strings(self.adversarial_receipt_cids, "adversarial_receipt_cids", identifiers=True),
        )
        if any(item.status is not CandidateStatus.CANDIDATE for item in self.surviving):
            raise InvariantMiningError("surviving candidates must remain in candidate status")
        if any(item.status is not CandidateStatus.REFUSED for item in self.refused):
            raise InvariantMiningError("vacuous candidates must be refused")
        if self.receipt.state is ArtifactState.VERIFIED or self.receipt.state is ArtifactState.PROMOTED:
            raise InvariantMiningError("non-vacuity receipts cannot claim verification")
        if self.receipt.facts.get("completeness_claimed") is True:
            raise InvariantMiningError("non-vacuity receipts cannot claim completeness")
        if self.receipt.facts.get("verified_count", 0) != 0:
            raise InvariantMiningError("non-vacuity cannot verify candidates")

    @property
    def upgraded_count(self) -> int:
        return 0


@dataclass(frozen=True)
class InvariantValidationResult:
    """Invariant-only non-vacuity output with candidate-status wire artifacts."""

    bindings: ArtifactBindings
    surviving: tuple[SpecificationCandidate, ...]
    refused: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    invariant_artifacts: tuple[InvariantCandidateArtifact, ...]
    tested_obligation_ids: tuple[str, ...]
    adversarial_receipt_cids: tuple[str, ...]
    assurance_api_status: AssuranceApiStatus
    campaign: AssuranceCampaignObservation
    receipt: InvariantValidationReceiptArtifact
    non_vacuity: NonVacuityValidationResult
    completeness_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        if self.completeness_claimed:
            raise InvariantMiningError("invariant validation cannot claim completeness")
        if any(item.property_kind is not PropertyKind.INVARIANT for item in self.surviving):
            raise InvariantMiningError("invariant validation retains only invariant survivors")
        if any(item.state is not ArtifactState.CANDIDATE for item in self.invariant_artifacts):
            raise InvariantMiningError("validated invariants cannot leave candidate state")
        if self.receipt.state is ArtifactState.VERIFIED or self.receipt.state is ArtifactState.PROMOTED:
            raise InvariantMiningError("invariant validation receipts cannot claim verification")

    @property
    def upgraded_count(self) -> int:
        return 0


def _coerce_candidates(
    value: Sequence[SpecificationCandidate]
    | InvariantMiningResult
    | SpecificationMiningResult,
    *,
    invariant_only: bool = False,
) -> tuple[SpecificationCandidate, ...]:
    if isinstance(value, InvariantMiningResult):
        items = value.candidates
    elif isinstance(value, SpecificationMiningResult):
        items = value.candidates
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, memoryview)):
            raise InvariantMiningError("candidates must be a bounded sequence")
        items = tuple(value)
    if len(items) > MAX_ITEMS:
        raise InvariantMiningError("candidates exceed their item bound")
    for item in items:
        if not isinstance(item, SpecificationCandidate):
            raise InvariantMiningError("candidates must be SpecificationCandidate values")
    if invariant_only:
        items = tuple(item for item in items if item.property_kind is PropertyKind.INVARIANT)
    return items


def _coerce_bindings(
    bindings: ArtifactBindings | None,
    current_bindings: ArtifactBindings | None,
    procedure: ProcedureSpec | None,
    trajectory: ExecutionTrajectory | None,
    source: Any,
) -> ArtifactBindings:
    if bindings is not None:
        return _bindings(bindings)
    if current_bindings is not None:
        return _bindings(current_bindings)
    if procedure is not None:
        return procedure.bindings
    if trajectory is not None:
        return trajectory.bindings
    if isinstance(source, (InvariantMiningResult, SpecificationMiningResult)):
        return source.bindings
    raise InvariantMiningError("exact current bindings are required")


class NonVacuityValidator:
    """Reject vacuous specification candidates; never certify survivors."""

    def __init__(
        self,
        *,
        adapter: AssuranceApiAdapter | None = None,
        campaign_api: Any | None = None,
        validator_revision: str = NON_VACUITY_VALIDATOR_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.validator_revision = _identifier(validator_revision, "validator_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self.adapter = adapter or AssuranceApiAdapter(
            campaign_api,
            emitted_at_ms=self.emitted_at_ms,
        )

    def validate(
        self,
        candidates: Sequence[SpecificationCandidate]
        | InvariantMiningResult
        | SpecificationMiningResult,
        *,
        bindings: ArtifactBindings | None = None,
        current_bindings: ArtifactBindings | None = None,
        procedure: ProcedureSpec | None = None,
        trajectory: ExecutionTrajectory | None = None,
        fixtures: Sequence[AdversarialFixture] = (),
    ) -> NonVacuityValidationResult:
        items = _coerce_candidates(candidates)
        if not items:
            raise InvariantMiningError("at least one candidate is required")
        if len(fixtures) > MAX_ITEMS:
            raise InvariantMiningError("adversarial fixtures exceed their item bound")
        sealed_fixtures = tuple(
            item if isinstance(item, AdversarialFixture) else AdversarialFixture(**item)
            for item in fixtures
        )
        current = _coerce_bindings(
            current_bindings, bindings, procedure, trajectory, candidates
        )
        source_bindings = _coerce_bindings(
            bindings, current_bindings, procedure, trajectory, candidates
        )
        if source_bindings != current:
            raise InvariantMiningError("candidate bindings are not the current bindings")

        tested_ids = _unique_identifiers(
            tuple(item.property_id for item in items), "tested_obligation_ids"
        )
        campaign = self.adapter.analyze_vacuity(
            bindings=current,
            fixtures=sealed_fixtures,
            candidate_ids=tested_ids,
            tested_obligation_ids=tested_ids,
        )
        api_hits = frozenset(campaign.finding_property_ids)
        receipt_cids = campaign.receipt_cids

        surviving: list[SpecificationCandidate] = []
        refused: list[SpecificationCandidate] = []
        counterexamples: list[SpecificationCounterexample] = []

        for candidate in items:
            if candidate.status is CandidateStatus.REFUSED:
                refused.append(candidate)
                continue
            hits = list(
                evaluate_vacuity_classes(
                    candidate,
                    current_bindings=current,
                    fixtures=sealed_fixtures,
                    peers=items,
                    procedure=procedure,
                    trajectory=trajectory,
                )
            )
            if candidate.property_id in api_hits and VacuityClass.INVARIANT_COUNTEREXAMPLE not in hits:
                if candidate.property_kind is PropertyKind.INVARIANT:
                    hits.append(VacuityClass.INVARIANT_COUNTEREXAMPLE)
                elif not hits:
                    hits.append(VacuityClass.FIXTURE_SHORTCUT)
            if hits:
                evidence_cid = _primary_evidence_cid(candidate, hits, sealed_fixtures)
                refused.append(_refuse(candidate, (evidence_cid, *receipt_cids)))
                if len(counterexamples) < MAX_ITEMS:
                    counterexamples.append(
                        _vacuity_counterexample(
                            candidate,
                            hits[0],
                            evidence_cid=evidence_cid,
                        )
                    )
                continue
            surviving.append(_retain_receipts(candidate, receipt_cids))

        counterexample_artifacts = tuple(
            item.to_artifact(current, emitted_at_ms=self.emitted_at_ms)
            for item in counterexamples
        )
        surviving_artifacts = tuple(
            item.to_artifact(current, emitted_at_ms=self.emitted_at_ms) for item in surviving
        )
        refused_artifacts = tuple(
            item.to_artifact(current, emitted_at_ms=self.emitted_at_ms) for item in refused
        )
        receipt = NonVacuityReceiptArtifact(
            bindings=current,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.validator_revision,
            reference_cids=_unique_identifiers(
                (
                    *(item.content_id for item in surviving_artifacts),
                    *(item.content_id for item in refused_artifacts),
                    *(item.content_id for item in counterexample_artifacts),
                    *receipt_cids,
                    campaign.plan_cid,
                ),
                "reference_cids",
            )
            if (surviving_artifacts or refused_artifacts or counterexample_artifacts or receipt_cids)
            else (campaign.plan_cid,),
            labels=("non-vacuity", "candidate"),
            facts={
                "validator_revision": self.validator_revision,
                "assurance_api_interface": campaign.interface_id,
                "assurance_api_status": campaign.status.value,
                "assurance_api_reason_code": campaign.reason_code,
                "campaign_plan_cid": campaign.plan_cid,
                "adversarial_receipt_cids": receipt_cids,
                "surviving_cids": tuple(item.content_id for item in surviving_artifacts),
                "refused_cids": tuple(item.content_id for item in refused_artifacts),
                "counterexample_cids": tuple(
                    item.content_id for item in counterexample_artifacts
                ),
                "tested_obligation_ids": tested_ids,
                "vacuity_classes_tested": tuple(
                    item.value for item in REQUIRED_VACUITY_CLASSES
                ),
                "fixture_ids": tuple(item.fixture_id for item in sealed_fixtures),
                "current_tree_id": current.tree_id,
                "current_repository_commit": current.repository_commit,
                "precise_nonclaims": campaign.precise_nonclaims
                or ("completeness-beyond-tested-obligations",),
                "completeness_claimed": False,
                "upgraded_count": 0,
                "verified_count": 0,
            },
            created_at_ms=self.emitted_at_ms,
        )
        return NonVacuityValidationResult(
            bindings=current,
            surviving=tuple(surviving),
            refused=tuple(refused),
            counterexamples=tuple(counterexamples),
            fixtures=sealed_fixtures,
            tested_obligation_ids=tested_ids,
            vacuity_classes_tested=REQUIRED_VACUITY_CLASSES,
            adversarial_receipt_cids=receipt_cids,
            assurance_api_status=campaign.status,
            campaign=campaign,
            receipt=receipt,
            completeness_claimed=False,
        )


class InvariantValidator:
    """Validate invariant candidates without leaving candidate status."""

    def __init__(
        self,
        *,
        adapter: AssuranceApiAdapter | None = None,
        campaign_api: Any | None = None,
        validator_revision: str = INVARIANT_VALIDATOR_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.validator_revision = _identifier(validator_revision, "validator_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self._non_vacuity = NonVacuityValidator(
            adapter=adapter,
            campaign_api=campaign_api,
            emitted_at_ms=self.emitted_at_ms,
        )

    def validate(
        self,
        candidates: Sequence[SpecificationCandidate]
        | InvariantMiningResult
        | SpecificationMiningResult,
        *,
        bindings: ArtifactBindings | None = None,
        current_bindings: ArtifactBindings | None = None,
        procedure: ProcedureSpec | None = None,
        trajectory: ExecutionTrajectory | None = None,
        fixtures: Sequence[AdversarialFixture] = (),
    ) -> InvariantValidationResult:
        items = _coerce_candidates(candidates, invariant_only=True)
        if not items:
            raise InvariantMiningError("at least one invariant candidate is required")
        result = self._non_vacuity.validate(
            items,
            bindings=bindings,
            current_bindings=current_bindings,
            procedure=procedure,
            trajectory=trajectory,
            fixtures=fixtures,
        )
        artifacts = tuple(
            item.to_invariant_artifact(result.bindings, emitted_at_ms=self.emitted_at_ms)
            for item in result.surviving
        )
        if any(item.state is not ArtifactState.CANDIDATE for item in artifacts):
            raise InvariantMiningError("validated invariants cannot leave candidate state")
        surviving_cids = tuple(item.content_id for item in artifacts)
        refused_cids = tuple(
            item.to_invariant_artifact(result.bindings, emitted_at_ms=self.emitted_at_ms).content_id
            for item in result.refused
        )
        counterexample_cids = tuple(
            item.to_artifact(result.bindings, emitted_at_ms=self.emitted_at_ms).content_id
            for item in result.counterexamples
        )
        receipt = InvariantValidationReceiptArtifact(
            bindings=result.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.validator_revision,
            reference_cids=_unique_identifiers(
                (
                    *surviving_cids,
                    *refused_cids,
                    *counterexample_cids,
                    *result.adversarial_receipt_cids,
                    result.campaign.plan_cid,
                    result.receipt.content_id,
                ),
                "reference_cids",
            ),
            labels=("invariant-validation", "candidate"),
            facts={
                "validator_revision": self.validator_revision,
                "non_vacuity_receipt_cid": result.receipt.content_id,
                "assurance_api_interface": result.campaign.interface_id,
                "assurance_api_status": result.assurance_api_status.value,
                "adversarial_receipt_cids": result.adversarial_receipt_cids,
                "surviving_cids": surviving_cids,
                "refused_cids": refused_cids,
                "counterexample_cids": counterexample_cids,
                "tested_obligation_ids": result.tested_obligation_ids,
                "vacuity_classes_tested": tuple(
                    item.value for item in REQUIRED_VACUITY_CLASSES
                ),
                "current_tree_id": result.bindings.tree_id,
                "current_repository_commit": result.bindings.repository_commit,
                "completeness_claimed": False,
                "upgraded_count": 0,
                "verified_count": 0,
            },
            created_at_ms=self.emitted_at_ms,
        )
        return InvariantValidationResult(
            bindings=result.bindings,
            surviving=result.surviving,
            refused=result.refused,
            counterexamples=result.counterexamples,
            invariant_artifacts=artifacts,
            tested_obligation_ids=result.tested_obligation_ids,
            adversarial_receipt_cids=result.adversarial_receipt_cids,
            assurance_api_status=result.assurance_api_status,
            campaign=result.campaign,
            receipt=receipt,
            non_vacuity=result,
            completeness_claimed=False,
        )


__all__ = [
    "ANALYZE_VACUITY_COMMAND",
    "ASSURANCE_API_ADAPTER_REVISION",
    "ASSURANCE_CAMPAIGN_API_INTERFACE_PIN",
    "INVARIANT_MINER_REVISION",
    "INVARIANT_VALIDATOR_REVISION",
    "NON_VACUITY_VALIDATOR_REVISION",
    "REQUIRED_VACUITY_CLASSES",
    "AdversarialFixture",
    "AssuranceApiAdapter",
    "AssuranceApiProbe",
    "AssuranceApiStatus",
    "AssuranceCampaignObservation",
    "InvariantCandidateArtifact",
    "InvariantMiner",
    "InvariantMiningError",
    "InvariantMiningResult",
    "InvariantValidationResult",
    "InvariantValidator",
    "NonVacuityValidationResult",
    "NonVacuityValidator",
    "VacuityClass",
    "evaluate_vacuity_classes",
    "project_invariant_sources",
]

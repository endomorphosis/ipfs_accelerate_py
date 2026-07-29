"""Fail-closed proof routing for compiled MCP contract obligations.

``McpContractProver@1`` is deliberately a small adapter over the supervisor's
existing proof contracts.  Graph and schema obligations are checked by local,
deterministic checkers.  Relation, deontic, and temporal obligations are sent
to optional SMT, CEC, and TDFOL providers only after an operation-specific
capability probe.

Provider responses are untrusted candidates.  In particular, a provider
cannot create kernel assurance by returning an ``assurance`` field, proof
receipt, or kernel-looking evidence.  An embedding application may supply a
trusted receipt validator; without it, positive provider output remains
inconclusive.  This module never imports or invokes an LLM.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderOperation,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    AttemptStatus,
    CanonicalContract,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofAttempt,
    ProofEvidence,
    ProofReceipt,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    assurance_satisfies,
    content_identity,
)
from .formal_verification_provider import (
    ProviderFailureCode,
    ProviderResponse,
    get_proof_provider,
)
from .mcp_contract_obligations import (
    LogicFragment,
    McpContractObligation,
)
from .multi_prover_router import MultiProverRouter, PropertyKind


MCP_CONTRACT_PROVER_INTERFACE: Final = "McpContractProver@1"
MCP_CONTRACT_PROOF_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-proof-result@1"
)
MCP_PROOF_RESULT_SCHEMA: Final = MCP_CONTRACT_PROOF_RESULT_SCHEMA
MCP_CONTRACT_PROVER_VERSION: Final = "1"
MCP_LOCAL_GRAPH_CHECKER_ID: Final = "mcp-local-graph-checker@1"
MCP_LOCAL_SCHEMA_CHECKER_ID: Final = "mcp-local-schema-checker@1"
MCP_PROVIDER_TRANSLATOR_ID: Final = "mcp-provider-logic-ir@1"
MCP_NO_KERNEL_ID: Final = "mcp-no-kernel@1"

_DEFAULT_PROVIDER_IDS: Final[Mapping[str, str]] = {
    "smt": "smt",
    "cec": "dcec",
    "tdfol": "tdfol",
    "kernel": "kernel",
}
_MAX_REASON_CODES: Final = 32
_MAX_FAILED_ITEMS: Final = 64


class McpContractProverError(ValueError):
    """A proof request or result violated the MCP prover contract."""


class ContractProofOutcome(str, Enum):
    """Closed, mutually distinct terminal outcomes."""

    PROVED = "proved"
    REFUTED = "refuted"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    TIMED_OUT = "timed_out"


# Compatibility-friendly names used by downstream adapters.
McpProofOutcome = ContractProofOutcome
ProofOutcome = ContractProofOutcome
McpContractProofOutcome = ContractProofOutcome


class ContractProofRoute(str, Enum):
    """Reviewed route selected from the obligation's canonical fragment."""

    LOCAL_GRAPH = "local_graph"
    LOCAL_SCHEMA = "local_schema"
    SMT = "smt"
    CEC = "cec"
    TDFOL = "tdfol"
    KERNEL = "kernel"
    NONE = "none"


McpProofRoute = ContractProofRoute


def _outcome(value: ContractProofOutcome | str) -> ContractProofOutcome:
    if isinstance(value, ContractProofOutcome):
        return value
    aliases = {
        "disproved": ContractProofOutcome.REFUTED,
        "timeout": ContractProofOutcome.TIMED_OUT,
        "unknown": ContractProofOutcome.INCONCLUSIVE,
        "candidate": ContractProofOutcome.INCONCLUSIVE,
    }
    normalized = str(getattr(value, "value", value)).strip().lower()
    if normalized in aliases:
        return aliases[normalized]
    try:
        return ContractProofOutcome(normalized)
    except ValueError as exc:
        raise McpContractProverError(f"unknown proof outcome: {value!r}") from exc


def _route(value: ContractProofRoute | str) -> ContractProofRoute:
    if isinstance(value, ContractProofRoute):
        return value
    try:
        return ContractProofRoute(str(getattr(value, "value", value)))
    except ValueError as exc:
        raise McpContractProverError(f"unknown proof route: {value!r}") from exc


def _ids(values: Sequence[Any] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise McpContractProverError(f"{name} must be a sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise McpContractProverError(f"{name} must contain non-empty strings")
        item = value.strip()
        if "\x00" in item or len(item.encode("utf-8")) > 2_048:
            raise McpContractProverError(f"{name} contains an invalid identifier")
        result.add(item)
    return tuple(sorted(result))


def _edge(value: Any, name: str) -> tuple[str, str]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise McpContractProverError(f"{name} must contain two-item edges")
    source, target = value
    if not isinstance(source, str) or not isinstance(target, str):
        raise McpContractProverError(f"{name} edge endpoints must be strings")
    source = source.strip()
    target = target.strip()
    if not source or not target or "\x00" in source or "\x00" in target:
        raise McpContractProverError(f"{name} has an invalid edge endpoint")
    return source, target


def _edges(values: Sequence[Any] | None, name: str) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise McpContractProverError(f"{name} must be a sequence of edges")
    return tuple(sorted({_edge(value, name) for value in values}))[:_MAX_FAILED_ITEMS]


def _budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
    if value is None:
        return ResourceBudget()
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, Mapping):
        return ResourceBudget.from_dict(value)
    raise McpContractProverError("resource_budget must be a ResourceBudget or object")


def _obligation(
    value: McpContractObligation | Mapping[str, Any],
) -> McpContractObligation:
    if isinstance(value, McpContractObligation):
        return value
    if isinstance(value, Mapping):
        try:
            return McpContractObligation.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise McpContractProverError(f"invalid MCP obligation: {exc}") from exc
    raise McpContractProverError(
        "obligation must be an McpContractObligation or canonical object"
    )


@dataclass(frozen=True, slots=True)
class LocalCheckResult:
    """Small result returned by deterministic local checkers."""

    outcome: ContractProofOutcome
    failed_premise_ids: tuple[str, ...] = ()
    failed_edges: tuple[tuple[str, str], ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", _outcome(self.outcome))
        object.__setattr__(
            self,
            "failed_premise_ids",
            _ids(self.failed_premise_ids, "failed_premise_ids")[:_MAX_FAILED_ITEMS],
        )
        object.__setattr__(
            self,
            "failed_edges",
            _edges(self.failed_edges, "failed_edges")[:_MAX_FAILED_ITEMS],
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes")[:_MAX_REASON_CODES],
        )
        if self.outcome is ContractProofOutcome.PROVED and (
            self.failed_premise_ids or self.failed_edges
        ):
            raise McpContractProverError(
                "a proved local check cannot contain failed premises or edges"
            )
        if self.outcome is ContractProofOutcome.REFUTED and not (
            self.failed_premise_ids or self.failed_edges
        ):
            raise McpContractProverError(
                "a refuted local check must identify a failed premise or edge"
            )

    @classmethod
    def from_value(cls, value: Any) -> "LocalCheckResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(
                ContractProofOutcome.PROVED
                if value
                else ContractProofOutcome.INCONCLUSIVE,
                reason_codes=("local_check_passed" if value else "local_check_unknown",),
            )
        if not isinstance(value, Mapping):
            raise McpContractProverError(
                "local checker must return LocalCheckResult, boolean, or object"
            )
        raw_outcome = value.get("outcome", value.get("status", value.get("verdict")))
        if raw_outcome is None and isinstance(value.get("proved"), bool):
            raw_outcome = "proved" if value["proved"] else (
                "refuted"
                if value.get("failed_premise_ids") or value.get("failed_edges")
                else "inconclusive"
            )
        return cls(
            outcome=raw_outcome or ContractProofOutcome.INCONCLUSIVE,
            failed_premise_ids=tuple(value.get("failed_premise_ids") or ()),
            failed_edges=tuple(value.get("failed_edges") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class McpContractProofResult(CanonicalContract):
    """Canonical terminal routing result with existing proof artifacts."""

    SCHEMA = MCP_CONTRACT_PROOF_RESULT_SCHEMA

    obligation_id: str
    outcome: ContractProofOutcome
    route: ContractProofRoute
    reason_codes: tuple[str, ...]
    receipt: ProofReceipt
    counterexample: FormalCounterexample | None = None
    capability: ProofProviderCapability | None = None
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise McpContractProverError("obligation_id is required")
        object.__setattr__(self, "obligation_id", self.obligation_id.strip())
        object.__setattr__(self, "outcome", _outcome(self.outcome))
        object.__setattr__(self, "route", _route(self.route))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes")[:_MAX_REASON_CODES],
        )
        if not isinstance(self.receipt, ProofReceipt):
            raise McpContractProverError("receipt must be a ProofReceipt")
        if self.receipt.obligation_id != self.obligation_id:
            raise McpContractProverError("receipt is bound to another obligation")
        if self.counterexample is not None and not isinstance(
            self.counterexample, FormalCounterexample
        ):
            raise McpContractProverError(
                "counterexample must be a FormalCounterexample"
            )
        if self.capability is not None and not isinstance(
            self.capability, ProofProviderCapability
        ):
            raise McpContractProverError(
                "capability must be a ProofProviderCapability"
            )
        if not isinstance(self.fallback_used, bool):
            raise McpContractProverError("fallback_used must be boolean")
        if not self.reason_codes:
            raise McpContractProverError("proof result requires a reason code")
        if self.outcome is ContractProofOutcome.REFUTED:
            if self.counterexample is None:
                raise McpContractProverError(
                    "refuted result requires a compact counterexample"
                )
            if self.receipt.authoritative_verdict is not ProofVerdict.DISPROVED:
                raise McpContractProverError(
                    "refuted result requires independently checked evidence"
                )
        elif self.counterexample is not None:
            raise McpContractProverError(
                "only a refuted result may carry a counterexample"
            )
        if self.outcome is ContractProofOutcome.PROVED:
            if self.receipt.authoritative_verdict is not ProofVerdict.PROVED:
                raise McpContractProverError(
                    "proved result requires independently accepted evidence"
                )
        expected_receipt_verdict = {
            ContractProofOutcome.PROVED: ProofVerdict.PROVED,
            ContractProofOutcome.REFUTED: ProofVerdict.DISPROVED,
            ContractProofOutcome.UNSUPPORTED: ProofVerdict.UNSUPPORTED,
            ContractProofOutcome.INCONCLUSIVE: ProofVerdict.INCONCLUSIVE,
            ContractProofOutcome.TIMED_OUT: ProofVerdict.INCONCLUSIVE,
        }[self.outcome]
        if self.receipt.verdict is not expected_receipt_verdict:
            raise McpContractProverError(
                "proof outcome and submitted receipt verdict disagree"
            )

    @property
    def status(self) -> ContractProofOutcome:
        return self.outcome

    @property
    def verdict(self) -> ContractProofOutcome:
        return self.outcome

    @property
    def proof_receipt(self) -> ProofReceipt:
        return self.receipt

    @property
    def formal_counterexample(self) -> FormalCounterexample | None:
        return self.counterexample

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": MCP_CONTRACT_PROVER_INTERFACE,
            "version": MCP_CONTRACT_PROVER_VERSION,
            "obligation_id": self.obligation_id,
            "outcome": self.outcome,
            "route": self.route,
            "reason_codes": self.reason_codes,
            "receipt": self.receipt,
            "counterexample": self.counterexample,
            "capability": (
                None if self.capability is None else self.capability.to_dict()
            ),
            "fallback_used": self.fallback_used,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "McpContractProofResult":
        if not isinstance(payload, Mapping):
            raise McpContractProverError("proof result must be an object")
        schema = payload.get("schema")
        if schema not in (None, cls.SCHEMA):
            raise McpContractProverError("unsupported MCP proof-result schema")
        if payload.get("interface") not in (None, MCP_CONTRACT_PROVER_INTERFACE):
            raise McpContractProverError("unsupported MCP prover interface")
        if payload.get("version") not in (None, MCP_CONTRACT_PROVER_VERSION):
            raise McpContractProverError("unsupported MCP prover version")
        receipt = payload.get("receipt")
        counterexample = payload.get("counterexample")
        capability = payload.get("capability")
        try:
            if receipt is None:
                raise McpContractProverError("proof result requires a receipt")
            result = cls(
                obligation_id=str(payload.get("obligation_id", "")),
                outcome=payload.get("outcome", ""),
                route=payload.get("route", ""),
                reason_codes=tuple(payload.get("reason_codes") or ()),
                receipt=(
                    receipt
                    if isinstance(receipt, ProofReceipt)
                    else ProofReceipt.from_dict(receipt)
                ),
                counterexample=(
                    None
                    if counterexample is None
                    else (
                        counterexample
                        if isinstance(counterexample, FormalCounterexample)
                        else FormalCounterexample.from_dict(counterexample)
                    )
                ),
                capability=(
                    None
                    if capability is None
                    else (
                        capability
                        if isinstance(capability, ProofProviderCapability)
                        else ProofProviderCapability.from_dict(capability)
                    )
                ),
                fallback_used=payload.get("fallback_used", False),
            )
        except (TypeError, ValueError) as exc:
            raise McpContractProverError(f"invalid MCP proof result: {exc}") from exc
        claimed_id = payload.get("content_id")
        if claimed_id and claimed_id != result.content_id:
            raise McpContractProverError(
                "proof-result identity does not match canonical content"
            )
        return result


McpProofResult = McpContractProofResult
ContractProofResult = McpContractProofResult


def route_contract_obligation(
    obligation: McpContractObligation | Mapping[str, Any],
) -> ContractProofRoute:
    """Select a route without importing or invoking an optional provider."""

    normalized = _obligation(obligation)
    if not normalized.supported:
        return ContractProofRoute.NONE
    return {
        LogicFragment.GRAPH: ContractProofRoute.LOCAL_GRAPH,
        LogicFragment.SCHEMA: ContractProofRoute.LOCAL_SCHEMA,
        LogicFragment.RELATION: ContractProofRoute.SMT,
        LogicFragment.DEONTIC: ContractProofRoute.CEC,
        LogicFragment.TEMPORAL: ContractProofRoute.TDFOL,
        LogicFragment.UNSUPPORTED: ContractProofRoute.NONE,
    }[normalized.logic_fragment]


route_mcp_contract_obligation = route_contract_obligation


def _premise_observations(
    obligation: McpContractObligation,
    facts: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    raw_results = facts.get("premise_results", {})
    if raw_results is None:
        raw_results = {}
    if not isinstance(raw_results, Mapping):
        raise McpContractProverError("premise_results must be an object")
    satisfied = set(_ids(facts.get("satisfied_premise_ids", ()), "satisfied_premise_ids"))
    failed = set(_ids(facts.get("failed_premise_ids", ()), "failed_premise_ids"))
    for premise_id, result in raw_results.items():
        if premise_id not in obligation.premise_ids:
            continue
        if result is True:
            satisfied.add(premise_id)
        elif result is False:
            failed.add(premise_id)
    satisfied.difference_update(failed)
    known = satisfied | failed
    complete = set(obligation.premise_ids).issubset(known)
    return tuple(sorted(satisfied)), tuple(sorted(failed)), complete


def _default_local_graph_check(
    obligation: McpContractObligation,
    facts: Mapping[str, Any],
) -> LocalCheckResult:
    _, failed_premises, premises_complete = _premise_observations(obligation, facts)
    explicit_failed_edges = _edges(facts.get("failed_edges", ()), "failed_edges")
    required_edges = set(_edges(facts.get("required_edges", ()), "required_edges"))
    observed_edges = set(_edges(facts.get("observed_edges", ()), "observed_edges"))
    failed_edges = tuple(sorted(set(explicit_failed_edges) | (required_edges - observed_edges)))
    if failed_premises or failed_edges:
        return LocalCheckResult(
            ContractProofOutcome.REFUTED,
            failed_premise_ids=failed_premises,
            failed_edges=failed_edges,
            reason_codes=("local_graph_counterexample",),
        )
    graph_valid = facts.get("graph_valid")
    edges_complete = bool(required_edges) and required_edges.issubset(observed_edges)
    if premises_complete and (graph_valid is True or edges_complete):
        return LocalCheckResult(
            ContractProofOutcome.PROVED,
            reason_codes=("local_graph_check_passed",),
        )
    return LocalCheckResult(
        ContractProofOutcome.INCONCLUSIVE,
        reason_codes=("local_graph_evidence_incomplete",),
    )


def _default_local_schema_check(
    obligation: McpContractObligation,
    facts: Mapping[str, Any],
) -> LocalCheckResult:
    _, failed_premises, premises_complete = _premise_observations(obligation, facts)
    schema_results = facts.get("schema_results", {})
    if schema_results is None:
        schema_results = {}
    if not isinstance(schema_results, Mapping):
        raise McpContractProverError("schema_results must be an object")
    failed_schema = tuple(
        sorted(
            str(key)
            for key, value in schema_results.items()
            if value is False and str(key).strip()
        )
    )
    failed = tuple(sorted(set(failed_premises) | set(failed_schema)))
    schema_valid = facts.get("schema_valid")
    if schema_valid is False and not failed:
        failed = (obligation.property_id,)
    if failed:
        return LocalCheckResult(
            ContractProofOutcome.REFUTED,
            failed_premise_ids=failed,
            reason_codes=("local_schema_counterexample",),
        )
    schema_complete = schema_valid is True or (
        bool(schema_results) and all(value is True for value in schema_results.values())
    )
    if premises_complete and schema_complete:
        return LocalCheckResult(
            ContractProofOutcome.PROVED,
            reason_codes=("local_schema_check_passed",),
        )
    return LocalCheckResult(
        ContractProofOutcome.INCONCLUSIVE,
        reason_codes=("local_schema_evidence_incomplete",),
    )


def _counterexample(
    obligation: McpContractObligation,
    check: LocalCheckResult,
    *,
    route: ContractProofRoute,
) -> FormalCounterexample:
    raw = {
        "provider_id": route.value,
        "obligation_id": obligation.obligation_id,
        "policy_id": obligation.policy_id,
        "tree_id": obligation.snapshot_id,
        "contradiction": {
            "failed_edges": [list(edge) for edge in check.failed_edges],
        },
        "premises": list(check.failed_premise_ids),
    }
    return normalize_counterexample(
        raw,
        kind=CounterexampleKind.TDFOL_CONTRADICTION,
        bindings=CounterexampleBindings(
            tree_ids=(obligation.snapshot_id,),
            ast_scope_ids=obligation.scope_ids,
            assumption_ids=check.failed_premise_ids,
            obligation_ids=(obligation.obligation_id,),
            provider_ids=(route.value,),
            policy_ids=(obligation.policy_id,),
        ),
        property_class=f"mcp_contract:{obligation.logic_view.family.value}",
        violated_property=obligation.property_id,
        assumption_ids=check.failed_premise_ids,
    )


def _evidence_for_local(
    obligation: McpContractObligation,
    check: LocalCheckResult,
    *,
    checker_id: str,
    fact_binding_id: str,
    counterexample: FormalCounterexample | None,
) -> tuple[ProofEvidence, ...]:
    if check.outcome is ContractProofOutcome.PROVED:
        return (
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id=content_identity(
                    {
                        "checker_id": checker_id,
                        "fact_binding_id": fact_binding_id,
                        "obligation_id": obligation.obligation_id,
                        "premise_ids": list(obligation.premise_ids),
                    }
                ),
                subject_id=obligation.obligation_id,
                verifier_id=checker_id,
                independent=True,
                metadata={
                    "deterministic_local_checker": True,
                    "fact_binding_id": fact_binding_id,
                },
            ),
        )
    if check.outcome is ContractProofOutcome.REFUTED and counterexample is not None:
        return (
            ProofEvidence(
                kind=EvidenceKind.SOLVER_RESULT,
                authority=EvidenceAuthority.VALIDATION_RUNNER,
                verdict=EvidenceVerdict.REJECTED,
                artifact_id=counterexample.counterexample_id,
                subject_id=obligation.obligation_id,
                verifier_id=checker_id,
                independent=True,
                metadata={
                    "counterexample_verified": True,
                    "deterministic_local_checker": True,
                    "fact_binding_id": fact_binding_id,
                },
            ),
        )
    return ()


def _make_receipt(
    obligation: McpContractObligation,
    *,
    route: ContractProofRoute,
    outcome: ContractProofOutcome,
    budget: ResourceBudget,
    evidence: tuple[ProofEvidence, ...] = (),
    provider_id: str = "",
    provider_claimed_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED,
    reason_codes: tuple[str, ...] = (),
) -> ProofReceipt:
    kernel_id = (
        MCP_LOCAL_GRAPH_CHECKER_ID
        if route is ContractProofRoute.LOCAL_GRAPH
        else (
            MCP_LOCAL_SCHEMA_CHECKER_ID
            if route is ContractProofRoute.LOCAL_SCHEMA
            else MCP_NO_KERNEL_ID
        )
    )
    plan_id = content_identity(
        {
            "interface": MCP_CONTRACT_PROVER_INTERFACE,
            "obligation_id": obligation.obligation_id,
            "route": route.value,
            "policy_id": obligation.policy_id,
        }
    )
    attempt = ProofAttempt(
        plan_id=plan_id,
        step_id=f"mcp-contract:{route.value}",
        obligation_id=obligation.obligation_id,
        repository_tree_id=obligation.snapshot_id,
        provider_id=provider_id or route.value,
        stage=(
            ProofStage.VALIDATE
            if route in {
                ContractProofRoute.LOCAL_GRAPH,
                ContractProofRoute.LOCAL_SCHEMA,
            }
            else ProofStage.SOLVE
        ),
        status={
            ContractProofOutcome.PROVED: AttemptStatus.SUCCEEDED,
            ContractProofOutcome.REFUTED: AttemptStatus.SUCCEEDED,
            ContractProofOutcome.UNSUPPORTED: AttemptStatus.UNSUPPORTED,
            ContractProofOutcome.INCONCLUSIVE: AttemptStatus.FAILED,
            ContractProofOutcome.TIMED_OUT: AttemptStatus.TIMED_OUT,
        }[outcome],
        evidence=evidence,
        provider_claimed_assurance=provider_claimed_assurance,
        error_code="" if outcome in {
            ContractProofOutcome.PROVED,
            ContractProofOutcome.REFUTED,
        } else outcome.value,
        metadata={"route": route.value, "reason_codes": list(reason_codes)},
    )
    verdict = {
        ContractProofOutcome.PROVED: ProofVerdict.PROVED,
        ContractProofOutcome.REFUTED: ProofVerdict.DISPROVED,
        ContractProofOutcome.UNSUPPORTED: ProofVerdict.UNSUPPORTED,
        ContractProofOutcome.INCONCLUSIVE: ProofVerdict.INCONCLUSIVE,
        ContractProofOutcome.TIMED_OUT: ProofVerdict.INCONCLUSIVE,
    }[outcome]
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=plan_id,
        attempt_id=attempt.attempt_id,
        repository_id=obligation.code_obligation.repository_id,
        repository_tree_id=obligation.snapshot_id,
        ast_scope_ids=obligation.scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id=(
            kernel_id
            if route in {
                ContractProofRoute.LOCAL_GRAPH,
                ContractProofRoute.LOCAL_SCHEMA,
            }
            else MCP_PROVIDER_TRANSLATOR_ID
        ),
        solver_id=provider_id or route.value,
        kernel_id=kernel_id,
        toolchain_id=obligation.toolchain_id,
        policy_id=obligation.policy_id,
        resource_budget=budget,
        verdict=verdict,
        evidence=evidence,
        provider_id=provider_id,
        provider_claimed_assurance=provider_claimed_assurance,
        freshness=EvidenceFreshness.CURRENT,
        metadata={
            "interface": MCP_CONTRACT_PROVER_INTERFACE,
            "route": route.value,
            "reason_codes": list(reason_codes),
            "catalog_id": obligation.catalog_id,
            "compiled_obligation_id": obligation.compiled_obligation_id,
        },
    )


def _claimed_assurance(result: Mapping[str, Any]) -> AssuranceLevel:
    raw = result.get(
        "provider_claimed_assurance",
        result.get("authoritative_assurance", result.get("assurance", "unverified")),
    )
    try:
        return AssuranceLevel(str(getattr(raw, "value", raw)))
    except ValueError:
        return AssuranceLevel.UNVERIFIED


def _provider_result(response: Any) -> tuple[Mapping[str, Any] | None, ProviderFailureCode | None]:
    if isinstance(response, ProviderResponse):
        if response.ok:
            return response.result or {}, None
        assert response.error is not None
        return None, response.error.code
    if isinstance(response, Mapping):
        if "ok" in response and "request_id" in response:
            parsed = ProviderResponse.from_dict(response)
            return _provider_result(parsed)
        result = dict(response)
        try:
            content_identity(result)
        except (TypeError, ValueError) as exc:
            raise McpContractProverError(
                "provider result must contain strict canonical JSON"
            ) from exc
        return result, None
    raise McpContractProverError("provider returned a malformed response")


def _capability_from_result(
    result: Mapping[str, Any],
) -> ProofProviderCapability:
    value: Any = result.get("capability", result)
    if not isinstance(value, Mapping):
        raise McpContractProverError("provider capability result must be an object")
    try:
        return ProofProviderCapability.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise McpContractProverError(
            f"provider returned an invalid capability: {exc}"
        ) from exc


TrustedReceiptValidator = Callable[
    [McpContractObligation, Mapping[str, Any]], ProofReceipt | Mapping[str, Any] | None
]
LocalChecker = Callable[
    [McpContractObligation, Mapping[str, Any]], LocalCheckResult | Mapping[str, Any] | bool
]


class McpContractProver:
    """Route and execute one MCP obligation without optional eager imports."""

    def __init__(
        self,
        *,
        providers: Mapping[ContractProofRoute | str, Any] | None = None,
        provider_ids: Mapping[ContractProofRoute | str, str] | None = None,
        provider_getter: Callable[[str], Any | None] | None = None,
        local_graph_checker: LocalChecker | None = None,
        local_schema_checker: LocalChecker | None = None,
        trusted_receipt_validator: TrustedReceiptValidator | None = None,
        multi_prover_router: MultiProverRouter | None = None,
    ) -> None:
        self._providers = {
            _route(key): value for key, value in (providers or {}).items()
        }
        ids = {
            ContractProofRoute.SMT: _DEFAULT_PROVIDER_IDS["smt"],
            ContractProofRoute.CEC: _DEFAULT_PROVIDER_IDS["cec"],
            ContractProofRoute.TDFOL: _DEFAULT_PROVIDER_IDS["tdfol"],
            ContractProofRoute.KERNEL: _DEFAULT_PROVIDER_IDS["kernel"],
        }
        for key, value in (provider_ids or {}).items():
            route = _route(key)
            if not isinstance(value, str) or not value.strip():
                raise McpContractProverError("provider ids must be non-empty strings")
            ids[route] = value.strip()
        self._provider_ids = ids
        self._provider_getter = provider_getter or get_proof_provider
        self._local_graph_checker = local_graph_checker or _default_local_graph_check
        self._local_schema_checker = local_schema_checker or _default_local_schema_check
        if trusted_receipt_validator is not None and not callable(
            trusted_receipt_validator
        ):
            raise McpContractProverError("trusted_receipt_validator must be callable")
        self._trusted_receipt_validator = trusted_receipt_validator
        # Retain the established portfolio router as the shared policy model.
        # MCP's closed fragment routing is performed before a portfolio exists.
        self.multi_prover_router = multi_prover_router or MultiProverRouter()

    def route(
        self, obligation: McpContractObligation | Mapping[str, Any]
    ) -> ContractProofRoute:
        return route_contract_obligation(obligation)

    route_obligation = route

    def _resolve_provider(self, route: ContractProofRoute) -> tuple[Any | None, str]:
        provider_id = self._provider_ids.get(route, route.value)
        value = self._providers.get(route)
        if value is None:
            return self._provider_getter(provider_id), provider_id
        if callable(value) and not any(
            callable(getattr(value, name, None))
            for name in ("capability", "prove", "verify", "call", "invoke")
        ):
            value = value()
        return value, provider_id

    @staticmethod
    def _call_capability(provider: Any, payload: Mapping[str, Any]) -> Any:
        method = getattr(provider, "capability", None)
        if callable(method):
            try:
                return method(payload)
            except TypeError:
                return method()
        call = getattr(provider, "call", None)
        if callable(call):
            return call(ProofProviderOperation.CAPABILITY, payload)
        raise McpContractProverError(
            "provider has no operation-specific capability method"
        )

    @staticmethod
    def _call_operation(
        provider: Any,
        operation: ProofProviderOperation,
        payload: Mapping[str, Any],
        *,
        budget: ResourceBudget,
    ) -> Any:
        method = getattr(provider, operation.value, None)
        if callable(method):
            try:
                return method(payload, resource_budget=budget, network_allowed=False)
            except TypeError:
                return method(payload)
        call = getattr(provider, "call", None)
        if callable(call):
            return call(
                operation,
                payload,
                resource_budget=budget,
                network_allowed=False,
            )
        raise McpContractProverError(
            f"provider has no {operation.value} operation"
        )

    @staticmethod
    def _validate_receipt_bindings(
        obligation: McpContractObligation,
        receipt: ProofReceipt,
    ) -> None:
        expected = obligation.code_obligation
        if (
            receipt.obligation_id != obligation.obligation_id
            or receipt.repository_id != expected.repository_id
            or receipt.repository_tree_id != obligation.snapshot_id
            or receipt.ast_scope_ids != obligation.scope_ids
            or receipt.premise_ids != obligation.premise_ids
            or receipt.toolchain_id != obligation.toolchain_id
            or receipt.policy_id != obligation.policy_id
        ):
            raise McpContractProverError(
                "trusted validator returned a receipt with detached bindings"
            )

    def _trusted_result(
        self,
        obligation: McpContractObligation,
        raw_result: Mapping[str, Any],
    ) -> ProofReceipt | None:
        if self._trusted_receipt_validator is None:
            return None
        value = self._trusted_receipt_validator(obligation, raw_result)
        if value is None:
            return None
        try:
            receipt = (
                value if isinstance(value, ProofReceipt) else ProofReceipt.from_dict(value)
            )
        except (TypeError, ValueError) as exc:
            raise McpContractProverError(
                f"trusted validator returned an invalid receipt: {exc}"
            ) from exc
        self._validate_receipt_bindings(obligation, receipt)
        return receipt

    def _kernel_verify_candidate(
        self,
        obligation: McpContractObligation,
        solver_result: Mapping[str, Any],
        budget: ResourceBudget,
    ) -> tuple[Mapping[str, Any], ProviderFailureCode | None]:
        """Route a candidate through a separately probed kernel boundary.

        The returned object is still untrusted.  It is passed only to the
        configured trusted receipt validator and can never establish assurance
        by itself.
        """

        provider, configured_id = self._resolve_provider(
            ContractProofRoute.KERNEL
        )
        if provider is None:
            return {
                "solver_result": dict(solver_result),
                "kernel_provider_id": configured_id,
            }, ProviderFailureCode.UNAVAILABLE
        try:
            capability_raw = self._call_capability(
                provider,
                {
                    "interface": MCP_CONTRACT_PROVER_INTERFACE,
                    "fragment": obligation.logic_fragment.value,
                    "required_operation": ProofProviderOperation.VERIFY.value,
                },
            )
            capability_result, failure = _provider_result(capability_raw)
            if failure is not None:
                return {
                    "solver_result": dict(solver_result),
                    "kernel_provider_id": configured_id,
                }, failure
            assert capability_result is not None
            capability = _capability_from_result(capability_result)
            if not capability.supports(ProofProviderOperation.VERIFY):
                return {
                    "solver_result": dict(solver_result),
                    "kernel_provider_id": capability.provider_id,
                }, ProviderFailureCode.UNSUPPORTED
            response = self._call_operation(
                provider,
                ProofProviderOperation.VERIFY,
                {
                    "interface": MCP_CONTRACT_PROVER_INTERFACE,
                    "compiled_obligation_id": obligation.compiled_obligation_id,
                    "obligation_id": obligation.obligation_id,
                    "candidate": dict(solver_result),
                    "snapshot_id": obligation.snapshot_id,
                    "policy_id": obligation.policy_id,
                    "toolchain_id": obligation.toolchain_id,
                },
                budget=budget,
            )
            kernel_result, failure = _provider_result(response)
            if failure is not None:
                return {
                    "solver_result": dict(solver_result),
                    "kernel_provider_id": capability.provider_id,
                }, failure
            assert kernel_result is not None
            return {
                "solver_result": dict(solver_result),
                "kernel_result": dict(kernel_result),
                "kernel_provider_id": capability.provider_id,
                "kernel_provider_version": capability.provider_version,
            }, None
        except TimeoutError:
            return {
                "solver_result": dict(solver_result),
                "kernel_provider_id": configured_id,
            }, ProviderFailureCode.TIMED_OUT
        except (McpContractProverError, TypeError, ValueError):
            return {
                "solver_result": dict(solver_result),
                "kernel_provider_id": configured_id,
            }, ProviderFailureCode.MALFORMED_RESPONSE

    def _local(
        self,
        obligation: McpContractObligation,
        route: ContractProofRoute,
        facts: Mapping[str, Any],
        budget: ResourceBudget,
    ) -> McpContractProofResult:
        checker = (
            self._local_graph_checker
            if route is ContractProofRoute.LOCAL_GRAPH
            else self._local_schema_checker
        )
        checker_id = (
            MCP_LOCAL_GRAPH_CHECKER_ID
            if route is ContractProofRoute.LOCAL_GRAPH
            else MCP_LOCAL_SCHEMA_CHECKER_ID
        )
        try:
            check = LocalCheckResult.from_value(checker(obligation, facts))
        except TimeoutError:
            check = LocalCheckResult(
                ContractProofOutcome.TIMED_OUT,
                reason_codes=("local_check_timed_out",),
            )
        counterexample = (
            _counterexample(obligation, check, route=route)
            if check.outcome is ContractProofOutcome.REFUTED
            else None
        )
        satisfied, failed, premises_complete = _premise_observations(
            obligation, facts
        )
        schema_results = facts.get("schema_results", {})
        fact_binding_id = content_identity(
            {
                "obligation_id": obligation.obligation_id,
                "checker_id": checker_id,
                "satisfied_premise_ids": list(satisfied),
                "failed_premise_ids": list(failed),
                "premises_complete": premises_complete,
                "required_edges": [
                    list(edge)
                    for edge in _edges(
                        facts.get("required_edges", ()), "required_edges"
                    )
                ],
                "observed_edges": [
                    list(edge)
                    for edge in _edges(
                        facts.get("observed_edges", ()), "observed_edges"
                    )
                ],
                "schema_results": (
                    {
                        str(key): value
                        for key, value in schema_results.items()
                        if isinstance(key, str) and isinstance(value, bool)
                    }
                    if isinstance(schema_results, Mapping)
                    else {}
                ),
                "schema_valid": (
                    facts.get("schema_valid")
                    if isinstance(facts.get("schema_valid"), bool)
                    else None
                ),
                "check": {
                    "outcome": check.outcome.value,
                    "failed_premise_ids": list(check.failed_premise_ids),
                    "failed_edges": [list(edge) for edge in check.failed_edges],
                    "reason_codes": list(check.reason_codes),
                },
            }
        )
        evidence = _evidence_for_local(
            obligation,
            check,
            checker_id=checker_id,
            fact_binding_id=fact_binding_id,
            counterexample=counterexample,
        )
        reasons = check.reason_codes or (f"{route.value}_{check.outcome.value}",)
        receipt = _make_receipt(
            obligation,
            route=route,
            outcome=check.outcome,
            budget=budget,
            evidence=evidence,
            reason_codes=reasons,
        )
        return McpContractProofResult(
            obligation_id=obligation.obligation_id,
            outcome=check.outcome,
            route=route,
            reason_codes=reasons,
            receipt=receipt,
            counterexample=counterexample,
            fallback_used=True,
        )

    def _provider(
        self,
        obligation: McpContractObligation,
        route: ContractProofRoute,
        budget: ResourceBudget,
    ) -> McpContractProofResult:
        portfolio_plan = self.multi_prover_router.plan(
            obligation.code_obligation,
            property_kind=(
                PropertyKind.FINITE_CONSTRAINT
                if route is ContractProofRoute.SMT
                else PropertyKind.TEMPORAL_DEONTIC
            ),
        )
        provider, configured_id = self._resolve_provider(route)
        if provider is None:
            reasons = ("provider_unavailable",)
            receipt = _make_receipt(
                obligation,
                route=route,
                outcome=ContractProofOutcome.UNSUPPORTED,
                budget=budget,
                provider_id=configured_id,
                reason_codes=reasons,
            )
            return McpContractProofResult(
                obligation.obligation_id,
                ContractProofOutcome.UNSUPPORTED,
                route,
                reasons,
                receipt,
                fallback_used=True,
            )

        request_payload = {
            "interface": MCP_CONTRACT_PROVER_INTERFACE,
            "obligation": obligation.logic_view.to_dict(),
            "compiled_obligation_id": obligation.compiled_obligation_id,
            "portfolio_plan_id": portfolio_plan.plan_id,
            "portfolio_prover_ids": [
                lane.prover_id for lane in portfolio_plan.lanes
            ],
            "required_assurance": obligation.required_assurance.value,
            "repository_id": obligation.code_obligation.repository_id,
            "snapshot_id": obligation.snapshot_id,
            "scope_ids": list(obligation.scope_ids),
            "premise_ids": list(obligation.premise_ids),
            "policy_id": obligation.policy_id,
            "toolchain_id": obligation.toolchain_id,
        }
        try:
            capability_raw = self._call_capability(
                provider,
                {
                    "interface": MCP_CONTRACT_PROVER_INTERFACE,
                    "fragment": obligation.logic_fragment.value,
                    "required_operation": ProofProviderOperation.PROVE.value,
                },
            )
            capability_result, failure = _provider_result(capability_raw)
            if failure is not None:
                outcome = (
                    ContractProofOutcome.TIMED_OUT
                    if failure is ProviderFailureCode.TIMED_OUT
                    else ContractProofOutcome.UNSUPPORTED
                )
                reasons = (f"capability_{failure.value}",)
                receipt = _make_receipt(
                    obligation,
                    route=route,
                    outcome=outcome,
                    budget=budget,
                    provider_id=configured_id,
                    reason_codes=reasons,
                )
                return McpContractProofResult(
                    obligation.obligation_id,
                    outcome,
                    route,
                    reasons,
                    receipt,
                    fallback_used=True,
                )
            assert capability_result is not None
            capability = _capability_from_result(capability_result)
            if not capability.supports(ProofProviderOperation.PROVE):
                reasons = ("provider_operation_unsupported",)
                receipt = _make_receipt(
                    obligation,
                    route=route,
                    outcome=ContractProofOutcome.UNSUPPORTED,
                    budget=budget,
                    provider_id=capability.provider_id,
                    reason_codes=reasons,
                )
                return McpContractProofResult(
                    obligation.obligation_id,
                    ContractProofOutcome.UNSUPPORTED,
                    route,
                    reasons,
                    receipt,
                    capability=capability,
                    fallback_used=True,
                )

            response = self._call_operation(
                provider,
                ProofProviderOperation.PROVE,
                request_payload,
                budget=budget,
            )
            raw_result, failure = _provider_result(response)
        except TimeoutError:
            capability = None
            raw_result = None
            failure = ProviderFailureCode.TIMED_OUT
        except (McpContractProverError, TypeError, ValueError):
            capability = None
            raw_result = None
            failure = ProviderFailureCode.MALFORMED_RESPONSE

        if failure is not None:
            outcome = (
                ContractProofOutcome.TIMED_OUT
                if failure is ProviderFailureCode.TIMED_OUT
                else (
                    ContractProofOutcome.UNSUPPORTED
                    if failure in {
                        ProviderFailureCode.UNAVAILABLE,
                        ProviderFailureCode.UNSUPPORTED,
                    }
                    else ContractProofOutcome.INCONCLUSIVE
                )
            )
            reasons = (f"provider_{failure.value}",)
            receipt = _make_receipt(
                obligation,
                route=route,
                outcome=outcome,
                budget=budget,
                provider_id=configured_id,
                reason_codes=reasons,
            )
            return McpContractProofResult(
                obligation.obligation_id,
                outcome,
                route,
                reasons,
                receipt,
                capability=capability,
                fallback_used=True,
            )

        assert raw_result is not None
        claimed = _claimed_assurance(raw_result)
        reported = str(
            raw_result.get(
                "outcome",
                raw_result.get("status", raw_result.get("verdict", "")),
            )
        ).lower()
        validation_result: Mapping[str, Any] = raw_result
        kernel_failure: ProviderFailureCode | None = None
        if (
            reported
            in {
                "candidate",
                "proved",
                "success",
                "succeeded",
                "unsat",
                "verified",
            }
            and obligation.required_assurance.rank
            >= AssuranceLevel.KERNEL_VERIFIED.rank
        ):
            validation_result, kernel_failure = self._kernel_verify_candidate(
                obligation,
                raw_result,
                budget,
            )
        trusted = self._trusted_result(obligation, validation_result)
        if trusted is not None:
            authoritative = trusted.authoritative_verdict
            if (
                authoritative is ProofVerdict.PROVED
                and assurance_satisfies(
                    trusted.authoritative_assurance,
                    obligation.required_assurance,
                )
            ):
                return McpContractProofResult(
                    obligation.obligation_id,
                    ContractProofOutcome.PROVED,
                    route,
                    ("trusted_receipt_accepted",),
                    trusted,
                    capability=capability,
                )
            # Refutation additionally needs a canonical compact counterexample.
            if authoritative is ProofVerdict.DISPROVED:
                raw_counterexample = raw_result.get("counterexample")
                if raw_counterexample is not None:
                    try:
                        formal = normalize_counterexample(
                            raw_counterexample,
                            bindings=CounterexampleBindings(
                                tree_ids=(obligation.snapshot_id,),
                                ast_scope_ids=obligation.scope_ids,
                                obligation_ids=(obligation.obligation_id,),
                                provider_ids=(capability.provider_id,),
                                policy_ids=(obligation.policy_id,),
                            ),
                            violated_property=obligation.property_id,
                        )
                    except (TypeError, ValueError):
                        formal = None
                    if formal is not None:
                        return McpContractProofResult(
                            obligation.obligation_id,
                            ContractProofOutcome.REFUTED,
                            route,
                            ("trusted_counterexample_accepted",),
                            trusted,
                            counterexample=formal,
                            capability=capability,
                        )

        if kernel_failure is ProviderFailureCode.TIMED_OUT:
            outcome = ContractProofOutcome.TIMED_OUT
            reasons = ("kernel_timed_out",)
        elif reported in {"timed_out", "timeout"}:
            outcome = ContractProofOutcome.TIMED_OUT
            reasons = ("provider_reported_timeout",)
        elif reported in {"unsupported", "unavailable"}:
            outcome = ContractProofOutcome.UNSUPPORTED
            reasons = ("provider_reported_unsupported",)
        elif (
            claimed.rank >= AssuranceLevel.SOLVER_CHECKED.rank
            or "receipt" in raw_result
            or "proof_receipt" in raw_result
        ):
            outcome = ContractProofOutcome.INCONCLUSIVE
            reasons = ("provider_assurance_rejected",)
        elif kernel_failure is not None:
            outcome = ContractProofOutcome.INCONCLUSIVE
            reasons = (f"kernel_{kernel_failure.value}",)
        else:
            outcome = ContractProofOutcome.INCONCLUSIVE
            reasons = ("provider_candidate_requires_independent_validation",)
        candidate_kind = (
            EvidenceKind.SMT_CANDIDATE
            if route is ContractProofRoute.SMT
            else EvidenceKind.ATP_CANDIDATE
        )
        candidate = ProofEvidence(
            kind=candidate_kind,
            authority=EvidenceAuthority.PROVIDER,
            verdict=EvidenceVerdict.CANDIDATE,
            artifact_id=content_identity(
                {
                    "provider_id": capability.provider_id,
                    "provider_version": capability.provider_version,
                    "obligation_id": obligation.obligation_id,
                    "reported_outcome": reported,
                    "provider_result_id": content_identity(raw_result),
                }
            ),
            subject_id=obligation.obligation_id,
            verifier_id=capability.provider_id,
            independent=False,
            metadata={"provider_output_is_non_authoritative": True},
        )
        receipt = _make_receipt(
            obligation,
            route=route,
            outcome=outcome,
            budget=budget,
            evidence=(candidate,),
            provider_id=capability.provider_id,
            provider_claimed_assurance=claimed,
            reason_codes=reasons,
        )
        return McpContractProofResult(
            obligation.obligation_id,
            outcome,
            route,
            reasons,
            receipt,
            capability=capability,
            fallback_used=False,
        )

    def prove(
        self,
        obligation: McpContractObligation | Mapping[str, Any],
        *,
        facts: Mapping[str, Any] | None = None,
        local_evidence: Mapping[str, Any] | None = None,
        resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
    ) -> McpContractProofResult:
        """Execute the selected bounded route and return one terminal result."""

        normalized = _obligation(obligation)
        if facts is not None and local_evidence is not None and facts != local_evidence:
            raise McpContractProverError("facts and local_evidence disagree")
        observations = facts if facts is not None else local_evidence
        if observations is None:
            observations = {}
        if not isinstance(observations, Mapping):
            raise McpContractProverError("facts must be an object")
        budget = _budget(resource_budget)
        route = self.route(normalized)
        if route is ContractProofRoute.NONE:
            reasons = (
                normalized.logic_view.unsupported_reason
                or "unsupported_logic_fragment",
            )
            receipt = _make_receipt(
                normalized,
                route=route,
                outcome=ContractProofOutcome.UNSUPPORTED,
                budget=budget,
                reason_codes=reasons,
            )
            return McpContractProofResult(
                normalized.obligation_id,
                ContractProofOutcome.UNSUPPORTED,
                route,
                reasons,
                receipt,
                fallback_used=True,
            )
        if route in {
            ContractProofRoute.LOCAL_GRAPH,
            ContractProofRoute.LOCAL_SCHEMA,
        }:
            return self._local(normalized, route, observations, budget)
        return self._provider(normalized, route, budget)

    execute = prove
    prove_obligation = prove


McpContractProofRouter = McpContractProver
McpMultiProverRouter = McpContractProver


def prove_contract_obligation(
    obligation: McpContractObligation | Mapping[str, Any],
    *,
    prover: McpContractProver | None = None,
    facts: Mapping[str, Any] | None = None,
    local_evidence: Mapping[str, Any] | None = None,
    resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
) -> McpContractProofResult:
    """Convenience entry point using the deterministic default router."""

    return (prover or McpContractProver()).prove(
        obligation,
        facts=facts,
        local_evidence=local_evidence,
        resource_budget=resource_budget,
    )


prove_mcp_contract = prove_contract_obligation
route_contract_proof = route_contract_obligation


def create_mcp_contract_prover_with_datasets_logic_backends(
    *,
    importer: Callable[[str], Any] | None = None,
    trusted_receipt_validator: TrustedReceiptValidator | None = None,
    local_graph_checker: LocalChecker | None = None,
    local_schema_checker: LocalChecker | None = None,
    multi_prover_router: MultiProverRouter | None = None,
    kinds: Sequence[str] | None = None,
    invocation_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
    extra_providers: Mapping[ContractProofRoute | str, Any] | None = None,
    provider_getter: Callable[[str], Any | None] | None = None,
) -> tuple[McpContractProver, Any]:
    """Bind only capability-probed exact datasets backends into the MCP prover.

    Unregistered or unavailable backends remain unsupported.  Capability labels
    alone cannot admit a backend: registration requires an exact-module
    signature probe performed by the datasets logic facade.
    """

    # Keep the optional datasets integration lazy so importing this module does
    # not require ``ipfs_datasets_py`` in minimal installations.
    from ..integrations.ipfs_datasets_logic_provider import (
        build_datasets_logic_backend_registry,
    )

    registry, probes = build_datasets_logic_backend_registry(
        importer=importer,
        kinds=kinds,
        invocation_hook=invocation_hook,
    )
    providers: dict[ContractProofRoute | str, Any] = {
        route: provider for route, provider in registry.mcp_providers().items()
    }
    for key, value in (extra_providers or {}).items():
        providers[_route(key)] = value
    # Only registered capability-probed backends may satisfy remote routes.
    # Fall through to an explicit empty getter rather than the global registry
    # so an unregistered id cannot silently resolve to an unrelated provider.
    def _registered_only(provider_id: str) -> Any | None:
        if provider_getter is not None:
            return provider_getter(provider_id)
        for registration in registry.registrations:
            if registration.provider_id == provider_id:
                return registration.provider
        return None

    prover = McpContractProver(
        providers=providers,
        provider_ids=registry.provider_ids() or None,
        provider_getter=_registered_only,
        local_graph_checker=local_graph_checker,
        local_schema_checker=local_schema_checker,
        trusted_receipt_validator=trusted_receipt_validator,
        multi_prover_router=multi_prover_router,
    )
    # Attach registry evidence for conformance and receipts without making it
    # part of the closed constructor contract.
    prover.datasets_logic_registry = registry
    prover.datasets_logic_probes = probes
    return prover, registry


def datasets_logic_backends_are_registered(
    prover: McpContractProver,
    *routes: ContractProofRoute | str,
) -> bool:
    """Return whether every requested MCP route has a registered provider."""

    if not routes:
        return False
    for route in routes:
        normalized = _route(route)
        provider, _provider_id = prover._resolve_provider(normalized)
        if provider is None:
            return False
    return True


__all__ = [
    "ContractProofOutcome",
    "ContractProofResult",
    "ContractProofRoute",
    "FormalCounterexample",
    "LocalCheckResult",
    "MCP_CONTRACT_PROOF_RESULT_SCHEMA",
    "MCP_CONTRACT_PROVER_INTERFACE",
    "MCP_CONTRACT_PROVER_VERSION",
    "MCP_PROOF_RESULT_SCHEMA",
    "McpContractProofOutcome",
    "McpContractProofResult",
    "McpContractProofRouter",
    "McpContractProver",
    "McpContractProverError",
    "McpMultiProverRouter",
    "McpProofOutcome",
    "McpProofResult",
    "McpProofRoute",
    "MultiProverRouter",
    "ProofOutcome",
    "ProofReceipt",
    "prove_contract_obligation",
    "prove_mcp_contract",
    "route_contract_obligation",
    "route_contract_proof",
    "route_mcp_contract_obligation",
    "create_mcp_contract_prover_with_datasets_logic_backends",
    "datasets_logic_backends_are_registered",
]

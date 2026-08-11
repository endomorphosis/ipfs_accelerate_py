"""Proof orchestration for contract-repair candidates.

This is intentionally an adapter, rather than another prover.  It sends the
immutable ``CodeProofObligation`` emitted by :mod:`contract_repair_obligations`
to the capability-probed ``ipfs_datasets_py`` logic provider and treats every
provider result as a candidate until an independent reconstruction (or an
explicit, policy-approved deterministic counterexample check) is reproduced.

In particular, neither a successful solver response nor a cache row can make a
repair candidate authoritative by itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contract_repair_obligations import (
    ContractRepairObligationCompilation,
    ProofObligation,
)
from .formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_cache import (
    CacheLookupStatus,
    FormalVerificationCache,
    ProofCacheKey,
)
from .formal_verification_capabilities import ProofProviderOperation
from .formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    canonical_json,
    content_identity,
)
from .formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from .kernel_verification import (
    KernelVerificationResult,
    build_kernel_verified_receipt,
)


CONTRACT_REPAIR_PROVER_INTERFACE: Final = "ContractRepairProver@1"
CONTRACT_REPAIR_PROOF_BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-proof-bundle@1"
)
CONTRACT_REPAIR_PROOF_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-proof-result@1"
)
IPFS_DATASETS_LOGIC_PROVIDER_ID: Final = "hammer"


class ContractRepairProverError(ValueError):
    """A caller tried to weaken a proof boundary or supplied malformed input."""


class ContractRepairProofDisposition(str, Enum):
    PROVED = "proved"
    REFUTED = "refuted"
    UNSUPPORTED = "unsupported"
    NON_CONCLUSIVE = "non_conclusive"


def _ids(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairProverError(f"{name} must be a sequence")
    result = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ContractRepairProverError(f"{name} must contain non-empty identifiers")
        result.add(value.strip())
    return tuple(sorted(result))


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractRepairProverError(f"{name} must be an object")
    try:
        import json

        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ContractRepairProverError(f"{name} must contain canonical JSON") from exc
    if not isinstance(normalized, dict):  # defensive; canonical JSON preserves mappings
        raise ContractRepairProverError(f"{name} must be an object")
    return normalized


def _failure_reason(code: ProviderFailureCode | None) -> str:
    return {
        ProviderFailureCode.TIMED_OUT: "proof_timed_out",
        ProviderFailureCode.UNSUPPORTED: "unsupported_semantics",
        ProviderFailureCode.UNAVAILABLE: "backend_unavailable",
        ProviderFailureCode.MALFORMED_RESPONSE: "malformed_backend_response",
        ProviderFailureCode.MALFORMED_REQUEST: "malformed_backend_request",
    }.get(code, "backend_non_conclusive")


@dataclass(frozen=True)
class CandidateProofResult:
    """One mandatory obligation outcome, with authority projected from receipt."""

    obligation_id: str
    receipt: ProofReceipt
    disposition: ContractRepairProofDisposition
    reason_codes: tuple[str, ...]
    cache_key_id: str
    counterexample: FormalCounterexample | None = None
    from_cache: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise ContractRepairProverError("obligation_id is required")
        object.__setattr__(self, "obligation_id", self.obligation_id.strip())
        if not isinstance(self.receipt, ProofReceipt):
            raise ContractRepairProverError("proof result requires a typed receipt")
        object.__setattr__(self, "disposition", ContractRepairProofDisposition(self.disposition))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        if not self.reason_codes:
            raise ContractRepairProverError("proof result requires a reason code")
        if not isinstance(self.cache_key_id, str) or not self.cache_key_id.strip():
            raise ContractRepairProverError("cache_key_id is required")
        if not isinstance(self.from_cache, bool):
            raise ContractRepairProverError("from_cache must be boolean")
        if self.counterexample is not None and not isinstance(self.counterexample, FormalCounterexample):
            raise ContractRepairProverError("counterexample must be a FormalCounterexample")
        if self.disposition is ContractRepairProofDisposition.PROVED:
            if not self.receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED):
                raise ContractRepairProverError("proved result requires current independent reconstruction")
            if self.counterexample is not None:
                raise ContractRepairProverError("proved result cannot carry a counterexample")
        elif self.disposition is ContractRepairProofDisposition.REFUTED:
            if self.receipt.authoritative_verdict is not ProofVerdict.DISPROVED:
                raise ContractRepairProverError("refuted result requires independently verified model")
            if self.counterexample is None:
                raise ContractRepairProverError("refuted result requires a minimal counterexample")

    @property
    def authoritative(self) -> bool:
        return self.disposition is ContractRepairProofDisposition.PROVED and self.receipt.satisfies_completion(
            AssuranceLevel.KERNEL_VERIFIED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_PROOF_RESULT_SCHEMA,
            "obligation_id": self.obligation_id,
            "receipt": self.receipt.to_dict(),
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "cache_key_id": self.cache_key_id,
            "counterexample_id": self.counterexample.counterexample_id if self.counterexample else "",
            "from_cache": self.from_cache,
            "candidate_authoritative": self.authoritative,
        }


@dataclass(frozen=True)
class CandidateProofBundle:
    """All mandatory claims for one exact candidate and repository tree."""

    candidate_id: str
    repository_id: str
    tree_id: str
    results: tuple[CandidateProofResult, ...]
    backend_id: str
    backend_version: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("candidate_id", "repository_id", "tree_id", "backend_id", "backend_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractRepairProverError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        if not self.results or not all(isinstance(item, CandidateProofResult) for item in self.results):
            raise ContractRepairProverError("proof bundle requires results")
        ids = [item.obligation_id for item in self.results]
        if len(ids) != len(set(ids)):
            raise ContractRepairProverError("proof bundle cannot repeat obligations")
        object.__setattr__(self, "results", tuple(sorted(self.results, key=lambda item: item.obligation_id)))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))

    @property
    def candidate_authoritative(self) -> bool:
        """True only after every mandatory obligation has an independent receipt."""

        return all(item.authoritative for item in self.results)

    @property
    def counterexample_refs(self) -> tuple[str, ...]:
        return tuple(sorted({item.counterexample.counterexample_id for item in self.results if item.counterexample}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_PROOF_BUNDLE_SCHEMA,
            "interface": CONTRACT_REPAIR_PROVER_INTERFACE,
            "candidate_id": self.candidate_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "results": [item.to_dict() for item in self.results],
            "reason_codes": list(self.reason_codes),
            "counterexample_refs": list(self.counterexample_refs),
            "candidate_authoritative": self.candidate_authoritative,
            "bundle_id": content_identity({
                "candidate_id": self.candidate_id, "tree_id": self.tree_id,
                "results": [item.to_dict() for item in self.results],
            }),
        }


CounterexampleVerifier = Callable[[ProofObligation, Mapping[str, Any]], FormalCounterexample | Mapping[str, Any] | None]


class ContractRepairProver:
    """Run compiled repair claims through the admitted datasets logic backend."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        cache: FormalVerificationCache | None = None,
        resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
        counterexample_verifier: CounterexampleVerifier | None = None,
    ) -> None:
        if cache is not None and not isinstance(cache, FormalVerificationCache):
            raise ContractRepairProverError("cache must be a FormalVerificationCache")
        if counterexample_verifier is not None and not callable(counterexample_verifier):
            raise ContractRepairProverError("counterexample_verifier must be callable")
        self.backend = backend if backend is not None else self._default_backend()
        self.cache = cache
        self.resource_budget = self._budget(resource_budget)
        self.counterexample_verifier = counterexample_verifier

    @staticmethod
    def _default_backend() -> Any:
        # Lazy import preserves capability probing/import isolation in minimal installs.
        from ..integrations.ipfs_datasets_logic_provider import IpfsDatasetsLogicProvider

        return IpfsDatasetsLogicProvider()

    @staticmethod
    def _budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
        if value is None:
            return ResourceBudget()
        if isinstance(value, ResourceBudget):
            return value
        if isinstance(value, Mapping):
            return ResourceBudget.from_dict(value)
        raise ContractRepairProverError("resource_budget must be a ResourceBudget or object")

    def _backend_identity(self) -> tuple[str, str]:
        provider_id = str(getattr(self.backend, "provider_id", "")).strip()
        provider_version = str(getattr(self.backend, "provider_version", "")).strip()
        if not provider_id or not provider_version:
            return "unavailable", "unavailable"
        return provider_id, provider_version

    def _backend_supports(self, operation: ProofProviderOperation) -> bool:
        # The stable adapter identifier is part of the compatibility contract;
        # do not route repair authority through an arbitrary generic provider.
        if self._backend_identity()[0] != IPFS_DATASETS_LOGIC_PROVIDER_ID:
            return False
        capability_method = getattr(self.backend, "capabilities", None)
        if not callable(capability_method):
            return False
        try:
            capability = capability_method()
            operations = getattr(capability, "operations", ())
            return operation in operations
        except (TypeError, ValueError, AttributeError):
            return False

    def _cache_key(self, obligation: ProofObligation, premises: tuple[dict[str, Any], ...]) -> ProofCacheKey:
        claim = obligation.claim
        backend_id, backend_version = self._backend_identity()
        return ProofCacheKey(
            obligation={"repair_obligation": obligation.to_dict(), "logic_ir": claim.to_logic_ir()},
            premises=premises,
            translator={"id": claim.translator_id, "capability": claim.capability_id, "revision": claim.capability_revision},
            solver={"provider_id": backend_id, "provider_version": backend_version},
            kernel={"required": "independent-reconstruction", "capability": claim.capability_id},
            toolchain={"id": claim.toolchain_id, "backend_version": backend_version},
            theorem_registry={"source_ids": list(claim.source_ids), "assumption_ids": list(claim.assumption_ids)},
            policy={"id": claim.policy_id, "required_assurance": AssuranceLevel.KERNEL_VERIFIED.value},
            resource_budget=self.resource_budget.to_dict(),
            candidate_tree={"repository_id": claim.repository_id, "tree_id": claim.tree_id, "candidate_id": obligation.candidate_id},
        )

    @staticmethod
    def _premises_for(obligation: ProofObligation, premises: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        if isinstance(premises, Mapping):
            raw = []
            for premise_id in obligation.claim.premise_ids:
                value = premises.get(premise_id)
                if value is None:
                    raise ContractRepairProverError("incomplete_premise_slice")
                if not isinstance(value, Mapping):
                    raise ContractRepairProverError("premise records must be objects")
                item = dict(value)
                item.setdefault("premise_id", premise_id)
                raw.append(item)
        elif isinstance(premises, Sequence) and not isinstance(premises, (str, bytes, bytearray)):
            by_id = {str(item.get("premise_id", "")): item for item in premises if isinstance(item, Mapping)}
            raw = [dict(by_id[premise_id]) for premise_id in obligation.claim.premise_ids if premise_id in by_id]
            if len(raw) != len(obligation.claim.premise_ids):
                raise ContractRepairProverError("incomplete_premise_slice")
        else:
            raise ContractRepairProverError("premises must be a mapping or sequence")
        normalized = tuple(_canonical_mapping(item, "premise") for item in raw)
        if {str(item.get("premise_id", "")) for item in normalized} != set(obligation.claim.premise_ids):
            raise ContractRepairProverError("premises do not bind the exact claim")
        return tuple(sorted(normalized, key=lambda item: str(item["premise_id"])))

    def _non_conclusive_receipt(self, obligation: ProofObligation, *, verdict: ProofVerdict, reason: str, backend_id: str) -> ProofReceipt:
        code = obligation.code_obligation
        return ProofReceipt(
            obligation_id=code.obligation_id,
            plan_id=content_identity({"interface": CONTRACT_REPAIR_PROVER_INTERFACE, "obligation": obligation.obligation_id}),
            attempt_id=content_identity({"reason": reason, "obligation": obligation.obligation_id}),
            repository_id=code.repository_id,
            repository_tree_id=code.repository_tree_id,
            ast_scope_ids=code.ast_scope_ids,
            premise_ids=code.premise_ids,
            translator_id=obligation.claim.translator_id,
            solver_id=backend_id or "unavailable",
            kernel_id="independent-reconstruction-required",
            toolchain_id=obligation.claim.toolchain_id,
            policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget,
            verdict=verdict,
            freshness=EvidenceFreshness.CURRENT,
            metadata={"reason_codes": [reason], "repair_obligation_id": obligation.obligation_id},
        )

    def _result(self, obligation: ProofObligation, *, receipt: ProofReceipt, disposition: ContractRepairProofDisposition,
                reasons: Sequence[str], key: ProofCacheKey, counterexample: FormalCounterexample | None = None,
                from_cache: bool = False) -> CandidateProofResult:
        return CandidateProofResult(obligation.obligation_id, receipt, disposition, tuple(reasons), key.key_id, counterexample, from_cache)

    def _reconstruction_receipt(
        self,
        obligation: ProofObligation,
        result: Mapping[str, Any],
        *,
        request_id: str,
        candidate_id: str,
    ) -> ProofReceipt | None:
        raw = result.get("kernel_verification")
        if not isinstance(raw, Mapping):
            return None
        try:
            verification = KernelVerificationResult.from_dict(raw)
        except (TypeError, ValueError):
            return None
        code = obligation.code_obligation
        if (
            verification.obligation_id != code.obligation_id
            or verification.request_id != request_id
            or verification.candidate_id != candidate_id
            or verification.toolchain_id != obligation.claim.toolchain_id
            or verification.verdict is not ProofVerdict.PROVED
        ):
            return None
        receipt = build_kernel_verified_receipt(
            verification, obligation=code,
            plan_id=content_identity({"interface": CONTRACT_REPAIR_PROVER_INTERFACE, "obligation": obligation.obligation_id}),
            attempt_id=verification.request_id,
            translator_id=obligation.claim.translator_id,
            solver_id=self._backend_identity()[0], policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget, provider_id=self._backend_identity()[0],
            theorem_registry_id=content_identity({"sources": list(obligation.claim.source_ids)}),
            metadata={"repair_obligation_id": obligation.obligation_id, "claim_id": obligation.claim.content_id},
        )
        if (
            receipt.repository_tree_id != obligation.claim.tree_id
            or receipt.toolchain_id != obligation.claim.toolchain_id
            or receipt.policy_id != obligation.claim.policy_id
            or not receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED)
        ):
            return None
        return receipt

    def _candidate_counterexample(
        self, obligation: ProofObligation, raw: Mapping[str, Any]
    ) -> FormalCounterexample | None:
        """Keep a minimized model reference even when it is not authoritative.

        This deliberately does *not* change the receipt verdict.  It gives the
        later decision/review path a compact witness while preserving the rule
        that solver models require an independent checker before refutation.
        """

        try:
            return normalize_counterexample(
                raw,
                kind=CounterexampleKind.SMT_MODEL,
                bindings=CounterexampleBindings(
                    tree_ids=(obligation.claim.tree_id,),
                    obligation_ids=(obligation.code_obligation.obligation_id,),
                    provider_ids=(self._backend_identity()[0],),
                    policy_ids=(obligation.claim.policy_id,),
                ),
                violated_property=obligation.claim.predicate,
            )
        except (TypeError, ValueError):
            return None

    def _verified_counterexample(self, obligation: ProofObligation, raw: Mapping[str, Any]) -> FormalCounterexample | None:
        if self.counterexample_verifier is None:
            return None
        try:
            candidate = self.counterexample_verifier(obligation, raw)
            if candidate is None:
                return None
            if isinstance(candidate, FormalCounterexample):
                result = candidate
            elif isinstance(candidate, Mapping):
                result = normalize_counterexample(
                    candidate, kind=CounterexampleKind.SMT_MODEL,
                    bindings=CounterexampleBindings(
                        tree_ids=(obligation.claim.tree_id,),
                        obligation_ids=(obligation.code_obligation.obligation_id,),
                        provider_ids=(self._backend_identity()[0],), policy_ids=(obligation.claim.policy_id,),
                    ),
                    violated_property=obligation.claim.predicate,
                )
            else:
                return None
        except (TypeError, ValueError):
            return None
        bindings = result.bindings
        if obligation.claim.tree_id not in bindings.tree_ids or obligation.code_obligation.obligation_id not in bindings.obligation_ids:
            return None
        return result

    def _refuted_receipt(self, obligation: ProofObligation, counterexample: FormalCounterexample) -> ProofReceipt:
        code = obligation.code_obligation
        evidence = ProofEvidence(
            kind=EvidenceKind.SOLVER_RESULT, authority=EvidenceAuthority.VALIDATION_RUNNER,
            verdict=EvidenceVerdict.REJECTED, artifact_id=counterexample.counterexample_id,
            subject_id=code.obligation_id, verifier_id="policy-approved-counterexample-checker",
            independent=True, metadata={"counterexample_verified": True, "repair_obligation_id": obligation.obligation_id},
        )
        return ProofReceipt(
            obligation_id=code.obligation_id, plan_id=content_identity({"counterexample": counterexample.counterexample_id}),
            attempt_id=counterexample.counterexample_id, repository_id=code.repository_id,
            repository_tree_id=code.repository_tree_id, ast_scope_ids=code.ast_scope_ids,
            premise_ids=code.premise_ids, translator_id=obligation.claim.translator_id,
            solver_id=self._backend_identity()[0], kernel_id="policy-approved-counterexample-checker",
            toolchain_id=obligation.claim.toolchain_id, policy_id=obligation.claim.policy_id,
            resource_budget=self.resource_budget, verdict=ProofVerdict.DISPROVED, evidence=(evidence,),
            freshness=EvidenceFreshness.CURRENT,
        )

    def prove_obligation(self, obligation: ProofObligation, *, premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                         reconstruction_inputs: Mapping[str, Any] | None = None) -> CandidateProofResult:
        if not isinstance(obligation, ProofObligation):
            raise ContractRepairProverError("obligation must be a ProofObligation")
        backend_id, _ = self._backend_identity()
        try:
            exact_premises = self._premises_for(obligation, premises)
        except ContractRepairProverError as exc:
            key = self._cache_key(obligation, ())
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.INCONCLUSIVE, reason=str(exc), backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.NON_CONCLUSIVE, reasons=(str(exc),), key=key)
        key = self._cache_key(obligation, exact_premises)
        if self.cache is not None:
            hit = self.cache.lookup(key, required_assurance=AssuranceLevel.KERNEL_VERIFIED, required_freshness=EvidenceFreshness.CURRENT)
            if hit.status is CacheLookupStatus.HIT and hit.receipt is not None:
                receipt = hit.receipt
                if (
                    receipt.obligation_id == obligation.code_obligation.obligation_id
                    and receipt.repository_tree_id == obligation.claim.tree_id
                    and receipt.toolchain_id == obligation.claim.toolchain_id
                    and receipt.policy_id == obligation.claim.policy_id
                    and receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED)
                ):
                    return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.PROVED,
                                        reasons=("authoritative_cache_hit",), key=key, from_cache=True)
        if not self._backend_supports(ProofProviderOperation.PROVE):
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.UNSUPPORTED, reason="missing_backend", backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.UNSUPPORTED, reasons=("missing_backend",), key=key)
        request = ProviderRequest(
            request_id=content_identity({"obligation": obligation.obligation_id, "cache_key": key.key_id})[-64:],
            operation=ProofProviderOperation.PROVE,
            payload={"obligation": obligation.code_obligation.to_dict(), "premises": list(exact_premises),
                     "logic_ir_claim": obligation.claim.to_logic_ir(), "contract_repair_obligation_id": obligation.obligation_id},
            resource_budget=self.resource_budget,
        )
        response = dispatch_provider_request(self.backend, request)
        if not response.ok:
            assert response.error is not None
            reason = _failure_reason(response.error.code)
            verdict = ProofVerdict.UNSUPPORTED if response.error.code in {ProviderFailureCode.UNSUPPORTED, ProviderFailureCode.UNAVAILABLE} else ProofVerdict.INCONCLUSIVE
            disposition = ContractRepairProofDisposition.UNSUPPORTED if verdict is ProofVerdict.UNSUPPORTED else ContractRepairProofDisposition.NON_CONCLUSIVE
            receipt = self._non_conclusive_receipt(obligation, verdict=verdict, reason=reason, backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=disposition, reasons=(reason,), key=key)
        result = response.result or {}
        status = str(result.get("status", "")).strip().lower()
        raw_counterexample = result.get("counterexample")
        if status in {"counterexample", "refuted", "disproved", "sat"} and isinstance(raw_counterexample, Mapping):
            counterexample = self._verified_counterexample(obligation, raw_counterexample)
            if counterexample is not None:
                receipt = self._refuted_receipt(obligation, counterexample)
                return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.REFUTED,
                                    reasons=("independently_verified_counterexample",), key=key, counterexample=counterexample)
            counterexample = self._candidate_counterexample(obligation, raw_counterexample)
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.INCONCLUSIVE, reason="unverified_counterexample", backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.NON_CONCLUSIVE,
                                reasons=("unverified_counterexample",), key=key, counterexample=counterexample)
        candidate = result.get("proof_candidate")
        if not isinstance(candidate, Mapping) and isinstance(result.get("hammer_result"), Mapping):
            candidate = result["hammer_result"].get("proof_candidate")
        if not isinstance(candidate, Mapping):
            reason = "unknown_backend_result" if status in {"", "unknown", "candidate"} else "backend_non_conclusive"
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.INCONCLUSIVE, reason=reason, backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.NON_CONCLUSIVE, reasons=(reason,), key=key)
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        candidate_request_id = str(candidate.get("request_id", "")).strip()
        if not candidate_id or candidate_request_id != request.request_id:
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.INCONCLUSIVE, reason="wrong_candidate_or_theorem", backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.NON_CONCLUSIVE,
                                reasons=("wrong_candidate_or_theorem",), key=key)
        if not self._backend_supports(ProofProviderOperation.RECONSTRUCT):
            receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.UNSUPPORTED, reason="independent_reconstruction_unavailable", backend_id=backend_id)
            return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.UNSUPPORTED, reasons=("independent_reconstruction_unavailable",), key=key)
        extras = _canonical_mapping(reconstruction_inputs or {}, "reconstruction_inputs")
        reconstruction_request = ProviderRequest(
            request_id=request.request_id, operation=ProofProviderOperation.RECONSTRUCT,
            payload={**dict(request.payload), **extras, "proof_candidate": dict(candidate)}, resource_budget=self.resource_budget,
        )
        reconstruction = dispatch_provider_request(self.backend, reconstruction_request)
        if reconstruction.ok:
            receipt = self._reconstruction_receipt(
                obligation, reconstruction.result or {}, request_id=request.request_id,
                candidate_id=candidate_id,
            )
            if receipt is not None:
                if self.cache is not None:
                    self.cache.put(key, receipt)
                return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.PROVED,
                                    reasons=("independent_reconstruction_accepted",), key=key)
        reason = "malformed_or_wrong_theorem_reconstruction" if reconstruction.ok else _failure_reason(reconstruction.error.code if reconstruction.error else None)
        receipt = self._non_conclusive_receipt(obligation, verdict=ProofVerdict.INCONCLUSIVE, reason=reason, backend_id=backend_id)
        return self._result(obligation, receipt=receipt, disposition=ContractRepairProofDisposition.NON_CONCLUSIVE, reasons=(reason,), key=key)

    def prove(self, compilation: ContractRepairObligationCompilation, *, premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
              reconstruction_inputs: Mapping[str, Any] | None = None) -> CandidateProofBundle:
        if not isinstance(compilation, ContractRepairObligationCompilation):
            raise ContractRepairProverError("compilation must be ContractRepairObligationCompilation")
        backend_id, backend_version = self._backend_identity()
        results = tuple(self.prove_obligation(item, premises=premises, reconstruction_inputs=reconstruction_inputs) for item in compilation.obligations)
        reasons = tuple(sorted({reason for item in results for reason in item.reason_codes}))
        return CandidateProofBundle(compilation.candidate_id, compilation.roots.repository_id, compilation.roots.tree_id,
                                    results, backend_id, backend_version, reasons)

    prove_candidate = prove


def reconstruct_contract_repair_proof(
    obligation: ProofObligation,
    *,
    backend: Any,
    premises: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    reconstruction_inputs: Mapping[str, Any] | None = None,
    cache: FormalVerificationCache | None = None,
) -> CandidateProofResult:
    """One-obligation convenience entry point retaining all fail-closed checks."""

    return ContractRepairProver(backend, cache=cache).prove_obligation(
        obligation, premises=premises, reconstruction_inputs=reconstruction_inputs
    )


__all__ = [
    "CONTRACT_REPAIR_PROOF_BUNDLE_SCHEMA", "CONTRACT_REPAIR_PROOF_RESULT_SCHEMA",
    "CONTRACT_REPAIR_PROVER_INTERFACE", "CandidateProofBundle", "CandidateProofResult",
    "ContractRepairProofDisposition", "ContractRepairProver", "ContractRepairProverError",
    "reconstruct_contract_repair_proof",
]

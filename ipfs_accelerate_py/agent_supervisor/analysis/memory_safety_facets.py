"""Fail-closed collection of memory-safety evidence at native boundaries.

``ProgramContract.max_memory_bytes`` limits resource consumption; it says
nothing about ownership, bounds checks, use-after-free, or ABI correctness.
This module deliberately keeps that distinction intact.  It turns *already
collected* receipts into the compact :class:`MemorySafetyFacet` contract, but
does not run compilers, sanitizers, package managers, or tests itself.

The collector is conservative by design.  A passing unit test, cgroup receipt,
or memory limit can make an observation empirical, never memory-safe.  Native
proof claims require a policy-selected set of current, scope-covering receipts.
Managed Python and TypeScript analysis remains a model-support result: their
reflection, monkey-patching, native-extension, and FFI frontiers cannot be
promoted to a general memory-safety claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    SourceSpan,
)


MEMORY_SAFETY_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/memory-safety-evidence@1"
)
MEMORY_SAFETY_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/memory-safety-policy@1"
)
MAX_EVIDENCE_ITEMS: Final[int] = 256


class MemorySafetyEvidenceError(ValueError):
    """Malformed, ambiguous, or non-local memory-safety evidence."""


class NativeBoundaryKind(str, Enum):
    """Boundaries that must be considered separately from resource bounds."""

    FFI = "ffi"
    NATIVE_EXTENSION = "native_extension"
    NATIVE_ADDON = "native_addon"
    UNSAFE = "unsafe"
    ALLOCATOR = "allocator"
    DEALLOCATOR = "deallocator"
    SERIALIZATION = "serialization"
    REFLECTION = "reflection"
    MONKEY_PATCH = "monkey_patch"
    UNMODELED_SERVICE = "unmodeled_service"
    UNKNOWN = "unknown"


class MemorySafetyReceiptKind(str, Enum):
    """Closed receipt vocabulary.  Not every receipt kind is proof evidence."""

    BORROW_CHECKER = "borrow_checker"
    MIRI = "miri"
    ASAN = "asan"
    UBSAN = "ubsan"
    EQUIVALENT_VERIFIER = "equivalent_verifier"
    STATIC_ANALYZER = "static_analyzer"
    MODEL_CHECKER = "model_checker"
    COMPILER = "compiler"
    UNIT_TEST = "unit_test"
    CGROUP = "cgroup"
    MEMORY_LIMIT = "memory_limit"


class MemorySafetyReceiptState(str, Enum):
    """Result state reported by the producer of one receipt."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    STALE = "stale"


_NATIVE_LANGUAGES: Final[frozenset[str]] = frozenset(
    {"c", "c++", "cpp", "rust", "zig", "swift", "objective-c", "objc"}
)
_MANAGED_LANGUAGES: Final[frozenset[str]] = frozenset(
    {"python", "typescript", "javascript", "node", "nodejs"}
)
_UNMODELED_BOUNDARIES: Final[frozenset[NativeBoundaryKind]] = frozenset(
    {
        NativeBoundaryKind.REFLECTION,
        NativeBoundaryKind.MONKEY_PATCH,
        NativeBoundaryKind.UNMODELED_SERVICE,
        NativeBoundaryKind.UNKNOWN,
    }
)
_MANAGED_NATIVE_BOUNDARIES: Final[frozenset[NativeBoundaryKind]] = frozenset(
    {
        NativeBoundaryKind.FFI,
        NativeBoundaryKind.NATIVE_EXTENSION,
        NativeBoundaryKind.NATIVE_ADDON,
    }
)
_OBSERVATION_ONLY_KINDS: Final[frozenset[MemorySafetyReceiptKind]] = frozenset(
    {
        MemorySafetyReceiptKind.UNIT_TEST,
        MemorySafetyReceiptKind.CGROUP,
        MemorySafetyReceiptKind.MEMORY_LIMIT,
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise MemorySafetyEvidenceError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise MemorySafetyEvidenceError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > 4096:
        raise MemorySafetyEvidenceError(f"{name} exceeds its byte bound")
    return value


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise MemorySafetyEvidenceError(f"{name} must be one of: {choices}") from exc


def _refs(values: Iterable[EvidenceReference], name: str) -> tuple[EvidenceReference, ...]:
    result = tuple(values)
    if len(result) > MAX_EVIDENCE_ITEMS:
        raise MemorySafetyEvidenceError(f"{name} exceeds its item bound")
    if not all(isinstance(item, EvidenceReference) for item in result):
        raise MemorySafetyEvidenceError(f"{name} must contain EvidenceReference values")
    return tuple(sorted(set(result), key=lambda item: item.content_id))


@dataclass(frozen=True)
class NativeBoundary:
    """A native, dynamic, allocator, or serialization boundary in one scope."""

    boundary_id: str
    kind: NativeBoundaryKind
    span: SourceSpan
    foreign_language_runtime: str = ""
    evidence_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_id", _text(self.boundary_id, "boundary_id"))
        object.__setattr__(self, "kind", _enum(self.kind, NativeBoundaryKind, "kind"))
        if not isinstance(self.span, SourceSpan):
            raise MemorySafetyEvidenceError("boundary span must be a SourceSpan")
        object.__setattr__(
            self,
            "foreign_language_runtime",
            _text(self.foreign_language_runtime, "foreign_language_runtime", required=False),
        )
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "boundary evidence_refs"))


@dataclass(frozen=True)
class ProofEvidence:
    """A receipt with the exact roots and scope needed to make it reusable.

    ``scope_ids`` are explicit because a repository-wide sanitizer run does not
    automatically cover an arbitrary generated module.  The subject span's
    content identity is accepted as a concise scope identifier by the collector.
    """

    evidence_ref: EvidenceReference
    receipt_kind: MemorySafetyReceiptKind
    language_runtime: str
    toolchain_id: str
    tree_id: str
    scope_ids: tuple[str, ...]
    state: MemorySafetyReceiptState = MemorySafetyReceiptState.PASSED

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_ref, EvidenceReference):
            raise MemorySafetyEvidenceError("evidence_ref must be an EvidenceReference")
        object.__setattr__(self, "receipt_kind", _enum(self.receipt_kind, MemorySafetyReceiptKind, "receipt_kind"))
        object.__setattr__(self, "language_runtime", _text(self.language_runtime, "language_runtime").lower())
        object.__setattr__(self, "toolchain_id", _text(self.toolchain_id, "toolchain_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "state", _enum(self.state, MemorySafetyReceiptState, "state"))
        scopes = tuple(sorted({_text(item, "scope_id") for item in self.scope_ids}))
        if not scopes:
            raise MemorySafetyEvidenceError("scope_ids must not be empty")
        if len(scopes) > MAX_EVIDENCE_ITEMS:
            raise MemorySafetyEvidenceError("scope_ids exceeds its item bound")
        object.__setattr__(self, "scope_ids", scopes)


# A more discoverable spelling for integrations that call all evidence a receipt.
MemorySafetyReceipt = ProofEvidence


@dataclass(frozen=True)
class MemorySafetyPolicy:
    """Policy-selected native proof requirements.

    Every inner group is an OR-set; every group is required.  For example,
    ``((BORROW_CHECKER,), (MIRI, ASAN, UBSAN))`` requires a borrow-checker
    receipt and one dynamic verifier.  An empty requirement is rejected rather
    than interpreted as a permissive native-proof policy.
    """

    policy_id: str = "memory-safety-policy@1"
    native_proof_groups: tuple[tuple[MemorySafetyReceiptKind, ...], ...] = (
        (
            MemorySafetyReceiptKind.BORROW_CHECKER,
            MemorySafetyReceiptKind.MIRI,
            MemorySafetyReceiptKind.ASAN,
            MemorySafetyReceiptKind.UBSAN,
            MemorySafetyReceiptKind.EQUIVALENT_VERIFIER,
        ),
    )
    require_native_proof: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        if not isinstance(self.require_native_proof, bool):
            raise MemorySafetyEvidenceError("require_native_proof must be a bool")
        groups: list[tuple[MemorySafetyReceiptKind, ...]] = []
        for group in self.native_proof_groups:
            if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
                raise MemorySafetyEvidenceError("native_proof_groups must contain sequences")
            normalized = tuple(sorted({_enum(item, MemorySafetyReceiptKind, "native receipt kind") for item in group}, key=lambda item: item.value))
            if not normalized:
                raise MemorySafetyEvidenceError("native proof groups must not be empty")
            if set(normalized) <= _OBSERVATION_ONLY_KINDS:
                raise MemorySafetyEvidenceError("native proof policy cannot require only observational receipts")
            groups.append(normalized)
        if self.require_native_proof and not groups:
            raise MemorySafetyEvidenceError("native proof policy must require at least one proof group")
        object.__setattr__(self, "native_proof_groups", tuple(groups))


@dataclass(frozen=True)
class MemorySafetyAssessment:
    """Detailed result; the embedded facet is the bounded interchange record."""

    facet: MemorySafetyFacet
    memory_safe: bool
    reason_codes: tuple[str, ...] = ()
    accepted_receipt_refs: tuple[EvidenceReference, ...] = ()
    rejected_receipt_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.facet, MemorySafetyFacet):
            raise MemorySafetyEvidenceError("facet must be a MemorySafetyFacet")
        if self.memory_safe != (self.facet.disposition is MemorySafetyDisposition.PROVED):
            raise MemorySafetyEvidenceError("memory_safe is true only for a proved facet")
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "accepted_receipt_refs", _refs(self.accepted_receipt_refs, "accepted_receipt_refs"))
        object.__setattr__(self, "rejected_receipt_refs", _refs(self.rejected_receipt_refs, "rejected_receipt_refs"))


class MemorySafetyEvidenceCollector:
    """Convert supplied evidence into a root-, scope-, and policy-bound facet."""

    def __init__(self, roots: AuthorityRoots, policy: MemorySafetyPolicy | None = None) -> None:
        if not isinstance(roots, AuthorityRoots):
            raise MemorySafetyEvidenceError("roots must be AuthorityRoots")
        self.roots = roots
        self.policy = policy if policy is not None else MemorySafetyPolicy()
        if not isinstance(self.policy, MemorySafetyPolicy):
            raise MemorySafetyEvidenceError("policy must be MemorySafetyPolicy")

    @staticmethod
    def _scope_ids(subject_span: SourceSpan, scope_ids: Iterable[str]) -> frozenset[str]:
        if not isinstance(subject_span, SourceSpan):
            raise MemorySafetyEvidenceError("subject_span must be a SourceSpan")
        supplied = {_text(item, "scope_id") for item in scope_ids}
        # A caller may use either a reviewed symbolic scope or this exact span.
        supplied.add(subject_span.content_id)
        return frozenset(supplied)

    def assess(
        self,
        *,
        subject_span: SourceSpan,
        language_runtime: str,
        scope_ids: Iterable[str] = (),
        boundaries: Iterable[NativeBoundary] = (),
        receipts: Iterable[ProofEvidence] = (),
        evidence_refs: Iterable[EvidenceReference] = (),
        ownership_refs: Iterable[EvidenceReference] = (),
        mutation_region_refs: Iterable[EvidenceReference] = (),
        borrow_lifetime_refs: Iterable[EvidenceReference] = (),
        aliasing_refs: Iterable[EvidenceReference] = (),
        nullability_refs: Iterable[EvidenceReference] = (),
        bounds_refs: Iterable[EvidenceReference] = (),
        unsupported_refs: Iterable[str] = (),
        max_memory_bytes: int | None = None,
        resource_bounds: Mapping[str, Any] | None = None,
    ) -> MemorySafetyAssessment:
        """Assess one source scope without treating resource data as safety proof.

        ``max_memory_bytes`` and ``resource_bounds`` are accepted solely so
        callers can pass a complete ProgramContract without extracting a
        resource number into the safety facet.  They never affect the verdict.
        """
        if not isinstance(subject_span, SourceSpan):
            raise MemorySafetyEvidenceError("subject_span must be a SourceSpan")
        language = _text(language_runtime, "language_runtime").lower()
        if max_memory_bytes is not None and (isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int) or max_memory_bytes < 0):
            raise MemorySafetyEvidenceError("max_memory_bytes must be a non-negative integer when supplied")
        if resource_bounds is not None and not isinstance(resource_bounds, Mapping):
            raise MemorySafetyEvidenceError("resource_bounds must be a mapping when supplied")

        checked_scope_ids = self._scope_ids(subject_span, scope_ids)
        checked_boundaries = tuple(boundaries)
        if len(checked_boundaries) > MAX_EVIDENCE_ITEMS or not all(isinstance(item, NativeBoundary) for item in checked_boundaries):
            raise MemorySafetyEvidenceError("boundaries must contain bounded NativeBoundary values")
        direct_evidence = _refs(
            (*evidence_refs, *ownership_refs, *mutation_region_refs, *borrow_lifetime_refs,
             *aliasing_refs, *nullability_refs, *bounds_refs),
            "evidence_refs",
        )
        boundary_evidence = _refs(
            (reference for boundary in checked_boundaries for reference in boundary.evidence_refs),
            "boundary evidence_refs",
        )
        checked_receipts = tuple(receipts)
        if len(checked_receipts) > MAX_EVIDENCE_ITEMS or not all(isinstance(item, ProofEvidence) for item in checked_receipts):
            raise MemorySafetyEvidenceError("receipts must contain bounded ProofEvidence values")
        explicit_unsupported = tuple(sorted({_text(item, "unsupported_ref") for item in unsupported_refs}))

        accepted: list[ProofEvidence] = []
        rejected: list[ProofEvidence] = []
        stale: list[ProofEvidence] = []
        errors: list[ProofEvidence] = []
        for receipt in checked_receipts:
            if receipt.state is MemorySafetyReceiptState.ERROR:
                errors.append(receipt)
            elif receipt.state is MemorySafetyReceiptState.STALE:
                stale.append(receipt)
            elif receipt.tree_id != self.roots.tree_id or receipt.toolchain_id != self.roots.toolchain_id:
                stale.append(receipt)
            elif receipt.language_runtime != language or not checked_scope_ids.intersection(receipt.scope_ids):
                rejected.append(receipt)
            elif receipt.state is MemorySafetyReceiptState.FAILED:
                errors.append(receipt)
            else:
                accepted.append(receipt)

        reasons: set[str] = set()
        unsupported: set[str] = set(explicit_unsupported)
        for boundary in checked_boundaries:
            if boundary.kind in _UNMODELED_BOUNDARIES:
                unsupported.add(f"unmodeled_{boundary.kind.value}_boundary")
            if language in _MANAGED_LANGUAGES and boundary.kind in _MANAGED_NATIVE_BOUNDARIES:
                unsupported.add(f"managed_{language}_{boundary.kind.value}_boundary")
        if language not in _NATIVE_LANGUAGES | _MANAGED_LANGUAGES:
            unsupported.add("language_runtime_memory_model_unsupported")

        if errors:
            disposition = MemorySafetyDisposition.ERROR
            reasons.add("receipt_error_or_failure")
        elif stale:
            disposition = MemorySafetyDisposition.STALE
            reasons.add("stale_tree_or_toolchain_receipt")
        elif unsupported:
            disposition = MemorySafetyDisposition.UNSUPPORTED
            reasons.update(unsupported)
        else:
            accepted_kinds = {receipt.receipt_kind for receipt in accepted}
            observational = [receipt for receipt in accepted if receipt.receipt_kind in _OBSERVATION_ONLY_KINDS]
            required_missing = language in _NATIVE_LANGUAGES and self.policy.require_native_proof and any(
                not accepted_kinds.intersection(group) for group in self.policy.native_proof_groups
            )
            if required_missing:
                if observational:
                    disposition = MemorySafetyDisposition.EMPIRICAL
                    reasons.add("native_proof_requirements_missing")
                else:
                    disposition = MemorySafetyDisposition.UNSUPPORTED
                    reasons.add("native_proof_receipts_missing")
                    unsupported.add("native_proof_receipts_missing")
            elif language in _NATIVE_LANGUAGES and self.policy.require_native_proof:
                disposition = MemorySafetyDisposition.PROVED
                reasons.add("policy_selected_native_proofs_current")
            elif observational:
                disposition = MemorySafetyDisposition.EMPIRICAL
                reasons.add("observation_only_receipts")
            else:
                # This says the language model is supported, not that the
                # program is safe.  ``memory_safe`` remains false below.
                disposition = MemorySafetyDisposition.SUPPORTED
                reasons.add("modeled_scope_without_memory_safety_promotion")

        proof_refs = tuple(item.evidence_ref for item in accepted if item.receipt_kind not in _OBSERVATION_ONLY_KINDS)
        accepted_refs = tuple(item.evidence_ref for item in accepted)
        rejected_refs = tuple(item.evidence_ref for item in (*rejected, *stale, *errors))
        if disposition is MemorySafetyDisposition.PROVED and not proof_refs:
            # Defensive: a policy bug must fail closed rather than construct an
            # invalid compact facet.
            disposition = MemorySafetyDisposition.UNSUPPORTED
            unsupported.add("native_proof_receipts_missing")
            reasons.add("native_proof_receipts_missing")
        if disposition is MemorySafetyDisposition.EMPIRICAL:
            facet_evidence = _refs((*direct_evidence, *boundary_evidence, *accepted_refs), "facet evidence_refs")
            if not facet_evidence:
                disposition = MemorySafetyDisposition.UNSUPPORTED
                unsupported.add("empirical_evidence_missing")
                reasons.add("empirical_evidence_missing")
        else:
            facet_evidence = _refs((*direct_evidence, *boundary_evidence), "facet evidence_refs")

        facet = MemorySafetyFacet(
            roots=self.roots,
            subject_span=subject_span,
            language_runtime=language,
            disposition=disposition,
            evidence_refs=facet_evidence,
            proof_refs=proof_refs if disposition is not MemorySafetyDisposition.UNSUPPORTED else (),
            unsupported_refs=tuple(sorted(unsupported)) if disposition is MemorySafetyDisposition.UNSUPPORTED else (),
        )
        return MemorySafetyAssessment(
            facet=facet,
            memory_safe=disposition is MemorySafetyDisposition.PROVED,
            reason_codes=tuple(reasons),
            accepted_receipt_refs=accepted_refs,
            rejected_receipt_refs=rejected_refs,
        )

    def collect(self, **kwargs: Any) -> MemorySafetyFacet:
        """Compatibility convenience: return only the interchange facet."""
        return self.assess(**kwargs).facet


def collect_memory_safety_facet(
    roots: AuthorityRoots, *, policy: MemorySafetyPolicy | None = None, **kwargs: Any
) -> MemorySafetyFacet:
    """Stateless convenience wrapper around :class:`MemorySafetyEvidenceCollector`."""
    return MemorySafetyEvidenceCollector(roots, policy).collect(**kwargs)


__all__ = (
    "MEMORY_SAFETY_EVIDENCE_SCHEMA",
    "MEMORY_SAFETY_POLICY_SCHEMA",
    "MemorySafetyEvidenceError",
    "NativeBoundaryKind",
    "MemorySafetyReceiptKind",
    "MemorySafetyReceiptState",
    "NativeBoundary",
    "ProofEvidence",
    "MemorySafetyReceipt",
    "MemorySafetyPolicy",
    "MemorySafetyAssessment",
    "MemorySafetyEvidenceCollector",
    "MemorySafetyFacet",
    "MemorySafetyDisposition",
    "collect_memory_safety_facet",
)

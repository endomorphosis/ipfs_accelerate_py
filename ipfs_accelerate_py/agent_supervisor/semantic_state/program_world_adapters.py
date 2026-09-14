"""Accelerator semantic-world operational adapters (SAWM-015).

Interface: ``SemanticWorldOperationalAdapters@1``.

Narrow datasets/kit consumers with typed capability gates, reuse/context
receipts, execution-transition compilation, and operational root publication
*requests*.  Adapters never redefine semantic identities, never decide kit
storage policy, never mutate the current root, and never replace
ContextCompiler, CLI/MCP registries, or supervisor validation/merge/event
authority.

ANN/similarity remains advisory.  Unavailable surfaces return a visible
typed fallback instead of a simulated capability.

Importing this module performs no I/O, starts no threads, and does not import
datasets or kit implementations.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    UnavailableResult,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ADAPTER_ID,
    ADAPTER_OWNED_AUTHORITIES,
    DATASETS_IDENTITY_AUTHORITY,
    DATASETS_RELATION_AUTHORITY,
    DATASETS_TRANSITION_AUTHORITY,
    FORBIDDEN_IDENTITY_FIELDS,
    KIT_VERIFIED_STORE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE,
    CompilationStatus,
    ExecutionTransitionCompilation,
    OperationalWorldRootPublicationRequest,
    ProgramWorldAdmissionError,
    ProgramWorldCapabilityReceipt,
    ProgramWorldContextReceipt,
    ProgramWorldReceiptError,
    ProgramWorldReuseDecision,
    PublicationStatus,
    ReuseVerdict,
    available_capability,
    fallback_capability,
    unavailable_capability,
)


DATASETS_SURFACE: Final[str] = "ipfs_datasets_py.program_world"
KIT_SURFACE: Final[str] = "ipfs_kit_py.verified_semantic_store"
ANN_SURFACE: Final[str] = "ipfs_kit_py.projection_index"

DATASETS_OPERATIONS: Final[tuple[str, ...]] = (
    "cite_semantic_object",
    "cite_semantic_world_root",
    "cite_domain",
    "cite_relation",
    "cite_transition",
)
KIT_OPERATIONS: Final[tuple[str, ...]] = (
    "consume_verified_semantic_object",
    "consume_verified_projection",
    "consume_verified_block",
)
_DATASETS_CID_ATTRS: Final[tuple[str, ...]] = (
    "semantic_object_cid",
    "semantic_world_root_cid",
    "relation_claim_cid",
    "query_cid",
    "candidate_cid",
    "prediction_cid",
    "observation_cid",
    "admission_cid",
    "transition_cid",
    "adapter_cid",
    "domain_identity",
    "state_cid",
    "capsule_cid",
    "root_cid",
)
_KIT_CID_ATTRS: Final[tuple[str, ...]] = (
    "semantic_object_cid",
    "projection_cid",
    "bytes_cid",
    "identity_cid",
    "cid",
)

_IDENTITY_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "SemanticObjectEnvelope",
        "SemanticWorldRootIdentity",
        "CanonicalProgramGraphIdentity",
        "RawExecutionStateIdentity",
        "AbstractExecutionStateIdentity",
        "ProjectionIdentity",
        "TransitionIdentity",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ProgramTransitionCandidate",
        "ProgramTransitionPrediction",
        "ProgramTransitionObservation",
        "ProgramWorldDomainAdapter",
        "DomainCapabilityUnavailable",
    }
)


class ProgramWorldAdapterError(ProgramWorldReceiptError):
    """Closed adapter validation or authority-boundary failure."""


class ProgramWorldCapabilityUnavailable(ProgramWorldAdapterError):
    """Named datasets/kit capability is missing, incompatible, or failed closed."""

    def __init__(
        self,
        operation: str,
        reason_code: str,
        diagnostic: str,
        *,
        retryable: bool = False,
        surface: str = DATASETS_SURFACE,
        adapter_id: str = ADAPTER_ID,
    ) -> None:
        self.operation = operation
        self.reason_code = reason_code
        self.diagnostic = diagnostic[:512]
        self.retryable = bool(retryable)
        self.surface = surface
        self.adapter_id = adapter_id
        super().__init__(f"{adapter_id}:{surface}:{operation}:{reason_code}: {self.diagnostic}")

    def to_capability_receipt(self) -> ProgramWorldCapabilityReceipt:
        return unavailable_capability(
            self.surface,
            reason_code=self.reason_code,
            diagnostic=self.diagnostic,
            retryable=self.retryable,
        )

    def to_unavailable_result(self) -> UnavailableResult:
        return UnavailableResult(
            operation=self.operation,
            adapter_id=self.adapter_id,
            reason_code=self.reason_code,
            retryable=self.retryable,
            diagnostic=self.diagnostic,
        )


def _module_of(value: Any) -> str:
    return str(getattr(type(value), "__module__", "") or "")


def _qualname_of(value: Any) -> str:
    return str(getattr(type(value), "__qualname__", "") or type(value).__name__)


def _attr(source: Any, *names: str) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if callable(value) and name.endswith("_cid"):
                continue
            return value
    return None


def _reject_ann_identity(source: Any, *, context: str) -> None:
    if isinstance(source, Mapping):
        forbidden = set(source) & FORBIDDEN_IDENTITY_FIELDS
        if forbidden:
            raise ProgramWorldAdapterError(
                f"{context} rejects ANN/score identity fields {sorted(forbidden)}"
            )
        return
    for name in FORBIDDEN_IDENTITY_FIELDS:
        if hasattr(source, name) and getattr(source, name) not in (None, False, (), [], {}):
            raise ProgramWorldAdapterError(
                f"{context} rejects ANN/score identity field {name!r}"
            )


def _require_datasets_authority(source: Any, *, context: str) -> None:
    module = _module_of(source)
    if module.startswith("ipfs_datasets_py."):
        return
    name = _qualname_of(source)
    if name in _IDENTITY_TYPE_NAMES:
        raise ProgramWorldAdapterError(
            f"{context} rejects fake datasets authority {module}.{name}"
        )
    if not isinstance(source, Mapping):
        claimed_schema = _attr(source, "SCHEMA", "INTERFACE", "schema", "interface")
        if isinstance(claimed_schema, str) and "ipfs-datasets" in claimed_schema:
            raise ProgramWorldAdapterError(
                f"{context} rejects fake datasets authority {module}.{name}"
            )
        return
    claimed = source.get("interface") or source.get("schema") or ""
    if isinstance(claimed, str) and "ipfs-datasets" in claimed:
        # Mappings may cite a datasets schema only as a public identity view.
        # They cannot claim to *be* the datasets type.
        return


def _require_kit_store_authority(store: Any, *, context: str) -> None:
    module = _module_of(store)
    if module.startswith("ipfs_kit_py."):
        return
    interface = getattr(store, "INTERFACE", None)
    if interface == "VerifiedSemanticStore@1" or _qualname_of(store) in {
        "VerifiedSemanticBlockStore",
        "VerifiedSemanticStore",
        "SemanticWorldArtifactStore",
        "DurableCoordinationStore",
    }:
        raise ProgramWorldAdapterError(
            f"{context} rejects fake kit authority {module}.{_qualname_of(store)}"
        )
    raise ProgramWorldAdapterError(
        f"{context} requires a kit VerifiedSemanticStore@1 from {KIT_VERIFIED_STORE_AUTHORITY}"
    )


def _extract_cid(source: Any, names: Sequence[str], *, context: str) -> str:
    for name in names:
        value = _attr(source, name)
        if value is None:
            continue
        return validate_opaque_cid(value, name)
    if type(source) is str:
        return validate_opaque_cid(source, context)
    raise ProgramWorldAdapterError(f"{context} is missing a datasets/kit identity CID")


def _identity_preserved(source: Any, cited_cid: str, names: Sequence[str]) -> bool:
    live = None
    for name in names:
        value = _attr(source, name)
        if value is not None:
            live = value
            break
    if live is None:
        return True
    try:
        return validate_opaque_cid(live, "live_identity") == cited_cid
    except Exception:
        return False


def _payload_snapshot(source: Any) -> Mapping[str, Any] | None:
    serializer = getattr(source, "to_dict", None)
    if callable(serializer):
        payload = serializer()
        if isinstance(payload, Mapping):
            return dict(payload)
    identity = getattr(source, "identity_payload", None)
    if callable(identity):
        payload = identity()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(source, Mapping):
        return dict(source)
    return None


# ---------------------------------------------------------------------------
# Lazy surfaces
# ---------------------------------------------------------------------------


def _load_datasets_surface() -> SimpleNamespace:
    try:
        from ipfs_datasets_py.logic.software_contracts.semantic_state import (
            program_domain_adapters as domain_mod,
        )
        from ipfs_datasets_py.logic.software_contracts.semantic_state import (
            program_identity as identity_mod,
        )
        from ipfs_datasets_py.logic.software_contracts.semantic_state import (
            program_relations as relations_mod,
        )
        from ipfs_datasets_py.logic.software_contracts.semantic_state import (
            program_transition as transition_mod,
        )
    except Exception as exc:
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "import_failed",
            f"datasets program-world surface import failed: {exc}",
            retryable=True,
            surface=DATASETS_SURFACE,
        ) from exc
    required = {
        "identity": (
            "SemanticObjectEnvelope",
            "SemanticWorldRootIdentity",
            "SEMANTIC_OBJECT_ENVELOPE_INTERFACE",
        ),
        "relations": ("ProgramRelationClaim", "SCOPED_PROGRAM_RELATION_INTERFACE"),
        "transition": ("ProgramTransitionQuery", "PROGRAM_TRANSITION_QUERY_INTERFACE"),
        "domain": ("adapt_program_world_domain", "PROGRAM_WORLD_DOMAIN_ADAPTER_INTERFACE"),
    }
    modules = {
        "identity": identity_mod,
        "relations": relations_mod,
        "transition": transition_mod,
        "domain": domain_mod,
    }
    for label, names in required.items():
        missing = [name for name in names if not hasattr(modules[label], name)]
        if missing:
            raise ProgramWorldCapabilityUnavailable(
                "load",
                "missing_exports",
                f"datasets {label} missing required exports: {', '.join(missing)}",
                retryable=False,
                surface=DATASETS_SURFACE,
            )
    if identity_mod.SEMANTIC_OBJECT_ENVELOPE_INTERFACE != "SemanticObjectEnvelope@1":
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "schema_mismatch",
            "datasets SemanticObjectEnvelope interface is not @1",
            retryable=False,
            surface=DATASETS_SURFACE,
        )
    return SimpleNamespace(
        identity=identity_mod,
        relations=relations_mod,
        transition=transition_mod,
        domain=domain_mod,
        capability=available_capability(DATASETS_SURFACE, operations=DATASETS_OPERATIONS),
    )


def _load_kit_surface() -> SimpleNamespace:
    try:
        from ipfs_kit_py.semantic_world_store import verified_store as verified_mod
    except Exception as exc:
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "import_failed",
            f"kit verified semantic store import failed: {exc}",
            retryable=True,
            surface=KIT_SURFACE,
        ) from exc
    required = (
        "VerifiedSemanticBlockStore",
        "VERIFIED_SEMANTIC_STORE_INTERFACE",
        "ProjectionRecord",
    )
    missing = [name for name in required if not hasattr(verified_mod, name)]
    if missing:
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "missing_exports",
            f"kit verified store missing required exports: {', '.join(missing)}",
            retryable=False,
            surface=KIT_SURFACE,
        )
    if verified_mod.VERIFIED_SEMANTIC_STORE_INTERFACE != "VerifiedSemanticStore@1":
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "schema_mismatch",
            "kit VerifiedSemanticStore interface is not @1",
            retryable=False,
            surface=KIT_SURFACE,
        )
    return SimpleNamespace(
        verified=verified_mod,
        capability=available_capability(KIT_SURFACE, operations=KIT_OPERATIONS),
    )


def inspect_datasets_capability(
    *, loader: Callable[[], SimpleNamespace] | None = None
) -> ProgramWorldCapabilityReceipt:
    try:
        surface = (loader or _load_datasets_surface)()
    except ProgramWorldCapabilityUnavailable as exc:
        return exc.to_capability_receipt()
    cap = getattr(surface, "capability", None)
    if isinstance(cap, ProgramWorldCapabilityReceipt):
        return cap
    return available_capability(DATASETS_SURFACE, operations=DATASETS_OPERATIONS)


def inspect_kit_capability(
    *, loader: Callable[[], SimpleNamespace] | None = None
) -> ProgramWorldCapabilityReceipt:
    try:
        surface = (loader or _load_kit_surface)()
    except ProgramWorldCapabilityUnavailable as exc:
        return exc.to_capability_receipt()
    cap = getattr(surface, "capability", None)
    if isinstance(cap, ProgramWorldCapabilityReceipt):
        return cap
    return available_capability(KIT_SURFACE, operations=KIT_OPERATIONS)


def inspect_ann_capability() -> ProgramWorldCapabilityReceipt:
    """ANN/projection indexes are always advisory, even when the kit surface loads."""

    kit = inspect_kit_capability()
    if not kit.available:
        return fallback_capability(
            ANN_SURFACE,
            reason_code=kit.reason_code,
            diagnostic=(
                "ANN retrieval is advisory and currently unavailable; "
                f"{kit.diagnostic}"
            ),
            operations=(),
            retryable=kit.retryable,
        )
    return fallback_capability(
        ANN_SURFACE,
        reason_code="ann_advisory_only",
        diagnostic=(
            "projection/ANN retrieval is available as advisory ranking only; "
            "it cannot admit identity, reuse, context omission, or operational acceptance"
        ),
        operations=("consume_verified_projection",),
        retryable=False,
    )


# ---------------------------------------------------------------------------
# Datasets adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetsIdentityCitation:
    """Operational citation of one datasets identity. Never replaces the CID."""

    kind: str
    identity_cid: str
    source_authority: str
    schema: str | None = None
    adapter_may_replace_identity: bool = False
    retains_source_authority: bool = True

    def __post_init__(self) -> None:
        if self.adapter_may_replace_identity:
            raise ProgramWorldAdapterError("adapters may not replace datasets identities")
        if not self.retains_source_authority:
            raise ProgramWorldAdapterError("adapters must retain datasets identity authority")


class DatasetsProgramWorldAdapter:
    """Consume exact datasets identities/relations/adapters without redefining them."""

    INTERFACE: Final[str] = SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE

    def __init__(
        self,
        *,
        surface: Any | None = None,
        loader: Callable[[], SimpleNamespace] | None = None,
    ) -> None:
        self._loader = loader or _load_datasets_surface
        self._surface = surface
        self._error: ProgramWorldCapabilityUnavailable | None = None

    def _resolve(self) -> Any:
        if self._surface is not None:
            return self._surface
        if self._error is not None:
            raise self._error
        try:
            self._surface = self._loader()
        except ProgramWorldCapabilityUnavailable as exc:
            self._error = exc
            raise
        except Exception as exc:
            wrapped = ProgramWorldCapabilityUnavailable(
                "load",
                "load_failed",
                f"datasets surface failed to load: {exc}",
                retryable=True,
                surface=DATASETS_SURFACE,
            )
            self._error = wrapped
            raise wrapped from exc
        return self._surface

    @property
    def capability(self) -> ProgramWorldCapabilityReceipt:
        try:
            surface = self._resolve()
        except ProgramWorldCapabilityUnavailable as exc:
            return exc.to_capability_receipt()
        cap = getattr(surface, "capability", None)
        if isinstance(cap, ProgramWorldCapabilityReceipt):
            return cap
        return available_capability(DATASETS_SURFACE, operations=DATASETS_OPERATIONS)

    def _require(self, operation: str) -> Any:
        cap = self.capability
        if not cap.available:
            raise ProgramWorldCapabilityUnavailable(
                operation,
                cap.reason_code,
                cap.diagnostic,
                retryable=cap.retryable,
                surface=DATASETS_SURFACE,
            )
        return self._resolve()

    def cite_semantic_object(self, source: Any) -> DatasetsIdentityCitation:
        self._require("cite_semantic_object")
        _reject_ann_identity(source, context="semantic_object")
        _require_datasets_authority(source, context="semantic_object")
        before = _payload_snapshot(source)
        cid = _extract_cid(source, ("semantic_object_cid",), context="semantic_object")
        if before is not None and _payload_snapshot(source) != before:
            raise ProgramWorldAdapterError("citing a semantic object must not mutate it")
        if not _identity_preserved(source, cid, ("semantic_object_cid",)):
            raise ProgramWorldAdapterError("semantic-object identity was not preserved")
        schema = _attr(source, "SCHEMA") or _attr(source, "schema")
        return DatasetsIdentityCitation(
            kind="semantic_object",
            identity_cid=cid,
            source_authority=DATASETS_IDENTITY_AUTHORITY,
            schema=schema if type(schema) is str else None,
        )

    def cite_semantic_world_root(self, source: Any) -> DatasetsIdentityCitation:
        self._require("cite_semantic_world_root")
        _reject_ann_identity(source, context="semantic_world_root")
        _require_datasets_authority(source, context="semantic_world_root")
        cid = _extract_cid(
            source, ("semantic_world_root_cid",), context="semantic_world_root"
        )
        if not _identity_preserved(source, cid, ("semantic_world_root_cid",)):
            raise ProgramWorldAdapterError("semantic-world-root identity was not preserved")
        return DatasetsIdentityCitation(
            kind="semantic_world_root",
            identity_cid=cid,
            source_authority=DATASETS_IDENTITY_AUTHORITY,
            schema=_attr(source, "SCHEMA") if type(_attr(source, "SCHEMA")) is str else None,
        )

    def cite_relation(self, source: Any) -> DatasetsIdentityCitation:
        self._require("cite_relation")
        _reject_ann_identity(source, context="relation")
        _require_datasets_authority(source, context="relation")
        kind = str(_attr(source, "relation_kind") or "")
        if kind.lower() in {"similar", "similarity", "ann", "knn", "nearest"}:
            raise ProgramWorldAdapterError("neural similarity is never a semantic relation")
        cid = _extract_cid(source, ("relation_claim_cid",), context="relation")
        if not _identity_preserved(source, cid, ("relation_claim_cid",)):
            raise ProgramWorldAdapterError("relation identity was not preserved")
        return DatasetsIdentityCitation(
            kind="relation_claim",
            identity_cid=cid,
            source_authority=DATASETS_RELATION_AUTHORITY,
        )

    def cite_transition(self, source: Any) -> DatasetsIdentityCitation:
        self._require("cite_transition")
        _reject_ann_identity(source, context="transition")
        _require_datasets_authority(source, context="transition")
        cid = _extract_cid(
            source,
            (
                "query_cid",
                "candidate_cid",
                "prediction_cid",
                "observation_cid",
                "admission_cid",
                "transition_cid",
            ),
            context="transition",
        )
        kind = _qualname_of(source)
        if kind == "ProgramTransitionPrediction" or _attr(source, "proposal_only") is True:
            if _attr(source, "admitted") is True:
                raise ProgramWorldAdmissionError("datasets predictions cannot self-admit")
        return DatasetsIdentityCitation(
            kind="transition",
            identity_cid=cid,
            source_authority=DATASETS_TRANSITION_AUTHORITY,
        )

    def cite_domain(
        self, domain_kind: str, source: Any | None = None, **kwargs: Any
    ) -> Any:
        surface = self._require("cite_domain")
        domain = getattr(surface, "domain", None)
        adapt = getattr(domain, "adapt_program_world_domain", None)
        if not callable(adapt):
            raise ProgramWorldCapabilityUnavailable(
                "cite_domain",
                "missing_exports",
                "adapt_program_world_domain is unavailable",
                surface=DATASETS_SURFACE,
            )
        if source is not None:
            _reject_ann_identity(source, context="domain")
            if source is not None and not isinstance(source, Mapping):
                module = _module_of(source)
                if module and not module.startswith("ipfs_datasets_py.") and not module.startswith("ipfs_kit_py."):
                    raise ProgramWorldAdapterError(
                        f"domain adapter rejects fake datasets/kit authority {module}"
                    )
        before = _payload_snapshot(source) if source is not None else None
        before_identity = None
        if source is not None:
            try:
                before_identity = _extract_cid(source, _DATASETS_CID_ATTRS, context="domain")
            except ProgramWorldAdapterError:
                before_identity = None
        adapted = adapt(domain_kind, source, **kwargs)
        unavailable_type = getattr(domain, "DomainCapabilityUnavailable", None)
        if unavailable_type is not None and isinstance(adapted, unavailable_type):
            return adapted
        identity = _attr(adapted, "domain_identity")
        if identity is None:
            raise ProgramWorldAdapterError("domain adapter did not retain domain_identity")
        if _attr(adapted, "adapter_may_replace_domain_identity") is True:
            raise ProgramWorldAdapterError("domain adapter replaced domain identity")
        if before_identity is not None and identity != before_identity:
            raise ProgramWorldAdapterError("domain identity was not preserved")
        if before is not None and source is not None and _payload_snapshot(source) != before:
            raise ProgramWorldAdapterError("domain adaptation mutated the source payload")
        preserved = getattr(domain, "domain_identity_preserved", None)
        if callable(preserved) and source is not None:
            if not preserved(
                source,
                adapted,
                before_payload=before,
                before_identity=before_identity,
            ):
                raise ProgramWorldAdapterError("domain identity was not preserved")
        return adapted


# ---------------------------------------------------------------------------
# Kit adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KitVerifiedCitation:
    """Operational citation of a kit-reverified result. Projections stay advisory."""

    kind: str
    identity_cid: str
    source_authority: str = KIT_VERIFIED_STORE_AUTHORITY
    advisory: bool = False
    authoritative: bool = False

    def __post_init__(self) -> None:
        if self.kind == "projection" and not self.advisory:
            raise ProgramWorldAdapterError("projections remain advisory")
        if self.authoritative and self.advisory:
            raise ProgramWorldAdapterError("advisory kit results cannot be authoritative")
        if self.kind == "projection" and self.authoritative:
            raise ProgramWorldAdapterError("ANN/projection results cannot be authoritative")


class KitProgramWorldAdapter:
    """Consume kit verified blocks. ANN/projections remain advisory ranking."""

    INTERFACE: Final[str] = SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE

    def __init__(
        self,
        *,
        surface: Any | None = None,
        loader: Callable[[], SimpleNamespace] | None = None,
        store: Any | None = None,
    ) -> None:
        self._loader = loader or _load_kit_surface
        self._surface = surface
        self._store = store
        self._error: ProgramWorldCapabilityUnavailable | None = None

    def _resolve(self) -> Any:
        if self._surface is not None:
            return self._surface
        if self._error is not None:
            raise self._error
        try:
            self._surface = self._loader()
        except ProgramWorldCapabilityUnavailable as exc:
            self._error = exc
            raise
        except Exception as exc:
            wrapped = ProgramWorldCapabilityUnavailable(
                "load",
                "load_failed",
                f"kit verified store failed to load: {exc}",
                retryable=True,
                surface=KIT_SURFACE,
            )
            self._error = wrapped
            raise wrapped from exc
        return self._surface

    @property
    def capability(self) -> ProgramWorldCapabilityReceipt:
        try:
            surface = self._resolve()
        except ProgramWorldCapabilityUnavailable as exc:
            return exc.to_capability_receipt()
        cap = getattr(surface, "capability", None)
        if isinstance(cap, ProgramWorldCapabilityReceipt):
            return cap
        return available_capability(KIT_SURFACE, operations=KIT_OPERATIONS)

    def _require(self, operation: str) -> Any:
        cap = self.capability
        if not cap.available:
            raise ProgramWorldCapabilityUnavailable(
                operation,
                cap.reason_code,
                cap.diagnostic,
                retryable=cap.retryable,
                surface=KIT_SURFACE,
            )
        return self._resolve()

    def _bound_store(self, store: Any | None) -> Any:
        bound = store if store is not None else self._store
        if bound is None:
            raise ProgramWorldCapabilityUnavailable(
                "consume",
                "store_absent",
                "kit verified store instance is absent; adapters do not open ambient storage",
                retryable=False,
                surface=KIT_SURFACE,
            )
        _require_kit_store_authority(bound, context="kit_store")
        return bound

    def consume_verified_semantic_object(
        self, cid: str, *, store: Any | None = None
    ) -> KitVerifiedCitation:
        self._require("consume_verified_semantic_object")
        cid = validate_opaque_cid(cid, "cid")
        bound = self._bound_store(store)
        getter = getattr(bound, "get_verified_semantic_object", None)
        if not callable(getter):
            raise ProgramWorldAdapterError(
                "kit store missing get_verified_semantic_object"
            )
        envelope = getter(cid)
        _reject_ann_identity(envelope, context="verified_semantic_object")
        live = _extract_cid(
            envelope, ("semantic_object_cid",), context="verified_semantic_object"
        )
        if live != cid:
            raise ProgramWorldAdapterError(
                "kit verified semantic object CID does not match the requested CID"
            )
        return KitVerifiedCitation(
            kind="semantic_object",
            identity_cid=live,
            advisory=False,
            authoritative=False,
        )

    def consume_verified_projection(
        self, cid: str, *, store: Any | None = None
    ) -> KitVerifiedCitation:
        self._require("consume_verified_projection")
        cid = validate_opaque_cid(cid, "cid")
        bound = self._bound_store(store)
        getter = getattr(bound, "get_verified_projection_record", None)
        if not callable(getter):
            raise ProgramWorldAdapterError(
                "kit store missing get_verified_projection_record"
            )
        record = getter(cid)
        _reject_ann_identity(
            getattr(record, "to_dict", lambda: {})(),
            context="verified_projection",
        )
        live = _extract_cid(record, ("projection_cid",), context="verified_projection")
        if live != cid:
            raise ProgramWorldAdapterError(
                "kit verified projection CID does not match the requested CID"
            )
        if getattr(getattr(record, "identity", record), "authoritative", False):
            raise ProgramWorldAdapterError("authoritative projections are rejected")
        return KitVerifiedCitation(
            kind="projection",
            identity_cid=live,
            advisory=True,
            authoritative=False,
        )

    def consume_verified_block(
        self, cid: str, *, store: Any | None = None
    ) -> KitVerifiedCitation:
        self._require("consume_verified_block")
        cid = validate_opaque_cid(cid, "cid")
        bound = self._bound_store(store)
        getter = getattr(bound, "get_verified_block", None) or getattr(
            bound, "get_verified_artifact", None
        )
        if not callable(getter):
            raise ProgramWorldAdapterError("kit store missing get_verified_block")
        block = getter(cid)
        if isinstance(block, Mapping):
            _reject_ann_identity(block, context="verified_block")
            live = block.get("identity_cid") or block.get("cid") or cid
            live = validate_opaque_cid(live, "identity_cid")
        else:
            live = _extract_cid(block, _KIT_CID_ATTRS, context="verified_block")
        if live != cid and (not isinstance(block, Mapping) or block.get("cid") not in {cid, live}):
            # Kit may return storage CID plus identity CID; the requested CID must appear.
            if isinstance(block, Mapping) and cid not in {
                block.get("identity_cid"),
                block.get("cid"),
                block.get("storage_cid"),
            }:
                raise ProgramWorldAdapterError("verified block CID was not preserved")
        return KitVerifiedCitation(
            kind="block",
            identity_cid=cid,
            advisory=False,
            authoritative=False,
        )


# ---------------------------------------------------------------------------
# Execution-transition compiler and root publication requests
# ---------------------------------------------------------------------------


class ExecutionTransitionCompiler:
    """Compile datasets transition contracts into operational proposals."""

    INTERFACE: Final[str] = SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE

    def __init__(self, *, datasets: DatasetsProgramWorldAdapter | None = None) -> None:
        self._datasets = datasets if datasets is not None else DatasetsProgramWorldAdapter()

    def compile(
        self,
        query: Any,
        *,
        candidate: Any | None = None,
        prediction: Any | None = None,
        observation: Any | None = None,
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
    ) -> ExecutionTransitionCompilation:
        cap = self._datasets.capability
        if not cap.available:
            return ExecutionTransitionCompilation(
                status=CompilationStatus.UNAVAILABLE,
                query_cid=_require_cid_or_raise(query, "query_cid"),
                subject_cid=_fallback_subject_cid(query),
                environment_cid=_fallback_env_cid(query),
                policy_cid=_fallback_policy_cid(query),
                reason_code=cap.reason_code,
                fallback=True,
                limitations=("datasets_transition_surface_unavailable",),
            )
        cited_query = self._datasets.cite_transition(query)
        query_cid = cited_query.identity_cid
        subject_cid = validate_opaque_cid(
            _attr(query, "subject_cid") or _attr(query, "current_subject_cid"),
            "subject_cid",
        )
        environment_cid = validate_opaque_cid(
            _attr(query, "environment_binding_cid")
            or _attr(query, "current_environment_binding_cid"),
            "environment_cid",
        )
        policy_cid = validate_opaque_cid(_attr(query, "policy_cid"), "policy_cid")
        candidate_cid = None
        if candidate is not None:
            candidate_cid = self._datasets.cite_transition(candidate).identity_cid
        prediction_cid = None
        if prediction is not None:
            prediction_cid = self._datasets.cite_transition(prediction).identity_cid
            if _attr(prediction, "admitted") is True or admission_authority in ADAPTER_OWNED_AUTHORITIES:
                raise ProgramWorldAdmissionError(
                    "predictions cannot self-admit operational transitions"
                )
        observation_cid = None
        if observation is not None:
            observation_cid = self._datasets.cite_transition(observation).identity_cid
        admitted = False
        proposal_only = True
        if admission_authority is not None:
            # Existing supervisor evidence may convert a proposal into admitted
            # operational evidence.  Adapters never mint that evidence.
            proposal_only = False
            admitted = True
        return ExecutionTransitionCompilation(
            status=CompilationStatus.COMPILED,
            query_cid=query_cid,
            subject_cid=subject_cid,
            environment_cid=environment_cid,
            policy_cid=policy_cid,
            candidate_cid=candidate_cid,
            prediction_cid=prediction_cid,
            observation_cid=observation_cid,
            datasets_transition_cid=query_cid,
            reason_code="compiled_proposal",
            proposal_only=proposal_only,
            admitted=admitted,
            admission_authority=admission_authority,
            admission_evidence_cid=admission_evidence_cid,
            limitations=(
                "compiler_does_not_execute",
                "compiler_does_not_admit",
                "operational_acceptance_uses_supervisor_validation_merge_event",
            ),
        )


def _require_cid_or_raise(source: Any, name: str) -> str:
    value = _attr(source, name)
    if value is None:
        raise ProgramWorldAdapterError(f"transition query missing {name}")
    return validate_opaque_cid(value, name)


def _fallback_subject_cid(query: Any) -> str:
    value = _attr(query, "subject_cid") or _attr(query, "current_subject_cid")
    if value is None:
        raise ProgramWorldAdapterError("transition query missing subject_cid")
    return validate_opaque_cid(value, "subject_cid")


def _fallback_env_cid(query: Any) -> str:
    value = _attr(query, "environment_binding_cid")
    if value is None:
        raise ProgramWorldAdapterError("unavailable compilation still requires environment_cid")
    return validate_opaque_cid(value, "environment_cid")


def _fallback_policy_cid(query: Any) -> str:
    return _require_cid_or_raise(query, "policy_cid")


class OperationalWorldRootPublisher:
    """Request generation-bearing publication. Never CAS the current root."""

    INTERFACE: Final[str] = SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE

    def __init__(
        self,
        *,
        datasets: DatasetsProgramWorldAdapter | None = None,
        kit: KitProgramWorldAdapter | None = None,
    ) -> None:
        self._datasets = datasets if datasets is not None else DatasetsProgramWorldAdapter()
        self._kit = kit if kit is not None else KitProgramWorldAdapter()

    def request_publication(
        self,
        semantic_world_root: Any,
        *,
        expected_generation: int,
        expected_root_cid: str | None = None,
        candidate_operational_manifest_cid: str | None = None,
        durable_port: Any | None = None,
    ) -> OperationalWorldRootPublicationRequest:
        if durable_port is not None and callable(
            getattr(durable_port, "compare_and_swap_root", None)
        ):
            # Presence of CAS is allowed for later merge authority, but this
            # publisher must not invoke it.
            cas = durable_port.compare_and_swap_root
            if getattr(cas, "_program_world_publisher_invoked", False):
                raise ProgramWorldAdapterError("publisher must not CAS the current root")
        cap = self._datasets.capability
        kit_cap = self._kit.capability
        if not cap.available:
            root_cid = _extract_cid(
                semantic_world_root,
                ("semantic_world_root_cid",),
                context="semantic_world_root",
            )
            return OperationalWorldRootPublicationRequest(
                status=PublicationStatus.UNAVAILABLE,
                semantic_world_root_cid=root_cid,
                expected_generation=expected_generation,
                expected_root_cid=expected_root_cid,
                candidate_operational_manifest_cid=candidate_operational_manifest_cid,
                kit_verified=False,
                reason_code=cap.reason_code,
                fallback=True,
                limitations=(
                    "cannot_change_current_root",
                    "datasets_identity_surface_unavailable",
                ),
            )
        cited = self._datasets.cite_semantic_world_root(semantic_world_root)
        kit_verified = kit_cap.available
        limitations = [
            "cannot_change_current_root",
            "publisher_does_not_cas",
            "existing_merge_event_authority_publishes",
        ]
        status = PublicationStatus.REQUESTED
        reason = "publication_requested"
        fallback = False
        if not kit_verified:
            limitations.append("kit_verified_store_unavailable")
            fallback = True
            reason = kit_cap.reason_code
            status = PublicationStatus.REQUESTED
        return OperationalWorldRootPublicationRequest(
            status=status,
            semantic_world_root_cid=cited.identity_cid,
            expected_generation=expected_generation,
            expected_root_cid=expected_root_cid,
            candidate_operational_manifest_cid=candidate_operational_manifest_cid,
            kit_verified=kit_verified,
            reason_code=reason,
            fallback=fallback,
            limitations=limitations,
        )


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


class SemanticWorldOperationalAdapters:
    """Unified capability gate, datasets/kit consumers, compiler, and publisher."""

    INTERFACE: Final[str] = SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE

    def __init__(
        self,
        *,
        datasets: DatasetsProgramWorldAdapter | None = None,
        kit: KitProgramWorldAdapter | None = None,
        compiler: ExecutionTransitionCompiler | None = None,
        publisher: OperationalWorldRootPublisher | None = None,
    ) -> None:
        self.datasets = datasets if datasets is not None else DatasetsProgramWorldAdapter()
        self.kit = kit if kit is not None else KitProgramWorldAdapter()
        self.compiler = (
            compiler
            if compiler is not None
            else ExecutionTransitionCompiler(datasets=self.datasets)
        )
        self.publisher = (
            publisher
            if publisher is not None
            else OperationalWorldRootPublisher(datasets=self.datasets, kit=self.kit)
        )

    def probe_capabilities(self) -> dict[str, ProgramWorldCapabilityReceipt]:
        return {
            "datasets": self.datasets.capability,
            "kit": self.kit.capability,
            "ann": inspect_ann_capability(),
        }

    def evaluate_reuse(
        self,
        *,
        state_cid: str,
        goal_cid: str,
        policy_cid: str,
        environment_cid: str,
        toolchain_cid: str,
        prior_state_cid: str | None = None,
        relation_claim: Any | None = None,
        procedure_revision_cid: str | None = None,
        ann_candidates: Sequence[Any] = (),
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
    ) -> ProgramWorldReuseDecision:
        cap = self.datasets.capability
        state_cid = validate_opaque_cid(state_cid, "state_cid")
        goal_cid = validate_opaque_cid(goal_cid, "goal_cid")
        policy_cid = validate_opaque_cid(policy_cid, "policy_cid")
        environment_cid = validate_opaque_cid(environment_cid, "environment_cid")
        toolchain_cid = validate_opaque_cid(toolchain_cid, "toolchain_cid")
        if not cap.available:
            return ProgramWorldReuseDecision(
                verdict=ReuseVerdict.UNAVAILABLE,
                state_cid=state_cid,
                goal_cid=goal_cid,
                policy_cid=policy_cid,
                environment_cid=environment_cid,
                toolchain_cid=toolchain_cid,
                reason_code=cap.reason_code,
                fallback=True,
                limitations=("datasets_identity_surface_unavailable",),
            )
        relation_cid = None
        if relation_claim is not None:
            relation_cid = self.datasets.cite_relation(relation_claim).identity_cid
        if ann_candidates:
            return ProgramWorldReuseDecision(
                verdict=ReuseVerdict.REJECT,
                state_cid=state_cid,
                goal_cid=goal_cid,
                policy_cid=policy_cid,
                environment_cid=environment_cid,
                toolchain_cid=toolchain_cid,
                relation_claim_cid=relation_cid,
                procedure_revision_cid=procedure_revision_cid,
                exact_match=False,
                reason_code="ann_not_authoritative",
                limitations=("similarity_is_not_reuse", "ann_advisory_only"),
            )
        exact = prior_state_cid is not None and prior_state_cid == state_cid
        if exact:
            admitted = admission_authority is not None
            return ProgramWorldReuseDecision(
                verdict=ReuseVerdict.REUSE,
                state_cid=state_cid,
                goal_cid=goal_cid,
                policy_cid=policy_cid,
                environment_cid=environment_cid,
                toolchain_cid=toolchain_cid,
                relation_claim_cid=relation_cid,
                procedure_revision_cid=procedure_revision_cid,
                exact_match=True,
                reason_code="exact_identity_match",
                proposal_only=not admitted,
                admitted=admitted,
                admission_authority=admission_authority,
                admission_evidence_cid=admission_evidence_cid,
                limitations=("reuse_is_proposal_until_supervisor_admission",),
            )
        return ProgramWorldReuseDecision(
            verdict=ReuseVerdict.ABSTAIN,
            state_cid=state_cid,
            goal_cid=goal_cid,
            policy_cid=policy_cid,
            environment_cid=environment_cid,
            toolchain_cid=toolchain_cid,
            relation_claim_cid=relation_cid,
            procedure_revision_cid=procedure_revision_cid,
            exact_match=False,
            reason_code="no_exact_identity_match",
            limitations=("exact_reuse_required",),
        )

    def record_context(
        self,
        *,
        included: Sequence[Any],
        omitted: Sequence[Any] = (),
        raw_fallbacks: Sequence[Any] = (),
        unresolved_questions: Sequence[str] = (),
        token_budget: int | None = None,
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
    ) -> ProgramWorldContextReceipt:
        admitted = admission_authority is not None
        return ProgramWorldContextReceipt(
            included=included,
            omitted=omitted,
            raw_fallbacks=raw_fallbacks,
            unresolved_questions=unresolved_questions,
            token_budget=token_budget,
            proposal_only=not admitted,
            admitted=admitted,
            admission_authority=admission_authority,
            admission_evidence_cid=admission_evidence_cid,
            reason_code="context_recorded",
        )

    def compile_execution_transition(
        self, query: Any, **kwargs: Any
    ) -> ExecutionTransitionCompilation:
        return self.compiler.compile(query, **kwargs)

    def request_root_publication(
        self, semantic_world_root: Any, **kwargs: Any
    ) -> OperationalWorldRootPublicationRequest:
        return self.publisher.request_publication(semantic_world_root, **kwargs)


def load_semantic_world_operational_adapters(
    *,
    datasets: DatasetsProgramWorldAdapter | None = None,
    kit: KitProgramWorldAdapter | None = None,
) -> SemanticWorldOperationalAdapters:
    return SemanticWorldOperationalAdapters(datasets=datasets, kit=kit)


def program_world_adapter_module_source() -> str:
    """Return this module's source for AST audits. Performs no I/O besides read."""

    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


def assert_no_duplicate_semantic_world_authority(source: str | None = None) -> None:
    """Fail if this module redefines datasets/kit/ContextCompiler authorities."""

    tree = ast.parse(source or program_world_adapter_module_source())
    banned = {
        "SemanticObjectEnvelope",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ContextCompiler",
        "DurableCoordinationStore",
        "VerifiedSemanticBlockStore",
        "SemanticWorldArtifactStore",
        "ProgramWorldReuseGate",
    }
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = defined & banned
    if overlap:
        raise ProgramWorldAdapterError(
            f"adapters must not duplicate semantic-state/procedure-world types {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_ID",
    "ANN_SURFACE",
    "DATASETS_OPERATIONS",
    "DATASETS_SURFACE",
    "KIT_OPERATIONS",
    "KIT_SURFACE",
    "OPERATIONAL_ACCEPTANCE_AUTHORITIES",
    "SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE",
    "DatasetsIdentityCitation",
    "DatasetsProgramWorldAdapter",
    "ExecutionTransitionCompiler",
    "KitProgramWorldAdapter",
    "KitVerifiedCitation",
    "OperationalWorldRootPublisher",
    "ProgramWorldAdapterError",
    "ProgramWorldCapabilityUnavailable",
    "SemanticWorldOperationalAdapters",
    "assert_no_duplicate_semantic_world_authority",
    "inspect_ann_capability",
    "inspect_datasets_capability",
    "inspect_kit_capability",
    "load_semantic_world_operational_adapters",
]

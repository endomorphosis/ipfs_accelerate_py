"""Locator-first collection seeds for proof-backed test reuse (PTR-143).

Splits stable collection identity from post-pass execution identity:

* Collection attaches a canonical :class:`ProofReuseCollectionSeed` and a
  stable :class:`TestLocatorKey` before any runtime trace exists.
* Collection performs no fixture setup, no test call, and attaches no final
  execution key, eligibility decision, cache hit, or skip authority.
* Parameterized nodes bind the exact canonical parameter-value CID on the
  locator.
* Explicit injected identity (locator / execution key / lookup request)
  remains an override.
* Incomplete or exceptional static facts attach no lookup authority and leave
  the item free to execute normally.
* Off mode retains cold-import behaviour (no optional providers).

This module has a standard-library-only import surface at load time.  Heavy
identity collectors are reached only through the existing default identity
factory / assembly services.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional

from .config import ProofReuseMode
from .item_identity import (
    ITEM_COLLECTION_SEED_ATTRIBUTE as ITEM_COLLECTION_SEED_ATTRIBUTE,
    ITEM_EXECUTION_KEY_ATTRIBUTE,
    ITEM_LOCATOR_ATTRIBUTE,
    ItemIdentityAssemblyServices,
)


PROOF_REUSE_COLLECTION_SEED_INTERFACE: Final = "ProofReuseCollectionSeed@1"
PROOF_REUSE_COLLECTION_SEED_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/collection-seed@1"
)
LOCATOR_FIRST_ASSEMBLER_INTERFACE: Final = (
    "LocatorFirstItemIdentityAssembler@1"
)

_LOOKUP_REQUEST_ATTRIBUTE: Final = "_ipfs_proof_reuse_lookup_request"
_TRUE_SEED_MODES: Final = frozenset(
    {
        ProofReuseMode.READ,
        ProofReuseMode.WRITE,
        ProofReuseMode.READWRITE,
        ProofReuseMode.SHADOW,
    }
)


class CollectionSeedReason(str, Enum):
    """Closed reason codes for locator-first collection seeding."""

    ADMITTED = "admitted"
    MODE_OFF = "mode_off"
    EXISTING_IDENTITY_OVERRIDE = "existing_identity_override"
    STATIC_IDENTITY_INCOMPLETE = "static_identity_incomplete"
    LOCATOR_UNAVAILABLE = "locator_unavailable"
    PARAMETER_CID_REQUIRED = "parameter_cid_required"
    ATTACHMENT_FAILED = "attachment_failed"
    FACTORY_UNAVAILABLE = "factory_unavailable"
    INTERNAL_ERROR_FAIL_OPEN = "internal_error_fail_open"


@dataclass(frozen=True, slots=True)
class ProofReuseCollectionSeed:
    """Canonical static collection seed for one collected pytest item.

    The seed is retrieval-narrowing and diagnostic only.  It never authorizes
    ``SKIP`` and never substitutes for a final execution key or lookup request.
    """

    __test__: ClassVar[bool] = False

    reason: CollectionSeedReason
    stage: str
    node_id: str = ""
    forest_id: str = ""
    locator: Any = None
    locator_artifact: Any = None
    locator_cid: str = ""
    parameter_id: str = ""
    parameter_values_cid: str = ""
    parameterized: bool = False
    static_trace_root_cid: str = ""
    component_root_cid: str = ""
    seed_cid: str = ""
    static_identity: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return PROOF_REUSE_COLLECTION_SEED_INTERFACE

    @property
    def admitted(self) -> bool:
        return (
            self.reason is CollectionSeedReason.ADMITTED
            and self.locator is not None
            and bool(self.locator_cid)
            and bool(self.seed_cid)
        )

    @property
    def has_stable_locator(self) -> bool:
        return self.admitted

    @property
    def reusable(self) -> bool:
        # "Reusable" here means the seed is complete for locator discovery.
        # It is not certificate authority and does not authorize skip.
        return self.admitted

    @property
    def action(self) -> str:
        return "RUN"

    @property
    def authorizes_skip(self) -> bool:
        return False

    @property
    def authorizes_lookup(self) -> bool:
        # Collection seeds deliberately attach no final execution key / request.
        return False

    @property
    def has_execution_key(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_COLLECTION_SEED_SCHEMA,
            "interface": self.interface,
            "reason": self.reason.value,
            "stage": self.stage,
            "admitted": self.admitted,
            "reusable": self.reusable,
            "action": self.action,
            "authorizes_skip": False,
            "authorizes_lookup": False,
            "has_execution_key": False,
            "node_id": self.node_id,
            "forest_id": self.forest_id,
            "locator_cid": self.locator_cid,
            "parameter_id": self.parameter_id,
            "parameter_values_cid": self.parameter_values_cid,
            "parameterized": self.parameterized,
            "static_trace_root_cid": self.static_trace_root_cid,
            "component_root_cid": self.component_root_cid,
            "seed_cid": self.seed_cid,
            "diagnostics": dict(self.diagnostics),
        }


def _bounded_diagnostics(**diagnostics: Any) -> Mapping[str, Any]:
    bounded: dict[str, Any] = {}
    for key, value in list(diagnostics.items())[:16]:
        name = str(key)[:64]
        if value is None or isinstance(value, (bool, int)):
            bounded[name] = value
        elif isinstance(value, str):
            bounded[name] = value[:128]
        else:
            bounded[name] = type(value).__name__[:64]
    return MappingProxyType(bounded)


def _failure(
    reason: CollectionSeedReason,
    stage: str,
    **diagnostics: Any,
) -> ProofReuseCollectionSeed:
    return ProofReuseCollectionSeed(
        reason=reason,
        stage=str(stage)[:64],
        diagnostics=_bounded_diagnostics(**diagnostics),
    )


def _parse_mode(value: Any) -> ProofReuseMode:
    if isinstance(value, ProofReuseMode):
        return value
    try:
        return ProofReuseMode.parse(value)
    except Exception:
        return ProofReuseMode.OFF


def _has_explicit_injected_identity(item: Any) -> bool:
    """True when a caller already attached authoritative identity overrides."""

    if getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE, None) is not None:
        return True
    if getattr(item, _LOOKUP_REQUEST_ATTRIBUTE, None) is not None:
        return True
    locator = getattr(item, ITEM_LOCATOR_ATTRIBUTE, None)
    seed = getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, None)
    # Locator without our collection seed is a manual/injected identity.
    if locator is not None and seed is None:
        return True
    return False


def _mint_seed_cid(payload: Mapping[str, Any]) -> str:
    from ...agent_supervisor.analysis.test_execution_identity import (
        mint_content_identity,
    )

    return mint_content_identity(dict(payload)).cid


def build_collection_seed_from_static_identity(
    static_identity: Any,
    *,
    stage: str = "static_identity",
) -> ProofReuseCollectionSeed:
    """Project a default static identity into a canonical collection seed.

    Does not call fixtures, tests, or runtime tracers.  Incomplete static
    identities become non-admitted seeds with no locator attachment authority.
    """

    if static_identity is None:
        return _failure(
            CollectionSeedReason.STATIC_IDENTITY_INCOMPLETE,
            stage,
            detail="static_identity_missing",
        )

    reason = getattr(static_identity, "reason", None)
    admitted_static = bool(getattr(static_identity, "reusable", False))
    locator_artifact = getattr(static_identity, "locator_artifact", None)
    locator = getattr(locator_artifact, "locator", None) if locator_artifact else None
    locator_cid = str(
        getattr(locator_artifact, "locator_cid", None)
        or getattr(locator, "locator_id", None)
        or getattr(locator, "content_id", None)
        or ""
    )
    components = getattr(static_identity, "components", None)
    static_trace = getattr(static_identity, "static_trace", None)
    forest_id = str(getattr(static_identity, "forest_id", "") or "")

    parameter_id = ""
    parameter_values_cid = ""
    parameterized = False
    if locator is not None:
        parameter_id = str(getattr(locator, "parameter_id", "") or "")
        parameter_values_cid = str(
            getattr(locator, "parameter_values_cid", "") or ""
        )
        parameterized = bool(parameter_id)
    if components is not None and parameterized and not parameter_values_cid:
        parameter_values_cid = str(
            getattr(components, "parameter_cid", "") or ""
        )

    if parameterized and not parameter_values_cid:
        return _failure(
            CollectionSeedReason.PARAMETER_CID_REQUIRED,
            "parameter",
            parameter_id=parameter_id[:128],
            static_reason=str(getattr(reason, "value", reason) or "")[:64],
        )

    if (
        not admitted_static
        or locator is None
        or not locator_cid
        or not getattr(locator_artifact, "reusable", False)
    ):
        return ProofReuseCollectionSeed(
            reason=CollectionSeedReason.STATIC_IDENTITY_INCOMPLETE,
            stage=str(
                getattr(static_identity, "stage", stage) or stage
            )[:64],
            forest_id=forest_id,
            locator=locator,
            locator_artifact=locator_artifact,
            locator_cid=locator_cid,
            parameter_id=parameter_id,
            parameter_values_cid=parameter_values_cid,
            parameterized=parameterized,
            static_trace_root_cid=str(
                getattr(static_trace, "trace_cid", "") or ""
            ),
            component_root_cid=str(
                getattr(components, "component_root_cid", "") or ""
            ),
            static_identity=static_identity,
            diagnostics=_bounded_diagnostics(
                static_reason=str(getattr(reason, "value", reason) or ""),
                has_locator=locator is not None,
            ),
        )

    node_id = str(getattr(locator, "node_id", "") or "")
    static_trace_root_cid = str(getattr(static_trace, "trace_cid", "") or "")
    component_root_cid = str(
        getattr(components, "component_root_cid", "") or ""
    )
    seed_payload = {
        "schema": PROOF_REUSE_COLLECTION_SEED_SCHEMA,
        "interface": PROOF_REUSE_COLLECTION_SEED_INTERFACE,
        "node_id": node_id,
        "forest_id": forest_id,
        "locator_cid": locator_cid,
        "parameter_id": parameter_id,
        "parameter_values_cid": parameter_values_cid,
        "static_trace_root_cid": static_trace_root_cid,
        "component_root_cid": component_root_cid,
        "collection_schema_version": str(
            getattr(locator, "collection_schema_version", "1") or "1"
        ),
    }
    try:
        seed_cid = _mint_seed_cid(seed_payload)
    except BaseException as exc:
        return _failure(
            CollectionSeedReason.INTERNAL_ERROR_FAIL_OPEN,
            "seed_cid",
            exception_type=type(exc).__name__,
        )

    return ProofReuseCollectionSeed(
        reason=CollectionSeedReason.ADMITTED,
        stage="complete",
        node_id=node_id,
        forest_id=forest_id,
        locator=locator,
        locator_artifact=locator_artifact,
        locator_cid=locator_cid,
        parameter_id=parameter_id,
        parameter_values_cid=parameter_values_cid,
        parameterized=parameterized,
        static_trace_root_cid=static_trace_root_cid,
        component_root_cid=component_root_cid,
        seed_cid=seed_cid,
        static_identity=static_identity,
        diagnostics=MappingProxyType({}),
    )


def attach_collection_seed(
    item: Any,
    seed: ProofReuseCollectionSeed,
) -> bool:
    """Attach an admitted seed and its stable locator; never an execution key.

    Returns ``True`` when the seed was attached.  Explicit injected identity is
    left untouched.  Incomplete seeds store only a diagnostic result attribute
    and attach no lookup authority.
    """

    if _has_explicit_injected_identity(item):
        return False

    try:
        setattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, seed)
    except BaseException:
        return False

    if not seed.admitted or seed.locator is None:
        # Diagnostic only — no locator authority, no lookup request.
        return True

    written: list[str] = []
    try:
        # Intermediate locator for candidate discovery.  No execution key.
        setattr(item, ITEM_LOCATOR_ATTRIBUTE, seed.locator)
        written.append(ITEM_LOCATOR_ATTRIBUTE)
        # Guard against accidental final-key attachment at collection.
        if getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE, None) is not None:
            delattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
        if getattr(item, _LOOKUP_REQUEST_ATTRIBUTE, None) is not None:
            delattr(item, _LOOKUP_REQUEST_ATTRIBUTE)
        return True
    except BaseException:
        for name in written:
            try:
                delattr(item, name)
            except BaseException:
                pass
        try:
            delattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE)
        except BaseException:
            pass
        return False


class LocatorFirstItemIdentityAssembler:
    """Assemble and attach a locator-first collection seed for one item.

    Uses :meth:`DefaultIdentityServiceFactory.obtain_static_identity` when a
    factory is available; otherwise falls back to an empty/non-admitted seed
    without calling fixtures or tests.
    """

    __test__ = False

    def __init__(
        self,
        *,
        factory: Any = None,
        services: Optional[ItemIdentityAssemblyServices] = None,
        mode: Any = None,
    ) -> None:
        self.factory = factory
        self.services = services
        if mode is not None:
            self.mode = _parse_mode(mode)
        elif factory is not None and hasattr(factory, "mode"):
            self.mode = _parse_mode(getattr(factory, "mode"))
        else:
            self.mode = ProofReuseMode.OFF

    @property
    def interface(self) -> str:
        return LOCATOR_FIRST_ASSEMBLER_INTERFACE

    def assemble(self, item: Any) -> ProofReuseCollectionSeed:
        """Return a collection seed; every uncertainty fails open to ``RUN``."""

        try:
            return self._assemble(item)
        except BaseException as exc:
            return _failure(
                CollectionSeedReason.INTERNAL_ERROR_FAIL_OPEN,
                "assembler",
                exception_type=type(exc).__name__,
            )

    def assemble_and_attach(self, item: Any) -> ProofReuseCollectionSeed:
        """Assemble the seed and attach locator/seed without execution key."""

        if _has_explicit_injected_identity(item):
            seed = _failure(
                CollectionSeedReason.EXISTING_IDENTITY_OVERRIDE,
                "item",
            )
            try:
                # Do not overwrite injected identity; record diagnostic only if
                # no seed is already present.
                if getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, None) is None:
                    setattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, seed)
            except BaseException:
                pass
            return seed

        seed = self.assemble(item)
        if not attach_collection_seed(item, seed):
            if seed.admitted:
                seed = _failure(
                    CollectionSeedReason.ATTACHMENT_FAILED,
                    "attachment",
                )
                try:
                    setattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, seed)
                except BaseException:
                    pass
        return seed

    def _assemble(self, item: Any) -> ProofReuseCollectionSeed:
        if self.mode not in _TRUE_SEED_MODES:
            return _failure(CollectionSeedReason.MODE_OFF, "mode")

        factory = self.factory
        if factory is None:
            return _failure(
                CollectionSeedReason.FACTORY_UNAVAILABLE,
                "factory",
            )

        obtain = getattr(factory, "obtain_static_identity", None)
        if not callable(obtain):
            return _failure(
                CollectionSeedReason.FACTORY_UNAVAILABLE,
                "factory",
                detail="obtain_static_identity_missing",
            )

        # obtain_static_identity is pure static: forest, AST, components,
        # locator.  It never invokes fixtures, test bodies, or runtime tracers.
        static_identity = obtain(item)
        return build_collection_seed_from_static_identity(
            static_identity,
            stage="obtain_static_identity",
        )


def assemble_and_attach_collection_seed(
    item: Any,
    *,
    factory: Any = None,
    services: Optional[ItemIdentityAssemblyServices] = None,
    mode: Any = None,
) -> ProofReuseCollectionSeed:
    """Plugin entry point: attach a locator-first collection seed for one item."""

    assembler = LocatorFirstItemIdentityAssembler(
        factory=factory,
        services=services,
        mode=mode,
    )
    return assembler.assemble_and_attach(item)


__all__ = (
    "ITEM_COLLECTION_SEED_ATTRIBUTE",
    "LOCATOR_FIRST_ASSEMBLER_INTERFACE",
    "PROOF_REUSE_COLLECTION_SEED_INTERFACE",
    "PROOF_REUSE_COLLECTION_SEED_SCHEMA",
    "CollectionSeedReason",
    "LocatorFirstItemIdentityAssembler",
    "ProofReuseCollectionSeed",
    "assemble_and_attach_collection_seed",
    "attach_collection_seed",
    "build_collection_seed_from_static_identity",
)

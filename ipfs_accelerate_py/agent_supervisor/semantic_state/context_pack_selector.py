"""Accelerate ContextPack freshness admission and minimal-pack selection.

Accelerate verifies Datasets semantic identity and Kit bytes/root, enforces
exact tree/objective/policy/interface/toolchain/environment freshness, selects
the current minimal adequate pack, and records reuse or invalidation. It does
not remint Datasets CIDs, bypass Kit verification, or treat stored bytes as
semantic proof.

Importing this module performs no I/O and starts no threads or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    DATASETS_CONTEXT_PACK_AUTHORITY,
    EXACT_FRESHNESS_FIELDS,
    KIT_CONTEXT_PACK_STORE_AUTHORITY,
    ContextPackError,
    CurrentPackIdentity,
    StaleIdentityError,
    decode_context_pack_envelope,
    evaluate_exact_freshness,
    kit_context_pack_store_of,
    load_datasets_context_pack_authority,
    load_kit_context_pack_store,
    load_verified_kit_bytes,
    pack_is_adequate,
    pack_minimality_key,
    verify_kit_bytes,
    verify_kit_current_root,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    _text,
    _unique_sorted_texts,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload

CONTEXT_PACK_SELECTOR_INTERFACE = "ContextPackSelector@1"
CONTEXT_PACK_SELECTOR_SCHEMA = "ipfs-accelerate.context-pack-selection@1"
CONTEXT_PACK_ADMISSION_SCHEMA = "ipfs-accelerate.context-pack-admission-record@1"

DISPOSITION_REUSE = "reuse"
DISPOSITION_SELECTED = "selected"
DISPOSITION_INVALIDATED = "invalidated"
DISPOSITION_REJECTED = "rejected"
DISPOSITIONS = frozenset(
    {
        DISPOSITION_REUSE,
        DISPOSITION_SELECTED,
        DISPOSITION_INVALIDATED,
        DISPOSITION_REJECTED,
    }
)


class ContextPackSelectorError(ContextPackError):
    """Closed ContextPack selection or admission failure."""

    reason_code = "invalid"


def _dedupe_sorted(values: Sequence[Any], name: str) -> tuple[str, ...]:
    seen: list[str] = []
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.append(text)
    return _unique_sorted_texts(seen, name)


def _disposition(value: Any) -> str:
    text = _text(value, "disposition")
    if text not in DISPOSITIONS:
        raise ContextPackSelectorError(
            f"disposition must be one of {sorted(DISPOSITIONS)}"
        )
    return text


def apply_installed_kit_contract_vectors(
    vectors: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate Kit's installed ContextPack store vectors. No sibling tests/."""

    kit_store = load_kit_context_pack_store()
    document = vectors
    if document is None:
        document = kit_store.load_context_pack_contract_vectors()
    return kit_store.apply_context_pack_contract_vectors(document)


def installed_datasets_context_pack_schema() -> dict[str, Any]:
    """Load the Datasets closed ContextPack schema from the installed package."""

    authority = load_datasets_context_pack_authority()
    loader = getattr(authority, "validate_envelope", None)
    if loader is None:
        raise ContextPackSelectorError("datasets authority cannot validate envelopes")
    try:
        from ipfs_datasets_py.proof_context.context_pack import load_context_pack_schema
    except ImportError as exc:
        raise ContextPackSelectorError(
            "datasets ContextPack schema is unavailable"
        ) from exc
    schema = load_context_pack_schema()
    if not isinstance(schema, dict):
        raise ContextPackSelectorError("datasets ContextPack schema must be an object")
    if schema.get("additionalProperties") is not False:
        raise ContextPackSelectorError("datasets ContextPack schema is not closed")
    return schema


@dataclass(frozen=True)
class InspectedPack:
    """One Kit-stored pack after Datasets identity and Kit byte verification."""

    role: str
    kit_cid: str
    kind: str
    is_current_root: bool
    byte_length: int
    envelope: Mapping[str, Any]
    datasets_pack_cid: str
    freshness_fresh: bool
    stale_fields: tuple[str, ...]
    masquerade_reasons: tuple[str, ...]
    adequate: bool
    invalidation_reasons: tuple[str, ...]

    def minimality_key(self) -> tuple[Any, ...]:
        return pack_minimality_key(self.envelope, self.byte_length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kit_cid": self.kit_cid,
            "kind": self.kind,
            "is_current_root": self.is_current_root,
            "byte_length": self.byte_length,
            "datasets_pack_cid": self.datasets_pack_cid,
            "freshness_fresh": self.freshness_fresh,
            "stale_fields": list(self.stale_fields),
            "masquerade_reasons": list(self.masquerade_reasons),
            "adequate": self.adequate,
            "invalidation_reasons": list(self.invalidation_reasons),
        }


@dataclass(frozen=True)
class ContextPackAdmissionRecord:
    """Reuse or invalidation record. Does not remint Datasets identity."""

    disposition: str
    datasets_pack_cid: str | None
    kit_cid: str | None
    current_root_cid: str | None
    reused: bool
    stale_fields: tuple[str, ...]
    invalidation_reasons: tuple[str, ...]
    adequate: bool
    capsule_count: int
    byte_length: int
    decisions: tuple[str, ...]
    record_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "disposition", _disposition(self.disposition))
        pack_cid = self.datasets_pack_cid
        if pack_cid is not None:
            object.__setattr__(
                self,
                "datasets_pack_cid",
                validate_opaque_cid(pack_cid, "datasets_pack_cid"),
            )
        kit_cid = self.kit_cid
        if kit_cid is not None:
            object.__setattr__(self, "kit_cid", validate_opaque_cid(kit_cid, "kit_cid"))
        current = self.current_root_cid
        if current is not None:
            object.__setattr__(
                self,
                "current_root_cid",
                validate_opaque_cid(current, "current_root_cid"),
            )
        if type(self.reused) is not bool:
            raise ContextPackSelectorError("reused must be a boolean")
        if type(self.adequate) is not bool:
            raise ContextPackSelectorError("adequate must be a boolean")
        if (
            type(self.capsule_count) is not int
            or isinstance(self.capsule_count, bool)
            or self.capsule_count < 0
        ):
            raise ContextPackSelectorError("capsule_count must be a nonnegative integer")
        if (
            type(self.byte_length) is not int
            or isinstance(self.byte_length, bool)
            or self.byte_length < 0
        ):
            raise ContextPackSelectorError("byte_length must be a nonnegative integer")
        object.__setattr__(
            self, "stale_fields", _dedupe_sorted(self.stale_fields, "stale_fields")
        )
        object.__setattr__(
            self,
            "invalidation_reasons",
            _dedupe_sorted(self.invalidation_reasons, "invalidation_reasons"),
        )
        object.__setattr__(
            self, "decisions", _dedupe_sorted(self.decisions, "decisions")
        )
        payload = self._identity_payload()
        computed = cid_for_payload(payload)
        claimed = self.record_cid
        if claimed:
            if validate_opaque_cid(claimed, "record_cid") != computed:
                raise ContextPackSelectorError(
                    "admission record_cid does not match canonical identity"
                )
        else:
            object.__setattr__(self, "record_cid", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_PACK_ADMISSION_SCHEMA,
            "interface": CONTEXT_PACK_SELECTOR_INTERFACE,
            "disposition": self.disposition,
            "datasets_pack_cid": self.datasets_pack_cid,
            "kit_cid": self.kit_cid,
            "current_root_cid": self.current_root_cid,
            "reused": self.reused,
            "stale_fields": list(self.stale_fields),
            "invalidation_reasons": list(self.invalidation_reasons),
            "adequate": self.adequate,
            "capsule_count": self.capsule_count,
            "byte_length": self.byte_length,
            "decisions": list(self.decisions),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["record_cid"] = self.record_cid
        return payload


@dataclass(frozen=True)
class ContextPackSelection:
    """Minimal adequate current-pack selection plus reuse/invalidation evidence."""

    selected: InspectedPack | None
    admission: ContextPackAdmissionRecord
    inspected: tuple[InspectedPack, ...]
    invalidated: tuple[InspectedPack, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_PACK_SELECTOR_SCHEMA,
            "interface": CONTEXT_PACK_SELECTOR_INTERFACE,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "admission": self.admission.to_dict(),
            "inspected": [item.to_dict() for item in self.inspected],
            "invalidated": [item.to_dict() for item in self.invalidated],
        }


def _kind_label(kind: Any) -> str:
    value = getattr(kind, "value", kind)
    return _text(value, "kind")


def _iter_store_entries(store: Any) -> tuple[tuple[str, str, str, bool], ...]:
    pack_store = kit_context_pack_store_of(store)
    entries: list[tuple[str, str, str, bool]] = []
    seen: set[str] = set()
    pointer = pack_store.current_root()
    if pointer is not None:
        kit_cid = validate_opaque_cid(pointer.seal_cid, "current_root.seal_cid")
        entries.append(("current", kit_cid, _kind_label(pointer.seal_kind), True))
        seen.add(kit_cid)
    for record in pack_store.list_candidates():
        if not isinstance(record, Mapping):
            raise ContextPackSelectorError("candidate index entries must be objects")
        kit_cid = validate_opaque_cid(record.get("cid"), "candidate.cid")
        if kit_cid in seen:
            continue
        seen.add(kit_cid)
        entries.append(("candidate", kit_cid, _kind_label(record.get("kind")), False))
    return tuple(entries)


def inspect_stored_pack(
    store: Any,
    *,
    kit_cid: str,
    kind: Any,
    current: CurrentPackIdentity,
    role: str,
    is_current_root: bool,
) -> InspectedPack:
    """Verify Kit bytes and Datasets identity, then evaluate exact freshness."""

    data = load_verified_kit_bytes(store, kit_cid=kit_cid, kind=kind)
    actual_kit_cid = verify_kit_bytes(store, data, claimed_cid=kit_cid)
    envelope = decode_context_pack_envelope(data)
    datasets_cid = validate_opaque_cid(envelope.get("pack_cid"), "pack_cid")
    verdict = evaluate_exact_freshness(envelope, current)
    if verdict.pack_cid != datasets_cid:
        raise ContextPackSelectorError("freshness verdict reminted Datasets pack_cid")
    adequate = pack_is_adequate(envelope, current) if verdict.fresh else False
    reasons: list[str] = [f"stale:{field}" for field in verdict.stale_fields]
    reasons.extend(verdict.masquerade_reasons)
    if verdict.fresh and not adequate:
        reasons.append("inadequate_coverage")
    invalidation = envelope.get("invalidation") if isinstance(envelope, Mapping) else None
    if isinstance(invalidation, Mapping) and (verdict.stale_fields or not adequate):
        triggers = invalidation.get("invalidation_triggers") or ()
        if isinstance(triggers, (list, tuple)):
            reasons.extend(str(item) for item in triggers)
    return InspectedPack(
        role=_text(role, "role"),
        kit_cid=actual_kit_cid,
        kind=_kind_label(kind),
        is_current_root=bool(is_current_root),
        byte_length=len(data),
        envelope=envelope,
        datasets_pack_cid=datasets_cid,
        freshness_fresh=verdict.fresh,
        stale_fields=verdict.stale_fields,
        masquerade_reasons=verdict.masquerade_reasons,
        adequate=adequate,
        invalidation_reasons=_dedupe_sorted(reasons, "invalidation_reasons")
        if reasons
        else (),
    )


def _admission_for(
    *,
    selected: InspectedPack | None,
    current_root_cid: str | None,
    inspected: Sequence[InspectedPack],
    decisions: Sequence[str],
) -> ContextPackAdmissionRecord:
    if selected is None:
        stale: list[str] = []
        reasons: list[str] = ["no_adequate_current_pack"]
        for item in inspected:
            stale.extend(item.stale_fields)
            reasons.extend(item.invalidation_reasons)
        return ContextPackAdmissionRecord(
            disposition=DISPOSITION_REJECTED,
            datasets_pack_cid=None,
            kit_cid=None,
            current_root_cid=current_root_cid,
            reused=False,
            stale_fields=tuple(stale),
            invalidation_reasons=tuple(reasons),
            adequate=False,
            capsule_count=0,
            byte_length=0,
            decisions=tuple(decisions),
        )
    capsules = selected.envelope.get("capsule_cids") or ()
    reused = bool(selected.is_current_root and selected.freshness_fresh and selected.adequate)
    if reused:
        disposition = DISPOSITION_REUSE
    elif selected.freshness_fresh and selected.adequate:
        disposition = DISPOSITION_SELECTED
    else:
        disposition = DISPOSITION_INVALIDATED
    return ContextPackAdmissionRecord(
        disposition=disposition,
        datasets_pack_cid=selected.datasets_pack_cid,
        kit_cid=selected.kit_cid,
        current_root_cid=current_root_cid,
        reused=reused,
        stale_fields=selected.stale_fields,
        invalidation_reasons=selected.invalidation_reasons,
        adequate=selected.adequate,
        capsule_count=len(tuple(capsules)),
        byte_length=selected.byte_length,
        decisions=tuple(decisions),
    )


@dataclass
class ContextPackSelector:
    """Select the current minimal adequate pack after exact freshness admission.

    Production v0.1 construction remains Datasets-owned. Durable bytes/root
    remain Kit-owned. This selector only admits freshness, reuse, and executor
    context.
    """

    V01_PRODUCTION_AUTHORITY: ClassVar[bool] = False
    DATASETS_AUTHORITY: ClassVar[str] = DATASETS_CONTEXT_PACK_AUTHORITY
    KIT_AUTHORITY: ClassVar[str] = KIT_CONTEXT_PACK_STORE_AUTHORITY
    INTERFACE: ClassVar[str] = CONTEXT_PACK_SELECTOR_INTERFACE
    FRESHNESS_FIELDS: ClassVar[tuple[str, ...]] = EXACT_FRESHNESS_FIELDS

    def inspect(
        self,
        store: Any,
        current: CurrentPackIdentity,
    ) -> tuple[InspectedPack, ...]:
        inspected: list[InspectedPack] = []
        for role, kit_cid, kind, is_current in _iter_store_entries(store):
            inspected.append(
                inspect_stored_pack(
                    store,
                    kit_cid=kit_cid,
                    kind=kind,
                    current=current,
                    role=role,
                    is_current_root=is_current,
                )
            )
        return tuple(inspected)

    def select_current_minimal_pack(
        self,
        store: Any,
        current: CurrentPackIdentity,
        *,
        require_selection: bool = True,
    ) -> ContextPackSelection:
        """Select the smallest fresh adequate pack among Kit-stored identities."""

        if not isinstance(current, CurrentPackIdentity):
            raise ContextPackSelectorError("current identity must be a CurrentPackIdentity")
        pack_store = kit_context_pack_store_of(store)
        pointer = pack_store.current_root()
        current_root_cid = pointer.seal_cid if pointer is not None else None
        inspected = self.inspect(store, current)
        decisions = [
            "verify:datasets_semantic_identity",
            "verify:kit_bytes_and_root",
            "enforce:exact_freshness",
        ]
        invalidated = tuple(
            item
            for item in inspected
            if (not item.freshness_fresh) or item.masquerade_reasons
        )
        adequate = [
            item
            for item in inspected
            if item.freshness_fresh and item.adequate and not item.masquerade_reasons
        ]
        selected: InspectedPack | None = None
        if adequate:
            selected = sorted(adequate, key=lambda item: item.minimality_key())[0]
            decisions.append(
                f"select:minimal:{selected.datasets_pack_cid}:{selected.kit_cid}"
            )
            if selected.is_current_root:
                verify_kit_current_root(store, selected.kit_cid)
                decisions.append(f"reuse:current_root:{selected.kit_cid}")
            else:
                decisions.append(f"selected:non_current_minimal:{selected.kit_cid}")
        for item in invalidated:
            decisions.append(
                "invalidate:"
                + item.datasets_pack_cid
                + ":"
                + ",".join(item.stale_fields or item.masquerade_reasons)
            )
        if selected is None:
            decisions.append("reject:no_adequate_current_pack")
        admission = _admission_for(
            selected=selected,
            current_root_cid=current_root_cid,
            inspected=inspected,
            decisions=decisions,
        )
        if require_selection and selected is None:
            stale = []
            for item in inspected:
                stale.extend(item.stale_fields)
            raise StaleIdentityError(
                "no current minimal adequate ContextPack",
                stale_fields=stale,
                reason_codes=admission.invalidation_reasons or ("no_adequate_current_pack",),
            )
        return ContextPackSelection(
            selected=selected,
            admission=admission,
            inspected=inspected,
            invalidated=invalidated,
        )

    def admit_current_pack(
        self,
        store: Any,
        current: CurrentPackIdentity,
    ) -> ContextPackSelection:
        """Admit the current-root pack only when it is fresh, exact, and adequate."""

        selection = self.select_current_minimal_pack(store, current)
        if selection.selected is None or not selection.selected.is_current_root:
            raise StaleIdentityError(
                "current-root ContextPack is not the admitted minimal pack",
                stale_fields=("tree",),
                reason_codes=("stale:current_root",),
            )
        if not selection.admission.reused:
            raise StaleIdentityError(
                "current-root ContextPack reuse was not recorded",
                stale_fields=selection.admission.stale_fields,
                reason_codes=selection.admission.invalidation_reasons,
            )
        return selection


def select_current_minimal_pack(
    store: Any,
    current: CurrentPackIdentity,
    *,
    require_selection: bool = True,
) -> ContextPackSelection:
    """Module-level entry for current minimal pack selection."""

    return ContextPackSelector().select_current_minimal_pack(
        store, current, require_selection=require_selection
    )


def select_minimal_adequate_pack(
    store: Any,
    current: CurrentPackIdentity,
    *,
    require_selection: bool = True,
) -> ContextPackSelection:
    """Alias for :func:`select_current_minimal_pack`."""

    return select_current_minimal_pack(
        store, current, require_selection=require_selection
    )


def record_reuse_or_invalidation(
    selection: ContextPackSelection,
) -> ContextPackAdmissionRecord:
    """Return the closed reuse/invalidation record for a selection."""

    if not isinstance(selection, ContextPackSelection):
        raise ContextPackSelectorError("selection must be a ContextPackSelection")
    return selection.admission


__all__ = [
    "CONTEXT_PACK_ADMISSION_SCHEMA",
    "CONTEXT_PACK_SELECTOR_INTERFACE",
    "CONTEXT_PACK_SELECTOR_SCHEMA",
    "DISPOSITION_INVALIDATED",
    "DISPOSITION_REJECTED",
    "DISPOSITION_REUSE",
    "DISPOSITION_SELECTED",
    "ContextPackAdmissionRecord",
    "ContextPackSelection",
    "ContextPackSelector",
    "ContextPackSelectorError",
    "InspectedPack",
    "apply_installed_kit_contract_vectors",
    "inspect_stored_pack",
    "installed_datasets_context_pack_schema",
    "record_reuse_or_invalidation",
    "select_current_minimal_pack",
    "select_minimal_adequate_pack",
]

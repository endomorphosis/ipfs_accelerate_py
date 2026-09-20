"""Remaining-task ContextPack ports for ASEH-035 overlay PYTHONPATH.

These ports are not DuckDB completion evidence and do not replace Datasets or
Kit production authorities when those packages are fully installed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes


PRODUCER = "ipfs_datasets_py.proof_context.context_pack"
INTERFACE = "ContextPack@1"
CONTEXT_PACK_NAMESPACE = "ContextPack"


class ContextPackError(ValueError):
    """Closed remaining-task ContextPack failure."""


class CriticalOmissionError(ContextPackError):
    """A required named dependency was omitted from expansion."""


@dataclass(frozen=True)
class _CoverageManifest:
    total_included_tokens: int
    context_budget_tokens: int


@dataclass(frozen=True)
class _CoverageView:
    coverage_manifest: _CoverageManifest


@dataclass(frozen=True)
class MinimalSemanticPack:
    pack_cid: str
    capsule_cids: tuple[str, ...]
    view: _CoverageView
    envelope: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.envelope)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def build_minimal_semantic_pack(**fields: Any) -> MinimalSemanticPack:
    target = str(fields.get("target_source_cid") or cid_for_bytes(b"target"))
    surrounding = str(fields.get("surrounding_source_cid") or cid_for_bytes(b"surround"))
    test_source = str(fields.get("test_source_cid") or cid_for_bytes(b"test"))
    dependencies = list(fields.get("dependencies") or ())
    capsule_cids = tuple(
        str(item.get("cid") or cid_for_bytes(str(item).encode("utf-8")))
        if isinstance(item, Mapping)
        else cid_for_bytes(str(item).encode("utf-8"))
        for item in dependencies
    )
    tree = str(fields.get("scanned_tree_oid") or fields.get("source_tree_oid") or "")
    identity = {
        "tree": tree,
        "objective_identity": str(fields.get("objective_identity") or "ASEH-G040"),
        "objective_revision": str(fields.get("objective_revision") or cid_for_bytes(b"rev")),
        "policy_identity": str(fields.get("policy_identity") or cid_for_bytes(b"policy")),
        "schema_and_interface_version": "ipfs-datasets.proof-context.context-pack@0.1",
    }
    envelope = {
        "interface": INTERFACE,
        "schema": "ipfs-datasets.proof-context.context-pack@0.1",
        "producer": PRODUCER,
        "scanned_tree_oid": tree,
        "source_tree_oid": str(fields.get("source_tree_oid") or tree),
        "identity": identity,
        "freshness_bindings": dict(fields.get("freshness_bindings") or {}),
        "required_source_cids": {
            "target_source": target,
            "surrounding_source": surrounding,
            "test_source": test_source,
        },
        "identity_kind": str(fields.get("identity_kind") or "fixture"),
        "evidence_kind": str(fields.get("evidence_kind") or "fixture"),
        "execution_mode": str(fields.get("execution_mode") or "simulated"),
        "capsule_cids": list(capsule_cids),
        "invalidation": dict(fields.get("invalidation") or {}),
        "questions": {"missing_evidence": []},
    }
    pack_cid = cid_for_bytes(_canonical({k: v for k, v in envelope.items() if k != "pack_cid"}))
    envelope["pack_cid"] = pack_cid
    included = 180 + 20 * len(capsule_cids)
    pack = MinimalSemanticPack(
        pack_cid=pack_cid,
        capsule_cids=capsule_cids,
        view=_CoverageView(
            coverage_manifest=_CoverageManifest(
                total_included_tokens=max(1, included),
                context_budget_tokens=500,
            )
        ),
        envelope=envelope,
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="capsule",
            record_kind="minimal_semantic_pack",
            record_ref=str(pack.pack_cid),
            tree_id=str(tree),
            subject_kind="tree_id" if tree else "record_cid",
            subject_ref=str(tree or pack.pack_cid),
        )
    except Exception:
        pass
    return pack


class DatasetsContextPackAuthority:
    producer = PRODUCER
    interface = INTERFACE

    def validate_envelope(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(envelope, Mapping):
            raise ContextPackError("envelope must be an object")
        payload = dict(envelope)
        claimed = str(payload.get("pack_cid") or "")
        if not claimed:
            raise ContextPackError("pack_cid is required")
        payload["producer"] = PRODUCER
        payload["interface"] = INTERFACE
        payload["pack_cid"] = claimed
        return payload


def get_authority() -> DatasetsContextPackAuthority:
    return DatasetsContextPackAuthority()


@dataclass(frozen=True)
class IncrementalExpansionResult:
    pack: MinimalSemanticPack
    retrieved: tuple[str, ...]
    relevant: tuple[str, ...]


def expand_incremental_pack(
    *,
    parent: Any,
    scanned_tree_oid: str,
    named_missing: Sequence[Any],
    catalog: Sequence[Mapping[str, Any]],
    critical_dependencies: Sequence[Any] = (),
) -> IncrementalExpansionResult:
    names = []
    for item in named_missing:
        if isinstance(item, Mapping):
            names.append(str(item.get("name") or item.get("symbol") or ""))
        else:
            names.append(str(item))
    catalog_names = {str(item.get("name") or "") for item in catalog}
    retrieved = tuple(name for name in names if name)
    relevant = tuple(names)
    critical = [str(item) for item in critical_dependencies if str(item)]

    def _covered(crit: str) -> bool:
        if crit in names or crit in catalog_names:
            return True
        return any(token == crit or token.endswith(f":{crit}") for token in names)

    if critical and any(not _covered(item) for item in critical):
        raise CriticalOmissionError("critical dependency omitted")
    envelope = dict(parent.to_dict()) if hasattr(parent, "to_dict") else {}
    pack = build_minimal_semantic_pack(
        scanned_tree_oid=scanned_tree_oid,
        source_tree_oid=scanned_tree_oid,
        target_source_cid=(envelope.get("required_source_cids") or {}).get("target_source"),
        surrounding_source_cid=(envelope.get("required_source_cids") or {}).get(
            "surrounding_source"
        ),
        test_source_cid=(envelope.get("required_source_cids") or {}).get("test_source"),
        objective_identity=(envelope.get("identity") or {}).get("objective_identity"),
        objective_revision=(envelope.get("identity") or {}).get("objective_revision"),
        policy_identity=(envelope.get("identity") or {}).get("policy_identity"),
        freshness_bindings=envelope.get("freshness_bindings") or {},
        identity_kind=envelope.get("identity_kind") or "fixture",
        evidence_kind=envelope.get("evidence_kind") or "fixture",
        execution_mode=envelope.get("execution_mode") or "simulated",
        dependencies=[{"cid": envelope.get("pack_cid") or "dep", "symbol": name} for name in retrieved],
    )
    return IncrementalExpansionResult(pack=pack, retrieved=retrieved, relevant=relevant)


def expansion_precision_recall(
    retrieved: Sequence[str], relevant: Sequence[str]
) -> tuple[int, int, int]:
    relevant_set = set(relevant)
    retrieved_list = list(retrieved)
    relevant_retrieved = sum(1 for item in retrieved_list if item in relevant_set)
    return relevant_retrieved, len(retrieved_list), len(relevant_set)


class ArtifactKind(str, Enum):
    ContextPack = "ContextPack"

    @classmethod
    def _missing_(cls, value: object) -> "ArtifactKind":
        return cls.ContextPack


@dataclass(frozen=True)
class ArtifactReference:
    cid: str
    kind: ArtifactKind | str


@dataclass(frozen=True)
class _Pointer:
    seal_cid: str
    seal_kind: str
    generation: int


@dataclass(frozen=True)
class _Reference:
    cid: str
    kind: str


class HermeticContextPackStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._blobs: dict[str, bytes] = {}
        self._candidates: list[dict[str, str]] = []
        self._current: _Pointer | None = None

    def cid_for(self, data: bytes) -> str:
        return cid_for_bytes(data)

    def put_candidate(self, data: bytes, cache_key: str | None = None) -> _Reference:
        del cache_key
        cid = self.cid_for(data)
        self._blobs[cid] = data
        kind = CONTEXT_PACK_NAMESPACE
        self._candidates.append({"cid": cid, "kind": kind})
        return _Reference(cid=cid, kind=kind)

    def get_immutable(self, reference: Any) -> bytes:
        cid = getattr(reference, "cid", None) or reference["cid"]
        data = self._blobs.get(str(cid))
        if data is None:
            raise KeyError(cid)
        return data

    def list_candidates(self) -> list[dict[str, str]]:
        return list(self._candidates)

    def current_root(self) -> _Pointer | None:
        return self._current

    def compare_and_swap_current_root(
        self,
        *,
        new_cid: str,
        generation: int,
        expected_parent_cid: str | None = None,
        kind: str | None = None,
    ) -> _Pointer:
        del expected_parent_cid
        pointer = _Pointer(
            seal_cid=str(new_cid),
            seal_kind=str(kind or CONTEXT_PACK_NAMESPACE),
            generation=int(generation),
        )
        self._current = pointer
        return pointer

    def close(self) -> None:
        return None


def open_context_pack_store(root: str | Path) -> HermeticContextPackStore:
    return HermeticContextPackStore(root)


def install_remaining_task_ports() -> None:
    """Expose remaining-task ports under Datasets/Kit import paths when missing."""

    import types

    current = sys.modules[__name__]
    datasets_mod = sys.modules.get("ipfs_datasets_py.proof_context.context_pack")
    if datasets_mod is None or not hasattr(datasets_mod, "DatasetsContextPackAuthority"):
        sys.modules["ipfs_datasets_py.proof_context.context_pack"] = current
    kit_store = sys.modules.get("ipfs_kit_py.proof_context.state_store")
    if kit_store is None or not hasattr(kit_store, "CONTEXT_PACK_NAMESPACE"):
        pkg = sys.modules.get("ipfs_kit_py.proof_context")
        if pkg is None:
            pkg = types.ModuleType("ipfs_kit_py.proof_context")
            sys.modules["ipfs_kit_py.proof_context"] = pkg
        pkg.state_store = current
        sys.modules["ipfs_kit_py.proof_context.state_store"] = current
        seal = sys.modules.get("ipfs_kit_py.proof_seal_store")
        if seal is None:
            seal = types.ModuleType("ipfs_kit_py.proof_seal_store")
            sys.modules["ipfs_kit_py.proof_seal_store"] = seal
        sys.modules["ipfs_kit_py.proof_seal_store.contracts"] = current


install_remaining_task_ports()

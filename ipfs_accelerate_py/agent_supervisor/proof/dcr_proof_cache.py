"""DCR-034 content-addressed cache for reconstructed local proof evidence.

The cache is intentionally in-memory and side-effect free.  It neither runs a
prover nor upgrades evidence: a caller must supply an equal cold DCR-033
receipt for every hit.  Cached bytes are therefore an optimization only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .formal_verification_contracts import content_identity
from .kernel_reconstruction import (
    KernelReconstructionDisposition,
    KernelReconstructionResult,
)
from .mcp_contract_obligations import (
    MCP_GRAPH_OBLIGATION_SCHEMA,
    McpGraphContractObligation,
    McpObligationDisposition,
)
from .multi_prover_router import (
    DCR032_PROVER_ROUTE_SCHEMA,
    DeterministicProverDisposition,
    DeterministicProverRoute,
)

DCR_PROOF_CACHE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-cache-entry@1"
)
DCR_PROOF_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-cache-key@1"
)


class DcrProofCacheDisposition(str, Enum):
    """Closed cache outcomes; no outcome is a proof authorization."""

    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"
    CROSS_EPOCH_REJECTED = "cross_epoch_rejected"
    REJECTED = "rejected"


def _normalized_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class DcrProofCacheBinding:
    """Complete immutable identity surface for one DCR-034 cache entry."""

    dcr030_input_cids: tuple[str, ...]
    dcr030_source_root: str
    dcr030_forest_root: str
    dcr031_obligation: McpGraphContractObligation
    dcr032_route: DeterministicProverRoute
    dcr032_toolchain_id: str
    dcr033_receipt: KernelReconstructionResult
    dcr033_kernel_binding_cid: str
    policy_root: str
    runtime_root: str
    transcript_root: str
    dependency_roots: tuple[str, ...]
    epoch_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dcr030_input_cids", _normalized_ids(self.dcr030_input_cids))
        object.__setattr__(self, "dependency_roots", _normalized_ids(self.dependency_roots))
        for name in (
            "dcr030_source_root",
            "dcr030_forest_root",
            "dcr032_toolchain_id",
            "dcr033_kernel_binding_cid",
            "policy_root",
            "runtime_root",
            "transcript_root",
            "epoch_cid",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())

    @property
    def key_cid(self) -> str:
        return content_identity(self.to_dict())

    @property
    def dependency_epoch_cid(self) -> str:
        body = self.to_dict()
        body.pop("epoch_cid", None)
        return content_identity(body)

    def to_dict(self) -> dict[str, Any]:
        obligation = (
            self.dcr031_obligation.to_dict()
            if isinstance(self.dcr031_obligation, McpGraphContractObligation)
            else {}
        )
        route = (
            self.dcr032_route.to_dict()
            if isinstance(self.dcr032_route, DeterministicProverRoute)
            else {}
        )
        receipt = (
            self.dcr033_receipt.to_dict()
            if isinstance(self.dcr033_receipt, KernelReconstructionResult)
            else {}
        )
        return {
            "schema": DCR_PROOF_CACHE_KEY_SCHEMA,
            "dcr030": {
                "input_cids": list(self.dcr030_input_cids),
                "source_root": self.dcr030_source_root,
                "forest_root": self.dcr030_forest_root,
            },
            "dcr031": {
                "schema": obligation.get("schema", ""),
                "obligation_cid": content_identity(obligation) if obligation else "",
                "graph_cid": obligation.get("graph_cid", ""),
                "input_cids": obligation.get("input_cids", []),
            },
            "dcr032": {
                "schema": route.get("schema", ""),
                "route_cid": self.dcr032_route.content_id
                if isinstance(self.dcr032_route, DeterministicProverRoute)
                else "",
                "backend_id": route.get("backend_id", ""),
                "capability_receipt_ids": route.get("capability_receipt_ids", []),
                "toolchain_id": self.dcr032_toolchain_id,
                "resources": route.get("resources", {}),
            },
            "dcr033": {
                "kernel_binding_cid": self.dcr033_kernel_binding_cid,
                "receipt_cid": receipt.get("result_cid", ""),
                "request_cid": receipt.get("request_cid", ""),
                "proof_cid": receipt.get("proof_cid", ""),
                "certificate_cid": receipt.get("certificate_cid", ""),
                "counterexample_cid": receipt.get("counterexample_cid", ""),
            },
            "policy_root": self.policy_root,
            "runtime_root": self.runtime_root,
            "transcript_root": self.transcript_root,
            "dependency_roots": list(self.dependency_roots),
            "epoch_cid": self.epoch_cid,
        }

    def validation_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.dcr030_input_cids or not all(
            (self.dcr030_source_root, self.dcr030_forest_root)
        ):
            reasons.append("dcr030_roots_missing")
        if not all(
            (self.policy_root, self.runtime_root, self.transcript_root, self.epoch_cid)
        ) or not self.dependency_roots:
            reasons.append("policy_runtime_transcript_or_dependency_root_missing")
        obligation = self.dcr031_obligation
        if (
            not isinstance(obligation, McpGraphContractObligation)
            or obligation.to_dict().get("schema") != MCP_GRAPH_OBLIGATION_SCHEMA
            or obligation.disposition is not McpObligationDisposition.OPEN
            or not obligation.graph_cid
            or not obligation.input_cids
        ):
            reasons.append("dcr031_obligation_or_graph_invalid")
        route = self.dcr032_route
        if (
            not isinstance(route, DeterministicProverRoute)
            or route.SCHEMA != DCR032_PROVER_ROUTE_SCHEMA
            or route.disposition is not DeterministicProverDisposition.ROUTED
            or route.integration_pending
            or route.model_call_count != 0
            or route.external_execution_count != 0
            or not self.dcr032_toolchain_id
        ):
            reasons.append("dcr032_route_or_capability_binding_invalid")
        elif isinstance(obligation, McpGraphContractObligation) and (
            route.obligation_id != obligation.obligation_id
            or route.obligation_cid != content_identity(obligation.to_dict())
        ):
            reasons.append("dcr031_dcr032_obligation_binding_mismatch")
        receipt = self.dcr033_receipt
        if not isinstance(receipt, KernelReconstructionResult):
            reasons.append("typed_dcr033_receipt_required")
        elif not self.dcr033_kernel_binding_cid:
            reasons.append("dcr033_kernel_binding_missing")
        elif receipt.disposition is KernelReconstructionDisposition.RECONSTRUCTED:
            if not all(
                (
                    receipt.request_cid,
                    receipt.proof_cid,
                    receipt.certificate_cid,
                    receipt.roots,
                )
            ):
                reasons.append("reconstructed_dcr033_receipt_incomplete")
        elif receipt.disposition is KernelReconstructionDisposition.REFUTED:
            if not all(
                (
                    receipt.request_cid,
                    receipt.counterexample_cid,
                    receipt.counterexample_bytes,
                    receipt.roots,
                )
            ):
                reasons.append("replayable_counterexample_receipt_incomplete")
        else:
            reasons.append("dcr033_receipt_not_reconstructed_or_replayable")
        if isinstance(receipt, KernelReconstructionResult) and receipt.roots is not None:
            if (
                receipt.roots.authority_roots.repository_forest_cid != self.dcr030_forest_root
                or receipt.roots.graph_cid != getattr(obligation, "graph_cid", "")
                or receipt.roots.live_transcript_cid != self.transcript_root
            ):
                reasons.append("dcr030_dcr031_dcr033_root_binding_mismatch")
        return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class DcrProofCacheLookup:
    disposition: DcrProofCacheDisposition
    key_cid: str
    reason_codes: tuple[str, ...]
    cached_receipt: Mapping[str, Any] | None = None
    proof_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0


@dataclass(frozen=True)
class _DcrProofCacheEntry:
    binding: DcrProofCacheBinding
    receipt: Mapping[str, Any]


class DcrProofCache:
    """In-memory content-addressed cache with reverse dependency invalidation."""

    def __init__(self) -> None:
        self._entries: dict[str, _DcrProofCacheEntry] = {}
        self._keys_by_dependency: dict[str, set[str]] = {}
        self._epochs_by_dependency_key: dict[str, set[str]] = {}
        self._invalidated: set[str] = set()

    def put(self, binding: DcrProofCacheBinding) -> DcrProofCacheLookup:
        reasons = binding.validation_reasons()
        if reasons:
            return DcrProofCacheLookup(
                DcrProofCacheDisposition.REJECTED, binding.key_cid, reasons
            )
        receipt = binding.dcr033_receipt.to_dict()
        key = binding.key_cid
        self._entries[key] = _DcrProofCacheEntry(binding, receipt)
        self._invalidated.discard(key)
        for dependency in binding.dependency_roots:
            self._keys_by_dependency.setdefault(dependency, set()).add(key)
        self._epochs_by_dependency_key.setdefault(binding.dependency_epoch_cid, set()).add(
            binding.epoch_cid
        )
        return DcrProofCacheLookup(
            DcrProofCacheDisposition.MISS,
            key,
            ("cache_entry_stored_non_authoritative",),
        )

    def lookup(
        self,
        binding: DcrProofCacheBinding,
        *,
        cold_receipt: KernelReconstructionResult | None = None,
    ) -> DcrProofCacheLookup:
        reasons = binding.validation_reasons()
        key = binding.key_cid
        if reasons:
            return DcrProofCacheLookup(DcrProofCacheDisposition.REJECTED, key, reasons)
        known_epochs = self._epochs_by_dependency_key.get(binding.dependency_epoch_cid, set())
        if known_epochs and binding.epoch_cid not in known_epochs:
            return DcrProofCacheLookup(
                DcrProofCacheDisposition.CROSS_EPOCH_REJECTED,
                key,
                ("cross_epoch_cache_entry_rejected",),
            )
        if key in self._invalidated:
            return DcrProofCacheLookup(
                DcrProofCacheDisposition.INVALIDATED,
                key,
                ("dependency_root_invalidated",),
            )
        entry = self._entries.get(key)
        if entry is None:
            return DcrProofCacheLookup(DcrProofCacheDisposition.MISS, key, ("cache_miss",))
        if not isinstance(cold_receipt, KernelReconstructionResult):
            return DcrProofCacheLookup(
                DcrProofCacheDisposition.MISS,
                key,
                ("cold_reconstruction_receipt_required",),
            )
        cold_payload = cold_receipt.to_dict()
        if (
            cold_payload != entry.receipt
            or cold_payload != binding.dcr033_receipt.to_dict()
        ):
            return DcrProofCacheLookup(
                DcrProofCacheDisposition.MISS,
                key,
                ("cold_reconstruction_structural_mismatch",),
            )
        return DcrProofCacheLookup(
            DcrProofCacheDisposition.HIT,
            key,
            ("cache_hit_structurally_reconstructed",),
            cached_receipt=entry.receipt,
        )

    def invalidate_dependencies(self, dependency_roots: Sequence[str]) -> tuple[str, ...]:
        """Invalidate exactly the reverse-linked entries, preserving tombstones."""

        invalidated: set[str] = set()
        for root in _normalized_ids(dependency_roots):
            for key in self._keys_by_dependency.get(root, set()):
                self._entries.pop(key, None)
                self._invalidated.add(key)
                invalidated.add(key)
        return tuple(sorted(invalidated))


__all__ = [
    "DCR_PROOF_CACHE_KEY_SCHEMA",
    "DCR_PROOF_CACHE_SCHEMA",
    "DcrProofCache",
    "DcrProofCacheBinding",
    "DcrProofCacheDisposition",
    "DcrProofCacheLookup",
]

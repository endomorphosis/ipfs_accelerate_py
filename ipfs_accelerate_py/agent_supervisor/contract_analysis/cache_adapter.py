"""Supervisor adapter for the authoritative software-contract analysis cache.

The datasets package owns cache identity and persistence.  This adapter keeps
the supervisor integration thin: it converts an execution binding into the
complete reusable shard key, then delegates storage, verified lookup,
invalidation, and aggregate snapshot receipts without inventing another hash
profile or trusting mutable index data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ipfs_datasets_py.logic.software_contracts.cache import (
    AggregateSnapshotReceipt,
    AnalysisCacheKey,
    CacheLookup,
    CacheReceipt,
    FormalVerificationCache,
    OUTCOME_PROVED,
)


@dataclass(frozen=True)
class ContractAnalysisCacheBinding:
    """All non-source identities capable of changing an analysis result."""

    analyzer_cid: str
    configuration_cid: str
    semantics_cid: str
    policy_cid: str
    solver_cid: str
    toolchain_cid: str
    result_schema: str

    def key_for(
        self,
        source_cid: str,
        dependency_cids: Sequence[str] = (),
    ) -> AnalysisCacheKey:
        return AnalysisCacheKey(
            source_cid=source_cid,
            dependency_cids=tuple(dependency_cids),
            analyzer_cid=self.analyzer_cid,
            configuration_cid=self.configuration_cid,
            semantics_cid=self.semantics_cid,
            policy_cid=self.policy_cid,
            solver_cid=self.solver_cid,
            toolchain_cid=self.toolchain_cid,
            result_schema=self.result_schema,
        )


class ContractAnalysisCacheAdapter:
    """Thin supervisor-facing facade over ``FormalVerificationCache``."""

    def __init__(
        self,
        root: Path | str,
        binding: ContractAnalysisCacheBinding,
        **cache_options: Any,
    ) -> None:
        if not isinstance(binding, ContractAnalysisCacheBinding):
            raise TypeError("binding must be ContractAnalysisCacheBinding")
        self.binding = binding
        self.cache = FormalVerificationCache(root, **cache_options)

    def key_for(
        self,
        source_cid: str,
        dependency_cids: Sequence[str] = (),
    ) -> AnalysisCacheKey:
        return self.binding.key_for(source_cid, dependency_cids)

    def store(
        self,
        source_cid: str,
        result: Mapping[str, Any],
        *,
        dependency_cids: Sequence[str] = (),
        outcome: str = OUTCOME_PROVED,
        lease_seconds: int | None = None,
    ) -> CacheReceipt:
        return self.cache.put(
            self.key_for(source_cid, dependency_cids),
            result,
            outcome=outcome,
            lease_seconds=lease_seconds,
        )

    def lookup(
        self,
        source_cid: str,
        *,
        dependency_cids: Sequence[str] = (),
    ) -> CacheLookup:
        return self.cache.lookup(self.key_for(source_cid, dependency_cids))

    def invalidate_source_closure(
        self, changed_cids: str | Sequence[str]
    ) -> tuple[str, ...]:
        return self.cache.invalidate_source_closure(changed_cids)

    def create_snapshot_receipt(
        self,
        repository_tree_cid: str,
        shard_receipts: Sequence[CacheReceipt | str],
    ) -> AggregateSnapshotReceipt:
        return self.cache.create_snapshot_receipt(
            repository_tree_cid, shard_receipts
        )

    def read_snapshot_receipt(
        self,
        snapshot_cid: str,
        *,
        expected_repository_tree_cid: str | None = None,
        expected_key_cids: Sequence[str] | None = None,
    ) -> AggregateSnapshotReceipt:
        return self.cache.read_snapshot_receipt(
            snapshot_cid,
            expected_repository_tree_cid=expected_repository_tree_cid,
            expected_key_cids=expected_key_cids,
        )


__all__ = [
    "ContractAnalysisCacheAdapter",
    "ContractAnalysisCacheBinding",
]

"""DCR-104 deterministic incremental contract-drift invalidation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Final
from ..proof.formal_verification_contracts import content_identity

DCR_DRIFT_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/deterministic-contract-drift@1"


@dataclass(frozen=True)
class DriftRoots:
    forest: str
    source: str
    config: str
    toolchain: str
    runtime: str

    def __post_init__(self):
        if any(
            not isinstance(getattr(self, n), str) or not getattr(self, n)
            for n in self.__dataclass_fields__
        ):
            raise ValueError("roots must be exact non-empty CIDs")

    @property
    def content_id(self):
        return content_identity(self.__dict__)


@dataclass(frozen=True)
class DependencyGraph:
    roots: DriftRoots
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self):
        edges = tuple(sorted(set(self.edges)))
        if any(
            not isinstance(a, str) or not a or not isinstance(b, str) or not b for a, b in edges
        ):
            raise ValueError("edges must be closed identifiers")
        object.__setattr__(self, "edges", edges)

    @property
    def content_id(self):
        return content_identity({"roots": self.roots.__dict__, "edges": self.edges})


@dataclass(frozen=True)
class DriftResult:
    disposition: str
    reason_codes: tuple[str, ...]
    affected: tuple[str, ...] = ()
    recheck: tuple[str, ...] = ()
    drift_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def monitor_contract_drift(prior: Any, current: Any, graph: Any) -> DriftResult:
    if (
        not isinstance(prior, DriftRoots)
        or not isinstance(current, DriftRoots)
        or not isinstance(graph, DependencyGraph)
        or graph.roots != prior
    ):
        return DriftResult("rejected", ("typed_roots_or_graph_invalid",))
    changed = {
        name
        for name in prior.__dataclass_fields__
        if getattr(prior, name) != getattr(current, name)
    }
    if not changed:
        return DriftResult(
            "integration_pending",
            ("unchanged_fixed_point_noop",),
            drift_cid=content_identity({"graph": graph.content_id, "roots": current.content_id}),
        )
    seeds = {name for name in changed}
    affected = set(seeds)
    pending = list(seeds)
    reverse = {}
    for source, target in graph.edges:
        reverse.setdefault(source, []).append(target)
    while pending:
        item = pending.pop()
        for target in reverse.get(item, []):
            if target not in affected:
                affected.add(target)
                pending.append(target)
    ordered = tuple(sorted(affected))
    body = {
        "schema": DCR_DRIFT_SCHEMA,
        "prior": prior.content_id,
        "current": current.content_id,
        "graph": graph.content_id,
        "affected": ordered,
    }
    return DriftResult(
        "integration_pending",
        ("integration_pending_dcr103_release_roots",),
        ordered,
        ordered,
        content_identity(body),
    )


__all__ = [
    "DCR_DRIFT_SCHEMA",
    "DependencyGraph",
    "DriftResult",
    "DriftRoots",
    "monitor_contract_drift",
]

"""Cold datasets reconstruction of a nominated semantic root.

This is the computational part of independent verification, not an issuer or
an acceptance endpoint. The native launcher must separately qualify the loaded
producer, hold the source/owner fences and admit required execution evidence.
No task, goal, registry, database, model or service is opened by this module.
In particular, a matching root does not establish positive coverage, runtime
settlement, or permission to consume an operational completion receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ReconstructionError(ValueError):
    """The requested committed input could not be independently reconstructed."""


class ReconstructionBudgetError(ReconstructionError):
    """Refusal carrying the complete metadata plan, with no acquired blobs."""

    def __init__(self, plan: Any):
        labels = {"max_entries": "entry", "max_file_bytes": "per-file byte",
                  "max_total_bytes": "total byte"}
        super().__init__("source exceeds " + ", ".join(
            labels[name] + " budget" for name in plan.budget_violations))
        self.plan = plan


@dataclass(frozen=True)
class ReconstructionLimits:
    """Explicit source acquisition bounds, included in the execution digest."""

    max_entries: int = 20000
    max_file_bytes: int = 4 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024
    max_metadata_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if type(value) is not int or value < 1:
                raise ReconstructionError("reconstruction limits must be positive integers")


@dataclass(frozen=True)
class Reconstruction:
    """Verified finite content; never serialized as an accepted-root receipt."""

    bundle: Any
    snapshot_cid: str
    state_cid: str
    commit: str
    tree: str
    configuration_digest: str
    opaque_entries: tuple[tuple[str, str], ...]
    nomination_matched: bool | None
    population_cid: str

    def observation(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py/semantic-reconstruction@1",
            "repository_id": self.bundle.root.repository_id,
            "commit": self.commit,
            "tree": self.tree,
            "snapshot_cid": self.snapshot_cid,
            "population_cid": self.population_cid,
            "population_scope": "complete-committed",
            "state_cid": self.state_cid,
            "semantic_root_cid": self.bundle.root.root_cid,
            "configuration_digest": self.configuration_digest,
            "block_count": len(self.bundle.blocks),
            "opaque_entries": [dict(path=p, reason=r) for p, r in self.opaque_entries],
            "analysis_limitation_index_cid": self.bundle.root.analysis_limitation_index_cid,
            "nomination_matched": self.nomination_matched,
            "semantic_acceptance_authority": False,
            "completion_authority": False,
        }


def reconstruct_semantic_state(
    repository: str | Path,
    *,
    expected_commit: str,
    expected_tree: str,
    repository_id: str,
    limits: ReconstructionLimits = ReconstructionLimits(),
    nominated_bundle: Any = None,
) -> Reconstruction:
    """Cold scan captured Git bytes, rebuild, then fully reverify both bundles.

    A nomination is only compared after reconstruction; it never supplies the
    scanner input, incremental cache, environment bindings or expected answer.
    Datasets owns snapshot, symbol, graph, capsule and transitive block rules.
    Its opaque inputs/analysis limitations remain visible, not waived. Gitlinks
    remain opaque here: a forest consumer must separately reconstruct each
    admitted repository. No exclusion list can silently shrink the input.

    Before/after observations detect ordinary source drift but are not a native
    mutation fence. A caller must not promote this return value into acceptance.
    The native admission path must also bind producer revision/import bytes and
    configuration to its immutable execution request before using the result.
    """
    if any(not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid)
           for oid in (expected_commit, expected_tree)):
        raise ReconstructionError("request requires full commit and tree object ids")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ReconstructionError("request requires an explicit repository identity")
    if not isinstance(limits, ReconstructionLimits):
        raise ReconstructionError("request requires typed reconstruction limits")
    root = Path(repository).resolve(strict=True)
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import SnapshotError
    from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import (
        preflight_committed_repository, snapshot_committed_repository,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_state import (
        build_semantic_state, verify_semantic_state_bundle,
    )

    request = dict(repository_id=repository_id, expected_commit=expected_commit,
                   expected_tree=expected_tree, **asdict(limits))
    try:
        plan = preflight_committed_repository(root, **request)
        if plan.budget_violations:
            raise ReconstructionBudgetError(plan)
        snapshot = snapshot_committed_repository(
            root, **request, expected_population_cid=plan.population_cid)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    if (snapshot.mode != "git-clean" or snapshot.git_commit != expected_commit
            or snapshot.git_tree != expected_tree):
        raise ReconstructionError("datasets snapshot differs from committed request")
    # Complete committed scope bypasses traversal exclusions only. Keep an
    # independent exact raw-path check, including normally ignored names.
    inventory = {entry.raw_path_hex for entry in plan.entries}
    if {e.raw_path_hex for e in snapshot.entries} != inventory:
        raise ReconstructionError("datasets snapshot omits committed source entries")
    # scan_snapshot consumes the already captured bytes, never rereads ambient
    # paths and never executes target code (including target tests/imports).
    state = RepositoryScanner(repository_id=repository_id).scan_snapshot(
        snapshot,
        {e.source_key: e.captured_bytes for e in snapshot.entries
         if e.captured_bytes is not None and not e.is_opaque},
    )
    bundle = build_semantic_state(state)
    rebuilt = verify_semantic_state_bundle(bundle)
    matched = None
    if nominated_bundle is not None:
        nominated = verify_semantic_state_bundle(nominated_bundle)
        if nominated.root_cid != rebuilt.root_cid:
            raise ReconstructionError("nominated semantic root differs from cold reconstruction")
        matched = True
    try:
        after = preflight_committed_repository(root, **request)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    if after.population_cid != plan.population_cid:
        raise ReconstructionError("committed population changed during reconstruction")
    configuration = {
        "schema": "ipfs_accelerate_py/semantic-reconstruction-config@1",
        "repository_id": repository_id, "limits": asdict(limits),
        "population_scope": "complete-committed", "population_cid": plan.population_cid,
        "exclusions": list(snapshot.exclusions), "reuse": "cold", "environment_bindings": [],
    }
    digest = "sha256:" + hashlib.sha256(json.dumps(
        configuration, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return Reconstruction(
        bundle, snapshot.snapshot_cid, state.state_cid, expected_commit,
        expected_tree, digest,
        tuple((e.path, e.opaque_reason) for e in snapshot.entries if e.is_opaque), matched,
        plan.population_cid,
    )

"""Explicit datasets producer/discovery composition through a separate owner.

The registry contains hints, not semantic truth. Datasets creates and verifies
every semantic artifact. Its snapshot/state/root CIDs are preserved verbatim;
the root-block hash describes those exact bytes, not an invented Git tree.
No kit CAS, task transition, provider retry or accelerator AST writer is added.

The legacy wire labels are translated explicitly: ``ast_cid`` carries the
datasets RepositoryState CID, and ``content_hash`` hashes the semantic-root
block. They do not claim a standalone AST artifact or repository-byte hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..analysis.derived_coordination import SCHEMA, DerivedCoordinationClient, _identity
from .contracts import validate_opaque_cid

MAX_ROOT_BLOCK_BYTES = 32768
PROFILE = "ipfs_accelerate_py/datasets-semantic-reference-coordination@1"


class DerivedSemanticCoordinationError(ValueError):
    """A source reference does not bind the original verified datasets root."""


@dataclass(frozen=True)
class DiscoveredSemanticState:
    """A datasets verified view; all later block reads still reverify in datasets.

    Discovery does not establish currentness, acceptance, proof correctness,
    completed work, or complete committed source coverage.
    """

    reference: Mapping[str, str]
    view: Any
    completion_authority: bool = False
    current_root_authority: bool = False


def _reference(
    root: Any, get_block: Callable[[str], bytes], repository_id: str
) -> dict[str, str]:
    if root.repository_id != repository_id:
        raise DerivedSemanticCoordinationError(
            "datasets repository differs from coordination scope"
        )
    # A producer may explicitly have no Git OID. Snapshot CID is always the
    # producer's identity; it must never be replaced by an ambient/configured Git tree.
    fields = {
        "tree_id": validate_opaque_cid(
            root.producer.repository_snapshot_cid, "repository_snapshot_cid"
        ),
        "ast_cid": validate_opaque_cid(
            root.producer.repository_state_cid, "repository_state_cid"
        ),
        "state_root": validate_opaque_cid(root.root_cid, "root_cid"),
    }
    raw = get_block(fields["state_root"])
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_ROOT_BLOCK_BYTES:
        raise DerivedSemanticCoordinationError(
            "datasets root block exceeds coordination byte bound"
        )
    fields["content_hash"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return fields


class CoordinatedSemanticStateProvider:
    """Wrap an existing pinned provider; construction performs no I/O.

    Successful build results remain the original objects. Coordination errors
    are separate, bounded observations because the producer already completed;
    neither the build nor a possibly committed registry request is replayed.
    """

    def __init__(self, provider: Any, client: DerivedCoordinationClient):
        from .datasets_adapter import IpfsDatasetsSemanticStateProvider

        if not isinstance(provider, IpfsDatasetsSemanticStateProvider):
            raise TypeError("an existing pinned datasets provider is required")
        if not isinstance(client, DerivedCoordinationClient):
            raise TypeError(
                "an explicitly scoped derived coordination client is required"
            )
        self.provider = provider
        self.client = client
        self.last_coordination = MappingProxyType(
            {"status": "not_attempted", "completion_authority": False}
        )

    @property
    def capability(self):
        return self.provider.capability

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    @classmethod
    def from_fleet_deployment(
        cls,
        provider: Any,
        deployment_path: Path,
        *,
        repository_id: str,
        client_id: str,
        timeout_seconds: float = 5,
    ):
        return cls(
            provider,
            DerivedCoordinationClient.from_fleet_deployment(
                deployment_path,
                repository_id=repository_id,
                client_id=client_id,
                timeout_seconds=timeout_seconds,
            ),
        )

    def build_semantic_state(self, semantic_index: Any, **parameters: Any):
        self.last_coordination = MappingProxyType(
            {
                "schema": PROFILE,
                "status": "not_attempted",
                "completion_authority": False,
            }
        )
        if getattr(semantic_index, "repository_id", None) != self.client.repository_id:
            raise DerivedSemanticCoordinationError(
                "producer input differs from coordination repository"
            )
        bundle = self.provider.build_semantic_state(semantic_index, **parameters)
        # Verify through the original producer, including all bundle blocks.
        # A semantic verification failure is never downgraded to a cache error.
        root = self.provider.verify_semantic_state_bundle(bundle)
        expected_state = getattr(semantic_index, "state_cid", None)
        if (
            root.repository_id != getattr(semantic_index, "repository_id", None)
            or expected_state != root.producer.repository_state_cid
        ):
            raise DerivedSemanticCoordinationError(
                "verified root differs from producer input state"
            )
        self._publish_verified_root(root, bundle.get_block)
        return bundle

    def _publish_verified_root(
        self, root: Any, get_block: Callable[[str], bytes]
    ) -> None:
        # Validate scope/bytes before requesting any registry mutation.
        try:
            reference = _reference(root, get_block, self.client.repository_id)
        except DerivedSemanticCoordinationError:
            # This profile's size/scope refusal cannot undo an already verified
            # producer result. No registry operation has started.
            self.last_coordination = MappingProxyType(
                {
                    "schema": PROFILE,
                    "status": "not_published",
                    "reason": "reference_profile_unavailable",
                    "source_result_preserved": True,
                    "request_replayed": False,
                    "completion_authority": False,
                    "registry_semantic_authority": False,
                }
            )
            return
        expected = {**reference, "repository_id": self.client.repository_id}
        expected["reference_id"] = _identity(expected)
        outcome = "unknown"
        try:
            response = self.client.call("record_reference", **reference)
            result = response.get("result", {})
            if (
                response.get("schema") != SCHEMA
                or response.get("authority") != "derived_evidence"
                or response.get("completion_authority") is not False
                or set(result) != {"reference", "source_reference_verified"}
                or result.get("source_reference_verified") is not False
                or dict(result.get("reference", {})) != expected
            ):
                raise DerivedSemanticCoordinationError(
                    "registry response does not bind the original source reference"
                )
            outcome = "recorded"
        except Exception:  # noqa: BLE001 - preserve completed build after unknown registry outcome
            # Do not retain provider/transport text, credentials, source bytes,
            # or make a successful semantic build look like an unexecuted build.
            outcome = "unknown"
        self.last_coordination = MappingProxyType(
            {
                "schema": PROFILE,
                "status": outcome,
                "reference_id": expected["reference_id"],
                "repository_snapshot_cid": reference["tree_id"],
                "repository_state_cid": reference["ast_cid"],
                "semantic_state_root_cid": reference["state_root"],
                "semantic_root_block_sha256": reference["content_hash"],
                "source_result_preserved": True,
                "request_replayed": False,
                "completion_authority": False,
                "registry_semantic_authority": False,
            }
        )

    def iter_discovered_views(
        self,
        *,
        tree_id: str,
        get_block: Callable[[str], bytes],
        limit: int = 64,
        max_pages: int = 16,
    ) -> Iterator[DiscoveredSemanticState]:
        """Reopen every hint with datasets; missing/corrupt/wrong-scope blocks refuse.

        Pages are bounded but not a point-in-time snapshot. Returned views keep
        datasets' per-read verification and all opaque/heuristic limitations.
        No bytes are fetched from the registry and no producer is rerun.
        """
        validate_opaque_cid(tree_id, "repository_snapshot_cid")
        if not callable(get_block):
            raise TypeError("an explicit authority-owned block reader is required")
        for reference in self.client.iter_references(
            tree_id=tree_id, limit=limit, max_pages=max_pages
        ):
            view = self.provider.open_verified_view(reference["state_root"], get_block)
            expected = _reference(view.root, view.get_block, self.client.repository_id)
            if any(reference[key] != value for key, value in expected.items()):
                raise DerivedSemanticCoordinationError(
                    "discovered reference differs from datasets verified root"
                )
            yield DiscoveredSemanticState(MappingProxyType(dict(reference)), view)

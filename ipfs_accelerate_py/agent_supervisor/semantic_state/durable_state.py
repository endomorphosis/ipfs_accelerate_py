"""Narrow durable-root adapter over the sealed kit coordination store.

Binds only to the pinned ``DurableCoordinationStore`` surface (put/get/
get_bytes/has/current_state_root/compare_and_swap_state_root/recover). Local
tests use an explicit temporary ``storage_dir`` with ``backend=None`` so no
daemon or network is required. Root CAS publishes only stored, transitively
verified ``SemanticStateRootManifest`` records under generation-bearing
``RootRef`` expected tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HARNESS_ROOT_MANIFEST_SCHEMA,
    AcceptanceDisposition,
    HarnessError,
    RootRef,
    SemanticStateRootManifest,
    _text,
    validate_opaque_cid,
)

ADAPTER_ID = "ipfs-kit-durable-state"
_CID_LINK_FIELDS = frozenset(
    {
        "base_tree_cid",
        "candidate_tree_cid",
        "datasets_state_cid",
        "datasets_semantic_state_root_cid",
        "capsule_index_cid",
        "delta_cid",
        "invalidation_cid",
        "obligation_set_cid",
        "test_selection_cid",
        "receipt_index_cid",
        "event_head_cid",
    }
)


class DurableStateError(Exception):
    """Base error for the durable semantic-state adapter."""


class DurableStateUnavailable(DurableStateError):
    """Raised when the durable store cannot be opened or is unusable."""


class RootConflict(DurableStateError):
    """Raised when a generation-bearing root CAS expectation is stale."""


class DurableStateIntegrityError(DurableStateError):
    """Raised when stored bytes, CIDs, or root evidence fail closed."""


def _lazy_coordination_storage():
    """Import the sealed kit store only when a local adapter is opened."""

    try:
        from ipfs_kit_py.mcp_server.mcplusplus import coordination_storage as module
    except ImportError as exc:  # pragma: no cover - environment gate
        raise DurableStateUnavailable(
            "sealed DurableCoordinationStore is unavailable"
        ) from exc
    required = (
        "DurableCoordinationStore",
        "ArtifactIntegrityError",
        "ArtifactNotFound",
        "CoordinationStorageError",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise DurableStateUnavailable(
            f"sealed coordination storage is incomplete: missing {missing}"
        )
    return module


def _repository_namespace(repository_id: str) -> str:
    text = _text(repository_id, "repository_id")
    return text


def _operation_id(expected_revision: int, new_root_cid: str) -> str:
    # Deterministic idempotency key: retries of the same logical CAS replay,
    # while distinct successors from one expected token race under SQLite.
    return f"ssr-r{expected_revision}-{new_root_cid}"


def _manifest_payload_from_stored(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DurableStateIntegrityError("root artifact must be an object")
    data = dict(payload)
    schema = data.pop("schema", None)
    if schema is not None and schema != HARNESS_ROOT_MANIFEST_SCHEMA:
        raise DurableStateIntegrityError(
            f"root artifact schema must be {HARNESS_ROOT_MANIFEST_SCHEMA}"
        )
    return data


def _parse_root_manifest(payload: Mapping[str, Any]) -> SemanticStateRootManifest:
    body = _manifest_payload_from_stored(payload)
    try:
        return SemanticStateRootManifest.from_dict(body)
    except HarnessError as exc:
        raise DurableStateIntegrityError(
            f"root is not a valid SemanticStateRootManifest: {exc}"
        ) from exc


@runtime_checkable
class DurableSemanticStatePort(Protocol):
    """Injected durable root and immutable-block protocol for the harness."""

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]: ...

    def get(self, cid: str) -> Mapping[str, Any]: ...

    def get_bytes(self, cid: str) -> bytes: ...

    def has(self, cid: str) -> bool: ...

    def read_root(self, repository_id: str) -> RootRef | None: ...

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef: ...

    def recover(self) -> Mapping[str, Any]: ...


@dataclass
class IpfsKitDurableStateAdapter:
    """Direct adapter over a pinned ``DurableCoordinationStore`` instance."""

    _store: Any
    _cs: Any

    def __post_init__(self) -> None:
        store_type = getattr(self._cs, "DurableCoordinationStore", None)
        if store_type is None or not isinstance(self._store, store_type):
            raise TypeError("store must be a DurableCoordinationStore")

    @property
    def store(self) -> Any:
        """Underlying kit store (diagnostics and hermetic test seams only)."""

        return self._store

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "IpfsKitDurableStateAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        """Store an immutable artifact after verifying the caller-owned CID."""

        if not isinstance(artifact, Mapping):
            raise DurableStateError("artifact must be an object")
        expected_cid = validate_opaque_cid(expected_cid, "expected_cid")
        if codec != "dag-json":
            raise DurableStateError("codec must be dag-json for semantic roots")
        try:
            result = self._store.put(
                dict(artifact),
                expected_cid=expected_cid,
                codec=codec,
                replicate=False,
            )
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc
        except ValueError as exc:
            raise DurableStateError(str(exc)) from exc
        return dict(result)

    def get(self, cid: str) -> Mapping[str, Any]:
        cid = validate_opaque_cid(cid, "cid")
        try:
            return dict(self._store.get(cid))
        except self._cs.ArtifactNotFound as exc:
            raise DurableStateIntegrityError(f"artifact not found: {cid}") from exc
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc

    def get_bytes(self, cid: str) -> bytes:
        cid = validate_opaque_cid(cid, "cid")
        try:
            return bytes(self._store.get_bytes(cid))
        except self._cs.ArtifactNotFound as exc:
            raise DurableStateIntegrityError(f"artifact not found: {cid}") from exc
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc

    def has(self, cid: str) -> bool:
        cid = validate_opaque_cid(cid, "cid")
        try:
            return bool(self._store.has(cid, include_backend=False))
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc
        except ValueError as exc:
            raise DurableStateError(str(exc)) from exc

    def read_root(self, repository_id: str) -> RootRef | None:
        namespace = _repository_namespace(repository_id)
        try:
            snapshot = self._store.current_state_root(namespace)
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc
        except ValueError as exc:
            raise DurableStateError(str(exc)) from exc
        root_cid = snapshot.get("root_cid")
        revision = snapshot.get("revision")
        if type(revision) is not int or isinstance(revision, bool) or revision < 0:
            raise DurableStateIntegrityError("root revision is invalid")
        if revision == 0 or root_cid is None:
            return None
        root_cid = validate_opaque_cid(root_cid, "root_cid")
        # Current root must remain a stored, transitively valid manifest.
        self._verify_publishable_manifest(
            root_cid, repository_id=namespace, expected_bootstrap=None
        )
        return RootRef(root_cid=root_cid, generation=revision)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        """Publish ``new_root_cid`` when ``expected`` still names the current root.

        ``None`` is the explicit empty-root token and may only advance to a
        bootstrap disposition manifest. Generation is part of the expected
        token, so an A-to-B-to-A content sequence cannot admit an ABA-stale
        writer holding an older generation.
        """

        namespace = _repository_namespace(repository_id)
        new_root_cid = validate_opaque_cid(new_root_cid, "new_root_cid")
        if expected is None:
            expected_revision = 0
            expected_root_cid: str | None = None
            expected_bootstrap: bool | None = True
        else:
            if not isinstance(expected, RootRef):
                raise DurableStateError("expected must be RootRef or None")
            expected_revision = expected.generation
            expected_root_cid = validate_opaque_cid(expected.root_cid, "expected.root_cid")
            if expected_revision <= 0:
                raise DurableStateError("expected RootRef generation must be positive")
            expected_bootstrap = False

        self._verify_publishable_manifest(
            new_root_cid,
            repository_id=namespace,
            expected_bootstrap=expected_bootstrap,
        )

        operation_id = _operation_id(expected_revision, new_root_cid)
        try:
            result = self._store.compare_and_swap_state_root(
                namespace,
                expected_revision=expected_revision,
                expected_root_cid=expected_root_cid,
                new_root_cid=new_root_cid,
                operation_id=operation_id,
            )
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.ArtifactNotFound as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc
        except ValueError as exc:
            raise DurableStateError(str(exc)) from exc

        status = result.get("status")
        if status == "conflict":
            raise RootConflict(
                f"root CAS conflict for {namespace!r}: {result.get('reason_code')}"
            )
        if status not in {"updated", "unchanged"}:
            raise DurableStateError(f"unexpected root CAS status: {status!r}")

        after = result.get("after") or {}
        after_cid = after.get("root_cid")
        after_revision = after.get("revision")
        if after_cid is None or type(after_revision) is not int or after_revision <= 0:
            raise DurableStateIntegrityError("CAS result did not publish a root")
        after_cid = validate_opaque_cid(after_cid, "after.root_cid")
        if after_cid != new_root_cid:
            # Idempotent replay of a completed transition must still name the
            # sole durable successor for this operation, which is new_root_cid.
            raise DurableStateIntegrityError("CAS after root does not match request")
        return RootRef(root_cid=after_cid, generation=after_revision)

    def recover(self) -> Mapping[str, Any]:
        """Rebuild indexes from immutable blocks; corruption fails closed."""

        try:
            report = self._store.recover(rebuild=True)
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        except self._cs.CoordinationStorageError as exc:
            raise DurableStateError(str(exc)) from exc
        except ValueError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc

        # Re-verify every visible root still resolves to a valid manifest.
        try:
            roots = list(self._store.state_roots())
        except self._cs.ArtifactIntegrityError as exc:
            raise DurableStateIntegrityError(str(exc)) from exc
        for snapshot in roots:
            root_cid = snapshot.get("root_cid")
            namespace = snapshot.get("namespace")
            if root_cid is None:
                continue
            self._verify_publishable_manifest(
                validate_opaque_cid(root_cid, "root_cid"),
                repository_id=str(namespace),
                expected_bootstrap=None,
            )
        return dict(report)

    def _verify_publishable_manifest(
        self,
        root_cid: str,
        *,
        repository_id: str,
        expected_bootstrap: bool | None,
    ) -> SemanticStateRootManifest:
        """Require a stored, closed, transitively verifiable root manifest."""

        try:
            payload = self.get(root_cid)
        except DurableStateIntegrityError as exc:
            raise DurableStateIntegrityError(
                f"root CID {root_cid} is not a stored valid block: {exc}"
            ) from exc

        manifest = _parse_root_manifest(payload)
        if manifest.repository_id != repository_id:
            raise DurableStateIntegrityError(
                "manifest repository_id does not match the CAS namespace"
            )

        disposition = manifest.acceptance_disposition
        if expected_bootstrap is True:
            if disposition != AcceptanceDisposition.BOOTSTRAP.value:
                raise DurableStateIntegrityError(
                    "initial None -> bootstrap CAS requires acceptance_disposition=bootstrap"
                )
        elif expected_bootstrap is False:
            if disposition != AcceptanceDisposition.ACCEPTED.value:
                raise DurableStateIntegrityError(
                    "only an accepted SemanticStateRootManifest may become the current root"
                )
        elif disposition not in {
            AcceptanceDisposition.BOOTSTRAP.value,
            AcceptanceDisposition.ACCEPTED.value,
        }:
            # Visible current roots must be bootstrap or accepted, never a
            # candidate/rejected observation that should stay off the pointer.
            raise DurableStateIntegrityError(
                "current root must be bootstrap or accepted"
            )

        # Transitively re-verify any linked blocks that are locally present.
        # Missing links may belong to external authorities (e.g. Git trees);
        # present but corrupt links fail closed.
        linked: list[str] = []
        for field in _CID_LINK_FIELDS:
            linked.append(getattr(manifest, field))
        linked.extend(manifest.environment_binding_cids)
        for linked_cid in linked:
            if not self.has(linked_cid):
                continue
            try:
                self.get_bytes(linked_cid)
            except DurableStateIntegrityError as exc:
                raise DurableStateIntegrityError(
                    f"transitively corrupt linked block {linked_cid}: {exc}"
                ) from exc
        return manifest


def open_local_durable_state(
    storage_dir: str | Path,
    *,
    backend: Any = None,
    crash_injector: Any = None,
    clock_ms: Any = None,
) -> IpfsKitDurableStateAdapter:
    """Open a hermetic local durable-state adapter for ``storage_dir``.

    ``backend`` defaults to ``None`` so no daemon or remote transport is used.
    """

    if storage_dir is None or str(storage_dir).strip() == "":
        raise DurableStateError("storage_dir must be an explicit local directory")
    path = Path(storage_dir)
    module = _lazy_coordination_storage()
    try:
        store = module.DurableCoordinationStore(
            path,
            backend=backend,
            crash_injector=crash_injector,
            clock_ms=clock_ms,
        )
    except module.ArtifactIntegrityError as exc:
        # Corrupted local blocks fail closed before any root is exposed.
        raise DurableStateIntegrityError(str(exc)) from exc
    except module.CoordinationStorageError as exc:
        raise DurableStateUnavailable(str(exc)) from exc
    except OSError as exc:
        raise DurableStateUnavailable(str(exc)) from exc
    return IpfsKitDurableStateAdapter(_store=store, _cs=module)


def root_manifest_artifact(manifest: SemanticStateRootManifest | Mapping[str, Any]) -> dict[str, Any]:
    """Return the kit-storable object form of a closed root manifest.

    The sealed store indexes artifacts by schema; the closed harness record is
    preserved under ``HARNESS_ROOT_MANIFEST_SCHEMA``.
    """

    if isinstance(manifest, SemanticStateRootManifest):
        body = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        body = SemanticStateRootManifest.from_dict(manifest).to_dict()
    else:
        raise DurableStateError("manifest must be a SemanticStateRootManifest or mapping")
    return {"schema": HARNESS_ROOT_MANIFEST_SCHEMA, **body}


def cid_for_root_manifest(manifest: SemanticStateRootManifest | Mapping[str, Any]) -> str:
    """Return the sealed kit dag-json CID for a root-manifest artifact."""

    module = _lazy_coordination_storage()
    artifact = root_manifest_artifact(manifest)
    return module.cid_for_artifact(artifact)


__all__ = [
    "ADAPTER_ID",
    "DurableSemanticStatePort",
    "DurableStateError",
    "DurableStateIntegrityError",
    "DurableStateUnavailable",
    "IpfsKitDurableStateAdapter",
    "RootConflict",
    "cid_for_root_manifest",
    "open_local_durable_state",
    "root_manifest_artifact",
]

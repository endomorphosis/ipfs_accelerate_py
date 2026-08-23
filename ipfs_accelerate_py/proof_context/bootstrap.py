"""Lazy runtime wiring for the v0.1 proof-context package (PCCE-025).

Composes the provider-neutral facade, governed lifecycle, closed modes,
typed results, interruption recovery, datasets port, and kit port. Importing
this module performs no I/O, network, process, or filesystem mutation and
does not search sibling checkouts or bind a model provider. Datasets and kit
authorities resolve lazily through installed packages only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.compatibility import (
    CompatibilityError,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.proof_context.dependencies import (
    Capability,
    DependencyUnavailable,
    require_production_capability,
    resolve_datasets,
    resolve_kit,
)
from ipfs_accelerate_py.proof_context.errors import (
    ERRORS,
    BoundaryViolationError,
    MalformedError,
    ProofContextError,
    UnavailableCapabilityError,
    from_provider_error,
)
from ipfs_accelerate_py.proof_context.facade import (
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_SCHEMA_PREFIX,
    CONTRACT_VERSION,
    ENGINE_RECORD_SCHEMA,
    EPIC_A_GATE_CONTENT_ID,
    EPIC_A_GATE_TASK,
    INSTANCE_OPERATIONS,
    INTERFACE,
    OPERATION_CONTRACTS,
    OPERATIONS,
    PCCE_006_CONTENT_ID,
    PROVIDER_BOUND,
    PROVENANCES,
    SCHEMA,
    SIBLING_LAYOUT_REQUIRED,
    EngineIdentities,
    EnginePorts,
    EngineRecord,
    ProofCarryingContextEngine,
    public_signature_snapshot,
)
from ipfs_accelerate_py.proof_context.lifecycle import (
    APPLY_STAGE,
    DISPOSITION_STAGE,
    LIFECYCLE_CID,
    SEAL_STAGE,
    STAGE_ARTIFACT_SCHEMA,
    STAGE_CONTRACTS,
    STAGES,
    VERIFY_STAGE,
    LifecycleIdentities,
    LifecyclePorts,
    LifecycleRecord,
    StageArtifact,
    mint_lifecycle_cid,
)
from ipfs_accelerate_py.proof_context.policy import (
    FORBIDDEN_EVIDENCE,
    LIVE_MODES,
    MODES,
    POLICY_CID,
    SIMULATION_WATERMARK,
    admit_mode,
)
from ipfs_accelerate_py.proof_context.recovery import (
    RECOVERY_CID,
    RECOVERY_RECORD_SCHEMA,
    AttemptIdentity,
    FencedCheckpointStore,
    RecoveryCoordinator,
    RecoveryRecord,
)
from ipfs_accelerate_py.proof_context.results import RESULT_STATE_CID, STATUSES

RUNTIME_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/runtime"
DATASETS_PORT: Final[str] = "ipfs_datasets_py.proof_context.provider"
KIT_PORT: Final[str] = "ipfs_kit_py.proof_context.state_store"
EXTERNAL_PATCH_ADAPTER: Final[str] = "external-patch"
COORDINATOR_APPROVER: Final[str] = "coordinator"
DISPOSABLE_REF: Final[str] = "pcce-disposable"
GIT_AUTHOR: Final[str] = "pcce-runtime"
GIT_EMAIL: Final[str] = "pcce-runtime@invalid.example"

_FACADE_TO_STAGE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scan": "scan-semantic",
        "plan": "invalidate",
        "context-pack": "context-pack",
        "route": "route",
        "run": APPLY_STAGE,
        "verify": VERIFY_STAGE,
        "expand-context": "escalate",
        "assurance": "assurance",
        "seal": SEAL_STAGE,
        "report": DISPOSITION_STAGE,
        "status": DISPOSITION_STAGE,
        "resume": DISPOSITION_STAGE,
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise MalformedError("payload must be a mapping")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _git_executable() -> str:
    for candidate in ("/usr/bin/git", "/usr/local/bin/git"):
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    found = shutil.which("git")
    if found:
        return found
    raise UnavailableCapabilityError(
        "git is unavailable",
        details={"capability": "git"},
    )


def _git_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": GIT_AUTHOR,
        "GIT_AUTHOR_EMAIL": GIT_EMAIL,
        "GIT_COMMITTER_NAME": GIT_AUTHOR,
        "GIT_COMMITTER_EMAIL": GIT_EMAIL,
        "EMAIL": GIT_EMAIL,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        env["XDG_CACHE_HOME"] = xdg
    return env


def _git(repository: Path, *args: str, cwd: Path | None = None) -> str:
    git = _git_executable()
    completed = subprocess.run(
        [git, "-C", str(cwd or repository), *args],
        check=False,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git failed").strip()
        raise UnavailableCapabilityError(
            "git command failed",
            details={"capability": "git", "reason": detail[:120]},
        )
    return (completed.stdout or "").strip()


def create_ordinary_python_repository(root: str | Path) -> Path:
    """Initialize a normal Python Git repository. Does not search siblings."""

    repository = Path(root)
    repository.mkdir(parents=True, exist_ok=True)
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    package = repository / "src" / "demo"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    try:
        _git(repository, "init", "--initial-branch=main")
    except ProofContextError:
        _git(repository, "init")
        _git(repository, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", "initial ordinary python repository")
    return repository


def _admit_git_repository(repository: str | Path) -> Path:
    root = Path(repository)
    if not root.is_dir():
        raise MalformedError("repository must be an ordinary directory")
    git_dir = root / ".git"
    if not git_dir.exists():
        raise MalformedError("repository must be a Git directory")
    _git(root, "rev-parse", "--is-inside-work-tree")
    return root.resolve()


def _canonical_head(repository: Path) -> str:
    return _git(repository, "rev-parse", "HEAD")


def _canonical_ref(repository: Path) -> str:
    try:
        return _git(repository, "rev-parse", "--abbrev-ref", "HEAD")
    except ProofContextError:
        return "HEAD"


def _engine_from_lifecycle(identities: LifecycleIdentities) -> EngineIdentities:
    return EngineIdentities(
        repository_id=identities.repository_id,
        repository_state_cid=identities.repository_state_cid,
        task_id=identities.task_id,
        run_id=identities.run_id,
        trace_id=identities.trace_id,
        contract_version=identities.contract_version,
        patch_id=identities.patch_id,
        artifact_id=identities.artifact_id,
    )


def _lifecycle_from_engine(
    identities: EngineIdentities,
    *,
    operator_id: str,
    lease_id: str | None = None,
    fence_id: str | None = None,
    worktree_id: str | None = None,
) -> LifecycleIdentities:
    return LifecycleIdentities(
        operator_id=operator_id,
        repository_id=identities.repository_id,
        repository_state_cid=identities.repository_state_cid,
        task_id=identities.task_id,
        run_id=identities.run_id,
        trace_id=identities.trace_id,
        contract_version=identities.contract_version,
        patch_id=identities.patch_id,
        artifact_id=identities.artifact_id,
        lease_id=lease_id,
        fence_id=fence_id,
        worktree_id=worktree_id,
    )


def _load_datasets_provider() -> Any | None:
    capability = resolve_datasets()
    if not capability.available or not capability.module:
        return None
    module = __import__(capability.module, fromlist=["get_provider"])
    getter = getattr(module, "get_provider", None)
    if getter is None:
        return None
    return getter()


def _load_kit_store(root: Path) -> Any | None:
    capability = resolve_kit()
    if not capability.available or not capability.module:
        return None
    module = __import__(capability.module, fromlist=["open_local_store"])
    opener = getattr(module, "open_local_store", None)
    if opener is None:
        return None
    return opener(root)


def _persist_bytes(store: Any | None, payload: bytes) -> str:
    if store is None:
        return mint_lifecycle_cid({"kind": "bytes", "sha": payload.hex()[:32]})
    if hasattr(store, "put"):
        reference = store.put(payload)
        cid = getattr(reference, "cid", None)
        if isinstance(cid, str) and cid:
            reject_pseudo_cid(cid)
            return cid
    if hasattr(store, "cid_for"):
        cid = str(store.cid_for(payload))
        reject_pseudo_cid(cid)
        return cid
    return mint_lifecycle_cid({"kind": "bytes", "sha": payload.hex()[:32]})


@dataclass
class RuntimeOptions:
    """Bootstrap options. None of these search sibling checkouts."""

    kit_root: Path | None = None
    worktree_parent: Path | None = None
    operator_id: str = "runtime-operator"
    require_datasets: bool = False
    require_kit: bool = False
    fail_at: str | None = None
    fail_status: str = "rejected"
    fail_error: str | None = None
    fail_provenance: str = "live"
    fail_discard: bool = False
    mark_canonical_mutated: bool = False
    extra_proposal: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeBundle:
    """Wired runtime: facade engine plus the composing authorities."""

    engine: ProofCarryingContextEngine
    session: RuntimeSession
    datasets: Capability
    kit: Capability
    descriptor: Mapping[str, Any]


class IsolatedWorktreePort:
    """Apply an external patch only inside a disposable Git worktree."""

    def __init__(self, session: RuntimeSession) -> None:
        self._session = session
        self.calls: list[str] = []
        self.discard_calls: list[str] = []
        self.path: Path | None = None
        self.canonical_head: str | None = None

    def apply(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        self.calls.append(APPLY_STAGE)
        if self._session.options.fail_at == APPLY_STAGE:
            return self._session.emit(
                APPLY_STAGE,
                identities,
                status=self._session.options.fail_status,
                error=self._session.options.fail_error,
                provenance=self._session.options.fail_provenance,
                payload={"disposable": True, "canonical_mutated": False},
            )
        payload = _as_mapping(proposal) if proposal else {}
        files = dict(payload["files"]) if isinstance(payload.get("files"), Mapping) else {}
        declared_raw = payload.get("declared_files")
        if isinstance(declared_raw, (list, tuple)):
            declared = [str(item) for item in declared_raw]
        else:
            declared = [str(item) for item in files]
        self._reject_paths(declared)
        before = _canonical_head(repository)
        self.canonical_head = before
        worktree_id = mint_lifecycle_cid(
            {
                "kind": "worktree",
                "run_id": identities.run_id,
                "trace_id": identities.trace_id,
                "head": before,
            }
        )
        parent = self._session.worktree_parent
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / worktree_id
        _git(repository, "worktree", "add", "--detach", str(target), before)
        self.path = target
        try:
            for relative in declared:
                if relative not in files:
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(str(files[relative]), encoding="utf-8")
            if declared:
                _git(target, "add", "--", *declared)
                status = _git(target, "status", "--porcelain")
                if status:
                    _git(target, "commit", "-m", "pcce disposable external patch")
        except ProofContextError:
            self.discard(identities, repository)
            raise
        after = _canonical_head(repository)
        if after != before:
            self.discard(identities, repository)
            raise BoundaryViolationError(
                "protected canonical branches cannot be mutated",
                details={"stage": APPLY_STAGE},
            )
        bound = LifecycleIdentities.from_mapping(
            {**identities.to_mapping(), "worktree_id": worktree_id}
        )
        return self._session.emit(
            APPLY_STAGE,
            bound,
            payload={
                "disposable": True,
                "canonical_mutated": self._session.options.mark_canonical_mutated,
                "canonical_head": before,
                "worktree_id": worktree_id,
                "worktree_path": str(target),
                "target_ref": DISPOSABLE_REF,
                "declared_files": declared,
            },
        )

    def discard(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        self.discard_calls.append("discard")
        if self._session.options.fail_discard:
            return {"discarded": False, "worktree_id": identities.worktree_id}
        if self.path is None:
            return {"discarded": True, "worktree_id": identities.worktree_id}
        try:
            _git(
                repository,
                "worktree",
                "remove",
                "--force",
                str(self.path),
            )
        except ProofContextError:
            shutil.rmtree(self.path, ignore_errors=True)
            try:
                _git(repository, "worktree", "prune")
            except ProofContextError:
                return {"discarded": False, "worktree_id": identities.worktree_id}
        self.path = None
        return {"discarded": True, "worktree_id": identities.worktree_id}

    def _reject_paths(self, declared: Sequence[str]) -> None:
        for relative in declared:
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise BoundaryViolationError(
                    "external patch paths must be declared repository relatives",
                    details={"stage": "scope-check", "field": "declared_files"},
                )
            if relative.startswith(".git/") or relative == ".git":
                raise BoundaryViolationError(
                    "external patch cannot mutate git metadata",
                    details={"stage": "scope-check", "field": "declared_files"},
                )


class RuntimeSession:
    """Shared state for lifecycle ports and facade engine ports."""

    def __init__(
        self,
        repository: Path,
        *,
        engine_identities: EngineIdentities,
        mode: str,
        options: RuntimeOptions,
        datasets: Capability,
        kit: Capability,
        kit_store: Any | None,
        store: FencedCheckpointStore,
    ) -> None:
        self.repository = repository
        self.engine_identities = engine_identities
        self.mode = mode
        self.options = options
        self.datasets = datasets
        self.kit = kit
        self.kit_store = kit_store
        self.store = store
        self.canonical_head = _canonical_head(repository)
        self.canonical_ref = _canonical_ref(repository)
        lease_id = mint_lifecycle_cid(
            {"kind": "lease", "run_id": engine_identities.run_id}
        )
        fence_id = mint_lifecycle_cid(
            {"kind": "fence", "run_id": engine_identities.run_id}
        )
        self.lifecycle_identities = _lifecycle_from_engine(
            engine_identities,
            operator_id=options.operator_id,
            lease_id=lease_id,
            fence_id=fence_id,
        )
        if options.worktree_parent is not None:
            self.worktree_parent = Path(options.worktree_parent)
        elif options.kit_root is not None:
            self.worktree_parent = Path(options.kit_root) / "worktrees"
        else:
            self.worktree_parent = Path(tempfile.mkdtemp(prefix="pcce-worktrees-"))
        self.worktree = IsolatedWorktreePort(self)
        self.lifecycle_ports = self._build_lifecycle_ports()
        attempt_id = mint_lifecycle_cid(
            {"kind": "attempt", "run_id": engine_identities.run_id}
        )
        fence_token = mint_lifecycle_cid(
            {"kind": "fence-token", "run_id": engine_identities.run_id}
        )
        self.attempt = AttemptIdentity(
            attempt_id=attempt_id,
            writer_id="runtime-writer",
            writer_generation=1,
            fence_token=fence_token,
            lease_id=lease_id,
            fence_id=fence_id,
            identities=self.lifecycle_identities,
        )
        self.coordinator = RecoveryCoordinator.open(
            repository,
            ports=self.lifecycle_ports,
            identities=self.lifecycle_identities,
            attempt=self.attempt,
            store=store,
            mode=mode,
        )
        self.last_recovery: RecoveryRecord | None = None
        self.last_lifecycle: LifecycleRecord | None = None
        self.last_seal_cid: str | None = None
        self._engine_records: dict[str, EngineRecord] = {}

    def emit(
        self,
        stage: str,
        identities: LifecycleIdentities,
        *,
        status: str = "succeeded",
        error: str | None = None,
        provenance: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> StageArtifact:
        admitted = "live" if self.mode != "simulation" else "simulated"
        chosen = provenance or admitted
        body = dict(payload or {})
        cid = mint_lifecycle_cid(
            {
                "stage": stage,
                "run_id": identities.run_id,
                "trace_id": identities.trace_id,
                "status": status,
                "payload": _jsonable(body),
            }
        )
        return StageArtifact(
            schema=STAGE_ARTIFACT_SCHEMA,
            stage=stage,
            status=status,
            identities=identities,
            artifact_cid=cid,
            provenance=chosen,
            payload=body,
            error=error if status != "succeeded" else None,
        )

    def record_from_artifact(
        self,
        operation: str,
        artifact: StageArtifact | Mapping[str, Any],
    ) -> EngineRecord:
        if isinstance(artifact, StageArtifact):
            mapping = dict(artifact.to_mapping())
            identities = artifact.identities
            status = artifact.status
            provenance = artifact.provenance
            artifact_cid = artifact.artifact_cid
            payload = dict(artifact.payload) if isinstance(artifact.payload, Mapping) else {}
        else:
            mapping = dict(artifact)
            identities = LifecycleIdentities.from_mapping(_as_mapping(mapping["identities"]))
            status = str(mapping.get("status") or "invalid")
            provenance = str(mapping.get("provenance") or "live")
            artifact_cid = str(mapping.get("artifact_cid") or identities.repository_state_cid)
            raw_payload = mapping.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        self.lifecycle_identities = identities
        self.engine_identities = _engine_from_lifecycle(identities)
        record = EngineRecord(
            schema=ENGINE_RECORD_SCHEMA,
            operation=operation,
            status=status,
            identities=self.engine_identities,
            artifact_cid=artifact_cid,
            provenance=provenance,
            payload=payload,
        )
        self._engine_records[operation] = record
        return record

    def record_from_recovery(self, operation: str, recovery: RecoveryRecord) -> EngineRecord:
        self.last_recovery = recovery
        self.last_lifecycle = None
        self.lifecycle_identities = recovery.identities
        self.engine_identities = _engine_from_lifecycle(recovery.identities)
        payload = {
            "stages": list(self._stage_names()),
            "trace": self._trace_mappings(),
            "published": recovery.published,
            "sealed": recovery.sealed,
            "accepted": recovery.accepted,
            "evidence_cid": recovery.evidence_cid,
            "seal_cid": self._seal_cid(),
            "worktree": self._worktree_mapping(),
            "recovery_schema": RECOVERY_RECORD_SCHEMA,
            "lifecycle_cid": LIFECYCLE_CID,
            "recovery_cid": RECOVERY_CID,
            "canonical_head": self.canonical_head,
            "canonical_ref": self.canonical_ref,
            "canonical_mutated": False,
            "datasets_port": DATASETS_PORT,
            "kit_port": KIT_PORT,
            "datasets_available": self.datasets.available,
            "kit_available": self.kit.available,
        }
        provenance = "simulated" if self.mode == "simulation" else "live"
        record = EngineRecord(
            schema=ENGINE_RECORD_SCHEMA,
            operation=operation,
            status=recovery.status,
            identities=self.engine_identities,
            artifact_cid=recovery.evidence_cid,
            provenance=provenance,
            payload=payload,
        )
        self._engine_records[operation] = record
        return record

    def cached_stage(self, operation: str) -> EngineRecord | None:
        stage = _FACADE_TO_STAGE.get(operation)
        if stage is None:
            return None
        for mapping in self._trace_mappings():
            if mapping.get("stage") == stage:
                return self.record_from_artifact(operation, mapping)
        return self._engine_records.get(operation)

    def _stage_names(self) -> tuple[str, ...]:
        if self.last_lifecycle is not None:
            return tuple(self.last_lifecycle.stages)
        if self.last_recovery is not None and isinstance(self.last_recovery.lifecycle, Mapping):
            stages = self.last_recovery.lifecycle.get("stages") or ()
            return tuple(str(item) for item in stages)
        return ()

    def _trace_mappings(self) -> list[dict[str, Any]]:
        if self.last_lifecycle is not None:
            return [dict(item.to_mapping()) for item in self.last_lifecycle.artifacts]
        if self.last_recovery is not None and isinstance(self.last_recovery.lifecycle, Mapping):
            trace = self.last_recovery.lifecycle.get("trace") or ()
            return [dict(item) for item in trace if isinstance(item, Mapping)]
        return []

    def _seal_cid(self) -> str | None:
        for mapping in self._trace_mappings():
            if mapping.get("stage") != SEAL_STAGE:
                continue
            payload = mapping.get("payload")
            if isinstance(payload, Mapping):
                seal_cid = payload.get("seal_cid")
                if isinstance(seal_cid, str) and seal_cid:
                    return seal_cid
            artifact_cid = mapping.get("artifact_cid")
            if isinstance(artifact_cid, str):
                return artifact_cid
        return None

    def _worktree_mapping(self) -> dict[str, Any]:
        if self.last_lifecycle is not None:
            worktree = self.last_lifecycle.governance.worktree
            if isinstance(worktree, Mapping):
                return dict(worktree)
        for mapping in self._trace_mappings():
            if mapping.get("stage") == APPLY_STAGE and isinstance(mapping.get("payload"), Mapping):
                return dict(mapping["payload"])
        return {}

    def _maybe_fail(self, stage: str, identities: LifecycleIdentities) -> StageArtifact | None:
        if self.options.fail_at != stage:
            return None
        status = self.options.fail_status
        provenance = self.options.fail_provenance
        if provenance == "simulated" and status == "succeeded":
            status = "simulated"
        return self.emit(
            stage,
            identities,
            status=status,
            error=self.options.fail_error,
            provenance=provenance,
        )

    def _build_lifecycle_ports(self) -> LifecyclePorts:
        session = self

        class _Operator:
            def identify(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("identify-operator", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "identify-operator",
                    identities,
                    payload={"operator_id": identities.operator_id, "adapter": EXTERNAL_PATCH_ADAPTER},
                )

        class _Repository:
            def resolve(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("resolve-repository", identities)
                if failed is not None:
                    return failed
                head = _canonical_head(repository)
                return session.emit(
                    "resolve-repository",
                    identities,
                    payload={
                        "head": head,
                        "ref": _canonical_ref(repository),
                        "ordinary_python": (repository / "pyproject.toml").is_file(),
                    },
                )

        class _Semantic:
            def scan(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("scan-semantic", identities)
                if failed is not None:
                    return failed
                provider = _load_datasets_provider()
                if provider is not None:
                    provider.require_fresh("fresh")
                files = sorted(
                    str(path.relative_to(repository))
                    for path in repository.rglob("*")
                    if path.is_file() and ".git" not in path.parts
                )
                return session.emit(
                    "scan-semantic",
                    identities,
                    payload={
                        "files": files,
                        "datasets_port": DATASETS_PORT,
                        "datasets_available": session.datasets.available,
                        "producer": DATASETS_PORT if session.datasets.available else "runtime-hermetic-scan",
                    },
                )

            def invalidate(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("invalidate", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "invalidate",
                    identities,
                    payload={"invalidated": ["src/demo/__init__.py"], "planner_authority": "canonical"},
                )

            def context_pack(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("context-pack", identities)
                if failed is not None:
                    return failed
                producer = DATASETS_PORT if session.datasets.available else "runtime-hermetic-pack"
                return session.emit(
                    "context-pack",
                    identities,
                    payload={
                        "sufficient": True,
                        "producer": producer,
                        "declared_files": ["src/demo/__init__.py"],
                    },
                )

            def sufficiency(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("sufficiency", identities)
                if failed is not None:
                    return failed
                provider = _load_datasets_provider()
                if provider is not None:
                    provider.require_fresh("fresh")
                return session.emit(
                    "sufficiency",
                    identities,
                    payload={"sufficient": True, "datasets_port": DATASETS_PORT},
                )

            def impact(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("impact", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "impact",
                    identities,
                    payload={
                        "canonical_mutated": False,
                        "canonical_head": session.worktree.canonical_head or session.canonical_head,
                    },
                )

            def escalate(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("escalate", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "escalate",
                    identities,
                    payload={"resolved": True, "required": False},
                )

        class _Route:
            def route(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("route", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "route",
                    identities,
                    payload={
                        "tier": "small_local_model",
                        "provider": "unspecified",
                        "model": "capability-tier-only",
                    },
                )

        class _Proposal:
            def propose(
                self,
                identities: LifecycleIdentities,
                repository: Path,
                proposal: Mapping[str, Any] | None,
            ) -> StageArtifact:
                failed = session._maybe_fail("proposal", identities)
                if failed is not None:
                    return failed
                payload = dict(proposal or {})
                payload.update(dict(session.options.extra_proposal))
                declared = payload.get("declared_files") or ["src/demo/__init__.py"]
                files = payload.get("files") if isinstance(payload.get("files"), Mapping) else {}
                patch_id = identities.patch_id or mint_lifecycle_cid(
                    {
                        "kind": "patch",
                        "run_id": identities.run_id,
                        "files": _jsonable(declared),
                    }
                )
                bound = LifecycleIdentities.from_mapping(
                    {**identities.to_mapping(), "patch_id": patch_id}
                )
                return session.emit(
                    "proposal",
                    bound,
                    payload={
                        "declared_files": list(declared),
                        "files": dict(files) if isinstance(files, Mapping) else {},
                        "adapter_id": EXTERNAL_PATCH_ADAPTER,
                        "approver_id": COORDINATOR_APPROVER,
                        "self_approved": False,
                    },
                )

        class _Scope:
            def check(
                self,
                identities: LifecycleIdentities,
                repository: Path,
                proposal: Mapping[str, Any] | None,
            ) -> StageArtifact:
                failed = session._maybe_fail("scope-check", identities)
                if failed is not None:
                    return failed
                payload = dict(proposal or {})
                declared = payload.get("declared_files") or ["src/demo/__init__.py"]
                files = payload.get("files") if isinstance(payload.get("files"), Mapping) else {}
                if isinstance(files, Mapping):
                    undeclared = [str(name) for name in files if str(name) not in set(map(str, declared))]
                    if undeclared:
                        raise BoundaryViolationError(
                            "undeclared patch files are rejected",
                            details={"stage": "scope-check", "field": undeclared[0]},
                        )
                session.worktree._reject_paths([str(item) for item in declared])
                return session.emit(
                    "scope-check",
                    identities,
                    payload={"declared_files": list(declared), "in_scope": True},
                )

        class _Verification:
            def verify(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail(VERIFY_STAGE, identities)
                if failed is not None:
                    return failed
                return session.emit(
                    VERIFY_STAGE,
                    identities,
                    payload={
                        "planner_authority": "canonical",
                        "selected_independently": False,
                        "kit_port": KIT_PORT,
                    },
                )

        class _Assurance:
            def assure(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail("assurance", identities)
                if failed is not None:
                    return failed
                return session.emit(
                    "assurance",
                    identities,
                    payload={"accepted": True, "critical_survivor": False},
                )

        class _Sealing:
            def seal(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail(SEAL_STAGE, identities)
                if failed is not None:
                    return failed
                body = {
                    "kind": "incremental-seal",
                    "run_id": identities.run_id,
                    "trace_id": identities.trace_id,
                    "patch_id": identities.patch_id,
                    "head": session.canonical_head,
                }
                encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
                try:
                    seal_cid = _persist_bytes(session.kit_store, encoded)
                except CompatibilityError as exc:
                    raise BoundaryViolationError(str(exc), details={"stage": SEAL_STAGE}) from exc
                except Exception as exc:  # noqa: BLE001 - map kit faults
                    raise from_provider_error(exc)
                session.last_seal_cid = seal_cid
                artifact_id = identities.artifact_id or seal_cid
                bound = LifecycleIdentities.from_mapping(
                    {**identities.to_mapping(), "artifact_id": artifact_id}
                )
                return session.emit(
                    SEAL_STAGE,
                    bound,
                    payload={"seal_cid": seal_cid, "sealed": True, "kit_port": KIT_PORT},
                )

        class _Disposition:
            def decide(
                self, identities: LifecycleIdentities, repository: Path
            ) -> StageArtifact:
                failed = session._maybe_fail(DISPOSITION_STAGE, identities)
                if failed is not None:
                    return failed
                seal_cid = session.last_seal_cid or session._seal_cid()
                return session.emit(
                    DISPOSITION_STAGE,
                    identities,
                    payload={
                        "sealed": True,
                        "seal_cid": seal_cid,
                        "published": False,
                        "canonical_mutated": False,
                    },
                )

        class _Governance:
            def acquire_lease(
                self, identities: LifecycleIdentities, repository: Path
            ) -> Mapping[str, Any]:
                return {
                    "lease_id": identities.lease_id
                    or mint_lifecycle_cid({"kind": "lease", "run_id": identities.run_id}),
                    "valid": True,
                    "receipt_cid": mint_lifecycle_cid(
                        {"kind": "lease-receipt", "run_id": identities.run_id}
                    ),
                }

            def acquire_fence(
                self, identities: LifecycleIdentities, repository: Path
            ) -> Mapping[str, Any]:
                return {
                    "fence_id": identities.fence_id
                    or mint_lifecycle_cid({"kind": "fence", "run_id": identities.run_id}),
                    "valid": True,
                    "receipt_cid": mint_lifecycle_cid(
                        {"kind": "fence-receipt", "run_id": identities.run_id}
                    ),
                }

            def admit_schedule(
                self, identities: LifecycleIdentities, repository: Path
            ) -> Mapping[str, Any]:
                return {
                    "admitted": True,
                    "status": "succeeded",
                    "receipt_cid": mint_lifecycle_cid(
                        {"kind": "schedule", "run_id": identities.run_id}
                    ),
                }

            def check_cancellation(
                self, identities: LifecycleIdentities, repository: Path
            ) -> Mapping[str, Any]:
                return {"status": "succeeded"}

        class _Persistence:
            def persist(
                self,
                artifact: StageArtifact | Mapping[str, Any],
                *,
                published: bool,
            ) -> Mapping[str, Any]:
                payload = (
                    dict(artifact.to_mapping())
                    if isinstance(artifact, StageArtifact)
                    else dict(artifact)
                )
                encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                evidence_cid = _persist_bytes(session.kit_store, encoded)
                return {"evidence_cid": evidence_cid, "published": published}

        return LifecyclePorts(
            operator=_Operator(),
            repository=_Repository(),
            semantic=_Semantic(),
            route=_Route(),
            proposal=_Proposal(),
            scope=_Scope(),
            worktree=self.worktree,
            verification=_Verification(),
            assurance=_Assurance(),
            sealing=_Sealing(),
            disposition=_Disposition(),
            governance=_Governance(),
            persistence=_Persistence(),
        )


class _EngineSurface:
    """Facade ports over the shared runtime session."""

    def __init__(self, session: RuntimeSession) -> None:
        self._session = session

    def _independent(self, operation: str, fn: Any, identities: EngineIdentities) -> EngineRecord:
        self._session.lifecycle_identities = _lifecycle_from_engine(
            identities,
            operator_id=self._session.lifecycle_identities.operator_id,
            lease_id=self._session.lifecycle_identities.lease_id,
            fence_id=self._session.lifecycle_identities.fence_id,
            worktree_id=self._session.lifecycle_identities.worktree_id,
        )
        artifact = fn(self._session.lifecycle_identities, self._session.repository)
        return self._session.record_from_artifact(operation, artifact)

    def scan(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._independent("scan", self._session.lifecycle_ports.semantic.scan, identities)

    def plan(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._independent(
            "plan", self._session.lifecycle_ports.semantic.invalidate, identities
        )

    def context_pack(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._independent(
            "context-pack", self._session.lifecycle_ports.semantic.context_pack, identities
        )

    def expand_context(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._independent(
            "expand-context", self._session.lifecycle_ports.semantic.escalate, identities
        )

    def route(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._independent("route", self._session.lifecycle_ports.route.route, identities)

    def run(
        self,
        identities: EngineIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None = None,
    ) -> EngineRecord:
        self._session.lifecycle_identities = _lifecycle_from_engine(
            identities,
            operator_id=self._session.lifecycle_identities.operator_id,
            lease_id=self._session.lifecycle_identities.lease_id,
            fence_id=self._session.lifecycle_identities.fence_id,
            worktree_id=self._session.lifecycle_identities.worktree_id,
        )
        try:
            recovery = self._session.coordinator.run(proposal)
        except ProofContextError as exc:
            status = "rejected"
            code = str(getattr(exc, "code", "") or "")
            if code == "malformed":
                status = "invalid"
            return EngineRecord(
                schema=ENGINE_RECORD_SCHEMA,
                operation="run",
                status=status,
                identities=identities,
                artifact_cid=identities.repository_state_cid,
                provenance="live",
                payload={
                    "published": False,
                    "error": code or "boundary_violation",
                    "reason": str(exc),
                },
            )
        return self._session.record_from_recovery("run", recovery)

    def verify(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        cached = self._session.cached_stage("verify")
        if cached is not None:
            return cached
        return self._independent(
            "verify", self._session.lifecycle_ports.verification.verify, identities
        )

    def assurance(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        cached = self._session.cached_stage("assurance")
        if cached is not None:
            return cached
        return self._independent(
            "assurance", self._session.lifecycle_ports.assurance.assure, identities
        )

    def seal(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        cached = self._session.cached_stage("seal")
        if cached is not None:
            return cached
        return self._independent("seal", self._session.lifecycle_ports.sealing.seal, identities)

    def status(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        cached = self._session.cached_stage("status")
        if cached is not None:
            return EngineRecord(
                schema=ENGINE_RECORD_SCHEMA,
                operation="status",
                status=cached.status,
                identities=cached.identities,
                artifact_cid=cached.artifact_cid,
                provenance=cached.provenance,
                payload=cached.payload,
            )
        return EngineRecord(
            schema=ENGINE_RECORD_SCHEMA,
            operation="status",
            status="succeeded",
            identities=identities,
            artifact_cid=identities.repository_state_cid,
            provenance="live",
            payload={"opened": True, "canonical_head": self._session.canonical_head},
        )

    def report(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        cached = self._session.cached_stage("report")
        if cached is not None:
            return cached
        return self._independent(
            "report", self._session.lifecycle_ports.disposition.decide, identities
        )

    def resume(
        self,
        identities: EngineIdentities,
        repository: Path,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> EngineRecord:
        recovery = self._session.coordinator.resume(checkpoint)
        return self._session.record_from_recovery("resume", recovery)


def _descriptor_body() -> dict[str, Any]:
    snapshot = {
        name: {
            "parameters": list(spec["parameters"]),
            "keyword_only": list(spec["keyword_only"]),
            "return": spec["return"],
        }
        for name, spec in public_signature_snapshot().items()
    }
    return {
        "schema": RUNTIME_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "interface": INTERFACE,
        "operations": list(OPERATIONS),
        "instance_operations": list(INSTANCE_OPERATIONS),
        "operation_contracts": dict(OPERATION_CONTRACTS),
        "public_signature_snapshot": snapshot,
        "modes": list(MODES),
        "live_modes": sorted(LIVE_MODES),
        "provenances": list(PROVENANCES),
        "statuses": list(STATUSES),
        "errors": list(ERRORS),
        "stages": list(STAGES),
        "stage_contracts": dict(STAGE_CONTRACTS),
        "forbidden_evidence": list(FORBIDDEN_EVIDENCE),
        "simulation_watermark": SIMULATION_WATERMARK,
        "sibling_layout_required": SIBLING_LAYOUT_REQUIRED,
        "provider_bound": PROVIDER_BOUND,
        "datasets_port": DATASETS_PORT,
        "kit_port": KIT_PORT,
        "lifecycle_cid": LIFECYCLE_CID,
        "policy_cid": POLICY_CID,
        "result_state_cid": RESULT_STATE_CID,
        "recovery_cid": RECOVERY_CID,
        "lifecycle_cid_binding": "ipfs_accelerate_py.proof_context.lifecycle.LIFECYCLE_CID",
        "policy_cid_binding": "ipfs_accelerate_py.proof_context.policy.POLICY_CID",
        "result_state_cid_binding": "ipfs_accelerate_py.proof_context.results.RESULT_STATE_CID",
        "recovery_cid_binding": "ipfs_accelerate_py.proof_context.recovery.RECOVERY_CID",
        "runtime_cid_binding": "ipfs_accelerate_py.proof_context.bootstrap.RUNTIME_CID",
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "epic_a_gate_task": EPIC_A_GATE_TASK,
        "epic_a_gate_content_id": EPIC_A_GATE_CONTENT_ID,
        "package": "ipfs_accelerate_py.proof_context",
        "bootstrap": "ipfs_accelerate_py.proof_context.bootstrap",
    }


_DESCRIPTOR_BODY: Final[dict[str, Any]] = _descriptor_body()
RUNTIME_CID: Final[str] = mint_lifecycle_cid(_DESCRIPTOR_BODY)
RUNTIME_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**_DESCRIPTOR_BODY, "cid": RUNTIME_CID}
)


def runtime_descriptor() -> Mapping[str, Any]:
    return RUNTIME_DESCRIPTOR


def open_runtime(
    repository: str | Path,
    *,
    identities: EngineIdentities | None = None,
    mode: str = "production",
    options: RuntimeOptions | None = None,
) -> RuntimeBundle:
    """Wire ports and open ``ProofCarryingContextEngine`` on a Git repository."""

    chosen = options or RuntimeOptions()
    requested_mode = "simulation" if chosen.fail_provenance == "simulated" else mode
    admitted_mode = admit_mode(requested_mode)
    root = _admit_git_repository(repository)
    datasets = resolve_datasets()
    kit = resolve_kit()
    if chosen.require_datasets:
        require_production_capability(datasets)
    if chosen.require_kit:
        require_production_capability(kit)
    kit_root = chosen.kit_root
    kit_store = None
    if kit_root is not None:
        kit_root.mkdir(parents=True, exist_ok=True)
        kit_store = _load_kit_store(kit_root)
        if kit_store is None and chosen.require_kit:
            raise UnavailableCapabilityError(
                "kit v0.1 port is unavailable",
                details={"capability": "kit"},
            )
    head = _canonical_head(root)
    if identities is None:
        state_cid = mint_lifecycle_cid(
            {"kind": "repository-state", "head": head, "ref": _canonical_ref(root)}
        )
        identities = EngineIdentities(
            repository_id="example/ordinary-python-repo",
            repository_state_cid=state_cid,
            task_id="PCCE-025",
            run_id=mint_lifecycle_cid({"kind": "run", "head": head}),
            trace_id=mint_lifecycle_cid({"kind": "trace", "head": head}),
        )
    store = FencedCheckpointStore()
    session = RuntimeSession(
        root,
        engine_identities=identities,
        mode=admitted_mode,
        options=chosen,
        datasets=datasets,
        kit=kit,
        kit_store=kit_store,
        store=store,
    )
    surface = _EngineSurface(session)
    ports = EnginePorts(
        semantic=surface,
        persistence=surface,
        route=surface,
        execution=surface,
        verification=surface,
        assurance=surface,
        sealing=surface,
        report=surface,
    )
    engine = ProofCarryingContextEngine.open(
        root,
        ports=ports,
        identities=identities,
        mode=admitted_mode,
    )
    return RuntimeBundle(
        engine=engine,
        session=session,
        datasets=datasets,
        kit=kit,
        descriptor=runtime_descriptor(),
    )


def open_engine(
    repository: str | Path,
    *,
    identities: EngineIdentities | None = None,
    mode: str = "production",
    options: RuntimeOptions | None = None,
) -> ProofCarryingContextEngine:
    return open_runtime(
        repository,
        identities=identities,
        mode=mode,
        options=options,
    ).engine


__all__ = [
    "DATASETS_PORT",
    "KIT_PORT",
    "RUNTIME_CID",
    "RUNTIME_DESCRIPTOR",
    "RUNTIME_SCHEMA",
    "RuntimeBundle",
    "RuntimeOptions",
    "RuntimeSession",
    "create_ordinary_python_repository",
    "open_engine",
    "open_runtime",
    "runtime_descriptor",
]

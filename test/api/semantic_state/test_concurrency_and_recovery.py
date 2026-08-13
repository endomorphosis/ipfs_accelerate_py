"""SCH-015 concurrency, generation CAS, ABA rejection, and safe recovery.

Covers:

* generation-bearing root CAS has exactly one winner under concurrent writers;
* ABA (stale generation token) is rejected even when content shape returns;
* interrupted publication recovers to zero or one valid root and retries safely;
* concurrent watchers coalesce and never publish roots by themselves;
* harness root conflicts are reported rather than overwritten.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
import textwrap
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    ContextPack,
    HarnessDisposition,
    HarnessMode,
    ModelRoute,
    RootRef,
    SemanticStateRootManifest,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    RootConflict,
    cid_for_root_manifest,
    open_local_durable_state,
    root_manifest_artifact,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessPolicy,
    HarnessRequest,
    SemanticCompressionHarness,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope


REPO_ROOT = Path(__file__).resolve().parents[3]
SEAL_PATH = REPO_ROOT / "config/semantic_state_dependencies.seal.json"
SEALED_COORDINATION_OID = "161c4631881607b4ed7c854f751f9fc3be0cfb45"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


# ---------------------------------------------------------------------------
# Hermetic memory durable port
# ---------------------------------------------------------------------------


class MemoryDurablePort:
    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}
        self.cas_calls: list[tuple[str, RootRef | None, str]] = []
        self.recover_calls = 0
        self._lock = threading.Lock()

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        with self._lock:
            self._objects[expected_cid] = dict(artifact)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        with self._lock:
            return json.dumps(
                self._objects[cid], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")

    def has(self, cid: str) -> bool:
        with self._lock:
            return cid in self._objects

    def read_root(self, repository_id: str) -> RootRef | None:
        with self._lock:
            return self._roots.get(repository_id)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        with self._lock:
            self.cas_calls.append((repository_id, expected, new_root_cid))
            current = self._roots.get(repository_id)
            body = dict(self._objects[new_root_cid])
            manifest = {k: v for k, v in body.items() if k != "schema"}
            disposition = manifest.get("acceptance_disposition")
            if expected is None:
                if current is not None:
                    raise RootConflict("root already exists")
                if disposition != AcceptanceDisposition.BOOTSTRAP.value:
                    raise RootConflict("initial CAS requires bootstrap disposition")
                ref = RootRef(root_cid=new_root_cid, generation=1)
                self._roots[repository_id] = ref
                return ref
            if current is None:
                raise RootConflict("expected root missing")
            if (
                current.root_cid != expected.root_cid
                or current.generation != expected.generation
            ):
                raise RootConflict("expected root token mismatch")
            if disposition != AcceptanceDisposition.ACCEPTED.value:
                raise RootConflict("only accepted manifests may advance the root")
            if current.root_cid == new_root_cid:
                return current
            ref = RootRef(root_cid=new_root_cid, generation=current.generation + 1)
            self._roots[repository_id] = ref
            return ref

    def recover(self) -> Mapping[str, Any]:
        with self._lock:
            self.recover_calls += 1
            return {"ok": True, "roots": dict(self._roots)}


# ---------------------------------------------------------------------------
# Sealed kit materialization (real DurableCoordinationStore)
# ---------------------------------------------------------------------------


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _coordination_path(root: Path) -> Path:
    return root / "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py"


def _is_sealed_kit_root(root: Path) -> bool:
    path = _coordination_path(root)
    if not path.is_file():
        return False
    return _git_blob_oid(path.read_bytes()) == SEALED_COORDINATION_OID


def _purge_ipfs_kit_modules() -> None:
    for name in list(sys.modules):
        if name == "ipfs_kit_py" or name.startswith("ipfs_kit_py."):
            del sys.modules[name]


def _sealed_kit_commit() -> str:
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    for authority in seal["authorities"]:
        if authority.get("role") == "kit_state_roots":
            return str(authority["commit"])
    raise RuntimeError("kit_state_roots authority missing from seal")


def _iter_kit_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    for key in (
        "SCH_KIT_CHECKOUT",
        "IPFS_KIT_CHECKOUT",
        "IPFS_KIT_PY_ROOT",
        "SEMANTIC_STATE_KIT_CHECKOUT",
    ):
        raw = os.environ.get(key)
        if raw:
            candidates.append(Path(raw))
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    candidates.extend(sorted(tmp.glob("kit-mat-*")))
    candidates.extend(sorted(tmp.glob("sch-seal-kit_state_roots.*/repo")))
    candidates.extend(sorted(tmp.glob("sch-003-kit-*")))
    for relative in (
        Path("ipfs_kit_py"),
        Path("../ipfs_kit_py"),
        Path("../../ipfs_kit"),
        Path("../../../external/ipfs_kit"),
        Path("/home/barberb/lift_coding/external/ipfs_kit"),
    ):
        path = relative if relative.is_absolute() else (REPO_ROOT / relative)
        candidates.append(path)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _materialize_sealed_kit(destination: Path) -> Path:
    commit = _sealed_kit_commit()
    errors: list[str] = []
    for source in _iter_kit_source_candidates():
        if _is_sealed_kit_root(source):
            return source
        if not (source / ".git").exists() and not (source / ".git").is_file():
            continue
        probe = subprocess.run(
            ["git", "-C", str(source), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        if probe.returncode != 0:
            errors.append(f"{source}: missing sealed commit")
            continue
        archived = subprocess.run(
            ["git", "-C", str(source), "archive", "--format=tar", commit],
            check=False,
            capture_output=True,
        )
        if archived.returncode != 0:
            errors.append(f"{source}: git archive failed")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
            archive.extractall(destination)
        if not _is_sealed_kit_root(destination):
            errors.append(f"{source}: archive did not match sealed coordination blob")
            continue
        return destination
    raise RuntimeError(
        "unable to materialize sealed DurableCoordinationStore; "
        "set SCH_KIT_CHECKOUT. " + "; ".join(errors[:6])
    )


@pytest.fixture(scope="module")
def sealed_kit_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("sch015-kit")
    try:
        root = _materialize_sealed_kit(dest)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return root


@pytest.fixture
def durable(sealed_kit_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _purge_ipfs_kit_modules()
    monkeypatch.syspath_prepend(str(sealed_kit_root))
    workspace_ns = str(REPO_ROOT / "ipfs_kit_py")
    sys.path[:] = [p for p in sys.path if p != workspace_ns]
    monkeypatch.syspath_prepend(str(sealed_kit_root))
    _purge_ipfs_kit_modules()
    storage = tmp_path / "durable-storage"
    with open_local_durable_state(storage, backend=None) as adapter:
        yield adapter
    _purge_ipfs_kit_modules()


def _manifest(
    *,
    disposition: str,
    label: str,
    repository_id: str = "example/repo",
) -> SemanticStateRootManifest:
    return SemanticStateRootManifest.from_dict(
        {
            "repository_id": repository_id,
            "base_tree_cid": _cid(f"{label}-base"),
            "candidate_tree_cid": _cid(f"{label}-cand"),
            "datasets_state_cid": _cid(f"{label}-datasets"),
            "datasets_semantic_state_root_cid": _cid(f"{label}-ds-root"),
            "capsule_index_cid": _cid(f"{label}-caps"),
            "delta_cid": _cid(f"{label}-delta"),
            "invalidation_cid": _cid(f"{label}-inv"),
            "obligation_set_cid": _cid(f"{label}-obl"),
            "test_selection_cid": _cid(f"{label}-sel"),
            "receipt_index_cid": _cid(f"{label}-rcpt"),
            "environment_binding_cids": [
                _cid(f"{label}-tool"),
                _cid(f"{label}-lock"),
            ],
            "event_head_cid": _cid(f"{label}-event"),
            "versions": {
                "semantic_index_schema": "ipfs-datasets.software-contracts.semantic-index@2",
                "semantic_state_schema": "ipfs-datasets.software-contracts.semantic-state@1",
                "capsule_schema": "ipfs-datasets.software-contracts.semantic-capsule@1",
                "selection_schema": "ipfs-datasets.software-contracts.semantic-test-selection@1",
            },
            "acceptance_disposition": disposition,
        }
    )


def _put_manifest(adapter: Any, manifest: SemanticStateRootManifest) -> str:
    artifact = root_manifest_artifact(manifest)
    cid = cid_for_root_manifest(manifest)
    adapter.put(artifact, expected_cid=cid)
    return cid


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _session_policy(tmp_path: Path, **overrides: object):
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    payload: dict[str, object] = {
        "repository_id": "example/repo",
        "event_log_path": tmp_path / "session-events.jsonl",
        "checkpoint_path": tmp_path / "session-cursor.json",
        "mode": HarnessMode.DEVELOPMENT.value,
        "debounce_ms": 0,
        "fence_ttl_ms": 60_000,
        "worker_enabled": False,
        "fail_closed_on_corrupt_log": True,
    }
    payload.update(overrides)
    return session_mod.SessionPolicy.from_dict(payload)


def _session(tmp_path: Path, **kwargs: Any):
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    policy = kwargs.pop("policy", None) or _session_policy(tmp_path)
    return session_mod.SemanticStateSession(policy, **kwargs)


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def _scope() -> PatchScope:
    return PatchScope.from_dict(
        {
            "allowed_paths": ("pkg/",),
            "effect_paths": ("pkg/target.py",),
            "task_owned_paths": ("pkg/",),
        }
    )


def _pack() -> ContextPack:
    return ContextPack.from_dict(
        {
            "objective": "concurrency acceptance",
            "target_source_cid": _cid("target-src"),
            "surrounding_source_cid": _cid("surround-src"),
            "test_source_cid": _cid("test-src"),
            "dependency_capsule_cids": [],
            "obligation_cids": [],
            "counterexample_cids": [],
            "delta_cid": _cid("pack-delta"),
            "interface_cids": [],
            "assumptions": [],
            "exclusions": [],
            "token_totals": {"total": 80, "target": 20},
            "estimator_version": "sch-test-estimator@1",
            "risk": RiskClass.LOW.value,
            "route": ModelRoute.DETERMINISTIC_ONLY.value,
            "escalation_recommendation": "none",
        }
    )


def _routing() -> RoutingDecision:
    inputs = RoutingInputs.from_dict(
        {
            "context_tokens": 100,
            "lowest_confidence": ConfidenceClass.EXACT.value,
            "risk": RiskClass.LOW.value,
            "dependency_cone_size": 1,
            "unresolved_obligations": 0,
            "prior_repair_failures": 0,
            "available_proofs": 1,
            "prior_route_failed": False,
        }
    )
    return RoutingDecision(
        route=ModelRoute.DETERMINISTIC_ONLY.value,
        reason_codes=("deterministic_only",),
        explanation="deterministic only",
        requires_provider=False,
        halt_before_dispatch=True,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=ModelRoutingPolicy.default(),
    )


def _simple_patch(*, new: str = "VALUE = 2") -> str:
    return textwrap.dedent(
        f"""\
        diff --git a/pkg/target.py b/pkg/target.py
        --- a/pkg/target.py
        +++ b/pkg/target.py
        @@ -1 +1 @@
        -VALUE = 1
        +{new}
        """
    )


def _env_cids() -> dict[str, str]:
    return {
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
    }


def _bootstrap(harness: SemanticCompressionHarness, repository_id: str = "repo:cas") -> RootRef:
    env = _env_cids()
    outcome = harness.bootstrap_scan(
        HarnessRequest(
            repository_id=repository_id,
            task_id="task-boot",
            objective="bootstrap",
            scope=_scope(),
            context_pack=_pack(),
            bootstrap_tree_cid=_cid("base-tree"),
            **env,
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    return outcome.result.current_root


def _accept_request(
    root: RootRef,
    *,
    repository_id: str = "repo:cas",
    attempt_key: str = "attempt-1",
    patch_text: str | None = None,
) -> HarnessRequest:
    env = _env_cids()
    return HarnessRequest(
        repository_id=repository_id,
        task_id="task-patch",
        objective="apply safe patch",
        scope=_scope(),
        expected_root=root,
        context_pack=_pack(),
        routing_decision=_routing(),
        patch_text=patch_text if patch_text is not None else _simple_patch(),
        base_tree=_cid("base-tree"),
        changed_symbol_ids=("pkg.target.VALUE",),
        obligation_cids=(_cid("obligation-a"),),
        visible_sources={"pkg/target.py": "VALUE = 1\n"},
        attempt_key=attempt_key,
        **env,
    )


# ---------------------------------------------------------------------------
# Generation CAS — one winner, ABA rejected
# ---------------------------------------------------------------------------


def test_root_cas_and_recovery(durable: Any) -> None:
    """Generation CAS: one concurrent winner, ABA rejected, recover is safe."""

    repo = "example/repo"
    boot = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.BOOTSTRAP.value, label="cas-boot"),
    )
    base = durable.compare_and_swap_root(repo, None, boot)
    assert base.generation == 1

    left = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="cas-left"),
    )
    right = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="cas-right"),
    )

    results: list[object] = []
    barrier = threading.Barrier(2)

    def writer(new_cid: str) -> None:
        barrier.wait()
        try:
            results.append(durable.compare_and_swap_root(repo, base, new_cid))
        except Exception as exc:  # noqa: BLE001 - collect race outcomes
            results.append(exc)

    threads = [
        threading.Thread(target=writer, args=(left,)),
        threading.Thread(target=writer, args=(right,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if not isinstance(item, Exception)]
    conflicts = [item for item in results if isinstance(item, RootConflict)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    winner = successes[0]
    assert winner.generation == 2
    assert winner.root_cid in {left, right}
    assert durable.read_root(repo) == winner

    # ABA: advance further, then attempt stale generation-1 write.
    next_acc = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="cas-next"),
    )
    gen3 = durable.compare_and_swap_root(repo, winner, next_acc)
    assert gen3.generation == 3

    stale = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="cas-stale"),
    )
    with pytest.raises(RootConflict):
        durable.compare_and_swap_root(repo, base, stale)
    assert durable.read_root(repo) == gen3

    # recover() is safe and does not drop the current root.
    recovered = durable.recover()
    assert isinstance(recovered, Mapping)
    assert durable.read_root(repo) == gen3


def test_aba_stale_writer_is_rejected_on_memory_port() -> None:
    """Memory CAS also rejects ABA: generation token is authoritative."""

    port = MemoryDurablePort()
    repo = "repo:aba"

    def store(label: str, disposition: str) -> str:
        manifest = _manifest(disposition=disposition, label=label, repository_id=repo)
        body = manifest.to_dict()
        body["schema"] = "ipfs-accelerate.semantic-state-root-manifest@1"
        cid = _cid(f"manifest-{label}")
        port.put(body, expected_cid=cid)
        return cid

    a1 = store("aba-a1", AcceptanceDisposition.BOOTSTRAP.value)
    ref_a = port.compare_and_swap_root(repo, None, a1)
    b = store("aba-b", AcceptanceDisposition.ACCEPTED.value)
    ref_b = port.compare_and_swap_root(repo, ref_a, b)
    a2 = store("aba-a2", AcceptanceDisposition.ACCEPTED.value)
    ref_a2 = port.compare_and_swap_root(repo, ref_b, a2)
    assert ref_a2.generation == 3

    stale = store("aba-stale", AcceptanceDisposition.ACCEPTED.value)
    with pytest.raises(RootConflict):
        port.compare_and_swap_root(repo, ref_a, stale)
    assert port.read_root(repo) == ref_a2


def test_two_memory_writers_yield_exactly_one_success() -> None:
    port = MemoryDurablePort()
    repo = "repo:race"

    def store(label: str, disposition: str) -> str:
        manifest = _manifest(disposition=disposition, label=label, repository_id=repo)
        body = manifest.to_dict()
        body["schema"] = "ipfs-accelerate.semantic-state-root-manifest@1"
        cid = _cid(f"manifest-{label}")
        port.put(body, expected_cid=cid)
        return cid

    boot = store("race-boot", AcceptanceDisposition.BOOTSTRAP.value)
    base = port.compare_and_swap_root(repo, None, boot)
    left = store("race-left", AcceptanceDisposition.ACCEPTED.value)
    right = store("race-right", AcceptanceDisposition.ACCEPTED.value)

    results: list[object] = []
    barrier = threading.Barrier(2)

    def writer(cid: str) -> None:
        barrier.wait()
        try:
            results.append(port.compare_and_swap_root(repo, base, cid))
        except Exception as exc:  # noqa: BLE001
            results.append(exc)

    threads = [
        threading.Thread(target=writer, args=(left,)),
        threading.Thread(target=writer, args=(right,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if not isinstance(item, Exception)]
    conflicts = [item for item in results if isinstance(item, RootConflict)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert port.read_root(repo) == successes[0]


# ---------------------------------------------------------------------------
# Interrupted publication recovery (real kit)
# ---------------------------------------------------------------------------


def test_interrupted_publication_recovers_safely(
    sealed_kit_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class InjectedInterruption(RuntimeError):
        pass

    _purge_ipfs_kit_modules()
    monkeypatch.syspath_prepend(str(sealed_kit_root))
    sys.path[:] = [p for p in sys.path if p != str(REPO_ROOT / "ipfs_kit_py")]
    monkeypatch.syspath_prepend(str(sealed_kit_root))
    _purge_ipfs_kit_modules()

    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        ROOT_CAS_INTERRUPTION_POINTS,
    )

    repo = "example/repo"
    for boundary in ROOT_CAS_INTERRUPTION_POINTS:
        storage = tmp_path / f"interrupt-{boundary}"
        bootstrap = _manifest(
            disposition=AcceptanceDisposition.BOOTSTRAP.value,
            label=f"i-{boundary}",
        )
        artifact = root_manifest_artifact(bootstrap)
        cid = cid_for_root_manifest(bootstrap)

        def interrupt(point: str, *, _boundary: str = boundary) -> None:
            if point == _boundary:
                raise InjectedInterruption(point)

        with open_local_durable_state(
            storage, backend=None, crash_injector=interrupt
        ) as adapter:
            adapter.put(artifact, expected_cid=cid)
            with pytest.raises(InjectedInterruption):
                adapter.compare_and_swap_root(repo, None, cid)

        # Reopen: zero or one valid root; retry is always safe.
        with open_local_durable_state(storage, backend=None) as recovered:
            current = recovered.read_root(repo)
            if boundary in {"before_transaction", "after_expectation_verification"}:
                assert current is None
            else:
                assert current is not None
                assert current.root_cid == cid
                assert current.generation == 1

            final = recovered.compare_and_swap_root(repo, None, cid)
            assert final.root_cid == cid
            assert final.generation == 1
            assert recovered.read_root(repo) == final

    _purge_ipfs_kit_modules()


# ---------------------------------------------------------------------------
# Concurrent watchers / session recovery
# ---------------------------------------------------------------------------


class SessionMemoryDurablePort:
    """Generation CAS without requiring pre-stored root-manifest bodies."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}
        self.cas_calls: list[tuple[str, RootRef | None, str]] = []
        self.recover_calls = 0

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        self._objects[expected_cid] = dict(artifact)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        return json.dumps(
            self._objects[cid], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def has(self, cid: str) -> bool:
        return cid in self._objects

    def read_root(self, repository_id: str) -> RootRef | None:
        return self._roots.get(repository_id)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        self.cas_calls.append((repository_id, expected, new_root_cid))
        current = self._roots.get(repository_id)
        if expected is None:
            if current is not None:
                raise RootConflict("expected empty root")
            published = RootRef(root_cid=new_root_cid, generation=1)
        else:
            if (
                current is None
                or current.root_cid != expected.root_cid
                or current.generation != expected.generation
            ):
                raise RootConflict("stale expected root")
            published = RootRef(
                root_cid=new_root_cid, generation=expected.generation + 1
            )
        self._roots[repository_id] = published
        return published

    def recover(self) -> Mapping[str, Any]:
        self.recover_calls += 1
        return {"recovered": True, "roots": len(self._roots)}


def test_concurrent_watchers_coalesce_and_do_not_publish(tmp_path: Path) -> None:
    durable = SessionMemoryDurablePort()
    calls: list[str] = []
    barrier = threading.Barrier(4)
    snap = _cid("snap-equal")

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs["snapshot_cid"])
        return {
            "status": "completed",
            "output_artifact_cids": [_cid("scan-out")],
            "verified": False,
        }

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    acks: list[Any] = []
    errors: list[BaseException] = []

    def notify() -> None:
        try:
            barrier.wait(timeout=2)
            acks.append(session.notify_watch(snap, source="watcher"))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=notify) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert errors == []
    scheduled = [ack for ack in acks if ack.scheduled]
    coalesced = [ack for ack in acks if ack.coalesced]
    assert len(scheduled) == 1
    assert len(coalesced) == 3
    results = session.drain()
    assert len(results) == 1
    assert calls == [snap]
    # Watch path alone never publishes a root.
    assert durable.read_root("example/repo") is None
    assert durable.cas_calls == []


def test_session_restart_preserves_accepted_without_unverified_publish(
    tmp_path: Path,
) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    durable = SessionMemoryDurablePort()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {
            "status": "completed",
            "new_root_cid": _cid("root-verified"),
            "verified": True,
            "output_artifact_cids": [_cid("out")],
        }

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-1"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    published = session.accept_transition(
        attempt_id=result.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=result.new_root_cid or _cid("root-verified"),
        expected=None,
        verified=True,
    )
    assert published.generation == 1
    assert durable.read_root("example/repo") == published

    # Journal an unverified candidate after acceptance.
    session.notify_watch(_cid("snap-unverified"))
    cand = session.drain()[0]
    cand_fence = session.fence_for(cand.attempt_id)
    assert cand_fence is not None
    with pytest.raises(session_mod.SessionRootPublishDenied):
        session.accept_transition(
            attempt_id=cand.attempt_id,
            fencing_token=cand_fence.fencing_token,
            new_root_cid=_cid("root-unverified"),
            expected=published,
            verified=False,
        )

    # Process restart over the same durable log preserves the accepted root.
    restarted = _session(
        tmp_path,
        durable_port=durable,
        scan_executor=executor,
        policy=_session_policy(
            tmp_path,
            event_log_path=session.policy.event_log_path,
            checkpoint_path=session.policy.checkpoint_path,
        ),
    )
    status = restarted.restart()
    assert status.current_root is not None
    assert status.current_root.root_cid == published.root_cid
    assert status.current_root.generation == published.generation
    assert durable.read_root("example/repo") == published
    assert durable.read_root("example/repo").root_cid != _cid("root-unverified")
    assert durable.recover_calls >= 1
    restarted.shutdown()


# ---------------------------------------------------------------------------
# Harness root conflict under concurrent logical writers
# ---------------------------------------------------------------------------


def test_harness_root_conflict_is_reported_not_overwritten() -> None:
    port = MemoryDurablePort()
    harness = SemanticCompressionHarness(
        durable=port,
        policy=HarnessPolicy(
            mode=HarnessMode.DEVELOPMENT.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)

    peer = SemanticCompressionHarness(
        durable=port,
        policy=HarnessPolicy(
            mode=HarnessMode.DEVELOPMENT.value,
            use_kit_root_cid=False,
        ),
    )
    peer_out = peer.run(_accept_request(root, attempt_key="peer-writer"))
    assert peer_out.result.disposition == HarnessDisposition.ACCEPTED.value
    advanced = port.read_root("repo:cas")
    assert advanced is not None
    assert advanced.generation == 2

    stale = harness.run(
        _accept_request(
            root,
            attempt_key="stale-writer",
            patch_text=_simple_patch(new="VALUE = 3"),
        )
    )
    assert stale.result.disposition == HarnessDisposition.REJECTED.value
    assert stale.root_conflict is True or "root_conflict" in stale.result.reasons
    assert port.read_root("repo:cas") == advanced
    assert stale.result.current_root == advanced


def test_recover_on_memory_port_is_idempotent() -> None:
    port = MemoryDurablePort()
    first = port.recover()
    second = port.recover()
    assert first["ok"] is True
    assert second["ok"] is True
    assert port.recover_calls == 2
    assert port.read_root("any") is None

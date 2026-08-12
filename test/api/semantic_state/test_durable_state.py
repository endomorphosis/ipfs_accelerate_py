"""SCH-003 hermetic durable-root adapter tests (local storage_dir, no daemon)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SEAL_PATH = REPO_ROOT / "config/semantic_state_dependencies.seal.json"
SEALED_COORDINATION_OID = "161c4631881607b4ed7c854f751f9fc3be0cfb45"


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


def _kit_importable() -> bool:
    try:
        from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (  # noqa: F401
            DurableCoordinationStore,
        )
    except ImportError:
        return False
    imported = Path(
        sys.modules["ipfs_kit_py.mcp_server.mcplusplus.coordination_storage"].__file__
    )
    return _git_blob_oid(imported.read_bytes()) == SEALED_COORDINATION_OID


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
    # Materializations left by prior seal validation / agent sessions.
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    candidates.extend(sorted(tmp.glob("kit-mat-*")))
    candidates.extend(sorted(tmp.glob("sch-seal-kit_state_roots.*/repo")))
    candidates.extend(sorted(tmp.glob("sch-003-kit-*")))
    # Common local checkouts near this repository tree.
    for relative in (
        Path("ipfs_kit_py"),
        Path("../ipfs_kit_py"),
        Path("../../ipfs_kit"),
        Path("../../../external/ipfs_kit"),
        Path("/home/barberb/lift_coding/external/ipfs_kit"),
        Path(
            "/home/barberb/lift_coding/data/agent_supervisor/"
            "ipfs_kit_semantic_state_roots/run-compatible/worktrees"
        ),
    ):
        path = relative if relative.is_absolute() else (REPO_ROOT / relative)
        candidates.append(path)
        if path.is_dir() and path.name == "worktrees":
            candidates.extend(sorted(path.glob("workspace-*"))[:8])
    # De-duplicate while preserving order.
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
        # Already a sealed tree (materialized archive or checkout).
        if _is_sealed_kit_root(source):
            return source
        # Git checkout that contains the sealed commit.
        if not (source / ".git").exists() and not (source / ".git").is_file():
            # Nested package root without .git: skip unless sealed files present.
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
        "unable to materialize sealed DurableCoordinationStore for hermetic tests; "
        "set SCH_KIT_CHECKOUT to the sealed kit pin. "
        + "; ".join(errors[:6])
    )


def _ensure_sealed_kit_on_path() -> None:
    """Make the SCH-000 sealed kit importable without a daemon or network."""

    if _kit_importable():
        return
    _purge_ipfs_kit_modules()
    cached = os.environ.get("SCH_003_MATERIALIZED_KIT")
    if cached and _is_sealed_kit_root(Path(cached)):
        root = Path(cached)
    else:
        destination = Path(tempfile.mkdtemp(prefix="sch-003-kit-"))
        root = _materialize_sealed_kit(destination)
        os.environ["SCH_003_MATERIALIZED_KIT"] = str(root)
    # Prefer the sealed root ahead of empty worktree submodule placeholders.
    worktree_placeholder = str((REPO_ROOT / "ipfs_kit_py").resolve())
    sys.path = [
        entry
        for entry in sys.path
        if entry.rstrip("/") not in {str(root), worktree_placeholder}
        and entry.rstrip("/") != str(REPO_ROOT / "ipfs_kit_py")
    ]
    sys.path.insert(0, str(root))
    _purge_ipfs_kit_modules()
    if not _kit_importable():
        raise RuntimeError(f"sealed kit materialized at {root} but import still fails")


_ensure_sealed_kit_on_path()

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    SemanticStateRootManifest,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableSemanticStatePort,
    DurableStateIntegrityError,
    IpfsKitDurableStateAdapter,
    RootConflict,
    cid_for_root_manifest,
    open_local_durable_state,
    root_manifest_artifact,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _manifest(
    repository_id: str = "example/repo",
    *,
    disposition: str = AcceptanceDisposition.BOOTSTRAP.value,
    label: str = "default",
) -> SemanticStateRootManifest:
    payload = {
        "repository_id": repository_id,
        "base_tree_cid": _cid(f"{label}-base-tree"),
        "candidate_tree_cid": _cid(f"{label}-candidate-tree"),
        "datasets_state_cid": _cid(f"{label}-datasets-state"),
        "datasets_semantic_state_root_cid": _cid(f"{label}-datasets-root"),
        "capsule_index_cid": _cid(f"{label}-capsule-index"),
        "delta_cid": _cid(f"{label}-delta"),
        "invalidation_cid": _cid(f"{label}-invalidation"),
        "obligation_set_cid": _cid(f"{label}-obligations"),
        "test_selection_cid": _cid(f"{label}-selection"),
        "receipt_index_cid": _cid(f"{label}-receipts"),
        "environment_binding_cids": [_cid(f"{label}-env-a"), _cid(f"{label}-env-b")],
        "event_head_cid": _cid(f"{label}-event-head"),
        "versions": {
            "capsule_schema": "ipfs-datasets.software-contracts.semantic-capsule@1",
            "selection_schema": "ipfs-datasets.software-contracts.semantic-test-selection@1",
            "semantic_index_schema": "ipfs-datasets.software-contracts.semantic-index@2",
            "semantic_state_schema": "ipfs-datasets.software-contracts.semantic-state@1",
        },
        "acceptance_disposition": disposition,
    }
    return SemanticStateRootManifest.from_dict(payload)


def _put_manifest(adapter: IpfsKitDurableStateAdapter, manifest: SemanticStateRootManifest) -> str:
    artifact = root_manifest_artifact(manifest)
    cid = cid_for_root_manifest(manifest)
    written = adapter.put(artifact, expected_cid=cid)
    assert written["cid"] == cid
    return cid


@pytest.fixture
def durable(tmp_path: Path) -> IpfsKitDurableStateAdapter:
    """Hermetic adapter: explicit temp directory, backend=None, no daemon."""

    storage = tmp_path / "coordination"
    with open_local_durable_state(storage, backend=None) as adapter:
        assert isinstance(adapter, DurableSemanticStatePort)
        yield adapter


def test_open_local_uses_explicit_temp_storage_and_no_daemon(tmp_path: Path) -> None:
    storage = tmp_path / "hermetic-store"
    with open_local_durable_state(storage, backend=None) as adapter:
        assert adapter.store.backend is None
        assert adapter.store.root == storage
        assert isinstance(adapter, IpfsKitDurableStateAdapter)
        assert adapter.read_root("example/repo") is None


def test_authoritative_cid_must_match_bytes(durable: IpfsKitDurableStateAdapter) -> None:
    manifest = _manifest(label="cid-match")
    artifact = root_manifest_artifact(manifest)
    correct = cid_for_root_manifest(manifest)
    wrong = cid_for_root_manifest(_manifest(label="other"))
    with pytest.raises(DurableStateIntegrityError, match="does not match expected"):
        durable.put(artifact, expected_cid=wrong)
    written = durable.put(artifact, expected_cid=correct)
    assert written["cid"] == correct
    assert durable.has(correct)
    assert durable.get(correct)["repository_id"] == "example/repo"
    assert durable.get_bytes(correct)


def test_only_stored_valid_manifest_may_be_published(
    durable: IpfsKitDurableStateAdapter,
) -> None:
    repo = "example/repo"
    absent = _cid("not-stored")
    with pytest.raises(DurableStateIntegrityError):
        durable.compare_and_swap_root(repo, None, absent)

    # Non-manifest object may be stored but cannot become the current root.
    non_manifest = {
        "schema": "example/not-a-root@1",
        "label": "nope",
    }
    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import cid_for_artifact

    non_cid = cid_for_artifact(non_manifest)
    durable.put(non_manifest, expected_cid=non_cid)
    with pytest.raises(
        DurableStateIntegrityError,
        match="SemanticStateRootManifest|schema must be",
    ):
        durable.compare_and_swap_root(repo, None, non_cid)

    # Candidate is storable but not publishable as the accepted current root.
    candidate = _manifest(disposition=AcceptanceDisposition.CANDIDATE.value, label="cand")
    cand_cid = _put_manifest(durable, candidate)
    with pytest.raises(DurableStateIntegrityError, match="bootstrap"):
        durable.compare_and_swap_root(repo, None, cand_cid)


def test_none_to_bootstrap_is_explicit(durable: IpfsKitDurableStateAdapter) -> None:
    repo = "example/repo"
    accepted = _manifest(
        disposition=AcceptanceDisposition.ACCEPTED.value, label="too-early"
    )
    accepted_cid = _put_manifest(durable, accepted)
    with pytest.raises(DurableStateIntegrityError, match="bootstrap"):
        durable.compare_and_swap_root(repo, None, accepted_cid)

    bootstrap = _manifest(
        disposition=AcceptanceDisposition.BOOTSTRAP.value, label="boot"
    )
    boot_cid = _put_manifest(durable, bootstrap)
    root = durable.compare_and_swap_root(repo, None, boot_cid)
    assert root.root_cid == boot_cid
    assert root.generation == 1
    assert durable.read_root(repo) == root


def test_accepted_successor_requires_prior_root(
    durable: IpfsKitDurableStateAdapter,
) -> None:
    repo = "example/repo"
    boot = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.BOOTSTRAP.value, label="boot2"),
    )
    current = durable.compare_and_swap_root(repo, None, boot)
    accepted = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="acc"),
    )
    next_root = durable.compare_and_swap_root(repo, current, accepted)
    assert next_root.generation == 2
    assert next_root.root_cid == accepted
    assert durable.read_root(repo) == next_root


def test_interrupted_publication_retains_or_completes_one_valid_root(
    tmp_path: Path,
) -> None:
    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        ROOT_CAS_INTERRUPTION_POINTS,
    )

    class InjectedInterruption(RuntimeError):
        pass

    repo = "example/repo"
    for boundary in ROOT_CAS_INTERRUPTION_POINTS:
        storage = tmp_path / f"interrupt-{boundary}"
        bootstrap = _manifest(
            disposition=AcceptanceDisposition.BOOTSTRAP.value, label=f"i-{boundary}"
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

        # Reopen without injector: recovery completes one valid root or retains none.
        with open_local_durable_state(storage, backend=None) as recovered:
            current = recovered.read_root(repo)
            if boundary in {"before_transaction", "after_expectation_verification"}:
                assert current is None
            else:
                assert current is not None
                assert current.root_cid == cid
                assert current.generation == 1

            # Retrying the same logical CAS is safe either way.
            final = recovered.compare_and_swap_root(repo, None, cid)
            assert final.root_cid == cid
            assert final.generation == 1
            assert recovered.read_root(repo) == final


def test_corrupted_blocks_fail_closed(tmp_path: Path) -> None:
    storage = tmp_path / "corrupt"
    repo = "example/repo"
    with open_local_durable_state(storage, backend=None) as adapter:
        boot = _put_manifest(
            adapter,
            _manifest(disposition=AcceptanceDisposition.BOOTSTRAP.value, label="c-boot"),
        )
        adapter.compare_and_swap_root(repo, None, boot)
        # Corrupt the published root block on disk.
        path = adapter.store._block_path(boot)
        path.write_bytes(b"tampered-not-json")

    with pytest.raises(DurableStateIntegrityError):
        with open_local_durable_state(storage, backend=None) as recovered:
            recovered.read_root(repo)

    # Fresh store with a present but corrupt linked block fails transitive check.
    storage2 = tmp_path / "corrupt-link"
    with open_local_durable_state(storage2, backend=None) as adapter:
        manifest = _manifest(
            disposition=AcceptanceDisposition.BOOTSTRAP.value, label="link"
        )
        # Plant a corrupt local block for a linked CID, then store the manifest.
        linked = manifest.delta_cid
        adapter.store._write_block(linked, b"not-the-real-linked-bytes")
        root_cid = _put_manifest(adapter, manifest)
        with pytest.raises(DurableStateIntegrityError, match="transitively corrupt|integrity|match"):
            # has(linked) is true via local path; get_bytes must re-verify CID.
            # Kit get_bytes raises when bytes do not rehash to the CID.
            adapter.compare_and_swap_root(repo, None, root_cid)


def test_two_writers_from_one_expected_token_yield_at_most_one_success(
    durable: IpfsKitDurableStateAdapter,
) -> None:
    repo = "example/repo"
    boot = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.BOOTSTRAP.value, label="race-boot"),
    )
    base = durable.compare_and_swap_root(repo, None, boot)

    left = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="race-left"),
    )
    right = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="race-right"),
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


def test_aba_stale_writer_is_rejected(durable: IpfsKitDurableStateAdapter) -> None:
    repo = "example/repo"
    a1 = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.BOOTSTRAP.value, label="aba-a1"),
    )
    ref_a = durable.compare_and_swap_root(repo, None, a1)

    b = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="aba-b"),
    )
    ref_b = durable.compare_and_swap_root(repo, ref_a, b)
    assert ref_b.generation == 2

    # Content returns to an A-shaped accepted manifest under a new generation.
    a2 = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="aba-a2"),
    )
    ref_a2 = durable.compare_and_swap_root(repo, ref_b, a2)
    assert ref_a2.generation == 3
    assert ref_a2.root_cid == a2

    stale = _put_manifest(
        durable,
        _manifest(disposition=AcceptanceDisposition.ACCEPTED.value, label="aba-stale"),
    )
    # Writer still holding the generation-1 RootRef must lose despite A-like content.
    with pytest.raises(RootConflict):
        durable.compare_and_swap_root(repo, ref_a, stale)
    assert durable.read_root(repo) == ref_a2


def test_repository_id_mismatch_fails_closed(durable: IpfsKitDurableStateAdapter) -> None:
    foreign = _put_manifest(
        durable,
        _manifest(
            repository_id="other/repo",
            disposition=AcceptanceDisposition.BOOTSTRAP.value,
            label="foreign",
        ),
    )
    with pytest.raises(DurableStateIntegrityError, match="repository_id"):
        durable.compare_and_swap_root("example/repo", None, foreign)

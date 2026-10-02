"""Native exact-location cache ownership, without checker/model execution.

The placement fixture changes only tempfile's classification root, because
pytest's ordinary /tmp paths intentionally bypass production relocation.
It uses an actual Git checkout and actual DuckDB storage throughout.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state


@pytest.fixture
def placement(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
    other_temp = tmp_path / "classification-only-temporary-root"
    other_temp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(other_temp))
    orchestration = tmp_path / "parent-orchestration"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(orchestration))
    return checkout, orchestration, other_temp


def _write_marker(cache, value="retained"):
    with cache._connect() as cx:
        cx.execute("CREATE TABLE exact_path_marker (value VARCHAR)")
        cx.execute("INSERT INTO exact_path_marker VALUES (?)", [value])


def _read_marker(cache):
    with cache._connect() as cx:
        return cx.execute("SELECT value FROM exact_path_marker").fetchone()[0]


def test_exact_path_survives_reopen_while_default_still_relocates(placement, monkeypatch):
    checkout, orchestration, _ = placement
    target = checkout / "owned" / "proof.duckdb"
    exact = FormalVerificationCache(target, exact_path=True)
    _write_marker(exact)
    inode = target.stat().st_ino
    assert exact.path == target and exact._legacy_path is None
    assert not orchestration.exists()
    changed = checkout.parent / "child-orchestration"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(changed))
    reopened = FormalVerificationCache(target, exact_path=True)
    assert reopened.path == target and target.stat().st_ino == inode
    assert _read_marker(reopened) == "retained"
    assert not changed.exists()
    default_target = checkout / "legacy-default.duckdb"
    default = FormalVerificationCache(default_target)
    assert default.path.is_relative_to(changed)
    assert default.path.is_file() and not default_target.exists()


def test_exact_reopen_in_fresh_process_with_changed_orchestration(placement):
    checkout, _, classification_root = placement
    target = checkout / "proof.duckdb"
    cache = FormalVerificationCache(target, exact_path=True)
    _write_marker(cache, "cold-reopen")
    inode = target.stat().st_ino
    changed = checkout.parent / "fresh-child-orchestration"
    code = """
import json,sys,tempfile
from pathlib import Path
tempfile.gettempdir=lambda:sys.argv[3]
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
path=Path(sys.argv[1]);cache=FormalVerificationCache(path,exact_path=True)
with cache._connect() as cx:
    assert cx.execute('SELECT value FROM exact_path_marker').fetchone()[0]=='cold-reopen'
assert cache.path==path and path.stat().st_ino==int(sys.argv[2])
print(json.dumps({'path':str(cache.path),'same_inode':True,'marker_retained':True}))
"""
    env = {**os.environ, "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR": str(changed)}
    child = subprocess.run([sys.executable, "-c", code, str(target), str(inode), str(classification_root)],
        env=env, text=True, capture_output=True, timeout=60, check=True)
    assert json.loads(child.stdout.splitlines()[-1]) == {
        "path": str(target), "same_inode": True, "marker_retained": True}
    assert not changed.exists()


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], {}])
def test_exact_option_requires_real_bool_before_storage_write(tmp_path, flag):
    target = tmp_path / "not-created" / "proof.duckdb"
    with pytest.raises(ValueError, match="boolean"):
        FormalVerificationCache(target, exact_path=flag)
    assert not target.parent.exists()


@pytest.mark.parametrize("kind", [
    "none", "relative", "directory", "no_suffix", "sqlite_suffix", "upper_suffix", "uri",
    "dot", "parent_dot", "double_separator", "target_symlink", "broken_symlink", "ancestor_symlink",
])
def test_invalid_exact_locations_refuse_before_storage_write(tmp_path, kind, monkeypatch):
    root = tmp_path / "case"
    root.mkdir()
    target = root / "new-parent" / "proof.duckdb"
    if kind == "none":
        target = None
    elif kind == "relative":
        monkeypatch.chdir(root)
        target = "new-parent/proof.duckdb"
    elif kind == "directory":
        target = root / "directory.duckdb"
        target.mkdir()
    elif kind == "no_suffix":
        target = root / "new-parent" / "proof"
    elif kind == "sqlite_suffix":
        target = root / "new-parent" / "proof.sqlite3"
    elif kind == "upper_suffix":
        target = root / "new-parent" / "proof.DUCKDB"
    elif kind == "uri":
        target = "quack://127.0.0.1:31415/proof.duckdb"
    elif kind == "dot":
        target = str(root) + "/./new-parent/proof.duckdb"
    elif kind == "parent_dot":
        target = str(root) + "/new-parent/../proof.duckdb"
    elif kind == "double_separator":
        target = str(root) + "//new-parent/proof.duckdb"
    elif kind in ("target_symlink", "broken_symlink"):
        destination = root / "destination"
        if kind == "target_symlink":
            destination.write_bytes(b"unchanged")
        target = root / "proof.duckdb"
        target.symlink_to(destination)
    elif kind == "ancestor_symlink":
        destination = root / "destination"
        destination.mkdir()
        (root / "alias").symlink_to(destination, target_is_directory=True)
        target = root / "alias" / "proof.duckdb"
    def snapshot():
        return {str(p.relative_to(root)): (p.is_symlink(),
            os.readlink(p) if p.is_symlink() else p.read_bytes() if p.is_file() else None)
            for p in root.rglob("*")}
    before = snapshot()
    with pytest.raises(ValueError, match="canonical absolute"):
        FormalVerificationCache(target, exact_path=True)
    assert snapshot() == before


def test_exact_path_does_not_probe_or_import_legacy_sibling(placement, monkeypatch):
    checkout, _, _ = placement
    target = checkout / "proof.duckdb"
    legacy = target.with_suffix(".sqlite3")
    with sqlite3.connect(legacy) as cx:
        cx.execute("CREATE TABLE proof_cache_entries (sentinel TEXT)")
        cx.execute("INSERT INTO proof_cache_entries VALUES ('do-not-import')")
    before = legacy.read_bytes()
    native_probe = duckdb_state.is_sqlite_database
    def probe(path):
        assert Path(path) != legacy, "exact mode must not even probe the legacy sibling"
        return native_probe(path)
    monkeypatch.setattr(duckdb_state, "is_sqlite_database", probe)
    monkeypatch.setenv(duckdb_state.DUCKDB_ONLY_ENV, "0")
    cache = FormalVerificationCache(target, exact_path=True)
    assert cache._legacy_path is None and legacy.read_bytes() == before
    with cache._connect() as cx:
        assert cx.execute("SELECT count(*) FROM proof_cache_entries").fetchone()[0] == 0
        assert cx.execute("SELECT count(*) FROM agent_supervisor_store_metadata WHERE key LIKE 'sqlite_migration:%'").fetchone()[0] == 0


def _bound_native_roots(checkout):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as gate
    from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
    source = checkout / "source.duckdb"
    artifacts = checkout / "artifacts"
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "64MB"}) as cx:
        store = gate.DuckDBASTStore(connection=cx)
        cas = gate.ImmutableCAS(artifacts)
        index = gate.RepositoryCodebaseIndex(ingestor=gate.DuckDBASTIngestor(store=store),
            artifacts=cas, catalog=gate.CodebaseCatalog(store, cas))
        catalog = gate.IntentCodebaseCatalog(index)
        cache = FiniteCheckedCache(FormalVerificationCache(checkout / "proof.duckdb", exact_path=True), cas)
        _write_marker(cache.cache, "bound-owner")
        source.chmod(0o600)
        artifacts.chmod(0o700)
        cache.cache.path.chmod(0o600)
        roots = gate._roots(catalog, cache)
    return roots


def test_native_behavioral_owner_reopens_bound_exact_path_after_environment_change(placement, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as gate
    checkout, orchestration, _ = placement
    roots = _bound_native_roots(checkout)
    changed = checkout.parent / "runtime-state" / "orchestration"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(changed))
    with gate._owners(deepcopy(roots)) as (catalog, cache):
        assert gate._roots(catalog, cache) == roots
        assert _read_marker(cache.cache) == "bound-owner"
        assert cache.cache.path == checkout / "proof.duckdb"
    assert not changed.exists() and not orchestration.exists()


@pytest.mark.parametrize("damage", ["missing", "replaced", "symlink"])
def test_native_behavioral_owner_refuses_missing_or_changed_proof_identity(placement, damage):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as gate
    checkout, _, _ = placement
    roots = _bound_native_roots(checkout)
    proof = Path(roots["proof_database"]["path"])
    before = proof.read_bytes()
    retained = proof.with_suffix(".retained")
    proof.rename(retained)
    if damage == "replaced":
        proof.write_bytes(before)
        proof.chmod(0o600)
    elif damage == "symlink":
        proof.symlink_to(retained)
    with pytest.raises((ValueError, FileNotFoundError)):
        with gate._owners(roots):
            pytest.fail("changed exact owner must not yield")
    assert retained.read_bytes() == before
    assert not proof.exists() if damage == "missing" else proof.read_bytes() == before

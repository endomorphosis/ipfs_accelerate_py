"""Real datasets SQL/CAS refresh, source fences and bounded query behavior."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_view as view
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_revision import revise_repository_view
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import (
    NativeRepositoryIndex, RepositorySourceIndexError,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


@pytest.fixture
def scheduler(tmp_path):
    return GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json",
        proof_resource_sampler=lambda: ProofHostResources(8, 16384, 16384),
        lane_reservations={}, auto_renew_leases=False,
        proof_backoff_seconds=0, poll_interval_seconds=0.005))


def snapshot(tmp_path, name="initial", files=None):
    source = tmp_path / (name + "-source")
    source.mkdir()
    files = files if files is not None else {
        "calc.py": "# Original comment\ndef add(value):\n    return value + 1\n",
        "stable.py": "def retained(value):\n    return value * 2\n",
        "deleted.py": "def removed():\n    return False\n",
        "data.txt": "retained data\n",
    }
    for path, content in files.items():
        selected = source / path
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(content)
    package = tmp_path / (name + "-package")
    receipt = view.create_repository_view(source, package)
    manifest = view.load_repository_view(package, expected_manifest_sha256=receipt["manifest_sha256"])
    original = tmp_path / (name + "-original")
    view.restore_repository_view(package, original, expected_manifest_sha256=receipt["manifest_sha256"])
    return source, package, original, receipt, manifest


def index(tmp_path, scheduler):
    return NativeRepositoryIndex(tmp_path / "index", scheduler=scheduler)


def test_native_refresh_is_json_safe_and_remains_datasets_owned(tmp_path, scheduler, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    source, _package, original, package_receipt, manifest = snapshot(tmp_path)
    original_bytes = (source / "calc.py").read_bytes()
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert receipt["package_manifest_sha256"] == package_receipt["manifest_sha256"]
        assert receipt["source_tree_sha256"] == package_receipt["source_tree_sha256"]
        assert receipt["head"]["generation"] == 1
        assert receipt["coverage"]["inventory_entries"] == 4
        assert receipt["semantic_reconstruction"] == "fresh"
        assert receipt["ast_stats_delta"]["parse_invocations"] == 3
        assert receipt["source_scope_verified"] is True
        assert all(receipt[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
        rows = owner.symbols(path="calc.py", query="ADD")
        assert len(rows) == 1 and rows[0]["name"] == "calc.add"
        assert rows[0]["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
        assert rows[0]["edit_handle_available"] is False
        assert json.loads(json.dumps(receipt)) == receipt
        assert json.loads(json.dumps(rows)) == rows
        assert len(owner.symbols(limit=1)) == 1
        assert owner.symbols(path="missing.py") == []
        assert owner.symbols(query="' OR TRUE --") == []
        assert owner._cx.execute("SELECT current_setting('threads')").fetchone()[0] == 1
        assert owner._cx.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
    assert (source / "calc.py").read_bytes() == original_bytes
    assert scheduler.snapshot()["active_lease_count"] == 0


def test_revision_refresh_reuses_ast_and_removes_deleted_symbols(tmp_path, scheduler):
    source, package, original, initial, manifest = snapshot(tmp_path)
    operations = [
        dict(operation="replace_symbol", path="calc.py", symbol="add",
             expected_sha256=manifest["entries"]["calc.py"]["original_sha256"],
             replacement="def add(value):\n    return value + 100\n"),
        dict(operation="delete_file", path="deleted.py",
             expected_sha256=manifest["entries"]["deleted.py"]["original_sha256"]),
        dict(operation="write_file", path="new.py", expected_sha256=None,
             content="def added():\n    return 7\n"),
    ]
    with index(tmp_path, scheduler) as owner:
        first = owner.refresh(original, manifest)
        old = owner.symbols(query="add")
        successor = tmp_path / "successor-package"
        revision = revise_repository_view(package, successor,
            expected_manifest_sha256=initial["manifest_sha256"], operations=operations)
        digest = revision["successor_manifest_sha256"]
        new_manifest = view.load_repository_view(successor, expected_manifest_sha256=digest)
        new_root = tmp_path / "successor-original"
        view.restore_repository_view(successor, new_root, expected_manifest_sha256=digest)
        second = owner.refresh(new_root, new_manifest)
        rows = owner.symbols()
        names = {row["name"] for row in rows}
        assert {"calc.add", "stable.retained", "new.added"} <= names
        assert not any(name.startswith("deleted") for name in names)
        assert second["head"]["generation"] == first["head"]["generation"] + 1
        assert second["head"]["snapshot_cid"] != first["head"]["snapshot_cid"]
        assert second["ast_stats_delta"]["reuse_hits"] >= 1
        assert second["ast_stats_delta"]["parse_invocations"] == 2
        changed = owner.symbols(query="calc.add")[0]
        assert changed["symbol_id"] == old[0]["symbol_id"]
        assert changed["version_cid"] != old[0]["version_cid"]
    assert (source / "deleted.py").exists() and (original / "deleted.py").exists()


@pytest.mark.parametrize("mutation", ["bytes", "added", "deleted", "mode"])
def test_current_queries_refuse_any_materialized_source_drift(tmp_path, scheduler, mutation):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        if mutation == "bytes":
            (original / "data.txt").write_text("changed unrelated data\n")
        elif mutation == "added":
            (original / "extra.py").write_text("def extra(): return 1\n")
        elif mutation == "deleted":
            (original / "deleted.py").unlink()
        else:
            (original / "data.txt").chmod(0o600)
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package"):
            owner.symbols(query="calc.add")
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()


def test_failed_successor_cannot_serve_predecessor_as_current(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        bad = copy.deepcopy(manifest)
        bad["source_tree_sha256"] = "0" * 64
        with pytest.raises(RepositorySourceIndexError):
            owner.refresh(original, bad)
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()
        assert owner.refresh(original, manifest)["source_scope_verified"]


def test_ancestor_git_is_refused_before_native_publication(tmp_path, scheduler):
    _source, package, _original, receipt, manifest = snapshot(tmp_path)
    ancestor = tmp_path / "git-parent"
    ancestor.mkdir()
    subprocess.run(["git", "init", "--quiet", str(ancestor)], check=True, timeout=10)
    original = ancestor / "nested-original"
    view.restore_repository_view(package, original, expected_manifest_sha256=receipt["manifest_sha256"])
    with index(tmp_path, scheduler) as owner:
        with pytest.raises(RepositorySourceIndexError, match="git_ancestor_not_supported"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None


def test_native_excluded_directory_is_explicitly_refused(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={".venv/hidden.py": "def hidden(): return 1\n"})
    with index(tmp_path, scheduler) as owner:
        with pytest.raises(RepositorySourceIndexError, match="native_excluded_path_not_supported"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None


def test_retained_symlink_is_opaque_and_never_followed(tmp_path, scheduler):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def real(): return 1\n")
    (source / "alias.py").symlink_to("a.py")
    receipt = view.create_repository_view(source, tmp_path / "package")
    manifest = view.load_repository_view(tmp_path / "package", expected_manifest_sha256=receipt["manifest_sha256"])
    view.restore_repository_view(tmp_path / "package", tmp_path / "original", expected_manifest_sha256=receipt["manifest_sha256"])
    with index(tmp_path, scheduler) as owner:
        result = owner.refresh(tmp_path / "original", manifest)
        assert result["coverage"]["opaque_entries"] == 1
        assert owner.symbols(path="alias.py") == []
        assert owner.symbols(path="a.py", query="real")


def test_binary_original_is_exactly_bound_but_remains_native_opaque(tmp_path, scheduler):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_bytes(b"")
    (source / "a.py").write_bytes(b"def real(): return 1\n")
    (source / "data.bin").write_bytes(b"\x00\xff\x10\x80")
    captured = view.create_repository_view(source, tmp_path / "package")
    manifest = view.load_repository_view(tmp_path / "package", expected_manifest_sha256=captured["manifest_sha256"])
    original = tmp_path / "original"
    view.restore_repository_view(tmp_path / "package", original, expected_manifest_sha256=captured["manifest_sha256"])
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert receipt["source_scope_verified"] is True
        assert receipt["coverage"]["inventory_entries"] == 3
        assert receipt["coverage"]["opaque_entries"] == 1
        assert receipt["coverage"]["formalized_properties"] == receipt["coverage"]["checked_properties"] == 0
        assert owner.symbols(path="data.bin") == []
        assert owner.symbols(path="a.py", query="real")
        (original / "data.bin").write_bytes(b"\x00\xff\x10\x81")
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package"):
            owner.symbols(path="a.py")


def test_parse_failure_is_visible_and_not_a_false_proof(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={"invalid.py": "def invalid(:\n", "ok.py": "def ok(): return 1\n"})
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert receipt["coverage"]["ast_failed"] == 1
        assert receipt["coverage"]["checked_properties"] == 0
        assert owner.symbols(path="invalid.py") == []
        assert owner.symbols(path="ok.py", query="ok")


def test_owner_lock_and_close_are_exclusive(tmp_path, scheduler):
    first = index(tmp_path, scheduler)
    with pytest.raises(RepositorySourceIndexError, match="index_owner_busy"):
        index(tmp_path, scheduler)
    first.close()
    first.close()
    with pytest.raises(RepositorySourceIndexError, match="index_closed_or_foreign_owner"):
        first.symbols()
    with index(tmp_path, scheduler):
        pass


@pytest.mark.parametrize("kwargs", [dict(limit=0), dict(limit=True), dict(limit=1001), dict(query=5),
                                   dict(query="x" * 4097), dict(query="\ud800"), dict(query="\u00e9" * 3000)])
def test_query_bounds(tmp_path, scheduler, kwargs):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        with pytest.raises(RepositorySourceIndexError, match="invalid_symbol"):
            owner.symbols(**kwargs)


def test_receipt_and_caller_manifest_mutation_do_not_change_active_index(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        receipt["head"]["generation"] = 999
        manifest["entries"].clear()
        assert owner.symbols(query="calc.add")[0]["name"] == "calc.add"


def test_wide_large_file_bound_refuses_before_native_publication(tmp_path, scheduler):
    files = {f"empty-{number}.txt": "" for number in range(128)}
    files["largest.txt"] = "x" * (1024 * 1024)
    _source, _package, original, _receipt, manifest = snapshot(tmp_path, files=files)
    with index(tmp_path, scheduler) as owner:
        with pytest.raises(RepositorySourceIndexError, match="native_index_reservation_limit"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None
        assert owner._ingestor.stats()["parse_invocations"] == 0


def test_external_memory_pressure_refuses_then_recovers_without_stale_rows(tmp_path, monkeypatch):
    available = [16384]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "pressure-resources.json",
        proof_resource_sampler=lambda: ProofHostResources(8, 16384, available[0]),
        lane_reservations={}, auto_renew_leases=False,
        proof_backoff_seconds=0, poll_interval_seconds=0.005))
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        controls = owner._controls
        monkeypatch.setattr(owner, "_controls", lambda memory_mb: {
            **controls(memory_mb), "admission_timeout_seconds": 0.05, "timeout_seconds": 0.2})
        available[0] = 64
        with pytest.raises(RepositorySourceIndexError, match="native_index_refresh_failed"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None
        assert owner._ingestor.stats()["parse_invocations"] == 0
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()
        assert scheduler.snapshot()["active_lease_count"] == 0
        available[0] = 16384
        monkeypatch.setattr(owner, "_controls", controls)
        assert owner.refresh(original, manifest)["source_scope_verified"]
        assert owner.symbols(query="calc.add")

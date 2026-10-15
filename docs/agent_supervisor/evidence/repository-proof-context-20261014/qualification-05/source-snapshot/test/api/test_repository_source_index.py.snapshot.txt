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


def capture_existing(tmp_path, source, name="initial"):
    package, original = tmp_path / (name + "-package"), tmp_path / (name + "-original")
    receipt = view.create_repository_view(source, package)
    manifest = view.load_repository_view(package, expected_manifest_sha256=receipt["manifest_sha256"])
    view.restore_repository_view(package, original, expected_manifest_sha256=receipt["manifest_sha256"])
    return original, receipt, manifest


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
        assert owner.symbol(symbol_id=old[0]["symbol_id"]) == changed
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


@pytest.mark.parametrize("directory", [".venv", "build", "dist", "vendor"])
@pytest.mark.parametrize("prefix", ["", "__pycache__/"])
def test_native_excluded_directory_is_explicitly_refused(tmp_path, scheduler, directory, prefix):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path,
        files={prefix + directory + "/hidden.py": "def hidden(): return 1\n"})
    with index(tmp_path, scheduler) as owner:
        with pytest.raises(RepositorySourceIndexError, match="native_excluded_path_not_supported"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None


def test_generated_cache_is_retained_exactly_with_separate_opaque_coverage(tmp_path, scheduler):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def real(): return 1\n")
    (source / "__pycache__").mkdir(mode=0o750)
    (source / "__pycache__" / "calc.pyc").write_bytes(b"\xa7\r\r\n\x00\xff")
    (source / "__pycache__" / "calc.pyc").chmod(0o600)
    # A cache name makes no claim that arbitrary source underneath is generated.
    (source / "__pycache__" / "source.py").write_text("def hidden(): raise RuntimeError('not executed')\n")
    (source / "__pycache__" / "alias").symlink_to("source.py")
    (source / "package" / ".pytest_cache" / "v" / "cache").mkdir(parents=True)
    (source / "package" / ".pytest_cache" / "v" / "cache" / "nodeids").write_text('["test_calc"]')
    (source / "package" / ".pytest_cache" / "empty").mkdir()
    original, captured, manifest = capture_existing(tmp_path, source)
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        cache = receipt["retained_opaque_cache"]
        assert cache["roots"] == ["__pycache__", "package/.pytest_cache"]
        expected = {name: entry for name, entry in view._original_entries(manifest["entries"]).items()
                    if name.startswith("__pycache__") or name.startswith("package/.pytest_cache")}
        assert cache["entries"] == expected
        payload = {key: cache[key] for key in ("schema", "roots", "entries")}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()
        assert cache["inventory_sha256"] == hashlib.sha256(encoded).hexdigest()
        assert cache["entry_count"] == 9
        assert cache["file_count"] == 3 and cache["directory_count"] == 5 and cache["symlink_count"] == 1
        assert cache["original_bytes"] == sum(entry.get("original_bytes", 0) for entry in expected.values())
        assert cache["classification"] == "directory_name_only"
        assert cache["native_indexed"] is False
        assert all(cache[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
        assert receipt["coverage"]["inventory_entries"] == 1
        assert receipt["coverage"]["opaque_entries"] == 0
        assert receipt["coverage"]["checked_properties"] == receipt["coverage"]["formalized_properties"] == 0
        assert receipt["ast_stats_delta"]["parse_invocations"] == 1
        assert receipt["package_manifest_sha256"] == captured["manifest_sha256"]
        assert owner.symbols(path="__pycache__/source.py") == []
        assert owner.symbols(query="hidden") == []
        assert owner.symbols(query="real")
        assert (original / "__pycache__" / "calc.pyc").read_bytes() == (source / "__pycache__" / "calc.pyc").read_bytes()
        assert (original / "__pycache__" / "calc.pyc").stat().st_mode & 0o777 == 0o600
        assert (original / "__pycache__" / "alias").readlink() == Path("source.py")
        assert (original / "package" / ".pytest_cache" / "empty").is_dir()
        # Caller mutation of this inventory cannot change its active fence.
        cache["entries"].clear()
        assert owner.symbols(query="real")


@pytest.mark.parametrize("name", ["__pycache__", ".pytest_cache"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_cache_spelling_does_not_exempt_files_or_symlinks(tmp_path, scheduler, name, kind):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def real(): return 1\n")
    if kind == "file":
        (source / name).write_text("must not silently disappear")
    else:
        (source / name).symlink_to("calc.py")
    original, _captured, manifest = capture_existing(tmp_path, source)
    with index(tmp_path, scheduler) as owner:
        with pytest.raises(RepositorySourceIndexError, match="native_cache_name_requires_directory"):
            owner.refresh(original, manifest)
        assert owner._index.current(owner.repository_id) is None


def test_empty_cache_directories_have_coverage_without_native_inventory(tmp_path, scheduler):
    source = tmp_path / "source"
    (source / ".pytest_cache" / "empty").mkdir(parents=True)
    (source / "__pycache__").mkdir()
    original, _captured, manifest = capture_existing(tmp_path, source)
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert receipt["coverage"]["inventory_entries"] == 0
        assert receipt["retained_opaque_cache"]["entry_count"] == 3
        assert receipt["retained_opaque_cache"]["directory_count"] == 3
        assert receipt["retained_opaque_cache"]["original_bytes"] == 0
        assert receipt["ast_stats_delta"]["parse_invocations"] == 0
        assert owner.symbols() == []


@pytest.mark.parametrize("mutation", ["bytes", "added", "deleted", "mode", "empty_directory", "symlink"])
def test_materialized_cache_drift_fences_native_queries(tmp_path, scheduler, mutation):
    _source, _package, original, _captured, manifest = snapshot(tmp_path,
        files={"calc.py": "def real(): return 1\n", ".pytest_cache/retained": "original"})
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        cache = original / ".pytest_cache"
        if mutation == "bytes":
            (cache / "retained").write_text("changed")
        elif mutation == "added":
            (cache / "new").write_text("new cache")
        elif mutation == "deleted":
            (cache / "retained").unlink()
        elif mutation == "mode":
            cache.chmod(0o700)
        elif mutation == "empty_directory":
            (cache / "empty").mkdir()
        else:
            (cache / "retained").unlink()
            (cache / "retained").symlink_to("../calc.py")
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package"):
            owner.symbols(query="real")
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()


def test_cache_change_during_native_observation_is_fenced(tmp_path, scheduler, monkeypatch):
    _source, _package, original, _captured, manifest = snapshot(tmp_path,
        files={"calc.py": "def real(): return 1\n", "__pycache__/retained": "original"})
    with index(tmp_path, scheduler) as owner:
        observe = owner._index.observe_current
        def mutate_cache_after_native_observation(*args, **kwargs):
            result = observe(*args, **kwargs)
            (original / "__pycache__" / "retained").write_text("changed outside native snapshot")
            return result
        monkeypatch.setattr(owner._index, "observe_current", mutate_cache_after_native_observation)
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package"):
            owner.refresh(original, manifest)
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbols()


def test_native_reservation_uses_eligible_inventory_but_retains_cache_bytes(tmp_path, scheduler):
    files = {f"__pycache__/empty-{number}": "" for number in range(128)}
    files["__pycache__/largest"] = "x" * (1024 * 1024)
    files["calc.py"] = "def real(): return 1\n"
    _source, _package, original, _captured, manifest = snapshot(tmp_path, files=files)
    with index(tmp_path, scheduler) as owner:
        receipt = owner.refresh(original, manifest)
        assert receipt["coverage"]["inventory_entries"] == 1
        assert receipt["retained_opaque_cache"]["entry_count"] == 130
        assert receipt["retained_opaque_cache"]["original_bytes"] == 1024 * 1024
        assert receipt["resource_limits"]["admission_memory_mb"] == 256
        assert owner.symbols(query="real")


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


def test_exact_symbol_lookup_is_independent_of_public_list_limits(tmp_path, scheduler, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index as native_index

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        expected = owner.symbols(path="stable.py", query="retained")[0]
        # Lower the public cap to exercise a real population larger than it
        # without creating a large parser workload just for this boundary.
        monkeypatch.setattr(native_index, "MAX_RESULTS", 1)
        assert expected not in owner.symbols(limit=1)
        with pytest.raises(RepositorySourceIndexError, match="invalid_symbol_limit"):
            owner.symbols(limit=2)
        assert owner.symbols(query=expected["symbol_id"], limit=1) == []
        actual = owner.symbol(symbol_id=expected["symbol_id"])
        assert actual == expected
        actual["span"]["start_line"] = 999
        assert owner.symbol(symbol_id=expected["symbol_id"]) == expected


def test_invalid_and_unknown_symbol_id_preserve_healthy_index(tmp_path, scheduler):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        expected = owner.symbols(query="calc.add")[0]
        for value in (None, True, 5, [], "", "not-a-cid", "x" * 257, "\ud800",
                      expected["symbol_id"].upper()):
            with pytest.raises(RepositorySourceIndexError, match="^invalid_symbol_id$") as failure:
                owner.symbol(symbol_id=value)
            assert failure.value.preserves_index is True
        with pytest.raises(RepositorySourceIndexError, match="^symbol_unavailable$") as failure:
            owner.symbol(symbol_id=cid_for_bytes(b"absent native symbol identity"))
        assert failure.value.preserves_index is True
        assert owner.symbol(symbol_id=expected["symbol_id"]) == expected


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("timing", ["before", "second_fence"])
def test_exact_symbol_lookup_checks_source_even_for_missing_identity(
        tmp_path, scheduler, monkeypatch, unknown, timing):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        symbol_id = (cid_for_bytes(b"missing symbol") if unknown
                     else owner.symbols(query="calc.add")[0]["symbol_id"])
        if timing == "before":
            (original / "data.txt").write_text("changed outside the requested symbol\n")
        else:
            observe = owner._index.observe_current
            calls = []

            def change_after_second_observation(*args, **kwargs):
                result = observe(*args, **kwargs)
                calls.append(True)
                if len(calls) == 2:
                    (original / "data.txt").write_text("changed during the final fence\n")
                return result

            monkeypatch.setattr(owner._index, "observe_current", change_after_second_observation)
        with pytest.raises(RepositorySourceIndexError, match="materialization_differs_from_package") as failure:
            owner.symbol(symbol_id=symbol_id)
        assert failure.value.preserves_index is False
        if timing == "second_fence":
            assert len(calls) == 2
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbol(symbol_id=symbol_id)


def test_exact_symbol_lookup_refuses_changed_native_head(tmp_path, scheduler):
    _source, _package, original, _receipt, manifest = snapshot(tmp_path)
    with index(tmp_path, scheduler) as owner:
        owner.refresh(original, manifest)
        symbol_id = owner.symbols(query="calc.add")[0]["symbol_id"]
        owner._cx.execute("UPDATE codebase_control.heads SET generation = generation + 1 WHERE repository_id = ?",
                          [owner.repository_id])
        with pytest.raises(RepositorySourceIndexError, match="native_index_query_failed") as failure:
            owner.symbol(symbol_id=symbol_id)
        assert failure.value.preserves_index is False
        with pytest.raises(RepositorySourceIndexError, match="index_not_current"):
            owner.symbol(symbol_id=symbol_id)


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

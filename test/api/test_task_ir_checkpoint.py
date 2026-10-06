"""Real retained bytes authenticate separately from decoder/runtime authority.

Mutation controls change actual fixture files/catalogs at identified read
boundaries. They exercise cooperative endpoint refusal, not an atomic lease.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys

import duckdb
import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.task_ir_checkpoint as owner
from test.api.test_task_ir_selection import store
from test.test_ir_persistent_catalog import _raw, _record, _selection


def retained(tmp_path, *, family="legal_ir", dimension=384, index=0, **declarations):
    payload = f"\x00opaque retained {family}/{dimension}/{index}\n".encode()
    row = _record(family, dimension, "latent" if dimension == 8 else "input_embedding",
                  checkpoint=payload.decode(), **declarations)
    path = tmp_path / f"retained-{family}-{dimension}-{index}.bin"
    path.write_bytes(payload)
    row["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"] = {
        "path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return row, path


def authenticate(path, rows):
    return owner.authenticate_task_ir_checkpoints(catalog_path=path,
        selections=[_selection(row) for row in rows])


def no_checkpoint_open(monkeypatch, paths):
    actual = os.open
    def checked(path, flags, *args, **kwargs):
        if Path(path) in paths:
            pytest.fail("all checkpoint plans must be validated before any checkpoint opens")
        return actual(path, flags, *args, **kwargs)
    monkeypatch.setattr(owner.os, "open", checked)


@pytest.mark.parametrize("database", [False, True])
def test_twelve_retained_family_lane_files_match_bytes_without_runtime_or_store_changes(tmp_path, monkeypatch, database):
    assets = [retained(tmp_path, family=family, dimension=dimension)
              for family in ("codebase_ir", "security_ir", "legal_ir", "intent_ir")
              for dimension in (8, 384, 768)]
    rows, paths = [item[0] for item in assets], [item[1] for item in assets]
    catalog = store(tmp_path, rows, database=database)
    before = {path: path.read_bytes() for path in paths + [catalog]}
    requests = [_selection(row) for row in rows]
    original = deepcopy(requests)
    native_before = owner.resolve_task_ir_selections(catalog_path=catalog, selections=requests)
    actual_open, opens = os.open, []
    def observed_open(path, flags, *args, **kwargs):
        if Path(path) in paths:
            assert flags & os.O_ACCMODE == os.O_RDONLY
            assert flags & (os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC) == (os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            opens.append(Path(path))
        return actual_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(owner.os, "open", observed_open)
    modules_before = set(sys.modules)
    result = owner.authenticate_task_ir_checkpoints(catalog_path=catalog, selections=requests)
    assert len(result) == 12 and opens == paths and requests == original
    assert not any(name == "torch" or name.startswith(("torch.", "transformers.", "sentence_transformers.", "ipfs_datasets_py."))
                   for name in set(sys.modules) - modules_before)
    assert "ipfs_accelerate_py.model_manager" not in set(sys.modules) - modules_before
    assert {path: path.read_bytes() for path in before} == before
    for request, native, row, path in zip(requests, native_before, result, paths):
        current = path.stat()
        assert row["schema"] == "task-ir-checkpoint-authentication/v1"
        assert row["selectors"] == request and row["native_resolution"] == native
        assert row["checkpoint_bytes_authenticated"] is True
        assert row["native_resolution"]["authority"]["checkpoint_bytes_authenticated"] is False
        assert not any(row["authority"].values()) and not any(native["authority"].values())
        assert row["observation_scope"] == "cooperative-endpoint-checks"
        assert row["file_witness"] == {
            **row["original_checkpoint_pin"], "device": current.st_dev, "inode": current.st_ino,
            "mode": current.st_mode, "nlink": 1, "mtime_ns": current.st_mtime_ns, "ctime_ns": current.st_ctime_ns,
        }
        assert stat.S_ISREG(row["file_witness"]["mode"])
    if database:
        with duckdb.connect(str(catalog), read_only=True) as connection:
            assert connection.execute("SELECT value FROM unrelated").fetchall() == [(73,)]


def test_schema_decoder_task_and_geometry_role_remain_distinct_despite_shared_bytes(tmp_path):
    first, file = retained(tmp_path, task_id="semantic_ir_reconstruction", schema_version="legal-ir/v1")
    # The native asset identity uses family/lane/role/checkpoint; duplicate
    # record IDs cannot share a catalog, so use different retained roles.
    second = _record(checkpoint=file.read_text(), role="text-head", task_id="legal_text_reconstruction", schema_version="legal-text/v1",
                     original_checkpoint_pin=deepcopy(first["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"]))
    rows = [first, second]
    result = authenticate(store(tmp_path, rows), rows)
    assert {row["selectors"]["task_id"] for row in result} == {"semantic_ir_reconstruction", "legal_text_reconstruction"}
    assert result[0]["file_witness"] == result[1]["file_witness"]
    assert all(row["native_resolution"]["selected_binding"]["runtime_observation_performed"] is False for row in result)


@pytest.mark.parametrize("change", ["contents", "shorter", "longer"])
def test_existing_file_bytes_drift_refuses_without_loading_or_rewriting_assets(tmp_path, change):
    row, file = retained(tmp_path)
    catalog = store(tmp_path, [row])
    payload = file.read_bytes()
    file.write_bytes((b"X" + payload[1:]) if change == "contents" else (payload[:-1] if change == "shorter" else payload + b"X"))
    changed = file.read_bytes()
    with pytest.raises(ValueError, match="declared"):
        authenticate(catalog, [row])
    assert file.read_bytes() == changed


@pytest.mark.parametrize("unsafe", ["missing", "symlink", "parent_symlink", "hardlink", "fifo", "directory", "relative", "traversal", "dot", "bool_bytes"])
def test_unsafe_pins_and_paths_refuse_before_any_checkpoint_descriptor(tmp_path, monkeypatch, unsafe):
    first, first_path = retained(tmp_path, family="intent_ir")
    bad, bad_path = retained(tmp_path, family="legal_ir")
    pin = bad["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"]
    if unsafe == "missing": bad_path.unlink()
    elif unsafe == "symlink":
        alias = tmp_path / "alias.bin"
        alias.symlink_to(bad_path)
        pin["path"] = str(alias)
    elif unsafe == "parent_symlink":
        alias = tmp_path / "parent-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        pin["path"] = str(alias / bad_path.name)
    elif unsafe == "hardlink": os.link(bad_path, tmp_path / "second-link.bin")
    elif unsafe == "fifo":
        bad_path.unlink()
        os.mkfifo(bad_path)
    elif unsafe == "directory":
        bad_path.unlink()
        bad_path.mkdir()
    elif unsafe == "relative": pin["path"] = "relative.bin"
    elif unsafe == "traversal": pin["path"] = str(tmp_path / ".." / tmp_path.name / bad_path.name)
    elif unsafe == "dot": pin["path"] = str(tmp_path) + "/./" + bad_path.name
    else: pin["bytes"] = True
    rows = [first, bad]
    catalog = store(tmp_path, rows)
    no_checkpoint_open(monkeypatch, {first_path, bad_path, Path(pin["path"])})
    with pytest.raises(ValueError):
        authenticate(catalog, rows)
    assert first_path.read_bytes().startswith(b"\x00opaque")


@pytest.mark.parametrize("bound", ["per_file", "aggregate"])
def test_real_sparse_file_byte_budgets_refuse_before_opening_any_checkpoint(tmp_path, monkeypatch, bound):
    count = 1 if bound == "per_file" else 3
    assets = [retained(tmp_path, index=index, task_id=f"head-{index}") for index in range(count)]
    rows, paths = [item[0] for item in assets], [item[1] for item in assets]
    size = owner.MAX_CHECKPOINT_BYTES + 1 if bound == "per_file" else owner.MAX_CHECKPOINT_BYTES
    for row, file in assets:
        with file.open("r+b") as stream: stream.truncate(size)
        row["huggingface_config"]["ir_checkpoint"]["original_checkpoint_pin"]["bytes"] = size
    catalog = store(tmp_path, rows)
    no_checkpoint_open(monkeypatch, set(paths))
    with pytest.raises(ValueError, match="byte bound"):
        authenticate(catalog, rows)


@pytest.mark.parametrize("selections", [[], (), None, [True], [None]])
def test_selection_types_refuse_without_store_creation(tmp_path, selections):
    missing = tmp_path / "never-created.json"
    with pytest.raises(ValueError):
        owner.authenticate_task_ir_checkpoints(catalog_path=missing, selections=selections)
    assert not missing.exists()


@pytest.mark.parametrize("count", [16, 17])
def test_exact_selection_upper_population_bound(tmp_path, count):
    assets = [retained(tmp_path, index=index, task_id=f"head-{index}") for index in range(count)]
    rows = [item[0] for item in assets]
    catalog = store(tmp_path, rows)
    if count == 16: assert len(authenticate(catalog, rows)) == 16
    else:
        with pytest.raises(ValueError, match="1 to 16"):
            authenticate(catalog, rows)


def test_competing_namespace_refuses_before_any_checkpoint_open(tmp_path, monkeypatch):
    first, file = retained(tmp_path)
    second, second_file = retained(tmp_path, index=1)
    rows = [first, second]
    catalog = store(tmp_path, rows)
    no_checkpoint_open(monkeypatch, {file, second_file})
    with pytest.raises(ValueError, match="complete namespace"):
        authenticate(catalog, rows)


@pytest.mark.parametrize("boundary", ["open", "read", "closing_catalog"])
def test_same_bytes_inode_exchange_at_actual_read_boundaries_refuses(tmp_path, monkeypatch, boundary):
    row, file = retained(tmp_path)
    catalog = store(tmp_path, [row])
    original = file.read_bytes()
    def exchange():
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(original)
        replacement.replace(file)
    if boundary == "open":
        actual = os.open
        def changed(path, flags, *args, **kwargs):
            if Path(path) == file: exchange()
            return actual(path, flags, *args, **kwargs)
        monkeypatch.setattr(owner.os, "open", changed)
    elif boundary == "read":
        actual = owner._read_pin
        def changed(descriptor, pin):
            actual(descriptor, pin)
            exchange()
        monkeypatch.setattr(owner, "_read_pin", changed)
    else:
        actual, calls = owner.resolve_task_ir_selections, []
        def changed(**kwargs):
            result = actual(**kwargs)
            calls.append(True)
            if len(calls) == 2: exchange()
            return result
        monkeypatch.setattr(owner, "resolve_task_ir_selections", changed)
    with pytest.raises(ValueError, match="identity changed|single-link"):
        authenticate(catalog, [row])
    assert file.read_bytes() == original


@pytest.mark.parametrize("change", ["same_size_bytes", "grow", "truncate", "mode", "hardlink"])
def test_genuine_file_changes_during_streaming_refuse(tmp_path, monkeypatch, change):
    row, file = retained(tmp_path)
    catalog = store(tmp_path, [row])
    actual = owner.os.read
    changed = []
    def changing_read(descriptor, count):
        block = actual(descriptor, count)
        if block and not changed and os.fstat(descriptor).st_ino == file.stat().st_ino:
            changed.append(True)
            if change == "same_size_bytes": file.write_bytes(b"X" * file.stat().st_size)
            elif change == "grow":
                with file.open("ab") as stream: stream.write(b"X")
            elif change == "truncate": file.write_bytes(b"X")
            elif change == "mode": file.chmod(0o600 if file.stat().st_mode & 0o777 != 0o600 else 0o644)
            else: os.link(file, tmp_path / "added-link.bin")
        return block
    monkeypatch.setattr(owner.os, "read", changing_read)
    with pytest.raises(ValueError):
        authenticate(catalog, [row])
    assert changed == [True]


def test_earlier_file_change_after_later_authentication_is_checked_at_final_endpoint(tmp_path, monkeypatch):
    first, file = retained(tmp_path, family="intent_ir")
    second, _ = retained(tmp_path, family="legal_ir")
    rows = [first, second]
    catalog = store(tmp_path, rows)
    actual, calls = owner._read_pin, []
    def changed(descriptor, pin):
        actual(descriptor, pin)
        calls.append(True)
        if len(calls) == 2: file.write_bytes(b"X" * file.stat().st_size)
    monkeypatch.setattr(owner, "_read_pin", changed)
    with pytest.raises(ValueError, match="identity changed"):
        authenticate(catalog, rows)


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("boundary", ["during_read", "after_closing_resolution"])
def test_actual_catalog_metadata_change_during_authentication_refuses(tmp_path, monkeypatch, boundary, database):
    row, file = retained(tmp_path)
    catalog = store(tmp_path, [row], database=database)
    def edit():
        row["huggingface_config"]["ir_checkpoint"]["trained"] = True
        if database:
            with duckdb.connect(str(catalog)) as connection:
                connection.execute("UPDATE model_metadata SET huggingface_config = ?",
                                   [_raw(row["huggingface_config"]).decode()])
        else:
            catalog.write_bytes(_raw([row]))
    if boundary == "during_read":
        actual = owner._read_pin
        def changed(descriptor, pin):
            actual(descriptor, pin)
            edit()
        monkeypatch.setattr(owner, "_read_pin", changed)
    else:
        # Inject into the native resolver's final load boundary, rather than
        # returning a forged resolution from the authentication owner.
        from ipfs_accelerate_py.model_catalog.sources.ir_persistent import IRPersistentCatalogSource
        actual, calls = IRPersistentCatalogSource.resolve_ir_binding, []
        def changed(self, request):
            result = actual(self, request)
            calls.append(True)
            if len(calls) == 2: edit()
            return result
        monkeypatch.setattr(IRPersistentCatalogSource, "resolve_ir_binding", changed)
    with pytest.raises(ValueError, match="generation changed"):
        authenticate(catalog, [row])
    assert file.read_bytes().startswith(b"\x00opaque")


def test_all_open_descriptors_close_when_a_later_checkpoint_refuses(tmp_path, monkeypatch):
    first, file = retained(tmp_path, family="intent_ir")
    second, second_file = retained(tmp_path, family="legal_ir")
    rows = [first, second]
    catalog = store(tmp_path, rows)
    second_file.write_bytes(b"X" * second_file.stat().st_size)
    actual_open, descriptors = os.open, []
    def observed(path, flags, *args, **kwargs):
        descriptor = actual_open(path, flags, *args, **kwargs)
        if Path(path) in (file, second_file): descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(owner.os, "open", observed)
    with pytest.raises(ValueError, match="declared SHA256"):
        authenticate(catalog, rows)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError): os.fstat(descriptor)


def test_returned_observations_are_detached_and_do_not_promote_declarations(tmp_path):
    row, _ = retained(tmp_path, trained=True, runtime_ready=True, teacher_qualified=True, proof_authority=True)
    catalog = store(tmp_path, [row])
    result = authenticate(catalog, [row])[0]
    assert not any(result["authority"].values())
    assert all(result["native_resolution"]["selected_binding"]["declaration"][name] is True
               for name in ("trained", "runtime_ready", "teacher_qualified", "proof_authority"))
    result["native_resolution"]["authority"]["runtime_admitted"] = True
    result["selectors"]["task_id"] = "foreign"
    result["original_checkpoint_pin"]["path"] = "/foreign"
    again = authenticate(catalog, [row])[0]
    assert not any(again["native_resolution"]["authority"].values())
    assert again["selectors"]["task_id"] is None
    assert again["original_checkpoint_pin"]["path"] != "/foreign"

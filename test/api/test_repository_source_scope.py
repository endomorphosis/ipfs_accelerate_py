"""Real historical Git/CAS nomination into explicit bounded source metadata.

Targets are never executed. This bridge returns no native task grant or source
body; actual semantic preparation and working reads retain their own owners.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as inventory
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_scope as scopes
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from test.api.test_repository_source_inventory import _repository, _retain, _commit_request, _git, _chunk_cid


MARKER = b"PRIVATE_SOURCE_BODY_MUST_NOT_APPEAR_IN_NOMINATION"
MAIN = b"raise RuntimeError('target_must_not_execute')\n# " + MARKER + b"\n"
SUPPORT = b"Public task requirement: preserve the source binding.\n"


def hexes(*paths):
    return tuple(sorted(os.fsencode(path).hex() for path in paths))


@pytest.fixture
def retained(tmp_path):
    files = {b"main.py": MAIN, b"instruction.md": SUPPORT, b"smoke.py": b"VALUE = 1\n",
        b"types.pyi": b"VALUE: int\n", b"coverage/retained.txt": b"retained tracked frontier\n",
        b"binary.bin": b"\xff\x00", b"nul.txt": b"a\x00b", b"bad-\xff.py": b"VALUE = 2\n",
        b"control-\n.py": b"VALUE = 3\n"}
    root = _repository(tmp_path, files)
    (root / "main.py").chmod(0o755)
    first = _commit_request(root)
    outside = tmp_path / "outside"
    outside.write_bytes(b"external target must remain unread")
    os.symlink(str(outside), root / "external-link")
    (root / "nested").mkdir()
    _git(root, "update-index", "--add", "--cacheinfo", "160000", first.expected_commit, "nested")
    return _retain(root, tmp_path, request=_commit_request(root))


def nominate(case, **changes):
    arguments = dict(cas=case.cas, expected_inventory_cid=case.ref.inventory_cid,
        request=case.request, limits=case.limits,
        selected_paths=hexes("main.py", "instruction.md", "smoke.py"),
        required_paths=hexes("instruction.md", "smoke.py"), program_paths=hexes("main.py"))
    arguments.update(changes)
    return scopes.nominate_source_scope(**arguments)


def frontier(scope, max_bytes=16384):
    rows, offset = [], 0
    for _ in range(1000):
        page = scope.frontier_page(offset=offset, limit=64, max_bytes=max_bytes)
        assert len(json.dumps(page, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()) <= max_bytes
        rows.extend(page["entries"])
        if page["next_offset"] is None:
            assert page["exhausted"] is True
            return rows
        assert page["next_offset"] > offset and page["exhausted"] is False
        offset = page["next_offset"]
    pytest.fail("bounded metadata paging did not terminate")


def test_full_population_identity_is_retained_and_bodies_never_exported(retained):
    scope = nominate(retained)
    observed = scope.observation()
    assert observed["schema"] == "supervisor-repository-source-scope-nomination@1"
    assert observed["source_inventory_cid"] == retained.ref.inventory_cid
    assert observed["population_cid"] == retained.request.expected_population_cid
    assert observed["git_commit"] == retained.request.expected_commit
    assert observed["git_tree"] == retained.request.expected_tree
    assert observed["inventory_entry_count"] == 11 and observed["selected_count"] == 3
    assert observed["selected_bytes"] == len(MAIN) + len(SUPPORT) + len(b"VALUE = 1\n")
    assert observed["frontier_count"] == 8
    assert observed["historical_only"] is True
    assert all(observed[key] is False for key in (
        "current_source_verified", "task_admission_verified", "proof_authority",
        "execution_authority", "completion_authority", "omission_authority"))
    selected = {row["path"]: row for row in observed["selected"]}
    assert selected["main.py"]["role"] == "program" and selected["main.py"]["required"] is False
    assert selected["main.py"]["git_mode"] == "100755"
    assert selected["main.py"]["source_cid"] == cid_for_bytes(MAIN)
    assert selected["main.py"]["source_sha256"] == hashlib.sha256(MAIN).hexdigest()
    assert selected["instruction.md"]["role"] == "support" and selected["instruction.md"]["required"] is True
    rows = frontier(scope)
    by_path = {bytes.fromhex(row["raw_path_hex"]): row for row in rows}
    assert by_path[b"external-link"]["status"] == "symlink_requires_separate_scope"
    assert by_path[b"nested"]["status"] == "gitlink_requires_separate_repository"
    assert by_path[b"nested"]["source_cid"] is None
    assert by_path[b"binary.bin"]["status"] == by_path[b"nul.txt"]["status"] == "unsupported_source_encoding"
    assert by_path[b"bad-\xff.py"]["status"] == by_path[b"control-\n.py"]["status"] == "unsupported_path"
    assert by_path[b"coverage/retained.txt"]["status"] == "not_selected_eligible"
    assert by_path[b"types.pyi"]["status"] == "not_selected_eligible"
    assert {row["raw_path_hex"] for row in rows}.isdisjoint(row["raw_path_hex"] for row in observed["selected"])
    assert MARKER not in json.dumps(observed).encode() and MARKER not in json.dumps(rows).encode()
    assert b"external target must remain unread" not in json.dumps(rows).encode()


def test_exports_are_detached_and_semantic_inputs_do_not_reduce_full_inventory_identity(retained):
    scope = nominate(retained)
    before = scope.observation()
    exported = scope.observation()
    exported["selected"].clear()
    exported["request"]["task_id"] = "foreign"
    first = scope.semantic_inputs()
    assert first == dict(paths=["instruction.md", "main.py", "smoke.py"],
        required_raw_paths=["instruction.md", "smoke.py"], program_paths=["main.py"],
        max_files=64, max_source_bytes=1_000_000)
    first["paths"].clear()
    page = scope.frontier_page()
    page["entries"].clear()
    assert scope.observation() == before
    assert scope.semantic_inputs()["paths"] == ["instruction.md", "main.py", "smoke.py"]
    assert scope.frontier_page()["entries"]
    assert set(scope.semantic_inputs()) == {"paths", "required_raw_paths", "program_paths", "max_files", "max_source_bytes"}


def test_historical_nomination_never_reads_or_recreates_the_source_checkout(retained, monkeypatch):
    shutil.rmtree(retained.root)
    def refuse(*args, **kwargs): pytest.fail("nomination created storage or acquired live source")
    monkeypatch.setattr(ImmutableCAS, "__init__", refuse)
    monkeypatch.setattr(ImmutableCAS, "put", refuse)
    monkeypatch.setattr(ImmutableCAS, "put_bytes", refuse)
    monkeypatch.setattr(inventory, "preflight_committed_repository", refuse)
    monkeypatch.setattr(inventory, "snapshot_chunked_repository", refuse)
    scope = nominate(retained)
    assert scope.observation()["current_source_verified"] is False
    assert scope.observation()["selected_count"] == 3
    assert not retained.root.exists()


@pytest.mark.parametrize("path", [b"external-link", b"nested", b"binary.bin", b"nul.txt", b"bad-\xff.py", b"control-\n.py", b"missing.py"])
def test_unsupported_or_absent_requested_source_refuses_the_entire_nomination(retained, path):
    with pytest.raises(ValueError):
        nominate(retained, selected_paths=hexes("instruction.md", "main.py", "smoke.py", path))


@pytest.mark.parametrize("change", ["empty", "list", "bool", "duplicate", "unsorted", "uppercase", "malformed_hex", "absolute", "parent", "dot", "git", "nul", "control", "backslash", "required_empty", "required_list", "required_foreign", "program_list", "program_foreign"])
def test_closed_selection_and_required_program_population_are_explicit(retained, change):
    kwargs = {}
    if change == "empty": kwargs["selected_paths"] = ()
    elif change == "list": kwargs["selected_paths"] = list(hexes("main.py"))
    elif change == "bool": kwargs["selected_paths"] = (True,)
    elif change == "duplicate": kwargs["selected_paths"] = (b"main.py".hex(),) * 2
    elif change == "unsorted": kwargs["selected_paths"] = tuple(reversed(hexes("main.py", "instruction.md", "smoke.py")))
    elif change == "uppercase": kwargs["selected_paths"] = (b"main.py".hex().upper(),)
    elif change == "malformed_hex": kwargs["selected_paths"] = ("not-hex",)
    elif change in {"absolute", "parent", "dot", "git", "nul", "control", "backslash"}:
        raw = {"absolute": b"/main.py", "parent": b"../main.py", "dot": b"./main.py", "git": b".git/config", "nul": b"main\x00.py", "control": b"main\n.py", "backslash": b"folder\\main.py"}[change]
        kwargs["selected_paths"] = (raw.hex(),)
    elif change == "required_empty": kwargs["required_paths"] = ()
    elif change == "required_list": kwargs["required_paths"] = list(hexes("instruction.md"))
    elif change == "required_foreign": kwargs["required_paths"] = hexes("types.pyi")
    elif change == "program_list": kwargs["program_paths"] = list(hexes("main.py"))
    else: kwargs["program_paths"] = hexes("types.pyi")
    with pytest.raises(ValueError): nominate(retained, **kwargs)


@pytest.mark.parametrize("field", ["task_id", "repository_id", "expected_commit", "expected_tree", "expected_population_cid"])
def test_independent_inventory_request_identity_cannot_be_relabelled(retained, field):
    value = getattr(retained.request, field)
    wrong = ("f" * len(value) if field in {"expected_commit", "expected_tree"} else
             cid_for_structured({"foreign": True}) if field == "expected_population_cid" else "foreign:" + value)
    with pytest.raises(ValueError): nominate(retained, request=replace(retained.request, **{field: wrong}))


@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_missing_or_corrupt_retained_source_never_returns_partial_scope(retained, change):
    cid = _chunk_cid(retained)
    path = retained.cas.path_for(cid, source=True)
    if change == "missing": path.unlink()
    else: path.write_bytes(b"corrupt")
    with pytest.raises((ValueError, FileNotFoundError)): nominate(retained)



def test_callback_mutation_of_independent_request_cannot_change_the_selected_scope(retained, monkeypatch):
    original = ImmutableCAS.get
    def changed(self, cid):
        value = original(self, cid)
        object.__setattr__(retained.request, "task_id", "changed-during-cas-callback")
        return value
    monkeypatch.setattr(ImmutableCAS, "get", changed)
    with pytest.raises(ValueError): nominate(retained)


@pytest.mark.parametrize("argument,value", [("offset", True), ("offset", -1), ("offset", 1000),
    ("limit", True), ("limit", 0), ("limit", 65), ("max_bytes", True), ("max_bytes", 2047), ("max_bytes", 65537)])
def test_frontier_paging_rejects_invalid_bounds_without_source_access(retained, argument, value):
    scope = nominate(retained)
    with pytest.raises(ValueError): scope.frontier_page(**{argument: value})


def test_small_pages_account_for_every_frontier_and_do_not_hide_excluded_files(tmp_path):
    files = {f"part_{i:03d}.py".encode(): b"VALUE = 1\n" for i in range(90)}
    files[b"instruction.md"] = SUPPORT
    case = _retain(_repository(tmp_path, files), tmp_path)
    scope = nominate(case, selected_paths=hexes("part_000.py", "instruction.md"),
        required_paths=hexes("instruction.md"), program_paths=hexes("part_000.py"))
    observed = scope.observation()
    assert observed["inventory_entry_count"] == 91 and observed["selected_count"] == 2 and observed["frontier_count"] == 89
    rows = frontier(scope, max_bytes=2048)
    assert len(rows) == 89 and len({r["raw_path_hex"] for r in rows}) == 89
    assert sum(observed["frontier_status_counts"].values()) == 89
    for offset in (0, 1, 89):
        page = scope.frontier_page(offset=offset, limit=1, max_bytes=2048)
        assert len(page["entries"]) <= 1


def test_file_count_cap_counts_required_support_and_never_silently_truncates(tmp_path):
    files = {f"part_{i:02d}.py".encode(): b"VALUE = 1\n" for i in range(64)}
    files[b"instruction.md"] = SUPPORT
    case = _retain(_repository(tmp_path, files), tmp_path)
    selected = hexes("instruction.md", *(f"part_{i:02d}.py" for i in range(63)))
    scope = nominate(case, selected_paths=selected, required_paths=hexes("instruction.md"), program_paths=hexes("part_00.py"))
    assert scope.observation()["selected_count"] == 64 and scope.observation()["frontier_count"] == 1
    with pytest.raises(ValueError):
        nominate(case, selected_paths=hexes(*files), required_paths=hexes("instruction.md"), program_paths=hexes("part_00.py"))


def test_original_per_path_bytes_and_exact_semantic_byte_boundary(tmp_path):
    root = _repository(tmp_path, {b"main.py": b"a" * 500_000, b"copy.py": b"a" * 500_000, b"extra.py": b"x"})
    case = _retain(root, tmp_path, limits=inventory.SourceInventoryLimits(frame_bytes=1_000_000))
    scope = nominate(case, selected_paths=hexes("main.py", "copy.py"), required_paths=hexes("copy.py"), program_paths=hexes("main.py"))
    assert scope.observation()["selected_bytes"] == 1_000_000
    with pytest.raises(ValueError):
        nominate(case, selected_paths=hexes("main.py", "copy.py", "extra.py"), required_paths=hexes("copy.py"), program_paths=hexes("main.py"))


def test_oversized_unselected_file_is_an_explicit_frontier_and_cannot_be_selected(tmp_path):
    case = _retain(_repository(tmp_path, {b"main.py": b"VALUE=1\n", b"instruction.md": SUPPORT,
        b"oversized.txt": b"x" * 1_000_001}), tmp_path,
        limits=inventory.SourceInventoryLimits(frame_bytes=1_000_000))
    scope = nominate(case, selected_paths=hexes("main.py", "instruction.md"), required_paths=hexes("instruction.md"), program_paths=hexes("main.py"))
    assert frontier(scope)[0]["status"] == "source_exceeds_task_byte_bound"
    with pytest.raises(ValueError):
        nominate(case, selected_paths=hexes("main.py", "instruction.md", "oversized.txt"), required_paths=hexes("instruction.md"), program_paths=hexes("main.py"))


def test_explicit_empty_program_partition_keeps_support_only_roles(retained):
    scope = nominate(retained, program_paths=())
    assert scope.semantic_inputs()["program_paths"] == []
    assert all(row["role"] == "support" for row in scope.observation()["selected"])
    assert scope.semantic_inputs()["required_raw_paths"] == ["instruction.md", "smoke.py"]


def test_frontier_identity_too_large_for_one_page_refuses_without_truncation(tmp_path):
    long_path = "/".join(["d" * 180] * 6) + "/remaining.py"
    case = _retain(_repository(tmp_path, {b"main.py": b"VALUE=1\n", b"instruction.md": SUPPORT,
        long_path.encode(): b"VALUE=2\n"}), tmp_path)
    scope = nominate(case, selected_paths=hexes("main.py", "instruction.md"),
        required_paths=hexes("instruction.md"), program_paths=hexes("main.py"))
    with pytest.raises(ValueError): scope.frontier_page(max_bytes=2048)
    page = scope.frontier_page(max_bytes=16384)
    assert page["entries"][0]["path"] == long_path and page["exhausted"] is True


def test_nomination_inputs_reuse_semantic_producer_and_current_reader_without_broadening(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews
    files = {f"part_{i:02d}.py".encode(): f"VALUE_{i} = {i}\n".encode() for i in range(70)}
    files[b"instruction.md"] = SUPPORT
    root = _repository(tmp_path, files)
    case = _retain(root, tmp_path)
    scope = nominate(case, selected_paths=hexes("part_00.py", "instruction.md"),
        required_paths=hexes("instruction.md"), program_paths=hexes("part_00.py"))
    output = root / ".runtime/semantic"
    semantic = prepare_semantic_context(repository=root, output=output,
        objective="Inspect the explicitly selected public source", task_id=case.request.task_id,
        **scope.semantic_inputs())
    payload = json.loads((output / "worker-context.json").read_text())
    assert set(payload["manifest"]) == {"part_00.py", "instruction.md"}
    assert payload["program_paths"] == ["part_00.py"]
    assert scope.observation()["inventory_entry_count"] == 71 and scope.observation()["frontier_count"] == 69
    workspace = tmp_path / "allocated"
    _git(root, "worktree", "add", "--detach", "-q", str(workspace))
    # The production reader independently checks current workspace bytes. Its
    # semantic pins are explicit authored host metadata for this reader seam.
    kwargs = dict(repository=root, workspace=workspace,
        source_hashes={row["path"]: row["source_sha256"] for row in scope.observation()["selected"]},
        task_id=case.request.task_id, scope_cid="scope:authored-inventory-nomination",
        semantic_root_cid="semantic:authored-inventory-nomination", semantic_sha256="a"*64,
        native_prompt_sha256="b"*64, source_manifest_sha256="c"*64)
    reader = WorkerSourceViews(**kwargs)
    page = reader.discovery(max_bytes=16384)
    assert {row["path"] for row in page["rows"]} == {"part_00.py", "instruction.md"}
    with pytest.raises(ValueError): reader.page(path="part_01.py", view="original", cursor=None, max_bytes=8192)
    (workspace / "part_00.py").write_bytes(b"VALUE_0 = 999\n")
    with pytest.raises(ValueError): reader.discovery(max_bytes=16384)
    assert scope.observation()["current_source_verified"] is False


def test_selected_hex_length_refuses_before_decode_or_cas_acquisition(retained, monkeypatch):
    monkeypatch.setattr(scopes, "_raw_path", lambda *a: pytest.fail("oversized selected hex was decoded"))
    monkeypatch.setattr(scopes, "load_source_inventory", lambda **k: pytest.fail("oversized selection acquired CAS"))
    with pytest.raises(ValueError):
        nominate(retained, selected_paths=("00" * 4097,))


def test_oversized_frontier_count_refuses_before_frontier_json_parsing(retained, monkeypatch):
    scope = nominate(retained)
    original = scopes._parse
    calls = []
    def parsed(raw):
        calls.append(raw)
        return original(raw)
    monkeypatch.setattr(scopes, "_parse", parsed)
    with pytest.raises(ValueError):
        scopes.RepositorySourceScope(scope._observation_bytes,
            (b"authored invalid row must not be parsed",) * (retained.limits.max_entries + 1))
    assert calls == [scope._observation_bytes]


def test_unicode_crlf_empty_source_and_partition_identities_remain_exact(tmp_path):
    path = "caf\u00e9.py"
    raw = 'VALUE = "\u03bb"\r\n'.encode()
    case = _retain(_repository(tmp_path, {path.encode(): raw, b"instruction.md": b""}), tmp_path)
    scope = nominate(case, selected_paths=hexes(path, "instruction.md"),
        required_paths=hexes("instruction.md"), program_paths=hexes(path))
    observed = scope.observation()
    selected = {row["path"]: row for row in observed["selected"]}
    assert selected[path]["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert selected[path]["source_cid"] == cid_for_bytes(raw)
    assert selected["instruction.md"]["source_bytes"] == 0
    assert selected["instruction.md"]["source_cid"] == cid_for_bytes(b"")
    assert scope.semantic_inputs()["paths"] == sorted([path, "instruction.md"])
    other = nominate(case, selected_paths=hexes(path, "instruction.md"),
        required_paths=hexes(path, "instruction.md"), program_paths=())
    assert other.observation()["scope_cid"] != observed["scope_cid"]
    assert other.observation()["source_inventory_cid"] == observed["source_inventory_cid"]
    assert other.observation()["frontier_cid"] == observed["frontier_cid"]
    assert scope.frontier_page()["entries"] == [] and scope.frontier_page()["exhausted"] is True

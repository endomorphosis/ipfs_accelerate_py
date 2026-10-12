"""Real worktree, exact byte, freshness and pagination tests; no provider calls."""
import base64
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_views as module
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def case(tmp_path):
    root, workspace = tmp_path / "canonical", tmp_path / "allocated"
    root.mkdir()
    sources = {
        "pkg/sample.py": ("# SPDX-License-Identifier: MIT\r\n"
            + 'UNICODE = "café λ"                 \r\n'
            + "".join(f'VALUE_{i} = {i}                 \r\n' for i in range(100))).encode(),
        "other.py": b"VALUE = 2\n",
        "empty.py": b"",
        "binary.dat": b"\xff\0\x80\xfe\r\n" * 500,
    }
    for name, raw in sources.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(raw)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Authored Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    git(root, "worktree", "add", "--detach", "-q", str(workspace))
    kwargs = dict(repository=root, workspace=workspace, source_hashes={k: sha(v) for k, v in sources.items()},
        task_id="authored-task", scope_cid="scope:authored", semantic_root_cid="semantic:authored",
        semantic_sha256="a" * 64, native_prompt_sha256="b" * 64, source_manifest_sha256="c" * 64)
    return dict(root=root, workspace=workspace, sources=sources, kwargs=kwargs)


def service(case, **kwargs):
    return module.WorkerSourceViews(**{**case["kwargs"], **kwargs})


def decode(page):
    raw = page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"], validate=True)
    assert sha(raw) == page["slice_sha256"]
    assert len(raw) == page["end_offset"] - page["offset"]
    return raw


def pages(svc, **kwargs):
    values, offset, cursor = [], 0, None
    for _ in range(100):
        result = svc.page(**kwargs, cursor=cursor)
        assert len(module.canonical_json(result)) <= kwargs.get("max_bytes", 8192)
        assert result["offset"] == offset
        values.append(result)
        offset = result["end_offset"]
        cursor = result["next_cursor"]
        assert result["exhausted"] is (cursor is None)
        if cursor is None:
            break
        assert result["end_offset"] > result["offset"]
    else:
        pytest.fail("pagination did not terminate")
    return b"".join(map(decode, values)), values


def test_compact_navigation_then_pinned_original_expansion_preserves_unicode_crlf_and_all_bytes(case):
    svc = service(case)
    source = case["sources"]["pkg/sample.py"]
    compact, compact_pages = pages(svc, path="pkg/sample.py", max_bytes=2300)
    assert len(compact_pages) > 1
    assert compact == build_source_view(source).view and len(compact) < len(source)
    first = compact_pages[0]
    original, original_pages = pages(svc, path="pkg/sample.py", view="original", max_bytes=2300,
        expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"])
    assert original == source
    assert all(row["representation_sha256"] == sha(source) for row in original_pages)
    assert all(row["representation_sha256"] == sha(compact) for row in compact_pages)
    assert (case["root"] / "pkg/sample.py").read_bytes() == source
    assert (case["workspace"] / "pkg/sample.py").read_bytes() == source
    for row in compact_pages + original_pages:
        assert row["binding"] == {key: case["kwargs"][key] for key in (
            "task_id", "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256",
            "source_manifest_sha256")} | {"profile": "python-whitespace@1"}
        assert row["proof_authority"] is False and row["mutation_authority"] is False
        assert row["omission_authority"] is False and row["source_obligations_discharged"] == 0
        assert not {"pieces", "codec_binding", "original", "restoration"} & set(row)


@pytest.mark.parametrize("name", ["empty.py", "binary.dat"])
@pytest.mark.parametrize("view", ["compact", "original"])
def test_empty_or_non_utf8_identity_fallback_has_lossless_slices(case, name, view):
    svc = service(case)
    result, rows = pages(svc, path=name, view=view, max_bytes=2400)
    assert result == case["sources"][name]
    assert rows[-1]["exhausted"] is True
    if name.endswith(".dat"):
        assert all(row["effective_profile"] == "identity@1" and row["fallback_reason"] == "non_python_identity"
            for row in rows)
        assert any(row["encoding"] == "base64" for row in rows)


@pytest.mark.parametrize("source,reason", [(b"not valid !!!\n", "invalid_python"),
    (b'VALUE = f"{1}"  \n', "unsupported_fstrings")])
def test_unsupported_python_profile_returns_exact_explicit_identity(case, source, reason):
    for root in (case["root"], case["workspace"]):
        (root / "other.py").write_bytes(source)
    case["kwargs"]["source_hashes"]["other.py"] = sha(source)
    result = service(case).page(path="other.py")
    assert decode(result) == source and result["effective_profile"] == "identity@1"
    assert result["fallback_reason"] == reason


def test_opt_in_comment_profile_does_not_remove_protected_contract_or_license(case):
    source = b"# SPDX-License-Identifier: MIT\nVALUE = 1  # ordinary explanation\n# must preserve contract\n"
    for root in (case["root"], case["workspace"]):
        (root / "other.py").write_bytes(source)
    case["kwargs"]["source_hashes"]["other.py"] = sha(source)
    result = service(case, profile="python-comments@1").page(path="other.py")
    value = decode(result)
    assert b"ordinary explanation" not in value
    assert b"SPDX-License-Identifier" in value and b"must preserve contract" in value
    assert result["semantic_equivalence_claimed"] is False


def test_discovery_covers_exact_population_with_no_source_bodies_or_maps(case):
    svc = service(case)
    rows, cursor = [], None
    while True:
        page = svc.discovery(cursor=cursor, max_bytes=1100)
        assert len(module.canonical_json(page)) <= 1100
        assert page["offset"] == len(rows)
        rows.extend(page["rows"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert [row["path"] for row in rows] == sorted(case["sources"])
    assert all(row["source_sha256"] == sha(case["sources"][row["path"]]) for row in rows)
    assert all(not {"content", "pieces", "original"} & set(row) for row in rows)


@pytest.mark.parametrize("name", ["../other.py", "/etc/passwd", "pkg/../other.py", "pkg//sample.py",
    "pkg/./sample.py", ".git/config", "other.py/", "pkg\\sample.py", "other.py\n", "", "unknown.py"])
def test_unadmitted_and_noncanonical_paths_refuse(case, name):
    with pytest.raises(ValueError):
        service(case).page(path=name)


@pytest.mark.parametrize("bound", [True, False, 0, -1, 1023, 65537, 8192.0, "8192"])
def test_page_byte_bound_is_exact_bounded_integer(case, bound):
    with pytest.raises(ValueError):
        service(case).page(path="other.py", max_bytes=bound)


def test_too_small_full_envelope_refuses_without_dropping_source(case):
    svc = service(case)
    with pytest.raises(ValueError, match="fit|bound"):
        svc.page(path="pkg/sample.py", max_bytes=1024)
    assert not svc._continuations


@pytest.mark.parametrize("pins", [{"expected_source_sha256": "0" * 64},
    {"expected_view_sha256": "0" * 64},
    {"expected_source_sha256": "0" * 64, "expected_view_sha256": "0" * 64},
    {"expected_source_sha256": True, "expected_view_sha256": "0" * 64}])
def test_original_expansion_refuses_missing_or_foreign_pin_pair(case, pins):
    with pytest.raises(ValueError, match="pins"):
        service(case).page(path="other.py", view="original", **pins)


def test_issued_cursor_cannot_be_replayed_or_shared_with_another_session(case):
    svc = service(case)
    first = svc.page(path="pkg/sample.py", max_bytes=2300)
    cursor = first["next_cursor"]
    assert cursor
    with pytest.raises(ValueError, match="foreign|forged|consumed"):
        service(case).page(path="pkg/sample.py", max_bytes=2300, cursor=cursor)
    second = svc.page(path="pkg/sample.py", max_bytes=2300, cursor=cursor)
    assert second["offset"] == first["end_offset"]
    with pytest.raises(ValueError, match="consumed"):
        svc.page(path="pkg/sample.py", max_bytes=2300, cursor=cursor)


@pytest.mark.parametrize("change", ["path", "view", "budget", "offset", "bool", "token", "digest", "extra", "schema"])
def test_cursor_is_bound_to_exact_representation_query_and_issued_boundary(case, change):
    svc = service(case)
    cursor = deepcopy(svc.page(path="pkg/sample.py", max_bytes=2300)["next_cursor"])
    query = dict(path="pkg/sample.py", max_bytes=2300, cursor=cursor)
    if change == "path":
        query["path"] = "other.py"
    elif change == "view":
        query["view"] = "original"
    elif change == "budget":
        query["max_bytes"] += 1
    elif change == "offset":
        cursor["offset"] += 1
    elif change == "bool":
        cursor["offset"] = True
    elif change == "token":
        cursor["token"] = "f" * 32
    elif change == "digest":
        cursor["request_sha256"] = "f" * 64
    elif change == "extra":
        cursor["extra"] = "ignored?"
    else:
        cursor["schema"] = []
    with pytest.raises(ValueError):
        svc.page(**query)


@pytest.mark.parametrize("root_key", ["root", "workspace"])
@pytest.mark.parametrize("path", ["pkg/sample.py", "other.py"])
def test_drift_of_requested_or_other_admitted_source_refuses_entire_session(case, root_key, path):
    svc = service(case)
    (case[root_key] / path).write_bytes(b"CHANGED = True\n")
    with pytest.raises(ValueError, match="stale"):
        svc.page(path="pkg/sample.py")
    with pytest.raises(ValueError, match="stale"):
        svc.discovery()


def test_source_changed_during_projection_refuses_before_cursor_is_consumed(case, monkeypatch):
    svc = service(case)
    cursor = svc.page(path="pkg/sample.py", max_bytes=2300)["next_cursor"]
    check, calls = svc._check_sources, []
    def changed():
        result = check()
        calls.append(1)
        if len(calls) == 1:
            (case["workspace"] / "other.py").write_bytes(b"CHANGED = True\n")
        return result
    monkeypatch.setattr(svc, "_check_sources", changed)
    with pytest.raises(ValueError, match="stale"):
        svc.page(path="pkg/sample.py", max_bytes=2300, cursor=cursor)
    assert cursor["token"] in svc._continuations


@pytest.mark.parametrize("kind", ["leaf_symlink", "parent_symlink", "hardlink", "fifo"])
@pytest.mark.parametrize("root_key", ["root", "workspace"])
def test_unsafe_source_kinds_refuse_before_any_slice(case, tmp_path, root_key, kind):
    svc = service(case)
    path = case[root_key] / "pkg/sample.py"
    if kind == "parent_symlink":
        target = tmp_path / "moved"
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    else:
        original = path.read_bytes()
        path.unlink()
        if kind == "leaf_symlink":
            target = tmp_path / "external.py"
            target.write_bytes(original)
            path.symlink_to(target)
        elif kind == "hardlink":
            target = tmp_path / "external.py"
            target.write_bytes(original)
            os.link(target, path)
        else:
            os.mkfifo(path)
    with pytest.raises(ValueError):
        svc.page(path="pkg/sample.py")


def test_detached_parent_descriptor_is_not_accepted_after_pathname_replacement(case, monkeypatch):
    svc = service(case)
    old_read, changed = module.os.read, []
    def swap(fd, count):
        raw = old_read(fd, count)
        target = os.readlink(f"/proc/self/fd/{fd}")
        if target == str(case["root"] / "pkg/sample.py") and not changed:
            old = case["root"] / "detached"
            (case["root"] / "pkg").rename(old)
            (case["root"] / "pkg").mkdir()
            (case["root"] / "pkg/sample.py").write_bytes(case["sources"]["pkg/sample.py"])
            changed.append(True)
        return raw
    monkeypatch.setattr(module.os, "read", swap)
    with pytest.raises(ValueError, match="parent pathname"):
        svc.page(path="pkg/sample.py")
    assert changed


def test_foreign_worktree_or_same_canonical_root_refuses(case, tmp_path):
    with pytest.raises(ValueError):
        service(case, workspace=case["root"])
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    git(foreign, "init", "-q")
    with pytest.raises(ValueError):
        service(case, workspace=foreign)


@pytest.mark.parametrize("field", ["profile", "hashes", "codec"])
def test_admission_and_codec_drift_refuse(case, monkeypatch, field):
    svc = service(case)
    if field == "profile":
        svc._profile = "python-comments@1"
    elif field == "hashes":
        svc._hashes["other.py"] = "0" * 64
    else:
        old = module.current_codec_binding
        monkeypatch.setattr(module, "current_codec_binding", lambda: {**old(), "codec_version": 999})
    with pytest.raises(ValueError, match="admission|codec"):
        svc.page(path="pkg/sample.py")


def test_constructor_copies_caller_map_and_returned_admission_has_no_mutable_authority(case):
    svc = service(case)
    case["kwargs"]["source_hashes"]["other.py"] = "0" * 64
    admission = svc.admission()
    admission["task_id"] = "foreign"
    assert svc.page(path="other.py")["binding"]["task_id"] == "authored-task"


def test_source_and_continuation_populations_are_bounded(case, monkeypatch):
    with pytest.raises(ValueError, match="bounded"):
        service(case, source_hashes={f"file_{i}.py": "a" * 64 for i in range(65)})
    svc = service(case)
    monkeypatch.setattr(module, "MAX_CONTINUATIONS", 1)
    first = svc.page(path="pkg/sample.py", max_bytes=2300)
    with pytest.raises(ValueError, match="population"):
        svc.page(path="pkg/sample.py", max_bytes=2300)
    assert svc.page(path="pkg/sample.py", max_bytes=2300, cursor=first["next_cursor"])["offset"] > 0


def test_oversized_source_scope_refuses_even_when_each_file_is_within_limit(case):
    hashes = dict(case["kwargs"]["source_hashes"])
    for name in ("first.dat", "second.dat"):
        raw = b"x" * 600000
        for root in (case["root"], case["workspace"]):
            (root / name).write_bytes(raw)
        hashes[name] = sha(raw)
    with pytest.raises(ValueError, match="scope exceeds"):
        service(case, source_hashes=hashes)

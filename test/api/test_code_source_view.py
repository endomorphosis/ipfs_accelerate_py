"""Real canonical-source edit cycles; all sources and commands are authored."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import code_source_view as views


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "calc.py").write_bytes(
        b"# Keep copyright and ordinary source comments.\n"
        b"VALUE = 2\n\n"
        b"def add(value):\n"
        b"    '''Public docstring stays in compact representations.'''\n"
        b"    # Deliberately verbose implementation commentary.\n"
        b"    return value + VALUE\n\n"
        b"# This untouched trailing comment stays byte-identical.\n"
        b"def untouched():\n"
        b"    return 'still here'\n")
    (root / "consumer.py").write_text("from calc import add\nprint(add(3))\n")
    return root


def invoke(repo, *args, audit=None):
    command = [sys.executable, views.__file__, "--root", str(repo)]
    if audit is not None:
        command.extend(["--audit-store", str(audit)])
    return subprocess.run(command + list(args), capture_output=True, check=False, timeout=10)


def assert_error(code, action):
    with pytest.raises(views.SourceViewError) as caught:
        action()
    assert caught.value.code == code


def test_read_compact_preserves_identifiers_literals_docstring_and_ast(repo):
    helper = views.CodeSourceView(repo)
    raw = helper.read("calc.py", "add", "raw")
    compact = helper.read("calc.py", "add", "compact")
    assert compact["content_bytes"] < raw["content_bytes"]
    assert "VALUE" in compact["content"]
    assert "Public docstring" in compact["content"]
    assert "verbose implementation" not in compact["content"]
    assert ast.dump(ast.parse(raw["content"])) == ast.dump(ast.parse(compact["content"]))
    assert compact["operation_counts"] == {"read": 2, "read_raw": 1, "read_compact": 1}
    assert compact["behavioral_equivalence_verified"] is False
    assert compact["symbol_id"] == "calc.py::add"


def test_unchanged_compact_round_trip_preserves_exact_original_bytes(repo):
    original = (repo / "calc.py").read_bytes()
    helper = views.CodeSourceView(repo)
    compact = helper.read("calc.py", "add")
    result = helper.replace("calc.py", "add", compact["source_sha256"], compact["content"])
    assert result["changed"] is False
    assert result["semantic_no_op"] is True
    assert result["result_sha256"] == sha(original)
    assert (repo / "calc.py").read_bytes() == original
    assert sorted(p.name for p in repo.iterdir()) == ["calc.py", "consumer.py"]


def test_edit_refresh_cross_file_behavior_and_untouched_bytes(repo):
    original = (repo / "calc.py").read_bytes()
    consumer = (repo / "consumer.py").read_bytes()
    helper = views.CodeSourceView(repo)
    before = helper.read("calc.py", "add")
    result = helper.replace("calc.py", "add", before["source_sha256"],
                            "def add(value):\n    return value + VALUE + 10")
    assert result["changed"] and result["untouched_bytes_preserved"]
    assert result["refreshed_view"]["source_sha256"] == result["result_sha256"]
    assert result["refreshed_view"]["symbol_id"] == before["symbol_id"]
    after = (repo / "calc.py").read_bytes()
    assert after.startswith(original[:original.index(b"def add")])
    assert after[after.index(b"\n\n# This untouched"):] == original[original.index(b"\n\n# This untouched"):]
    assert (repo / "consumer.py").read_bytes() == consumer
    run = subprocess.run([sys.executable, "consumer.py"], cwd=repo, capture_output=True,
                         check=True, timeout=10)
    assert run.stdout == b"15\n"
    assert_error("stale_source", lambda: helper.replace("calc.py", "add", before["source_sha256"],
                                                        "def add(value): return -1"))


def test_cli_read_edit_and_usage_receipts(repo, tmp_path):
    audit = tmp_path / "operations.jsonl"
    read = invoke(repo, "read", "calc.py", "--symbol", "add", "--format", "compact", audit=audit)
    assert read.returncode == 0, read.stderr
    result = json.loads(read.stdout)
    assert result["response_bytes"] == len(read.stdout)
    (repo / "replacement.txt").write_text("def add(value):\n    return value + 100\n")
    edit = invoke(repo, "replace", "calc.py", "--symbol", "add", "--expected-sha256",
                  result["source_sha256"], "--replacement-file", "replacement.txt", audit=audit)
    assert edit.returncode == 0, edit.stderr
    receipt = json.loads(edit.stdout)
    assert receipt["response_bytes"] == len(edit.stdout)
    usage = views.summarize_audit_store(audit, expected_root=repo)
    assert usage["status"] == "observed"
    assert usage["operation_counts"] == {"read": 1, "replace": 1}
    assert usage["format_counts"] == {"compact": 1}
    assert usage["changed_edits"] == 1
    assert usage["response_bytes"] == len(read.stdout) + len(edit.stdout)
    assert "Public docstring" not in audit.read_text()
    assert "calc.py" not in audit.read_text()
    stale = invoke(repo, "replace", "calc.py", "--symbol", "add", "--expected-sha256",
                   result["source_sha256"], "--replacement-file", "replacement.txt")
    assert stale.returncode == 2 and b"stale_source" in stale.stderr
    assert stale.stdout == b""


@pytest.mark.parametrize("path", ["/etc/passwd", "../calc.py", "sub/../calc.py", "./calc.py",
                                      ".hidden.py", "sub/.secret.py", "calc.py/", "a//b.py",
                                      "a\\b.py", "calc.py\n", "source.txt"])
def test_explicit_visible_paths_only(repo, path):
    with pytest.raises(views.SourceViewError):
        views.CodeSourceView(repo).read(path)


def test_symlink_file_and_directory_and_hardlink_rejected(repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 'authored sensitive marker'\n")
    (repo / "link.py").symlink_to(outside / "secret.py")
    (repo / "linked_dir").symlink_to(outside, target_is_directory=True)
    os.link(outside / "secret.py", repo / "hardlink.py")
    for path in ("link.py", "linked_dir/secret.py", "hardlink.py"):
        with pytest.raises(views.SourceViewError):
            views.CodeSourceView(repo).read(path)
    linked_root = tmp_path / "linked_root"
    linked_root.symlink_to(repo, target_is_directory=True)
    assert_error("invalid_root", lambda: views.CodeSourceView(linked_root))


@pytest.mark.parametrize("snippet,code", [
    ("def different(value): return 0", "replacement_symbol_mismatch"),
    ("class add: pass", "replacement_symbol_mismatch"),
    ("def add(value): return 0\nEXTRA = True", "replacement_symbol_mismatch"),
    ("def add(:", "invalid_python"),
    ("not python source!", "invalid_python"),
])
def test_invalid_replacement_never_writes(repo, snippet, code):
    original = (repo / "calc.py").read_bytes()
    assert_error(code, lambda: views.CodeSourceView(repo).replace("calc.py", "add", sha(original), snippet))
    assert (repo / "calc.py").read_bytes() == original
    assert not list(repo.glob(".source-view-*.tmp"))


def test_module_edit_unsupported_and_no_arbitrary_execution(repo):
    assert_error("unsupported_module_edit", lambda: views.CodeSourceView(repo).replace(
        "calc.py", "module", sha((repo / "calc.py").read_bytes()), "pass"))
    (repo / "danger.py").write_text("raise RuntimeError('executed!')\ndef public(): return 1\n")
    assert views.CodeSourceView(repo).read("danger.py")["ast_equivalent"]


def test_decorators_unicode_crlf_and_mode_preserved(repo):
    source = ("# π untouched\r\n"
              "def decorate(fn): return fn\r\n\r\n"
              "@decorate\r\n"
              "def café(value):\r\n"
              "    return value + 'é'  # trailing comment retained\r\n\r\n"
              "# ω untouched\r\n").encode()
    path = repo / "unicode.py"
    path.write_bytes(source)
    path.chmod(0o751)
    helper = views.CodeSourceView(repo)
    compact = helper.read("unicode.py", "café")
    helper.replace("unicode.py", "café", compact["source_sha256"], compact["content"])
    assert path.read_bytes() == source
    receipt = helper.replace("unicode.py", "café", compact["source_sha256"],
                             "@decorate\ndef café(value):\n    return value + 'ø'")
    assert receipt["changed"]
    assert path.read_bytes() == source.replace("'é'".encode(), "'ø'".encode())
    assert path.stat().st_mode & 0o777 == 0o751


def test_utf8_bom_roundtrip_and_edit(repo):
    original = b"\xef\xbb\xbf# retained\ndef f(): return 1\n"
    (repo / "bom.py").write_bytes(original)
    helper = views.CodeSourceView(repo)
    before = helper.read("bom.py", "f")
    helper.replace("bom.py", "f", before["source_sha256"], "def f(): return 2")
    assert (repo / "bom.py").read_bytes() == original.replace(b"return 1", b"return 2")


def test_direct_method_edit_preserves_class_and_other_method(repo):
    original = "# top\nclass Example:\n    def value(self):\n        return 1\n\n    def other(self): return 4\n"
    (repo / "classy.py").write_text(original)
    helper = views.CodeSourceView(repo)
    read = helper.read("classy.py", "Example.value")
    helper.replace("classy.py", "Example.value", read["source_sha256"], "def value(self):\n    return 2")
    assert (repo / "classy.py").read_text() == original.replace("return 1", "return 2")


def test_multiline_literal_dedent_cannot_silently_change_values(repo):
    (repo / "classy.py").write_text('class C:\n    def f(self):\n        return """first\n        second"""\n')
    assert_error("unsupported_symbol_layout", lambda: views.CodeSourceView(repo).read("classy.py", "C.f"))
    # A full module view still supports a conservative, AST-checked read.
    assert views.CodeSourceView(repo).read("classy.py")["ast_equivalent"]


def test_method_replacement_with_indentation_sensitive_literal_rejected(repo):
    path = repo / "classy.py"
    path.write_text("class C:\n    def f(self): return 1\n")
    original = path.read_bytes()
    assert_error("replacement_context_mismatch", lambda: views.CodeSourceView(repo).replace(
        "classy.py", "C.f", sha(original), 'def f(self):\n    return """first\n    second"""'))
    assert path.read_bytes() == original


def test_ambiguous_redefinitions_and_nested_functions_rejected(repo):
    (repo / "ambiguous.py").write_text("def f(): return 1\ndef f(): return 2\n")
    helper = views.CodeSourceView(repo)
    assert helper.index(["ambiguous.py"])["files"][0]["symbols"][0]["ambiguous"]
    assert_error("ambiguous_symbol", lambda: helper.read("ambiguous.py", "f"))
    (repo / "nested.py").write_text("def outer():\n    def inner(): return 1\n    return inner()\n")
    assert_error("unsupported_or_missing_symbol", lambda: helper.read("nested.py", "outer.inner"))


def test_duplicate_class_ancestor_rejected(repo):
    (repo / "ambiguous.py").write_text("class C:\n    def f(self): return 1\nclass C: pass\n")
    assert_error("ambiguous_symbol", lambda: views.CodeSourceView(repo).read("ambiguous.py", "C.f"))


def test_type_ignore_compact_falls_back_without_losing_comment(repo):
    source = "# retained header\n\nvalue = 1  # type: ignore[assignment]\n"
    (repo / "typed.py").write_text(source)
    read = views.CodeSourceView(repo).read("typed.py")
    assert read["format"] == "raw"
    assert read["fallback_reason"] == "ast_mismatch"
    assert read["content"] == source


def test_source_and_output_bounds(repo, monkeypatch):
    monkeypatch.setattr(views, "MAX_SOURCE_BYTES", 32)
    assert_error("source_limit", lambda: views.CodeSourceView(repo).read("calc.py"))
    monkeypatch.setattr(views, "MAX_SOURCE_BYTES", 1024 * 1024)
    monkeypatch.setattr(views, "MAX_CONTENT_BYTES", 8)
    assert_error("view_limit", lambda: views.CodeSourceView(repo).read("calc.py"))


def test_external_edit_during_precommit_is_detected(repo, monkeypatch):
    helper = views.CodeSourceView(repo)
    original = (repo / "calc.py").read_bytes()
    original_read = views._read_fd
    count = 0
    def racing_read(parent, name):
        nonlocal count
        count += 1
        if count == 2:
            (repo / "calc.py").write_bytes(original + b"# concurrent external change\n")
        return original_read(parent, name)
    monkeypatch.setattr(views, "_read_fd", racing_read)
    assert_error("stale_source", lambda: helper.replace("calc.py", "add", sha(original), "def add(value): return 9"))
    assert (repo / "calc.py").read_bytes() == original + b"# concurrent external change\n"
    assert not list(repo.glob(".source-view-*.tmp"))


def test_missing_audit_and_wrong_root_fail_without_source_leak(repo, tmp_path):
    audit = tmp_path / "audit.jsonl"
    assert views.summarize_audit_store(audit, expected_root=repo)["status"] == "not_observed"
    views.CodeSourceView(repo, audit_store=audit).read("calc.py")
    assert views.summarize_audit_store(audit, expected_root=tmp_path)["status"] == "invalid"


@pytest.mark.parametrize("mutation", ["truncated", "duplicate", "unknown_field", "negative_bytes", "bool_bytes", "content_leak"])
def test_corrupt_receipts_rejected(repo, tmp_path, mutation):
    audit = tmp_path / "audit.jsonl"
    views.CodeSourceView(repo, audit_store=audit).read("calc.py")
    payload = audit.read_text()
    row = json.loads(payload)
    if mutation == "truncated":
        audit.write_text(payload.rstrip())
    elif mutation == "duplicate":
        audit.write_text(payload + payload)
    else:
        if mutation == "unknown_field": row["surprise"] = "source"
        elif mutation == "negative_bytes": row["response_bytes"] = -1
        elif mutation == "bool_bytes": row["response_bytes"] = True
        elif mutation == "content_leak": row["source_sha256"] = "secret source"
        audit.write_text(json.dumps(row) + "\n")
    receipt = views.summarize_audit_store(audit, expected_root=repo)
    assert receipt["status"] == "invalid"
    assert receipt["operations"] is None
    assert "secret source" not in json.dumps(receipt)


def test_audit_symlink_and_size_cap_do_not_mask_committed_edit(repo, tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.write_text("retained")
    link = tmp_path / "linked_audit"
    link.symlink_to(target)
    helper = views.CodeSourceView(repo, audit_store=link)
    assert helper.read("calc.py")["audit_status"] == "unavailable"
    assert target.read_text() == "retained"
    assert views.summarize_audit_store(link, expected_root=repo)["status"] == "invalid"
    monkeypatch.setattr(views, "MAX_AUDIT_BYTES", 1)
    helper = views.CodeSourceView(repo, audit_store=tmp_path / "full.jsonl")
    source = (repo / "calc.py").read_bytes()
    receipt = helper.replace("calc.py", "add", sha(source), "def add(value): return 9")
    assert receipt["changed"] and receipt["audit_status"] == "capacity_reached"
    assert "return 9" in (repo / "calc.py").read_text()


def test_direct_cli_is_standard_library_only(repo):
    # -I excludes the repository and user site from import resolution.
    run = subprocess.run([sys.executable, "-I", views.__file__, "--root", str(repo),
                          "index", "calc.py", "consumer.py"], capture_output=True, timeout=10)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert len(result["files"]) == 2
    assert result["response_bytes"] == len(run.stdout)


@pytest.mark.parametrize("format", ["raw", "compact"])
def test_replace_refresh_honors_selected_format(repo, format):
    helper = views.CodeSourceView(repo)
    read = helper.read("calc.py", "add", format)
    receipt = helper.replace("calc.py", "add", read["source_sha256"],
                             "def add(value):\n    # Comment in intentional edit.\n    return value + 8",
                             format=format)
    assert receipt["refreshed_view"]["requested_format"] == format
    assert receipt["refreshed_view"]["format"] == format
    assert ("Comment in intentional" in receipt["refreshed_view"]["content"]) == (format == "raw")


def test_index_does_not_overpromise_unsupported_edits(repo):
    (repo / "mixed.py").write_text('class C:\n    def f(self):\n        return """a\n        b"""\n'
                                    'class D:\n    def g(self): return 1\nclass D: pass\n')
    rows = {row["symbol"]: row for row in views.CodeSourceView(repo).index(["mixed.py"])["files"][0]["symbols"]}
    assert rows["C.f"]["edit_supported"] is False
    assert rows["C.f"]["unsupported_reason"] == "unsupported_symbol_layout"
    assert rows["D.g"]["edit_supported"] is False
    assert rows["D.g"]["ambiguous"]


def test_parallel_same_hash_edits_have_one_winner(repo):
    helper = views.CodeSourceView(repo)
    expected = helper.read("calc.py", "add")["source_sha256"]
    commands = []
    for index in range(8):
        snippet = f"replacement{index}.txt"
        (repo / snippet).write_text(f"def add(value): return value + {index + 100}\n")
        commands.append([sys.executable, views.__file__, "--root", str(repo), "replace", "calc.py",
                         "--symbol", "add", "--expected-sha256", expected,
                         "--replacement-file", snippet])
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for command in commands]
    completed = [(worker, worker.communicate(timeout=15)) for worker in workers]
    assert sum(worker.returncode == 0 for worker, _ in completed) == 1
    assert all(worker.returncode == 0 or b"stale_source" in output[1] for worker, output in completed)
    assert helper.read("calc.py", "add")["ast_equivalent"]
    assert not list(repo.glob(".source-view-*.tmp"))


def test_audit_parallel_reads_count_every_distinct_operation(repo, tmp_path):
    audit = tmp_path / "parallel.jsonl"
    commands = [[sys.executable, views.__file__, "--root", str(repo), "--audit-store", str(audit),
                 "read", "calc.py", "--symbol", "add", "--format", fmt]
                for fmt in ("raw", "compact") * 4]
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for command in commands]
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=15)
        assert worker.returncode == 0, stderr
        assert json.loads(stdout)["audit_status"] == "recorded"
    receipt = views.summarize_audit_store(audit, expected_root=repo)
    assert receipt["operations"] == 8
    assert receipt["format_counts"] == {"raw": 4, "compact": 4}


@pytest.mark.parametrize("snippet", ["def add(value, value): return value",
                                       "def add(value):\n    nonlocal missing\n    return value",
                                       "def add(value):\n    continue"])
def test_compile_invalid_replacement_never_publishes(repo, snippet):
    original = (repo / "calc.py").read_bytes()
    assert_error("invalid_python", lambda: views.CodeSourceView(repo).replace(
        "calc.py", "add", sha(original), snippet))
    assert (repo / "calc.py").read_bytes() == original


def test_invalid_format_and_stale_noop_fail_before_mutation(repo, monkeypatch):
    helper = views.CodeSourceView(repo)
    view = helper.read("calc.py", "add")
    original = (repo / "calc.py").read_bytes()
    def unexpected_lock(_path):
        raise AssertionError("invalid format reached locking")
    monkeypatch.setattr(helper, "_locked", unexpected_lock)
    assert_error("invalid_format", lambda: helper.replace("calc.py", "add", view["source_sha256"],
                                                          view["content"], format="unsupported"))
    assert (repo / "calc.py").read_bytes() == original
    helper = views.CodeSourceView(repo)
    (repo / "calc.py").write_bytes(original + b"# valid external change\n")
    assert_error("stale_source", lambda: helper.replace("calc.py", "add", view["source_sha256"], view["content"]))

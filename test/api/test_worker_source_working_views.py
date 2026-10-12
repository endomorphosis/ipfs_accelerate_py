"""Working-source snapshots preserve canonical evidence while following edits.

These tests use a real Git worktree and codec, with no model or proof calls.
"""
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_views as module
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
from test.api.test_worker_source_views import case, decode, service, sha  # noqa: F401


def working_pages(svc, *, path, view="compact", max_bytes=2800, **pins):
    rows, cursor = [], None
    generation = pins.pop("expected_workspace_generation", None)
    for _ in range(100):
        row = svc.working_page(path=path, view=view, max_bytes=max_bytes, cursor=cursor,
            expected_workspace_generation=generation, **pins)
        assert len(module.canonical_json(row)) <= max_bytes
        assert row["offset"] == (rows[-1]["end_offset"] if rows else 0)
        generation = row["workspace_generation"]
        rows.append(row)
        cursor = row["next_cursor"]
        if cursor is None:
            assert row["exhausted"] is True
            return b"".join(decode(row) for row in rows), rows
    pytest.fail("working source pagination did not terminate")


def pins(row):
    return dict(expected_source_sha256=row["source_sha256"],
        expected_view_sha256=row["view_sha256"],
        expected_workspace_generation=row["workspace_generation"])


def edit(case, name="pkg/sample.py", raw=None):
    value = raw if raw is not None else case["sources"][name] + b"CHANGED = True  # worktree edit\r\n"
    (case["workspace"] / name).write_bytes(value)
    return value


def test_working_pages_after_edit_expand_exact_bytes_without_relabeling_baseline_evidence(case):
    svc = service(case, profile="python-comments@1")
    baseline = svc.baseline_admission()
    changed = edit(case)
    compact, rows = working_pages(svc, path="pkg/sample.py")
    assert compact == build_source_view(changed, profile="python-comments@1").view
    original, originals = working_pages(svc, path="pkg/sample.py", view="original", **pins(rows[0]))
    assert original == changed and len(rows) > 1 and len(originals) > 1
    assert svc.baseline_admission() == baseline
    for row in rows + originals:
        assert row["schema"] == "supervisor-worker-source-view-page@2"
        assert row["source_state"] == "working-candidate"
        assert row["source_check_profile"] == "owner-working-source-candidate@1"
        assert row["baseline_source_sha256"] == sha(case["sources"]["pkg/sample.py"])
        assert row["source_sha256"] == sha(changed)
        assert row["baseline_evidence_applicable"] is False
        assert row["proof_authority"] is row["mutation_authority"] is row["omission_authority"] is False
        assert row["source_obligations_discharged"] == 0
        assert row["binding"]["semantic_root_cid"] == case["kwargs"]["semantic_root_cid"]
        assert row["binding"]["source_manifest_sha256"] == case["kwargs"]["source_manifest_sha256"]
    with pytest.raises(ValueError):
        svc.page(path="pkg/sample.py")
    assert (case["root"] / "pkg/sample.py").read_bytes() == case["sources"]["pkg/sample.py"]
    assert (case["workspace"] / "pkg/sample.py").read_bytes() == changed


def test_unedited_candidate_is_not_presented_as_new_proof_and_noop_reuses_generation(case):
    svc = service(case)
    first = svc.working_page(path="other.py")
    second = svc.working_page(path="other.py")
    assert first["workspace_generation"] == second["workspace_generation"]
    assert decode(first) == case["sources"]["other.py"]
    assert first["baseline_evidence_applicable"] is False
    assert svc.working_admission(expected_workspace_generation=first["workspace_generation"])


def test_whole_population_edit_invalidates_cursor_and_generation_for_unchanged_file(case):
    svc = service(case)
    first = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    assert first["next_cursor"]
    edit(case, "other.py", b"VALUE = 200\n")
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py", max_bytes=2800,
            cursor=first["next_cursor"], expected_workspace_generation=first["workspace_generation"])
    current = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    assert current["source_sha256"] == first["source_sha256"]
    assert current["workspace_generation"] != first["workspace_generation"]
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py", **pins(first))
    with pytest.raises(ValueError):
        svc.working_admission(expected_workspace_generation=first["workspace_generation"])
    assert decode(svc.working_page(path="other.py")) == b"VALUE = 200\n"


def test_observed_aba_does_not_revive_prior_generation_or_cursor(case):
    svc = service(case)
    first = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    edit(case, "other.py", b"VALUE = 200\n")
    second = svc.working_page(path="other.py")
    edit(case, "other.py", case["sources"]["other.py"])
    third = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    assert len({row["workspace_generation"] for row in (first, second, third)}) == 3
    assert first["source_sha256"] == third["source_sha256"]
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py", max_bytes=2800,
            cursor=first["next_cursor"], expected_workspace_generation=first["workspace_generation"])


@pytest.mark.parametrize("query", ["pins", "cursor"])
def test_continuation_or_pinned_expansion_requires_explicit_generation(case, query):
    svc = service(case)
    first = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    options = {key: value for key, value in pins(first).items() if key != "expected_workspace_generation"}
    if query == "cursor":
        options = dict(cursor=first["next_cursor"])
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py", max_bytes=2800, **options)


def test_pinned_old_source_does_not_trigger_implicit_refresh(case):
    svc = service(case)
    first = svc.working_page(path="other.py")
    edit(case, "other.py", b"VALUE = 3\n")
    with pytest.raises(ValueError):
        svc.working_page(path="other.py", view="original", **pins(first))
    current = svc.working_page(path="other.py")
    assert current["source_sha256"] != first["source_sha256"]
    with pytest.raises(ValueError):
        svc.working_page(path="other.py", view="original", **pins(first))
    assert decode(svc.working_page(path="other.py", view="original", **pins(current))) == b"VALUE = 3\n"


@pytest.mark.parametrize("root_key", ["root", "workspace"])
def test_deletion_refuses_without_serving_partial_candidate(case, root_key):
    svc = service(case)
    first = svc.working_page(path="pkg/sample.py", max_bytes=2800)
    (case[root_key] / "other.py").unlink()
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py", max_bytes=2800)
    (case[root_key] / "other.py").write_bytes(case["sources"]["other.py"])
    current = svc.working_page(path="other.py")
    assert decode(current) == case["sources"]["other.py"]
    if root_key == "workspace":
        assert current["workspace_generation"] != first["workspace_generation"]
        with pytest.raises(ValueError):
            svc.working_page(path="pkg/sample.py", max_bytes=2800,
                cursor=first["next_cursor"], expected_workspace_generation=first["workspace_generation"])


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_allocated_source_refuses_new_snapshot_and_cannot_leak_external_bytes(case, tmp_path, kind):
    svc = service(case)
    svc.working_page(path="other.py")
    path = case["workspace"] / "other.py"
    path.unlink()
    external = tmp_path / "private.py"
    external.write_bytes(b"PRIVATE = 'outside scope'\n")
    if kind == "symlink":
        path.symlink_to(external)
    elif kind == "hardlink":
        os.link(external, path)
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py")
    path.unlink()
    edit(case, "other.py", b"VALUE = 10\n")
    assert decode(svc.working_page(path="other.py")) == b"VALUE = 10\n"


def test_unadmitted_new_file_does_not_expand_scope_or_invalidate_existing_population(case):
    svc = service(case)
    first = svc.working_page(path="other.py")
    (case["workspace"] / "new.py").write_bytes(b"UNADMITTED = True\n")
    with pytest.raises(ValueError):
        svc.working_page(path="new.py")
    assert svc.working_page(path="other.py")["workspace_generation"] == first["workspace_generation"]


def test_canonical_change_cannot_be_adopted_as_working_baseline(case):
    svc = service(case)
    first = svc.working_page(path="other.py")
    (case["root"] / "other.py").write_bytes(b"VALUE = 3\n")
    edit(case, "other.py", b"VALUE = 3\n")
    for operation in (lambda: svc.working_page(path="other.py"),
                      lambda: svc.working_page(path="other.py", **pins(first)),
                      svc.baseline_admission):
        with pytest.raises(ValueError):
            operation()


def test_edit_during_projection_refuses_entire_candidate_before_any_bytes(case, monkeypatch):
    svc = service(case)
    edit(case, "other.py", b"VALUE = 4\n")
    build, invoked = module.build_source_view, []
    def race(raw, **kwargs):
        projected = build(raw, **kwargs)
        if not invoked:
            invoked.append(True)
            edit(case, "other.py", b"VALUE = 5\n")
        return projected
    monkeypatch.setattr(module, "build_source_view", race)
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py")
    assert invoked
    monkeypatch.setattr(module, "build_source_view", build)
    assert decode(svc.working_page(path="other.py")) == b"VALUE = 5\n"


def test_oversized_working_population_refuses_even_when_baseline_remains_admitted(case):
    svc = service(case)
    edit(case, "pkg/sample.py", b"x" * 600000)
    edit(case, "other.py", b"y" * 600000)
    with pytest.raises(ValueError):
        svc.working_page(path="pkg/sample.py")
    assert svc.baseline_admission()


@pytest.mark.parametrize("name", ["empty.py", "binary.dat"])
def test_working_empty_and_binary_data_round_trip_exactly(case, name):
    svc = service(case)
    changed = b"" if name == "empty.py" else b"\xff\x80\0\xfe" * 900
    edit(case, name, changed)
    compact, rows = working_pages(svc, path=name)
    original, _ = working_pages(svc, path=name, view="original", **pins(rows[0]))
    assert compact == original == changed


def test_separate_sessions_cannot_share_working_generation_even_for_identical_sources(case):
    left, right = service(case), service(case)
    first = left.working_page(path="pkg/sample.py", max_bytes=2800)
    second = right.working_page(path="pkg/sample.py", max_bytes=2800)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["workspace_generation"] != second["workspace_generation"]
    with pytest.raises(ValueError):
        right.working_page(path="pkg/sample.py", **pins(first))
    with pytest.raises(ValueError):
        right.working_page(path="pkg/sample.py", max_bytes=2800,
            cursor=first["next_cursor"], expected_workspace_generation=first["workspace_generation"])

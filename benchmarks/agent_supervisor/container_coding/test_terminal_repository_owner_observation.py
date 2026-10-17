"""Authored sealed owner evidence; no native lease, worker or provider execution.

Filesystem protection checks are real. Descriptor, task and invocation history
comes from explicit authored records, so these cases establish collector
behavior rather than live cross-UID provenance or model tool consumption.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as collector
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_audit import authored_owner_evidence
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_lifecycle import authored_lifecycle_envelope
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _scope():
    return dict(task_cid="authored-task-cid", task_alias="authored-task",
        baseline_commit="a" * 40,
        outputs=[dict(path="result.py", effect="create", baseline_sha256=None)])


def _authored(session_id="1" * 32, invocation_id="2" * 32):
    value = authored_owner_evidence(session_id=session_id, invocation_id=invocation_id)
    value["row"]["public_instruction"] = dict(schema="supervisor-public-instruction-inclusion@1",
        task_cid="authored-task-cid", task_id="authored-task", artifact_sha256="a" * 64,
        source_sha256="b" * 64, block_sha256="c" * 64,
        manifest_signature_verified=True, verbatim_utf8=True, source_freshness_verified=True,
        historical_replay=False, completion_authority=False, scope_expansion_authority=False,
        artifact="/app/.supervisor-public-instructions/authored.json", context_cid="authored-context-cid",
        manifest_cid="authored-manifest-cid", source_path="instruction.md", source_bytes=18,
        semantic_minification_applied=False)
    return value


def _owner_instruction(row):
    inclusion = row["public_instruction"]
    return {**{key: inclusion[key] for key in ("artifact", "context_cid", "manifest_cid", "task_cid",
        "source_path", "source_sha256", "source_bytes", "completion_authority", "scope_expansion_authority")},
        "sha256": inclusion["artifact_sha256"]}


def _write_sealed(path, raw):
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)


@pytest.fixture
def evidence():
    # The workspace has writable group ancestors; /tmp has a public ancestor.
    # A private home child preserves the actual full-ancestor protection.
    with tempfile.TemporaryDirectory(prefix="repository-owner-observer-", dir=Path.home()) as name:
        parent = Path(name) / "observations"
        parent.mkdir(mode=0o750)
        parent.chmod(0o750)
        first = _authored()

        def write(value):
            session = value["row"]["repository_source_policy"]["session_id"]
            stem = "repository-mcp-" + session + "-audit"
            paths = dict(audit=parent / (stem + ".json"),
                         operations=parent / (stem + "-operations.jsonl"))
            _write_sealed(paths["audit"], value["audit_raw"])
            _write_sealed(paths["operations"], value["operations_raw"])
            return paths

        paths = write(first)
        report = dict(repository_source_policy=first["row"]["repository_source_policy"]["requested_policy"],
                      provider_invocations=[first["row"]], worker_cleanup_returncode=0,
                      public_instruction=_owner_instruction(first["row"]))
        yield dict(parent=parent, first=first, paths=paths, write=write,
                   report=report, scope=_scope())


def _collect(evidence):
    return collector.collect(parent=evidence["parent"], scope=evidence["scope"], report=evidence["report"])


def _unavailable(result, reason):
    assert result == collector.unavailable(reason)
    assert result["session_count"] is result["transport"] is result["prepared_mcp_counts"] is None
    assert result["delivered_tool_calls"] is result["delivered_reads"] is result["model_consumed_tokens"] is None


def test_real_sealed_files_are_recounted_without_mutating_worker_usage(evidence):
    original = deepcopy(evidence["report"])
    before = {name: path.read_bytes() for name, path in evidence["paths"].items()}
    result = _collect(evidence)
    assert result["status"] == "observed" and result["session_count"] == 1
    assert stat.S_IMODE(evidence["parent"].stat().st_mode) == 0o750
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 for path in evidence["paths"].values())
    assert evidence["report"] == original
    assert before == {name: path.read_bytes() for name, path in evidence["paths"].items()}
    session = result["sessions"][0]
    assert session["audit_sha256"] == _sha(before["audit"])
    assert session["operations_sha256"] == _sha(before["operations"])
    assert result["prepared_mcp_counts"]["successful_reads"] == session["prepared_mcp_usage"]["successful_reads"]
    assert result["delivered_tool_calls"] is result["delivered_reads"] is result["model_consumed_tokens"] is None
    assert result["native_parent_release_verified"] is result["worker_cleanup_independently_verified"] is False
    assert result["provider_ingestion_verified"] is result["token_savings_measured"] is False
    assert result["raw_content_exported"] is result["completion_authority"] is False
    retirement = result["lifecycle_observation"]
    assert retirement["status"] == "unavailable"
    assert retirement["known_session_count"] == 0 and retirement["unknown_session_count"] == 1
    assert retirement["own_service_reservation_release_reported"] is None
    assert retirement["sessions"][0]["reason"] == "no_retained_lifecycle_receipt"
    text = json.dumps(result)
    assert "authored-task" not in text and "/authored/candidate" not in text
    assert evidence["first"]["row"]["invocation_id"] not in text


def test_complete_distinct_sessions_sum_only_supported_counts(evidence):
    second = _authored("3" * 32, "4" * 32)
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    result = _collect(evidence)
    assert result["status"] == "observed" and result["session_count"] == 2
    assert len(result["sessions"]) == 2
    for key, value in result["prepared_mcp_counts"].items():
        assert value == sum(row["prepared_mcp_usage"][key] for row in result["sessions"])
    assert result["delivered_reads"] is None and result["model_consumed_tokens"] is None


def test_missing_second_session_never_exports_a_qualified_subtotal(evidence):
    second = _authored("3" * 32, "4" * 32)
    evidence["write"](second)["operations"].unlink()
    evidence["report"]["provider_invocations"].append(second["row"])
    _unavailable(_collect(evidence), "retained_evidence_unavailable")


@pytest.mark.parametrize("report", [{}, {"repository_source_policy": "legacy"},
    {"provider_invocations": [{"phase": "planning"}]}])
def test_unselected_population_never_accesses_parent_or_scope(report):
    class NoPath:
        def __fspath__(self):
            raise AssertionError("unselected evidence must not touch a parent")
    result = collector.collect(parent=NoPath(), scope=None, report=report)
    expected = collector.unavailable("not_selected")
    expected["status"] = "not_selected"
    assert result == expected


@pytest.mark.parametrize("bad", [None, [], "report", 7])
def test_invalid_report_is_unavailable_without_io(bad):
    _unavailable(collector.collect(parent=None, scope=None, report=bad), "population_unavailable")


@pytest.mark.parametrize("change", ["rows_none", "rows_object", "nondict_row", "too_many_rows",
    "too_many_sessions", "unselected_coding", "no_owner_binding", "unreceipted_false", "unreceipted_true"])
def test_population_refusals_happen_before_artifact_access(evidence, monkeypatch, change):
    report = evidence["report"]
    if change == "rows_none": report["provider_invocations"] = None
    elif change == "rows_object": report["provider_invocations"] = {}
    elif change == "nondict_row": report["provider_invocations"].append(None)
    elif change == "too_many_rows": report["provider_invocations"] += [{"phase": "planning"}] * 128
    elif change == "too_many_sessions": report["provider_invocations"] *= 17
    elif change == "unselected_coding": report["provider_invocations"].append({"phase": "coding"})
    elif change == "no_owner_binding": report["provider_invocations"] = []
    elif change == "unreceipted_false": report["unreceipted_provider_attempt"] = False
    elif change == "unreceipted_true": report["unreceipted_provider_attempt"] = True
    touched = []
    monkeypatch.setattr(collector, "_protected_directory", lambda *args, **kwargs: touched.append(True))
    _unavailable(_collect(evidence), "population_unavailable")
    assert touched == []


@pytest.mark.parametrize("cleanup", [None, True, False, 1, -1, "0"])
def test_cleanup_requires_exact_success_record(evidence, cleanup):
    evidence["report"]["worker_cleanup_returncode"] = cleanup
    _unavailable(_collect(evidence), "cleanup_unavailable")


@pytest.mark.parametrize("change", ["absent", "empty", "empty_outputs", "extra_field", "bad_baseline"])
def test_captured_scope_is_required_without_reconstruction(evidence, change):
    if change == "absent": evidence["scope"] = None
    elif change == "empty": evidence["scope"] = {}
    elif change == "empty_outputs": evidence["scope"]["outputs"] = []
    elif change == "extra_field": evidence["scope"]["unknown"] = True
    else: evidence["scope"]["baseline_commit"] = "missing"
    _unavailable(_collect(evidence), "scope_unavailable")


@pytest.mark.parametrize("duplicate", ["session", "invocation", "both", "planning_row"])
def test_duplicate_attribution_is_unavailable(evidence, duplicate):
    if duplicate == "planning_row":
        evidence["report"]["provider_invocations"].append(dict(phase="planning",
            invocation_id=evidence["first"]["row"]["invocation_id"]))
        _unavailable(_collect(evidence), "session_join_unavailable")
        return
    if duplicate == "invocation":
        second = _authored("3" * 32, "2" * 32)
    else:
        second = deepcopy(evidence["first"])
        if duplicate == "session": second["row"]["invocation_id"] = "4" * 32
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    _unavailable(_collect(evidence), "session_join_unavailable")


@pytest.mark.parametrize("change", ["task_cid", "task_id", "instruction_schema", "instruction_hash",
    "signature", "verbatim", "freshness", "historical", "completion", "scope_authority",
    "owner_uid", "worker_uid", "boundary_manifest", "boundary_purpose", "boundary_authority",
    "workspace", "invocation", "policy", "legacy_policy", "minification"])
def test_task_boundary_and_invocation_must_match(evidence, change):
    row = evidence["report"]["provider_invocations"][0]
    inclusion = row["public_instruction"]
    boundary = row["external_container_boundary"]
    if change in {"task_cid", "task_id"}: inclusion[change] = "different-task"
    elif change == "instruction_schema": inclusion["schema"] = "unrecognized"
    elif change == "instruction_hash": inclusion["artifact_sha256"] = "unknown"
    elif change == "signature": inclusion["manifest_signature_verified"] = False
    elif change == "verbatim": inclusion["verbatim_utf8"] = False
    elif change == "freshness": inclusion["source_freshness_verified"] = False
    elif change == "historical": inclusion["historical_replay"] = True
    elif change == "completion": inclusion["completion_authority"] = True
    elif change == "scope_authority": inclusion["scope_expansion_authority"] = True
    elif change in {"owner_uid", "worker_uid"}: boundary[change] += 10
    elif change == "boundary_manifest": boundary["manifest_sha256"] = "f" * 64
    elif change == "boundary_purpose": boundary["purpose"] = "planning"
    elif change == "boundary_authority": boundary["completion_authority"] = True
    elif change == "workspace": row["workspace"] = "/authored/another-candidate"
    elif change == "invocation": row["invocation_id"] = "unknown"
    elif change == "policy": evidence["report"]["repository_source_policy"] = "repository-source-raw@1"
    elif change == "legacy_policy": evidence["report"]["repository_source_policy"] = "legacy"
    elif change == "minification": inclusion["semantic_minification_applied"] = True
    _unavailable(_collect(evidence), "session_join_unavailable")


@pytest.mark.parametrize("change", ["absent", "sha256", "artifact", "context_cid", "manifest_cid", "task_cid",
    "source_path", "source_sha256", "source_bytes", "completion_authority", "scope_expansion_authority"])
def test_owner_prepared_instruction_is_independently_joined(evidence, change):
    prepared = evidence["report"]["public_instruction"]
    if change == "absent": evidence["report"].pop("public_instruction")
    elif change in {"completion_authority", "scope_expansion_authority"}: prepared[change] = True
    elif change == "source_bytes": prepared[change] += 1
    elif change in {"sha256", "source_sha256"}: prepared[change] = "d" * 64
    else: prepared[change] = "different-owner-selection"
    _unavailable(_collect(evidence), "session_join_unavailable")


@pytest.mark.parametrize("which", ["audit", "operations"])
@pytest.mark.parametrize("change", ["missing", "tampered", "empty"])
def test_missing_or_changed_retained_bytes_are_not_zero_usage(evidence, which, change):
    path = evidence["paths"][which]
    if change == "missing": path.unlink()
    else: _write_sealed(path, b"" if change == "empty" else b'{"tampered":true}\n')
    _unavailable(_collect(evidence), "retained_evidence_unavailable")


@pytest.mark.parametrize("which", ["audit", "operations"])
@pytest.mark.parametrize("change", ["mode600", "mode440", "mode000", "symlink", "hardlink", "directory"])
def test_evidence_requires_exact_sealed_regular_single_link_files(evidence, which, change):
    path = evidence["paths"][which]
    if change.startswith("mode"):
        path.chmod(int(change[4:], 8))
    elif change == "hardlink":
        os.link(path, path.with_name(path.name + ".other-link"))
    else:
        renamed = path.with_name(path.name + ".retained")
        path.rename(renamed)
        if change == "symlink": path.symlink_to(renamed)
        else: path.mkdir(mode=0o700)
    _unavailable(_collect(evidence), "retained_evidence_unavailable")


@pytest.mark.parametrize("mode", [0o700, 0o755, 0o770])
def test_parent_requires_actual_protected_0750_directory(evidence, mode):
    evidence["parent"].chmod(mode)
    _unavailable(_collect(evidence), "retained_evidence_unavailable")


def test_parent_symlink_is_refused(evidence):
    parent = evidence["parent"]
    retained = parent.with_name("retained-observations")
    parent.rename(retained)
    parent.symlink_to(retained, target_is_directory=True)
    _unavailable(_collect(evidence), "retained_evidence_unavailable")


@pytest.mark.parametrize("change", ["replace", "permissions"])
def test_parent_change_during_real_reads_is_detected(evidence, monkeypatch, change):
    actual = collector._sealed_read
    touched = []
    def changing_read(fd, name, **kwargs):
        raw = actual(fd, name, **kwargs)
        if name.endswith("-operations.jsonl"):
            parent = evidence["parent"]
            if change == "replace":
                parent.rename(parent.with_name("retained-observations"))
                parent.mkdir(mode=0o750)
                parent.chmod(0o750)
            else:
                parent.chmod(0o770)
            touched.append(True)
        return raw
    monkeypatch.setattr(collector, "_sealed_read", changing_read)
    _unavailable(_collect(evidence), "retained_evidence_unavailable")
    assert touched == [True]


@pytest.mark.parametrize("which", ["audit", "operations"])
def test_file_replaced_between_read_and_final_stat_is_detected(evidence, monkeypatch, which):
    actual = collector._read
    touched = []
    target = evidence["paths"][which]
    def changing_read(fd, name, **kwargs):
        raw = actual(fd, name, **kwargs)
        if name == target.name:
            target.rename(target.with_name(target.name + ".retained"))
            _write_sealed(target, raw)
            touched.append(True)
        return raw
    monkeypatch.setattr(collector, "_read", changing_read)
    _unavailable(_collect(evidence), "retained_evidence_unavailable")
    assert touched == [True]


def test_unrelated_artifacts_are_never_discovered_or_exported(evidence):
    unrelated = evidence["parent"] / "unrelated-private.txt"
    unrelated.write_text("authored private source that must not be exported")
    unrelated.chmod(0o000)
    (evidence["parent"] / "unused-link").symlink_to("/nonexistent-private-location")
    result = _collect(evidence)
    assert result["status"] == "observed" and result["session_count"] == 1
    assert "authored private source" not in json.dumps(result)


def _retirement_path(evidence, value=None):
    value = evidence["first"] if value is None else value
    session = value["row"]["repository_source_policy"]["session_id"]
    return evidence["parent"] / ("repository-mcp-" + session + "-lifecycle.json")


def _write_retirement(evidence, value=None):
    value = evidence["first"] if value is None else value
    target = _retirement_path(evidence, value)
    _write_sealed(target, protocol.canonical_json(authored_lifecycle_envelope(value)))
    return target


def test_optional_retirement_adds_only_scoped_release_observation(evidence):
    historical = _collect(evidence)
    target = _write_retirement(evidence)
    before = target.read_bytes()
    result = _collect(evidence)
    retirement = result.pop("lifecycle_observation")
    historical.pop("lifecycle_observation")
    assert result == historical
    assert retirement["status"] == "observed"
    assert retirement["known_session_count"] == 1 and retirement["unknown_session_count"] == 0
    assert retirement["own_service_reservation_release_reported"] is True
    assert retirement["independent_scheduler_capacity_verified"] is False
    assert retirement["borrowed_caller_release"] is None
    assert retirement["sessions"][0]["retirement_sha256"] == _sha(before)
    assert retirement["sessions"][0]["invocation_sha256"] == result["sessions"][0]["invocation_sha256"]
    assert target.read_bytes() == before and stat.S_IMODE(target.stat().st_mode) == 0o400
    assert result["native_parent_release_verified"] is False
    assert result["sessions"][0]["cleanup"]["native_parent_released"] is None


@pytest.mark.parametrize("change", ["tampered", "empty", "wrong_session", "mode600", "mode440",
    "mode000", "symlink", "dangling_symlink", "hardlink", "directory"])
def test_bad_optional_retirement_preserves_valid_phase12_accounting(evidence, change):
    target = _write_retirement(evidence)
    if change == "tampered":
        _write_sealed(target, b'{"tampered":true}')
    elif change == "empty":
        _write_sealed(target, b"")
    elif change == "wrong_session":
        other = _authored("3" * 32, "4" * 32)
        _write_sealed(target, protocol.canonical_json(authored_lifecycle_envelope(other)))
    elif change.startswith("mode"):
        target.chmod(int(change[4:], 8))
    elif change == "hardlink":
        os.link(target, target.with_name(target.name + ".other-link"))
    else:
        retained = target.with_name(target.name + ".retained")
        target.rename(retained)
        if change == "directory":
            target.mkdir(mode=0o700)
        else:
            target.symlink_to(retained if change == "symlink" else "does-not-exist")
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["prepared_mcp_counts"]["successful_reads"] == 1
    retirement = result["lifecycle_observation"]
    assert retirement["status"] == "unavailable"
    assert retirement["own_service_reservation_release_reported"] is None
    assert retirement["known_session_count"] == 0 and retirement["unknown_session_count"] == 1
    assert retirement["sessions"][0]["reason"] == "invalid_retained_lifecycle_receipt"


def test_optional_retirement_requires_all_selected_sessions_for_aggregate_release(evidence):
    second = _authored("3" * 32, "4" * 32)
    evidence["write"](second)
    evidence["report"]["provider_invocations"].append(second["row"])
    _write_retirement(evidence)
    partial = _collect(evidence)
    assert partial["status"] == "observed" and partial["prepared_mcp_counts"]["successful_reads"] == 2
    retirement = partial["lifecycle_observation"]
    assert retirement["known_session_count"] == retirement["unknown_session_count"] == 1
    assert retirement["own_service_reservation_release_reported"] is None
    _write_retirement(evidence, second)
    result = _collect(evidence)
    assert result["prepared_mcp_counts"] == partial["prepared_mcp_counts"]
    assert result["lifecycle_observation"]["own_service_reservation_release_reported"] is True
    assert result["lifecycle_observation"]["known_session_count"] == 2
    assert result["lifecycle_observation"]["unknown_session_count"] == 0

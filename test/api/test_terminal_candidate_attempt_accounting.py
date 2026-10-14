"""Private lifecycle joins and downstream token totals; no provider calls."""
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_candidate_attempt_accounting as accounting
from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage, populate_harbor_usage
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_lifecycle as lifecycle
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import canonical_json


def observation(rounds):
    return dict(schema="owner-candidate-rounds-observation@1", rounds=rounds,
        max_rounds=2, elapsed_seconds=1.25, outcome="workers_returned", failure_type=None,
        completion_authority=False, proof_authority=False, source_obligations_discharged=0,
        accepted_index_published=False)


def receipt(retained, number):
    attempt = retained["observation"]["rounds"][number-1]
    binding = dict(schema="worker-source-dispatch-observation@4" if number == 1 else "supervisor-candidate-dispatch-observation@1",
        session_id=attempt["session_id"], descriptor_sha256=attempt["descriptor_sha256"],
        task_id=retained["task_id"], native_prompt_sha256=retained["native_prompt_sha256"],
        proof_authority=False, execution_authority=False, completion_authority=False)
    if number == 2:
        binding.update(prior_session_id=retained["prior_session_id"],
            prior_descriptor_sha256=retained["prior_descriptor_sha256"], round_number=2)
    row = dict(schema="router-implementation-invocation@1", invocation_id="authored-round-"+str(number),
        phase="coding", purpose="coding", provider="codex_cli", workspace="/authored/workspace",
        native_prompt_sha256=retained["native_prompt_sha256"], status="failed" if number == 1 else "provider_returned",
        native_rollout_usage=dict(schema="native-codex-cumulative-usage@1", task_complete_observed=True,
            cache_included_in_input=True, usage=dict(input_tokens=100*number, cached_input_tokens=80*number,
                output_tokens=10*number, total_tokens=110*number)))
    row["source_view_retrieval" if number == 1 else "candidate_round_dispatch"] = binding
    return row


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Only container parent-location policy is authored. UID, mode, hardlink,
    # nofollow descriptor reads, canonical JSON and content joins remain real.
    monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
    parent = tmp_path / "worker-observations"
    parent.mkdir(mode=0o700)
    rows = [dict(round_number=number, process_started=True, session_id=str(number)*32,
        descriptor_sha256=str(number+2)*64, returncode=3 if number == 1 else 0,
        wait_completed=True, retirement_observed=True) for number in (1, 2)]
    retained = dict(schema=lifecycle.SCHEMA, prior_session_id=rows[0]["session_id"],
        prior_descriptor_sha256=rows[0]["descriptor_sha256"], task_id="TASK-AUTHORED",
        native_prompt_sha256="a"*64, workspace_sha256=hashlib.sha256(b"/authored/workspace").hexdigest(),
        owner_pid=os.getpid(), owner_start_time_ticks=1, boundary_manifest_sha256="b"*64,
        observation=observation(rows), **lifecycle.FLAGS)
    path = parent / lifecycle.lifecycle_name(retained["prior_session_id"])
    def write(value=None, raw=None):
        if path.exists():
            path.chmod(0o600)
        path.write_bytes(canonical_json(retained if value is None else value) if raw is None else raw)
        path.chmod(0o400)
    write()
    report = dict(candidate_rounds=2, provider_invocations=[receipt(retained, n) for n in (1, 2)])
    return SimpleNamespace(parent=parent, retained=retained, path=path, write=write, report=report)


def test_two_distinct_receipts_preserve_native_cache_semantics_and_sum(evidence):
    report = evidence.report
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "matched_round_receipts"
    assert not value["potential_unreceipted_attempt"] and not value["complete_usage_established"]
    usage = measured_usage(report)
    assert [usage[key] for key in ("provider_calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")] == [2, 300, 240, 30, 330]
    assert usage["cache_included_in_input"] is True and usage["all_invocations_receipted"] is True
    assert usage["billing_total_verified"] is False
    exported = SimpleNamespace()
    populate_harbor_usage(exported, usage)
    assert (exported.n_input_tokens, exported.n_cache_tokens, exported.n_output_tokens) == (300, 240, 30)


@pytest.mark.parametrize("existing", [None, False, {}, []])
def test_missing_second_receipt_keeps_overall_unknown_with_false_like_previous_marker(evidence, existing):
    report = evidence.report
    report["provider_invocations"].pop()
    report["unreceipted_provider_attempt"] = existing
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "incomplete" and value["potential_unreceipted_attempt"]
    assert value["rounds"][0]["receipt_observed"] and not value["rounds"][1]["receipt_observed"]
    usage = measured_usage(report)
    assert usage["provider_calls"] == 1 and not usage["all_invocations_receipted"]
    assert all(usage[key] is None for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"))
    exported = SimpleNamespace()
    populate_harbor_usage(exported, usage)
    assert exported.n_input_tokens is exported.n_cache_tokens is exported.n_output_tokens is None


def test_truthy_planning_unknown_is_preserved(evidence):
    report = evidence.report
    report["provider_invocations"].pop()
    marker = dict(phase="planning", reason="missing_planner_receipt", usage=None)
    report["unreceipted_provider_attempt"] = marker
    accounting.apply(report, parent=evidence.parent)
    assert report["unreceipted_provider_attempt"] is marker
    assert measured_usage(report)["total_tokens"] is None


@pytest.mark.parametrize("change", ["duplicate", "extra", "sibling", "wrong_prior", "wrong_workspace", "wrong_task", "promoted", "noncoding", "malformed_rows"])
def test_malformed_or_extra_receipts_cannot_establish_completeness(evidence, change):
    report = evidence.report
    second = report["provider_invocations"][1]
    binding = second["candidate_round_dispatch"]
    if change == "duplicate":
        report["provider_invocations"].append(deepcopy(second))
    elif change == "extra":
        extra = deepcopy(second); extra["invocation_id"] = "extra-receipt"
        report["provider_invocations"].append(extra)
    elif change == "sibling":
        binding["session_id"] = "f"*32
    elif change == "wrong_prior":
        binding["prior_descriptor_sha256"] = "f"*64
    elif change == "wrong_workspace":
        second["workspace"] += "/sibling"
    elif change == "wrong_task":
        binding["task_id"] = "SIBLING"
    elif change == "promoted":
        binding["completion_authority"] = True
    elif change == "noncoding":
        second["phase"] = "planning"
    else:
        report["provider_invocations"] = [None]
    result = accounting.apply(report, parent=evidence.parent)
    assert result["status"] == "unavailable" and result["potential_unreceipted_attempt"]
    assert report["unreceipted_provider_attempt"]


@pytest.mark.parametrize("change", ["sibling_file", "bad_json", "duplicate_key", "nonfinite", "mode", "hardlink", "symlink", "missing_file_after_discovery"])
def test_bad_private_evidence_cannot_hide_as_legacy_without_markers(evidence, monkeypatch, change):
    report = dict(provider_invocations=evidence.report["provider_invocations"][:1])
    if change == "sibling_file":
        sibling = evidence.parent / lifecycle.lifecycle_name("f"*32)
        sibling.write_bytes(evidence.path.read_bytes()); sibling.chmod(0o400)
    elif change == "bad_json":
        evidence.write(raw=b"{}")
    elif change == "duplicate_key":
        raw = evidence.path.read_bytes()
        evidence.write(raw=b'{"schema":"forged",'+raw[1:])
    elif change == "nonfinite":
        evidence.write(raw=evidence.path.read_bytes().replace(b'"elapsed_seconds":1.25', b'"elapsed_seconds":NaN'))
    elif change == "mode":
        evidence.path.chmod(0o640)
    elif change == "hardlink":
        os.link(evidence.path, evidence.parent / "other-link")
    elif change == "symlink":
        target = evidence.parent / "other-file"
        evidence.path.rename(target); evidence.path.symlink_to(target)
    else:
        def missing(*args, **kwargs):
            raise FileNotFoundError("authored disappearing retained evidence")
        monkeypatch.setattr(accounting, "_read", missing)
    result = accounting.apply(report, parent=evidence.parent)
    assert result["status"] == "unavailable" and result["potential_unreceipted_attempt"]
    assert measured_usage(report)["total_tokens"] is None


def test_forged_log_success_cannot_replace_missing_private_evidence(evidence):
    evidence.path.unlink()
    evidence.report["owner_candidate_rounds"] = [observation(evidence.retained["observation"]["rounds"])]
    result = accounting.apply(evidence.report, parent=evidence.parent)
    assert result["status"] == "unavailable" and measured_usage(evidence.report)["total_tokens"] is None


def test_registration_uncertainty_cannot_count_as_zero(evidence):
    row = evidence.retained["observation"]["rounds"][1]
    row.update(process_started=None, returncode=None, wait_completed=False, retirement_observed=False)
    evidence.write()
    evidence.report["provider_invocations"].pop()
    result = accounting.apply(evidence.report, parent=evidence.parent)
    assert result["status"] == "incomplete" and result["rounds"][1]["process_started"] is None
    assert result["rounds"][1]["possible_unreceipted_attempt"] is True
    assert measured_usage(evidence.report)["total_tokens"] is None


def test_private_lifecycle_is_discovered_without_any_log_marker(evidence):
    report = dict(provider_invocations=evidence.report["provider_invocations"][:1])
    assert accounting.apply(report, parent=evidence.parent)["status"] == "incomplete"
    assert measured_usage(report)["total_tokens"] is None


def test_unselected_legacy_total_unchanged(evidence):
    evidence.path.unlink()
    report = dict(provider_invocations=evidence.report["provider_invocations"][:1])
    before = measured_usage(report)
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "not_selected" and measured_usage(report) == before
    assert "unreceipted_provider_attempt" not in report
    assert accounting.collect(parent=evidence.parent / "absent", report=report)["status"] == "not_selected"


def candidate_archive(parent, session, *, shape="complete"):
    """Authored names model complete or interrupted private publication."""
    name = "candidate-audit-" + session
    if shape in ("complete", "directory_only"):
        (parent / name).mkdir(mode=0o700)
    if shape in ("complete", "manifest_only"):
        path = parent / (name + ".json")
        path.write_bytes(b"{}")
        path.chmod(0o400)
    if shape == "malformed_name":
        (parent / "candidate-audit-interrupted").mkdir(mode=0o700)


@pytest.mark.parametrize("shape", ["complete", "directory_only", "manifest_only", "malformed_name"])
def test_archive_without_final_lifecycle_keeps_killed_owner_usage_unknown(evidence, shape):
    evidence.path.unlink()
    candidate_archive(evidence.parent, evidence.retained["observation"]["rounds"][1]["session_id"], shape=shape)
    # Simulate an owner killed before finalization: no selection config, owner
    # marker or second router receipt. The first receipt alone is insufficient.
    report = dict(provider_invocations=evidence.report["provider_invocations"][:1])
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "unavailable" and value["potential_unreceipted_attempt"]
    usage = measured_usage(report)
    assert usage["provider_calls"] == 1 and usage["all_invocations_receipted"] is False
    assert all(usage[key] is None for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"))


def test_matching_archive_does_not_change_two_round_usage(evidence):
    candidate_archive(evidence.parent, evidence.retained["observation"]["rounds"][1]["session_id"])
    report = dict(provider_invocations=evidence.report["provider_invocations"])
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "matched_round_receipts" and not value["potential_unreceipted_attempt"]
    assert measured_usage(report)["total_tokens"] == 330


@pytest.mark.parametrize("shape", ["complete", "directory_only", "manifest_only", "malformed_name"])
def test_sibling_archive_cannot_be_hidden_by_matching_lifecycle(evidence, shape):
    candidate_archive(evidence.parent, "f"*32, shape=shape)
    report = dict(provider_invocations=evidence.report["provider_invocations"])
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "unavailable" and value["potential_unreceipted_attempt"]
    assert measured_usage(report)["total_tokens"] is None


def test_archive_cannot_be_ignored_by_first_round_only_lifecycle(evidence):
    second = evidence.retained["observation"]["rounds"].pop()
    evidence.write()
    candidate_archive(evidence.parent, second["session_id"])
    report = dict(provider_invocations=evidence.report["provider_invocations"][:1])
    value = accounting.apply(report, parent=evidence.parent)
    assert value["status"] == "unavailable" and measured_usage(report)["total_tokens"] is None

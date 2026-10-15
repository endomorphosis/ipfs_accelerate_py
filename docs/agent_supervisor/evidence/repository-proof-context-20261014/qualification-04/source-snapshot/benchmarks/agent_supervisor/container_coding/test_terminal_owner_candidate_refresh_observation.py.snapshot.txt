"""Authored owner checkpoint observation; no providers, target execution or proofs."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_owner_candidate_refresh_observation as collector
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_checkpoint as checkpoint
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import canonical_json


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _data(session="a" * 32):
    bindings = dict(prior_session_id=session, descriptor_sha256="b" * 64,
        task_id="sample", native_prompt_sha256="c" * 64, workspace_generation="d" * 64)
    bodies = {name: canonical_json(dict(private_source="never export this source", artifact=name))
        for name in ("summary", "capsules", "formal")}
    pins = {name: dict(sha256=_sha(raw), bytes=len(raw)) for name, raw in bodies.items()}
    context = dict(schema="supervisor-owner-candidate-round-context@1", **bindings,
        prior_returncode=17, round_number=2, source_state="working-candidate", artifacts=pins,
        baseline_evidence_applicable=False, proof_authority=False, execution_authority=False,
        completion_authority=False, next_round_admitted=False, accepted_index_published=False,
        source_obligations_discharged=0, provider_calls=0, prover_calls=0,
        model_inference_calls=0, training_steps=0, private_source_path="source-private.py")
    observation = dict(schema=checkpoint.OBSERVATION_SCHEMA, status="prepared",
        reason_code="candidate_context_prepared", **bindings, artifacts_count=3,
        artifact_bytes=sum(map(len, bodies.values())), elapsed_seconds=1.25, **checkpoint.FLAGS)
    artifacts = {name: dict(path="candidate-round-" + session + "/" + name + ".json", **pin)
        for name, pin in pins.items()}
    manifest = dict(schema=checkpoint.SCHEMA, context=context, artifacts=artifacts,
        observation=observation, **checkpoint.FLAGS)
    receipt = dict(schema="worker-source-dispatch-observation@4", session_id=session,
        **{key: bindings[key] for key in ("descriptor_sha256", "task_id", "native_prompt_sha256")},
        proof_authority=False, execution_authority=False, completion_authority=False)
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="invocation-" + session, status="failed", source_view_retrieval=receipt)
    return manifest, bodies, row


@pytest.fixture
def evidence():
    with tempfile.TemporaryDirectory(prefix="candidate-refresh-observer-", dir=Path.home()) as name:
        parent = Path(name); parent.chmod(0o750)
        manifest, bodies, row = _data()
        scope = dict(task_alias="sample")
        report = dict(provider_invocations=[row], worker_cleanup_returncode=0)

        def write(value=manifest, values=bodies):
            session = value["observation"]["prior_session_id"]
            child = parent / ("candidate-round-" + session)
            child.mkdir(mode=0o700, exist_ok=True)
            for artifact, raw in values.items():
                path = child / (artifact + ".json")
                if path.exists(): path.chmod(0o600)
                path.write_bytes(raw); path.chmod(0o400)
            path = parent / (child.name + ".json")
            if path.exists(): path.chmod(0o600)
            path.write_bytes(canonical_json(value)); path.chmod(0o400)
            return path

        path = write()
        yield dict(parent=parent, manifest=manifest, bodies=bodies, row=row,
            scope=scope, report=report, write=write, path=path)


def _collect(evidence):
    return collector.collect(parent=evidence["parent"], scope=evidence["scope"], report=evidence["report"])


def test_current_trial_candidate_hashes_and_counts_have_no_completion_or_token_claim(evidence):
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=1, artifacts=3,
        artifact_bytes=sum(map(len, evidence["bodies"].values())), elapsed_seconds=1.25)
    assert result["sessions"][0]["artifacts"] == evidence["manifest"]["context"]["artifacts"]
    assert result["sessions"][0]["checkpoint_manifest_sha256"] == _sha(evidence["path"].read_bytes())
    assert result["provider_tokens"] is None and result["artifact_delivery_observed"] is False
    assert result["total_token_savings_measured"] is result["accepted_index_published"] is False
    assert all(result[key] == expected for key, expected in checkpoint.FLAGS.items())
    text = canonical_json(result)
    assert b"never export" not in text and b"source-private.py" not in text
    assert b'"task_id"' not in text and b'"path"' not in text and b"invocation-" not in text


def test_multiple_unique_current_sessions_sum_retained_bytes(evidence):
    manifest, bodies, row = _data("e" * 32)
    evidence["write"](manifest, bodies)
    evidence["report"]["provider_invocations"].append(row)
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"] == dict(sessions=2, artifacts=6,
        artifact_bytes=2 * sum(map(len, bodies.values())), elapsed_seconds=2.5)


def test_unrelated_orphan_checkpoint_is_not_discovered(evidence):
    # Even malformed unrelated files must not select a prior or other trial.
    (evidence["parent"] / ("candidate-round-" + "e" * 32 + ".json")).write_text("garbage")
    assert _collect(evidence)["counts"]["sessions"] == 1


@pytest.mark.parametrize("rows", [[], [dict(phase="planning", purpose="planning")]])
def test_no_source_session_is_not_selected_and_not_known_zero(evidence, rows):
    evidence["report"]["provider_invocations"] = rows
    result = _collect(evidence)
    assert result["status"] == "not_selected" and result["counts"] is None
    assert result["provider_tokens"] is None


def test_successful_session_does_not_require_a_failure_checkpoint(evidence):
    evidence["row"]["status"] = "provider_returned"
    evidence["path"].unlink()
    result = _collect(evidence)
    assert result["status"] == "not_selected" and result["counts"] is None


def test_router_returned_session_with_later_wrapper_failure_retains_prepared_checkpoint(evidence):
    evidence["row"]["status"] = "provider_returned"
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["sessions"] == 1


def test_successful_session_without_checkpoint_does_not_discard_other_prepared_session(evidence):
    _, _, row = _data("e" * 32)
    row["status"] = "provider_returned"
    evidence["report"]["provider_invocations"].append(row)
    result = _collect(evidence)
    assert result["status"] == "observed" and result["counts"]["sessions"] == 1


@pytest.mark.parametrize("change", [
    "missing_manifest", "missing_artifact", "artifact_changed", "manifest_duplicate_key",
    "manifest_symlink", "artifact_symlink", "directory_symlink", "manifest_hardlink",
    "artifact_hardlink", "manifest_world_readable", "artifact_group_readable", "directory_group_readable",
    "parent_writable", "cleanup_failed", "cleanup_unknown", "unreceipted", "unreceipted_empty", "duplicate_session",
    "duplicate_invocation", "scope_mismatch", "missing_scope", "wrong_task", "wrong_descriptor",
    "wrong_prompt", "wrong_context_generation", "wrong_context_pin", "bad_count", "bad_bytes",
    "path_escape", "absolute_path", "wrong_schema", "extra_manifest_field", "extra_observation_field",
    "round_admitted", "proof_claim", "model_calls", "boolean_bytes", "boolean_elapsed",
    "boolean_calls", "success_prior", "negative_prior", "oversized_prior", "unknown_receipt_status", "malformed_row", "oversized_manifest", "overflow_elapsed",
])
def test_missing_ambiguous_unsafe_or_unbound_evidence_is_unavailable(evidence, change):
    manifest, row = evidence["manifest"], evidence["row"]
    artifact = evidence["parent"] / manifest["artifacts"]["summary"]["path"]
    child = artifact.parent
    rewrite = True
    if change == "missing_manifest": evidence["path"].unlink(); rewrite = False
    elif change == "missing_artifact": artifact.unlink(); rewrite = False
    elif change == "artifact_changed": artifact.chmod(0o600); artifact.write_bytes(b"different"); artifact.chmod(0o400); rewrite = False
    elif change == "manifest_duplicate_key":
        evidence["path"].chmod(0o600); evidence["path"].write_bytes(b'{"schema":"one","schema":"two"}'); evidence["path"].chmod(0o400); rewrite = False
    elif change in {"manifest_symlink", "artifact_symlink"}:
        path = evidence["path"] if change == "manifest_symlink" else artifact
        replacement = path.with_name(path.name + ".original"); path.rename(replacement); path.symlink_to(replacement.name); rewrite = False
    elif change == "directory_symlink":
        replacement = child.with_name(child.name + "-original"); child.rename(replacement); child.symlink_to(replacement.name); rewrite = False
    elif change in {"manifest_hardlink", "artifact_hardlink"}:
        path = evidence["path"] if change == "manifest_hardlink" else artifact
        os.link(path, path.with_name(path.name + ".other")); rewrite = False
    elif change == "manifest_world_readable": evidence["path"].chmod(0o444); rewrite = False
    elif change == "artifact_group_readable": artifact.chmod(0o440); rewrite = False
    elif change == "directory_group_readable": child.chmod(0o750); rewrite = False
    elif change == "parent_writable": evidence["parent"].chmod(0o770)
    elif change == "cleanup_failed": evidence["report"]["worker_cleanup_returncode"] = 1
    elif change == "cleanup_unknown": evidence["report"].pop("worker_cleanup_returncode")
    elif change == "unreceipted": evidence["report"]["unreceipted_provider_attempt"] = dict(usage=None)
    elif change == "unreceipted_empty": evidence["report"]["unreceipted_provider_attempt"] = {}
    elif change == "duplicate_session": evidence["report"]["provider_invocations"].append(deepcopy(row))
    elif change == "duplicate_invocation":
        other = deepcopy(row); other["source_view_retrieval"]["session_id"] = "e" * 32; evidence["report"]["provider_invocations"].append(other)
    elif change == "scope_mismatch": evidence["scope"]["task_alias"] = "different"
    elif change == "missing_scope": evidence["scope"] = None
    elif change == "wrong_task": row["source_view_retrieval"]["task_id"] = "different"
    elif change == "wrong_descriptor": row["source_view_retrieval"]["descriptor_sha256"] = "e" * 64
    elif change == "wrong_prompt": row["source_view_retrieval"]["native_prompt_sha256"] = "e" * 64
    elif change == "wrong_context_generation": manifest["context"]["workspace_generation"] = "e" * 64
    elif change == "wrong_context_pin": manifest["context"]["artifacts"]["summary"]["sha256"] = "e" * 64
    elif change == "bad_count": manifest["observation"]["artifacts_count"] = 2
    elif change == "bad_bytes": manifest["observation"]["artifact_bytes"] += 1
    elif change == "path_escape": manifest["artifacts"]["summary"]["path"] = "../summary.json"
    elif change == "absolute_path": manifest["artifacts"]["summary"]["path"] = str(artifact)
    elif change == "wrong_schema": manifest["schema"] = "unknown"
    elif change == "extra_manifest_field": manifest["native_prompt"] = "private"
    elif change == "extra_observation_field": manifest["observation"]["native_prompt"] = "private"
    elif change == "round_admitted": manifest["context"]["next_round_admitted"] = True
    elif change == "proof_claim": manifest["proof_authority"] = True
    elif change == "model_calls": manifest["observation"]["model_inference_calls"] = 1
    elif change == "boolean_bytes": manifest["artifacts"]["summary"]["bytes"] = True
    elif change == "boolean_elapsed": manifest["observation"]["elapsed_seconds"] = False
    elif change == "boolean_calls": manifest["provider_calls"] = False
    elif change == "success_prior": manifest["context"]["prior_returncode"] = 0
    elif change == "negative_prior": manifest["context"]["prior_returncode"] = -9
    elif change == "oversized_prior": manifest["context"]["prior_returncode"] = 256
    elif change == "unknown_receipt_status": row.pop("status")
    elif change == "malformed_row": evidence["report"]["provider_invocations"].append(None)
    elif change == "oversized_manifest": manifest["context"]["padding"] = "x" * checkpoint.MAX_MANIFEST_BYTES
    elif change == "overflow_elapsed": manifest["observation"]["elapsed_seconds"] = 10 ** 1000
    else: raise AssertionError(change)
    if rewrite: evidence["write"]()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None
    assert result["sessions"] == [] and result["provider_tokens"] is None


def test_partial_session_population_does_not_report_partial_totals(evidence):
    _, _, row = _data("e" * 32)
    evidence["report"]["provider_invocations"].append(row)
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


def test_selected_population_is_bounded_before_file_reads(evidence, monkeypatch):
    evidence["report"]["provider_invocations"] = [_data(f"{i:032x}")[2] for i in range(collector.MAX_SESSIONS + 1)]
    monkeypatch.setattr(collector, "_open_directory", lambda *a, **kw: pytest.fail("must refuse before filesystem reads"))
    assert _collect(evidence)["status"] == "unavailable"


def test_artifact_accounting_refuses_changed_manifest_during_read(evidence, monkeypatch):
    original = collector._private_read
    changed = False

    def read(*args, **kwargs):
        nonlocal changed
        value = original(*args, **kwargs)
        if args[1] == "summary.json" and not changed:
            changed = True
            evidence["manifest"]["observation"]["elapsed_seconds"] += 1
            path = evidence["path"]; path.chmod(0o600)
            path.write_bytes(canonical_json(evidence["manifest"])); path.chmod(0o400)
        return value

    monkeypatch.setattr(collector, "_private_read", read)
    assert _collect(evidence)["status"] == "unavailable"


def test_earlier_artifact_changed_during_later_artifact_read_is_unavailable(evidence, monkeypatch):
    original = collector._private_read

    def read(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[1] == "summary.json":
            path = evidence["parent"] / evidence["manifest"]["artifacts"]["capsules"]["path"]
            path.chmod(0o600); path.write_bytes(b"changed after its own hash check"); path.chmod(0o400)
        return value

    monkeypatch.setattr(collector, "_private_read", read)
    assert _collect(evidence)["status"] == "unavailable"


def test_nonfinite_aggregate_elapsed_accounting_is_unavailable(evidence):
    evidence["manifest"]["observation"]["elapsed_seconds"] = 1e308
    evidence["write"]()
    manifest, bodies, row = _data("e" * 32)
    manifest["observation"]["elapsed_seconds"] = 1e308
    evidence["write"](manifest, bodies)
    evidence["report"]["provider_invocations"].append(row)
    assert _collect(evidence)["status"] == "unavailable"

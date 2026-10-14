"""Retained world inputs reconstruct without live source or owner authority."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_world_packet_audit as world_audit
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as replay
from test.api.test_repository_world_dispatch import TASK, dispatch_case, invoke, world_session  # noqa: F401
from test.api.test_repository_context_compiler import required  # noqa: F401
from test.api.test_repository_source_index import scheduler  # noqa: F401


@pytest.fixture
def retained(dispatch_case, tmp_path):
    state = tmp_path / "audit-state"
    state.mkdir(mode=0o700)
    root = state / "world_packet_artifacts"
    root.mkdir(mode=0o700)
    _output, receipt = invoke(dispatch_case, world_packet_artifact_dir=root)
    receipt["phase"] = "coding"
    assert receipt["world_packet_artifacts"]["status"] == "retained"
    return dict(case=dispatch_case, receipt=receipt, state=state, root=root, workspace_root=tmp_path)


def reconstruct(case, **changes):
    options = dict(receipt=case["receipt"], artifact_root=case["root"], workspace_root=case["workspace_root"])
    options.update(changes)
    return world_audit.replay_retained_world_packet_input(**options)


def inspect(case, **changes):
    options = dict(state=case["state"], receipts=[case["receipt"]], workspace_root=case["workspace_root"])
    options.update(changes)
    return audit.audit_terminal_context(**options)


def test_actual_router_receipt_reconstructs_complete_input_and_false_authority(retained):
    replayed = reconstruct(retained)
    assert replayed.model_prompt == retained["case"][3]["seen"][-1][0]
    receipt = replayed.receipt()
    assert receipt["model_input_verified"] is True
    assert receipt["verification_scope"] == "historical_world_packet_input_reconstruction"
    assert all(receipt[key] is False for key in world_audit.FALSE_FIELDS)
    assert receipt["replay"]["owner_signature_replayed"] is False
    assert receipt["replay"]["provider_delivery_verified"] is False
    assert TASK not in json.dumps(receipt)
    assert str(retained["case"][0]) not in json.dumps(receipt)


def test_direct_projection_requires_independent_root_and_exact_task(retained):
    parsed = audit._receipt(retained["receipt"], retained["workspace_root"])
    model, _advisory, checks = audit._model_projection(rendered=TASK, receipt=parsed, repository=None,
        world_packet_artifact_root=retained["root"], workspace_root=retained["workspace_root"])
    assert model == retained["case"][3]["seen"][-1][0] and all(checks.values())
    with pytest.raises(ValueError):
        audit._model_projection(rendered=TASK, receipt=parsed, repository=None)
    with pytest.raises(ValueError):
        audit._model_projection(rendered="different task", receipt=parsed, repository=None,
            world_packet_artifact_root=retained["root"], workspace_root=retained["workspace_root"])


def test_default_private_state_root_needs_no_legacy_capsule_files(retained):
    result = inspect(retained)
    assert result["status"] == "verified_all_observed_model_inputs"
    assert result["coding_receipts_seen"] == result["usable_coding_receipts"] == 1
    assert result["capsules"] == [] and result["read_errors"] == []
    assert result["matches"][0]["world_packet_replay"]["source_currentness_verified"] is False
    assert result["invocations"][0]["verification_scope"] == "historical_world_packet_input_reconstruction"
    assert result["raw_prompts_exported"] is False


def test_child_collector_retains_full_world_markers_and_independent_root(retained):
    result = audit.collect_terminal_context_audit(state=retained["state"], receipts=[retained["receipt"]],
        workspace_root=retained["workspace_root"], world_packet_artifact_root=retained["root"], timeout_seconds=15)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["matches"][0]["world_packet_replay"]["owner_currentness_verified"] is False
    assert result["provider_calls"] == 0 and result["task_correctness_established"] is False


def test_historical_reconstruction_survives_source_loss_without_live_operations(retained, monkeypatch):
    import duckdb
    import sqlite3
    import socket
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    source, session, _packet, _provider = retained["case"]
    session.close()
    shutil.rmtree(source)
    def forbidden(*_args, **_kwargs):
        pytest.fail("historical audit opened a live database/source/provider owner")
    for owner, name in ((duckdb, "connect"), (sqlite3, "connect"), (socket, "socket"),
                        (llm_router, "get_llm_provider"), (llm_router, "generate_text"),
                        (RepositorySourceSession, "__init__")):
        monkeypatch.setattr(owner, name, forbidden)
    result = inspect(retained, state=retained["workspace_root"] / "unavailable-state",
        world_packet_artifact_root=retained["root"])
    assert result["status"] == "verified_all_observed_model_inputs"
    assert result["matches"][0]["world_packet_replay"]["source_currentness_verified"] is False


@pytest.mark.parametrize("field", ["invocation_id", "model", "model_prompt_sha256", "workspace"])
def test_original_receipt_projection_cannot_be_substituted(retained, field):
    changed = deepcopy(retained["receipt"])
    changed[field] = {"invocation_id": "f" * 32, "model": "foreign-model",
        "model_prompt_sha256": "0" * 64, "workspace": str(retained["workspace_root"] / "foreign-workspace")}[field]
    with pytest.raises((ValueError, OSError)):
        reconstruct(retained, receipt=changed)


@pytest.mark.parametrize("field", ["semantic_translation", "semantic_metadata_view", "source_formalization_context",
    "query_reference_view", "source_input_policy", "source_view_retrieval"])
def test_unsupported_compositions_are_refused_and_not_stripped_by_child(retained, field):
    changed = deepcopy(retained["receipt"])
    changed[field] = "source-input-raw@1" if field == "source_input_policy" else {"unexpected": True}
    with pytest.raises(ValueError):
        reconstruct(retained, receipt=changed)
    observed = audit.collect_terminal_context_audit(state=retained["state"], receipts=[changed],
        workspace_root=retained["workspace_root"], world_packet_artifact_root=retained["root"], timeout_seconds=15)
    assert observed["all_observed_coding_inputs_verified"] is not True
    assert observed["matches"] == []


@pytest.mark.parametrize("kind", ["packet", "invocation"])
@pytest.mark.parametrize("change", ["missing", "bytes", "symlink", "hardlink", "public_mode"])
def test_artifact_pin_type_and_private_custody_are_enforced(retained, kind, change):
    descriptor = retained["receipt"]["world_packet_artifacts"][kind]
    path = retained["root"] / descriptor["name"]
    if change == "missing": path.unlink()
    elif change == "bytes": path.write_bytes(b"tampered")
    elif change == "symlink":
        other = retained["root"] / "other"
        path.rename(other); path.symlink_to(other)
    elif change == "hardlink": os.link(path, retained["root"] / "other")
    else: path.chmod(0o644)
    with pytest.raises((ValueError, OSError)):
        reconstruct(retained)
    assert inspect(retained)["all_observed_coding_inputs_verified"] is False


@pytest.mark.parametrize("change", ["escape", "extra_root", "boolean_bytes", "wrong_sha"])
def test_retention_descriptor_schema_and_pins_are_closed(retained, change):
    changed = deepcopy(retained["receipt"])
    binding = changed["world_packet_artifacts"]
    if change == "escape": binding["packet"]["name"] = "../foreign.json"
    elif change == "extra_root": binding["root"] = str(retained["root"])
    elif change == "boolean_bytes": binding["invocation"]["bytes"] = True
    else: binding["packet"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, OSError)):
        reconstruct(retained, receipt=changed)


def test_rehashed_invocation_artifact_still_requires_original_receipt_and_packet(retained):
    binding = deepcopy(retained["receipt"]["world_packet_artifacts"])
    path = retained["root"] / binding["invocation"]["name"]
    artifact = json.loads(path.read_bytes())
    artifact["invocation"]["model"] = "forged-model"
    raw = replay._canonical_bytes(artifact)
    path.write_bytes(raw)
    changed = deepcopy(retained["receipt"])
    changed["world_packet_artifacts"]["invocation"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="original_invocation_join"):
        reconstruct(retained, receipt=changed)


def test_packet_descriptor_is_not_replaced_by_embedded_invocation_copy(retained):
    changed = deepcopy(retained["receipt"])
    descriptor = changed["world_packet_artifacts"]["packet"]
    path = retained["root"] / descriptor["name"]
    raw = b'{"different":"packet"}'
    path.write_bytes(raw)
    descriptor.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="packet_invocation_join"):
        reconstruct(retained, receipt=changed)


def test_missing_or_unavailable_retention_cannot_downgrade_to_legacy(retained):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_artifact_retention import (
        unavailable_world_packet_artifacts,
    )
    missing = deepcopy(retained["receipt"])
    missing.pop("world_packet_artifacts")
    assert inspect(retained, receipts=[missing])["matches"] == []
    unavailable = deepcopy(retained["receipt"])
    unavailable["world_packet_artifacts"] = unavailable_world_packet_artifacts(
        invocation_id=unavailable["invocation_id"], reason="packet_retention_failed")
    observed = inspect(retained, receipts=[unavailable])
    assert observed["all_observed_coding_inputs_verified"] is False
    assert observed["invocations"][0]["reason"] == "retained_world_packet_input_unavailable"


def test_root_is_independent_private_and_symlink_free(retained):
    foreign = retained["workspace_root"] / "foreign-root"
    foreign.mkdir(mode=0o700)
    with pytest.raises((ValueError, OSError)):
        reconstruct(retained, artifact_root=foreign)
    alias = retained["workspace_root"] / "alias-root"
    alias.symlink_to(retained["root"], target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        reconstruct(retained, artifact_root=alias)
    retained["state"].chmod(0o755)
    assert inspect(retained)["all_observed_coding_inputs_verified"] is False
    assert inspect(retained, world_packet_artifact_root=retained["root"])["all_observed_coding_inputs_verified"] is True


def test_changed_file_during_replay_is_refused_without_rewriting_artifacts(retained, monkeypatch):
    original = replay.replay_repository_world_invocation
    path = retained["root"] / retained["receipt"]["world_packet_artifacts"]["packet"]["name"]
    def raced(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_bytes(b"foreign replacement")
        return result
    monkeypatch.setattr(replay, "replay_repository_world_invocation", raced)
    with pytest.raises(ValueError, match="changed_after_replay"):
        reconstruct(retained)
    assert path.read_bytes() == b"foreign replacement"


def test_receipt_aggregate_budget_precedes_owner_projection(retained, monkeypatch):
    changed = deepcopy(retained["receipt"])
    changed["world_task_packet"]["unexpected"] = "x" * 4096
    monkeypatch.setattr(world_audit, "MAX_RECEIPT_BYTES", 1024)
    monkeypatch.setattr(replay, "_projection", lambda *_args: pytest.fail("copied before bounded preflight"))
    with pytest.raises(ValueError, match="receipt_byte_bound"):
        reconstruct(retained, receipt=changed)


def test_duplicate_receipts_preserve_first_observation_without_claiming_complete_population(retained):
    observed = inspect(retained, receipts=[retained["receipt"], retained["receipt"]])
    assert observed["any_model_input_verified"] is True
    assert observed["all_observed_coding_inputs_verified"] is False
    assert observed["coding_receipts_seen"] == 2 and observed["usable_coding_receipts"] == 1


@pytest.mark.parametrize("legacy_capture", ["missing_capture_files", "absent_state"])
def test_mixed_population_missing_legacy_capture_keeps_world_match_and_open_frontier(retained, legacy_capture):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    legacy = deepcopy(retained["receipt"])
    legacy.pop("world_task_packet"); legacy.pop("world_packet_artifacts")
    legacy["invocation_id"] = "e" * 32
    model, advisory = render_model_prompt(prompt=TASK, purpose="coding", workspace=Path(legacy["workspace"]))
    for prefix, text in (("prompt", TASK), ("native_prompt", TASK), ("router_prompt", TASK),
                         ("model_prompt", model), ("workspace_advisory", advisory)):
        legacy[prefix + "_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        legacy[prefix + "_bytes"] = len(text.encode())
    state = retained["state"] if legacy_capture == "missing_capture_files" else retained["workspace_root"] / "absent-state"
    observed = inspect(retained, receipts=[retained["receipt"], legacy], state=state,
        world_packet_artifact_root=retained["root"])
    assert observed["status"] == "verified_some_observed_model_inputs"
    assert observed["usable_coding_receipts"] == 2
    assert len(observed["matches"]) == 1 and observed["matches"][0]["invocation_id"] == retained["receipt"]["invocation_id"]
    assert observed["all_observed_coding_inputs_verified"] is False
    assert any(row["kind"] == "capture" for row in observed["read_errors"])
    assert all(observed[key] is False for key in ("proof_authority", "formalization_authority",
        "execution_authority", "completion_authority", "task_correctness_established"))
    assert all(observed["matches"][0]["world_packet_replay"][key] is False
        for key in world_audit.FALSE_FIELDS)

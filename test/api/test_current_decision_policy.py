"""Closed launch controls and real selected archive members; no model calls."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_decision_policy as policy


PIN = "d" * 64
SELECTED = {"semantic_decision_policy_artifact": "/opt/ipfs-supervisor/policy.json",
            "semantic_decision_policy_sha256": PIN}
NOMINATED = {"semantic_decision_nomination_artifact": "/opt/ipfs-supervisor/nomination.json",
             "semantic_decision_nomination_sha256": PIN}


def test_default_group_is_inert_and_does_not_inspect_unrelated_route_state():
    assert policy.decision_selection() == {}
    assert policy.semantic_decision_arguments() == []
    policy.validate_decision_route({}, arm="no-index", provider="grok_cli", unrelated=object())
    assert policy.decision_runtime_sources({}) == {}
    assert policy.require_decision_archive(object(), object(), {}) == {}


@pytest.mark.parametrize("options", [
    {"semantic_decision_policy_artifact": "/p.json"},
    {"semantic_decision_policy_sha256": PIN},
    {"semantic_decision_nomination_artifact": "/n.json"},
    {"semantic_decision_nomination_sha256": PIN},
    {**SELECTED, **NOMINATED},
    {**NOMINATED, "semantic_decision_nomination_source": "/host/n.json"},
    {**SELECTED, "semantic_decision_policy_artifact": "relative.json"},
    {**SELECTED, "semantic_decision_policy_artifact": "/a/../p.json"},
    {**SELECTED, "semantic_decision_policy_artifact": "/a//p.json"},
    {**SELECTED, "semantic_decision_policy_sha256": "D" * 64},
    {**SELECTED, "semantic_decision_max_calls": True},
    {**SELECTED, "semantic_decision_max_calls": 5},
    {**SELECTED, "semantic_decision_max_total_input_tokens": False},
    {**SELECTED, "semantic_decision_max_total_input_tokens": 0},
    {"semantic_decision_max_calls": 2},
    {"semantic_decision_max_total_input_tokens": 100},
    {"disable_semantic_typed_nomination": True},
    {**SELECTED, "disable_semantic_typed_nomination": True},
    {**SELECTED, "semantic_retrieval_query_reference": True},
    {"semantic_retrieval_query_reference": True},
    {**NOMINATED, "semantic_retrieval_query_reference": 1},
    {**NOMINATED, "disable_semantic_typed_nomination": 1},
])
def test_invalid_controls_refuse_without_reading_artifacts(options):
    with pytest.raises(ValueError):
        policy.decision_selection(**options)


@pytest.mark.parametrize("options", [
    {"arm": "no-index"}, {"provider": "grok_cli"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"source_view_profile": "source384@1"}, {"source384_config": Path("/unread/config")},
    {"numerical_selection_config": Path("/unread/numerical")},
    {"metadata_census_only": True}, {"metadata_census_only": 0},
    {"planning_mode": "declared"}, {"timeout": 601}, {"timeout": True},
    {"purpose": "planning"}, {"unknown_mode": "enabled"},
])
def test_selected_unsupported_routes_refuse_before_effects(options):
    values = {"arm": "full", "provider": "codex_cli", **options}
    with pytest.raises(ValueError):
        policy.validate_decision_route(policy.decision_selection(**SELECTED), **values)


def test_owner_arguments_keep_task_identity_out_of_host_selection():
    nominated = {**NOMINATED, "disable_semantic_typed_nomination": True,
                 "semantic_retrieval_query_reference": True,
                 "semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 30000}
    result = policy.semantic_decision_arguments(**nominated, owner_binding=True)
    assert result == ["--semantic-decision-nomination-artifact", NOMINATED["semantic_decision_nomination_artifact"],
        "--semantic-decision-nomination-sha256", PIN, "--semantic-decision-typed-nomination", "disabled",
        "--semantic-decision-max-calls", "2", "--semantic-decision-max-total-input-tokens", "30000"]
    assert not any("task" in value or "bundle" in value for value in result)
    host = {"semantic_decision_nomination_source": "/host/n.json", "semantic_decision_nomination_sha256": PIN}
    with pytest.raises(ValueError, match="delivered"):
        policy.semantic_decision_arguments(**host)


@pytest.mark.parametrize("disabled", [False, True])
def test_retrieval_reference_preserves_typed_default_and_exact_combined_inventory(disabled):
    options = {**NOMINATED, "semantic_retrieval_query_reference": True}
    if disabled:
        options["disable_semantic_typed_nomination"] = True
    selected = policy.decision_selection(**options)
    assert selected == {**options, "semantic_decision_max_calls": 3,
                        "semantic_decision_max_total_input_tokens": None}
    owner = policy.semantic_decision_arguments(**selected, owner_binding=True)
    worker = policy.semantic_decision_arguments(**selected)
    mode = "disabled" if disabled else "enabled"
    assert owner == ["--semantic-decision-nomination-artifact", NOMINATED["semantic_decision_nomination_artifact"],
        "--semantic-decision-nomination-sha256", PIN, "--semantic-decision-typed-nomination", mode,
        "--semantic-decision-max-calls", "3"]
    assert "--semantic-retrieval-query-reference" not in owner
    assert worker.count("--semantic-retrieval-query-reference") == 1
    runtime = "source/ipfs_accelerate_py/agent_supervisor/runtime/"
    retrieval = {runtime + name for name in (
        "semantic_retrieval_query_reference.py", "code_retrieval_context.py", "task_context_bundle.py")}
    base_options = {key: value for key, value in options.items() if key != "semantic_retrieval_query_reference"}
    base = policy.decision_runtime_sources(policy.decision_selection(**base_options))
    combined = policy.decision_runtime_sources(selected)
    assert set(combined) == set(base) | retrieval
    assert len(combined) == (36 if disabled else 43)
    assert (runtime + "semantic_typed_nomination_lineage.py" in combined) is not disabled


def _archive(tmp_path, *, damage=None, retrieval=False):
    selected = policy.decision_selection(**NOMINATED, semantic_retrieval_query_reference=retrieval)
    sources = policy.decision_runtime_sources(selected)
    target = next(iter(sources))
    archive = tmp_path / "runtime.tar.gz"
    rows = []
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sources.items():
            raw = path.read_bytes()
            mode = policy._source_row(path)["mode"]
            row = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mode": mode}
            if name == target and damage == "stale_row":
                row["sha256"] = "0" * 64
            if name == target and damage == "boolean_mode":
                row["mode"] = True
            if name != target or damage != "missing_row":
                rows.append(row)
            if name == target and damage == "duplicate_row":
                rows.append(deepcopy(row))
            if name == target and damage == "missing_member":
                continue
            if name == target and damage == "payload":
                raw = b"!" + raw[1:]
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), mode
            if name == target and damage == "link":
                member.type, member.linkname, member.size = tarfile.SYMTYPE, "/outside", 0
                bundle.addfile(member)
            else:
                bundle.addfile(member, io.BytesIO(raw))
                if name == target and damage == "duplicate_member":
                    bundle.addfile(member, io.BytesIO(raw))
    manifest = {"files": rows, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    return selected, archive, manifest


@pytest.mark.parametrize("retrieval", [False, True])
def test_real_archive_requires_exact_selected_bytes_and_rechecks_local_source(tmp_path, retrieval):
    selected, archive, manifest = _archive(tmp_path, retrieval=retrieval)
    pins = policy.require_decision_archive(archive, manifest, selected)
    assert pins == {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in policy.decision_runtime_sources(selected).items()}
    policy.require_decision_source_pins(pins, selected)
    changed = dict(pins)
    changed[next(iter(changed))] = "0" * 64
    with pytest.raises(ValueError, match="source pins"):
        policy.require_decision_source_pins(changed, selected)


@pytest.mark.parametrize("damage", ["missing_row", "duplicate_row", "stale_row", "boolean_mode",
                                     "missing_member", "duplicate_member", "payload", "link"])
@pytest.mark.parametrize("retrieval", [False, True])
def test_self_consistent_tar_digest_does_not_authenticate_missing_or_changed_sources(tmp_path, damage, retrieval):
    selected, archive, manifest = _archive(tmp_path, damage=damage, retrieval=retrieval)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with pytest.raises(ValueError, match="selected decision"):
        policy.require_decision_archive(archive, manifest, selected)

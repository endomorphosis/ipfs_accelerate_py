"""Owner bridge controls with real nomination-file custody and authored source seams.

These tests exercise policy selection, independent bindings, closing fences and
immutable results. Native source verification, graph projection and launch
construction are explicit substitutes; the companion native probe qualifies
those integrations. No model, provider, scanner, database or Git fixture runs.
"""

from __future__ import annotations

from base64 import b32encode
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import (
    SemanticDecisionPolicy,
)

TASK = "TASK-AUTHORED-TYPED-OWNER"
ARTIFACT = ".runtime/semantic/worker-context.json"
GRAPH_ROOT = "decision-root:authored-typed-owner"
SOURCE = b"def helper(value):\n    return value + 1\n\ndef target(value):\n    return helper(value)\n"


def _cid(label):
    return "b" + b32encode(b"\x01\x55\x12\x20" + sha256(label.encode()).digest()).decode().lower().rstrip("=")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(value):
    return sha256(value).hexdigest()


@pytest.fixture
def authored(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", "0")
    repository, workspace = tmp_path / "repository", tmp_path / "candidate"
    repository.mkdir()
    workspace.mkdir()
    source = repository / "operations.py"
    source.write_bytes(SOURCE)
    artifact = repository / ARTIFACT
    artifact.parent.mkdir(parents=True)
    names = ("target", "helper", "unrelated_required", "already_selected", "omitted")
    symbols = {name: _cid("symbol:" + name) for name in names}
    capsules = {name: _cid("capsule:" + name) for name in names}
    root_cid = _cid("authored-semantic-root")
    manifest = {"operations.py": {"sha256": _sha(SOURCE), "source_cid": _cid("source")}}
    payload = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": TASK,
        "semantic_root_cid": root_cid, "scope_cid": _cid("source-scope"),
        "manifest": manifest, "required_raw_paths": ["operations.py"],
        "raw_sources": {"operations.py": SOURCE.decode()},
        "capsules": [{"stable_symbol_id": symbols[name], "capsule_cid": capsules[name]}
                     for name in names],
        "admissions": [], "completion_authority": False,
    }
    artifact.write_bytes(_encoded(payload))
    policy = SemanticDecisionPolicy(
        required_capsule_ids=(capsules["unrelated_required"], capsules["target"]),
        optional_capsule_ids=(capsules["omitted"], capsules["helper"], capsules["already_selected"]),
        selected_optional_capsule_ids=(capsules["already_selected"],),
    )
    owner = tmp_path / "owner-nomination.json"
    owner.write_bytes(_encoded(bridge.build_decision_nomination(policy)))
    owner.chmod(0o444)
    parent_text = artifact.read_text()
    wire = {"objective_id": TASK, "repository_id": "authored-repository",
            "tree_id": "authored-tree", "evidence": []}
    prompt = _encoded(wire).decode() + "\nPreserve literal public instruction café.\n"
    unresolved = {"edge_id": _cid("authored-global-read"), "source_stable_id": symbols["target"],
                  "relation": "reads_state", "target_kind": "unresolved", "target_stable_id": None,
                  "unresolved_target": "global:helper", "obligation_node_id": "authored-obligation"}
    proposal = {
        "schema": "authored-source-bound-typed-dependency@1", "root_cid": root_cid,
        "semantic_root_cid": root_cid, "graph_root_id": GRAPH_ROOT,
        "full_symbol_ids": sorted(symbols.values()),
        "nominee_symbol_ids": sorted(symbols[name] for name in ("target", "helper", "unrelated_required")),
        "graph": {"authored_fixture": True, "nodes": [
            {"record": {"native_symbol_id": symbols[name], "capsule_cid": capsules[name],
                        "source_module_path": "operations.py", "source_cid": manifest["operations.py"]["source_cid"]}}
            for name in names]},
        "source_binding": {"task_id": TASK, "artifact": ARTIFACT, "artifact_sha256": _sha(artifact.read_bytes()),
                           "semantic_root_cid": root_cid, "scope_cid": payload["scope_cid"],
                           "source_manifest": manifest, "required_raw_paths": ["operations.py"],
                           "raw_source_paths": ["operations.py"]},
        "unresolved_state_binding_rows": [unresolved], "unresolved_other_reference_rows": [],
        "incoming_relations_automatically_selected": False,
        "source_freshness_verified": True, "owner_task_policy_verified": False,
        "dependency_semantics_complete": False, "effective_task_coverage_verified": False,
        "source_substitution_authority": False,
        "execution_authority": False, "proof_authority": False, "completion_authority": False,
        "runtime_activation": False, "deployment_ready": False,
    }
    state = SimpleNamespace(repository=repository, workspace=workspace, source=source,
                            artifact=artifact, owner=owner, policy=policy, prompt=prompt,
                            parent_text=parent_text, payload=payload, root_cid=root_cid,
                            symbols=symbols, capsules=capsules, proposal=proposal,
                            project_calls=[], verify_calls=[], launch_calls=[],
                            after_projection=None, after_launch=None)

    def prompt_parts(received):
        parsed, end = json.JSONDecoder().raw_decode(received)
        return parsed, received[end:], (), (), state.parent_text, ARTIFACT

    def verified(root, *, artifact, text, task_id, current):
        state.verify_calls.append((root, artifact, task_id, current))
        assert current is True
        raw = bridge.codec._read(root, artifact)
        if raw != text.encode():
            raise ValueError("authored source verifier: parent changed")
        value = json.loads(raw)
        if value["task_id"] != task_id:
            raise ValueError("authored source verifier: task differs")
        for path, binding in value["manifest"].items():
            actual = bridge.codec._read(root, path)
            if actual != SOURCE or _sha(actual) != binding["sha256"]:
                raise ValueError("authored source verifier: current source stale")
        return value, tuple(sorted(row["stable_symbol_id"] for row in value["capsules"]))

    def project(**kwargs):
        state.project_calls.append(kwargs)
        verified(kwargs["repository"], artifact=kwargs["artifact"], text=state.parent_text,
                 task_id=kwargs["task_id"], current=True)
        assert kwargs["expected_sha256"] == _sha(state.parent_text.encode())
        assert kwargs["expected_semantic_root_cid"] == state.root_cid
        assert kwargs["graph_root_id"] == GRAPH_ROOT
        if state.after_projection is not None:
            state.after_projection()
        return deepcopy(state.proposal)

    def launch(*, native_prompt, policy, repository):
        state.launch_calls.append((native_prompt, policy, repository))
        if state.after_launch is not None:
            state.after_launch()
        return {"schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
                "task_id": TASK, "parent_artifact": ARTIFACT,
                "parent_sha256": _sha(state.parent_text.encode()), "policy": policy.to_dict()}

    monkeypatch.setattr(bridge.codec, "_prompt_parts", prompt_parts)
    monkeypatch.setattr(bridge.codec, "_verified_semantic", verified)
    monkeypatch.setattr(bridge, "project_source_bound_typed_links", project)
    monkeypatch.setattr(bridge, "build_decision_launch_policy", launch)
    return state


def _build(state, **changes):
    arguments = {
        "native_prompt": state.prompt, "repository": state.repository, "workspace": state.workspace,
        "nomination_artifact": state.owner, "nomination_sha256": _sha(state.owner.read_bytes()),
        "task_id": TASK, "expected_parent_sha256": _sha(state.parent_text.encode()),
        "expected_semantic_root_cid": state.root_cid, "graph_root_id": GRAPH_ROOT,
    }
    arguments.update(changes)
    return bridge.build_typed_owner_decision_nomination(**arguments)


def _replace_owner(state, policy):
    state.owner.chmod(0o600)
    state.owner.write_bytes(_encoded(bridge.build_decision_nomination(policy)))
    state.owner.chmod(0o444)


def test_owner_partition_order_is_preserved_and_known_optional_dependencies_are_selected(authored):
    before = (authored.source.read_bytes(), authored.artifact.read_bytes(), authored.owner.read_bytes())
    policy_before = authored.policy.to_dict()
    result = _build(authored)
    selected = result.policy
    assert type(selected) is SemanticDecisionPolicy
    assert selected.required_capsule_ids == authored.policy.required_capsule_ids
    assert selected.optional_capsule_ids == authored.policy.optional_capsule_ids
    assert set(selected.selected_optional_capsule_ids) == {
        authored.capsules["helper"], authored.capsules["already_selected"]}
    assert selected.selected_optional_capsule_ids == (
        authored.capsules["already_selected"], authored.capsules["helper"])
    assert authored.capsules["helper"] in selected.optional_capsule_ids
    assert authored.capsules["helper"] not in selected.required_capsule_ids
    assert authored.capsules["omitted"] not in selected.selected_optional_capsule_ids
    assert result.nomination == bridge.build_decision_nomination(selected)
    assert set(result.nomination) == {"schema", "policy"}
    assert set(result.launch_policy) == {"schema", "task_id", "parent_artifact", "parent_sha256", "policy"}
    assert result.launch_policy["policy"] == selected.to_dict()
    assert authored.policy.to_dict() == policy_before
    assert before == (authored.source.read_bytes(), authored.artifact.read_bytes(), authored.owner.read_bytes())
    assert authored.owner.stat().st_mode & 0o777 == 0o444
    assert authored.project_calls[0]["seed_symbol_ids"] == (
        authored.symbols["unrelated_required"], authored.symbols["target"])


def test_receipt_keeps_unresolved_bindings_and_independent_policy_bindings_without_authority(authored):
    result = _build(authored)
    receipt = result.receipt
    assert receipt["nomination_sha256"] == _sha(authored.owner.read_bytes())
    assert receipt["original_policy_sha256"] == _sha(_encoded(authored.policy.to_dict()))
    assert receipt["selected_policy_sha256"] == _sha(_encoded(result.policy.to_dict()))
    assert receipt["original_policy_sha256"] != receipt["selected_policy_sha256"]
    assert receipt["parent_sha256"] == _sha(authored.parent_text.encode())
    assert receipt["semantic_root_cid"] == authored.root_cid
    assert receipt["added_selected_optional_capsule_ids"] == [authored.capsules["helper"]]
    assert receipt["source_proposal"]["unresolved_state_binding_rows"] == authored.proposal["unresolved_state_binding_rows"]
    assert receipt["source_proposal"]["incoming_relations_automatically_selected"] is False
    for flag in ("owner_required_partition_preserved", "owner_optional_partition_preserved",
                 "prior_selected_optional_preserved", "source_freshness_verified", "owner_declaration_rechecked"):
        assert receipt[flag] is True
    for flag in ("owner_task_coverage_verified", "effective_task_coverage_verified",
                 "source_substitution_authority", "proof_authority", "execution_authority",
                 "completion_authority", "deployment_ready", "runtime_activation"):
        assert receipt[flag] is False
    assert SOURCE.decode() not in json.dumps(receipt)


@pytest.mark.parametrize("location", ["candidate", "symlink", "ancestor_symlink", "directory", "over_bound"])
def test_independent_owner_file_uses_existing_bounded_canonical_outside_candidate_loader(authored, tmp_path, location):
    if location == "candidate":
        path = authored.workspace / "owner.json"
        path.write_bytes(authored.owner.read_bytes())
    elif location == "symlink":
        path = tmp_path / "owner-link.json"
        path.symlink_to(authored.owner)
    elif location == "ancestor_symlink":
        linked = tmp_path / "linked-parent"
        linked.symlink_to(authored.owner.parent, target_is_directory=True)
        path = linked / authored.owner.name
    elif location == "directory":
        path = tmp_path / "owner-directory"
        path.mkdir()
    else:
        path = tmp_path / "oversized-owner.json"
        path.write_bytes(b" " * 256_001)
    digest = _sha(path.read_bytes()) if path.is_file() else "0" * 64
    with pytest.raises(ValueError):
        _build(authored, nomination_artifact=path, nomination_sha256=digest)
    assert authored.project_calls == authored.launch_calls == []


@pytest.mark.parametrize("damage", ["hash", "duplicate_top", "duplicate_policy", "extra", "schema"])
def test_independent_owner_hash_and_closed_json_cannot_be_relaxed(authored, damage):
    if damage == "hash":
        with pytest.raises(ValueError):
            _build(authored, nomination_sha256="0" * 64)
    else:
        raw = authored.owner.read_bytes()
        if damage == "duplicate_top":
            raw = raw.replace(b'{"policy":', b'{"schema":"shadow","policy":', 1)
        elif damage == "duplicate_policy":
            raw = raw.replace(b'"policy":{', b'"policy":{"optional_capsule_ids":[],', 1)
        else:
            value = json.loads(raw)
            if damage == "extra":
                value["allow_inferred_optional"] = True
            else:
                value["schema"] = "unsupported-owner@9"
            raw = _encoded(value)
        authored.owner.chmod(0o600)
        authored.owner.write_bytes(raw)
        authored.owner.chmod(0o444)
        with pytest.raises(ValueError):
            _build(authored)
    assert authored.project_calls == authored.launch_calls == []


@pytest.mark.parametrize("damage", ["foreign", "stale", "missing", "no_required", "revision2"])
def test_complete_owner_partition_and_supported_revision_are_required_before_projection(authored, damage):
    policy = authored.policy
    if damage in {"foreign", "stale"}:
        policy = SemanticDecisionPolicy(
            required_capsule_ids=policy.required_capsule_ids,
            optional_capsule_ids=(_cid(damage + "-capsule"), *policy.optional_capsule_ids[1:]),
            selected_optional_capsule_ids=policy.selected_optional_capsule_ids)
    elif damage == "missing":
        policy = SemanticDecisionPolicy(
            required_capsule_ids=policy.required_capsule_ids,
            optional_capsule_ids=policy.optional_capsule_ids[1:],
            selected_optional_capsule_ids=policy.selected_optional_capsule_ids)
    elif damage == "no_required":
        policy = SemanticDecisionPolicy(required_capsule_ids=(),
            optional_capsule_ids=policy.required_capsule_ids + policy.optional_capsule_ids,
            selected_optional_capsule_ids=policy.selected_optional_capsule_ids)
    else:
        policy = SemanticDecisionPolicy(required_capsule_ids=policy.required_capsule_ids,
            optional_capsule_ids=policy.optional_capsule_ids,
            selected_optional_capsule_ids=policy.selected_optional_capsule_ids,
            task_text_source_path="instruction.md")
    _replace_owner(authored, policy)
    with pytest.raises(ValueError):
        _build(authored)
    assert authored.project_calls == authored.launch_calls == []


@pytest.mark.parametrize("changes", [
    {"task_id": "TASK-FOREIGN"}, {"expected_parent_sha256": "0" * 64},
    {"expected_semantic_root_cid": _cid("foreign-root")},
])
def test_parent_task_root_and_hash_bindings_are_independent(authored, changes):
    with pytest.raises(ValueError):
        _build(authored, **changes)
    assert authored.project_calls == authored.launch_calls == []


@pytest.mark.parametrize("damage", ["missing", "duplicate_symbol", "duplicate_capsule"])
def test_native_parent_population_cannot_be_narrowed_or_aliased(authored, damage):
    payload = deepcopy(authored.payload)
    if damage == "missing":
        payload["capsules"].pop()
    elif damage == "duplicate_symbol":
        payload["capsules"][1]["stable_symbol_id"] = payload["capsules"][0]["stable_symbol_id"]
    else:
        payload["capsules"][1]["capsule_cid"] = payload["capsules"][0]["capsule_cid"]
    authored.artifact.write_bytes(_encoded(payload))
    authored.parent_text = authored.artifact.read_text()
    with pytest.raises(ValueError):
        _build(authored)
    assert authored.launch_calls == []


@pytest.mark.parametrize("damage", ["full_missing", "full_duplicate", "foreign_nominee",
    "duplicate_nominee", "required_seed_omitted", "binding_task", "binding_parent",
    "graph_root", "freshness", "owner_authority", "execution_authority"])
def test_typed_source_proposal_cannot_mint_foreign_selection_or_authority(authored, damage):
    proposal = authored.proposal
    if damage == "full_missing":
        proposal["full_symbol_ids"].pop()
    elif damage == "full_duplicate":
        proposal["full_symbol_ids"].append(proposal["full_symbol_ids"][0])
    elif damage == "foreign_nominee":
        proposal["nominee_symbol_ids"].append(_cid("foreign-proposal-symbol"))
    elif damage == "duplicate_nominee":
        proposal["nominee_symbol_ids"].append(proposal["nominee_symbol_ids"][0])
    elif damage == "required_seed_omitted":
        proposal["nominee_symbol_ids"].remove(authored.symbols["unrelated_required"])
    elif damage == "binding_task":
        proposal["source_binding"]["task_id"] = "TASK-FOREIGN"
    elif damage == "binding_parent":
        proposal["source_binding"]["artifact_sha256"] = "0" * 64
    elif damage == "graph_root":
        proposal["graph_root_id"] = "foreign-graph-root"
    elif damage == "freshness":
        proposal["source_freshness_verified"] = False
    elif damage == "owner_authority":
        proposal["owner_task_policy_verified"] = True
    else:
        proposal["execution_authority"] = True
    with pytest.raises(ValueError):
        _build(authored)
    assert authored.launch_calls == []


@pytest.mark.parametrize("field", ["task_id", "parent_artifact", "parent_sha256", "policy"])
def test_launch_builder_result_cannot_change_parent_task_or_selected_policy(authored, monkeypatch, field):
    original = bridge.build_decision_launch_policy
    def changed(**kwargs):
        value = original(**kwargs)
        value[field] = {} if field == "policy" else "foreign-binding"
        return value
    monkeypatch.setattr(bridge, "build_decision_launch_policy", changed)
    with pytest.raises(ValueError):
        _build(authored)


@pytest.mark.parametrize("stage", ["projection", "launch"])
@pytest.mark.parametrize("target", ["source", "parent", "owner"])
def test_source_parent_or_owner_drift_after_work_cannot_pass_closing_fences(authored, stage, target):
    def change():
        path = {"source": authored.source, "parent": authored.artifact, "owner": authored.owner}[target]
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n# authored concurrent change\n")
    setattr(authored, "after_" + stage, change)
    with pytest.raises(ValueError):
        _build(authored)
    assert authored.verify_calls


@pytest.mark.parametrize("limits", [
    {"max_selected_symbols": True}, {"max_selected_symbols": 0}, {"max_selected_symbols": 33},
    {"max_graph_nodes": 2049}, {"max_graph_edges": 8193}, {"max_report_bytes": 2_000_001},
    {"max_report_bytes": 1}, {"max_selected_symbols": 1},
])
def test_bridge_budgets_refuse_without_truncating_required_material(authored, limits):
    with pytest.raises(ValueError):
        _build(authored, **limits)


def test_authored_source_proposal_refusal_is_not_replaced_by_an_inferred_policy(authored, monkeypatch):
    def refuse(**_kwargs):
        raise ValueError("authored proposal exceeds complete dependency bound")
    monkeypatch.setattr(bridge, "project_source_bound_typed_links", refuse)
    with pytest.raises(ValueError, match="complete dependency bound"):
        _build(authored)
    assert authored.launch_calls == []


def test_result_properties_return_copies_and_result_binding_is_frozen(authored):
    result = _build(authored)
    snapshot = (result.nomination, result.launch_policy, result.policy.to_dict(), result.receipt)
    nomination = result.nomination
    nomination["policy"]["required_capsule_ids"].clear()
    launch = result.launch_policy
    launch["policy"]["optional_capsule_ids"].clear()
    receipt = result.receipt
    receipt["source_proposal"]["nominee_symbol_ids"].clear()
    assert snapshot == (result.nomination, result.launch_policy, result.policy.to_dict(), result.receipt)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.result_json = "{}"


def _scoped_authored_proposal(authored):
    """Authored source seam; native full-root composition is qualified separately."""
    proposal = authored.proposal
    records = sorted((deepcopy(node["record"]) for node in proposal["graph"]["nodes"]),
                     key=lambda record: record["native_symbol_id"])
    nominees = proposal["nominee_symbol_ids"]
    proposal.update(schema=bridge.SCOPED_SCHEMA, adapter_version="native-typed-dependency-proposal@2",
                    graph_scope="task_closure", root_symbol_records=records,
                    root_symbol_records_count=len(records), root_symbol_records_sha256=_sha(_encoded(records)),
                    materialized_symbol_ids=list(nominees),
                    omitted_root_symbol_ids=sorted(set(proposal["full_symbol_ids"])-set(nominees)))
    proposal["graph"]["nodes"] = [node for node in proposal["graph"]["nodes"]
                                  if node["record"]["native_symbol_id"] in nominees]


def test_owner_task_graph_defaults_on_and_preserves_independent_partition(authored, monkeypatch):
    _scoped_authored_proposal(authored)
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", raising=False)
    result = _build(authored)
    assert authored.project_calls[0]["graph_scope"] == "task_closure"
    assert result.policy.required_capsule_ids == authored.policy.required_capsule_ids
    assert result.policy.optional_capsule_ids == authored.policy.optional_capsule_ids
    assert result.policy.selected_optional_capsule_ids == (
        authored.capsules["already_selected"], authored.capsules["helper"])
    assert len(result.receipt["source_proposal"]["graph"]["nodes"]) == 3
    assert result.receipt["source_proposal"]["root_symbol_records_count"] == 5


def test_owner_task_graph_explicit_optout_forwards_full_root(authored):
    _build(authored)
    assert authored.project_calls[0]["graph_scope"] == "full_root"


@pytest.mark.parametrize("damage", ["wrong_version", "wrong_scope", "missing_root_record",
                                  "root_digest", "graph_record", "omitted_population"])
def test_owner_scoped_population_cannot_use_incomplete_or_foreign_root_metadata(authored, monkeypatch, damage):
    _scoped_authored_proposal(authored)
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", raising=False)
    proposal = authored.proposal
    if damage == "wrong_version":
        proposal["schema"] = "supervisor-source-bound-typed-context-proposal@1"
    elif damage == "wrong_scope":
        proposal["graph_scope"] = "full_root"
    elif damage == "missing_root_record":
        proposal["root_symbol_records"].pop()
        proposal["root_symbol_records_count"] = len(proposal["root_symbol_records"])
        proposal["root_symbol_records_sha256"] = _sha(_encoded(proposal["root_symbol_records"]))
    elif damage == "root_digest":
        proposal["root_symbol_records_sha256"] = "0" * 64
    elif damage == "graph_record":
        proposal["graph"]["nodes"][0]["record"]["capsule_cid"] = _cid("foreign-scoped-capsule")
    else:
        proposal["omitted_root_symbol_ids"] = []
    with pytest.raises(ValueError):
        _build(authored)
    assert authored.launch_calls == []

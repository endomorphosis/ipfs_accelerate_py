"""Projected-worker custody controls with real files and explicit authored seams.

The retained full graph is authored, not a native producer result. Source
verification, graph loading and launch building are substitutes. Owner policy
loading and bounded codec reads remain real. Native @2 representation and
producer integration are qualified by separate tests and the root-owned probe.
"""
from __future__ import annotations

from base64 import b32encode
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy as bridge
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_source_context as wrapper
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy

TASK = "TASK-AUTHORED-PROJECTED-OWNER"
ARTIFACT = ".runtime/query/worker-context.json"
GRAPH_ROOT = "decision-root:authored-projected-owner"
SOURCE = b"def helper(value):\n    return value + 1\n\ndef target(value):\n    return helper(value)\n"
INSTRUCTION = "target helper public_test café\nKeep every trailing space.  \n"
INSTRUCTION_PATH = "instruction.md"


def _cid(label):
    return "b" + b32encode(b"\x01\x55\x12\x20" + sha256(label.encode()).digest()).decode().lower().rstrip("=")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(value):
    return sha256(value).hexdigest()


def _case(tmp_path):
    repository, workspace = tmp_path / "repository", tmp_path / "candidate"
    repository.mkdir()
    workspace.mkdir()
    (repository / "operations.py").write_bytes(SOURCE)
    (repository / INSTRUCTION_PATH).write_text(INSTRUCTION)
    names = ("target", "helper", "public_test", "disconnected", "preselected", "outside_worker")
    delivered = names[:-1]
    symbols = {name: _cid("symbol:" + name) for name in names}
    capsules = {name: _cid("capsule:" + name) for name in names}
    root_cid = _cid("authored-full-root")
    scope_cid = _cid("authored-source-scope")
    source_cid = _cid("authored-module-source")
    manifest = {
        "operations.py": {"sha256": _sha(SOURCE), "source_cid": source_cid},
        INSTRUCTION_PATH: {"sha256": _sha(INSTRUCTION.encode()), "source_cid": _cid("instruction-source")},
    }
    inventory = [{"stable_symbol_id": symbols[name], "capsule_cid": capsules[name]} for name in delivered]
    payload = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": TASK,
        "semantic_root_cid": root_cid, "scope_cid": scope_cid, "manifest": manifest,
        "required_raw_paths": ["operations.py", INSTRUCTION_PATH],
        "raw_sources": {"operations.py": SOURCE.decode(), INSTRUCTION_PATH: INSTRUCTION},
        "objective": " ".join(INSTRUCTION.split()),
        "capsules": deepcopy(inventory),
        "admissions": [{"ref": {**row, "semantic_state_root_cid": root_cid,
                                  "source_cid": source_cid, "raw_source_required": True}}
                       for row in inventory],
        "pack": {"schema": "ipfs-accelerate.context-pack-result@1", "interface": "ContextPack@1",
                 "pack": {"objective": " ".join(INSTRUCTION.split()), "obligation_cids": [_cid("public-obligation")]}},
        "worker_projection": {
            "schema": "supervisor-semantic-worker-projection@1", "selector": "lexical-query-symbol-name@1",
            "query": INSTRUCTION, "max_capsules": 8, "max_bytes": 65536,
            "full_capsule_count": len(names), "selected_capsule_count": len(delivered),
            "omitted_capsule_count": 1,
            "selected_symbols": [{"path": "operations.py", "qualified_name": "operations." + name,
                                  "stable_symbol_id": symbols[name]} for name in delivered],
            "capsule_index_cid": _cid("full-capsule-index"), "raw_source_fetch_required": {},
            "omission_reason": "Explicit query-selected worker inventory; full native graph retained.",
            "raw_required_caveat": "Read exact current source before editing.",
            "semantic_equivalence_claimed": False, "completion_authority": False,
        },
        "completion_authority": False,
    }
    artifact = repository / ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(_encoded(payload))
    graph_records = [{"native_symbol_id": symbols[name], "capsule_cid": capsules[name],
                      "qualified_name": "operations." + name, "source_module_path": "operations.py",
                      "source_cid": source_cid, "version_cid": _cid("version:" + name),
                      "fact_cid": _cid("fact:" + name)} for name in names]
    proposal = {
        "schema": "authored-typed-dependency-proposal@1", "semantic_root_cid": root_cid,
        "root_cid": root_cid, "graph_root_id": GRAPH_ROOT,
        "full_symbol_ids": sorted(symbols.values()),
        "nominee_symbol_ids": sorted(symbols[name] for name in ("target", "helper", "public_test", "disconnected")),
        "graph": {"nodes": [{"record": record} for record in graph_records], "edges": []},
        "unresolved_state_binding_rows": [{"source_stable_id": symbols["target"],
                                           "unresolved_target": "global:helper", "relation": "reads_state"}],
        "unresolved_other_reference_rows": [],
        "incoming_relations_automatically_selected": False,
        "source_freshness_verified": False, "owner_task_policy_verified": False,
        "effective_task_coverage_verified": False, "dependency_semantics_complete": False,
        "source_substitution_authority": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
        "deployment_ready": False, "runtime_activation": False,
    }
    bundle = SimpleNamespace(root=SimpleNamespace(root_cid=root_cid,
        capsule_index_cid=payload["worker_projection"]["capsule_index_cid"]), authored_fixture=True)
    wire = {"objective_id": TASK, "repository_id": "authored-repository", "tree_id": "authored-tree"}
    return SimpleNamespace(repository=repository, workspace=workspace, artifact=artifact,
        source_path=repository / "operations.py", instruction_path=repository / INSTRUCTION_PATH,
        payload=payload, root_cid=root_cid, symbols=symbols, capsules=capsules, names=names,
        delivered=delivered, inventory=inventory, proposal=proposal, bundle=bundle,
        parent_text=artifact.read_text(), prompt=_encoded(wire).decode() + "\nLiteral native suffix.\n",
        verify_calls=[], core_calls=[], load_calls=[], worker_calls=[], full_calls=[], launch_calls=[],
        after_core=None, after_launch=None, owner=None, policy=None)


def _install_verifier(state, monkeypatch):
    def verified(root, *, artifact, text, task_id, current):
        state.verify_calls.append((root, artifact, task_id, current))
        assert current is True
        raw = wrapper.codec._read(root, artifact)
        if raw != text.encode():
            raise ValueError("authored current verifier: parent changed")
        value = json.loads(raw)
        if value["task_id"] != task_id:
            raise ValueError("authored current verifier: foreign task")
        for path, binding in value["manifest"].items():
            original = SOURCE if path == "operations.py" else INSTRUCTION.encode()
            current_bytes = wrapper.codec._read(root, path)
            if current_bytes != original or _sha(current_bytes) != binding["sha256"]:
                raise ValueError("authored current verifier: source stale")
        return value, tuple(sorted(row["stable_symbol_id"] for row in value["capsules"]))
    monkeypatch.setattr(wrapper.codec, "_verified_semantic", verified)
    return verified


@pytest.fixture
def projected_source(tmp_path, monkeypatch):
    # Preserve the legacy authored full-graph seam explicitly.
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", "0")
    state = _case(tmp_path)
    _install_verifier(state, monkeypatch)
    def load(repository, artifact, semantic_root_cid):
        state.load_calls.append((repository, artifact, semantic_root_cid))
        return state.bundle
    def core(**kwargs):
        state.core_calls.append(kwargs)
        assert kwargs["bundle"] is state.bundle
        if state.after_core is not None:
            state.after_core()
        return deepcopy(state.proposal)
    monkeypatch.setattr(wrapper, "_load_bundle", load)
    monkeypatch.setattr(wrapper, "nominate_typed_semantic_dependencies", core)
    return state


def _project(state, **changes):
    arguments = dict(repository=state.repository, artifact=ARTIFACT,
        expected_sha256=_sha(state.artifact.read_bytes()), task_id=TASK,
        expected_semantic_root_cid=state.root_cid, graph_root_id=GRAPH_ROOT,
        seed_symbol_ids=(state.symbols["target"],))
    arguments.update(changes)
    return wrapper.project_worker_source_bound_typed_links(**arguments)


def _rewrite_parent(state):
    state.artifact.write_bytes(_encoded(state.payload))
    state.parent_text = state.artifact.read_text()


def test_projected_source_preserves_full_graph_and_exact_delivered_inventory(projected_source):
    state = projected_source
    before = state.artifact.read_bytes()
    result = _project(state)
    assert result["full_symbol_ids"] == sorted(state.symbols.values())
    assert result["worker_inventory"] == state.inventory
    assert result["nominee_symbol_ids"] == state.proposal["nominee_symbol_ids"]
    assert len(result["graph"]["nodes"]) == 6 and len(result["worker_inventory"]) == 5
    assert result["source_freshness_verified"] is True
    assert result["owner_task_policy_verified"] is False
    assert result["source_binding"]["artifact_sha256"] == _sha(before)
    assert result["source_binding"]["source_manifest"] == state.payload["manifest"]
    assert len(state.verify_calls) == 2 and all(call[-1] is True for call in state.verify_calls)
    assert len(state.core_calls) == len(state.load_calls) == 1
    assert result["worker_projection_binding"]["full_symbol_count"] == 6
    assert result["worker_projection_binding"]["worker_symbol_count"] == 5
    assert result["worker_projection_binding"]["omitted_symbol_count"] == 1
    assert result["worker_projection_binding"]["worker_inventory_complete"] is False
    assert state.artifact.read_bytes() == before
    assert state.source_path.read_bytes() == SOURCE
    assert result["unresolved_state_binding_rows"] == state.proposal["unresolved_state_binding_rows"]


def _require_instruction_only(state):
    """Keep module bytes outside delivery while retaining their exact frontier."""
    state.payload["required_raw_paths"] = [INSTRUCTION_PATH]
    state.payload["raw_sources"] = {INSTRUCTION_PATH: INSTRUCTION}
    state.payload["worker_projection"]["raw_source_fetch_required"] = {
        "operations.py": deepcopy(state.payload["manifest"]["operations.py"])}


def test_optional_raw_fetch_frontier_preserves_full_source_graph_and_required_instruction(projected_source):
    state = projected_source
    _require_instruction_only(state)
    _rewrite_parent(state)
    artifact_before = state.artifact.read_bytes()
    result = _project(state)
    assert result["full_symbol_ids"] == sorted(state.symbols.values())
    assert result["worker_inventory"] == state.inventory
    binding = result["source_binding"]
    assert binding["source_manifest"] == state.payload["manifest"]
    assert binding["required_raw_paths"] == [INSTRUCTION_PATH]
    assert binding["raw_source_paths"] == [INSTRUCTION_PATH]
    assert state.payload["raw_sources"] == {INSTRUCTION_PATH: INSTRUCTION}
    assert state.payload["worker_projection"]["raw_source_fetch_required"] == {
        "operations.py": state.payload["manifest"]["operations.py"]}
    assert state.source_path.read_bytes() == SOURCE
    assert state.instruction_path.read_text() == INSTRUCTION
    assert state.artifact.read_bytes() == artifact_before
    assert len(state.verify_calls) == 2 and all(call[-1] is True for call in state.verify_calls)
    assert result["source_substitution_authority"] is False


@pytest.mark.parametrize("damage", [
    "missing_fetch", "foreign_fetch", "stale_fetch_cid", "stale_fetch_sha",
    "bool_full_count", "selected_count", "omitted_count", "max_capsules",
    "actual_byte_overflow", "extra_field", "authority",
])
def test_projected_fetch_frontier_counts_and_complete_payload_budget_are_exact(projected_source, damage):
    state = projected_source
    _require_instruction_only(state)
    projection = state.payload["worker_projection"]
    if damage == "missing_fetch":
        projection["raw_source_fetch_required"].clear()
    elif damage == "foreign_fetch":
        projection["raw_source_fetch_required"]["foreign.py"] = deepcopy(
            state.payload["manifest"]["operations.py"])
    elif damage == "stale_fetch_cid":
        projection["raw_source_fetch_required"]["operations.py"]["source_cid"] = _cid("stale-fetch-source")
    elif damage == "stale_fetch_sha":
        projection["raw_source_fetch_required"]["operations.py"]["sha256"] = "0" * 64
    elif damage == "bool_full_count":
        projection["full_capsule_count"] = True
    elif damage == "selected_count":
        projection["selected_capsule_count"] -= 1
    elif damage == "omitted_count":
        projection["omitted_capsule_count"] += 1
    elif damage == "max_capsules":
        projection["max_capsules"] = len(state.inventory) - 1
    elif damage == "actual_byte_overflow":
        # The ceiling itself is valid. The complete authored parent exceeds it;
        # accepting it would mean measuring a subset rather than delivered bytes.
        projection["max_bytes"] = 8192
        state.payload["objective"] = "authored literal pack body " * 500
        state.payload["pack"]["pack"]["objective"] = state.payload["objective"]
    elif damage == "extra_field":
        projection["infer_optional_from_graph"] = True
    else:
        projection["completion_authority"] = True
    _rewrite_parent(state)
    if damage == "actual_byte_overflow":
        assert len(state.artifact.read_bytes()) > 8192
        assert len(projection["query"].encode()) <= 8192
    with pytest.raises(ValueError):
        _project(state)


@pytest.mark.parametrize("damage", ["foreign_path", "foreign_source", "foreign_capsule",
                                    "duplicate_symbol", "duplicate_capsule", "missing_symbol"])
def test_projected_inventory_cannot_borrow_foreign_graph_bindings(projected_source, damage):
    state = projected_source
    records = state.proposal["graph"]["nodes"]
    if damage == "foreign_path":
        records[0]["record"]["source_module_path"] = "foreign.py"
    elif damage == "foreign_source":
        records[0]["record"]["source_cid"] = _cid("foreign-source")
    elif damage == "foreign_capsule":
        records[0]["record"]["capsule_cid"] = _cid("foreign-capsule")
    elif damage == "duplicate_symbol":
        records.append(deepcopy(records[0]))
    elif damage == "duplicate_capsule":
        records[1]["record"]["capsule_cid"] = records[0]["record"]["capsule_cid"]
    else:
        records.pop()
    with pytest.raises(ValueError):
        _project(state)


@pytest.mark.parametrize("damage", ["counts", "index", "selected_symbol", "qualified_name", "path"])
def test_query_projection_metadata_matches_retained_graph_and_actual_inventory(projected_source, damage):
    state = projected_source
    projection = state.payload["worker_projection"]
    if damage == "counts":
        projection["full_capsule_count"] -= 1
    elif damage == "index":
        projection["capsule_index_cid"] = _cid("foreign-index")
    elif damage == "selected_symbol":
        projection["selected_symbols"][0]["stable_symbol_id"] = state.symbols["outside_worker"]
    elif damage == "qualified_name":
        projection["selected_symbols"][0]["qualified_name"] = "foreign.target"
    else:
        projection["selected_symbols"][0]["path"] = "foreign.py"
    _rewrite_parent(state)
    with pytest.raises(ValueError):
        _project(state)


@pytest.mark.parametrize("damage", ["none", "list", "string", "role_view", "decision_packet"])
def test_worker_projection_requires_actual_native_query_parent(projected_source, damage):
    state = projected_source
    if damage in {"role_view", "decision_packet"}:
        state.payload["schema"] = "supervisor-semantic-worker-role-view@1" if damage == "role_view" else "supervisor-semantic-decision-packet@2"
    else:
        state.payload["worker_projection"] = {"none": None, "list": [], "string": "projection"}[damage]
    _rewrite_parent(state)
    with pytest.raises(ValueError):
        _project(state)
    if damage in {"role_view", "decision_packet"}:
        assert state.core_calls == []


@pytest.mark.parametrize("target", ["source", "parent"])
def test_projected_source_closing_fence_refuses_concurrent_changes(projected_source, target):
    state = projected_source
    path = state.source_path if target == "source" else state.artifact
    before = path.read_bytes()
    state.after_core = lambda: path.write_bytes(before + b"\n# external change\n")
    with pytest.raises(ValueError):
        _project(state)
    assert len(state.core_calls) == 1 and path.read_bytes() != before


@pytest.fixture
def projected_owner(tmp_path, monkeypatch):
    # Preserve the legacy authored full-graph seam explicitly.
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", "0")
    state = _case(tmp_path)
    verified = _install_verifier(state, monkeypatch)
    state.policy = SemanticDecisionPolicy(
        required_capsule_ids=(state.capsules["disconnected"], state.capsules["target"]),
        optional_capsule_ids=(state.capsules["public_test"], state.capsules["preselected"], state.capsules["helper"]),
        selected_optional_capsule_ids=(state.capsules["preselected"],))
    state.owner = tmp_path / "owner-policy.json"
    def save_policy():
        state.owner.chmod(0o600) if state.owner.exists() else None
        state.owner.write_bytes(_encoded(bridge.build_decision_nomination(state.policy)))
        state.owner.chmod(0o444)
    state.save_policy = save_policy
    save_policy()
    def parts(prompt):
        wire, end = json.JSONDecoder().raw_decode(prompt)
        return wire, prompt[end:], (), (), state.parent_text, ARTIFACT
    def full(**kwargs):
        state.full_calls.append(kwargs)
        raise AssertionError("projected worker must not use full-parent wrapper")
    def worker(**kwargs):
        state.worker_calls.append(kwargs)
        verified(kwargs["repository"], artifact=kwargs["artifact"], text=state.parent_text,
                 task_id=kwargs["task_id"], current=True)
        if type(state.payload.get("worker_projection")) is not dict:
            raise ValueError("authored worker verifier: invalid projection")
        proposal = deepcopy(state.proposal)
        proposal.update(source_freshness_verified=True, worker_inventory=deepcopy(state.inventory),
            source_binding={"task_id": TASK, "artifact": ARTIFACT,
                            "artifact_sha256": _sha(state.parent_text.encode()),
                            "semantic_root_cid": state.root_cid, "scope_cid": state.payload["scope_cid"],
                            "source_manifest": deepcopy(state.payload["manifest"]),
                            "required_raw_paths": list(state.payload["required_raw_paths"]),
                            "raw_source_paths": sorted(state.payload["raw_sources"])})
        if state.after_core is not None:
            state.after_core()
        return proposal
    def launch(*, native_prompt, policy, repository):
        state.launch_calls.append((native_prompt, policy, repository))
        if state.after_launch is not None:
            state.after_launch()
        return {"schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
                "task_id": TASK, "parent_artifact": ARTIFACT,
                "parent_sha256": _sha(state.parent_text.encode()), "policy": policy.to_dict()}
    monkeypatch.setattr(bridge.codec, "_prompt_parts", parts)
    monkeypatch.setattr(bridge, "project_source_bound_typed_links", full)
    monkeypatch.setattr(bridge, "project_worker_source_bound_typed_links", worker)
    monkeypatch.setattr(bridge, "build_decision_launch_policy", launch)
    return state


def _build(state, **changes):
    arguments = dict(native_prompt=state.prompt, repository=state.repository, workspace=state.workspace,
        nomination_artifact=state.owner, nomination_sha256=_sha(state.owner.read_bytes()), task_id=TASK,
        expected_parent_sha256=_sha(state.parent_text.encode()), expected_semantic_root_cid=state.root_cid,
        graph_root_id=GRAPH_ROOT)
    arguments.update(changes)
    return bridge.build_typed_owner_decision_nomination(**arguments)


@pytest.mark.parametrize("revision", ["1", "2"])
def test_projected_owner_keeps_actual_partition_and_adds_known_optional_nominees(projected_owner, revision):
    state = projected_owner
    if revision == "2":
        state.policy = SemanticDecisionPolicy(
            required_capsule_ids=state.policy.required_capsule_ids,
            optional_capsule_ids=state.policy.optional_capsule_ids,
            selected_optional_capsule_ids=state.policy.selected_optional_capsule_ids,
            task_text_source_path=INSTRUCTION_PATH)
        state.save_policy()
    before = (state.owner.read_bytes(), state.artifact.read_bytes(), state.source_path.read_bytes())
    result = _build(state)
    assert result.policy.required_capsule_ids == state.policy.required_capsule_ids
    assert result.policy.optional_capsule_ids == state.policy.optional_capsule_ids
    assert result.policy.selected_optional_capsule_ids == (
        state.capsules["preselected"], state.capsules["public_test"], state.capsules["helper"])
    assert result.policy.task_text_source_path == state.policy.task_text_source_path
    assert result.nomination["schema"] == "supervisor-semantic-decision-nomination@" + revision
    assert result.launch_policy["schema"] == "supervisor-semantic-decision-launch-policy@" + revision
    assert state.worker_calls[0]["seed_symbol_ids"] == (state.symbols["disconnected"], state.symbols["target"])
    assert state.full_calls == [] and len(state.launch_calls) == 1
    assert all(call[-1] is True for call in state.verify_calls) and len(state.verify_calls) >= 2
    receipt = result.receipt
    assert receipt["projection_aware"] is True
    assert receipt["full_capsule_count"] == 6
    assert len([node for node in receipt["source_proposal"]["graph"]["nodes"]
                if "native_symbol_id" in node["record"]]) == 6
    assert receipt["delivered_capsule_count"] == 5
    assert receipt["source_proposal"]["unresolved_state_binding_rows"] == state.proposal["unresolved_state_binding_rows"]
    assert before == (state.owner.read_bytes(), state.artifact.read_bytes(), state.source_path.read_bytes())
    assert state.payload["worker_projection"]["query"] == state.payload["raw_sources"][INSTRUCTION_PATH]
    assert state.payload["objective"] == state.payload["pack"]["pack"]["objective"]
    assert state.payload["objective"] != INSTRUCTION
    assert receipt["effective_task_coverage_verified"] is False


def test_known_dependency_missing_from_worker_refuses_even_with_complete_raw_sources(projected_owner):
    state = projected_owner
    helper_cid = state.capsules["helper"]
    state.payload["capsules"] = [row for row in state.payload["capsules"] if row["capsule_cid"] != helper_cid]
    state.inventory = deepcopy(state.payload["capsules"])
    state.payload["admissions"] = [row for row in state.payload["admissions"] if row["ref"]["capsule_cid"] != helper_cid]
    projection = state.payload["worker_projection"]
    projection["selected_symbols"] = [row for row in projection["selected_symbols"]
                                      if row["stable_symbol_id"] != state.symbols["helper"]]
    projection["selected_capsule_count"] = 4
    projection["omitted_capsule_count"] = 2
    state.policy = SemanticDecisionPolicy(required_capsule_ids=state.policy.required_capsule_ids,
        optional_capsule_ids=tuple(cid for cid in state.policy.optional_capsule_ids if cid != helper_cid),
        selected_optional_capsule_ids=state.policy.selected_optional_capsule_ids)
    _rewrite_parent(state)
    state.save_policy()
    with pytest.raises(ValueError):
        _build(state)
    assert state.launch_calls == [] and state.worker_calls
    assert state.payload["raw_sources"]["operations.py"] == SOURCE.decode()
    assert state.symbols["helper"] in state.proposal["nominee_symbol_ids"]


def test_owner_partition_for_full_graph_cannot_relabel_missing_worker_evidence(projected_owner):
    state = projected_owner
    state.policy = SemanticDecisionPolicy(required_capsule_ids=state.policy.required_capsule_ids,
        optional_capsule_ids=state.policy.optional_capsule_ids + (state.capsules["outside_worker"],),
        selected_optional_capsule_ids=state.policy.selected_optional_capsule_ids)
    state.save_policy()
    with pytest.raises(ValueError):
        _build(state)
    assert state.worker_calls == state.launch_calls == []


@pytest.mark.parametrize("damage", ["duplicate_full_id", "foreign_full_id", "duplicate_graph_symbol",
    "duplicate_graph_capsule", "foreign_actual_capsule", "missing_graph_symbol", "worker_inventory"])
def test_full_graph_and_actual_worker_identity_join_is_exact(projected_owner, monkeypatch, damage):
    state = projected_owner
    if damage == "duplicate_full_id":
        state.proposal["full_symbol_ids"].append(state.proposal["full_symbol_ids"][0])
    elif damage == "foreign_full_id":
        state.proposal["full_symbol_ids"][0] = _cid("foreign-full-symbol")
    elif damage == "duplicate_graph_symbol":
        state.proposal["graph"]["nodes"].append(deepcopy(state.proposal["graph"]["nodes"][0]))
    elif damage == "duplicate_graph_capsule":
        state.proposal["graph"]["nodes"][1]["record"]["capsule_cid"] = state.proposal["graph"]["nodes"][0]["record"]["capsule_cid"]
    elif damage == "foreign_actual_capsule":
        state.proposal["graph"]["nodes"][0]["record"]["capsule_cid"] = _cid("foreign-current-capsule")
    elif damage == "missing_graph_symbol":
        state.proposal["graph"]["nodes"].pop()
    else:
        original = bridge.project_worker_source_bound_typed_links
        def different(**kwargs):
            result = original(**kwargs)
            result["worker_inventory"].reverse()
            return result
        monkeypatch.setattr(bridge, "project_worker_source_bound_typed_links", different)
    with pytest.raises(ValueError):
        _build(state)
    assert state.launch_calls == []


@pytest.mark.parametrize("target", ["source", "parent", "owner"])
def test_projected_owner_rechecks_current_inputs_after_descriptor_construction(projected_owner, target):
    state = projected_owner
    path = {"source": state.source_path, "parent": state.artifact, "owner": state.owner}[target]
    before = path.read_bytes()
    def change():
        path.chmod(0o600)
        path.write_bytes(before + b"\n# external concurrent change\n")
    state.after_launch = change
    with pytest.raises(ValueError):
        _build(state)
    assert len(state.launch_calls) == 1 and path.read_bytes() != before


def test_projected_results_remain_detached_and_do_not_claim_task_authority(projected_owner):
    state = projected_owner
    result = _build(state)
    expected = result.to_dict()
    nomination = result.nomination
    nomination["policy"]["required_capsule_ids"].clear()
    receipt = result.receipt
    receipt["source_proposal"]["worker_inventory"].clear()
    assert result.to_dict() == expected
    for flag in ("owner_task_coverage_verified", "effective_task_coverage_verified",
                 "source_substitution_authority", "proof_authority", "execution_authority", "completion_authority"):
        assert result.receipt[flag] is False


# These scoped fixtures are authored wrapper/owner seams. The native adapter's
# complete root verification and actual unknown-node joins are independently
# qualified in test_semantic_typed_dependency_proposal.py.
def _authored_scoped_proposal(state, *, worker_wrapper):
    proposal = deepcopy(state.proposal)
    records = sorted((deepcopy(node["record"]) for node in proposal["graph"]["nodes"]
                      if "native_symbol_id" in node["record"]),
                     key=lambda row: row["native_symbol_id"])
    materialized = list(proposal["nominee_symbol_ids"])
    selected = set(materialized)
    proposal.update({
        "schema": (wrapper.SCOPED_WORKER_SCHEMA if worker_wrapper
                   else "supervisor-semantic-typed-dependency-proposal@2"),
        "adapter_version": "native-typed-dependency-proposal@2",
        "graph_scope": "task_closure",
        "root_symbol_records": records,
        "root_symbol_records_count": len(records),
        "root_symbol_records_sha256": _sha(_encoded(records)),
        "materialized_symbol_ids": materialized,
        "omitted_root_symbol_ids": sorted(set(proposal["full_symbol_ids"]) - selected),
    })
    proposal["graph"]["nodes"] = [node for node in proposal["graph"]["nodes"]
                                      if node["record"].get("native_symbol_id") in selected]
    # Retain a body-free authored unknown observation/obligation to verify
    # passthrough custody. Native joining/completeness is not inferred here.
    observation_id, obligation_id = _cid("authored-unknown-observation"), _cid("authored-unknown-obligation")
    proposal["graph"]["nodes"].extend([
        {"node_id": observation_id, "record": {
            "source_stable_id": state.symbols["target"], "relation": "reads_state",
            "target_kind": "unresolved", "unresolved_target": "global:helper"}},
        {"node_id": obligation_id, "record": {
            "obligation_kind": "unresolved_state_binding", "recorded_target": "global:helper",
            "binding_value_verified": False}},
    ])
    proposal["unresolved_state_binding_rows"][0]["obligation_node_id"] = obligation_id
    return proposal


@pytest.fixture
def scoped_projected_source(projected_source):
    state = projected_source
    state.proposal = _authored_scoped_proposal(state, worker_wrapper=False)
    return state


@pytest.fixture
def scoped_projected_owner(projected_owner, monkeypatch):
    state = projected_owner
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", raising=False)
    state.proposal = _authored_scoped_proposal(state, worker_wrapper=True)
    return state


def test_scoped_source_keeps_delivered_optional_outside_graph_closure(scoped_projected_source):
    state = scoped_projected_source
    before = (state.artifact.read_bytes(), state.source_path.read_bytes())
    result = _project(state, graph_scope="task_closure")
    extra = state.symbols["preselected"]
    assert extra in [row["stable_symbol_id"] for row in result["worker_inventory"]]
    assert extra in result["omitted_root_symbol_ids"] and extra not in result["materialized_symbol_ids"]
    assert extra in [row["native_symbol_id"] for row in result["root_symbol_records"]]
    assert result["root_symbol_records_count"] == 6
    assert result["schema"] == wrapper.SCOPED_WORKER_SCHEMA
    assert result["source_freshness_verified"] is True
    assert state.core_calls[0]["graph_scope"] == "task_closure"
    assert {node["record"]["native_symbol_id"] for node in result["graph"]["nodes"]
            if "native_symbol_id" in node["record"]} == set(result["nominee_symbol_ids"])
    assert result["worker_projection_binding"]["full_symbol_count"] == 6
    assert result["worker_projection_binding"]["worker_symbol_count"] == 5
    assert before == (state.artifact.read_bytes(), state.source_path.read_bytes())


def test_scoped_owner_keeps_delivered_optional_outside_graph_closure(scoped_projected_owner):
    state = scoped_projected_owner
    result = _build(state)
    proposal = result.receipt["source_proposal"]
    assert proposal["schema"] == wrapper.SCOPED_WORKER_SCHEMA
    assert state.worker_calls[0]["graph_scope"] == "task_closure"
    assert state.symbols["preselected"] in proposal["omitted_root_symbol_ids"]
    assert state.capsules["preselected"] in result.policy.selected_optional_capsule_ids
    assert result.policy.required_capsule_ids == state.policy.required_capsule_ids
    assert result.policy.optional_capsule_ids == state.policy.optional_capsule_ids
    assert result.policy.selected_optional_capsule_ids == (
        state.capsules["preselected"], state.capsules["public_test"], state.capsules["helper"])
    assert result.receipt["full_capsule_count"] == 6 and result.receipt["delivered_capsule_count"] == 5
    assert proposal["worker_inventory"] == state.inventory
    assert len(state.launch_calls) == 1
    assert result.receipt["effective_task_coverage_verified"] is False


def test_scoped_owner_missing_known_nominee_refuses_before_launch(scoped_projected_owner):
    state = scoped_projected_owner
    helper_cid = state.capsules["helper"]
    state.payload["capsules"] = [row for row in state.payload["capsules"] if row["capsule_cid"] != helper_cid]
    state.inventory = deepcopy(state.payload["capsules"])
    state.payload["admissions"] = [row for row in state.payload["admissions"] if row["ref"]["capsule_cid"] != helper_cid]
    projection = state.payload["worker_projection"]
    projection["selected_symbols"] = [row for row in projection["selected_symbols"]
                                      if row["stable_symbol_id"] != state.symbols["helper"]]
    projection["selected_capsule_count"] = 4
    projection["omitted_capsule_count"] = 2
    state.policy = SemanticDecisionPolicy(required_capsule_ids=state.policy.required_capsule_ids,
        optional_capsule_ids=tuple(cid for cid in state.policy.optional_capsule_ids if cid != helper_cid),
        selected_optional_capsule_ids=state.policy.selected_optional_capsule_ids)
    _rewrite_parent(state)
    state.save_policy()
    with pytest.raises(ValueError, match="missing known typed dependencies"):
        _build(state)
    assert state.launch_calls == [] and state.worker_calls
    assert state.symbols["helper"] in state.proposal["nominee_symbol_ids"]
    assert state.payload["raw_sources"]["operations.py"] == SOURCE.decode()


@pytest.mark.parametrize("damage", [
    "missing_records", "bool_count", "bad_digest", "duplicate_record_id", "missing_record",
    "missing_graph_symbol", "changed_graph_symbol", "wrong_omitted_ids", "wrong_materialized_ids",
    "wrong_schema", "wrong_adapter",
])
def test_scoped_source_complete_metadata_and_materialized_symbol_join_is_exact(scoped_projected_source, damage):
    state = scoped_projected_source
    proposal = state.proposal
    if damage == "missing_records":
        del proposal["root_symbol_records"]
    elif damage == "bool_count":
        proposal["root_symbol_records_count"] = True
    elif damage == "bad_digest":
        proposal["root_symbol_records_sha256"] = "0" * 64
    elif damage in {"duplicate_record_id", "missing_record"}:
        records = proposal["root_symbol_records"]
        if damage == "duplicate_record_id":
            records[1] = deepcopy(records[0])
        else:
            records.pop()
        proposal["root_symbol_records_count"] = len(records)
        proposal["root_symbol_records_sha256"] = _sha(_encoded(records))
    elif damage == "missing_graph_symbol":
        proposal["graph"]["nodes"].pop(0)
    elif damage == "changed_graph_symbol":
        proposal["graph"]["nodes"][0]["record"]["fact_cid"] = _cid("rebound-scoped-fact")
    elif damage == "wrong_omitted_ids":
        proposal["omitted_root_symbol_ids"].clear()
    elif damage == "wrong_materialized_ids":
        proposal["materialized_symbol_ids"].pop()
    elif damage == "wrong_schema":
        proposal["schema"] = "authored-typed-dependency-proposal@1"
    else:
        proposal["adapter_version"] = "native-typed-dependency-proposal@1"
    with pytest.raises(ValueError):
        _project(state, graph_scope="task_closure")


@pytest.mark.parametrize("damage", ["foreign_path", "foreign_source"])
def test_scoped_source_checks_full_root_frontier_of_delivered_optional_outside_closure(scoped_projected_source, damage):
    state = scoped_projected_source
    record = next(row for row in state.proposal["root_symbol_records"]
                  if row["native_symbol_id"] == state.symbols["preselected"])
    if damage == "foreign_path":
        record["source_module_path"] = "foreign.py"
    else:
        record["source_cid"] = _cid("foreign-root-source")
    state.proposal["root_symbol_records_sha256"] = _sha(_encoded(state.proposal["root_symbol_records"]))
    assert state.symbols["preselected"] not in state.proposal["materialized_symbol_ids"]
    with pytest.raises(ValueError, match="symbol escapes current source frontier"):
        _project(state, graph_scope="task_closure")


@pytest.mark.parametrize("setting,scope", [(None, "task_closure"), ("1", "task_closure"), ("0", "full_root")])
def test_owner_scope_default_and_explicit_optout_are_forwarded(projected_owner, monkeypatch, setting, scope):
    state = projected_owner
    if setting is None:
        monkeypatch.delenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", raising=False)
    else:
        monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", setting)
    if scope == "task_closure":
        state.proposal = _authored_scoped_proposal(state, worker_wrapper=True)
    result = _build(state)
    assert state.worker_calls[0]["graph_scope"] == scope
    if scope == "task_closure":
        assert result.receipt["source_proposal"]["schema"] == wrapper.SCOPED_WORKER_SCHEMA
    else:
        assert len([node for node in result.receipt["source_proposal"]["graph"]["nodes"]
                    if "native_symbol_id" in node["record"]]) == 6
    assert len(state.launch_calls) == 1


def test_scoped_unknown_rows_remain_body_free_advisory_passthrough(scoped_projected_source):
    state = scoped_projected_source
    result = _project(state, graph_scope="task_closure")
    assert result["unresolved_state_binding_rows"] == state.proposal["unresolved_state_binding_rows"]
    row = result["unresolved_state_binding_rows"][0]
    assert row["unresolved_target"] == "global:helper"
    assert row["obligation_node_id"] in {node.get("node_id") for node in result["graph"]["nodes"]}
    graph_text = _encoded(result["graph"]).decode()
    assert "normalized_ast" not in graph_text and SOURCE.decode() not in graph_text
    assert result["dependency_semantics_complete"] is False
    assert result["source_substitution_authority"] is False
    assert result["execution_authority"] is False
    assert result["proof_authority"] is False

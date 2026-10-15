"""Retained graph-scope replay with actual immutable owner-material custody.

Native source projection is an explicitly authored seam. These controls prove
version selection, opt-out independence, closed commitments, and exact replay
comparison. Native qualification separately authenticates producer evidence.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner as owner
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy as typed_policy
import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_nomination_lineage as lineage_api
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_owner_policy import TypedOwnerNomination

TASK = "TASK-AUTHORED-TYPED-HANDOFF"
ARTIFACT = ".runtime/semantic/worker-context.json"
SOURCE = b"def helper(value):\n    return value + 1\n\ndef target(value):\n    return helper(value)\n"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(raw):
    return sha256(raw).hexdigest()




@pytest.fixture
def handoff(tmp_path, monkeypatch):
    # This retained historical seam deliberately uses the authentic legacy
    # source proposal contract. Scope replay must stay exact even after the
    # owner default changes to task_closure.
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", "0")
    repository, workspace = tmp_path / "source", tmp_path / "candidate"
    repository.mkdir(mode=0o755)
    workspace.mkdir(mode=0o755)
    source_path = repository / "operations.py"
    source_path.write_bytes(SOURCE)
    parent_path = repository / ARTIFACT
    parent_path.parent.mkdir(parents=True, mode=0o755)
    parent_path.parent.chmod(0o755)
    parent_path.parent.parent.chmod(0o755)
    parent = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": TASK,
        "objective": "Preserve the public target and helper behavior.",
        "semantic_root_cid": "authored-semantic-root", "scope_cid": "authored-source-scope",
        "manifest": {"operations.py": {"sha256": _sha(SOURCE), "source_cid": "authored-source-cid"}},
        "capsules": [{"stable_symbol_id": "symbol-target", "capsule_cid": "capsule-target"},
                     {"stable_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}],
        "admissions": [
            {"ref": {"stable_symbol_id": "symbol-target", "capsule_cid": "capsule-target"}},
            {"ref": {"stable_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}},
        ],
        "required_raw_paths": ["operations.py"], "raw_sources": {"operations.py": SOURCE.decode()},
        "preparation_bounds": {"max_capsules": 2, "max_bytes": 8192},
        "pack": {"schema": "ipfs-accelerate.context-pack-result@1", "interface": "ContextPack@1",
                 "pack": {}},
        "doctor_snapshot_id": "authored-doctor-snapshot",
        "doctor_manifest_cid": "authored-doctor-manifest",
        "reconstruction": {"semantic_equivalence_claimed": False, "completion_authority": False},
        "refresh_lineage": None,
        "completion_authority": False,
    }
    parent_path.write_bytes(_encoded(parent))
    policy = SemanticDecisionPolicy(required_capsule_ids=("capsule-target",),
                                    optional_capsule_ids=("capsule-helper",))
    selected = SemanticDecisionPolicy(required_capsule_ids=policy.required_capsule_ids,
        optional_capsule_ids=policy.optional_capsule_ids, selected_optional_capsule_ids=("capsule-helper",))
    nomination_path = tmp_path / "owner declaration café.json"
    # Whitespace and UTF-8 spelling must be retained exactly in lineage, while
    # the standard generated nomination uses the publisher's canonical bytes.
    origin_text = json.dumps(owner.build_decision_nomination(policy), ensure_ascii=False, indent=2) + "\n"
    nomination_path.write_text(origin_text)
    nomination_path.chmod(0o444)
    wire = {"objective_id": TASK, "repository_id": "authored-repository", "tree_id": "authored-tree"}
    prompt = _encoded(wire) + "\nPreserve literal public requirement café.  \n".encode()
    state = SimpleNamespace(repository=repository, workspace=workspace, source_path=source_path,
        parent_path=parent_path, parent=parent, parent_text=parent_path.read_text(),
        nomination_path=nomination_path, origin_text=origin_text, policy=policy, selected=selected,
        prompt=prompt, typed_calls=[], launch_calls=[], verify_calls=[], diagnostic_bytes=0,
        typed_error=None, after_typed=None, replay_calls=[], latest_source_proposal=None)

    def parts(text):
        parsed, end = json.JSONDecoder().raw_decode(text)
        return parsed, text[end:], (), (), state.parent_text, ARTIFACT

    def verified(root, *, artifact, text, task_id, current):
        state.verify_calls.append((root, artifact, task_id, current))
        assert current is True
        raw = owner.codec._read(root, artifact)
        if raw != text.encode() or task_id != TASK:
            raise ValueError("authored current verifier: parent/task changed")
        if owner.codec._read(root, "operations.py") != SOURCE:
            raise ValueError("authored current verifier: source changed")
        return json.loads(raw), ("symbol-helper", "symbol-target")

    def launch(*, native_prompt, policy, repository):
        state.launch_calls.append((native_prompt, policy, repository))
        return {"schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
                "task_id": TASK, "parent_artifact": ARTIFACT,
                "parent_sha256": _sha(state.parent_text.encode()), "policy": policy.to_dict()}

    def typed(**kwargs):
        state.typed_calls.append(kwargs)
        if state.typed_error is not None:
            raise state.typed_error
        original = owner.load_decision_nomination(
            artifact=kwargs["nomination_artifact"], expected_sha256=kwargs["nomination_sha256"],
            workspace=state.workspace)
        assert original == state.policy
        descriptor = launch(native_prompt=kwargs["native_prompt"], policy=state.selected,
                            repository=state.repository)
        proposal = {
            "schema": "supervisor-source-bound-typed-context-proposal@1",
            "adapter_version": "native-typed-dependency-proposal@1",
            "semantic_root_cid": state.parent["semantic_root_cid"],
            "graph_root_id": kwargs["graph_root_id"],
            "full_symbol_ids": ["symbol-helper", "symbol-target"],
            "nominee_symbol_ids": ["symbol-helper", "symbol-target"],
            "graph": {"nodes": [
                {"record": {"native_symbol_id": "symbol-target", "capsule_cid": "capsule-target"}},
                {"record": {"native_symbol_id": "symbol-helper", "capsule_cid": "capsule-helper"}}],
                "edges": []},
            "unresolved_state_binding_rows": [{"unresolved_target": "global:helper"}],
            "source_freshness_verified": True, "owner_task_policy_verified": False,
            "source_check_scope": "opening_and_closing_current_native_verification",
            "effective_task_coverage_verified": False, "source_substitution_authority": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False,
            "deployment_ready": False, "runtime_activation": False,
        }
        if state.diagnostic_bytes:
            proposal["authored_diagnostic"] = "x" * state.diagnostic_bytes
        state.latest_source_proposal = deepcopy(proposal)
        receipt = {
            "schema": "supervisor-semantic-typed-owner-policy-receipt@1",
            "task_id": TASK, "nomination_artifact": str(state.nomination_path),
            "nomination_sha256": _sha(state.origin_text.encode()),
            "original_policy_sha256": _sha(_encoded(state.policy.to_dict())),
            "selected_policy_sha256": _sha(_encoded(state.selected.to_dict())),
            "native_prompt_sha256": _sha(state.prompt), "parent_artifact": ARTIFACT,
            "parent_sha256": _sha(state.parent_text.encode()),
            "semantic_root_cid": state.parent["semantic_root_cid"],
            "graph_root_id": kwargs["graph_root_id"], "source_proposal": proposal,
            "added_selected_optional_capsule_ids": ["capsule-helper"],
            "known_nominee_capsule_ids": ["capsule-helper", "capsule-target"],
            "owner_required_partition_preserved": True, "owner_optional_partition_preserved": True,
            "prior_selected_optional_preserved": True, "source_freshness_verified": True,
            "owner_declaration_rechecked": True, "owner_task_coverage_verified": False,
            "effective_task_coverage_verified": False, "source_substitution_authority": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False,
            "deployment_ready": False, "runtime_activation": False,
            "provider_called": False, "descriptor_published": False,
            "projection_aware": False, "full_capsule_count": 2, "delivered_capsule_count": 2,
            "source_check_scope": "current_source_wrapper_and_post_builder_native_verification",
            "selection_basis": "recorded_typed_nominees_within_independent_owner_partition",
        }
        value = {"schema": "supervisor-semantic-typed-owner-nomination@1",
                 "nomination": owner.build_decision_nomination(state.selected),
                 "launch_policy": descriptor, "policy": state.selected.to_dict(), "receipt": receipt}
        if state.after_typed is not None:
            state.after_typed()
        return TypedOwnerNomination(_encoded(value).decode())

    monkeypatch.setattr(owner.codec, "_prompt_parts", parts)
    monkeypatch.setattr(owner.codec, "_verified_semantic", verified)
    monkeypatch.setattr(owner, "build_decision_launch_policy", launch)
    monkeypatch.setattr(typed_policy, "build_typed_owner_decision_nomination", typed)
    # Allow either root's lazy module import or a patchable forwarding alias.
    monkeypatch.setattr(owner, "build_typed_owner_decision_nomination", typed, raising=False)
    def current_replay(**kwargs):
        state.replay_calls.append(("current", kwargs))
        verified(kwargs["repository"], artifact=kwargs["artifact"], text=state.parent_text,
                 task_id=kwargs["task_id"], current=True)
        return deepcopy(state.latest_source_proposal)
    def historical_replay(**kwargs):
        state.replay_calls.append(("historical", kwargs))
        value = deepcopy(state.latest_source_proposal)
        value["source_freshness_verified"] = False
        value["source_check_scope"] = "authored_retained_native_replay"
        return value
    monkeypatch.setattr(lineage_api, "project_source_bound_typed_links", current_replay)
    monkeypatch.setattr(lineage_api, "project_worker_source_bound_typed_links", current_replay)
    monkeypatch.setattr(lineage_api, "replay_source_bound_typed_links_for_audit", historical_replay)
    return state

def _arguments(state, mode=None):
    values = ["--model", "fixture-model", "--purpose", "coding", "--semantic-repository", str(state.repository),
        "--semantic-decision-nomination-artifact", str(state.nomination_path),
        "--semantic-decision-nomination-sha256", _sha(state.origin_text.encode()),
        "--semantic-decision-task-id", TASK]
    if mode is not None:
        values += ["--semantic-decision-typed-nomination", mode]
    return values

def _prepare(state, mode="enabled", *, arguments=None):
    return owner.prepare_decision_owner_dispatch(arguments=_arguments(state, mode) if arguments is None else arguments,
        prompt_bytes=state.prompt, workspace=state.workspace)

def _verify_lineage(state, result, **changes):
    receipt = result.receipt
    binding = receipt["typed_nomination"]
    arguments = dict(artifact=Path(binding["lineage_artifact"]),
        expected_sha256=binding["lineage_sha256"], native_prompt=state.prompt.decode(),
        repository=state.repository, workspace=state.workspace,
        expected_origin_nomination_sha256=_sha(state.origin_text.encode()), expected_task_id=TASK,
        owner_binding=receipt, require_current_source=False)
    arguments.update(changes)
    return lineage_api.load_and_verify_typed_nomination_lineage(**arguments)


def _scoped_proposal(state, *, projected=False):
    """Authored @2 replay value with internally exact population commitments."""
    proposal = deepcopy(state.latest_source_proposal)
    proposal["schema"] = ("supervisor-worker-source-bound-typed-context-proposal@2" if projected
                          else "supervisor-source-bound-typed-context-proposal@2")
    proposal["adapter_version"] = "native-typed-dependency-proposal@2"
    ids = sorted(proposal["full_symbol_ids"])
    records = sorted((deepcopy(node["record"]) for node in proposal["graph"]["nodes"]),
                     key=lambda record: record["native_symbol_id"])

    def population(values):
        return {"count": len(values), "sha256": _sha(_encoded(values))}

    populations = {
        "symbols": {"full_root": population(ids), "materialized": population(ids),
                    "omitted_root": population([])},
        "relations": {"full_root": population([]), "materialized": population([]),
                      "omitted_root": population([])},
        "referenced_artifact_endpoints": {
            "full_root": population([]), "materialized": population([]),
            "omitted_root": population([])},
    }
    proposal.update(graph_scope="task_closure", materialized_symbol_ids=ids,
        omitted_root_symbol_ids=[], root_symbol_records=records,
        root_symbol_records_count=len(records), root_symbol_records_sha256=_sha(_encoded(records)),
        graph_population=populations, incoming_boundary_relation_rows=[],
        incoming_boundary_relation_count=0, incoming_boundary_relations_sha256=_sha(_encoded([])),
        graph_sha256=_sha(_encoded(proposal["graph"])), full_root_graph_materialized=True)
    commitment = {
        "schema": "supervisor-semantic-typed-graph-scope@1",
        "root_cid": proposal["semantic_root_cid"], "graph_root_id": proposal["graph_root_id"],
        "graph_scope": "task_closure",
        "materialization_basis": "all_outgoing_recorded_relations_of_explicit_seed_resolved_symbol_closure",
        "seed_symbol_ids": ["symbol-target"], "materialized_symbol_ids": ids,
        "omitted_root_symbol_ids": [], "index_cids": {"authored_index": "authored-index-cid"},
        "root_symbol_records_count": len(records),
        "root_symbol_records_sha256": proposal["root_symbol_records_sha256"],
        "full_capsule_identities_sha256": _sha(_encoded(state.parent["capsules"])),
        "full_relation_ledger_sha256": _sha(_encoded([])), "graph_population": populations,
        "incoming_boundary_relation_count": 0,
        "incoming_boundary_relations_sha256": proposal["incoming_boundary_relations_sha256"],
        "graph_sha256": proposal["graph_sha256"], "full_root_graph_materialized": True,
    }
    proposal.update(graph_scope_commitment=commitment,
                    graph_scope_commitment_sha256=_sha(_encoded(commitment)))
    return proposal


def _rehashed_retained_source(state, result, proposal):
    """Retain changed source bytes with real digest-addressed immutable custody."""
    receipt = result.receipt
    old_path = Path(receipt["typed_nomination"]["lineage_artifact"])
    lineage = json.loads(old_path.read_bytes())
    lineage["typed_nomination"]["receipt"]["source_proposal"] = deepcopy(proposal)
    raw = _encoded(lineage)
    digest = _sha(raw)
    path = old_path.parent / (digest + ".json")
    if path.exists():
        assert path.read_bytes() == raw
    else:
        path.write_bytes(raw)
        path.chmod(0o444)
    receipt["typed_nomination"].update(lineage_artifact=str(path), lineage_sha256=digest)
    return dict(artifact=path, expected_sha256=digest, owner_binding=receipt)


@pytest.mark.parametrize("environment", ["0", "1", ""])
@pytest.mark.parametrize("current", [False, True])
def test_retained_legacy_scope_replays_full_root_independently_of_live_default(handoff, monkeypatch,
                                                                            environment, current):
    state = handoff
    result = _prepare(state)
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", environment)
    verification = _verify_lineage(state, result, require_current_source=current)
    assert state.replay_calls[-1][1]["graph_scope"] == "full_root"
    assert state.replay_calls[-1][0] == ("current" if current else "historical")
    assert verification["source_freshness_revalidated"] is current
    assert verification["schema"] == "supervisor-semantic-typed-nomination-verification@1"


@pytest.mark.parametrize("environment", ["0", "1", ""])
@pytest.mark.parametrize("current", [False, True])
def test_retained_scoped_result_replays_task_closure_even_with_live_optout(handoff, monkeypatch,
                                                                         environment, current):
    state = handoff
    result = _prepare(state)
    proposal = _scoped_proposal(state)
    state.latest_source_proposal = deepcopy(proposal)
    changes = _rehashed_retained_source(state, result, proposal)
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_TASK_GRAPH", environment)
    verification = _verify_lineage(state, result, require_current_source=current, **changes)
    assert state.replay_calls[-1][1]["graph_scope"] == "task_closure"
    assert verification["source_freshness_revalidated"] is current
    assert verification["lineage_sha256"] == changes["expected_sha256"]
    assert verification["execution_authority"] is verification["proof_authority"] is False


@pytest.mark.parametrize("damage", [
    "missing_schema", "missing_adapter", "pure_schema", "foreign_schema", "foreign_adapter",
    "worker_schema_for_full_parent", "legacy_with_scope", "scoped_wrong_scope",
    "scoped_missing_commitment", "scoped_extra_commitment", "scoped_boolean_count",
    "scoped_duplicate_symbols", "scoped_overlapping_omission", "scoped_wrong_record_digest",
    "scoped_wrong_graph_digest", "scoped_wrong_scope_digest",
])
def test_unvalidated_retained_scope_contract_refuses_before_native_replay(handoff, damage):
    state = handoff
    result = _prepare(state)
    proposal = (_scoped_proposal(state) if damage.startswith("scoped_")
                else deepcopy(state.latest_source_proposal))
    if damage == "missing_schema":
        del proposal["schema"]
    elif damage == "missing_adapter":
        del proposal["adapter_version"]
    elif damage == "pure_schema":
        proposal["schema"] = "supervisor-semantic-typed-dependency-proposal@1"
    elif damage == "foreign_schema":
        proposal["schema"] = "supervisor-source-bound-typed-context-proposal@99"
    elif damage == "foreign_adapter":
        proposal["adapter_version"] = "native-typed-dependency-proposal@99"
    elif damage == "worker_schema_for_full_parent":
        proposal["schema"] = "supervisor-worker-source-bound-typed-context-proposal@1"
    elif damage == "legacy_with_scope":
        proposal["graph_scope"] = "task_closure"
    elif damage == "scoped_wrong_scope":
        proposal["graph_scope"] = "full_root"
    elif damage == "scoped_missing_commitment":
        del proposal["graph_scope_commitment"]
    elif damage == "scoped_extra_commitment":
        proposal["graph_scope_commitment"]["self_verified"] = True
        proposal["graph_scope_commitment_sha256"] = _sha(_encoded(proposal["graph_scope_commitment"]))
    elif damage == "scoped_boolean_count":
        proposal["graph_population"]["symbols"]["materialized"]["count"] = True
    elif damage == "scoped_duplicate_symbols":
        proposal["materialized_symbol_ids"].append("symbol-target")
    elif damage == "scoped_overlapping_omission":
        proposal["omitted_root_symbol_ids"] = ["symbol-target"]
    else:
        field = {"scoped_wrong_record_digest": "root_symbol_records_sha256",
                 "scoped_wrong_graph_digest": "graph_sha256",
                 "scoped_wrong_scope_digest": "graph_scope_commitment_sha256"}[damage]
        proposal[field] = "0" * 64
    changes = _rehashed_retained_source(state, result, proposal)
    before = len(state.replay_calls)
    with pytest.raises(ValueError):
        _verify_lineage(state, result, **changes)
    assert len(state.replay_calls) == before


@pytest.mark.parametrize("damage", ["population_digest", "population_count", "omitted_membership",
                                     "full_record_binding", "incoming_boundary", "graph_binding"])
def test_self_consistent_rehashed_scoped_claims_require_complete_native_replay(handoff, damage):
    state = handoff
    result = _prepare(state)
    correct = _scoped_proposal(state)
    state.latest_source_proposal = deepcopy(correct)
    proposal = deepcopy(correct)
    if damage == "population_digest":
        proposal["graph_population"]["relations"]["full_root"]["sha256"] = "0" * 64
    elif damage == "population_count":
        proposal["graph_population"]["relations"]["full_root"]["count"] = 1
    elif damage == "omitted_membership":
        proposal["materialized_symbol_ids"] = ["symbol-target"]
        proposal["omitted_root_symbol_ids"] = ["symbol-helper"]
    elif damage == "full_record_binding":
        proposal["root_symbol_records"][0]["capsule_cid"] = "capsule-foreign"
        proposal["root_symbol_records_sha256"] = _sha(_encoded(proposal["root_symbol_records"]))
    elif damage == "incoming_boundary":
        proposal["incoming_boundary_relation_rows"] = [{"edge_id": "authored-foreign-edge"}]
        proposal["incoming_boundary_relation_count"] = 1
        proposal["incoming_boundary_relations_sha256"] = _sha(_encoded(proposal["incoming_boundary_relation_rows"]))
    else:
        proposal["graph"]["nodes"][0]["record"]["capsule_cid"] = "capsule-foreign"
        proposal["graph_sha256"] = _sha(_encoded(proposal["graph"]))
    # Recompute every public repeated field and scope digest. These internal
    # commitments cannot replace the independently replayed native evidence.
    commitment = proposal["graph_scope_commitment"]
    for name in ("graph_population", "materialized_symbol_ids", "omitted_root_symbol_ids",
                 "root_symbol_records_sha256", "incoming_boundary_relation_count",
                 "incoming_boundary_relations_sha256", "graph_sha256"):
        commitment[name] = deepcopy(proposal[name])
    proposal["graph_scope_commitment_sha256"] = _sha(_encoded(commitment))
    changes = _rehashed_retained_source(state, result, proposal)
    before = len(state.replay_calls)
    with pytest.raises(ValueError, match="differs from replayed retained native evidence"):
        _verify_lineage(state, result, **changes)
    assert len(state.replay_calls) == before + 1
    assert state.replay_calls[-1][1]["graph_scope"] == "task_closure"


@pytest.mark.parametrize("projected", [False, True])
def test_schema_mode_must_match_independently_reconstructed_parent(handoff, projected):
    state = handoff
    _prepare(state)
    proposal = _scoped_proposal(state, projected=projected)
    assert lineage_api._retained_graph_scope(source_proposal=proposal, projected=projected) == "task_closure"
    with pytest.raises(ValueError):
        lineage_api._retained_graph_scope(source_proposal=proposal, projected=not projected)



@pytest.mark.parametrize("field", ["schema", "adapter_version", "graph_scope"])
def test_scoped_header_requires_literal_strings_even_before_serialization(handoff, field):
    state = handoff
    _prepare(state)
    proposal = _scoped_proposal(state)
    class Text(str):
        pass
    proposal[field] = Text(proposal[field])
    with pytest.raises(ValueError):
        lineage_api._retained_graph_scope(source_proposal=proposal, projected=False)


@pytest.mark.parametrize("scoped", [False, True])
def test_cross_version_scope_and_adapter_pair_cannot_choose_a_replay(handoff, scoped):
    state = handoff
    _prepare(state)
    proposal = (_scoped_proposal(state) if scoped else deepcopy(state.latest_source_proposal))
    proposal["adapter_version"] = "native-typed-dependency-proposal@" + ("1" if scoped else "2")
    with pytest.raises(ValueError):
        lineage_api._retained_graph_scope(source_proposal=proposal, projected=False)

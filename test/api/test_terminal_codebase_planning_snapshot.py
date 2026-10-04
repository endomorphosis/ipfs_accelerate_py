"""Real signed source/request binding and proposal-only semantic material reuse."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_planning_snapshot as api
from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_control as intent
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_codebase_proof_index as proof_index
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptSource, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    RepositoryAllowlist, scan_prompt_directory_detailed,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence, PromptGoalPlannerConfig
from ipfs_accelerate_py.agent_supervisor.planning import intent_codebase_matching as matching


ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts/codebase_ir_terminal_bench/qualification-20261001-04/supervisor/repository"
LEARNED_LOGIC = ROOT / "artifacts/codebase_ir_terminal_bench/decoder-qualification-20261001-01/logic/decoder-logic-result.json"
ACTUAL_INDEX = ROOT / "artifacts/codebase_ir_terminal_bench/intent-qualification-20261001-01/proof-index/manifest.json"


def _git(repository, *args):
    return subprocess.run(["git", "-C", str(repository), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _native_match(control, index, *, query=None, native_document=None, evidence_rows=()):
    authored = control["authored_control"]
    native = authored["native_document"] if native_document is None else native_document
    ref = native["sources"][0]
    identity = {field: ref[field] for field in (
        "ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}
    if query is None:
        atom = native["statements"][0]
        query = {"schema": matching.QUERY_SCHEMA, "review_ref": "authored-development-control:header-focus-test@1",
            "statement": {field: atom[field] for field in ("statement_id", "predicate", "arguments")},
            "source_path": "bottle.py", "symbols": ["_hkey", "_hval"], "property": "header_delimiter_rejection",
            "polarity": "positive", "domain": matching.reviewed_header_matching_domain(),
            "semantic_alignment_verified": False}
    return matching.match_intent_codebase(intent_document=native, source_text=authored["text"],
        source_identity=identity, query=query, evidence_rows=list(evidence_rows), current_source_snapshot=index["source_snapshot"])


@pytest.fixture(scope="module")
def signed_case(tmp_path_factory):
    """Create a new native profile; preserve both exact public and authored inputs."""
    output = tmp_path_factory.mktemp("signed-repository-planning")
    repository = output / "repository"
    repository.mkdir()
    control = intent.build_terminal_intent_control(public_instruction_bytes=(PUBLIC / prep.INSTRUCTION).read_bytes())
    (repository / "bottle.py").write_bytes((PUBLIC / "bottle.py").read_bytes())
    (repository / intent.AUTHORED_CONTROL_PATH).write_text(control["authored_control"]["text"])
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Isolated snapshot test")
    _git(repository, "config", "user.email", "snapshot@example.invalid")
    _git(repository, "add", "--", "bottle.py", intent.AUTHORED_CONTROL_PATH)
    _git(repository, "commit", "-qm", "Preserve exact public Bottle and separate authored control")
    prepared = prep.prepare(repository=repository, instruction=PUBLIC / prep.INSTRUCTION,
                            state=output / "state", disable_intent_autoencoder=True)
    original_manifest = deepcopy(prepared["manifest"])
    old = prepared["manifest"]["payload"]
    specs = deepcopy(old["tasks"])
    request = PromptWorkflowRequest.from_dict(old["planning_inputs"]["request"])
    request = replace(request,
        prompt_source=PromptSource.inline(control["authored_control"]["text"], redacted_metadata={
            "summary": "Separate authored atomic development control; complete public request remains unresolved"}),
        planning_policy=replace(request.planning_policy, allow_model=False),
        scan_policy=replace(request.scan_policy, include_patterns=("bottle.py", prep.INSTRUCTION,
                                                                  prep.SMOKE, intent.AUTHORED_CONTROL_PATH)))
    allowlist = RepositoryAllowlist.from_roots([repository])
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist)
    request = replace(request, program_root=details.receipt.program_root)
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist, previous=details)
    evidence = _select_evidence(request, details.receipt, PromptGoalPlannerConfig())
    specs[0]["acceptance"][0]["evidence_cids"] = [evidence[0].evidence_cid]
    domains = local.local_planning_domain_declarations(repository=repository,
        profile_dir=Path(old["profile_dir"]), lifecycle_dir=Path(old["lifecycle_dir"]), task_specs=specs)
    request = replace(request, intent_ir_root=local.content_identity(domains["intent"]),
        legal_ir_root=local.content_identity(domains["legal"]), security_ir_root=local.content_identity(domains["security"]))
    # Domain values remain unchanged when acceptance evidence references change.
    assert request.request_cid == details.receipt.request_cid
    manifest = local.author_local_benchmark_manifest(repository=repository,
        profile_dir=Path(old["profile_dir"]), lifecycle_dir=Path(old["lifecycle_dir"]),
        task_specs=specs, planning_roots={"request_cid": request.request_cid,
            "scan_cid": details.receipt.scan_cid, "program_root": request.program_root},
        planning_inputs={"request": request.to_dict(), "scan": details.receipt.to_dict(),
            "domain_declarations": domains, "selected_evidence": [row.to_dict() for row in evidence]},
        intent_requirements=control["authored_control"]["contract"])
    learned = json.loads(LEARNED_LOGIC.read_bytes())
    source = {"schema": "terminal-codebase-proof-source-snapshot@1", "source_path": "bottle.py",
        "source_sha256": hashlib.sha256((repository / "bottle.py").read_bytes()).hexdigest(),
        "source_bytes": len((repository / "bottle.py").read_bytes()),
        "source_unit_bindings": learned["native_header_derivation"]["modeled_symbols"],
        "source_context_sha256": hashlib.sha256(b"explicitly authored unverified source context").hexdigest(),
        "environment_sha256": hashlib.sha256(b"explicitly authored unverified environment").hexdigest(),
        "environment_ref_sha256": hashlib.sha256(b"explicitly authored unverified environment reference").hexdigest(),
        "translation_sha256": hashlib.sha256(b"explicitly authored unverified translation").hexdigest()}
    index_output = output / "authored-index-context"
    index_output.mkdir()
    index = {"schema": "authored-repository-index-context-test@1", "output": str(index_output), "source_snapshot": dict(source),
        "records": [{"kind": "authored_conditional_model_context", "source_semantics_verified": False}],
        "entries": [], "environment": {"scope": "explicitly_authored_unverified_environment_context"},
        **api.AUTHORITY}
    index["manifest_id"] = api._digest(index)
    (index_output / "manifest.json").write_bytes(api._wire(index) + b"\n")
    match = _native_match(control, index)
    inputs = {"manifest": manifest, "workflow_request": request.to_dict(), "control": control,
              "proof_index_manifest": index, "match_result": match}
    result = api.build_repository_proof_planning_snapshot(**inputs)
    return {"inputs": inputs, "result": result, "repository": repository,
            "prepared": prepared, "original_manifest": original_manifest}


def test_actual_signed_native_snapshot_preserves_complete_sources_and_separate_control(signed_case):
    inputs, result = signed_case["inputs"], signed_case["result"]
    declared, _, observed = local._manifest(inputs["manifest"], initial=True)
    assert declared["schema"] == "supervisor-local-benchmark-manifest@4"
    assert set(observed) == {"bottle.py", prep.INSTRUCTION, prep.SMOKE, intent.AUTHORED_CONTROL_PATH}
    assert (signed_case["repository"] / prep.INSTRUCTION).read_bytes() == (PUBLIC / prep.INSTRUCTION).read_bytes()
    assert (signed_case["repository"] / "bottle.py").read_bytes() == (PUBLIC / "bottle.py").read_bytes()
    assert local.decode_intent_requirement_contract(declared) == inputs["control"]["authored_control"]["contract"]
    assert signed_case["prepared"]["manifest"] == signed_case["original_manifest"]
    assert signed_case["original_manifest"]["payload"]["schema"] == "supervisor-local-benchmark-manifest@3"
    snapshot = result["snapshot"]
    assert snapshot["current_source_inventory"] == observed
    assert snapshot["current_code_leaf"] == {"source_path": "bottle.py", "source_sha256": observed["bottle.py"]["sha256"]}
    reference = snapshot["full_materials"]["conditional_proof_index"]
    manifest_bytes = (Path(inputs["proof_index_manifest"]["output"]) / "manifest.json").read_bytes()
    assert reference["schema"] == "repository-proof-index-artifact-reference@1"
    assert reference["manifest_id"] == inputs["proof_index_manifest"]["manifest_id"]
    assert reference["bytes"] == len(manifest_bytes)
    assert reference["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert reference["canonical_body_sha256"] == api._digest(inputs["proof_index_manifest"])
    assert "records" not in reference  # Complete body remains in the pinned file.
    assert snapshot["native_symbolic_receipt"]["observed_facts_supplied"] == 0
    assert snapshot["current_behavioral_facts"] == snapshot["behavioral_satisfied_requirements"] == []
    assert all(snapshot[key] is False for key in api.AUTHORITY)
    assert snapshot["public_request_fully_interpreted"] is snapshot["public_request_planned"] is False
    assert snapshot["repository_evidence_admitted"] is snapshot["worker_launched"] is False


def test_actual_native_model_budget_is_zero_not_just_context_metadata(signed_case):
    request = signed_case["inputs"]["workflow_request"]
    snapshot = signed_case["result"]["snapshot"]
    assert request["planning_policy"]["allow_model"] is False
    assert snapshot["native_input_snapshot"]["budget"]["max_model_calls"] == 0
    assert snapshot["full_materials"]["model_off"]["max_model_candidates"] == 0
    assert snapshot["provider_calls"] == snapshot["training_steps"] == 0
    assert snapshot["native_input_snapshot"]["material_binding"]["reuse_supported"]


def _pin_changed_index(index, tmp_path):
    output = tmp_path / "separately-pinned-index"
    output.mkdir()
    index["output"] = str(output)
    index.pop("manifest_id", None)
    index["manifest_id"] = api._digest(index)
    (output / "manifest.json").write_bytes(api._wire(index) + b"\n")


def test_semantic_context_changes_native_extra_digest_and_snapshot_without_changing_graph(signed_case, tmp_path):
    inputs = deepcopy(signed_case["inputs"])
    inputs["proof_index_manifest"]["records"][0]["diagnostic"] = "new unverified model-only context"
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    changed = api.build_repository_proof_planning_snapshot(**inputs)
    old = signed_case["result"]["snapshot"]
    new = changed["snapshot"]
    assert new["snapshot_id"] != old["snapshot_id"]
    assert new["full_materials_sha256"] != old["full_materials_sha256"]
    assert new["native_input_snapshot"]["snapshot_cid"] != old["native_input_snapshot"]["snapshot_cid"]
    assert new["native_input_snapshot"]["material_binding"]["field_digests"]["extra"] != old["native_input_snapshot"]["material_binding"]["field_digests"]["extra"]
    assert new["native_graph_cid"] == old["native_graph_cid"]
    assert changed["symbolic_plan"]["graph"].to_dict() == signed_case["result"]["symbolic_plan"]["graph"].to_dict()
    with pytest.raises(ValueError, match="exact current material replay"):
        api.replay_repository_proof_planning_snapshot(expected=old, **inputs)


def test_exact_replay_rebuilds_same_snapshot_without_aliasing_inputs(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    old = signed_case["result"]["snapshot"]
    actual = api.replay_repository_proof_planning_snapshot(expected=old, **inputs)
    assert actual["snapshot"] == old
    inputs["proof_index_manifest"]["records"].append({"kind": "caller later mutation"})
    inputs["control"]["public_request"]["text"] = "caller mutation"
    assert actual["snapshot"] == old


def test_native_empty_lookup_context_preserves_exact_intent_roots_and_all_residuals(signed_case):
    match = signed_case["inputs"]["match_result"]
    assert match["schema"] == matching.SCHEMA
    assert match["status"] == "unknown"
    assert match["evidence_rows"] == match["model_nominations"] == []
    assert len(match["residual_requirements"]) == 1
    assert match["residual_requirements"][0]["status"] == "unresolved_software_behavior"
    assert match["native_document_sha256"] == match["roots"]["native_document_sha256"]
    assert match["intent_document"] == signed_case["inputs"]["control"]["authored_control"]["native_document"]
    assert match["native_checker_invocations_here"] == 0
    assert match["source_inference_replayed_here"] is False


@pytest.mark.parametrize("field", ["residual_requirements", "model_nominations", "roots", "query"])
def test_false_flags_cannot_hide_modified_native_matching_materials(signed_case, field):
    inputs = deepcopy(signed_case["inputs"])
    result = inputs["match_result"]
    if field == "residual_requirements":
        result[field] = []
    elif field == "model_nominations":
        result[field] = [{"entry_id": "sha256:" + "0" * 64, "behavioral_satisfaction": False}]
    elif field == "roots":
        result[field]["native_document_sha256"] = "sha256:" + "0" * 64
    else:
        result[field]["review_ref"] = "altered:wellformed-reviewed-query"
    with pytest.raises(ValueError, match="exact native intent/evidence/residual reconstruction"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_foreign_native_document_cannot_supply_matching_context_for_authored_control(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    foreign = deepcopy(inputs["control"]["authored_control"]["native_document"])
    foreign["document_id"] += ":foreign-root"
    inputs["match_result"] = _native_match(inputs["control"], inputs["proof_index_manifest"], native_document=foreign)
    with pytest.raises(ValueError, match="exact native intent/evidence/residual reconstruction"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_rebuilt_reviewed_query_changes_material_digest_but_keeps_administrative_plan(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    query = deepcopy(inputs["match_result"]["query"])
    query["property"] = "explicitly_authored_unsupported_property"
    inputs["match_result"] = _native_match(inputs["control"], inputs["proof_index_manifest"], query=query)
    changed = api.build_repository_proof_planning_snapshot(**inputs)["snapshot"]
    old = signed_case["result"]["snapshot"]
    assert inputs["match_result"]["status"] == "unknown"
    assert "unsupported_property" in inputs["match_result"]["residual_requirements"][0]["reasons"]
    assert changed["native_graph_cid"] == old["native_graph_cid"]
    assert changed["snapshot_id"] != old["snapshot_id"]
    assert changed["native_input_snapshot"]["material_binding"]["field_digests"]["extra"] != old["native_input_snapshot"]["material_binding"]["field_digests"]["extra"]
    assert changed["current_behavioral_facts"] == changed["behavioral_satisfied_requirements"] == []


def test_lookup_claim_outside_pinned_index_is_refused_before_native_nomination(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    inputs["match_result"]["evidence_rows"] = [{"status": "hit", "entry_id": "sha256:" + "0" * 64}]
    with pytest.raises(ValueError, match="outside the complete pinned proof index"):
        api.build_repository_proof_planning_snapshot(**inputs)


@pytest.fixture(scope="module")
def actual_lookup_case(signed_case, tmp_path_factory):
    """Copy one immutable actual model row; no cache, checker, or inference call."""
    actual = json.loads(ACTUAL_INDEX.read_bytes())
    entry = next(row for row in actual["entries"]
                 if row["evidence"]["symbol"] == "_hkey"
                 and row["evidence"]["property"] == "unsafe_converted_input_accepted")
    assert entry["evidence"]["classification"] == "conditional_model_sat_witness"
    inputs = deepcopy(signed_case["inputs"])
    index = inputs["proof_index_manifest"]
    index["source_snapshot"] = actual["source_snapshot"]
    index["environment"] = actual["environment"]
    index["entries"] = [entry]
    index["records"] = [{"kind": "copied_native_conditional_model_test_context",
                         "historical_index_manifest_id": actual["manifest_id"],
                         "checker_invocations_here": 0, "source_semantics_verified": False}]
    _pin_changed_index(index, tmp_path_factory.mktemp("copied-native-lookup"))
    lookup = proof_index._lookup(entry, index["environment"])
    inputs["match_result"] = _native_match(inputs["control"], index, evidence_rows=[lookup])
    return inputs


def test_actual_native_lookup_is_bound_to_complete_index_and_stays_conditional(actual_lookup_case, monkeypatch):
    def forbidden_runtime_work(*args, **kwargs):
        raise AssertionError("snapshot validation must not execute native cache or checkers")
    monkeypatch.setattr(proof_index, "lookup_terminal_codebase_model_evidence", forbidden_runtime_work)
    monkeypatch.setattr(proof_index, "capture_terminal_codebase_proof_environment", forbidden_runtime_work)
    inputs = deepcopy(actual_lookup_case)
    snapshot = api.build_repository_proof_planning_snapshot(**inputs)["snapshot"]
    match = inputs["match_result"]
    assert match["evidence_rows"][0]["evidence"]["classification"] == "conditional_model_sat_witness"
    assert match["model_nominations"]
    assert len(match["residual_requirements"]) == 1
    assert match["residual_requirements"][0]["status"] == "unresolved_software_behavior"
    assert match["native_checker_invocations_here"] == 0
    assert match["source_inference_replayed_here"] is False
    assert snapshot["full_materials"]["intent_codebase_match"] == match
    assert snapshot["current_behavioral_facts"] == snapshot["behavioral_satisfied_requirements"] == []
    assert snapshot["repository_evidence_admitted"] is False
    assert all(snapshot[key] is False for key in api.AUTHORITY)


@pytest.mark.parametrize("field", ["witness", "open_frontiers", "key_relationship", "expected_environment_sha256"])
def test_modified_native_lookup_is_refused_despite_unchanged_entry_id_and_false_authority(actual_lookup_case, field):
    inputs = deepcopy(actual_lookup_case)
    lookup = inputs["match_result"]["evidence_rows"][0]
    if field == "witness":
        lookup["evidence"]["checker_receipt"]["model_text"] += "\ncaller-authored witness"
    elif field == "open_frontiers":
        lookup["evidence"][field] = []
    elif field == "key_relationship":
        lookup[field]["dimensions"]["policy"]["caller-authored"] = "different policy"
    else:
        lookup[field] = "0" * 64
    assert all(lookup[key] is False for key in proof_index.AUTHORITY)
    with pytest.raises(ValueError, match="lookup differs from complete expected entry"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_valid_native_lookup_cannot_name_entry_outside_pinned_index(actual_lookup_case, tmp_path):
    inputs = deepcopy(actual_lookup_case)
    inputs["proof_index_manifest"]["entries"] = []
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    with pytest.raises(ValueError, match="outside the complete pinned proof index"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_duplicate_pinned_entry_ids_are_refused_before_matching(actual_lookup_case, tmp_path):
    inputs = deepcopy(actual_lookup_case)
    inputs["proof_index_manifest"]["entries"] *= 2
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    with pytest.raises(ValueError, match="ambiguous complete proof-index entries"):
        api.build_repository_proof_planning_snapshot(**inputs)


@pytest.mark.parametrize("field", ["current_behavioral_facts", "behavioral_satisfied_requirements"])
def test_caller_supplied_behavioral_success_is_refused(signed_case, field):
    inputs = deepcopy(signed_case["inputs"])
    inputs["match_result"][field] = ["fake behavior success"]
    with pytest.raises(ValueError, match="cannot supply behavioral facts"):
        api.build_repository_proof_planning_snapshot(**inputs)


@pytest.mark.parametrize("container", ["match_result", "proof_index_manifest"])
@pytest.mark.parametrize("field", ["proof_authority", "execution_authority", "source_semantics_verified"])
def test_caller_fake_index_or_match_authority_is_refused(signed_case, container, field):
    inputs = deepcopy(signed_case["inputs"])
    inputs[container][field] = True
    with pytest.raises(ValueError, match="authority"):
        api.build_repository_proof_planning_snapshot(**inputs)


@pytest.mark.parametrize("change", ["index_hash", "match_hash", "index_path"])
def test_index_and_match_must_equal_actual_signed_code_leaf(signed_case, change, tmp_path):
    inputs = deepcopy(signed_case["inputs"])
    if change == "index_hash":
        inputs["proof_index_manifest"]["source_snapshot"]["source_sha256"] = "0" * 64
    elif change == "match_hash":
        inputs["match_result"]["current_source_snapshot"]["source_sha256"] = "0" * 64
    else:
        inputs["proof_index_manifest"]["source_snapshot"]["source_path"] = prep.INSTRUCTION
    if change != "match_hash":
        _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    with pytest.raises(ValueError, match="current signed code leaf"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_workflow_request_edit_cannot_bypass_signed_native_inputs(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    request = PromptWorkflowRequest.from_dict(inputs["workflow_request"])
    inputs["workflow_request"] = replace(request,
        planning_policy=replace(request.planning_policy, allow_model=True)).to_dict()
    with pytest.raises(ValueError, match="signed native planning inputs"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_genuine_signed_model_enabled_request_cannot_be_called_model_off(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    original = signed_case["original_manifest"]["payload"]
    assert original["planning_inputs"]["request"]["planning_policy"]["allow_model"] is True
    inputs["manifest"] = local.author_local_benchmark_manifest(repository=signed_case["repository"],
        profile_dir=Path(original["profile_dir"]), lifecycle_dir=Path(original["lifecycle_dir"]),
        task_specs=deepcopy(original["tasks"]), planning_roots=deepcopy(original["planning_roots"]),
        planning_inputs=deepcopy(original["planning_inputs"]),
        intent_requirements=inputs["control"]["authored_control"]["contract"])
    inputs["workflow_request"] = deepcopy(original["planning_inputs"]["request"])
    with pytest.raises(ValueError, match="signed native planning inputs|model"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_native_profile_rejects_signed_manifest_payload_tamper(signed_case):
    inputs = deepcopy(signed_case["inputs"])
    inputs["manifest"]["payload"]["sources"]["bottle.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_complete_proof_index_body_is_checked_before_freezing_reference(signed_case, tmp_path):
    inputs = deepcopy(signed_case["inputs"])
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    path = Path(inputs["proof_index_manifest"]["output"]) / "manifest.json"
    modified = deepcopy(inputs["proof_index_manifest"])
    modified["records"].append({"kind": "unbound additional model context"})
    path.write_bytes(api._wire(modified) + b"\n")
    with pytest.raises(ValueError, match="artifact differs"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_index_manifest_symlink_is_refused_before_source_or_planning(signed_case, tmp_path):
    inputs = deepcopy(signed_case["inputs"])
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    path = Path(inputs["proof_index_manifest"]["output"]) / "manifest.json"
    alternate = path.with_name("alternate.json")
    path.rename(alternate)
    path.symlink_to(alternate)
    with pytest.raises(ValueError, match="canonical immutable proof-index manifest"):
        api.build_repository_proof_planning_snapshot(**inputs)


def test_same_json_with_changed_raw_index_bytes_invalidates_snapshot_reuse(signed_case, tmp_path):
    inputs = deepcopy(signed_case["inputs"])
    _pin_changed_index(inputs["proof_index_manifest"], tmp_path)
    original = api.build_repository_proof_planning_snapshot(**inputs)["snapshot"]
    path = Path(inputs["proof_index_manifest"]["output"]) / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    assert json.loads(path.read_bytes()) == inputs["proof_index_manifest"]
    with pytest.raises(ValueError, match="exact current material replay"):
        api.replay_repository_proof_planning_snapshot(expected=original, **inputs)


@pytest.mark.parametrize("path", ["bottle.py", prep.INSTRUCTION, intent.AUTHORED_CONTROL_PATH])
def test_edited_signed_sources_refuse_current_replay_and_restore_cleanly(signed_case, path):
    source = signed_case["repository"] / path
    before = source.read_bytes()
    try:
        source.write_bytes(before + b"\n")
        with pytest.raises(ValueError):
            api.replay_repository_proof_planning_snapshot(expected=signed_case["result"]["snapshot"],
                                                        **signed_case["inputs"])
    finally:
        source.write_bytes(before)
    replay = api.replay_repository_proof_planning_snapshot(expected=signed_case["result"]["snapshot"],
                                                        **signed_case["inputs"])
    assert replay["snapshot"] == signed_case["result"]["snapshot"]

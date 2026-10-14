"""Declarative context selection preserves the complete public task contract."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as owner


def scoped_profile():
    return {
        "schema": owner.SCOPED_SCHEMA,
        "instruction_sha256": owner.instruction_sha256("Repair source.py.\n"),
        "input_paths": ["notes.md", "source.py", *[f"unselected_{index:03d}.py" for index in range(73)]],
        "outputs": [{"path": "source.py", "effect": "modify", "media_type": "text/x-python"}],
        "semantic_scope": {
            "schema": owner.SEMANTIC_SCOPE_SCHEMA,
            "input_paths": ["notes.md", "source.py"],
            "required_input_paths": ["notes.md", "source.py"],
            "program_paths": ["source.py"],
        },
    }


def legacy_profile(declared):
    return {**{key: value for key, value in declared.items() if key != "semantic_scope"},
            "schema": owner.SCHEMA}


def test_selected_context_retains_complete_inventory_task_scope_and_smoke():
    declared = scoped_profile()
    base = legacy_profile(declared)
    checked = owner.validate_task_profile(declared, instruction="Repair source.py.\n")
    assert checked == owner.validate_task_profile(checked)
    assert checked["input_paths"] == base["input_paths"]
    assert len(owner.task_profile_worker_inputs(checked)) == 78
    assert owner.task_profile_worker_inputs(checked) == owner.task_profile_worker_inputs(base)
    assert owner.task_profile_semantic_inputs(checked) == [
        owner.INSTRUCTION, owner.SMOKE, owner.PROFILE, "notes.md", "source.py"]
    assert owner.task_profile_semantic_required_paths(checked) == [
        owner.INSTRUCTION, owner.SMOKE, owner.PROFILE, "notes.md", "source.py"]
    assert owner.task_profile_index_paths(checked) == ["source.py"]
    assert owner.task_profile_spec(checked, policy_cid="fixture") == owner.task_profile_spec(base, policy_cid="fixture")
    assert owner.task_profile_specs(checked, policy_cid="fixture") == owner.task_profile_specs(base, policy_cid="fixture")
    assert owner.task_profile_smoke(checked) == owner.task_profile_smoke(base)
    assert set(owner.task_profile_spec(checked, policy_cid="fixture")["scope_paths"]) == set(
        owner.task_profile_worker_inputs(base))
    with pytest.raises(ValueError, match="64-file vector bound"):
        owner.task_profile_index_paths(base)


def test_compatibility_module_exports_the_same_producer_and_helpers():
    for name in ("SCOPED_SCHEMA", "SEMANTIC_SCOPE_SCHEMA", "validate_task_profile",
                 "task_profile_semantic_inputs", "task_profile_semantic_required_paths",
                 "task_profile_index_paths"):
        assert getattr(profiles, name) is getattr(owner, name)


def test_profile_and_helper_results_do_not_alias_declarations():
    declared = scoped_profile()
    frozen = deepcopy(declared)
    checked = owner.validate_task_profile(declared)
    checked["semantic_scope"]["input_paths"].clear()
    checked["input_paths"].clear()
    owner.task_profile_semantic_inputs(declared).clear()
    owner.task_profile_semantic_required_paths(declared).clear()
    owner.task_profile_index_paths(declared).clear()
    assert declared == frozen


def test_complete_inputs_still_canonicalize_while_scope_order_is_exact():
    declared = scoped_profile()
    declared["input_paths"].reverse()
    checked = owner.validate_task_profile(declared)
    assert checked["input_paths"] == sorted(declared["input_paths"])
    assert checked["semantic_scope"] == declared["semantic_scope"]
    assert json.loads(owner.task_profile_bytes(declared)) == checked
    assert owner.task_profile_bytes(declared).endswith(b"\n")
    assert set(checked) == {"schema", "instruction_sha256", "input_paths", "outputs", "semantic_scope"}


@pytest.mark.parametrize("mutation", [
    "unknown_top", "data_inputs", "tasks", "nested_task", "missing_scope", "scope_not_object",
    "scope_unknown", "scope_schema", "scope_missing", "empty_selected", "empty_program",
    "selected_absent", "required_absent", "program_absent", "program_not_python",
    "modify_not_selected", "modify_not_required", "python_modify_not_program", "support_not_required",
    "instruction_digest", "inventory_cid", "source_hashes",
])
def test_scoped_profile_rejects_ambiguous_or_incomplete_context_contracts(mutation):
    declared = scoped_profile()
    scope = declared["semantic_scope"]
    if mutation == "unknown_top": declared["execution_authority"] = True
    elif mutation == "data_inputs": declared["data_inputs"] = []
    elif mutation == "tasks": declared["tasks"] = []
    elif mutation == "nested_task": declared["task_profile"] = legacy_profile(declared)
    elif mutation == "missing_scope": declared.pop("semantic_scope")
    elif mutation == "scope_not_object": declared["semantic_scope"] = []
    elif mutation == "scope_unknown": scope["proof_authority"] = False
    elif mutation == "scope_schema": scope["schema"] = "terminal-public-task-semantic-scope@2"
    elif mutation == "scope_missing": scope.pop("program_paths")
    elif mutation == "empty_selected": scope["input_paths"] = []
    elif mutation == "empty_program": scope["program_paths"] = []
    elif mutation == "selected_absent": scope["input_paths"].append("unknown.py")
    elif mutation == "required_absent": scope["required_input_paths"].append("unselected_000.py")
    elif mutation == "program_absent": scope["program_paths"].append("unselected_000.py")
    elif mutation == "program_not_python": scope["program_paths"] = ["notes.md", "source.py"]
    elif mutation == "modify_not_selected": declared["outputs"][0]["path"] = "unselected_000.py"
    elif mutation == "modify_not_required": scope["required_input_paths"] = ["notes.md"]
    elif mutation == "python_modify_not_program":
        scope["input_paths"].append("unselected_000.py")
        scope["required_input_paths"].append("unselected_000.py")
        scope["program_paths"] = ["unselected_000.py"]
    elif mutation == "support_not_required": scope["required_input_paths"] = ["source.py"]
    elif mutation == "instruction_digest": declared["instruction_sha256"] = "0" * 64
    elif mutation == "inventory_cid": scope["source_inventory_cid"] = "postcommit-cycle"
    elif mutation == "source_hashes": scope["source_sha256"] = {"source.py": "0" * 64}
    with pytest.raises(ValueError):
        owner.validate_task_profile(declared, instruction="Repair source.py.\n")


@pytest.mark.parametrize("field", ["input_paths", "required_input_paths", "program_paths"])
@pytest.mark.parametrize("mutation", ["tuple", "duplicate", "unsorted", "nonstring", "reserved", "alias"])
def test_scope_lists_are_exact_without_silent_normalization(field, mutation):
    declared = scoped_profile()
    scope = declared["semantic_scope"]
    scope["input_paths"].append("unselected_000.py")
    scope["required_input_paths"].append("unselected_000.py")
    scope["program_paths"].append("unselected_000.py")
    if mutation == "tuple": scope[field] = tuple(scope[field])
    elif mutation == "duplicate": scope[field].append(scope[field][-1])
    elif mutation == "unsorted": scope[field].reverse()
    elif mutation == "nonstring": scope[field].append(True)
    elif mutation == "reserved": scope[field] = [owner.INSTRUCTION]
    elif mutation == "alias": scope[field] = ["./source.py"]
    with pytest.raises(ValueError):
        owner.validate_task_profile(declared)


@pytest.mark.parametrize("name", [".supervisor-instruction.md", ".supervisor-public-smoke.py",
                                 ".supervisor-task-profile.json", "sub/.supervisor-extra"])
def test_generated_support_paths_cannot_be_nominated_as_ordinary_inputs(name):
    declared = scoped_profile()
    declared["input_paths"].append(name)
    declared["semantic_scope"]["input_paths"] = sorted([name, "source.py"])
    declared["semantic_scope"]["required_input_paths"] = sorted([name, "source.py"])
    with pytest.raises(ValueError, match="nonreserved"):
        owner.validate_task_profile(declared)


@pytest.mark.parametrize("count,accepted", [(61, True), (62, False)])
def test_scope_file_limit_counts_three_mandatory_supports(count, accepted):
    declared = scoped_profile()
    names = [f"source_{index:03d}.py" for index in range(75)]
    declared["input_paths"] = names
    declared["outputs"][0]["path"] = names[0]
    declared["semantic_scope"].update(input_paths=names[:count], required_input_paths=names[:1],
                                      program_paths=names[:count])
    if accepted:
        assert len(owner.task_profile_semantic_inputs(declared)) == 64
        assert len(owner.task_profile_worker_inputs(declared)) == 78
    else:
        with pytest.raises(ValueError, match="61 exact"):
            owner.validate_task_profile(declared)


def test_required_ordinary_inputs_can_be_empty_for_create_only_task():
    declared = scoped_profile()
    declared["outputs"] = [{"path": "created.py", "effect": "create", "media_type": "text/x-python"}]
    declared["semantic_scope"].update(input_paths=["source.py"], required_input_paths=[])
    assert owner.task_profile_semantic_required_paths(declared) == [owner.INSTRUCTION, owner.SMOKE, owner.PROFILE]
    assert owner.task_profile_index_paths(declared) == ["source.py"]


def test_nonpython_modified_input_requires_raw_coverage_but_not_program_role():
    declared = scoped_profile()
    declared["outputs"].append({"path": "notes.md", "effect": "modify", "media_type": "text/markdown"})
    checked = owner.validate_task_profile(declared)
    assert "notes.md" in owner.task_profile_semantic_required_paths(checked)
    assert "notes.md" not in owner.task_profile_index_paths(checked)


def test_scoped_canonical_bytes_retain_existing_profile_byte_bound():
    declared = scoped_profile()
    names = [f"source_{index:03d}_" + "x" * 480 + ".py" for index in range(60)]
    declared["input_paths"] = names
    declared["outputs"][0]["path"] = names[0]
    declared["semantic_scope"].update(input_paths=names, required_input_paths=names[:1], program_paths=names)
    assert len(owner.task_profile_bytes(legacy_profile(declared))) < 65536
    with pytest.raises(ValueError, match="65536-byte"):
        owner.task_profile_bytes(declared)


@pytest.mark.parametrize("schema", [owner.SCHEMA, owner.DATA_SCHEMA])
def test_legacy_context_helpers_fall_back_to_complete_inventory(schema):
    declared = legacy_profile(scoped_profile())
    declared["schema"] = schema
    if schema == owner.DATA_SCHEMA:
        declared["data_inputs"] = []
    assert owner.task_profile_semantic_inputs(declared) == owner.task_profile_worker_inputs(declared)
    assert owner.task_profile_semantic_required_paths(declared) == [owner.INSTRUCTION, owner.SMOKE, owner.PROFILE]
    assert "semantic_scope" not in owner.validate_task_profile(declared)


@pytest.mark.parametrize("mixed,profile_digest,smoke_digest,spec_digest", [
    (True, "8c3a9518fb8ed888bb24fc1f23e48deaa8b79b13514819c9cdea2ff754bdec47",
     "488e32462a4d2bf595b50bccc37cf74f1be351461b7ee2ab040f74527fdd8549",
     "2df2f197bf9187bf708b12124a03bcb8ac3d18266e1029708fbcf38a2ebb55c9"),
    (False, "93a47f93d1aa4a4de018f403328196f75232080c1599acf238353ee2d394e42b",
     "779b97b174ae07b680dbe08e3a9e2ca00ed8e52caf0ba3df742399d58aa71301",
     "e9015b0824cd21ed1b82f04f4d2046fa7dd9822422ba0b7e0e7bb090a5f76f95"),
])
def test_version_one_published_golden_bytes_remain_exact(mixed, profile_digest, smoke_digest, spec_digest):
    declared = {"schema": owner.SCHEMA,
        "instruction_sha256": owner.instruction_sha256("Implement public_source and write result.py.\n"),
        "input_paths": ["source.py"] if mixed else [],
        "outputs": [{"path": "result.py", "effect": "create", "media_type": "text/x-python"}]}
    if mixed:
        declared["outputs"].insert(0, {"path": "source.py", "effect": "modify", "media_type": "text/x-python"})
    assert hashlib.sha256(owner.task_profile_bytes(declared)).hexdigest() == profile_digest
    assert hashlib.sha256(owner.task_profile_smoke(declared).encode()).hexdigest() == smoke_digest
    raw = json.dumps(owner.task_profile_spec(declared, policy_cid="fixture"), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == spec_digest

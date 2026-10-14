"""Generic public declarations through native Git, signing, indexing and smoke."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as owner


def profile(text="Implement public_source and write result.py.\n", inputs=("source.py",), output="result.py"):
    return {"schema": profiles.SCHEMA, "instruction_sha256": profiles.instruction_sha256(text),
            "input_paths": list(inputs), "outputs": [{"path": output,
                "effect": "modify" if output in inputs else "create", "media_type": "text/x-python"}]}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def original(tmp_path, *, empty=False, dirty=False):
    root = tmp_path / "app"
    root.mkdir()
    git(root, "init", "-q")
    if not empty:
        (root / "source.py").write_text("def public_source():\n    return 1\n")
        git(root, "add", "source.py")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "public inputs")
    if dirty:
        (root / "source.py").write_text("def public_source():\n    return 2\n")
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Implement public_source and write result.py.\n")
    declared = profile(instruction.read_text(), inputs=() if empty else ("source.py",))
    state = tmp_path / "state"
    return root, instruction, state, declared


def prepare(args):
    root, instruction, state, declared = args
    return prep.prepare(repository=root, instruction=instruction, state=state,
                        task_profile=declared, disable_intent_autoencoder=True)


@pytest.mark.parametrize("name", ["/app/x.py", "../x.py", "a/../x.py", "./x.py", "a//x.py",
    ".git/config", "a/.git/config", ".runtime/a", ".supervisor-any", "x/*.py", "x?.py", "a\\b.py", "a\nb.py"])
def test_profile_rejects_nonexact_or_reserved_paths(name):
    declared = profile(inputs=(name,))
    with pytest.raises(ValueError):
        profiles.validate_task_profile(declared)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate_input", "duplicate_output", "absent_modify",
    "present_create", "directory_overlap", "bad_media", "too_many_sources", "instruction"])
def test_profile_closes_population_and_effect_contracts(mutation):
    declared = profile()
    if mutation == "unknown": declared["proof_authority"] = True
    elif mutation == "duplicate_input": declared["input_paths"] *= 2
    elif mutation == "duplicate_output": declared["outputs"] *= 2
    elif mutation == "absent_modify": declared["outputs"][0]["effect"] = "modify"
    elif mutation == "present_create": declared["outputs"][0]["path"] = "source.py"
    elif mutation == "directory_overlap": declared["outputs"][0]["path"] = "source.py/child"
    elif mutation == "bad_media": declared["outputs"][0]["media_type"] = "text/python; executable"
    elif mutation == "too_many_sources": declared["input_paths"] = [f"s{i}.py" for i in range(253)]
    elif mutation == "instruction": declared["instruction_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        profiles.validate_task_profile(declared, instruction="Implement public_source and write result.py.\n")


def test_profile_normalizes_line_endings_without_rewording():
    assert profiles.instruction_sha256("A\r\nB\r") == profiles.instruction_sha256("A\nB\n")
    assert profiles.instruction_sha256("A B") != profiles.instruction_sha256("A  B")


def output_population(declared, *, creates, modifies):
    inputs = [f"existing_{index:02d}.py" for index in range(modifies)]
    declared["input_paths"] = inputs
    declared["outputs"] = [
        {"path": name, "effect": "modify", "media_type": "text/x-python"} for name in inputs
    ] + [
        {"path": f"new_{index:02d}.py", "effect": "create", "media_type": "text/x-python"}
        for index in range(creates)
    ]
    return declared


@pytest.mark.parametrize("creates,modifies", [(32, 0), (32, 32), (0, 64)])
def test_profile_preserves_native_output_population_boundaries(creates, modifies):
    declared = output_population(profile(), creates=creates, modifies=modifies)
    checked = profiles.validate_task_profile(declared)
    assert len(checked["outputs"]) == creates + modifies
    assert sum(item["effect"] == "create" for item in checked["outputs"]) == creates
    assert checked == profiles.validate_task_profile(checked)


@pytest.mark.parametrize("creates,modifies", [(33, 0), (33, 31), (64, 0)])
def test_profile_rejects_created_population_above_native_bound(creates, modifies):
    declared = output_population(profile(), creates=creates, modifies=modifies)
    with pytest.raises(ValueError, match="native 32-created-output manifest bound"):
        profiles.validate_task_profile(declared)


@pytest.mark.parametrize("inputs,creates,accepted", [(252, 1, True), (252, 2, False),
    (221, 32, True), (222, 32, False)])
def test_profile_counts_support_and_future_creations_in_source_bound(inputs, creates, accepted):
    declared = output_population(profile(), creates=creates, modifies=0)
    declared["input_paths"] = [f"source_{index:03d}.py" for index in range(inputs)]
    if accepted:
        checked = profiles.validate_task_profile(declared)
        assert len(profiles.task_profile_worker_inputs(checked)) + len(checked["outputs"]) == 256
    else:
        with pytest.raises(ValueError, match="native 256-source published manifest bound"):
            profiles.validate_task_profile(declared)


@pytest.mark.parametrize("inputs,accepted", [(125, True), (126, False)])
def test_modify_only_profile_retains_native_source_bound(inputs, accepted):
    names = [f"source_{index:03d}.py" for index in range(inputs)]
    declared = profile(inputs=names, output=names[0])
    if accepted:
        assert len(profiles.task_profile_worker_inputs(declared)) == 128
    else:
        with pytest.raises(ValueError, match="native 128-source manifest bound"):
            profiles.validate_task_profile(declared)


@pytest.mark.parametrize("overflow", ["created_population", "published_sources"])
def test_excess_creations_rejected_before_advice_or_workspace_mutation(tmp_path, monkeypatch, overflow):
    from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as advisor

    args = original(tmp_path, dirty=True)
    root, _, state, declared = args
    creates = 33 if overflow == "created_population" else 2
    declared["outputs"] = output_population(profile(), creates=creates, modifies=0)["outputs"]
    expected_bound = "32-created-output manifest" if overflow == "created_population" else "256-source published manifest"
    if overflow == "published_sources":
        extras = [f"extra_{index:03d}.py" for index in range(251)]
        for name in extras:
            (root / name).write_text("pass\n")
        git(root, "add", "--", *extras)
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "additional public inputs")
        declared["input_paths"].extend(extras)
    original_head = git(root, "rev-parse", "HEAD")
    original_status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    original_source = (root / "source.py").read_bytes()

    def unexpected_advice(**kwargs):
        pytest.fail("invalid output population must fail before preprocessing")

    monkeypatch.setattr(advisor, "prepare_intent_advice", unexpected_advice)
    with pytest.raises(ValueError, match="native " + expected_bound + " bound"):
        prepare(args)
    assert git(root, "rev-parse", "HEAD") == original_head
    assert git(root, "status", "--porcelain=v1", "--untracked-files=all") == original_status
    assert (root / "source.py").read_bytes() == original_source
    assert not state.exists()
    assert not (root / ".runtime").exists()
    assert not any((root / name).exists() for name in (profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE))


def test_native_prepare_accepts_exactly_32_created_outputs(tmp_path):
    args = original(tmp_path, empty=True)
    output_population(args[3], creates=32, modifies=0)
    result = prepare(args)
    assert prep._load_prepared(args[2]) == result
    assert result["manifest"]["payload"]["created_outputs"] == sorted(
        item["path"] for item in args[3]["outputs"])
    assert result["provider_calls"] == 0


@pytest.mark.parametrize("empty,dirty", [(False, False), (False, True), (True, False)])
def test_native_prepare_load_preserves_sources_and_signs_exact_profile(tmp_path, empty, dirty):
    args = original(tmp_path, empty=empty, dirty=dirty)
    root, instruction, state, declared = args
    before = {name: (root / name).read_bytes() for name in declared["input_paths"]}
    result = prepare(args)
    assert result == prep._load_prepared(state)
    assert result["task_profile"] == declared
    assert result["worker_inputs"] == profiles.task_profile_worker_inputs(declared)
    assert result["manifest"]["payload"]["created_outputs"] == ["result.py"]
    assert (root / profiles.PROFILE).read_bytes() == profiles.task_profile_bytes(declared)
    assert not (root / "result.py").exists()
    assert {name: (root / name).read_bytes() for name in before} == before
    assert result["provider_calls"] == 0


@pytest.mark.parametrize("mutation", ["profile", "worker_inputs", "spec", "smoke", "remove_profile"])
def test_unsigned_preparation_changes_cannot_change_signed_task(tmp_path, mutation):
    args = original(tmp_path)
    root, _, state, _ = args
    result = prepare(args)
    if mutation == "profile": result["task_profile"]["outputs"][0]["path"] = "other.py"
    elif mutation == "worker_inputs": result["worker_inputs"].remove("source.py")
    elif mutation == "spec": result["spec"]["outputs"][0]["path"] = "other.py"
    elif mutation == "smoke": (root / prep.SMOKE).write_text("pass\n")
    elif mutation == "remove_profile": del result["task_profile"]
    prep._write(state / "prepared.json", result)
    with pytest.raises(ValueError): prep._load_prepared(state)


@pytest.mark.parametrize("mutation", ["untracked", "ignored", "input_symlink", "output_parent_symlink", "inventory"])
def test_preparation_rejects_undeclared_source_or_unsafe_paths_before_state(tmp_path, mutation):
    args = original(tmp_path)
    root, _, state, declared = args
    if mutation in {"untracked", "ignored"}:
        (root / "extra.py").write_text("pass\n")
        if mutation == "ignored": (root / ".git/info/exclude").write_text("extra.py\n")
    elif mutation == "input_symlink":
        (root / "source.py").unlink()
        (root / "source.py").symlink_to(args[1])
    elif mutation == "output_parent_symlink":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        declared["outputs"][0]["path"] = "link/result.py"
    elif mutation == "inventory": declared["input_paths"] = []
    with pytest.raises(ValueError): prepare(args)
    assert not state.exists()


def smoke(tmp_path, declared):
    (tmp_path / prep.SMOKE).write_text(profiles.task_profile_smoke(declared))
    spec = profiles.task_profile_spec(declared, policy_cid="fixture")
    argv = spec["validations"][0]["argv"]
    assert argv == prep.ARGV
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=10)


@pytest.mark.parametrize("name", ["ast.py", "json.py", "pathlib.py", "ast/__init__.py", "sitecustomize.py"])
@pytest.mark.parametrize("valid", [True, False])
def test_structural_smoke_cannot_import_candidate_modules(tmp_path, monkeypatch, name, valid):
    shadow = tmp_path / name
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_text("open('EXECUTED', 'w').write('candidate import')\nraise RuntimeError('candidate import')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    (tmp_path / "result.py").write_text("pass\n" if valid else "def broken(:\n")
    assert (smoke(tmp_path, profile(inputs=(name,))).returncode == 0) is valid
    assert not (tmp_path / "EXECUTED").exists()


def test_structural_smoke_parses_without_executing_candidate(tmp_path):
    declared = profile(inputs=())
    (tmp_path / "result.py").write_text("from pathlib import Path\nPath('EXECUTED').write_text('bad')\nraise RuntimeError('not executed')\n")
    assert smoke(tmp_path, declared).returncode == 0
    assert not (tmp_path / "EXECUTED").exists()


@pytest.mark.parametrize("mutation", ["missing", "syntax", "symlink", "parent_symlink", "oversize", "fifo"])
def test_structural_smoke_rejects_missing_invalid_or_unbounded_outputs(tmp_path, mutation):
    declared = profile(inputs=())
    output = tmp_path / "result.py"
    if mutation == "syntax": output.write_text("def broken(:\n")
    elif mutation == "symlink": output.symlink_to(tmp_path / prep.SMOKE)
    elif mutation == "parent_symlink":
        declared["outputs"][0]["path"] = "link/result.py"
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        output.write_text("pass\n")
    elif mutation == "oversize": output.write_bytes(b" " * (profiles.MAX_OUTPUT_BYTES + 1))
    elif mutation == "fifo": os.mkfifo(output)
    assert smoke(tmp_path, declared).returncode != 0


def test_empty_task_prepares_native_absence_without_source_vectors(tmp_path):
    args = original(tmp_path, empty=True)
    prepare(args)
    result = prep.initial_context(state=args[2])
    assert result["disposition"] == "no_program_inputs"
    assert result["index_id"] is None
    assert result["indexed_symbols"] == result["full_capsules"] == result["embedding_calls"] == 0
    assert result["learned_embeddings"] is result["completion_authority"] is False
    assert not (args[0] / ".runtime/terminal-vectors/vectors.duckdb").exists()
    assert (args[2] / "initial-context-result.json").exists()


def test_native_modify_only_profile_remains_exact(tmp_path):
    args = original(tmp_path)
    args[3]["outputs"] = [{"path": "source.py", "effect": "modify", "media_type": "text/x-python"}]
    result = prepare(args)
    assert prep._load_prepared(args[2]) == result
    assert "created_outputs" not in result["manifest"]["payload"]


def test_oversize_planner_source_is_rejected_without_partial_scan(tmp_path):
    args = original(tmp_path)
    (args[0] / "source.py").write_bytes(b"#" * 262145)
    with pytest.raises(ValueError, match="complete planner scan byte bound"):
        prepare(args)
    assert not args[2].exists()


def test_source_change_during_domain_declaration_cannot_replace_original_inventory(tmp_path, monkeypatch):
    args = original(tmp_path)
    root, _, state, _ = args
    native_declarations = prep.local.local_planning_domain_declarations

    def changed_source(**kwargs):
        (root / "source.py").write_text("def public_source():\n    return 99\n")
        return native_declarations(**kwargs)

    monkeypatch.setattr(prep.local, "local_planning_domain_declarations", changed_source)
    with pytest.raises(ValueError, match="public source bytes changed during task preparation"):
        prepare(args)
    assert not (state / "prepared.json").exists()
    assert not (state / "provider-request.json").exists()


def test_actual_generic_source_indexes_and_world_bind_task_inputs(tmp_path):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    args = original(tmp_path)
    prepared = prepare(args)
    result = prep.initial_context(state=args[2])
    loaded = initial.load_initial_context(state=args[2], prepared=prepared, require_empty_owner=True)
    assert result["indexed_symbols"] == 1
    assert set(loaded["retrieval"]["source_sha256"]) == {"source.py"}
    assert result["public_task_index_scope"] == {"task_source_paths": ["source.py"],
        "instruction_only": False, "source_semantics_verified": False}
    assert result["world_task_count"] == 0
    assert result["provider_calls"] == 0


@pytest.mark.parametrize("mixed,profile_digest,smoke_digest,spec_digest", [
    (True, "8c3a9518fb8ed888bb24fc1f23e48deaa8b79b13514819c9cdea2ff754bdec47",
     "488e32462a4d2bf595b50bccc37cf74f1be351461b7ee2ab040f74527fdd8549",
     "2df2f197bf9187bf708b12124a03bcb8ac3d18266e1029708fbcf38a2ebb55c9"),
    (False, "93a47f93d1aa4a4de018f403328196f75232080c1599acf238353ee2d394e42b",
     "779b97b174ae07b680dbe08e3a9e2ca00ed8e52caf0ba3df742399d58aa71301",
     "e9015b0824cd21ed1b82f04f4d2046fa7dd9822422ba0b7e0e7bb090a5f76f95"),
])
def test_version_one_bytes_preserve_frozen_published_baseline(mixed, profile_digest, smoke_digest, spec_digest):
    # Golden bytes were generated from the unchanged 7dda779c9 owner, before
    # this profile increment, including its fixed structural acceptance text.
    declared = profile(inputs=("source.py",) if mixed else ())
    if mixed:
        declared["outputs"].insert(0, {"path": "source.py", "effect": "modify", "media_type": "text/x-python"})
    assert hashlib.sha256(owner.task_profile_bytes(declared)).hexdigest() == profile_digest
    assert hashlib.sha256(owner.task_profile_smoke(declared).encode()).hexdigest() == smoke_digest
    spec = owner.task_profile_spec(declared, policy_cid="fixture")
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == spec_digest
    assert owner.task_profile_specs(declared, policy_cid="fixture") == [spec]
    assert set(owner.validate_task_profile(declared)) == {"schema", "instruction_sha256", "input_paths", "outputs"}


def multitask_profile(count=2):
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
    declared = profile(inputs=())
    declared.update(schema=owner.MULTITASK_SCHEMA,
                    intent_requirement_contract_cid=cid_for_dag_json({"reviewed": "structural-fixture"}),
                    outputs=[], tasks=[])
    for index in range(count):
        path, key = f"result_{index:02d}.py", f"task:{index:02d}"
        declared["outputs"].append({"path": path, "effect": "create", "media_type": "text/x-python"})
        declared["tasks"].append({"task_key": key, "operation_id": f"operation:{index:02d}",
            "output_paths": [path], "dependencies": [], "validation_key": f"check:{index:02d}",
            "criterion_key": f"criterion:{index:02d}"})
    return declared


@pytest.mark.parametrize("count", [2, 16])
def test_multi_task_profile_canonicalizes_bounded_population(count):
    declared = multitask_profile(count)
    declared["tasks"].reverse()
    declared["outputs"].reverse()
    declared["tasks"][0]["dependencies"] = ["task:00"]
    canonical = owner.validate_task_profile(declared)
    assert canonical == owner.validate_task_profile(canonical)
    assert canonical["tasks"][0]["task_key"] == "task:00"
    assert len(owner.task_profile_specs(canonical, policy_cid="fixture")) == count
    assert owner.task_profile_bytes(declared) == owner.task_profile_bytes(canonical)


@pytest.mark.parametrize("mutation", ["one_task", "too_many", "unknown_top", "unknown_task", "empty_outputs",
    "duplicate_ownership", "unowned_output", "extra_output", "unknown_dependency", "self_dependency",
    "duplicate_dependency", "cycle", "duplicate_task", "duplicate_operation", "duplicate_validation",
    "duplicate_criterion", "bad_task_key", "oversize_key", "invalid_cid", "boolean_cid", "reserved_version"])
def test_multi_task_profile_refuses_ambiguous_or_unbounded_bindings(mutation):
    declared = multitask_profile(17 if mutation == "too_many" else 2)
    if mutation == "one_task": declared["tasks"].pop()
    elif mutation == "unknown_top": declared["completion_authority"] = True
    elif mutation == "unknown_task": declared["tasks"][0]["proof_authority"] = True
    elif mutation == "empty_outputs": declared["tasks"][0]["output_paths"] = []
    elif mutation == "duplicate_ownership": declared["tasks"][1]["output_paths"] = declared["tasks"][0]["output_paths"][:]
    elif mutation == "unowned_output": declared["outputs"].append({"path": "unowned.py", "effect": "create", "media_type": "text/x-python"})
    elif mutation == "extra_output": declared["tasks"][0]["output_paths"].append("extra.py")
    elif mutation == "unknown_dependency": declared["tasks"][0]["dependencies"] = ["task:absent"]
    elif mutation == "self_dependency": declared["tasks"][0]["dependencies"] = ["task:00"]
    elif mutation == "duplicate_dependency": declared["tasks"][1]["dependencies"] = ["task:00", "task:00"]
    elif mutation == "cycle":
        declared["tasks"][0]["dependencies"] = ["task:01"]
        declared["tasks"][1]["dependencies"] = ["task:00"]
    elif mutation.startswith("duplicate_"):
        name = {"duplicate_task": "task_key", "duplicate_operation": "operation_id",
                "duplicate_validation": "validation_key", "duplicate_criterion": "criterion_key"}[mutation]
        declared["tasks"][1][name] = declared["tasks"][0][name]
    elif mutation == "bad_task_key": declared["tasks"][0]["task_key"] = "task with spaces"
    elif mutation == "oversize_key": declared["tasks"][0]["criterion_key"] = "x" * 129
    elif mutation == "invalid_cid": declared["intent_requirement_contract_cid"] = "not-a-cid"
    elif mutation == "boolean_cid": declared["intent_requirement_contract_cid"] = True
    elif mutation == "reserved_version": declared["schema"] = "terminal-public-task-profile@2"
    with pytest.raises(ValueError):
        owner.validate_task_profile(declared)


def test_multi_task_disjoint_ownership_covers_modify_outputs_too():
    declared = multitask_profile()
    declared["input_paths"] = [item["path"] for item in declared["outputs"]]
    for output in declared["outputs"]: output["effect"] = "modify"
    specs = owner.task_profile_specs(declared, policy_cid="fixture")
    assert all(spec["outputs"][0]["effect"] == "modify" for spec in specs)
    assert set(specs[0]["scope_paths"]) == {owner.INSTRUCTION, owner.PROFILE, owner.SMOKE,
                                           "result_00.py", "result_01.py"}
    declared["tasks"][1]["output_paths"] = declared["tasks"][0]["output_paths"][:]
    with pytest.raises(ValueError, match="disjoint"):
        owner.validate_task_profile(declared)


def test_multi_task_specs_keep_closed_per_task_structural_checks():
    declared = multitask_profile()
    specs = owner.task_profile_specs(declared, policy_cid="fixture")
    assert set(specs[0]) == {"task_key", "scope_paths", "dependencies", "outputs", "validations", "acceptance"}
    assert "result_01.py" not in specs[0]["scope_paths"]
    assert specs[0]["validations"] == [{"validation_key": "check:00",
        "argv": ["python3", "-I", "-B", owner.SMOKE, "--task", "task:00"],
        "cwd": ".", "expected_exit_codes": [0], "policy_cid": "fixture"}]
    assert specs[0]["acceptance"][0]["evidence_cids"] == []
    assert specs[0]["acceptance"][0]["validation_keys"] == ["check:00"]
    assert specs[0]["acceptance"][0]["criterion"].endswith("benchmark correctness remains unverified")
    with pytest.raises(ValueError, match="singleton"):
        owner.task_profile_spec(declared, policy_cid="fixture")


@pytest.mark.parametrize("selector", [[], ["--task"], ["--task", "missing"],
    ["task:00"], ["--task", "task:00", "extra"], ["--task=task:00"]])
def test_multi_task_smoke_refuses_missing_or_nonexact_task_selector(tmp_path, selector):
    declared = multitask_profile()
    for output in declared["outputs"]: (tmp_path / output["path"]).write_text("pass\n")
    (tmp_path / owner.SMOKE).write_text(owner.task_profile_smoke(declared))
    result = subprocess.run(["python3", "-I", "-B", owner.SMOKE, *selector], cwd=tmp_path,
                            capture_output=True, timeout=10)
    assert result.returncode != 0


def test_multi_task_smoke_checks_only_selected_outputs_without_candidate_execution(tmp_path):
    declared = multitask_profile()
    (tmp_path / "result_00.py").write_text("open('EXECUTED','w').write('candidate')\nraise RuntimeError('candidate')\n")
    (tmp_path / owner.SMOKE).write_text(owner.task_profile_smoke(declared))
    specs = owner.task_profile_specs(declared, policy_cid="fixture")
    selected = subprocess.run(specs[0]["validations"][0]["argv"], cwd=tmp_path, capture_output=True, timeout=10)
    missing = subprocess.run(specs[1]["validations"][0]["argv"], cwd=tmp_path, capture_output=True, timeout=10)
    assert selected.returncode == 0 and missing.returncode != 0
    assert not (tmp_path / "EXECUTED").exists()


def reviewed_multi_task_profile(*, ordered=False):
    from test.api.test_intent_requirement_adapter import prepared
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import validate_intent_requirement_contract
    contract, _ = prepared(two=True, ordered=ordered)
    contract["source_path"] = owner.INSTRUCTION
    contract = validate_intent_requirement_contract(contract)
    source = "".join(unit["text"] for unit in contract["ledger"]["source_units"])
    outputs = [deepcopy(output) for operation in contract["symbolic_operations"]["operations"]
               for output in operation["outputs"]]
    op_tasks = {op["operation_id"]: op["task_key"] for op in contract["symbolic_operations"]["operations"]}
    declared = {"schema": owner.MULTITASK_SCHEMA, "instruction_sha256": owner.instruction_sha256(source),
        "input_paths": [item["path"] for item in outputs], "outputs": outputs,
        "intent_requirement_contract_cid": cid_for_dag_json(contract), "tasks": []}
    for index, operation in enumerate(contract["symbolic_operations"]["operations"]):
        declared["tasks"].append({"task_key": operation["task_key"], "operation_id": operation["operation_id"],
            "output_paths": [item["path"] for item in operation["outputs"]],
            "dependencies": [op_tasks[key] for key in operation["dependency_operation_ids"]],
            "validation_key": operation["validation_keys"][0], "criterion_key": f"criterion:{index}"})
    return source, contract, declared


@pytest.mark.parametrize("ordered", [False, True])
def test_multi_task_contract_joins_exact_atomic_reviewed_operations(ordered):
    source, contract, declared = reviewed_multi_task_profile(ordered=ordered)
    assert owner.validate_task_profile_contract(declared, contract, instruction=source) == contract
    assert owner.validate_task_profile_contract(declared, contract) == contract


@pytest.mark.parametrize("contract", [None, [], "reviewed-contract", True])
def test_multi_task_contract_requires_an_actual_reviewed_object(contract):
    with pytest.raises(ValueError, match="reviewed requirement contract"):
        owner.validate_task_profile_contract(multitask_profile(), contract)


@pytest.mark.parametrize("mutation", ["wrong_contract", "wrong_source_path", "wrong_instruction", "wrong_digest",
    "wrong_operation", "wrong_task", "wrong_output", "wrong_validation", "missing_dependency", "extra_dependency",
    "extra_operation_dependency", "schema_one"])
def test_multi_task_contract_refuses_identity_effect_or_ordering_drift(mutation):
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
    source, contract, declared = reviewed_multi_task_profile(ordered=mutation == "missing_dependency")
    if mutation == "wrong_contract": declared["intent_requirement_contract_cid"] = cid_for_dag_json({"different": True})
    elif mutation == "wrong_source_path": contract["source_path"] = "other.md"
    elif mutation == "wrong_instruction": source += "Extra instruction."
    elif mutation == "wrong_digest": declared["instruction_sha256"] = "0" * 64
    elif mutation == "wrong_operation": declared["tasks"][0]["operation_id"] = "operation:absent"
    elif mutation == "wrong_task": declared["tasks"][0]["task_key"] = "task:other"
    elif mutation == "wrong_output": declared["outputs"][0]["media_type"] = "text/plain"
    elif mutation == "wrong_validation": declared["tasks"][0]["validation_key"] = "check:other"
    elif mutation == "missing_dependency": declared["tasks"][1]["dependencies"] = []
    elif mutation == "extra_dependency": declared["tasks"][1]["dependencies"] = [declared["tasks"][0]["task_key"]]
    elif mutation == "extra_operation_dependency":
        operations = contract["symbolic_operations"]["operations"]
        operations[1]["dependency_operation_ids"] = [operations[0]["operation_id"]]
        declared["tasks"][1]["dependencies"] = [declared["tasks"][0]["task_key"]]
    elif mutation == "schema_one":
        contract["schema"] = "intent-plan-requirement-contract@1"
        del contract["symbolic_operations"]
    if mutation in {"wrong_source_path", "extra_operation_dependency", "schema_one"}:
        declared["intent_requirement_contract_cid"] = cid_for_dag_json(contract)
    with pytest.raises(ValueError):
        owner.validate_task_profile_contract(declared, contract, instruction=source)

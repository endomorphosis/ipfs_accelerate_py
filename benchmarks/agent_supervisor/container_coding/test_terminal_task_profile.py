"""Generic public declarations through native Git, signing, indexing and smoke."""
import os
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles


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

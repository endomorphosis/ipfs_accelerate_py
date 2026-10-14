"""Real signed mixed-code/data preparation, partition replay and format gates."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_task_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original, prepare, git, smoke
from ipfs_accelerate_py.agent_supervisor.runtime import source384_program_scope as scope_owner
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_source_partition import terminal_profile_partition


def test_version_one_smoke_retains_published_producer_bytes():
    profile = dict(schema=profiles.SCHEMA, instruction_sha256=profiles.instruction_sha256("fixture"),
        input_paths=["source.py"], outputs=[dict(path="result.py", effect="create", media_type="text/x-python")])
    assert hashlib.sha256(profiles.task_profile_smoke(profile).encode()).hexdigest() == (
        "779b97b174ae07b680dbe08e3a9e2ca00ed8e52caf0ba3df742399d58aa71301")


def mixed(tmp_path, *, media="application/xml", name="config.xml", raw=b"<config/>\n", empty=False):
    args = original(tmp_path, empty=empty)
    root, _, _, profile = args
    (root / name).write_bytes(raw)
    git(root, "add", "--", name)
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "public data")
    profile.update(schema=profiles.DATA_SCHEMA, data_inputs=[dict(path=name, media_type=media)])
    profile["input_paths"].append(name)
    return args


@pytest.mark.parametrize("media,raw", [
    ("application/json", b'{"key":1}'), ("application/x-ndjson", b'{"key":1}\n{"key":2}\n'),
    ("application/xml", b'<config name="test"/>'),
    ("text/calendar", b'BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:fixture\nBEGIN:VEVENT\nEND:VEVENT\nEND:VCALENDAR\n')])
def test_closed_data_formats_are_nonexecuting(media, raw):
    profiles.validate_task_data(raw, media)


@pytest.mark.parametrize("media,raw", [
    ("application/json", b'{"x":1,"x":2}'), ("application/json", b'{"x":NaN}'),
    ("application/json", b'{"x":1e9999}'), ("application/json", b'not json'),
    ("application/x-ndjson", b'{}\n\n{}'), ("application/x-ndjson", b'[]'),
    ("application/x-ndjson", b''), ("application/xml", b'<broken>'),
    ("application/xml", b'<!DOCTYPE a [<!ENTITY x SYSTEM "file:///etc/passwd">]><a>&x;</a>'),
    ("text/calendar", b'BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:x\nBEGIN:VEVENT\nEND:VCALENDAR'),
    ("text/calendar", b'BEGIN:VCALENDAR\nEND:VCALENDAR')])
def test_bad_data_never_becomes_accepted_semantic_input(media, raw):
    with pytest.raises(ValueError):
        profiles.validate_task_data(raw, media)


@pytest.mark.parametrize("damage", ["code_suffix", "unknown_media", "absent", "duplicate", "mutable", "extra"])
def test_data_classification_is_closed_and_immutable(tmp_path, damage):
    args = mixed(tmp_path)
    profile = args[3]
    if damage == "code_suffix": profile["data_inputs"][0]["path"] = "source.py"
    elif damage == "unknown_media": profile["data_inputs"][0]["media_type"] = "text/plain"
    elif damage == "absent": profile["data_inputs"][0]["path"] = "missing.xml"
    elif damage == "duplicate": profile["data_inputs"] *= 2
    elif damage == "mutable": profile["outputs"] = [dict(path="config.xml", media_type="application/xml", effect="modify")]
    elif damage == "extra": profile["data_inputs"][0]["proof_authority"] = True
    with pytest.raises(ValueError):
        profiles.validate_task_profile(profile)


def test_real_prepare_initial_context_preserves_complete_mixed_input_ledger(tmp_path):
    args = mixed(tmp_path)
    root, _, state, _ = args
    prepared = prepare(args)
    partition = terminal_profile_partition(repository=root, manifest=prepared["manifest"]["payload"])
    assert partition.program_paths == ("source.py",)
    assert ("config.xml", "task_data", hashlib.sha256(b"<config/>\n").hexdigest()) in partition.support_hashes
    result = prep.initial_context(state=state)
    assert result["indexed_symbols"] == 1
    assert result["provider_calls"] == 0
    assert "config.xml" in prepared["worker_inputs"]
    assert set(prepared["manifest"]["payload"]["sources"]) == set(prepared["worker_inputs"])
    assert (root / "config.xml").read_bytes() == b"<config/>\n"


def test_data_only_inputs_have_no_fabricated_program_symbols(tmp_path):
    args = mixed(tmp_path, empty=True)
    prepare(args)
    result = prep.initial_context(state=args[2])
    assert result["disposition"] == "no_program_inputs"
    assert result["indexed_symbols"] == result["full_capsules"] == result["embedding_calls"] == 0
    assert result["provider_calls"] == 0


def test_partition_does_not_impose_vector_limit_on_other_consumers(tmp_path):
    args = mixed(tmp_path)
    root, _, _, profile = args
    for index in range(64):
        name = f"source_{index:02d}.py"
        (root / name).write_text("def function():\n    return 1\n")
        profile["input_paths"].append(name)
    git(root, "add", "--all")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "many public sources")
    prepared = prepare(args)
    partition = terminal_profile_partition(repository=root, manifest=prepared["manifest"]["payload"])
    assert len(partition.program_paths) == 65
    with pytest.raises(ValueError, match="64-file vector bound"):
        profiles.task_profile_index_paths(profile)


@pytest.mark.parametrize("path", ["config.xml", "source.py"])
def test_any_signed_input_change_refuses_current_source384_scope(tmp_path, path):
    args = mixed(tmp_path)
    prepared = prepare(args)
    hashes = {name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()}
    (args[0] / path).write_bytes((args[0] / path).read_bytes() + b"\n")
    with pytest.raises(ValueError):
        scope_owner.program_scope(repository=args[0], source_hashes=hashes, envelope=prepared["manifest"], current=True)


def test_source384_historical_data_scope_replays_signed_profile_and_all_hashes(tmp_path):
    args = mixed(tmp_path)
    prepared = prepare(args)
    hashes = {name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()}
    scope = scope_owner.program_scope(repository=args[0], source_hashes=hashes, envelope=prepared["manifest"], current=True)
    assert scope["program_paths"] == ["source.py"]
    assert len(scope["harness_support"]) == 3
    assert [item["path"] for item in scope["task_data"]] == ["config.xml"]
    assert scope_owner.recorded_scope(source_hashes=hashes, envelope=prepared["manifest"], task_profile=scope["task_profile"]) == scope
    forged = deepcopy(scope["task_profile"])
    forged["data_inputs"] = []
    with pytest.raises(ValueError, match="signed profile"):
        scope_owner.recorded_scope(source_hashes=hashes, envelope=prepared["manifest"], task_profile=forged)
    hashes["config.xml"] = "f" * 64
    with pytest.raises(ValueError, match="support hashes changed"):
        scope_owner.recorded_scope(source_hashes=hashes, envelope=prepared["manifest"], task_profile=scope["task_profile"])


@pytest.mark.parametrize("media,path,valid,invalid", [
    ("application/json", "result.json", b'{"ok":1}', b'{"x":NaN}'),
    ("application/x-ndjson", "result.jsonl", b'{"ok":1}\n', b'[]\n'),
    ("application/xml", "result.xml", b'<root/>', b'<!DOCTYPE root><root/>')])
def test_versioned_smoke_checks_selected_formats_without_executing_candidate(tmp_path, media, path, valid, invalid):
    profile = dict(schema=profiles.DATA_SCHEMA, instruction_sha256=profiles.instruction_sha256("Fixture"),
        input_paths=[], data_inputs=[], outputs=[dict(path=path, effect="create", media_type=media)])
    (tmp_path / path).write_bytes(valid)
    assert smoke(tmp_path, profile).returncode == 0
    (tmp_path / path).write_bytes(invalid)
    assert smoke(tmp_path, profile).returncode != 0


def test_symbolic_capabilities_distinguish_data_from_generated_support(tmp_path):
    from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_capabilities import inputs
    from benchmarks.agent_supervisor.container_coding.terminal_symbolic_capabilities import assess_terminal_symbolic_capabilities
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    args = inputs()
    profile = dict(schema=profiles.DATA_SCHEMA, instruction_sha256=profiles.instruction_sha256("Fixture"),
        input_paths=["source.py", "config.xml"], data_inputs=[dict(path="config.xml", media_type="application/xml")],
        outputs=args["task_spec"]["outputs"])
    sources = args["manifest"]["payload"]["sources"]
    sources["config.xml"] = dict(sha256="a" * 64, executable=False)
    sources[profiles.PROFILE]["sha256"] = hashlib.sha256(profiles.task_profile_bytes(profile)).hexdigest()
    report = args["doctor_result"]
    report["manifest_cid"] = content_identity(args["manifest"])
    report["source_hashes"] = {name: row["sha256"] for name, row in sources.items()}
    report["reason_codes"] = ["doctor_task_data_contract_unavailable"]
    part = report["source_partition"]
    part.update(task_profile=profiles.validate_task_profile(profile), manifest_cid=report["manifest_cid"],
        profile_sha256=sources[profiles.PROFILE]["sha256"])
    for row in part["harness_support"]:
        row["sha256"] = sources[row["path"]]["sha256"]
    part["harness_support"].append(dict(path="config.xml", role="task_data", sha256="a" * 64))
    part["partition_cid"] = content_identity({key:value for key,value in part.items() if key != "partition_cid"})
    result = assess_terminal_symbolic_capabilities(**args)
    assert result["inventory"]["partition"]["harness_support_count"] == 3
    assert result["inventory"]["partition"]["task_data_input_count"] == 1
    assert "task_data_semantic_contract_unavailable" in result["gap_codes"]
    assert result["operators"]["known_residual_reasons"] == ["doctor_task_data_contract_unavailable"]
    assert result["proof"]["whole_program_verified"] is False

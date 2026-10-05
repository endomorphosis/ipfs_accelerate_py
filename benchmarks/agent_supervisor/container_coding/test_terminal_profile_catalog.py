"""Reviewed catalog bindings reject drift before constructing a worker checkout."""
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_profile_catalog as catalog


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    task = dataset / "public-task"
    (task / "environment").mkdir(parents=True)
    contents = {"instruction.md": b"Write result.xml from public input.\n",
        "task.toml": b"[agent]\ntimeout_sec=900\n",
        "environment/Dockerfile": b"FROM python:3.13-slim\nWORKDIR /app\nCOPY config.xml /app/\n",
        "environment/config.xml": b"<configuration/>\n"}
    for name, raw in contents.items():
        (task / name).write_bytes(raw)
    metadata = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()
        if name != "environment/config.xml"}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(dict(schema="terminal-reviewed-public-profile-catalog@1",
        dataset_commit="fixture", tasks={"public-task": dict(public_metadata=metadata,
        inputs=[dict(environment_path="config.xml", path="config.xml", media_type="application/xml",
            sha256=hashlib.sha256(contents["environment/config.xml"]).hexdigest())],
        outputs=[dict(path="result.xml", effect="create", media_type="application/xml")])})))
    monkeypatch.setattr(catalog, "CATALOG", path)
    return dataset, task


def test_catalog_records_exact_public_inputs_without_reading_verifier_or_solution(reviewed):
    dataset, task = reviewed
    for directory in ("tests", "solution"):
        (task / directory).mkdir()
        (task / directory / "not-readable").symlink_to("/not/a/real/file")
    selected = catalog.reviewed_profile(dataset=dataset, task_name="public-task")
    assert selected["profile"]["input_paths"] == ["config.xml"]
    assert selected["native_agent_seconds"] == 900
    assert selected["official_reward"] is None
    assert selected["source384_inference_qualified"] is False
    assert selected["hidden_verifier_or_solution_read"] is False


@pytest.mark.parametrize("name", ["instruction.md", "task.toml", "environment/Dockerfile", "environment/config.xml"])
def test_any_reviewed_public_input_change_is_rejected_before_output(reviewed, tmp_path, name):
    dataset, task = reviewed
    with (task / name).open("ab") as stream:
        stream.write(b"\n")
    output = tmp_path / "qualification"
    with pytest.raises(ValueError, match="changed"):
        catalog.qualify_public_profile(dataset=dataset, task_name="public-task", output=output)
    assert not output.exists()


def test_public_input_symlinks_are_not_accepted(reviewed):
    dataset, task = reviewed
    path = task / "environment/config.xml"
    raw = path.read_bytes()
    path.unlink()
    target = task / "elsewhere.xml"
    target.write_bytes(raw)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="canonical regular"):
        catalog.reviewed_profile(dataset=dataset, task_name="public-task")


def test_unknown_task_cannot_be_inferred_from_filename(reviewed):
    with pytest.raises(ValueError, match="no reviewed public profile"):
        catalog.reviewed_profile(dataset=reviewed[0], task_name="../private")


def test_builtin_catalog_is_bounded_explicit_and_contains_three_new_profiles():
    value = json.loads(catalog.CATALOG.read_text())
    assert set(value["tasks"]) == {"tune-mjcf", "llm-inference-batching-scheduler", "constraints-scheduling"}
    for entry in value["tasks"].values():
        assert set(entry["public_metadata"]) == {"instruction.md", "environment/Dockerfile", "task.toml"}
        assert entry["outputs"]
        assert all(item["effect"] == "create" for item in entry["outputs"])
        assert all(len(item["sha256"]) == 64 for item in entry["inputs"])

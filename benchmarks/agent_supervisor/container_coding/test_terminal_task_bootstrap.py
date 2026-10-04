"""Real isolated Git baseline controls; no Docker, provider or package install."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_task_bootstrap import bootstrap_script


def profile(inputs):
    return {"schema": "terminal-public-task-profile@1", "instruction_sha256": "a" * 64,
            "input_paths": inputs, "outputs": [
                {"path": "answer.py", "effect": "create", "media_type": "text/x-python"}]}


def execute(root, selected):
    # The production script only accepts /app; bind its literal to our fixture.
    script = bootstrap_script(selected).replace("pathlib.Path('/app')", "pathlib.Path(" + repr(str(root)) + ")")
    return subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("has_source", [False, True])
def test_new_baseline_retains_exact_original_bytes_and_reopens(tmp_path, has_source):
    selected = profile(["base.py"] if has_source else [])
    if has_source:
        (tmp_path / "base.py").write_bytes(b"def base():\n    return 42\n")
    result = execute(tmp_path, selected)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["git_initialized"] and receipt["original_source_bytes_preserved"]
    if has_source:
        assert receipt["source_files"]["base.py"]["sha256"] == hashlib.sha256((tmp_path / "base.py").read_bytes()).hexdigest()
    reopened = execute(tmp_path, selected)
    assert reopened.returncode == 0, reopened.stderr
    again = json.loads(reopened.stdout)
    assert not again["git_initialized"]
    assert again["source_files"] == receipt["source_files"] and again["head"] == receipt["head"]


@pytest.mark.parametrize("kind", ["undeclared", "missing", "linked_file", "linked_directory", "git_link", "hardlink"])
def test_invalid_original_population_refused_before_git_initialization(tmp_path, kind):
    root = tmp_path / "task"
    root.mkdir()
    other = tmp_path / "private"
    other.write_text("outside data")
    selected = profile(["base.py"])
    if kind == "undeclared":
        (root / "base.py").write_text("x=1")
        (root / "unexpected.txt").write_text("unlisted")
    elif kind == "linked_file":
        (root / "base.py").symlink_to(other)
    elif kind == "linked_directory":
        (root / "nested").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "git_link":
        (root / ".git").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "hardlink":
        (root / "base.py").hardlink_to(other)
    result = execute(root, selected)
    assert result.returncode != 0
    if kind != "git_link":
        assert not (root / ".git").exists()
    assert other.read_text() == "outside data"

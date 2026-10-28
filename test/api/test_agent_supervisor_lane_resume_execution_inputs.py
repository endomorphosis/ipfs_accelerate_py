"""Explicit imported/configuration inputs rearm only repaired task contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume


pytestmark = pytest.mark.proof_reuse_disabled(reason="Exercise real imported-helper and configuration repairs")


@pytest.fixture(autouse=True)
def _one_failure_before_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane_resume, "_ATTEMPT_LIMIT", 1)


def _task(root: Path, *, inputs: str = "helper.py, settings.json", producer: str | None = None) -> None:
    (root / "helper.py").write_text("def ready(): return False\n")
    (root / "settings.json").write_text('{"enabled":true}\n')
    (root / "produce.py").write_text(
        "from pathlib import Path\n"
        "counter=Path('producer-count')\n"
        "counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n"
        + (producer if producer is not None else
           "import sys,json\nsys.dont_write_bytecode=True\nfrom helper import ready\n"
           "if not ready() or not json.loads(Path('settings.json').read_text())['enabled']: raise SystemExit(1)\n"
           "Path('report.json').write_text('{\"decision\":\"GO\"}')\n"),
        encoding="utf-8",
    )
    (root / "validate.py").write_text(
        "from pathlib import Path\nimport json\nPath('validator-ran').touch()\n"
        "assert json.loads(Path('report.json').read_text())['decision']=='GO'\n",
        encoding="utf-8",
    )
    (root / "board.md").write_text(
        "## TEST-001 Imported implementation\n- Status: ready\n- Depends on:\n"
        "- Deterministic command: python3 produce.py\n- Validation: python3 validate.py\n"
        f"- Execution inputs: {inputs}\n- Outputs: report.json\n"
        "- Acceptance: Genuine output from the current imported implementation\n",
        encoding="utf-8",
    )


def _cycle(root: Path) -> dict[str, Any]:
    return lane_resume.run_resume_cycle(
        repo_root=root, todo_path=root / "board.md", task_header_prefix="## TEST-",
        attempt_path=root / "attempts.json", log_path=root / "lane.log",
        heartbeat=lambda **_kwargs: None,
    )


def _assert_waiting(root: Path) -> None:
    result = _cycle(root)
    assert result["phase"] == "attempt_retry_wait"
    assert result["work_complete"] is False
    assert result["completion_authority"] is False
    assert "- Status: ready" in (root / "board.md").read_text()


@pytest.mark.parametrize("declaration", [None, "", "   "])
def test_no_declared_execution_inputs_retains_the_exact_legacy_fingerprint(
    tmp_path: Path, declaration: str | None,
) -> None:
    for name in ["produce.py", "validate.py"]:
        (tmp_path / name).write_text("raise SystemExit(1)\n")
    fields = {"deterministic_command": "python3 produce.py", "validation": "python3 validate.py", "outputs": "report.json"}
    if declaration is not None:
        fields["execution_inputs"] = declaration
    digest = lane_resume._command_contract(
        fields, ["python3", "produce.py"], ["python3", "validate.py"], repo_root=tmp_path,
    )
    # Captured from the prior direct-script-only implementation. A new empty
    # payload key would silently invalidate every historical retry contract.
    assert digest == "9b890ad72ea4406faf1c79f43254a31737c7b5aa589cb8a10f8d3a7bea6b32b5"


def test_repaired_imported_helper_rearms_exhausted_task_and_completes(tmp_path: Path) -> None:
    _task(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)
    producer_before = (tmp_path / "produce.py").read_bytes()
    (tmp_path / "helper.py").write_text("def ready(): return True\n")

    result = _cycle(tmp_path)

    assert result["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "producer-count").read_text() == "2"
    assert (tmp_path / "produce.py").read_bytes() == producer_before
    assert (tmp_path / "validator-ran").exists()
    assert (tmp_path / "lane.log").read_text().count("retry rearmed TEST-001:") == 1
    assert result["completion_authority"] is False


def test_repaired_declared_json_configuration_rearms_without_changing_the_worker(tmp_path: Path) -> None:
    _task(tmp_path)
    (tmp_path / "helper.py").write_text("def ready(): return True\n")
    (tmp_path / "settings.json").write_text('{"enabled":false}\n')
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)

    (tmp_path / "settings.json").write_text('{"enabled":true}\n')

    assert _cycle(tmp_path)["advanced_task_id"] == "TEST-001"
    assert (tmp_path / "producer-count").read_text() == "2"


def test_timestamp_only_input_change_preserves_the_existing_cooldown(tmp_path: Path) -> None:
    _task(tmp_path)
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    before = json.loads((tmp_path / "attempts.json").read_text())
    helper = tmp_path / "helper.py"
    stamp = helper.stat()
    os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))

    _assert_waiting(tmp_path)

    after = json.loads((tmp_path / "attempts.json").read_text())
    assert after["_retry_after_ms"] == before["_retry_after_ms"]
    assert (tmp_path / "producer-count").read_text() == "1"
    assert "retry rearmed" not in (tmp_path / "lane.log").read_text()


def test_missing_declared_file_is_bound_as_none_and_its_creation_rearms_only_once(tmp_path: Path) -> None:
    _task(tmp_path, inputs="future-settings.json", producer="raise SystemExit(1)\n")
    binding = lane_resume._execution_input_contract("future-settings.json", repo_root=tmp_path)
    assert binding["future-settings.json"]["sha256"] is None
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)

    (tmp_path / "future-settings.json").write_text('{"fixed":true}\n')
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)

    assert (tmp_path / "producer-count").read_text() == "2"
    assert (tmp_path / "lane.log").read_text().count("retry rearmed TEST-001:") == 1


def test_changed_import_cannot_complete_using_old_green_evidence(tmp_path: Path) -> None:
    _task(tmp_path, producer="Path('report.json').write_text('{\"decision\":\"GO\"}')\nraise SystemExit(1)\n")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    # The first failed attempt made a fresh artifact. Changing its declared
    # imported input invalidates that invocation-start proof before validation.
    (tmp_path / "helper.py").write_text("def ready(): return True\n")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)

    assert (tmp_path / "producer-count").read_text() == "2"
    assert not (tmp_path / "validator-ran").exists()
    assert json.loads((tmp_path / "report.json").read_text())["decision"] == "GO"


def test_producer_written_input_does_not_renew_its_own_budget(tmp_path: Path) -> None:
    _task(tmp_path, inputs="helper.py", producer="Path('helper.py').write_text('generated version '+counter.read_text())\nraise SystemExit(1)\n")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)
    assert (tmp_path / "producer-count").read_text() == "1"

    (tmp_path / "helper.py").write_text("external repaired version")
    assert _cycle(tmp_path)["phase"] == "deterministic_command_failed"
    _assert_waiting(tmp_path)
    assert (tmp_path / "producer-count").read_text() == "2"


@pytest.mark.parametrize("kind", ["absolute", "parent", "symlink", "directory", "null"])
def test_invalid_inputs_are_recorded_without_opening_outside_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    secret = tmp_path / "outside-private.json"
    secret.write_text('{"private":"do not read"}')
    if kind == "symlink":
        (root / "imported.json").symlink_to(secret)
    names = {"absolute": str(secret), "parent": "../outside-private.json", "symlink": "imported.json", "directory": ".", "null": "bad\x00path"}

    def unexpected(_path: Path) -> Any:
        raise AssertionError("invalid input reached the content reader")

    monkeypatch.setattr(lane_resume, "_file_digest", unexpected)
    record = lane_resume._execution_input_contract(names[kind], repo_root=root)[names[kind]]

    assert record["sha256"] is None
    assert record["invalid"]


def test_outside_file_changes_do_not_rearm_an_exhausted_task(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"revision":1}')
    _task(root, inputs="../outside.json", producer="raise SystemExit(1)\n")
    assert _cycle(root)["phase"] == "deterministic_command_failed"

    outside.write_text('{"revision":2}')
    _assert_waiting(root)

    assert (root / "producer-count").read_text() == "1"

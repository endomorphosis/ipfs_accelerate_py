"""Deterministic tests for the EAAEF-111 supervisor handoff CLI."""

from __future__ import annotations

import argparse
import inspect
import io
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import ExternalHandoffAPI
from ipfs_accelerate_py.cli.supervisor_handoff import (
    CLI_COMMANDS,
    CLI_TO_OPERATION,
    EXIT_AUTHORITY,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    EXIT_USAGE,
    EXIT_WORKER_APPROVAL,
    GROUP,
    build_parser,
    dispatch,
    main,
    parse_argv,
    register_supervisor_handoff_cli,
    supervisor_handoff_cli_discovery_manifest,
)


OPERATOR = "principal:operator"
WORKER = "principal:worker"
REVIEWER = "principal:reviewer"
SESSION = "session:example"
REPO = "repo:example"

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "cli"
    / "supervisor_handoff.py"
)


def _handoff_argv(*extra: str) -> list[str]:
    return [
        "handoff",
        "--principal-id",
        OPERATOR,
        "--worker-principal-id",
        WORKER,
        "--session-id",
        SESSION,
        "--repository-id",
        REPO,
        "--objective-id",
        "objective:handoff",
        "--idempotency-key",
        "idem:cli-1",
        "--detach",
        *extra,
    ]


def _run(
    argv: list[str],
    *,
    api: ExternalHandoffAPI | None = None,
    stdin: str = "",
) -> tuple[int, dict[str, object]]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        argv,
        api=api,
        stdout=stdout,
        stderr=stderr,
        stdin=io.StringIO(stdin),
    )
    text = stdout.getvalue() or stderr.getvalue()
    payload = json.loads(text) if text.strip().startswith("{") else {"raw": text, "ok": False}
    return code, payload


def _start(api: ExternalHandoffAPI) -> dict[str, object]:
    code, payload = _run(_handoff_argv(), api=api)
    assert code == EXIT_SUCCESS
    assert payload["ok"] is True
    assert payload["detached"] is True
    assert payload["run_id"]
    assert payload["authority_id"]
    return payload


def _bound_argv(command: str, started: dict[str, object], *extra: str) -> list[str]:
    argv = [
        command,
        "--principal-id",
        OPERATOR,
        "--worker-principal-id",
        WORKER,
        "--run-id",
        str(started["run_id"]),
        "--session-id",
        SESSION,
    ]
    if command in {"steer", "pause", "resume", "cancel"}:
        argv.extend(["--authority-id", str(started["authority_id"])])
    argv.extend(extra)
    return argv


def test_discovery_manifest_is_closed() -> None:
    manifest = supervisor_handoff_cli_discovery_manifest()
    assert manifest["group"] == GROUP
    assert manifest["shell"] is False
    assert manifest["self_approval"] is False
    assert manifest["detach"] is True
    assert manifest["large_input_references"] is True
    assert manifest["live_quack"] is False
    assert manifest["live_docker"] is False
    assert tuple(manifest["commands"]) == CLI_COMMANDS
    assert "export-result" in manifest["commands"]
    assert "export" in manifest["operations"]


@pytest.mark.parametrize("command", CLI_COMMANDS)
def test_parse_argv_accepts_each_command(command: str) -> None:
    argv = [command, "--principal-id", OPERATOR, "--run-id", "run:example"]
    if command in {"steer", "pause", "resume", "cancel"}:
        argv.extend(["--authority-id", "auth:example"])
    if command == "steer":
        argv.extend(["--instruction", "keep owned files only"])
    if command == "handoff":
        argv = _handoff_argv()
    args = parse_argv(argv)
    assert args.command == command
    assert args.operation == CLI_TO_OPERATION[command]


def test_parse_and_dispatch_use_argv_lists_not_shell() -> None:
    api = ExternalHandoffAPI()
    args = parse_argv(_handoff_argv())
    stdout = io.StringIO()
    code = dispatch(args, api=api, stdout=stdout)
    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "handoff"
    assert payload["operation"] == "handoff"
    assert payload["detached"] is True
    assert payload["shell"] is False


def test_source_has_no_shell_or_live_backends() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "shell = True" not in source
    assert "subprocess" not in source
    assert "os.system" not in source
    module_source = inspect.getsource(inspect.getmodule(main))
    assert "import docker" not in module_source
    assert "import quack" not in module_source
    assert "from docker" not in module_source
    assert "from quack" not in module_source


@pytest.mark.parametrize("command", CLI_COMMANDS)
def test_each_command_dispatches_to_api(command: str) -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    extra: list[str] = []
    if command == "handoff":
        argv = _handoff_argv("--idempotency-key", "idem:cli-1")
        code, payload = _run(argv, api=api)
        assert code == EXIT_SUCCESS
        assert payload["run_id"] == started["run_id"]
        return
    if command == "steer":
        extra = ["--instruction", "narrow the owned-file patch"]
    if command == "approve":
        extra = ["--reviewer-principal-id", REVIEWER]
    if command == "reject":
        extra = ["--reviewer-principal-id", REVIEWER]
    if command == "resume":
        pause_code, _ = _run(_bound_argv("pause", started), api=api)
        assert pause_code == EXIT_SUCCESS
    argv = _bound_argv(command, started, *extra)
    code, payload = _run(argv, api=api)
    assert code == EXIT_SUCCESS, payload
    assert payload["ok"] is True
    assert payload["command"] == command
    assert payload["operation"] == CLI_TO_OPERATION[command]
    assert payload["run_id"] == started["run_id"]
    assert payload["receipt"]["run_id"] == started["run_id"]


def test_detach_returns_identities_without_follow() -> None:
    api = ExternalHandoffAPI()
    code, payload = _run(_handoff_argv(), api=api)
    assert code == EXIT_SUCCESS
    assert payload["detached"] is True
    assert payload["receipt"]["reason_code"] in {"admitted", "idempotent"}
    assert "followed" not in str(payload["receipt"]["reason_code"])
    follow_code, follow_payload = _run(_bound_argv("follow", payload), api=api)
    assert follow_code == EXIT_SUCCESS
    assert follow_payload["receipt"]["reason_code"] == "followed"
    assert follow_payload["detached"] is False


def test_large_input_instruction_file_is_a_reference(tmp_path: Path) -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    body = "keep the patch inside owned files and do not widen authority"
    instruction_file = tmp_path / "instruction.txt"
    instruction_file.write_text(body, encoding="utf-8")
    argv = _bound_argv(
        "steer",
        started,
        "--instruction-file",
        str(instruction_file),
    )
    assert body not in argv
    args = parse_argv(argv)
    assert args.instruction_file == instruction_file
    assert getattr(args, "instruction", "") in {"", None}
    code, payload = _run(argv, api=api)
    assert code == EXIT_SUCCESS
    assert payload["receipt"]["reason_code"] == "steered"


def test_large_input_request_file_reference(tmp_path: Path) -> None:
    api = ExternalHandoffAPI()
    request_file = tmp_path / "handoff.json"
    request_file.write_text(
        json.dumps(
            {
                "principal_id": OPERATOR,
                "worker_principal_id": WORKER,
                "session_id": "session:file",
                "repository_id": REPO,
                "idempotency_key": "idem:file",
            }
        ),
        encoding="utf-8",
    )
    argv = ["handoff", "--request-file", str(request_file), "--detach"]
    args = parse_argv(argv)
    assert args.request_file == request_file
    code, payload = _run(argv, api=api)
    assert code == EXIT_SUCCESS
    assert payload["ok"] is True
    assert payload["detached"] is True
    assert payload["receipt"]["session_id"] == "session:file"


def test_worker_cannot_approve() -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    code, payload = _run(
        _bound_argv(
            "approve",
            started,
            "--reviewer-principal-id",
            WORKER,
        ),
        api=api,
    )
    assert code == EXIT_WORKER_APPROVAL
    assert payload["ok"] is False
    assert payload["reason_code"] == "worker_self_approval"
    assert payload["exit_code"] == EXIT_WORKER_APPROVAL

    caller_is_worker = _run(
        [
            "approve",
            "--principal-id",
            WORKER,
            "--worker-principal-id",
            WORKER,
            "--reviewer-principal-id",
            REVIEWER,
            "--run-id",
            str(started["run_id"]),
        ],
        api=api,
    )
    assert caller_is_worker[0] == EXIT_WORKER_APPROVAL
    assert caller_is_worker[1]["reason_code"] == "worker_self_approval"

    missing = _run(_bound_argv("approve", started), api=api)
    assert missing[0] == EXIT_WORKER_APPROVAL
    assert missing[1]["reason_code"] == "missing_reviewer"


def test_independent_reviewer_may_approve() -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    code, payload = _run(
        _bound_argv("approve", started, "--reviewer-principal-id", REVIEWER),
        api=api,
    )
    assert code == EXIT_SUCCESS
    assert payload["receipt"]["run_status"] == "approved"
    assert payload["receipt"]["reviewer_principal_id"] == REVIEWER
    assert payload["receipt"]["worker_principal_id"] == WORKER
    assert payload["receipt"]["reviewer_principal_id"] != payload["receipt"][
        "worker_principal_id"
    ]


def test_control_requires_matching_authority() -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    other = _run(
        _handoff_argv(
            "--idempotency-key",
            "idem:other",
            "--session-id",
            "session:other",
        ),
        api=api,
    )[1]
    code, payload = _run(
        _bound_argv(
            "cancel",
            started,
            "--authority-id",
            str(other["authority_id"]),
        ),
        api=api,
    )
    assert code == EXIT_AUTHORITY
    assert payload["reason_code"] == "authority_mismatch"


def test_unknown_run_is_not_found() -> None:
    api = ExternalHandoffAPI()
    code, payload = _run(
        [
            "status",
            "--principal-id",
            OPERATOR,
            "--run-id",
            "run:missing",
        ],
        api=api,
    )
    assert code == EXIT_NOT_FOUND
    assert payload["reason_code"] == "unknown_run"


def test_unknown_command_is_usage() -> None:
    code, payload = _run(["not-a-command", "--principal-id", OPERATOR])
    assert code == EXIT_USAGE
    assert payload.get("ok") in {False, None}


def test_export_result_maps_to_export() -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    args = parse_argv(_bound_argv("export-result", started))
    assert args.command == "export-result"
    assert args.operation == "export"
    code, payload = _run(_bound_argv("export-result", started), api=api)
    assert code == EXIT_SUCCESS
    assert payload["operation"] == "export"
    assert payload["export_id"]
    assert payload["receipt"]["export_id"] == payload["export_id"]


def test_follow_uses_cursor_from_argv() -> None:
    api = ExternalHandoffAPI()
    started = _start(api)
    first = _run(_bound_argv("follow", started), api=api)[1]
    event_ids = first["receipt"]["event_ids"]
    assert event_ids
    cursor = event_ids[-1]
    code, payload = _run(_bound_argv("follow", started, "--cursor", cursor), api=api)
    assert code == EXIT_SUCCESS
    assert payload["receipt"]["event_ids"] == []


def test_host_registration_is_parser_only() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group")
    group = register_supervisor_handoff_cli(sub)
    assert GROUP in group.prog
    args = parser.parse_args(
        ["supervisor-handoff", "status", "--principal-id", OPERATOR, "--run-id", "run:x"]
    )
    assert args.command == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["supervisor-handoff", "not-a-command"])


def test_missing_principal_is_usage() -> None:
    code, payload = _run(["status", "--run-id", "run:x"])
    assert code == EXIT_USAGE
    assert payload["reason_code"] == "malformed"

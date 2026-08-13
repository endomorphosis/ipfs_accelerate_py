"""SCG-037 semantic-governor CLI: ten commands, privacy, promotion gates."""

from __future__ import annotations

import importlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_MODULE = "ipfs_accelerate_py.agent_supervisor.semantic_governor.cli"
CLI_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py"
)

REQUIRED_COMMANDS = (
    "audit",
    "shadow",
    "diagnose",
    "expand",
    "calibrate",
    "propose-rules",
    "evaluate-policy",
    "promote-policy",
    "report",
    "dashboard-data",
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


@pytest.fixture
def cli():
    sys.modules.pop(CLI_MODULE, None)
    return importlib.import_module(CLI_MODULE)


def _run(
    cli,
    argv: list[str],
    *,
    apis: dict[str, Any] | None = None,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
    stdin: str | None = None,
) -> tuple[int, dict[str, Any], str]:
    out = io.StringIO()
    err = io.StringIO()
    in_stream = io.StringIO(stdin) if stdin is not None else None
    code = cli.main(
        argv,
        stdout=out,
        stderr=err,
        apis=apis,
        policy_repository=policy_repository,
        promotion_repository=promotion_repository,
        stdin=in_stream,
    )
    text = out.getvalue()
    payload = json.loads(text) if text.strip() else {}
    return code, payload, err.getvalue()


def _ok_fn(label: str, **extra: Any):
    def _handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
        body = {
            "status": "ok",
            "label": label,
            "cid": _cid(label),
            **extra,
        }
        return body

    return _handler


def _apis(**overrides: Any) -> dict[str, Any]:
    base = {
        "evaluate_context_sufficiency": _ok_fn("audit"),
        "create_shadow_plan": _ok_fn("shadow-plan"),
        "compare_shadow_results": _ok_fn("shadow-compare"),
        "diagnose_omission": _ok_fn("diagnose"),
        "plan_context_expansion": _ok_fn("expand-plan"),
        "execute_expansion_loop": _ok_fn("expand-exec"),
        "update_calibration": _ok_fn("calibrate"),
        "propose_rule_change": _ok_fn("propose-rules"),
        "evaluate_rule_candidate": _ok_fn("evaluate-policy"),
        "promote_compression_policy": _ok_fn(
            "promote-policy",
            status="promoted",
            head_mutated=True,
            authorization_cid=_cid("auth"),
        ),
        "build_governor_report": _ok_fn("report", report_cid=_cid("report")),
        "build_dashboard_data": _ok_fn(
            "dashboard-data", report_cid=_cid("report")
        ),
        "audit_task": _ok_fn("audit-runtime"),
        "shadow_task": _ok_fn("shadow-runtime"),
        "expand_audit": _ok_fn("expand-runtime"),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Parser / help / descriptor
# ---------------------------------------------------------------------------


def test_module_exists(cli) -> None:
    assert CLI_PATH.is_file()
    assert cli.CLI_INTERFACE == "SemanticGovernorCLI@1"
    assert cli.CLI_EVIDENCE == "scg/cli@1"
    assert cli.CONSOLE_ENTRY == "semantic-governor"


def test_exact_ten_commands(cli) -> None:
    commands = cli.required_cli_commands()
    assert commands == REQUIRED_COMMANDS
    assert len(commands) == 10
    assert len(set(commands)) == 10


def test_build_parser_exposes_all_plan_commands(cli) -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "semantic-governor" in help_text
    for name in REQUIRED_COMMANDS:
        assert name in help_text


def test_descriptor_declares_interface_and_invariants(cli) -> None:
    desc = cli.semantic_governor_cli_descriptor()
    assert desc["interface"] == "SemanticGovernorCLI@1"
    assert desc["bundle"] == "scg/cli@1"
    assert desc["console_entry"] == "semantic-governor"
    assert desc["evidence"] == "scg/cli@1"
    assert desc["commands"] == list(REQUIRED_COMMANDS)
    assert "promotion_requires_explicit_authorization_and_cas" in desc["invariants"]
    assert "private_raw_source_never_in_output" in desc["invariants"]
    assert "no_implicit_promotion" in desc["invariants"]
    assert desc["exit_codes"]["unavailable"] == 3
    assert desc["exit_codes"]["production_gate"] == 4


def test_subparser_help_is_bounded_and_stable(cli) -> None:
    for command in REQUIRED_COMMANDS:
        code = cli.main([command, "--help"], stdout=io.StringIO(), stderr=io.StringIO())
        assert code == 0


def test_missing_command_is_usage_error(cli) -> None:
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main([], stdout=out, stderr=err)
    assert code == cli.EXIT_USAGE


def test_exit_codes_are_stable_constants(cli) -> None:
    assert cli.EXIT_OK == 0
    assert cli.EXIT_ERROR == 1
    assert cli.EXIT_USAGE == 2
    assert cli.EXIT_UNAVAILABLE == 3
    assert cli.EXIT_PRODUCTION_GATE == 4


# ---------------------------------------------------------------------------
# Cold import / --help: no mutation
# ---------------------------------------------------------------------------


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = CLI_MODULE
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}
    started_threads: list[str] = []
    real_thread_start = threading.Thread.start

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started_threads.append(self.name)
        return real_thread_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    popen_calls: list[Any] = []

    def guarded_popen(*args: Any, **kwargs: Any):
        popen_calls.append((args, kwargs))
        raise AssertionError("cold import must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    socket_mod = importlib.import_module("socket")
    real_socket = socket_mod.socket
    socket_calls: list[Any] = []

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            socket_calls.append((args, kwargs))
            raise AssertionError("cold import must not open sockets")

    monkeypatch.setattr(socket_mod, "socket", GuardedSocket)

    real_run = subprocess.run

    def guarded_run(*args: Any, **kwargs: Any):
        cmd = args[0] if args else kwargs.get("args")
        text = " ".join(str(x) for x in (cmd or ()))
        if "pip" in text or "install" in text:
            raise AssertionError(f"cold import must not install: {text}")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)

    mod = importlib.import_module(module_name)
    assert mod.CLI_INTERFACE == "SemanticGovernorCLI@1"
    assert started_threads == []
    assert popen_calls == []
    assert socket_calls == []

    after_threads = {t.ident for t in threading.enumerate()}
    assert after_threads - before_threads == set()


def test_cold_help_does_not_mutate_environment_or_cwd(
    cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_before = dict(os.environ)
    cwd_before = Path.cwd()
    monkeypatch.chdir(tmp_path)
    before_names = set(os.listdir(tmp_path))

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(["--help"], stdout=out, stderr=err)
    assert code == 0
    text = out.getvalue() + err.getvalue()
    assert "semantic-governor" in text

    assert dict(os.environ) == env_before
    assert Path.cwd() == tmp_path
    assert set(os.listdir(tmp_path)) == before_names
    monkeypatch.chdir(cwd_before)


# ---------------------------------------------------------------------------
# Exact ten commands work (injected APIs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", REQUIRED_COMMANDS)
def test_each_command_returns_deterministic_json(cli, command: str) -> None:
    apis = _apis()
    argv = [command, "--json", "{}"]
    if command == "promote-policy":
        # Authorization + CAS injections required.
        argv = [
            "promote-policy",
            "--authorization",
            _cid("explicit-auth"),
            "--operation-id",
            "op-test-1",
            "--json",
            json.dumps(
                {
                    "candidate": {"candidate_id": "c1"},
                    "evaluation_report": {"verdict": "pass"},
                    "release_qualification": {"promotion_allowed": True},
                }
            ),
        ]
        code, payload, _ = _run(
            cli,
            argv,
            apis=apis,
            policy_repository=MagicMock(name="policy_repo"),
        )
    else:
        code, payload, _ = _run(cli, argv, apis=apis)

    assert code == cli.EXIT_OK
    assert payload["ok"] is True
    assert payload["command"] == command
    assert payload["interface"] == "SemanticGovernorCLI@1"
    assert payload["bundle"] == "scg/cli@1"
    assert payload["evidence"] == "scg/cli@1"
    assert "result" in payload
    assert payload["exit_code"] == 0
    # Deterministic key ordering is enforced by sort_keys=True; re-parse ok.
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_command_to_api_mapping(cli) -> None:
    mapping = cli.CLI_TO_API
    assert mapping["audit"] == "evaluate_context_sufficiency"
    assert mapping["shadow"] == "create_shadow_plan"
    assert mapping["diagnose"] == "diagnose_omission"
    assert mapping["expand"] == "execute_expansion_loop"
    assert mapping["calibrate"] == "update_calibration"
    assert mapping["propose-rules"] == "propose_rule_change"
    assert mapping["evaluate-policy"] == "evaluate_rule_candidate"
    assert mapping["promote-policy"] == "promote_compression_policy"
    assert mapping["report"] == "build_governor_report"
    assert mapping["dashboard-data"] == "build_dashboard_data"


def test_shadow_compare_mode(cli) -> None:
    called: list[str] = []

    def compare(**kwargs: Any) -> dict[str, Any]:
        called.append("compare")
        return {"status": "compared", "cid": _cid("cmp")}

    apis = _apis(compare_shadow_results=compare)
    code, payload, _ = _run(
        cli,
        [
            "shadow",
            "--mode",
            "compare",
            "--json",
            json.dumps(
                {
                    "compressed_result": {"cid": _cid("c")},
                    "expanded_result": {"cid": _cid("e")},
                }
            ),
        ],
        apis=apis,
    )
    assert code == 0
    assert called == ["compare"]
    assert payload["api"] == "compare_shadow_results"


def test_expand_plan_mode(cli) -> None:
    called: list[str] = []

    def plan(**kwargs: Any) -> dict[str, Any]:
        called.append("plan")
        return {"plan_cid": _cid("plan"), "status": "planned"}

    apis = _apis(plan_context_expansion=plan)
    code, payload, _ = _run(
        cli,
        [
            "expand",
            "--mode",
            "plan",
            "--json",
            json.dumps(
                {
                    "audit_case": {"case_id": "a1"},
                    "omission_hypotheses": [],
                    "token_budget": 1000,
                }
            ),
        ],
        apis=apis,
    )
    assert code == 0
    assert called == ["plan"]
    assert payload["api"] == "plan_context_expansion"


def test_input_file_payload(cli, tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps({"context_pack": {"cid": _cid("pack")}}),
        encoding="utf-8",
    )
    code, payload, _ = _run(
        cli,
        ["audit", "--input", str(path)],
        apis=_apis(),
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "audit"


def test_stdin_payload(cli) -> None:
    code, payload, _ = _run(
        cli,
        ["report", "--input", "-"],
        apis=_apis(),
        stdin="{}",
    )
    assert code == 0
    assert payload["command"] == "report"


# ---------------------------------------------------------------------------
# Private raw source stays out of output
# ---------------------------------------------------------------------------


def test_private_raw_source_stripped_from_output(cli) -> None:
    def leaky(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "cid": _cid("safe"),
            "raw_private_source": "class Secret: pass\nAPI_KEY=sk-leaked",
            "source_text": "def private(): ...",
            "nested": {
                "private_source": "leak-body",
                "report_cid": _cid("nested"),
            },
            "password": "should-not-appear",
            "managed_ref": _cid("managed"),
        }

    apis = _apis(build_governor_report=leaky)
    code, payload, _ = _run(cli, ["report", "--json", "{}"], apis=apis)
    assert code == 0
    text = json.dumps(payload)
    assert "raw_private_source" not in text
    assert "source_text" not in text
    assert "private_source" not in text
    assert "password" not in text
    assert "class Secret" not in text
    assert "sk-leaked" not in text
    assert "should-not-appear" not in text
    assert payload["result"]["status"] == "ok"
    assert payload["result"]["managed_ref"] == _cid("managed")
    assert payload["result"]["nested"]["report_cid"] == _cid("nested")


def test_host_paths_redacted_from_output(cli) -> None:
    def with_paths(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "cwd": "/home/secret/project",
            "note": "value is fine",
            "repo_relative": "src/mod.py",
        }

    apis = _apis(build_dashboard_data=with_paths)
    code, payload, _ = _run(
        cli, ["dashboard-data", "--json", "{}"], apis=apis
    )
    assert code == 0
    text = json.dumps(payload)
    assert "/home/secret/project" not in text
    assert payload["result"]["note"] == "value is fine"


def test_project_cli_output_helper(cli) -> None:
    raw = {
        "ok": True,
        "raw_source": "SECRET",
        "cid": _cid("x"),
        "items": [{"api_key": "x", "value": 1}],
    }
    projected = cli.project_cli_output(raw)
    assert "raw_source" not in projected
    assert projected["cid"] == _cid("x")
    assert "api_key" not in projected["items"][0]
    assert projected["items"][0]["value"] == 1


# ---------------------------------------------------------------------------
# Promotion: explicit authorization + CAS
# ---------------------------------------------------------------------------


def test_promote_without_authorization_is_production_gate(cli) -> None:
    promote = MagicMock(
        return_value={"status": "promoted", "head_mutated": True}
    )
    apis = _apis(promote_compression_policy=promote)
    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--operation-id",
            "op-no-auth",
            "--json",
            json.dumps(
                {
                    "candidate": {"candidate_id": "c1"},
                    "evaluation_report": {"verdict": "pass"},
                }
            ),
        ],
        apis=apis,
        policy_repository=MagicMock(),
    )
    assert code == cli.EXIT_PRODUCTION_GATE
    assert payload["ok"] is False
    assert payload["error"]["reason_code"] == "absent_authorization"
    assert payload["error"]["head_mutated"] is False
    assert payload["error"]["implicit_promotion"] is False
    promote.assert_not_called()


def test_promote_without_cas_is_unavailable(cli) -> None:
    promote = MagicMock()
    apis = _apis(promote_compression_policy=promote)
    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--authorization",
            _cid("auth-1"),
            "--operation-id",
            "op-no-cas",
            "--json",
            json.dumps({"candidate": {}, "evaluation_report": {}}),
        ],
        apis=apis,
        # No policy_repository and no --store-dir
    )
    assert code == cli.EXIT_UNAVAILABLE
    assert payload["ok"] is False
    assert payload["error"]["reason_code"] == "cas_unavailable"
    assert payload["error"]["cas_required"] is True
    promote.assert_not_called()


def test_promote_with_authorization_and_cas_calls_api(cli) -> None:
    captured: dict[str, Any] = {}

    def promote(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "status": "promoted",
            "head_mutated": True,
            "authorization_cid": kwargs["authorization"],
            "operation_id": kwargs["operation_id"],
            "workspace": kwargs["workspace"],
            "policy_cas": {"status": "updated", "generation": 2},
        }

    repo = MagicMock(name="policy_repo")
    promo_repo = MagicMock(name="promo_repo")
    auth = _cid("explicit-board-auth")
    apis = _apis(promote_compression_policy=promote)
    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--authorization",
            auth,
            "--operation-id",
            "promo-cas-1",
            "--workspace",
            "default",
            "--expected-generation",
            "1",
            "--json",
            json.dumps(
                {
                    "candidate": {"candidate_id": "cand-1"},
                    "evaluation_report": {"verdict": "pass"},
                    "release_qualification": {
                        "promotion_allowed": True,
                        "authorization_cid": auth,
                    },
                }
            ),
        ],
        apis=apis,
        policy_repository=repo,
        promotion_repository=promo_repo,
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "promote-policy"
    assert payload["api"] == "promote_compression_policy"
    assert captured["authorization"] == auth
    assert captured["operation_id"] == "promo-cas-1"
    assert captured["policy_repository"] is repo
    assert captured["promotion_repository"] is promo_repo
    assert captured["expected_generation"] == 1
    assert payload["result"]["head_mutated"] is True
    assert payload["result"]["authorization_required"] is True
    assert payload["result"]["cas_required"] is True
    assert payload["result"]["implicit_promotion"] is False


def test_promote_authorization_from_payload(cli) -> None:
    captured: dict[str, Any] = {}

    def promote(**kwargs: Any) -> dict[str, Any]:
        captured["authorization"] = kwargs["authorization"]
        return {
            "status": "promoted",
            "head_mutated": True,
            "authorization_cid": kwargs["authorization"],
        }

    auth = _cid("payload-auth")
    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--operation-id",
            "op-payload-auth",
            "--json",
            json.dumps(
                {
                    "authorization": auth,
                    "candidate": {"candidate_id": "c"},
                    "evaluation_report": {"verdict": "pass"},
                }
            ),
        ],
        apis=_apis(promote_compression_policy=promote),
        policy_repository=MagicMock(),
    )
    assert code == 0
    assert captured["authorization"] == auth


def test_promote_rejected_by_api_exits_nonzero_without_mutation(cli) -> None:
    def promote(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "rejected",
            "head_mutated": False,
            "blocking_reasons": ["stale_candidate"],
            "diagnostic": "candidate base does not match live head",
        }

    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--authorization",
            _cid("auth"),
            "--operation-id",
            "op-reject",
            "--json",
            json.dumps({"candidate": {}, "evaluation_report": {}}),
        ],
        apis=_apis(promote_compression_policy=promote),
        policy_repository=MagicMock(),
    )
    assert code == cli.EXIT_ERROR
    assert payload["ok"] is False
    assert payload["error"]["head_mutated"] is False
    assert "stale_candidate" in payload["error"]["reason_code"]


def test_promote_requires_operation_id(cli) -> None:
    code, payload, _ = _run(
        cli,
        [
            "promote-policy",
            "--authorization",
            _cid("auth"),
            "--json",
            json.dumps({"candidate": {}, "evaluation_report": {}}),
        ],
        apis=_apis(),
        policy_repository=MagicMock(),
    )
    assert code == cli.EXIT_ERROR
    assert payload["ok"] is False
    assert "operation" in payload["error"]["diagnostic"].lower() or payload[
        "error"
    ]["reason_code"]


# ---------------------------------------------------------------------------
# Packaging discovery
# ---------------------------------------------------------------------------


def test_setup_py_registers_console_entry() -> None:
    """Console entry is declared without pyproject edits.

    ``pyproject.toml`` is a validation-config path; the proposal gate hard-denies
    unauthorised edits. Packaging therefore uses setuptools ``setup.py`` only:
    console entry via ``entry_points`` + generated ``scripts=`` wrapper (same
    pattern as ``semantic-state`` / SCH-013).
    """

    setup_text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "semantic-governor=" in setup_text
    assert (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.cli:main"
        in setup_text
    )
    assert (
        "_semantic_governor_console" in setup_text
        or "_SEMANTIC_GOVERNOR_CONSOLE" in setup_text
        or "scripts=" in setup_text
    )


def test_compact_json_flag(cli) -> None:
    out = io.StringIO()
    code = cli.main(
        ["report", "--json", "{}", "--compact"],
        stdout=out,
        stderr=io.StringIO(),
        apis=_apis(),
    )
    assert code == 0
    text = out.getvalue()
    assert "\n  " not in text  # no indented pretty-print
    payload = json.loads(text)
    assert payload["ok"] is True


def test_invalid_json_payload_is_usage_error(cli) -> None:
    code, payload, _ = _run(
        cli,
        ["audit", "--json", "not-json"],
        apis=_apis(),
    )
    assert code == cli.EXIT_USAGE
    assert payload["ok"] is False
    assert payload["error"]["reason_code"] == "invalid_payload"


def test_unknown_command_rejected(cli) -> None:
    # argparse should reject before handler
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(["not-a-command"], stdout=out, stderr=err, apis=_apis())
    assert code == cli.EXIT_USAGE


def test_api_exception_is_typed_error(cli) -> None:
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("leaf failed")

    code, payload, _ = _run(
        cli,
        ["diagnose", "--json", "{}"],
        apis=_apis(diagnose_omission=boom),
    )
    assert code == cli.EXIT_ERROR
    assert payload["ok"] is False
    assert "leaf failed" in payload["error"]["diagnostic"]


def test_unavailable_api_exits_three(cli) -> None:
    class Unavailable(Exception):
        reason_code = "api_unavailable"
        diagnostic = "surface missing"
        retryable = False

    def boom(**kwargs: Any) -> Any:
        raise Unavailable()

    code, payload, _ = _run(
        cli,
        ["calibrate", "--json", "{}"],
        apis=_apis(update_calibration=boom),
    )
    assert code == cli.EXIT_UNAVAILABLE
    assert payload["ok"] is False
    assert payload["error"]["reason_code"] == "api_unavailable"

"""VGO-060: standalone gui-opt CLI tests.

Acceptance coverage:

* help/schema snapshots are stable
* all eight command families emit closed JSON receipts
* targets and aliases resolve only through the repository registry
* path/command injection and unknown flags reject fail-closed
* verify/improve refuse production/canonical defaults
* report recovers interrupted journals and never treats process exit
  as completion
* TypeScript bridge argv is fixed and never caller-supplied
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.cli import (
    BENCHMARK_REGISTRY,
    COMMAND_INTERFACES,
    GUI_OPTIMIZER_CLI_INTERFACE,
    GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE,
    GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA,
    GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE,
    GUI_OPT_COMMANDS,
    HELP_TEXT,
    HOST_PYTHON_EXECUTABLE,
    HOST_VALIDATION_PATH,
    REPORT_ALIAS_REGISTRY,
    TARGET_REGISTRY,
    TYPESCRIPT_CLI_BRIDGE_ARGV,
    TYPESCRIPT_CLI_MODULE,
    VERIFY_ALIAS_REGISTRY,
    GuiOptimizerCliError,
    default_repo_root,
    parse_argv,
    resolve_target,
    run_cli,
    sealed_cli_environment,
    typescript_bridge_plan,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.run_journal import (
    JournalPhase,
    PhaseRecordStatus,
    RunStatus,
    default_run_journal,
)

REVISION = "b" * 40
IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def test_cli_interface_and_command_schema_snapshot() -> None:
    assert GUI_OPTIMIZER_CLI_INTERFACE == "GuiOptimizerCli@1"
    assert (
        GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE
        == "GuiOptimizerTypeScriptCliBridge@1"
    )
    assert GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE == "GuiOptimizerCliReceipt@1"
    assert GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA.endswith("cli-receipt@1")
    assert GUI_OPT_COMMANDS == (
        "scan",
        "baseline",
        "impact",
        "evaluate",
        "pack-context",
        "verify",
        "improve",
        "report",
    )
    assert COMMAND_INTERFACES["scan"] == "gui-opt scan@1"
    assert COMMAND_INTERFACES["baseline"] == "gui-opt baseline@1"
    assert COMMAND_INTERFACES["impact"] == "gui-opt impact@1"
    assert COMMAND_INTERFACES["evaluate"] == "gui-opt evaluate@1"
    assert COMMAND_INTERFACES["pack-context"] == "gui-opt pack-context@1"
    assert COMMAND_INTERFACES["verify"] == "gui-opt verify@1"
    assert COMMAND_INTERFACES["improve"] == "gui-opt improve@1"
    assert COMMAND_INTERFACES["report"] == "gui-opt report@1"
    assert "agent-supervisor" in TARGET_REGISTRY
    assert "agent-supervisor-target" in VERIFY_ALIAS_REGISTRY
    assert "current-tree" in VERIFY_ALIAS_REGISTRY
    assert "final-current-tree" in VERIFY_ALIAS_REGISTRY
    assert "final-current-tree" in REPORT_ALIAS_REGISTRY
    assert "benchmark-agent-supervisor" in REPORT_ALIAS_REGISTRY
    assert "acceptance-security-audit" in REPORT_ALIAS_REGISTRY
    assert "benchmark-v1" in BENCHMARK_REGISTRY
    assert HOST_PYTHON_EXECUTABLE == "/usr/bin/python3.12"
    assert HOST_VALIDATION_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert TYPESCRIPT_CLI_MODULE == "swissknife/src/services/gui-optimizer/cli.ts"
    assert TYPESCRIPT_CLI_BRIDGE_ARGV == (TYPESCRIPT_CLI_MODULE,)


def test_default_repo_root_finds_the_gui_opt_adapter() -> None:
    root = default_repo_root()
    assert (root / "scripts" / "gui-opt").is_file()
    assert (root / IN_SCOPE).is_file()


def test_help_snapshot_lists_all_command_families() -> None:
    result = run_cli(["--help"])
    assert result.exit_code == 0
    assert result.receipt is None
    assert result.human_text == HELP_TEXT
    for command in GUI_OPT_COMMANDS:
        assert command in HELP_TEXT
    assert "gui-opt --help" in HELP_TEXT
    assert "isolated worktrees" in HELP_TEXT


def test_sealed_environment_uses_validation_path_and_xdg(tmp_path: Path) -> None:
    home = tmp_path / "ipfs-accelerate-validation-home-test"
    env = sealed_cli_environment(str(home))
    assert env["PATH"] == HOST_VALIDATION_PATH
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["XDG_CACHE_HOME"] == f"{home}/.cache"
    assert env["XDG_CONFIG_HOME"] == f"{home}/.config"
    assert env["XDG_DATA_HOME"] == f"{home}/.local/share"
    assert env["XDG_STATE_HOME"] == f"{home}/.local/state"


def test_fixed_target_resolution() -> None:
    target = resolve_target("agent-supervisor")
    assert target.application_id == "app:agent-supervisor"
    assert target.screen_id == "screen:agent-supervisor"
    assert IN_SCOPE in target.source_paths
    with pytest.raises(GuiOptimizerCliError) as exc:
        resolve_target("unknown-app")
    assert exc.value.reason_code == "unknown_target"


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "../secrets.env"],
        ["scan", "/etc/passwd"],
        ["scan", "C:\\Windows\\System32\\cmd.exe"],
        ["baseline", "file:///etc/passwd"],
        ["evaluate", "swissknife/web/js/apps/agent-supervisor.js"],
        ["improve", "../../etc/passwd", "--objective", "accessible-name"],
    ],
)
def test_target_path_injection_rejected(argv: list[str], tmp_path: Path) -> None:
    result = run_cli(argv, repo_root=tmp_path)
    assert result.exit_code != 0
    assert result.receipt is not None
    assert result.receipt["ok"] is False
    assert result.receipt["reason_codes"][0] in {
        "path_injection",
        "path_absolute_or_traversal",
        "unknown_target",
        "invalid_argument",
    }


@pytest.mark.parametrize(
    "subject",
    [
        "../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
        "swissknife/web/js/apps/../../etc/passwd",
        "file:///tmp/x",
        "swissknife/web/js/apps/node_modules/evil.js",
        "swissknife/src/services/control/authorization.ts",
    ],
)
def test_impact_path_injection_and_unregistered_paths_reject(
    subject: str, tmp_path: Path
) -> None:
    result = run_cli(["impact", subject], repo_root=tmp_path)
    assert result.exit_code != 0
    assert result.receipt is not None
    assert result.receipt["ok"] is False


def test_command_metacharacters_and_forbidden_flags_reject(tmp_path: Path) -> None:
    injected = run_cli(["scan", "agent-supervisor;rm -rf /"], repo_root=tmp_path)
    assert injected.exit_code != 0
    assert injected.receipt is not None
    assert "command_string_forbidden" in injected.receipt["reason_codes"]

    forbidden = run_cli(
        ["scan", "agent-supervisor", "--shell", "bash"], repo_root=tmp_path
    )
    assert forbidden.exit_code != 0
    assert forbidden.receipt is not None
    assert "forbidden_flag" in forbidden.receipt["reason_codes"]

    unknown = run_cli(["not-a-command", "agent-supervisor"], repo_root=tmp_path)
    assert unknown.exit_code != 0
    assert unknown.receipt is not None
    assert "unknown_command" in unknown.receipt["reason_codes"]


def test_scan_and_baseline_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / IN_SCOPE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("export const GoalForm = () => null;\n", encoding="utf-8")
    first = run_cli(["scan", "agent-supervisor"], repo_root=tmp_path)
    second = run_cli(["scan", "agent-supervisor"], repo_root=tmp_path)
    assert first.ok and second.ok
    assert first.receipt is not None and second.receipt is not None
    assert first.receipt["receipt_id"] == second.receipt["receipt_id"]
    assert first.receipt["interface"] == GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE
    assert first.receipt["command_interface"] == "gui-opt scan@1"
    assert first.receipt["payload"]["effectful"] is False
    assert first.receipt["payload"]["typescript_bridge"]["executed"] is False
    assert first.receipt["payload"]["typescript_bridge"]["argv"][0] == TYPESCRIPT_CLI_MODULE

    baseline_a = run_cli(["baseline", "agent-supervisor"], repo_root=tmp_path)
    baseline_b = run_cli(["baseline", "agent-supervisor"], repo_root=tmp_path)
    assert baseline_a.ok and baseline_b.ok
    assert baseline_a.receipt is not None and baseline_b.receipt is not None
    assert baseline_a.receipt["receipt_id"] == baseline_b.receipt["receipt_id"]
    assert baseline_a.receipt["payload"]["baseline_seed_digest"].startswith("sha256:")


def test_impact_resolves_registered_component_and_path(tmp_path: Path) -> None:
    component = run_cli(["impact", "comp:goal-form"], repo_root=tmp_path)
    assert component.ok
    assert component.receipt is not None
    assert component.receipt["payload"]["impact"]["kind"] == "component"
    assert component.receipt["payload"]["impact"]["source_path"] == IN_SCOPE

    path = run_cli(["impact", IN_SCOPE], repo_root=tmp_path)
    assert path.ok
    assert path.receipt is not None
    assert path.receipt["payload"]["impact"]["kind"] == "path"


def test_pack_context_requires_registered_objective(tmp_path: Path) -> None:
    missing = run_cli(["pack-context", "agent-supervisor"], repo_root=tmp_path)
    assert not missing.ok
    assert missing.receipt is not None
    assert "missing_objective" in missing.receipt["reason_codes"]

    ok = run_cli(
        ["pack-context", "agent-supervisor", "--objective", "accessible-name"],
        repo_root=tmp_path,
    )
    assert ok.ok
    assert ok.receipt is not None
    assert (
        ok.receipt["payload"]["objective"]["objective_id"]
        == "objective:accessible-name"
    )


def test_evaluate_unknown_or_missing_benchmark_is_fail_closed(tmp_path: Path) -> None:
    plain = run_cli(["evaluate", "agent-supervisor"], repo_root=tmp_path)
    assert plain.ok
    assert plain.receipt is not None
    assert plain.receipt["payload"]["effectful"] is False

    unknown = run_cli(
        ["evaluate", "agent-supervisor", "--benchmark", "not-registered"],
        repo_root=tmp_path,
    )
    assert not unknown.ok
    assert unknown.receipt is not None
    assert "unknown_subject" in unknown.receipt["reason_codes"]

    missing = run_cli(
        [
            "evaluate",
            "agent-supervisor",
            "--benchmark",
            "benchmark-v1",
            "--expected-tasks",
            "15",
            "--progress-interval-seconds",
            "60",
        ],
        repo_root=tmp_path,
    )
    assert not missing.ok
    assert missing.receipt is not None
    assert "benchmark_catalog_unavailable" in missing.receipt["reason_codes"]

    mismatch = run_cli(
        [
            "evaluate",
            "agent-supervisor",
            "--benchmark",
            "benchmark-v1",
            "--expected-tasks",
            "3",
        ],
        repo_root=tmp_path,
    )
    assert not mismatch.ok
    assert mismatch.receipt is not None
    assert "benchmark_task_count_mismatch" in mismatch.receipt["reason_codes"]


def test_verify_named_target_and_current_tree_aliases(tmp_path: Path) -> None:
    missing = run_cli(
        [
            "verify",
            "agent-supervisor-target",
            "--receipt",
            "implementation_plan/evidence/verified_gui_optimizer/"
            "agent-supervisor-target-improvement-receipt.json",
        ],
        repo_root=tmp_path,
    )
    assert not missing.ok
    assert missing.receipt is not None
    assert "missing_receipt" in missing.receipt["reason_codes"]

    receipt_rel = (
        "implementation_plan/evidence/verified_gui_optimizer/"
        "agent-supervisor-target-improvement-receipt.json"
    )
    _write_json(
        tmp_path / receipt_rel,
        {
            "interface": "GuiImprovementReceipt@1",
            "decision": "accept",
            "verification_status": "verified",
        },
    )
    present = run_cli(
        ["verify", "agent-supervisor-target", "--receipt", receipt_rel],
        repo_root=tmp_path,
    )
    assert present.ok
    assert present.receipt is not None
    assert present.receipt["payload"]["receipt_present"] is True

    current = run_cli(
        [
            "verify",
            "current-tree",
            "--full",
            "--receipt",
            "implementation_plan/evidence/verified_gui_optimizer/"
            "current-tree-verification.json",
        ],
        repo_root=tmp_path,
    )
    assert not current.ok
    assert current.receipt is not None
    assert current.receipt["payload"]["full"] is True


def test_verify_receipt_path_injection_rejected(tmp_path: Path) -> None:
    result = run_cli(
        ["verify", "agent-supervisor-target", "--receipt", "../secrets.json"],
        repo_root=tmp_path,
    )
    assert result.exit_code != 0
    assert result.receipt is not None
    assert result.receipt["reason_codes"][0] in {
        "path_absolute_or_traversal",
        "path_injection",
        "path_outside_allowed_roots",
    }


def test_improve_has_no_production_or_canonical_defaults(tmp_path: Path) -> None:
    missing_obj = run_cli(["improve", "agent-supervisor"], repo_root=tmp_path)
    assert not missing_obj.ok
    assert missing_obj.receipt is not None
    assert "missing_objective" in missing_obj.receipt["reason_codes"]

    no_isolated = run_cli(
        ["improve", "agent-supervisor", "--objective", "accessible-name"],
        repo_root=tmp_path,
    )
    assert not no_isolated.ok
    assert no_isolated.receipt is not None
    codes = set(no_isolated.receipt["reason_codes"])
    assert "isolated_worktree_required" in codes
    assert "no_production_defaults" in codes
    assert no_isolated.receipt["payload"]["canonical_merge"] is False
    assert no_isolated.receipt["payload"]["production_defaults"] is False


def test_report_require_complete_missing_alias_fails_closed(tmp_path: Path) -> None:
    result = run_cli(
        [
            "report",
            "final-current-tree",
            "--require-complete",
            "--verify-receipts",
        ],
        repo_root=tmp_path,
    )
    assert not result.ok
    assert result.receipt is not None
    assert "incomplete_evidence" in result.receipt["reason_codes"]

    benchmark = run_cli(
        [
            "report",
            "benchmark-agent-supervisor",
            "--require-complete",
            "--expected-tasks",
            "15",
            "--verify-receipts",
        ],
        repo_root=tmp_path,
    )
    assert not benchmark.ok
    audit = run_cli(
        [
            "report",
            "acceptance-security-audit",
            "--require-complete",
            "--verify-receipts",
        ],
        repo_root=tmp_path,
    )
    assert not audit.ok


def test_interrupted_report_recovery(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "cas")
    journal.open_run(
        run_id="run:interrupted-label",
        application_id="app:agent-supervisor",
        screen_id="screen:agent-supervisor",
        objective_id="objective:accessible-name",
        source_revision=REVISION,
        canonical_branch="main",
        canonical_revision=REVISION,
        canonical_porcelain="",
        attempt=1,
    )
    journal.append_phase(
        run_id="run:interrupted-label",
        phase=JournalPhase.BASELINE,
        effect_id="effect:run:interrupted-label:baseline:1",
        payload={"status": "started"},
        status=PhaseRecordStatus.INTERRUPTED,
    )
    checkpoint = journal.require_checkpoint("run:interrupted-label")
    assert checkpoint.status is not RunStatus.COMPLETED

    result = run_cli(
        ["report", "run:interrupted-label"],
        repo_root=tmp_path,
        journal=journal,
    )
    assert result.ok
    assert result.receipt is not None
    assert "interrupted" in result.receipt["reason_codes"]
    assert "process_exit_not_completion" in result.receipt["reason_codes"]
    assert result.receipt["payload"]["process_exit_is_completion"] is False
    assert result.receipt["payload"]["status"] != RunStatus.COMPLETED.value

    required = run_cli(
        ["report", "run:interrupted-label", "--require-complete"],
        repo_root=tmp_path,
        journal=journal,
    )
    assert not required.ok
    assert required.receipt is not None
    assert "interrupted" in required.receipt["reason_codes"]


def test_report_rejects_path_subjects(tmp_path: Path) -> None:
    result = run_cli(["report", "../journals/head.json"], repo_root=tmp_path)
    assert result.exit_code != 0
    assert result.receipt is not None
    assert "path_injection" in result.receipt["reason_codes"]


def test_typescript_bridge_plan_is_fixed() -> None:
    plan = typescript_bridge_plan("scan", "agent-supervisor")
    assert plan["interface"] == GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE
    assert plan["argv"] == [TYPESCRIPT_CLI_MODULE, "scan", "agent-supervisor"]
    assert plan["executed"] is False


def test_parse_argv_rejects_extra_subjects() -> None:
    with pytest.raises(GuiOptimizerCliError) as exc:
        parse_argv(["scan", "agent-supervisor", "other"])
    assert exc.value.reason_code == "invalid_argument"


def test_all_eight_commands_emit_closed_receipts(tmp_path: Path) -> None:
    cases = [
        ["scan", "agent-supervisor"],
        ["baseline", "agent-supervisor"],
        ["impact", "comp:goal-form"],
        ["evaluate", "agent-supervisor"],
        ["pack-context", "agent-supervisor", "--objective", "accessible-name"],
        ["verify", "agent-supervisor-target"],
        ["improve", "agent-supervisor", "--objective", "accessible-name"],
        ["report", "final-current-tree"],
    ]
    for argv in cases:
        result = run_cli(argv, repo_root=tmp_path)
        assert result.receipt is not None
        assert result.receipt["interface"] == GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE
        assert result.receipt["schema_version"] == GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA
        assert result.receipt["command"] == argv[0]
        assert result.receipt["command_interface"] == COMMAND_INTERFACES[argv[0]]
        assert result.receipt["receipt_id"].startswith("sha256:")
        assert type(result.receipt["reason_codes"]) is list
        assert type(result.receipt["ok"]) is bool
        assert type(result.receipt["exit_code"]) is int
        clone = dict(result.receipt)
        # Re-running the same argv yields the same identity for observation cmds.
        if argv[0] in {"scan", "baseline", "impact", "evaluate", "pack-context"}:
            again = run_cli(argv, repo_root=tmp_path)
            assert again.receipt is not None
            assert again.receipt["receipt_id"] == clone["receipt_id"]

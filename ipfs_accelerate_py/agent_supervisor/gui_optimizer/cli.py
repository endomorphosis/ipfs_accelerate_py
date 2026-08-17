"""Standalone ``gui-opt`` development CLI (VGO-060).

Interfaces owned by this module:

* ``GuiOptimizerCli@1`` — fixed scan/baseline/impact/evaluate/pack-context/
  verify/improve/report command surface
* ``GuiOptimizerTypeScriptCliBridge@1`` — sealed TypeScript observation bridge
* ``gui-opt scan@1`` … ``gui-opt report@1`` — per-command receipts

The root adapter and this module accept only repository-registry target IDs
and allowlisted repository-relative paths.  Callers cannot select arbitrary
host paths, shell strings, or executables.  Observation commands are
non-effectful.  ``verify`` / ``improve`` refuse canonical-tree defaults and
operate only against isolated worktree or durable-receipt aliases.
``report`` resolves immutable journal or registered evidence and never
treats process exit as completion.

Fail-closed invariants:

* unknown commands, targets, flags, and path/command injection reject;
* JSON receipts are schema-versioned and deterministic;
* missing required evidence exits nonzero;
* interrupted journals report interruption, never completion;
* TypeScript bridge argv is looked up from a fixed registry.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .authority import (
    AuthorityReasonCode,
    GuiAuthorityError,
    GuiPatchAuthority,
    path_has_forbidden_segment,
    path_under_allowed_roots,
)
from .check_plan import HOST_PYTHON_EXECUTABLE, HOST_VALIDATION_PATH
from .run_journal import (
    GuiRunJournal,
    GuiRunJournalError,
    JournalReasonCode,
    RunStatus,
    default_run_journal,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_OPTIMIZER_CLI_INTERFACE: Final[str] = "GuiOptimizerCli@1"
GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE: Final[str] = (
    "GuiOptimizerTypeScriptCliBridge@1"
)
GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE: Final[str] = "GuiOptimizerCliReceipt@1"
GUI_OPTIMIZER_CLI_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/cli@1"
)
GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/cli-receipt@1"
)
GUI_OPTIMIZER_CLI_VERSION: Final[str] = "gui-opt@1.0.0"

GUI_OPT_SCAN_INTERFACE: Final[str] = "gui-opt scan@1"
GUI_OPT_BASELINE_INTERFACE: Final[str] = "gui-opt baseline@1"
GUI_OPT_IMPACT_INTERFACE: Final[str] = "gui-opt impact@1"
GUI_OPT_EVALUATE_INTERFACE: Final[str] = "gui-opt evaluate@1"
GUI_OPT_PACK_CONTEXT_INTERFACE: Final[str] = "gui-opt pack-context@1"
GUI_OPT_VERIFY_INTERFACE: Final[str] = "gui-opt verify@1"
GUI_OPT_IMPROVE_INTERFACE: Final[str] = "gui-opt improve@1"
GUI_OPT_REPORT_INTERFACE: Final[str] = "gui-opt report@1"

COMMAND_INTERFACES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scan": GUI_OPT_SCAN_INTERFACE,
        "baseline": GUI_OPT_BASELINE_INTERFACE,
        "impact": GUI_OPT_IMPACT_INTERFACE,
        "evaluate": GUI_OPT_EVALUATE_INTERFACE,
        "pack-context": GUI_OPT_PACK_CONTEXT_INTERFACE,
        "verify": GUI_OPT_VERIFY_INTERFACE,
        "improve": GUI_OPT_IMPROVE_INTERFACE,
        "report": GUI_OPT_REPORT_INTERFACE,
    }
)
GUI_OPT_COMMANDS: Final[tuple[str, ...]] = tuple(COMMAND_INTERFACES)

TYPESCRIPT_CLI_MODULE: Final[str] = "swissknife/src/services/gui-optimizer/cli.ts"
TYPESCRIPT_CLI_BRIDGE_ARGV: Final[tuple[str, ...]] = (TYPESCRIPT_CLI_MODULE,)

EVIDENCE_ROOT: Final[str] = "implementation_plan/evidence/verified_gui_optimizer/"
AGENT_SUPERVISOR_SOURCE: Final[str] = "swissknife/web/js/apps/agent-supervisor.js"
DEFAULT_APPLICATION_ID: Final[str] = "app:agent-supervisor"
DEFAULT_SCREEN_ID: Final[str] = "screen:agent-supervisor"

_SAFE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_COMMAND_META_RE: Final = re.compile(r"[;&|`$<>\n\r]|\$\(|\)")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[a-zA-Z]:")
_URI_RE: Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

FORBIDDEN_CLI_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--argv",
        "--cmd",
        "--command",
        "--cwd",
        "--env",
        "--exec",
        "--executable",
        "--file-path",
        "--host-path",
        "--python-process",
        "--shell",
        "--subprocess",
        "--working-directory",
    }
)

GLOBAL_FLAGS: Final[frozenset[str]] = frozenset({"--help", "-h", "--json"})
COMMAND_FLAGS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "scan": frozenset(),
        "baseline": frozenset(),
        "impact": frozenset(),
        "evaluate": frozenset(
            {
                "--benchmark",
                "--expected-tasks",
                "--progress-interval-seconds",
            }
        ),
        "pack-context": frozenset({"--objective"}),
        "verify": frozenset({"--receipt", "--full"}),
        "improve": frozenset({"--objective", "--isolated", "--request"}),
        "report": frozenset(
            {"--require-complete", "--verify-receipts", "--expected-tasks"}
        ),
    }
)
FLAGS_TAKING_VALUE: Final[frozenset[str]] = frozenset(
    {
        "--benchmark",
        "--expected-tasks",
        "--objective",
        "--progress-interval-seconds",
        "--receipt",
        "--request",
    }
)

HELP_TEXT: Final[str] = """gui-opt — Verified GUI Optimizer development CLI

Usage:
  gui-opt --help
  gui-opt scan <target>
  gui-opt baseline <target>
  gui-opt impact <path-or-component>
  gui-opt evaluate <target> [--benchmark ID] [--expected-tasks N]
      [--progress-interval-seconds N]
  gui-opt pack-context <target> --objective <objective>
  gui-opt verify <worktree-or-patch-or-alias> [--receipt PATH] [--full]
  gui-opt improve <target> --objective <objective> [--isolated] [--request PATH]
  gui-opt report <run-id-or-alias> [--require-complete] [--verify-receipts]
      [--expected-tasks N]

Targets resolve through the repository registry. Paths are repository-relative
and allowlisted. Commands are fixed argument vectors. JSON receipts are
schema-versioned. verify/improve operate only in isolated worktrees.
"""


class CliReasonCode(str):
    """Stable CLI reason codes.  Values are plain strings for JSON."""


class _Codes:
    OK = "ok"
    HELP = "help"
    TARGET_RESOLVED = "target_resolved"
    ALIAS_RESOLVED = "alias_resolved"
    BRIDGE_PLANNED = "typescript_bridge_planned"
    UNKNOWN_COMMAND = "unknown_command"
    UNKNOWN_TARGET = "unknown_target"
    UNKNOWN_FLAG = "unknown_flag"
    UNKNOWN_SUBJECT = "unknown_subject"
    MISSING_SUBJECT = "missing_subject"
    MISSING_OBJECTIVE = "missing_objective"
    MISSING_RECEIPT = "missing_receipt"
    MISSING_EVIDENCE = "missing_evidence"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    PATH_INJECTION = "path_injection"
    PATH_ABSOLUTE_OR_TRAVERSAL = AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    PATH_OUTSIDE_ALLOWED_ROOTS = AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
    COMMAND_STRING_FORBIDDEN = "command_string_forbidden"
    FORBIDDEN_FLAG = "forbidden_flag"
    INVALID_ARGUMENT = "invalid_argument"
    ISOLATED_WORKTREE_REQUIRED = "isolated_worktree_required"
    NO_PRODUCTION_DEFAULTS = "no_production_defaults"
    BENCHMARK_CATALOG_UNAVAILABLE = "benchmark_catalog_unavailable"
    BENCHMARK_TASK_COUNT_MISMATCH = "benchmark_task_count_mismatch"
    RECEIPT_INVALID = "receipt_invalid"
    INTERRUPTED = "interrupted"
    JOURNAL_MISSING = "journal_missing"
    JOURNAL_CORRUPT = "journal_corrupt"
    PROCESS_EXIT_NOT_COMPLETION = JournalReasonCode.PROCESS_EXIT_NOT_COMPLETION.value


@dataclass(frozen=True)
class RegisteredTarget:
    """Closed application/screen target.  Never a free filesystem path."""

    target_id: str
    application_id: str
    screen_id: str
    source_paths: tuple[str, ...]
    component_ids: tuple[str, ...]
    kind: str = "application"

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "component_ids": list(self.component_ids),
            "kind": self.kind,
            "screen_id": self.screen_id,
            "source_paths": list(self.source_paths),
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class RegisteredComponent:
    component_id: str
    target_id: str
    source_path: str


@dataclass(frozen=True)
class RegisteredVerifyAlias:
    alias_id: str
    kind: str
    target_id: str
    default_receipt: str
    isolated_worktree_required: bool = True
    full: bool = False


@dataclass(frozen=True)
class RegisteredReportAlias:
    alias_id: str
    kind: str
    receipt_path: str
    report_path: str = ""
    expected_tasks: int | None = None


@dataclass(frozen=True)
class RegisteredBenchmark:
    benchmark_id: str
    catalog_path: str
    expected_tasks: int


@dataclass(frozen=True)
class RegisteredObjective:
    objective_id: str
    metric_id: str
    label: str


_AGENT_SUPERVISOR = RegisteredTarget(
    target_id="agent-supervisor",
    application_id=DEFAULT_APPLICATION_ID,
    screen_id=DEFAULT_SCREEN_ID,
    source_paths=(AGENT_SUPERVISOR_SOURCE,),
    component_ids=("comp:console-root", "comp:goal-form"),
)

TARGET_REGISTRY: Final[Mapping[str, RegisteredTarget]] = MappingProxyType(
    {_AGENT_SUPERVISOR.target_id: _AGENT_SUPERVISOR}
)

COMPONENT_REGISTRY: Final[Mapping[str, RegisteredComponent]] = MappingProxyType(
    {
        "comp:console-root": RegisteredComponent(
            "comp:console-root", "agent-supervisor", AGENT_SUPERVISOR_SOURCE
        ),
        "comp:goal-form": RegisteredComponent(
            "comp:goal-form", "agent-supervisor", AGENT_SUPERVISOR_SOURCE
        ),
    }
)

VERIFY_ALIAS_REGISTRY: Final[Mapping[str, RegisteredVerifyAlias]] = MappingProxyType(
    {
        "agent-supervisor-target": RegisteredVerifyAlias(
            alias_id="agent-supervisor-target",
            kind="named_target_receipt",
            target_id="agent-supervisor",
            default_receipt=(
                f"{EVIDENCE_ROOT}agent-supervisor-target-improvement-receipt.json"
            ),
        ),
        "current-tree": RegisteredVerifyAlias(
            alias_id="current-tree",
            kind="current_tree",
            target_id="agent-supervisor",
            default_receipt=f"{EVIDENCE_ROOT}current-tree-verification.json",
            full=True,
        ),
        "final-current-tree": RegisteredVerifyAlias(
            alias_id="final-current-tree",
            kind="current_tree",
            target_id="agent-supervisor",
            default_receipt=f"{EVIDENCE_ROOT}current-tree-verification.json",
            full=True,
        ),
    }
)

REPORT_ALIAS_REGISTRY: Final[Mapping[str, RegisteredReportAlias]] = MappingProxyType(
    {
        "final-current-tree": RegisteredReportAlias(
            alias_id="final-current-tree",
            kind="final_current_tree",
            receipt_path=f"{EVIDENCE_ROOT}final-current-tree-receipt.json",
            report_path=f"{EVIDENCE_ROOT}final-report.md",
        ),
        "benchmark-agent-supervisor": RegisteredReportAlias(
            alias_id="benchmark-agent-supervisor",
            kind="benchmark",
            receipt_path=f"{EVIDENCE_ROOT}agent-supervisor-benchmark.json",
            expected_tasks=15,
        ),
        "acceptance-security-audit": RegisteredReportAlias(
            alias_id="acceptance-security-audit",
            kind="audit",
            receipt_path=f"{EVIDENCE_ROOT}acceptance-security-audit.json",
        ),
    }
)

BENCHMARK_REGISTRY: Final[Mapping[str, RegisteredBenchmark]] = MappingProxyType(
    {
        "benchmark-v1": RegisteredBenchmark(
            benchmark_id="benchmark-v1",
            catalog_path=(
                "external/ipfs_accelerate/test/fixtures/gui_optimizer/"
                "benchmark-tasks.json"
            ),
            expected_tasks=15,
        )
    }
)

OBJECTIVE_REGISTRY: Final[Mapping[str, RegisteredObjective]] = MappingProxyType(
    {
        "accessible-name": RegisteredObjective(
            "objective:accessible-name",
            "accessible_name_coverage",
            "Repair accessible names on the selected screen.",
        ),
        "objective:accessible-name": RegisteredObjective(
            "objective:accessible-name",
            "accessible_name_coverage",
            "Repair accessible names on the selected screen.",
        ),
        "accessible_name_coverage": RegisteredObjective(
            "objective:accessible-name",
            "accessible_name_coverage",
            "Repair accessible names on the selected screen.",
        ),
    }
)


class GuiOptimizerCliError(GuiAuthorityError):
    """Malformed or unsafe CLI input.  Never grants an effect."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = _Codes.INVALID_ARGUMENT,
        details: Mapping[str, Any] | None = None,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)
        self.exit_code = exit_code


@dataclass(frozen=True)
class CliRequest:
    command: str
    subject: str
    flags: Mapping[str, Any] = field(default_factory=dict)
    help_requested: bool = False


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    receipt: Mapping[str, Any] | None
    human_text: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def sealed_cli_environment(home: str | None = None) -> dict[str, str]:
    """Host-fixed environment for the root adapter and any child process."""

    env = {
        "PATH": HOST_VALIDATION_PATH,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    if type(home) is str and home:
        env["HOME"] = home
        env["XDG_CACHE_HOME"] = f"{home}/.cache"
        env["XDG_CONFIG_HOME"] = f"{home}/.config"
        env["XDG_DATA_HOME"] = f"{home}/.local/share"
        env["XDG_STATE_HOME"] = f"{home}/.local/state"
    return env


def typescript_bridge_plan(command: str, subject: str) -> dict[str, Any]:
    """Return the fixed TypeScript bridge invocation.  Argv is never caller-set."""

    return {
        "argv": list(TYPESCRIPT_CLI_BRIDGE_ARGV) + [command, subject],
        "command": command,
        "executed": False,
        "interface": GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE,
        "module": TYPESCRIPT_CLI_MODULE,
        "subject": subject,
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _receipt_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _looks_like_path(value: str) -> bool:
    # Registry identifiers such as ``run:interrupted-label`` match a naive
    # ``scheme:`` URI pattern. They are never filesystem paths.
    if _SAFE_ID_RE.fullmatch(value):
        return False
    return (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or _WINDOWS_DRIVE_RE.match(value) is not None
        or _URI_RE.match(value) is not None
    )


def _reject_meta(value: str, name: str) -> str:
    if _COMMAND_META_RE.search(value):
        raise GuiOptimizerCliError(
            f"{name} contains forbidden command metacharacters",
            reason_code=_Codes.COMMAND_STRING_FORBIDDEN,
            details={"field": name},
            exit_code=3,
        )
    if "\x00" in value:
        raise GuiOptimizerCliError(
            f"{name} must not contain NUL",
            reason_code=_Codes.INVALID_ARGUMENT,
            details={"field": name},
        )
    return value


def _require_identifier(value: str, name: str) -> str:
    text = _reject_meta(value, name)
    if not _SAFE_ID_RE.fullmatch(text):
        raise GuiOptimizerCliError(
            f"{name} is not a registry identifier",
            reason_code=_Codes.INVALID_ARGUMENT,
            details={"field": name, "value": value},
        )
    return text


def _allowlisted_repo_path(value: str, name: str = "path") -> str:
    raw = _reject_meta(value.replace("\\", "/"), name)
    if (
        raw.startswith("/")
        or raw.startswith("~")
        or ".." in raw.split("/")
        or _WINDOWS_DRIVE_RE.match(raw) is not None
        or _URI_RE.match(raw) is not None
    ):
        raise GuiOptimizerCliError(
            f"{name} must be a repository-relative allowlisted path",
            reason_code=_Codes.PATH_ABSOLUTE_OR_TRAVERSAL,
            details={"field": name, "value": value},
            exit_code=3,
        )
    authority = GuiPatchAuthority()
    try:
        decision = authority.evaluate_path(raw)
    except GuiAuthorityError as exc:
        raise GuiOptimizerCliError(
            str(exc),
            reason_code=exc.reason_code or _Codes.PATH_INJECTION,
            details=dict(exc.details),
            exit_code=3,
        ) from exc
    if decision.rejected or not path_under_allowed_roots(raw):
        codes = list(decision.reason_codes) or [_Codes.PATH_OUTSIDE_ALLOWED_ROOTS]
        raise GuiOptimizerCliError(
            f"{name} is outside the CLI allowlist",
            reason_code=codes[0],
            details={"field": name, "value": raw, "reason_codes": codes},
            exit_code=3,
        )
    if path_has_forbidden_segment(raw):
        raise GuiOptimizerCliError(
            f"{name} contains a forbidden path segment",
            reason_code=_Codes.PATH_INJECTION,
            details={"field": name, "value": raw},
            exit_code=3,
        )
    return raw


def resolve_target(target_id: str) -> RegisteredTarget:
    if _looks_like_path(target_id):
        raise GuiOptimizerCliError(
            "targets resolve through the repository registry and cannot be paths",
            reason_code=_Codes.PATH_INJECTION,
            details={"target": target_id},
            exit_code=3,
        )
    ident = _require_identifier(target_id, "target")
    target = TARGET_REGISTRY.get(ident)
    if target is None:
        raise GuiOptimizerCliError(
            f"unknown target: {ident}",
            reason_code=_Codes.UNKNOWN_TARGET,
            details={"target": ident, "known": sorted(TARGET_REGISTRY)},
        )
    return target


def resolve_objective(raw: str) -> RegisteredObjective:
    ident = _require_identifier(raw, "objective")
    objective = OBJECTIVE_REGISTRY.get(ident)
    if objective is None:
        raise GuiOptimizerCliError(
            f"unknown objective: {ident}",
            reason_code=_Codes.UNKNOWN_SUBJECT,
            details={"objective": ident, "known": sorted(OBJECTIVE_REGISTRY)},
        )
    return objective


def resolve_impact_subject(subject: str) -> dict[str, Any]:
    if subject in COMPONENT_REGISTRY:
        component = COMPONENT_REGISTRY[subject]
        target = TARGET_REGISTRY[component.target_id]
        return {
            "component_id": component.component_id,
            "kind": "component",
            "source_path": component.source_path,
            "target": target.to_dict(),
        }
    path = _allowlisted_repo_path(subject, "impact_subject")
    for target in TARGET_REGISTRY.values():
        if path in target.source_paths:
            return {
                "component_id": "",
                "kind": "path",
                "source_path": path,
                "target": target.to_dict(),
            }
    raise GuiOptimizerCliError(
        "impact subject is not a registered component or target path",
        reason_code=_Codes.UNKNOWN_SUBJECT,
        details={"subject": subject},
        exit_code=3,
    )


def parse_argv(argv: Sequence[str]) -> CliRequest:
    tokens = [str(item) for item in argv]
    if any(token in FORBIDDEN_CLI_FLAGS for token in tokens):
        forbidden = [token for token in tokens if token in FORBIDDEN_CLI_FLAGS]
        raise GuiOptimizerCliError(
            f"forbidden CLI flag: {forbidden[0]}",
            reason_code=_Codes.FORBIDDEN_FLAG,
            details={"flag": forbidden[0]},
            exit_code=3,
        )
    for token in tokens:
        _reject_meta(token, "argv")
    if not tokens or tokens[0] in {"--help", "-h"}:
        return CliRequest(command="help", subject="", help_requested=True)
    command = tokens[0]
    if command not in COMMAND_INTERFACES:
        raise GuiOptimizerCliError(
            f"unknown command: {command}",
            reason_code=_Codes.UNKNOWN_COMMAND,
            details={"command": command, "known": list(GUI_OPT_COMMANDS)},
        )
    allowed = COMMAND_FLAGS[command] | GLOBAL_FLAGS
    flags: dict[str, Any] = {}
    subject = ""
    rest = list(tokens[1:])
    while rest:
        token = rest.pop(0)
        if token in {"--help", "-h"}:
            return CliRequest(command=command, subject=subject, help_requested=True)
        if token == "--json":
            flags["json"] = True
            continue
        if token.startswith("-"):
            if token not in allowed:
                raise GuiOptimizerCliError(
                    f"unknown flag for {command}: {token}",
                    reason_code=_Codes.UNKNOWN_FLAG,
                    details={"command": command, "flag": token},
                )
            key = token[2:]
            if token in FLAGS_TAKING_VALUE:
                if not rest or rest[0].startswith("-"):
                    raise GuiOptimizerCliError(
                        f"{token} requires a value",
                        reason_code=_Codes.INVALID_ARGUMENT,
                        details={"flag": token},
                    )
                flags[key] = rest.pop(0)
            else:
                flags[key] = True
            continue
        if subject:
            raise GuiOptimizerCliError(
                f"{command} accepts exactly one subject",
                reason_code=_Codes.INVALID_ARGUMENT,
                details={"command": command, "extra": token},
            )
        subject = token
    if not subject:
        raise GuiOptimizerCliError(
            f"{command} requires a registry subject",
            reason_code=_Codes.MISSING_SUBJECT,
            details={"command": command},
        )
    return CliRequest(command=command, subject=subject, flags=MappingProxyType(flags))


def _make_receipt(
    *,
    command: str,
    ok: bool,
    reason_codes: Sequence[str],
    payload: Mapping[str, Any],
    exit_code: int,
    subject: str = "",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "command": command,
        "command_interface": COMMAND_INTERFACES.get(command, GUI_OPTIMIZER_CLI_INTERFACE),
        "exit_code": exit_code,
        "interface": GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE,
        "ok": ok,
        "payload": dict(payload),
        "reason_codes": list(reason_codes),
        "schema_version": GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA,
        "subject": subject,
        "version": GUI_OPTIMIZER_CLI_VERSION,
    }
    body["receipt_id"] = _receipt_digest(body)
    return body


def _error_result(exc: GuiOptimizerCliError, *, command: str = "", subject: str = "") -> CliResult:
    receipt = _make_receipt(
        command=command or "unknown",
        ok=False,
        reason_codes=(exc.reason_code,),
        payload={"details": dict(exc.details), "message": str(exc)},
        exit_code=exc.exit_code,
        subject=subject,
    )
    return CliResult(
        exit_code=exc.exit_code,
        receipt=receipt,
        human_text=f"{exc.reason_code}: {exc} [{receipt['receipt_id']}]",
        reason_codes=(exc.reason_code,),
    )


def _ok(
    command: str,
    subject: str,
    reason_codes: Sequence[str],
    payload: Mapping[str, Any],
) -> CliResult:
    receipt = _make_receipt(
        command=command,
        ok=True,
        reason_codes=reason_codes,
        payload=payload,
        exit_code=0,
        subject=subject,
    )
    return CliResult(
        exit_code=0,
        receipt=receipt,
        human_text=f"{command} {subject} ok [{receipt['receipt_id']}]",
        reason_codes=tuple(reason_codes),
    )


def _fail(
    command: str,
    subject: str,
    reason_codes: Sequence[str],
    payload: Mapping[str, Any],
    exit_code: int = 4,
) -> CliResult:
    receipt = _make_receipt(
        command=command,
        ok=False,
        reason_codes=reason_codes,
        payload=payload,
        exit_code=exit_code,
        subject=subject,
    )
    return CliResult(
        exit_code=exit_code,
        receipt=receipt,
        human_text=f"{command} {subject} failed [{receipt['receipt_id']}]",
        reason_codes=tuple(reason_codes),
    )


def _read_json_if_present(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuiOptimizerCliError(
            "durable receipt is not valid JSON",
            reason_code=_Codes.RECEIPT_INVALID,
            details={"path": str(path), "error": str(exc)},
            exit_code=4,
        ) from exc


def _cmd_scan(request: CliRequest, repo_root: Path) -> CliResult:
    target = resolve_target(request.subject)
    sources = []
    for relative in target.source_paths:
        path = repo_root / relative
        digest = ""
        present = path.is_file()
        if present:
            digest = (
                "sha256:"
                + hashlib.sha256(path.read_bytes()).hexdigest()
            )
        sources.append(
            {"digest": digest, "path": relative, "present": present}
        )
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
        {
            "effectful": False,
            "sources": sources,
            "target": target.to_dict(),
            "typescript_bridge": typescript_bridge_plan(
                request.command, request.subject
            ),
        },
    )


def _cmd_baseline(request: CliRequest, repo_root: Path) -> CliResult:
    target = resolve_target(request.subject)
    seed = {
        "application_id": target.application_id,
        "command": "baseline",
        "screen_id": target.screen_id,
        "source_paths": list(target.source_paths),
        "target_id": target.target_id,
    }
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
        {
            "baseline_seed": seed,
            "baseline_seed_digest": _receipt_digest(seed),
            "effectful": False,
            "target": target.to_dict(),
            "typescript_bridge": typescript_bridge_plan(
                request.command, request.subject
            ),
        },
    )


def _cmd_impact(request: CliRequest, repo_root: Path) -> CliResult:
    resolved = resolve_impact_subject(request.subject)
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
        {
            "effectful": False,
            "impact": resolved,
            "typescript_bridge": typescript_bridge_plan(
                request.command, request.subject
            ),
        },
    )


def _cmd_evaluate(request: CliRequest, repo_root: Path) -> CliResult:
    target = resolve_target(request.subject)
    payload: dict[str, Any] = {
        "effectful": False,
        "target": target.to_dict(),
        "typescript_bridge": typescript_bridge_plan(
            request.command, request.subject
        ),
    }
    benchmark_id = request.flags.get("benchmark")
    if benchmark_id is None:
        return _ok(
            request.command,
            request.subject,
            (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
            payload,
        )
    ident = _require_identifier(str(benchmark_id), "benchmark")
    spec = BENCHMARK_REGISTRY.get(ident)
    if spec is None:
        raise GuiOptimizerCliError(
            f"unknown benchmark: {ident}",
            reason_code=_Codes.UNKNOWN_SUBJECT,
            details={"benchmark": ident, "known": sorted(BENCHMARK_REGISTRY)},
        )
    expected = spec.expected_tasks
    if "expected-tasks" in request.flags:
        try:
            expected = int(str(request.flags["expected-tasks"]))
        except ValueError as exc:
            raise GuiOptimizerCliError(
                "--expected-tasks must be an integer",
                reason_code=_Codes.INVALID_ARGUMENT,
            ) from exc
        if expected != spec.expected_tasks:
            return _fail(
                request.command,
                request.subject,
                (_Codes.BENCHMARK_TASK_COUNT_MISMATCH,),
                {
                    **payload,
                    "benchmark": spec.benchmark_id,
                    "expected_tasks": spec.expected_tasks,
                    "requested_tasks": expected,
                },
            )
    catalog = _read_json_if_present(repo_root / spec.catalog_path)
    payload["benchmark"] = {
        "catalog_path": spec.catalog_path,
        "catalog_present": catalog is not None,
        "expected_tasks": spec.expected_tasks,
        "id": spec.benchmark_id,
        "progress_interval_seconds": request.flags.get(
            "progress-interval-seconds", ""
        ),
    }
    if catalog is None:
        return _fail(
            request.command,
            request.subject,
            (_Codes.BENCHMARK_CATALOG_UNAVAILABLE,),
            payload,
        )
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
        payload,
    )


def _cmd_pack_context(request: CliRequest, repo_root: Path) -> CliResult:
    target = resolve_target(request.subject)
    raw_objective = request.flags.get("objective")
    if not raw_objective:
        raise GuiOptimizerCliError(
            "pack-context requires --objective",
            reason_code=_Codes.MISSING_OBJECTIVE,
            details={"command": "pack-context"},
        )
    objective = resolve_objective(str(raw_objective))
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED, _Codes.BRIDGE_PLANNED),
        {
            "effectful": False,
            "objective": {
                "label": objective.label,
                "metric_id": objective.metric_id,
                "objective_id": objective.objective_id,
            },
            "target": target.to_dict(),
            "typescript_bridge": typescript_bridge_plan(
                request.command, request.subject
            ),
        },
    )


def _cmd_verify(request: CliRequest, repo_root: Path) -> CliResult:
    subject = request.subject
    alias = VERIFY_ALIAS_REGISTRY.get(subject)
    receipt_flag = request.flags.get("receipt")
    if receipt_flag is not None:
        receipt_rel = _allowlisted_repo_path(str(receipt_flag), "receipt")
    elif alias is not None:
        receipt_rel = alias.default_receipt
    else:
        if _looks_like_path(subject):
            receipt_rel = _allowlisted_repo_path(subject, "verify_subject")
        else:
            _require_identifier(subject, "verify_subject")
            return _fail(
                request.command,
                subject,
                (_Codes.ISOLATED_WORKTREE_REQUIRED, _Codes.MISSING_RECEIPT),
                {
                    "effectful": False,
                    "isolated_worktree_required": True,
                    "message": (
                        "verify of an unregistered worktree/patch requires an "
                        "isolated worktree alias or an explicit --receipt"
                    ),
                    "subject_kind": "worktree_or_patch",
                },
            )
    full = bool(request.flags.get("full")) or (alias is not None and alias.full)
    document = _read_json_if_present(repo_root / receipt_rel)
    payload = {
        "alias": None
        if alias is None
        else {
            "alias_id": alias.alias_id,
            "full": alias.full,
            "kind": alias.kind,
            "target_id": alias.target_id,
        },
        "effectful": False,
        "full": full,
        "isolated_worktree_required": True
        if alias is None
        else alias.isolated_worktree_required,
        "receipt_path": receipt_rel,
        "receipt_present": document is not None,
    }
    if document is None:
        return _fail(
            request.command,
            subject,
            (_Codes.MISSING_RECEIPT,),
            payload,
        )
    if type(document) is not dict:
        return _fail(
            request.command,
            subject,
            (_Codes.RECEIPT_INVALID,),
            {**payload, "message": "receipt must be a JSON object"},
        )
    payload["receipt_keys"] = sorted(str(key) for key in document)
    return _ok(
        request.command,
        subject,
        (_Codes.ALIAS_RESOLVED if alias is not None else _Codes.OK,),
        payload,
    )


def _cmd_improve(request: CliRequest, repo_root: Path) -> CliResult:
    target = resolve_target(request.subject)
    raw_objective = request.flags.get("objective")
    if not raw_objective:
        raise GuiOptimizerCliError(
            "improve requires --objective",
            reason_code=_Codes.MISSING_OBJECTIVE,
            details={"command": "improve"},
        )
    objective = resolve_objective(str(raw_objective))
    isolated = bool(request.flags.get("isolated"))
    request_path = request.flags.get("request")
    payload: dict[str, Any] = {
        "canonical_merge": False,
        "effectful_default": False,
        "isolated": isolated,
        "objective": {
            "label": objective.label,
            "metric_id": objective.metric_id,
            "objective_id": objective.objective_id,
        },
        "production_defaults": False,
        "target": target.to_dict(),
    }
    if request_path is not None:
        relative = _allowlisted_repo_path(str(request_path), "request")
        document = _read_json_if_present(repo_root / relative)
        payload["request_path"] = relative
        payload["request_present"] = document is not None
        if document is None:
            return _fail(
                request.command,
                request.subject,
                (_Codes.MISSING_EVIDENCE,),
                payload,
            )
    if not isolated:
        return _fail(
            request.command,
            request.subject,
            (_Codes.ISOLATED_WORKTREE_REQUIRED, _Codes.NO_PRODUCTION_DEFAULTS),
            {
                **payload,
                "message": (
                    "improve refuses canonical-tree and production defaults; "
                    "pass --isolated and an explicit request"
                ),
            },
        )
    if request_path is None:
        return _fail(
            request.command,
            request.subject,
            (_Codes.NO_PRODUCTION_DEFAULTS, _Codes.MISSING_EVIDENCE),
            {
                **payload,
                "message": "improve requires an explicit --request payload",
            },
        )
    return _ok(
        request.command,
        request.subject,
        (_Codes.TARGET_RESOLVED,),
        payload,
    )


def _validate_reported_document(
    document: Mapping[str, Any],
    *,
    expected_tasks: int | None,
    verify_receipts: bool,
) -> list[str]:
    reasons: list[str] = []
    if expected_tasks is not None:
        tasks = document.get("tasks")
        expected_field = document.get("expected_tasks")
        count = None
        if type(tasks) is list:
            count = len(tasks)
        elif type(expected_field) is int:
            count = expected_field
        elif type(document.get("task_count")) is int:
            count = document["task_count"]
        if count != expected_tasks:
            reasons.append(_Codes.BENCHMARK_TASK_COUNT_MISMATCH)
    if verify_receipts:
        digest = document.get("receipt_id") or document.get("digest")
        if type(digest) is str and digest.startswith("sha256:"):
            clone = {key: value for key, value in document.items() if key != "receipt_id"}
            recomputed = _receipt_digest(clone) if "schema_version" in clone else digest
            if digest != recomputed and "receipt_id" in document:
                body = dict(document)
                body.pop("receipt_id", None)
                if _receipt_digest(body) != digest:
                    reasons.append(_Codes.RECEIPT_INVALID)
        status = document.get("verification_status")
        if status in {"stale", "invalid", "simulated"}:
            reasons.append(_Codes.RECEIPT_INVALID)
    return reasons


def _cmd_report(
    request: CliRequest,
    repo_root: Path,
    journal: GuiRunJournal | None,
) -> CliResult:
    subject = request.subject
    require_complete = bool(request.flags.get("require-complete"))
    verify_receipts = bool(request.flags.get("verify-receipts"))
    expected_tasks: int | None = None
    if "expected-tasks" in request.flags:
        try:
            expected_tasks = int(str(request.flags["expected-tasks"]))
        except ValueError as exc:
            raise GuiOptimizerCliError(
                "--expected-tasks must be an integer",
                reason_code=_Codes.INVALID_ARGUMENT,
            ) from exc

    alias = REPORT_ALIAS_REGISTRY.get(subject)
    if alias is not None:
        if expected_tasks is None:
            expected_tasks = alias.expected_tasks
        path = repo_root / alias.receipt_path
        document = _read_json_if_present(path)
        payload = {
            "alias": {
                "alias_id": alias.alias_id,
                "expected_tasks": alias.expected_tasks,
                "kind": alias.kind,
                "receipt_path": alias.receipt_path,
                "report_path": alias.report_path,
            },
            "effectful": False,
            "receipt_present": document is not None,
            "require_complete": require_complete,
            "verify_receipts": verify_receipts,
        }
        if document is None:
            code = (
                _Codes.INCOMPLETE_EVIDENCE
                if require_complete
                else _Codes.MISSING_EVIDENCE
            )
            return _fail(request.command, subject, (code,), payload)
        if type(document) is not dict:
            return _fail(
                request.command,
                subject,
                (_Codes.RECEIPT_INVALID,),
                payload,
            )
        extra = _validate_reported_document(
            document,
            expected_tasks=expected_tasks,
            verify_receipts=verify_receipts,
        )
        payload["receipt_keys"] = sorted(str(key) for key in document)
        if extra:
            return _fail(request.command, subject, tuple(extra), payload)
        return _ok(
            request.command,
            subject,
            (_Codes.ALIAS_RESOLVED,),
            payload,
        )

    if _looks_like_path(subject):
        raise GuiOptimizerCliError(
            "report run IDs are registry identifiers, not paths",
            reason_code=_Codes.PATH_INJECTION,
            details={"subject": subject},
            exit_code=3,
        )
    run_id = _require_identifier(subject, "run_id")
    if journal is None:
        return _fail(
            request.command,
            subject,
            (_Codes.JOURNAL_MISSING,),
            {
                "effectful": False,
                "message": "no host journal is bound for this report",
                "run_id": run_id,
            },
        )
    try:
        checkpoint = journal.load_checkpoint(run_id)
    except GuiRunJournalError as exc:
        code = exc.reason_code or _Codes.JOURNAL_CORRUPT
        if code == JournalReasonCode.MISSING_RUN.value:
            code = _Codes.JOURNAL_MISSING
        return _fail(
            request.command,
            subject,
            (code,),
            {"effectful": False, "run_id": run_id, "message": str(exc)},
        )
    if checkpoint is None:
        return _fail(
            request.command,
            subject,
            (_Codes.JOURNAL_MISSING,),
            {"effectful": False, "run_id": run_id},
        )
    status = (
        checkpoint.status.value
        if type(checkpoint.status) is RunStatus
        else str(checkpoint.status)
    )
    payload = {
        "application_id": checkpoint.application_id,
        "checkpoint_cid": checkpoint.cid,
        "effectful": False,
        "objective_id": checkpoint.objective_id,
        "phase": (
            checkpoint.phase.value
            if hasattr(checkpoint.phase, "value")
            else str(checkpoint.phase)
        ),
        "process_exit_is_completion": False,
        "run_id": checkpoint.run_id,
        "screen_id": checkpoint.screen_id,
        "status": status,
        "terminal_receipt_cid": checkpoint.terminal_receipt_cid,
    }
    if status in {RunStatus.INTERRUPTED.value, RunStatus.OPEN.value, RunStatus.IN_PROGRESS.value}:
        codes = [_Codes.INTERRUPTED, _Codes.PROCESS_EXIT_NOT_COMPLETION]
        if require_complete:
            return _fail(request.command, subject, tuple(codes), payload)
        return _ok(request.command, subject, tuple(codes), payload)
    if require_complete and status not in {
        RunStatus.COMPLETED.value,
        RunStatus.REJECTED.value,
        RunStatus.FAILED.value,
    }:
        return _fail(
            request.command,
            subject,
            (_Codes.INCOMPLETE_EVIDENCE,),
            payload,
        )
    if require_complete and status == RunStatus.COMPLETED.value and not checkpoint.terminal_receipt_cid:
        return _fail(
            request.command,
            subject,
            (_Codes.INCOMPLETE_EVIDENCE,),
            payload,
        )
    return _ok(request.command, subject, (_Codes.OK,), payload)


def default_repo_root() -> Path:
    """Locate the superproject root from this module's fixed path."""

    here = Path(__file__).resolve()
    candidate = here.parents[5]
    if (candidate / "scripts" / "gui-opt").is_file():
        return candidate
    return Path.cwd()


def run_cli(
    argv: Sequence[str],
    *,
    repo_root: Path | str | None = None,
    journal: GuiRunJournal | None = None,
    host_root: Path | str | None = None,
) -> CliResult:
    """Parse and execute a fixed ``gui-opt`` argument vector."""

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    bound_journal = journal
    if bound_journal is None and host_root is not None:
        bound_journal = default_run_journal(host_root)
    try:
        request = parse_argv(argv)
        if request.help_requested:
            return CliResult(
                exit_code=0,
                receipt=None,
                human_text=HELP_TEXT,
                reason_codes=(_Codes.HELP,),
            )
        if request.command == "scan":
            return _cmd_scan(request, root)
        if request.command == "baseline":
            return _cmd_baseline(request, root)
        if request.command == "impact":
            return _cmd_impact(request, root)
        if request.command == "evaluate":
            return _cmd_evaluate(request, root)
        if request.command == "pack-context":
            return _cmd_pack_context(request, root)
        if request.command == "verify":
            return _cmd_verify(request, root)
        if request.command == "improve":
            return _cmd_improve(request, root)
        if request.command == "report":
            return _cmd_report(request, root, bound_journal)
        raise GuiOptimizerCliError(
            f"unknown command: {request.command}",
            reason_code=_Codes.UNKNOWN_COMMAND,
        )
    except GuiOptimizerCliError as exc:
        command = ""
        subject = ""
        try:
            # Best-effort context when parse succeeded enough to know the verb.
            if argv:
                command = str(argv[0])
            if len(argv) > 1 and not str(argv[1]).startswith("-"):
                subject = str(argv[1])
        except Exception:
            pass
        return _error_result(exc, command=command, subject=subject)


def main(argv: Sequence[str] | None = None) -> int:
    """Process entrypoint used by ``scripts/gui-opt``."""

    args = list(sys.argv[1:] if argv is None else argv)
    result = run_cli(args)
    if result.receipt is None:
        sys.stdout.write(result.human_text)
        if result.human_text and not result.human_text.endswith("\n"):
            sys.stdout.write("\n")
        return result.exit_code
    sys.stdout.write(
        json.dumps(dict(result.receipt), sort_keys=True, indent=2, ensure_ascii=True)
    )
    sys.stdout.write("\n")
    if result.human_text:
        sys.stderr.write(result.human_text)
        if not result.human_text.endswith("\n"):
            sys.stderr.write("\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BENCHMARK_REGISTRY",
    "COMMAND_INTERFACES",
    "COMPONENT_REGISTRY",
    "GUI_OPTIMIZER_CLI_INTERFACE",
    "GUI_OPTIMIZER_CLI_RECEIPT_INTERFACE",
    "GUI_OPTIMIZER_CLI_RECEIPT_SCHEMA",
    "GUI_OPTIMIZER_CLI_SCHEMA",
    "GUI_OPTIMIZER_CLI_VERSION",
    "GUI_OPTIMIZER_TYPESCRIPT_CLI_BRIDGE_INTERFACE",
    "GUI_OPT_BASELINE_INTERFACE",
    "GUI_OPT_COMMANDS",
    "GUI_OPT_EVALUATE_INTERFACE",
    "GUI_OPT_IMPACT_INTERFACE",
    "GUI_OPT_IMPROVE_INTERFACE",
    "GUI_OPT_PACK_CONTEXT_INTERFACE",
    "GUI_OPT_REPORT_INTERFACE",
    "GUI_OPT_SCAN_INTERFACE",
    "GUI_OPT_VERIFY_INTERFACE",
    "HELP_TEXT",
    "HOST_PYTHON_EXECUTABLE",
    "HOST_VALIDATION_PATH",
    "OBJECTIVE_REGISTRY",
    "REPORT_ALIAS_REGISTRY",
    "TARGET_REGISTRY",
    "TYPESCRIPT_CLI_BRIDGE_ARGV",
    "TYPESCRIPT_CLI_MODULE",
    "VERIFY_ALIAS_REGISTRY",
    "CliRequest",
    "CliResult",
    "GuiOptimizerCliError",
    "RegisteredTarget",
    "default_repo_root",
    "main",
    "parse_argv",
    "resolve_target",
    "run_cli",
    "sealed_cli_environment",
    "typescript_bridge_plan",
)

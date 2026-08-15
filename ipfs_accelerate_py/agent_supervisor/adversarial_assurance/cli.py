"""Adversarial Assurance Engine host CLI (AAE-056).

Registers the parser-only ``assurance`` group on the product
``ipfs-accelerate`` host and dispatches campaign commands lazily:

* ``assurance mutate plan|run|target|explain``
* ``assurance report``

Importing this module performs no I/O, starts no processes or network
activity, and does not construct campaign APIs. Product logic stays in the
AAE public API (``AssuranceCampaignApi`` / leaf modules). Dispatch is fail
closed: unknown commands, absolute host-path exposure, missing run authority,
and cancelled/resource-exceeded states produce typed envelopes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, TextIO

# ---------------------------------------------------------------------------
# Interface / evidence pins (stdlib only at import)
# ---------------------------------------------------------------------------

AAE_CLI_EVIDENCE: Final[str] = "aae/cli@1"
ASSURANCE_CLI_INTERFACE: Final[str] = "AssuranceCLI@1"
ASSURANCE_CLI_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-cli@1"
)
ASSURANCE_CLI_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-cli-result@1"
)
ASSURANCE_GROUP: Final[str] = "assurance"
CONSOLE_ENTRY: Final[str] = "ipfs-accelerate"

EXIT_SUCCESS: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXIT_UNAVAILABLE: Final[int] = 3
EXIT_CANCELLED: Final[int] = 4
EXIT_AUTHORITY: Final[int] = 5
EXIT_RESOURCE: Final[int] = 6

MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_JSON_BYTES: Final[int] = 4_194_304
MAX_HUMAN_LINES: Final[int] = 256
MAX_OUTPUT_CHARS: Final[int] = 262_144

MUTATE_SUBCOMMANDS: Final[tuple[str, ...]] = (
    "plan",
    "run",
    "target",
    "explain",
)
TOP_LEVEL_COMMANDS: Final[tuple[str, ...]] = ("mutate", "report")
# Closed vocabulary of dispatch keys used by the campaign handlers.
CAMPAIGN_COMMANDS: Final[tuple[str, ...]] = (
    "mutate.plan",
    "mutate.run",
    "mutate.target",
    "mutate.explain",
    "report",
)

_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
_HOME_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/home/",
    "/Users/",
    "\\Users\\",
    "/tmp/",
    "\\Temp\\",
    "C:\\",
    "c:\\",
)

# Host flags that must never admit arbitrary external repository roots.
_FORBIDDEN_HOST_ROOT_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "repository",
        "repository_root",
        "repo_root",
        "repo",
        "workdir",
        "worktree",
        "worktree_path",
        "absolute_path",
        "host_path",
        "local_path",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssuranceCLIError(ValueError):
    """Typed CLI failure before or during campaign dispatch."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "cli_error",
        exit_code: int = EXIT_ERROR,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "cli_error")
        self.exit_code = int(exit_code)
        self.details = dict(details or {})


class AssuranceCLIUsageError(AssuranceCLIError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "usage_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=reason_code,
            exit_code=EXIT_USAGE,
            details=details,
        )


class AssuranceCLIAuthorityError(AssuranceCLIError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "run_authority_required",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=reason_code,
            exit_code=EXIT_AUTHORITY,
            details=details,
        )


class AssuranceCLICancelledError(AssuranceCLIError):
    def __init__(
        self,
        message: str = "operation cancelled",
        *,
        reason_code: str = "cancelled",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=reason_code,
            exit_code=EXIT_CANCELLED,
            details=details,
        )


class AssuranceCLIResourceError(AssuranceCLIError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "resource_exceeded",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=reason_code,
            exit_code=EXIT_RESOURCE,
            details=details,
        )


class AssuranceCLIPathError(AssuranceCLIError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "path_exposure",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=reason_code,
            exit_code=EXIT_USAGE,
            details=details,
        )


# ---------------------------------------------------------------------------
# Cancellation / resources
# ---------------------------------------------------------------------------


@dataclass
class CliResourceBudget:
    """Bounded CLI resource envelope (seconds / candidates / worktrees)."""

    timeout_seconds: int | None = None
    max_candidates: int | None = None
    max_worktrees: int | None = None
    max_json_bytes: int = MAX_JSON_BYTES

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None:
            value = int(self.timeout_seconds)
            if value < 0 or value > 86_400:
                raise AssuranceCLIUsageError(
                    "timeout_seconds must be in [0, 86400]",
                    reason_code="invalid_timeout",
                )
            self.timeout_seconds = value
        if self.max_candidates is not None:
            value = int(self.max_candidates)
            if value < 0 or value > 65_536:
                raise AssuranceCLIUsageError(
                    "max_candidates must be in [0, 65536]",
                    reason_code="invalid_max_candidates",
                )
            self.max_candidates = value
        if self.max_worktrees is not None:
            value = int(self.max_worktrees)
            if value < 0 or value > 1_024:
                raise AssuranceCLIUsageError(
                    "max_worktrees must be in [0, 1024]",
                    reason_code="invalid_max_worktrees",
                )
            self.max_worktrees = value
        self.max_json_bytes = max(1, min(int(self.max_json_bytes), MAX_JSON_BYTES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_candidates": self.max_candidates,
            "max_worktrees": self.max_worktrees,
            "max_json_bytes": self.max_json_bytes,
        }


class CliCancellationToken:
    """Cooperative cancellation token for CLI dispatch."""

    def __init__(self, *, cancellation_id: str | None = None) -> None:
        self.cancellation_id = str(cancellation_id or "aae-cli-cancel").strip()
        self._cancelled = False
        self._reason = ""

    def cancel(self, *, reason: str = "cancelled") -> None:
        self._cancelled = True
        self._reason = str(reason or "cancelled")

    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def check(self) -> None:
        if self._cancelled:
            raise AssuranceCLICancelledError(
                self._reason or "operation cancelled",
                details={"cancellation_id": self.cancellation_id},
            )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)] + "..."


def looks_like_host_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _ABSOLUTE_PATH_RE.match(text):
        return True
    for marker in _HOME_PATH_MARKERS:
        if marker in text:
            return True
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        return True
    return False


def reject_path_exposure(value: Any, *, path: str = "$") -> None:
    """Reject absolute host paths on public CLI inputs (fail closed)."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered in _FORBIDDEN_HOST_ROOT_FLAGS or lowered.endswith("_path"):
                if isinstance(item, str) and looks_like_host_path(item):
                    raise AssuranceCLIPathError(
                        f"{path}.{key_str} exposes an absolute host path or external root",
                        details={"field": f"{path}.{key_str}"},
                    )
            reject_path_exposure(item, path=f"{path}.{key_str}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_path_exposure(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and looks_like_host_path(value):
        raise AssuranceCLIPathError(
            f"{path} exposes an absolute host path",
            details={"field": path},
        )


def repo_relative_path(value: Any, name: str) -> str:
    """Normalize a repository-relative path; reject absolute host paths."""

    text = str(value or "").strip()
    if not text:
        raise AssuranceCLIUsageError(f"{name} must be a non-empty path")
    if looks_like_host_path(text):
        raise AssuranceCLIPathError(
            f"{name} must be repository-relative, not an absolute host path",
            details={"field": name},
        )
    normalized = str(PurePosixPath(text))
    if normalized.startswith("..") or "/../" in f"/{normalized}/":
        raise AssuranceCLIPathError(
            f"{name} must not escape the repository via '..'",
            details={"field": name},
        )
    if normalized.startswith("/"):
        raise AssuranceCLIPathError(
            f"{name} must be repository-relative",
            details={"field": name},
        )
    return normalized


def project_result(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    """Project API results into JSON-safe, path-scrubbed structures."""

    if depth > max_depth:
        return _clip(repr(value), limit=128)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if looks_like_host_path(value):
            return "<redacted-host-path>"
        return _clip(value, limit=MAX_DIAGNOSTIC)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = project_result(item, depth=depth + 1, max_depth=max_depth)
        return out
    if isinstance(value, (list, tuple)):
        return [
            project_result(item, depth=depth + 1, max_depth=max_depth) for item in value
        ]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return project_result(value.to_dict(), depth=depth + 1, max_depth=max_depth)
        except Exception:  # noqa: BLE001 - projection must not raise raw API noise
            return _clip(repr(value), limit=256)
    if hasattr(value, "to_canonical") and callable(value.to_canonical):
        try:
            return project_result(
                value.to_canonical(), depth=depth + 1, max_depth=max_depth
            )
        except Exception:  # noqa: BLE001
            return _clip(repr(value), limit=256)
    return _clip(repr(value), limit=256)


def envelope(
    *,
    ok: bool,
    command: str,
    status: str,
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
    reason_code: str | None = None,
    exit_code: int = EXIT_SUCCESS,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": ASSURANCE_CLI_RESULT_SCHEMA,
        "interface": ASSURANCE_CLI_INTERFACE,
        "evidence": AAE_CLI_EVIDENCE,
        "ok": bool(ok),
        "command": str(command),
        "status": str(status),
        "exit_code": int(exit_code),
        "production_policy_change": False,
        "path_exposure": False,
        "side_effects": {
            "network": False,
            "process_spawn": False,
            "key_generation": False,
            "production_policy_change": False,
        },
    }
    if result is not None:
        body["result"] = project_result(dict(result))
    if error is not None:
        body["error"] = _clip(error)
    if reason_code is not None:
        body["reason_code"] = str(reason_code)
    if details:
        body["details"] = project_result(dict(details))
    return body


def emit(
    payload: Mapping[str, Any],
    *,
    output_json: bool,
    stream: TextIO,
) -> None:
    """Write bounded deterministic JSON or human summary."""

    if output_json:
        text = json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=True)
        if len(text) > MAX_OUTPUT_CHARS:
            truncated = {
                "schema": ASSURANCE_CLI_RESULT_SCHEMA,
                "interface": ASSURANCE_CLI_INTERFACE,
                "evidence": AAE_CLI_EVIDENCE,
                "ok": payload.get("ok"),
                "command": payload.get("command"),
                "status": payload.get("status"),
                "exit_code": payload.get("exit_code"),
                "reason_code": "output_truncated",
                "error": "output exceeded bound; truncated",
                "production_policy_change": False,
                "path_exposure": False,
            }
            text = json.dumps(truncated, sort_keys=True, indent=2, ensure_ascii=True)
        if not text.endswith("\n"):
            text += "\n"
        stream.write(text)
        stream.flush()
        return

    lines: list[str] = []
    ok = bool(payload.get("ok"))
    command = payload.get("command")
    status = payload.get("status")
    lines.append(f"assurance {command}: {'ok' if ok else 'error'} ({status})")
    if payload.get("reason_code"):
        lines.append(f"reason_code={payload['reason_code']}")
    if payload.get("error"):
        lines.append(f"error={_clip(str(payload['error']))}")
    result = payload.get("result")
    if isinstance(result, Mapping):
        for key in (
            "plan_id",
            "plan_cid",
            "result_cid",
            "report_cid",
            "terminal_status",
            "killed_count",
            "survivor_count",
            "invalid_count",
            "inconclusive_count",
            "target_count",
            "candidate_count",
            "summary",
        ):
            if key in result and result[key] is not None:
                lines.append(f"{key}={result[key]}")
        reasons = result.get("reason_codes")
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
            clipped = ", ".join(str(item) for item in list(reasons)[:16])
            if clipped:
                lines.append(f"reason_codes={clipped}")
    text = "\n".join(lines[:MAX_HUMAN_LINES]) + "\n"
    stream.write(text)
    stream.flush()


def load_json_mapping(
    path: str | Path | None,
    *,
    field: str,
    budget: CliResourceBudget | None = None,
    required: bool = True,
) -> dict[str, Any] | None:
    """Load a JSON object from a local file path (not an external repo root)."""

    if path is None:
        if required:
            raise AssuranceCLIUsageError(f"{field} is required")
        return None
    file_path = Path(str(path))
    # Absolute paths for *input files* are allowed only as local CLI I/O
    # (tmp fixtures / pipes). Payload *content* still rejects host roots.
    if not file_path.is_file():
        raise AssuranceCLIUsageError(
            f"{field} not found: {file_path.name}",
            reason_code="input_not_found",
            details={"field": field},
        )
    limit = (budget.max_json_bytes if budget else MAX_JSON_BYTES)
    raw = file_path.read_bytes()
    if len(raw) > limit:
        raise AssuranceCLIResourceError(
            f"{field} exceeds max_json_bytes={limit}",
            details={"field": field, "size": len(raw), "limit": limit},
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssuranceCLIUsageError(
            f"{field} is not valid UTF-8 JSON: {_clip(str(exc), limit=256)}",
            reason_code="invalid_json",
            details={"field": field},
        ) from exc
    if not isinstance(payload, Mapping):
        raise AssuranceCLIUsageError(
            f"{field} must be a JSON object",
            reason_code="invalid_json_type",
            details={"field": field},
        )
    data = dict(payload)
    reject_path_exposure(data, path=field)
    return data


def load_json_value(
    path: str | Path | None,
    *,
    field: str,
    budget: CliResourceBudget | None = None,
    required: bool = True,
) -> Any:
    if path is None:
        if required:
            raise AssuranceCLIUsageError(f"{field} is required")
        return None
    file_path = Path(str(path))
    if not file_path.is_file():
        raise AssuranceCLIUsageError(
            f"{field} not found: {file_path.name}",
            reason_code="input_not_found",
            details={"field": field},
        )
    limit = (budget.max_json_bytes if budget else MAX_JSON_BYTES)
    raw = file_path.read_bytes()
    if len(raw) > limit:
        raise AssuranceCLIResourceError(
            f"{field} exceeds max_json_bytes={limit}",
            details={"field": field, "size": len(raw), "limit": limit},
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssuranceCLIUsageError(
            f"{field} is not valid UTF-8 JSON: {_clip(str(exc), limit=256)}",
            reason_code="invalid_json",
            details={"field": field},
        ) from exc
    reject_path_exposure(payload, path=field)
    return payload


def read_stdin_json(
    stdin: TextIO,
    *,
    field: str = "stdin",
    budget: CliResourceBudget | None = None,
) -> Any:
    limit = (budget.max_json_bytes if budget else MAX_JSON_BYTES)
    raw = stdin.read(limit + 1)
    if len(raw) > limit:
        raise AssuranceCLIResourceError(
            f"{field} exceeds max_json_bytes={limit}",
            details={"field": field, "limit": limit},
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssuranceCLIUsageError(
            f"{field} is not valid JSON: {_clip(str(exc), limit=256)}",
            reason_code="invalid_json",
        ) from exc
    reject_path_exposure(payload, path=field)
    return payload


def resource_budget_from_args(args: argparse.Namespace) -> CliResourceBudget:
    return CliResourceBudget(
        timeout_seconds=getattr(args, "timeout_seconds", None),
        max_candidates=getattr(args, "max_candidates", None),
        max_worktrees=getattr(args, "max_worktrees", None),
    )


def cancellation_from_args(
    args: argparse.Namespace,
    *,
    token: CliCancellationToken | None = None,
) -> CliCancellationToken:
    active = token if token is not None else CliCancellationToken()
    cancel_flag = bool(getattr(args, "cancel", False))
    cancel_file = getattr(args, "cancel_file", None)
    if cancel_flag:
        active.cancel(reason="cancelled_by_flag")
    if cancel_file is not None:
        path = Path(str(cancel_file))
        if path.is_file():
            try:
                body = path.read_text(encoding="utf-8").strip()
            except OSError:
                body = "cancelled"
            if body == "" or body.lower() in {"1", "true", "yes", "cancel", "cancelled"}:
                active.cancel(reason="cancelled_by_file")
            else:
                # Non-empty custom reason still means cancelled when file exists.
                active.cancel(reason=_clip(body, limit=256))
    return active


def resolve_dispatch_command(args: argparse.Namespace) -> str:
    """Map parsed host args to a closed campaign dispatch key."""

    assurance_command = getattr(args, "assurance_command", None)
    if assurance_command == "report":
        return "report"
    if assurance_command == "mutate":
        mutate_command = getattr(args, "assurance_mutate_command", None)
        if mutate_command in MUTATE_SUBCOMMANDS:
            return f"mutate.{mutate_command}"
        raise AssuranceCLIUsageError(
            "assurance mutate requires one of: plan, run, target, explain",
            reason_code="missing_mutate_command",
        )
    raise AssuranceCLIUsageError(
        "assurance requires a subcommand: mutate or report",
        reason_code="missing_assurance_command",
    )


# ---------------------------------------------------------------------------
# Registration (parser-only)
# ---------------------------------------------------------------------------


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-json",
        action="store_true",
        default=True,
        help="Emit a deterministic JSON envelope (default).",
    )
    parser.add_argument(
        "--output-human",
        action="store_true",
        help="Emit a bounded human summary instead of JSON.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Hard wall-clock budget for the operation (honored by the CLI gate).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Upper bound on candidates admitted for this invocation.",
    )
    parser.add_argument(
        "--max-worktrees",
        type=int,
        default=None,
        help="Upper bound on disposable worktrees for this invocation.",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Treat the invocation as already cancelled (cooperative cancel gate).",
    )
    parser.add_argument(
        "--cancel-file",
        help="If this file exists, cancel cooperatively before dispatch.",
    )


def _add_json_input(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    help_text: str,
    required: bool = False,
) -> None:
    parser.add_argument(
        *flags,
        dest=dest,
        required=required,
        help=help_text,
    )


def register_assurance_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the lightweight ``assurance`` group (parser-only, cold-safe)."""

    group = subparsers.add_parser(
        ASSURANCE_GROUP,
        help="Adversarial assurance campaign CLI (mutate plan/run/target/explain, report).",
        description=(
            "Hermetic adversarial-assurance commands. Inputs are typed JSON "
            "identity bindings and sealed artifacts — never arbitrary external "
            "repository roots or host paths. Mutate run requires explicit "
            "--authorize-run authority. Product logic is lazy-loaded from the "
            "AAE public API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ipfs-accelerate assurance mutate plan --repository-state-json state.json "
            "--manifest-json manifest.json --policy-json policy.json "
            "--resource-budget-json budget.json --baseline-json baseline.json\n"
            "  ipfs-accelerate assurance mutate run --plan-json plan.json "
            "--verification-policy-json policy.json --precomputed-reports-json reports.json "
            "--authorize-run\n"
            "  ipfs-accelerate assurance mutate target --properties-json props.json "
            "--repository-id repository:sha256:… --repository-state-cid bagu…\n"
            "  ipfs-accelerate assurance mutate explain --candidate-json cand.json "
            "--manifest-json manifest.json\n"
            "  ipfs-accelerate assurance report --campaign-result-json result.json\n"
        ),
    )
    commands = group.add_subparsers(
        dest="assurance_command",
        metavar="COMMAND",
        help="Assurance operation family.",
    )

    # -- mutate ------------------------------------------------------------
    mutate = commands.add_parser(
        "mutate",
        help="Mutation campaign plan/run/target/explain.",
    )
    mutate_cmds = mutate.add_subparsers(
        dest="assurance_mutate_command",
        metavar="MUTATE_COMMAND",
        help="Mutation campaign subcommand.",
    )

    plan_p = mutate_cmds.add_parser("plan", help="Plan a mutation campaign.")
    _add_common_flags(plan_p)
    _add_json_input(
        plan_p,
        "--repository-state-json",
        dest="repository_state_json",
        help_text="JSON mapping for repository_state (identity bindings only).",
    )
    _add_json_input(
        plan_p,
        "--manifest-json",
        dest="manifest_json",
        help_text="JSON AssuranceManifest@1 mapping.",
    )
    _add_json_input(
        plan_p,
        "--policy-json",
        dest="policy_json",
        help_text="JSON MutationCampaignPolicy@1 mapping.",
    )
    _add_json_input(
        plan_p,
        "--resource-budget-json",
        dest="resource_budget_json",
        help_text="JSON CampaignResourceBudget mapping.",
    )
    _add_json_input(
        plan_p,
        "--baseline-json",
        dest="baseline_json",
        help_text="JSON BaselineRequirements or receipt CID object.",
    )
    _add_json_input(
        plan_p,
        "--targets-json",
        dest="targets_json",
        help_text="Optional JSON array of MutationTarget mappings.",
    )
    _add_json_input(
        plan_p,
        "--operators-json",
        dest="operators_json",
        help_text="Optional JSON array of MutationOperatorDefinition mappings.",
    )
    _add_json_input(
        plan_p,
        "--properties-json",
        dest="properties_json",
        help_text="Optional JSON array of asserted properties/claims.",
    )
    _add_json_input(
        plan_p,
        "--generation-manifest-json",
        dest="generation_manifest_json",
        help_text="Optional MutationGenerationManifest mapping.",
    )
    _add_json_input(
        plan_p,
        "--seed-config-json",
        dest="seed_config_json",
        help_text="Optional SeedConfigBinding mapping.",
    )
    plan_p.add_argument("--plan-id", default=None, help="Optional stable plan id.")
    plan_p.add_argument(
        "--no-partition",
        action="store_true",
        help="Disable held-out partition (default: partition enabled).",
    )
    plan_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    run_p = mutate_cmds.add_parser(
        "run",
        help="Execute a planned mutation campaign (requires --authorize-run).",
    )
    _add_common_flags(run_p)
    _add_json_input(
        run_p,
        "--plan-json",
        dest="plan_json",
        help_text="JSON MutationCampaignPlan@1 or plan result mapping.",
    )
    _add_json_input(
        run_p,
        "--verification-policy-json",
        dest="verification_policy_json",
        help_text="JSON verification policy mapping.",
    )
    _add_json_input(
        run_p,
        "--precomputed-reports-json",
        dest="precomputed_reports_json",
        help_text="JSON array of sealed per-candidate reports (hermetic execution).",
    )
    _add_json_input(
        run_p,
        "--candidates-json",
        dest="candidates_json",
        help_text="Optional JSON array of MutationCandidate mappings.",
    )
    _add_json_input(
        run_p,
        "--expected-detections-json",
        dest="expected_detections_json",
        help_text="Optional JSON array of ExpectedDetectionSet mappings.",
    )
    run_p.add_argument(
        "--authorize-run",
        action="store_true",
        help="Explicit run authority. Required for mutate run (fail closed without it).",
    )
    run_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    target_p = mutate_cmds.add_parser(
        "target",
        help="Select risk-weighted mutation targets from asserted properties.",
    )
    _add_common_flags(target_p)
    _add_json_input(
        target_p,
        "--properties-json",
        dest="properties_json",
        help_text="JSON array of AssertedProperty/ClaimRecord mappings.",
    )
    target_p.add_argument(
        "--repository-id",
        required=True,
        help="Repository identity token (not a filesystem path).",
    )
    target_p.add_argument(
        "--repository-state-cid",
        required=True,
        help="Repository state CID.",
    )
    _add_json_input(
        target_p,
        "--sampling-budget-json",
        dest="sampling_budget_json",
        help_text="Optional SamplingBudget mapping.",
    )
    target_p.add_argument(
        "--return-result",
        action="store_true",
        help="Return TargetSelectionResult envelope instead of bare targets.",
    )

    explain_p = mutate_cmds.add_parser(
        "explain",
        help="Predict and explain the expected detection set for one candidate.",
    )
    _add_common_flags(explain_p)
    _add_json_input(
        explain_p,
        "--candidate-json",
        dest="candidate_json",
        help_text="JSON MutationCandidate mapping.",
    )
    _add_json_input(
        explain_p,
        "--manifest-json",
        dest="manifest_json",
        help_text="JSON AssuranceManifest or DetectionAssuranceManifest mapping.",
    )
    explain_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    # -- report ------------------------------------------------------------
    report_p = commands.add_parser(
        "report",
        help="Project a bounded deterministic campaign report.",
    )
    _add_common_flags(report_p)
    _add_json_input(
        report_p,
        "--campaign-result-json",
        dest="campaign_result_json",
        help_text="JSON MutationCampaignExecutionResult mapping.",
    )
    _add_json_input(
        report_p,
        "--plan-json",
        dest="plan_json",
        help_text="Optional plan mapping to bind into the report.",
    )
    report_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    return group


def assurance_cli_discovery_manifest() -> dict[str, Any]:
    """Static vocabulary for help/conformance without constructing services."""

    return {
        "schema": ASSURANCE_CLI_SCHEMA,
        "interface": ASSURANCE_CLI_INTERFACE,
        "evidence": AAE_CLI_EVIDENCE,
        "group": ASSURANCE_GROUP,
        "console_entry": CONSOLE_ENTRY,
        "commands": list(TOP_LEVEL_COMMANDS),
        "mutate_commands": list(MUTATE_SUBCOMMANDS),
        "campaign_commands": list(CAMPAIGN_COMMANDS),
        "cold_help": True,
        "side_effect_free_parse": True,
        "lazy_dispatch": True,
        "production_policy_change": False,
        "arbitrary_external_repositories": False,
        "explicit_run_authority_required": True,
        "honors_cancellation": True,
        "honors_resources": True,
        "output": ("json", "human"),
    }


def assurance_cli_descriptor() -> Mapping[str, Any]:
    return MappingProxyType(assurance_cli_discovery_manifest())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _output_json_mode(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "output_human", False)):
        return False
    return True


def run_assurance_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    api: Any | None = None,
    cancellation_token: CliCancellationToken | None = None,
    campaign_handlers: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
) -> int:
    """Dispatch one assurance command through the campaign handlers / public API."""

    output_json = _output_json_mode(args)
    command = "assurance"
    try:
        command = resolve_dispatch_command(args)
        budget = resource_budget_from_args(args)
        token = cancellation_from_args(args, token=cancellation_token)
        token.check()

        # Lazy import of campaign handlers keeps registration cold.
        if campaign_handlers is None:
            from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.cli_campaign import (
                CAMPAIGN_HANDLERS,
            )

            handlers = CAMPAIGN_HANDLERS
        else:
            handlers = campaign_handlers

        handler = handlers.get(command)
        if handler is None:
            raise AssuranceCLIUsageError(
                f"unknown assurance command: {command}",
                reason_code="unknown_command",
            )

        token.check()
        result = handler(
            args,
            api=api,
            budget=budget,
            cancellation_token=token,
            stdin=stdin,
        )
        if not isinstance(result, Mapping):
            raise AssuranceCLIError(
                "campaign handler returned a non-mapping result",
                reason_code="invalid_handler_result",
            )
        status = str(result.get("status") or "ok")
        payload = envelope(
            ok=True,
            command=command,
            status=status,
            result=dict(result),
            exit_code=EXIT_SUCCESS,
        )
        emit(payload, output_json=output_json, stream=stdout)
        return EXIT_SUCCESS
    except AssuranceCLIError as exc:
        payload = envelope(
            ok=False,
            command=command,
            status=exc.reason_code,
            error=str(exc),
            reason_code=exc.reason_code,
            exit_code=exc.exit_code,
            details=exc.details,
        )
        emit(payload, output_json=output_json, stream=stdout if output_json else stderr)
        return exc.exit_code
    except Exception as exc:  # map public API errors
        reason = "public_api_error"
        exit_code = EXIT_ERROR
        details: dict[str, Any] = {}
        try:
            from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
                AssuranceApiUnavailableError,
                AssurancePublicApiError,
                PathExposureError,
                UnknownCommandError,
                UnknownFieldError,
            )

            if isinstance(exc, PathExposureError):
                reason = "path_exposure"
                exit_code = EXIT_USAGE
                details = dict(getattr(exc, "details", {}) or {})
            elif isinstance(exc, UnknownCommandError):
                reason = "unknown_command"
                exit_code = EXIT_USAGE
            elif isinstance(exc, UnknownFieldError):
                reason = "unknown_field"
                exit_code = EXIT_USAGE
            elif isinstance(exc, AssuranceApiUnavailableError):
                reason = str(getattr(exc, "reason_code", None) or "unavailable")
                exit_code = EXIT_UNAVAILABLE
                details = {
                    "command": getattr(exc, "command", None),
                    "diagnostic": getattr(exc, "diagnostic", None),
                }
            elif isinstance(exc, AssurancePublicApiError):
                reason = str(getattr(exc, "reason_code", None) or "public_api_error")
                details = dict(getattr(exc, "details", {}) or {})
        except Exception:  # noqa: BLE001 - mapping layer must not mask original
            pass
        # Leaf planning/execution errors often carry reason_code.
        if hasattr(exc, "reason_code") and reason == "public_api_error":
            reason = str(getattr(exc, "reason_code") or reason)
        payload = envelope(
            ok=False,
            command=command,
            status=reason,
            error=_clip(str(exc)),
            reason_code=reason,
            exit_code=exit_code,
            details=details,
        )
        emit(payload, output_json=output_json, stream=stdout if output_json else stderr)
        return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for focused tests (mirrors host registration)."""

    parser = argparse.ArgumentParser(
        prog=f"{CONSOLE_ENTRY} {ASSURANCE_GROUP}",
        description="Adversarial assurance campaign CLI.",
    )
    sub = parser.add_subparsers(dest="command")
    # Host registers under dest='command' == 'assurance'; for standalone we
    # expose the same nested structure under a synthetic root.
    register_assurance_cli(sub)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    api: Any | None = None,
    cancellation_token: CliCancellationToken | None = None,
) -> int:
    """Standalone entry for assurance CLI (used by focused tests)."""

    parser = argparse.ArgumentParser(
        prog=f"{CONSOLE_ENTRY}",
        description="IPFS Accelerate — assurance subset.",
    )
    sub = parser.add_subparsers(dest="command")
    register_assurance_cli(sub)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if getattr(args, "command", None) != ASSURANCE_GROUP:
        parser.print_help(stderr)
        return EXIT_USAGE
    if not getattr(args, "assurance_command", None):
        # Print assurance group help via a nested parse.
        nested = argparse.ArgumentParser(prog=f"{CONSOLE_ENTRY} {ASSURANCE_GROUP}")
        nested_sub = nested.add_subparsers(dest="assurance_command")
        # Re-register children onto a temporary parser for help text only.
        # Simpler: just emit usage.
        stderr.write(
            "usage: ipfs-accelerate assurance {mutate,report} ...\n"
            "assurance requires a subcommand: mutate or report\n"
        )
        return EXIT_USAGE
    return run_assurance_cli(
        args,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        api=api,
        cancellation_token=cancellation_token,
    )


__all__ = [
    "AAE_CLI_EVIDENCE",
    "ASSURANCE_CLI_INTERFACE",
    "ASSURANCE_CLI_RESULT_SCHEMA",
    "ASSURANCE_CLI_SCHEMA",
    "ASSURANCE_GROUP",
    "CAMPAIGN_COMMANDS",
    "CONSOLE_ENTRY",
    "CliCancellationToken",
    "CliResourceBudget",
    "EXIT_AUTHORITY",
    "EXIT_CANCELLED",
    "EXIT_ERROR",
    "EXIT_RESOURCE",
    "EXIT_SUCCESS",
    "EXIT_UNAVAILABLE",
    "EXIT_USAGE",
    "MUTATE_SUBCOMMANDS",
    "TOP_LEVEL_COMMANDS",
    "AssuranceCLIAuthorityError",
    "AssuranceCLICancelledError",
    "AssuranceCLIError",
    "AssuranceCLIPathError",
    "AssuranceCLIResourceError",
    "AssuranceCLIUsageError",
    "assurance_cli_descriptor",
    "assurance_cli_discovery_manifest",
    "build_parser",
    "cancellation_from_args",
    "emit",
    "envelope",
    "load_json_mapping",
    "load_json_value",
    "looks_like_host_path",
    "main",
    "project_result",
    "read_stdin_json",
    "register_assurance_cli",
    "reject_path_exposure",
    "repo_relative_path",
    "resolve_dispatch_command",
    "resource_budget_from_args",
    "run_assurance_cli",
]

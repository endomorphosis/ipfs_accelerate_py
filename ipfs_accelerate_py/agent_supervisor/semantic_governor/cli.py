"""Semantic-governor console CLI (SCG-037).

Interface: ``SemanticGovernorCLI@1`` / scg/cli@1

Closed ``semantic-governor`` entrypoint with exactly ten subcommands:

``audit``, ``shadow``, ``diagnose``, ``expand``, ``calibrate``,
``propose-rules``, ``evaluate-policy``, ``promote-policy``, ``report``,
``dashboard-data``.

Emits bounded deterministic JSON by default. Starts no public service, GUI,
listener, or provider configuration. Importing this module performs no I/O,
starts no threads/processes/network clients, and does not install packages.
``--help`` is cold and free of side effects.

Private raw source, secrets, and arbitrary host paths are stripped from
outputs. Promotion requires an explicit authorization input and a CAS-capable
policy repository; there is no implicit promotion path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Interface pins (stdlib only at module import)
# ---------------------------------------------------------------------------

CLI_INTERFACE = "SemanticGovernorCLI@1"
CLI_SCHEMA = "ipfs-accelerate.semantic-governor-cli@1"
CLI_BUNDLE = "scg/cli@1"
CLI_ADAPTER_ID = "ipfs-accelerate.semantic-governor.cli"
CLI_EVIDENCE = "scg/cli@1"
BOARD_NAMESPACE = "semantic-compression-governor-v1"
CONSOLE_ENTRY = "semantic-governor"
ENTRY_POINT = "ipfs_accelerate_py.agent_supervisor.semantic_governor.cli:main"

# Stable exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_PRODUCTION_GATE = 4

_MAX_DIAGNOSTIC = 512
_MAX_JSON_BYTES = 8_000_000

# Exact closed command vocabulary (plan §11 / SCG-037).
REQUIRED_CLI_COMMANDS: tuple[str, ...] = (
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

# CLI command -> primary typed API (evidence mapping).
CLI_TO_API: dict[str, str] = {
    "audit": "evaluate_context_sufficiency",
    "shadow": "create_shadow_plan",
    "diagnose": "diagnose_omission",
    "expand": "execute_expansion_loop",
    "calibrate": "update_calibration",
    "propose-rules": "propose_rule_change",
    "evaluate-policy": "evaluate_rule_candidate",
    "promote-policy": "promote_compression_policy",
    "report": "build_governor_report",
    "dashboard-data": "build_dashboard_data",
}

# Secondary APIs reachable via payload mode fields (not separate commands).
_SHADOW_COMPARE_API = "compare_shadow_results"
_EXPAND_PLAN_API = "plan_context_expansion"
_AUDIT_RUNTIME_API = "audit_task"
_SHADOW_RUNTIME_API = "shadow_task"
_EXPAND_RUNTIME_API = "expand_audit"

# Keys that must never appear in CLI JSON output (private raw source / secrets).
_PRIVATE_OUTPUT_KEY_MARKERS: frozenset[str] = frozenset(
    {
        "raw_private_source",
        "raw_source",
        "raw_source_text",
        "private_source",
        "private_source_text",
        "source_bytes",
        "source_text",
        "source_body",
        "source_code",
        "file_content",
        "file_contents",
        "expanded_private_source",
        "private_expanded_source",
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "bearer_token",
        "client_secret",
        "password",
        "passphrase",
        "private_key",
        "refresh_token",
        "session_token",
        "secret",
        "credentials",
        "credential",
    }
)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O beyond argument shaping)
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = _MAX_DIAGNOSTIC) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    return body[: limit - 3] + "..."


def _normalized_key(name: str) -> str:
    return str(name).strip().casefold().replace("-", "_").replace(" ", "_")


def _key_is_private_output(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in _PRIVATE_OUTPUT_KEY_MARKERS:
        return True
    for marker in _PRIVATE_OUTPUT_KEY_MARKERS:
        if marker in lowered:
            return True
    return False


def _string_looks_like_host_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "~/", "\\\\")) or value.startswith("file:"):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}:
        return True
    return False


def _emit_json(payload: Mapping[str, Any], stream: TextIO, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=True)
    else:
        text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
    if not text.endswith("\n"):
        text += "\n"
    stream.write(text)
    stream.flush()


def _success_envelope(
    command: str,
    result: Mapping[str, Any],
    *,
    exit_code: int = EXIT_OK,
    api: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "command": command,
        "api": api or CLI_TO_API.get(command),
        "exit_code": exit_code,
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "evidence": CLI_EVIDENCE,
        "board_namespace": BOARD_NAMESPACE,
        "result": dict(result),
    }


def _error_envelope(
    command: str | None,
    *,
    reason_code: str,
    diagnostic: str,
    exit_code: int,
    retryable: bool = False,
    operation: str | None = None,
    api: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "command": command,
        "api": api or (CLI_TO_API.get(command) if command else None),
        "exit_code": exit_code,
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "evidence": CLI_EVIDENCE,
        "board_namespace": BOARD_NAMESPACE,
        "error": {
            "operation": operation or (command or "cli"),
            "adapter_id": CLI_ADAPTER_ID,
            "reason_code": reason_code,
            "retryable": bool(retryable),
            "diagnostic": _clip(diagnostic),
        },
    }
    if extra:
        body["error"].update(dict(extra))
    return body


def _object_to_dict(value: Any) -> Any:
    """Convert API results to JSON-serializable structures (pre-privacy filter)."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _object_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_object_to_dict(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _object_to_dict(to_dict())
        except Exception:
            pass
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float, bool)):
        try:
            return _object_to_dict(value.value)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    # Bounded representation for opaque producer objects.
    attrs: dict[str, Any] = {"type": type(value).__name__}
    for attr in (
        "cid",
        "plan_cid",
        "report_cid",
        "claim_cid",
        "result_cid",
        "policy_cid",
        "candidate_cid",
        "authorization_cid",
        "status",
        "head_mutated",
        "workspace",
        "operation_id",
        "diagnostic",
    ):
        if hasattr(value, attr):
            try:
                raw = getattr(value, attr)
            except Exception:
                continue
            if isinstance(raw, (str, int, float, bool)) or raw is None:
                attrs[attr] = raw
    if len(attrs) > 1:
        return attrs
    return {"type": type(value).__name__, "repr": _clip(repr(value), limit=200)}


def project_cli_output(value: Any) -> Any:
    """Project a CLI result with private raw source and secrets removed.

    Fail-closed for free-form authority is not applied here: the projection
    strips private/secret keys and host-path strings so operator JSON never
    contains private raw source, regardless of intermediate API payloads.
    """

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _key_is_private_output(name):
                continue
            projected = project_cli_output(item)
            # Drop path-like fields that carry absolute host paths.
            if isinstance(projected, str) and _string_looks_like_host_path(projected):
                if name.endswith(("_path", "_dir", "_directory", "path", "cwd")):
                    continue
                projected = "<host-path-redacted>"
            out[name] = projected
        return out
    if isinstance(value, (list, tuple)):
        return [project_cli_output(item) for item in value]
    if isinstance(value, str):
        if _string_looks_like_host_path(value):
            return "<host-path-redacted>"
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    return project_cli_output(_object_to_dict(value))


def required_cli_commands() -> tuple[str, ...]:
    return REQUIRED_CLI_COMMANDS


def semantic_governor_cli_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticGovernorCLI@1."""

    return {
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "evidence": CLI_EVIDENCE,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": CLI_ADAPTER_ID,
        "console_entry": CONSOLE_ENTRY,
        "entry_point": ENTRY_POINT,
        "commands": list(REQUIRED_CLI_COMMANDS),
        "command_apis": dict(CLI_TO_API),
        "exit_codes": {
            "ok": EXIT_OK,
            "error": EXIT_ERROR,
            "usage": EXIT_USAGE,
            "unavailable": EXIT_UNAVAILABLE,
            "production_gate": EXIT_PRODUCTION_GATE,
        },
        "invariants": [
            "deterministic_json_default",
            "exact_ten_commands",
            "bounded_help_and_errors",
            "stable_exit_codes",
            "no_gui_or_public_server",
            "no_provider_configuration",
            "no_implicit_promotion",
            "promotion_requires_explicit_authorization_and_cas",
            "private_raw_source_never_in_output",
            "cold_help_and_import_no_mutation",
            "typed_unavailable_never_exit_zero",
        ],
        "symbols": [
            "build_parser",
            "main",
            "semantic_governor_cli_descriptor",
            "required_cli_commands",
            "project_cli_output",
        ],
    }


# ---------------------------------------------------------------------------
# Payload loading
# ---------------------------------------------------------------------------


def _load_json_text(text: str, *, source: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{source} exceeds maximum JSON size")
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must be a JSON object")
    return {str(k): v for k, v in payload.items()}


def load_payload(
    *,
    input_path: str | None,
    json_text: str | None,
    stdin: TextIO | None = None,
) -> dict[str, Any]:
    """Load a closed JSON object payload from --json, --input, or empty."""

    if json_text is not None and input_path is not None:
        raise ValueError("provide only one of --json or --input")
    if json_text is not None:
        return _load_json_text(json_text, source="--json")
    if input_path is None:
        return {}
    if input_path == "-":
        stream = stdin if stdin is not None else sys.stdin
        return _load_json_text(stream.read(), source="stdin")
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    return _load_json_text(path.read_text(encoding="utf-8"), source=str(path))


def _payload_get(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


# ---------------------------------------------------------------------------
# Default API resolution (lazy)
# ---------------------------------------------------------------------------


def _default_apis() -> dict[str, Callable[..., Any]]:
    """Resolve typed APIs lazily so cold import stays side-effect free."""

    from ipfs_accelerate_py.agent_supervisor.semantic_governor.differential import (
        compare_shadow_results,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
        execute_expansion_loop,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
        evaluate_rule_candidate,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
        promote_compression_policy,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.report import (
        build_dashboard_data,
        build_governor_report,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.runtime import (
        audit_task,
        expand_audit,
        shadow_task,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
        create_shadow_plan,
    )

    # Datasets analysis surfaces (optional at import time of leaf package).
    from ipfs_datasets_py.logic.software_contracts.semantic_governor import (
        diagnose_omission,
        evaluate_context_sufficiency,
        plan_context_expansion,
        propose_rule_change,
        update_calibration,
    )

    return {
        "evaluate_context_sufficiency": evaluate_context_sufficiency,
        "create_shadow_plan": create_shadow_plan,
        "compare_shadow_results": compare_shadow_results,
        "diagnose_omission": diagnose_omission,
        "plan_context_expansion": plan_context_expansion,
        "execute_expansion_loop": execute_expansion_loop,
        "update_calibration": update_calibration,
        "propose_rule_change": propose_rule_change,
        "evaluate_rule_candidate": evaluate_rule_candidate,
        "promote_compression_policy": promote_compression_policy,
        "build_governor_report": build_governor_report,
        "build_dashboard_data": build_dashboard_data,
        "audit_task": audit_task,
        "shadow_task": shadow_task,
        "expand_audit": expand_audit,
    }


def _resolve_api(
    apis: Mapping[str, Callable[..., Any]] | None,
    name: str,
) -> Callable[..., Any]:
    if apis is not None and name in apis:
        return apis[name]
    if apis is not None:
        # Allow CLI-command-named injections for tests.
        for command, api_name in CLI_TO_API.items():
            if api_name == name and command in apis:
                return apis[command]
    try:
        defaults = _default_apis()
    except Exception as exc:  # pragma: no cover - dependency surface
        raise RuntimeError(f"required API surface unavailable: {exc}") from exc
    if name not in defaults:
        raise KeyError(f"unknown API {name!r}")
    return defaults[name]


def _call_api(
    apis: Mapping[str, Callable[..., Any]] | None,
    name: str,
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    fn = _resolve_api(apis, name)
    return fn(*tuple(args), **dict(kwargs or {}))


# ---------------------------------------------------------------------------
# CAS repository helpers (promote-policy only)
# ---------------------------------------------------------------------------


def _open_cas_repositories(store_dir: str) -> tuple[Any, Any, Any]:
    """Open durable CAS policy + promotion repositories under *store_dir*."""

    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        DurableCoordinationStore,
    )
    from ipfs_kit_py.semantic_governor_store.policy import DurablePolicyCASRepositories

    store = DurableCoordinationStore(store_dir)
    cas = DurablePolicyCASRepositories(store)
    return store, cas.policy, cas.promotion


class _AuthorizationGateError(RuntimeError):
    """Promotion rejected before CAS because authorization is missing."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "absent_authorization",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class _CasGateError(RuntimeError):
    """Promotion rejected because CAS repository surface is missing."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "cas_unavailable",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_audit(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """audit → evaluate_context_sufficiency (or runtime audit_task)."""

    mode = str(_payload_get(payload, "mode", default=getattr(args, "mode", None) or "sufficiency"))
    if mode in {"runtime", "full", "audit_task"}:
        api = _AUDIT_RUNTIME_API
        result = _call_api(
            apis,
            api,
            kwargs={
                "task": _payload_get(payload, "task"),
                "compressed_context": _payload_get(
                    payload, "compressed_context", "context_pack"
                ),
                "repository_state": _payload_get(payload, "repository_state"),
                "audit_policy": _payload_get(payload, "audit_policy"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "task",
                        "compressed_context",
                        "context_pack",
                        "repository_state",
                        "audit_policy",
                    }
                },
            },
        )
    else:
        api = "evaluate_context_sufficiency"
        result = _call_api(
            apis,
            api,
            kwargs={
                "context_pack": _payload_get(
                    payload, "context_pack", "compressed_context"
                ),
                "repository_state": _payload_get(payload, "repository_state"),
                "verification_policy": _payload_get(payload, "verification_policy"),
                "calibration_profile": _payload_get(payload, "calibration_profile"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "context_pack",
                        "compressed_context",
                        "repository_state",
                        "verification_policy",
                        "calibration_profile",
                    }
                    and k
                    in {"claim_id"}
                },
            },
        )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_shadow(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """shadow → create_shadow_plan / compare_shadow_results / shadow_task."""

    mode = str(
        _payload_get(payload, "mode", default=getattr(args, "mode", None) or "plan")
    )
    if mode in {"compare", "compare_shadow_results"}:
        api = _SHADOW_COMPARE_API
        result = _call_api(
            apis,
            api,
            kwargs={
                "compressed_result": _payload_get(payload, "compressed_result"),
                "expanded_result": _payload_get(payload, "expanded_result"),
                "verification_evidence": _payload_get(
                    payload, "verification_evidence"
                ),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "compressed_result",
                        "expanded_result",
                        "verification_evidence",
                    }
                },
            },
        )
    elif mode in {"runtime", "full", "shadow_task"}:
        api = _SHADOW_RUNTIME_API
        result = _call_api(
            apis,
            api,
            kwargs={
                "task": _payload_get(payload, "task"),
                "compressed_context": _payload_get(
                    payload, "compressed_context", "context_pack"
                ),
                "repository_state": _payload_get(payload, "repository_state"),
                "audit_policy": _payload_get(payload, "audit_policy"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "task",
                        "compressed_context",
                        "context_pack",
                        "repository_state",
                        "audit_policy",
                    }
                },
            },
        )
    else:
        api = "create_shadow_plan"
        result = _call_api(
            apis,
            api,
            kwargs={
                "task": _payload_get(payload, "task"),
                "compressed_context": _payload_get(
                    payload, "compressed_context", "context_pack"
                ),
                "repository_state": _payload_get(payload, "repository_state"),
                "audit_policy": _payload_get(payload, "audit_policy"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "task",
                        "compressed_context",
                        "context_pack",
                        "repository_state",
                        "audit_policy",
                    }
                },
            },
        )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_diagnose(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "diagnose_omission"
    result = _call_api(
        apis,
        api,
        kwargs={
            "audit_case": _payload_get(payload, "audit_case"),
            "repository_state": _payload_get(payload, "repository_state"),
            "dependency_graph": _payload_get(payload, "dependency_graph"),
            **{
                k: v
                for k, v in payload.items()
                if k
                not in {"audit_case", "repository_state", "dependency_graph"}
            },
        },
    )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_expand(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    mode = str(
        _payload_get(payload, "mode", default=getattr(args, "mode", None) or "auto")
    )
    if mode in {"plan", "plan_context_expansion"} or (
        mode == "auto"
        and "plan" not in payload
        and _payload_get(payload, "omission_hypotheses") is not None
    ):
        api = _EXPAND_PLAN_API
        result = _call_api(
            apis,
            api,
            kwargs={
                "audit_case": _payload_get(payload, "audit_case"),
                "omission_hypotheses": _payload_get(payload, "omission_hypotheses"),
                "token_budget": _payload_get(payload, "token_budget"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "audit_case",
                        "omission_hypotheses",
                        "token_budget",
                    }
                },
            },
        )
    elif mode in {"runtime", "expand_audit"}:
        api = _EXPAND_RUNTIME_API
        result = _call_api(
            apis,
            api,
            kwargs={
                "plan": _payload_get(payload, "plan"),
                "model_policy": _payload_get(payload, "model_policy"),
                "verification_policy": _payload_get(payload, "verification_policy"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "plan",
                        "model_policy",
                        "verification_policy",
                    }
                },
            },
        )
    else:
        api = "execute_expansion_loop"
        result = _call_api(
            apis,
            api,
            kwargs={
                "plan": _payload_get(payload, "plan"),
                "model_policy": _payload_get(payload, "model_policy"),
                "verification_policy": _payload_get(payload, "verification_policy"),
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "mode",
                        "plan",
                        "model_policy",
                        "verification_policy",
                    }
                },
            },
        )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_calibrate(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "update_calibration"
    result = _call_api(
        apis,
        api,
        kwargs={
            "audit_case": _payload_get(payload, "audit_case"),
            "current_profile": _payload_get(
                payload, "current_profile", "calibration_profile"
            ),
            **{
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "audit_case",
                    "current_profile",
                    "calibration_profile",
                }
            },
        },
    )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_propose_rules(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "propose_rule_change"
    result = _call_api(
        apis,
        api,
        kwargs={
            "calibration_profile": _payload_get(payload, "calibration_profile"),
            "audit_cases": _payload_get(payload, "audit_cases"),
            **{
                k: v
                for k, v in payload.items()
                if k not in {"calibration_profile", "audit_cases"}
            },
        },
    )
    return api, project_cli_output(_object_to_dict(result))


def _cmd_evaluate_policy(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "evaluate_rule_candidate"
    result = _call_api(
        apis,
        api,
        kwargs={
            "candidate": _payload_get(payload, "candidate"),
            "held_out_benchmark": _payload_get(
                payload, "held_out_benchmark", "benchmark"
            ),
            **{
                k: v
                for k, v in payload.items()
                if k not in {"candidate", "held_out_benchmark", "benchmark"}
            },
        },
    )
    return api, project_cli_output(_object_to_dict(result))


def _extract_authorization(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> str | Mapping[str, Any] | None:
    """Resolve explicit authorization input (never implicit)."""

    if getattr(args, "authorization", None):
        return str(args.authorization)
    auth = _payload_get(
        payload,
        "authorization",
        "authorization_cid",
        "external_authorization_cid",
        "promotion_authorization_cid",
    )
    if auth is None or auth is False or auth == "":
        return None
    return auth


def _cmd_promote_policy(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """promote-policy → promote_compression_policy with explicit auth + CAS."""

    api = "promote_compression_policy"
    authorization = _extract_authorization(payload, args)
    if authorization is None:
        raise _AuthorizationGateError(
            "promote-policy requires explicit --authorization or "
            "payload.authorization (CID or mapping); implicit promotion is forbidden",
            reason_code="absent_authorization",
        )

    store_dir = getattr(args, "store_dir", None) or _payload_get(payload, "store_dir")
    store_handle = None
    try:
        if policy_repository is None:
            if store_dir:
                store_handle, policy_repository, promotion_repository = (
                    _open_cas_repositories(str(store_dir))
                )
            else:
                raise _CasGateError(
                    "promote-policy requires CAS via --store-dir or an injected "
                    "policy_repository; expected-version CAS cannot be skipped",
                    reason_code="cas_unavailable",
                )

        operation_id = (
            getattr(args, "operation_id", None)
            or _payload_get(payload, "operation_id")
        )
        if not operation_id:
            raise ValueError(
                "promote-policy requires --operation-id or payload.operation_id "
                "for CAS publication"
            )
        workspace = (
            getattr(args, "workspace", None)
            or _payload_get(payload, "workspace")
            or "default"
        )
        expected_generation = getattr(args, "expected_generation", None)
        if expected_generation is None:
            expected_generation = _payload_get(payload, "expected_generation")
        expected_policy_cid = getattr(args, "expected_policy_cid", None) or _payload_get(
            payload, "expected_policy_cid"
        )

        kwargs: dict[str, Any] = {
            "candidate": _payload_get(payload, "candidate"),
            "evaluation_report": _payload_get(
                payload, "evaluation_report", "evaluation"
            ),
            "authorization": authorization,
            "release_qualification": _payload_get(
                payload, "release_qualification", "qualification"
            ),
            "policy_repository": policy_repository,
            "workspace": workspace,
            "operation_id": str(operation_id),
            "expected_generation": expected_generation,
            "expected_policy_cid": expected_policy_cid,
            "promoted_policy": _payload_get(payload, "promoted_policy"),
            "promoted_policy_version": _payload_get(
                payload, "promoted_policy_version"
            )
            or getattr(args, "promoted_policy_version", None),
            "promotion_repository": promotion_repository,
            "repository_state_cid": _payload_get(payload, "repository_state_cid"),
            "notes": _payload_get(payload, "notes"),
            "metadata": _payload_get(payload, "metadata"),
        }
        # Drop None-valued optional kwargs so leaf defaults apply cleanly.
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k in {
            "candidate",
            "evaluation_report",
            "authorization",
            "release_qualification",
            "policy_repository",
            "workspace",
            "operation_id",
        }}

        result = _call_api(apis, api, kwargs=kwargs)
        projected = project_cli_output(_object_to_dict(result))
        # Surface head mutation / CAS status for operators without private data.
        if isinstance(projected, dict):
            projected.setdefault("cas_required", True)
            projected.setdefault("authorization_required", True)
            projected.setdefault("implicit_promotion", False)
        return api, projected if isinstance(projected, dict) else {"value": projected}
    finally:
        if store_handle is not None:
            close = getattr(store_handle, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _cmd_report(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "build_governor_report"
    # Keyword-only builder: pass payload fields as kwargs.
    kwargs = dict(payload)
    kwargs.pop("mode", None)
    result = _call_api(apis, api, kwargs=kwargs)
    return api, project_cli_output(_object_to_dict(result))


def _cmd_dashboard_data(
    payload: Mapping[str, Any],
    *,
    apis: Mapping[str, Callable[..., Any]] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api = "build_dashboard_data"
    kwargs = dict(payload)
    kwargs.pop("mode", None)
    report = kwargs.pop("report", None)
    if report is not None:
        result = _call_api(apis, api, args=(report,), kwargs=kwargs)
    else:
        result = _call_api(apis, api, kwargs=kwargs)
    return api, project_cli_output(_object_to_dict(result))


_HANDLERS: dict[str, Callable[..., tuple[str, dict[str, Any]]]] = {
    "audit": _cmd_audit,
    "shadow": _cmd_shadow,
    "diagnose": _cmd_diagnose,
    "expand": _cmd_expand,
    "calibrate": _cmd_calibrate,
    "propose-rules": _cmd_propose_rules,
    "evaluate-policy": _cmd_evaluate_policy,
    "promote-policy": _cmd_promote_policy,
    "report": _cmd_report,
    "dashboard-data": _cmd_dashboard_data,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the closed ``semantic-governor`` argument parser."""

    parser = argparse.ArgumentParser(
        prog=CONSOLE_ENTRY,
        description=(
            "Narrowly scoped Semantic Compression Governor CLI. Deterministic "
            "JSON by default. Exactly ten commands. No public server, GUI, "
            "provider configuration, or implicit promotion."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0=ok, 1=error, 2=usage, 3=unavailable, "
            "4=production-gate. Promotion requires explicit authorization "
            "and expected-version CAS."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s ({CLI_INTERFACE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(child: argparse.ArgumentParser) -> None:
        child.add_argument(
            "--input",
            "-i",
            type=str,
            default=None,
            dest="input_path",
            help="JSON object payload path (use '-' for stdin).",
        )
        child.add_argument(
            "--json",
            type=str,
            default=None,
            dest="json_text",
            help="Inline JSON object payload.",
        )
        child.add_argument(
            "--compact",
            action="store_true",
            help="Emit compact JSON (no indentation).",
        )

    def _add_mode(child: argparse.ArgumentParser, choices: Sequence[str]) -> None:
        child.add_argument(
            "--mode",
            type=str,
            default=None,
            choices=list(choices),
            help="Command mode selecting a typed API surface.",
        )

    # audit
    p = sub.add_parser(
        "audit",
        help="Evaluate context sufficiency (or full runtime audit).",
    )
    _add_common(p)
    _add_mode(p, ("sufficiency", "runtime", "full", "audit_task"))

    # shadow
    p = sub.add_parser(
        "shadow",
        help="Create a shadow plan, compare results, or run shadow_task.",
    )
    _add_common(p)
    _add_mode(
        p,
        (
            "plan",
            "compare",
            "compare_shadow_results",
            "runtime",
            "full",
            "shadow_task",
        ),
    )

    # diagnose
    p = sub.add_parser("diagnose", help="Diagnose omission for an audit case.")
    _add_common(p)

    # expand
    p = sub.add_parser(
        "expand",
        help="Plan or execute bounded context expansion.",
    )
    _add_common(p)
    _add_mode(
        p,
        (
            "auto",
            "plan",
            "plan_context_expansion",
            "execute",
            "execute_expansion_loop",
            "runtime",
            "expand_audit",
        ),
    )

    # calibrate
    p = sub.add_parser("calibrate", help="Update a calibration profile.")
    _add_common(p)

    # propose-rules
    p = sub.add_parser("propose-rules", help="Propose rule changes from calibration.")
    _add_common(p)

    # evaluate-policy
    p = sub.add_parser(
        "evaluate-policy",
        help="Evaluate a rule candidate against held-out evidence.",
    )
    _add_common(p)

    # promote-policy (authorization + CAS gates)
    p = sub.add_parser(
        "promote-policy",
        help=(
            "Promote a compression policy (requires explicit authorization "
            "and expected-version CAS)."
        ),
    )
    _add_common(p)
    p.add_argument(
        "--authorization",
        type=str,
        default=None,
        help="Explicit promotion authorization CID (required; never implicit).",
    )
    p.add_argument(
        "--store-dir",
        type=str,
        default=None,
        dest="store_dir",
        help="Local durable coordination store directory for policy CAS.",
    )
    p.add_argument(
        "--operation-id",
        type=str,
        default=None,
        help="Stable CAS operation id (required).",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Policy workspace token (default: default).",
    )
    p.add_argument(
        "--expected-generation",
        type=int,
        default=None,
        help="Expected policy head generation for CAS.",
    )
    p.add_argument(
        "--expected-policy-cid",
        type=str,
        default=None,
        help="Expected policy head CID for CAS.",
    )
    p.add_argument(
        "--promoted-policy-version",
        type=str,
        default=None,
        help="Optional promoted policy version label.",
    )

    # report
    p = sub.add_parser(
        "report",
        help="Build a privacy-filtered governor final report projection.",
    )
    _add_common(p)

    # dashboard-data
    p = sub.add_parser(
        "dashboard-data",
        help="Build a privacy-filtered dashboard-data summary projection.",
    )
    _add_common(p)

    return parser


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    apis: Mapping[str, Callable[..., Any]] | None = None,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Run the ``semantic-governor`` CLI. Returns a stable exit code."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()

    # argparse writes help/usage to the process streams; bind them for this call
    # so tests and embedded hosts capture bounded help without side channels.
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = out
        sys.stderr = err
        try:
            args = parser.parse_args(list(argv) if argv is not None else None)
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return EXIT_OK
            if isinstance(code, int):
                return (
                    code
                    if code in (EXIT_OK, EXIT_USAGE, EXIT_ERROR)
                    else EXIT_USAGE
                )
            return EXIT_USAGE
    finally:
        sys.stdout = old_out
        sys.stderr = old_err

    command = str(args.command)
    compact = bool(getattr(args, "compact", False))

    def _write(payload: Mapping[str, Any]) -> None:
        # Always project envelope through privacy filter (defense in depth).
        safe = project_cli_output(dict(payload))
        if not isinstance(safe, dict):
            safe = {"ok": False, "result": safe}
        _emit_json(safe, out, compact=compact)

    if command not in REQUIRED_CLI_COMMANDS:
        envelope = _error_envelope(
            command,
            reason_code="unknown_command",
            diagnostic=f"unknown command: {command!r}",
            exit_code=EXIT_USAGE,
        )
        _write(envelope)
        return EXIT_USAGE

    try:
        payload = load_payload(
            input_path=getattr(args, "input_path", None),
            json_text=getattr(args, "json_text", None),
            stdin=stdin,
        )
    except FileNotFoundError as exc:
        envelope = _error_envelope(
            command,
            reason_code="not_found",
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
        )
        _write(envelope)
        return EXIT_ERROR
    except ValueError as exc:
        envelope = _error_envelope(
            command,
            reason_code="invalid_payload",
            diagnostic=str(exc),
            exit_code=EXIT_USAGE,
        )
        _write(envelope)
        return EXIT_USAGE

    handler = _HANDLERS[command]
    try:
        if command == "promote-policy":
            api_name, result = handler(
                payload,
                apis=apis,
                args=args,
                policy_repository=policy_repository,
                promotion_repository=promotion_repository,
            )
        else:
            api_name, result = handler(payload, apis=apis, args=args)
    except _AuthorizationGateError as exc:
        envelope = _error_envelope(
            command,
            reason_code=exc.reason_code,
            diagnostic=str(exc),
            exit_code=EXIT_PRODUCTION_GATE,
            operation="promote-policy",
            api="promote_compression_policy",
            extra={
                "head_mutated": False,
                "implicit_promotion": False,
                "authorization_required": True,
                "cas_required": True,
            },
        )
        _write(envelope)
        return EXIT_PRODUCTION_GATE
    except _CasGateError as exc:
        envelope = _error_envelope(
            command,
            reason_code=exc.reason_code,
            diagnostic=str(exc),
            exit_code=EXIT_UNAVAILABLE,
            operation="promote-policy",
            api="promote_compression_policy",
            extra={
                "head_mutated": False,
                "cas_required": True,
                "authorization_required": True,
            },
        )
        _write(envelope)
        return EXIT_UNAVAILABLE
    except FileNotFoundError as exc:
        envelope = _error_envelope(
            command,
            reason_code="not_found",
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
        )
        _write(envelope)
        return EXIT_ERROR
    except Exception as exc:
        reason_code = getattr(exc, "reason_code", None)
        operation = getattr(exc, "operation", command)
        retryable = bool(getattr(exc, "retryable", False))
        name = type(exc).__name__
        unavailable_names = {
            "GovernorApiUnavailableError",
            "GovernorApiUnavailableResult",
            "SemanticStateUnavailable",
            "DurableStateUnavailable",
            "UnavailableResult",
        }
        unavailable_codes = {
            "import_failed",
            "capability_unavailable",
            "missing_exports",
            "provider_unavailable",
            "api_unavailable",
            "cas_unavailable",
        }
        if name in unavailable_names or str(reason_code) in unavailable_codes:
            diagnostic = getattr(exc, "diagnostic", None) or str(exc)
            envelope = _error_envelope(
                command,
                reason_code=str(reason_code or "unavailable"),
                diagnostic=str(diagnostic),
                exit_code=EXIT_UNAVAILABLE,
                retryable=retryable,
                operation=str(operation),
            )
            _write(envelope)
            return EXIT_UNAVAILABLE

        envelope = _error_envelope(
            command,
            reason_code=str(reason_code or name),
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
            retryable=retryable,
            operation=str(operation),
        )
        _write(envelope)
        return EXIT_ERROR

    # Soft failures from promotion: rejected/conflict still emit structured JSON.
    if command == "promote-policy" and isinstance(result, Mapping):
        status = str(result.get("status") or "")
        head_mutated = result.get("head_mutated")
        if status in {"rejected", "conflict", "unavailable", "corrupt"} or (
            head_mutated is False and status not in {"promoted", "unchanged", ""}
        ):
            # Still a successful CLI invocation that reported a typed gate result.
            # Exit nonzero only for unavailable; rejected gates exit 1.
            exit_code = (
                EXIT_UNAVAILABLE
                if status in {"unavailable", "corrupt"}
                else EXIT_ERROR
            )
            if status == "rejected" and "absent_authorization" in list(
                result.get("blocking_reasons") or ()
            ):
                exit_code = EXIT_PRODUCTION_GATE
            envelope = _error_envelope(
                command,
                reason_code=str(
                    (result.get("blocking_reasons") or ["promotion_rejected"])[0]
                    if result.get("blocking_reasons")
                    else status or "promotion_rejected"
                ),
                diagnostic=str(
                    result.get("diagnostic") or status or "promotion rejected"
                ),
                exit_code=exit_code,
                operation="promote-policy",
                api=api_name,
                extra={"result": result, "head_mutated": bool(head_mutated)},
            )
            _write(envelope)
            return exit_code

    envelope = _success_envelope(command, result, exit_code=EXIT_OK, api=api_name)
    _write(envelope)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

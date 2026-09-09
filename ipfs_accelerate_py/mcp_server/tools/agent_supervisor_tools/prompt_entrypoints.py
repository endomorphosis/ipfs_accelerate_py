"""MCP / MCP++ prompt-lifecycle tools for the production Supervisor facade (ASE3-011).

Normal run/preview input is a prompt. Tools never accept raw client filesystem
authority as authorization; repository roots must be server-configured aliases
or already-authorized context. Listing tools is cold and side-effect free.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any, Final

PROMPT_LIFECYCLE_CATEGORY: Final = "agent_supervisor"
PROMPT_LIFECYCLE_TOOL_PREFIX: Final = "agent_supervisor_"
REPOSITORY_ALLOWLIST_ENV: Final = "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST"
STATE_ALLOWLIST_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST"

PROMPT_LIFECYCLE_TOOLS: Final[tuple[str, ...]] = (
    "agent_supervisor_run",
    "agent_supervisor_preview",
    "agent_supervisor_steer",
    "agent_supervisor_status",
    "agent_supervisor_follow",
    "agent_supervisor_explain",
    "agent_supervisor_doctor",
)

# These submission tools intentionally live alongside the established prompt
# tools.  The manager registers this module for both MCP transports, so adding
# a second registry or an MCP++-only service would create a competing surface.
OBJECTIVE_SUBMISSION_MCP_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.objective-submission-mcp@1"
)
CANONICAL_MCP_OBJECTIVE_SUBMISSION_ADAPTER: Final = "ObjectiveSubmissionMCPAdapter@1"
MCP_OBJECTIVE_SUBMISSION_TOOLS: Final[tuple[str, ...]] = (
    "agent_supervisor_submit_objective",
    "agent_supervisor_submit_objective_mcpplusplus",
)

_lock = RLock()
_injected_supervisor: Any = None


class PromptEntrypointError(RuntimeError):
    """Typed MCP prompt-lifecycle failure."""


class PathInjectionDenied(PromptEntrypointError):
    """Client-supplied path is not on the server allowlist."""


class ObjectiveSubmissionMCPError(PromptEntrypointError):
    """Typed failure at the objective-submission MCP boundary."""


def configure_prompt_lifecycle_supervisor(supervisor: Any | None) -> None:
    """Inject a Supervisor instance for later tool invocations (tests/embedders)."""

    global _injected_supervisor
    with _lock:
        _injected_supervisor = supervisor


def prompt_lifecycle_discovery_manifest() -> dict[str, Any]:
    """Static tool vocabulary without constructing a Supervisor."""

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-lifecycle-mcp@1",
        "category": PROMPT_LIFECYCLE_CATEGORY,
        "tools": list(PROMPT_LIFECYCLE_TOOLS),
        "normal_input": "prompt",
        "path_authority": "server_allowlist_only",
        "cold_registration": True,
    }


def _allowlist(env_name: str) -> tuple[Path, ...]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return ()
    paths: list[Path] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        paths.append(Path(item).resolve())
    return tuple(paths)


def _authorize_repository(repository: str | None) -> Path | None:
    if repository is None or repository == "":
        return None
    if not isinstance(repository, str):
        raise PathInjectionDenied("repository must be a string path")
    candidate = Path(repository).resolve()
    # Reject path traversal / symlink escape attempts by requiring allowlist hit.
    allowed = _allowlist(REPOSITORY_ALLOWLIST_ENV)
    if not allowed:
        # Without an explicit server allowlist, refuse client-supplied paths.
        raise PathInjectionDenied(
            "client repository paths require server allowlist "
            f"({REPOSITORY_ALLOWLIST_ENV})"
        )
    for root in allowed:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PathInjectionDenied(f"repository path not allowlisted: {candidate}")


def _open_supervisor(*, repository: str | None = None) -> Any:
    # Always enforce allowlist policy before any facade use (including tests).
    root = _authorize_repository(repository)
    with _lock:
        if _injected_supervisor is not None:
            return _injected_supervisor
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor

    return Supervisor.open(repository=root)


def _prompt_schema(*, require_prompt: bool = True) -> dict[str, Any]:
    props: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "minLength": 1,
            "description": "Intent prompt (never authority).",
        },
        "repository": {
            "type": "string",
            "description": "Optional server-allowlisted repository alias/path.",
        },
        "run_id": {
            "type": "string",
            "description": "Exact run identifier when required or ambiguous.",
        },
    }
    required = ["prompt"] if require_prompt else []
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _result_ok(payload: Mapping[str, Any], *, composition_cid: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": True,
        "result": dict(payload),
    }
    if composition_cid:
        body["composition_cid"] = composition_cid
    return body


def _result_err(error: str, *, code: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "error_code": code}


def _trusted_admission(admission: Any, *, require_delegation: bool) -> Any:
    """Accept only a host-produced DOEP-016 admission record.

    Raw principal, policy, UCAN, and delegation values are deliberately not
    accepted here.  The transport must authenticate them and invoke the
    canonical authority resolver before reaching this thin adapter.
    """

    from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
        AuthorityAdmission,
    )

    if not isinstance(admission, AuthorityAdmission):
        raise ObjectiveSubmissionMCPError(
            "objective submission requires trusted AuthorityAdmission"
        )
    if not admission.authorized:
        raise ObjectiveSubmissionMCPError("objective submission authority denied")
    if require_delegation and admission.delegation is None:
        raise ObjectiveSubmissionMCPError(
            "MCP++ objective submission requires verified delegation"
        )
    return admission


def _receipt_payload(receipt: Any) -> dict[str, Any]:
    if hasattr(receipt, "to_dict"):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        raise ObjectiveSubmissionMCPError(
            "canonical objective submission returned an unreadable receipt"
        )
    return dict(receipt)


def _submit_objective_adapter(
    intent: Any,
    *,
    admission: Any,
    require_delegation: bool,
    submit_objective: Any | None = None,
) -> dict[str, Any]:
    """Delegate objective submission to DOEP-012 after DOEP-016 admission."""

    _trusted_admission(admission, require_delegation=require_delegation)
    if submit_objective is None:
        from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
            submit_objective as _submit_objective,
        )

        submit_objective = _submit_objective
    return _receipt_payload(submit_objective(intent))


def submit_objective_mcp(
    intent: Any,
    *,
    admission: Any,
    submit_objective: Any | None = None,
) -> dict[str, Any]:
    """Authenticated MCP adapter over the canonical objective service."""

    return _submit_objective_adapter(
        intent,
        admission=admission,
        require_delegation=False,
        submit_objective=submit_objective,
    )


def submit_objective_mcpplusplus(
    intent: Any,
    *,
    admission: Any,
    submit_objective: Any | None = None,
) -> dict[str, Any]:
    """Authenticated and delegated MCP++ adapter over the same service."""

    return _submit_objective_adapter(
        intent,
        admission=admission,
        require_delegation=True,
        submit_objective=submit_objective,
    )


async def agent_supervisor_submit_objective(
    intent: Any,
    *,
    authority_admission: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Submit a datasets-owned objective intent through authenticated MCP."""

    try:
        return _result_ok(
            submit_objective_mcp(intent, admission=authority_admission),
        )
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_submit_objective_mcpplusplus(
    intent: Any,
    *,
    authority_admission: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Submit a datasets-owned objective intent through delegated MCP++."""

    try:
        return _result_ok(
            submit_objective_mcpplusplus(intent, admission=authority_admission),
        )
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


def objective_submission_mcp_discovery_manifest() -> dict[str, Any]:
    """Static description of the two transport adapters; no service is opened."""

    return {
        "schema": OBJECTIVE_SUBMISSION_MCP_SCHEMA,
        "adapter": CANONICAL_MCP_OBJECTIVE_SUBMISSION_ADAPTER,
        "tools": list(MCP_OBJECTIVE_SUBMISSION_TOOLS),
        "submission_delegate": (
            "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service"
            ".submit_objective"
        ),
        "authority_delegate": (
            "ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver"
            ".admit_authority"
        ),
        "mcp_requires_authenticated_admission": True,
        "mcpplusplus_requires_verified_delegation": True,
        "callers_supply_authoritative_policy": False,
        "completion_authority": False,
        "cold_registration": True,
    }


async def agent_supervisor_run(
    prompt: str,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Start or resume a durable run from a prompt."""

    try:
        if not isinstance(prompt, str) or not prompt.strip():
            return _result_err("prompt must be a non-empty string", code="invalid")
        supervisor = _open_supervisor(repository=repository)
        run = supervisor.run(prompt)
        return _result_ok(
            {
                "run_id": run.run_id,
                "state": run.state,
                "health": run.health,
                "event_cursor": run.event_cursor,
                "effect_receipt_cids": list(run.effect_receipt_cids),
            },
            composition_cid=supervisor.composition_cid,
        )
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:  # typed facade failures
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_preview(
    prompt: str,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            return _result_err("prompt must be a non-empty string", code="invalid")
        supervisor = _open_supervisor(repository=repository)
        obs = supervisor.preview(prompt)
        payload = obs.to_dict()
        # Ensure prompt body does not leak.
        blob = str(payload)
        if prompt in blob:
            return _result_err("prompt body leak denied", code="prompt_leak")
        return _result_ok(payload, composition_cid=supervisor.composition_cid)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_steer(
    prompt: str,
    run_id: str,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            return _result_err("prompt must be a non-empty string", code="invalid")
        if not isinstance(run_id, str) or not run_id.strip():
            return _result_err("run_id is required", code="invalid")
        supervisor = _open_supervisor(repository=repository)
        obs = supervisor.steer(run_id, prompt)
        return _result_ok(obs.to_dict(), composition_cid=supervisor.composition_cid)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_status(
    run_id: str | None = None,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        supervisor = _open_supervisor(repository=repository)
        obs = supervisor.status(run_id)
        return _result_ok(obs.to_dict(), composition_cid=supervisor.composition_cid)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_follow(
    run_id: str | None = None,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        supervisor = _open_supervisor(repository=repository)
        events = [obs.to_dict() for obs in supervisor.follow(run_id)]
        return _result_ok(
            {"events": events, "count": len(events)},
            composition_cid=supervisor.composition_cid,
        )
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_explain(
    run_id: str | None = None,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        supervisor = _open_supervisor(repository=repository)
        obs = supervisor.explain(run_id)
        return _result_ok(obs.to_dict(), composition_cid=supervisor.composition_cid)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


async def agent_supervisor_doctor(
    run_id: str | None = None,
    repository: str | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        supervisor = _open_supervisor(repository=repository)
        obs = supervisor.doctor(run_id)
        return _result_ok(obs.to_dict(), composition_cid=supervisor.composition_cid)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code="path_denied")
    except Exception as exc:
        return _result_err(str(exc), code=type(exc).__name__)


_TOOL_FUNCS: Final[Mapping[str, Any]] = {
    "agent_supervisor_run": agent_supervisor_run,
    "agent_supervisor_preview": agent_supervisor_preview,
    "agent_supervisor_steer": agent_supervisor_steer,
    "agent_supervisor_status": agent_supervisor_status,
    "agent_supervisor_follow": agent_supervisor_follow,
    "agent_supervisor_explain": agent_supervisor_explain,
    "agent_supervisor_doctor": agent_supervisor_doctor,
    "agent_supervisor_submit_objective": agent_supervisor_submit_objective,
    "agent_supervisor_submit_objective_mcpplusplus": (
        agent_supervisor_submit_objective_mcpplusplus
    ),
}

_TOOL_SCHEMAS: Final[Mapping[str, dict[str, Any]]] = {
    "agent_supervisor_run": _prompt_schema(require_prompt=True),
    "agent_supervisor_preview": _prompt_schema(require_prompt=True),
    "agent_supervisor_steer": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "run_id": {"type": "string", "minLength": 1},
            "repository": {"type": "string"},
        },
        "required": ["prompt", "run_id"],
        "additionalProperties": False,
    },
    "agent_supervisor_status": _prompt_schema(require_prompt=False),
    "agent_supervisor_follow": _prompt_schema(require_prompt=False),
    "agent_supervisor_explain": _prompt_schema(require_prompt=False),
    "agent_supervisor_doctor": _prompt_schema(require_prompt=False),
    # AuthorityAdmission is transport-owned context and intentionally absent
    # from this public schema: clients cannot forge an authority decision.
    "agent_supervisor_submit_objective": {
        "type": "object",
        "properties": {"intent": {"type": "object"}},
        "required": ["intent"],
        "additionalProperties": False,
    },
    "agent_supervisor_submit_objective_mcpplusplus": {
        "type": "object",
        "properties": {"intent": {"type": "object"}},
        "required": ["intent"],
        "additionalProperties": False,
    },
}


def register_prompt_lifecycle_tools(manager: Any) -> None:
    """Register prompt-lifecycle tools without resolving a Supervisor."""

    for name in (*PROMPT_LIFECYCLE_TOOLS, *MCP_OBJECTIVE_SUBMISSION_TOOLS):
        manager.register_tool(
            category=PROMPT_LIFECYCLE_CATEGORY,
            name=name,
            func=_TOOL_FUNCS[name],
            description=(
                (
                    f"Objective submission {name.removeprefix(PROMPT_LIFECYCLE_TOOL_PREFIX)} "
                    "through the canonical intent and authority services."
                    if name in MCP_OBJECTIVE_SUBMISSION_TOOLS
                    else f"Prompt-first supervisor {name.removeprefix(PROMPT_LIFECYCLE_TOOL_PREFIX)} "
                    "via the shared production facade (ASE3-011)."
                )
            ),
            input_schema=_TOOL_SCHEMAS[name],
            runtime="fastapi",
            tags=[
                "native",
                "agent-supervisor",
                "prompt-lifecycle",
                "policy-controlled",
                "body-free",
                *( ["objective-submission", "authenticated"]
                   if name in MCP_OBJECTIVE_SUBMISSION_TOOLS else [] ),
            ],
        )


__all__ = [
    "PROMPT_LIFECYCLE_CATEGORY",
    "PROMPT_LIFECYCLE_TOOLS",
    "CANONICAL_MCP_OBJECTIVE_SUBMISSION_ADAPTER",
    "MCP_OBJECTIVE_SUBMISSION_TOOLS",
    "OBJECTIVE_SUBMISSION_MCP_SCHEMA",
    "ObjectiveSubmissionMCPError",
    "PathInjectionDenied",
    "PromptEntrypointError",
    "REPOSITORY_ALLOWLIST_ENV",
    "agent_supervisor_doctor",
    "agent_supervisor_explain",
    "agent_supervisor_follow",
    "agent_supervisor_preview",
    "agent_supervisor_run",
    "agent_supervisor_status",
    "agent_supervisor_steer",
    "agent_supervisor_submit_objective",
    "agent_supervisor_submit_objective_mcpplusplus",
    "configure_prompt_lifecycle_supervisor",
    "prompt_lifecycle_discovery_manifest",
    "objective_submission_mcp_discovery_manifest",
    "register_prompt_lifecycle_tools",
    "submit_objective_mcp",
    "submit_objective_mcpplusplus",
]

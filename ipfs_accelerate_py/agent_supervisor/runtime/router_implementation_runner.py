"""Bounded native implementation command dispatched through the shared router.

The supervisor owns worktree allocation, credentials isolation, validation and
completion. This runner performs one pinned coding-provider invocation in the
current worktree and retains native usage; it grants no task completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

MAX_IMPLEMENTATION_TIMEOUT_SECONDS = 600

# Diagnostic vocabulary is intentionally closed: exception messages, dynamic
# type names, caller paths and locals never enter the child-reported record.
RUNNER_FAILURE_PHASES = frozenset({
    "runner_initialization", "argument_validation", "planning_contract",
    "container_boundary", "workspace_identity", "semantic_context",
    "doctor_residual", "public_instruction", "model_prompt", "coding_reply_contract",
    "credential_isolation", "provider_discovery", "provider_allocation",
    "provider_initialization", "provider_invocation", "provider_result_validation",
    "semantic_response_decode", "coding_reply_validation", "unclassified",
})
RUNNER_FAILURE_TYPES = frozenset({
    "ValueError", "TypeError", "RuntimeError", "PermissionError", "FileNotFoundError",
    "CalledProcessError", "TimeoutExpired", "TimeoutError", "JSONDecodeError",
    "UnicodeDecodeError", "ImportError", "ModuleNotFoundError", "OSError",
    "SemanticTranslationError", "other",
})
RUNNER_FAILURE_FILES = frozenset({
    "router_implementation_runner.py", "container_worker_boundary.py",
    "semantic_router_translation.py", "doctor_residual_context.py",
    "router_public_instruction.py", "local_planning_admission.py",
    "terminal_planner_contract.py", "doctor_candidate_runner.py", "coding_reply_contract.py",
})


def validate_runner_failure_diagnostic(value: object) -> dict:
    """Validate child-reported metadata, never evidence of dispatch or custody."""
    if (type(value) is not dict or set(value) != {
            "schema", "phase", "exceptions", "chain_truncated", "completion_authority",
            "automatic_retry_admitted", "settlement_authority", "provider_dispatch_observed"}
            or value["schema"] != "router-implementation-failure-diagnostic@1"
            or type(value["phase"]) is not str or value["phase"] not in RUNNER_FAILURE_PHASES
            or type(value["chain_truncated"]) is not bool
            or any(value[key] is not False for key in (
                "completion_authority", "automatic_retry_admitted", "settlement_authority"))
            or value["provider_dispatch_observed"] is not None
            or type(value["exceptions"]) is not list or not 1 <= len(value["exceptions"]) <= 4):
        raise ValueError("invalid closed router failure diagnostic")
    for item in value["exceptions"]:
        if (type(item) is not dict or set(item) != {"exception_type", "frames"}
                or type(item["exception_type"]) is not str or item["exception_type"] not in RUNNER_FAILURE_TYPES
                or type(item["frames"]) is not list or len(item["frames"]) > 12):
            raise ValueError("invalid closed router exception diagnostic")
        for frame in item["frames"]:
            if (type(frame) is not dict or set(frame) != {"file", "line"}
                    or type(frame["file"]) is not str or frame["file"] not in RUNNER_FAILURE_FILES
                    or type(frame["line"]) is not int or not 1 <= frame["line"] <= 1_000_000):
                raise ValueError("invalid closed router failure frame")
    return json.loads(json.dumps(value, allow_nan=False))


def _runner_failure_diagnostic(error: BaseException) -> dict:
    """Walk bounded code metadata directly; do not load traceback source text."""
    phase = "unclassified"
    exceptions = []
    seen = set()
    walked = 0
    truncated = False
    runtime_dir = os.path.dirname(os.path.abspath(__file__))
    current = error
    while current is not None and len(exceptions) < 4 and id(current) not in seen:
        seen.add(id(current))
        kind = type(current).__name__
        kind = kind if kind in RUNNER_FAILURE_TYPES else "other"
        frames = []
        trace = current.__traceback__
        while trace is not None and walked < 256:
            walked += 1
            code = trace.tb_frame.f_code
            if code is run.__code__:
                selected = trace.tb_frame.f_locals.get("failure_phase")
                if phase == "unclassified" and type(selected) is str and selected in RUNNER_FAILURE_PHASES:
                    phase = selected
            filename = os.path.abspath(code.co_filename)
            name = os.path.basename(filename)
            if (os.path.dirname(filename) == runtime_dir and name in RUNNER_FAILURE_FILES
                    and 1 <= trace.tb_lineno <= 1_000_000):
                frames.append({"file": name, "line": trace.tb_lineno})
                frames = frames[-12:]
            trace = trace.tb_next
        truncated = truncated or trace is not None
        exceptions.append({"exception_type": kind, "frames": frames})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return validate_runner_failure_diagnostic({
        "schema": "router-implementation-failure-diagnostic@1", "phase": phase,
        "exceptions": exceptions, "chain_truncated": truncated or current is not None,
        "completion_authority": False, "automatic_retry_admitted": False,
        "settlement_authority": False, "provider_dispatch_observed": None,
    })


def validate_runner_error_envelope(value: object) -> dict:
    """Require an exact wrapper error, without assigning trust to its author."""
    if (type(value) is not dict or set(value) != {"schema", "error_type", "diagnostic"}
            or value["schema"] != "router-implementation-error@2"):
        raise ValueError("invalid closed router error envelope")
    diagnostic = validate_runner_failure_diagnostic(value["diagnostic"])
    if (type(value["error_type"]) is not str
            or value["error_type"] != diagnostic["exceptions"][0]["exception_type"]):
        raise ValueError("router error type differs from its diagnostic")
    return {"schema": value["schema"], "error_type": value["error_type"], "diagnostic": diagnostic}


def _provider_failure_diagnostic(error: BaseException, *, provider: str) -> dict:
    """Project the shared provider classifier without retaining its message."""
    from ipfs_accelerate_py.llm_allocation.observations import CallErrorKind, classify_provider_failure

    reason = CallErrorKind.UNKNOWN
    typed_kind = getattr(error, "codex_error_kind", None) if provider == "codex_cli" else None
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        reason = CallErrorKind.TIMEOUT
    elif isinstance(typed_kind, CallErrorKind) and typed_kind is not CallErrorKind.SUCCESS:
        reason = typed_kind
    else:
        try:
            failure = classify_provider_failure(provider, str(error)[:16384], exc=error)
            if isinstance(failure.kind, CallErrorKind) and failure.kind is not CallErrorKind.SUCCESS:
                reason = failure.kind
        except Exception:
            # Diagnostics must not replace the original provider failure.
            pass
    result = {"phase": "provider_invocation", "reason_code": reason.value}
    status = getattr(error, "codex_error_status", None) if provider == "codex_cli" else None
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    return result


def render_model_prompt(*, prompt: str, purpose: str, workspace: Path,
                        semantic_transport: bool = False) -> tuple[str, str]:
    """Preserve native task bytes while making the allocated coding path explicit."""
    if purpose not in {"planning", "coding"}:
        raise ValueError("explicit planning or coding purpose required")
    if purpose == "planning":
        return prompt, ""
    location = json.dumps(str(workspace), ensure_ascii=True)
    advisory = (
        "Supervisor coding workspace contract:\n"
        f"Your allocated checkout and current working directory is {location}.\n"
        "When the task names /app/<relative-path>, use the corresponding file under "
        "this allocated checkout. Edit the declared outputs there, including any "
        "declared new output. Do not write to canonical /app.\n"
        "This mapping changes tool filesystem locations only. In report contents, "
        "keep the task's requested logical path names, including original /app paths; "
        "do not substitute allocated checkout paths.\n"
        "The supervisor owner validates and publishes the candidate. Do not stage "
        "or commit Git changes, and do not modify Git control files. This path "
        "mapping does not expand the task's allowed files or grant completion authority.\n\n"
        + ("Task context follows using a reversible semantic symbol table; source code and authority remain canonical:\n"
           if semantic_transport else "Original native task prompt follows without modification:\n")
    )
    if len(advisory.encode()) > 8192:
        raise ValueError("coding workspace advisory exceeds its byte bound")
    return advisory + prompt, advisory


def run(*, prompt: str, provider: str, model: str, timeout: int, max_output_tokens: int,
        reasoning_effort: str = "high", container_boundary: Path | None = None,
        container_boundary_sha256: str = "", purpose: str = "coding",
        semantic_repository: Path | None = None,
        semantic_transport_schema: str = "supervisor-semantic-router-input@1",
        coding_reply_mode: str = "legacy",
        doctor_residual_artifact: Path | None = None,
        doctor_residual_sha256: str = "", doctor_residual_task_cid: str = "",
        public_instruction_artifact: Path | None = None, public_instruction_sha256: str = "",
        public_instruction_task_cid: str = "") -> tuple[str, dict]:
    failure_phase = "runner_initialization"
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation.intelligence_index import discover_available_providers, select_efficient_route
    from ipfs_accelerate_py.llm_router import generate_text, get_llm_provider
    from ipfs_accelerate_py.router_deps import RouterDeps

    failure_phase = "argument_validation"
    if provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("explicit supported shared-router coding provider required")
    if provider == "grok_cli" and model != "grok-4.7":
        raise ValueError("explicit pinned Grok benchmark model required")
    if not model or not 1 <= timeout <= MAX_IMPLEMENTATION_TIMEOUT_SECONDS or not 1 <= max_output_tokens <= 16_384:
        raise ValueError("explicit model and bounded invocation settings required")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("explicit supported reasoning effort required")
    from .coding_reply_contract import validate_coding_reply_mode
    validate_coding_reply_mode(mode=coding_reply_mode, purpose=purpose, provider=provider)
    if semantic_transport_schema not in {"supervisor-semantic-router-input@1",
                                         "supervisor-semantic-router-input@2"}:
        raise ValueError("unsupported semantic transport schema")
    if (semantic_transport_schema != "supervisor-semantic-router-input@1"
            and (purpose != "coding" or semantic_repository is None)):
        raise ValueError("compact semantic transport requires a semantic coding route")
    if not prompt.strip() or len(prompt.encode()) > 256_000:
        raise ValueError("implementation prompt exceeds its byte bound")
    failure_phase = "planning_contract"
    response_format = None
    task_contract = None
    codex_planning_schema = None
    codex_schema_projection = None
    codex_planning_validator = None
    if provider == "codex_cli" and purpose == "planning":
        from .codex_planning_schema import planning_schema
        selection = planning_schema(prompt)
        if selection is not None:
            codex_planning_schema, codex_schema_projection, codex_planning_validator = selection
    if provider == "grok_cli" and purpose == "planning":
        from ipfs_accelerate_py.cli_runtime.grok_structured_output import planning_response_format
        response_format = planning_response_format(prompt)
        from .terminal_planner_contract import contract_from_prompt
        task_contract = contract_from_prompt(prompt)
        if task_contract is not None and response_format is None:
            raise ValueError("terminal task constraints require canonical structured planning")
    failure_phase = "container_boundary"
    root = Path.cwd().resolve()
    boundary = None
    if container_boundary is not None or container_boundary_sha256:
        from .container_worker_boundary import verify_container_worker_boundary
        if container_boundary is None:
            raise ValueError("container boundary artifact required")
        boundary = verify_container_worker_boundary(
            artifact=container_boundary, expected_sha256=container_boundary_sha256, workspace=root,
        )
    if provider == "grok_cli" and purpose == "coding" and boundary is None:
        raise ValueError("Grok coding requires the verified isolated container worker boundary")
    failure_phase = "workspace_identity"
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError("implementation must start at its allocated Git worktree root")
    failure_phase = "semantic_context"
    encoded = None
    residual_receipt = None
    residual_advisory = ""
    router_prompt = prompt
    if semantic_repository is not None:
        semantic_repository = Path(semantic_repository).resolve(strict=True)
        if semantic_repository == root:
            raise ValueError("semantic coding requires a separate allocated worktree")
        # The trusted launch chooses the canonical source root. A model cannot
        # nominate a foreign catalog or substitute a different checkout.
        def common_dir(repository):
            value = subprocess.check_output(
                ["/usr/bin/git", "-c", "safe.directory=" + str(repository), "-C", str(repository),
                 "rev-parse", "--path-format=absolute", "--git-common-dir"],
                text=True, timeout=10,
            ).strip()
            return Path(value).resolve(strict=True)
        if common_dir(semantic_repository) != common_dir(root):
            raise ValueError("semantic source repository differs from allocated worktree")
        from .semantic_router_translation import encode_semantic_router_prompt
        encoded = encode_semantic_router_prompt(prompt=prompt, repository=semantic_repository,
                                               transport_schema=semantic_transport_schema)
        router_prompt = encoded.provider_prompt
    failure_phase = "doctor_residual"
    residual_args = (doctor_residual_artifact, doctor_residual_sha256, doctor_residual_task_cid)
    if any(residual_args):
        if not all(residual_args) or semantic_repository is None or purpose != "coding":
            raise ValueError("exact residual binding requires the full semantic coding route")
        from .doctor_residual_context import load_doctor_residual_advisory
        residual_advisory, residual_receipt = load_doctor_residual_advisory(
            artifact=doctor_residual_artifact, expected_sha256=doctor_residual_sha256,
            repository=semantic_repository, task_cid=doctor_residual_task_cid, prompt=prompt, workspace=root)
        router_prompt += residual_advisory
    failure_phase = "public_instruction"
    instruction_receipt = None
    instruction_args = (public_instruction_artifact, public_instruction_sha256, public_instruction_task_cid)
    if any(instruction_args):
        if not all(instruction_args) or purpose != "coding":
            raise ValueError("exact public instruction binding requires a coding route")
        from .router_public_instruction import load_public_instruction
        instruction, instruction_receipt = load_public_instruction(
            artifact=public_instruction_artifact, expected_sha256=public_instruction_sha256,
            task_cid=public_instruction_task_cid, prompt=prompt, workspace=root,
            repository=semantic_repository)
        # Preserve the literal requirements after semantic encoding. Native
        # capsule identity remains the stdin identity, not this projection.
        router_prompt += instruction
    failure_phase = "model_prompt"
    model_prompt, advisory = render_model_prompt(prompt=router_prompt, purpose=purpose, workspace=root,
                                                semantic_transport=encoded is not None)
    coding_reply_receipt = None
    if coding_reply_mode != "legacy":
        failure_phase = "coding_reply_contract"
        from .coding_reply_contract import apply_coding_reply_contract
        model_prompt, coding_reply_receipt = apply_coding_reply_contract(
            model_prompt=model_prompt, mode=coding_reply_mode, purpose=purpose, provider=provider)
    if len(model_prompt.encode()) > 256_000:
        raise ValueError("model prompt with workspace contract exceeds its byte bound")
    # Native provider children must never inherit a typed owner token, private
    # bootstrap channel, or database program. The daemon supplies this scrub.
    failure_phase = "credential_isolation"
    from .multi_supervisor_runner import DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES
    if any(name in os.environ for name in (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
                                           "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN")):
        raise ValueError("implementation inherited database authority")
    failure_phase = "provider_discovery"
    available = discover_available_providers()
    if provider not in available:
        raise ValueError("configured shared-router provider is unavailable")
    failure_phase = "provider_allocation"
    route = select_efficient_route(model_name=model, provider=provider, task_kind="coding", available_providers=available)
    if route.provider != provider or route.model_name != model:
        raise ValueError("model manager changed the explicit benchmark route")
    failure_phase = "provider_initialization"
    deps = RouterDeps()
    instance = get_llm_provider(provider, deps=deps, use_cache=False)
    set_last_cli_observation(provider, {})
    invocation = uuid.uuid4().hex
    started = time.monotonic()
    previous = os.environ.get("ipfs_accelerate_py_CODEX_SANDBOX")
    previous_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    # External mode is available only inside the explicitly deployed Docker
    # worker identity. The ordinary host route keeps its workspace sandbox.
    os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] = "danger-full-access" if boundary else "workspace-write"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = {
        "schema": "router-implementation-invocation@1", "invocation_id": invocation,
        "provider": provider, "model": model, "allocation_catalog": route.catalog_revision,
        "reasoning_effort": reasoning_effort, "purpose": purpose,
        # Keep the original fields bound to native stdin for capsule comparison.
        # The distinct model fields attest the complete actual provider input.
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "native_prompt_bytes": len(prompt.encode()),
        "model_prompt_sha256": hashlib.sha256(model_prompt.encode()).hexdigest(),
        "model_prompt_bytes": len(model_prompt.encode()),
        "workspace_advisory_sha256": hashlib.sha256(advisory.encode()).hexdigest() if advisory else None,
        "workspace_advisory_bytes": len(advisory.encode()),
        "workspace": str(root),
        "router_prompt_sha256": hashlib.sha256(router_prompt.encode()).hexdigest(),
        "router_prompt_bytes": len(router_prompt.encode()),
        "semantic_translation": dict(encoded.receipt) if encoded is not None else None,
        "doctor_residual_context": residual_receipt,
        "public_instruction": instruction_receipt,
        "timeout_seconds": timeout, "requested_output_tokens": max_output_tokens,
        "provider_output_token_cap_enforced": False, "router_calls": 1,
        "usage": None, "completion_authority": False,
        "external_container_boundary": boundary,
    }
    if coding_reply_receipt is not None:
        receipt["coding_reply_contract"] = coding_reply_receipt
    failure_phase = "provider_invocation"
    provider_options = {}
    if codex_planning_schema is not None:
        provider_options["codex_output_schema"] = codex_planning_schema
        receipt["provider_invocation_policy"] = {"structured_output": {
            "schema": "codex-native-planning-json-schema@1",
            "native_schema_projection": codex_schema_projection,
            "native_schema_requested": True,
            "response_schema_validated": False,
            "plan_admitted": False,
        }}
    if coding_reply_receipt is not None:
        from .coding_reply_contract import coding_completion_schema
        provider_options["codex_output_schema"] = coding_completion_schema()
        receipt["provider_invocation_policy"] = {"structured_output": {
            "schema": "codex-native-coding-completion-json-schema@1",
            "native_schema_requested": True,
            "response_schema_validated": False,
            "execution_authority": False,
            "completion_authority": False,
            "settlement_authority": False,
        }}
    if provider == "grok_cli":
        # Grok treats an empty allowlist as its default toolset. A nonempty
        # singleton followed by its explicit denial requests zero planning
        # tools; MCP dispatch is denied separately for both purposes. Actual
        # native tool exposure still requires independent qualification.
        provider_options = {"grok_tools": "read_file" if purpose == "planning" else
            "read_file,search_replace,grep,list_dir,todo_write,run_terminal_cmd",
            "grok_disallowed_tools": "read_file,search_tool,use_tool" if purpose == "planning" else
            "search_tool,use_tool",
            "grok_permission_mode": "dontAsk" if purpose == "planning" else "bypassPermissions",
            "grok_max_turns": 2 if purpose == "planning" else 128}
        receipt["provider_invocation_policy"] = {
            "max_turns": provider_options["grok_max_turns"],
            "tools_profile": "none" if purpose == "planning" else "isolated_coding",
            "permission_mode": provider_options["grok_permission_mode"],
            "native_tool_allowlist": provider_options["grok_tools"].split(","),
            "native_tool_denylist": provider_options["grok_disallowed_tools"].split(","),
            "disallowed_tools": provider_options["grok_disallowed_tools"].split(","),
            "effective_toolset_verified": False,
        }
        if response_format is not None:
            provider_options["response_format"] = response_format
            from ipfs_accelerate_py.cli_runtime.grok_structured_output import native_schema_projection
            _wire_schema, _wire_argument, projection = native_schema_projection(
                response_format["json_schema"]["schema"], task_contract=task_contract)
            if task_contract is not None:
                provider_options["grok_task_contract"] = task_contract
            schema_json = json.dumps(response_format["json_schema"]["schema"],
                ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            receipt["provider_invocation_policy"]["structured_output"] = {
                "schema": "grok-native-json-schema@1",
                "response_schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
                "response_schema_bytes": len(schema_json.encode()),
                "native_schema_projection": projection,
                "native_schema_requested": True,
                "response_schema_validated": False,
                "plan_admitted": False,
            }
    try:
        output = generate_text(
            model_prompt, provider=provider, model_name=model, provider_instance=instance, deps=deps,
            allow_local_fallback=False, allow_cross_provider_fallback=False,
            timeout=timeout, max_tokens=max_output_tokens, max_new_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            # Both purposes launch a workspace-capable agent and require a fresh
            # native receipt. Cached response text cannot replay those effects.
            side_effecting=True,
            task_kind="coding", allocation_path="cli", allocation_session_id=invocation,
            **provider_options,
        )
        failure_phase = "provider_result_validation"
        observation = get_last_cli_observation(provider)
        if observation.get("exit_code") != 0:
            raise RuntimeError("coding CLI did not return an observed successful exit")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("coding router returned an empty response")
        if codex_planning_validator is not None:
            if (observation.get("codex_output_schema_sha256") != codex_schema_projection["native_wire_schema_sha256"]
                    or observation.get("codex_output_schema_bytes") != codex_schema_projection["native_wire_schema_bytes"]):
                raise RuntimeError("Codex planning schema invocation metadata is missing or differs")
            from .codex_planning_schema import validate_planning_response
            output = validate_planning_response(output, codex_planning_validator)
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        if response_format is not None:
            # The shared adapter enforces this too. Keep the runner boundary
            # explicit in case an injected provider returns unchecked text.
            from ipfs_accelerate_py.cli_runtime.grok_structured_output import validate_response_format, decode_response
            _schema, _argument, validator = validate_response_format(response_format)
            wire_validator = None
            if task_contract is not None:
                wire_format = {**response_format, "json_schema": {**response_format["json_schema"], "schema": _wire_schema}}
                _wire, _argument, wire_validator = validate_response_format(wire_format)
            output = decode_response(json.dumps({"text": output}), validator, additional_validator=wire_validator)
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        if encoded is not None:
            failure_phase = "semantic_response_decode"
            from .semantic_router_translation import decode_semantic_router_response
            decoded = decode_semantic_router_response(response=output, encoded=encoded,
                                                      repository=semantic_repository)
            receipt["semantic_response_translation"] = dict(decoded.receipt)
            output = decoded.text
        if coding_reply_receipt is not None:
            # Reserved semantic envelopes always pass through the unchanged
            # strict decoder above; the ordinary contract cannot relabel them.
            failure_phase = "coding_reply_validation"
            if (observation.get("codex_output_schema_sha256") != coding_reply_receipt["native_output_schema_sha256"]
                    or observation.get("codex_output_schema_bytes") != coding_reply_receipt["native_output_schema_bytes"]):
                raise RuntimeError("Codex coding reply schema invocation metadata is missing or differs")
            from .coding_reply_contract import validate_coding_completion_reply
            output = validate_coding_completion_reply(output)
            coding_reply_receipt["response_validated"] = True
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        receipt["status"] = "provider_returned"
        return output, receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, failure_phase=failure_phase)
        if failure_phase == "provider_invocation":
            receipt["provider_failure"] = _provider_failure_diagnostic(error, provider=provider)
        if failure_phase == "semantic_response_decode":
            from .semantic_router_translation import SemanticTranslationError
            # The closed projection never serializes model text or arbitrary
            # exception messages. Candidate rejection remains unchanged.
            if type(error) is SemanticTranslationError:
                receipt["semantic_response_failure"] = {
                    "phase": failure_phase, "reason_code": error.reason_code,
                }
        raise
    finally:
        observation = get_last_cli_observation(provider)
        # Keep native categories intact; do not add cached tokens to input or
        # infer a missing total. Provider failures remain measured attempts.
        receipt["usage"] = {key: observation[key] for key in (
            "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
            "total_cost_usd", "model_id", "session_id", "thread_id", "num_turns", "exit_code", "timed_out",
            "codex_output_schema_sha256", "codex_output_schema_bytes",
        ) if key in observation}
        if provider == "codex_cli":
            from .codex_usage_receipt import recover_codex_usage
            receipt["native_rollout_usage"] = recover_codex_usage(
                home=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"),
                thread_id=str(observation.get("thread_id") or observation.get("session_id") or ""),
                workspace=root)
        else:
            from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_outcome_receipt, grok_usage_receipt
            receipt["native_rollout_usage"] = grok_usage_receipt(observation)
            receipt["native_provider_outcome"] = grok_outcome_receipt(observation)
            if receipt["native_provider_outcome"]["stop_reason"] != "unknown":
                receipt["usage"]["stop_reason"] = receipt["native_provider_outcome"]["stop_reason"]
        receipt["seconds"] = time.monotonic() - started
        print(json.dumps(receipt, sort_keys=True), flush=True)
        if previous is None:
            os.environ.pop("ipfs_accelerate_py_CODEX_SANDBOX", None)
        else:
            os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] = previous
        if previous_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="codex_cli", choices=["codex_cli", "grok_cli"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--purpose", default="coding", choices=["planning", "coding"])
    parser.add_argument("--container-boundary", type=Path)
    parser.add_argument("--container-boundary-sha256", default="")
    parser.add_argument("--semantic-repository", type=Path)
    parser.add_argument("--semantic-transport-schema", default="supervisor-semantic-router-input@1",
                        choices=["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
    parser.add_argument("--coding-reply-mode", default="legacy",
                        choices=["legacy", "ordinary-completion@1"])
    parser.add_argument("--doctor-residual-artifact", type=Path)
    parser.add_argument("--doctor-residual-sha256", default="")
    parser.add_argument("--doctor-residual-task-cid", default="")
    parser.add_argument("--public-instruction-artifact", type=Path)
    parser.add_argument("--public-instruction-sha256", default="")
    parser.add_argument("--public-instruction-task-cid", default="")
    args = parser.parse_args()
    try:
        prompt = sys.stdin.buffer.read(256_001).decode("utf-8")
        output, _ = run(prompt=prompt, provider=args.provider, model=args.model,
                        timeout=args.timeout, max_output_tokens=args.max_output_tokens,
                        reasoning_effort=args.reasoning_effort,
                        container_boundary=args.container_boundary,
                        container_boundary_sha256=args.container_boundary_sha256,
                        purpose=args.purpose, semantic_repository=args.semantic_repository,
                        semantic_transport_schema=args.semantic_transport_schema,
                        coding_reply_mode=args.coding_reply_mode,
                        doctor_residual_artifact=args.doctor_residual_artifact,
                        doctor_residual_sha256=args.doctor_residual_sha256,
                        doctor_residual_task_cid=args.doctor_residual_task_cid,
                        public_instruction_artifact=args.public_instruction_artifact,
                        public_instruction_sha256=args.public_instruction_sha256,
                        public_instruction_task_cid=args.public_instruction_task_cid)
    except Exception as error:
        try:
            diagnostic = _runner_failure_diagnostic(error)
            failure = {"schema": "router-implementation-error@2",
                       "error_type": diagnostic["exceptions"][0]["exception_type"],
                       "diagnostic": diagnostic}
        except Exception:
            # Preserve the failure exit even if observation itself fails.
            failure = {"schema": "router-implementation-error@1", "error_type": "other"}
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

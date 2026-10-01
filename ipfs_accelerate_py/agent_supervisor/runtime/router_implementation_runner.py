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
        semantic_repository: Path | None = None, doctor_residual_artifact: Path | None = None,
        doctor_residual_sha256: str = "", doctor_residual_task_cid: str = "",
        public_instruction_artifact: Path | None = None, public_instruction_sha256: str = "",
        public_instruction_task_cid: str = "") -> tuple[str, dict]:
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation.intelligence_index import discover_available_providers, select_efficient_route
    from ipfs_accelerate_py.llm_router import generate_text, get_llm_provider
    from ipfs_accelerate_py.router_deps import RouterDeps

    if provider != "codex_cli":
        raise ValueError("this version qualifies the router's codex_cli coding route only")
    if not model or not 1 <= timeout <= 600 or not 1 <= max_output_tokens <= 16_384:
        raise ValueError("explicit model and bounded invocation settings required")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("explicit supported reasoning effort required")
    if not prompt.strip() or len(prompt.encode()) > 256_000:
        raise ValueError("implementation prompt exceeds its byte bound")
    root = Path.cwd().resolve()
    boundary = None
    if container_boundary is not None or container_boundary_sha256:
        from .container_worker_boundary import verify_container_worker_boundary
        if container_boundary is None:
            raise ValueError("container boundary artifact required")
        boundary = verify_container_worker_boundary(
            artifact=container_boundary, expected_sha256=container_boundary_sha256, workspace=root,
        )
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError("implementation must start at its allocated Git worktree root")
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
        encoded = encode_semantic_router_prompt(prompt=prompt, repository=semantic_repository)
        router_prompt = encoded.provider_prompt
    residual_args = (doctor_residual_artifact, doctor_residual_sha256, doctor_residual_task_cid)
    if any(residual_args):
        if not all(residual_args) or semantic_repository is None or purpose != "coding":
            raise ValueError("exact residual binding requires the full semantic coding route")
        from .doctor_residual_context import load_doctor_residual_advisory
        residual_advisory, residual_receipt = load_doctor_residual_advisory(
            artifact=doctor_residual_artifact, expected_sha256=doctor_residual_sha256,
            repository=semantic_repository, task_cid=doctor_residual_task_cid, prompt=prompt, workspace=root)
        router_prompt += residual_advisory
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
    model_prompt, advisory = render_model_prompt(prompt=router_prompt, purpose=purpose, workspace=root,
                                                semantic_transport=encoded is not None)
    if len(model_prompt.encode()) > 256_000:
        raise ValueError("model prompt with workspace contract exceeds its byte bound")
    # Native provider children must never inherit a typed owner token, private
    # bootstrap channel, or database program. The daemon supplies this scrub.
    from .multi_supervisor_runner import DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES
    if any(name in os.environ for name in (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
                                           "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN")):
        raise ValueError("implementation inherited database authority")
    available = discover_available_providers()
    if provider not in available:
        raise ValueError("configured shared-router provider is unavailable")
    route = select_efficient_route(model_name=model, provider=provider, task_kind="coding", available_providers=available)
    if route.provider != provider or route.model_name != model:
        raise ValueError("model manager changed the explicit benchmark route")
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
    try:
        output = generate_text(
            model_prompt, provider=provider, model_name=model, provider_instance=instance, deps=deps,
            allow_local_fallback=False, allow_cross_provider_fallback=False,
            timeout=timeout, max_tokens=max_output_tokens, max_new_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            task_kind="coding", allocation_path="cli", allocation_session_id=invocation,
        )
        observation = get_last_cli_observation(provider)
        if observation.get("exit_code") != 0:
            raise RuntimeError("coding CLI did not return an observed successful exit")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("coding router returned an empty response")
        if encoded is not None:
            from .semantic_router_translation import decode_semantic_router_response
            decoded = decode_semantic_router_response(response=output, encoded=encoded,
                                                      repository=semantic_repository)
            receipt["semantic_response_translation"] = dict(decoded.receipt)
            output = decoded.text
        receipt["status"] = "provider_returned"
        return output, receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        observation = get_last_cli_observation(provider)
        # Keep native categories intact; do not add cached tokens to input or
        # infer a missing total. Provider failures remain measured attempts.
        receipt["usage"] = {key: observation[key] for key in (
            "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
            "total_cost_usd", "model_id", "session_id", "thread_id", "num_turns", "exit_code", "timed_out",
        ) if key in observation}
        from .codex_usage_receipt import recover_codex_usage
        receipt["native_rollout_usage"] = recover_codex_usage(
            home=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"),
            thread_id=str(observation.get("thread_id") or observation.get("session_id") or ""),
            workspace=root,
        )
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
    parser.add_argument("--provider", default="codex_cli", choices=["codex_cli"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--purpose", default="coding", choices=["planning", "coding"])
    parser.add_argument("--container-boundary", type=Path)
    parser.add_argument("--container-boundary-sha256", default="")
    parser.add_argument("--semantic-repository", type=Path)
    parser.add_argument("--doctor-residual-artifact", type=Path)
    parser.add_argument("--doctor-residual-sha256", default="")
    parser.add_argument("--doctor-residual-task-cid", default="")
    parser.add_argument("--public-instruction-artifact", type=Path)
    parser.add_argument("--public-instruction-sha256", default="")
    parser.add_argument("--public-instruction-task-cid", default="")
    args = parser.parse_args()
    prompt = sys.stdin.buffer.read(256_001).decode("utf-8")
    try:
        output, _ = run(prompt=prompt, provider=args.provider, model=args.model,
                        timeout=args.timeout, max_output_tokens=args.max_output_tokens,
                        reasoning_effort=args.reasoning_effort,
                        container_boundary=args.container_boundary,
                        container_boundary_sha256=args.container_boundary_sha256,
                        purpose=args.purpose, semantic_repository=args.semantic_repository,
                        doctor_residual_artifact=args.doctor_residual_artifact,
                        doctor_residual_sha256=args.doctor_residual_sha256,
                        doctor_residual_task_cid=args.doctor_residual_task_cid,
                        public_instruction_artifact=args.public_instruction_artifact,
                        public_instruction_sha256=args.public_instruction_sha256,
                        public_instruction_task_cid=args.public_instruction_task_cid)
    except Exception as error:
        print(json.dumps({"schema": "router-implementation-error@1", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

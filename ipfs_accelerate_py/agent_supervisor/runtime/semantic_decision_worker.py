"""Owner-bound, opt-in semantic evidence dialogue for a coding worker.

The launch pins an external selection descriptor independently of model input.
This entry point preserves native public requirements, source verification,
the container boundary and the complete dialogue receipt. It grants neither
candidate acceptance nor authority to classify capsules as optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from . import semantic_router_translation as codec
from .local_prompt_tokenizer import (
    _bounded_local_bytes, _json_object, _local_path, _sha,
)
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter
from .semantic_decision_dialogue import SemanticDecisionDialogueError, SemanticDecisionResult
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_decision_provider import run_shared_router_decision_dialogue
from .semantic_projection_dispatch import project_semantic_router_prompt

if TYPE_CHECKING:
    from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding

SCHEMA = "supervisor-semantic-decision-launch-policy@1"
RETRIEVAL_SCHEMA = "supervisor-semantic-decision-launch-policy@3"
SCHEMAS = frozenset({SCHEMA, "supervisor-semantic-decision-launch-policy@2", RETRIEVAL_SCHEMA})
REFUSAL_SCHEMA = "supervisor-semantic-decision-worker-refusal@1"
MAX_POLICY_BYTES = 256_000
_FIELDS = frozenset({"schema", "task_id", "parent_artifact", "parent_sha256", "policy"})


class SemanticDecisionWorkerPreparationError(ValueError):
    """A refusal known to precede the shared provider dialogue."""


def _canonical_directory(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    if candidate.resolve(strict=True) != candidate or not candidate.is_dir():
        raise ValueError(f"canonical existing {label} required")
    return candidate


def build_decision_launch_policy(
    *, native_prompt: str, policy: SemanticDecisionPolicy, repository: Path,
    retrieval_query_binding: SemanticRetrievalQueryBinding | None = None,
) -> dict:
    """Authenticate the native producer and explicit partition without a call."""
    root = _canonical_directory(repository, "semantic repository")
    if type(policy) is not SemanticDecisionPolicy:
        raise ValueError("independent typed semantic selection policy required")
    wire, _, _, _, text, artifact = codec._prompt_parts(native_prompt)
    # This verifies the full native artifact, source freshness and the complete
    # required/optional partition; scores and raw-source flags infer nothing.
    options = {}
    if retrieval_query_binding is not None:
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        if (type(retrieval_query_binding) is not SemanticRetrievalQueryBinding
                or policy.task_text_source_path is None):
            raise ValueError("retrieval query reference requires a bound parent and required task text policy")
        options["retrieval_query_binding"] = retrieval_query_binding
    project_semantic_router_prompt(prompt=native_prompt, repository=root, decision_policy=policy, **options)
    descriptor = {
        "schema": "supervisor-semantic-decision-launch-policy@" + policy.schema_revision,
        "task_id": wire["objective_id"],
        "parent_artifact": artifact,
        "parent_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "policy": policy.to_dict(),
    }
    if retrieval_query_binding is not None:
        descriptor.update(schema=RETRIEVAL_SCHEMA, retrieval_query_binding=retrieval_query_binding.to_dict())
    if len(codec._json(descriptor).encode()) > MAX_POLICY_BYTES:
        raise ValueError("semantic decision launch policy exceeds its byte bound")
    return descriptor


def load_decision_launch_configuration(
    *, artifact: Path, expected_sha256: str, native_prompt: str,
    repository: Path, workspace: Path,
) -> tuple[SemanticDecisionPolicy, SemanticRetrievalQueryBinding | None]:
    """Load a bounded external descriptor and replay its exact native binding."""
    expected_sha256 = _sha(expected_sha256, "semantic decision launch policy")
    work = _canonical_directory(workspace, "allocated workspace")
    path = _local_path(artifact)
    if path.resolve(strict=True) != path or path.is_relative_to(work):
        raise ValueError("semantic decision launch policy must be canonical and outside the candidate")
    raw = _bounded_local_bytes(path, maximum=MAX_POLICY_BYTES, expected_sha256=expected_sha256)
    value = _json_object(raw, "semantic decision launch policy")
    retrieval = value.get("schema") == RETRIEVAL_SCHEMA
    fields = _FIELDS | {"retrieval_query_binding"} if retrieval else _FIELDS
    if (set(value) != fields or type(value.get("schema")) is not str
            or value["schema"] not in SCHEMAS):
        raise ValueError("exact semantic decision launch policy schema and fields required")
    policy = SemanticDecisionPolicy.from_dict(value["policy"])
    if not retrieval and value["schema"] != "supervisor-semantic-decision-launch-policy@" + policy.schema_revision:
        raise ValueError("semantic decision launch policy schema and policy revision differ")
    binding = None
    if retrieval:
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        binding = SemanticRetrievalQueryBinding.from_dict(value["retrieval_query_binding"])
    if binding is not None and (Path(repository) / binding.artifact).resolve(strict=True).is_relative_to(work):
        raise ValueError("retrieval parent must remain outside the allocated candidate")
    options = {"retrieval_query_binding": binding} if binding is not None else {}
    expected = build_decision_launch_policy(native_prompt=native_prompt, policy=policy, repository=repository, **options)
    if value != expected:
        raise ValueError("semantic decision launch policy differs from its native parent or task")
    return policy, binding


def load_decision_launch_policy(
    *, artifact: Path, expected_sha256: str, native_prompt: str,
    repository: Path, workspace: Path,
) -> SemanticDecisionPolicy:
    """Load legacy policy-only launches without discarding extra parent custody."""
    policy, binding = load_decision_launch_configuration(
        artifact=artifact, expected_sha256=expected_sha256, native_prompt=native_prompt,
        repository=repository, workspace=workspace,
    )
    if binding is not None:
        raise ValueError("retrieval launch requires the complete launch configuration loader")
    return policy


def _prepare_launch(
    *, prompt: str, model: str, semantic_repository: Path,
    semantic_decision_policy_artifact: Path, semantic_decision_policy_sha256: str,
    provider: str = "codex_cli", purpose: str = "coding", reasoning_effort: str = "high",
    timeout: int = 90, max_output_tokens: int = 4096,
    semantic_decision_max_calls: int = 3,
    semantic_decision_max_total_input_tokens: int | None = None,
    max_input_tokens: int | None = None,
    container_boundary: Path | None = None, container_boundary_sha256: str = "",
    prompt_tokenizer_artifact: Path | None = None, prompt_tokenizer_sha256: str = "",
    doctor_residual_artifact: Path | None = None, doctor_residual_sha256: str = "",
    doctor_residual_task_cid: str = "", public_instruction_artifact: Path | None = None,
    public_instruction_sha256: str = "", public_instruction_task_cid: str = "",
) -> tuple[dict, dict]:
    """Prepare literal suffixes before any complete-prompt count or invocation."""
    if provider != "codex_cli" or purpose != "coding":
        raise ValueError("Terminal semantic dialogue qualifies codex_cli coding only")
    if type(model) is not str or not model or model != model.strip():
        raise ValueError("explicit pinned model required")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("explicit supported reasoning effort required")
    for value, maximum in ((timeout, 600), (max_output_tokens, 16_384),
                           (semantic_decision_max_calls, 4)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("positive bounded dialogue settings required")
    for limit in (max_input_tokens, semantic_decision_max_total_input_tokens):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("positive complete prompt token budgets required")
    if type(prompt) is not str or not prompt.strip() or len(prompt.encode()) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("bounded nonempty native implementation prompt required")
    work = _canonical_directory(Path.cwd(), "allocated workspace")
    root = _canonical_directory(semantic_repository, "semantic repository")
    if work == root:
        raise ValueError("semantic decision requires a separate allocated worktree")
    boundary_args = (container_boundary, container_boundary_sha256)
    if any(boundary_args) and not all(boundary_args):
        raise ValueError("complete container boundary binding required")
    if container_boundary is not None:
        _sha(container_boundary_sha256, "container boundary")
        from .container_worker_boundary import verify_container_worker_boundary
        verify_container_worker_boundary(
            artifact=container_boundary, expected_sha256=container_boundary_sha256, workspace=work,
        )
    policy, retrieval_binding = load_decision_launch_configuration(
        artifact=semantic_decision_policy_artifact, expected_sha256=semantic_decision_policy_sha256,
        native_prompt=prompt, repository=root, workspace=work,
    )
    suffix = ""
    bindings = {
        "launch_policy_artifact": str(_local_path(semantic_decision_policy_artifact)),
        "launch_policy_sha256": semantic_decision_policy_sha256,
        "original_native_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "original_native_prompt_bytes": len(prompt.encode()),
    }
    if retrieval_binding is not None:
        bindings["retrieval_query_binding"] = retrieval_binding.to_dict()
    residual_args = (doctor_residual_artifact, doctor_residual_sha256, doctor_residual_task_cid)
    if any(residual_args):
        if not all(residual_args):
            raise ValueError("complete Doctor residual binding required")
        from .doctor_residual_context import load_doctor_residual_advisory
        advisory, receipt = load_doctor_residual_advisory(
            artifact=doctor_residual_artifact, expected_sha256=doctor_residual_sha256,
            repository=root, task_cid=doctor_residual_task_cid, prompt=prompt, workspace=work,
        )
        suffix += advisory
        bindings["doctor_residual_context"] = receipt
    instruction_args = (public_instruction_artifact, public_instruction_sha256, public_instruction_task_cid)
    if any(instruction_args):
        if not all(instruction_args):
            raise ValueError("complete public instruction binding required")
        from .router_public_instruction import load_public_instruction
        instruction, receipt = load_public_instruction(
            artifact=public_instruction_artifact, expected_sha256=public_instruction_sha256,
            task_cid=public_instruction_task_cid, prompt=prompt, workspace=work, repository=root,
        )
        suffix += instruction
        bindings["public_instruction"] = receipt
    prepared = prompt + suffix
    if len(prepared.encode()) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("native prompt with literal required suffix exceeds its byte bound")
    tokenizer_args = (prompt_tokenizer_artifact, prompt_tokenizer_sha256)
    if any(tokenizer_args):
        if not all(tokenizer_args):
            raise ValueError("complete local tokenizer artifact binding required")
        from .local_prompt_tokenizer import load_owner_prompt_tokenizer
        local = load_owner_prompt_tokenizer(
            artifact=prompt_tokenizer_artifact, expected_sha256=prompt_tokenizer_sha256,
            provider=provider, model=model,
        )
        counter = PromptTokenCounter(**local.binding(), count_tokens=local.count,
                                     compatibility_basis="owner_declaration")
        tokenizer_source = local.provenance()
    else:
        from .automatic_prompt_tokenizer import resolve_prompt_tokenizer
        counter, tokenizer_source = resolve_prompt_tokenizer(provider=provider, model=model)
    bindings.update(
        required_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
        required_suffix_bytes=len(suffix.encode()), prompt_tokenizer_source=tokenizer_source,
    )
    return dict(
        native_prompt=prepared, repository=root, workspace=work, policy=policy,
        provider=provider, model=model, prompt_token_counter=counter,
        reasoning_effort=reasoning_effort, container_boundary=container_boundary,
        container_boundary_sha256=container_boundary_sha256,
        max_calls=semantic_decision_max_calls, timeout=timeout, max_output_tokens=max_output_tokens,
        max_input_tokens=max_input_tokens,
        max_total_input_tokens=semantic_decision_max_total_input_tokens,
        **({"retrieval_query_binding": retrieval_binding} if retrieval_binding is not None else {}),
    ), bindings


def run(**kwargs) -> SemanticDecisionResult:
    """Retain dialogue failures without treating later errors as refusals."""
    try:
        arguments, bindings = _prepare_launch(**kwargs)
    except Exception as error:
        raise SemanticDecisionWorkerPreparationError("worker preparation refused") from error
    try:
        result = run_shared_router_decision_dialogue(**arguments)
    except SemanticDecisionDialogueError as error:
        receipt = error.receipt
        receipt["worker_launch"] = bindings
        raise SemanticDecisionDialogueError(str(error), receipt) from error
    receipt = result.receipt
    receipt["worker_launch"] = bindings
    return SemanticDecisionResult(result.text, json.dumps(receipt, sort_keys=True, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="codex_cli", choices=["codex_cli"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--purpose", default="coding", choices=["coding"])
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--semantic-repository", required=True, type=Path)
    parser.add_argument("--semantic-decision-policy-artifact", required=True, type=Path)
    parser.add_argument("--semantic-decision-policy-sha256", required=True)
    parser.add_argument("--semantic-decision-max-calls", type=int, default=3)
    parser.add_argument("--semantic-decision-max-total-input-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    for name in ("container-boundary", "prompt-tokenizer", "doctor-residual", "public-instruction"):
        artifact_name = name if name == "container-boundary" else name + "-artifact"
        parser.add_argument("--" + artifact_name, type=Path)
        parser.add_argument("--" + name + "-sha256", default="")
        if name in {"doctor-residual", "public-instruction"}:
            parser.add_argument("--" + name + "-task-cid", default="")
    args = parser.parse_args(argv)
    try:
        try:
            prompt = sys.stdin.buffer.read(MAX_MODEL_PROMPT_BYTES + 1).decode("utf-8")
        except Exception as error:
            raise SemanticDecisionWorkerPreparationError("native worker input unavailable") from error
        result = run(prompt=prompt, **vars(args))
    except SemanticDecisionDialogueError as error:
        print(json.dumps(error.receipt, sort_keys=True, allow_nan=False), flush=True)
        return 1
    except Exception as error:
        preparation = isinstance(error, SemanticDecisionWorkerPreparationError)
        retained = getattr(error, "receipt", None)
        receipt = {
            "schema": REFUSAL_SCHEMA, "status": "refused" if preparation else "failed",
            "error_type": type(error.__cause__ if preparation else error).__name__,
            "provider_called": False if preparation else None,
            "usage": None, "completion_authority": False,
        }
        if type(retained) is dict:
            receipt["retained_receipt"] = retained
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 1
    print(json.dumps(result.receipt, sort_keys=True, allow_nan=False), flush=True)
    print(result.text, flush=True)
    return 0


if __name__ == "__main__":
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker import main as _canonical_main

    raise SystemExit(_canonical_main())

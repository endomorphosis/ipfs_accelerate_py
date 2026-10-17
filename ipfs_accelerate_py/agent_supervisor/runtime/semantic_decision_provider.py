"""Pinned shared-router adapters for supervisor-owned semantic dialogue.

The supervisor retains policy and authenticates source around every call. This
adapter grants no expansion or completion authority. Codex uses the existing
coding runner; Grok is a separate, tool-free structured-response route, with
one fresh native CLI session per supervisor call and no provider fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from .router_implementation_runner import PromptTokenCounter, render_model_prompt
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_decision_packet import SemanticDecisionPolicy

SCHEMA = "router-semantic-decision-provider@1"
_USAGE_FIELDS = (
    "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
    "total_cost_usd", "model_id", "session_id", "thread_id", "request_id",
    "num_turns", "exit_code", "timed_out",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _copy(value: dict) -> dict:
    return json.loads(json.dumps(value, allow_nan=False))


def _canonical_directory(path: Path, name: str) -> Path:
    candidate = Path(path).absolute()
    if candidate.resolve(strict=True) != candidate or not candidate.is_dir():
        raise ValueError(f"canonical {name} directory required")
    return candidate


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-c", "safe.directory=" + str(root), "-C", str(root),
         *arguments], text=True, timeout=10,
    ).strip()


class SharedRouterDecisionProvider:
    """One exact provider/model/workspace binding for checked dialogue calls."""

    def __init__(
        self, *, provider: str, model: str, repository: Path, workspace: Path,
        prompt_token_counter: PromptTokenCounter, reasoning_effort: str = "high",
        container_boundary: Path | None = None, container_boundary_sha256: str = "",
    ):
        if provider not in {"codex_cli", "grok_cli"}:
            raise ValueError("explicit supported semantic decision provider required")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("explicit pinned semantic decision model required")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("explicit supported reasoning effort required")
        if (not isinstance(prompt_token_counter, PromptTokenCounter)
                or prompt_token_counter.provider != provider
                or prompt_token_counter.model != model):
            raise ValueError("token counter differs from pinned decision provider/model")
        if (container_boundary is None) != (container_boundary_sha256 == ""):
            raise ValueError("complete container boundary binding required")
        self.provider, self.model = provider, model
        self.repository = _canonical_directory(repository, "source repository")
        self.workspace = _canonical_directory(workspace, "allocated workspace")
        self.prompt_token_counter = prompt_token_counter
        self.reasoning_effort = reasoning_effort
        self.container_boundary = container_boundary
        self.container_boundary_sha256 = container_boundary_sha256

    def _validate(self, call):
        from .semantic_decision_dialogue import SemanticDecisionCall
        from .multi_supervisor_runner import (
            DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES,
        )

        if not isinstance(call, SemanticDecisionCall):
            raise ValueError("typed semantic decision call required")
        if call.provider != self.provider or call.model != self.model:
            raise ValueError("decision call differs from pinned provider/model")
        if (Path.cwd().resolve() != self.workspace
                or _canonical_directory(self.workspace, "allocated workspace") != self.workspace
                or _canonical_directory(self.repository, "source repository") != self.repository):
            raise ValueError("decision call must start in its pinned allocated workspace")
        if self.repository == self.workspace:
            raise ValueError("semantic decision requires a separate allocated worktree")
        if Path(_git(self.workspace, "rev-parse", "--show-toplevel")).resolve() != self.workspace:
            raise ValueError("decision workspace must be its allocated Git worktree root")
        if Path(_git(self.repository, "rev-parse", "--show-toplevel")).resolve() != self.repository:
            raise ValueError("semantic source must be its pinned Git repository root")
        common = "rev-parse", "--path-format=absolute", "--git-common-dir"
        if Path(_git(self.repository, *common)).resolve() != Path(_git(self.workspace, *common)).resolve():
            raise ValueError("semantic source repository differs from allocated worktree")
        if any(name in os.environ for name in (
            *DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
            "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN",
        )):
            raise ValueError("semantic decision provider inherited database authority")
        if (not isinstance(call.prompt, str) or not call.prompt.strip()
                or len(call.prompt.encode()) > 256_000):
            raise ValueError("bounded semantic decision prompt required")
        if (type(call.timeout) is not int or not 1 <= call.timeout <= 600
                or type(call.max_output_tokens) is not int
                or not 1 <= call.max_output_tokens <= 16_384):
            raise ValueError("bounded semantic decision invocation required")
        rendered, _ = render_model_prompt(
            prompt=call.prompt, purpose="coding", workspace=self.workspace,
        )
        if len(rendered.encode()) > 256_000 or _sha(rendered) != call.model_prompt_sha256:
            raise ValueError("decision call differs from the complete rendered model prompt")
        return rendered

    def __call__(self, call):
        from .semantic_decision_dialogue import (
            SemanticDecisionProviderFailure, SemanticDecisionReply,
        )

        model_prompt = self._validate(call)
        receipt = {
            "schema": SCHEMA, "call_id": call.call_id, "session_id": call.session_id,
            "generation": call.generation, "binding_sha256": call.binding_sha256,
            "provider": self.provider, "model": self.model,
            "prompt_sha256": _sha(call.prompt), "prompt_bytes": len(call.prompt.encode()),
            "model_prompt_sha256": _sha(model_prompt),
            "model_prompt_bytes": len(model_prompt.encode()),
            "workspace": str(self.workspace), "source_repository": str(self.repository),
            "timeout_seconds": call.timeout, "requested_output_tokens": call.max_output_tokens,
            "provider_output_token_cap_enforced": False,
            "status": "not_started", "usage": None, "usage_status": "unknown",
            "provider_called": False, "completion_authority": False,
        }
        started = time.monotonic()
        try:
            if self.provider == "codex_cli":
                output = self._codex(call, receipt)
            else:
                output = self._grok(call, model_prompt, receipt)
            receipt["status"] = "provider_returned"
        except Exception as error:
            receipt.update(status="failed", error_type=type(error).__name__)
            receipt["seconds"] = time.monotonic() - started
            raise SemanticDecisionProviderFailure(
                "semantic decision provider failed", receipt=_copy(receipt),
            ) from error
        receipt["seconds"] = time.monotonic() - started
        return SemanticDecisionReply(text=output, receipt=_copy(receipt))

    def _codex(self, call, receipt):
        from . import router_implementation_runner as runner

        def observe(native):
            native = _copy(native)
            receipt["native_invocation"] = native
            receipt["usage"] = native.get("usage")
            receipt["native_rollout_usage"] = native.get("native_rollout_usage")
            receipt["provider_called"] = True
            receipt["usage_status"] = self._usage_status(native.get("usage"))

        output, native = runner.run(
            prompt=call.prompt, provider=self.provider, model=self.model,
            timeout=call.timeout, max_output_tokens=call.max_output_tokens,
            reasoning_effort=self.reasoning_effort, purpose="coding",
            semantic_repository=None, prompt_token_counter=self.prompt_token_counter,
            container_boundary=self.container_boundary,
            container_boundary_sha256=self.container_boundary_sha256,
            codex_cli_literal_response=True,
            receipt_observer=observe,
        )
        if "native_invocation" not in receipt:
            observe(native)
        if native.get("model_prompt_sha256") != call.model_prompt_sha256:
            raise ValueError("native decision runner changed the complete model prompt")
        if native.get("provider") != self.provider or native.get("model") != self.model:
            raise ValueError("native decision runner changed the pinned provider/model")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("decision coding CLI returned an empty response")
        return output

    @staticmethod
    def _usage_status(usage):
        if not isinstance(usage, dict):
            return "unknown"
        if all(type(usage.get(field)) is int and usage[field] >= 0
               for field in ("prompt_tokens", "completion_tokens")):
            return "reported"
        return "partial" if any(field in usage for field in (
            "prompt_tokens", "completion_tokens", "total_cost_usd",
        )) else "unknown"

    def _grok(self, call, model_prompt, receipt):
        from ipfs_accelerate_py import llm_router
        from ipfs_accelerate_py.cli_runtime.cli_metadata import (
            get_last_cli_observation, set_last_cli_observation,
        )
        from ipfs_accelerate_py.cli_runtime.grok_native_usage import (
            grok_outcome_receipt, grok_usage_receipt,
        )
        from ipfs_accelerate_py.llm_allocation.intelligence_index import (
            discover_available_providers, select_efficient_route,
        )
        from ipfs_accelerate_py.router_deps import RouterDeps

        if self.container_boundary is not None:
            from .container_worker_boundary import verify_container_worker_boundary
            receipt["external_container_boundary"] = verify_container_worker_boundary(
                artifact=self.container_boundary, expected_sha256=self.container_boundary_sha256,
                workspace=self.workspace,
            )
        available = discover_available_providers()
        if self.provider not in available:
            raise ValueError("configured semantic decision provider is unavailable")
        route = select_efficient_route(
            model_name=self.model, provider=self.provider, task_kind="coding",
            available_providers=available,
        )
        if route.provider != self.provider or route.model_name != self.model:
            raise ValueError("model manager changed the pinned semantic decision route")
        command = shlex.split(llm_router._grok_cli_command())
        if len(command) != 1 or Path(command[0]).name not in {"grok", "agent"}:
            raise ValueError("tool-free Grok dialogue requires one structured CLI executable")
        from .grok_cli_runner import _grok_executable_extension_paths, _resolve_trusted_grok_bin
        trusted = _resolve_trusted_grok_bin(configured=command[0], workspace=self.workspace)
        if not trusted:
            raise ValueError("Grok dialogue executable must be trusted and outside the writable workspace")
        if _grok_executable_extension_paths(self.workspace):
            raise ValueError("Grok dialogue refuses project MCP, hook, plugin or LSP configuration")
        command = [trusted]
        deps = RouterDeps()
        instance = llm_router.get_llm_provider(self.provider, deps=deps, use_cache=False)
        set_last_cli_observation(self.provider, {})
        # An empty Grok allowlist retains its default tools. Match the native
        # planning boundary: request one named tool, then deny it and both MCP
        # dispatch tools. These argv settings still need native qualification.
        receipt.update(provider_called=True, native_tool_profile="none", native_max_turns=1,
                       native_tool_allowlist=["read_file"],
                       native_tool_denylist=["read_file", "search_tool", "use_tool"],
                       effective_toolset_verified=False)
        try:
            output = llm_router.generate_text(
                model_prompt, provider=self.provider, model_name=self.model,
                provider_instance=instance, deps=deps,
                allow_local_fallback=False, allow_cross_provider_fallback=False,
                timeout=call.timeout, max_tokens=call.max_output_tokens,
                max_new_tokens=call.max_output_tokens, reasoning_effort=self.reasoning_effort,
                temperature=0, task_kind="coding", allocation_path="cli",
                allocation_session_id=call.call_id,
                grok_cli_cmd=command, grok_tools="read_file",
                grok_disallowed_tools="read_file,search_tool,use_tool",
                grok_permission_mode="dontAsk", grok_max_turns=1,
                chat_session_id=call.call_id,
            )
            observed = get_last_cli_observation(self.provider)
            if observed.get("exit_code") != 0:
                raise RuntimeError("decision Grok CLI did not return an observed successful exit")
            if not isinstance(output, str) or not output.strip():
                raise RuntimeError("decision Grok CLI returned an empty response")
            return output
        finally:
            observed = get_last_cli_observation(self.provider)
            usage = {key: observed[key] for key in _USAGE_FIELDS if key in observed}
            receipt["usage"] = usage
            receipt["usage_status"] = self._usage_status(usage)
            # Keep native final-envelope evidence on both return and failure.
            # These closed projections do not establish complete task usage.
            receipt["native_rollout_usage"] = grok_usage_receipt(observed)
            try:
                from ipfs_accelerate_py.cli_runtime.grok_prompt_accounting import grok_prompt_accounting_receipt
                receipt["native_prompt_accounting"] = grok_prompt_accounting_receipt(observed)
            except Exception:
                receipt["native_prompt_accounting"] = None
            receipt["native_provider_outcome"] = grok_outcome_receipt(observed)


def run_shared_router_decision_dialogue(
    *, native_prompt: str, repository: Path, workspace: Path,
    policy: SemanticDecisionPolicy, provider: str, model: str,
    prompt_token_counter: PromptTokenCounter | None = None,
    reasoning_effort: str = "high", container_boundary: Path | None = None,
    container_boundary_sha256: str = "", max_calls: int = 3, timeout: int = 90,
    max_output_tokens: int = 4096, max_input_tokens: int | None = None,
    max_total_input_tokens: int | None = None,
    max_observed_native_tokens_before_next_dispatch: int | None = None,
    max_expansion_bytes: int = 16384, max_expansion_tokens: int = 4096, cancelled=None,
    retrieval_query_binding=None,
):
    """Compose the checked dialogue and its pinned native-provider adapter.

    Automatic tokenizer provenance remains distinct from reported provider
    usage. This library helper does not register a Terminal Bench launch route
    or grant a Grok worker tools for applying its structured response.
    """
    budget_type = None
    if max_observed_native_tokens_before_next_dispatch is not None:
        from .observed_native_usage_budget import (
            ObservedNativeUsageBudget, validate_observed_native_token_threshold,
        )
        validate_observed_native_token_threshold(max_observed_native_tokens_before_next_dispatch)
        budget_type = ObservedNativeUsageBudget
    from .semantic_decision_dialogue import (
        SemanticDecisionDialogueError, SemanticDecisionResult, run_semantic_decision_dialogue,
    )

    dialogue_options = {}
    if retrieval_query_binding is not None:
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        if (type(retrieval_query_binding) is not SemanticRetrievalQueryBinding
                or type(policy) is not SemanticDecisionPolicy or policy.task_text_source_path is None):
            raise ValueError("trusted retrieval binding and task-text reference policy required")
        dialogue_options["retrieval_query_binding"] = retrieval_query_binding
    tokenizer_source = None
    if prompt_token_counter is None:
        from .automatic_prompt_tokenizer import resolve_prompt_tokenizer
        prompt_token_counter, tokenizer_source = resolve_prompt_tokenizer(
            provider=provider, model=model,
        )
    adapter = SharedRouterDecisionProvider(
        provider=provider, model=model, repository=repository, workspace=workspace,
        prompt_token_counter=prompt_token_counter, reasoning_effort=reasoning_effort,
        container_boundary=container_boundary,
        container_boundary_sha256=container_boundary_sha256,
    )
    budget = (budget_type(adapter, provider=provider, model=model, workspace=workspace,
        max_observed_native_tokens_before_next_dispatch=max_observed_native_tokens_before_next_dispatch)
        if budget_type is not None else None)
    try:
        result = run_semantic_decision_dialogue(
            native_prompt=native_prompt, repository=repository, workspace=workspace,
            policy=policy, provider=provider, model=model, invoke=budget if budget is not None else adapter,
            count_tokens=prompt_token_counter, max_calls=max_calls, timeout=timeout,
            max_output_tokens=max_output_tokens, max_input_tokens=max_input_tokens,
            max_total_input_tokens=max_total_input_tokens,
            max_expansion_bytes=max_expansion_bytes, max_expansion_tokens=max_expansion_tokens,
            cancelled=cancelled, **dialogue_options,
        )
    except SemanticDecisionDialogueError as error:
        if budget is None:
            raise
        receipt = error.receipt
        receipt["observed_native_usage_stop"] = budget.observation()
        # The original dialogue error and its exact provider cause remain in
        # the cause chain. Only this explicitly selected helper adds metadata.
        raise SemanticDecisionDialogueError(error.args[0], receipt) from error
    if tokenizer_source is None and budget is None:
        return result
    receipt = result.receipt
    if tokenizer_source is not None:
        receipt["prompt_tokenizer_source"] = tokenizer_source
    if budget is not None:
        receipt["observed_native_usage_stop"] = budget.observation()
    return SemanticDecisionResult(result.text, json.dumps(receipt, allow_nan=False))

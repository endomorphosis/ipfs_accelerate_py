"""Model-free source-input comparison on reviewed public Terminal Bench bytes.

The real producer, DuckDB selector, router, and retained-evidence audit run.
Only the provider boundary is replaced with a local capture. No benchmark
source executes, no task is solved, and proxy counts are never provider usage.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from . import terminal_profile_catalog as catalog
from .terminal_source_view_experiment import local_token_counter

SCHEMA = "terminal-source-input-experiment@1"
POLICIES = ("legacy", "source-input-raw@1", "source-input-compact@1",
            "source-input-ast-ref-raw@1", "source-input-ast-ref-compact@1")
DEFAULT_TASKS = ("tune-mjcf", "llm-inference-batching-scheduler")
MAX_TOTAL_BYTES = 4 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")


def _path(name):
    path = Path(name)
    if (type(name) is not str or not name or path.is_absolute() or path.as_posix() != name
            or any(part in {".", ".."} or part.startswith(".") for part in path.parts)):
        raise ValueError("reviewed input must be a visible canonical relative path")
    return path


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout


@contextmanager
def _captured_provider(workspace):
    """Intercept the sole provider call without consulting host credentials."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    seen = []

    def capture(prompt, **kwargs):
        if (kwargs.get("allow_local_fallback") is not False
                or kwargs.get("allow_cross_provider_fallback") is not False
                or kwargs.get("side_effecting") is not True):
            raise ValueError("qualification requires the explicit provider boundary")
        seen.append(prompt)
        # No fabricated token counters. Exit status describes only this stub.
        set_last_cli_observation("codex_cli", {"exit_code": 0})
        return "Local source-input qualification only; no task execution."

    previous = Path.cwd()
    with ExitStack() as stack:
        stack.enter_context(patch.object(intelligence_index, "discover_available_providers", lambda: ["codex_cli"]))
        stack.enter_context(patch.object(intelligence_index, "select_efficient_route", lambda **kwargs:
            SimpleNamespace(provider="codex_cli", model_name="local-capture", catalog_revision="qualification")))
        stack.enter_context(patch.object(llm_router, "get_llm_provider", lambda *args, **kwargs: object()))
        stack.enter_context(patch.object(llm_router, "generate_text", capture))
        stack.enter_context(patch.dict(os.environ, {"CODEX_HOME": str(workspace.parent / "unused-cli-home")}))
        os.chdir(workspace)
        try:
            yield seen
        finally:
            os.chdir(previous)


def _native_prompt(repository, artifact, task_id, paths):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget

    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(repository).as_posix(),
        repository_id="repo:public-source-input", tree_id="tree:reviewed-inputs", required=True, chunk_bytes=8192)
    result = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:public-source-input", tree_id="tree:reviewed-inputs",
        objective_id=task_id, objective_revision="sha256:public-input-qualification",
        policy_id="policy:local-capture", policy_revision="sha256:source-input@1",
        caller="supervisor:qualification", stage="implementation", goal={"id": task_id},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": paths}, acceptance={"criteria": ["representation qualification only"]}, evidence=refs)
    return render_context_capsule(result.capsule)


def qualify_task(*, name, sources, instruction, output, counter=None, provenance=None, public_profile=None):
    """Capture all five arms on one fixed producer selection and workspace.

    Sources must already be reviewed public inputs or an authored fixture.
    Output keeps canonical evidence; captured model inputs stay in memory.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from .terminal_context_audit import _model_projection
    from .terminal_semantic_context_policy import semantic_context_objective, semantic_context_options

    if (not sources or len(sources) > 64 or any(type(raw) is not bytes for raw in sources.values())
            or sum(map(len, sources.values())) > MAX_TOTAL_BYTES
            or type(instruction) is not str or not instruction.strip() or len(instruction.encode()) > 8192):
        raise ValueError("bounded explicit public inputs and instruction required")
    for path in sources:
        _path(path)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification requires a new canonical output directory")
    output.mkdir(parents=True)
    result = dict(schema=SCHEMA, task_name=name, provenance=provenance or {}, provider_calls=0, local_capture_calls=0,
        containers_started=0, benchmark_source_executed=False, task_solved=None, official_reward=None,
        provider_usage_tokens=None, helper_operations=0, arms=[],
        input_sha256={path: _sha(raw) for path, raw in sorted(sources.items())},
        public_instruction_sha256=_sha(instruction),
        measurement_scope="full captured coding input including router wrappers; local provider stub; no reasoning or tool exchanges")
    if not any(path.endswith(".py") for path in sources):
        result.update(disposition="no_supported_python_input", qualified=False)
        _save(output / "qualification.json", result)
        return result
    repository = output / "repository"
    repository.mkdir()
    for name_, raw in sources.items():
        target = repository / name_
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    from .terminal_task_profile import (INSTRUCTION, SMOKE, PROFILE, task_profile_bytes,
        task_profile_smoke, task_profile_index_paths, validate_task_profile)
    support = INSTRUCTION
    support_sources = {support: instruction.encode()}
    required = [support]
    program_paths, defer_data = sorted(sources), False
    if public_profile is not None:
        public_profile = validate_task_profile(public_profile, instruction=instruction)
        if set(public_profile["input_paths"]) != set(sources):
            raise ValueError("reviewed profile input population differs")
        support_sources.update({PROFILE: task_profile_bytes(public_profile),
                               SMOKE: task_profile_smoke(public_profile).encode()})
        required.append(SMOKE)
        program_paths = task_profile_index_paths(public_profile)
        defer_data = bool(public_profile.get("data_inputs"))
    for support_path, raw in support_sources.items():
        (repository / support_path).write_bytes(raw)
    all_paths = sorted([*sources, *support_sources])
    _git(repository, "init", "-q")
    _git(repository, "add", "--all")
    _git(repository, "-c", "user.name=Source input qualification", "-c",
         "user.email=qualification@example.invalid", "commit", "-qm", "Reviewed copied inputs")
    workspace = output / "allocated-worktree"
    _git(repository, "worktree", "add", "--detach", "-q", str(workspace))
    artifact = repository / ".runtime/semantic/worker-context.json"
    prepared = prepare_semantic_context(repository=repository, paths=all_paths,
        required_raw_paths=required, program_paths=program_paths, defer_task_data=defer_data,
        objective=semantic_context_objective("indexed-default@1", instruction),
        task_id="SOURCE-INPUT-QUALIFICATION", output=artifact.parent, worker_query=instruction,
        max_symbols=1024 if public_profile is not None else 256,
        **semantic_context_options("indexed-default@1"))
    payload = json.loads(artifact.read_text())
    from .terminal_source_payload_census import census_worker_payload
    result.update(selected_capsules=prepared["worker_capsules"], full_capsules=prepared["capsules"],
        selected_symbols_sha256=_sha(_json(payload["worker_projection"]["selected_symbols"])),
        canonical_payload_sha256=_sha(artifact.read_bytes()),
        optional_program_raw_omitted=not bool(set(program_paths) & set(payload["raw_sources"])),
        required_instruction_exact=payload["raw_sources"].get(support) == instruction,
        worker_payload_census=census_worker_payload(payload),
        preparation_scope=("reviewed COPY inputs, canonical task profile/smoke and literal required instruction; production indexed-default selection/bounds; no full supervisor planning/admission or container setup"
            if public_profile is not None else "authored inputs plus literal required instruction; no full supervisor planning/admission or container setup"))
    prompt = _native_prompt(repository, artifact, "SOURCE-INPUT-QUALIFICATION", all_paths)
    captured = []
    for policy in POLICIES:
        with _captured_provider(workspace) as seen, redirect_stdout(io.StringIO()):
            try:
                _, receipt = runner.run(prompt=prompt, provider="codex_cli", model="local-capture", timeout=1,
                    max_output_tokens=128, semantic_repository=repository, source_input_policy=policy)
            except ValueError as error:
                if seen:
                    raise
                # Known ineligibility is retained, never imputed as a treatment.
                from ipfs_accelerate_py.agent_supervisor.runtime.source_input_view import SourceInputViewError
                if (not isinstance(error, SourceInputViewError)
                        or error.code not in {"no_eligible_source_views", "no_compact_source_views"}):
                    raise
                result["arms"].append(dict(policy=policy, disposition="ineligible_before_provider_boundary",
                    reason_code=error.code, positive_input_observed=False,
                    full_input_bytes=None, full_input_proxy_tokens=None))
                continue
        if len(seen) != 1:
            raise ValueError("exactly one local provider-boundary capture required")
        result["local_capture_calls"] += len(seen)
        model = seen[0]
        replay, _, checks = _model_projection(rendered=prompt, receipt=receipt, repository=repository)
        if replay != model or not all(checks.values()):
            raise ValueError("captured full input did not replay exactly")
        binding = receipt.get("source_input_view")
        if policy != "legacy" and binding is None:
            raise ValueError("source-input treatment receipt is missing")
        if binding is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.source_input_view import validate_source_input_view_receipt
            validate_source_input_view_receipt(binding)
            if (binding["views_included"] < 1
                    or "compact" in policy and binding["compact_views_included"] < 1):
                raise ValueError("source-input treatment has no actual requested source views")
        arm = dict(policy=policy, disposition="captured", full_input_sha256=_sha(model),
            full_input_bytes=len(model.encode()), full_input_proxy_tokens=counter(model) if counter else None,
            exact_input_replay=True, positive_input_observed=policy != "legacy",
            source_input_receipt=binding, provider_usage_tokens=None)
        result["arms"].append(arm)
        captured.append((model, receipt, arm))
    # The historical audit must not depend on any mutable allocated source.
    source = next(path for path in sources if path.endswith(".py"))
    with (workspace / source).open("ab") as stream:
        stream.write(b"\n# Authored stale-copy probe; benchmark source is never executed.\n")
    stale_rejected = None
    if any(arm["policy"] == POLICIES[1] and arm["disposition"] == "captured" for arm in result["arms"]):
        with _captured_provider(workspace) as seen, redirect_stdout(io.StringIO()):
            from ipfs_accelerate_py.agent_supervisor.runtime.source_input_view import SourceInputViewError
            stale_rejected = False
            try:
                runner.run(prompt=prompt, provider="codex_cli", model="local-capture", timeout=1,
                    max_output_tokens=128, semantic_repository=repository, source_input_policy=POLICIES[1])
            except SourceInputViewError as error:
                if error.code != "stale_workspace":
                    raise
                stale_rejected = not seen
            result["local_capture_calls"] += len(seen)
    _git(repository, "worktree", "remove", "--force", str(workspace))
    for model, receipt, arm in captured:
        replay, _, checks = _model_projection(rendered=prompt, receipt=receipt, repository=repository)
        arm["historical_replay_after_workspace_removed"] = replay == model and all(checks.values())
    result.update(stale_allocated_source_refused=stale_rejected,
        allocated_worktree_removed=not workspace.exists(),
        canonical_public_inputs_unchanged=all((repository / path).read_bytes() == raw for path, raw in sources.items()),
        canonical_payload_unchanged=_sha(artifact.read_bytes()) == result["canonical_payload_sha256"],
        disposition="qualified" if len(captured) == len(POLICIES) else "partial_input_eligibility")
    selections = {arm["source_input_receipt"]["ordered_selection_sha256"] for _, _, arm in captured
                  if arm["source_input_receipt"] is not None}
    result["matched_source_selection"] = len(selections) == 1
    result["qualified"] = (len(captured) == len(POLICIES) and stale_rejected
        and result["optional_program_raw_omitted"] and result["required_instruction_exact"]
        and result["canonical_public_inputs_unchanged"] and result["canonical_payload_unchanged"] and result["matched_source_selection"]
        and all(arm["historical_replay_after_workspace_removed"] for _, _, arm in captured))
    by_policy = {arm["policy"]: arm for arm in result["arms"]}
    comparisons = ((POLICIES[0], POLICIES[1]), (POLICIES[1], POLICIES[2]),
                   (POLICIES[1], POLICIES[3]), (POLICIES[3], POLICIES[4]), (POLICIES[0], POLICIES[4]))
    result["comparisons"] = []
    for before, after in comparisons:
        a, b = by_policy[before], by_policy[after]
        result["comparisons"].append(dict(before=before, after=after,
            input_bytes_change=(b["full_input_bytes"] - a["full_input_bytes"])
                if a["full_input_bytes"] is not None and b["full_input_bytes"] is not None else None,
            input_proxy_tokens_change=(b["full_input_proxy_tokens"] - a["full_input_proxy_tokens"])
                if a["full_input_proxy_tokens"] is not None and b["full_input_proxy_tokens"] is not None else None))
    _save(output / "qualification.json", result)
    return result


def run_experiment(*, dataset, output, tasks=DEFAULT_TASKS, tokenizer_cache=None):
    tasks = tuple(tasks)
    if not tasks or len(tasks) > 8 or len(set(tasks)) != len(tasks):
        raise ValueError("bounded unique task selection required")
    dataset = Path(dataset).resolve(strict=True)
    loaded, total = [], 0
    for task in tasks:
        selected = catalog.reviewed_profile(dataset=dataset, task_name=task)
        sources = {}
        for row in selected["bindings"]["inputs"]:
            _path(row["path"])
            if row["path"] in sources:
                raise ValueError("duplicate reviewed input")
            sources[row["path"]] = catalog._read(dataset / task / "environment" / row["environment_path"], row["sha256"])
        from .terminal_task_profile import normalized_instruction
        instruction = normalized_instruction(catalog._read(dataset / task / "instruction.md",
            selected["bindings"]["public_metadata"]["instruction.md"]).decode())
        total += sum(map(len, sources.values())) + len(instruction.encode())
        if total > MAX_TOTAL_BYTES:
            raise ValueError("aggregate public input bytes exceed experiment bound")
        loaded.append((task, selected, sources, instruction))
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output or output == dataset or dataset in output.parents:
        raise ValueError("experiment output must be new and outside the dataset")
    output.mkdir(parents=True)
    counter, tokenizer = local_token_counter(tokenizer_cache)
    rows = [qualify_task(name=task, sources=sources, instruction=instruction, output=output / task,
        counter=counter, public_profile=selected["profile"], provenance=dict(dataset_revision_reviewed=selected["dataset_revision_reviewed"],
            source_boundary=selected["source_boundary"], hidden_verifier_or_solution_read=False))
        for task, selected, sources, instruction in loaded]
    unchanged = all(catalog._read(dataset / task / "environment" / row["environment_path"], row["sha256"])
        == sources[row["path"]] for task, selected, sources, _ in loaded for row in selected["bindings"]["inputs"])
    result = dict(schema=SCHEMA, qualified=unchanged and all(row["qualified"] for row in rows),
        tasks=rows, local_tokenizer=tokenizer, original_dataset_unchanged=unchanged,
        provider_calls=0, local_capture_calls=sum(row["local_capture_calls"] for row in rows),
        containers_started=0, provider_usage_tokens=None, official_reward=None,
        limitations=["One fixed indexed selection per task; no task solving or live model behavior.",
            "Full captured input proxy counts exclude later tool calls, expansions, reasoning, repairs, and provider framing.",
            "Representation changes do not prove behavioral equivalence or confer completion authority.",
            "Selected class/module read views include nested source bodies that their normalized facet projections elide; both rendering arms include identical source scope.",
            "Copied public inputs and canonical support use production indexed-default settings; full signed supervisor planning and container setup are excluded."])
    _save(output / "qualification.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(dataset=args.dataset, output=args.output, tasks=args.task or DEFAULT_TASKS,
                            tokenizer_cache=args.tokenizer_cache)
    print(_json(dict(schema=SCHEMA, qualified=result["qualified"], provider_calls=0,
                     tasks=[row["task_name"] for row in result["tasks"]], output=str(args.output))))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

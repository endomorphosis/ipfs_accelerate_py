"""Model-free, complete-input query-reference comparison on reviewed inputs.

Both arms use one immutable indexed selection and the actual coding router.
Only the provider boundary is replaced with a local capture; no benchmark
program executes, and local tokenizer measurements are not provider usage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

from . import terminal_profile_catalog as catalog
from .terminal_source_input_experiment import (
    _captured_provider, _git, _json, _native_prompt, _path, _save, _sha,
)
from .terminal_source_view_experiment import local_token_counter

SCHEMA = "terminal-query-reference-experiment@1"
POLICIES = ("legacy", "required-instruction-ref@1")
DEFAULT_TASKS = ("tune-mjcf", "llm-inference-batching-scheduler")
MAX_TOTAL_BYTES = 4 * 1024 * 1024
INELIGIBLE_CODES = {"no_exact_required_source", "ambiguous_required_source", "unprofitable_query_reference"}


@contextmanager
def _captured_completion(workspace):
    """Reuse the isolated local boundary, with the production completion shape."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    with _captured_provider(workspace) as seen:
        original = llm_router.generate_text

        def capture(prompt, **kwargs):
            original(prompt, **kwargs)
            schema = kwargs.get("codex_output_schema")
            if type(schema) is not dict:
                raise ValueError("qualification requires the ordinary completion schema")
            raw = _json(schema).encode()
            set_last_cli_observation("codex_cli", {"exit_code": 0,
                "codex_output_schema_sha256": _sha(raw), "codex_output_schema_bytes": len(raw)})
            return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'

        with patch.object(llm_router, "generate_text", capture):
            yield seen


def _receipt_projection(receipt):
    """Retain audit inputs only; no native observation, prompt, or response body."""
    from .terminal_context_audit import RECEIPT_FIELDS, TRANSLATION_FIELDS
    return {key: receipt[key] for key in (*RECEIPT_FIELDS, *TRANSLATION_FIELDS, "provider") if key in receipt}


def _measure(text, counter):
    count = counter(text) if counter else None
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("local tokenizer proxy must return a nonnegative integer")
    return {"full_input_sha256": _sha(text), "full_input_bytes": len(text.encode()),
            "full_input_proxy_tokens": count}


def _router_text(model, receipt):
    raw = model.encode()
    start = receipt["workspace_advisory_bytes"]
    router = raw[start:start + receipt["router_prompt_bytes"]]
    if _sha(router) != receipt["router_prompt_sha256"]:
        raise ValueError("captured router boundaries differ from the complete input")
    return router.decode()


def qualify_task(*, name, sources, instruction, output, counter=None, provenance=None,
                 public_profile=None, worker_query=None):
    """Compare two presentations without changing canonical worker evidence.

    ``worker_query`` is an authored-fixture control for ineligible cases. Public
    profile runs always use the exact instruction as the production query.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import coding_completion_schema
    from .terminal_context_audit import _receipt, _model_projection
    from .terminal_semantic_context_policy import semantic_context_objective, semantic_context_options
    from .terminal_task_profile import (INSTRUCTION, SMOKE, PROFILE, task_profile_bytes,
        task_profile_smoke, task_profile_index_paths, validate_task_profile)

    if (type(name) is not str or not name or len(name) > 128
            or type(sources) is not dict or not sources or len(sources) > 64
            or any(type(raw) is not bytes for raw in sources.values())
            or sum(map(len, sources.values())) > MAX_TOTAL_BYTES
            or type(instruction) is not str or not instruction.strip() or len(instruction.encode()) > 8192):
        raise ValueError("bounded explicit public inputs and instruction required")
    for path in sources:
        _path(path)
    query = instruction if worker_query is None else worker_query
    if (type(query) is not str or not query.strip() or len(query.encode()) > 8192
            or public_profile is not None and query != instruction):
        raise ValueError("public qualification requires the exact production query")
    if public_profile is not None:
        public_profile = validate_task_profile(public_profile, instruction=instruction)
        if set(public_profile["input_paths"]) != set(sources):
            raise ValueError("reviewed profile input population differs")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification requires a new canonical output directory")
    output.mkdir(parents=True)
    result = dict(schema=SCHEMA, task_name=name, provenance=provenance or {},
        provider_calls=0, local_capture_calls=0, containers_started=0, benchmark_source_executed=False,
        task_solved=None, official_reward=None, provider_usage_tokens=None, arms=[],
        input_sha256={path: _sha(raw) for path, raw in sorted(sources.items())},
        public_instruction_sha256=_sha(instruction), worker_query_sha256=_sha(query),
        coding_reply_mode="ordinary-completion@1", source_input_policy="legacy",
        semantic_metadata_view="legacy", source_view_advisory_policy="legacy", tool_output_policy="legacy",
        measurement_scope="complete captured coding input including ordinary completion and workspace wrappers; no provider framing, reasoning or tool exchanges")
    if not any(path.endswith(".py") for path in sources):
        result.update(disposition="no_supported_python_input", qualified=False, observations_valid=True)
        _save(output / "qualification.json", result)
        return result
    repository = output / "repository"
    repository.mkdir()
    for relative, raw in sources.items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    support = {INSTRUCTION: instruction.encode()}
    required, program_paths, defer_data = [INSTRUCTION], sorted(sources), False
    if public_profile is not None:
        support.update({PROFILE: task_profile_bytes(public_profile), SMOKE: task_profile_smoke(public_profile).encode()})
        required.append(SMOKE)
        program_paths = task_profile_index_paths(public_profile)
        defer_data = bool(public_profile.get("data_inputs"))
    for relative, raw in support.items():
        (repository / relative).write_bytes(raw)
    all_paths = sorted([*sources, *support])
    _git(repository, "init", "-q")
    _git(repository, "add", "--all")
    _git(repository, "-c", "user.name=Query reference qualification", "-c",
         "user.email=qualification@example.invalid", "commit", "-qm", "Reviewed copied inputs")
    workspace = output / "allocated-worktree"
    _git(repository, "worktree", "add", "--detach", "-q", str(workspace))
    artifact = repository / ".runtime/semantic/worker-context.json"
    prepared = prepare_semantic_context(repository=repository, paths=all_paths,
        required_raw_paths=required, program_paths=program_paths, defer_task_data=defer_data,
        objective=semantic_context_objective("indexed-default@1", instruction),
        task_id="QUERY-REFERENCE-QUALIFICATION", output=artifact.parent, worker_query=query,
        max_symbols=1024 if public_profile is not None else 256,
        **semantic_context_options("indexed-default@1"))
    payload = json.loads(artifact.read_text())
    from .terminal_source_payload_census import census_worker_payload
    selected_hash = _sha(_json(payload["worker_projection"]["selected_symbols"]))
    result.update(selected_capsules=prepared["worker_capsules"], full_capsules=prepared["capsules"],
        selected_symbols_sha256=selected_hash, canonical_payload_sha256=_sha(artifact.read_bytes()),
        optional_program_raw_omitted=not bool(set(program_paths) & set(payload["raw_sources"])),
        required_instruction_exact=payload["raw_sources"].get(INSTRUCTION) == instruction,
        support_input_sha256={path: _sha(raw) for path, raw in sorted(support.items())},
        required_raw_paths=required, program_paths=program_paths,
        worker_payload_census=census_worker_payload(payload),
        preparation_scope=("reviewed COPY inputs and canonical profile/smoke/instruction; production indexed-default query/selection/bounds; no signed supervisor planning/admission or container setup"
            if public_profile is not None else "authored inputs and required literal instruction; no supervisor planning/admission or container setup"))
    prompt = _native_prompt(repository, artifact, "QUERY-REFERENCE-QUALIFICATION", all_paths)
    # This is the deterministic native producer object, never a model log or
    # captured provider prompt. It makes later replay independent of defaults.
    native_context = json.loads(prompt)
    if _json(native_context) != prompt:
        raise ValueError("native context renderer must produce canonical JSON")
    _save(output / "native-context-input.json", native_context)
    result["native_context_input_sha256"] = _sha((output / "native-context-input.json").read_bytes())
    result["native_prompt_sha256"] = _sha(prompt)
    native_output_schema = _json(coding_completion_schema())
    captured, legacy_router = [], None
    try:
        for policy in POLICIES:
            with _captured_completion(workspace) as seen, redirect_stdout(io.StringIO()):
                try:
                    _, receipt = runner.run(prompt=prompt, provider="codex_cli", model="local-capture", timeout=1,
                        max_output_tokens=128, semantic_repository=repository, query_reference_policy=policy,
                        coding_reply_mode="ordinary-completion@1")
                except ValueError as error:
                    if seen or policy == "legacy" or getattr(error, "code", None) not in INELIGIBLE_CODES:
                        raise
                    result["arms"].append(dict(policy=policy, disposition="ineligible_before_provider_boundary",
                        reason_code=error.code, positive_reference_observed=False, reference_count=0,
                        full_input_bytes=None, full_input_proxy_tokens=None, provider_usage_tokens=None))
                    continue
            if len(seen) != 1:
                raise ValueError("exactly one local provider-boundary capture required")
            result["local_capture_calls"] += len(seen)
            model = seen[0]
            receipt = {**receipt, "phase": "coding"}
            parsed = _receipt(receipt, workspace.parent)
            replay, _, checks = _model_projection(rendered=prompt, receipt=parsed, repository=repository)
            if replay != model or not all(value is True for value in checks.values()):
                raise ValueError("captured complete input did not replay exactly")
            binding = receipt.get("query_reference_view")
            selected = receipt.get("query_reference_selected") is True
            dispatched = receipt.get("query_reference_dispatched") is True
            positive = policy != "legacy" and selected and dispatched and type(binding) is dict and binding.get("reference_count") == 1
            if policy != "legacy" and not positive:
                raise ValueError("requested reference did not cross the captured coding boundary")
            router_text = _router_text(model, receipt)
            wire = json.loads(router_text)
            restored_exact = True
            if policy == "legacy":
                legacy_router = router_text
                transport = wire
            else:
                from ipfs_accelerate_py.agent_supervisor.runtime.semantic_query_reference import restore_semantic_query_reference
                restored_exact = restore_semantic_query_reference(provider_prompt=router_text, receipt=binding) == legacy_router
                transport = wire["transport"]
                if binding["source_path"] != INSTRUCTION:
                    raise ValueError("reference does not select the declared required instruction")
            literal_exact = transport["translated_semantic"]["raw_sources"].get(INSTRUCTION) == instruction
            selection_wire_hash = _sha(_json(transport["translated_semantic"]["worker_projection"]["selected_symbols"]))
            if not literal_exact or not restored_exact:
                raise ValueError("reference changed the required literal or original semantic transport")
            reply = receipt["coding_reply_contract"]
            if (reply["native_output_schema_sha256"] != _sha(native_output_schema)
                    or reply["native_output_schema_bytes"] != len(native_output_schema.encode())):
                raise ValueError("native completion schema differs between coding arms")
            audit_receipt = _receipt_projection(receipt)
            _save(output / ("legacy-receipt.json" if policy == "legacy" else "reference-receipt.json"), audit_receipt)
            arm = dict(policy=policy, disposition="captured", **_measure(model, counter),
                exact_input_replay=True, audit_checks=checks, positive_reference_observed=positive,
                reference_count=binding["reference_count"] if binding else 0,
                query_reference_receipt=binding, provider_usage_tokens=None,
                required_instruction_inline_exact=literal_exact, exact_legacy_transport_restored=restored_exact,
                source_selection_wire_sha256=selection_wire_hash,
                canonical_payload_sha256=_sha(artifact.read_bytes()), selected_symbols_sha256=selected_hash,
                receipt_sha256=_sha(_json(audit_receipt) + "\n"))
            arm.update(native_output_schema_sha256=reply["native_output_schema_sha256"],
                native_output_schema_bytes=reply["native_output_schema_bytes"],
                native_output_schema_proxy_tokens=_measure(native_output_schema, counter)["full_input_proxy_tokens"],
                model_prompt_and_native_output_schema_bytes=arm["full_input_bytes"] + reply["native_output_schema_bytes"])
            result["arms"].append(arm)
            captured.append((model, parsed, arm))
        stale_rejected = None
        if any(arm["positive_reference_observed"] for _, _, arm in captured):
            (workspace / INSTRUCTION).write_text(instruction + "\nAuthored freshness probe.\n")
            with _captured_completion(workspace) as seen, redirect_stdout(io.StringIO()):
                from ipfs_accelerate_py.agent_supervisor.runtime.source_input_view import SourceInputViewError
                stale_rejected = False
                try:
                    runner.run(prompt=prompt, provider="codex_cli", model="local-capture", timeout=1,
                        max_output_tokens=128, semantic_repository=repository, query_reference_policy=POLICIES[1],
                        coding_reply_mode="ordinary-completion@1")
                except SourceInputViewError as error:
                    if error.code != "stale_workspace":
                        raise
                    stale_rejected = not seen
                result["local_capture_calls"] += len(seen)
    finally:
        _git(repository, "worktree", "remove", "--force", str(workspace))
    retained_prompt = _json(json.loads((output / "native-context-input.json").read_text()))
    if retained_prompt != prompt:
        raise ValueError("retained native input changed before historical replay")
    for model, receipt, arm in captured:
        replay, _, checks = _model_projection(rendered=retained_prompt, receipt=receipt, repository=repository)
        arm["historical_replay_after_workspace_removed"] = replay == model and all(value is True for value in checks.values())
    result.update(allocated_worktree_removed=not workspace.exists(), stale_required_instruction_refused=stale_rejected,
        canonical_public_inputs_unchanged=all((repository / path).read_bytes() == raw for path, raw in sources.items()),
        canonical_support_inputs_unchanged=all((repository / path).read_bytes() == raw for path, raw in support.items()),
        canonical_payload_unchanged=_sha(artifact.read_bytes()) == result["canonical_payload_sha256"],
        matched_source_selection=len({arm["source_selection_wire_sha256"] for _, _, arm in captured}) == 1)
    result["observations_valid"] = (len(result["arms"]) == len(POLICIES) and bool(captured)
        and result["required_instruction_exact"] and result["canonical_public_inputs_unchanged"]
        and result["canonical_support_inputs_unchanged"] and result["canonical_payload_unchanged"]
        and result["matched_source_selection"] and all(arm["historical_replay_after_workspace_removed"] for _, _, arm in captured))
    result["qualified"] = result["observations_valid"] and len(captured) == len(POLICIES) and stale_rejected is True
    before, after = result["arms"]
    available = before["full_input_bytes"] is not None and after["full_input_bytes"] is not None
    result["comparison"] = dict(before=POLICIES[0], after=POLICIES[1],
        input_bytes_change=after["full_input_bytes"] - before["full_input_bytes"] if available else None,
        input_proxy_tokens_change=(after["full_input_proxy_tokens"] - before["full_input_proxy_tokens"])
            if available and before["full_input_proxy_tokens"] is not None and after["full_input_proxy_tokens"] is not None else None)
    result["complete_input_bytes_reduced"] = available and result["comparison"]["input_bytes_change"] < 0
    result["disposition"] = "qualified" if result["qualified"] else "retained_ineligible_treatment"
    import inspect
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_query_reference
    from . import terminal_context_audit
    result["implementation_sha256"] = {
        "experiment": _sha(Path(__file__).read_bytes()),
        "query_reference_codec": _sha(Path(inspect.getfile(semantic_query_reference)).read_bytes()),
        "router": _sha(Path(inspect.getfile(runner)).read_bytes()),
        "input_audit": _sha(Path(inspect.getfile(terminal_context_audit)).read_bytes()),
    }
    _save(output / "qualification.json", result)
    return result


def run_experiment(*, dataset, output, tasks=DEFAULT_TASKS, tokenizer_cache=None):
    tasks = tuple(tasks)
    if not tasks or len(tasks) > 8 or len(set(tasks)) != len(tasks):
        raise ValueError("bounded unique task selection required")
    dataset = Path(dataset).resolve(strict=True)
    loaded, total = [], 0
    from .terminal_task_profile import normalized_instruction
    for task in tasks:
        selected = catalog.reviewed_profile(dataset=dataset, task_name=task)
        sources = {}
        for row in selected["bindings"]["inputs"]:
            _path(row["path"])
            if row["path"] in sources:
                raise ValueError("duplicate reviewed input")
            sources[row["path"]] = catalog._read(dataset / task / "environment" / row["environment_path"], row["sha256"])
        instruction = normalized_instruction(catalog._read(dataset / task / "instruction.md",
            selected["bindings"]["public_metadata"]["instruction.md"]).decode())
        total += sum(map(len, sources.values())) + len(instruction.encode())
        if total > MAX_TOTAL_BYTES:
            raise ValueError("aggregate public inputs exceed experiment bound")
        loaded.append((task, selected, sources, instruction))
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output or output == dataset or dataset in output.parents:
        raise ValueError("experiment output must be new and outside the dataset")
    output.mkdir(parents=True)
    counter, tokenizer = local_token_counter(tokenizer_cache)
    rows = [qualify_task(name=task, sources=sources, instruction=instruction, output=output / task,
        counter=counter, public_profile=selected["profile"], provenance=dict(
            dataset_revision_reviewed=selected["dataset_revision_reviewed"], source_boundary=selected["source_boundary"],
            hidden_verifier_or_solution_read=False)) for task, selected, sources, instruction in loaded]
    unchanged = all(catalog._read(dataset / task / "environment" / row["environment_path"], row["sha256"])
        == sources[row["path"]] for task, selected, sources, _ in loaded for row in selected["bindings"]["inputs"])
    unchanged = unchanged and all(catalog.reviewed_profile(dataset=dataset, task_name=task)["bindings"]
        == selected["bindings"] for task, selected, _, _ in loaded)
    result = dict(schema=SCHEMA, qualified=unchanged and all(row["qualified"] for row in rows),
        observations_valid=unchanged and all(row["observations_valid"] for row in rows),
        tasks=rows, local_tokenizer=tokenizer, original_dataset_unchanged=unchanged,
        provider_calls=0, local_capture_calls=sum(row["local_capture_calls"] for row in rows), containers_started=0,
        provider_usage_tokens=None, official_reward=None,
        limitations=["One fixed production indexed selection per task; no task solving or live model behavior.",
            "Complete input proxy counts exclude later tool calls, reasoning, repairs and provider framing.",
            "The identical native output-schema argument is reported separately from model_prompt; combined bytes are declared text, not serialized provider request size or tokens.",
            "Literal instruction stays inline; the reference only presents an exact repeated ranking query.",
            "Canonical evidence and selection stay fixed; lossless presentation grants no semantic, proof or completion authority.",
            "Public source preparation uses production support/query/limits; signed supervisor planning, Doctor/public appendices and container setup are excluded."])
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
    print(_json(dict(schema=SCHEMA, qualified=result["qualified"], observations_valid=result["observations_valid"],
                     provider_calls=0, tasks=[row["task_name"] for row in result["tasks"]], output=str(args.output))))
    return 0 if result["observations_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect bounded self-contained Grok read sets, fresh unless reuse is explicit.

This command contacts the selected provider. Each actual dispatch repeats the
complete literal requirements; responses are retained privately and never become hidden
history for the next call. Local rendered-input allowances and an optional
observed-native stop do not enforce a native request, billing or task token cap.
Explicit result reuse is limited to this series and an unchanged complete input;
it reuses a read-only answer and establishes no proof or task acceptance.
Only admitted source and declared input bindings are checked; external state or
nondeterministic answers can differ, and requested tool restrictions are not containment.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile

from . import repository_read_set as preparation
from .repository_read_set_inspect import MAX_RESPONSE_BYTES, _publish

SCHEMA = "repository-read-set-series-cli@1"
MAX_QUERIES = 8
MAX_PROMPT_BYTES = 256000


def run_repository_read_set_series(*, root, queries, required_paths, requirements_file,
        task_id, provider, model, max_total_input_tokens, max_input_tokens, output,
        limit=8, timeout=1800, max_output_tokens=2048,
        max_observed_native_tokens_before_next_dispatch=None,
        reuse_identical_inputs=False,
        scheduler_state=None, scheduler_policy_sha256=None, failure_publication_observer=None):
    """Hold one real source session; preserve paid-attempt metadata on failure."""
    root, output, requirements_file = map(preparation._absolute, (root, output, requirements_file))
    if Path.cwd() != root:
        raise ValueError("series must start in its explicit source root")
    if (provider, model) != ("grok_cli", "grok-4.7"):
        raise ValueError("explicit supported Grok inspection route required")
    if type(reuse_identical_inputs) is not bool:
        raise ValueError("explicit boolean result reuse required")
    if type(task_id) is not str or preparation._ID.fullmatch(task_id) is None:
        raise ValueError("explicit task identity required")
    if (type(queries) not in {list, tuple} or not 1 <= len(queries) <= MAX_QUERIES
            or any(type(query) is not str or not query.strip() or "\x00" in query
                or len(query.encode("utf-8")) > 4096 for query in queries)):
        raise ValueError("bounded explicit query series required")
    queries = tuple(queries)
    if type(required_paths) not in {list, tuple} or len(required_paths) > preparation.MAX_REQUIRED_PATHS:
        raise ValueError("bounded required paths required")
    required_paths = tuple(preparation._relative(path) for path in required_paths)
    if len(set(required_paths)) != len(required_paths):
        raise ValueError("unique required paths required")
    for value, maximum in ((limit, 32), (timeout, 1800), (max_output_tokens, 16384),
            (max_input_tokens, preparation.MAX_COUNTER), (max_total_input_tokens, preparation.MAX_COUNTER)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("bounded series settings required")
    if (max_observed_native_tokens_before_next_dispatch is not None
            and (type(max_observed_native_tokens_before_next_dispatch) is not int
                or not 1 <= max_observed_native_tokens_before_next_dispatch <= preparation.MAX_COUNTER)):
        raise ValueError("bounded observed-native stop required")
    if failure_publication_observer is not None and not callable(failure_publication_observer):
        raise ValueError("explicit failure publication observer required")
    scheduler_path = preparation._scheduler_input(scheduler_state, scheduler_policy_sha256)
    descriptor = preparation._directory(root)
    os.close(descriptor)
    parent = preparation._output_parent(output, root)
    os.close(parent)
    required_raw, required_text = preparation._capture_requirements(requirements_file)
    temporary_root = Path("/tmp")
    descriptor = preparation._directory(temporary_root)
    os.close(descriptor)
    if (root == temporary_root or root in temporary_root.parents
            or any(os.path.lexists(path / ".git") for path in (temporary_root, *temporary_root.parents))):
        raise ValueError("private native state must remain outside Git")
    from ipfs_accelerate_py.agent_supervisor.runtime.automatic_prompt_tokenizer import resolve_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_read_set_series import inspect_repository_read_set_series
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    index_factory = None
    if scheduler_path is not None:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
        owner = open_existing_scheduler(scheduler_path, scheduler_policy_sha256)
        index_factory = lambda path: NativeRepositoryIndex(path, scheduler=owner)
    counter, provenance = resolve_prompt_tokenizer(provider=provider, model=model)
    last_receipt, observation_status, phase = None, "not_observed", "native_session_initialization"

    def requirements_guard():
        if preparation._capture_requirements(requirements_file)[0] != required_raw:
            raise ValueError("literal series requirements changed")

    def observe(value):
        nonlocal last_receipt, observation_status
        try:
            if type(value) is not dict:
                raise ValueError("closed series receipt required")
            raw = preparation._json(value)
            if len(raw) > preparation.MAX_RECEIPT_BYTES:
                raise ValueError("series receipt exceeds bound")
            last_receipt = json.loads(raw)
            observation_status = "retained"
        except Exception:
            observation_status = "invalid_observation"
            raise

    common = dict(schema=SCHEMA, task_id=task_id, provider=provider, model=model,
        query_bindings=[dict(ordinal=index + 1, sha256=preparation._sha(query.encode("utf-8")),
            bytes=len(query.encode("utf-8"))) for index, query in enumerate(queries)],
        tokenizer_provenance=provenance, requirements_sha256=preparation._sha(required_raw),
        requirements_bytes=len(required_raw), live_provider_operation_requested=True,
        resource_scheduler_binding=dict(mode="existing_pinned" if scheduler_path is not None else "native_default",
            state_path_sha256=None if scheduler_path is None else preparation._sha(str(scheduler_path).encode()),
            policy_sha256=scheduler_policy_sha256, policy_reconfiguration_requested=False,
            state_creation_requested=False if scheduler_path is not None else None),
        source_guard_scope="cooperative snapshot checks; no writer lease",
        reuse_identical_inputs_requested=reuse_identical_inputs,
        independent_fresh_contexts=True, prior_responses_in_next_prompt=False,
        native_history_complete=None, whole_session_token_savings=None, task_acceptance=False,
        billing_total_verified=False, native_cumulative_cap_enforced=False,
        source_bodies_in_receipt=False, proof_authority=False, execution_authority=False,
        completion_authority=False)
    try:
        with tempfile.TemporaryDirectory(prefix="repository-read-set-series-", dir=temporary_root) as temporary:
            with RepositorySourceSession(root, Path(temporary) / "state", representation="original",
                    index_factory=index_factory) as session:
                before = session.status()
                phase = "guarded_series"
                result = inspect_repository_read_set_series(session=session, generation=before["generation"],
                    queries=queries, required_paths=required_paths, required_texts=(required_text,),
                    task_id=task_id, provider=provider, model=model, workspace=root, token_counter=counter,
                    max_total_input_tokens=max_total_input_tokens, max_input_tokens=max_input_tokens,
                    limit=limit, timeout=timeout, max_output_tokens=max_output_tokens,
                    max_observed_native_tokens_before_next_dispatch=max_observed_native_tokens_before_next_dispatch,
                    reuse_identical_inputs=reuse_identical_inputs,
                    requirements_guard=requirements_guard, receipt_observer=observe)
                if observation_status != "retained" or last_receipt != result.receipt:
                    raise ValueError("final series observation unavailable")
                phase = "closing_requirements_check"
                requirements_guard()
                phase = "closing_source_check"
                after = session.status()
                if (after.get("source_current") is not True or after.get("refresh_required") is not False
                        or any(before.get(key) != after.get(key) for key in ("session_id", "generation",
                            "source_tree_sha256", "package_manifest_sha256", "datasets_head", "semantic_state_cid"))):
                    raise ValueError("series source generation changed")
                phase = "bounded_output_capture"
                prompts, responses = result.selected_prompts, result.response_texts
                if len(prompts) != len(responses) or not 0 <= len(prompts) <= len(queries):
                    raise ValueError("bounded completed series required")
                files, bindings = {}, []
                for ordinal, (prompt, response) in enumerate(zip(prompts, responses), 1):
                    if type(prompt) is not str or type(response) is not str:
                        raise ValueError("bounded series text required")
                    prompt, response = prompt.encode("utf-8"), response.encode("utf-8")
                    if not 0 < len(prompt) <= MAX_PROMPT_BYTES or not 0 < len(response) <= MAX_RESPONSE_BYTES:
                        raise ValueError("bounded series text required")
                    prompt_name, response_name = f"prompt-{ordinal:02d}.txt", f"response-{ordinal:02d}.txt"
                    files.update({prompt_name: prompt, response_name: response})
                    bindings.append(dict(ordinal=ordinal, prompt_file=prompt_name, response_file=response_name,
                        prompt_sha256=preparation._sha(prompt), prompt_bytes=len(prompt),
                        response_sha256=preparation._sha(response), response_bytes=len(response)))
                completed_calls = sum(row["status"] == "returned" and row["backend_dispatches"] == 1
                    for row in last_receipt["attempts"])
                reused_results = sum(row["status"] == "reused" for row in last_receipt["attempts"])
                if completed_calls + reused_results != len(bindings):
                    raise ValueError("completed series result bindings differ")
                receipt = dict(common, status=last_receipt["status"], series=last_receipt,
                    series_observation_status=observation_status, source_current_at_close=True,
                    source_generation=after["generation"], source_tree_sha256=after["source_tree_sha256"],
                    completed_calls=completed_calls, completed_results=len(bindings),
                    reused_results=reused_results, independent_fresh_contexts=reused_results == 0,
                    output_bindings=bindings)
        encoded = preparation._json(receipt)
        if len(encoded) > preparation.MAX_RECEIPT_BYTES:
            raise ValueError("bounded series receipt required")
        phase = "output_publication"
        files["receipt.json"] = encoded
        _publish(output, root, files)
    except BaseException:
        publication = dict(schema=SCHEMA, status="failed", reason="series_failed", failure_phase=phase,
            receipt_retained=False, receipt_sha256=None, receipt_bytes=None)
        try:
            failed = dict(common, status="failed", failure_reason="series_failed", failure_phase=phase,
                series=last_receipt, series_observation_status=observation_status,
                source_current_at_close=None, output_bindings=[])
            if last_receipt is not None:
                failed["independent_fresh_contexts"] = not any(
                    row.get("status") == "reused" for row in last_receipt.get("attempts", [])
                    if type(row) is dict)
            encoded = preparation._json(failed)
            if len(encoded) <= preparation.MAX_RECEIPT_BYTES:
                _publish(output, root, {"receipt.json": encoded})
                publication.update(receipt_retained=True, receipt_sha256=preparation._sha(encoded), receipt_bytes=len(encoded))
        except BaseException:
            pass  # Preserve the original provider/cancellation/closing failure.
        if failure_publication_observer is not None:
            try:
                failure_publication_observer(publication)
            except BaseException:
                pass
        raise
    return dict(schema=SCHEMA, status="written", series_status=last_receipt["status"],
        completed_calls=completed_calls, completed_results=len(bindings), reused_results=reused_results,
        receipt_sha256=preparation._sha(encoded), receipt_bytes=len(encoded),
        live_provider_operation_requested=True, task_acceptance=False, raw_content_exported_to_stdout=False)


def main(argv=None):
    parser = preparation._Arguments(description=__doc__)
    for flag in ("root", "requirements-file", "task-id", "provider", "model", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--required-path", action="append", default=[])
    for flag in ("max-input-tokens", "max-total-input-tokens"):
        parser.add_argument("--" + flag, type=int, required=True)
    parser.add_argument("--max-observed-native-tokens-before-next-dispatch", type=int)
    parser.add_argument("--reuse-identical-inputs", action="store_true",
        help="Reuse an earlier read-only answer within this series only for an exact complete input and unchanged admitted source; external state is not checked, and no proof or task acceptance follows.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--scheduler-state")
    parser.add_argument("--scheduler-policy-sha256")
    publication = None
    def remember(value):
        nonlocal publication
        publication = value
    try:
        arguments = vars(parser.parse_args(argv))
        arguments["required_paths"] = arguments.pop("required_path")
        arguments["queries"] = arguments.pop("query")
        result = run_repository_read_set_series(**arguments, failure_publication_observer=remember)
    except Exception:
        result = publication or dict(schema=SCHEMA, status="refused", reason="invalid_input_or_local_failure",
            receipt_retained=False)
        print(preparation._json(result).decode(), end="")
        return 2
    print(preparation._json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

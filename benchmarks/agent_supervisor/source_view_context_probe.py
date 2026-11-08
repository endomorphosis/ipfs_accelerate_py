"""Offline real-source rendering census over explicitly authored input histories.

This measures the generic source-view API inside complete authored user inputs,
not an MCP delivery, provider transcript, source inverse or accepted task.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import (
    MAX_SOURCE_BYTES, SourceViewError, render_source_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime.context_trajectory_comparison import (
    compare_context_trajectories,
)
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES, PromptTokenCounter, render_model_prompt,
)

SCHEMA = "source-view-context-probe@1"
MAX_FILES = 8
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_BINDING = {"task_id", "provider", "model", "tree_id", "acceptance_binding"}
GUIDANCE = (
    "Source responses below are an offline authored history. Canonical original source "
    "remains authoritative. A compact AST view may omit comments and formatting; "
    "it grants no proof, edit, execution or completion authority. Request exact current "
    "source when omitted context matters. All earlier visible responses are replayed.\n"
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return sha256(raw).hexdigest()


def _path(value):
    if (type(value) is not str or len(value) > 1024
            or re.fullmatch(r"[A-Za-z0-9_./-]+\.py", value) is None
            or any(not part or part.startswith(".") for part in value.split("/"))):
        raise ValueError("bounded explicit visible relative Python path required")
    return value


def _binding(value):
    if (type(value) is not dict or set(value) != _BINDING
            or any(type(item) is not str or re.fullmatch(r"[A-Za-z0-9_:/@.+-]{1,256}", item) is None
                   for item in value.values())):
        raise ValueError("closed explicit identity binding required")
    return dict(value)


def compare_source_view_context(*, source: bytes, path: str, binding: dict,
        common_prefix: str, common_suffix: str, workspace: Path,
        token_counter: PromptTokenCounter | None, rounds: int = 3) -> dict:
    """Compare repeated reads and an extra exact-source expansion, without IO.

    Each repeated-read callback appends one response and replays every predecessor.
    The expansion control has one raw read plus rounds-1 continuations; treatment
    has one compact read, an extra raw expansion callback, then those continuations.
    Both comparisons lose all counts if any shared counter observation fails.
    """
    path, bound = _path(path), _binding(binding)
    if type(source) is not bytes or len(source) > MAX_SOURCE_BYTES:
        raise ValueError("bounded captured source bytes required")
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("rounds must be an integer from one through eight")
    for text in (common_prefix, common_suffix):
        if type(text) is not str or len(text.encode("utf-8")) > 16384:
            raise ValueError("bounded complete common instructions required")
    if type(workspace) is not Path and not isinstance(workspace, Path):
        raise ValueError("explicit workspace Path required")
    if (not workspace.is_absolute() or ".." in workspace.parts
            or len(str(workspace).encode("utf-8")) > 4096):
        raise ValueError("bounded absolute workspace Path required")
    if token_counter is not None and (type(token_counter) is not PromptTokenCounter
            or token_counter.provider != bound["provider"] or token_counter.model != bound["model"]):
        raise ValueError("token counter differs from pinned provider/model")
    metadata = None if token_counter is None else (
        token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis)
    result = dict(schema=SCHEMA, measurement_status="unavailable", failure_reason=None,
        source_sha256=_sha(source), source_bytes=len(source), path_sha256=_sha(path.encode()),
        common_prefix_sha256=_sha(common_prefix.encode()), common_suffix_sha256=_sha(common_suffix.encode()),
        guidance_sha256=_sha(GUIDANCE.encode()), rounds=rounds, responses=None, comparisons=None,
        tokenizer_binding=None if metadata is None else metadata[0],
        token_count_kind=None if metadata is None else metadata[1],
        tokenizer_compatibility_basis=None if metadata is None else metadata[2],
        counter_evaluation_count=0, all_comparison_counts_available=False,
        count_scope="complete_authored_rendered_user_inputs_only",
        extra_expansion_callback_in_recipe=True, prior_visible_response_replay_in_recipe=True,
        generic_view_has_alias_dictionary=False, original_source_authoritative=True,
        original_source_obligations_discharged=False, actual_MCP_delivery=False,
        native_history=False, task_acceptance=False, provider_invoked=False,
        provider_usage_complete=None, whole_session_token_savings=None,
        token_budget_guaranteed=False, execution_authority=False, completion_authority=False)
    try:
        raw = render_source_bytes(path, source, format="raw")
        compact = render_source_bytes(path, source, format="compact")
    except (SourceViewError, UnicodeError, RecursionError, ValueError):
        result.update(measurement_status="unsupported", failure_reason="source_view_unavailable")
        return result
    responses = [_json(value).decode() + "\n" for value in (raw, compact)]
    result["responses"] = {name: dict(response_sha256=_sha(text.encode()),
        response_bytes=len(text.encode()), content_bytes=value["content_bytes"],
        actual_format=value["format"], fallback_to_raw=value["format"] != requested)
        for name, text, value, requested in zip(("raw", "compact"), responses,
            (raw, compact), ("raw", "compact"), strict=True)}

    def trajectory(histories, *, expansion=False):
        inputs = []
        for index, history in enumerate(histories):
            prompt = common_prefix + "\n" + GUIDANCE + "\n".join(history) + common_suffix
            rendered, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
            if len(rendered.encode()) > MAX_MODEL_PROMPT_BYTES:
                raise ValueError("complete_input_bound")
            inputs.append(dict(input_id=f"input-{index + 1}",
                kind="expansion" if expansion and index == 1 else "coding", model_prompt=rendered))
        return dict(binding=bound, inputs=inputs)

    try:
        repeated = [trajectory([[text] * count for count in range(1, rounds + 1)]) for text in responses]
        expansion = [trajectory([[responses[0]]] * rounds), trajectory(
            [[responses[1]]] + [[responses[1], responses[0]]] * rounds, expansion=True)]
        # Validate every trace before consulting a possibly expensive local counter.
        for pair in (repeated, expansion):
            compare_context_trajectories(*pair, expected_binding=bound, token_counter=None)
    except (ValueError, UnicodeError, RecursionError):
        result.update(measurement_status="unsupported", failure_reason="complete_input_bound")
        return result
    observations, failure = {}, None

    def count(text):
        nonlocal failure
        result["counter_evaluation_count"] += 1
        if (token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) != metadata:
            failure = "counter_changed"
            raise ValueError("counter observation refused")
        amount = token_counter.count(text)
        if ((token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) != metadata):
            failure = "counter_changed"
            raise ValueError("counter observation refused")
        if text in observations and observations[text] != amount:
            failure = "counter_not_repeatable"
            raise ValueError("counter observation refused")
        observations[text] = amount
        return amount

    wrapped = None if metadata is None else PromptTokenCounter(**metadata[0], count_tokens=count,
        count_kind=metadata[1], compatibility_basis=metadata[2])
    comparisons = {}
    for label, pair in (("repeated_reads", repeated), ("exact_source_expansion", expansion)):
        comparisons[label] = compare_context_trajectories(*pair, expected_binding=bound, token_counter=wrapped)
        if comparisons[label]["measurement_status"] == "failed":
            failure = failure or "counter_failed"
            break
    if failure is not None:
        comparisons = {label: compare_context_trajectories(*pair, expected_binding=bound, token_counter=None)
            for label, pair in (("repeated_reads", repeated), ("exact_source_expansion", expansion))}
        result.update(measurement_status="failed", failure_reason=failure)
    else:
        result["measurement_status"] = comparisons["repeated_reads"]["measurement_status"]
        result["all_comparison_counts_available"] = wrapped is not None
    result["comparisons"] = comparisons
    return result


def _directory(path):
    path = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _capture(root, relative):
    relative = _path(relative)
    directory = _directory(root)
    try:
        for part in relative.split("/")[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.split("/")[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_SOURCE_BYTES:
                raise ValueError("source capture refused")
            chunks, size = [], 0
            while size <= MAX_SOURCE_BYTES:
                chunk = os.read(descriptor, min(65536, MAX_SOURCE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            witness = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if size != before.st_size or size > MAX_SOURCE_BYTES or witness(before) != witness(after):
                raise ValueError("source capture refused")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _write_new(path, raw):
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("report exceeds bound")
    path = Path(os.path.abspath(path))
    directory = _directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if written < 1:
                    raise ValueError("report write refused")
                remaining = remaining[written:]
            os.fsync(descriptor)
            if os.fstat(descriptor).st_nlink != 1:
                raise ValueError("report publication refused")
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


class _Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("explicit bounded arguments required")


def main(argv=None):
    parser = _Arguments(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default="grok_cli")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    try:
        args = parser.parse_args(argv)
        if not 1 <= len(args.paths) <= MAX_FILES or len(set(args.paths)) != len(args.paths):
            raise ValueError("bounded unique population required")
        if not 1 <= args.rounds <= 8:
            raise ValueError("bounded rounds required")
        bound = _binding(dict(task_id="offline-source-probe", provider=args.provider, model=args.model,
            tree_id="captured-source-population", acceptance_binding="unverified-offline-rendering"))
        sources = [(path, _capture(args.root, path)) for path in args.paths]
        population_sha256 = _sha(_json([dict(path_sha256=_sha(path.encode()),
            source_sha256=_sha(source), bytes=len(source)) for path, source in sources]))
        bound["tree_id"] = "sha256:" + population_sha256
        from ipfs_accelerate_py.agent_supervisor.runtime.automatic_prompt_tokenizer import resolve_prompt_tokenizer
        counter, provenance = resolve_prompt_tokenizer(provider=args.provider, model=args.model)
        cases = [compare_source_view_context(source=source, path=path, binding=bound,
            common_prefix="Inspect this source without editing; preserve its documented requirements.",
            common_suffix="\nNo task acceptance or source-change result is requested by this offline probe.",
            workspace=Path("/offline/source-probe"), token_counter=counter, rounds=args.rounds)
            for path, source in sources]
        if any(_capture(args.root, path) != source for path, source in sources):
            raise ValueError("source population changed")
        report = dict(schema="source-view-context-probe-population@1", source_count=len(cases),
            population_captured_source_sha256=population_sha256,
            tokenizer_provenance=provenance, cases=cases, source_bodies_exported=False,
            provider_invoked=False, actual_MCP_delivery=False, native_history=False,
            task_acceptance=False, whole_session_token_savings=None, population_tokens=None)
        raw = _json(report) + b"\n"
        _write_new(args.output, raw)
        print(_json(dict(schema=SCHEMA, status="report_written", output_sha256=_sha(raw), output_bytes=len(raw))).decode())
        return 0
    except Exception:
        print(_json(dict(schema=SCHEMA, status="refused", reason="invalid_input_or_local_failure")).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

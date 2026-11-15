"""Offline measured representation comparison over actual indexed capsules.

Only the provider seam is scripted. Source scanning, indexed selection, native
context compilation, source replay, codecs, complete prompt selection and host
capsule expansion use their existing production owners. Authored sources and
complete prompt text remain private transient fixtures, never report content.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch


SCHEMA = "terminal-capsule-columns-offline-experiment@1"
FUNCTION_COUNTS = (4, 8)
BUDGETS = ("legacy", "one-capsule@1", "two-capsules@1")
MAX_REPORT_BYTES = 1024 * 1024
# Six genuine indexed contexts retain local DuckDB/DuckLake files (~91 MB).
# This bounds authored scratch storage, independently of prompt/report limits.
MAX_FIXTURE_BYTES = 128 * 1024 * 1024
MAX_FIXTURE_FILES = 4096
MAX_MESSAGE_BYTES = 256000
PROVIDER = "codex_cli"
MODEL = "gpt-6.1-sol"
_LOCK = threading.Lock()
_FALSE = dict(live_provider_calls=0, model_calls=0, training_executed=False,
    provider_usage_measured=False, grok_tokenizer_compatibility_verified=False,
    codex_tokenizer_compatibility_verified=False,
    provider_ingestion_verified=False, deployed_worker_retrieval_verified=False,
    source_body_omission_performed=False, token_savings_established=False,
    raw_content_exported=False, proof_authority=False, completion_authority=False)
_CHECKS = frozenset({"exact_transport_restoration", "exact_native_prompt_restoration",
    "required_raw_sources_identical", "native_constraints_identical", "instruction_literal_preserved",
    "workspace_guidance_identical", "all_selected_capsules_and_admissions_identical",
    "source_bytes_unchanged", "allocated_source_bytes_unchanged", "producer_payload_unchanged",
    "host_expansion_rows_exact", "complete_router_inputs_exact"})
_CLEANUP_FAILURE_CODES = frozenset({"private_fixture_mode_required",
    "private_fixture_identity_changed", "private_fixture_file_bound_exceeded",
    "unexpected_private_fixture_symlink", "unexpected_private_fixture_file_kind",
    "private_fixture_byte_bound_exceeded"})


def _need(condition, code):
    if not condition:
        raise ValueError(code)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_sha256(report):
    """FULL canonical report digest, including the internal report seal."""
    return _sha(_wire(report))


def _measure(text, counter):
    _need(type(text) is str, "measurement_requires_literal_text")
    raw = text.encode("utf-8")
    _need(len(raw) <= MAX_MESSAGE_BYTES, "measurement_text_exceeds_bound")
    count = counter(text) if counter is not None else None
    _need(count is None or type(count) is int and 0 <= count <= (1 << 63) - 1,
          "invalid_local_token_count")
    return dict(sha256=_sha(raw), utf8_bytes=len(raw), local_tokenizer_proxy_tokens=count)


def _trace(messages, counter):
    _need(type(messages) is list and len(messages) <= 8, "authored_trace_population_exceeds_bound")
    parts = [_measure(text, counter) for text in messages]
    return dict(message_count=len(parts), messages=parts,
        utf8_bytes=sum(part["utf8_bytes"] for part in parts),
        local_tokenizer_proxy_tokens=sum(part["local_tokenizer_proxy_tokens"] for part in parts)
            if counter is not None else None,
        scope="authored host-only discovery, request, exact expansion response and guidance once; no agent/model delivery observation")


@contextmanager
def _private_fixture(parent):
    parent = Path(parent).absolute()
    _need(parent.resolve(strict=True) == parent and not parent.is_symlink(), "canonical_scratch_parent_required")
    info = parent.stat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "owned_scratch_parent_required")
    path = Path(tempfile.mkdtemp(prefix="capsule-columns-", dir=parent))
    before = path.lstat()
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    cleanup = dict(private_fixture_cleanup_complete=False, private_fixture_files=None,
                   private_fixture_bytes=None, private_fixture_storage_bound_bytes=MAX_FIXTURE_BYTES)
    entries, total = 0, 0
    try:
        _need(stat.S_IMODE(before.st_mode) == 0o700, "private_fixture_mode_required")
        yield path, cleanup
    finally:
        try:
            current = path.lstat()
            held = os.fstat(descriptor)
            _need((current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
                == (held.st_dev, held.st_ino) and stat.S_ISDIR(current.st_mode)
                and path.resolve(strict=True) == path, "private_fixture_identity_changed")
            pending = [path]
            while pending:
                directory = pending.pop()
                with os.scandir(directory) as children:
                    for entry in children:
                        entries += 1
                        _need(entries <= MAX_FIXTURE_FILES, "private_fixture_file_bound_exceeded")
                        row = entry.stat(follow_symlinks=False)
                        _need(not stat.S_ISLNK(row.st_mode), "unexpected_private_fixture_symlink")
                        if stat.S_ISDIR(row.st_mode):
                            pending.append(Path(entry.path))
                        else:
                            _need(stat.S_ISREG(row.st_mode), "unexpected_private_fixture_file_kind")
                            total += row.st_size
                            _need(total <= MAX_FIXTURE_BYTES, "private_fixture_byte_bound_exceeded")
            cleanup.update(private_fixture_files=entries, private_fixture_bytes=total)
            current = path.lstat()
            _need((current.st_dev, current.st_ino) == (held.st_dev, held.st_ino)
                and stat.S_ISDIR(current.st_mode) and path.resolve(strict=True) == path,
                "private_fixture_identity_changed")
            shutil.rmtree(path)
            cleanup["private_fixture_cleanup_complete"] = not path.exists()
        except (OSError, ValueError) as error:
            code = str(error) if type(error) is ValueError else ""
            cleanup["private_fixture_cleanup_reason"] = (code if code in _CLEANUP_FAILURE_CODES
                else "owned_fixture_cleanup_io_unavailable" if isinstance(error, OSError)
                else "owned_fixture_cleanup_unavailable")
            cleanup["private_fixture_observed_prefix_files"] = entries
            cleanup["private_fixture_observed_prefix_bytes"] = total
        finally:
            os.close(descriptor)


@contextmanager
def _scripted_provider(captured):
    """Explicit offline seam; none of these calls reaches a CLI/provider."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    def generate(text, **options):
        _need(options.get("provider") == PROVIDER and options.get("model_name") == MODEL
            and options.get("allow_local_fallback") is False
            and options.get("allow_cross_provider_fallback") is False,
            "scripted_route_changed")
        captured.append(text)
        set_last_cli_observation(PROVIDER, {"exit_code": 0})
        return "Authored offline comparison response."

    with ExitStack() as stack:
        stack.enter_context(patch.object(intelligence_index, "discover_available_providers", return_value=[PROVIDER]))
        stack.enter_context(patch.object(intelligence_index, "select_efficient_route",
            return_value=SimpleNamespace(provider=PROVIDER, model_name=MODEL,
                                         catalog_revision="authored-offline-provider-seam")))
        stack.enter_context(patch.object(llm_router, "get_llm_provider", return_value=object()))
        stack.enter_context(patch.object(llm_router, "generate_text", side_effect=generate))
        yield


def _source(function_count):
    return "# Authored public fixture; keep source_cid and columns literally.\n" + "\n".join(
        f"def helper_{index}(value: int = {index}) -> int:\n"
        "    '''Preserve literal words: unsupported, unknown, café.'''\n"
        f"    return value + {index}\n" for index in range(function_count))


def _compile_prompt(repository, artifact, instruction):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    semantic = json.loads(artifact.read_text(encoding="utf-8"))
    references = build_text_context_references(artifact.read_text(encoding="utf-8"),
        reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(), repository_id="repo:authored-columns",
        tree_id="tree:authored-columns", required=True, chunk_bytes=6000)
    constraints = {"allowed_paths": ["module.py"], "security_obligations": ["No network, install, or scope expansion."],
        "unresolved_facts": ["Source behavior and complete task correctness remain unproved."],
        "validation_commands": [["python", "smoke.py"]]}
    compiled = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:authored-columns", tree_id="tree:authored-columns",
        objective_id=semantic["task_id"], objective_revision="sha256:authored-task",
        policy_id="policy:authored-columns", policy_revision="sha256:authored-policy",
        caller="supervisor:authored-offline", stage="implementation", goal={"id": semantic["task_id"]},
        authority={"mode": "candidate_only", "completion_authority": False}, scope=constraints,
        acceptance={"criteria": ["Independent validation remains pending."]}, evidence=references)
    suffix = "\nComplete literal public instruction:\n" + instruction
    return render_context_capsule(compiled.capsule) + suffix, semantic, constraints, suffix


def _case(repository, workspace, function_count, budget, counter, counter_identity):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_columns as columns
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_expansion import expand_semantic_capsules
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import SemanticMetadataTokenCounter
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
    from .terminal_capsule_selection_policy import capsule_budget_limit

    source_before = {name: _sha((repository / name).read_bytes())
                     for name in ("module.py", "instruction.md", "smoke.py")}
    allocated_before = {name: _sha((workspace / name).read_bytes()) for name in source_before}
    _need(allocated_before == source_before, "allocated_source_bytes_differ")
    instruction = (repository / "instruction.md").read_text(encoding="utf-8")
    output = repository / ".runtime" / ("semantic-" + budget.replace("@", "-"))
    with public_replay_without_metadata():
        prepared = prepare_semantic_context(repository=repository,
            paths=["module.py", "instruction.md", "smoke.py"], program_paths=["module.py"],
            required_raw_paths=["instruction.md", "smoke.py"], objective="Inspect helper functions; preserve required evidence.",
            task_id="AUTHORED-CAPSULE-COLUMNS", output=output, worker_query=instruction,
            worker_capsule_limit=capsule_budget_limit(budget), worker_max_bytes=32768,
            worker_selector="duckdb-query-symbol-name@1")
        artifact = output / "worker-context.json"
        prompt, semantic, constraints, suffix = _compile_prompt(repository, artifact, instruction)
        artifact_pin = _sha(artifact.read_bytes())
        encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
        candidate = columns.project_capsule_columns(encoded.provider_prompt)
        restored = columns.restore_capsule_columns(provider_prompt=candidate.provider_prompt, receipt=candidate.receipt)
        _need(restored == encoded.provider_prompt, "columns_transport_restore_mismatch")
        _need(codec.restore_semantic_router_prompt(provider_prompt=restored,
            table=encoded.table, repository=repository) == prompt, "columns_native_restore_mismatch")
        native_wire = json.loads(encoded.provider_prompt)
        restored_wire = json.loads(restored)
        native_model, native_advisory = router.render_model_prompt(prompt=encoded.provider_prompt,
            purpose="coding", workspace=workspace, semantic_transport=True)
        candidate_model, candidate_advisory = router.render_model_prompt(prompt=candidate.provider_prompt,
            purpose="coding", workspace=workspace, semantic_transport=True)
        typed_counter = (SemanticMetadataTokenCounter(count=counter, provider=PROVIDER, model=MODEL,
            tokenizer="o200k_base_local_proxy", tokenizer_revision=counter_identity["ranks_sha256"],
            qualification="estimate") if counter is not None else None)
        selection = columns.select_capsule_columns(view=candidate, native_complete_prompt=native_model,
            candidate_complete_prompt=candidate_model, token_counter=typed_counter, provider=PROVIDER, model=MODEL)
        arms = []
        for mode in ("legacy", columns.MODE):
            captured = []
            with _scripted_provider(captured), redirect_stdout(io.StringIO()):
                reply, receipt = router.run(prompt=prompt, provider=PROVIDER, model=MODEL,
                    timeout=30, max_output_tokens=512, semantic_repository=repository,
                    semantic_metadata_view=mode, semantic_metadata_token_counter=typed_counter if mode != "legacy" else None)
            expected = native_model if mode == "legacy" else selection.selected_prompt
            _need(captured == [expected] and receipt["model_prompt_sha256"] == _sha(expected.encode())
                and receipt["model_prompt_bytes"] == len(expected.encode()), "router_complete_input_differs")
            _need(reply == "Authored offline comparison response.", "scripted_response_changed")
            arms.append(dict(requested_mode=mode,
                selected_mode="legacy" if mode == "legacy" else receipt["semantic_metadata_view"]["selected_mode"],
                complete_model_prompt=_measure(expected, counter),
                scripted_provider_calls=len(captured), source_freshness_checked=True))
        requested = [row["stable_symbol_id"] for row in semantic["capsules"]]
        _need(bool(requested), "fixture_selected_capsule_population_empty")
        if function_count == 8 and budget == "legacy":
            _need(len(requested) >= 4, "largest_fixture_requires_four_selected_rows")
        expanded = expand_semantic_capsules(repository=repository,
            artifact=artifact.relative_to(repository).as_posix(), expected_sha256=artifact_pin,
            task_id=semantic["task_id"], stable_symbol_ids=requested, max_bytes=65536)
        _need(expanded["capsules"] == semantic["capsules"]
            and expanded["admissions"] == semantic["admissions"], "host_expansion_population_differs")
        discovery = dict(schema="authored-host-expansion-discovery@1", semantic_root_cid=semantic["semantic_root_cid"],
                         selected_symbols=semantic["worker_projection"]["selected_symbols"])
        request = dict(schema="authored-host-expansion-request@1", task_id=semantic["task_id"],
                       semantic_root_cid=semantic["semantic_root_cid"], stable_symbol_ids=requested)
        guidance = "Host-only authored expansion workload. " + expanded["raw_source_policy"]
        workload = _trace([_wire(discovery).decode(), _wire(request).decode(), _wire(expanded).decode(), guidance], counter)
        for arm in arms:
            arm["complete_authored_trace"] = dict(utf8_bytes=arm["complete_model_prompt"]["utf8_bytes"] + workload["utf8_bytes"],
                local_tokenizer_proxy_tokens=arm["complete_model_prompt"]["local_tokenizer_proxy_tokens"]
                    + workload["local_tokenizer_proxy_tokens"] if counter is not None else None,
                scope="one complete rendered coding prompt plus the same host-only authored expansion workload once")
        source_after = {name: _sha((repository / name).read_bytes()) for name in source_before}
        allocated_after = {name: _sha((workspace / name).read_bytes()) for name in source_before}
        checks = dict(exact_transport_restoration=restored == encoded.provider_prompt,
            exact_native_prompt_restoration=True,
            required_raw_sources_identical=native_wire["translated_semantic"]["raw_sources"] == restored_wire["translated_semantic"]["raw_sources"],
            native_constraints_identical=native_wire["native_context"] == restored_wire["native_context"],
            instruction_literal_preserved=semantic["raw_sources"]["instruction.md"] == instruction
                and native_wire["native_suffix"] == suffix,
            workspace_guidance_identical=native_advisory == candidate_advisory,
            all_selected_capsules_and_admissions_identical=restored_wire["translated_semantic"]["capsules"]
                == native_wire["translated_semantic"]["capsules"] and restored_wire["translated_semantic"]["admissions"]
                == native_wire["translated_semantic"]["admissions"],
            source_bytes_unchanged=source_before == source_after,
            allocated_source_bytes_unchanged=allocated_before == allocated_after == source_before,
            producer_payload_unchanged=artifact_pin == _sha(artifact.read_bytes()),
            host_expansion_rows_exact=True, complete_router_inputs_exact=True)
        _need(all(checks.values()), "required_fact_preservation_failed")
        return dict(function_count=function_count, capsule_budget=budget,
            selected_capsule_count=len(semantic["capsules"]), full_capsule_count=prepared["capsules"],
            omitted_capsule_count=prepared["capsules"] - len(semantic["capsules"]),
            source_sha256=source_before, required_raw_paths=list(semantic["required_raw_paths"]),
            producer_payload_sha256=artifact_pin, semantic_root_cid=semantic["semantic_root_cid"],
            constraints_sha256=_sha(_wire(constraints)), checks=checks,
            candidate_complete_model_prompt=_measure(candidate_model, counter),
            selection=selection.receipt, arms=arms, host_authored_expansion_workload=workload,
            selected_token_proxy_delta=arms[1]["complete_authored_trace"]["local_tokenizer_proxy_tokens"]
                - arms[0]["complete_authored_trace"]["local_tokenizer_proxy_tokens"] if counter is not None else None,
            selected_utf8_byte_delta=arms[1]["complete_authored_trace"]["utf8_bytes"] - arms[0]["complete_authored_trace"]["utf8_bytes"])


def _pins():
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_columns as columns
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime, semantic_router_translation, semantic_capsule_expansion
    from ipfs_datasets_py.logic.software_contracts.semantic_state import models, capsules
    from ipfs_accelerate_py.agent_supervisor.context import context_compiler
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_selection, semantic_candidate_index
    from . import terminal_source_view_experiment, terminal_capsule_selection_policy
    modules = [columns, router, semantic_context_runtime, semantic_router_translation, semantic_capsule_expansion,
        semantic_capsule_selection, semantic_candidate_index, context_compiler, models, capsules,
        terminal_source_view_experiment, terminal_capsule_selection_policy]
    pins = {module.__name__: _sha(Path(module.__file__).read_bytes()) for module in modules}
    pins[__name__] = _sha(Path(__file__).read_bytes())
    return dict(scope="listed_source_files_only; not transitive provenance", files=pins)


def run_experiment(*, scratch_root: Path, tokenizer_cache: Path | None = None) -> dict:
    """Run six within-budget pairs serially; no models, downloads or target corpus."""
    from .terminal_source_view_experiment import local_token_counter
    _need(_LOCK.acquire(blocking=False), "offline_experiment_already_active")
    previous_directory = Path.cwd()
    try:
        counter, tokenizer = local_token_counter(tokenizer_cache)
        pins_before = _pins()
        cases = []
        with _private_fixture(scratch_root) as (fixture, cleanup):
            for count in FUNCTION_COUNTS:
                repository = fixture / ("functions-" + str(count))
                repository.mkdir(mode=0o700)
                subprocess.run(["git", "init", "-q", str(repository)], check=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
                (repository / "module.py").write_text(_source(count), encoding="utf-8")
                (repository / "instruction.md").write_text("Inspect helper functions. Preserve exact defaults and café literals.\n"
                    "Keep unsupported and unknown facts; do not claim proof or widen write scope.\n", encoding="utf-8")
                (repository / "smoke.py").write_text("# Authored validation remains unexecuted.\n", encoding="utf-8")
                subprocess.run(["git", "-C", str(repository), "add", "--", "module.py", "instruction.md", "smoke.py"],
                    check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
                subprocess.run(["git", "-c", "user.name=Authored Offline Fixture", "-c", "user.email=authored@example.invalid",
                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(repository),
                    "commit", "-q", "-m", "Capture authored offline comparison inputs"], check=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
                workspace = fixture / ("allocated-" + str(count))
                subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", str(workspace), "HEAD"],
                    check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
                os.chdir(workspace)
                try:
                    for budget in BUDGETS:
                        cases.append(_case(repository, workspace, count, budget, counter, tokenizer))
                finally:
                    os.chdir(previous_directory)
        unchanged = pins_before == _pins()
        result = dict(schema=SCHEMA, status="qualified_offline" if counter is not None
            and unchanged and cleanup["private_fixture_cleanup_complete"] else "incomplete_offline",
            qualified=counter is not None and unchanged and cleanup["private_fixture_cleanup_complete"],
            population=dict(function_counts=list(FUNCTION_COUNTS), capsule_budgets=list(BUDGETS), within_budget_pairs=6),
            source_corpus_scope="authored 4/8-function fixtures; not public benchmark task coverage",
            source_pins=pins_before, source_pins_unchanged=unchanged,
            tokenizer={**tokenizer, "qualification": "estimate", "grok_compatibility_verified": False},
            cases=cases, **cleanup, **_FALSE)
        result["report_sha256"] = _sha(_wire(result))
        _need(len(_wire(result)) <= MAX_REPORT_BYTES, "experiment_report_exceeds_bound")
        return result
    finally:
        os.chdir(previous_directory)
        _LOCK.release()


def validate_report(report, *, expected_report_sha256):
    """Check an independent FULL pin, finite population and retained arithmetic.

    This does not rerun source preparation or authenticate an unpinned report.
    """
    _need(type(report) is dict and len(_wire(report)) <= MAX_REPORT_BYTES, "bounded_experiment_report_required")
    _need(report_sha256(report) == expected_report_sha256, "independent_report_pin_differs")
    _need(report.get("schema") == SCHEMA and report.get("report_sha256") == _sha(_wire(
        {key: value for key, value in report.items() if key != "report_sha256"})), "experiment_seal_differs")
    _need(all(type(report.get(key)) is type(value) and report.get(key) == value for key, value in _FALSE.items()),
          "experiment_authority_or_execution_scope_differs")
    _need(type(report.get("qualified")) is bool and type(report.get("source_pins_unchanged")) is bool
        and type(report.get("private_fixture_cleanup_complete")) is bool
        and type(report.get("tokenizer")) is dict and type(report["tokenizer"].get("available")) is bool,
        "experiment_qualification_fields_differ")
    qualified = report["tokenizer"]["available"] and report["source_pins_unchanged"] and report["private_fixture_cleanup_complete"]
    _need(report["qualified"] is qualified and report.get("status") == ("qualified_offline" if qualified else "incomplete_offline"),
          "experiment_qualification_claim_differs")
    _need(type(report.get("private_fixture_storage_bound_bytes")) is int
        and report["private_fixture_storage_bound_bytes"] == MAX_FIXTURE_BYTES,
        "experiment_fixture_storage_bound_differs")
    cases = report.get("cases")
    _need(type(cases) is list and [(row["function_count"], row["capsule_budget"]) for row in cases]
        == [(count, budget) for count in FUNCTION_COUNTS for budget in BUDGETS], "experiment_population_differs")
    for case in cases:
        _need(type(case["checks"]) is dict and set(case["checks"]) == _CHECKS
            and all(value is True for value in case["checks"].values()), "experiment_preservation_not_verified")
        shared = case["host_authored_expansion_workload"]
        _need(type(shared.get("messages")) is list and len(shared["messages"]) == shared.get("message_count") == 4,
              "authored_trace_message_population_differs")
        for message in [*shared["messages"], case["candidate_complete_model_prompt"],
                        *[arm["complete_model_prompt"] for arm in case["arms"]]]:
            _need(type(message) is dict and set(message) == {"sha256", "utf8_bytes", "local_tokenizer_proxy_tokens"}
                and type(message["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", message["sha256"])
                and type(message["utf8_bytes"]) is int and 0 <= message["utf8_bytes"] <= MAX_MESSAGE_BYTES,
                "experiment_message_measurement_differs")
            token_count = message["local_tokenizer_proxy_tokens"]
            _need((type(token_count) is int and 0 <= token_count <= (1 << 63) - 1)
                if report["tokenizer"]["available"] else token_count is None, "experiment_token_measurement_differs")
        _need(shared["utf8_bytes"] == sum(message["utf8_bytes"] for message in shared["messages"]),
              "authored_trace_byte_subtotal_differs")
        if report["tokenizer"]["available"]:
            _need(shared["local_tokenizer_proxy_tokens"] == sum(message["local_tokenizer_proxy_tokens"] for message in shared["messages"]),
                  "authored_trace_token_subtotal_differs")
        _need(len(case["arms"]) == 2 and [arm["requested_mode"] for arm in case["arms"]]
            == ["legacy", "capsule-columns@1"], "experiment_arms_differ")
        for arm in case["arms"]:
            _need(arm["complete_authored_trace"]["utf8_bytes"] == arm["complete_model_prompt"]["utf8_bytes"] + shared["utf8_bytes"],
                  "authored_trace_byte_total_differs")
            if report["tokenizer"]["available"]:
                _need(arm["complete_authored_trace"]["local_tokenizer_proxy_tokens"]
                    == arm["complete_model_prompt"]["local_tokenizer_proxy_tokens"] + shared["local_tokenizer_proxy_tokens"],
                    "authored_trace_token_total_differs")
        _need(case["selected_utf8_byte_delta"] == case["arms"][1]["complete_authored_trace"]["utf8_bytes"]
            - case["arms"][0]["complete_authored_trace"]["utf8_bytes"], "paired_byte_delta_differs")
        if report["tokenizer"]["available"]:
            _need(case["selected_token_proxy_delta"] == case["arms"][1]["complete_authored_trace"]["local_tokenizer_proxy_tokens"]
                - case["arms"][0]["complete_authored_trace"]["local_tokenizer_proxy_tokens"], "paired_token_delta_differs")
    return dict(schema="terminal-capsule-columns-report-validation@1", valid=True,
        validation_scope="independently pinned report population, flags and arithmetic; no independent experiment rerun", **_FALSE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args()
    _need(args.output.parent.resolve(strict=True) == args.output.parent.absolute(), "canonical_output_parent_required")
    result = run_experiment(scratch_root=args.scratch_root, tokenizer_cache=args.tokenizer_cache)
    validate_report(result, expected_report_sha256=report_sha256(result))
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_wire(result) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(schema=SCHEMA, qualified=result["qualified"],
        report_sha256=report_sha256(result), output_bytes=len(_wire(result)) + 1, **_FALSE), sort_keys=True))
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

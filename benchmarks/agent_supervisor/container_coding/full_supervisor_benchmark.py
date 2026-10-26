"""One-shot original Harbor task trials for the full and no-index supervisor arms."""
from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import (
    DEFAULT_GROK_CODING_MAX_TURNS, validate_grok_coding_max_turns, grok_coding_selection,
    require_grok_coding_archive, grok_coding_runtime_matches,
    grok_coding_not_applicable_provider_free,
)

import argparse
from datetime import datetime
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time

from benchmarks.agent_supervisor.container_coding.native_codex_baseline import (
    TASK, MODEL, CLI_VERSION, REASONING, _hash, _json, _task_hashes, _task_path, _prepared_task,
    _trial_task_matches,
    config_for as baseline_config,
)
from benchmarks.agent_supervisor.container_coding import benchmark_controls
from .benchmark_resource_profile import PROFILES, validate_resource_profile
from .terminal_retrieval_selection import (
    selected_retrieval_revision, require_retrieval_revision, validate_model_revision,
)
from .benchmark_provider_profile import (prepared_provider_identity, require_runtime_provider_profile,
    resolve_provider_profile, PROVIDER_PROFILES)
from .terminal_semantic_transport_policy import (
    DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, SEMANTIC_TRANSPORT_SCHEMAS,
    validate_semantic_transport_schema, semantic_transport_selection, require_semantic_transport_archive,
)
from .terminal_coding_reply_policy import (
    DEFAULT_CODING_REPLY_MODE, CODING_REPLY_MODES, validate_coding_reply_mode,
    coding_reply_selection, require_coding_reply_archive,
)
from .terminal_source_input_policy import (
    DEFAULT_SOURCE_INPUT_POLICY, SOURCE_INPUT_POLICIES, validate_source_input_policy,
    source_input_selection, require_source_input_archive, source_input_dispatch_observation,
)
from .terminal_query_reference_policy import (
    DEFAULT_QUERY_REFERENCE_POLICY, QUERY_REFERENCE_POLICIES, validate_query_reference_policy,
    query_reference_selection, require_query_reference_archive, query_reference_dispatch_observation,
)
from .terminal_repository_source_policy import (
    DEFAULT_REPOSITORY_SOURCE_POLICY, REPOSITORY_SOURCE_POLICIES, validate_repository_source_policy,
    repository_source_selection, require_repository_source_archive, repository_source_dispatch_observation,
)
from .terminal_source_view_policy import (
    DEFAULT_SOURCE_VIEW_POLICY, SOURCE_VIEW_POLICIES, validate_source_view_advisory_policy,
    source_view_advisory_selection, require_source_view_advisory_archive, source_view_dispatch_observation,
)
from .terminal_tool_output_policy import (
    DEFAULT_TOOL_OUTPUT_POLICY, TOOL_OUTPUT_POLICIES, validate_tool_output_policy,
    tool_output_selection, require_tool_output_archive, tool_output_dispatch_observation,
)
from .terminal_semantic_metadata_policy import (
    DEFAULT_SEMANTIC_METADATA_VIEW, SEMANTIC_METADATA_VIEWS, validate_semantic_metadata_view,
    semantic_metadata_selection, semantic_metadata_observation, require_semantic_metadata_archive,
)
from .terminal_capsule_selection_policy import (
    DEFAULT_CAPSULE_SELECTION, CAPSULE_SELECTIONS, validate_capsule_selection,
    capsule_selection_options, capsule_selection_observation, require_capsule_selection_archive,
    DEFAULT_CAPSULE_BUDGET, CAPSULE_BUDGETS, validate_capsule_budget,
    capsule_budget_options, capsule_budget_observation, require_capsule_budget_archive,
)
from .terminal_planning_policy import (
    DEFAULT_PLANNING_MODE, PLANNING_MODES, validate_planning_mode,
    planning_selection, planning_observation, require_planning_archive,
)
from .terminal_semantic_working_set_policy import (
    DEFAULT_SEMANTIC_WORKING_SET, SEMANTIC_WORKING_SETS, validate_semantic_working_set,
    semantic_working_set_selection, semantic_working_set_observation, require_semantic_working_set_archive,
)
from .terminal_source_edit_policy import (
    DEFAULT_SOURCE_EDIT_MODE, SOURCE_EDIT_MODES, validate_source_edit_mode,
    source_edit_selection, source_edit_observation, require_source_edit_archive,
)
from .terminal_source_view_policy import (
    DEFAULT_SOURCE_VIEW_PROFILE, SOURCE_VIEW_PROFILES, validate_source_view_profile,
    source_view_selection, source_view_observation, require_source_view_archive,
    validate_source_view_policy,
)
from .terminal_semantic_context_policy import (
    DEFAULT_SEMANTIC_CONTEXT_PROFILE, SEMANTIC_CONTEXT_PROFILES,
    validate_semantic_context_profile, semantic_context_selection, require_semantic_context_archive,
)
from .terminal_numerical_policy import (
    PROFILES as NUMERICAL_PROFILES, validate_numerical_profile, numerical_selection,
    numerical_observation, require_numerical_archive,
)

from .container_worker_deployment import (
    doctor_dispatch_arguments, prompt_tokenizer_arguments, semantic_context_arguments,
    semantic_decision_arguments, validate_legacy_semantic_route,
)
from .semantic_decision_nomination_delivery import (
    capture_nomination_snapshot, host_nomination_path, read_nomination_source,
    verify_nomination_snapshot,
)

ADAPTER = "benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent"
INTENT_SOURCE_PATH = ".supervisor-instruction.md"
MAX_INTENT_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_INSTRUCTION_BYTES = 32768
MAX_TASK_PROFILE_BYTES = 65536
MAX_SOURCE_VIEW_POLICY_BYTES = 65536


MAX_RUNTIME_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_RUNTIME_ARCHIVE_MEMBERS = 100000
MAX_RUNTIME_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
MAX_TIMEOUT_RUNTIME_SOURCE_BYTES = 2 * 1024 * 1024
TIMEOUT_RUNTIME_SOURCE_NAMES = (
    "timeout_policy.py",
    "terminal_container_supervisor.py",
    "terminal_indexed_preparation.py",
    "container_worker_deployment.py",
)
TIMEOUT_RUNTIME_ARCHIVE_PREFIX = "source/benchmarks/agent_supervisor/container_coding/"
DECISION_RUNTIME_SOURCE_NAMES = (
    "semantic_decision_worker.py", "semantic_decision_owner.py", "semantic_decision_provider.py",
    "semantic_decision_dialogue.py", "semantic_decision_packet.py",
    "semantic_decision_usage.py", "semantic_projection_dispatch.py",
    "semantic_router_translation.py", "semantic_context_projection.py",
    "router_implementation_runner.py", "automatic_prompt_tokenizer.py",
    "local_prompt_tokenizer.py", "container_worker_boundary.py",
    "doctor_residual_context.py", "router_public_instruction.py",
    "cumulative_usage_reconciliation.py", "multi_supervisor_runner.py",
    "semantic_typed_owner_policy.py", "semantic_typed_source_context.py",
    "semantic_typed_dependency_proposal.py", "semantic_typed_nomination_lineage.py",
    "semantic_selection_relation_audit.py", "semantic_selection_dependency_audit.py",
)
DECISION_RUNTIME_ARCHIVE_PREFIX = "source/ipfs_accelerate_py/agent_supervisor/runtime/"
DECISION_BENCHMARK_SOURCE_NAMES = (
    "terminal_decision_policy.py", "benchmark_controls.py", "full_supervisor_benchmark.py",
    "terminal_decision_usage_projection.py",
    "terminal_doctor_dispatch.py", "terminal_context_audit.py", "terminal_context_evidence.py",
    "semantic_decision_input_audit.py", "automatic_tokenizer_audit.py",
    "semantic_decision_nomination_delivery.py", "full_supervisor_harbor_agent.py",
    "terminal_deployment.py",
)



def _decision_runtime_sources(*, retrieval_query_reference: bool = False) -> dict[str, Path]:
    """Declared worker dependencies, argv adapter and historical input observers."""
    runtime = Path(__file__).resolve().parents[3] / "ipfs_accelerate_py/agent_supervisor/runtime"
    sources = {DECISION_RUNTIME_ARCHIVE_PREFIX + name: runtime / name for name in DECISION_RUNTIME_SOURCE_NAMES}
    sources.update({TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name: Path(__file__).with_name(name)
                    for name in DECISION_BENCHMARK_SOURCE_NAMES})
    sources["source/ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py"] = (
        runtime.parent / "analysis/semantic_dependency_graph.py")
    if type(retrieval_query_reference) is not bool:
        raise ValueError("retrieval query reference must be boolean")
    if retrieval_query_reference:
        sources.update({DECISION_RUNTIME_ARCHIVE_PREFIX + name: runtime / name for name in (
            "semantic_retrieval_query_reference.py", "code_retrieval_context.py", "task_context_bundle.py",
        )})
    return sources

def _require_current_decision_runtime_pins(pins, *, native_input: bool = False,
                                         retrieval_query_reference: bool = False) -> None:
    route = "native input" if native_input else "decision"
    failure = f"runtime {route} source pins differ; prepare a new immutable trial"
    sources = _decision_runtime_sources(**({"retrieval_query_reference": True}
        if retrieval_query_reference else {}))
    if type(pins) is not dict or set(pins) != set(sources):
        raise ValueError(failure)
    for name, source in sources.items():
        if (source.is_symlink() or not source.is_file()
                or not 0 < source.stat().st_size <= MAX_TIMEOUT_RUNTIME_SOURCE_BYTES
                or _hash(source) != pins[name]):
            raise ValueError(failure)

def _require_current_timeout_runtime(
    archive: Path, manifest: dict, *, selected_sources: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    """Bind deadline and optional selected-route sources to archive bytes.

    The decision list is a bounded capability check over declared direct
    dependencies, not a proof of the complete imported dependency closure.
    """
    failure = ("runtime selected decision sources differ; rebuild the runtime archive" if selected_sources is not None
               else "runtime deadline sources differ; rebuild the runtime archive")
    if (type(manifest) is not dict
            or manifest.get("schema") != "terminal-supervisor-runtime-archive@1"
            or type(manifest.get("files")) is not list
            or len(manifest["files"]) > MAX_RUNTIME_ARCHIVE_MEMBERS):
        raise ValueError(failure)
    expected = {}
    for name in TIMEOUT_RUNTIME_SOURCE_NAMES:
        source = Path(__file__).with_name(name)
        size = source.stat().st_size
        if not 0 < size <= MAX_TIMEOUT_RUNTIME_SOURCE_BYTES:
            raise ValueError(failure)
        expected[TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name] = (size, _hash(source))
    if selected_sources is not None:
        maximum_selected_sources = len(_decision_runtime_sources(retrieval_query_reference=True))
        if not isinstance(selected_sources, Mapping) or not 1 <= len(selected_sources) <= maximum_selected_sources:
            raise ValueError(failure)
        for name, source in selected_sources.items():
            if type(name) is not str:
                raise ValueError(failure)
            path = PurePosixPath(name)
            if (not name.startswith("source/") or path.is_absolute() or ".." in path.parts
                    or path.as_posix() != name or name in expected):
                raise ValueError(failure)
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError(failure)
            size = source.stat().st_size
            if not 0 < size <= MAX_TIMEOUT_RUNTIME_SOURCE_BYTES:
                raise ValueError(failure)
            expected[name] = (size, _hash(source))
    required_rows = {}
    for row in manifest["files"]:
        if type(row) is not dict:
            raise ValueError(failure)
        name = row.get("path")
        if type(name) is not str:
            raise ValueError(failure)
        if name in expected:
            if name in required_rows:
                raise ValueError(failure)
            size, digest = expected[name]
            if (type(row.get("bytes")) is not int or row["bytes"] != size
                    or row.get("sha256") != digest):
                raise ValueError(failure)
            required_rows[name] = row
    if required_rows.keys() != expected.keys():
        raise ValueError(failure)
    observed = set()
    count = total_size = 0
    try:
        with tarfile.open(archive / "runtime.tar.gz", mode="r|gz") as bundle:
            for member in bundle:
                count += 1
                total_size += member.size
                path = PurePosixPath(member.name)
                if (count > MAX_RUNTIME_ARCHIVE_MEMBERS
                        or total_size > MAX_RUNTIME_ARCHIVE_UNCOMPRESSED_BYTES
                        or member.size < 0 or not member.isfile()
                        or path.is_absolute() or ".." in path.parts
                        or member.name != path.as_posix()):
                    raise ValueError(failure)
                if member.name not in expected:
                    continue
                size, digest = expected[member.name]
                if member.name in observed or member.size != size:
                    raise ValueError(failure)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(failure)
                with stream:
                    raw = stream.read(size + 1)
                if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(failure)
                observed.add(member.name)
    except (OSError, tarfile.TarError, EOFError) as error:
        raise ValueError(failure) from error
    if observed != expected.keys():
        raise ValueError(failure)
    return {name: digest for name, (_, digest) in expected.items()}



_DECISION_FIELDS = ('semantic_decision_policy_artifact', 'semantic_decision_policy_sha256', 'semantic_decision_nomination_artifact', 'semantic_decision_nomination_source', 'semantic_decision_nomination_sha256', 'disable_semantic_typed_nomination', 'semantic_retrieval_query_reference', 'semantic_decision_max_calls', 'semantic_decision_max_total_input_tokens')
_DECISION_ROUTE_DEFAULTS = {'planning_mode': 'model', 'semantic_transport_schema': 'supervisor-semantic-router-input@1', 'coding_reply_mode': 'legacy', 'tool_output_policy': 'legacy', 'source_view_advisory_policy': 'legacy', 'repository_source_policy': 'legacy', 'source_input_policy': 'legacy', 'query_reference_policy': 'legacy', 'semantic_metadata_view': 'legacy', 'semantic_working_set': 'legacy', 'source_view_profile': 'disabled', 'source_edit_mode': 'disabled', 'semantic_context_profile': 'legacy-bounded@1', 'capsule_selection': 'lexical-query-symbol-name@1', 'capsule_budget': 'legacy', 'numerical_formalization_profile': None, 'metadata_census_only': False, 'resource_profile': None, 'task_profile': None, 'intent_requirement_contract': None, 'source384_config': None, 'numerical_selection_config': None, 'source_view_policy': None, 'grok_coding_max_turns': 128}

def _decision_options(options, *, arm, provider):
    """Validate one additive decision group while ordinary shapes stay intact."""
    from .terminal_decision_policy import decision_selection, validate_decision_route
    selected = decision_selection(**{key:options[key] for key in _DECISION_FIELDS if key in options})
    validate_decision_route(selected, arm=arm, provider=provider,
        **{key:options.get(key, default) for key,default in _DECISION_ROUTE_DEFAULTS.items()})
    return selected

def _decision_dispatch_observation(report, selected):
    """Bind the requested route to original retained rows; never repair them."""
    from .terminal_decision_usage_projection import measured_decision_usage
    problems = []
    value = report if type(report) is dict else {}
    if type(report) is not dict:
        problems.append("decision_report_unavailable")
    if value.get("semantic_decision_dialogue_expected") is not True:
        problems.append("decision_dialogue_expectation_missing")
    nomination = any(key in selected for key in (
        "semantic_decision_nomination_source", "semantic_decision_nomination_artifact"))
    dialogues = value.get("semantic_decision_dialogues")
    bounded_dialogues = type(dialogues) is list and 0 < len(dialogues) <= 4096
    if not bounded_dialogues or any(type(item) is not dict for item in dialogues):
        problems.append("decision_dialogue_population_unavailable")
        dialogues = []
    if nomination:
        if value.get("semantic_decision_owner_binding_expected") is not True:
            problems.append("decision_owner_expectation_missing")
        if value.get("semantic_decision_nomination_sha256") != selected["semantic_decision_nomination_sha256"]:
            problems.append("decision_nomination_pin_mismatch")
        typed = value.get("semantic_decision_typed_nomination_expected")
        if type(typed) is not bool or typed != (not selected.get("disable_semantic_typed_nomination", False)):
            problems.append("decision_typed_expectation_mismatch")
    else:
        for dialogue in dialogues:
            launch = dialogue.get("worker_launch")
            if (type(launch) is not dict
                    or launch.get("launch_policy_artifact") != selected["semantic_decision_policy_artifact"]
                    or launch.get("launch_policy_sha256") != selected["semantic_decision_policy_sha256"]):
                problems.append("decision_launch_policy_mismatch")
                break
    if selected.get("semantic_retrieval_query_reference"):
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        for dialogue in dialogues:
            launch = dialogue.get("worker_launch")
            held = dialogue.get("retrieval_query_reference")
            try:
                if type(launch) is not dict or type(held) is not dict:
                    raise ValueError("query_reference_unavailable")
                binding = SemanticRetrievalQueryBinding.from_dict(launch.get("retrieval_query_binding"))
                if (held.get("binding") != binding.to_dict()
                        or binding.task_id != value.get("semantic_decision_expected_task_id")
                        or any(type(held.get(key)) is not str or re.fullmatch("[0-9a-f]{64}", held[key]) is None
                            for key in ("native_context_sha256", "view_sha256", "native_reference_records_sha256"))):
                    raise ValueError("query_reference_mismatch")
            except (TypeError, ValueError):
                problems.append("decision_query_reference_binding_unavailable_or_mismatched")
                break
    usage = None
    error_type = None
    try:
        usage = measured_decision_usage(value)
    except Exception as error:
        error_type = type(error).__name__[:128]
        problems.append("decision_usage_projection_unavailable")
    coverage = usage.get("semantic_decision_coverage", {}) if type(usage) is dict else {}
    authoritative = coverage.get("authoritative_usage_coverage_complete") is True
    owner_complete = coverage.get("owner_binding_coverage_complete") is True if nomination else None
    native_complete = (type(usage) is dict
        and usage.get("native_scoped_usage_complete_observed") is True
        and usage.get("harbor_usage_exportable") is True
        and type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0)
    if not authoritative:
        problems.append("decision_authoritative_usage_coverage_incomplete")
    if nomination and not owner_complete:
        problems.append("decision_owner_binding_coverage_incomplete")
    if not native_complete:
        problems.append("decision_native_usage_incomplete")
    return {"schema": "terminal-semantic-decision-dispatch-observation@1",
        "requested_selection": dict(selected), "verified": not problems,
        "authoritative_usage_coverage_complete": authoritative,
        "owner_binding_coverage_complete": owner_complete,
        "native_usage_complete": native_complete, "problems": problems,
        **({"usage_projection_error_type": error_type} if error_type is not None else {}),
        "retained_receipt_usage_recomputed": True, "stored_usage_summary_trusted": False,
        "verification_scope": "frozen_requested_bindings_and_retained_native_usage_only",
        "actual_packet_ingestion_independently_verified": None,
        "source_freshness_verified": False,
        "source_audit_rerun": False, "completion_authority": False,
        "production_token_savings_established": False}

def _prompt_options(arm: str, *,
                    prompt_tokenizer_artifact: str | Path | None = None,
                    prompt_tokenizer_sha256: str | None = None,
                    max_input_tokens: int | None = None,
                    semantic_slot_policy_revision: str = "legacy@1",
                    semantic_context_projection: bool = False,
                    semantic_native_input: bool = False,
                    semantic_decision_policy_artifact: str | Path | None = None,
                    semantic_decision_policy_sha256: str | None = None,
                    semantic_decision_nomination_artifact: str | Path | None = None,
                    semantic_decision_nomination_source: str | Path | None = None,
                    semantic_decision_nomination_sha256: str | None = None,
                    disable_semantic_typed_nomination: bool = False,
                    semantic_retrieval_query_reference: bool = False,
                    semantic_decision_max_calls: int = 3,
                    semantic_decision_max_total_input_tokens: int | None = None) -> dict:
    """Validate requested options without reading or staging assets."""
    if arm not in {"full", "no-index"}:
        raise ValueError("unknown supervisor ablation")
    if type(semantic_native_input) is not bool:
        raise ValueError("native input control must be boolean")
    if type(semantic_retrieval_query_reference) is not bool:
        raise ValueError("retrieval query reference must be boolean")
    prompt_tokenizer_arguments(prompt_tokenizer_artifact=prompt_tokenizer_artifact,
                              prompt_tokenizer_sha256=prompt_tokenizer_sha256,
                              max_input_tokens=max_input_tokens)
    semantic_arguments = semantic_context_arguments(
        semantic_slot_policy_revision=semantic_slot_policy_revision,
        semantic_context_projection=semantic_context_projection,
        **({"semantic_native_input": semantic_native_input} if semantic_native_input is not False else {}))
    if semantic_arguments and arm != "full":
        raise ValueError("semantic context options require the full indexed arm")
    nomination_artifact = semantic_decision_nomination_artifact
    if semantic_decision_nomination_source is not None:
        if (semantic_decision_nomination_artifact is not None
                or semantic_decision_policy_artifact is not None
                or semantic_decision_policy_sha256 is not None):
            raise ValueError("host nomination source cannot combine with child-local decision artifacts")
        nomination_artifact = host_nomination_path(semantic_decision_nomination_source)
    decision_arguments = semantic_decision_arguments(
        semantic_decision_policy_artifact=semantic_decision_policy_artifact,
        semantic_decision_policy_sha256=semantic_decision_policy_sha256,
        semantic_decision_nomination_artifact=nomination_artifact,
        semantic_decision_nomination_sha256=semantic_decision_nomination_sha256,
        disable_semantic_typed_nomination=disable_semantic_typed_nomination,
        **({"semantic_retrieval_query_reference": semantic_retrieval_query_reference}
           if semantic_retrieval_query_reference is not False else {}),
        semantic_decision_max_calls=semantic_decision_max_calls,
        semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
    if decision_arguments and arm != "full":
        raise ValueError("semantic decision options require the full indexed arm")
    if decision_arguments and semantic_arguments:
        raise ValueError("semantic decision options cannot combine with semantic context alternatives")
    options = {
        "prompt_tokenizer_artifact": (
            str(prompt_tokenizer_artifact) if prompt_tokenizer_artifact is not None else None
        ),
        "prompt_tokenizer_sha256": prompt_tokenizer_sha256,
        "max_input_tokens": max_input_tokens,
        "semantic_slot_policy_revision": semantic_slot_policy_revision,
        "semantic_context_projection": semantic_context_projection,
    }
    if semantic_native_input:
        options["semantic_native_input"] = True
    if decision_arguments:
        options.update(
            semantic_decision_max_calls=semantic_decision_max_calls,
            semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
        if semantic_decision_nomination_source is not None:
            options.update(
                semantic_decision_nomination_source=str(nomination_artifact),
                semantic_decision_nomination_sha256=semantic_decision_nomination_sha256)
        elif semantic_decision_nomination_artifact is not None:
            options.update(
                semantic_decision_nomination_artifact=str(semantic_decision_nomination_artifact),
                semantic_decision_nomination_sha256=semantic_decision_nomination_sha256)
        else:
            options.update(
                semantic_decision_policy_artifact=str(semantic_decision_policy_artifact),
                semantic_decision_policy_sha256=semantic_decision_policy_sha256)
    if nomination_artifact is not None:
        options["disable_semantic_typed_nomination"] = disable_semantic_typed_nomination
    if semantic_retrieval_query_reference:
        options["semantic_retrieval_query_reference"] = True
    return options

def _prompt_option_observation(prepared: dict, agent_kwargs: dict | None) -> dict:
    """Report requested configuration independently of actual router selection."""
    legacy_keys = (
        "prompt_tokenizer_artifact", "prompt_tokenizer_sha256", "max_input_tokens",
        "semantic_slot_policy_revision", "semantic_context_projection",
    )
    native_keys = (*legacy_keys, "semantic_native_input")
    budget_keys = ("semantic_decision_max_calls", "semantic_decision_max_total_input_tokens")
    policy_keys = (*legacy_keys, *budget_keys, "semantic_decision_policy_artifact", "semantic_decision_policy_sha256")
    nomination_keys = (*legacy_keys, *budget_keys, "semantic_decision_nomination_artifact", "semantic_decision_nomination_sha256")
    source_keys = (*legacy_keys, *budget_keys, "semantic_decision_nomination_source", "semantic_decision_nomination_sha256")
    typed_nomination_keys = (*nomination_keys, "disable_semantic_typed_nomination")
    typed_source_keys = (*source_keys, "disable_semantic_typed_nomination")
    retrieval_nomination_keys = (*typed_nomination_keys, "semantic_retrieval_query_reference")
    retrieval_source_keys = (*typed_source_keys, "semantic_retrieval_query_reference")
    keys = (*policy_keys, "semantic_native_input", "semantic_decision_nomination_artifact",
            "semantic_decision_nomination_source", "semantic_decision_nomination_sha256",
            "disable_semantic_typed_nomination", "semantic_retrieval_query_reference")
    pinned = "prompt_comparison_options" in prepared
    if type(agent_kwargs) is not dict:
        return {
            "prompt_comparison_options": None,
            "prompt_comparison_options_valid": None if agent_kwargs is None else False,
            "prompt_comparison_basis": "unavailable_configuration",
            "prompt_comparison_config_unchanged": None if agent_kwargs is None else False,
        }
    try:
        configured = _prompt_options(prepared["arm"], **{
            key: agent_kwargs[key] for key in keys if key in agent_kwargs
        })
        renderer_options = {key: agent_kwargs[key] for key in (
            'semantic_transport_schema',
            'semantic_metadata_view',
            'semantic_working_set',
            'source_view_profile',
            'source_edit_mode',
            'source_view_policy',
            'coding_reply_mode',
            'tool_output_policy',
            'source_view_advisory_policy',
            'repository_source_policy',
            'source_input_policy',
            'query_reference_policy',
            'semantic_context_profile',
            'capsule_selection',
            'capsule_budget',
            'metadata_census_only',
        ) if key in agent_kwargs}
        renderer_options["provider"] = resolve_provider_profile(agent_kwargs.get("provider_profile"))["provider"]
        validate_legacy_semantic_route(policyoptions=configured, rendereroptions=renderer_options)
        expected = prepared.get("prompt_comparison_options")
        matches = None
        if pinned:
            matches = False
            if type(expected) is dict and set(expected) in (
                    set(legacy_keys), set(native_keys), set(policy_keys), set(nomination_keys),
                    set(source_keys), set(typed_nomination_keys), set(typed_source_keys),
                    set(retrieval_nomination_keys), set(retrieval_source_keys)):
                # Older option records retain their closed key sets; the
                # current API supplies its documented false opt-out default.
                canonical_expected = _prompt_options(prepared["arm"], **expected)
                matches = canonical_expected == configured
        valid = True
    except (TypeError, ValueError):
        configured, matches, valid = None, False, False
    return {
        "prompt_comparison_options": configured,
        "prompt_comparison_options_valid": valid,
        "prompt_comparison_basis": (
            "preparation" if pinned else "retained_configuration_without_preparation_pin"
        ),
        "prompt_comparison_config_unchanged": matches,
        **({"semantic_native_input_requested": True}
           if configured is not None and configured.get("semantic_native_input") is True else {}),
        **({"semantic_retrieval_query_reference_requested": True}
           if configured is not None and configured.get("semantic_retrieval_query_reference") is True else {}),
    }

def _doctor_policy_observation(prepared: dict, agent_kwargs: dict | None) -> dict:
    """Observe requested dispatch policy, independently of the executed route."""
    if type(agent_kwargs) is not dict:
        return {
            "doctor_dispatch_policy": None,
            "doctor_dispatch_policy_valid": None if agent_kwargs is None else False,
            "doctor_dispatch_policy_config_unchanged": None if agent_kwargs is None else False,
        }
    policy = agent_kwargs.get("doctor_dispatch_policy", "auto")
    try:
        doctor_dispatch_arguments(arm=prepared["arm"], doctor_dispatch_policy=policy)
        matches = None
        if "doctor_dispatch_policy" in prepared:
            expected = prepared["doctor_dispatch_policy"]
            doctor_dispatch_arguments(arm=prepared["arm"], doctor_dispatch_policy=expected)
            matches = expected == policy
        valid = True
    except (TypeError, ValueError):
        policy, matches, valid = None, False, False
    return {
        "doctor_dispatch_policy": policy,
        "doctor_dispatch_policy_valid": valid,
        "doctor_dispatch_policy_config_unchanged": matches,
    }


def _load_source_view_policy(path: Path | None) -> dict | None:
    if path is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError("source view policy must be a regular canonical file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_VIEW_POLICY_BYTES + 1)
    if not raw or len(raw) > MAX_SOURCE_VIEW_POLICY_BYTES:
        raise ValueError("source view policy exceeds its byte bound")
    return SourceViewPolicy.from_json(raw.decode("utf-8")).to_dict()


def _source_view_policy_state(prepared, config, *, source_view, configured_source_view,
                              current_config_sha256):
    """Recheck the recipe and implementation binding without backfilling history."""
    kwargs = config["agents"][0].get("kwargs", {})
    fields = {"source_view_policy", "source_view_policy_sha256", "source_view_policy_binding_sha256"}
    present = bool(fields & set(prepared)) or "source_view_policy" in kwargs
    declared_controls = prepared.get("comparison_controls")
    controls_unchanged = declared_controls is None or benchmark_controls.observe_controls(
        prepared, config, current_task_hashes=prepared.get("task_input_sha256"))[
            "configuration_unchanged"] is True
    if not present:
        # A removed recipe cannot be inferred from the remaining config alone.
        # Its existing frozen controls still detect that edited configuration;
        # unchanged legacy records and records without controls keep their shape.
        return None, controls_unchanged, not controls_unchanged
    try:
        selected = validate_source_view_policy(prepared.get("source_view_policy"),
            profile=source_view, arm=prepared["arm"])
        configured = validate_source_view_policy(kwargs.get("source_view_policy"),
            profile=configured_source_view, arm=prepared["arm"])
        if selected is None or configured is None:
            return selected, False, True
        observation = source_view_observation(source_view, selected)
        same = (selected == configured and controls_unchanged
            and current_config_sha256 == prepared.get("config_sha256")
            and all(prepared.get(key) == observation[key] for key in fields))
        return selected, same, True
    except (KeyError, TypeError, ValueError):
        return None, False, True


def _transport_task_profile(profile: dict | None, *, instruction: str | None = None) -> dict | None:
    if profile is None:
        return None
    from .terminal_task_profile import validate_task_profile
    encoded = json.dumps(profile, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_TASK_PROFILE_BYTES:
        raise ValueError("task profile exceeds its byte bound")
    return validate_task_profile(json.loads(encoded), instruction=instruction)


def _load_task_profile(path: Path | None, *, task: Path) -> dict | None:
    if path is None:
        if task.name != TASK:
            raise ValueError("nondefault supervisor task requires an explicit task profile")
        return None
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError("task profile must be a regular canonical file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_TASK_PROFILE_BYTES + 1)
    if not raw or len(raw) > MAX_TASK_PROFILE_BYTES:
        raise ValueError("task profile exceeds its byte bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate task profile field")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite task profile value")
    public = task / "instruction.md"
    if public.resolve(strict=True) != public or public.is_symlink() or not public.is_file():
        raise ValueError("public instruction must be a regular canonical file")
    with public.open("rb") as stream:
        source = stream.read(MAX_PUBLIC_INSTRUCTION_BYTES + 1)
    if not source or len(source) > MAX_PUBLIC_INSTRUCTION_BYTES:
        raise ValueError("public instruction exceeds its byte bound")
    instruction = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return _transport_task_profile(json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite),
                                   instruction=instruction)


def _transport_intent_contract(contract: dict | None) -> dict | None:
    """Match the adapter's bounded JSON transport without inferring requirements."""
    if contract is None:
        return None
    if type(contract) is not dict:
        raise ValueError("Intent coverage requires an explicit requirement contract object")
    encoded = json.dumps(contract, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_INTENT_CONTRACT_BYTES:
        raise ValueError("Intent requirement contract exceeds its byte bound")
    return json.loads(encoded)


def _load_intent_requirement_contract(path: Path | None, *, dataset: Path,
                                      task_name: str = TASK) -> dict | None:
    """Validate candidate requirements against the selected public task instruction."""
    if path is None:
        return None
    return load_intent_requirements_for_instruction(path, _task_path(dataset, task_name) / "instruction.md")


def load_intent_requirements_for_instruction(path: Path | None, instruction: Path) -> dict | None:
    """Read a bounded reviewed mapping for one explicitly selected public instruction."""
    if path is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
        validate_intent_requirement_contract,
    )

    path = Path(path).absolute()
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError("intent requirement contract must be a regular canonical file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INTENT_CONTRACT_BYTES + 1)
    if len(raw) > MAX_INTENT_CONTRACT_BYTES:
        raise ValueError("intent requirement contract exceeds its byte bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate intent requirement contract key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite intent requirement contract value")
    public = Path(instruction).absolute()
    if public.resolve(strict=True) != public or public.is_symlink() or not public.is_file():
        raise ValueError("public instruction must be a regular canonical file")
    with public.open("rb") as stream:
        source = stream.read(MAX_PUBLIC_INSTRUCTION_BYTES + 1)
    if not source or len(source) > MAX_PUBLIC_INSTRUCTION_BYTES:
        raise ValueError("public instruction exceeds its byte bound")
    # Harbor exposes the instruction as text; the container preparation also
    # reads that text before signing the immutable supervisor input.
    text = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    contract = validate_intent_requirement_contract(
        json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite), source_text=text,
    )
    if contract["source_path"] != INTENT_SOURCE_PATH:
        raise ValueError("benchmark intent requirements must bind the original public instruction")
    return _transport_intent_contract(contract)


def _intent_selection(contract: dict | None, *, planning_mode=DEFAULT_PLANNING_MODE) -> dict:
    validate_planning_mode(planning_mode, intent_requirement_contract=contract)
    strategy = ("intent_symbolic" if contract is not None and contract.get("schema") in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3", "intent-plan-requirement-contract@4", "intent-plan-requirement-contract@5"}
                else "intent_coverage" if contract is not None else "direct")
    if planning_mode != DEFAULT_PLANNING_MODE:
        strategy = "declared_contract"
    result = {"planning_strategy": strategy,
              "intent_requirement_contract_cid": None, "intent_requirement_contract_sha256": None}
    if contract is not None:
        from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json

        contract = _transport_intent_contract(contract)
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        result.update(intent_requirement_contract_cid=cid_for_dag_json(contract),
                      intent_requirement_contract_sha256=hashlib.sha256(encoded).hexdigest())
    return result


def config_for(dataset: Path, output: Path, archive: Path, arm: str,
               *, intent_requirement_contract: dict | None = None, resource_profile=None,
               setup_cache_selection: dict | None = None, task_name: str = TASK,
               task_profile: dict | None = None, provider_profile: str | None = None,
               semantic_transport_schema: str = DEFAULT_SEMANTIC_TRANSPORT_SCHEMA,
               tool_output_policy: str = DEFAULT_TOOL_OUTPUT_POLICY,
               source_view_advisory_policy: str = DEFAULT_SOURCE_VIEW_POLICY,
               repository_source_policy: str = DEFAULT_REPOSITORY_SOURCE_POLICY,
               source_input_policy: str = DEFAULT_SOURCE_INPUT_POLICY,
               query_reference_policy: str = DEFAULT_QUERY_REFERENCE_POLICY,
               coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
               semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
               semantic_working_set: str = DEFAULT_SEMANTIC_WORKING_SET,
               grok_coding_max_turns: int = DEFAULT_GROK_CODING_MAX_TURNS,
               capsule_selection: str = DEFAULT_CAPSULE_SELECTION,
               capsule_budget: str = DEFAULT_CAPSULE_BUDGET,
               source_view_profile: str = DEFAULT_SOURCE_VIEW_PROFILE,
               source_view_policy: dict | None = None,
               source_edit_mode: str = DEFAULT_SOURCE_EDIT_MODE,
               semantic_context_profile: str = DEFAULT_SEMANTIC_CONTEXT_PROFILE,
               numerical_formalization_profile: str | None = None,
               planning_mode: str = DEFAULT_PLANNING_MODE,
               prompt_tokenizer_artifact: str | Path | None = None,
               prompt_tokenizer_sha256: str | None = None,
               max_input_tokens: int | None = None,
               semantic_slot_policy_revision: str = "legacy@1",
               semantic_context_projection: bool = False,
               semantic_native_input: bool = False,
               semantic_decision_policy_artifact: str | Path | None = None,
               semantic_decision_policy_sha256: str | None = None,
               semantic_decision_nomination_artifact: str | Path | None = None,
               semantic_decision_nomination_source: str | Path | None = None,
               semantic_decision_nomination_sha256: str | None = None,
               disable_semantic_typed_nomination: bool = False,
               semantic_retrieval_query_reference: bool = False,
               semantic_decision_max_calls: int = 3,
               semantic_decision_max_total_input_tokens: int | None = None,
               doctor_dispatch_policy: str = "auto",
               model_revision: str = "") -> dict:
    _decision_options(locals(), arm=arm,
        provider=resolve_provider_profile(provider_profile)["provider"])
    doctor_arguments = doctor_dispatch_arguments(arm=arm, doctor_dispatch_policy=doctor_dispatch_policy)
    options = _prompt_options(arm, prompt_tokenizer_artifact=prompt_tokenizer_artifact,
        prompt_tokenizer_sha256=prompt_tokenizer_sha256, max_input_tokens=max_input_tokens,
        semantic_slot_policy_revision=semantic_slot_policy_revision,
        semantic_context_projection=semantic_context_projection,
        semantic_native_input=semantic_native_input,
        semantic_decision_policy_artifact=semantic_decision_policy_artifact,
        semantic_decision_policy_sha256=semantic_decision_policy_sha256,
        semantic_decision_nomination_artifact=semantic_decision_nomination_artifact,
        semantic_decision_nomination_source=semantic_decision_nomination_source,
        semantic_decision_nomination_sha256=semantic_decision_nomination_sha256,
        disable_semantic_typed_nomination=disable_semantic_typed_nomination,
        **({"semantic_retrieval_query_reference": semantic_retrieval_query_reference}
           if semantic_retrieval_query_reference is not False else {}),
        semantic_decision_max_calls=semantic_decision_max_calls,
        semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
    validate_legacy_semantic_route(policyoptions=options, rendereroptions={
        "semantic_transport_schema": semantic_transport_schema,
        "semantic_metadata_view": semantic_metadata_view,
        "semantic_working_set": semantic_working_set,
        "source_view_profile": source_view_profile,
        "source_edit_mode": source_edit_mode,
        "source_view_policy": source_view_policy,
        "coding_reply_mode": coding_reply_mode,
        "tool_output_policy": tool_output_policy,
        "source_view_advisory_policy": source_view_advisory_policy,
        "repository_source_policy": repository_source_policy,
        "source_input_policy": source_input_policy,
        "query_reference_policy": query_reference_policy,
        "semantic_context_profile": semantic_context_profile,
        "capsule_selection": capsule_selection,
        "capsule_budget": capsule_budget,
        "provider": resolve_provider_profile(provider_profile)["provider"],
    })
    validate_planning_mode(planning_mode, intent_requirement_contract=intent_requirement_contract,
        task_profile=task_profile)
    validate_semantic_context_profile(semantic_context_profile, arm=arm)
    validate_numerical_profile(numerical_formalization_profile)
    validate_capsule_selection(capsule_selection, arm=arm)
    validate_capsule_budget(capsule_budget, arm=arm)
    if capsule_budget != DEFAULT_CAPSULE_BUDGET:
        from .terminal_task_profile import MULTITASK_SCHEMA
        if isinstance(task_profile, dict) and task_profile.get("schema") == MULTITASK_SCHEMA:
            raise ValueError("bounded capsules support one independent signed task")
    validate_source_view_profile(source_view_profile, arm=arm)
    validate_source_edit_mode(source_edit_mode, arm=arm, source_view_profile=source_view_profile)
    validate_semantic_working_set(semantic_working_set, arm=arm, source_view_profile=source_view_profile,
        transport_schema=semantic_transport_schema, metadata_view=semantic_metadata_view)
    source_view_policy = validate_source_view_policy(source_view_policy, profile=source_view_profile, arm=arm)
    from .terminal_source_view_policy import validate_source_context_combination
    validate_source_context_combination(source_view_profile=source_view_profile,
        source_view_advisory_policy=source_view_advisory_policy, repository_source_policy=repository_source_policy,
        source_input_policy=source_input_policy, query_reference_policy=query_reference_policy,
        capsule_selection=capsule_selection, semantic_context_profile=semantic_context_profile)
    model_revision = validate_model_revision(model_revision)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_grok_coding_max_turns(grok_coding_max_turns, provider=selected_provider["provider"])
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
    validate_tool_output_policy(tool_output_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_view_advisory_policy(source_view_advisory_policy, arm=arm, provider=selected_provider["provider"])
    validate_repository_source_policy(repository_source_policy, arm=arm, provider=selected_provider["provider"],
        source_view_advisory_policy=source_view_advisory_policy)
    validate_source_input_policy(source_input_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view)
    validate_query_reference_policy(query_reference_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view, source_input_policy=source_input_policy)
    if arm not in {"full", "no-index"}:
        raise ValueError("unknown supervisor ablation")
    validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
    validate_semantic_metadata_view(semantic_metadata_view, arm=arm,
        transport_schema=semantic_transport_schema, provider=selected_provider["provider"])
    from .benchmark_resource_profile import execution_budget
    budget = execution_budget(resource_profile)
    config = baseline_config(dataset, output, resource_profile=resource_profile, task_name=task_name)
    config["job_name"] = "supervisor-" + arm + "-" + task_name
    config["agents"] = [{
        "import_path": ADAPTER, "model_name": selected_provider["model"],
        "override_timeout_sec": float(budget["harbor_seconds"]), "max_timeout_sec": float(budget["harbor_seconds"]),
        "override_setup_timeout_sec": 1800.0,
        "kwargs": {"runtime_archive": str(archive), "arm": arm,
                   "model_revision": model_revision if arm == "full" else "",
                   **semantic_transport_selection(semantic_transport_schema),
                   **coding_reply_selection(coding_reply_mode),
                   **tool_output_selection(tool_output_policy),
                   **source_view_advisory_selection(source_view_advisory_policy),
                   **repository_source_selection(repository_source_policy),
                   **source_input_selection(source_input_policy),
                   **query_reference_selection(query_reference_policy),
                   **numerical_selection(numerical_formalization_profile),
                   **planning_selection(planning_mode),
                   **semantic_metadata_selection(semantic_metadata_view),
                   **semantic_working_set_selection(semantic_working_set),
                   **grok_coding_selection(grok_coding_max_turns),
                   **capsule_selection_options(capsule_selection),
                   **capsule_budget_options(capsule_budget),
                   **source_view_selection(source_view_profile, source_view_policy),
                   **source_edit_selection(source_edit_mode),
                   **semantic_context_selection(semantic_context_profile)},
    }]
    if provider_profile is not None:
        config["agents"][0]["kwargs"]["provider_profile"] = selected_provider["id"]
    if intent_requirement_contract is not None:
        config["agents"][0]["kwargs"]["intent_requirement_contract"] = _transport_intent_contract(intent_requirement_contract)
    if task_profile is not None:
        config["agents"][0]["kwargs"]["task_profile"] = _transport_task_profile(task_profile)
    if resource_profile is not None:
        config["agents"][0]["kwargs"]["resource_profile"] = resource_profile
    if setup_cache_selection is not None:
        if selected_provider["provider"] != "codex_cli":
            raise ValueError("Codex setup cache is incompatible with Grok")
        from .terminal_setup_cache_advice import _selection_shape
        _selection_shape(setup_cache_selection)
        config["agents"][0]["kwargs"]["setup_cache_selection"] = json.loads(json.dumps(setup_cache_selection))
    defaults = _prompt_options(arm)
    config["agents"][0]["kwargs"].update({
        key: value for key, value in options.items() if key not in defaults or value != defaults[key]
    })
    if doctor_arguments:
        config["agents"][0]["kwargs"]["doctor_dispatch_policy"] = doctor_dispatch_policy
    return config


def validate_header_planning_selection(binding, contract, arm):
    """Bind explicit reviewed intent to its portable runtime profile before setup."""
    profile = None if binding is None else binding["config"].get("header_applicability")
    header = isinstance(contract, dict) and contract.get("schema") == "intent-plan-requirement-contract@3"
    if profile is None and not header:
        return
    if arm != "full" or profile is None or not header:
        raise ValueError("header applicability requires the full arm, reviewed intent and selected Source384 profile")
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import validate_intent_requirement_contract
    from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import validate_runtime_profile
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
    selected = validate_intent_requirement_contract(contract)["source_applicability"]
    profile = validate_runtime_profile(profile)
    if (binding["config"]["schema"] != "terminal-source384-config@2"
            or profile["selector_cid"] != cid_for_dag_json(selected)
            or profile["checker_profile"] != selected["checker_profile"]):
        raise ValueError("header runtime profile differs from reviewed intent selector")


def prepare(*, dataset: Path, output: Path, archive: Path, arm: str,
            intent_requirement_contract: Path | None = None,
            intent_action_384_config: Path | None = None,
            source384_config: Path | None = None, resource_profile=None,
            setup_cache_policy: str | None = None, task_name: str = TASK,
            task_profile: Path | None = None, provider_profile: str | None = None,
            semantic_transport_schema: str = DEFAULT_SEMANTIC_TRANSPORT_SCHEMA,
            tool_output_policy: str = DEFAULT_TOOL_OUTPUT_POLICY,
            source_view_advisory_policy: str = DEFAULT_SOURCE_VIEW_POLICY,
            repository_source_policy: str = DEFAULT_REPOSITORY_SOURCE_POLICY,
            source_input_policy: str = DEFAULT_SOURCE_INPUT_POLICY,
            query_reference_policy: str = DEFAULT_QUERY_REFERENCE_POLICY,
            coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
            semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
            semantic_working_set: str = DEFAULT_SEMANTIC_WORKING_SET,
            grok_coding_max_turns: int = DEFAULT_GROK_CODING_MAX_TURNS,
            capsule_selection: str = DEFAULT_CAPSULE_SELECTION,
            capsule_budget: str = DEFAULT_CAPSULE_BUDGET,
            source_view_profile: str = DEFAULT_SOURCE_VIEW_PROFILE,
            source_view_policy: dict | None = None,
            source_edit_mode: str = DEFAULT_SOURCE_EDIT_MODE,
            semantic_context_profile: str = DEFAULT_SEMANTIC_CONTEXT_PROFILE,
            numerical_formalization_profile: str | None = None,
            prompt_tokenizer_artifact: str | Path | None = None,
            prompt_tokenizer_sha256: str | None = None,
            max_input_tokens: int | None = None,
            semantic_slot_policy_revision: str = "legacy@1",
            semantic_context_projection: bool = False,
            semantic_native_input: bool = False,
            semantic_decision_policy_artifact: str | Path | None = None,
            semantic_decision_policy_sha256: str | None = None,
            semantic_decision_nomination_artifact: str | Path | None = None,
            semantic_decision_nomination_source: str | Path | None = None,
            semantic_decision_nomination_sha256: str | None = None,
            disable_semantic_typed_nomination: bool = False,
            semantic_retrieval_query_reference: bool = False,
            semantic_decision_max_calls: int = 3,
            semantic_decision_max_total_input_tokens: int | None = None,
            doctor_dispatch_policy: str = "auto",
            planning_mode: str = DEFAULT_PLANNING_MODE) -> dict:
    _decision_options(locals(), arm=arm,
        provider=resolve_provider_profile(provider_profile)["provider"])
    doctor_arguments = doctor_dispatch_arguments(arm=arm, doctor_dispatch_policy=doctor_dispatch_policy)
    options = _prompt_options(arm, prompt_tokenizer_artifact=prompt_tokenizer_artifact,
        prompt_tokenizer_sha256=prompt_tokenizer_sha256, max_input_tokens=max_input_tokens,
        semantic_slot_policy_revision=semantic_slot_policy_revision,
        semantic_context_projection=semantic_context_projection,
        semantic_native_input=semantic_native_input,
        semantic_decision_policy_artifact=semantic_decision_policy_artifact,
        semantic_decision_policy_sha256=semantic_decision_policy_sha256,
        semantic_decision_nomination_artifact=semantic_decision_nomination_artifact,
        semantic_decision_nomination_source=semantic_decision_nomination_source,
        semantic_decision_nomination_sha256=semantic_decision_nomination_sha256,
        disable_semantic_typed_nomination=disable_semantic_typed_nomination,
        **({"semantic_retrieval_query_reference": semantic_retrieval_query_reference}
           if semantic_retrieval_query_reference is not False else {}),
        semantic_decision_max_calls=semantic_decision_max_calls,
        semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
    validate_legacy_semantic_route(policyoptions=options, rendereroptions={
        "semantic_transport_schema": semantic_transport_schema,
        "semantic_metadata_view": semantic_metadata_view,
        "semantic_working_set": semantic_working_set,
        "source_view_profile": source_view_profile,
        "source_edit_mode": source_edit_mode,
        "source_view_policy": source_view_policy,
        "coding_reply_mode": coding_reply_mode,
        "tool_output_policy": tool_output_policy,
        "source_view_advisory_policy": source_view_advisory_policy,
        "repository_source_policy": repository_source_policy,
        "source_input_policy": source_input_policy,
        "query_reference_policy": query_reference_policy,
        "semantic_context_profile": semantic_context_profile,
        "capsule_selection": capsule_selection,
        "capsule_budget": capsule_budget,
        "provider": resolve_provider_profile(provider_profile)["provider"],
    })
    if semantic_decision_nomination_source is not None:
        read_nomination_source(source=semantic_decision_nomination_source,
                               expected_sha256=semantic_decision_nomination_sha256)
    validate_planning_mode(planning_mode, intent_requirement_contract=intent_requirement_contract)
    validate_semantic_context_profile(semantic_context_profile, arm=arm)
    validate_numerical_profile(numerical_formalization_profile)
    validate_capsule_selection(capsule_selection, arm=arm)
    validate_capsule_budget(capsule_budget, arm=arm)
    validate_source_view_profile(source_view_profile, arm=arm)
    validate_source_edit_mode(source_edit_mode, arm=arm, source_view_profile=source_view_profile)
    validate_semantic_working_set(semantic_working_set, arm=arm, source_view_profile=source_view_profile,
        transport_schema=semantic_transport_schema, metadata_view=semantic_metadata_view)
    source_view_policy = validate_source_view_policy(source_view_policy, profile=source_view_profile, arm=arm)
    from .terminal_source_view_policy import validate_source_context_combination
    validate_source_context_combination(source_view_profile=source_view_profile,
        source_view_advisory_policy=source_view_advisory_policy, repository_source_policy=repository_source_policy,
        source_input_policy=source_input_policy, query_reference_policy=query_reference_policy,
        capsule_selection=capsule_selection, semantic_context_profile=semantic_context_profile)
    validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_grok_coding_max_turns(grok_coding_max_turns, provider=selected_provider["provider"])
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
    validate_tool_output_policy(tool_output_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_view_advisory_policy(source_view_advisory_policy, arm=arm, provider=selected_provider["provider"])
    validate_repository_source_policy(repository_source_policy, arm=arm, provider=selected_provider["provider"],
        source_view_advisory_policy=source_view_advisory_policy)
    validate_source_input_policy(source_input_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view)
    validate_query_reference_policy(query_reference_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view, source_input_policy=source_input_policy)
    validate_semantic_metadata_view(semantic_metadata_view, arm=arm,
        transport_schema=semantic_transport_schema, provider=selected_provider["provider"])
    from harbor.models.job.config import JobConfig
    dataset = dataset.resolve(strict=True)
    task = _task_path(dataset, task_name)
    selected_task_profile = _load_task_profile(task_profile, task=task)
    if capsule_budget != DEFAULT_CAPSULE_BUDGET:
        from .terminal_task_profile import MULTITASK_SCHEMA
        if isinstance(selected_task_profile, dict) and selected_task_profile.get("schema") == MULTITASK_SCHEMA:
            raise ValueError("bounded capsules support one independent signed task")
    validate_planning_mode(planning_mode, task_profile=selected_task_profile)
    archive = archive.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh trial output required")
    with (archive / "manifest.json").open("rb") as stream:
        manifest_bytes = stream.read(MAX_RUNTIME_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > MAX_RUNTIME_MANIFEST_BYTES:
        raise ValueError("runtime manifest exceeds its bound; rebuild the runtime archive")
    manifest = json.loads(manifest_bytes)
    if _hash(archive / "runtime.tar.gz") != manifest["archive_sha256"]:
        raise ValueError("runtime archive integrity failed")
    selected_sources = (_decision_runtime_sources(**({"retrieval_query_reference": True}
                         if semantic_retrieval_query_reference else {}))
                        if semantic_native_input or any(key in options for key in ("semantic_decision_policy_artifact", "semantic_decision_nomination_artifact", "semantic_decision_nomination_source"))
                        else None)
    if selected_sources is None:
        runtime_timeout_sources = _require_current_timeout_runtime(archive, manifest)
        runtime_decision_sources = None
    else:
        verified_sources = _require_current_timeout_runtime(archive, manifest, selected_sources=selected_sources)
        runtime_timeout_sources = {name: digest for name, digest in verified_sources.items()
                                   if name not in selected_sources}
        runtime_decision_sources = {name: verified_sources[name] for name in selected_sources}
    model_revision = selected_retrieval_revision(manifest, arm)
    from .terminal_worker_capability import require_worker_capability
    require_worker_capability(manifest, resource_profile)
    require_runtime_provider_profile(manifest, provider_profile)
    require_semantic_transport_archive(manifest, semantic_transport_schema)
    require_coding_reply_archive(manifest, coding_reply_mode)
    require_tool_output_archive(manifest, tool_output_policy)
    require_source_view_advisory_archive(manifest, source_view_advisory_policy)
    require_repository_source_archive(manifest, repository_source_policy, archive=archive / "runtime.tar.gz")
    require_source_input_archive(manifest, source_input_policy)
    require_query_reference_archive(manifest, query_reference_policy)
    require_semantic_metadata_archive(manifest, semantic_metadata_view)
    require_semantic_working_set_archive(manifest, semantic_working_set)
    require_grok_coding_archive(manifest, grok_coding_max_turns)
    require_capsule_selection_archive(manifest, capsule_selection)
    require_capsule_budget_archive(manifest, capsule_budget)
    require_source_view_archive(manifest, source_view_profile, resource_profile)
    require_source_edit_archive(manifest, source_edit_mode)
    require_semantic_context_archive(manifest, semantic_context_profile)
    require_numerical_archive(manifest, numerical_formalization_profile)
    require_planning_archive(manifest, planning_mode)
    if selected_provider["provider"] == "grok_cli":
        from .terminal_grok_deployment import verify_grok_archive
        verify_grok_archive(archive / "runtime.tar.gz", manifest)
        if setup_cache_policy is not None:
            raise ValueError("Codex setup cache is incompatible with Grok")
    from .terminal_deployment import (
        load_intent_action_384_config, _intent_action_384_assets,
        validate_intent_action_384_binding, verify_intent_action_384_archive,
        load_source384_config, _source384_assets, validate_source384_binding, verify_source384_archive,
    )
    selected_intent = validate_intent_action_384_binding(manifest)
    verify_intent_action_384_archive(archive / "runtime.tar.gz", manifest)
    selected_source384 = validate_source384_binding(manifest)
    verify_source384_archive(archive / "runtime.tar.gz", manifest)
    from .terminal_numerical_selection_deployment import verify_numerical_selection_archive
    selected_numerical = verify_numerical_selection_archive(archive / "runtime.tar.gz", manifest)
    _decision_options({**locals(), "source384_config": selected_source384,
        "numerical_selection_config": selected_numerical}, arm=arm,
        provider=selected_provider["provider"])
    if setup_cache_policy is not None and manifest.get("numerical_selection_384") is not None:
        raise ValueError("setup cache is not qualified for learned numerical selection")
    if source384_config is not None:
        _, expected_binding = _source384_assets(load_source384_config(source384_config))
        if selected_source384 != expected_binding:
            raise ValueError("selected Source384 config differs from the runtime archive")
    host_config = None
    if intent_action_384_config is not None:
        host_config = load_intent_action_384_config(intent_action_384_config)
        _, expected_binding = _intent_action_384_assets(host_config)
        if selected_intent != expected_binding:
            raise ValueError("selected Intent384 config differs from the runtime archive")
    from .terminal_intent_action_readiness import intent_action_readiness
    task_hashes = _task_hashes(task)
    with (task / "instruction.md").open("rb") as stream:
        public_instruction = stream.read(MAX_PUBLIC_INSTRUCTION_BYTES + 1)
    if len(public_instruction) > MAX_PUBLIC_INSTRUCTION_BYTES:
        raise ValueError("public instruction exceeds its byte bound")
    intent_readiness = intent_action_readiness(
        public_instruction.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"),
        manifest=manifest, config=host_config)
    from .terminal_setup_cache_advice import select_setup_cache, validate_setup_cache_prerequisites
    setup_cache_selection = select_setup_cache(archive, setup_cache_policy)
    validate_setup_cache_prerequisites(setup_cache_selection, install_codex=True,
        auth_json=Path.home() / (".codex/auth.json" if selected_provider["provider"] == "codex_cli" else ".grok/auth.json"), arm=arm, resource_profile=resource_profile)
    requirements = _load_intent_requirement_contract(intent_requirement_contract, dataset=dataset, task_name=task_name)
    validate_header_planning_selection(selected_source384, requirements, arm)
    declared_config = config_for(dataset, output, archive, arm,
        model_revision=model_revision, doctor_dispatch_policy=doctor_dispatch_policy, **options,
        intent_requirement_contract=requirements, resource_profile=resource_profile,
        setup_cache_selection=setup_cache_selection, task_name=task_name, task_profile=selected_task_profile,
        semantic_transport_schema=semantic_transport_schema,
        coding_reply_mode=coding_reply_mode,
        tool_output_policy=tool_output_policy,
        source_view_advisory_policy=source_view_advisory_policy,
        repository_source_policy=repository_source_policy,
        source_input_policy=source_input_policy,
        query_reference_policy=query_reference_policy,
        semantic_metadata_view=semantic_metadata_view,
        **semantic_working_set_selection(semantic_working_set),
        **grok_coding_selection(grok_coding_max_turns),
        capsule_selection=capsule_selection,
        capsule_budget=capsule_budget,
        source_view_profile=source_view_profile,
        source_edit_mode=source_edit_mode,
        source_view_policy=source_view_policy,
        semantic_context_profile=semantic_context_profile,
        numerical_formalization_profile=numerical_formalization_profile,
        planning_mode=planning_mode,
        **({"provider_profile": provider_profile} if provider_profile is not None else {}))
    if (selected_source384 is not None or selected_numerical is not None) and arm == "full":
        validate_resource_profile(declared_config, resource_profile)
    config = JobConfig.model_validate(declared_config, extra="forbid")
    output.mkdir(parents=True)
    nomination_snapshot = None
    if semantic_decision_nomination_source is not None:
        nomination_snapshot = capture_nomination_snapshot(
            source=semantic_decision_nomination_source,
            expected_sha256=semantic_decision_nomination_sha256, directory=output)
        options["semantic_decision_nomination_source"] = nomination_snapshot["artifact"]
        declared_config["agents"][0]["kwargs"]["semantic_decision_nomination_source"] = nomination_snapshot["artifact"]
        config = JobConfig.model_validate(declared_config, extra="forbid")
    wire = benchmark_controls.canonicalize_serialized_retry_sets(
        config.model_dump(mode="json", context={"redact_sensitive_env": False}))
    _json(output / "config.json", wire)
    harbor = Path(sys.executable).with_name("harbor")
    command = [str(harbor), "run", "--config", str(output / "config.json"), "--yes"]
    if nomination_snapshot is not None:
        verify_nomination_snapshot(nomination_snapshot, directory=output)
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, timeout=60)
    (output / "dry-run.stdout").write_text(dry.stdout)
    (output / "dry-run.stderr").write_text(dry.stderr)
    if _task_hashes(task) != task_hashes:
        raise ValueError("original task changed during supervisor preflight")
    sources = {str(path): _hash(path) for path in Path(__file__).parent.glob("*.py")}
    if any(sources.get(str(Path(__file__).with_name(name))) != runtime_timeout_sources[
            TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name] for name in TIMEOUT_RUNTIME_SOURCE_NAMES):
        raise ValueError("runtime deadline sources changed during preflight; rebuild the runtime archive")
    if runtime_decision_sources is not None:
        _require_current_decision_runtime_pins(runtime_decision_sources, native_input=semantic_native_input,
            **({"retrieval_query_reference": True} if semantic_retrieval_query_reference else {}))
    result = {"schema": "terminal-full-supervisor-preparation@1", "prepared": dry.returncode == 0,
              "arm": arm, "task": task_name, "dataset": str(dataset), "archive": str(archive),
              "archive_sha256": manifest["archive_sha256"], "manifest_sha256": _hash(archive / "manifest.json"),
              "host_source_sha256": sources,
              "runtime_timeout_source_sha256": runtime_timeout_sources,
              "prompt_comparison_options": options,
              "doctor_dispatch_policy": doctor_dispatch_policy,
              "task_input_sha256": task_hashes, "config_sha256": _hash(output / "config.json"),
              "comparison_controls": benchmark_controls.build_controls(
                  json.loads((output / "config.json").read_text()),
                  task_input_sha256=task_hashes, task=task_name, model=selected_provider["model"],
                  reasoning_effort=selected_provider["reasoning_effort"], cli_version=selected_provider["cli_version"]),
              "command": command, **{key: selected_provider[key] for key in ("model", "reasoning_effort", "cli_version")},
              **({"provider_profile": selected_provider["id"]} if provider_profile is not None else {}),
              **semantic_transport_selection(semantic_transport_schema),
              **coding_reply_selection(coding_reply_mode),
              **tool_output_selection(tool_output_policy),
              **source_view_advisory_selection(source_view_advisory_policy),
              **repository_source_selection(repository_source_policy),
              **source_input_selection(source_input_policy),
              **query_reference_selection(query_reference_policy),
              **semantic_metadata_observation(semantic_metadata_view),
              **semantic_working_set_observation(semantic_working_set),
              **grok_coding_selection(grok_coding_max_turns),
              **capsule_selection_observation(capsule_selection),
              **capsule_budget_observation(capsule_budget),
              **source_view_observation(source_view_profile, source_view_policy),
              **source_edit_observation(source_edit_mode),
              **semantic_context_selection(semantic_context_profile),
              **numerical_observation(numerical_formalization_profile, arm),
              **planning_observation(planning_mode),
              "agent_timeout_seconds": declared_config["agents"][0]["override_timeout_sec"], "provider_calls": 0,
              "planning_and_cold_index_charged_to_agent_time": True,
              "benchmark_advantage_claimed": False, **_intent_selection(requirements, planning_mode=planning_mode),
              "intent_action_384": selected_intent, "source384": selected_source384,
              "intent_action_384_readiness": intent_readiness,
              "source384_enabled": selected_source384 is not None and arm == "full",
              **({"numerical_selection_384": selected_numerical,
                  "numerical_selection_enabled": arm == "full"} if selected_numerical is not None else {}),
              "resource_profile": resource_profile,
              **({"setup_cache_selection": setup_cache_selection} if setup_cache_selection is not None else {})}
    if runtime_decision_sources is not None:
        pin_field = "runtime_native_input_source_sha256" if semantic_native_input else "runtime_decision_source_sha256"
        result[pin_field] = runtime_decision_sources
    if semantic_native_input:
        result["semantic_native_input_requested"] = True
    if semantic_retrieval_query_reference:
        result["semantic_retrieval_query_reference_requested"] = True
    if nomination_snapshot is not None:
        result["semantic_decision_nomination_snapshot"] = nomination_snapshot
    _json(output / "preparation.json", result)
    if dry.returncode:
        raise RuntimeError("Harbor preflight failed; see retained logs")
    return result


def _metadata_dispatch_observation(report, requested_mode: str) -> dict:
    """Join retained selection receipts to the separate exact-input audit.

    This read-only observation does not rerun the source audit or measure usage.
    A valid legacy fallback remains a completed requested-arm attempt; only
    verified common-bindings inputs count as an effective metadata treatment.
    """
    validate_semantic_metadata_view(requested_mode)
    report = report if type(report) is dict else {}
    declaration = report.get("semantic_metadata_view")
    declared = declaration.get("requested_mode") if type(declaration) is dict else None
    declared_matches = (declared == requested_mode if requested_mode != "legacy"
                        else declaration is None or declared == "legacy")
    problems = [] if declared_matches else ["requested_mode_report_mismatch"]
    if report.get("unreceipted_provider_attempt"):
        problems.append("unreceipted_provider_attempt")
    population = report.get("provider_invocations")
    if type(population) is not list or len(population) > 4096:
        problems.append("provider_invocation_population_unavailable_or_unbounded")
        population = []
    if any(type(row) is not dict for row in population):
        problems.append("malformed_provider_invocation")
    coding = [row for row in population if type(row) is dict and row.get("phase") == "coding"]
    audit = report.get("context_input_audit")
    audit = audit if type(audit) is dict else {}
    audit_rows, matches = audit.get("invocations"), audit.get("matches")
    audit_rows = audit_rows if type(audit_rows) is list and len(audit_rows) <= 16 else []
    matches = matches if type(matches) is list and len(matches) <= 512 else []
    identities = [row.get("invocation_id") for row in coding]
    valid_id = lambda value: type(value) is str and 0 < len(value) <= 4096
    valid_hash = lambda value: type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None
    valid_bytes = lambda value: type(value) is int and 0 < value <= 2_000_000
    distinct = all(valid_id(value) for value in identities) and len(set(identities)) == len(identities)
    audit_ids = [row.get("invocation_id") for row in audit_rows if type(row) is dict]
    audit_population = (distinct and len(audit_rows) == len(coding) == len(audit_ids)
        and all(valid_id(value) for value in audit_ids) and set(audit_ids) == set(identities)
        and len(set(audit_ids)) == len(audit_ids)
        and audit.get("status") == "verified_all_observed_model_inputs"
        and audit.get("all_observed_coding_inputs_verified") is True
        and type(audit.get("coding_receipts_seen")) is int and audit["coding_receipts_seen"] == len(coding)
        and type(audit.get("usable_coding_receipts")) is int and audit["usable_coding_receipts"] == len(coding)
        and audit.get("read_errors") == [])
    if not coding:
        problems.append("no_observed_coding_receipts")
    elif not distinct:
        problems.append("coding_invocation_identities_missing_or_duplicate")
    observations = []
    for row in coding:
        identity = row.get("invocation_id")
        selection = row.get("semantic_metadata_view")
        selection = selection if type(selection) is dict else {}
        mode, fallback = selection.get("selected_mode"), selection.get("fallback_reason")
        valid_selection = (selection.get("schema") == "supervisor-semantic-metadata-selection@1"
            and type(mode) is str and mode in {"legacy", "common-bindings@1"}
            and selection.get("candidate_only") is True
            and selection.get("token_proxy") == "utf8-bytes-ceil-div4@1"
            and all(selection.get(key) is False for key in (
                "source_freshness_verified", "program_semantics_proved", "omission_authority",
                "proof_authority", "execution_authority", "completion_authority", "publication_authority"))
            and all(valid_hash(selection.get(key)) for key in (
                "view_receipt_sha256", "native_complete_sha256", "candidate_complete_sha256", "selected_complete_sha256"))
            and all(valid_bytes(selection.get(key)) for key in (
                "native_complete_bytes", "candidate_complete_bytes", "selected_complete_bytes"))
            and all(type(selection.get(prefix + "_complete_proxy_tokens")) is int
                and selection[prefix + "_complete_proxy_tokens"] == (selection[prefix + "_complete_bytes"] + 3) // 4
                for prefix in ("native", "candidate", "selected")))
        if valid_selection:
            smaller = selection["candidate_complete_bytes"] < selection["native_complete_bytes"] and (
                (selection["candidate_complete_bytes"] + 3) // 4 < (selection["native_complete_bytes"] + 3) // 4)
            prefix = "candidate" if mode == "common-bindings@1" else "native"
            valid_selection = (smaller == (mode == "common-bindings@1")
                and fallback == (None if smaller else "complete_input_not_smaller_under_bytes_and_proxy")
                and selection["selected_complete_sha256"] == selection[prefix + "_complete_sha256"]
                and selection["selected_complete_bytes"] == selection[prefix + "_complete_bytes"])
        bound = (valid_selection and valid_hash(row.get("model_prompt_sha256"))
            and valid_bytes(row.get("model_prompt_bytes"))
            and selection["selected_complete_sha256"] == row["model_prompt_sha256"]
            and selection["selected_complete_bytes"] == row["model_prompt_bytes"])
        per_audit = [item for item in audit_rows if type(item) is dict and item.get("invocation_id") == identity]
        per_matches = [item for item in matches if type(item) is dict and item.get("invocation_id") == identity]
        def verified_match(item):
            checks = item.get("model_input_checks")
            return (item.get("model_input_verified") is True
                and item.get("model_prompt_sha256") == row.get("model_prompt_sha256")
                and type(item.get("model_prompt_bytes")) is int
                and item.get("model_prompt_bytes") == row.get("model_prompt_bytes")
                and type(checks) is dict and checks.get("semantic_metadata_selection_receipt") is True
                and all(value is True for value in checks.values()))
        verified = (bound and audit_population and len(per_audit) == 1
            and per_audit[0].get("model_input_verified") is True
            and per_audit[0].get("status") == "verified_exact_model_input"
            and bool(per_matches) and all(verified_match(item) for item in per_matches))
        row_problems = []
        if not valid_selection: row_problems.append("metadata_selection_missing_or_invalid")
        elif not bound: row_problems.append("selected_input_binding_mismatch")
        if not verified: row_problems.append("retained_exact_input_audit_not_correlated")
        observations.append({"invocation_id": identity if valid_id(identity) else None,
            "selected_mode": mode if type(mode) is str and mode in {"legacy", "common-bindings@1"} else None,
            "fallback_reason": fallback if fallback in (None, "complete_input_not_smaller_under_bytes_and_proxy") else "unrecognized",
            "selection_receipt_matches_model_input": bool(bound),
            "retained_exact_input_audit_verified": bool(verified), "problems": row_problems})
    modes = sorted({row["selected_mode"] for row in observations if row["selected_mode"] is not None})
    bound = bool(observations) and all(row["selection_receipt_matches_model_input"] for row in observations)
    verified = (not problems and bound
        and all(row["retained_exact_input_audit_verified"] for row in observations))
    status = ("verified_treatment" if modes == ["common-bindings@1"] else
              "verified_fallback" if modes == ["legacy"] else "verified_mixed_selection") if verified else "unverified"
    return {"schema": "terminal-semantic-metadata-dispatch-observation@1", "requested_mode": requested_mode,
        "requested_mode_matches_report": declared_matches, "coding_receipts_seen": len(coding),
        "invocations": observations, "selected_modes": modes,
        "fallback_reasons": sorted({row["fallback_reason"] for row in observations if row["fallback_reason"] is not None}),
        "all_selection_receipts_bound_to_model_inputs": bound,
        "all_coding_inputs_verified_by_retained_audit": verified,
        "effective_treatment_observed": verified and modes == ["common-bindings@1"],
        "status": status, "problems": problems, "source_audit_rerun": False,
        "provider_usage_modified": False, "token_savings_established": False}


def collect(output: Path, *, task_name: str | None = None) -> dict:
    from .terminal_provider_stop import provider_stop_observation
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, config, job_prefix="supervisor-" + prepared["arm"] + "-",
                          task_name=task_name)
    profile = prepared_provider_identity(prepared, config)
    prompt_observation = _prompt_option_observation(prepared, config["agents"][0].get("kwargs"))
    decision = _decision_options(config["agents"][0].get("kwargs", {}), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    held_options = prepared.get("prompt_comparison_options")
    held_decision = _decision_options(held_options if type(held_options) is dict else {},
        arm=prepared["arm"], provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    if (decision != held_decision or decision and (
            prompt_observation["prompt_comparison_options_valid"] is not True
            or prompt_observation["prompt_comparison_config_unchanged"] is not True)):
        raise ValueError("prepared semantic decision selection changed")
    doctor_observation = _doctor_policy_observation(prepared, config["agents"][0].get("kwargs"))
    planning_mode = validate_planning_mode(prepared.get("planning_mode", DEFAULT_PLANNING_MODE),
        intent_requirement_contract=config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        task_profile=config["agents"][0].get("kwargs", {}).get("task_profile"))
    configured_planning = validate_planning_mode(
        config["agents"][0].get("kwargs", {}).get("planning_mode", DEFAULT_PLANNING_MODE),
        intent_requirement_contract=config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        task_profile=config["agents"][0].get("kwargs", {}).get("task_profile"))
    numerical_profile = validate_numerical_profile(prepared.get("numerical_formalization_profile"))
    configured_numerical = validate_numerical_profile(
        config["agents"][0].get("kwargs", {}).get("numerical_formalization_profile"))
    reply_mode = validate_coding_reply_mode(prepared.get("coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
        arm=prepared["arm"], provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_reply_mode = validate_coding_reply_mode(
        config["agents"][0].get("kwargs", {}).get("coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
        arm=prepared["arm"])
    tool_output = validate_tool_output_policy(
        prepared.get("tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_tool_output = validate_tool_output_policy(
        config["agents"][0].get("kwargs", {}).get("tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY),
        arm=prepared["arm"])
    source_view_advisory = validate_source_view_advisory_policy(
        prepared.get("source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_source_view_advisory = validate_source_view_advisory_policy(
        config["agents"][0].get("kwargs", {}).get("source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY),
        arm=prepared["arm"])
    repository_source = validate_repository_source_policy(
        prepared.get("repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"], source_view_advisory_policy=source_view_advisory)
    configured_repository_source = validate_repository_source_policy(
        config["agents"][0].get("kwargs", {}).get("repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY),
        arm=prepared["arm"], provider=resolve_provider_profile(config["agents"][0].get("kwargs", {}).get(
            "provider_profile"))["provider"], source_view_advisory_policy=configured_source_view_advisory)
    source_input = validate_source_input_policy(
        prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    query_reference = validate_query_reference_policy(
        prepared.get("query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        source_input_policy=prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
    configured_source_input = validate_source_input_policy(
        config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    configured_query_reference = validate_query_reference_policy(
        config["agents"][0].get("kwargs", {}).get("query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        source_input_policy=config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
    transport = validate_semantic_transport_schema(
        prepared.get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=prepared["arm"])
    configured_transport = validate_semantic_transport_schema(
        config["agents"][0].get("kwargs", {}).get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
        arm=prepared["arm"])
    metadata_view = validate_semantic_metadata_view(
        prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=prepared["arm"],
        transport_schema=transport,
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_metadata_view = validate_semantic_metadata_view(
        config["agents"][0].get("kwargs", {}).get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        arm=prepared["arm"], transport_schema=configured_transport)
    capsule_selection = validate_capsule_selection(
        prepared.get("capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=prepared["arm"])
    configured_capsules = validate_capsule_selection(
        config["agents"][0].get("kwargs", {}).get("capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=prepared["arm"])
    capsule_budget = validate_capsule_budget(
        prepared.get("capsule_budget", DEFAULT_CAPSULE_BUDGET), arm=prepared["arm"])
    configured_capsule_budget = validate_capsule_budget(
        config["agents"][0].get("kwargs", {}).get("capsule_budget", DEFAULT_CAPSULE_BUDGET), arm=prepared["arm"])
    source_view = validate_source_view_profile(
        prepared.get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=prepared["arm"])
    configured_source_view = validate_source_view_profile(
        config["agents"][0].get("kwargs", {}).get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE),
        arm=prepared["arm"])
    from .terminal_source_view_policy import validate_source_context_combination
    for signed_profile, advisory, repository_policy, input_policy, reference_policy in (
            (source_view, source_view_advisory, repository_source, source_input, query_reference),
            (configured_source_view, configured_source_view_advisory, configured_repository_source,
             configured_source_input, configured_query_reference)):
        validate_source_context_combination(source_view_profile=signed_profile,
            source_view_advisory_policy=advisory, repository_source_policy=repository_policy,
            source_input_policy=input_policy, query_reference_policy=reference_policy)
    grok_turns = validate_grok_coding_max_turns(
        prepared.get("grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS),
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_grok_turns = validate_grok_coding_max_turns(
        config["agents"][0].get("kwargs", {}).get("grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS),
        provider=resolve_provider_profile(config["agents"][0].get("kwargs", {}).get("provider_profile"))["provider"])
    working_set = validate_semantic_working_set(
        prepared.get("semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET), arm=prepared["arm"],
        source_view_profile=source_view, transport_schema=transport, metadata_view=metadata_view)
    configured_working_set = validate_semantic_working_set(
        config["agents"][0].get("kwargs", {}).get("semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET),
        arm=prepared["arm"], source_view_profile=configured_source_view,
        transport_schema=config["agents"][0].get("kwargs", {}).get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
        metadata_view=configured_metadata_view)
    source_edit = validate_source_edit_mode(prepared.get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=source_view)
    configured_source_edit = validate_source_edit_mode(
        config["agents"][0].get("kwargs", {}).get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=configured_source_view)
    source_policy, source_policy_unchanged, source_policy_present = _source_view_policy_state(
        prepared, config, source_view=source_view, configured_source_view=configured_source_view,
        current_config_sha256=_hash(output / "config.json"))
    semantic_profile = validate_semantic_context_profile(
        prepared.get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE), arm=prepared["arm"])
    configured_semantic_profile = validate_semantic_context_profile(
        config["agents"][0].get("kwargs", {}).get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE),
        arm=prepared["arm"])
    selection = {key: prepared.get(key, value) for key, value in _intent_selection(None).items()}
    configured_selection = _intent_selection(config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        planning_mode=configured_planning)
    job = Path(config["jobs_dir"]) / config["job_name"]
    trials = []
    for path in sorted(job.glob("*/result.json")):
        native = json.loads(path.read_text())
        report_path = path.parent / "agent/supervisor-result.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        provider_stop = provider_stop_observation(report) if type(report) is dict else None
        durations = {}
        for field in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
            phase = native.get(field) or {}
            start, end = phase.get("started_at"), phase.get("finished_at")
            durations[field] = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None
        trials.append({"trial": path.parent.name, "native_result_sha256": _hash(path),
                       "task": native.get("task_name"), "exact_trial_task_matches": _trial_task_matches(native, task),
                       **_prompt_option_observation(prepared, ((native.get("config") or {}).get("agent") or {}).get("kwargs")),
                       **_doctor_policy_observation(prepared, ((native.get("config") or {}).get("agent") or {}).get("kwargs")),
                       "reward": (native.get("verifier_result") or {}).get("rewards"),
                       "exception_type": (native.get("exception_info") or {}).get("exception_type"),
                       "durations_seconds": durations, "agent_context": native.get("agent_result"),
                       "native_agent_execution_started": bool((native.get("agent_execution") or {}).get("started_at")),
                       "supervisor": report,
                       **({"semantic_decision_dispatch": _decision_dispatch_observation(report, decision)}
                          if decision else {}),
                       **({"provider_stop": provider_stop} if provider_stop is not None else {}),
                       **({"source_input_dispatch": source_input_dispatch_observation(report, source_input)}
                          if source_input != DEFAULT_SOURCE_INPUT_POLICY else {}),
                       **({"query_reference_dispatch": query_reference_dispatch_observation(report, query_reference)}
                          if query_reference != DEFAULT_QUERY_REFERENCE_POLICY else {}),
                       **({"repository_source_dispatch": repository_source_dispatch_observation(report, repository_source)}
                          if repository_source != DEFAULT_REPOSITORY_SOURCE_POLICY else {}),
                       **({"source_view_dispatch": source_view_dispatch_observation(report, source_view_advisory)}
                          if source_view_advisory != DEFAULT_SOURCE_VIEW_POLICY else {}),
                       **({"tool_output_dispatch": tool_output_dispatch_observation(report, tool_output)}
                          if tool_output != DEFAULT_TOOL_OUTPUT_POLICY else {}),
                       **({"semantic_metadata_dispatch": _metadata_dispatch_observation(report, metadata_view)}
                          if metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW else {}),
                       **({"semantic_context_profile_matches": isinstance(report, dict)
                           and report.get("semantic_context_profile") == semantic_profile}
                          if semantic_profile != DEFAULT_SEMANTIC_CONTEXT_PROFILE else {})})
    unchanged = _task_hashes(task) == prepared["task_input_sha256"]
    budget_runtime_matches = None
    if capsule_budget != DEFAULT_CAPSULE_BUDGET and trials:
        expected_budget = capsule_budget_observation(capsule_budget)
        reported_budgets = [trial["supervisor"] for trial in trials]
        if all(type(report) is dict for report in reported_budgets):
            budget_runtime_matches = all(
                all(type(report.get(key)) is type(value) and report.get(key) == value
                    for key, value in expected_budget.items())
                for report in reported_budgets)
    grok_turns_runtime_matches = None
    grok_turns_policy_matches = None
    grok_turns_not_applicable = None
    if grok_turns != DEFAULT_GROK_CODING_MAX_TURNS and trials:
        if all(type(trial["supervisor"]) is dict for trial in trials):
            grok_turns_runtime_matches = all(
                type(trial["supervisor"].get("grok_coding_max_turns")) is int
                and trial["supervisor"]["grok_coding_max_turns"] == grok_turns for trial in trials)
            policies = [grok_coding_runtime_matches(trial["supervisor"], grok_turns) for trial in trials]
            grok_turns_policy_matches = False if False in policies else True if all(value is True for value in policies) else None
            grok_turns_not_applicable = all(grok_coding_not_applicable_provider_free(
                trial["supervisor"], grok_turns) for trial in trials)
    working_set_runtime_matches = None
    if working_set != DEFAULT_SEMANTIC_WORKING_SET and trials:
        expected_working_set = semantic_working_set_observation(working_set)
        reported_working_sets = [trial["supervisor"] for trial in trials]
        if all(type(report) is dict for report in reported_working_sets):
            working_set_runtime_matches = all(
                all(type(report.get(key)) is type(value) and report.get(key) == value
                    for key, value in expected_working_set.items())
                for report in reported_working_sets)
    result = {"schema": "terminal-full-supervisor-receipt@1", "arm": prepared["arm"],
              "task": task.name, **profile,
              **prompt_observation, **doctor_observation,
              **({"provider_profile": prepared["provider_profile"]} if "provider_profile" in prepared else {}),
              **semantic_transport_selection(transport),
              **coding_reply_selection(reply_mode),
              **tool_output_selection(tool_output),
              **source_view_advisory_selection(source_view_advisory),
              **repository_source_selection(repository_source),
              **source_input_selection(source_input),
              **query_reference_selection(query_reference),
              **({"source_input_config_unchanged": configured_source_input == source_input}
                 if "source_input_policy" in prepared or "source_input_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"query_reference_config_unchanged": configured_query_reference == query_reference}
                 if "query_reference_policy" in prepared or "query_reference_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"repository_source_config_unchanged": configured_repository_source == repository_source}
                 if "repository_source_policy" in prepared or "repository_source_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"source_view_advisory_config_unchanged": configured_source_view_advisory == source_view_advisory}
                 if "source_view_advisory_policy" in prepared or "source_view_advisory_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"tool_output_config_unchanged": configured_tool_output == tool_output}
                 if "tool_output_policy" in prepared or "tool_output_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **semantic_metadata_observation(metadata_view),
              **semantic_working_set_observation(working_set),
              **grok_coding_selection(grok_turns),
              **({"grok_coding_config_unchanged": configured_grok_turns == grok_turns}
                 if "grok_coding_max_turns" in prepared or "grok_coding_max_turns" in config["agents"][0].get("kwargs", {}) else {}),
              **({"grok_coding_runtime_declared_matches": grok_turns_runtime_matches,
                  "grok_coding_runtime_policy_matches": grok_turns_policy_matches,
                  "grok_coding_not_applicable_provider_free": grok_turns_not_applicable}
                 if grok_turns != DEFAULT_GROK_CODING_MAX_TURNS else {}),
              **({"semantic_working_set_config_unchanged": configured_working_set == working_set}
                 if "semantic_working_set" in prepared or "semantic_working_set" in config["agents"][0].get("kwargs", {}) else {}),
              **({"semantic_working_set_runtime_declared_matches": working_set_runtime_matches}
                 if working_set != DEFAULT_SEMANTIC_WORKING_SET else {}),
              **capsule_selection_observation(capsule_selection),
              **capsule_budget_observation(capsule_budget),
              **source_view_observation(source_view, source_policy),
              **source_edit_observation(source_edit),
              **({"source_edit_config_unchanged": configured_source_edit == source_edit}
                 if "source_edit_mode" in prepared or "source_edit_mode" in config["agents"][0].get("kwargs", {}) else {}),
              **({"source_view_policy_config_unchanged": source_policy_unchanged}
                 if source_policy_present else {}),
              **({"source_view_config_unchanged": configured_source_view == source_view}
                 if "source_view_profile" in prepared or "source_view_profile" in config["agents"][0].get("kwargs", {}) else {}),
              **({"capsule_selection_config_unchanged": configured_capsules == capsule_selection}
                 if "capsule_selection" in prepared or "capsule_selection" in config["agents"][0].get("kwargs", {}) else {}),
              **semantic_context_selection(semantic_profile),
              **({"semantic_context_config_unchanged": configured_semantic_profile == semantic_profile}
                 if "semantic_context_profile" in prepared or "semantic_context_profile" in config["agents"][0].get("kwargs", {}) else {}),
              **({"capsule_budget_config_unchanged": configured_capsule_budget == capsule_budget}
                 if "capsule_budget" in prepared or "capsule_budget" in config["agents"][0].get("kwargs", {}) else {}),
              **({"capsule_budget_runtime_declared_matches": budget_runtime_matches}
                 if capsule_budget != DEFAULT_CAPSULE_BUDGET else {}),
              **numerical_observation(numerical_profile, prepared["arm"]),
              **planning_observation(planning_mode),
              **({"planning_mode_config_unchanged": configured_planning == planning_mode}
                 if "planning_mode" in prepared or "planning_mode" in config["agents"][0].get("kwargs", {}) else {}),
              **({"numerical_formalization_config_unchanged": configured_numerical == numerical_profile}
                 if numerical_profile is not None or configured_numerical is not None else {}),
              **({"semantic_metadata_config_unchanged": configured_metadata_view == metadata_view}
                 if "semantic_metadata_view" in prepared or "semantic_metadata_view" in config["agents"][0].get("kwargs", {}) else {}),
              **({"coding_reply_config_unchanged": configured_reply_mode == reply_mode}
                 if "coding_reply_mode" in prepared or "coding_reply_mode" in config["agents"][0].get("kwargs", {}) else {}),
              **({"semantic_transport_config_unchanged": configured_transport == transport}
                 if "semantic_transport_schema" in prepared or "semantic_transport_schema" in config["agents"][0].get("kwargs", {}) else {}),
              "original_task_inputs_unchanged": unchanged,
              "comparison_controls": benchmark_controls.observe_controls(
                  prepared, config, current_task_hashes=_task_hashes(task)),
              "trials": trials, "trial_count": len(trials), "native_job_result_present": (job / "result.json").is_file(),
              "complete_single_trial_receipt": unchanged and len(trials) == 1
                  and (not decision or trials[0]["semantic_decision_dispatch"]["verified"] is True)
                  and prompt_observation["prompt_comparison_options_valid"] is True
                  and prompt_observation["prompt_comparison_config_unchanged"] is not False
                  and doctor_observation["doctor_dispatch_policy_valid"] is True
                  and doctor_observation["doctor_dispatch_policy_config_unchanged"] is not False
                  and trials[0]["exact_trial_task_matches"] is True and (job / "result.json").is_file()
                  and configured_transport == transport and configured_reply_mode == reply_mode
                  and configured_numerical == numerical_profile and configured_planning == planning_mode
                  and configured_metadata_view == metadata_view
                  and configured_grok_turns == grok_turns
                  and (grok_turns == DEFAULT_GROK_CODING_MAX_TURNS or
                       grok_turns_runtime_matches is True and
                       (grok_turns_policy_matches is True or grok_turns_not_applicable is True))
                  and configured_working_set == working_set
                  and (working_set == DEFAULT_SEMANTIC_WORKING_SET or working_set_runtime_matches is True)
                  and configured_capsules == capsule_selection
                  and configured_capsule_budget == capsule_budget
                  and (capsule_budget == DEFAULT_CAPSULE_BUDGET or budget_runtime_matches is True)
                  and configured_source_view == source_view and source_policy_unchanged
                  and configured_source_edit == source_edit
                  and configured_source_input == source_input
                  and (source_input == DEFAULT_SOURCE_INPUT_POLICY or
                       trials[0]["source_input_dispatch"]["effective_input_observed"] is True)
                  and configured_query_reference == query_reference
                  and (query_reference == DEFAULT_QUERY_REFERENCE_POLICY or
                       trials[0]["query_reference_dispatch"]["effective_input_observed"] is True)
                  and configured_repository_source == repository_source
                  and (repository_source == DEFAULT_REPOSITORY_SOURCE_POLICY or
                       trials[0]["repository_source_dispatch"]["effective_repository_tools_observed"] is True)
                  and configured_source_view_advisory == source_view_advisory
                  and (source_view_advisory == DEFAULT_SOURCE_VIEW_POLICY or
                       trials[0]["source_view_dispatch"]["effective_advisory_observed"] is True)
                  and configured_tool_output == tool_output
                  and (tool_output == DEFAULT_TOOL_OUTPUT_POLICY or
                       trials[0]["tool_output_dispatch"]["effective_advisory_observed"] is True)
                  and configured_numerical == numerical_profile
                  and configured_metadata_view == metadata_view
                  and (metadata_view == DEFAULT_SEMANTIC_METADATA_VIEW or
                       trials[0]["semantic_metadata_dispatch"]["requested_mode_matches_report"] is True)
                  and configured_semantic_profile == semantic_profile
                  and (semantic_profile == DEFAULT_SEMANTIC_CONTEXT_PROFILE
                       or trials[0].get("semantic_context_profile_matches") is True),
              "planning_and_cold_index_charged_to_agent_time": True, "benchmark_advantage_claimed": False,
              "parallel_workers": 1, "dollar_cost": None, **selection,
              "intent_action_384": prepared.get("intent_action_384"),
              "source384": prepared.get("source384"), "source384_enabled": prepared.get("source384_enabled", False),
              "resource_profile": prepared.get("resource_profile"),
              "intent_selection_config_unchanged": configured_selection == selection}
    if decision:
        result.update(semantic_decision_selection=decision, semantic_decision_config_unchanged=True,
            semantic_decision_verification_scope="frozen_requested_bindings_and_retained_native_usage_only",
            semantic_decision_runtime_verified=bool(trials) and all(
                trial["semantic_decision_dispatch"]["verified"] is True for trial in trials))
    _json(output / "receipt.json", result)
    return result


def _execute_once(output: Path, *, task_name: str | None = None) -> dict:
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, config, job_prefix="supervisor-" + prepared["arm"] + "-",
                          task_name=task_name)
    if not prepared["prepared"] or _hash(output / "config.json") != prepared["config_sha256"]:
        raise ValueError("prepared configuration changed")
    agent = config["agents"][0]
    decision = _decision_options(agent.get("kwargs", {}), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    prompt_observation = _prompt_option_observation(prepared, agent.get("kwargs"))
    if (prompt_observation["prompt_comparison_options_valid"] is not True
            or prompt_observation["prompt_comparison_config_unchanged"] is False
            or ("prompt_comparison_options" in prepared
                and prompt_observation["prompt_comparison_config_unchanged"] is not True)):
        raise ValueError("prepared supervisor prompt options differ from its configuration")
    decision_selected = any(key in prompt_observation["prompt_comparison_options"]
                            for key in ("semantic_decision_policy_artifact", "semantic_decision_nomination_artifact", "semantic_decision_nomination_source"))
    native_selected = prompt_observation["prompt_comparison_options"].get("semantic_native_input") is True
    retrieval_selected = prompt_observation["prompt_comparison_options"].get("semantic_retrieval_query_reference") is True
    selected_pin_field = "runtime_native_input_source_sha256" if native_selected else "runtime_decision_source_sha256"
    nomination_source = prompt_observation["prompt_comparison_options"].get("semantic_decision_nomination_source")
    if nomination_source is not None:
        snapshot = prepared.get("semantic_decision_nomination_snapshot")
        if (type(snapshot) is not dict or snapshot.get("artifact") != nomination_source
                or snapshot.get("sha256") != prompt_observation["prompt_comparison_options"]["semantic_decision_nomination_sha256"]):
            raise ValueError("prepared host nomination snapshot differs from its configuration")
        verify_nomination_snapshot(snapshot, directory=output.absolute())
    elif "semantic_decision_nomination_snapshot" in prepared:
        raise ValueError("prepared host nomination snapshot requires its host source configuration")
    if decision_selected or native_selected:
        _require_current_decision_runtime_pins(prepared.get(selected_pin_field), native_input=native_selected,
            **({"retrieval_query_reference": True} if retrieval_selected else {}))
    doctor_observation = _doctor_policy_observation(prepared, agent.get("kwargs"))
    if (doctor_observation["doctor_dispatch_policy_valid"] is not True
            or doctor_observation["doctor_dispatch_policy_config_unchanged"] is False
            or ("doctor_dispatch_policy" in prepared
                and doctor_observation["doctor_dispatch_policy_config_unchanged"] is not True)):
        raise ValueError("prepared supervisor Doctor policy differs from its configuration")
    planning_mode = validate_planning_mode(prepared.get("planning_mode", DEFAULT_PLANNING_MODE),
        intent_requirement_contract=config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        task_profile=config["agents"][0].get("kwargs", {}).get("task_profile"))
    configured_planning = validate_planning_mode(
        config["agents"][0].get("kwargs", {}).get("planning_mode", DEFAULT_PLANNING_MODE),
        intent_requirement_contract=config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        task_profile=config["agents"][0].get("kwargs", {}).get("task_profile"))
    if planning_mode != configured_planning:
        raise ValueError("prepared planning mode changed")
    numerical_profile = validate_numerical_profile(prepared.get("numerical_formalization_profile"))
    configured_numerical = validate_numerical_profile(
        config["agents"][0].get("kwargs", {}).get("numerical_formalization_profile"))
    if numerical_profile != configured_numerical:
        raise ValueError("prepared numerical formalization selection changed")
    reply_mode = validate_coding_reply_mode(prepared.get("coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
        arm=prepared["arm"], provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    if reply_mode != config["agents"][0].get("kwargs", {}).get("coding_reply_mode", DEFAULT_CODING_REPLY_MODE):
        raise ValueError("prepared coding reply selection changed")
    tool_output = validate_tool_output_policy(
        prepared.get("tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_tool_output = validate_tool_output_policy(
        config["agents"][0].get("kwargs", {}).get("tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY),
        arm=prepared["arm"])
    source_view_advisory = validate_source_view_advisory_policy(
        prepared.get("source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_source_view_advisory = validate_source_view_advisory_policy(
        config["agents"][0].get("kwargs", {}).get("source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY),
        arm=prepared["arm"])
    repository_source = validate_repository_source_policy(
        prepared.get("repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"], source_view_advisory_policy=source_view_advisory)
    configured_repository_source = validate_repository_source_policy(
        config["agents"][0].get("kwargs", {}).get("repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY),
        arm=prepared["arm"], provider=resolve_provider_profile(config["agents"][0].get("kwargs", {}).get(
            "provider_profile"))["provider"], source_view_advisory_policy=configured_source_view_advisory)
    source_input = validate_source_input_policy(
        prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    query_reference = validate_query_reference_policy(
        prepared.get("query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        source_input_policy=prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
    configured_source_input = validate_source_input_policy(
        config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    configured_query_reference = validate_query_reference_policy(
        config["agents"][0].get("kwargs", {}).get("query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        source_input_policy=config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
    transport = validate_semantic_transport_schema(
        prepared.get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=prepared["arm"])
    if source_input != configured_source_input:
        raise ValueError("prepared source input policy selection changed")
    if query_reference != configured_query_reference:
        raise ValueError("prepared query reference policy selection changed")
    if repository_source != configured_repository_source:
        raise ValueError("prepared repository source policy selection changed")
    if source_view_advisory != configured_source_view_advisory:
        raise ValueError("prepared source view policy selection changed")
    if tool_output != configured_tool_output:
        raise ValueError("prepared tool output policy selection changed")
    if transport != config["agents"][0].get("kwargs", {}).get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA):
        raise ValueError("prepared semantic transport selection changed")
    metadata_view = validate_semantic_metadata_view(
        prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=prepared["arm"],
        transport_schema=transport,
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_metadata_view = validate_semantic_metadata_view(
        config["agents"][0].get("kwargs", {}).get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
        arm=prepared["arm"], transport_schema=transport)
    if metadata_view != configured_metadata_view:
        raise ValueError("prepared semantic metadata view selection changed")
    capsule_selection = validate_capsule_selection(
        prepared.get("capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=prepared["arm"])
    configured_capsules = validate_capsule_selection(
        config["agents"][0].get("kwargs", {}).get("capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=prepared["arm"])
    capsule_budget = validate_capsule_budget(
        prepared.get("capsule_budget", DEFAULT_CAPSULE_BUDGET), arm=prepared["arm"])
    configured_capsule_budget = validate_capsule_budget(
        config["agents"][0].get("kwargs", {}).get("capsule_budget", DEFAULT_CAPSULE_BUDGET), arm=prepared["arm"])
    if capsule_selection != configured_capsules:
        raise ValueError("prepared capsule selection changed")
    if capsule_budget != configured_capsule_budget:
        raise ValueError("prepared capsule budget changed")
    source_view = validate_source_view_profile(
        prepared.get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=prepared["arm"])
    configured_source_view = validate_source_view_profile(
        config["agents"][0].get("kwargs", {}).get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE),
        arm=prepared["arm"])
    from .terminal_source_view_policy import validate_source_context_combination
    for signed_profile, advisory, repository_policy, input_policy, reference_policy in (
            (source_view, source_view_advisory, repository_source, source_input, query_reference),
            (configured_source_view, configured_source_view_advisory, configured_repository_source,
             configured_source_input, configured_query_reference)):
        validate_source_context_combination(source_view_profile=signed_profile,
            source_view_advisory_policy=advisory, repository_source_policy=repository_policy,
            source_input_policy=input_policy, query_reference_policy=reference_policy)
    grok_turns = validate_grok_coding_max_turns(
        prepared.get("grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS),
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_grok_turns = validate_grok_coding_max_turns(
        config["agents"][0].get("kwargs", {}).get("grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS),
        provider=resolve_provider_profile(config["agents"][0].get("kwargs", {}).get("provider_profile"))["provider"])
    working_set = validate_semantic_working_set(
        prepared.get("semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET), arm=prepared["arm"],
        source_view_profile=source_view, transport_schema=transport, metadata_view=metadata_view)
    configured_working_set = validate_semantic_working_set(
        config["agents"][0].get("kwargs", {}).get("semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET),
        arm=prepared["arm"], source_view_profile=configured_source_view,
        transport_schema=config["agents"][0].get("kwargs", {}).get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
        metadata_view=configured_metadata_view)
    source_edit = validate_source_edit_mode(prepared.get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=source_view)
    configured_source_edit = validate_source_edit_mode(
        config["agents"][0].get("kwargs", {}).get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=configured_source_view)
    if grok_turns != configured_grok_turns:
        raise ValueError("prepared Grok coding turn bound changed")
    if working_set != configured_working_set:
        raise ValueError("prepared semantic working set changed")
    if source_edit != configured_source_edit:
        raise ValueError("prepared source edit mode changed")
    if source_view != configured_source_view:
        raise ValueError("prepared source view profile changed")
    _, source_policy_unchanged, _ = _source_view_policy_state(prepared, config,
        source_view=source_view, configured_source_view=configured_source_view,
        current_config_sha256=_hash(output / "config.json"))
    if not source_policy_unchanged:
        raise ValueError("prepared source view policy changed")
    semantic_profile = validate_semantic_context_profile(
        prepared.get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE), arm=prepared["arm"])
    configured_semantic_profile = validate_semantic_context_profile(
        config["agents"][0].get("kwargs", {}).get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE),
        arm=prepared["arm"])
    if semantic_profile != configured_semantic_profile:
        raise ValueError("prepared semantic context profile selection changed")
    if _task_hashes(task) != prepared["task_input_sha256"]:
        raise ValueError("original task changed")
    if _hash(Path(prepared["archive"]) / "runtime.tar.gz") != prepared["archive_sha256"]:
        raise ValueError("runtime archive changed")
    if _hash(Path(prepared["archive"]) / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("runtime dependency manifest changed")
    current_manifest = json.loads((Path(prepared["archive"]) / "manifest.json").read_text())
    if decision:
        from .terminal_deployment import validate_source384_binding
        _decision_options({**agent.get("kwargs", {}),
            "source384_config": validate_source384_binding(current_manifest),
            "numerical_selection_config": current_manifest.get("numerical_selection_384")},
            arm=prepared["arm"], provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    if decision_selected or native_selected:
        # Selected capabilities are replayed even for externally retained
        # preparation records; an internally consistent archive is insufficient.
        manifest = json.loads((Path(prepared["archive"]) / "manifest.json").read_text())
        verified = _require_current_timeout_runtime(
            Path(prepared["archive"]), manifest, selected_sources=_decision_runtime_sources(
                **({"retrieval_query_reference": True} if retrieval_selected else {})))
        if any(verified[name] != digest for name, digest in prepared[selected_pin_field].items()):
            route = "native input" if native_selected else "decision"
            raise ValueError(f"runtime {route} source pins differ from the packaged worker")
    require_coding_reply_archive(current_manifest, reply_mode)
    require_tool_output_archive(current_manifest, tool_output)
    require_source_view_advisory_archive(current_manifest, source_view_advisory)
    require_repository_source_archive(current_manifest, repository_source,
        archive=Path(prepared["archive"]) / "runtime.tar.gz")
    require_source_input_archive(current_manifest, source_input)
    require_query_reference_archive(current_manifest, query_reference)
    require_semantic_metadata_archive(current_manifest, metadata_view)
    require_semantic_working_set_archive(current_manifest, working_set)
    require_grok_coding_archive(current_manifest, grok_turns)
    require_capsule_selection_archive(current_manifest, capsule_selection)
    require_capsule_budget_archive(current_manifest, capsule_budget)
    require_source_view_archive(current_manifest, source_view,
        config["agents"][0].get("kwargs", {}).get("resource_profile"))
    require_semantic_context_archive(current_manifest, semantic_profile)
    require_numerical_archive(current_manifest, numerical_profile)
    require_planning_archive(current_manifest, planning_mode)
    require_source_edit_archive(current_manifest, source_edit)
    from .terminal_numerical_selection_deployment import require_numerical_selection_archive
    require_numerical_selection_archive(current_manifest)
    require_retrieval_revision(current_manifest, prepared["arm"],
        config["agents"][0].get("kwargs", {}).get("model_revision", ""))
    from .terminal_worker_capability import require_worker_capability
    require_worker_capability(current_manifest,
        config["agents"][0].get("kwargs", {}).get("resource_profile"))
    require_runtime_provider_profile(current_manifest, prepared.get("provider_profile"))
    require_semantic_transport_archive(json.loads((Path(prepared["archive"]) / "manifest.json").read_text()), transport)
    if any(_hash(Path(path)) != value for path, value in prepared["host_source_sha256"].items()):
        raise ValueError("host adapter changed; prepare a new immutable trial")
    with (output / "invocation.json").open("x") as stream:
        json.dump({"started_at": datetime.now().astimezone().isoformat(), "command": prepared["command"]}, stream)
    started = time.monotonic()
    with (output / "harbor.stdout").open("w") as stdout, (output / "harbor.stderr").open("w") as stderr:
        native = subprocess.run(prepared["command"], stdout=stdout, stderr=stderr, env=os.environ.copy())
    result = collect(output)
    result.update(harbor_returncode=native.returncode, invocation_seconds=time.monotonic() - started)
    _json(output / "receipt.json", result)
    return result


def execute(output: Path, *, task_name: str | None = None,
            provider_backoff_directory: Path | None = None, provider_max_wait_seconds=0) -> dict:
    """Admit fresh work through shared cooldown; dispatched trials never replay."""
    from .terminal_provider_backoff import ProviderBackoff, BackoffStateError
    from .terminal_provider_stop import provider_stop_observation
    output = Path(output).absolute()
    # Check the one-shot boundary before touching any recovery state.
    if (output / 'invocation.json').exists():
        raise FileExistsError('trial already invoked; prepare a fresh trial')
    prepared = json.loads((output / 'preparation.json').read_text())
    selected = resolve_provider_profile(prepared.get('provider_profile'))
    provider = selected['provider']
    try:
        gate = ProviderBackoff(provider_backoff_directory)
        permit = gate.acquire_with_wait(provider, max_wait_seconds=provider_max_wait_seconds)
    except (BackoffStateError, OSError) as error:
        result = {'schema': 'terminal-provider-deferral@1', 'state': 'provider_deferred',
                  'provider': provider, 'reason': 'provider_admission_unavailable',
                  'error_type': type(error).__name__, 'retry_after_seconds': None,
                  'provider_calls': 0, 'containers_launched': 0, 'trial_invoked': False,
                  'automatic_replay_allowed': False}
        _json(output / 'provider-deferral.json', result)
        return result
    if not permit.allowed:
        result = {'schema': 'terminal-provider-deferral@1', 'state': 'provider_deferred',
                  **permit.observation(), 'provider_calls': 0, 'containers_launched': 0,
                  'trial_invoked': False}
        result['schema'] = 'terminal-provider-deferral@1'
        _json(output / 'provider-deferral.json', result)
        return result
    result = None
    try:
        # Full preparation and source validation happens AFTER any waiting.
        result = _execute_once(output, task_name=task_name)
        trials = result.get('trials', [])
        single_trial = type(trials) is list and len(trials) == 1 and type(trials[0]) is dict
        exact_trial = single_trial and trials[0].get('exact_trial_task_matches') is True
        report = trials[0].get('supervisor') if single_trial else {}
        report = report if type(report) is dict else {}
        stop = provider_stop_observation(report)
        rows = report.get('provider_invocations', [])
        receipts_bound = (type(rows) is list and bool(rows) and len(rows) <= 1024
                     and all(type(row) is dict and row.get('status') in {'provider_returned', 'failed'}
                             and row.get('schema') == 'router-implementation-invocation@1'
                             and type(row.get('invocation_id')) is str and 0 < len(row['invocation_id']) <= 256
                             and row.get('model') == selected['model']
                             and row.get('provider') == provider for row in rows)
                     and len({row['invocation_id'] for row in rows}) == len(rows))
        outcome_unknown = (bool(report.get('unreceipted_provider_attempt'))
                           or (bool(rows) and not receipts_bound)
                           or (single_trial and trials[0].get('native_agent_execution_started') is True
                               and not receipts_bound)
                           or (bool(trials) and not single_trial))
        recovered = (exact_trial and receipts_bound and not outcome_unknown
                     and all(row['status'] == 'provider_returned' for row in rows))
        result['provider_admission'] = permit.observation()
        result['provider_backoff'] = gate.settle(permit, stop=stop, provider_succeeded=recovered,
                                               outcome_unknown=outcome_unknown)
        _json(output / 'receipt.json', result)
        return result
    except BaseException as error:
        if (output / 'invocation.json').exists():
            try:
                gate.mark_unsettled(permit)
            except Exception as persistence_error:
                error.add_note('Provider admission blocker could not be persisted: '
                               + type(persistence_error).__name__)
        raise
    finally:
        # Unsettled recovery retains its persisted marker. After a crash or
        # failed receipt collection, another process cannot assume availability.
        try:
            if not permit._settled and not (output / 'invocation.json').exists():
                gate.abandon_before_dispatch(permit)
        finally:
            permit.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "execute", "collect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-backoff-directory", type=Path,
                        help="Shared host provider cooldown state; execute only")
    parser.add_argument("--provider-max-wait-seconds", type=float, default=0,
                        help="Bounded pre-dispatch wait, 0..60 seconds; default returns a deferral")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--task", help="Dataset task name; prepare defaults to fix-code-vulnerability")
    parser.add_argument("--task-profile", type=Path,
                        help="Explicit public-instruction-bound task profile for a nondefault supervisor task")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--arm", choices=("full", "no-index"), default="full")
    parser.add_argument("--provider-profile", choices=PROVIDER_PROFILES, help="Explicit immutable router route; no cross-provider fallback")
    parser.add_argument("--planning-mode", choices=PLANNING_MODES, default=DEFAULT_PLANNING_MODE,
        help="Explicit signed single-task planning route; frozen by preparation")
    parser.add_argument("--semantic-transport-schema", choices=SEMANTIC_TRANSPORT_SCHEMAS,
        default=DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, help="Explicit coding-context transport; planning remains unchanged")
    parser.add_argument("--source-input-policy", choices=SOURCE_INPUT_POLICIES,
        default=DEFAULT_SOURCE_INPUT_POLICY, help="Frozen explicit raw/compact code input and optional AST references")
    parser.add_argument("--query-reference-policy", choices=QUERY_REFERENCE_POLICIES,
        default=DEFAULT_QUERY_REFERENCE_POLICY, help="Frozen reference to the exact required inline instruction")
    parser.add_argument("--repository-source-policy", choices=REPOSITORY_SOURCE_POLICIES,
        default=DEFAULT_REPOSITORY_SOURCE_POLICY, help="Opt-in matched repository MCP tools; frozen by preparation")
    parser.add_argument("--source-view-advisory-policy", choices=SOURCE_VIEW_POLICIES,
        default=DEFAULT_SOURCE_VIEW_POLICY, help="Opt-in matched raw/compact source view; frozen by preparation")
    parser.add_argument("--tool-output-policy", choices=TOOL_OUTPUT_POLICIES,
        default=DEFAULT_TOOL_OUTPUT_POLICY, help="Opt-in coding output previews; frozen by preparation")
    parser.add_argument("--coding-reply-mode", choices=CODING_REPLY_MODES,
        default=DEFAULT_CODING_REPLY_MODE, help="Explicit coding acknowledgment contract; frozen by preparation")
    parser.add_argument("--semantic-metadata-view", choices=SEMANTIC_METADATA_VIEWS,
        default=DEFAULT_SEMANTIC_METADATA_VIEW,
        help="Experimental readable coding metadata view; planning unchanged and selection frozen by preparation")
    parser.add_argument("--semantic-working-set", choices=SEMANTIC_WORKING_SETS,
        default=DEFAULT_SEMANTIC_WORKING_SET, help="Opt-in capsule references with live exact expansion; frozen by preparation")
    parser.add_argument("--capsule-selection", choices=CAPSULE_SELECTIONS, default=DEFAULT_CAPSULE_SELECTION,
        help="Experimental source-bound capsule ranking; requires an explicit public task profile and is frozen by preparation")
    parser.add_argument("--capsule-budget", choices=CAPSULE_BUDGETS, default=DEFAULT_CAPSULE_BUDGET,
        help="Experimental maximum capsule count; full indexed arm only and frozen by preparation")
    parser.add_argument("--source-view-profile", choices=SOURCE_VIEW_PROFILES, default=DEFAULT_SOURCE_VIEW_PROFILE,
        help="Opt-in read-only coding source views; full indexed arm only and frozen by preparation")
    parser.add_argument("--source-edit-mode", choices=SOURCE_EDIT_MODES, default=DEFAULT_SOURCE_EDIT_MODE,
        help="Opt-in mapped source edit proposals; requires full arm and source views; frozen by preparation")
    parser.add_argument("--source-view-policy", type=Path,
        help="Explicit shared source view recipe JSON file; frozen by preparation")
    parser.add_argument("--semantic-context-profile", choices=SEMANTIC_CONTEXT_PROFILES,
        default=DEFAULT_SEMANTIC_CONTEXT_PROFILE,
        help="Frozen semantic capsule selection; select identically except for a matched context ablation")
    parser.add_argument("--source384-config", type=Path, help="Verify the offline pinned parent matches the archive")
    parser.add_argument("--numerical-formalization-profile", choices=NUMERICAL_PROFILES,
        help="Opt-in source-bound numerical obligations; frozen equally for both arms, consumed only by full")
    from .terminal_setup_cache_advice import POLICIES
    parser.add_argument("--setup-cache-policy", choices=POLICIES,
        help="Explicit matching runtime setup-cache policy; absent preserves default")
    parser.add_argument("--resource-profile", choices=PROFILES, help="Explicit common limits; select identically for all comparison arms")
    parser.add_argument("--intent-action-384-config", type=Path,
        help="Verify the explicit local Intent384 configuration matches the packaged assets")
    parser.add_argument("--intent-requirement-contract", type=Path,
        help="Use source-bound @1 provider coverage or @2 reviewed symbolic operations before admission")
    parser.add_argument("--grok-coding-max-turns", type=int, default=DEFAULT_GROK_CODING_MAX_TURNS)
    parser.add_argument("--semantic-slot-policy-revision", default="legacy@1",
        choices=("legacy@1", "closed-typed-identifiers@2"),
        help="Request existing full-arm semantic coding alternatives")
    parser.add_argument("--semantic-native-input", action="store_true",
                        help="Full indexed control preserving native input without semantic encoding or projection")
    parser.add_argument("--semantic-context-projection", action="store_true",
        help="Request the existing V2 semantic role-view comparison")
    parser.add_argument("--semantic-decision-policy-artifact",
        help="Already deployed absolute child-local decision policy; preparation does not stage it")
    parser.add_argument("--semantic-decision-policy-sha256",
        help="Exact lowercase SHA-256 of the task-specific decision policy")
    parser.add_argument("--semantic-decision-nomination-artifact",
        help="Already deployed child-local capsule nomination; task identity is derived by the owner")
    parser.add_argument("--semantic-decision-nomination-source",
        help="Absolute host nomination file to snapshot and deliver to the child owner")
    parser.add_argument("--semantic-decision-nomination-sha256",
        help="Exact lowercase SHA-256 of the nomination; mutually exclusive with a prebound policy")
    parser.add_argument("--disable-semantic-typed-nomination", action="store_true",
                        help="opt out of source-bound typed dependency selection for an owner nomination")
    parser.add_argument("--semantic-retrieval-query-reference", action="store_true",
        help="Reference an exact repeated retrieval query using an owner nomination with typed selection disabled")
    parser.add_argument("--semantic-decision-max-calls", type=int, default=3,
        help="Maximum dialogue calls (1 to 4); nondefault values require a policy or nomination")
    parser.add_argument("--semantic-decision-max-total-input-tokens", type=int,
        help="Optional sum of rendered input replay counts; estimates cannot guarantee provider usage")
    parser.add_argument("--prompt-tokenizer-artifact",
        help="Absolute container-local tokenizer descriptor; not staged by preparation")
    parser.add_argument("--prompt-tokenizer-sha256",
        help="Exact lowercase SHA-256 of the supplied tokenizer descriptor")
    parser.add_argument("--max-input-tokens", type=int,
        help="Complete user-prompt budget; automatic estimates cannot guarantee provider limits")
    parser.add_argument("--doctor-dispatch-policy", default="auto", choices=("auto", "model-only"),
        help="Full-arm ablation: ordinary model coding without Doctor candidate selection")
    args = parser.parse_args()
    selected_prompt = _prompt_options(args.arm,
        prompt_tokenizer_artifact=args.prompt_tokenizer_artifact,
        prompt_tokenizer_sha256=args.prompt_tokenizer_sha256, max_input_tokens=args.max_input_tokens,
        semantic_slot_policy_revision=args.semantic_slot_policy_revision,
        semantic_context_projection=args.semantic_context_projection,
        semantic_native_input=args.semantic_native_input,
        semantic_decision_policy_artifact=args.semantic_decision_policy_artifact,
        semantic_decision_policy_sha256=args.semantic_decision_policy_sha256,
        semantic_decision_nomination_artifact=args.semantic_decision_nomination_artifact,
        semantic_decision_nomination_source=args.semantic_decision_nomination_source,
        semantic_decision_nomination_sha256=args.semantic_decision_nomination_sha256,
        disable_semantic_typed_nomination=args.disable_semantic_typed_nomination,
        **({"semantic_retrieval_query_reference": True} if args.semantic_retrieval_query_reference else {}),
        semantic_decision_max_calls=args.semantic_decision_max_calls,
        semantic_decision_max_total_input_tokens=args.semantic_decision_max_total_input_tokens)
    if args.operation != "prepare" and (selected_prompt != _prompt_options(args.arm)
            or args.doctor_dispatch_policy != "auto"):
        parser.error("prompt and decision options are frozen by preparation; collect/execute cannot select them")
    if args.operation != "prepare" and args.grok_coding_max_turns != DEFAULT_GROK_CODING_MAX_TURNS:
        parser.error("Grok coding turn bound is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.planning_mode != DEFAULT_PLANNING_MODE:
        parser.error("planning mode is frozen during prepare; collect/execute cannot select it")
    if args.operation != 'execute' and (args.provider_backoff_directory is not None or args.provider_max_wait_seconds != 0):
        parser.error('provider admission options apply only to execute')
    if args.operation != "prepare" and args.numerical_formalization_profile is not None:
        parser.error("numerical formalization is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
        parser.error("semantic transport is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_input_policy != DEFAULT_SOURCE_INPUT_POLICY:
        parser.error("source input policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.query_reference_policy != DEFAULT_QUERY_REFERENCE_POLICY:
        parser.error("query reference policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.repository_source_policy != DEFAULT_REPOSITORY_SOURCE_POLICY:
        parser.error("repository source policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_advisory_policy != DEFAULT_SOURCE_VIEW_POLICY:
        parser.error("source view policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.tool_output_policy != DEFAULT_TOOL_OUTPUT_POLICY:
        parser.error("tool output policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.coding_reply_mode != DEFAULT_CODING_REPLY_MODE:
        parser.error("coding reply mode is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW:
        parser.error("semantic metadata view is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_working_set != DEFAULT_SEMANTIC_WORKING_SET:
        parser.error("semantic working set is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.capsule_selection != DEFAULT_CAPSULE_SELECTION:
        parser.error("capsule selection is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.capsule_budget != DEFAULT_CAPSULE_BUDGET:
        parser.error("capsule budget is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_profile != DEFAULT_SOURCE_VIEW_PROFILE:
        parser.error("source view profile is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_edit_mode != DEFAULT_SOURCE_EDIT_MODE:
        parser.error("source edit mode is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_policy is not None:
        parser.error("source view policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_context_profile != DEFAULT_SEMANTIC_CONTEXT_PROFILE:
        parser.error("semantic context profile is frozen by preparation; collect/execute cannot select it")
    if args.operation == "prepare":
        if args.dataset is None or args.archive is None:
            parser.error("prepare requires --dataset and --archive")
        result = prepare(dataset=args.dataset, output=args.output, archive=args.archive, arm=args.arm,
                         doctor_dispatch_policy=args.doctor_dispatch_policy, **selected_prompt,
                         intent_requirement_contract=args.intent_requirement_contract,
                         intent_action_384_config=args.intent_action_384_config,
                         source384_config=args.source384_config, resource_profile=args.resource_profile,
                         setup_cache_policy=args.setup_cache_policy,
                         task_name=TASK if args.task is None else args.task,
                         task_profile=args.task_profile, provider_profile=args.provider_profile,
                         semantic_transport_schema=args.semantic_transport_schema,
                         coding_reply_mode=args.coding_reply_mode,
                         tool_output_policy=args.tool_output_policy,
                         source_view_advisory_policy=args.source_view_advisory_policy,
                         repository_source_policy=args.repository_source_policy,
                         source_input_policy=args.source_input_policy,
                         query_reference_policy=args.query_reference_policy,
                         semantic_metadata_view=args.semantic_metadata_view,
                         semantic_context_profile=args.semantic_context_profile,
                         **semantic_working_set_selection(args.semantic_working_set),
                         **grok_coding_selection(args.grok_coding_max_turns),
                         capsule_selection=args.capsule_selection,
                         capsule_budget=args.capsule_budget,
                         source_view_profile=args.source_view_profile,
                         source_edit_mode=args.source_edit_mode,
                         source_view_policy=_load_source_view_policy(args.source_view_policy),
                         numerical_formalization_profile=args.numerical_formalization_profile,
                         planning_mode=args.planning_mode)
    elif args.operation == 'execute':
        result = execute(args.output, task_name=args.task,
                         provider_backoff_directory=args.provider_backoff_directory,
                         provider_max_wait_seconds=args.provider_max_wait_seconds)
    else:
        result = collect(args.output, task_name=args.task)
    print(json.dumps(result, sort_keys=True, indent=2))
    if result.get('state') == 'provider_deferred':
        raise SystemExit(75)


if __name__ == "__main__":
    main()

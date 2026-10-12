"""One-shot original Harbor task trials for the full and no-index supervisor arms."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
from .terminal_source_view_policy import (
    DEFAULT_SOURCE_VIEW_POLICY, SOURCE_VIEW_POLICIES, validate_source_view_policy,
    source_view_selection, require_source_view_archive, source_view_dispatch_observation,
)
from .terminal_tool_output_policy import (
    DEFAULT_TOOL_OUTPUT_POLICY, TOOL_OUTPUT_POLICIES, validate_tool_output_policy,
    tool_output_selection, require_tool_output_archive, tool_output_dispatch_observation,
)
from .terminal_semantic_metadata_policy import (
    DEFAULT_SEMANTIC_METADATA_VIEW, SEMANTIC_METADATA_VIEWS, validate_semantic_metadata_view,
    semantic_metadata_selection, semantic_metadata_observation, require_semantic_metadata_archive,
)
from .terminal_semantic_context_policy import (
    DEFAULT_SEMANTIC_CONTEXT_PROFILE, SEMANTIC_CONTEXT_PROFILES,
    validate_semantic_context_profile, semantic_context_selection, require_semantic_context_archive,
)
from .terminal_numerical_policy import (
    PROFILES as NUMERICAL_PROFILES, validate_numerical_profile, numerical_selection,
    numerical_observation, require_numerical_archive,
)

ADAPTER = "benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent"
INTENT_SOURCE_PATH = ".supervisor-instruction.md"
MAX_INTENT_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_INSTRUCTION_BYTES = 32768
MAX_TASK_PROFILE_BYTES = 65536


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


def _intent_selection(contract: dict | None) -> dict:
    strategy = ("intent_symbolic" if contract is not None and contract.get("schema") in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3", "intent-plan-requirement-contract@4", "intent-plan-requirement-contract@5"}
                else "intent_coverage" if contract is not None else "direct")
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
               source_view_policy: str = DEFAULT_SOURCE_VIEW_POLICY,
               source_input_policy: str = DEFAULT_SOURCE_INPUT_POLICY,
               coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
               semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
               semantic_context_profile: str = DEFAULT_SEMANTIC_CONTEXT_PROFILE,
               numerical_formalization_profile: str | None = None,
               model_revision: str = "") -> dict:
    validate_semantic_context_profile(semantic_context_profile, arm=arm)
    validate_numerical_profile(numerical_formalization_profile)
    model_revision = validate_model_revision(model_revision)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
    validate_tool_output_policy(tool_output_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_view_policy(source_view_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_input_policy(source_input_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view)
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
                   **source_view_selection(source_view_policy),
                   **source_input_selection(source_input_policy),
                   **numerical_selection(numerical_formalization_profile),
                   **semantic_metadata_selection(semantic_metadata_view),
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
            source_view_policy: str = DEFAULT_SOURCE_VIEW_POLICY,
            source_input_policy: str = DEFAULT_SOURCE_INPUT_POLICY,
            coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
            semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
            semantic_context_profile: str = DEFAULT_SEMANTIC_CONTEXT_PROFILE,
            numerical_formalization_profile: str | None = None) -> dict:
    validate_semantic_context_profile(semantic_context_profile, arm=arm)
    validate_numerical_profile(numerical_formalization_profile)
    validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
    validate_tool_output_policy(tool_output_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_view_policy(source_view_policy, arm=arm, provider=selected_provider["provider"])
    validate_source_input_policy(source_input_policy, arm=arm, provider=selected_provider["provider"],
        semantic_metadata_view=semantic_metadata_view)
    validate_semantic_metadata_view(semantic_metadata_view, arm=arm,
        transport_schema=semantic_transport_schema, provider=selected_provider["provider"])
    from harbor.models.job.config import JobConfig
    dataset = dataset.resolve(strict=True)
    task = _task_path(dataset, task_name)
    selected_task_profile = _load_task_profile(task_profile, task=task)
    archive = archive.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh trial output required")
    manifest = json.loads((archive / "manifest.json").read_text())
    if _hash(archive / "runtime.tar.gz") != manifest["archive_sha256"]:
        raise ValueError("runtime archive integrity failed")
    model_revision = selected_retrieval_revision(manifest, arm)
    from .terminal_worker_capability import require_worker_capability
    require_worker_capability(manifest, resource_profile)
    require_runtime_provider_profile(manifest, provider_profile)
    require_semantic_transport_archive(manifest, semantic_transport_schema)
    require_coding_reply_archive(manifest, coding_reply_mode)
    require_tool_output_archive(manifest, tool_output_policy)
    require_source_view_archive(manifest, source_view_policy)
    require_source_input_archive(manifest, source_input_policy)
    require_semantic_metadata_archive(manifest, semantic_metadata_view)
    require_semantic_context_archive(manifest, semantic_context_profile)
    require_numerical_archive(manifest, numerical_formalization_profile)
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
    if source384_config is not None:
        _, expected_binding = _source384_assets(load_source384_config(source384_config))
        if selected_source384 != expected_binding:
            raise ValueError("selected Source384 config differs from the runtime archive")
    if intent_action_384_config is not None:
        host_config = load_intent_action_384_config(intent_action_384_config)
        _, expected_binding = _intent_action_384_assets(host_config)
        if selected_intent != expected_binding:
            raise ValueError("selected Intent384 config differs from the runtime archive")
    from .terminal_setup_cache_advice import select_setup_cache, validate_setup_cache_prerequisites
    setup_cache_selection = select_setup_cache(archive, setup_cache_policy)
    validate_setup_cache_prerequisites(setup_cache_selection, install_codex=True,
        auth_json=Path.home() / (".codex/auth.json" if selected_provider["provider"] == "codex_cli" else ".grok/auth.json"), arm=arm, resource_profile=resource_profile)
    requirements = _load_intent_requirement_contract(intent_requirement_contract, dataset=dataset, task_name=task_name)
    validate_header_planning_selection(selected_source384, requirements, arm)
    task_hashes = _task_hashes(task)
    declared_config = config_for(dataset, output, archive, arm,
        model_revision=model_revision,
        intent_requirement_contract=requirements, resource_profile=resource_profile,
        setup_cache_selection=setup_cache_selection, task_name=task_name, task_profile=selected_task_profile,
        semantic_transport_schema=semantic_transport_schema,
        coding_reply_mode=coding_reply_mode,
        tool_output_policy=tool_output_policy,
        source_view_policy=source_view_policy,
        source_input_policy=source_input_policy,
        semantic_metadata_view=semantic_metadata_view,
        semantic_context_profile=semantic_context_profile,
        numerical_formalization_profile=numerical_formalization_profile,
        **({"provider_profile": provider_profile} if provider_profile is not None else {}))
    if selected_source384 is not None and arm == "full":
        validate_resource_profile(declared_config, resource_profile)
    config = JobConfig.model_validate(declared_config, extra="forbid")
    output.mkdir(parents=True)
    wire = benchmark_controls.canonicalize_serialized_retry_sets(
        config.model_dump(mode="json", context={"redact_sensitive_env": False}))
    _json(output / "config.json", wire)
    harbor = Path(sys.executable).with_name("harbor")
    command = [str(harbor), "run", "--config", str(output / "config.json"), "--yes"]
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, timeout=60)
    (output / "dry-run.stdout").write_text(dry.stdout)
    (output / "dry-run.stderr").write_text(dry.stderr)
    if _task_hashes(task) != task_hashes:
        raise ValueError("original task changed during supervisor preflight")
    sources = {str(path): _hash(path) for path in Path(__file__).parent.glob("*.py")}
    result = {"schema": "terminal-full-supervisor-preparation@1", "prepared": dry.returncode == 0,
              "arm": arm, "task": task_name, "dataset": str(dataset), "archive": str(archive),
              "archive_sha256": manifest["archive_sha256"], "manifest_sha256": _hash(archive / "manifest.json"),
              "host_source_sha256": sources,
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
              **source_view_selection(source_view_policy),
              **source_input_selection(source_input_policy),
              **semantic_metadata_observation(semantic_metadata_view),
              **semantic_context_selection(semantic_context_profile),
              **numerical_observation(numerical_formalization_profile, arm),
              "agent_timeout_seconds": declared_config["agents"][0]["override_timeout_sec"], "provider_calls": 0,
              "planning_and_cold_index_charged_to_agent_time": True,
              "benchmark_advantage_claimed": False, **_intent_selection(requirements),
              "intent_action_384": selected_intent, "source384": selected_source384,
              "source384_enabled": selected_source384 is not None and arm == "full",
              "resource_profile": resource_profile,
              **({"setup_cache_selection": setup_cache_selection} if setup_cache_selection is not None else {})}
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
    source_view = validate_source_view_policy(
        prepared.get("source_view_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_source_view = validate_source_view_policy(
        config["agents"][0].get("kwargs", {}).get("source_view_policy", DEFAULT_SOURCE_VIEW_POLICY),
        arm=prepared["arm"])
    source_input = validate_source_input_policy(
        prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    configured_source_input = validate_source_input_policy(
        config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
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
    semantic_profile = validate_semantic_context_profile(
        prepared.get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE), arm=prepared["arm"])
    configured_semantic_profile = validate_semantic_context_profile(
        config["agents"][0].get("kwargs", {}).get("semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE),
        arm=prepared["arm"])
    selection = {key: prepared.get(key, value) for key, value in _intent_selection(None).items()}
    configured_selection = _intent_selection(config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"))
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
                       "reward": (native.get("verifier_result") or {}).get("rewards"),
                       "exception_type": (native.get("exception_info") or {}).get("exception_type"),
                       "durations_seconds": durations, "agent_context": native.get("agent_result"),
                       "native_agent_execution_started": bool((native.get("agent_execution") or {}).get("started_at")),
                       "supervisor": report,
                       **({"provider_stop": provider_stop} if provider_stop is not None else {}),
                       **({"source_input_dispatch": source_input_dispatch_observation(report, source_input)}
                          if source_input != DEFAULT_SOURCE_INPUT_POLICY else {}),
                       **({"source_view_dispatch": source_view_dispatch_observation(report, source_view)}
                          if source_view != DEFAULT_SOURCE_VIEW_POLICY else {}),
                       **({"tool_output_dispatch": tool_output_dispatch_observation(report, tool_output)}
                          if tool_output != DEFAULT_TOOL_OUTPUT_POLICY else {}),
                       **({"semantic_metadata_dispatch": _metadata_dispatch_observation(report, metadata_view)}
                          if metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW else {}),
                       **({"semantic_context_profile_matches": isinstance(report, dict)
                           and report.get("semantic_context_profile") == semantic_profile}
                          if semantic_profile != DEFAULT_SEMANTIC_CONTEXT_PROFILE else {})})
    unchanged = _task_hashes(task) == prepared["task_input_sha256"]
    result = {"schema": "terminal-full-supervisor-receipt@1", "arm": prepared["arm"],
              "task": task.name, **profile,
              **({"provider_profile": prepared["provider_profile"]} if "provider_profile" in prepared else {}),
              **semantic_transport_selection(transport),
              **coding_reply_selection(reply_mode),
              **tool_output_selection(tool_output),
              **source_view_selection(source_view),
              **source_input_selection(source_input),
              **({"source_input_config_unchanged": configured_source_input == source_input}
                 if "source_input_policy" in prepared or "source_input_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"source_view_config_unchanged": configured_source_view == source_view}
                 if "source_view_policy" in prepared or "source_view_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **({"tool_output_config_unchanged": configured_tool_output == tool_output}
                 if "tool_output_policy" in prepared or "tool_output_policy" in config["agents"][0].get("kwargs", {}) else {}),
              **semantic_metadata_observation(metadata_view),
              **semantic_context_selection(semantic_profile),
              **({"semantic_context_config_unchanged": configured_semantic_profile == semantic_profile}
                 if "semantic_context_profile" in prepared or "semantic_context_profile" in config["agents"][0].get("kwargs", {}) else {}),
              **numerical_observation(numerical_profile, prepared["arm"]),
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
                  and trials[0]["exact_trial_task_matches"] is True and (job / "result.json").is_file()
                  and configured_transport == transport and configured_reply_mode == reply_mode
                  and configured_source_input == source_input
                  and (source_input == DEFAULT_SOURCE_INPUT_POLICY or
                       trials[0]["source_input_dispatch"]["effective_input_observed"] is True)
                  and configured_source_view == source_view
                  and (source_view == DEFAULT_SOURCE_VIEW_POLICY or
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
    source_view = validate_source_view_policy(
        prepared.get("source_view_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"])
    configured_source_view = validate_source_view_policy(
        config["agents"][0].get("kwargs", {}).get("source_view_policy", DEFAULT_SOURCE_VIEW_POLICY),
        arm=prepared["arm"])
    source_input = validate_source_input_policy(
        prepared.get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=prepared["arm"],
        provider=resolve_provider_profile(prepared.get("provider_profile"))["provider"],
        semantic_metadata_view=prepared.get("semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    configured_source_input = validate_source_input_policy(
        config["agents"][0].get("kwargs", {}).get("source_input_policy", DEFAULT_SOURCE_INPUT_POLICY),
        arm=prepared["arm"], semantic_metadata_view=config["agents"][0].get("kwargs", {}).get(
            "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
    transport = validate_semantic_transport_schema(
        prepared.get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=prepared["arm"])
    if source_input != configured_source_input:
        raise ValueError("prepared source input policy selection changed")
    if source_view != configured_source_view:
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
    require_coding_reply_archive(current_manifest, reply_mode)
    require_tool_output_archive(current_manifest, tool_output)
    require_source_view_archive(current_manifest, source_view)
    require_source_input_archive(current_manifest, source_input)
    require_semantic_metadata_archive(current_manifest, metadata_view)
    require_semantic_context_archive(current_manifest, semantic_profile)
    require_numerical_archive(current_manifest, numerical_profile)
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
    parser.add_argument("--semantic-transport-schema", choices=SEMANTIC_TRANSPORT_SCHEMAS,
        default=DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, help="Explicit coding-context transport; planning remains unchanged")
    parser.add_argument("--source-input-policy", choices=SOURCE_INPUT_POLICIES,
        default=DEFAULT_SOURCE_INPUT_POLICY, help="Frozen explicit raw/compact code input and optional AST references")
    parser.add_argument("--source-view-policy", choices=SOURCE_VIEW_POLICIES,
        default=DEFAULT_SOURCE_VIEW_POLICY, help="Opt-in matched raw/compact source view; frozen by preparation")
    parser.add_argument("--tool-output-policy", choices=TOOL_OUTPUT_POLICIES,
        default=DEFAULT_TOOL_OUTPUT_POLICY, help="Opt-in coding output previews; frozen by preparation")
    parser.add_argument("--coding-reply-mode", choices=CODING_REPLY_MODES,
        default=DEFAULT_CODING_REPLY_MODE, help="Explicit coding acknowledgment contract; frozen by preparation")
    parser.add_argument("--semantic-metadata-view", choices=SEMANTIC_METADATA_VIEWS,
        default=DEFAULT_SEMANTIC_METADATA_VIEW,
        help="Experimental readable coding metadata view; planning unchanged and selection frozen by preparation")
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
    args = parser.parse_args()
    if args.operation != 'execute' and (args.provider_backoff_directory is not None or args.provider_max_wait_seconds != 0):
        parser.error('provider admission options apply only to execute')
    if args.operation != "prepare" and args.numerical_formalization_profile is not None:
        parser.error("numerical formalization is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
        parser.error("semantic transport is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_input_policy != DEFAULT_SOURCE_INPUT_POLICY:
        parser.error("source input policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_policy != DEFAULT_SOURCE_VIEW_POLICY:
        parser.error("source view policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.tool_output_policy != DEFAULT_TOOL_OUTPUT_POLICY:
        parser.error("tool output policy is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.coding_reply_mode != DEFAULT_CODING_REPLY_MODE:
        parser.error("coding reply mode is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW:
        parser.error("semantic metadata view is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_context_profile != DEFAULT_SEMANTIC_CONTEXT_PROFILE:
        parser.error("semantic context profile is frozen by preparation; collect/execute cannot select it")
    if args.operation == "prepare":
        if args.dataset is None or args.archive is None:
            parser.error("prepare requires --dataset and --archive")
        result = prepare(dataset=args.dataset, output=args.output, archive=args.archive, arm=args.arm,
                         intent_requirement_contract=args.intent_requirement_contract,
                         intent_action_384_config=args.intent_action_384_config,
                         source384_config=args.source384_config, resource_profile=args.resource_profile,
                         setup_cache_policy=args.setup_cache_policy,
                         task_name=TASK if args.task is None else args.task,
                         task_profile=args.task_profile, provider_profile=args.provider_profile,
                         semantic_transport_schema=args.semantic_transport_schema,
                         coding_reply_mode=args.coding_reply_mode,
                         tool_output_policy=args.tool_output_policy,
                         source_view_policy=args.source_view_policy,
                         source_input_policy=args.source_input_policy,
                         semantic_metadata_view=args.semantic_metadata_view,
                         semantic_context_profile=args.semantic_context_profile,
                         numerical_formalization_profile=args.numerical_formalization_profile)
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

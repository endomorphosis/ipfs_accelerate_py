"""One-shot original Harbor task trials for the full and no-index supervisor arms."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
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
from .terminal_semantic_metadata_policy import (
    DEFAULT_SEMANTIC_METADATA_VIEW, SEMANTIC_METADATA_VIEWS, validate_semantic_metadata_view,
    semantic_metadata_selection, semantic_metadata_observation, require_semantic_metadata_archive,
)
from .terminal_capsule_selection_policy import (
    DEFAULT_CAPSULE_SELECTION, CAPSULE_SELECTIONS, validate_capsule_selection,
    capsule_selection_options, capsule_selection_observation, require_capsule_selection_archive,
)
from .terminal_planning_policy import (
    DEFAULT_PLANNING_MODE, PLANNING_MODES, validate_planning_mode,
    planning_selection, planning_observation, require_planning_archive,
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
from .terminal_numerical_policy import (
    PROFILES as NUMERICAL_PROFILES, validate_numerical_profile, numerical_selection,
    numerical_observation, require_numerical_archive,
)

ADAPTER = "benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent"
INTENT_SOURCE_PATH = ".supervisor-instruction.md"
MAX_INTENT_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_INSTRUCTION_BYTES = 32768
MAX_TASK_PROFILE_BYTES = 65536
MAX_SOURCE_VIEW_POLICY_BYTES = 65536


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
               coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
               semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
               capsule_selection: str = DEFAULT_CAPSULE_SELECTION,
               source_view_profile: str = DEFAULT_SOURCE_VIEW_PROFILE,
               source_view_policy: dict | None = None,
               source_edit_mode: str = DEFAULT_SOURCE_EDIT_MODE,
               numerical_formalization_profile: str | None = None,
               planning_mode: str = DEFAULT_PLANNING_MODE,
               model_revision: str = "") -> dict:
    validate_planning_mode(planning_mode, intent_requirement_contract=intent_requirement_contract,
        task_profile=task_profile)
    validate_numerical_profile(numerical_formalization_profile)
    validate_capsule_selection(capsule_selection, arm=arm)
    validate_source_view_profile(source_view_profile, arm=arm)
    validate_source_edit_mode(source_edit_mode, arm=arm, source_view_profile=source_view_profile)
    source_view_policy = validate_source_view_policy(source_view_policy, profile=source_view_profile, arm=arm)
    model_revision = validate_model_revision(model_revision)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
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
                   **numerical_selection(numerical_formalization_profile),
                   **planning_selection(planning_mode),
                   **semantic_metadata_selection(semantic_metadata_view),
                   **capsule_selection_options(capsule_selection),
                   **source_view_selection(source_view_profile, source_view_policy),
                   **source_edit_selection(source_edit_mode)},
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
            coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
            semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
            capsule_selection: str = DEFAULT_CAPSULE_SELECTION,
            source_view_profile: str = DEFAULT_SOURCE_VIEW_PROFILE,
            source_view_policy: dict | None = None,
               source_edit_mode: str = DEFAULT_SOURCE_EDIT_MODE,
            numerical_formalization_profile: str | None = None,
            planning_mode: str = DEFAULT_PLANNING_MODE) -> dict:
    validate_planning_mode(planning_mode, intent_requirement_contract=intent_requirement_contract)
    validate_numerical_profile(numerical_formalization_profile)
    validate_capsule_selection(capsule_selection, arm=arm)
    validate_source_view_profile(source_view_profile, arm=arm)
    validate_source_edit_mode(source_edit_mode, arm=arm, source_view_profile=source_view_profile)
    source_view_policy = validate_source_view_policy(source_view_policy, profile=source_view_profile, arm=arm)
    validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
    selected_provider = resolve_provider_profile(provider_profile)
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=selected_provider["provider"])
    validate_semantic_metadata_view(semantic_metadata_view, arm=arm,
        transport_schema=semantic_transport_schema, provider=selected_provider["provider"])
    from harbor.models.job.config import JobConfig
    dataset = dataset.resolve(strict=True)
    task = _task_path(dataset, task_name)
    selected_task_profile = _load_task_profile(task_profile, task=task)
    validate_planning_mode(planning_mode, task_profile=selected_task_profile)
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
    require_semantic_metadata_archive(manifest, semantic_metadata_view)
    require_capsule_selection_archive(manifest, capsule_selection)
    require_source_view_archive(manifest, source_view_profile, resource_profile)
    require_source_edit_archive(manifest, source_edit_mode)
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
        model_revision=model_revision,
        intent_requirement_contract=requirements, resource_profile=resource_profile,
        setup_cache_selection=setup_cache_selection, task_name=task_name, task_profile=selected_task_profile,
        semantic_transport_schema=semantic_transport_schema,
        coding_reply_mode=coding_reply_mode,
        semantic_metadata_view=semantic_metadata_view,
        capsule_selection=capsule_selection,
        source_view_profile=source_view_profile,
        source_edit_mode=source_edit_mode,
        source_view_policy=source_view_policy,
        numerical_formalization_profile=numerical_formalization_profile,
        planning_mode=planning_mode,
        **({"provider_profile": provider_profile} if provider_profile is not None else {}))
    if (selected_source384 is not None or selected_numerical is not None) and arm == "full":
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
              **semantic_metadata_observation(semantic_metadata_view),
              **capsule_selection_observation(capsule_selection),
              **source_view_observation(source_view_profile, source_view_policy),
              **source_edit_observation(source_edit_mode),
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
    _json(output / "preparation.json", result)
    if dry.returncode:
        raise RuntimeError("Harbor preflight failed; see retained logs")
    return result


def collect(output: Path, *, task_name: str | None = None) -> dict:
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, config, job_prefix="supervisor-" + prepared["arm"] + "-",
                          task_name=task_name)
    profile = prepared_provider_identity(prepared, config)
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
    source_view = validate_source_view_profile(
        prepared.get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=prepared["arm"])
    configured_source_view = validate_source_view_profile(
        config["agents"][0].get("kwargs", {}).get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE),
        arm=prepared["arm"])
    source_edit = validate_source_edit_mode(prepared.get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=source_view)
    configured_source_edit = validate_source_edit_mode(
        config["agents"][0].get("kwargs", {}).get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=configured_source_view)
    source_policy, source_policy_unchanged, source_policy_present = _source_view_policy_state(
        prepared, config, source_view=source_view, configured_source_view=configured_source_view,
        current_config_sha256=_hash(output / "config.json"))
    selection = {key: prepared.get(key, value) for key, value in _intent_selection(None).items()}
    configured_selection = _intent_selection(config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"),
        planning_mode=configured_planning)
    job = Path(config["jobs_dir"]) / config["job_name"]
    trials = []
    for path in sorted(job.glob("*/result.json")):
        native = json.loads(path.read_text())
        report_path = path.parent / "agent/supervisor-result.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
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
                       "supervisor": report})
    unchanged = _task_hashes(task) == prepared["task_input_sha256"]
    result = {"schema": "terminal-full-supervisor-receipt@1", "arm": prepared["arm"],
              "task": task.name, **profile,
              **({"provider_profile": prepared["provider_profile"]} if "provider_profile" in prepared else {}),
              **semantic_transport_selection(transport),
              **coding_reply_selection(reply_mode),
              **semantic_metadata_observation(metadata_view),
              **capsule_selection_observation(capsule_selection),
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
                  and trials[0]["exact_trial_task_matches"] is True and (job / "result.json").is_file()
                  and configured_transport == transport and configured_reply_mode == reply_mode
                  and configured_numerical == numerical_profile and configured_planning == planning_mode
                  and configured_metadata_view == metadata_view
                  and configured_capsules == capsule_selection
                  and configured_source_view == source_view and source_policy_unchanged
                  and configured_source_edit == source_edit,
              "planning_and_cold_index_charged_to_agent_time": True, "benchmark_advantage_claimed": False,
              "parallel_workers": 1, "dollar_cost": None, **selection,
              "intent_action_384": prepared.get("intent_action_384"),
              "source384": prepared.get("source384"), "source384_enabled": prepared.get("source384_enabled", False),
              "resource_profile": prepared.get("resource_profile"),
              "intent_selection_config_unchanged": configured_selection == selection}
    _json(output / "receipt.json", result)
    return result


def execute(output: Path, *, task_name: str | None = None) -> dict:
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, config, job_prefix="supervisor-" + prepared["arm"] + "-",
                          task_name=task_name)
    if not prepared["prepared"] or _hash(output / "config.json") != prepared["config_sha256"]:
        raise ValueError("prepared configuration changed")
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
    transport = validate_semantic_transport_schema(
        prepared.get("semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=prepared["arm"])
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
    if capsule_selection != configured_capsules:
        raise ValueError("prepared capsule selection changed")
    source_view = validate_source_view_profile(
        prepared.get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=prepared["arm"])
    configured_source_view = validate_source_view_profile(
        config["agents"][0].get("kwargs", {}).get("source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE),
        arm=prepared["arm"])
    source_edit = validate_source_edit_mode(prepared.get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=source_view)
    configured_source_edit = validate_source_edit_mode(
        config["agents"][0].get("kwargs", {}).get("source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
        arm=prepared["arm"], source_view_profile=configured_source_view)
    if source_edit != configured_source_edit:
        raise ValueError("prepared source edit mode changed")
    if source_view != configured_source_view:
        raise ValueError("prepared source view profile changed")
    _, source_policy_unchanged, _ = _source_view_policy_state(prepared, config,
        source_view=source_view, configured_source_view=configured_source_view,
        current_config_sha256=_hash(output / "config.json"))
    if not source_policy_unchanged:
        raise ValueError("prepared source view policy changed")
    if _task_hashes(task) != prepared["task_input_sha256"]:
        raise ValueError("original task changed")
    if _hash(Path(prepared["archive"]) / "runtime.tar.gz") != prepared["archive_sha256"]:
        raise ValueError("runtime archive changed")
    if _hash(Path(prepared["archive"]) / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("runtime dependency manifest changed")
    current_manifest = json.loads((Path(prepared["archive"]) / "manifest.json").read_text())
    require_coding_reply_archive(current_manifest, reply_mode)
    require_semantic_metadata_archive(current_manifest, metadata_view)
    require_capsule_selection_archive(current_manifest, capsule_selection)
    require_source_view_archive(current_manifest, source_view,
        config["agents"][0].get("kwargs", {}).get("resource_profile"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "execute", "collect"))
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--coding-reply-mode", choices=CODING_REPLY_MODES,
        default=DEFAULT_CODING_REPLY_MODE, help="Explicit coding acknowledgment contract; frozen by preparation")
    parser.add_argument("--semantic-metadata-view", choices=SEMANTIC_METADATA_VIEWS,
        default=DEFAULT_SEMANTIC_METADATA_VIEW,
        help="Experimental readable coding metadata view; planning unchanged and selection frozen by preparation")
    parser.add_argument("--capsule-selection", choices=CAPSULE_SELECTIONS, default=DEFAULT_CAPSULE_SELECTION,
        help="Experimental source-bound capsule ranking; requires an explicit public task profile and is frozen by preparation")
    parser.add_argument("--source-view-profile", choices=SOURCE_VIEW_PROFILES, default=DEFAULT_SOURCE_VIEW_PROFILE,
        help="Opt-in read-only coding source views; full indexed arm only and frozen by preparation")
    parser.add_argument("--source-edit-mode", choices=SOURCE_EDIT_MODES, default=DEFAULT_SOURCE_EDIT_MODE,
        help="Opt-in mapped source edit proposals; requires full arm and source views; frozen by preparation")
    parser.add_argument("--source-view-policy", type=Path,
        help="Explicit shared source view recipe JSON file; frozen by preparation")
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
    if args.operation != "prepare" and args.planning_mode != DEFAULT_PLANNING_MODE:
        parser.error("planning mode is frozen during prepare; collect/execute cannot select it")
    if args.operation != "prepare" and args.numerical_formalization_profile is not None:
        parser.error("numerical formalization is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
        parser.error("semantic transport is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.coding_reply_mode != DEFAULT_CODING_REPLY_MODE:
        parser.error("coding reply mode is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.semantic_metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW:
        parser.error("semantic metadata view is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.capsule_selection != DEFAULT_CAPSULE_SELECTION:
        parser.error("capsule selection is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_profile != DEFAULT_SOURCE_VIEW_PROFILE:
        parser.error("source view profile is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_edit_mode != DEFAULT_SOURCE_EDIT_MODE:
        parser.error("source edit mode is frozen by preparation; collect/execute cannot select it")
    if args.operation != "prepare" and args.source_view_policy is not None:
        parser.error("source view policy is frozen by preparation; collect/execute cannot select it")
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
                         semantic_metadata_view=args.semantic_metadata_view,
                         capsule_selection=args.capsule_selection,
                         source_view_profile=args.source_view_profile,
                         source_edit_mode=args.source_edit_mode,
                         source_view_policy=_load_source_view_policy(args.source_view_policy),
                         numerical_formalization_profile=args.numerical_formalization_profile,
                         planning_mode=args.planning_mode)
    else:
        result = {"execute": execute, "collect": collect}[args.operation](args.output, task_name=args.task)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

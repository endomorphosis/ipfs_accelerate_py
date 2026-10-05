"""Frozen public controls for the existing single-task Harbor pilot.

These are declared configuration identities, not proof of runtime enforcement
or a paired-campaign qualification. No prompts, credentials or task contents
are exported. Legacy preparations cannot acquire a retrospective declaration.
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "terminal-benchmark-declared-controls@1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
JOB_KEYS = ("n_attempts", "n_concurrent_trials", "retry", "timeout_multiplier",
            "agent_timeout_multiplier", "agent_setup_timeout_multiplier",
            "environment_build_timeout_multiplier", "verifier_timeout_multiplier",
            "environment", "verifier", "extra_instruction_paths", "extra_instructions",
            "install_only", "source_jobs", "datasets")
AGENT_KEYS = ("override_timeout_sec", "max_timeout_sec", "override_setup_timeout_sec",
              "n_concurrent", "concurrency_group", "mcp_servers", "skills",
              "extra_allowed_hosts", "load_trajectory", "resume_trajectory")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def build_controls(config, *, task_input_sha256, task, model, reasoning_effort, cli_version):
    """Build before execution from the validated, serialized Harbor config."""
    if len(config.get("agents", [])) != 1 or len(config.get("tasks", [])) != 1:
        raise ValueError("declared pilot controls require one agent and one task")
    if not task_input_sha256 or any(
        not isinstance(k, str) or not isinstance(v, str) or not _DIGEST.fullmatch(v)
        for k, v in task_input_sha256.items()
    ):
        raise ValueError("complete task input hashes required")
    agent = config["agents"][0]
    environment = config.get("environment") or {}
    if (config.get("user_agent") or config.get("extra_instruction_paths")
            or environment.get("extra_docker_compose") or environment.get("mounts")
            or environment.get("kwargs")):
        raise ValueError("external input sources require a separately sealed comparison profile")
    if agent.get("model_name") != model:
        raise ValueError("declared model differs from configuration")
    kwargs = agent.get("kwargs", {})
    if agent.get("name") == "codex" and not agent.get("import_path"):
        if kwargs != {"version": cli_version, "reasoning_effort": reasoning_effort}:
            raise ValueError("native model controls differ from declared CLI/reasoning")
        if set(agent.get("env") or {}) - {"CODEX_AUTH_JSON_PATH"}:
            raise ValueError("unsupported native environment requires a new comparison profile")
    elif agent.get("import_path") == (
        "benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent"
    ):
        # This fixed adapter has no runtime CLI/reasoning selectors. Its source
        # and archive are pinned by the supervisor preparation/execute owner.
        from .native_codex_baseline import MODEL, REASONING, CLI_VERSION
        if (model, reasoning_effort, cli_version) != (MODEL, REASONING, CLI_VERSION):
            raise ValueError("supervisor declaration differs from fixed source profile")
        if set(kwargs) - {"runtime_archive", "arm", "model_revision", "intent_requirement_contract", "resource_profile", "setup_cache_selection", "task_profile"}:
            raise ValueError("unsupported supervisor kwargs require a new comparison profile")
        if "task_profile" in kwargs:
            from .terminal_task_profile import validate_task_profile
            validate_task_profile(kwargs["task_profile"])
        if "resource_profile" in kwargs:
            from .benchmark_resource_profile import validate_resource_profile
            validate_resource_profile(config, kwargs["resource_profile"])
        if "setup_cache_selection" in kwargs:
            from .terminal_setup_cache_advice import _selection_shape
            from .benchmark_resource_profile import PROFILES
            _selection_shape(kwargs["setup_cache_selection"])
            if kwargs.get("arm") != "full" or kwargs.get("resource_profile") not in PROFILES:
                raise ValueError("setup cache selection requires the full arm and supported resource profile")
            # The complete configuration digest below binds this adapter-specific
            # selection. Common-control equality does not prove equal setup work.
        if agent.get("env"):
            raise ValueError("unsupported supervisor environment requires a new comparison profile")
    else:
        raise ValueError("unsupported benchmark adapter")
    return _control_record(config, task_input_sha256=task_input_sha256, task=task,
                           model=model, reasoning_effort=reasoning_effort, cli_version=cli_version)


def _control_record(config, *, task_input_sha256, task, model, reasoning_effort, cli_version):
    """Hash an already declared configuration without selecting a new profile."""
    if (type(config) is not dict or type(config.get("agents")) is not list
            or len(config["agents"]) != 1 or type(config["agents"][0]) is not dict):
        raise ValueError("frozen controls require one configured agent")
    if (type(task_input_sha256) is not dict or not task_input_sha256
            or any(type(key) is not str or type(value) is not str or not _DIGEST.fullmatch(value)
                   for key, value in task_input_sha256.items())):
        raise ValueError("complete task input hashes required")
    agent = config["agents"][0]
    if agent.get("model_name") != model:
        raise ValueError("frozen model differs from configuration")
    if agent.get("name") == "codex" and not agent.get("import_path"):
        if agent.get("kwargs") != {"version": cli_version, "reasoning_effort": reasoning_effort}:
            raise ValueError("frozen native profile differs from configuration")
    # Commit complete common objects, including resource overrides, retry filters,
    # additional instructions, mounts and tools. Only their digests are exported.
    controls = {
        "schema": SCHEMA,
        # Full config binding detects changes even to intentionally unequal arm
        # settings and task/output paths. It is not compared between arms.
        "configuration_sha256": _digest(config),
        "identity": {"task": task, "model": model, "reasoning_effort": reasoning_effort,
                     "cli_version": cli_version},
        "task_input_sha256": dict(sorted(task_input_sha256.items())),
        "job_controls_sha256": {key: _digest(config.get(key)) for key in JOB_KEYS},
        "agent_controls_sha256": {key: _digest(agent.get(key)) for key in AGENT_KEYS},
    }
    controls["sha256"] = _digest(controls)
    return controls


def validate_controls(value):
    """Reject edited, partial or malformed declarations without inventing defaults."""
    keys = {"schema", "configuration_sha256", "identity", "task_input_sha256", "job_controls_sha256",
            "agent_controls_sha256", "sha256"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != SCHEMA:
        return False
    if not isinstance(value["configuration_sha256"], str) or not _DIGEST.fullmatch(value["configuration_sha256"]):
        return False
    identity = value["identity"]
    if not isinstance(identity, dict) or set(identity) != {"task", "model", "reasoning_effort", "cli_version"}:
        return False
    if any(not isinstance(v, str) or not v for v in identity.values()):
        return False
    for name in ("task_input_sha256", "job_controls_sha256", "agent_controls_sha256"):
        fields = value[name]
        if not isinstance(fields, dict) or not fields:
            return False
        expected = {"job_controls_sha256": JOB_KEYS, "agent_controls_sha256": AGENT_KEYS}.get(name)
        if expected is not None and set(fields) != set(expected):
            return False
        if any(not isinstance(k, str) or not isinstance(v, str) or not _DIGEST.fullmatch(v)
               for k, v in fields.items()):
            return False
    try:
        return value["sha256"] == _digest({k: v for k, v in value.items() if k != "sha256"})
    except (TypeError, ValueError):
        return False


def observe_controls(prepared, config, *, current_task_hashes):
    """Recheck a pre-execution declaration; never backfill old preparations."""
    declared = prepared.get("comparison_controls")
    if declared is None:
        return {"status": "unavailable", "reason": "legacy_preparation_without_controls",
                "declared": None, "configuration_unchanged": None}
    if not validate_controls(declared):
        return {"status": "invalid", "reason": "invalid_frozen_controls",
                "declared": None, "configuration_unchanged": False}
    try:
        # Historical runs retain their original model/CLI identity. Observation
        # replays exact frozen controls; only build_controls selects today's
        # executable benchmark profile before a new trial.
        current = _control_record(config, task_input_sha256=current_task_hashes,
                                  **declared["identity"])
    except (KeyError, TypeError, ValueError):
        return {"status": "mismatch", "reason": "configuration_or_task_inputs_invalid",
                "declared": declared, "configuration_unchanged": False}
    same = current == declared
    return {"status": "observed" if same else "mismatch",
            "reason": None if same else "configuration_or_task_inputs_changed",
            "declared": declared, "configuration_unchanged": same}


def compare_controls(baseline, candidate):
    """Return tri-state match plus public field names, not sensitive config values."""
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return {"matches": None, "status": "unavailable", "differences": []}
    if baseline.get("status") == "unavailable" or candidate.get("status") == "unavailable":
        return {"matches": None, "status": "unavailable", "differences": []}
    left, right = baseline.get("declared"), candidate.get("declared")
    if not validate_controls(left) or not validate_controls(right):
        return {"matches": False, "status": "invalid", "differences": []}
    differences = [f"{section}.{key}" for section in
                   ("identity", "task_input_sha256", "job_controls_sha256", "agent_controls_sha256")
                   for key in sorted(set(left[section]) | set(right[section]))
                   if left[section].get(key) != right[section].get(key)]
    unchanged = all(item.get("status") == "observed" and item.get("configuration_unchanged") is True
                    for item in (baseline, candidate))
    return {"matches": not differences and unchanged,
            "status": "matched" if not differences and unchanged else "mismatch",
            "differences": differences,
            "basis": "predeclared_configuration_only; runtime_enforcement_not_established"}

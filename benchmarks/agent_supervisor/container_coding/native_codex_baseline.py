"""Prepare/run the native Harbor Codex counterpart and retain raw usage.

Run ``prepare`` with Harbor's Python. Preparation calls Harbor's actual dry-run
and native adapter preflight, never agent.run or a model. ``execute`` is a
separate, one-shot operation for use after the supervisor arm is qualified.
The original task instructions, environment and verifier remain untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

from benchmarks.agent_supervisor.container_coding import benchmark_controls


TASK = "fix-code-vulnerability"
MODEL = "gpt-5.6-sol"
CLI_VERSION = "0.158.0"
REASONING = "high"
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_name(value: str) -> str:
    """Accept one bounded dataset child name, never a path or glob."""
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError("task must be a simple dataset child name")
    return value


def _task_path(dataset: Path, task_name: str) -> Path:
    dataset = Path(dataset).absolute()
    task = dataset / _task_name(task_name)
    if (dataset.resolve(strict=True) != dataset or not dataset.is_dir()
            or task.is_symlink() or not task.is_dir() or task.resolve(strict=True) != task):
        raise ValueError("task must be a canonical non-symlink direct dataset child")
    return task


def _prepared_task(output: Path, prepared: dict, config: dict, *, job_prefix: str,
                   task_name: str | None = None) -> Path:
    """Recheck the selected task against the frozen Harbor configuration."""
    selected = _task_name(prepared.get("task", TASK))
    if task_name is not None and _task_name(task_name) != selected:
        raise ValueError("requested task differs from the prepared task")
    task = _task_path(Path(prepared["dataset"]), selected)
    tasks = config.get("tasks")
    if (_hash(output / "config.json") != prepared["config_sha256"]
            or type(tasks) is not list or len(tasks) != 1 or type(tasks[0]) is not dict
            or tasks[0].get("path") != str(task)
            or config.get("job_name") != job_prefix + selected
            or config.get("jobs_dir") != str(output / "jobs")):
        raise ValueError("prepared task binding differs from the frozen configuration")
    controls = prepared.get("comparison_controls")
    if controls is not None and (not benchmark_controls.validate_controls(controls)
            or controls["identity"]["task"] != selected):
        raise ValueError("prepared task differs from the declared comparison controls")
    return task


def _trial_task_matches(result: dict, task: Path) -> bool | None:
    """Do not turn absent historical trial identity into a matching task."""
    import tomllib
    public_config = tomllib.loads((task / "task.toml").read_text())
    expected_name = (public_config.get("task") or {}).get("name", task.name)
    config = result.get("config")
    task_config = config.get("task") if type(config) is dict else None
    observed = {"name": result.get("task_name"),
                "path": task_config.get("path") if type(task_config) is dict else None}
    expected = {"name": expected_name, "path": str(task)}
    if any(value is not None and value != expected[key] for key, value in observed.items()):
        return False
    return True if all(value is not None for value in observed.values()) else None


def _task_hashes(task: Path) -> dict:
    # Hash verifier inputs for integrity, but never inspect or send the oracle.
    if not task.is_dir() or task.is_symlink() or task.resolve() != task:
        raise ValueError("canonical regular task directory required")
    files = [task / "instruction.md", task / "task.toml"]
    for name in ("environment", "tests"):
        root = task / name
        if not root.is_dir() or root.is_symlink():
            raise ValueError("regular environment and verifier directories required")
        for path in root.rglob("*"):
            mode = path.lstat().st_mode
            if path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("task input inventory cannot omit links or special files")
            if stat.S_ISREG(mode):
                files.append(path)
    if any(path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
           or not path.resolve().is_relative_to(task) for path in files):
        raise ValueError("task inputs must be local regular files")
    return {path.relative_to(task).as_posix(): _hash(path) for path in sorted(files)}


def config_for(dataset: Path, output: Path, *, resource_profile=None, task_name: str = TASK) -> dict:
    from .benchmark_resource_profile import apply_resource_profile, execution_budget
    budget = execution_budget(resource_profile)
    task_name = _task_name(task_name)
    return apply_resource_profile({
        "job_name": "native-codex-" + task_name,
        "jobs_dir": str(output / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "force_build": True, "delete": True},
        "verifier": {"disable": False},
        "agents": [
            {
                "name": "codex",
                "model_name": MODEL,
                "override_timeout_sec": float(budget["harbor_seconds"]),
                "max_timeout_sec": float(budget["harbor_seconds"]),
                "override_setup_timeout_sec": 1800.0,
                "kwargs": {"version": CLI_VERSION, "reasoning_effort": REASONING},
                "env": {"CODEX_AUTH_JSON_PATH": str(Path.home() / ".codex/auth.json")},
            }
        ],
        "tasks": [{"path": str(dataset / task_name)}],
    }, resource_profile)


def prepare(*, dataset: Path, output: Path, harbor: Path | None = None, resource_profile=None,
            task_name: str = TASK) -> dict:
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.codex import Codex
    from harbor.models.job.config import JobConfig
    from harbor.models.task.task import Task

    dataset = Path(dataset).resolve(strict=True)
    task = _task_path(dataset, task_name)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("baseline preparation requires a fresh output directory")
    executable = str(harbor or Path(sys.executable).with_name("harbor"))
    if not Path(executable).is_file():
        raise ValueError("run preparation with the installed Harbor Python")
    cli = shutil.which("codex")
    if not cli:
        raise ValueError("the pinned local Codex CLI is unavailable")
    observed = subprocess.check_output([cli, "--version"], text=True, timeout=15).strip()
    if observed != "codex-cli " + CLI_VERSION:
        raise ValueError("local Codex CLI does not match the independently pinned version")
    original = Task(task)
    if original.has_steps:
        raise ValueError("baseline profile requires exactly one original task step")
    config = JobConfig.model_validate(config_for(dataset, output, resource_profile=resource_profile,
                                                 task_name=task_name), extra="forbid")
    AgentFactory.run_preflight(config.agents[0])
    adapter = AgentFactory.create_agent_from_config(
        config.agents[0], logs_dir=output / "adapter-preflight"
    )
    if type(adapter) is not Codex or adapter.version() != CLI_VERSION:
        raise ValueError("baseline must use Harbor's native pinned Codex adapter")
    if adapter.model_name != MODEL or adapter.build_cli_flags() != "-c model_reasoning_effort=high":
        raise ValueError("native Codex model/reasoning options differ")
    # Native supported resolver checks presence only; credentials are never read,
    # copied into the preparation artifacts, hashed, or put in process argv.
    auth = adapter._resolve_auth_json_path()
    if auth != Path.home() / ".codex/auth.json":
        raise ValueError("native auth resolver did not select the existing default auth file")
    hashes = _task_hashes(task)
    output.mkdir(parents=True)
    wire = config.model_dump(mode="json", context={"redact_sensitive_env": False})
    _json(output / "config.json", wire)
    command = [executable, "run", "--config", str(output / "config.json"), "--yes"]
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, timeout=60)
    (output / "native-dry-run.stdout").write_text(dry.stdout)
    (output / "native-dry-run.stderr").write_text(dry.stderr)
    if _task_hashes(task) != hashes:
        raise ValueError("original task changed during baseline preflight")
    source_files = [Path(inspect.getfile(item)) for item in (Codex, AgentFactory, JobConfig, Task)]
    record = {
        "resource_profile": resource_profile,
        "schema": "native-codex-harbor-baseline-preparation@1",
        "prepared": dry.returncode == 0,
        "native_dry_run_returncode": dry.returncode,
        "dataset": str(dataset),
        "task": task_name,
        "task_input_sha256": hashes,
        "harbor_version": version("harbor"),
        "host_codex_version": observed,
        "native_adapter": "harbor.agents.installed.codex:Codex",
        "native_adapter_source_sha256": {str(path): _hash(path) for path in source_files},
        "config_sha256": _hash(output / "config.json"),
        "collector_source_sha256": _hash(Path(__file__)),
        "controls_source_sha256": _hash(Path(benchmark_controls.__file__)),
        "comparison_controls": benchmark_controls.build_controls(
            wire, task_input_sha256=hashes, task=task_name, model=MODEL,
            reasoning_effort=REASONING, cli_version=CLI_VERSION),
        "model": MODEL,
        "reasoning_effort": REASONING,
        "cli_version": CLI_VERSION,
        "agent_timeout_seconds": wire["agents"][0]["override_timeout_sec"],
        "attempts": 1,
        "max_retries": 0,
        "concurrency": 1,
        "original_verifier_timeout_seconds": original.config.verifier.timeout_sec,
        "native_cli_flags": adapter.build_cli_flags(),
        "command": command,
        "auth": {
            "mechanism": "CODEX_AUTH_JSON_PATH",
            "existing_file_found": True,
            "credential_contents_recorded": False,
        },
        "provider_calls": 0,
        "container_cli_version_verified": False,
        "original_verifier_executed": False,
        "benchmark_advantage_claimed": False,
        "usage": {name: None for name in TOKEN_FIELDS},
        "official_codex_config_reference": "https://learn.chatgpt.com/docs/config-file/config-reference",
        "notes": [
            f"{wire['agents'][0]['override_timeout_sec']:g} seconds bounds the native agent execution phase; environment build, CLI install, and original verifier are measured separately.",
            "Container installation uses Harbor's native version pin and is not exercised by dry-run.",
            "This is one selected task and one trial, not a Terminal-Bench score or an advantage measurement.",
        ],
    }
    _json(output / "preparation.json", record)
    if dry.returncode:
        raise RuntimeError("native Harbor dry-run failed; see retained local diagnostics")
    return record


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def rollout_usage(agent: Path, *, known_redacted_literal: str | None = None) -> dict:
    """Keep each session's last observed counters and explicit completion.

    A structured native task_complete event can establish session completion;
    model text, verifier reward and process exit cannot. Malformed records or
    resumed/interrupted work leave complete usage unknown, while preserving
    the observed cumulative counters as lower bounds. Never sum snapshots.
    """
    _recover_known_literal("", known_redacted_literal)
    sessions = {}
    for path in sorted((agent / "sessions").rglob("rollout-*.jsonl")):
        identity, totals, count, malformed = None, None, 0, 0
        completed, completion_event_seen = False, False
        cli_version, observed_models, observed_efforts = None, set(), set()
        with path.open() as stream:
            for line in stream:
                try:
                    item = json.loads(_recover_known_literal(line, known_redacted_literal))
                except (ValueError, RecursionError):
                    malformed += 1
                    continue
                payload = item.get("payload") if isinstance(item, dict) else None
                if not isinstance(payload, dict):
                    malformed += 1
                    continue
                if item.get("type") == "session_meta":
                    observed_identity = payload.get("id") or payload.get("session_id")
                    if (not isinstance(observed_identity, str) or not observed_identity
                            or identity is not None and observed_identity != identity):
                        malformed += 1
                    elif identity is None:
                        identity = observed_identity
                    cli_version = payload.get("cli_version")
                if item.get("type") == "turn_context":
                    completed = False
                    if isinstance(payload.get("model"), str):
                        observed_models.add(payload["model"])
                    if isinstance(payload.get("effort"), str):
                        observed_efforts.add(payload["effort"])
                if item.get("type") != "event_msg":
                    continue
                event = payload.get("type")
                if event == "task_complete":
                    completed = completion_event_seen = True
                elif event in ("task_started", "turn_started", "turn_aborted", "task_aborted", "user_message"):
                    completed = False
                elif event == "token_count":
                    # New usage after an earlier completion requires another
                    # exact completion event before counters can be final.
                    completed = False
                    count += 1
                    info = payload.get("info")
                    if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                        totals = {
                            name: _integer(info["total_token_usage"].get(name))
                            for name in TOKEN_FIELDS
                        }
        identified = isinstance(identity, str) and bool(identity)
        if not identified:
            identity = "unidentified:" + path.relative_to(agent).as_posix()
        row = {
            "session_id": identity,
            "path": path.relative_to(agent).as_posix(),
            "sha256": _hash(path),
            "token_count_events": count,
            "malformed_lines": malformed,
            "task_complete_event_observed": completion_event_seen,
            "task_complete_observed": completed and identified and malformed == 0,
            "billing_total_verified": False,
            "cli_version": cli_version,
            "observed_models": sorted(observed_models),
            "observed_reasoning_efforts": sorted(observed_efforts),
            "usage": totals or {name: None for name in TOKEN_FIELDS},
        }
        if identity in sessions:
            if sessions[identity]["sha256"] != row["sha256"]:
                raise ValueError("different transcripts claim the same native session identity")
            continue
        sessions[identity] = row
    rows = list(sessions.values())
    usage = {
        name: sum(row["usage"][name] for row in rows)
        if rows and all(row["usage"][name] is not None for row in rows)
        else None
        for name in TOKEN_FIELDS
    }
    return {
        "source": "native_rollout_observed_cumulative_totals",
        "sessions": rows,
        "usage": usage,
        "task_complete_observed": bool(rows) and all(row["task_complete_observed"] for row in rows),
        "billing_total_verified": False,
        "reported_cost_usd": None,
        "unknown_fields_are_zero": False,
        "known_redacted_literal_restored": known_redacted_literal,
    }


def _recover_known_literal(text: str, literal: str | None) -> str:
    # Only this historical non-secret auth flag is recoverable. Never guess
    # or restore arbitrary credential values; original files stay untouched.
    if literal not in (None, "1"):
        raise ValueError("unsupported redaction recovery")
    return text.replace("[REDACTED]", "1") if literal == "1" else text


def _historical_redaction_literal(output: Path, prepared: dict) -> str | None:
    import tomllib
    config_path = output / "config.json"
    if _hash(config_path) != prepared["config_sha256"]:
        raise ValueError("baseline config changed before receipt collection")
    config = json.loads(config_path.read_text())
    env = config["agents"][0].get("env", {})
    if env != {"CODEX_FORCE_AUTH_JSON": "1", "CODEX_AUTH_JSON_PATH": ""}:
        return None
    task = tomllib.loads((_task_path(Path(prepared["dataset"]), prepared.get("task", TASK)) / "task.toml").read_text())
    # These are the complete sources used by Harbor0.23.0's trial scrubber.
    if (prepared.get("harbor_version") != "0.23.0"
            or (config.get("verifier") or {}).get("env")
            or (task.get("verifier") or {}).get("env")
            or config.get("user_agent") or config.get("user_agents")):
        raise ValueError("cannot establish the sole historical redaction value")
    return "1"


def _elapsed(record) -> float | None:
    if (
        not isinstance(record, dict)
        or not record.get("started_at")
        or not record.get("finished_at")
    ):
        return None
    return (
        datetime.fromisoformat(record["finished_at"].replace("Z", "+00:00"))
        - datetime.fromisoformat(record["started_at"].replace("Z", "+00:00"))
    ).total_seconds()


def collect(output: Path, *, task_name: str | None = None) -> dict:
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    selected_config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, selected_config, job_prefix="native-codex-", task_name=task_name)
    task_name = task.name
    from .benchmark_resource_profile import execution_budget
    budget = execution_budget(prepared.get("resource_profile"))
    recovered = _historical_redaction_literal(output, prepared)
    job = output / "jobs" / ("native-codex-" + task_name)
    rows = []
    for result_path in sorted(job.glob("*/result.json")):
        result = json.loads(_recover_known_literal(result_path.read_text(), recovered))
        config = result.get("config") or {}
        agent = config.get("agent") or {}
        task_matches = _trial_task_matches(result, task)
        profile_matches = (
            task_matches is True
            and agent.get("name") == "codex"
            and not agent.get("import_path")
            and agent.get("model_name") == MODEL
            and agent.get("kwargs") == {"version": CLI_VERSION, "reasoning_effort": REASONING}
            and agent.get("override_timeout_sec") == budget["harbor_seconds"]
            and agent.get("max_timeout_sec") == budget["harbor_seconds"]
            and not (config.get("verifier") or {}).get("disable", True)
        )
        rows.append(
            {
                "trial": result.get("trial_name"),
                "task": result.get("task_name"),
                "exact_trial_task_matches": task_matches,
                "exact_trial_profile_matches": profile_matches,
                "agent_info": result.get("agent_info"),
                "reward": (result.get("verifier_result") or {}).get("rewards"),
                "exception_type": (result.get("exception_info") or {}).get("exception_type"),
                "seconds": {
                    name: _elapsed(result.get(name))
                    for name in ("environment_setup", "agent_setup", "agent_execution", "verifier")
                },
                "total_seconds": _elapsed(result),
                "raw_usage": rollout_usage(result_path.parent / "agent", known_redacted_literal=recovered),
                "harbor_context": result.get("agent_result"),
                "result_sha256": _hash(result_path),
            }
        )
    integrity = _task_hashes(task) == prepared["task_input_sha256"]
    job_path = job / "result.json"
    job_stats = json.loads(job_path.read_text()).get("stats", {}) if job_path.is_file() else {}
    native_job_usage = {field: job_stats.get(key) for field, key in (
        ("input_tokens", "n_input_tokens"), ("cached_input_tokens", "n_cache_tokens"),
        ("output_tokens", "n_output_tokens"),
    )}
    corroborated = bool(rows) and all(
        value is not None
        and all(row["raw_usage"]["usage"].get(field) is not None for row in rows)
        and value == sum(row["raw_usage"]["usage"][field] for row in rows)
        for field, value in native_job_usage.items()
    )
    summary = {
        "schema": "native-codex-harbor-baseline-receipt@1",
        "task": task_name,
        "model": MODEL,
        "reasoning_effort": REASONING,
        "cli_version": CLI_VERSION,
        "original_task_inputs_unchanged": integrity,
        "comparison_controls": benchmark_controls.observe_controls(
            prepared, json.loads((output / "config.json").read_text()),
            current_task_hashes=_task_hashes(task)),
        "trial_count": len(rows),
        "trials": rows,
        "complete_single_trial_receipt": integrity
        and len(rows) == 1
        and rows[0]["exact_trial_profile_matches"]
        and (job / "result.json").is_file(),
        "benchmark_advantage_claimed": False,
        "collector_source_sha256": _hash(Path(__file__)),
        "native_job_usage": native_job_usage,
        "usage_corroborated_by_unscrubbed_native_job": corroborated,
        "native_job_sha256": _hash(job_path) if job_path.is_file() else None,
        "artifact_recovery": {
            "restored_nonsecret_flag": "CODEX_FORCE_AUTH_JSON=1" if recovered else None,
            "raw_artifacts_modified": False,
            "reason": "Harbor treated the boolean auth flag as a secret and redacted every digit 1" if recovered else None,
        },
        "cost_note": "Harbor context cost may be a LiteLLM estimate; native Codex totals do not report dollar cost.",
    }
    _json(output / "receipt.json", summary)
    return summary


def execute(output: Path, *, task_name: str | None = None) -> dict:
    output = Path(output).absolute()
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    task = _prepared_task(output, prepared, config, job_prefix="native-codex-", task_name=task_name)
    if not prepared["prepared"] or _hash(output / "config.json") != prepared["config_sha256"]:
        raise ValueError("prepared native baseline configuration changed")
    if _task_hashes(task) != prepared["task_input_sha256"]:
        raise ValueError("original benchmark task changed")
    if _hash(Path(__file__)) != prepared["collector_source_sha256"]:
        raise ValueError("prepared baseline driver changed")
    if _hash(Path(benchmark_controls.__file__)) != prepared.get("controls_source_sha256"):
        raise ValueError("prepared benchmark controls owner changed")
    if any(
        _hash(Path(path)) != digest
        for path, digest in prepared["native_adapter_source_sha256"].items()
    ):
        raise ValueError("native Harbor adapter changed after preflight")
    # An exclusive invocation marker prevents silently rerunning an expensive
    # or partially failed benchmark attempt under the same result directory.
    with (output / "invocation.json").open("x") as stream:
        json.dump(
            {"command": prepared["command"], "started_at": datetime.now().astimezone().isoformat()},
            stream,
        )
    started = time.monotonic()
    with (
        (output / "harbor.stdout").open("w") as stdout,
        (output / "harbor.stderr").open("w") as stderr,
    ):
        result = subprocess.run(prepared["command"], stdout=stdout, stderr=stderr)
    summary = collect(output)
    summary["harbor_returncode"] = result.returncode
    summary["invocation_seconds"] = time.monotonic() - started
    _json(output / "receipt.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "execute", "collect"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--task", help="Dataset task name; prepare defaults to fix-code-vulnerability")
    parser.add_argument("--harbor", type=Path)
    from .benchmark_resource_profile import PROFILES
    parser.add_argument("--resource-profile", choices=PROFILES)
    args = parser.parse_args()
    output = args.output.absolute()
    if args.operation == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset")
        result = prepare(dataset=args.dataset, output=output, harbor=args.harbor,
                         resource_profile=args.resource_profile,
                         task_name=TASK if args.task is None else args.task)
    else:
        result = {"execute": execute, "collect": collect}[args.operation](output, task_name=args.task)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

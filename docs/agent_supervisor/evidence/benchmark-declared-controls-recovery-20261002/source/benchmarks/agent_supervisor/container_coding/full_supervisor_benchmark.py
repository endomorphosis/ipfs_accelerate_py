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
    TASK, MODEL, CLI_VERSION, REASONING, _hash, _json, _task_hashes, config_for as baseline_config,
)
from benchmarks.agent_supervisor.container_coding import benchmark_controls

ADAPTER = "benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent"
INTENT_SOURCE_PATH = ".supervisor-instruction.md"
MAX_INTENT_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_INSTRUCTION_BYTES = 32768


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


def _load_intent_requirement_contract(path: Path | None, *, dataset: Path) -> dict | None:
    """Validate candidate requirements against the selected public task instruction."""
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
    public = Path(dataset).resolve(strict=True) / TASK / "instruction.md"
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
    strategy = ("intent_symbolic" if contract is not None and contract.get("schema") == "intent-plan-requirement-contract@2"
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
               *, intent_requirement_contract: dict | None = None) -> dict:
    if arm not in {"full", "no-index"}:
        raise ValueError("unknown supervisor ablation")
    config = baseline_config(dataset, output)
    config["job_name"] = "supervisor-" + arm + "-" + TASK
    config["agents"] = [{
        "import_path": ADAPTER, "model_name": MODEL,
        "override_timeout_sec": 300.0, "max_timeout_sec": 300.0,
        "override_setup_timeout_sec": 1800.0,
        "kwargs": {"runtime_archive": str(archive), "arm": arm,
                   "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"},
    }]
    if intent_requirement_contract is not None:
        config["agents"][0]["kwargs"]["intent_requirement_contract"] = _transport_intent_contract(intent_requirement_contract)
    return config


def prepare(*, dataset: Path, output: Path, archive: Path, arm: str,
            intent_requirement_contract: Path | None = None,
            intent_action_384_config: Path | None = None) -> dict:
    from harbor.models.job.config import JobConfig
    dataset = dataset.resolve(strict=True)
    archive = archive.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh trial output required")
    manifest = json.loads((archive / "manifest.json").read_text())
    if _hash(archive / "runtime.tar.gz") != manifest["archive_sha256"]:
        raise ValueError("runtime archive integrity failed")
    from .terminal_deployment import (
        load_intent_action_384_config, _intent_action_384_assets,
        validate_intent_action_384_binding, verify_intent_action_384_archive,
    )
    selected_intent = validate_intent_action_384_binding(manifest)
    verify_intent_action_384_archive(archive / "runtime.tar.gz", manifest)
    if intent_action_384_config is not None:
        host_config = load_intent_action_384_config(intent_action_384_config)
        _, expected_binding = _intent_action_384_assets(host_config)
        if selected_intent != expected_binding:
            raise ValueError("selected Intent384 config differs from the runtime archive")
    requirements = _load_intent_requirement_contract(intent_requirement_contract, dataset=dataset)
    task_hashes = _task_hashes(dataset / TASK)
    config = JobConfig.model_validate(config_for(dataset, output, archive, arm,
        intent_requirement_contract=requirements), extra="forbid")
    output.mkdir(parents=True)
    _json(output / "config.json", config.model_dump(mode="json", context={"redact_sensitive_env": False}))
    harbor = Path(sys.executable).with_name("harbor")
    command = [str(harbor), "run", "--config", str(output / "config.json"), "--yes"]
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, timeout=60)
    (output / "dry-run.stdout").write_text(dry.stdout)
    (output / "dry-run.stderr").write_text(dry.stderr)
    if _task_hashes(dataset / TASK) != task_hashes:
        raise ValueError("original task changed during supervisor preflight")
    sources = {str(path): _hash(path) for path in Path(__file__).parent.glob("*.py")}
    result = {"schema": "terminal-full-supervisor-preparation@1", "prepared": dry.returncode == 0,
              "arm": arm, "task": TASK, "dataset": str(dataset), "archive": str(archive),
              "archive_sha256": manifest["archive_sha256"], "manifest_sha256": _hash(archive / "manifest.json"),
              "host_source_sha256": sources,
              "task_input_sha256": task_hashes, "config_sha256": _hash(output / "config.json"),
              "comparison_controls": benchmark_controls.build_controls(
                  json.loads((output / "config.json").read_text()),
                  task_input_sha256=task_hashes, task=TASK, model=MODEL,
                  reasoning_effort=REASONING, cli_version=CLI_VERSION),
              "command": command, "model": MODEL, "reasoning_effort": REASONING, "cli_version": CLI_VERSION,
              "agent_timeout_seconds": 300, "provider_calls": 0,
              "planning_and_cold_index_charged_to_agent_time": True,
              "benchmark_advantage_claimed": False, **_intent_selection(requirements),
              "intent_action_384": selected_intent}
    _json(output / "preparation.json", result)
    if dry.returncode:
        raise RuntimeError("Harbor preflight failed; see retained logs")
    return result


def collect(output: Path) -> dict:
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    selection = {key: prepared.get(key, value) for key, value in _intent_selection(None).items()}
    configured_selection = _intent_selection(config["agents"][0].get("kwargs", {}).get("intent_requirement_contract"))
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
                       "reward": (native.get("verifier_result") or {}).get("rewards"),
                       "exception_type": (native.get("exception_info") or {}).get("exception_type"),
                       "durations_seconds": durations, "agent_context": native.get("agent_result"),
                       "supervisor": report})
    result = {"schema": "terminal-full-supervisor-receipt@1", "arm": prepared["arm"],
              "task": TASK, "model": MODEL, "reasoning_effort": REASONING, "cli_version": CLI_VERSION,
              "original_task_inputs_unchanged": _task_hashes(Path(prepared["dataset"]) / TASK) == prepared["task_input_sha256"],
              "comparison_controls": benchmark_controls.observe_controls(
                  prepared, config, current_task_hashes=_task_hashes(Path(prepared["dataset"]) / TASK)),
              "trials": trials, "trial_count": len(trials), "native_job_result_present": (job / "result.json").is_file(),
              "planning_and_cold_index_charged_to_agent_time": True, "benchmark_advantage_claimed": False,
              "parallel_workers": 1, "dollar_cost": None, **selection,
              "intent_action_384": prepared.get("intent_action_384"),
              "intent_selection_config_unchanged": configured_selection == selection}
    _json(output / "receipt.json", result)
    return result


def execute(output: Path) -> dict:
    prepared = json.loads((output / "preparation.json").read_text())
    if not prepared["prepared"] or _hash(output / "config.json") != prepared["config_sha256"]:
        raise ValueError("prepared configuration changed")
    if _task_hashes(Path(prepared["dataset"]) / TASK) != prepared["task_input_sha256"]:
        raise ValueError("original task changed")
    if _hash(Path(prepared["archive"]) / "runtime.tar.gz") != prepared["archive_sha256"]:
        raise ValueError("runtime archive changed")
    if _hash(Path(prepared["archive"]) / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("runtime dependency manifest changed")
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
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--arm", choices=("full", "no-index"), default="full")
    parser.add_argument("--intent-action-384-config", type=Path,
        help="Verify the explicit local Intent384 configuration matches the packaged assets")
    parser.add_argument("--intent-requirement-contract", type=Path,
        help="Use source-bound @1 provider coverage or @2 reviewed symbolic operations before admission")
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.dataset is None or args.archive is None:
            parser.error("prepare requires --dataset and --archive")
        result = prepare(dataset=args.dataset, output=args.output, archive=args.archive, arm=args.arm,
                         intent_requirement_contract=args.intent_requirement_contract,
                         intent_action_384_config=args.intent_action_384_config)
    else:
        result = {"execute": execute, "collect": collect}[args.operation](args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

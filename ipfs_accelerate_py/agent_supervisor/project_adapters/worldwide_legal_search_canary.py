"""Isolated, noncharging canary of native repair validation and promotion.

The executable provider is explicitly a deterministic provider_stub. Native
failure admission, Portal execution, immutable red/green validation, merge,
and task completion remain real. This does not certify model routing or an
operating service's restart/reload bindings.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any

DEFECT = "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_canary_defect.py"
BROKEN = 'def health():\n    return "broken"\n\ndef reproduce():\n    assert health() == "healthy", "intentional isolated canary defect"\n'
FIXED = BROKEN.replace('return "broken"', 'return "healthy"')
MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_canary"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(root: Path, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, check=check, timeout=60)


def _worker(spec_path: Path) -> dict[str, Any]:
    """Produce one exact fixture patch; never admit production workspaces."""
    spec = json.loads(spec_path.read_text())
    root = Path.cwd().resolve()
    fixture = Path(spec["fixture_root"]).resolve()
    state = Path(spec["state_root"]).resolve()
    if spec.get("schema") != "wwl.repair-canary-provider-stub@1" or not fixture.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("provider_stub requires a dedicated temporary fixture")
    if not root.is_relative_to(state / "repairs") or not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("provider_stub refuses a non-canary native worktree")
    if os.environ.get("IPFS_ACCELERATE_AGENT_TASK_ID") != spec["task_id"]:
        raise ValueError("provider_stub requires the exact native repair task")
    subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True, cwd=root)
    target = root / DEFECT
    if target.read_text() != BROKEN:
        raise ValueError("provider_stub baseline changed")
    target.write_text(FIXED)
    regression = root / spec["regression_path"]
    regression.parent.mkdir(parents=True, exist_ok=True)
    regression.write_text(
        "import pathlib, runpy\n\n"
        "def test_reproduced_canary_defect():\n"
        "    source = pathlib.Path(__file__).resolve().parents[2] / " + repr(DEFECT) + "\n"
        "    runpy.run_path(str(source))['reproduce']()\n"
    )
    evidence = root / spec["evidence_path"]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"schema": "wwl.repair-canary-provider-output@1", "provider_stub": True,
        "task_id": spec["task_id"], "action_id": spec["action_id"], "completion_authority": False}, sort_keys=True) + "\n")
    return {"schema": "wwl.repair-canary-provider-stub-result@1", "provider_stub": True, "completion_authority": False}


@contextlib.contextmanager
def _stub_provider_command(spec_path: Path, python: str):
    # Process-local dependency substitution is confined to this canary. No
    # production config, routing function source, or native gate is changed.
    from . import worldwide_legal_search_provider as provider
    original = provider.implementation_command
    worker = shlex.join([python, "-I", str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
    provider.implementation_command = lambda *_args, **_kwargs: worker
    try:
        yield
    finally:
        provider.implementation_command = original


def _native_proofs(state: Path, action_id: str, baseline: str, promoted: str) -> dict[str, Any]:
    """Retain real native gate identities instead of inferring from exit zero."""
    repair_state = state / "repairs" / action_id
    plan = json.loads((repair_state / "repair-plan.json").read_text())
    task = plan["tasks"][0]
    expected_command = shlex.join(task["validation_argv"][0])
    event_paths = list((repair_state / "native/wwl_database_portal_attempts").glob("*/portal-events.jsonl"))
    if len(event_paths) != 1 or event_paths[0].stat().st_size > 8 * 1024 * 1024:
        raise RuntimeError("canary requires one bounded retained native event stream")
    events = [json.loads(line) for line in event_paths[0].read_text().splitlines()]
    finishes = [row for row in events if row.get("type") == "implementation_finished" and row.get("task_id") == task["id"]]
    merges = [row for row in events if row.get("type") == "merge_finished" and row.get("merged") is True and row.get("merge_commit") == promoted]
    if len(finishes) != 1 or len(merges) != 1:
        raise RuntimeError("canary lacks unique native implementation and promotion evidence")
    finish, merge = finishes[0], merges[0]
    validation = finish.get("validation_result", {})
    dag = validation.get("validation_dag_receipt", {})
    nodes = dag.get("nodes", [])
    if finish.get("returncode") != 0 or validation.get("passed") is not True or dag.get("passed") is not True or not dag.get("receipt_id") or not any(
        node.get("command") == expected_command and node.get("mandatory") is True and node.get("selected") is True
        and node.get("disposition") == "succeeded" and node.get("returncode") == 0 and node.get("result_digest")
        for node in nodes
    ):
        raise RuntimeError("canary lacks the accepted mandatory immutable validator node")
    integration = finish.get("merge_result", {}).get("integration_commit_proof", {})
    if integration.get("passed") is not True or integration.get("integration_commit") != promoted:
        raise RuntimeError("canary lacks native accepted integration ancestry evidence")
    log = Path(finish["log_path"]).resolve()
    if not log.is_relative_to(repair_state.resolve()) or log.stat().st_size > 8 * 1024 * 1024:
        raise RuntimeError("canary validation log is outside retained bounded native state")
    reports = []
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get("schema") == "wwl.native-repair-validation@1":
                reports.append(value)
    if len(reports) != 1 or reports[0].get("action_id") != action_id or reports[0].get("baseline_revision") != baseline or not all(
        reports[0].get(key) is True for key in ("baseline_failure_verified", "candidate_regressions_verified")
    ):
        raise RuntimeError("canary lacks the bound independent red/green validator report")
    return {"native_merge_receipt": merge["event_id"], "native_validation_receipt": dag["receipt_id"],
        "native_implementation_event": finish["event_id"], "integration_commit_proof": integration,
        "red_green_validation": reports[0], "retained_event_stream": str(event_paths[0]),
        "retained_event_stream_sha256": digest(event_paths[0]), "retained_validation_log": str(log),
        "retained_validation_log_sha256": digest(log), "immutable_validation_command_sha256": hashlib.sha256(expected_command.encode()).hexdigest()}


def run_canary(*, parent_dir: Path | None = None, reference_config: Path | None = None) -> dict[str, Any]:
    """Keep implicit native catalogs inside the fixture, restoring caller env."""
    keys = ("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", "IPFS_ACCELERATE_AGENT_HOME")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        return _run_canary(parent_dir=parent_dir, reference_config=reference_config)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_canary(*, parent_dir: Path | None = None, reference_config: Path | None = None) -> dict[str, Any]:
    from . import worldwide_legal_search_runtime as native

    parent = (parent_dir or Path(tempfile.gettempdir())).resolve()
    if not parent.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("repair canary must run beneath the system temporary directory")
    if reference_config is None:
        raise ValueError("repair canary requires a read-only reference config for exact pipeline and policy binding")
    reference = native.config_from_args(native.parser().parse_args(["run", "--config", str(reference_config), "--once"]))
    pipeline_binding = native.repair_pipeline_binding(reference)
    reference_plan = native.read_json(reference.plan_path)
    reference_policy = native.read_json(reference.plan_path.with_name("watchdog.policy.json"))
    parent.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="wwl-native-repair-canary-", dir=parent))
    fixture, state = home / "fixture", home / "state"
    fixture.mkdir()
    state.mkdir(mode=0o700)
    os.environ["IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR"] = str(state / "orchestration")
    os.environ["IPFS_ACCELERATE_AGENT_HOME"] = str(state / "agent-home")
    (fixture / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n*.pyc\n")
    for path, expected in pipeline_binding["source_sha256"].items():
        original = reference.framework_root / path
        if digest(original) != expected:
            raise RuntimeError("pipeline source changed during canary admission")
        copied = fixture / path
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(original.read_bytes())
    started = time.time()
    run_id = "worldwide-legal-search-canary"
    plan_path = fixture / "implementation_plan/worldwide_legal_search/plan.json"
    objectives = plan_path.with_name("goals.objectives.md")
    source = fixture / DEFECT
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(BROKEN)
    task = {"id": "WWL-000", "title": "Intentional isolated canary task", "repository": "accelerate", "depends_on": [],
        "subgoal_id": "WWL-G000", "outputs": [DEFECT], "evidence_output": "artifacts/canary.json", "acceptance": "The isolated fixture defect is healthy.",
        "validation_argv": [[sys.executable, "-c", "raise SystemExit(0)"]], "bounds": {"max_attempts": 1, "wall_seconds_per_attempt": 600}}
    plan = {"schema": "worldwide-legal-search.plan@1", "namespace": run_id,
        "goals": [{"id": "WWL-G000", "title": "Repair the isolated fixture defect", "parent_ids": [], "task_ids": ["WWL-000"], "acceptance": "The exact isolated defect passes its independent regression."}],
        "tasks": [task], "execution": {"protected_acceptance_inputs": [str(plan_path.relative_to(fixture)), str(objectives.relative_to(fixture)), str(plan_path.with_name("watchdog.policy.json").relative_to(fixture))]},
        "routing": reference_plan["routing"]}
    native.atomic_json(plan_path, plan)
    objectives.write_text("# Isolated canary\n\n## WWL-G000 Repair the isolated fixture defect\n- Status: active\n- Parent goal ids json: []\n- Required task ids json: [\"WWL-000\"]\n")
    policy = reference_policy
    native.atomic_json(plan_path.with_name("watchdog.policy.json"), policy)
    command(fixture, "git", "init", "-b", "canary")
    command(fixture, "git", "config", "user.email", "canary@localhost")
    command(fixture, "git", "config", "user.name", "Isolated Repair Canary")
    command(fixture, "git", "add", ".")
    command(fixture, "git", "commit", "-m", "Admit intentional isolated repair canary defect")
    baseline = native.git(fixture, "rev-parse", "HEAD")
    config_path = state / "config.json"
    native.atomic_json(config_path, {"site_root": str(fixture), "framework_root": str(fixture), "state_root": str(state),
        "plan_path": str(plan_path), "python_executable": sys.executable, "run_id": run_id,
        "poll_seconds": 0.1, "task_timeout_seconds": 600, "max_task_attempts": 1, "repair_command": []})
    config = native.config_from_args(native.parser().parse_args(["run", "--config", str(config_path), "--once"]))
    with native.owner_lock(config):
        bootstrap = native.bootstrap(config, already_locked=True)
        runtime = native.CampaignRuntime(config, bootstrap)
        runtime.open()
        original = runtime._daemon.run_once
        try:
            # Explicit fault injection at the native pre-provider scheduling
            # boundary, captured by the unmodified native failure recorder.
            runtime._daemon.run_once = runpy.run_path(str(source))["reproduce"]
            runtime._run_pass()
            status = runtime.publish(alive=False)
        finally:
            runtime._daemon.run_once = original
            runtime._daemon.close()
    failure = status["native_error"]
    if failure.get("effects_unknown") is not False or failure.get("provider_callback_entered") or [frame["path"] for frame in failure.get("source_frames", [])] != [DEFECT]:
        raise RuntimeError("canary failed to capture an exact known-effects native defect")
    action_id = hashlib.sha256((failure["receipt_id"] + baseline).encode()).hexdigest()
    request = {"schema": "wwl.watchdog.action@1", "kind": "repair", "action_id": action_id, "run_id": run_id,
        "policy_id": policy["policy_id"], "authority_revision": status["authority_revision"], "accepted_tree": native.identity(native.observed_forest(config)),
        "issued_at": time.time(), "reproducer_receipt": failure["receipt_id"]}
    request_path = state / "control/requests" / (action_id + ".json")
    native.atomic_json(request_path, request)
    admission = native.admit_repair(config, request_path)
    task_id = "WWL-REPAIR-" + action_id[:16]
    spec_path = state / "provider-stub.json"
    native.atomic_json(spec_path, {"schema": "wwl.repair-canary-provider-stub@1", "fixture_root": str(fixture), "state_root": str(state),
        "action_id": action_id, "task_id": task_id, "regression_path": admission["regression_path"], "evidence_path": admission["evidence_path"]})
    previous_seal = os.environ.get("IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB")
    os.environ["IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"] = str(state / "canary-pytest-seal.duckdb")
    try:
        with _stub_provider_command(spec_path, sys.executable):
            result = native.repair_once(config, request_path)
    finally:
        if previous_seal is None:
            os.environ.pop("IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB", None)
        else:
            os.environ["IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"] = previous_seal
    promoted = native.git(fixture, "rev-parse", "HEAD")
    if not result["validated"] or not result.get("native_completion_receipt") or not result.get("native_projection") or promoted == baseline or source.read_text() != FIXED:
        raise RuntimeError("native repair canary did not validate, merge and record accepted completion: " + json.dumps(result, sort_keys=True))
    if command(fixture, "git", "status", "--porcelain").stdout.strip():
        raise RuntimeError("native repair canary left an uncommitted fixture tree")
    runpy.run_path(str(source))["reproduce"]()
    proofs = _native_proofs(state, action_id, baseline, promoted)
    code = Path(native.__file__).resolve()
    if native.repair_pipeline_binding(reference) != pipeline_binding or native.repair_pipeline_binding(config) != pipeline_binding:
        raise RuntimeError("native repair pipeline or policy changed during the canary")
    receipt = {"schema": "wwl.native-repair-canary@1", "provider_stub": True, "repair_canary_verified": True,
        "provider_kind": "deterministic-canary", "pipeline_binding": pipeline_binding,
        "baseline_failure_verified": True, "candidate_regressions_verified": True, "validated_promotion": True,
        "native_completion_receipt": result["native_completion_receipt"], "native_projection": result["native_projection"],
        "native_gate_evidence": proofs,
        "source_provenance": {"framework_head": native.git(reference.framework_root, "rev-parse", "HEAD"),
            "framework_tree": native.git(reference.framework_root, "rev-parse", "HEAD^{tree}"), "canary_code_sha256": digest(Path(__file__)),
            "reference_config_sha256": digest(reference_config), "source_sha256": pipeline_binding["source_sha256"]},
        "scope": "isolated native repair admission, red/green validation, Portal merge and accepted task receipt",
        "model_routing_verified": False, "service_reload_verified": False, "completion_authority": False,
        "native_orchestration_root": str(state / "orchestration"), "native_agent_home": str(state / "agent-home"),
        "fixture_root": str(fixture), "state_root": str(state), "baseline_revision": baseline, "promoted_revision": promoted,
        "runtime_source_sha256": digest(code), "validator_sha256": hashlib.sha256(native.REPAIR_VALIDATOR.encode()).hexdigest(),
        "canary_source_sha256": digest(Path(__file__)), "policy_sha256": digest(plan_path.with_name("watchdog.policy.json")),
        "native_failure_receipt": failure["receipt_id"], "native_repair_result": result,
        "limits": {"max_attempts": 1, "repair_max_attempts": 2, "wall_seconds_per_attempt": 600, "paid_provider_calls": 0},
        "started_at": started, "completed_at": time.time()}
    receipt["receipt_id"] = native.identity(receipt)
    native.atomic_json(state / "repair-canary-receipt.json", receipt)
    return receipt


def refresh_argv(python_executable: str, reference_config: Path) -> tuple[str, ...]:
    """A fresh process loads the promoted pipeline; this never performs reload."""
    return (str(python_executable), "-B", "-m", MODULE, "refresh", "--reference-config", str(reference_config))


def refresh_canary(reference_config: Path, *, force: bool = False) -> dict[str, Any]:
    """Refresh stale proof using the real isolated pipeline, without live DB I/O.

    Reload remains a separate mandatory fence. Callers must not wait for this
    optional capability refresh before reloading already promoted native code.
    A failed refresh leaves repair capability unverified, never fabricates a
    replacement receipt, and never edits the host configuration.
    """
    from . import worldwide_legal_search_runtime as native

    config = native.config_from_args(native.parser().parse_args(["run", "--config", str(reference_config), "--once"]))
    raw = native.read_json(reference_config)
    destination = Path(raw.get("repair_canary_receipt") or config.state_root / "repair-canary.json")
    if not destination.is_absolute() or destination.is_symlink():
        raise ValueError("canary receipt destination must be an absolute owned regular path")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent or destination.parent.stat().st_uid != os.getuid():
        raise ValueError("canary receipt directory ownership or path is unsafe")
    with (destination.parent / ".repair-canary-refresh.lock").open("a+") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("native repair canary refresh already has an owner") from error
        existing = native.verified_repair_canary(config)
        if existing is not None and not force:
            receipt, refreshed = existing, False
        else:
            receipt = run_canary(reference_config=reference_config)
            if receipt.get("pipeline_binding") != native.repair_pipeline_binding(config):
                raise RuntimeError("pipeline changed before canary refresh publication")
            # run_canary has already required actual mandatory validation,
            # accepted integration, native completion and retained evidence.
            native.atomic_json(destination, receipt)
            refreshed = True
        verified = native.verified_repair_canary(config)
        if verified is None or verified["receipt_id"] != receipt["receipt_id"]:
            raise RuntimeError("published canary does not verify against the current native pipeline")
    return {"schema": "wwl.native-repair-canary-refresh@1", "verified": True, "refreshed": refreshed,
        "provider_stub": True, "provider_kind": "deterministic-canary", "paid_provider_calls": 0,
        "receipt_path": str(destination), "receipt_id": verified["receipt_id"],
        "pipeline_binding": verified["pipeline_binding"], "native_completion_receipt": verified["native_completion_receipt"],
        "native_projection": verified["native_projection"], "service_reload_performed": False, "completion_authority": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "refresh"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "worker":
        result = _worker(args.spec)
    elif args.command == "refresh":
        if args.reference_config is None:
            parser.error("refresh requires --reference-config")
        result = refresh_canary(args.reference_config, force=args.force)
    else:
        result = run_canary(reference_config=args.reference_config)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

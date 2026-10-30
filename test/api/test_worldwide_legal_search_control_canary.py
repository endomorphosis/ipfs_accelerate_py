"""WWL-005: one small canary after the fault, blocker and restart suites.

The canary admits a native plan, executes the admitted Grok route runner,
checks the isolated defect red then green, merges that fix in git, fences a
dead worker, and promotes the repair. The readiness receipt binds the native
component bytes and the accepted revision. It does not authorize sustained
watchdog repair or corpus expansion, and it does not grant completion.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_runtime as native
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import (
    ADMITTED_ROOT_BREADTH,
    ADMISSION_SCHEMA,
    CANONICAL_ROUTING,
    ROOT_GOAL_ID,
    WorldwideRouteError,
    admitted_tiny_canary,
    admit_goals_and_tasks,
    run_admitted_tiny_canary,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_recovery import (
    RecoveryOwner,
    read_pid_birth,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_repair import (
    CANARY_SCHEMA,
    CANONICAL_PYTHON,
    PRIMARY_MODEL,
    PRIMARY_PROVIDER,
    REGRESSION_SCHEMA,
    ROUTE_POLICY,
    ROUTER_NAME,
    RepairSession,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/worldwide/control-canary.json"
RECEIPT_PATH = ROOT / "artifacts/worldwide-legal-search/receipts/WWL-005.json"
RUN = "wwl-control-canary"
SOURCE = "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_canary_defect.py"
REPRO = "test/api/test_worldwide_legal_search_control_canary_repro.py"
BROKEN = 'def health():\n    return "broken"\n'
FIXED = 'def health():\n    return "healthy"\n'
REPRO_TEXT = (
    "def test_reproduced_control_canary_defect():\n"
    "    import pathlib, runpy\n"
    "    source = pathlib.Path(__file__).resolve().parents[2] / " + repr(SOURCE) + "\n"
    "    assert runpy.run_path(str(source))['health']() == 'healthy', 'control_canary_defect'\n"
    "test_reproduced_control_canary_defect()\n"
)
ARGV = [CANONICAL_PYTHON, REPRO]
PROOFS = (
    "native_admission",
    "grok_execution",
    "independent_validation",
    "merge",
    "worker_death_recovery",
    "validated_code_repair_promotion",
)
SUMMARY = re.compile(r"=+ (.+) =+")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ControlCanaryError(RuntimeError):
    """The control canary is not allowed to open the next campaign phase."""


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str, strip: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git " + " ".join(args) + " failed")
    return completed.stdout.strip() if strip else completed.stdout


def component_sha256(paths: list[str]) -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in paths}


def pipeline_component_paths() -> list[str]:
    source = inspect.getsource(native.repair_pipeline_binding)
    return re.findall(r'"(ipfs_accelerate_py/[^"]+)"', source)


def adapter_component_paths() -> list[str]:
    directory = ROOT / "ipfs_accelerate_py/agent_supervisor/project_adapters"
    return sorted(path.relative_to(ROOT).as_posix() for path in directory.glob("worldwide_legal_search*.py"))


def accepted_git_identity() -> tuple[str, str]:
    revision = git(ROOT, "rev-parse", "HEAD")
    tree = git(ROOT, "rev-parse", "HEAD^{tree}")
    if COMMIT.fullmatch(revision) is None or COMMIT.fullmatch(tree) is None:
        raise ControlCanaryError("accepted revision is not a git object")
    return revision, tree


def run_predecessor_suites(contract: dict, seal_root: Path) -> dict[str, dict]:
    """Re-execute the named suites in a fresh seal catalog so a prior seal cannot skip them."""
    results = {}
    for suite in contract["predecessor_suites"]:
        seal = seal_root / (suite["name"] + ".duckdb")
        env = os.environ.copy()
        env["IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"] = str(seal)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", suite["test"], "-q", "--tb=line"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        output = completed.stdout + "\n" + completed.stderr
        summary = ""
        for line in reversed(output.splitlines()):
            match = SUMMARY.search(line)
            if match:
                summary = match.group(1)
                break
        passed = re.search(r"(\d+) passed", summary)
        results[suite["name"]] = {
            "returncode": completed.returncode,
            "summary": summary,
            "passed": int(passed.group(1)) if passed else 0,
            "skipped": "skipped" in summary,
            "failed": "failed" in summary or "error" in summary,
            "argv": list(suite["validation_argv"]),
        }
        if completed.returncode != 0 or results[suite["name"]]["failed"] or results[suite["name"]]["skipped"]:
            tail = "\n".join(output.splitlines()[-40:])
            raise AssertionError(suite["name"] + " suite did not pass:\n" + tail)
        if results[suite["name"]]["passed"] != suite["passed"]:
            raise AssertionError(suite["name"] + " passed " + summary + ", expected " + str(suite["passed"]))
    return results


def campaign_documents() -> tuple[dict, str, str]:
    acceptance = "Native current-tree receipts bind this contract"
    tasks, goals = [], []
    for index in range(ADMITTED_ROOT_BREADTH):
        task_id = f"WWL-{index:03d}"
        goal_id = f"WWL-G{(index + 1) * 10:03d}"
        tasks.append({
            "id": task_id,
            "title": "Control canary task " + task_id,
            "status": "done",
            "depends_on": [] if index == 0 else [f"WWL-{index - 1:03d}"],
            "repository": "site" if index % 2 == 0 else "accelerate",
            "goal_id": goal_id,
            "subgoal_id": goal_id,
            "outputs": [f"artifacts/worldwide-legal-search/receipts/{task_id}.json"],
            "evidence_output": f"artifacts/worldwide-legal-search/receipts/{task_id}.json",
            "acceptance": acceptance,
            "validation_argv": [["python3", "-c", "print('validated fixture')"]],
            "bounds": {"max_attempts": 3, "wall_seconds_per_attempt": 5400},
        })
        goals.append({
            "id": goal_id,
            "title": "Control canary child " + goal_id,
            "parent_ids": [ROOT_GOAL_ID],
            "task_ids": [task_id],
            "acceptance": acceptance,
        })
    plan = {
        "schema": "worldwide-legal-search.plan@1",
        "namespace": RUN,
        "goals": [{
            "id": ROOT_GOAL_ID,
            "title": "Control canary root",
            "parent_ids": [],
            "task_ids": [task["id"] for task in tasks],
            "acceptance": acceptance,
        }, *goals],
        "tasks": tasks,
    }
    heap = ["# Worldwide legal search goal bootstrap", ""]
    for goal in plan["goals"]:
        heap.extend([
            f"## {goal['id']} {goal['title']}",
            "- Status: active",
            "- Parent goal ids json: " + json.dumps(goal["parent_ids"]),
            "- Required task ids json: " + json.dumps(goal["task_ids"]),
            "- Acceptance: " + goal["acceptance"],
            "",
        ])
    markdown = "".join(f"## {task['id']} {task['title']}\n- Status: done\n" for task in tasks)
    return plan, "\n".join(heap), markdown


def execute_isolated(source_text: str) -> int:
    stage = Path(tempfile.mkdtemp(prefix="wwl-control-canary-validate-"))
    try:
        source = stage / SOURCE
        repro = stage / REPRO
        source.parent.mkdir(parents=True)
        repro.parent.mkdir(parents=True)
        source.write_text(source_text, encoding="utf-8")
        repro.write_text(REPRO_TEXT, encoding="utf-8")
        completed = subprocess.run(
            [CANONICAL_PYTHON, str(repro)],
            cwd=stage,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return completed.returncode
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        rows.append(path.relative_to(root).as_posix() + ":" + sha256_file(path))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def prepare_repair_repository(parent: Path) -> tuple[Path, str]:
    repo = parent / "repair-git"
    repo.mkdir()
    source = repo / SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(BROKEN, encoding="utf-8")
    git(repo, "init", "-b", "baseline")
    git(repo, "config", "user.email", "canary@localhost")
    git(repo, "config", "user.name", "Control Canary")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "add", SOURCE)
    git(repo, "commit", "-m", "Admit the isolated control canary defect")
    return repo, git(repo, "rev-parse", "HEAD")


def promote_merge(repo: Path, baseline: str) -> dict:
    git(repo, "checkout", "-b", "candidate")
    (repo / SOURCE).write_text(FIXED, encoding="utf-8")
    git(repo, "add", SOURCE)
    git(repo, "commit", "-m", "Repair the isolated control canary defect")
    git(repo, "checkout", "baseline")
    git(repo, "merge", "--no-ff", "candidate", "-m", "Promote the validated control canary repair")
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merged = git(repo, "show", commit + ":" + SOURCE, strip=False)
    if commit == baseline or COMMIT.fullmatch(commit) is None or merged != FIXED:
        raise RuntimeError("git merge did not promote the validated fix")
    return {
        "merged": True,
        "merge_commit": commit,
        "merge_receipt": digest(commit),
        "tree": digest(tree),
        "effects_unknown": False,
        "completion_authority": False,
    }


def red_green(worktree: Path) -> tuple[int, int, bool]:
    before = tree_digest(worktree)
    baseline_code = execute_isolated(BROKEN)
    candidate_code = execute_isolated((worktree / SOURCE).read_text(encoding="utf-8"))
    unchanged = tree_digest(worktree) == before
    return baseline_code, candidate_code, unchanged


def regression_report(context: dict) -> dict:
    baseline_code, candidate_code, unchanged = red_green(Path(context["worktree"]))
    passed = baseline_code != 0 and candidate_code == 0 and unchanged
    return {
        "schema": REGRESSION_SCHEMA,
        "baseline_failure_verified": baseline_code != 0,
        "candidate_passed": candidate_code == 0,
        "focused_passed": candidate_code == 0,
        "protected_hashes_unchanged": unchanged,
        "exit_status": 0 if passed else 1,
        "argv": list(context["reproducer_argv"]),
        "completion_authority": False,
    }


def canary_report(context: dict, *, promote: bool = False) -> dict:
    baseline_code, candidate_code, unchanged = red_green(Path(context["worktree"]))
    verified = baseline_code != 0 and candidate_code == 0 and unchanged
    return {
        "schema": CANARY_SCHEMA,
        "baseline_failure_verified": baseline_code != 0 and unchanged,
        "candidate_regressions_verified": candidate_code == 0 and unchanged,
        "repair_canary_verified": verified,
        "validated_promotion": promote,
        "paid_provider_calls": 0,
        "hosted_model_invocations": 0,
        "completion_authority": False,
    }


def recover_dead_worker() -> dict:
    process = subprocess.Popen(["/usr/bin/sleep", "30"])
    birth = read_pid_birth(process.pid)
    try:
        if birth["alive"] is not True or birth["start_ticks"] <= 0:
            raise RuntimeError("canary worker did not publish a kernel birth")
        owner = RecoveryOwner(RUN, probe=read_pid_birth)
        bound = owner.bind_owner(
            worker_id="canary-worker",
            pid=process.pid,
            birth=birth["start_ticks"],
            process_generation=1,
            lease_generation=1,
        )
        if bound["disposition"] != "owner_bound" or bound["completion_authority"] is not False:
            raise RuntimeError("live canary worker was not admitted")
    finally:
        process.kill()
        process.wait(timeout=5)
    deadline = time.time() + 5
    while read_pid_birth(process.pid)["alive"] and time.time() < deadline:
        time.sleep(0.05)
    if read_pid_birth(process.pid)["alive"]:
        raise RuntimeError("canary worker stayed alive after it was killed")
    identity = {
        "worker_id": "canary-worker",
        "pid": process.pid,
        "birth": birth["start_ticks"],
        "process_generation": 1,
        "lease_generation": 1,
    }
    decision = owner.recover({
        "action_id": "dead-worker",
        "observed_at": 100,
        "deadline_at": 200,
        "incident": {"task_id": "WWL-005", "stage": "task", "defect": {"failure_key": "worker_lost"}},
        "owner": identity,
        "claims": [{**identity, "claim_id": "claim-dead", "task_id": "WWL-005", "effect_id": "effect-dead"}],
        "effects": [{
            "effect_id": "effect-dead",
            "task_id": "WWL-005",
            "status": "known_none",
            "digest": digest("none"),
        }],
        "completion_authority": False,
    })
    class BirthThenDeath:
        def __init__(self) -> None:
            self.alive = True

        def __call__(self, pid: int) -> dict:
            if pid != process.pid:
                return {"pid": pid, "alive": False, "start_ticks": 0, "state": "dead"}
            if self.alive:
                return {"pid": pid, "alive": True, "start_ticks": birth["start_ticks"], "state": "alive"}
            return {"pid": pid, "alive": False, "start_ticks": 0, "state": "dead"}

    probe = BirthThenDeath()
    preserved = RecoveryOwner(RUN, probe=probe)
    preserved.bind_owner(
        worker_id="canary-worker",
        pid=process.pid,
        birth=birth["start_ticks"],
        process_generation=1,
        lease_generation=1,
    )
    probe.alive = False
    preserved_decision = preserved.recover({
        "action_id": "dead-unknown",
        "observed_at": 100,
        "deadline_at": 200,
        "incident": {"task_id": "WWL-005", "stage": "task", "defect": {"failure_key": "worker_lost"}},
        "owner": identity,
        "claims": [{**identity, "claim_id": "claim-unknown", "task_id": "WWL-005", "effect_id": "effect-unknown"}],
        "effects": [{
            "effect_id": "effect-unknown",
            "task_id": "WWL-005",
            "status": "unknown",
            "digest": digest("in-flight"),
        }],
        "runnable_task_ids": ["WWL-006"],
        "idle_since": 10,
        "queue_deadline_at": 50,
        "completion_authority": False,
    })
    return {"recovered": decision, "unknown": preserved_decision, "pid": process.pid, "birth": birth["start_ticks"]}


def bind_readiness(contract: dict, proofs: dict[str, bool], components: dict[str, str], revision: str, tree: str) -> dict:
    if list(proofs) != list(PROOFS) or any(proofs[name] is not True for name in PROOFS):
        raise ControlCanaryError("control canary proofs are incomplete")
    if components != contract["component_sha256"]:
        raise ControlCanaryError("native component binding does not match the accepted bytes")
    if COMMIT.fullmatch(revision) is None or COMMIT.fullmatch(tree) is None:
        raise ControlCanaryError("accepted revision is not a git object")
    if contract["sustained_watchdog_repair_authorized"] is not False or contract["corpus_expansion_authorized"] is not False:
        raise ControlCanaryError("control canary contract already authorizes scaling")
    return {
        "schema": contract["readiness_schema"],
        "task_id": "WWL-005",
        "control_canary_verified": True,
        "accepted_revision": revision,
        "accepted_tree": tree,
        "component_sha256": dict(components),
        "proofs": {name: True for name in PROOFS},
        "hosted_model_invocations": 0,
        "paid_provider_calls": 0,
        "sustained_watchdog_repair_authorized": False,
        "corpus_expansion_authorized": False,
        "host_publish_readiness_unchanged": True,
        "host_control_canary_verified": False,
        "host_bootstrap_only": True,
        "completion_authority": False,
        "execution_completed": False,
    }


def authorize_next_phase(readiness: dict) -> dict:
    if readiness.get("control_canary_verified") is not True:
        raise ControlCanaryError("readiness receipt is not verified")
    if readiness.get("sustained_watchdog_repair_authorized") is not False or readiness.get("corpus_expansion_authorized") is not False:
        raise ControlCanaryError("canary readiness cannot authorize sustained repair or corpus expansion")
    if readiness.get("completion_authority") is not False or readiness.get("execution_completed") is not False:
        raise ControlCanaryError("canary readiness cannot grant completion authority")
    required = set(PROOFS)
    if set(readiness.get("proofs") or {}) != required or any(readiness["proofs"][name] is not True for name in required):
        raise ControlCanaryError("readiness receipt is missing a canary proof")
    if not readiness.get("component_sha256") or not readiness.get("accepted_revision") or not readiness.get("accepted_tree"):
        raise ControlCanaryError("readiness receipt does not bind native components and the accepted revision")
    return {"sustained_watchdog_repair": False, "corpus_expansion": False, "completion_authority": False}


def run_small_canary(parent: Path) -> dict:
    contract = load_contract()
    plan, heap, markdown = campaign_documents()
    forest = {
        "site": {"path": str(parent / "site"), "head": "1" * 40, "tree": "2" * 40},
        "accelerate": {"path": str(parent / "framework"), "head": "3" * 40, "tree": "4" * 40},
    }
    database = parent / "control.duckdb"
    board = parent / "tasks.md"
    board.write_text(markdown, encoding="utf-8")
    before_board = board.read_bytes()
    admission = admit_goals_and_tasks(
        plan, heap, database_path=database, source_forest=forest,
        config_path=parent / "runtime.json", python_executable=CANONICAL_PYTHON, task_markdown=markdown,
    )
    replay = admit_goals_and_tasks(
        plan, heap, database_path=database, source_forest=forest,
        config_path=parent / "runtime.json", python_executable=CANONICAL_PYTHON, task_markdown=markdown,
    )
    if board.read_bytes() != before_board:
        raise AssertionError("native admission rewrote the bootstrap markdown")
    workspace = parent / "candidate"
    state = parent / "route-state"
    workspace.mkdir()
    state.mkdir()
    (workspace / "seed.txt").write_text("seed\n", encoding="utf-8")
    grok = run_admitted_tiny_canary(admitted_tiny_canary(), workspace=workspace, state_root=state, python_executable=sys.executable)
    with pytest.raises(WorldwideRouteError, match="already ran"):
        run_admitted_tiny_canary(admitted_tiny_canary(), workspace=workspace, state_root=state, python_executable=sys.executable)
    recovery = recover_dead_worker()
    repo, baseline = prepare_repair_repository(parent)
    holder: dict = {"merge": None}
    campaign = parent / "campaign"
    live = parent / "live"
    campaign.mkdir()
    live.mkdir()
    (live / "sentinel.txt").write_text("live\n", encoding="utf-8")

    def merger(context: dict) -> dict:
        if context["baseline_revision"] != baseline:
            raise RuntimeError("repair baseline is not the git revision")
        if (Path(context["worktree"]) / SOURCE).read_text(encoding="utf-8") != FIXED:
            raise RuntimeError("repair worktree does not contain the validated fix")
        holder["merge"] = promote_merge(repo, baseline)
        return holder["merge"]

    def reloader(context: dict) -> dict:
        commit = context["merge"]["merge_commit"]
        if git(repo, "rev-parse", "HEAD") != commit or git(repo, "show", commit + ":" + SOURCE, strip=False) != FIXED:
            raise RuntimeError("reloaded lane is not the merged fix")
        return {"reloaded": True, "fenced": True, "lane_revision": commit, "completion_authority": False}

    session = RepairSession(RUN, regression_runner=regression_report, canary_runner=canary_report, merger=merger, reloader=reloader)
    decision = session.repair({
        "action_id": "control-canary-repair",
        "observed_at": 100,
        "deadline_at": 500,
        "failure": {
            "schema": "wwl.native-failure@1",
            "run_id": RUN,
            "task_id": "WWL-005",
            "stage": "task",
            "defect": {"failure_key": "control_canary_defect", "message": "isolated canary defect"},
            "effects_unknown": False,
            "source_revision": baseline,
            "reproducer": {"path": REPRO, "argv": list(ARGV)},
            "source_frames": [{"path": SOURCE, "source_sha256": hashlib.sha256(BROKEN.encode("utf-8")).hexdigest()}],
            "baseline_files": {SOURCE: BROKEN},
            "receipt_id": digest("control-canary-failure"),
            "pid": 1,
            "attempt": 1,
        },
        "campaign_root": str(parent / "campaign"),
        "live_checkout": str(live),
        "plan": {"schema": "worldwide-legal-search.plan@1", "routing": json.loads(json.dumps(CANONICAL_ROUTING))},
        "provider": {
            "grok_bin": "/usr/bin/grok",
            "codex_bin": "/usr/bin/codex",
            "codex_model": "deployment-configured-model",
            "codex_reasoning_effort": "high",
        },
        "python_executable": CANONICAL_PYTHON,
        "edits": [
            {"path": REPRO, "role": "reproducer", "text": REPRO_TEXT},
            {"path": SOURCE, "role": "source", "text": FIXED},
        ],
        "completion_authority": False,
    })
    revision, tree = accepted_git_identity()
    components = component_sha256(sorted(contract["component_sha256"]))
    proofs = {
        "native_admission": (
            admission["schema"] == ADMISSION_SCHEMA
            and admission["task_count"] == ADMITTED_ROOT_BREADTH
            and admission["ready_task_ids"] == ["WWL-000"]
            and admission["markdown_bootstrap_only"] is True
            and admission["markdown_is_execution_authority"] is False
            and admission["execution_completed"] is False
            and admission["completion_authority"] is False
            and replay["idempotent_replay"] is True
            and replay["task_cids_by_alias"] == admission["task_cids_by_alias"]
        ),
        "grok_execution": (
            grok["schema"] == contract["grok_execution"]["admitted_canary_schema"]
            and grok["decision"] == "fallback"
            and grok["failure_kind"] == "launch_failure"
            and grok["primary_model"] == PRIMARY_MODEL
            and grok["paid_provider_calls"] == 0
            and grok["hosted_model_invocations"] == 0
            and grok["probe_route_readiness"] is False
            and grok["workspace_unchanged"] is True
            and grok["completion_authority"] is False
            and decision["route"]["router"] == ROUTER_NAME
            and decision["route"]["primary_provider"] == PRIMARY_PROVIDER
            and decision["route"]["primary_model"] == PRIMARY_MODEL
            and decision["route"]["fallback_policy"] == ROUTE_POLICY
            and decision["route"]["launched"] is False
            and Path(decision["route"]["argv"][1]).name == "provider_fallback_runner.py"
        ),
        "independent_validation": (
            decision["regression"]["baseline_failure_verified"] is True
            and decision["regression"]["candidate_passed"] is True
            and decision["regression"]["exit_status"] == 0
            and decision["regression"]["argv"] == ARGV
            and decision["canary"]["repair_canary_verified"] is True
            and decision["canary"]["validated_promotion"] is False
            and decision["canary"]["paid_provider_calls"] == 0
        ),
        "merge": (
            decision["merge"]["merged"] is True
            and decision["merge"]["effects_unknown"] is False
            and decision["merge"]["merge_commit"] != baseline
            and decision["merge"]["merge_commit"] == holder["merge"]["merge_commit"]
        ),
        "worker_death_recovery": (
            recovery["recovered"]["disposition"] == "dead_worker"
            and recovery["recovered"]["dead_worker"] is True
            and recovery["recovered"]["generation_advanced"] is True
            and recovery["recovered"]["effects_unknown"] is False
            and recovery["recovered"]["owner_live"] is False
            and recovery["recovered"]["completion_authority"] is False
            and recovery["unknown"]["disposition"] == "effects_preserved"
            and recovery["unknown"]["effects_unknown"] is True
            and recovery["unknown"]["wake_authorized"] is False
            and "WWL-005" not in recovery["unknown"]["runnable_task_ids"]
        ),
        "validated_code_repair_promotion": (
            decision["disposition"] == "reloaded"
            and decision["promoted"] is True
            and decision["reloaded"] is True
            and decision["fenced"] is True
            and decision["completion_authority"] is False
            and decision["rollback"]["promoted_revision"] == decision["merge"]["merge_commit"]
            and decision["rollback"]["baseline_revision"] == baseline
            and (live / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
            and not (live / SOURCE).exists()
        ),
    }
    readiness = bind_readiness(contract, proofs, components, revision, tree)
    return {
        "admission": admission,
        "grok": grok,
        "recovery": recovery,
        "repair": decision,
        "baseline": baseline,
        "proofs": proofs,
        "readiness": readiness,
        "components": components,
        "revision": revision,
        "tree": tree,
    }


@pytest.fixture(scope="module")
def predecessor(tmp_path_factory):
    return run_predecessor_suites(load_contract(), tmp_path_factory.mktemp("seals"))


@pytest.fixture(scope="module")
def canary(tmp_path_factory):
    return run_small_canary(tmp_path_factory.mktemp("canary"))


def test_fault_blocker_and_restart_suites_pass(predecessor):
    contract = load_contract()
    assert [suite["name"] for suite in contract["predecessor_suites"]] == ["fault", "blocker", "restart"]
    for suite in contract["predecessor_suites"]:
        result = predecessor[suite["name"]]
        assert result["returncode"] == 0
        assert result["passed"] == suite["passed"]
        assert result["failed"] is False
        assert result["skipped"] is False
        receipt = json.loads((ROOT / suite["receipt"]).read_text(encoding="utf-8"))
        assert receipt["task_id"] == suite["task_id"]
        assert receipt["completion_authority"] is False
        assert receipt["self_certified_completion"] is False
        assert receipt["execution_completed"] is False
        assert receipt["validation_argv"] == suite["validation_argv"]
    for dependency in contract["dependency_receipts"]:
        receipt = json.loads((ROOT / dependency["receipt"]).read_text(encoding="utf-8"))
        assert receipt["task_id"] == dependency["task_id"]
        assert receipt["completion_authority"] is False
        assert receipt["validation_argv"] == dependency["validation_argv"]


def test_small_canary_proves_admission_grok_validation_merge_recovery_and_promotion(predecessor, canary):
    assert all(predecessor[name]["passed"] > 0 for name in ("fault", "blocker", "restart"))
    assert canary["proofs"] == {name: True for name in PROOFS}
    route = canary["repair"]["route"]
    primary = json.loads(route["argv"][route["argv"].index("--primary-command-json") + 1])
    assert primary[primary.index("--model") + 1] == "grok-4.7"
    assert primary[primary.index("--mode") + 1] == "agent"
    assert Path(primary[1]).name == "grok_cli_runner.py"
    assert canary["grok"]["hosted_model_invocations"] == 0
    assert canary["grok"]["fallback_executable"] == "admitted-local-canary"
    assert canary["admission"]["root_goal_id"] == ROOT_GOAL_ID
    assert canary["admission"]["admitted_root_breadth"] == ADMITTED_ROOT_BREADTH
    assert canary["recovery"]["recovered"]["process_generation"] == 2
    assert canary["repair"]["budget"]["repair"]["spent"] == 1
    assert canary["repair"]["budget"]["validation"]["spent"] == 1
    assert canary["repair"]["budget"]["merge"]["spent"] == 1
    assert canary["repair"]["edit_order"] == [REPRO, SOURCE]


def test_readiness_binds_every_component_and_refuses_scaling(canary):
    contract = load_contract()
    readiness = canary["readiness"]
    assert readiness["schema"] == "wwl.control-canary.readiness@1"
    assert readiness["control_canary_verified"] is True
    assert readiness["accepted_revision"] == canary["revision"]
    assert readiness["accepted_tree"] == canary["tree"]
    assert readiness["component_sha256"] == contract["component_sha256"] == canary["components"]
    assert set(pipeline_component_paths()) <= set(readiness["component_sha256"])
    assert set(adapter_component_paths()) <= set(readiness["component_sha256"])
    assert "ipfs_accelerate_py/llm_router.py" in readiness["component_sha256"]
    assert "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py" in readiness["component_sha256"]
    assert "ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py" in readiness["component_sha256"]
    phase = authorize_next_phase(readiness)
    assert phase == {"sustained_watchdog_repair": False, "corpus_expansion": False, "completion_authority": False}
    assert contract["blocked_until_this_canary"] == ["WWL-010"]
    assert contract["sustained_watchdog_repair_authorized"] is False
    assert contract["corpus_expansion_authorized"] is False
    assert contract["hosted_model_invocations"] == 0
    assert contract["paid_provider_calls"] == 0
    source = inspect.getsource(native.CampaignRuntime.publish_readiness)
    assert '"control_canary_verified": False' in source
    assert '"bootstrap_only": True' in source
    assert readiness["host_control_canary_verified"] is False
    assert readiness["host_bootstrap_only"] is True


def test_incomplete_binding_cannot_authorize_scaling():
    contract = load_contract()
    components = component_sha256(sorted(contract["component_sha256"]))
    revision, tree = accepted_git_identity()
    proofs = {name: True for name in PROOFS}
    proofs["merge"] = False
    with pytest.raises(ControlCanaryError, match="incomplete"):
        bind_readiness(contract, proofs, components, revision, tree)
    drifted = dict(components)
    drifted[next(iter(drifted))] = "0" * 64
    with pytest.raises(ControlCanaryError, match="component binding"):
        bind_readiness(contract, {name: True for name in PROOFS}, drifted, revision, tree)
    readiness = bind_readiness(contract, {name: True for name in PROOFS}, components, revision, tree)
    readiness["corpus_expansion_authorized"] = True
    with pytest.raises(ControlCanaryError, match="cannot authorize"):
        authorize_next_phase(readiness)
    readiness["corpus_expansion_authorized"] = False
    readiness["completion_authority"] = True
    with pytest.raises(ControlCanaryError, match="completion authority"):
        authorize_next_phase(readiness)


def test_canary_cannot_promote_itself(tmp_path: Path):
    parent = tmp_path
    (parent / "campaign").mkdir()
    (parent / "live").mkdir()
    (parent / "live" / "sentinel.txt").write_text("live\n", encoding="utf-8")
    repo, baseline = prepare_repair_repository(parent)

    def merger(context: dict) -> dict:
        return promote_merge(repo, baseline)

    session = RepairSession(
        RUN,
        regression_runner=regression_report,
        canary_runner=lambda context: canary_report(context, promote=True),
        merger=merger,
        reloader=lambda context: {"reloaded": True, "fenced": True, "lane_revision": context["merge"]["merge_commit"], "completion_authority": False},
    )
    decision = session.repair({
        "action_id": "self-promote",
        "observed_at": 100,
        "deadline_at": 500,
        "failure": {
            "schema": "wwl.native-failure@1",
            "run_id": RUN,
            "task_id": "WWL-005",
            "stage": "task",
            "defect": {"failure_key": "control_canary_defect", "message": "isolated canary defect"},
            "effects_unknown": False,
            "source_revision": baseline,
            "reproducer": {"path": REPRO, "argv": list(ARGV)},
            "source_frames": [{"path": SOURCE, "source_sha256": hashlib.sha256(BROKEN.encode("utf-8")).hexdigest()}],
            "baseline_files": {SOURCE: BROKEN},
            "receipt_id": digest("self-promote"),
        },
        "campaign_root": str(parent / "campaign"),
        "live_checkout": str(parent / "live"),
        "plan": {"schema": "worldwide-legal-search.plan@1", "routing": json.loads(json.dumps(CANONICAL_ROUTING))},
        "provider": {
            "grok_bin": "/usr/bin/grok",
            "codex_bin": "/usr/bin/codex",
            "codex_model": "deployment-configured-model",
            "codex_reasoning_effort": "high",
        },
        "python_executable": CANONICAL_PYTHON,
        "edits": [
            {"path": REPRO, "role": "reproducer", "text": REPRO_TEXT},
            {"path": SOURCE, "role": "source", "text": FIXED},
        ],
        "completion_authority": False,
    })
    assert decision["disposition"] == "canary_failed"
    assert decision["promoted"] is False
    assert decision["completion_authority"] is False
    assert git(repo, "rev-parse", "HEAD") == baseline


def test_receipt_records_the_canary_without_completion_authority():
    contract = load_contract()
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "worldwide-legal-search.task-receipt@1"
    assert payload["task_id"] == "WWL-005"
    assert payload["title"] == contract["title"]
    assert payload["namespace"] == "worldwide-legal-search-v1"
    assert payload["root_goal_id"] == "WWL-G000"
    assert payload["goal_id"] == "WWL-G010"
    assert payload["subgoal_id"] == "WWL-G010.S2"
    assert payload["depends_on"] == ["WWL-004", "WWL-105"]
    assert payload["predecessor_suites"] == ["fault", "blocker", "restart"]
    assert payload["proofs"] == list(PROOFS)
    assert payload["control_canary"] == "docs/worldwide/control-canary.json"
    assert payload["hosted_model_invocations"] == 0
    assert payload["paid_provider_calls"] == 0
    assert payload["sustained_watchdog_repair_authorized"] is False
    assert payload["corpus_expansion_authorized"] is False
    assert payload["execution_completed"] is False
    assert payload["completion_authority"] is False
    assert payload["self_certified_completion"] is False
    assert payload["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_control_canary.py", "-q"]
    assert payload["outputs"] == [
        "docs/worldwide/control-canary.json",
        "test/api/test_worldwide_legal_search_control_canary.py",
        "artifacts/worldwide-legal-search/receipts/WWL-005.json",
    ]
    assert "does not grant campaign completion" in " ".join(payload["limitations"])
    assert "hosted" in " ".join(payload["limitations"]).lower()
    assert contract["component_sha256"]
    assert contract["host_publish_readiness"] == {"control_canary_verified": False, "bootstrap_only": True}

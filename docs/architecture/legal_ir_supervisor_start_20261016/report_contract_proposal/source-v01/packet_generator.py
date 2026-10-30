"""Root-owned real report-contract packet proposal; no accepting/dispatch CLI.

Importing this module imports stdlib only and writes nothing. The public
prepare_real_packet function requires a genuine native preparation owner and
actual pinned host observations. It returns bytes for a separately charged
materializer; it never edits a board, imports a live database, resets history,
dispatches a provider, writes reviewer evidence or issues acceptance.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ACCELERATOR = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
OLD_PREFIX = "docs/architecture/legal_ir_supervisor_start_20261016"
NEW_PREFIX = OLD_PREFIX + "/real_report_contract_v01"
OLD_ROOT = ACCELERATOR / OLD_PREFIX
ADAPTER = OLD_ROOT / "factual_acceptance/adapter-v02"
EVIDENCE_ROOT = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs")
TASK = "LIR-9001"
REPORT = "data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json"
CANDIDATE_SCHEMA = "legal-ir-source-factual-candidate@1"
RETRY_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
)
NATIVE_SOURCES = tuple("ipfs_datasets_py/optimizers/logic_theorem_optimizer/" + name + ".py" for name in (
    "autoencoder_daemon_resources", "resource_scheduler", "existing_resource_scheduler",
    "parented_phase_resources", "validation_run_resources"))
MISSING_REAL_BACKEND_FUNCTIONS = (
    "native-regression/v03/native_fixture_roundtrip.py:prepare_isolated_fixture synthesizes fixture provider/context receipts; a real dispatch/outcome/context observer and report-only candidate commit freezer are absent",
    "native-regression/v04/docker_fixture_roundtrip.py:prepare_docker_fixture rejects nonfixture CID/namespace and writes a fixture-scoped inner request; real-context preparation is absent",
    "native-regression/v04/docker_fixture_roundtrip.py:consume_or_publish_fixture supports only fixture board publication; real board/context handoff is not integrated",
    "native-regression/v04/entry.py:main stages only synthetic fixture inputs; no real provider/candidate/native coordinator entry is implemented",
)
OBJECTIVE = ("Read the exact protected source packet and unaccepted candidate template. Produce only the declared "
    "report using legal-ir-source-factual-candidate@1, preserving every source commitment, derived fact, closed "
    "finding, unresolved scope item, false authority field and unactivated LIR-9002. This is a source-only worker "
    "report pending independent host acceptance; do not import product code, run tests, execute native owner or "
    "coordinator APIs, modify source/status/counters, create reviewer receipts or claim legal/training/proof authority. "
    "Standalone check-report deliberately exits3; retain the actual report for the distinct native host handoff.")


class PacketRefusal(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise PacketRefusal(reason)


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(PacketRefusal("nonfinite JSON value")))


def read(path, maximum=4 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and ".." not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), "canonical unaliased input required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
             "bounded regular single-link input required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(witness(before) == witness(after) == witness(path.lstat()) and len(raw) == before.st_size,
         "input changed while reading")
    return raw


def reject_identity_shortcuts(task):
    forbidden = {"canonical task key", "canonical task cid", "dedupe key"}
    for mapping in (task, task.get("metadata", {})):
        need(type(mapping) is dict, "task metadata mapping required")
        normalized = {str(key).strip().lower().replace("_", " ") for key in mapping}
        need(not normalized & forbidden, "provided canonical identity or dedupe shortcut forbidden")


def transform_projection(projection, config):
    """Pure narrow transformation. It neither authenticates inputs nor writes."""
    projected, configured = copy.deepcopy(projection), copy.deepcopy(config)
    tasks = projected.get("tasks")
    need(type(tasks) is list and len(tasks) == 64 and sum(t.get("task_id") == TASK for t in tasks) == 1,
         "exact64-task frontier required")
    need(len(projected.get("goal_hierarchy", [])) == 34
         and len(projected.get("legacy_parent_crosswalk", [])) == 24
         and sum(len(t["depends_on"]) for t in tasks) == 115, "original graph population differs")
    for task in tasks:
        reject_identity_shortcuts(task)
        need(task.get("status") == ("todo" if task["task_id"] == TASK else "blocked"), "frontier already changed")
    selected = next(t for t in tasks if t["task_id"] == TASK)
    selected["objective"] = OBJECTIVE
    selected["validation"] = ["python3 -B " + NEW_PREFIX + "/check_report.py --check-report " + REPORT]
    selected["inputs"]["references"] = [NEW_PREFIX + "/" + name for name in (
        "authoring/tasks.json", "authoring/source_baseline.json", "source-copy-manifest.json",
        "CURRENT_SOURCE_BINDINGS.json", "unaccepted-candidate-template.json", "REPORT_CONTRACT.md")]
    projected["bootstrap_install_prefix"] = NEW_PREFIX
    configured["max_task_attempts"] = 3
    configured["preserve_task_attempts"] = True
    configured["taskboard_path"] = NEW_PREFIX + "/tasks.todo.md"
    configured["validator_path"] = NEW_PREFIX + "/check_report.py"
    configured["status"] = "prepared_real_report_contract_pending_host_adoption"
    configured["bootstrap_execution_contract"]["launch_started"] = False
    configured["bootstrap_execution_contract"]["native_grant_supplied"] = False
    need(configured["board_namespace"] == config["board_namespace"] and configured["runtime_paths"] == config["runtime_paths"],
         "history namespace or runtime paths changed")
    return projected, configured


def transform_board(raw, selected):
    text = raw.decode("utf-8")
    parts = re.split(r"(?m)^(## LIR-\d{4} .*)$", text)
    need(len(parts) == 129, "exact64-block board required")
    count = 0
    for index in range(1, len(parts), 2):
        if parts[index].split()[1] != TASK:
            continue
        count += 1
        replacements = {"Objective": selected["objective"], "Inputs": ", ".join(selected["inputs"]["references"]),
                        "Validation": "; ".join(selected["validation"])}
        seen = set()
        lines = []
        for line in parts[index + 1].splitlines(keepends=True):
            match = re.match(r"^- ([^:]+):", line)
            if match and match[1] in replacements:
                need(match[1] not in seen, "duplicate revised board field")
                seen.add(match[1])
                line = "- " + match[1] + ": " + replacements[match[1]] + "\n"
            lines.append(line)
        need(seen == set(replacements), "revised board fields missing")
        parts[index + 1] = "".join(lines)
    need(count == 1, "single LIR-9001 block required")
    return "".join(parts).encode()


def assess_prerequisite_documents(docker_terminal, retry_publication, history):
    """Pure shape/claim check only; actual root observation is required below."""
    outcome = docker_terminal.get("outcome", {})
    need(docker_terminal.get("schema") == "legal-ir-native-docker-fixture-entry-terminal@4"
         and docker_terminal.get("actual_preparation_finalized") is True
         and docker_terminal.get("scope") == "synthetic-native-acceptance-regression-only"
         and all(outcome.get(k) is True for k in ("actual_native_validation_finalized", "actual_separate_orchestration_finalized",
             "actual_handle_consumed_once", "fixture_board_published", "actual_container_exit_and_cleanup_verified"))
         and outcome.get("actual_kernel_backend") == "source-bound-ordinary-A91-Docker"
         and outcome.get("provider_calls_performed") is False and outcome.get("real_LIR_task_accepted") is False
         and docker_terminal.get("provider_calls_performed") is False and docker_terminal.get("real_LIR_task_accepted") is False,
         "actual full engineering Docker qualification required")
    need(retry_publication.get("schema") == "legal-ir-attempt-preservation-publication@1"
         and retry_publication.get("pushed") is True and retry_publication.get("controlled_policy_testcases") == 51
         and all(type(retry_publication.get(k)) is int and retry_publication[k] == 0
                 for k in ("failures", "errors", "skipped", "canonical_tasks_accepted"))
         and retry_publication.get("new_runtime_setting_enabled") is False
         and retry_publication.get("taskboard_or_taskstate_changed") is False
         and retry_publication.get("branch") == "agent/legal-ir-ordinary-worker-20261016"
         and type(retry_publication.get("accelerator_commit")) is str
         and re.fullmatch(r"[0-9a-f]{40}", retry_publication["accelerator_commit"]) is not None
         and set(retry_publication.get("production_source_sha256", {})) == set(RETRY_PATHS),
         "actual qualified retry preservation deployment required")
    need(type(history) is dict and set(history) == {"schema", "task_id", "canonical_task_cid", "consumed_attempts",
         "board_namespace", "orchestration_namespace", "runtime_paths", "source_observation_pin", "history_reset_performed"}
         and history["schema"] == "legal-ir-root-observed-attempt-history@1" and history["task_id"] == TASK
         and type(history["consumed_attempts"]) is int and history["consumed_attempts"] == 2
         and history["history_reset_performed"] is False, "actual two-consumed-attempt observation required")


def _guard_owner(owner):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    need(type(owner) is ParentedPhaseResourceReservation and owner.heartbeat(), "genuine native preparation owner required")
    owner.check_usage(owner.directory, full=True)
    public = owner.native_lease.to_dict()
    need(public["owner_pid"] == os.getpid() and public["lane"] == "orchestration" and not public["released"]
         and not public["cancelled"], "actual held native preparation required")


def _git(path, *args):
    need(path in (ACCELERATOR, DATASETS), "only exact owned repository metadata allowed")
    result = subprocess.run(["/usr/bin/git", "--no-optional-locks", "-c", "core.hooksPath=/dev/null", *args],
        cwd=path, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    need(result.returncode == 0 and len(result.stdout) <= 4096, "actual bounded owned repository metadata unavailable")
    return result.stdout.decode().strip()


def _load_module(path, expected, name):
    need(sha(read(path)) == expected, "protected helper source pin differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_real_packet(*, preparation_owner, prerequisite_pins, helper_pins, root_observation):
    """Observe actual protected sources/evidence; return an unaccepted byte plan.

The trusted root supplies exact pins and explicit actual observation metadata,
not a model report. JSON evidence authenticates no native capability; genuine
preparation admission is checked independently. No materializer or dispatch
function is provided in this proposal.
"""
    need(sys.dont_write_bytecode is True, "root preparation must disable bytecode writes")
    _guard_owner(preparation_owner)
    need(type(root_observation) is dict and set(root_observation) == {"origin", "observed_at_utc", "evidence_path", "evidence_sha256"}
         and root_observation["origin"] == "actual root-owned native preparation observation"
         and all(type(v) is str and v for v in root_observation.values()), "actual root observation binding required")
    need(EVIDENCE_ROOT in Path(root_observation["evidence_path"]).parents,
         "actual owned public root observation required")
    need(sha(read(Path(root_observation["evidence_path"]))) == root_observation["evidence_sha256"],
         "actual root observation evidence differs")
    need(type(prerequisite_pins) is dict and set(prerequisite_pins) == {"docker_terminal", "retry_publication", "attempt_history"},
         "all actual prerequisite pins required")
    prerequisites, archived = {}, {}
    evidence_root = EVIDENCE_ROOT
    for role, pin in prerequisite_pins.items():
        need(type(pin) is dict and set(pin) == {"absolute_path", "bytes", "sha256"}, "exact actual prerequisite pin required")
        path = Path(pin["absolute_path"])
        need(evidence_root in path.parents and "fixtures" not in path.parts, "actual owned runtime evidence required")
        raw = read(path)
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "actual prerequisite bytes differ")
        if role == "docker_terminal":
            matching = []
            for line in raw.decode("utf-8").splitlines():
                try:
                    value = load(line.encode())
                except (ValueError, TypeError):
                    continue
                if type(value) is dict and value.get("schema") == "legal-ir-native-docker-fixture-entry-terminal@4":
                    matching.append(value)
            need(len(matching) == 1, "single actual Docker terminal observation required")
            prerequisites[role] = matching[0]
        else:
            prerequisites[role] = load(raw)
        archived["prerequisites/" + role + ".json"] = raw
    assess_prerequisite_documents(prerequisites["docker_terminal"], prerequisites["retry_publication"], prerequisites["attempt_history"])
    need(type(helper_pins) is dict and set(helper_pins) == {"task_identity", "factual_candidate", "native_acceptance", "controller_coordinator", "native_owner_sources"},
         "complete protected helper source pins required")
    need(type(helper_pins["native_owner_sources"]) is dict and set(helper_pins["native_owner_sources"]) == set(NATIVE_SOURCES),
         "complete actual native owner source closure required")
    for name in NATIVE_SOURCES:
        need(sha(read(DATASETS / name)) == helper_pins["native_owner_sources"][name], "actual native source closure differs")
    identity_path = ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py"
    identity = _load_module(identity_path, helper_pins["task_identity"], "real_packet_public_task_identity")
    facts = _load_module(ADAPTER / "factual_candidate.py", helper_pins["factual_candidate"], "real_packet_factual_candidate")
    documents, frozen = facts.load_source_bundle(OLD_ROOT)
    facts.derive_expected(documents, frozen)
    projection, config = load(documents["execution_projection"]), load(documents["scheduler_config"])
    revised, configured = transform_projection(projection, config)
    selected = next(t for t in revised["tasks"] if t["task_id"] == TASK)
    board = transform_board(documents["execution_board"], selected)
    # Use the real public daemon parser, which calls the actual identity API;
    # comparing JSON-only prose can miss board metadata used by that parser.
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_text
    need(Path(parse_task_text.__code__.co_filename).absolute() == ACCELERATOR / RETRY_PATHS[0],
         "actual owned public daemon parser required")
    old_tasks = parse_task_text(documents["execution_board"].decode(), path=OLD_ROOT / "tasks.todo.md", task_header_prefix="## LIR-")
    new_tasks = parse_task_text(board.decode(), path=ACCELERATOR / NEW_PREFIX / "tasks.todo.md", task_header_prefix="## LIR-")
    need(len(old_tasks) == len(new_tasks) == 64, "actual parser task population differs")
    for old, new in zip(old_tasks, new_tasks):
        reject_identity_shortcuts({"metadata": dict(old.metadata)})
        reject_identity_shortcuts({"metadata": dict(new.metadata)})
        before = identity.canonical_task_identity(old, board_namespace=old.board_namespace)
        after = identity.canonical_task_identity(new, board_namespace=new.board_namespace)
        need(before.canonical_task_cid == old.canonical_task_cid == new.canonical_task_cid == after.canonical_task_cid
             and before.canonical_task_key == after.canonical_task_key, "actual canonical semantic task identity changed")
    actual = next(t for t in old_tasks if t.task_id == TASK)
    history = prerequisites["attempt_history"]
    pin = history["source_observation_pin"]
    need(type(pin) is dict and set(pin) == {"absolute_path", "bytes", "sha256"}, "actual history source observation pin required")
    source_path = Path(pin["absolute_path"])
    need(evidence_root in source_path.parents and "fixtures" not in source_path.parts,
         "actual owned public history metadata required")
    history_raw = read(source_path)
    need(len(history_raw) == pin["bytes"] and sha(history_raw) == pin["sha256"], "actual history source observation differs")
    need(history["canonical_task_cid"] == actual.canonical_task_cid
         and history["board_namespace"] == actual.board_namespace
         and history["orchestration_namespace"] == config["board_namespace"] and history["runtime_paths"] == config["runtime_paths"],
         "actual consumed history identity or state lineage differs")
    retry = prerequisites["retry_publication"]
    head = _git(ACCELERATOR, "rev-parse", "HEAD")
    need(_git(ACCELERATOR, "status", "--porcelain=v1", "--untracked-files=all") == ""
         and _git(ACCELERATOR, "branch", "--show-current") == "agent/legal-ir-ordinary-worker-20261016",
         "actual clean owned deployment checkout required")
    _git(ACCELERATOR, "merge-base", "--is-ancestor", retry["accelerator_commit"], head)
    for name in RETRY_PATHS:
        need(sha(read(ACCELERATOR / name, 8 * 1024 * 1024)) == retry["production_source_sha256"][name],
             "actual deployed retry source differs")
    configured["source_binding"]["accelerator_required_ancestor"] = head
    observation = {"origin": root_observation["origin"], "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "evidence_path": root_observation["evidence_path"]}
    bindings = load(documents["current_source_bindings"])
    old_bindings = documents["current_source_bindings"]
    for row, repository in zip(bindings["repositories"], (ACCELERATOR, DATASETS)):
        row.update(status="observed_metadata", checkout_path=str(repository), head=_git(repository, "rev-parse", "HEAD"),
                   tree=_git(repository, "rev-parse", "HEAD^{tree}"), branch=_git(repository, "rev-parse", "--abbrev-ref", "HEAD"))
        # Preserve only source pins actually re-read; never relabel old hashes current.
        for pin in row["source_files"]:
            raw = read(repository / pin["path"], 8 * 1024 * 1024)
            pin.update(bytes=len(raw), sha256=sha(raw))
    bindings.update(status="observed_metadata", observation=observation)
    output = {name: raw for name, raw in frozen.items()}
    output.update({"source-copy-manifest.json": documents["source_copy_manifest"], "tasks.execution.json": wire(revised),
        "tasks.todo.md": board, "scheduler_config.json": wire(configured), "CURRENT_SOURCE_BINDINGS.json": wire(bindings),
        "history/CURRENT_SOURCE_BINDINGS.before.json": old_bindings,
        "native-owner-source-pins.json": wire(helper_pins["native_owner_sources"]),
        "protected-helper-source-pins.json": wire({k: v for k, v in helper_pins.items() if k != "native_owner_sources"}), **archived})
    for name in ("factual_candidate", "native_acceptance", "controller_coordinator"):
        raw = read(ADAPTER / (name + ".py"))
        need(sha(raw) == helper_pins[name], "protected adapter source pin differs")
        output["host_adapter/" + name + ".py"] = raw
    roles = {role: output[name] for role, name in facts.SOURCE_PATHS.items()}
    template = facts.candidate_template(roles, frozen)
    output["unaccepted-candidate-template.json"] = facts.canonical(template)
    # Host packet sources remain read-only worker inputs, including the exact
    # closed CLI. No advisory --check-facts command replaces completion validation.
    output["check_report.py"] = CHECKER_SOURCE.encode()
    output["REPORT_CONTRACT.md"] = REPORT_CONTRACT.encode()
    configured["protected_paths"] = list(dict.fromkeys(configured["protected_paths"] + [NEW_PREFIX + "/" + n for n in output]
                                                      + [NEW_PREFIX + "/packet-manifest.json"]))
    output["scheduler_config.json"] = wire(configured)
    roles["scheduler_config"] = output["scheduler_config.json"]
    output["unaccepted-candidate-template.json"] = facts.canonical(facts.candidate_template(roles, frozen))
    need(_git(ACCELERATOR, "rev-parse", "HEAD") == head, "owned repository changed during observation")
    total = sum(len(raw) for raw in output.values())
    need(total <= 8 * 1024 * 1024, "finite packet source/evidence byte budget exceeded")
    output["packet-manifest.json"] = wire({"schema": "legal-ir-real-report-source-packet@1",
        "canonical_task_identity_derived_via_public_helper": actual.canonical_task_cid,
        "consumed_attempts_retained": 2, "max_task_attempts": 3, "preserve_task_attempts": True,
        "history_namespace_and_runtime_paths_unchanged": True, "source_copy_manifest_sha256": sha(documents["source_copy_manifest"]),
        "root_observation": root_observation, "actual_prerequisite_pins": prerequisite_pins,
        "dispatch_authority": False, "completion_authority": False,
        "missing_real_backend_functions": list(MISSING_REAL_BACKEND_FUNCTIONS),
        "files": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(output.items())]})
    _guard_owner(preparation_owner)
    return {"schema": "legal-ir-real-report-packet-proposal@1", "files": output,
        "canonical_task_cid": actual.canonical_task_cid, "consumed_attempts_retained": 2, "max_task_attempts": 3,
        "canonical_task_board_namespace": actual.board_namespace, "orchestration_namespace": config["board_namespace"],
        "preserve_task_attempts": True, "worker_writable_report": REPORT, "actual_repo_head_observed": head,
        "materialization_performed": False, "dispatch_authority": False, "completion_authority": False,
        "required_materialization_precharge_bytes": sum(len(raw) for raw in output.values()) + 65536,
        "missing_real_backend_functions": list(MISSING_REAL_BACKEND_FUNCTIONS)}


CHECKER_SOURCE = '''"""Source-integrity preflight and closed completion-facing report gate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
sys.dont_write_bytecode = True

def _read(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("canonical source file required")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4 * 1024 * 1024:
        raise ValueError("bounded regular source required")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("source changed while reading")
    return raw

def _verify_manifest(root):
    manifest = json.loads(_read(root / "packet-manifest.json"))
    if manifest.get("schema") != "legal-ir-real-report-source-packet@1" or manifest.get("dispatch_authority") is not False or manifest.get("completion_authority") is not False:
        raise ValueError("unaccepted source packet manifest required")
    files = manifest.get("files")
    if type(files) is not list or not 18 <= len(files) <= 64:
        raise ValueError("bounded complete source file inventory required")
    seen, total = set(), 0
    for row in files:
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("closed source file pin required")
        name = Path(row["path"])
        if name.is_absolute() or ".." in name.parts or str(name) != row["path"] or row["path"] in seen:
            raise ValueError("unique canonical source name required")
        seen.add(row["path"])
        raw = _read(root / name)
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("source file pin differs")
        total += len(raw)
    required = {"authoring/tasks.json", "source-copy-manifest.json", "tasks.execution.json", "tasks.todo.md", "scheduler_config.json", "CURRENT_SOURCE_BINDINGS.json", "host_adapter/factual_candidate.py", "unaccepted-candidate-template.json"}
    if not required <= seen or total > 8 * 1024 * 1024:
        raise ValueError("complete bounded source roles required")

def _load_facts(root):
    spec = importlib.util.spec_from_file_location("packet_factual_candidate", root / "host_adapter/factual_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-all", action="store_true")
    modes.add_argument("--check-report", type=Path)
    modes.add_argument("--check-facts", type=Path)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    try:
        _verify_manifest(root)
        facts = _load_facts(root)
        if args.check_all:
            sources, frozen = facts.load_source_bundle(root)
            expected = facts.derive_expected(sources, frozen)
            config = facts.strict_json(sources["scheduler_config"])
            if config.get("max_task_attempts") != 3 or config.get("preserve_task_attempts") is not True or config.get("completion_policy", {}).get("task_completion_mode") != "manual_independent_report_acceptance":
                raise ValueError("finite manual report contract required")
            facts.check_facts(_read(root / "unaccepted-candidate-template.json"), sources, frozen)
            print(json.dumps({"valid": True, "scope": "source-integrity-only", "population": expected["facts"]["population"], "authority": facts.NO_AUTHORITY, "completion_authority": False, "dispatch_authority": False, "independent_acceptance_required": True}, sort_keys=True))
            return 0
        mode = "--check-report" if args.check_report is not None else "--check-facts"
        selected = args.check_report if args.check_report is not None else args.check_facts
        return facts.main(["--source-root", str(root), mode, str(selected)])
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"valid": False, "scope": "source-integrity-only", "errors": [str(error)], "completion_authority": False, "dispatch_authority": False}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
'''

REPORT_CONTRACT = '''# Real worker report contract, unaccepted

Produce only the existing declared handoff-reconciliation.json report using
legal-ir-source-factual-candidate@1. The supplied source-derived template is
unaccepted input metadata, not provider output or a reviewer receipt. Preserve
all source commitments, full facts, findings, unresolved scope, false authority
fields and unactivated LIR-9002. The completion-facing --check-report remains
closed with exit3 even when facts match. Do not replace it with --check-facts.

The worker cannot call native_acceptance or controller_coordinator. A distinct
genuine host validation owner must bind the real provider attempt, fenced
outcome, exact report-only candidate commit and source pins. The current V04
Docker entry is fixture-only; real-context launch/observation/adoption functions
remain explicit implementation gaps. No packet file supplies native, reviewer,
training, legal or proof authority. Preserve both consumed attempts; max3 is a
total cap with preserve_task_attempts:true, not permission to reset history.
'''


__all__ = ["PacketRefusal", "transform_projection", "transform_board", "reject_identity_shortcuts",
           "assess_prerequisite_documents", "prepare_real_packet"]

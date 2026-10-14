#!/usr/bin/env python3
"""Check T01's declared source closure without granting semantic admission.

Input schema is intentionally explicit. Receipts remain historical observations;
review, model fidelity, training qualification and Lean admission are out of scope.
This CLI performs read-only Git/file observations and one pinned-tree import probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile

SCHEMA = "legal-ir-source-integration-input/v1"
REPORT_SCHEMA = "legal-ir-source-integration-report/v1"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BYTE_BUDGET = 256 * 1024 * 1024
MAX_GIT_BYTES = 4 * 1024 * 1024
MAX_FILES = 4096
MODULE_PATHS = {
    "compiler": "ipfs_datasets_py/logic/legal_ir/canonical_compiler.py",
    "decompiler": "ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py",
    "parser": "ipfs_datasets_py/logic/deontic/utils/deontic_parser.py",
}
TREE_PIN_PATH = "ipfs_datasets_py/logic/autoformal/tree_pin.py"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class IntegrationError(ValueError):
    pass


def need(value, message):
    if not value:
        raise IntegrationError(message)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object(value, keys, label):
    need(type(value) is dict and set(value) == set(keys), f"{label}: exact fields required")


def _text(value, label):
    need(type(value) is str and 0 < len(value) <= 4096, f"{label}: nonempty bounded text required")
    return value


def _path(value):
    path = Path(_text(value, "path"))
    need(path.is_absolute() and path.resolve(strict=True) == path, "path must be absolute and exact, without symlinks")
    return path


def _relative(value):
    _text(value, "relative path")
    path = PurePosixPath(value)
    need(bool(path.parts) and value not in {"", ".", ".."} and not path.is_absolute()
         and value == path.as_posix() and all(p not in {"", ".", ".."} for p in path.parts),
         "nontraversing nonempty relative path required")
    return value


def _sha(value):
    need(type(value) is str and _SHA.fullmatch(value), "full lowercase SHA256 required")
    return value


def _commit(value):
    need(type(value) is str and _COMMIT.fullmatch(value), "full lowercase commit required")
    return value


def _load_json(raw):
    def pairs(rows):
        result = {}
        for k, v in rows:
            need(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    def nonfinite(value):
        raise IntegrationError("nonfinite JSON number")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise IntegrationError(f"invalid bounded JSON: {type(exc).__name__}") from exc


def _pin(row):
    _object(row, {"path", "bytes", "sha256"}, "file pin")
    _path(row["path"])
    need(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE_BYTES, "file size outside bound")
    _sha(row["sha256"])


def _hash_file(path, expected_size, budget):
    path = _path(str(path))
    need(stat.S_ISREG(path.stat().st_mode), "only regular pinned files may be read")
    before = path.stat()
    need(before.st_size == expected_size and expected_size <= MAX_FILE_BYTES, "pinned file byte size changed")
    need(budget[0] >= expected_size, "declared file byte budget exhausted")
    budget[0] -= expected_size
    digest = hashlib.sha256()
    observed = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(stat.S_ISREG(opened.st_mode) and identity(before) == identity(opened),
             "pinned file replaced before read")
        for chunk in iter(lambda: stream.read(65536), b""):
            observed += len(chunk)
            need(observed <= expected_size, "pinned file grew during observation")
            digest.update(chunk)
        ended = os.fstat(stream.fileno())
    need(path.resolve(strict=True) == path, "pinned file path redirected during read")
    after = path.stat()
    need(identity(before) == identity(opened) == identity(ended) == identity(after), "pinned file changed during observation")
    need(observed == expected_size, "pinned file byte count changed")
    return digest.hexdigest()


def file_pin(path):
    """Capture one operator-selected regular file; never scan a directory."""
    path = _path(str(path))
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "sha256": _hash_file(path, size, [MAX_FILE_BYTES])}


def _pinned_json(pin, budget):
    """Recheck bounded exact bytes when interpreting a previously pinned JSON file."""
    path = _path(pin["path"])
    before = path.stat()
    need(budget[0] >= pin["bytes"], "declared file byte budget exhausted")
    budget[0] -= pin["bytes"]
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and identity(before) == identity(opened),
             "JSON file replaced before read")
        raw = stream.read(pin["bytes"] + 1)
        ended = os.fstat(stream.fileno())
    need(path.resolve(strict=True) == path and identity(before) == identity(ended) == identity(path.stat()),
         "JSON file changed during read")
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
         "JSON bytes differ from immutable pin")
    return _load_json(raw)


def _git(root, *args, allowed=(0,)):
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "GIT_OPTIONAL_LOCKS": "0", "HOME": str(Path.home())}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        run = subprocess.run(["git", "-C", str(root), *args], stdin=subprocess.DEVNULL,
                             stdout=out, stderr=err, env=env, timeout=30, check=False)
        out.seek(0); raw = out.read(MAX_GIT_BYTES + 1)
        err.seek(0); error = err.read(4096)
    need(len(raw) <= MAX_GIT_BYTES, "Git observation exceeds byte bound")
    need(run.returncode in allowed, f"Git observation failed ({run.returncode}): {error.decode(errors='replace')[:512]}")
    return run.returncode, raw


def _source_changes(status, source_roots):
    rows = status.split(b"\0")
    changed = set()
    index = 0
    while index < len(rows) and rows[index]:
        row = rows[index]; index += 1
        need(len(row) >= 4 and row[2:3] == b" ", "malformed Git status observation")
        paths = [row[3:].decode("utf-8")]
        if b"R" in row[:2] or b"C" in row[:2]:
            need(index < len(rows) and rows[index], "missing Git rename source")
            paths.append(rows[index].decode("utf-8")); index += 1
        for path in paths:
            _relative(path)
            if any(path == scope or path.startswith(scope + "/") for scope in source_roots):
                changed.add(path)
    return sorted(changed)


def capture_repository(path, source_roots):
    """Capture HEAD/status and changed source paths; no diff or untracked-byte scan."""
    root = _path(str(path))
    need(root.is_dir(), "repository directory required")
    scopes = [_relative(scope) for scope in source_roots]
    _, actual_root = _git(root, "rev-parse", "--show-toplevel")
    need(Path(actual_root.decode().strip()).resolve() == root, "repository root differs")
    _, head = _git(root, "rev-parse", "HEAD")
    _, status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    return {"head": _commit(head.decode().strip()), "status_sha256": hashlib.sha256(status).hexdigest(),
            "changed_source_paths": _source_changes(status, scopes)}


def _manifest(value):
    fields = {"schema", "workspace_root", "byte_budget", "runtime", "repositories", "receipts",
              "producer_file_pins", "protected_file_pins", "hierarchy_pin", "unchecked_requirements"}
    _object(value, fields, "manifest")
    need(value["schema"] == SCHEMA, "unsupported manifest schema")
    _path(value["workspace_root"])
    need(type(value["byte_budget"]) is int and 0 < value["byte_budget"] <= MAX_BYTE_BUDGET, "declared byte budget outside bound")
    _object(value["runtime"], {"python_executable", "dependency_roots"}, "runtime")
    need(_path(value["runtime"]["python_executable"]).is_file(), "exact Python executable required")
    need(type(value["runtime"]["dependency_roots"]) is list and len(value["runtime"]["dependency_roots"]) <= 8, "dependency root bound exceeded")
    for path in value["runtime"]["dependency_roots"]:
        need(_path(path).is_dir(), "dependency root directory required")
    for key in ("repositories", "receipts", "producer_file_pins", "protected_file_pins", "unchecked_requirements"):
        need(type(value[key]) is list and len(value[key]) <= MAX_FILES, f"{key}: bounded list required")
    need(value["repositories"] and value["protected_file_pins"], "repository and protected source/checkpoint pins required")
    ids = set(); roots = set()
    for repo in value["repositories"]:
        _object(repo, {"repository_id", "role", "path", "head", "status_sha256", "source_roots", "required_ancestors"}, "repository")
        key = _text(repo["repository_id"], "repository id")
        need(key not in ids and repo["path"] not in roots, "duplicate repository id/path")
        ids.add(key); roots.add(repo["path"])
        need(repo["role"] in {"canonical_datasets", "accelerate", "source_contract"}, "unsupported repository role")
        _path(repo["path"]); _commit(repo["head"]); _sha(repo["status_sha256"])
        need(type(repo["source_roots"]) is list and 0 < len(repo["source_roots"]) <= 32, "source root declarations required")
        for scope in repo["source_roots"]: _relative(scope)
        need(type(repo["required_ancestors"]) is list and len(repo["required_ancestors"]) <= 128, "ancestor bound exceeded")
        commits = set()
        for row in repo["required_ancestors"]:
            _object(row, {"commit", "kind"}, "ancestor")
            _commit(row["commit"])
            need(row["kind"] in {"reviewed_main", "source_contract_tip", "pilot_implementation"}, "unsupported ancestor kind")
            need((row["commit"], row["kind"]) not in commits, "duplicate ancestor declaration")
            commits.add((row["commit"], row["kind"]))
    canonical = [r for r in value["repositories"] if r["role"] == "canonical_datasets"]
    need(len(canonical) == 1, "exactly one canonical datasets root required")
    need(Path(canonical[0]["path"]) == Path(value["workspace_root"]) / "external/ipfs_datasets", "canonical root differs from workspace/external/ipfs_datasets; redirects forbidden")
    need(len({a["commit"] for r in value["repositories"] for a in r["required_ancestors"]
              if a["kind"] == "source_contract_tip"}) >= 4, "all four distinct source-contract tip declarations required")
    need(any(a["kind"] == "reviewed_main" for r in value["repositories"] for a in r["required_ancestors"]), "reviewed main ancestor declaration required")
    receipt_ids = set(); pilot_commits = set()
    for receipt in value["receipts"]:
        _object(receipt, {"receipt_id", "kind", "path", "bytes", "sha256", "implementation_repository_id", "implementation_commit", "original_parent_commit", "original_task_parents"}, "receipt")
        key = _text(receipt["receipt_id"], "receipt id"); need(key not in receipt_ids, "duplicate receipt id"); receipt_ids.add(key)
        need(receipt["kind"] in {"pilot_completion", "supporting"}, "unsupported receipt kind")
        _pin({k: receipt[k] for k in ("path", "bytes", "sha256")})
        need(receipt["implementation_repository_id"] in ids, "receipt repository not declared")
        _commit(receipt["implementation_commit"]); _commit(receipt["original_parent_commit"])
        need(type(receipt["original_task_parents"]) is dict, "original task hierarchy mapping required")
        if receipt["kind"] == "pilot_completion":
            need(receipt["original_task_parents"], "pilot original task parents required")
            pilot_commits.add(receipt["implementation_commit"])
        for task, goal in receipt["original_task_parents"].items():
            need(re.fullmatch(r"LAC-\d{3}", task) and re.fullmatch(r"LAC-G\d{3}", goal), "invalid original hierarchy id")
    need(len([r for r in value["receipts"] if r["kind"] == "pilot_completion"]) == 5 and len(pilot_commits) == 5, "five distinct pilot implementation completion receipts required")
    producer_keys = set()
    for pin in value["producer_file_pins"]:
        _object(pin, {"repository_id", "relative_path", "bytes", "sha256"}, "producer pin")
        need(pin["repository_id"] in ids, "producer repository not declared"); _relative(pin["relative_path"])
        key = (pin["repository_id"], pin["relative_path"])
        need(key not in producer_keys, "duplicate producer file pin"); producer_keys.add(key)
        root = next(r["path"] for r in value["repositories"] if r["repository_id"] == pin["repository_id"])
        _pin({"path": str(Path(root) / pin["relative_path"]), "bytes": pin["bytes"], "sha256": pin["sha256"]})
    need({(canonical[0]["repository_id"], p) for p in (TREE_PIN_PATH, *MODULE_PATHS.values())} <= producer_keys, "actual canonical tree_pin and three logic module producer pins required")
    for pin in value["protected_file_pins"]: _pin(pin)
    _pin(value["hierarchy_pin"])
    need(value["hierarchy_pin"]["bytes"] <= MAX_MANIFEST_BYTES, "original hierarchy exceeds JSON byte bound")
    for text in value["unchecked_requirements"]: _text(text, "unchecked requirement")
    return canonical[0]


def _tree_probe(manifest, canonical):
    runtime = manifest["runtime"]
    environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": str(Path.home()),
        "PYTHONPATH": os.pathsep.join([canonical["path"], *runtime["dependency_roots"]]),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1", "IPFS_ACCELERATE_SKIP_CORE_IMPORTS": "1",
        "IPFS_AUTO_INSTALL": "false", "IPFS_DATASETS_AUTO_INSTALL": "false",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "IPFS_DATASETS_LEGAL_IR_METRIC_DISK_CACHE": "0", "CUDA_VISIBLE_DEVICES": ""}
    code = ("import json; from pathlib import Path; "
            "import ipfs_datasets_py.logic.autoformal.tree_pin as pin; "
            "result=pin.require_workspace_logic_tree(); "
            "print('LEGAL_IR_TREE_PIN_RESULT='+json.dumps({'tree_pin':str(Path(pin.__file__).resolve()),'root':str(pin.workspace_root()),'modules':result},sort_keys=True))")
    with tempfile.TemporaryDirectory(prefix="legal-ir-tree-pin-") as temporary:
        environment["PYTHONPYCACHEPREFIX"] = str(Path(temporary) / "empty-bytecode")
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            run = subprocess.run([runtime["python_executable"], "-P", "-B", "-c", code], cwd=temporary,
                env=environment, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=60, check=False)
            out.seek(0); stdout = out.read(MAX_MANIFEST_BYTES + 1)
            err.seek(0); stderr = err.read(MAX_MANIFEST_BYTES + 1)
        need(len(stdout) <= MAX_MANIFEST_BYTES and len(stderr) <= MAX_MANIFEST_BYTES, "tree pin output exceeds bound")
        need(run.returncode == 0, f"canonical tree pin failed ({run.returncode}): {stderr.decode(errors='replace')[-1024:]}")
        markers = [line.split(b"=", 1)[1] for line in stdout.splitlines() if line.startswith(b"LEGAL_IR_TREE_PIN_RESULT=")]
        need(len(markers) == 1, "one actual tree pin observation required")
        result = _load_json(markers[0])
        need(result == {"tree_pin": str(Path(canonical["path"]) / TREE_PIN_PATH), "root": canonical["path"],
                      "modules": {k: str(Path(canonical["path"]) / p) for k, p in MODULE_PATHS.items()}}, "canonical tree pin module/root origins drifted")
        need(not list(Path(temporary).rglob("*.pyc")), "tree pin subprocess wrote bytecode")
    return {"observed": result, "exit_code": run.returncode, "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "no_user_site": True,
            "metric_disk_cache": False, "fresh_empty_bytecode_prefix": True,
            "pythonpath": environment["PYTHONPATH"], "python_executable": runtime["python_executable"]}


def check(manifest):
    """Return an honest scoped disposition; never imports model weights or opens a DB."""
    report = {"schema": REPORT_SCHEMA, "scope": "manifest_declared_source_closure",
        "input_sha256": hashlib.sha256(wire(manifest)).hexdigest(), "status": "blocked", "passed": False,
        "findings": [], "repositories": [], "receipts": [], "tree_pin": None,
        "source_closure_preserved": False, "original_task_acceptance": "partial",
        "execution_admitted": False, "training_qualified": False, "legal_semantics_verified": False,
        "Lean_admitted": False, "Constitution_formalized": False, "database_opened": False,
        "weights_imported": False, "weights_downloaded": False, "provider_calls": 0}
    def finding(code, detail): report["findings"].append({"code": code, "detail": str(detail)})
    try:
        canonical = _manifest(manifest)
    except (IntegrationError, OSError, TypeError) as exc:
        finding("invalid_manifest", exc); return report
    budget = [manifest["byte_budget"]]
    repositories = {r["repository_id"]: r for r in manifest["repositories"]}
    all_pins = [manifest["hierarchy_pin"], *manifest["protected_file_pins"],
                *[{k: r[k] for k in ("path", "bytes", "sha256")} for r in manifest["receipts"]]]
    all_pins.extend({"path": str(Path(repositories[p["repository_id"]]["path"]) / p["relative_path"]),
                     "bytes": p["bytes"], "sha256": p["sha256"]} for p in manifest["producer_file_pins"])
    unique_pins = {}
    for pin in all_pins:
        if pin["path"] in unique_pins and unique_pins[pin["path"]] != pin:
            finding("contradictory_file_pins", pin["path"])
        unique_pins[pin["path"]] = pin
    def verify_pins(stage):
        for pin in unique_pins.values():
            try:
                need(_hash_file(pin["path"], pin["bytes"], budget) == pin["sha256"], "pinned SHA256 differs")
            except (IntegrationError, OSError) as exc:
                finding("file_pin_" + stage, pin["path"] + ": " + str(exc))
    verify_pins("before")
    before = {}
    for key, repo in repositories.items():
        try:
            observation = capture_repository(repo["path"], repo["source_roots"]); before[key] = observation
            report["repositories"].append({"repository_id": key, **observation})
            need(observation["head"] == repo["head"], "exact HEAD changed")
            need(observation["status_sha256"] == repo["status_sha256"], "Git working-state observation changed")
            covered = {p["relative_path"] for p in manifest["producer_file_pins"] if p["repository_id"] == key}
            for path in set(observation["changed_source_paths"]) - covered:
                finding("unreconciled_source_change", key + ":" + path)
            for ancestor in repo["required_ancestors"]:
                code, _ = _git(repo["path"], "merge-base", "--is-ancestor", ancestor["commit"], observation["head"], allowed=(0, 1, 128))
                if code != 0:
                    finding("repository_binding", key + f": missing/unintegrated {ancestor['kind']} ancestor {ancestor['commit']}")
                if ancestor["kind"] == "source_contract_tip":
                    code, _ = _git(canonical["path"], "merge-base", "--is-ancestor", ancestor["commit"], canonical["head"], allowed=(0, 1, 128))
                    if code != 0:
                        finding("repository_binding", key + f": source-contract tip absent from canonical tree {ancestor['commit']}")
        except (IntegrationError, OSError, subprocess.SubprocessError) as exc:
            finding("repository_binding", key + ": " + str(exc))
    try:
        hierarchy = _pinned_json(manifest["hierarchy_pin"], budget)
        need(type(hierarchy) is dict and hierarchy.get("schema") == "legal-autoformalization-plan/v1" and type(hierarchy.get("tasks")) is list, "retained original hierarchy required")
        task_parents = {task["id"]: task["goal_id"] for task in hierarchy["tasks"]}
        need(len(task_parents) == len(hierarchy["tasks"]), "duplicate original task id")
        for receipt in manifest["receipts"]:
            repo = repositories[receipt["implementation_repository_id"]]
            code, _ = _git(repo["path"], "merge-base", "--is-ancestor", receipt["implementation_commit"], repo["head"], allowed=(0, 1, 128))
            need(code == 0, f"receipt implementation not integrated: {receipt['receipt_id']}")
            code, _ = _git(canonical["path"], "merge-base", "--is-ancestor", receipt["implementation_commit"], canonical["head"], allowed=(0, 1, 128))
            if receipt["kind"] == "pilot_completion":
                need(code == 0, f"pilot implementation absent from canonical tree: {receipt['receipt_id']}")
            _, parents = _git(repo["path"], "rev-list", "--parents", "-n", "1", receipt["implementation_commit"])
            need(receipt["original_parent_commit"] in parents.decode().split()[1:], "implementation original Git parent differs")
            if receipt["kind"] == "pilot_completion":
                payload = _pinned_json({k: receipt[k] for k in ("path", "bytes", "sha256")}, budget)
                need(payload.get("datasets", {}).get("commit") == receipt["implementation_commit"], "pilot receipt implementation differs from actual receipt bytes")
                native_ids = payload.get("completed_native_task_ids")
                need(type(native_ids) is list and native_ids and all(type(t) is str and re.fullmatch(r"LAC-S\d+-\d{3}", t) for t in native_ids), "pilot completed task identities required")
                wanted = {"LAC-" + t.rsplit("-", 1)[1] for t in native_ids}
                need(wanted == set(receipt["original_task_parents"]), "pilot original task mapping differs from completed identities")
            for task, goal in receipt["original_task_parents"].items():
                need(task_parents.get(task) == goal, "retained original goal parent differs: " + task)
            report["receipts"].append({"receipt_id": receipt["receipt_id"], "sha256": receipt["sha256"],
                "implementation_commit": receipt["implementation_commit"], "original_task_parents": receipt["original_task_parents"],
                "scope": "immutable_historical_source_contract_observation"})
    except (IntegrationError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        finding("receipt_hierarchy_binding", exc)
    if not any(f["code"] in {"file_pin_before", "contradictory_file_pins"} for f in report["findings"]):
        try:
            report["tree_pin"] = _tree_probe(manifest, canonical)
        except (IntegrationError, OSError, subprocess.SubprocessError) as exc:
            finding("canonical_import_binding", exc)
    else:
        finding("canonical_import_not_run", "unverified input bytes are never executed")
    verify_pins("after")
    for key, observation in before.items():
        try:
            need(capture_repository(repositories[key]["path"], repositories[key]["source_roots"]) == observation,
                 "repository HEAD/source status changed during qualification")
        except (IntegrationError, OSError, subprocess.SubprocessError) as exc:
            finding("repository_drift", key + ": " + str(exc))
    for requirement in manifest["unchecked_requirements"]: finding("unchecked_requirement", requirement)
    report["bytes_rehashed"] = manifest["byte_budget"] - budget[0]
    report["passed"] = not report["findings"]
    report["status"] = "verified_declared_source_closure" if report["passed"] else "blocked"
    report["source_closure_preserved"] = not any(r["code"].startswith("file_pin_") or r["code"] == "repository_drift" for r in report["findings"])
    report["full_T01_acceptance"] = False
    report["claim_limit"] = "Only manifest-declared source custody/integration is checked; entire-worktree custody, independent legal review and semantic/model/Lake qualification are not inferred."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    need(args.manifest.resolve() != args.report.resolve(), "input/output paths must differ")
    need(args.report.suffix == ".json" and not args.report.exists(), "new JSON report path required; existing evidence is immutable")
    with args.manifest.open("rb") as stream: raw = stream.read(MAX_MANIFEST_BYTES + 1)
    need(len(raw) <= MAX_MANIFEST_BYTES, "manifest exceeds byte bound")
    value = _load_json(raw)
    if type(value) is dict:
        for repo in value.get("repositories", []):
            if type(repo) is dict and type(repo.get("path")) is str:
                need(Path(repo["path"]).resolve() not in args.report.resolve().parents, "report writes inside a checked source repository are forbidden")
    report = check(value)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    need(args.report.parent.resolve(strict=True) == args.report.parent.absolute(),
         "report parent must be exact and without symlinks")
    descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(args.report.resolve()), "passed": report["passed"], "status": report["status"], "finding_count": len(report["findings"])}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IntegrationError, OSError, subprocess.SubprocessError) as exc:
        print(f"source integration input rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)

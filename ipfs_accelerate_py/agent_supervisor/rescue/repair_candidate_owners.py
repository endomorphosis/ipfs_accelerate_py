"""Maintained finite validation and deferred activation owners.

Only the public D resource CLI allocates validation resources. A distinct trusted
worker checks natural tests and owns source READ custody; its result is joined
with the original resource terminal, real command wait, and durable artifacts.
The adoption command stages an immutable release. A separate serialized service
may cut it over after the original repair job has genuinely closed. Verifiers
and resume never repeat an owner command or a service action.
"""
from __future__ import annotations

import argparse
import base64
import ast
from collections import Counter
import contextlib
from contextlib import ExitStack
import fcntl
import dataclasses
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types
import threading
import weakref
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPAIR_PROFILE = "legal-ir"

POLICY_SCHEMA = "agent-supervisor/repair-public-owner-policy@2"
CONTROL_SCHEMA = "agent-supervisor/repair-control-source-release@1"
_CONTROL_GUARD = None
WORK_SCHEMA = "agent-supervisor/repair-trusted-validation@1"
DEFERRED_SCHEMA = "agent-supervisor/repair-adoption-deferred@1"
ACTIVATION_SCHEMA = "agent-supervisor/repair-actual-activation@1"
VALIDATION_SCHEMA = "agent-supervisor/isolated-repair-candidate-validation@1"
ADOPTION_SCHEMA = "agent-supervisor/isolated-repair-candidate-adoption@1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ORIGINAL_POPEN = subprocess.Popen
_ORIGINAL_COMPILE = compile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ordinary_bytes(path, maximum=8 * 1024**2):
    """Stdlib-only bootstrap read; no project package may precede the guard."""
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased absolute owner file required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
             and 0 <= before.st_size <= maximum, "bounded ordinary owner file required")
        raw = bytearray()
        while len(raw) <= maximum:
            part = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                           s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) == before.st_size and len(raw) <= maximum
             and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), "owner file changed while read")
        return bytes(raw)
    finally:
        os.close(fd)


def descriptor(path):
    path = Path(path); raw = ordinary_bytes(path)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked_descriptor(pin):
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}
         and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 8 * 1024**2
         and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]), "closed typed owner descriptor required")
    raw = ordinary_bytes(Path(pin["path"]))
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "owner descriptor changed")
    return raw


def json_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate owner JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite owner JSON")))
    need(type(value) is dict, "owner JSON object required")
    return value


def require_current_authorization(*args, **kwargs):
    # Deliberately lazy: validation-worker must first install its source guard
    # and normal candidate package search, before ANY project initialization.
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import require_current_authorization as require
    return require(*args, **kwargs)


def need(value, message):
    if not value:
        raise ValueError(message)


def new_json(path: Path, value) -> dict:
    raw = canonical(value) + b"\n"
    need(len(raw) <= 8 * 1024**2, "owner metadata bound exceeded")
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(path)


def proc_identity(pid):
    need(type(pid) is int and pid > 0, "positive process ID required")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "birth": fields[19]}


def checked_control_descriptor(pin):
    """The immutable manifest and each source file have a 4 MiB read ceiling."""
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}
         and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 4 * 1024**2
         and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]),
         "closed bounded control descriptor required")
    raw = ordinary_bytes(Path(pin["path"]), 4 * 1024**2)
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
         "control descriptor changed")
    return raw


def checked_control_release(policy, *, current_owner=True):
    """Authenticate one fixed control population, without project imports.

    The complete source manifest excludes itself and the separately pinned
    owner policy, avoiding a hash cycle. An active source successor cannot
    silently replace this independently granted control population.
    """
    value = policy.get("control_release")
    need(type(value) is dict and set(value) == {"root", "manifest"}, "immutable control release binding required")
    root = Path(value["root"])
    need(root.is_absolute() and ".." not in root.parts and root.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)), "ordinary absolute control release required")
    info = root.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "owned control release directory required")
    manifest_pin = value["manifest"]
    need(Path(manifest_pin["path"]).parent == root, "explicit control manifest must be directly inside its immutable root")
    manifest = json_object(checked_control_descriptor(manifest_pin))
    need(set(manifest) == {"schema", "root", "files", "namespace_profiles"} and manifest["schema"] == CONTROL_SCHEMA
         and manifest["root"] == str(root), "closed control release manifest required")
    rows = manifest["files"]
    need(type(rows) is list and 0 < len(rows) <= 4096
         and all(type(row) is dict and type(row.get("path")) is str for row in rows)
         and [row["path"] for row in rows] == sorted({row["path"] for row in rows}),
         "sorted finite unique complete control source population required")
    need(all(type(row.get("bytes")) is int and 0 <= row["bytes"] <= 4 * 1024**2 for row in rows)
         and sum(row["bytes"] for row in rows) <= 128 * 1024**2,
         "declared control source byte population exceeds its finite bound")
    total = 0; pins = {}
    for row in rows:
        path = Path(row["path"])
        need(root in path.parents and path != Path(manifest_pin["path"]), "control source escaped its immutable root")
        raw = checked_control_descriptor(row)
        need(len(raw) <= 4 * 1024**2 and not path.lstat().st_mode & 0o222,
             "control source must be bounded and sealed before use")
        total += len(raw)
        need(total <= 128 * 1024**2, "control source population byte bound exceeded")
        pins[str(path)] = row["sha256"]
    need(not Path(manifest_pin["path"]).lstat().st_mode & 0o222, "control manifest must be sealed before use")
    profiles = manifest["namespace_profiles"]
    need(type(profiles) is list and len(profiles) <= 1, "finite explicitly published normal namespace profile required")
    d = Path(policy["datasets_root"])
    for profile in profiles:
        need(type(profile) is dict and set(profile) == {"fullname", "path", "device", "inode", "initializer_absent"}
             and profile["fullname"] == "ipfs_datasets_py.logic.backends"
             and profile["path"] == str(d / "ipfs_datasets_py/logic/backends")
             and type(profile["device"]) is int and profile["device"] >= 0
             and type(profile["inode"]) is int and profile["inode"] > 0 and profile["initializer_absent"] is True,
             "closed exact canonical D backend namespace profile required")
        namespace = Path(profile["path"])
        need(namespace.is_absolute() and ".." not in namespace.parts
             and not any(path.is_symlink() for path in (namespace, *namespace.parents)),
             "normal namespace directory must be ordinary and unaliased")
        observed = namespace.lstat()
        need(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid()
             and (observed.st_dev, observed.st_ino) == (profile["device"], profile["inode"]),
             "published normal namespace directory identity changed")
        namespace_fd = os.open(namespace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(namespace_fd)
            need((opened.st_dev, opened.st_ino) == (profile["device"], profile["inode"]),
                 "published namespace changed during descriptor open")
            try:
                os.stat("__init__.py", dir_fd=namespace_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("published normal namespace acquired an initializer")
            after = namespace.lstat()
            need(stat.S_ISDIR(after.st_mode) and (after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino),
                 "published namespace changed after initializer inspection")
        finally:
            os.close(namespace_fd)
    owner = root / "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py"
    worker = owner.with_name("repair_validation_worker.py")
    for relative in ("ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py"):
        need(str(root / relative) in pins, "real normal package initializer/authorization source is missing")
    need(str(owner) in pins and str(worker) in pins, "complete owner and worker control files must be pinned")
    worker_tree = ast.parse(checked_control_descriptor(next(row for row in rows if row["path"] == str(worker))))
    guard_classes = [node for node in worker_tree.body if isinstance(node, ast.ClassDef) and node.name == "PinnedSources"]
    initializers = [node for node in guard_classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "__init__"] if len(guard_classes) == 1 else []
    need(len(initializers) == 1 and "normal_namespace_profiles" in {
        argument.arg for argument in [*initializers[0].args.args, *initializers[0].args.kwonlyargs]},
        "trusted full-file guard lacks the explicit normal namespace interface")
    if current_owner:
        need(Path(__file__).absolute() == owner and descriptor(owner)["sha256"] == pins[str(owner)],
             "actual owner is outside its authenticated immutable control release")
    control = {"root": root, "manifest": manifest_pin, "pins": pins, "owner": owner, "worker": worker,
        "file_count": len(rows), "source_bytes": total, "namespace_profiles": profiles}
    if type(policy.get("validation")) is dict and "external_fixture_profile" in policy["validation"]:
        checked_v3_validation_worker(control)
    return control


def install_control_guard(path, sha):
    """Cold, isolated entrypoint using the full maintained worker guard file."""
    global _CONTROL_GUARD
    need(_CONTROL_GUARD is None, "control bootstrap is not replayable in one interpreter")
    need(sys.flags.isolated == 1 and sys.dont_write_bytecode is True, "isolated no-bytecode owner process required")
    need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.")
                 or name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.") for name in sys.modules),
         "control guard must precede every project package initialization")
    need(type(sha) is str and _SHA.fullmatch(sha), "exact owner policy hash required")
    raw = ordinary_bytes(path)
    need(hashlib.sha256(raw).hexdigest() == sha, "owner policy changed before control bootstrap")
    policy = json_object(raw)
    need(set(policy) == {"schema", "datasets_root", "python", "source_descriptors", "validation", "adoption", "control_release"}
         and policy["schema"] == POLICY_SCHEMA, "closed immutable-control owner policy required")
    need(policy["python"] == sys.executable, "control owner interpreter differs from the original grant")
    need(type(_ORIGINAL_COMPILE) is types.BuiltinFunctionType
         and _ORIGINAL_COMPILE.__module__ == "builtins" and _ORIGINAL_COMPILE.__name__ == "compile",
         "original trusted control compiler primitive required")
    control = checked_control_release(policy)
    need(str(Path(path)) not in control["pins"], "control policy must be excluded from its source manifest")
    d = Path(policy["datasets_root"])
    need(d.is_absolute() and ".." not in d.parts and d.is_dir()
         and not any(p.is_symlink() for p in (d, *d.parents)), "ordinary canonical D source root required")
    need(d != control["root"] and d not in control["root"].parents and control["root"] not in d.parents,
         "control and canonical D source roots must be separate")
    sources = policy["source_descriptors"]
    imports = policy["validation"].get("import_sources")
    need(type(sources) is list and 0 < len(sources) <= 128 and type(imports) is list and len(imports) <= 4096,
         "finite original control and normal import populations required")
    pins = dict(control["pins"])
    for row in [*sources, *imports]:
        checked_descriptor(row)
        need(row["path"] not in pins or pins[row["path"]] == row["sha256"], "control/import source pin collision")
        pins[row["path"]] = row["sha256"]
    for profile in control["namespace_profiles"]:
        need(str(Path(profile["path"]) / "process.py") in pins,
             "published normal namespace must retain its raw-pinned process.py descendant")
    need(len(pins) <= 4096, "complete control and normal import manifest exceeds its finite bound")
    need(descriptor(control["owner"]) in sources and descriptor(control["worker"]) in sources,
         "original owner command controls omit the immutable owner or worker")
    worker = control["worker"]; worker_raw = ordinary_bytes(worker, 4 * 1024**2)
    need(hashlib.sha256(worker_raw).hexdigest() == pins[str(worker)], "guard bootstrap source changed")
    # Execute the complete maintained stdlib-only file, with an honest normal
    # file specification and its actual filename. No project namespace or
    # extracted function body is constructed, and no bytecode is consulted.
    class BootstrapLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            need(ordinary_bytes(worker, 4 * 1024**2) == worker_raw, "guard bootstrap changed before compilation")
            return _ORIGINAL_COMPILE(worker_raw, str(worker), "exec", dont_inherit=True)
    name = "_ipfs_repair_control_source_guard"
    need(name not in sys.modules, "control bootstrap module already cached")
    loader = BootstrapLoader(name, str(worker))
    spec = importlib.util.spec_from_file_location(name, worker, loader=loader)
    need(spec is not None and spec.origin == str(worker), "real control bootstrap file specification required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    need(descriptor(worker)["sha256"] == pins[str(worker)], "guard bootstrap changed after compilation")
    guard = module.PinnedSources(pins, [control["root"], d], normal_namespace_profiles=control["namespace_profiles"])
    sys.meta_path.insert(0, guard)
    # -I removes environment/cwd search. Replace any earlier script-root hint
    # explicitly, then retain only the actual isolated interpreter's roots.
    sys.path[:] = [str(control["root"]), str(d), *[entry for entry in sys.path if entry not in {str(control["root"]), str(d)}]]
    os.environ["IPFS_DATASETS_PY_MINIMAL_IMPORTS"] = "1"
    _CONTROL_GUARD = guard
    return policy, control


def inspect_control(args):
    """Read-only source readiness joined to the same pinned policy twice."""
    raw = ordinary_bytes(args.policy)
    need(hashlib.sha256(raw).hexdigest() == args.policy_sha256, "inspection owner policy changed before parsing")
    policy = json_object(raw); control = checked_control_release(policy)
    need(_CONTROL_GUARD is not None, "cold control guard is required")
    module = importlib.import_module("ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization")
    path = Path(module.__file__).absolute()
    need(control["root"] in path.parents and str(path) in _CONTROL_GUARD.executed,
         "normal immutable control package did not execute its actual authorization source")
    need(not _CONTROL_GUARD.denials, "control source guard recorded a permanent refusal")
    namespaces = _CONTROL_GUARD.verify_normal_namespaces()
    closing = ordinary_bytes(args.policy)
    need(closing == raw and hashlib.sha256(closing).hexdigest() == args.policy_sha256,
         "inspection owner policy changed before closing observation")
    need(checked_control_release(policy) == control, "inspection immutable control population changed")
    print(canonical({"schema": "agent-supervisor/repair-control-source-inspection@1",
        "policy": {"path": str(args.policy), "bytes": len(raw), "sha256": args.policy_sha256},
        "control_release": policy["control_release"], "file_count": control["file_count"],
        "source_bytes": control["source_bytes"], "actual_owner": descriptor(Path(__file__).absolute()),
        "declared_normal_namespace_profiles": control["namespace_profiles"],
        "observed_normal_namespace_origins": namespaces,
        "executed_source_buffers": _CONTROL_GUARD.executed, "source_guard_denials": _CONTROL_GUARD.denials,
        "python_isolated": sys.flags.isolated, "no_bytecode": sys.dont_write_bytecode,
        "state_written": False, "resource_admission_performed": False, "task_acceptance_asserted": False}).decode(), flush=True)
    return 0


def expected_natural_nodes(plan):
    """Authenticate the finite natural cases behind reviewed file selectors."""
    selectors = plan["selected_checks"]
    if "expected_nodeids" not in plan:
        need("selector_case_counts" not in plan, "partial selector expansion is forbidden")
        return selectors
    nodes, counts = plan["expected_nodeids"], plan.get("selector_case_counts")
    need(type(selectors) is list and 0 < len(selectors) <= 128 and len(set(selectors)) == len(selectors)
         and type(nodes) is list and 0 < len(nodes) <= 1024 and len(set(nodes)) == len(nodes)
         and type(counts) is dict and set(counts) == set(selectors)
         and all(type(count) is int and count > 0 for count in counts.values())
         and sum(counts.values()) == len(nodes) == plan["expected_cases"], "closed finite selector expansion required")
    observed = Counter()
    for node in nodes:
        need(type(node) is str and "::" in node and Path(node.partition("::")[0]).is_absolute(), "absolute natural case ID required")
        matches = [selector for selector in selectors if node == selector or node.startswith(selector + "::") or node.startswith(selector + "[")]
        need(len(matches) == 1, "natural case must belong to one exact reviewed selector")
        observed[matches[0]] += 1
    need(dict(observed) == counts, "natural case counts differ from the original selector expansion")
    return nodes


def checked_tool_descriptor(pin):
    """Original installed tools use their separate qualified 32 MiB boundary."""
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}
         and type(pin["bytes"]) is int and 0 < pin["bytes"] <= 32 * 1024**2
         and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]), "bounded original tool descriptor required")
    path = Path(pin["path"])
    need(path.is_absolute() and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "ordinary absolute installed tool required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.getuid()} and before.st_nlink == 1
             and before.st_mode & 0o111 and before.st_size == pin["bytes"], "original installed executable identity differs")
        digest = hashlib.sha256(); count = 0
        while True:
            raw = os.read(fd, 65536)
            if not raw:
                break
            count += len(raw); need(count <= pin["bytes"], "installed tool grew while authenticated"); digest.update(raw)
        stamp = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        need(count == pin["bytes"] and digest.hexdigest() == pin["sha256"]
             and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), "original installed tool changed")
    finally:
        os.close(fd)
    return pin


def trusted_validation_test_copies(policy, frozen, freeze_pin, root):
    """Join existing candidate test files to the real prepared Git baseline."""
    prepared_pin = descriptor(Path(freeze_pin["path"]).parent / "prepared.json")
    prepared = json_object(checked_descriptor(prepared_pin))
    need(prepared["binding"] == frozen["binding"] and digest(prepared["policy"]) == frozen["binding"]["policy_sha256"]
         and digest(prepared["baseline"]) == frozen["baseline_sha256"], "original candidate preparation does not bind the frozen baseline")
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate import parse_candidate_policy
    original_policy = parse_candidate_policy({"candidate_policy": prepared["policy"]})
    need(original_policy is not None and original_policy.namespace == frozen["binding"]["namespace"] == "ptr_lane_0",
         "genuine fixed candidate lane differs from the original prepared policy")
    control = Path(policy["control_release"]["root"]); copies = []
    for trusted in policy["validation"]["trusted_tests"]:
        path = Path(trusted["path"])
        need(control in path.parents, "trusted test source must remain in the immutable control population")
        relative = path.relative_to(control).as_posix()
        need(relative in prepared["policy"]["source_paths"] and relative not in prepared["policy"]["allowed_paths"]
             and relative not in frozen["changed_paths"], "candidate test copy entered provider write authority")
        before = prepared["baseline"].get(relative); after = frozen["manifest"].get(relative)
        need(type(before) is dict and after == before and before["bytes"] == trusted["bytes"]
             and before["sha256"] == trusted["sha256"], "candidate test was not already present unchanged at preparation")
        actual = descriptor(root / relative)
        need((actual["bytes"], actual["sha256"]) == (trusted["bytes"], trusted["sha256"])
             and checked_descriptor(trusted) == checked_descriptor(actual), "actual candidate test bytes differ from the maintained trusted input")
        copies.append({"trusted": trusted, "actual": actual})
    return prepared_pin, copies


def mapped_validation_fixture_profile(profile, copies):
    """Map only immutable test callsites; tool/control bodies remain exact."""
    copies_by_path = {row["trusted"]["path"]: row for row in copies}
    result = json_object(canonical(profile))
    def mapped_pin(pin):
        row = copies_by_path.get(pin["path"])
        need(row is not None and row["trusted"] == pin, "fixture caller is not one original trusted test copy")
        return row["actual"]
    def selected(node):
        path, separator, suffix = node.partition("::")
        row = copies_by_path.get(path)
        need(row is not None, "fixture selector does not belong to an immutable original test file")
        return row["actual"]["path"] + (separator + suffix if separator else "")
    def slot(row):
        row["selector"] = selected(row["selector"])
        row["owner_source"] = mapped_pin(row["owner_source"])
    for name in ("alias_slots", "source_slots"):
        for row in result[name]:
            slot(row)
    children = result["children"]
    for name in ("git_node_selectors", "frozen_owner_node_selectors"):
        children[name] = [selected(node) for node in children[name]]
    for name in ("cold_control_slots", "python_file_slots", "inline_slots"):
        for row in children[name]:
            slot(row)
    contract = children["installer_child_contract"]
    if "private_stub_node" in contract:
        contract["private_stub_node"] = selected(contract["private_stub_node"])
    if "systemd_allowed_nodes" in contract:
        contract["systemd_allowed_nodes"] = [selected(node) for node in contract["systemd_allowed_nodes"]]
    return result


def checked_worker_fixture_profile(value, *, python, tests, selectors, expected_nodes):
    """Metadata-only validation of explicit immutable fixture callsite grants."""
    need(type(value) is dict and set(value) == {"schema", "alias_slots", "source_slots", "children"}
         and value["schema"] == "agent-supervisor/bounded-external-fixtures@1", "closed bounded worker fixture profile required")
    pins = {pin["path"]: pin for pin in tests}
    def reviewed_selector(selector):
        return type(selector) is str and any(node == selector or node.startswith(selector + "::")
            or node.startswith(selector + "[") for node in expected_nodes)
    for name in ("alias_slots", "source_slots"):
        slots = value[name]
        need(type(slots) is list and len(slots) <= 1024, "finite explicit fixture slots required")
        for slot in slots:
            fields = {"selector", "owner_source", "function", "line"} | ({"event"} if name == "alias_slots" else {"path_suffix", "bytes", "sha256"})
            need(type(slot) is dict and set(slot) == fields and reviewed_selector(slot["selector"]),
                 "fixture slot must name one original reviewed selector")
            owner = slot["owner_source"]
            need(type(owner) is dict and owner == pins.get(owner.get("path")), "fixture slot must pin its immutable owning test source")
            raw = checked_descriptor(owner)
            need(type(slot["function"]) is str and 0 < len(slot["function"]) <= 128
                 and type(slot["line"]) is int and slot["line"] > 0, "exact fixture function and line required")
            functions = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name == slot["function"] and node.lineno <= slot["line"] <= node.end_lineno]
            need(bool(functions), "fixture callsite does not belong to its original source function")
            if name == "alias_slots":
                need(slot["event"] in {"os.symlink", "os.link", "os.mkfifo"}, "unsupported special fixture event")
            else:
                suffix = Path(slot["path_suffix"])
                need(type(slot["path_suffix"]) is str and bool(slot["path_suffix"]) and not suffix.is_absolute()
                     and ".." not in suffix.parts and type(slot["bytes"]) is int and 0 <= slot["bytes"] <= 4 * 1024**2
                     and type(slot["sha256"]) is str and _SHA.fullmatch(slot["sha256"]), "bounded exact fixture source payload required")
    children = value["children"]
    fields = {"schema", "python_executable", "installed_tool_inputs", "installer_tool", "installer_child_contract",
        "frozen_child_source", "landlock_launcher_source", "landlock_test_source", "git_node_selectors",
        "frozen_owner_node_selectors", "cold_control_slots", "python_file_slots", "inline_slots"}
    need(type(children) is dict and set(children) == fields and children["schema"] == "agent-supervisor/bounded-validation-children@1"
         and children["python_executable"] == python and len(canonical(children)) <= 1024**2,
         "closed finite original child fixture grammar required")
    for name in ("git_node_selectors", "frozen_owner_node_selectors", "cold_control_slots", "python_file_slots", "inline_slots"):
        rows = children[name]
        need(type(rows) is list and len(rows) <= 128, "finite child selector/slot population required")
        if name.endswith("node_selectors"):
            need(len(set(rows)) == len(rows) and all(reviewed_selector(row) for row in rows), "child selectors differ from reviewed test selection")
        else:
            need(all(type(row) is dict and len(canonical(row)) <= 256 * 1024 for row in rows), "finite typed child slot required")
            for row in rows:
                need(reviewed_selector(row.get("selector")) and type(row.get("owner_source")) is dict
                     and row["owner_source"] == pins.get(row["owner_source"].get("path")),
                     "child slot must name its original immutable test callsite")
                for key, content in row.items():
                    if type(content) is str:
                        need(len(content.encode()) <= 64 * 1024, "child literal exceeds the fixed metadata bound")
    need(children["python_file_slots"] == [] and children["inline_slots"] == [],
         "initial v3 reserves generic Python child slots without execution authority")
    for row in children["cold_control_slots"]:
        need(set(row) == {"selector", "owner_source", "function", "line", "owner_sha256s"}
             and type(row["function"]) is str and type(row["line"]) is int and row["line"] > 0
             and type(row["owner_sha256s"]) is list and 0 < len(row["owner_sha256s"]) <= 8
             and len(set(row["owner_sha256s"])) == len(row["owner_sha256s"])
             and all(type(sha) is str and _SHA.fullmatch(sha) for sha in row["owner_sha256s"]),
             "cold control child requires its exact finite reviewed owner hashes")
    tools = children["installed_tool_inputs"]
    need(type(tools) is list and 0 < len(tools) <= 8 and len({pin["path"] for pin in tools}) == len(tools), "finite original installed tools required")
    for pin in tools:
        checked_tool_descriptor(pin)
    checked_tool_descriptor(children["installer_tool"])
    need(children["installer_tool"] in tools, "installer tool must be an original installed tool input")
    for name in ("frozen_child_source", "landlock_launcher_source", "landlock_test_source"):
        need(type(children[name]) is str and 0 < len(children[name].encode()) <= 64 * 1024,
             "bounded exact original child literal source required")
    need(type(children["installer_child_contract"]) is dict and len(canonical(children["installer_child_contract"])) <= 1024**2,
         "bounded original installer child contract required")
    return value


def checked_v3_validation_worker(control):
    """Refuse an incompatible pure worker during metadata enrollment.

    A literal capability and the complete maintained interfaces are readiness
    gates only. The selected full worker still needs its own qualification and
    exact immutable source pin before a runtime enrollment can be authorized.
    No worker body is imported or executed by this inspection.
    """
    worker = Path(control["worker"])
    worker_pin = descriptor(worker)
    need(control["pins"].get(str(worker)) == worker_pin["sha256"],
         "v3 validation worker differs from its immutable control pin")
    tree = ast.parse(checked_control_descriptor(worker_pin))
    capabilities = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "V3_VALIDATION_CAPABILITY" for target in node.targets):
            capabilities.append(node.value)
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "V3_VALIDATION_CAPABILITY"):
            capabilities.append(node.value)
    need(len(capabilities) == 1 and isinstance(capabilities[0], ast.Constant)
         and type(capabilities[0].value) is str
         and capabilities[0].value == "agent-supervisor/repair-validation-v3@1",
         "v3 validation worker capability is unavailable")
    for name in ("expected_natural_nodes", "authenticated_fixture_mapping"):
        need(sum(isinstance(node, ast.FunctionDef) and node.name == name
                 for node in tree.body) == 1,
             "v3 validation worker interface is unavailable: " + name)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TrustedFixtures"]
    need(len(classes) == 1, "v3 validation worker fixture interface is unavailable")
    for name in ("__init__", "verify", "set_node", "alias", "compile_source",
            "exec_source", "setup_children", "checked_popen", "close_children",
            "deadline_enter", "deadline_exit"):
        need(sum(isinstance(node, ast.FunctionDef) and node.name == name
                 for node in classes[0].body) == 1,
             "v3 validation worker fixture method is unavailable: " + name)
    guards = [node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == "PinnedSources"]
    initializers = [node for node in guards[0].body if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__"] if len(guards) == 1 else []
    need(len(initializers) == 1 and "normal_namespace_profiles" in {
        arg.arg for arg in [*initializers[0].args.args, *initializers[0].args.kwonlyargs]},
        "v3 validation worker normal namespace interface is unavailable")
    return {"capability": capabilities[0].value, "worker": worker_pin,
            "source_interface_only": True, "worker_executed": False}


VALIDATION_ADMISSION_RETRY_SCHEMA = "agent-supervisor/validation-admission-retry@1"
SOURCE_ADMISSION_DEFER_SCHEMA = "agent-supervisor/validation-source-admission-deferred@1"
SOURCE_ADMISSION_VERIFIED_SCHEMA = "agent-supervisor/verified-validation-source-admission-defer@1"



LEDGER_BUSY_RETRY_SCHEMA = 'agent-supervisor/validation-ledger-busy-retry@1'
LEDGER_BUSY_SCHEMA = 'agent-supervisor/closed-operational-ledger-busy@1'
LEDGER_BUSY_VERIFIED_SCHEMA = 'agent-supervisor/verified-operational-ledger-busy@1'
LEDGER_BUSY_CONTEXT_SCHEMA = 'agent-supervisor/validation-ledger-busy-generation@1'


def checked_validation_ledger_busy_retry(value):
    need(type(value) is dict and set(value) == {'schema','max_additional_invocations','cooldown_seconds',
        'authorization_epoch_sha256','expires_at'} and value['schema'] == LEDGER_BUSY_RETRY_SCHEMA
        and type(value['max_additional_invocations']) is int and value['max_additional_invocations'] == 2
        and type(value['cooldown_seconds']) is int and value['cooldown_seconds'] == 60
        and type(value['authorization_epoch_sha256']) is str and _SHA.fullmatch(value['authorization_epoch_sha256'])
        and type(value['expires_at']) in (int,float) and math.isfinite(value['expires_at']) and value['expires_at'] > 0,
        'exact explicit finite ledger-busy retry profile required')
    return json_object(canonical(value))


def ledger_busy_request_attempt_id(frozen, epoch_sha256, index):
    need(type(index) is int and 1 <= index <= 2 and type(epoch_sha256) is str and _SHA.fullmatch(epoch_sha256)
         and type(frozen['freeze_sha256']) is str and _SHA.fullmatch(frozen['freeze_sha256']),
         'bounded original ledger-busy generation identity required')
    return digest({'freeze_sha256':frozen['freeze_sha256'],'authorization_epoch_sha256':epoch_sha256,
                   'validation_ledger_busy_generation':index})[:32]


def validation_request_generation(policy, frozen, invocation=None, *, live=False):
    """Fresh D request identity; the original frozen candidate is unchanged."""
    original = frozen['binding']['attempt_id']
    if 'source_qualification' in policy['validation']:
        need(invocation is None or not (invocation/'ledger-busy-generation.json').exists(),
             'immutable qualification cannot use repair retry generation')
        return original
    if invocation is None or not (invocation/'ledger-busy-generation.json').exists():
        return original
    profile = checked_validation_ledger_busy_retry(policy['validation']['validation_ledger_busy_retry'])
    reference = descriptor(invocation/'ledger-busy-generation.json')
    context = json_object(checked_descriptor(reference))
    need(not Path(reference['path']).lstat().st_mode & 0o222
         and set(context) == {'schema','index','authorization_epoch_sha256','expires_at','freeze_sha256',
              'source_commitment','request_attempt_id','prior_operational_ref','spent_slot_ref'}
         and context['schema'] == LEDGER_BUSY_CONTEXT_SCHEMA
         and context['authorization_epoch_sha256'] == profile['authorization_epoch_sha256']
         and type(context['expires_at']) in (int,float) and context['expires_at'] == profile['expires_at']
         and context['freeze_sha256'] == frozen['freeze_sha256']
         and context['source_commitment'] == frozen['source_commitment']
         and context['request_attempt_id'] == ledger_busy_request_attempt_id(frozen,profile['authorization_epoch_sha256'],context['index']),
         'ledger-busy request generation differs from the original frozen authority')
    ticket = json_object(checked_descriptor(context['spent_slot_ref']))
    need(not Path(context['spent_slot_ref']['path']).lstat().st_mode & 0o222
         and Path(context['spent_slot_ref']['path']) == invocation.parent/('ledger-busy-slot-%02d.json' % context['index'])
         and set(ticket)=={'schema','index','attempt_number','attempt_token','authorization_epoch_sha256',
             'expires_at','freeze_sha256','source_commitment','request_attempt_id','invocation','prior_operational_ref','spent_at'}
         and ticket['schema'] == 'agent-supervisor/spent-validation-ledger-busy-slot@1'
         and type(ticket['attempt_number'])is int and ticket['attempt_number']==frozen['binding']['attempt_number']
         and type(ticket['attempt_token'])is str and ticket['attempt_token']==frozen['binding']['attempt_id']
         and all(canonical(ticket[k]) == canonical(context[k]) for k in ('index','authorization_epoch_sha256',
             'expires_at','freeze_sha256','source_commitment','request_attempt_id','prior_operational_ref'))
         and ticket['invocation'] == str(invocation) and type(ticket['spent_at']) in (int,float)
         and math.isfinite(ticket['spent_at']) and 0 < ticket['spent_at'] < context['expires_at'],
         'ledger-busy generation lacks its original immutable spent slot')
    prior = json_object(checked_descriptor(context['prior_operational_ref']))
    expected_previous = 'original-validate-invocation' if context['index'] == 1 else 'original-validate-ledger-busy-01'
    need(invocation.name == 'original-validate-ledger-busy-%02d' % context['index']
         and Path(prior['invocation_root']) == invocation.parent/expected_previous,
         'ledger-busy generation chain is cyclic, forked or not the preceding original')
    need(prior['schema'] == LEDGER_BUSY_SCHEMA and prior['binding'] == frozen['binding']
         and prior['freeze_sha256'] == frozen['freeze_sha256'] and prior['source_commitment'] == frozen['source_commitment']
         and prior['retry_authorized'] is False and prior['semantic_negative'] is False,
         'ledger-busy generation lacks a same-candidate original operational disposition')
    verify_v5_ledger_busy_closeout(policy,prior['request'],Path(prior['invocation_root']),prior['parent_source_closeout'])
    if live:
        need(time.time() < profile['expires_at'], 'original ledger-busy retry authorization expired')
    return context['request_attempt_id']


def _current_v5_cli_classifier(policy):
    """Normally load the whole current pinned CLI; never an old body/extraction."""
    path=Path(policy['datasets_root'])/'scripts/ops/legal_ir/run_owned_validation.py'
    pin=next(pin for pin in policy['source_descriptors'] if pin['path']==str(path))
    raw=checked_descriptor(pin);name='_repair_current_ledger_busy_classifier_DATA'
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and type(spec.loader) is importlib.machinery.SourceFileLoader
         and spec.origin==str(path), 'current original ledger-busy classifier file required')
    module=importlib.util.module_from_spec(spec)
    exec(_ORIGINAL_COMPILE(raw,str(path),'exec',dont_inherit=True),vars(module))
    function=module.classify_owned_validation_source_custody
    need(type(function) is types.FunctionType and function.__globals__ is vars(module)
         and function.__code__.co_filename==str(path) and checked_descriptor(pin)==raw,
         'current original ledger-busy classifier origin changed')
    return function


def verify_v5_ledger_busy_closeout(policy,request_pin,invocation,closeout_pin):
    """Closed operational DATA only: no child/grant/record, never retry authority."""
    value=json_object(checked_descriptor(closeout_pin));wait=json_object(ordinary_bytes(invocation/'runner-wait.json'))
    terminal=json_object(wait['stdout'].encode());raw=checked_descriptor(request_pin);request=json_object(raw)
    expected_argv=[policy['python'],'-I','-B',str(Path(policy['datasets_root'])/'scripts/ops/legal_ir/run_owned_validation.py'),
        '--request',request_pin['path'],'--expected-request-sha256',request_pin['sha256'],
        '--resource-ledger',policy['validation']['resource_ledger']]
    need(wait['argv']==expected_argv,'operational ledger-busy original CLI argv differs')
    life=terminal.get('source_custody_lifecycle',{})
    need(life.get('schema')=='owned-validation-pre-grant-ledger-lock-refusal/v1'
         and life.get('stage')=='owner_open_ledger_lock_refused_before_record'
         and terminal.get('reservation',{}).get('status')=='not_entered',
         'only the original pregrant ledger-lock refusal is operationally eligible')
    verdict=_current_v5_cli_classifier(policy)(terminal,raw,expected_request_sha256=request_pin['sha256'],
        owner_pid=wait['runner_identity']['pid'],owner_birth=wait['runner_identity']['birth'],
        returncode=wait['returncode'],expected_resource_ledger=policy['validation']['resource_ledger'])
    need(verdict=={'verdict':'proven_original_no_child_no_native_grant','source_custody_closed':True,
                   'storage_release_authorized':False}, 'original ledger-busy no-grant classifier refused')
    _verify_v5_parent_source_closeout(policy,request_pin,invocation,closeout_pin,None,_ledger_busy=True)
    need(original_execution_request(invocation)==request_pin
         and request==validation_execution_request(policy,json_object(ordinary_bytes(invocation/'validation-plan.json')),
                                                   descriptor(invocation/'validation-plan.json')),
         'operational ledger-busy original request publication differs')
    roots=[Path(policy['datasets_root']),Path(request['cwd'])]
    for receipt,root in zip(value['parent_source_custody']['source_receipts'],roots,strict=True):
        need(type(receipt)is dict and set(receipt)=={'schema_version','canonical_root','cooperative','lock_path','mode','waited_seconds'}
             and receipt['schema_version']=='autoencoder-source-lease/v1' and receipt['canonical_root']==str(root)
             and receipt['cooperative']is True and receipt['mode']=='read'
             and receipt['lock_path']==str(root/'workspace/control-plane/source-leases/source.lock')
             and type(receipt['waited_seconds'])in(int,float) and math.isfinite(receipt['waited_seconds'])
             and receipt['waited_seconds']>=0, 'operational ledger-busy original Source@2 receipt differs')
    attempt=Path(request['attempt_directory']);need(attempt.is_dir() and not attempt.is_symlink(),
                                                    'original no-grant attempt directory is unknown')
    with os.scandir(attempt)as entries:
        need(next(entries,None)is None,'pregrant ledger-busy attempt has unexpected outputs')
    return {'request':request_pin,'parent_source_closeout':closeout_pin,
        'source_closed':True,'native_closed':True,'storage_status':'not_entered',
        'record_created':False,'native_grant_created':False,'productive_child_created':False,
        'semantic_negative':False,'retry_authorized':False,'storage_release_authorized':False}


def operational_ledger_busy_material(policy,frozen,request_pin,invocation,closeout_pin):
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair operational_ledger_busy_material")
    checked=verify_v5_ledger_busy_closeout(policy,request_pin,invocation,closeout_pin)
    return {'schema':LEDGER_BUSY_SCHEMA,'reason':'pregrant_ledger_busy','binding':frozen['binding'],
        'freeze_sha256':frozen['freeze_sha256'],'source_commitment':frozen['source_commitment'],
        'invocation_root':str(invocation),'policy':descriptor(Path(json_object(ordinary_bytes(invocation/'validation-plan.json'))['policy']['path'])),
        'plan':descriptor(invocation/'validation-plan.json'),**checked}


def verify_operational_validation(args):
    policy=checked_policy(args.policy,args.policy_sha256)
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair verify_operational_validation")
    frozen=frozen_candidate(args.freeze_path,args.source_commitment,args.candidate_root)
    pin=descriptor(args.invocation_root/'operational-ledger-busy.json')
    need(not Path(pin['path']).lstat().st_mode & 0o222,'original operational disposition must be immutable')
    value=json_object(checked_descriptor(pin));request=original_execution_request(args.invocation_root)
    actual=operational_ledger_busy_material(policy,frozen,request,args.invocation_root,
                                           descriptor(args.invocation_root/'parent-source-closeout.json'))
    need(canonical(value)==canonical(actual),'original operational ledger-busy disposition changed')
    print(canonical({'schema':LEDGER_BUSY_VERIFIED_SCHEMA,'operational':pin,'disposition':actual,
        'read_only':True,'retry_authorized':False,'storage_release_authorized':False}).decode(),flush=True)


def checked_validation_admission_retry(value):
    need(type(value) is dict and set(value) == {"schema", "max_additional_invocations", "cooldown_seconds", "source_probe_timeout_seconds"}
         and value["schema"] == VALIDATION_ADMISSION_RETRY_SCHEMA
         and all(type(value[key]) is int and value[key] == expected for key, expected in {
             "max_additional_invocations": 2, "cooldown_seconds": 60, "source_probe_timeout_seconds": 60}.items()),
         "exact finite validation admission retry profile required")
    return dict(value)


def validation_unstarted_paths(policy, frozen, invocation):
    binding = frozen["binding"]
    attempt = Path(policy["validation"]["attempt_root"]) / binding["namespace"] / binding["board_id"] / binding["attempt_id"]
    return [invocation / "validation-plan.json", invocation / "execution-request.json",
        validation_control_directory(policy, frozen) / "execution-request.json", attempt,
        attempt.parent / (attempt.name + "-external-fixtures")]


def validation_path_absent(path):
    """Prove only an anchored missing leaf/ancestor, never an OSError absence."""
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts and len(path.parts) <= 128,
         "finite ordinary validation path required")
    held = []
    try:
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append((Path(path.anchor), fd))
        prefix = Path(path.anchor)
        absent = False
        for offset, name in enumerate(path.parts[1:], 1):
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                absent = True
                break
            if offset == len(path.parts) - 1:
                break
            need(stat.S_ISDIR(info.st_mode), "validation path ancestor is absent or aliased")
            prefix = prefix / name
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            observed = os.fstat(child)
            held.append((prefix, child))
            need((observed.st_dev, observed.st_ino) == (info.st_dev, info.st_ino),
                 "validation path ancestor changed")
            fd = child
        for named, descriptor in held:
            current, opened = named.lstat(), os.fstat(descriptor)
            need(stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                 "validation path ancestor changed during absence observation")
        return absent
    finally:
        for _, descriptor in reversed(held):
            os.close(descriptor)


def require_validation_unstarted(policy, frozen, invocation):
    paths = validation_unstarted_paths(policy, frozen, invocation)
    need(all(validation_path_absent(path) for path in paths),
         "validation admission already has a plan, request, or attempt")
    return [str(path) for path in paths]


def probe_validation_source_admission(policy, args, frozen):
    """Original source READ eligibility only; close before any D admission."""
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair probe_validation_source_admission")
    profile = checked_validation_admission_retry(policy["validation"]["validation_admission_retry"])
    invocation = args.invocation_root
    absent = require_validation_unstarted(policy, frozen, invocation)
    intent = descriptor(invocation / "intent.json")
    source = public_source_lease(policy)
    module = sys.modules.get(source.__module__)
    source_pin = next(row for row in policy["source_descriptors"]
        if row["path"] == str(Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"))
    wrapped = getattr(source, "__wrapped__", None)
    need(type(module) is types.ModuleType and source is getattr(module, "source_lease", None)
         and descriptor(Path(module.__file__).absolute()) == source_pin
         and type(wrapped) is types.FunctionType and wrapped.__code__.co_filename == source_pin["path"],
         "original source-admission primitive origin differs")
    original_timeout = module.SourceLeaseTimeout
    entered = []; failure = None; original_source_lease_calls = 0
    started = time.monotonic(); deadline = started + profile["source_probe_timeout_seconds"]
    with ExitStack() as stack:
        try:
            for root in (Path(policy["datasets_root"]), args.candidate_root):
                remaining = deadline - time.monotonic()
                need(remaining >= 0, "source admission eligibility deadline elapsed before original wait")
                original_source_lease_calls += 1
                lease = stack.enter_context(source(root, mode="read", timeout_seconds=remaining))
                entered.append(lease.to_dict())
        except BaseException as error:
            if type(error) is not original_timeout:
                raise
            frames = []; traceback = error.__traceback__
            while traceback is not None:
                frame = traceback.tb_frame
                if frame.f_globals is vars(module) and frame.f_code.co_filename == source_pin["path"]:
                    frames.append({"function": frame.f_code.co_name, "line": traceback.tb_lineno,
                        "source": source_pin})
                traceback = traceback.tb_next
            need(any(frame["function"] == "_acquire" for frame in frames),
                 "SourceLeaseTimeout lacks its original admission traceback")
            failure = {"module": module.__name__, "exception_type": "SourceLeaseTimeout",
                "source": source_pin, "original_frames": frames}
    # Original Git identity checks have their existing independent bounds.
    # No hard wall-time claim or synthetic SourceLeaseTimeout is introduced.
    if failure is None and time.monotonic() > deadline:
        raise TimeoutError("source admission probe elapsed without an original SourceLeaseTimeout")
    # Any original __exit__ failure propagated above; no closed flag is guessed.
    require_validation_unstarted(policy, frozen, invocation)
    need(descriptor(invocation / "intent.json") == intent, "source admission original intent changed")
    record = {"schema": SOURCE_ADMISSION_DEFER_SCHEMA if failure else "agent-supervisor/validation-source-admission-probe@1",
        "policy": descriptor(args.policy), "freeze": descriptor(args.freeze_path), "original_intent": intent,
        "binding": frozen["binding"], "freeze_sha256": frozen["freeze_sha256"], "source_commitment": frozen["source_commitment"],
        "owner": proc_identity(os.getpid()), "retry_profile": profile,
        "entered_source_READs": entered, "source_READ_closed": True, "source_exception": failure,
        "probe_started_monotonic": started, "probe_deadline_monotonic": deadline,
        "probe_finished_monotonic": time.monotonic(), "observed_at": time.time(), "unstarted_paths": absent,
        "clock_scope": "cumulative_original_lock_wait_allowance_with_existing_separate_Git_identity_bounds",
        "hard_wall_time_60_seconds_claimed": False,
        "original_source_lease_calls": original_source_lease_calls,
        "zero_D_calls_scope": "public_validation_CLI_and_validation_compute_or_storage_allocation_only",
        "validation_admission_performed": False, "native_disposition": "not_requested", "storage_disposition": "not_requested",
        "validation_worker_started": False, "tests_started": False, "edited_body_executed": False,
        "D_owner_calls": 0, "D_resource_API_calls": 0, "semantic_negative": False, "retry_authorized": False,
        "parent_repair_resource_closure_claimed": False, "completion_authority": False}
    pin = new_json(invocation / ("source-admission-deferred.json" if failure else "source-admission-completed.json"), record)
    return pin if failure else None


def verify_validation_admission(args):
    """Read-only authentication; never reacquire/retry source or D work."""
    policy = checked_policy(args.policy, args.policy_sha256)
    profile = checked_validation_admission_retry(policy["validation"]["validation_admission_retry"])
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    invocation = args.invocation_root
    need(args.receipt_path.parent == invocation and args.freeze_path.parent == invocation.parent,
         "original source-admission invocation binding required")
    path = invocation / "source-admission-deferred.json"
    need(not path.lstat().st_mode & 0o222, "original source-admission defer must be immutable")
    pin = descriptor(path); value = json_object(checked_descriptor(pin))
    intent = descriptor(invocation / "intent.json")
    need(value["schema"] == SOURCE_ADMISSION_DEFER_SCHEMA and value["policy"] == descriptor(args.policy)
         and value["freeze"] == descriptor(args.freeze_path) and value["original_intent"] == intent
         and value["binding"] == frozen["binding"] and value["freeze_sha256"] == frozen["freeze_sha256"]
         and value["source_commitment"] == frozen["source_commitment"] and value["retry_profile"] == profile,
         "original source-admission defer belongs to another candidate or policy")
    source_pin = next(row for row in policy["source_descriptors"]
        if row["path"] == str(Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"))
    exception = value["source_exception"]
    need(type(exception) is dict and exception["module"] == "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease"
         and exception["exception_type"] == "SourceLeaseTimeout" and exception["source"] == source_pin
         and type(exception["original_frames"]) is list and 0 < len(exception["original_frames"]) <= 16
         and all(row["source"] == source_pin and type(row["line"]) is int and row["line"] > 0 for row in exception["original_frames"])
         and any(row["function"] == "_acquire" for row in exception["original_frames"]),
         "exact original source-admission timeout is unavailable")
    need(value["source_READ_closed"] is True and type(value["entered_source_READs"]) is list
         and len(value["entered_source_READs"]) <= 1 and value["validation_admission_performed"] is False
         and value["native_disposition"] == value["storage_disposition"] == "not_requested"
         and all(value[key] is False for key in ("validation_worker_started", "tests_started", "edited_body_executed",
             "semantic_negative", "retry_authorized", "parent_repair_resource_closure_claimed", "completion_authority"))
         and type(value["D_owner_calls"]) is int and value["D_owner_calls"] == 0
         and type(value["D_resource_API_calls"]) is int and value["D_resource_API_calls"] == 0
         and value["zero_D_calls_scope"] == "public_validation_CLI_and_validation_compute_or_storage_allocation_only"
         and type(value["original_source_lease_calls"]) is int
         and value["original_source_lease_calls"] == len(value["entered_source_READs"]) + 1,
         "source-admission defer does not establish the zero-D-work boundary")
    for key in ("probe_started_monotonic", "probe_deadline_monotonic", "probe_finished_monotonic", "observed_at"):
        need(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] > 0,
             "original source-admission clock is invalid")
    need(value["probe_deadline_monotonic"] == value["probe_started_monotonic"] + 60
         and value["probe_finished_monotonic"] >= value["probe_started_monotonic"]
         and value["clock_scope"] == "cumulative_original_lock_wait_allowance_with_existing_separate_Git_identity_bounds"
         and value["hard_wall_time_60_seconds_claimed"] is False
         and value["unstarted_paths"] == require_validation_unstarted(policy, frozen, invocation),
         "original source-admission eligibility changed")
    checked_descriptor(source_pin)
    need(descriptor(path) == pin and descriptor(invocation / "intent.json") == intent,
         "original source-admission metadata changed during observation")
    print(canonical({"schema": SOURCE_ADMISSION_VERIFIED_SCHEMA, "defer": pin,
        "source_admission": value, "read_only": True, "D_owner_calls": 0,
        "zero_D_calls_scope": "public_validation_CLI_and_validation_compute_or_storage_allocation_only",
        "original_source_lease_calls": 0,
        "resource_recovery_performed": False, "coding_attempts_incremented": False}).decode(), flush=True)
    return 0



def checked_validation_profile(value, control):
    validation = value["validation"]
    base_fields = {"trusted_tests", "selected_checks", "pytest_module", "import_sources", "attempt_root",
        "cpu_slots", "memory_mb", "storage_bytes", "timeout_seconds", "child_process_slots",
        "scheduler_state_path", "expected_scheduler_policy_sha256", "roots", "resource_ledger"}
    extra = {"external_fixture_profile", "normal_namespace_profiles", "expected_nodeids", "selector_case_counts"}
    source_mode = base_fields | {"ordinary_scratch_profile", "source_qualification"}
    v5 = type(validation) is dict and set(validation) in (base_fields | {"ordinary_scratch_profile"}, base_fields | {"ordinary_scratch_profile", "validation_ledger_busy_retry"}, source_mode)
    need(type(validation) is dict and set(validation) in (base_fields, base_fields | extra, base_fields | extra | {"validation_admission_retry"}, base_fields | {"ordinary_scratch_profile"}, base_fields | {"ordinary_scratch_profile", "validation_ledger_busy_retry"}, source_mode), "closed finite validation profile required")
    if "source_qualification" in validation:
        checked_source_qualification(value, control)
    v3 = set(validation) in (base_fields | extra, base_fields | extra | {"validation_admission_retry"})
    if "validation_ledger_busy_retry" in validation:
        checked_validation_ledger_busy_retry(validation["validation_ledger_busy_retry"])
    if "validation_admission_retry" in validation:
        checked_validation_admission_retry(validation["validation_admission_retry"])
    for name, maximum in (("cpu_slots", 1), ("memory_mb", 2048),
            ("storage_bytes", 67108864), ("timeout_seconds", 300), ("child_process_slots", 4 if v3 or v5 else 2)):
        need(type(validation[name]) is int and 0 < validation[name] <= maximum, "finite public validation envelope required")
    if v3:
        need(all(validation[key] == wanted and type(validation[key]) is int for key, wanted in {
            "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 67108864, "timeout_seconds": 300, "child_process_slots": 4}.items()),
            "v3 requires the explicitly reviewed unchanged 64/32 bounded validation envelope")
        checked_v3_validation_worker(control)
    if v5:
        need(type(validation["ordinary_scratch_profile"]) is dict
             and validation["ordinary_scratch_profile"] == _V5_ORDINARY_PROFILE
             and all(type(validation[k]) is int and validation[k] == n for k, n in {
                 "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 16777216,
                 "timeout_seconds": 300, "child_process_slots": 4}.items()),
             "explicit V5 ordinary 16MiB/300/1/1536/4 profile required")
    tests, checks = validation["trusted_tests"], validation["selected_checks"]
    checked_descriptor(validation["pytest_module"])
    need(type(validation["import_sources"]) is list and len(validation["import_sources"]) <= 4096,
         "finite complete normal import manifest required")
    for pin in validation["import_sources"]:
        checked_descriptor(pin)
    pytest_support = str(Path(validation["pytest_module"]["path"]).parent.parent / "_pytest/python.py")
    need(any(pin["path"] == pytest_support for pin in validation["import_sources"]),
         "original Pytest collection wrapper source must be pinned for attributed syntax failures")
    need(type(tests) is list and 0 < len(tests) <= 32, "finite maintained tests required")
    for pin in tests:
        checked_descriptor(pin)
    need(type(checks) is list and 0 < len(checks) <= 128 and len(set(checks)) == len(checks), "exact natural test nodes required")
    need(all(type(node) is str and (v3 or "::" in node) and node.split("::", 1)[0] in {r["path"] for r in tests}
             for node in checks), "only pinned maintained test nodes may execute")
    maintained = control["root"]
    need(all(any((maintained / name) in Path(pin["path"]).parents for name in ("test", "tests")) for pin in tests),
         "trusted tests must be maintained repository tests, not historical controls")
    need(type(validation["roots"]) is list and 0 < len(validation["roots"]) <= 4
         and all(Path(r).is_absolute() for r in validation["roots"]), "original finite named roots required")
    attempt_root = Path(validation["attempt_root"])
    need(attempt_root.is_absolute() and ".." not in attempt_root.parts
         and not any(p.is_symlink() for p in (attempt_root, *attempt_root.parents))
         and any(attempt_root == Path(r) or Path(r) in attempt_root.parents for r in validation["roots"]),
         "validation attempt must remain below an existing original named root")
    if v3:
        need(len({pin["path"] for pin in tests}) == len(tests)
             and all(pin["bytes"] <= 4 * 1024**2 for pin in [*tests, *validation["import_sources"]]),
             "v3 requires unique bounded immutable test and import files")
        package = Path(value["datasets_root"]) / "ipfs_datasets_py"
        directory = package / "optimizers/logic_theorem_optimizer"
        required = {str(path) for path in [*(directory / name for name in ("validation_run_resources.py", "autoencoder_daemon_resources.py",
            "resource_scheduler.py", "existing_resource_scheduler.py", "proof_resource_safety.py", "runtime_telemetry.py", "__init__.py")),
            package / "optimizers/__init__.py", package / "__init__.py", package / "router_deps.py", package / "ipfs_backend_router.py",
            package / "logic/__init__.py", package / "logic/backends/process.py"]}
        need(required <= {pin["path"] for pin in [*value["source_descriptors"], *validation["import_sources"]]},
             "complete original public v3 normal control source population is unavailable")
        pytest_main = str(Path(validation["pytest_module"]["path"]).parent.parent / "_pytest/main.py")
        need(any(pin["path"] == pytest_main for pin in validation["import_sources"]),
             "original Pytest selected-module collection source must be pinned for v3 syntax failures")
        expected_natural_nodes({"selected_checks": checks, "expected_nodeids": validation["expected_nodeids"],
            "selector_case_counts": validation["selector_case_counts"], "expected_cases": len(validation["expected_nodeids"])})
        need(validation["normal_namespace_profiles"] == control["namespace_profiles"] and len(control["namespace_profiles"]) == 1,
             "v3 normal D namespace must be the independently published control profile")
        checked_worker_fixture_profile(validation["external_fixture_profile"], python=value["python"], tests=tests, selectors=checks, expected_nodes=validation["expected_nodeids"])
    return validation


def original_external_validation_profile(policy, plan, request_pin, record):
    """Read original typed custody metadata; never inventory EXTERNAL payloads."""
    request = json_object(checked_descriptor(request_pin))
    need(request["schema"] == "owned-validation-request/v3" and "external_fixture_profile" in plan,
         "original explicit v3 validation profile required")
    external = plan["external_fixture_directory"]; attempt = Path(plan["attempt_directory"])
    need(request["external_fixture_profile"] == {"schema": "owned-validation-external-fixtures/v1",
        "external_fixture_directory": external, "max_bytes": 33554432}
        and request["declared_external_writable_paths"] == [external]
        and request["normal_namespace_profiles"] == plan["normal_namespace_profiles"]
        and request_pin["path"] == plan["public_request_path"]
        and request["storage_bytes"] == record["storage_bytes"] == 67108864,
        "original v3 request/profile/path/storage binding differs")
    bound = record["external_fixture_profile"]
    fields = {"schema", "external_fixture_directory", "request_sha256", "charge_key", "charged_bytes", "root_identity"}
    key = "external-fixtures-" + request_pin["sha256"][:32]
    need(type(bound) is dict and set(bound) == fields and bound["schema"] == "owned-validation-external-fixtures/v1"
         and bound["external_fixture_directory"] == external and bound["request_sha256"] == request_pin["sha256"]
         and bound["charge_key"] == key and bound["charged_bytes"] == 33554432
         and record["external_charges"] == {key: 33554432}, "original immutable external32 charge is unproven")
    identity = bound["root_identity"]
    need(type(identity) is dict and set(identity) == {"path", "device", "inode"} and identity["path"] == external
         and type(identity["device"]) is type(identity["inode"]) is int and identity["device"] >= 0 and identity["inode"] > 0, "original typed external root identity required")
    receipts = record["external_fixture_original_receipts"]
    need(type(receipts) is dict and set(receipts) == {"path", "charge"}, "two original external prechild receipts required")
    need(receipts["path"]["path"] == str(attempt / "logs/external-fixture-path.json")
         and receipts["charge"]["path"] == str(attempt / "logs/external-fixture-charge.json"), "original receipt locations differ")
    path = json_object(checked_descriptor(receipts["path"])); charge = json_object(checked_descriptor(receipts["charge"]))
    need(set(path) == {"schema", "request_sha256", "reservation_id", "external_fixture_directory", "root_identity", "outside_attempt", "owner",
            "owner_started_monotonic", "owner_deadline_monotonic", "admitted", "Lean_admitted", "test_or_task_acceptance_asserted"}
         and path["schema"] == "owned-validation-external-original-path/v2" and path["request_sha256"] == request_pin["sha256"]
         and path["reservation_id"] == record["reservation_id"] and path["external_fixture_directory"] == external
         and path["root_identity"] == identity and path["outside_attempt"] is True
         and path["owner"] == {"pid": record["owner_pid"], "birth": record["owner_birth"]}, "original external path custody differs")
    need(type(record["validation_timeout_seconds"]) is int
         and record["validation_timeout_seconds"] == request["timeout_seconds"] == policy["validation"]["timeout_seconds"],
         "original v3 row timeout differs from its validated request and policy")
    started, deadline = path["owner_started_monotonic"], path["owner_deadline_monotonic"]
    need(type(started) in (int, float) and type(deadline) in (int, float) and math.isfinite(started) and math.isfinite(deadline)
         and started > 0 and deadline == started + float(record["validation_timeout_seconds"]),
         "original bounded SDK clock differs from its own request timeout")
    accounting = charge["accounting_result"]
    need(charge["schema"] == "owned-validation-external-original-charge/v1" and charge["request_sha256"] == request_pin["sha256"]
         and charge["reservation_id"] == record["reservation_id"] and charge["path_receipt"] == receipts["path"]
         and accounting["key"] == key and accounting["bytes"] == 33554432 and accounting["already_charged"] is False,
         "original prechild immutable external charge receipt differs")
    for receipt in (path, charge):
        need(receipt["admitted"] is receipt["Lean_admitted"] is receipt["test_or_task_acceptance_asserted"] is False,
             "resource receipts cannot assert semantic acceptance")
    disposition = record["external_fixture_disposition"]
    need(disposition["schema"] == "owned-validation-external-disposition-observation/v1"
         and disposition["request_sha256"] == request_pin["sha256"] and disposition["reservation_id"] == record["reservation_id"]
         and disposition["root_identity"] == identity and disposition["stable_repeated_inventory"] is True
         and disposition["all_regular_hardlinks_internal"] is True
         and all(disposition[field] is False for field in ("targets_followed", "special_payloads_opened", "regular_payloads_opened",
             "ordinary_external_artifact_durability_asserted", "admitted", "Lean_admitted", "test_or_task_acceptance_asserted")),
         "original terminal typed external disposition is unproven")
    need(type(disposition["apparent_bytes"]) is int and 0 <= disposition["apparent_bytes"] <= 33554432
         and type(disposition["entry_count"]) is int and disposition["entry_count"] >= 0
         and type(disposition["typed_counts"]) is dict and set(disposition["typed_counts"]) == {"regular", "directory", "symlink", "fifo"}
         and all(type(value) is int and value >= 0 for value in disposition["typed_counts"].values())
         and sum(disposition["typed_counts"].values()) == disposition["entry_count"],
         "original typed external accounting counters differ")
    pin = disposition["manifest"]
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"} and type(pin["bytes"]) is int
         and 0 < pin["bytes"] <= 16 * 1024**2 and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"])
         and Path(pin["path"]).parent == attempt / "logs", "bounded original OWNED typed manifest required")
    raw = ordinary_bytes(Path(pin["path"]), 16 * 1024**2)
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "original typed manifest bytes changed")
    manifest = json_object(raw)
    need(manifest["schema"] == "qualified-external-synthetic-filesystem-disposition/v1" and manifest["root_identity"] == identity
         and hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
             ensure_ascii=True, allow_nan=False).encode()).hexdigest() == disposition["inventory_sha256"]
         and Path(pin["path"]).name == "external-synthetic-disposition-" + disposition["inventory_sha256"] + ".json"
         and manifest["typed_counts"] == disposition["typed_counts"] and manifest["entry_count"] == disposition["entry_count"]
         and manifest["apparent_bytes"] == disposition["apparent_bytes"] <= 33554432
         and manifest["all_regular_hardlinks_internal"] is True
         and all(manifest[field] is False for field in ("targets_followed", "special_payloads_opened", "regular_payloads_opened", "ordinary_external_artifact_durability_asserted")),
         "original terminal typed manifest header differs")
    return {"request_schema": request["schema"], "external_fixture_profile": bound,
        "original_receipts": receipts, "terminal_disposition": disposition}


def _v5_lifecycle_owner_paths(datasets_root: Path) -> set[str]:
    directory = datasets_root / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    return {str(datasets_root / "scripts/ops/legal_ir/run_owned_validation.py"),
            *(str(directory / name) for name in ("validation_run_resources.py",
                "autoencoder_daemon_resources.py", "resource_scheduler.py",
                "existing_resource_scheduler.py", "proof_resource_safety.py"))}


def _require_v5_lifecycle_source_pins(datasets_root: Path, pins: dict) -> None:
    need(all(type(pins.get(path)) is str and _SHA.fullmatch(pins[path])
             for path in _v5_lifecycle_owner_paths(datasets_root)),
         "explicit original lifecycle owner source pins required")


def validation_execution_request(policy, plan, plan_pin):
    """Bind the public original request to its immutable plan without a cycle."""
    v = policy["validation"]; attempt = Path(plan["attempt_directory"])
    module = Path(policy["control_release"]["root"]) / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"
    request = {"schema": "owned-validation-request/v1", "argv": [policy["python"], "-I", "-B", str(module),
        "--plan", plan_pin["path"], "--plan-sha256", plan_pin["sha256"]],
        "cwd": plan["candidate_root"], "selected_checks": plan["selected_checks"],
        "source_sha256": {**plan["original_public_control_pins"], plan_pin["path"]: plan_pin["sha256"]},
        **{key: v[key] for key in ("timeout_seconds", "storage_bytes", "memory_mb", "cpu_slots", "child_process_slots",
            "scheduler_state_path", "expected_scheduler_policy_sha256", "roots")},
        "attempt_directory": str(attempt), "outputs": {key: str(attempt / key) for key in ("pytest", "tmp", "cache", "db", "exports", "logs")},
        "declared_external_writable_paths": [], "process_scope": "sampled_birth_bound_descendants"}
    if "external_fixture_profile" in plan:
        external = plan["external_fixture_directory"]
        request.update(schema="owned-validation-request/v3", normal_namespace_profiles=plan["normal_namespace_profiles"],
            external_fixture_profile={"schema": "owned-validation-external-fixtures/v1", "external_fixture_directory": external, "max_bytes": 33554432},
            declared_external_writable_paths=[external])
        package = Path(policy["datasets_root"]) / "ipfs_datasets_py"
        directory = package / "optimizers/logic_theorem_optimizer"
        required = [module, Path(policy["datasets_root"]) / "scripts/ops/legal_ir/run_owned_validation.py",
            *(directory / name for name in ("validation_run_resources.py", "autoencoder_daemon_resources.py", "resource_scheduler.py",
                "existing_resource_scheduler.py", "proof_resource_safety.py", "runtime_telemetry.py", "__init__.py")),
            package / "optimizers/__init__.py", package / "__init__.py", package / "router_deps.py", package / "ipfs_backend_router.py",
            package / "logic/__init__.py", package / "logic/backends/process.py"]
        need(all(str(path) in request["source_sha256"] for path in required), "complete original public v3 control source pins required")
    if "ordinary_scratch_profile" in plan:
        request.update(schema="owned-validation-request/v5", ordinary_scratch_profile=dict(plan["ordinary_scratch_profile"]))
        _require_v5_lifecycle_source_pins(Path(policy["datasets_root"]), request["source_sha256"])
    need(len(request["source_sha256"]) <= 256 and len(request["selected_checks"]) <= 256,
         "public original request source/selection caps remain unchanged")
    return request


def checked_policy(path: Path, sha: str, *, current_owner=True) -> dict:
    need(_SHA.fullmatch(sha) is not None, "owner policy hash required")
    raw = ordinary_bytes(path)
    need(hashlib.sha256(raw).hexdigest() == sha, "owner policy changed")
    value = json_object(raw)
    need(set(value) == {"schema", "datasets_root", "python", "source_descriptors",
        "validation", "adoption", "control_release"} and value["schema"] == POLICY_SCHEMA,
        "closed maintained owner policy required")
    control = checked_control_release(value, current_owner=current_owner)
    need(str(path) not in control["pins"], "control policy must be excluded from its source manifest")
    sources = value["source_descriptors"]
    need(type(sources) is list and 0 < len(sources) <= 128
         and len({r["path"] for r in sources}) == len(sources), "finite owner source population required")
    for row in sources:
        checked_descriptor(row)
    need(descriptor(control["owner"]) in sources, "authenticated maintained control owner source is not pinned")
    need(descriptor(control["worker"]) in sources, "trusted stdlib-only control worker is not pinned")
    d = Path(value["datasets_root"])
    need(d.is_absolute() and not any(p.is_symlink() for p in (d, *d.parents)), "canonical D root required")
    required = {str(d / "scripts/ops/legal_ir/run_owned_validation.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_run_resources.py")}
    need(required <= {r["path"] for r in sources}, "public D owner, source lease and storage disposition inspector must be pinned")
    recovery_path = str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py")
    recovery_pin = next(pin for pin in sources if pin["path"] == recovery_path)
    # Metadata-only interface preflight: deploying a profile without its public
    # negative-storage observer must fail before any coding attempt is spent.
    declared = {node.name for node in ast.parse(checked_descriptor(recovery_pin)).body if isinstance(node, ast.FunctionDef)}
    need({"inspect_validation_reservation", "inspect_validation_disposition"} <= declared,
         "qualified public negative-storage observation API is unavailable")
    if _v5_ordinary_validation(value):
        cli_pin = next(pin for pin in sources if pin["path"] == str(d / "scripts/ops/legal_ir/run_owned_validation.py"))
        functions = {n.name for n in ast.parse(checked_descriptor(cli_pin)).body if isinstance(n, ast.FunctionDef)}
        need("inspect_owned_validation_productive_native_closed_v1" in functions,
             "public V5 productive-native source inspector is unavailable")
        need("inspect_owned_validation_no_child_native_closed_v5" in functions,
             "public V5 acquired-no-child source inspector is unavailable")
        directory = d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
        required_v5 = _v5_lifecycle_owner_paths(d)
        required_v5.update(str(path) for path in (directory / "autoencoder_source_lease.py",
            directory / "autoencoder_resource_recovery.py", directory / "runtime_telemetry.py", directory / "validation_pytest.py",
            directory / "__init__.py", d / "ipfs_datasets_py/optimizers/__init__.py",
            d / "ipfs_datasets_py/__init__.py", d / "ipfs_datasets_py/router_deps.py",
            d / "ipfs_datasets_py/ipfs_backend_router.py", d / "ipfs_datasets_py/logic/__init__.py",
            d / "ipfs_datasets_py/logic/backends/process.py"))
        all_sources = [*sources, *value["validation"]["import_sources"]]
        need(required_v5 <= {pin["path"] for pin in all_sources}, "complete V5 normal lifecycle control closure must be pinned")
        need(all(pin["bytes"] <= 4 * 1024**2 for pin in [*all_sources, *value["validation"]["trusted_tests"]]),
             "V5 source/test files must fit the unchanged 4MiB boundary")
    need(Path(value["python"]).is_absolute(), "exact Python executable required")
    checked_validation_profile(value, control)
    if "external_fixture_profile" in value["validation"]:
        need(len(raw) <= 4 * 1024**2 and all(pin["bytes"] <= 4 * 1024**2 for pin in sources),
             "v3 original public source-control files must fit the unchanged 4 MiB read boundary")
    return value


def frozen_candidate(path: Path, commitment: str, root: Path) -> dict:
    value = json_object(ordinary_bytes(path))
    need(value.get("stage") == "candidate_ready" and value.get("validation_passed") is False
         and value.get("runtime_adopted") is False, "original frozen candidate required")
    unsigned = {k: v for k, v in value.items() if k != "freeze_sha256"}
    need(value.get("freeze_sha256") == digest(unsigned), "candidate freeze commitment differs")
    need(value.get("source_commitment") == commitment and value["binding"]["candidate_root"] == str(root),
         "candidate source/root differs from the original binding")
    need(type(value["manifest"]) is dict and 0 < len(value["manifest"]) <= 4096, "finite supported source manifest required")
    for relative, row in value["manifest"].items():
        rel = Path(relative)
        need(not rel.is_absolute() and ".." not in rel.parts and rel.parts, "relative candidate file required")
        need(type(row) is dict and set(row) == {"bytes", "sha256", "executable"}, "typed frozen source row required")
        observed = descriptor(root / rel)
        need((observed["bytes"], observed["sha256"]) == (row["bytes"], row["sha256"]), "candidate source changed")
    return value


def _v5_control_namespace_profiles(policy):
    """Project only the already authenticated immutable control declaration.

    The public validate entry checks the complete policy first. Shape-only old
    plan-builder callers without a control manifest keep the empty projection;
    they still cannot pass checked_policy or acquire any original authority.
    """
    if 'manifest' not in policy['control_release']:
        return []
    return checked_control_release(policy, current_owner=False)['namespace_profiles']


def _v5_validation_namespace_evidence(policy, plan, report):
    """Bind saved V5 namespace DATA without changing legacy/V3 validation."""
    if not _v5_ordinary_validation(policy):
        return
    profiles = (policy["validation"]["normal_namespace_profiles"] if _reviewed_parent_policy(policy)
                else _v5_control_namespace_profiles(policy))
    need(canonical(plan.get('normal_namespace_profiles', [])) == canonical(profiles)
         and canonical(report.get('declared_normal_namespace_profiles', [])) == canonical(profiles),
         'original V5 immutable control namespace declaration differs')


def validation_plan(policy, policy_pin, frozen, freeze_pin, root, invocation, *, source_parent_context=None):
    binding = frozen["binding"]; v = policy["validation"]
    qualification = checked_source_qualification(policy)
    if qualification is not None:
        need(root == qualification["root"] and freeze_pin == qualification["snapshot_ref"]
             and canonical(frozen) == canonical(qualification["snapshot"]), "immutable plan snapshot differs")
    request_generation = validation_request_generation(policy, frozen, invocation, live=True)
    attempt = Path(v["attempt_root"]) / binding["namespace"] / binding["board_id"] / request_generation
    tests = v["trusted_tests"]; v3 = "external_fixture_profile" in v
    need(qualification is not None or all(Path(pin["path"]) != root and root not in Path(pin["path"]).parents for pin in tests),
         "original trusted tests must stay outside the mutable candidate")
    pins = {}
    for pin in [*policy["source_descriptors"], *tests, v["pytest_module"], *v["import_sources"],
            *[{"path": str(root / relative), "sha256": row["sha256"]} for relative, row in frozen["manifest"].items()]]:
        need(pin["path"] not in pins or pins[pin["path"]] == pin["sha256"], "normal source manifest collision")
        pins[pin["path"]] = pin["sha256"]
    need(len(pins) <= 4096, "complete normal import manifest exceeds its original finite bound")
    controls = {pin["path"]: pin["sha256"] for pin in policy["source_descriptors"] + tests}
    controls[policy_pin["path"]] = policy_pin["sha256"]; controls[freeze_pin["path"]] = freeze_pin["sha256"]
    required = [] if qualification is not None else [str(root / relative) for relative in frozen["changed_paths"] if relative.endswith(".py")]
    need(qualification is not None or required, "this finite validation profile requires an actually changed Python body")
    plan = {"schema": WORK_SCHEMA, "policy": policy_pin, "freeze": freeze_pin,
        "source_commitment": frozen["source_commitment"], "candidate_root": str(root),
        "attempt_directory": str(attempt), "source_pins": pins, "original_public_control_pins": controls,
        "selected_checks": v["selected_checks"], "expected_cases": len(v["selected_checks"]), "required_candidate_bodies": required}
    if qualification is not None:
        plan.update(source_qualification=qualification["profile"], required_source_bodies=qualification["required_source_bodies"])
        manifest_pin = policy["control_release"]["manifest"]
        pins[manifest_pin["path"]] = manifest_pin["sha256"]
        pins[freeze_pin["path"]] = freeze_pin["sha256"]
        controls[manifest_pin["path"]] = manifest_pin["sha256"]
    if v3:
        prepared_pin, copies = trusted_validation_test_copies(policy, frozen, freeze_pin, root)
        by_path = {row["trusted"]["path"]: row["actual"]["path"] for row in copies}
        def mapped(node):
            path, separator, suffix = node.partition("::")
            need(path in by_path, "selected test does not belong to one immutable test copy")
            return by_path[path] + (separator + suffix if separator else "")
        selectors = {node: mapped(node) for node in v["selected_checks"]}
        plan.update(selected_checks=[selectors[node] for node in v["selected_checks"]],
            expected_nodeids=[mapped(node) for node in v["expected_nodeids"]],
            selector_case_counts={selectors[node]: count for node, count in v["selector_case_counts"].items()},
            expected_cases=len(v["expected_nodeids"]), trusted_test_preparation=prepared_pin, trusted_test_copies=copies,
            normal_namespace_profiles=v["normal_namespace_profiles"],
            external_fixture_profile=mapped_validation_fixture_profile(v["external_fixture_profile"], copies),
            external_fixture_directory=str(attempt.parent / (attempt.name + "-external-fixtures")),
            public_request_path=str(validation_control_directory(policy, frozen) / "execution-request.json"))
        expected_natural_nodes(plan)
        d = Path(policy["datasets_root"])
        for pin in v["import_sources"]:
            if d in Path(pin["path"]).parents:
                controls[pin["path"]] = pin["sha256"]
        for row in copies:
            controls[row["actual"]["path"]] = row["actual"]["sha256"]
        controls[prepared_pin["path"]] = prepared_pin["sha256"]
    if _v5_ordinary_validation(policy):
        plan["ordinary_scratch_profile"] = dict(v["ordinary_scratch_profile"])
        profiles = _v5_control_namespace_profiles(policy)
        if profiles:
            plan["normal_namespace_profiles"] = json_object(canonical({"profiles": profiles}))["profiles"]
        # Source closure only: no V3 aliases or external fixture grants. The
        # sole ordinary namespace is the existing sealed control declaration.
        d = Path(policy["datasets_root"])
        for pin in v["import_sources"]:
            if d in Path(pin["path"]).parents:
                controls[pin["path"]] = pin["sha256"]
    if _immutable_source_parent_enabled(policy):
        need(source_parent_context is not None, "trusted immutable parent plan context required")
        parent_fields = _immutable_source_parent_plan_fields(source_parent_context, invocation)
        need(parent_fields["execution_request_path"] == str(validation_control_directory(policy, frozen, invocation) / "execution-request.json"),
             "immutable parent plan must bind actual public request path")
        plan.update(parent_fields)
        plan.setdefault("normal_namespace_profiles", [])
    else:
        need(source_parent_context is None, "legacy plan cannot borrow immutable parent context")
    generation_path = invocation/"ledger-busy-generation.json"
    if generation_path.exists():
        generation_pin = descriptor(generation_path)
        pins[generation_pin["path"]] = generation_pin["sha256"]
        controls[generation_pin["path"]] = generation_pin["sha256"]
        ticket_pin = json_object(checked_descriptor(generation_pin))["spent_slot_ref"]
        checked_descriptor(ticket_pin)
        pins[ticket_pin["path"]] = ticket_pin["sha256"]
        controls[ticket_pin["path"]] = ticket_pin["sha256"]
    need(len(pins) <= 4096, "complete normal import manifest exceeds its original finite bound")
    need(len(controls) <= 253, "original public request control cap reached; no cap increase")
    return plan




_V5_ORDINARY_PROFILE = {"schema": "owned-validation-ordinary-scratch-live/v1"}
_V5_PROOF_RESERVE_SECONDS = 20.0
_V5_ENGINE_CLEANUP_SECONDS = 10.0  # Existing TERM5 plus final drain5, inside D.
_V5_EXECUTION_FIELDS = ("schema", "completion_kind", "process_pid", "process_returncode",
    "leader_waited", "output_capture", "work_deadline_monotonic", "cleanup_deadline_monotonic",
    "observed_at_monotonic")

def _v5_ordinary_validation(policy):
    value = policy.get("validation")
    return type(value) is dict and "ordinary_scratch_profile" in value


class _V5ValidationSourceCustody:
    """Source custody only; original engine retains all process cleanup."""
    def __init__(self, policy, argv, cwd, timeout, context):
        self.policy, self.argv, self.process = policy, list(argv), None
        self.stack = ExitStack(); self.receipts = []; self.closed = False
        self.started = time.monotonic(); self.deadline = self.started + timeout
        self.identity = None; self.context = context; self.identity_error = None
        self.engine_source = None; self.engine_returned = False; self.wait_ref = None
        self.work_deadline = self.deadline - _V5_PROOF_RESERVE_SECONDS - _V5_ENGINE_CLEANUP_SECONDS
        self.cleanup_deadline = self.deadline - _V5_PROOF_RESERVE_SECONDS
        self.spawn_attempted = False; self.spawn_returned_at = None; self.identity_observed_at = None
        self.execution_ref = None; self.execution_observation = None; self.engine = None
        self.engine_started_at = None; self.engine_finished_at = None; self.engine_runner = None
        self.process_count = 0; self.source_exit_finished_at = None
        v = policy["validation"]
        delegated_source = _immutable_source_parent_enabled(policy)
        need(v["ordinary_scratch_profile"] == _V5_ORDINARY_PROFILE
             and type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
             and type(context) is dict and set(context) == ({"plan", "request_pin", "invocation", "frozen", "source_parent"}
                 if delegated_source else {"plan", "request_pin", "invocation", "frozen"}),
             "explicit original V5 validation custody context required")
        self.request_pin = context["request_pin"]; self.request_path = Path(self.request_pin["path"])
        self.raw = checked_descriptor(self.request_pin); self.request = json_object(self.raw)
        self.d = Path(policy["datasets_root"]); self.cwd = Path(cwd)
        self.cli_path = self.d / "scripts/ops/legal_ir/run_owned_validation.py"
        self.ledger = Path(v["resource_ledger"])
        need(self.argv == [policy["python"], "-I", "-B", str(self.cli_path), "--request",
             str(self.request_path), "--expected-request-sha256", self.request_pin["sha256"],
             "--resource-ledger", str(self.ledger)], "exact original V5 owner invocation required")
        q = self.request
        need(q["schema"] == "owned-validation-request/v5" and q["ordinary_scratch_profile"] == _V5_ORDINARY_PROFILE
             and q["declared_external_writable_paths"] == [] and q["process_scope"] == "sampled_birth_bound_descendants"
             and q["selected_checks"] == context["plan"]["selected_checks"]
             and q["attempt_directory"] == context["plan"]["attempt_directory"]
             and q["cwd"] == str(self.cwd) == context["plan"]["candidate_root"]
             and all(q[k] == v[k] for k in ("cpu_slots", "memory_mb", "storage_bytes", "timeout_seconds",
                 "child_process_slots", "scheduler_state_path", "expected_scheduler_policy_sha256", "roots")),
             "V5 original request/context differs")
        need(type(q["source_sha256"]) is dict and 0 < len(q["source_sha256"]) <= 256, "bounded V5 source pins required")
        _require_v5_lifecycle_source_pins(self.d, q["source_sha256"])
        self.cli_pin = next(p for p in policy["source_descriptors"] if p["path"] == str(self.cli_path))
        need(q["source_sha256"].get(str(self.cli_path)) == self.cli_pin["sha256"], "current V5 CLI pin differs")
        if delegated_source:
            _immutable_source_parent_check_context(self)
        try:
            # Source42 performs Git subprocesses: acquire before GLOBAL Popen interposition.
            lease = public_source_lease(policy)
            if _reviewed_parent_policy(policy):
                self.launch_clock()
                self.reviewed_source_root_pin = _reviewed_parent_source_root_pin(policy)
                self.reviewed_source_root_binding = _reviewed_parent_prepare_source_root_binding(
                    self.reviewed_source_root_pin, (self.d, self.cwd))
                self.launch_clock()
            elif delegated_source:
                self.immutable_source_binding = _immutable_source_owner(policy)
                need(lease is self.immutable_source_binding[0].source_lease, "same original Source42 owner required")
            for root in (self.d, self.cwd):
                # Source waits consume the one allowance before the full CLI window.
                source_wait = self.launch_clock() - q["timeout_seconds"]
                manager = lease(root, mode="read", timeout_seconds=source_wait)
                held = self.stack.enter_context(manager)
                self.receipts.append(held.to_dict())
                if _reviewed_parent_policy(policy):
                    _reviewed_parent_record_source_fd(self, manager, held)
                elif delegated_source:
                    _immutable_source_parent_record_fd(self, manager, held, root)
            self.launch_clock()
            self.guard()
            if _reviewed_parent_policy(policy):
                _reviewed_parent_publish_source_observation(self)
            elif delegated_source:
                _immutable_source_parent_publish(self)
            name = "_repair_original_v5_source_custody_cli"
            spec = importlib.util.spec_from_file_location(name, self.cli_path)
            need(spec is not None and type(spec.loader) is importlib.machinery.SourceFileLoader
                 and spec.origin == str(self.cli_path), "original canonical CLI loader required")
            self.clock()
            cli_raw = checked_descriptor(self.cli_pin)
            need(type(_ORIGINAL_COMPILE) is types.BuiltinFunctionType
                 and _ORIGINAL_COMPILE.__module__ == "builtins"
                 and _ORIGINAL_COMPILE.__name__ == "compile", "original raw-source compiler required")
            self.cli = importlib.util.module_from_spec(spec)
            # Execute exactly authenticated canonical raw bytes, never a pyc.
            exec(_ORIGINAL_COMPILE(cli_raw, str(self.cli_path), "exec", dont_inherit=True), vars(self.cli))
            self.inspector = self.cli.inspect_owned_validation_productive_native_closed_v1
            self.no_child = self.cli.inspect_owned_validation_no_child_native_closed_v5
            self.no_grant = self.cli.classify_owned_validation_source_custody
            need(type(self.cli) is types.ModuleType and self.cli.__file__ == str(self.cli_path)
                 and self.cli.__spec__.origin == str(self.cli_path)
                 and type(self.inspector) is types.FunctionType
                 and self.inspector.__globals__ is vars(self.cli)
                 and self.inspector.__code__.co_filename == str(self.cli_path)
                 and self.inspector.__name__ == "inspect_owned_validation_productive_native_closed_v1",
                 "original canonical productive-native inspector origin differs")
            for function, expected_name in ((self.no_child, "inspect_owned_validation_no_child_native_closed_v5"),
                    (self.no_grant, "classify_owned_validation_source_custody")):
                need(type(function) is types.FunctionType and function.__globals__ is vars(self.cli)
                     and function.__code__.co_filename == str(self.cli_path)
                     and function.__name__ == expected_name, "original canonical no-child inspector origin differs")
            self.guard()
        except BaseException as original_error:
            try:
                self.stack.close(); self.closed = True
            except BaseException as close_error:
                self.hold(close_error)
            raise original_error

    def clock(self):
        need(time.monotonic() < self.deadline, "original V5 source/native proof budget expired")

    def launch_clock(self):
        """Refuse before actual Popen unless the unchanged request window fits."""
        self.clock()
        remaining = self.work_deadline - time.monotonic()
        need(remaining > 0, "original V5 worker window expired before launch")
        need(remaining >= self.request["timeout_seconds"],
             "original V5 full nominal request window does not fit before launch")
        return remaining

    def record_execution(self, completed, engine, records):
        """Bounded observations only, persisted even when later proof is unknown."""
        self.engine = engine; self.engine_finished_at = time.monotonic()
        self.process_count = len(records)
        exact_result = type(completed) is getattr(engine, "CommandResult", None)
        observation = getattr(completed, "execution_observation", None) if exact_result else None
        exact_observation = type(observation) is getattr(engine, "CommandExecutionObservation", None)
        self.execution_observation = observation if exact_observation else None
        # Never serialize arbitrary caller objects or a valid-looking stdout prefix.
        def finite(value):
            if type(value) not in (int, float):
                return None
            try:
                return value if math.isfinite(value) else None
            except OverflowError:
                return None
        observed = None
        if exact_observation:
            observed = {}
            for name in _V5_EXECUTION_FIELDS:
                value = getattr(observation, name, None)
                observed[name] = (value if type(value) is bool or value is None else
                    value if type(value) is str and len(value) <= 128 else finite(value))
        process = self.process
        identity = self.identity
        bounded_identity = (dict(identity) if type(identity) is dict and set(identity) == {"pid", "birth"}
            and type(identity["pid"]) is int and type(identity["birth"]) is str and len(identity["birth"]) <= 32 else None)
        value = {"schema": "agent-supervisor/v5-runner-execution-observation@1", "request": self.request_pin,
            "engine_source": self.engine_source, "exact_command_result": exact_result,
            "exact_execution_observation": exact_observation, "execution_observation": observed,
            "public_returncode": completed.returncode if exact_result and type(completed.returncode) is int else None,
            "actual_process_pid": None if process is None else process.pid,
            "actual_process_returncode": None if process is None else process.returncode,
            "actual_Popen_returned": process is not None, "spawn_attempted": self.spawn_attempted,
            "process_count": self.process_count, "runner_identity": bounded_identity,
            "identity_observation_error": self.identity_error,
            "engine_returned": self.engine_returned, "engine_started_at_monotonic": finite(self.engine_started_at),
            "engine_finished_at_monotonic": finite(self.engine_finished_at),
            "spawn_returned_at_monotonic": finite(self.spawn_returned_at),
            "identity_observed_at_monotonic": finite(self.identity_observed_at),
            "authority_deadline_monotonic": self.deadline, "work_deadline_monotonic": self.work_deadline,
            "cleanup_deadline_monotonic": self.cleanup_deadline,
            "requested_nominal_window_seconds": self.request["timeout_seconds"],
            "window_scope": "CLI startup and scheduling consume this window; no inner SDK productive-time guarantee",
            "source_closed": False, "native_closed": False,
            "storage_release_authorized": False, "retry_authorized": False}
        self.execution_ref = new_json(Path(self.context["invocation"]) / "runner-execution-observation.json", value)

    def checked_execution(self, completed, witness):
        engine = self.engine; observation = self.execution_observation; process = self.process
        need(type(engine) is types.ModuleType and engine.run_command is self.engine_runner
             and type(engine.run_command) is types.FunctionType
             and engine.run_command.__globals__ is vars(engine)
             and engine.run_command.__code__.co_filename == self.engine_source["path"]
             and engine.__file__ == self.engine_source["path"], "original V5 engine emitter origin differs")
        need(type(completed) is engine.CommandResult and type(completed.command) is tuple
             and list(completed.command) == self.argv
             and type(completed.returncode) is int and type(completed.stdout) is str and type(completed.stderr) is str,
             "exact original V5 engine command result required")
        need(type(observation) is engine.CommandExecutionObservation
             and completed.execution_observation is observation
             and observation.__class__.__module__ == engine.__name__
             and dataclasses.is_dataclass(observation) and observation.__dataclass_params__.frozen
             and tuple(field.name for field in dataclasses.fields(observation)) == _V5_EXECUTION_FIELDS,
             "typed immutable original V5 execution observation is unknown")
        need(observation.schema == "command-execution-observation/v1"
             and type(observation.completion_kind) is str and observation.completion_kind in ("completed", "timeout")
             and type(observation.output_capture) is str and observation.output_capture in ("complete", "incomplete", "unknown")
             and type(observation.leader_waited) in (bool, type(None)), "closed V5 execution observation vocabulary required")
        need(type(process) is _ORIGINAL_POPEN and process.args == self.argv and self.process_count == 1
             and witness["actual_Popen_returned"] is True
             and type(process.pid) is int and type(observation.process_pid) is int
             and observation.process_pid == process.pid
             and type(process.returncode) is int and type(observation.process_returncode) is int
             and process.returncode == observation.process_returncode == witness["actual_process_returncode"]
             and observation.leader_waited is True and witness["actual_process_waited"] is True,
             "genuine original V5 same-Popen wait is unknown")
        identity = witness["runner_identity"]
        need(type(identity) is dict and set(identity) == {"pid", "birth"} and identity == self.identity
             and type(identity["pid"]) is int and identity["pid"] == process.pid
             and type(identity["birth"]) is str and 0 < len(identity["birth"]) <= 32
             and identity["birth"].isascii() and identity["birth"].isdigit() and int(identity["birth"]) > 0,
             "genuine original V5 process birth is unknown")
        times = (self.engine_started_at, self.engine_finished_at, self.spawn_returned_at,
            self.identity_observed_at, observation.observed_at_monotonic,
            observation.work_deadline_monotonic, observation.cleanup_deadline_monotonic)
        need(all(type(value) in (int, float) and math.isfinite(value) for value in times)
             and self.started <= self.engine_started_at <= self.spawn_returned_at
             <= self.identity_observed_at <= observation.observed_at_monotonic <= self.engine_finished_at
             and observation.work_deadline_monotonic == self.work_deadline
             and observation.cleanup_deadline_monotonic == self.cleanup_deadline,
             "original V5 observation clock/deadline binding differs")
        if observation.completion_kind == "completed":
            need(completed.returncode == process.returncode, "original V5 completed public/actual code differs")
        else:
            need(completed.returncode == 124, "original V5 timeout public policy code differs")
        return observation

    def guard(self):
        self.clock()
        need(checked_descriptor(self.request_pin) == self.raw, "original V5 request changed")
        for path, expected in self.request["source_sha256"].items():
            self.clock()
            need(type(path) is str and type(expected) is str and _SHA.fullmatch(expected)
                 and hashlib.sha256(ordinary_bytes(Path(path), 4 * 1024**2)).hexdigest() == expected,
                 "original V5 source changed")

    def hold(self, error):
        """No authority transfer: keep this actual Popen and Source stack reachable."""
        value = {"schema": "agent-supervisor/v5-validation-source-custody-unknown@1",
            "request": self.request_pin, "source_closed": self.closed, "native_closed": False,
            "original_Popen_recorded": self.process is not None, "original_owner_identity": self.identity,
            "actual_process_pid": None if self.process is None else self.process.pid,
            "actual_process_returncode": None if self.process is None else self.process.returncode,
            "identity_observation_error": self.identity_error, "engine_source": self.engine_source,
            "engine_returned": self.engine_returned, "original_wait": self.wait_ref,
            "execution_observation": self.execution_ref, "spawn_attempted": self.spawn_attempted,
            "source_receipts": self.receipts, "authority_deadline_monotonic": self.deadline,
            "failure_type": type(error).__name__[:128], "storage_release_authorized": False,
            "retry_authorized": False, "further_productive_work": False, "Root_intervention_required": True}
        try:
            new_json(Path(self.context["invocation"]) / "source-custody-unknown.json", value)
        except BaseException:
            pass  # Preserve actual process/context even if evidence storage fails.
        try: print(canonical(value).decode(), flush=True)
        except BaseException: pass
        while True:
            try: time.sleep(.1)
            except BaseException: pass

    def close_failed_operational_verification(self, completed, witness, error):
        """Close only this proven original's Source scopes, never accept/retry it.

        Any absent, changed, incomplete or expired original proof stays UNKNOWN.
        No inspector, resource mutation, new census or new deadline is added.
        """
        try:
            need(isinstance(error, Exception) and not self.closed,
                 "ordinary failed verification on the same open custody required")
            self.clock()
            if self.policy.get("schema") in ("reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3", "reviewed-definition-realization-parent/v1"):
                if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
                        or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
                    raise ValueError("reviewed parent callable bindings changed: guard")
                _reviewed_parent_guard()
            need(type(getattr(self, "_released_operational_raw", None)) is bytes
                 and type(getattr(self, "_released_stdout_raw", None)) is bytes
                 and type(getattr(self, "_released_wait_raw", None)) is bytes,
                 "independently verified pre-metadata original closure is absent")
            observation = self.checked_execution(completed, witness)
            need(observation.completion_kind == "completed" and observation.output_capture == "complete"
                 and completed.returncode == self.process.returncode == 0
                 and completed.stdout.encode("utf-8") == self._released_stdout_raw,
                 "same complete original successful SDK terminal required")
            need(checked_descriptor(self.request_pin) == self.raw
                 and checked_descriptor(self.wait_ref) == self._released_wait_raw,
                 "original request or waited terminal bytes changed")
            terminal = json_object(self._released_stdout_raw)
            operational = _validation_released_operational_evidence_v1(
                self.context["plan"], self.request_pin, terminal, Path(self.context["invocation"]))
            need(_validation_released_operational_projection_v1(operational, terminal)
                 == self._released_operational_raw, "original operational proof changed")
            self.guard(); self.clock()
        except Exception:
            return False
        try:
            self.stack.close()
        except BaseException:
            return False
        # Irreversible Source closure precedes every later evidence write.
        self.closed = True
        self.source_exit_finished_at = time.monotonic()
        value = {"schema": "agent-supervisor/v5-validation-failed-operational-source-closeout@1",
            "request": self.request_pin, "original_wait": self.wait_ref,
            "execution_observation": self.execution_ref,
            "closure_kind": "released_original_metadata_verification_failed",
            "SDK_returncode": completed.returncode,
            "verification_error_type": type(error).__name__[:128],
            "source_closed": True, "native_closed": True, "storage_status": "released",
            "source_receipts": self.receipts,
            "operational_projection_sha256": hashlib.sha256(self._released_operational_raw).hexdigest(),
            "authority_deadline_monotonic": self.deadline,
            "source_exit_finished_at_monotonic": self.source_exit_finished_at,
            "late_source_close": self.source_exit_finished_at >= self.deadline,
            "validation_accepted": False, "original_success_authenticated": False,
            "native_execution_authenticated": False, "parent_handle_issued": False,
            "storage_release_authorized": False, "retry_authorized": False,
            "further_productive_work": False}
        try:
            new_json(Path(self.context["invocation"]) / "failed-operational-source-closeout.json", value)
        except Exception as write_error:
            # Preserve the original failed operation; closed Source cannot be
            # put back into UNKNOWN merely because its metadata write failed.
            try:
                BaseException.add_note(error, "Failed operational Source closeout evidence write: "
                    + type(write_error).__name__[:128])
            except Exception:
                pass
        return True

    def finish(self, completed, witness):
        # The bounded observation was persisted before record-count/code/proof gates.
        self.clock(); self.guard()
        observation = self.checked_execution(completed, witness)
        need(observation.output_capture == "complete", "original V5 terminal EOF is incomplete or unknown")
        need(0 < len(completed.stdout) <= 1024**2 and len(completed.stderr) <= 1024**2,
             "bounded original V5 terminal required")
        raw = completed.stdout.encode("utf-8")
        need(len(raw) <= 1024**2 and len(completed.stderr.encode("utf-8")) <= 1024**2,
             "bounded original V5 terminal required")
        # Only a genuinely waited, complete terminal enters existing classifiers.
        wait = {"schema": WORK_SCHEMA, "request": self.request_pin, **witness,
            "argv": self.argv, "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "original_process_owner_returned": True,
            "execution_observation": self.execution_ref}
        self.wait_ref = new_json(Path(self.context["invocation"]) / "runner-wait.json", wait)
        self.clock()
        terminal = json_object(raw)
        if completed.returncode == 0:
            need(observation.completion_kind == "completed" and self.process.returncode == 0,
                 "original V5 success requires actual/public zero and completed capture")
            self.clock()
            if self.policy.get("schema") in ("reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3", "reviewed-definition-realization-parent/v1"):
                if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
                        or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
                    raise ValueError("reviewed parent callable bindings changed: guard")
                _reviewed_parent_guard()
                operational = _validation_released_operational_evidence_v1(
                    self.context["plan"], self.request_pin, terminal, Path(self.context["invocation"]))
                self._released_operational_raw = _validation_released_operational_projection_v1(operational, terminal)
                self._released_stdout_raw = raw
                self._released_wait_raw = checked_descriptor(self.wait_ref)
                self.clock()
            summary = verify_validation_evidence(self.policy, self.context["plan"], self.request_pin,
                terminal, Path(self.context["invocation"]))
            value = {"closure_kind": "existing_verified_success", "storage_status": "released",
                "verified_checks": summary}
        else:
            arguments = {"expected_request_sha256": self.request_pin["sha256"],
                "owner_pid": self.identity["pid"], "owner_birth": self.identity["birth"],
                "returncode": self.process.returncode, "expected_resource_ledger": str(self.ledger)}
            self.clock()
            verdict = self.no_grant(terminal, self.raw, **arguments)
            if verdict == {"verdict": "proven_original_no_child_no_native_grant",
                    "source_custody_closed": True, "storage_release_authorized": False}:
                value = {"closure_kind": "original_no_child_no_native_grant",
                    "storage_status": terminal.get("reservation", {}).get("status", "not_requested")}
            else:
                reservation = terminal.get("reservation", {}); row = reservation.get("record")
                if type(row) is dict and row.get("child") is None:
                    self.clock()
                    verdict = self.no_child(terminal, self.raw, **arguments,
                        cleanup_timeout_seconds=min(5, self.deadline - time.monotonic()))
                    need(verdict == {"verdict": "proven_original_no_child_native_closed",
                        "source_custody_closed": True, "storage_release_authorized": False},
                        "original V5 no-child native closure remains unknown")
                    value = {"closure_kind": "original_no_child_native_closed", "storage_status": reservation["status"]}
                elif (type(row) is dict and row.get("status") == "retained" and reservation.get("cleanup_error") is None
                      and "source_qualification" in self.policy["validation"]
                      and observation.completion_kind == "timeout" and completed.returncode == 124):
                    self.clock()
                    _source_qualification_plan(self.policy, self.context["plan"], Path(self.context["invocation"]))
                    verdict = self.inspector(terminal, self.raw, **arguments,
                        original_wait_deadline_monotonic=self.deadline, cleanup_profile="cancelled_retained_v5")
                    need(verdict == {"verdict": "proven_original_cancelled_retained_native_closed",
                        "source_custody_closed": True, "storage_release_authorized": False},
                        "original V5 cancelled retained native closure remains unknown")
                    value = {"closure_kind": "original_cancelled_retained_native_closed", "storage_status": "retained"}
                elif (type(row) is dict and row.get("status") == "retained" and reservation.get("cleanup_error") is None
                      and "source_qualification" in self.policy["validation"]
                      and observation.completion_kind == "completed"
                      and completed.returncode == self.process.returncode == 1
                      and type(terminal.get("failure")) is dict
                      and terminal["failure"].get("error_type") == "RuntimeError"
                      and type(terminal["failure"].get("returncode")) is int
                      and terminal["failure"]["returncode"] != 0
                      and terminal["failure"].get("error") == "validation command failed with returncode "+str(terminal["failure"]["returncode"])
                      and terminal["failure"].get("received_signals") == []):
                    self.clock()
                    _source_qualification_plan(self.policy, self.context["plan"], Path(self.context["invocation"]))
                    verdict = self.inspector(terminal, self.raw, **arguments,
                        original_wait_deadline_monotonic=self.deadline, cleanup_profile="failed_retained_v5")
                    need(verdict == {"verdict": "proven_original_failed_retained_native_closed",
                        "source_custody_closed": True, "storage_release_authorized": False},
                        "original V5 failed retained native closure remains unknown")
                    value = {"closure_kind": "original_failed_retained_native_closed", "storage_status": "retained"}
                elif type(row) is dict and row.get("status") == "retained" and reservation.get("cleanup_error") is None:
                    self.clock()
                    if "source_qualification" in self.policy["validation"]:
                        negative = _source_qualification_failed_custody(self.policy, self.context["plan"],
                            self.request_pin, Path(self.context["invocation"]), terminal, wait)
                    else:
                        negative = negative_validation_material(self.policy, self.context["plan"], self.context["frozen"],
                            self.request_pin, Path(self.context["invocation"]), terminal, wait)
                    need(all(negative[k] is True for k in (("parent_source_ready_to_close", "native_closed", "processes_closed")
                         if _reviewed_parent_policy(self.policy) else ("source_closed", "native_closed", "processes_closed"))),
                        "original V5 retained failure custody remains unknown")
                    value = {"closure_kind": "existing_verified_retained_failure", "storage_status": "retained"}
                else:
                    self.clock()
                    verdict = self.inspector(terminal, self.raw, **arguments,
                        original_wait_deadline_monotonic=self.deadline)
                    need(type(verdict) is dict and verdict == {"verdict": "proven_original_productive_native_closed_storage_separate",
                        "source_custody_closed": True, "storage_release_authorized": False},
                        "original V5 native/source closure remains unknown")
                    need(reservation["status"] == row["status"] == "active"
                         and row["storage_bytes"] == self.request["storage_bytes"],
                         "failed retention must preserve original ACTIVE storage")
                    value = {"closure_kind": "productive_native_closed_storage_separate", "storage_status": "active"}
        self.guard(); self.clock()
        if _reviewed_parent_policy(self.policy):
            _reviewed_parent_check_worker_delegation(self, require_worker_report=completed.returncode == 0)
        elif _immutable_source_parent_enabled(self.policy):
            _immutable_source_parent_check_delegation(self, require_worker_report=completed.returncode == 0)
            value.update(_immutable_source_parent_summary(self.context["plan"], self.immutable_source_observation))
        self.stack.close(); self.closed = True
        self.source_exit_finished_at = time.monotonic()
        if self.source_exit_finished_at >= self.deadline:
            # Record actual irreversible closure; do not grant acceptance after D.
            new_json(Path(self.context["invocation"]) / "source-custody-late-close.json", {
                "schema": "agent-supervisor/v5-validation-late-source-close@1", "request": self.request_pin,
                "source_closed": True, "native_closed": True, "execution_observation": self.execution_ref,
                "authority_deadline_monotonic": self.deadline,
                "source_exit_finished_at_monotonic": self.source_exit_finished_at,
                "storage_release_authorized": False, "retry_authorized": False, "validation_accepted": False})
            need(False, "original V5 Source exit completed after the original authority deadline")
        # A later metadata failure cannot relabel this closed Source stack as open.
        return {"schema": "agent-supervisor/v5-validation-parent-source-custody@2", **value,
            "source_closed": True, "native_closed": True, "source_receipts": self.receipts,
            "storage_release_authorized": False, "retry_authorized": False,
            "original_wait": self.wait_ref, "authority_deadline_monotonic": self.deadline,
            "execution_observation": self.execution_ref, "work_deadline_monotonic": self.work_deadline,
            "cleanup_deadline_monotonic": self.cleanup_deadline,
            "source_exit_finished_at_monotonic": self.source_exit_finished_at,
            "clock_scope": "Source entry and closure proof use the original timeout; unchanged engine stop/reap and nonpreemptible IO may exceed it, in which case no postdeadline proof is begun"}


def original_runner_wait(policy, argv, cwd, timeout, *, validation_context=None, capture_parent_success=False):
    need(type(capture_parent_success) is bool, "exact original parent capture option required")
    if capture_parent_success:
        if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
                or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
            raise ValueError("reviewed parent callable bindings changed: guard")
        _reviewed_parent_guard()
        need(_reviewed_parent_policy(policy) and timeout == 420, "explicit reviewed parent420 profile required")
        need(type(validation_context) is dict and type(validation_context.get("plan")) is dict,
             "predeclared reviewed parent validation context required")
        declared = validation_context["plan"]["policy"]
        checked = _reviewed_parent_checked_policy_v3(Path(declared["path"]), declared["sha256"], policy)
        need(canonical(checked) == canonical(policy), "closed original reviewed parent declaration required")
        _reviewed_parent_check_plan_v1(policy, validation_context)
    joined_opening_binding = (_reviewed_parent_joined_opening_v3(policy)
                              if capture_parent_success else None)
    custody = (_V5ValidationSourceCustody(policy, argv, cwd, timeout, validation_context)
               if _v5_ordinary_validation(policy) else None)
    completed = None; witness = None
    try:
        """Observe the actual Popen owned by the unchanged public engine helper."""
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
        if custody is not None:
            custody.clock()
        source = descriptor(Path(engine.__file__).absolute())
        need(source in policy["source_descriptors"], "original process owner source is not pinned")
        if capture_parent_success:
            _reviewed_parent_engine_guard(engine, source)
        if custody is not None:
            custody.engine_source = source; custody.engine_runner = engine.run_command
        actual = engine.subprocess.Popen
        need(actual is _ORIGINAL_POPEN, "original process primitive was replaced before owner invocation")
        records = []
        def observed(command, *positional, **kwargs):
            frame = sys._getframe(1)
            need(frame.f_code is engine.run_command.__code__ and frame.f_globals is vars(engine)
                 and frame.f_code.co_filename == source["path"] and list(command) == argv
                 and kwargs.get("cwd") == str(cwd) and kwargs.get("start_new_session") is True,
                 "only the exact original owned runner launch may be observed")
            if custody is not None:
                custody.launch_clock()
                custody.spawn_attempted = True
            process = actual(command, *positional, **kwargs)
            # Record uncertainty without raising after spawn: the original engine
            # must retain its actual process and perform its own stop/reap logic.
            if custody is None:
                records.append((process, proc_identity(process.pid)))
            else:
                custody.process = process; custody.spawn_returned_at = time.monotonic()
                try:
                    identity = proc_identity(process.pid)
                except (OSError, ValueError, IndexError) as error:
                    identity = None; custody.identity_error = type(error).__name__
                custody.identity = identity; custody.identity_observed_at = time.monotonic()
                records.append((process, identity))
            return process
        completed = None
        try:
            engine.subprocess.Popen = observed
            if custody is not None:
                remaining = custody.launch_clock()
                custody.engine_started_at = time.monotonic()
                custody.clock()
                completed = engine.run_command(argv, cwd=cwd, timeout_seconds=remaining,
                    work_deadline_monotonic=custody.work_deadline,
                    cleanup_deadline_monotonic=custody.cleanup_deadline)
                custody.engine_returned = True
            else:
                completed = engine.run_command(argv, cwd=cwd, timeout_seconds=timeout)
        finally:
            engine.subprocess.Popen = actual
            if custody is not None:
                custody.record_execution(completed, engine, records)
        need(len(records) == 1, "one genuine original runner process required")
        process, identity = records[0]
        if custody is None:
            need(identity is not None and process.returncode is not None
                 and process.poll() == process.returncode == completed.returncode, "actual runner wait is unknown")
        else:
            custody.clock()
        need(descriptor(Path(engine.__file__).absolute()) == source, "original process owner source changed")
        observation = None if custody is None else custody.execution_observation
        witness = {"runner_identity": identity, "actual_Popen_returned": True,
            "actual_process_waited": True if custody is None else (
                None if observation is None else observation.leader_waited),
            "actual_process_returncode": process.returncode, "engine_source": source}
        if custody is not None:
            witness["parent_source_custody"] = custody.finish(completed, witness)
            # Canonical parent observation, persisted after normal Source exit.
            # It grants no storage release, retry or independently trusted success.
            witness["parent_source_closeout"] = new_json(Path(validation_context["invocation"]) / "parent-source-closeout.json", {
                "schema": "agent-supervisor/v5-validation-parent-source-closeout@2",
                "request": custody.request_pin, "original_wait": custody.wait_ref,
                "producer": descriptor(Path(__file__).absolute()),
                "parent_source_custody": witness["parent_source_custody"]})
        if capture_parent_success:
            need(custody is not None and custody.closed, "original reviewed parent Source closure required")
            return completed, witness, _reviewed_parent_issue(custody, completed, witness)
        return completed, witness

    except BaseException as error:
        if custody is not None:
            # Source closure is irreversible. A later metadata failure is a
            # failed observation, not UNKNOWN custody of an open Source stack.
            if custody.closed:
                raise
            failed_close = None
            if capture_parent_success or (type(policy) is dict and policy.get("schema") in (
                    "reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3", "reviewed-definition-realization-parent/v1")):
                try:
                    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
                            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
                        raise ValueError("reviewed parent callable bindings changed: guard")
                    _reviewed_parent_guard()
                    failed_close = _reviewed_parent_failed_close_method_v1(custody)
                except BaseException:
                    failed_close = None
            if isinstance(error, Exception) and type(failed_close) is types.MethodType and failed_close.__self__ is custody:
                failed_close(completed, witness, error)
                if custody.closed:
                    raise
            if custody.process is not None or custody.spawn_attempted:
                custody.hold(error)
            try:
                custody.stack.close(); custody.closed = True
            except BaseException as close_error:
                custody.hold(close_error)
        raise



CONTROL_PUBLICATION_SCHEMA = "agent-supervisor/public-validation-request-publication@1"
CLOSED_CONTROL_SCHEMA = "agent-supervisor/public-validation-closed-owner@1"


def immutable_bytes(path, raw):
    """Byte-exact append-only metadata; existing same bytes are read-only resume."""
    need(type(raw) is bytes and 0 < len(raw) <= 8 * 1024**2, "bounded immutable original bytes required")
    if path.exists():
        need(ordinary_bytes(path) == raw and not path.lstat().st_mode & 0o222,
             "existing immutable original copy differs; no replay")
        return descriptor(path)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(path)


def validation_control_directory(policy, frozen, invocation=None):
    """Separate the fixed programme registry from the actual candidate lane."""
    binding = frozen["binding"]; v3 = "external_fixture_profile" in policy["validation"]
    expected = "ptr_lane_0" if v3 or _v5_ordinary_validation(policy) else "legal_ir_unattended_repair"
    need(binding["namespace"] == expected and binding["board_id"] == REPAIR_PROFILE
         and type(binding["attempt_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", binding["attempt_id"]),
         "fixed original candidate namespace/board/attempt required")
    registry_namespace = "legal_ir_unattended_repair"
    directory = Path(policy["validation"]["attempt_root"]) / "control-requests" / registry_namespace / binding["board_id"] / validation_request_generation(policy, frozen, invocation)
    need(not any(parent.is_symlink() for parent in (directory, *directory.parents)), "original control registry must not be aliased")
    return directory


def publish_validation_request(policy, frozen, invocation, local_request):
    """Publish the actual executed D request before invoking its original owner."""
    directory = validation_control_directory(policy, frozen, invocation)
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    count = 0
    with os.scandir(directory.parent) as entries:
        for entry in entries:
            count += 1
            need(count <= 64 and entry.is_dir(follow_symlinks=False)
                 and re.fullmatch(r"[0-9a-f]{32}", entry.name), "finite original control registry capacity/identity exceeded")
    need(count < 64 and not directory.exists(), "fresh original control generation required; do not replay")
    directory.mkdir(mode=0o700)
    request_pin = immutable_bytes(directory / "execution-request.json", checked_descriptor(local_request))
    plan_pin = descriptor(invocation / "validation-plan.json")
    plan = json_object(checked_descriptor(plan_pin))
    new_json(invocation / "execution-request-publication.json", {"schema": CONTROL_PUBLICATION_SCHEMA,
        "request": request_pin, "invocation_copy": local_request, "plan": plan_pin,
        "freeze": plan["freeze"], "owner_policy": plan["policy"], "source_commitment": frozen["source_commitment"]})
    return request_pin


def original_execution_request(invocation):
    """Resolve only the exact byte-identical original request publication."""
    local = descriptor(invocation / "execution-request.json")
    path = invocation / "execution-request-publication.json"
    if not path.exists():
        return local  # Existing closed local-only invocations remain readable.
    publication = json_object(ordinary_bytes(path))
    plan_pin = descriptor(invocation / "validation-plan.json"); plan = json_object(checked_descriptor(plan_pin))
    policy = json_object(checked_descriptor(plan["policy"])); frozen = json_object(checked_descriptor(plan["freeze"]))
    expected = validation_control_directory(policy, frozen, invocation) / "execution-request.json"
    need(set(publication) == {"schema", "request", "invocation_copy", "plan", "freeze", "owner_policy", "source_commitment"}
         and publication["schema"] == CONTROL_PUBLICATION_SCHEMA and publication["request"]["path"] == str(expected)
         and publication["invocation_copy"] == local and publication["plan"] == plan_pin
         and publication["freeze"] == plan["freeze"] and publication["owner_policy"] == plan["policy"]
         and publication["source_commitment"] == plan["source_commitment"]
         and checked_descriptor(publication["request"]) == checked_descriptor(local),
         "original executed request publication changed")
    return publication["request"]


def publish_closed_validation_control(policy, plan, frozen, request_pin, invocation, negative_pin):
    """Advertise closed failed resource custody, never semantic acceptance.

    The watchdog independently rechecks the fixed original D producer/resource
    version, exact current ledger row/native policy/PID/groups and typed failed
    inventory. The changing A publisher descriptor is historical metadata.
    """
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair publish_closed_validation_control")
    proof = json_object(checked_descriptor(negative_pin))
    if not all(proof[key] is True for key in ("source_closed", "native_closed", "processes_closed")):
        return None
    terminal = json_object(checked_descriptor(proof["original_failure_terminal"]))
    record = terminal["reservation"]["record"]
    request = json_object(checked_descriptor(request_pin))
    need(record["status"] == "retained" and record.get("retention_reason") == "context_failed"
         and proof["storage_closed"] is False and proof["original_record_sha256"]
             == hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
         "only the exact original nocharge context-failed retained claim may be advertised")
    v3_profile = None
    if request["schema"] == "owned-validation-request/v3":
        v3_profile = original_external_validation_profile(policy, plan, request_pin, record)
        need(terminal["failure"]["external_fixture_disposition"] == v3_profile["terminal_disposition"]
             and not terminal["failure"].get("external_fixture_disposition_error"),
             "original post-exit terminal does not bind the completed typed external observation")
    else:
        need(request["schema"] == "owned-validation-request/v1" and record.get("external_charges") == {}
             and request["declared_external_writable_paths"] == [], "legacy closed-owner publication requires the original no-charge profile")
    wait = json_object(checked_descriptor(proof["original_runner_wait"]))
    need(wait["actual_Popen_returned"] is True and wait["actual_process_waited"] is True
         and wait["runner_identity"] == proof["original_owner_identity"]
         and wait["request"] == request_pin and json_object(wait["stdout"].encode()) == terminal,
         "genuine original wait/terminal bytes do not bind this failed claim")
    directory = validation_control_directory(policy, frozen)
    need(request_pin == descriptor(directory / "execution-request.json"), "original detached D request differs")
    wait_copy = immutable_bytes(directory / "runner-wait.json", checked_descriptor(proof["original_runner_wait"]))
    terminal_copy = immutable_bytes(directory / "original-failure-terminal.json", checked_descriptor(proof["original_failure_terminal"]))
    negative_copy = immutable_bytes(directory / "negative-validation.json", checked_descriptor(negative_pin))
    d = Path(policy["datasets_root"])
    producer = next(pin for pin in policy["source_descriptors"] if pin["path"] == str(d / "scripts/ops/legal_ir/run_owned_validation.py"))
    resources = next(pin for pin in policy["source_descriptors"] if pin["path"] == str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_run_resources.py"))
    checked_descriptor(producer); checked_descriptor(resources)
    value = {"schema": "agent-supervisor/public-validation-closed-owner@2" if v3_profile is not None else CLOSED_CONTROL_SCHEMA,
        "binding": {key: frozen["binding"][key] for key in ("namespace", "board_id", "attempt_id")},
        "producer": producer, "publisher": descriptor(Path(__file__).absolute()), "validation_resources": resources,
        "owner_policy": plan["policy"], "request": request_pin, "plan": descriptor(invocation / "validation-plan.json"),
        "freeze": plan["freeze"], "original_wait": wait_copy, "failure_terminal": terminal_copy,
        "negative_disposition": negative_copy, "invocation_copy": descriptor(invocation / "execution-request.json"),
        "owner": proof["original_owner_identity"], "reservation_id": proof["reservation_id"],
        "retained_record_sha256": proof["original_record_sha256"],
        "native_inspection": {"state_path": policy["validation"]["scheduler_state_path"],
            "policy_sha256": policy["validation"]["expected_scheduler_policy_sha256"],
            "request_id": "validation:" + proof["reservation_id"]},
        "genuine_wait_completed": True, "source_closed": True, "native_closed": True, "processes_closed": True,
        "storage_claim_released": False, "admitted": False, "Lean_admitted": False, "retry_authorized": False}
    if v3_profile is not None:
        value["original_validation_profile"] = v3_profile
    return immutable_bytes(directory / "closed-owner.json", canonical(value) + b"\n")

def validate(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    qualification = checked_source_qualification(policy)
    frozen = (_source_qualification_arguments(args, qualification) if qualification is not None else
              frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root))
    invocation = args.invocation_root
    need(args.receipt_path.parent == invocation and args.freeze_path.parent == invocation.parent,
         "original candidate owner invocation required")
    if "validation_admission_retry" in policy["validation"]:
        deferred = probe_validation_source_admission(policy, args, frozen)
        if deferred is not None:
            print(canonical({"schema": "agent-supervisor/validation-source-admission-return@1", "defer": deferred}).decode(), flush=True)
            return 75
    source_parent = (_immutable_source_parent_runtime(policy, invocation)
                     if _immutable_source_parent_enabled(policy) else None)
    plan = validation_plan(policy, descriptor(args.policy), frozen, descriptor(args.freeze_path), args.candidate_root, invocation,
        **({"source_parent_context": source_parent} if source_parent is not None else {}))
    need(len(canonical(plan)) + 1 <= 4 * 1024**2 and descriptor(args.freeze_path)["bytes"] <= 4 * 1024**2,
         "trusted immutable manifest exceeds the unchanged public source-file bound")
    plan_pin = new_json(invocation / "validation-plan.json", plan)
    attempt = Path(plan["attempt_directory"]); attempt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (attempt, *attempt.parents)), "attempt parent changed")
    attempt.mkdir(mode=0o700)
    v = policy["validation"]
    request = validation_execution_request(policy, plan, plan_pin)
    local_request = new_json(invocation / "execution-request.json", request)
    request_pin = publish_validation_request(policy, frozen, invocation, local_request)
    if request["schema"] == "owned-validation-request/v3":
        need(request_pin["path"] == plan["public_request_path"], "executed original v3 request path differs from its prechild plan")
    # The original engine owns stop/reap, and the public D CLI owns all native
    # and storage allocation/finalization. No historical operator code runs.
    argv = [policy["python"], "-I", "-B", str(Path(policy["datasets_root"]) / "scripts/ops/legal_ir/run_owned_validation.py"),
        "--request", request_pin["path"], "--expected-request-sha256", request_pin["sha256"],
        "--resource-ledger", v["resource_ledger"]]
    validation_request_generation(policy,frozen,invocation,live=True) # Fixed expiry rechecked after publication I/O.
    completed, witness = original_runner_wait(policy, argv, args.candidate_root, v["timeout_seconds"] + 60,
        **({"validation_context": {"plan": plan, "request_pin": request_pin, "invocation": invocation, "frozen": frozen,
                **({"source_parent": source_parent} if source_parent is not None else {})}}
           if _v5_ordinary_validation(policy) else {}))
    if not _v5_ordinary_validation(policy):
        new_json(invocation / "runner-wait.json", {"schema": WORK_SCHEMA, "request": request_pin, **witness,
            "argv": argv, "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "original_process_owner_returned": True})
    if type(completed.returncode) is int and completed.returncode != 0:
        if _v5_ordinary_validation(policy):
            need(witness["parent_source_custody"]["source_closed"] is True
                 and witness["parent_source_custody"]["native_closed"] is True, "V5 failed source custody is unknown")
            if (qualification is None and witness["parent_source_custody"]["closure_kind"] == "original_no_child_no_native_grant"
                    and json_object(completed.stdout.encode()).get("source_custody_lifecycle", {}).get("schema")
                        == "owned-validation-pre-grant-ledger-lock-refusal/v1"):
                material = operational_ledger_busy_material(policy, frozen, request_pin, invocation,
                    witness["parent_source_closeout"])
                new_json(invocation/"operational-ledger-busy.json", material)
                return completed.returncode
            # Source/native closure is operational evidence only; ACTIVE storage
            # must never enter the legacy retained-negative publisher or retry path.
            new_json(invocation / ("failed-source-qualification.json" if qualification is not None else
                "failed-validation-source-closeout.json"), {
                "schema": ("agent-supervisor/v5-source-qualification-failed-source-closeout@1" if qualification is not None else
                    "agent-supervisor/v5-validation-failed-source-closeout@1"),
                "request": request_pin, "parent_source_custody": witness["parent_source_custody"],
                "storage_closed": False, "semantic_negative": False, "passed": False, "retry_authorized": False})
        else:
            capture_negative_validation(policy, plan, frozen, request_pin, invocation, completed)
        return completed.returncode if 0 < completed.returncode < 126 else 1
    need(type(completed.returncode) is int and completed.returncode == 0,
         "original validation owner failed or is unknown; retain without retry")
    terminal = json_object(completed.stdout.encode())
    original_pin = new_json(invocation / "original-terminal.json", terminal)
    summary = (witness["parent_source_custody"]["verified_checks"] if _v5_ordinary_validation(policy)
               else verify_validation_evidence(policy, plan, request_pin, terminal, invocation))
    if qualification is not None:
        new_json(args.receipt_path, {"schema": _source_qualification_receipt_schema(qualification),
            "snapshot": qualification["snapshot_ref"], "source_qualification": qualification["profile"],
            "source_commitment": args.source_commitment, "passed": True,
            "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True,
            "cleanup_errors": [], "original_receipt_reference": original_pin["path"],
            "original_receipt_descriptor": original_pin, "verified_checks": summary,
            "parent_source_closeout": witness["parent_source_closeout"],
            "repair_authority": False, "adoption_authority": False,
            "activation_authority": False, "completion_authority": False})
        return
    new_json(args.receipt_path, {"schema": VALIDATION_SCHEMA, "freeze_sha256": frozen["freeze_sha256"],
        "source_commitment": args.source_commitment, "passed": True,
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True,
        "cleanup_errors": [], "original_receipt_reference": original_pin["path"],
        "original_receipt_descriptor": original_pin, "verified_checks": summary,
        **({"parent_source_closeout": witness["parent_source_closeout"]} if _v5_ordinary_validation(policy) else {})})


def _reviewed_parent_failed_close_method_v1(custody):
    """Authenticate the fixed owner method; this lookup closes no resource."""
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    _reviewed_parent_guard()
    need(type(custody) is _V5ValidationSourceCustody, "same authenticated original custody class required")
    method = getattr(custody, "close_failed_operational_verification", None)
    need(type(method) is types.MethodType and method.__self__ is custody
         and method.__func__ is vars(_V5ValidationSourceCustody)["close_failed_operational_verification"],
         "reviewed parent callable bindings changed: failed operational close method")
    return method


def _validation_released_operational_evidence_v1(plan, request_pin, terminal, invocation):
    """Existing released-success checks only; no semantic or authority grant."""
    request = json_object(checked_descriptor(request_pin))
    wait = json_object(ordinary_bytes(invocation / "runner-wait.json"))
    need(wait["request"] == request_pin and wait["original_process_owner_returned"] is True
         and type(wait["returncode"]) is int and wait["returncode"] == 0
         and json_object(wait["stdout"].encode()) == terminal, "genuine original runner wait required")
    reservation = terminal["reservation"]; record = reservation["record"]; native = reservation["resource_lease"]
    need(reservation["status"] == record["status"] == "released" and record["artifacts_durable_asserted"] is True
         and native["released"] is True and native["lease_id"] == record["native_lease_id"]
         and record["validation_request_sha256"] == request_pin["sha256"] and reservation["cleanup_error"] is None,
         "original native/storage closure is not proven")
    need(wait["runner_identity"] == {"pid": record["owner_pid"], "birth": record["owner_birth"]}
         and native["owner_pid"] == record["owner_pid"]
         and wait["actual_Popen_returned"] is True and wait["actual_process_waited"] is True
         and wait["actual_process_returncode"] == 0, "original runner PID/birth/wait binding differs")
    checked_descriptor(wait["engine_source"])
    process = reservation["process_observation"]
    need(process["live_processes"] == 0 and process["out_of_group_live_identities"] == []
         and not process["tracking_error"], "original descendant custody remains unknown")
    attempt = Path(plan["attempt_directory"])
    owner = json_object(ordinary_bytes(Path(terminal["result"])))
    need(Path(terminal["result"]) == attempt / "logs/result.json" and owner["schema"] == "owned-validation-result/v1"
         and owner["completed"] is True and type(owner["returncode"]) is int and owner["returncode"] == 0
         and owner["request_sha256"] == request_pin["sha256"], "original result mismatch")
    return request, wait, reservation, record, native, attempt


def _validation_released_operational_projection_v1(operational, terminal):
    """Detached bounded original DATA; it is not a capability or close action."""
    return canonical({"request": operational[0], "wait": operational[1],
        "terminal": terminal, "attempt_directory": str(operational[5])})


def verify_validation_evidence(policy, plan, request_pin, terminal, invocation):
    qualification = _source_qualification_plan(policy, plan, invocation)
    request, wait, reservation, record, native, attempt = _validation_released_operational_evidence_v1(
        plan, request_pin, terminal, invocation)
    report = json_object(ordinary_bytes(attempt / "exports/trusted-validation.json"))
    native_profile = _reviewed_parent_policy(policy)
    delegated_source = _immutable_source_parent_enabled(policy)
    need(report["schema"] == (_reviewed_parent_domain_v3(_reviewed_parent_version_v3(policy))["worker_schema"] if native_profile else
         IMMUTABLE_PARENT_WORKER_SCHEMA if delegated_source else WORK_SCHEMA) and report["passed"] is True
         and (report["source_READ_closed"] is False if native_profile or delegated_source else report["source_READ_closed"] is True)
         and report["source_snapshot_stable"] is True
         and report["cleanup_errors"] == [] and report["error"] is None and report["junit_error"] is None and report["guard_denials"] == []
         and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"],
         "trusted source/semantic result missing")
    need(report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"]), "complete trusted source guard was not applied")
    need(report["pytest_module_origin"] == policy["validation"]["pytest_module"]
         and report["startup"]["python"] == policy["python"]
         and report["startup"]["isolated"] == 1 and report["startup"]["no_bytecode"] is True,
         "trusted interpreter/Pytest origin differs")
    _source_qualification_report(qualification, plan, report)
    need(all(report["executed_source_buffers"].get(path) == plan["source_pins"][path]
             for path in plan["required_candidate_bodies"]), "edited candidate body was not actually tested")
    _v5_validation_namespace_evidence(policy, plan, report)
    if request["schema"] == "owned-validation-request/v3":
        original_external_validation_profile(policy, plan, request_pin, record)
        need(report["collected_cases"] == expected_natural_nodes(plan)
             and report["expected_natural_nodes"] == expected_natural_nodes(plan)
             and report["selector_case_counts"] == plan["selector_case_counts"]
             and report["declared_normal_namespace_profiles"] == plan["normal_namespace_profiles"],
             "actual v3 natural collection/namespace declaration differs")
    for path, sha in plan["source_pins"].items():
        need(descriptor(Path(path))["sha256"] == sha, "full original validation source population changed")
    expected = Counter((node, phase) for node in expected_natural_nodes(plan) for phase in ("setup", "call", "teardown"))
    need(all(row["outcome"] == "passed" for row in report["phases"])
         and Counter((r["nodeid"], r["when"]) for r in report["phases"]) == expected,
         "every natural phase must actually pass")
    suites = ET.fromstring(ordinary_bytes(attempt / "exports/test-results.xml")).findall(".//testsuite")
    need({k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
         == {"tests": plan["expected_cases"], "failures": 0, "errors": 0, "skipped": 0}, "JUnit result differs")
    durability = json_object(ordinary_bytes(attempt / "logs/validation-durability.json"))
    need(durability["schema"] == "owned-validation-durability/v1"
         and durability["request_sha256"] == request_pin["sha256"]
         and durability["reservation_id"] == record["reservation_id"]
         and durability["result_sha256"] == descriptor(Path(terminal["result"]))["sha256"], "original durability mismatch")
    if request["schema"] == "owned-validation-request/v3":
        need(durability["external_fixture_disposition"] == record["external_fixture_disposition"],
             "original durability receipt external disposition differs")
    seen = set()
    for row in durability["files"]:
        rel = Path(row["path"])
        need(not rel.is_absolute() and ".." not in rel.parts and row["path"] not in seen, "durability path mismatch")
        seen.add(row["path"]); path = attempt / rel; pin = descriptor(path); info = path.lstat()
        need((pin["bytes"], pin["sha256"], info.st_dev, info.st_ino) == (row["bytes"], row["sha256"], row["device"], row["inode"]), "durable artifact changed")
    for path, sha in request["source_sha256"].items():
        need(descriptor(Path(path))["sha256"] == sha, "validation source changed")
    summary = {"cases": plan["expected_cases"], "phases": sum(expected.values()), "reservation_id": record["reservation_id"],
            "native_lease_id": native["lease_id"], "natural_phase_counter_verified": True}
    if native_profile:
        need(report.get("native_return_refusal") is None, "reviewed worker recorded a failed native build return")
        need(terminal.get("admitted") is False
             and plan["schema"] == _reviewed_parent_domain_v3(_reviewed_parent_version_v3(policy))["worker_schema"] and plan["operation"] == policy["operation"]
             and report["source_custody_mode"] == "delegated-original-parent-read/v1"
             and report["source_READ_admission_complete"] is False
             and report["parent_nonce_sha256"] == plan["parent_nonce_sha256"],
             "reviewed worker delegated source scope differs")
        binding_pin = descriptor(attempt / "exports/reviewed-preservation-native-binding.json")
        need(report["native_binding"] == binding_pin, "worker original native binding descriptor differs")
        binding = json_object(checked_descriptor(binding_pin))
        inventory = _reviewed_parent_binding_v3(policy, policy["operation"], binding, durability["files"],
            attempt_directory=attempt)
        summary.update(native_binding=binding_pin, packet_inventory=[list(row) for row in inventory],
            durability=descriptor(attempt / "logs/validation-durability.json"),
            parent_source_observation=report["parent_source_observation"])
    if delegated_source:
        summary.update(_immutable_source_parent_report_binding(policy, plan, request_pin, report, invocation))
    return summary


def verify_validation(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair verify_validation")
    receipt = json_object(ordinary_bytes(args.receipt_path))
    plan = json_object(ordinary_bytes(args.invocation_root / "validation-plan.json"))
    frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    need(receipt["schema"] == VALIDATION_SCHEMA and receipt["passed"] is True
         and receipt["source_commitment"] == args.source_commitment, "positive bound validation required")
    terminal = json_object(checked_descriptor(receipt["original_receipt_descriptor"]))
    request_pin = original_execution_request(args.invocation_root)
    actual_summary = verify_validation_evidence(policy, plan, request_pin, terminal, args.invocation_root)
    if _v5_ordinary_validation(policy):
        need(receipt["verified_checks"] == actual_summary, "V5 verified summary differs from actual evidence")
        verify_v5_parent_source_closeout(policy, request_pin, args.invocation_root,
            receipt["parent_source_closeout"], actual_summary)



# Immutable incoming SDK evidence is DATA only. It is not the currently running
# producer or engine, and never supplies a lease, retry, Source or native grant.
_ARCHIVED_V5_SDK_SOURCES = {}
_ARCHIVED_V5_SOURCE_STAMP_FIELDS = ('st_dev','st_ino','st_mode','st_uid','st_gid',
    'st_nlink','st_size','st_mtime_ns','st_ctime_ns')


class _ArchivedV5SDKSource:
    __slots__ = ()


def _archived_v5_frozen_input(reference):
    # Use the existing exact typed descriptor before any path/hash association.
    raw = checked_descriptor(reference); path = Path(reference['path']); info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
         and not info.st_mode & 0o222, 'archived V5 SDK input must be sealed and owned')
    return raw, tuple(getattr(info,key) for key in _ARCHIVED_V5_SOURCE_STAMP_FIELDS)


def _archived_v5_current_reader_sources():
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    producer = descriptor(Path(__file__).absolute()); engine_ref = descriptor(Path(engine.__file__).absolute())
    need(producer['sha256'] == _REVIEWED_PARENT_SELF_PIN,
         'actual current V5 SDK reader source changed after import')
    need(all(function.__globals__ is globals() and function.__code__.co_filename == producer['path']
             for function in (checked_policy, checked_control_release, verify_validation_evidence,
                              verify_v5_parent_source_closeout, _verify_v5_parent_source_closeout)),
         'actual V5 SDK reader callable origin changed')
    return producer, engine_ref


def _archived_v5_sdk_source_record(witness, policy=None):
    need(type(witness) is _ArchivedV5SDKSource and witness in _ARCHIVED_V5_SDK_SOURCES,
         'archived V5 SDK capability is not authenticated')
    record = _ARCHIVED_V5_SDK_SOURCES[witness]
    need(policy is None or canonical(policy) == record['policy_encoded'],
         'archived V5 SDK capability belongs to another policy')
    current_owner, current_engine = _archived_v5_current_reader_sources()
    current_stamps = [tuple(getattr(Path(pin['path']).lstat(), key) for key in _ARCHIVED_V5_SOURCE_STAMP_FIELDS)
                      for pin in (current_owner,current_engine)]
    need(current_stamps == record['current_stamps']
         and current_owner == record['current_producer'] and current_engine == record['current_engine'],
         'archived V5 SDK current reader or engine changed')
    for reference, stamp in record['inputs']:
        _, observed = _archived_v5_frozen_input(reference)
        need(observed == stamp, 'archived V5 SDK frozen input identity changed')
    # Every declared control source is checked by the existing whole manifest
    # reader. Every other declared policy/import source retains ordinary current
    # exact reads in checked_policy; there is no general source substitution.
    selected = checked_policy(Path(record['policy_ref']['path']), record['policy_ref']['sha256'], current_owner=False)
    need(canonical(selected) == record['policy_encoded'], 'archived V5 SDK original policy changed')
    control = checked_control_release(selected, current_owner=False)
    need(descriptor(control['owner']) == record['producer']
         and record['engine'] in selected['source_descriptors']
         and control['pins'].get(record['engine']['path']) == record['engine']['sha256']
         and checked_descriptor(record['producer']) == checked_descriptor(current_owner)
         and checked_descriptor(record['engine']) == checked_descriptor(current_engine),
         'archived V5 SDK incoming/current bodies are not the same source generation')
    # NEW body compatibility is mandatory. Old7ad receipts do not become
    # qualified by a later reader that merely adds this API.
    return record


def authenticate_archived_v5_sdk_source(policy_ref):
    """Bind the exact incoming immutable SDK/engine without loading old code.

    The current SDK body must already be the SAME bytes as both incoming SDK
    files. Freeze NEW code first, then policy/plan/receipts outside that body.
    This cannot upgrade an old SDK lineage or assign current producer identity.
    """
    policy_raw, _ = _archived_v5_frozen_input(policy_ref)
    policy = checked_policy(Path(policy_ref['path']), policy_ref['sha256'], current_owner=False)
    need(_v5_ordinary_validation(policy), 'archived SDK requires explicit existing V5 ordinary profile')
    control = checked_control_release(policy, current_owner=False)
    producer = descriptor(control['owner'])
    engine_path = control['root']/'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py'
    engine_ref = descriptor(engine_path)
    need(engine_ref in policy['source_descriptors']
         and control['pins'].get(engine_ref['path']) == engine_ref['sha256'],
         'archived V5 SDK exact incoming engine must be in the sealed control manifest and policy')
    current_owner, current_engine = _archived_v5_current_reader_sources()
    need(checked_descriptor(producer) == checked_descriptor(current_owner)
         and checked_descriptor(engine_ref) == checked_descriptor(current_engine),
         'archived V5 SDK incoming/current bodies are not the same source generation')
    manifest = json_object(checked_descriptor(control['manifest']))
    references = [policy_ref, control['manifest'], *manifest['files']]
    unique = {}; inputs = []
    for reference in references:
        if reference['path'] in unique:
            need(canonical(unique[reference['path']]) == canonical(reference), 'archived V5 SDK reference alias')
            continue
        unique[reference['path']] = reference
        _, stamp = _archived_v5_frozen_input(reference)
        inputs.append((dict(reference),stamp))
    need(len(_ARCHIVED_V5_SDK_SOURCES) < 64, 'archived V5 SDK capability population exceeded')
    witness = _ArchivedV5SDKSource()
    _ARCHIVED_V5_SDK_SOURCES[witness] = dict(policy_ref=dict(policy_ref),policy_encoded=canonical(policy),
        producer=producer,engine=engine_ref,current_producer=current_owner,current_engine=current_engine,
        inputs=tuple(inputs),control_root=str(control['root']),
        current_stamps=[tuple(getattr(Path(pin['path']).lstat(), key) for key in _ARCHIVED_V5_SOURCE_STAMP_FIELDS)
                        for pin in (current_owner,current_engine)])
    try:
        _archived_v5_sdk_source_record(witness)
        return witness
    except BaseException:
        del _ARCHIVED_V5_SDK_SOURCES[witness]; raise


def close_archived_v5_sdk_source(witness):
    need(type(witness) is _ArchivedV5SDKSource and witness in _ARCHIVED_V5_SDK_SOURCES,
         'archived V5 SDK capability is not authenticated')
    del _ARCHIVED_V5_SDK_SOURCES[witness]


def archived_v5_sdk_source_data(witness):
    """Return a fresh DATA copy; it is not a source/producer capability."""
    record = _archived_v5_sdk_source_record(witness)
    return {'schema':'agent-supervisor/archived-v5-SDK-source-DATA@1',
        'policy':json_object(record['policy_encoded']), 'policy_ref':dict(record['policy_ref']),
        'control_root':record['control_root'], 'producer_ref':dict(record['producer']),
        'engine_ref':dict(record['engine']), 'same_source_generation_bytes_authenticated':True,
        'current_producer_verified':False, 'current_engine_verified':False,
        'Source_authority':False, 'native_authority':False, 'dispatch_authority':False,
        'retry_authority':False, 'storage_release_authority':False,
        'runtime_activation_authority':False,'completion_authority':False,'current_health_authority':False}


def _archived_v5_sdk_origin(witness, policy, role):
    need(role in ('producer','engine'), 'archived V5 SDK unknown source role')
    return dict(_archived_v5_sdk_source_record(witness, policy)[role])


def verify_archived_v5_parent_source_closeout(witness, request_pin, invocation, closeout_pin, summary):
    """Recheck original typed wait/Source@2 clocks at their unchanged origins."""
    record = _archived_v5_sdk_source_record(witness)
    policy = json_object(record['policy_encoded'])
    value = _verify_v5_parent_source_closeout(policy, request_pin, invocation, closeout_pin, summary,
                                             archived_sdk_source=witness)
    _archived_v5_sdk_source_record(witness, policy)
    return {'schema':'agent-supervisor/archived-v5-parent-source-closeout-DATA@1',
        'original_parent_source_closeout':value,'original_policy_ref':dict(record['policy_ref']),
        'original_producer_ref':dict(record['producer']),'original_engine_ref':dict(record['engine']),
        'historical_evidence_verified':True,'current_producer_verified':False,
        'current_engine_verified':False,'Source_authority':False,'native_authority':False,
        'dispatch_authority':False,'retry_authority':False,'storage_release_authority':False,
        'runtime_activation_authority':False,'completion_authority':False,'current_health_authority':False,
        'new_original_process_or_native_execution_performed':False}


def verify_v5_parent_source_closeout(policy, request_pin, invocation, closeout_pin, summary):
    """Recheck CURRENT canonical parent origins; archived capabilities are not accepted."""
    return _verify_v5_parent_source_closeout(policy, request_pin, invocation, closeout_pin, summary)


def _verify_v5_parent_source_closeout(policy, request_pin, invocation, closeout_pin, summary, *, archived_sdk_source=None, _ledger_busy=False):
    """Recheck the canonical parent's linked observation; no new authority."""
    need(type(_ledger_busy) is bool and not (_ledger_busy and archived_sdk_source is not None), "operational DATA cannot borrow archived authority")
    expected_code = 1 if _ledger_busy else 0
    need(_v5_ordinary_validation(policy) and closeout_pin == descriptor(invocation / "parent-source-closeout.json"),
         "original V5 parent Source closeout descriptor required")
    value = json_object(checked_descriptor(closeout_pin))
    need(type(value) is dict and set(value) == {"schema", "request", "original_wait", "producer", "parent_source_custody"}
         and value["schema"] == "agent-supervisor/v5-validation-parent-source-closeout@2"
         and value["request"] == request_pin and value["producer"] == (descriptor(Path(__file__).absolute()) if archived_sdk_source is None else
             _archived_v5_sdk_origin(archived_sdk_source, policy, 'producer'))
         and value["producer"] in policy["source_descriptors"], "V5 parent/request/current producer differs")
    wait_pin = descriptor(invocation / "runner-wait.json")
    need(value["original_wait"] == wait_pin, "V5 parent genuine wait descriptor differs")
    wait = json_object(checked_descriptor(wait_pin)); custody = value["parent_source_custody"]
    need(wait["request"] == request_pin and wait["actual_Popen_returned"] is True
         and wait["actual_process_waited"] is True and wait["original_process_owner_returned"] is True
         and type(wait["returncode"]) is int and wait["returncode"] == wait["actual_process_returncode"] == expected_code,
         "V5 parent completed genuine wait differs")
    need(type(custody) is dict and custody["schema"] == "agent-supervisor/v5-validation-parent-source-custody@2"
         and custody["closure_kind"] == ("original_no_child_no_native_grant" if _ledger_busy else "existing_verified_success")
         and custody["storage_status"] == ("not_entered" if _ledger_busy else "released")
         and custody["source_closed"] is True and custody["native_closed"] is True
         and custody["storage_release_authorized"] is False and custody["retry_authorized"] is False
         and custody["original_wait"] == wait_pin
         and (("verified_checks" not in custody) if _ledger_busy else custody["verified_checks"] == summary)
         and type(custody["source_receipts"]) is list and len(custody["source_receipts"]) == 2
         and type(custody["authority_deadline_monotonic"]) in (int, float)
         and math.isfinite(custody["authority_deadline_monotonic"]), "V5 positive parent Source observation differs")
    execution_pin = descriptor(invocation / "runner-execution-observation.json")
    need(wait["execution_observation"] == custody["execution_observation"] == execution_pin,
         "V5 parent typed execution descriptor differs")
    execution = json_object(checked_descriptor(execution_pin)); observed = execution.get("execution_observation")
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    engine_source = (descriptor(Path(engine.__file__).absolute()) if archived_sdk_source is None else
                     _archived_v5_sdk_origin(archived_sdk_source, policy, 'engine'))
    need(type(execution) is dict and set(execution) == {"schema", "request", "engine_source", "exact_command_result",
        "exact_execution_observation", "execution_observation", "public_returncode", "actual_process_pid",
        "actual_process_returncode", "actual_Popen_returned", "spawn_attempted", "process_count", "runner_identity",
        "identity_observation_error", "engine_returned", "engine_started_at_monotonic", "engine_finished_at_monotonic",
        "spawn_returned_at_monotonic", "identity_observed_at_monotonic", "authority_deadline_monotonic",
        "work_deadline_monotonic", "cleanup_deadline_monotonic", "requested_nominal_window_seconds", "window_scope",
        "source_closed", "native_closed", "storage_release_authorized", "retry_authorized"}
         and execution["schema"] == "agent-supervisor/v5-runner-execution-observation@1"
         and execution["request"] == request_pin and execution["engine_source"] == wait["engine_source"] == engine_source
         and engine_source in policy["source_descriptors"]
         and execution["exact_command_result"] is True and execution["exact_execution_observation"] is True
         and execution["actual_Popen_returned"] is True and execution["spawn_attempted"] is True
         and execution["engine_returned"] is True and type(execution["process_count"]) is int and execution["process_count"] == 1
         and execution["runner_identity"] == wait["runner_identity"]
         and type(execution["actual_process_pid"]) is int and execution["actual_process_pid"] > 0
         and type(execution["runner_identity"]) is dict and set(execution["runner_identity"]) == {"pid", "birth"}
         and type(execution["runner_identity"]["pid"]) is int
         and execution["runner_identity"]["pid"] == execution["actual_process_pid"]
         and type(execution["runner_identity"]["birth"]) is str and 0 < len(execution["runner_identity"]["birth"]) <= 32
         and execution["runner_identity"]["birth"].isascii() and execution["runner_identity"]["birth"].isdigit()
         and int(execution["runner_identity"]["birth"]) > 0 and execution["identity_observation_error"] is None
         and type(execution["public_returncode"]) is int and type(execution["actual_process_returncode"]) is int
         and execution["public_returncode"] == execution["actual_process_returncode"] == expected_code,
         "V5 parent original engine observation differs")
    need(type(wait["actual_process_returncode"]) is int
         and type(execution["requested_nominal_window_seconds"]) is int and execution["requested_nominal_window_seconds"] == 300
         and execution["source_closed"] is False and execution["native_closed"] is False
         and execution["storage_release_authorized"] is False and execution["retry_authorized"] is False,
         "V5 parent observation cannot grant authority")
    need(type(observed) is dict and tuple(sorted(observed)) == tuple(sorted(_V5_EXECUTION_FIELDS))
         and observed["schema"] == "command-execution-observation/v1"
         and observed["completion_kind"] == "completed" and observed["leader_waited"] is True
         and observed["output_capture"] == "complete"
         and type(observed["process_pid"]) is int and observed["process_pid"] == execution["actual_process_pid"] == wait["runner_identity"]["pid"]
         and type(observed["process_returncode"]) is int and observed["process_returncode"] == expected_code,
         "V5 parent complete typed terminal observation differs")
    deadline = custody["authority_deadline_monotonic"]
    times = [execution[key] for key in ("engine_started_at_monotonic", "spawn_returned_at_monotonic",
        "identity_observed_at_monotonic", "engine_finished_at_monotonic")]
    times += [observed["observed_at_monotonic"], custody["source_exit_finished_at_monotonic"],
        execution["authority_deadline_monotonic"], execution["work_deadline_monotonic"],
        execution["cleanup_deadline_monotonic"], custody["work_deadline_monotonic"], custody["cleanup_deadline_monotonic"],
        observed["work_deadline_monotonic"], observed["cleanup_deadline_monotonic"]]
    need(all(type(value) in (int, float) and math.isfinite(value) for value in times)
         and times[0] <= times[1] <= times[2] <= times[4] <= times[3] <= times[5] < deadline
         and execution["authority_deadline_monotonic"] == deadline
         and execution["work_deadline_monotonic"] == custody["work_deadline_monotonic"]
             == observed["work_deadline_monotonic"] == deadline - _V5_PROOF_RESERVE_SECONDS - _V5_ENGINE_CLEANUP_SECONDS
         and execution["cleanup_deadline_monotonic"] == custody["cleanup_deadline_monotonic"]
             == observed["cleanup_deadline_monotonic"] == deadline - _V5_PROOF_RESERVE_SECONDS,
         "V5 parent original deadline observation differs")
    if _immutable_source_parent_enabled(policy):
        plan = json_object(ordinary_bytes(invocation / "validation-plan.json"))
        need(not _ledger_busy and summary is not None, "immutable delegated closure cannot borrow repair operational proof")
        saved = _immutable_source_parent_saved_observation(policy, plan, request_pin, invocation)
        need(canonical(custody['verified_checks']) == canonical(summary)
             and all(canonical(custody.get(key)) == canonical(summary.get(key)) == canonical(item)
                 for key, item in saved.items()),
             "immutable parent archived mapping/observation summary differs")
        _immutable_source_parent_receipts(custody["source_receipts"], plan["source_root_bindings"],
            json_object(checked_descriptor(saved["parent_source_observation"])))
    checked_descriptor(request_pin)
    return value


NEGATIVE_SCHEMA = "agent-supervisor/isolated-repair-candidate-negative-validation@1"
NEGATIVE_CLOSURE_SCHEMA = "agent-supervisor/negative-validation-closure-observation@1"
NEGATIVE_VERIFIED_SCHEMA = "agent-supervisor/verified-negative-validation@1"


def verify_negative_semantic_report(policy, plan, report, invocation):
    """Recheck original failed-report attribution and raw compilation facts."""
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair verify_negative_semantic_report")
    need("source_qualification" not in plan and "required_source_bodies" not in plan
         and "source_qualification" not in report and "required_source_bodies" not in report,
         "repair negative report cannot carry immutable qualification fields")
    need(report["schema"] == WORK_SCHEMA and report["passed"] is False
         and report["source_READ_closed"] is True and report["source_snapshot_stable"] is True
         and report["cleanup_errors"] == [] and report["guard_denials"] == [] and report["error"] is None
         and report["junit_error"] is None
         and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"]
         and report["freeze"] == plan["freeze"] and report["source_commitment"] == plan["source_commitment"],
         "original negative source/report closure is unproven")
    need(report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"])
         and report["pytest_module_origin"] == policy["validation"]["pytest_module"]
         and report["startup"] == {"python": policy["python"], "isolated": 1, "no_bytecode": True}
         and report["original_compile_primitive"] == {"module": "builtins", "name": "compile", "builtin": True},
         "trusted negative interpreter/manifest/compiler identity differs")
    for path, sha in plan["source_pins"].items():
        need(descriptor(Path(path))["sha256"] == sha, "negative source identity changed")
    if "external_fixture_profile" in plan:
        need(report["expected_natural_nodes"] == expected_natural_nodes(plan)
             and report["selector_case_counts"] == plan["selector_case_counts"]
             and report["declared_normal_namespace_profiles"] == plan["normal_namespace_profiles"],
             "original negative v3 selector expansion/namespace declaration differs")
    code = report["pytest_returncode"]
    need(type(code) is int and code in {1, 2, 4}, "nonsemantic worker failure cannot authorize a coding attempt")
    suites = ET.fromstring(ordinary_bytes(Path(plan["attempt_directory"]) / "exports/test-results.xml")).findall(".//testsuite")
    junit = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    need(report["JUnit"] == junit, "original worker/JUnit counts differ")
    expected = Counter((node, phase) for node in expected_natural_nodes(plan) for phase in ("setup", "call", "teardown"))
    phases = report["phases"]
    complete = (report["collected_cases"] == expected_natural_nodes(plan)
        and Counter((row["nodeid"], row["when"]) for row in phases) == expected)
    bodies = all(report["executed_source_buffers"].get(path) == plan["source_pins"][path] for path in plan["required_candidate_bodies"])
    selected_files = {node.partition("::")[0] for node in plan["selected_checks"]}
    failures = report["failed_report_attributions"]
    need(type(failures) is list and bool(failures)
         and all(type(row) is dict and row["report_path"] in selected_files
             and row["when"] in {"call", "collect"} for row in failures),
         "original selected failed-report attribution is unavailable")
    failed_calls = [row["nodeid"] for row in phases if row["when"] == "call" and row["outcome"] == "failed"]
    call_shape = (complete and report["natural_phase_counter_complete"] is True and bool(failed_calls)
        and all(row["outcome"] == "passed" if row["when"] != "call" else row["outcome"] in {"passed", "failed"} for row in phases)
        and junit == {"tests": plan["expected_cases"], "failures": len(failed_calls), "errors": 0, "skipped": 0}
        and Counter(row["nodeid"] for row in failures if row["when"] == "call") == Counter(failed_calls)
        and all(row["when"] == "call" for row in failures))
    def attributed_call(row):
        if row["kind"] == "trusted_test_assertion":
            frame = row["trusted_test_frame"]
            return (row["exception_type"] == "AssertionError" and type(frame) is dict
                and frame["path"] in selected_files and frame["sha256"] == plan["source_pins"][frame["path"]]
                and report["raw_compiled_sources"].get(frame["path"]) == frame["sha256"])
        if row["kind"] == "changed_candidate_runtime_error":
            frames = row["candidate_traceback"]
            return (type(frames) is list and bool(frames) and all(frame["path"] in plan["required_candidate_bodies"]
                and frame["sha256"] == plan["source_pins"][frame["path"]]
                and report["executed_source_buffers"].get(frame["path"]) == frame["sha256"] for frame in frames))
        return False
    if code == 1 and call_shape and bodies and all(attributed_call(row) for row in failures):
        profile = "trusted_pytest_call_failure"
    else:
        valid_indices = {index for index, row in enumerate(report["syntax_failures"])
            if row["path"] in plan["required_candidate_bodies"] and row["filename"] == row["path"]
            and row["sha256"] == plan["source_pins"][row["path"]]
            and report["raw_compiled_sources"].get(row["path"]) == row["sha256"]
            and row["body_executed"] is False and row["path"] not in report["executed_source_buffers"]
            and row["exception_category"] == "SyntaxError" and row["exception_type"] in {"SyntaxError", "IndentationError", "TabError"}}
        wrapper_path = str(Path(policy["validation"]["pytest_module"]["path"]).parent.parent / "_pytest/python.py")
        def attributed_syntax(row):
            indices = row["syntax_witness_indices"]
            if not (type(indices) is list and bool(indices)
                    and all(type(index) is int and index in valid_indices for index in indices)):
                return False
            if row["kind"] == "original_source_syntax_error":
                return row["exception_type"] in {"SyntaxError", "IndentationError", "TabError"}
            if row["kind"] == "trusted_pytest_collect_wrapper":
                wrapper = row["trusted_wrapper_source"]
                return (row["exception_type"] == "CollectError" and type(wrapper) is dict
                    and wrapper["path"] == wrapper_path and descriptor(Path(wrapper_path)) == wrapper
                    and plan["source_pins"].get(wrapper_path) == wrapper["sha256"]
                    and report["executed_source_buffers"].get(wrapper_path) == wrapper["sha256"])
            return False
        collection_shape = (not phases and all(row["when"] == "collect" for row in failures)
            and junit["errors"] == len(failures) and junit["failures"] == 0 and junit["skipped"] == 0)
        usage_shape = False
        if code == 4:
            witness = report.get("collection_usage_attribution")
            main_path = str(Path(policy["validation"]["pytest_module"]["path"]).parent.parent / "_pytest/main.py")
            if type(witness) is dict and set(witness) == {"schema", "exception_type", "collector_source",
                    "original_raise_line", "no_collector_selectors"}:
                pin = witness["collector_source"]; line = witness["original_raise_line"]
                source_lines = ordinary_bytes(Path(main_path)).decode("utf-8").splitlines()
                failed_files = {row["report_path"] for row in failures}
                expected_selectors = [node for node in plan["selected_checks"] if node.partition("::")[0] in failed_files]
                usage_shape = (witness["schema"] == "agent-supervisor/original-pytest-no-collectors@1"
                    and witness["exception_type"] == "UsageError" and type(pin) is dict
                    and pin == descriptor(Path(main_path)) and plan["source_pins"].get(main_path) == pin["sha256"]
                    and report["executed_source_buffers"].get(main_path) == pin["sha256"]
                    and type(line) is int and 0 < line <= len(source_lines)
                    and source_lines[line - 1].strip() == "raise UsageError(*errors)"
                    and witness["no_collector_selectors"] == expected_selectors and bool(expected_selectors)
                    and len(expected_selectors) == len(set(expected_selectors))
                    and all("::" in node for node in expected_selectors)
                    and failed_files == {node.partition("::")[0] for node in expected_selectors}
                    and all(row["kind"] == "trusted_pytest_collect_wrapper" for row in failures)
                    and collection_shape and report["collected_cases"] == []
                    and report["natural_phase_counter_complete"] is False and junit["tests"] == len(failures))
        need(all(attributed_syntax(row) for row in failures)
             and ((code == 1 and call_shape) or (code == 2 and collection_shape) or usage_shape),
             "original failed reports do not prove the changed-candidate compilation SyntaxError")
        profile = "changed_candidate_source_syntax_error"
    need(report["semantic_negative"] is True and report["negative_profile"] == profile,
         "trusted negative classification differs from its actual evidence")
    return profile


def negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait):
    """Join immutable original facts; retained storage remains explicitly false."""
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair negative_validation_material")
    need(type(wait["returncode"]) is int and wait["returncode"] != 0 and wait["request"] == request_pin
         and wait["original_process_owner_returned"] is True and wait["actual_Popen_returned"] is True
         and wait["actual_process_waited"] is True and wait["actual_process_returncode"] == wait["returncode"]
         and json_object(wait["stdout"].encode()) == terminal, "original failed process wait is unknown")
    checked_descriptor(wait["engine_source"])
    need(wait["engine_source"] in policy["source_descriptors"], "original failed process source is not pinned")
    need(terminal["schema"] == "owned-validation-failure-terminal/v1" and terminal["completed"] is False
         and terminal["request_sha256"] == request_pin["sha256"]
         and terminal["process_cleanup_error"] is None and terminal["owner_exit_error"] is None,
         "original post-exit validation failure is incomplete")
    reservation = terminal["reservation"]; record = reservation["record"]; native = reservation["resource_lease"]
    identity = {"pid": record["owner_pid"], "birth": record["owner_birth"]}
    need(record["validation_request_sha256"] == request_pin["sha256"] and identity == wait["runner_identity"]
         and native["lease_id"] == record["native_lease_id"] and native["owner_pid"] == identity["pid"],
         "original failed request/native/PID/birth binding differs")
    lifecycle = terminal["source_custody_lifecycle"]
    need(lifecycle["request_sha256"] == request_pin["sha256"] and lifecycle["owner"] == identity
         and lifecycle["reservation_id"] == record["reservation_id"]
         and lifecycle["events"]["owner_exit_handling_completed"] is True
         and lifecycle["source_guards"]["errors"] == []
         and all(lifecycle["source_guards"][key] is True for key in ("request_unchanged", "source_pins_unchanged", "owner_identity_unchanged")),
         "original failed owner exit/source control flow is unknown")
    producer = lifecycle["producer"]
    need(type(producer) is dict and set(producer) == {"path", "sha256"}, "typed original failed producer required")
    producer_pin = next(pin for pin in policy["source_descriptors"] if pin["path"] == producer["path"])
    checked_descriptor(producer_pin)
    need(producer_pin["sha256"] == producer["sha256"], "original failed producer identity is unpinned")
    if _reviewed_parent_policy(policy):
        # Reuse the original SDK's post-exit request/native/source provenance
        # above. Failed scientific work cannot claim a semantic negative or a
        # child Source lease; only the actual parent closes its held READs.
        process = reservation["process_observation"]
        process_closed = (process["live_processes"] == 0
            and process["out_of_group_live_identities"] == [] and not process["tracking_error"])
        need(reservation["cleanup_error"] is None and native["released"] is True and process_closed
             and reservation["status"] == record["status"] == "retained",
             "original scientific failure native/process closure remains unknown")
        return {"semantic_negative": False, "negative_profile": None, "source_closed": False,
            "parent_source_ready_to_close": True, "native_closed": True, "processes_closed": True,
            "storage_closed": False, "reservation_id": record["reservation_id"],
            "native_lease_id": native["lease_id"]}
    report_path = Path(plan["attempt_directory"]) / "exports/trusted-validation.json"
    need(not report_path.lstat().st_mode & 0o222, "trusted worker report is not immutable")
    report = json_object(ordinary_bytes(report_path))
    need(report["worker_identity"] == record["child"], "trusted failed worker is not the original attached PID and birth")
    need(report["schema"] == WORK_SCHEMA and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"]
         and report["freeze"] == plan["freeze"] and report["source_commitment"] == plan["source_commitment"]
         and report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"])
         and report["startup"] == {"python": policy["python"], "isolated": 1, "no_bytecode": True}
         and type(report["source_READ_closed"]) is bool and type(report["cleanup_errors"]) is list,
         "original failed worker/report provenance is unknown")
    # Resource custody and semantic diagnosis are separate. An unsupported
    # failed test may still close its own unused storage promise without any
    # task, retry, inference or validation-acceptance authority.
    source_closed = report["source_READ_closed"] is True and report["cleanup_errors"] == []
    profile = None; semantic_error = None
    try:
        profile = verify_negative_semantic_report(policy, plan, report, invocation)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ET.ParseError) as error:
        semantic_error = {"type": type(error).__name__, "message": str(error)[:1000]}
    process = reservation["process_observation"]
    process_closed = (process["live_processes"] == 0 and process["out_of_group_live_identities"] == [] and not process["tracking_error"])
    native_closed = native["released"] is True
    storage_closed = reservation["status"] == record["status"] == "released"
    need(reservation["cleanup_error"] is None, "original resource cleanup is unknown")
    return {"semantic_negative": profile is not None, "negative_profile": profile, "semantic_error": semantic_error,
        "source_closed": source_closed, "native_closed": native_closed,
        "processes_closed": process_closed, "storage_closed": storage_closed,
        "reservation_id": record["reservation_id"], "native_lease_id": native["lease_id"],
        "process_scope": record["process_scope"], "original_owner_identity": identity,
        "original_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "trusted_semantic_report": descriptor(report_path),
        "original_JUnit_report": descriptor(Path(plan["attempt_directory"]) / "exports/test-results.xml")
            if (Path(plan["attempt_directory"]) / "exports/test-results.xml").exists() else None,
        "original_child_identity": record["child"]}


def capture_negative_validation(policy, plan, frozen, request_pin, invocation, completed):
    """Preserve original partial failure now; future storage release is separate."""
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair capture_negative_validation")
    wait_pin = descriptor(invocation / "runner-wait.json"); wait = json_object(checked_descriptor(wait_pin))
    value = {"schema": NEGATIVE_SCHEMA, "passed": False, "semantic_negative": False, "negative_profile": None, "semantic_error": None,
        "freeze_sha256": frozen["freeze_sha256"], "source_commitment": frozen["source_commitment"],
        "binding": frozen["binding"], "request": request_pin,
        "original_owner_returncode": completed.returncode, "original_owner_identity": wait["runner_identity"],
        "reservation_id": None, "native_lease_id": None, "process_scope": None, "original_record_sha256": None,
        "source_closed": False, "native_closed": False, "processes_closed": False, "storage_closed": False,
        "cleanup_errors": [], "original_failure_terminal": None, "original_runner_wait": wait_pin,
        "trusted_semantic_report": None, "original_JUnit_report": None, "original_child_identity": None,
        "observation_errors": [], "validation_replayed": False, "resource_recovery_performed": False}
    report_path = Path(plan["attempt_directory"]) / "exports/trusted-validation.json"
    if report_path.exists():
        # Retain the original file even when incomplete resource provenance
        # prevents treating its contents as accepted semantic/source evidence.
        value["trusted_semantic_report"] = descriptor(report_path)
    try:
        terminal = json_object(completed.stdout.encode())
        value["original_failure_terminal"] = immutable_bytes(invocation / "original-failure-terminal.json", completed.stdout.encode())
        value.update(negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration, ET.ParseError) as error:
        value["observation_errors"].append({"type": type(error).__name__, "message": str(error)[:1000]})
    proof_pin = new_json(invocation / "negative-validation.json", value)
    try:
        publish_closed_validation_control(policy, plan, frozen, request_pin, invocation, proof_pin)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as error:
        new_json(invocation / "closed-owner-publication-failure.json", {"schema": CLOSED_CONTROL_SCHEMA,
            "negative_disposition": proof_pin, "publication_completed": False,
            "error": {"type": type(error).__name__, "message": str(error)[:1000]},
            "original_validation_replayed": False, "resource_recovery_performed": False})
    return proof_pin


def verify_negative_validation(args):
    """Read-only join of original failed work and later public storage history.

    No allocation, recovery, validation replay, source write or service action
    is authorized by this observation. Missing or ambiguous history stays held.
    """
    policy = checked_policy(args.policy, args.policy_sha256)
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair verify_negative_validation")
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    invocation = args.invocation_root
    need(args.receipt_path.parent == invocation and args.freeze_path.parent == invocation.parent,
         "original negative invocation binding required")
    proof_path = invocation / "negative-validation.json"; proof_pin = descriptor(proof_path)
    need(not proof_path.lstat().st_mode & 0o222, "original negative proof must be immutable")
    proof = json_object(checked_descriptor(proof_pin))
    need(proof["schema"] == NEGATIVE_SCHEMA and proof["passed"] is False and proof["semantic_negative"] is True
         and proof["freeze_sha256"] == frozen["freeze_sha256"] and proof["source_commitment"] == frozen["source_commitment"]
         and proof["binding"] == frozen["binding"] and proof["cleanup_errors"] == [] and proof["observation_errors"] == [],
         "original trusted negative validation is unknown")
    plan = json_object(ordinary_bytes(invocation / "validation-plan.json"))
    need(plan["freeze"] == descriptor(args.freeze_path) and plan["source_commitment"] == frozen["source_commitment"],
         "negative original frozen plan differs")
    request_pin = original_execution_request(invocation)
    need(proof["request"] == request_pin and proof["original_runner_wait"] == descriptor(invocation / "runner-wait.json"),
         "negative original request/wait descriptors differ")
    terminal = json_object(checked_descriptor(proof["original_failure_terminal"]))
    wait = json_object(checked_descriptor(proof["original_runner_wait"]))
    observed = negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait)
    need(all(proof[key] == value for key, value in observed.items())
         and type(proof["original_owner_returncode"]) is int and proof["original_owner_returncode"] == wait["returncode"]
         and all(proof[key] is True for key in ("source_closed", "native_closed", "processes_closed")),
         "original failed source/native/process evidence does not establish closure")
    source = public_source_lease(policy)
    module_name = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_resource_recovery"
    expected_path = str(Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py")
    expected = next(pin for pin in policy["source_descriptors"] if pin["path"] == expected_path)
    with source(Path(policy["datasets_root"]), mode="read", timeout_seconds=policy["validation"]["timeout_seconds"]):
        checked_descriptor(expected)
        recovery = importlib.import_module(module_name)
        need(descriptor(Path(recovery.__file__).absolute()) == expected, "public disposition inspector origin differs")
        validation = policy["validation"]
        # This original selected-row inspector supplies an exact current CAS
        # preimage; status alone supplies no release authority.
        current = recovery.inspect_validation_reservation(validation["resource_ledger"], roots=validation["roots"],
            reservation_id=proof["reservation_id"], expected_request_sha256=request_pin["sha256"],
            attempt_directory=plan["attempt_directory"], process_scope=proof["process_scope"])
        need(current["present"] is True and current["binding_verified"] is True,
             "original selected validation custody is unavailable")
        disposition = recovery.inspect_validation_disposition(validation["resource_ledger"], roots=validation["roots"],
            reservation_id=proof["reservation_id"], expected_request_sha256=request_pin["sha256"],
            attempt_directory=plan["attempt_directory"], expected_record_sha256=current["record_sha256"],
            expected_previous_record_sha256=proof["original_record_sha256"], process_scope=proof["process_scope"])
    # Final key spelling must match the separately qualified public inspector.
    need(disposition["storage_release_established"] is True
         and disposition["reservation_id"] == proof["reservation_id"]
         and disposition["expected_request_sha256"] == request_pin["sha256"]
         and disposition["record_sha256"] == current["record_sha256"]
         and disposition["expected_previous_record_sha256"] == proof["original_record_sha256"]
         and disposition["schema"] == "validation-storage-disposition-inspection/v1"
         and disposition["present"] is True and disposition["binding_verified"] is True
         and disposition["status"] == "released" and disposition["native_release_established"] is False
         and disposition["state_written"] is False and disposition["recovery_performed"] is False and disposition["admitted"] is False
         and disposition["custody"]["native_lease_id"] == proof["native_lease_id"]
         and {"pid": disposition["custody"]["owner_pid"], "birth": disposition["custody"]["owner_birth"]} == proof["original_owner_identity"]
         and disposition["release_evidence"]["previous_record_sha256"] == proof["original_record_sha256"]
         and disposition["release_evidence"]["native_release_established"] is False
         and disposition["release_evidence"]["test_or_task_acceptance_asserted"] is False
         and disposition["release_evidence"]["historical_capacity_is_admission_authority"] is False,
         "genuine original selected-storage release/history is unproven")
    closure = {"schema": NEGATIVE_CLOSURE_SCHEMA, "negative_validation": proof_pin,
        **{key: proof[key] for key in ("freeze_sha256", "source_commitment", "binding", "negative_profile", "request", "original_owner_returncode",
            "original_owner_identity", "reservation_id", "native_lease_id")},
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True,
        "cleanup_errors": [], "storage_disposition": disposition, "observed_at": time.time()}
    print(canonical({"schema": NEGATIVE_VERIFIED_SCHEMA, "negative_validation": proof_pin,
        "closure_observation": closure}).decode(), flush=True)
    return 0

def public_source_lease(policy):
    d = Path(policy["datasets_root"])
    expected = str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py")
    pin = next(row for row in policy["source_descriptors"] if row["path"] == expected)
    checked_descriptor(pin)
    if _CONTROL_GUARD is None:
        sys.path.insert(0, str(d))
    else:
        roots = _CONTROL_GUARD.roots
        need(len(roots) == 2 and roots[1] == d, "authenticated cold Source roots differ")
        prefix = [str(root) for root in roots]
        sys.path[:] = prefix + [entry for entry in sys.path if entry not in prefix]
    os.environ["IPFS_DATASETS_PY_MINIMAL_IMPORTS"] = "1"
    module = importlib.import_module("ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease")
    need(descriptor(Path(module.__file__).absolute()) == pin, "canonical public source owner origin differs")
    if _reviewed_parent_policy(policy):
        _reviewed_parent_source_guard(module, pin)
    return module.source_lease


def git_command(root, argv, *, timeout, environment=None, stdin=None):
    """Fixed callers below provide all argv; provider strings are never commands."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import run_command
    value = run_command(["git", *argv], cwd=Path(root), timeout_seconds=timeout,
                        environment=environment, stdin=stdin, preserve_output_newlines=True)
    need(type(value.returncode) is int and value.returncode == 0, "original Git command failed; no replay")
    return value.stdout


def stage_source_successor(args, policy, frozen, contract):
    """Write real immutable Git objects; leave active HEAD/index/files unchanged."""
    preparing = json_object(ordinary_bytes(args.freeze_path.parent / "preparing.json"))
    prior = preparing["policy"]; active = Path(prior["repository_root"])
    need(active == Path(frozen["binding"]["active_root"]), "source snapshot root differs")
    need(set(frozen["changed_paths"]) <= set(prior["allowed_paths"]), "source path authority widened")
    need(contract["source_successor"] is None, "operator contract cannot inject a source successor")
    timeout = min(prior["git_timeout_seconds"], policy["validation"]["timeout_seconds"])
    clean = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    clean.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0", GIT_INDEX_FILE=str(args.invocation_root / "source-snapshot.index"))
    author_time = str(int(time.time())) + " +0000"
    clean.update(GIT_AUTHOR_NAME="Fleet repair owner", GIT_AUTHOR_EMAIL="repair-owner@localhost",
        GIT_COMMITTER_NAME="Fleet repair owner", GIT_COMMITTER_EMAIL="repair-owner@localhost",
        GIT_AUTHOR_DATE=author_time, GIT_COMMITTER_DATE=author_time)
    commands = []
    def run(argv, text=None):
        output = git_command(active, argv, timeout=timeout, environment=clean, stdin=text)
        commands.append({"argv": ["git", *argv], "returncode": 0, "stdout": output})
        return output.strip()
    lease = public_source_lease(policy)
    with lease(active, mode="write", timeout_seconds=timeout):
        need(run(["rev-parse", "--verify", "HEAD"]) == prior["base_commit"], "active source HEAD changed")
        index_environment = {k: v for k, v in clean.items() if k != "GIT_INDEX_FILE"}
        index = Path(git_command(active, ["rev-parse", "--git-path", "index"], timeout=timeout, environment=index_environment).strip())
        if not index.is_absolute():
            index = active / index
        before_index = descriptor(index)
        head = Path(git_command(active, ["rev-parse", "--git-path", "HEAD"], timeout=timeout, environment=index_environment).strip())
        if not head.is_absolute():
            head = active / head
        common = Path(git_command(active, ["rev-parse", "--git-common-dir"], timeout=timeout, environment=index_environment).strip())
        if not common.is_absolute():
            common = (active / common).resolve(strict=True)
        need(head.resolve() != (common / "HEAD").resolve(), "repair publication requires the explicitly selected linked worktree")
        head_pin = descriptor(head); primary = [descriptor(common / "HEAD"), descriptor(common / "index")]
        head_text = ordinary_bytes(head).decode("ascii").strip()
        target = head_text.removeprefix("ref: ") if head_text.startswith("ref: ") else "HEAD"
        need(target == "HEAD" or target.startswith("refs/heads/") and ".." not in target,
             "owned active worktree ref required")
        need(not (head_text.startswith("ref: ") and ordinary_bytes(common / "HEAD").decode().strip() == head_text),
             "repair cannot update the primary worktree branch")
        before = {}
        for rel in prior["source_paths"]:
            pin = descriptor(active / rel); baseline = preparing["baseline"][rel]
            need((pin["bytes"], pin["sha256"]) == (baseline["bytes"], baseline["sha256"]), "active source beforeimage differs")
            before[rel] = pin
        run(["read-tree", prior["base_commit"]])
        for rel in frozen["changed_paths"]:
            raw = ordinary_bytes(args.candidate_root / rel, 4 * 1024**2)
            need(hashlib.sha256(raw).hexdigest() == frozen["manifest"][rel]["sha256"], "frozen source changed")
            oid = run(["hash-object", "-w", "--stdin"], raw.decode("utf-8"))
            need(re.fullmatch(r"[0-9a-f]{40}", oid), "actual source blob missing")
            mode = "100755" if frozen["manifest"][rel]["executable"] else "100644"
            run(["update-index", "--add", "--cacheinfo", mode + "," + oid + "," + rel])
        tree = run(["write-tree"])
        commit = run(["commit-tree", tree, "-p", prior["base_commit"]], "Validated repair " + frozen["freeze_sha256"] + "\n")
        header = run(["cat-file", "-p", commit])
        need(header.startswith("tree " + tree + "\nparent " + prior["base_commit"] + "\n"), "new commit parent/tree differs")
        need(descriptor(index) == before_index and descriptor(head) == head_pin
             and run(["rev-parse", "--verify", "HEAD"]) == prior["base_commit"], "staging changed the active index or HEAD")
        for pin in primary:
            checked_descriptor(pin)
    successor = dict(prior, base_commit=commit)
    source_refs = [{"path": str(active / rel), "bytes": row["bytes"], "sha256": row["sha256"]}
                   for rel, row in sorted(frozen["manifest"].items()) if rel in prior["source_paths"]]
    need(len(source_refs) == len(prior["source_paths"]), "successor selected source population differs")
    return {"schema": "agent-supervisor/repair-source-snapshot@1", "prior_policy": prior,
        "successor_policy": successor, "new_commit": commit, "new_tree": tree,
        "source_commitment": frozen["source_commitment"], "freeze": descriptor(args.freeze_path),
        "validation": new_json(args.invocation_root / "validated-owner-receipt.json", json_object(ordinary_bytes(args.validation_path))["validation"]),
        "candidate_root": str(args.candidate_root), "before": before, "active_index": before_index,
        "active_head": head_pin, "ref_target": target, "primary_Git_preserved": primary,
        "changed_paths": frozen["changed_paths"], "manifest": frozen["manifest"],
        "successor_source_descriptors": source_refs, "commands": commands,
        "public_owner_policy": descriptor(args.policy), "source_WRITE_closed": True,
        "attempt_number": frozen["binding"]["attempt_number"], "attempt_token": frozen["binding"]["attempt_id"],
        "board_id": frozen["binding"]["board_id"]}



def cutover_remaining(deadline):
    """One finite original monotonic budget; callers may only shorten it."""
    need(type(deadline) in (int, float) and 0 < deadline < float("inf"), "finite original cutover deadline required")
    left = deadline - time.monotonic()
    need(left > 0, "original cutover deadline expired; retain partial effects without replay")
    return left


def original_cutover_clock(unit, identity, runner):
    """Use the actual systemd execution start, within its existing120s limit."""
    observed = runner({"argv": ["systemctl", "--user", "show", unit,
        "--property=MainPID,InvocationID,ExecMainStartTimestampMonotonic"]}, cwd="/", timeout=5)
    need(type(observed["returncode"]) is int and observed["returncode"] == 0, "original cutover clock observation failed")
    props = dict(line.split("=", 1) for line in observed["stdout"].splitlines() if "=" in line)
    need(props.get("MainPID") == str(identity["pid"]) and proc_identity(identity["pid"]) == identity
         and re.fullmatch(r"[0-9a-f]{32}", props.get("InvocationID", ""))
         and props.get("ExecMainStartTimestampMonotonic", "").isdecimal(), "original external invocation clock differs")
    started = int(props["ExecMainStartTimestampMonotonic"]) / 1000000
    need(0 < started <= time.monotonic(), "original external start time is unknown")
    deadline = started + 120
    cutover_remaining(deadline)
    return {"owner": identity, "invocation_id": props["InvocationID"], "started_monotonic": started,
        "deadline_monotonic": deadline, "existing_kernel_runtime_seconds": 120}

def install_source_successor(snapshot, *, deadline=None):
    """Original checkout lease and source WRITE, exact finite files/HEAD CAS."""
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        acquire_checkout_mutation_lease, checkout_lock_metadata, checkout_mutation_lock_path, release_checkout_mutation_lease)
    from scripts.ops.agent_supervisor.install_fleet_watchdog import atomic_write
    policy = json_object(checked_descriptor(snapshot["public_owner_policy"]))
    active = Path(snapshot["prior_policy"]["repository_root"])
    timeout = snapshot["prior_policy"]["git_timeout_seconds"]
    def budget():
        return min(timeout, cutover_remaining(deadline)) if deadline is not None else timeout
    metadata = checkout_lock_metadata(kind="fleet-repair-cutover", repo_root=active,
        task_id="infrastructure-repair:" + snapshot["board_id"], attempt=snapshot["attempt_number"],
        extra={"process_birth": proc_identity(os.getpid())["birth"], "repair_unit_token": snapshot["attempt_token"]})
    def alive(old):
        identity = proc_identity(int(old.get("pid") or 0)) if old.get("pid") else None
        return identity is not None and (not old.get("process_birth") or identity["birth"] == old["process_birth"])
    held, reason, _, _ = acquire_checkout_mutation_lease(checkout_mutation_lock_path(active), metadata,
        owner_active=alive, timeout_seconds=budget())
    need(held is not None, "original checkout owner unavailable: " + str(reason))
    try:
        with public_source_lease(policy)(active, mode="write", timeout_seconds=budget()):
            need(git_command(active, ["rev-parse", "--verify", "HEAD"], timeout=budget()).strip()
                 == snapshot["prior_policy"]["base_commit"], "source HEAD preimage differs")
            checked_descriptor(snapshot["active_index"])
            checked_descriptor(snapshot["active_head"])
            for pin in snapshot["primary_Git_preserved"]:
                checked_descriptor(pin)
            for pin in snapshot["before"].values():
                checked_descriptor(pin)
            # Exact owned branch update only; the primary worktree's HEAD and
            # index are untouched. No broad add/reset/checkout/merge occurs.
            for rel in snapshot["changed_paths"]:
                budget()
                source = Path(snapshot["candidate_root"]) / rel
                row = snapshot["manifest"][rel]; raw = ordinary_bytes(source, 4 * 1024**2)
                need(hashlib.sha256(raw).hexdigest() == row["sha256"], "qualified source origin changed")
                checked_descriptor(snapshot["before"][rel])
                atomic_write(active / rel, raw, mode=stat.S_IMODE((active / rel).lstat().st_mode))
            args = ["update-ref"] + (["--no-deref"] if snapshot["ref_target"] == "HEAD" else [])
            git_command(active, [*args, snapshot["ref_target"], snapshot["new_commit"], snapshot["prior_policy"]["base_commit"]], timeout=budget())
            checked_descriptor(snapshot["active_index"])
            for pin in snapshot["primary_Git_preserved"]:
                checked_descriptor(pin)
            for pin in snapshot["successor_source_descriptors"]:
                checked_descriptor(pin)
        with public_source_lease(policy)(active, mode="read", timeout_seconds=budget()):
            need(git_command(active, ["rev-parse", "--verify", "HEAD"], timeout=budget()).strip() == snapshot["new_commit"], "source HEAD postimage differs")
            for pin in snapshot["successor_source_descriptors"]:
                checked_descriptor(pin)
    finally:
        need(release_checkout_mutation_lease(held, require_exact_metadata=True), "original checkout release unknown")
    return {"completed": True, "source_WRITE_closed": True, "postwrite_source_READ_closed": True,
            "new_commit": snapshot["new_commit"], "new_tree": snapshot["new_tree"]}


def verify_original_job_closed(unit, runner, *, deadline=None):
    result = runner({"argv": ["systemctl", "--user", "show", unit,
        "--property=MainPID,ActiveState,ControlGroup"]}, cwd="/", timeout=min(10, cutover_remaining(deadline)) if deadline is not None else 10)
    need(result["returncode"] == 0, "original job state unavailable")
    props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    need(props.get("MainPID") == "0" and props.get("ActiveState") in {"inactive", "failed"}, "original job is not closed")
    group = props.get("ControlGroup", "")
    if group:
        need(group.startswith("/") and ".." not in Path(group).parts, "original cgroup identity differs")
        path = Path("/sys/fs/cgroup") / group.lstrip("/")
        if path.exists():
            need(not (path / "cgroup.procs").read_text().split() and not any(p.is_dir() for p in path.iterdir()), "original cgroup is not stopped")
    return props


def verify_source_successor_activation(link, frozen, activation):
    """Historical actual activation + exact original Git parent/tree/source."""
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and activation.get("source_commitment") == frozen["source_commitment"]
         and activation.get("old_owner_job_closed") is True, "unknown or foreign source activation")
    stage = json_object(checked_descriptor(activation["stage"]))
    need(link["git_snapshot"] == activation["stage"], "source snapshot descriptor differs")
    snapshot = stage["source_successor"]
    need(snapshot["prior_policy"] == link["prior_policy"] and snapshot["successor_policy"] == link["successor_policy"]
         and snapshot["freeze"] == link["freeze"] and snapshot["attempt_number"] == link["attempt_number"],
         "source activation chain differs from its original snapshot")
    need(frozen["source_commitment"] == digest({"base_commit": link["prior_policy"]["base_commit"], "manifest": frozen["manifest"]})
         and frozen["freeze_sha256"] == digest({k: v for k, v in frozen.items() if k != "freeze_sha256"}),
         "original source freeze commitment differs")
    timeout = link["prior_policy"]["git_timeout_seconds"]
    header = git_command(Path(link["prior_policy"]["repository_root"]), ["cat-file", "-p", snapshot["new_commit"]], timeout=timeout)
    need(header.split("\n\n", 1)[0].splitlines()[:2] == ["tree " + snapshot["new_tree"], "parent " + link["prior_policy"]["base_commit"]]
         and len([line for line in header.split("\n\n", 1)[0].splitlines() if line.startswith("parent ")]) == 1,
         "genuine source successor commit parent/tree differs")
    for relative, row in frozen["manifest"].items():
        raw = git_command(Path(link["prior_policy"]["repository_root"]),
            ["show", snapshot["new_commit"] + ":" + relative], timeout=timeout).encode("utf-8")
        need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
             "genuine source successor tree is not the validated source")
    installation = json_object(checked_descriptor(activation["installation"]))
    need(installation["completed"] is True and installation["source_WRITE_closed"] is True
         and installation["postwrite_source_READ_closed"] is True
         and all(type(c["returncode"]) is int and c["returncode"] == 0 for c in installation["commands"]),
         "original source/service cutover did not return known completion")
    verify_external_activation_closed(activation["external_owner"], link["external_terminal"], link["activation"])
    for suffix, pin in activation["loaded_runtime"].items():
        observed = json_object(checked_descriptor(pin))
        need(suffix in {"watchdog", "repair"} and observed["source_commitment"] == frozen["source_commitment"]
             and observed["loaded_module"] == stage["release_sources"]["fleet_" + suffix],
             "historical actual source-body attestation differs")


def validate_external_activation_terminal(owner, properties, observed_process):
    """Pure check of an original manager terminal and birth-bound identity.

    A recycled PID is evidence that the original process has gone. Its new
    owner never inherits this activation's identity or completion authority.
    """
    need(type(owner) is dict and set(owner) == {"unit", "owner", "invocation_id"},
         "closed original external owner identity required")
    identity = owner["owner"]
    need(type(identity) is dict and set(identity) == {"pid", "birth"}
         and type(identity["pid"]) is int and identity["pid"] > 0
         and type(identity["birth"]) is str and identity["birth"].isdecimal(),
         "original external PID and birth required")
    need(type(owner["unit"]) is str and re.fullmatch(r"ipfs-taskboard-" + re.escape(REPAIR_PROFILE)
         + r"-cutover-[0-9a-f]{32}\.service", owner["unit"])
         and type(owner["invocation_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", owner["invocation_id"]),
         "finite original external unit/invocation required")
    keys = {"MainPID", "InvocationID", "Result", "ExecMainStatus", "SubState", "ControlGroup", "TasksCurrent", "KillMode"}
    need(type(properties) is dict and set(properties) == keys
         and all(type(value) is str for value in properties.values())
         and properties["MainPID"] == "0" and properties["InvocationID"] == owner["invocation_id"]
         and properties["Result"] == "success" and properties["ExecMainStatus"] == "0"
         and properties["SubState"] in {"exited", "dead"}
         and properties["TasksCurrent"] == "0" and properties["KillMode"] == "control-group",
         "external activation original manager terminal is unknown")
    if observed_process is not None:
        need(type(observed_process) is dict and set(observed_process) == {"pid", "birth"}
             and type(observed_process["pid"]) is int and observed_process["pid"] == identity["pid"]
             and type(observed_process["birth"]) is str and observed_process["birth"].isdecimal()
             and observed_process["birth"] != identity["birth"],
             "original external activation PID and birth remain live or unknown")
    return properties


def observe_external_activation_closed(owner):
    """Capture the original manager's genuine terminal once, before sealing."""
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    need(type(owner) is dict and type(owner.get("unit")) is str, "original external unit required")
    result = command({"argv": ["systemctl", "--user", "show", owner["unit"],
        "--property=MainPID,InvocationID,Result,ExecMainStatus,SubState,ControlGroup,TasksCurrent,KillMode"]}, cwd="/", timeout=10)
    need(type(result["returncode"]) is int and result["returncode"] == 0,
         "original external manager observation failed")
    props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    process = proc_identity(owner["owner"]["pid"])
    validate_external_activation_terminal(owner, props, process)
    return {"terminal_properties": props, "process_observation": process}


def verify_external_activation_closed(owner, terminal_pin, activation_pin):
    """Authenticate immutable historical closure; never infer it from absence.

    The hash-bound observation is required even after transient manager state
    disappears. Missing, mutable or foreign history stays held for inspection.
    """
    need(terminal_pin is not None and activation_pin is not None,
         "immutable authenticated external terminal proof required")
    activation = json_object(checked_descriptor(activation_pin))
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and activation.get("external_owner") == owner, "external terminal activation binding differs")
    proof = json_object(checked_descriptor(terminal_pin))
    need(not Path(terminal_pin["path"]).lstat().st_mode & 0o222,
         "external terminal proof must be immutable")
    need(set(proof) == {"schema", "activation", "external_owner", "terminal_properties",
        "process_observation", "observed_at", "original_manager_terminal_verified"}
         and proof["schema"] == "agent-supervisor/repair-external-owner-terminal@1"
         and proof["activation"] == activation_pin and proof["external_owner"] == owner
         and proof["original_manager_terminal_verified"] is True
         and type(proof["observed_at"]) in (int, float) and proof["observed_at"] > 0,
         "external terminal proof is incomplete or belongs to another activation")
    return validate_external_activation_terminal(owner, proof["terminal_properties"], proof["process_observation"])


def seal_external_activation_terminal(activation_pin, invocation):
    """One original observation; existing proof is read-only and never replayed."""
    activation = json_object(checked_descriptor(activation_pin))
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and Path(activation_pin["path"]) == invocation / "actual-activation.json",
         "fixed original actual activation required")
    path = invocation / "external-owner-terminal.json"
    if path.exists():
        pin = descriptor(path)
    else:
        observed = observe_external_activation_closed(activation["external_owner"])
        pin = new_json(path, {"schema": "agent-supervisor/repair-external-owner-terminal@1",
            "activation": activation_pin, "external_owner": activation["external_owner"],
            **observed, "observed_at": time.time(), "original_manager_terminal_verified": True})
    verify_external_activation_closed(activation["external_owner"], pin, activation_pin)
    return pin


INITIAL_PROFILE_BOOTSTRAP_SCHEMA = "agent-supervisor/repair-initial-profile-bootstrap@1"
INITIAL_PROFILE_POLICY_HASH = "__CHECKED_OWNER_POLICY_SHA256__"
INITIAL_PROFILE_POLICY_BYTES = "__CHECKED_OWNER_POLICY_BYTES__"


def _bootstrap_marker_paths(value, prefix=()):
    result = set()
    if type(value) is str:
        markers = (INITIAL_PROFILE_POLICY_HASH, INITIAL_PROFILE_POLICY_BYTES)
        need(all(marker not in value or value == marker for marker in markers),
             "reserved bootstrap marker embedded in unrelated text")
        if value in markers:
            result.add(prefix)
    elif type(value) is dict:
        for key, item in value.items():
            need(type(key) is str and all(marker not in key for marker in
                 (INITIAL_PROFILE_POLICY_HASH, INITIAL_PROFILE_POLICY_BYTES)),
                 "reserved bootstrap marker used as a key")
            result.update(_bootstrap_marker_paths(item, prefix + (key,)))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.update(_bootstrap_marker_paths(item, prefix + (index,)))
    return result


def _bootstrap_config_projection(config, policy, checked_policy_reference, owner_policy_path):
    need(type(checked_policy_reference) is dict and
         checked_policy_reference.get("path") == owner_policy_path,
         "actual checked owner policy reference required")
    raw = checked_descriptor(checked_policy_reference)
    need(json_object(raw) == policy, "actual checked policy differs from the invoked policy")
    need(not _bootstrap_marker_paths(config), "live configuration contains a bootstrap marker")
    value = json_object(canonical(config))
    paths = set()
    runtime = value["repair_worker"]["candidate_runtime"]
    for role in ("validation_owner", "adoption_owner"):
        for command in ("argv", "verify_argv"):
            argv = runtime[role][command]
            need(type(argv) is list and all(type(arg) is str for arg in argv)
                 and argv.count("--policy") == argv.count("--policy-sha256") == 1
                 and not any(arg.startswith(("--policy=", "--policy-sha256=")) for arg in argv),
                 "four exact separate owner policy arguments required")
            policy_index = argv.index("--policy") + 1
            hash_index = argv.index("--policy-sha256") + 1
            need(policy_index < len(argv) and hash_index < len(argv)
                 and argv[policy_index] == owner_policy_path
                 and argv[hash_index] == checked_policy_reference["sha256"],
                 "live owner policy path/hash differs from the actual checked policy")
            argv[hash_index] = INITIAL_PROFILE_POLICY_HASH
            paths.add(("repair_worker", "candidate_runtime", role, command, hash_index))
    descriptor_roles = []
    for role in ("validation_owner", "adoption_owner"):
        refs = runtime[role].get("source_descriptors")
        need(type(refs) is list and 0 < len(refs) <= 128,
             "closed nested owner source descriptors required")
        matching = [(index, pin) for index, pin in enumerate(refs)
                    if type(pin) is dict and pin.get("path") == owner_policy_path]
        need(len(matching) <= 1, "duplicate nested policy source descriptor")
        descriptor_roles.append(bool(matching))
        if matching:
            index, pin = matching[0]
            need(pin == checked_policy_reference,
                 "live owner policy source descriptor differs from the actual checked policy")
            # The current immutable-control renderer includes this policy as
            # source DATA twice. Authenticate all real fields before replacing
            # only these two structural self-reference slots in the projection.
            refs[index] = {"path": owner_policy_path, "bytes": INITIAL_PROFILE_POLICY_BYTES,
                           "sha256": INITIAL_PROFILE_POLICY_HASH}
            paths.add(("repair_worker", "candidate_runtime", role, "source_descriptors", index, "bytes"))
            paths.add(("repair_worker", "candidate_runtime", role, "source_descriptors", index, "sha256"))
    need(all(descriptor_roles) or not any(descriptor_roles),
         "nested policy source descriptors must use the same renderer shape")
    need(_bootstrap_marker_paths(value) == paths and len(paths) == (8 if all(descriptor_roles) else 4),
         "only exact authenticated owner policy structural slots may be normalized")
    return value



def _bootstrap_current_profile_contract(policy, frozen, anchor, config, board, checked_policy_reference, config_raw):
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import digest as source_digest
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate import parse_candidate_policy
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime import parse_runtime_policy
    from scripts.ops.agent_supervisor.install_fleet_watchdog import observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    need(set(anchor) == {"schema", "owner_policy_path", "original_contract", "source_snapshot",
         "expected_config_projection", "expected_unit_postimages"}
         and anchor["schema"] == INITIAL_PROFILE_BOOTSTRAP_SCHEMA,
         "closed immutable initial profile bootstrap anchor required")
    anchor_path = Path(policy["adoption"]["path"])
    need(not anchor_path.lstat().st_mode & 0o222, "bootstrap anchor must be immutable")
    base = anchor["original_contract"]
    fields = {"schema", "profile", "config", "inventory", "units", "prior_job",
              "expected_loaded_before", "job_unit", "source_successor", "initial_release"}
    need(type(base) is dict and set(base) == fields
         and base["schema"] == "agent-supervisor/repair-profile-upgrade-contract@1"
         and base["profile"] == "legal-ir" and base["source_successor"] is None,
         "original existing-profile bootstrap grant differs")
    need(type(config_raw) is bytes and json_object(config_raw) == config,
         "bootstrap config object is not bound to its original bytes")
    expected = anchor["expected_config_projection"]
    projection = _bootstrap_config_projection(config, policy, checked_policy_reference,
                                              anchor["owner_policy_path"])
    need(type(expected) is dict and projection == expected
         and _bootstrap_marker_paths(expected) == _bootstrap_marker_paths(projection),
         "installed profile differs from the complete operator bootstrap projection")
    checked_descriptor(base["inventory"])
    checked_descriptor(base["prior_job"])
    snapshot = anchor["source_snapshot"]
    need(type(snapshot) is dict and set(snapshot) == {"prior_policy", "new_commit", "new_tree",
         "source_commitment", "manifest"}, "closed initial source snapshot required")
    worker = config["repair_worker"]
    selected = parse_candidate_policy(worker)
    need(selected is not None and snapshot["prior_policy"] == worker["candidate_policy"]
         and snapshot["new_commit"] == worker["candidate_policy"]["base_commit"]
         and type(snapshot["new_tree"]) is str and len(snapshot["new_tree"]) in (40, 64)
         and re.fullmatch(r"[0-9a-f]+", snapshot["new_tree"]) is not None,
         "bootstrap source policy/commit/tree differs")
    parse_runtime_policy(worker, selected)
    manifest = snapshot["manifest"]
    need(type(manifest) is dict and set(manifest) == set(selected.source_paths),
         "bootstrap selected source manifest differs")
    initial = json_object(checked_descriptor(base["initial_release"]))
    need(initial["schema"] == "agent-supervisor/repair-source-release@1"
         and initial["snapshot_commit"] == snapshot["new_commit"]
         and initial["snapshot_tree"] == snapshot["new_tree"]
         and initial["source_commitment"] == snapshot["source_commitment"]
         == source_digest({"base_commit": snapshot["new_commit"], "manifest": manifest}),
         "bootstrap release/source commitment differs")
    release = Path(config["runtime_release"])
    need(Path(base["initial_release"]["path"]).parent == release,
         "bootstrap release is not the explicit initial source grant")
    refs = {pin["path"]: pin for pin in worker["candidate_runtime"]["source_descriptors"]}
    for rel, row in manifest.items():
        need(type(row) is dict and set(row) == {"bytes", "sha256", "executable"}
             and type(row["executable"]) is bool, "closed bootstrap source row required")
        actual = refs[str(selected.repository_root / rel)]
        released = initial["selected_sources"][rel]
        need(actual["bytes"] == released["bytes"] == row["bytes"]
             and actual["sha256"] == released["sha256"] == row["sha256"]
             and released["path"] == str(release / rel)
             and bool((selected.repository_root / rel).lstat().st_mode & 0o111) == row["executable"],
             "bootstrap current/released source differs")
        checked_descriptor(released)
    posts = anchor["expected_unit_postimages"]
    need(type(posts) is dict and set(posts) == set(base["units"]) == {"watchdog", "repair"},
         "two exact bootstrap unit postimages required")
    current_units = {}
    unit_rows = []
    config_path = Path(base["config"]["path"])
    need(config_path.is_absolute() and len(config_path.parents) >= 4
         and config_path.parents[3] / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json" == config_path,
         "original canonical fleet configuration target differs")
    need(ordinary_bytes(config_path) == config_raw, "bootstrap configuration changed before descriptor")
    config_pin = descriptor(config_path)
    need(config_pin == {"path": str(config_path), "bytes": len(config_raw),
         "sha256": hashlib.sha256(config_raw).hexdigest()},
         "bootstrap configuration changed while its descriptor was authenticated")
    for suffix, post in posts.items():
        need(type(post) is dict and set(post) == {"path", "bytes", "sha256", "mode"}
             and post["path"] == base["units"][suffix]["path"]
             and Path(post["path"]).name == "ipfs-taskboard-legal-ir-" + suffix + ".service"
             and type(post["mode"]) is int and 0 <= post["mode"] <= 0o777,
             "bootstrap unit target/mode differs")
        pin = {key: post[key] for key in ("path", "bytes", "sha256")}
        raw = checked_descriptor(pin)
        need(stat.S_IMODE(Path(pin["path"]).lstat().st_mode) == post["mode"]
             and checked_policy_reference["sha256"].encode() not in raw
             and config_pin["sha256"].encode() not in raw,
             "bootstrap unit postimage differs or contains a self-referential hash")
        current_units[suffix] = pin
        unit_rows.append({"before": pin, "postimage_utf8": raw.decode(), "mode": post["mode"]})
    current = dict(base, config=config_pin, units=current_units)
    bindings = {"candidate_policy": worker["candidate_policy"],
        "candidate_runtime": worker["candidate_runtime"],
        "authorization_epoch": board["repair_authorization_epoch"], "source_snapshot": snapshot}
    seed = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1",
        "release": str(release), "source_commitment": snapshot["source_commitment"],
        "state_dir": config["state_dir"], "contract": current,
        "release_sources": initial["sources"], "selected_release_sources": initial["selected_sources"],
        "release_commitment": base["initial_release"], "source_successor": None,
        "initial_bindings": bindings, "files": unit_rows + [
            {"before": config_pin, "postimage_utf8": config_raw.decode(),
             "mode": stat.S_IMODE(config_path.lstat().st_mode)}]}
    science = next((item for item in config["boards"] if item["id"] == "legal-ir-scientific"), None)
    if science is not None and "native_executor_binding" in science:
        seed["scientific_executor_binding"] = science["native_executor_binding"]
    if science is not None and "native_executor_git_binding" in science:
        need("scientific_executor_binding" in seed, "scientific Git binding requires its exact profile")
        seed["scientific_executor_git_binding"] = science["native_executor_git_binding"]
    need(ordinary_bytes(config_path) == config_raw and descriptor(config_path) == config_pin
         and _bootstrap_config_projection(json_object(config_raw), policy, checked_policy_reference,
             anchor["owner_policy_path"]) == expected,
         "bootstrap configuration/policy changed before readiness")
    observed = observe_repair_upgrade(seed, command)
    need(ordinary_bytes(config_path) == config_raw and descriptor(config_path) == config_pin,
         "bootstrap configuration changed during readiness")
    need(set(observed["loaded_runtime"]) == {"watchdog", "repair"},
         "both genuine imported-source observations required")
    return dict(current, expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                for suffix, pin in observed["loaded_runtime"].items()})


def current_profile_contract(policy, frozen, *, checked_policy_reference=None):
    """Advance only exact prior profile postimages from the anchored chain."""
    need("source_qualification" not in policy["validation"], "immutable qualification has no adoption profile")
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import repair_job_directory
    grant = json_object(checked_descriptor(policy["adoption"]))
    bootstrap = grant.get("schema") == INITIAL_PROFILE_BOOTSTRAP_SCHEMA
    if bootstrap:
        need(set(grant) == {"schema", "owner_policy_path", "original_contract", "source_snapshot",
             "expected_config_projection", "expected_unit_postimages"}
             and type(grant["original_contract"]) is dict,
             "closed initial profile bootstrap anchor required")
        base = grant["original_contract"]
    else:
        base = grant
    need(base["source_successor"] is None, "immutable adoption contract must not contain a supplied successor")
    config_raw = ordinary_bytes(Path(base["config"]["path"]))
    config = json_object(config_raw)
    board = next((b for b in config["boards"] if b["id"] == frozen["binding"]["board_id"]), None)
    need(board is not None, "original configured repair board required")
    path = repair_job_directory(Path(config["state_dir"]), board) / "job.json"
    job = json_object(ordinary_bytes(path))
    require_current_authorization(config, board, job=job)
    chain = job.get("repair_source_successors", [])
    if not chain:
        if bootstrap:
            return _bootstrap_current_profile_contract(policy, frozen, grant, config, board,
                                                       checked_policy_reference, config_raw)
        checked_descriptor(base["config"])
        initial = json_object(checked_descriptor(base["initial_release"]))
        need(initial["snapshot_commit"] == config["repair_worker"]["candidate_policy"]["base_commit"]
             and Path(base["initial_release"]["path"]).parent == Path(config["runtime_release"]),
             "first profile source is not the explicit initial source grant")
        from scripts.ops.agent_supervisor.install_fleet_watchdog import observe_repair_upgrade
        from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
        # A legitimate restart can change a PID while preserving the exact
        # anchored source/config. Refresh only through genuine imported-source
        # attestation and original manager observations, never a lookup PID.
        seed = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1", "release": config["runtime_release"],
            "source_commitment": initial["source_commitment"], "state_dir": config["state_dir"], "contract": base,
            "release_sources": initial["sources"], "selected_release_sources": initial["selected_sources"],
            "release_commitment": base["initial_release"], "files": [
                {"before": pin, "postimage_utf8": ordinary_bytes(Path(pin["path"])).decode(),
                 "mode": stat.S_IMODE(Path(pin["path"]).lstat().st_mode)}
                for pin in [*base["units"].values(), base["config"]]]}
        science = next((item for item in config["boards"] if item["id"] == "legal-ir-scientific"), None)
        if science is not None and "native_executor_binding" in science:
            seed["scientific_executor_binding"] = science["native_executor_binding"]
        if science is not None and "native_executor_git_binding" in science:
            need("scientific_executor_binding" in seed, "scientific Git binding requires its exact profile")
            seed["scientific_executor_git_binding"] = science["native_executor_git_binding"]
        observed = observe_repair_upgrade(seed, command)
        return dict(base, expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                    for suffix, pin in observed["loaded_runtime"].items()})
    latest = json_object(checked_descriptor(chain[-1]))
    actual = json_object(checked_descriptor(latest["activation"]))
    previous = json_object(checked_descriptor(actual["stage"]))
    from scripts.ops.agent_supervisor.install_fleet_watchdog import verify_repair_upgrade, observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    verify_repair_upgrade(previous, installed=True)
    observed = observe_repair_upgrade(previous, command)
    need(previous["source_successor"]["successor_policy"] == config["repair_worker"]["candidate_policy"],
         "profile preimage is not the authenticated source-chain head")
    current = dict(base, config=descriptor(Path(base["config"]["path"])),
        units={suffix: descriptor(Path(pin["path"])) for suffix, pin in base["units"].items()},
        expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                               for suffix, pin in observed["loaded_runtime"].items()})
    # No new target, route, budget, epoch, inventory or legacy-job authority.
    need({k: v for k, v in current.items() if k not in {"config", "units", "expected_loaded_before"}}
         == {k: v for k, v in base.items() if k not in {"config", "units", "expected_loaded_before"}},
         "source successor widened the immutable profile contract")
    return current


def adopt(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    need("source_qualification" not in policy["validation"], "immutable qualification cannot enter repair adopt")
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    pending = json_object(ordinary_bytes(args.validation_path))
    need(pending.get("stage") == "adoption_pending" and pending.get("freeze_sha256") == frozen["freeze_sha256"], "accepted original validation required")
    validation_receipt = pending["validation"]
    validation_invocation = Path(validation_receipt["original_receipt_reference"]).parent
    actual_summary = verify_validation_evidence(policy,
        json_object(ordinary_bytes(validation_invocation / "validation-plan.json")),
        original_execution_request(validation_invocation),
        json_object(checked_descriptor(validation_receipt["original_receipt_descriptor"])), validation_invocation)
    if _v5_ordinary_validation(policy):
        need(validation_receipt["verified_checks"] == actual_summary, "V5 verified summary differs from actual evidence")
        verify_v5_parent_source_closeout(policy, original_execution_request(validation_invocation), validation_invocation,
            validation_receipt["parent_source_closeout"], actual_summary)
    checked_reference = descriptor(args.policy)
    need(checked_reference["sha256"] == args.policy_sha256,
         "actual checked owner policy changed before adoption")
    contract = current_profile_contract(policy, frozen, checked_policy_reference=checked_reference)
    contract["source_successor"] = stage_source_successor(args, policy, frozen, contract)
    from scripts.ops.agent_supervisor.install_fleet_watchdog import prepare_repair_upgrade
    stage = prepare_repair_upgrade(args.candidate_root, contract, frozen["source_commitment"])
    stage_pin = new_json(args.invocation_root / "profile-upgrade-stage.json", stage)
    body = {"schema": DEFERRED_SCHEMA, "adopted": False, "known_deferred": True,
        "freeze_sha256": frozen["freeze_sha256"], "source_commitment": args.source_commitment,
        "validation_sha256": pending["validation_sha256"], "stage": stage_pin,
        "activation_path": str(args.invocation_root / "activation-receipt.json"),
        "policy": descriptor(args.policy), "original_receipt_reference": stage_pin["path"],
        "original_receipt_descriptor": stage_pin, "owner_command_replay_authorized": False}
    new_json(args.receipt_path, body)


def consume_deferred_adoption(receipt: dict, invocation: Path):
    need(receipt.get("schema") == DEFERRED_SCHEMA and receipt.get("known_deferred") is True
         and receipt.get("adopted") is False and receipt.get("owner_command_replay_authorized") is False,
         "known staged adoption required")
    checked_descriptor(receipt["stage"])
    path = Path(receipt["activation_path"])
    need(path == invocation / "activation-receipt.json", "fixed original activation receipt path required")
    if not path.exists():
        return None
    result = json_object(ordinary_bytes(path))
    need(result.get("schema") == ADOPTION_SCHEMA and result.get("adopted") is True
         and result.get("freeze_sha256") == receipt["freeze_sha256"]
         and result.get("validation_sha256") == receipt["validation_sha256"]
         and result.get("loaded_source_commitment") == receipt["source_commitment"], "actual activation belongs to another candidate")
    verify_actual_activation(json_object(checked_descriptor(result["original_receipt_descriptor"])),
        external_terminal=result.get("external_terminal"), activation_pin=result["original_receipt_descriptor"])
    return result, path


def verify_adoption(args):
    checked_policy(args.policy, args.policy_sha256)
    receipt = json_object(ordinary_bytes(args.receipt_path))
    need(receipt["freeze_sha256"] == json_object(ordinary_bytes(args.freeze_path))["freeze_sha256"]
         and receipt["source_commitment"] == args.source_commitment, "adoption binding differs")
    if receipt["schema"] == DEFERRED_SCHEMA:
        consume_deferred_adoption(receipt, args.invocation_root)
    else:
        verify_actual_activation(json_object(checked_descriptor(receipt["original_receipt_descriptor"])),
            external_terminal=receipt.get("external_terminal"), activation_pin=receipt["original_receipt_descriptor"])


def verify_actual_activation(value, *, external_terminal=None, activation_pin=None):
    need(value.get("schema") == ACTIVATION_SCHEMA and value.get("completed") is True
         and value.get("old_owner_job_closed") is True, "actual independently closed activation required")
    from scripts.ops.agent_supervisor.install_fleet_watchdog import verify_repair_upgrade, observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    stage = json_object(checked_descriptor(value["stage"]))
    verify_repair_upgrade(stage, installed=True)
    verify_external_activation_closed(value["external_owner"], external_terminal, activation_pin)
    # A normal subsequent service restart does not invalidate the historical
    # cutover. Require a fresh original observation of the SAME source/config,
    # with its own real current PID/birth/invocation attestation.
    observe_repair_upgrade(stage, command)
    return value


def request_deferred_activation(config, board, job_path, runner):
    """Outer dispatcher only, after its separate original job actually exited."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime
    job = json_object(ordinary_bytes(job_path))
    bootstrap_pin = job.get("candidate_runtime", {}).get("bootstrap_result")
    if not bootstrap_pin:
        return False
    bootstrap = json_object(checked_descriptor(bootstrap_pin))
    pending_pin = bootstrap.get("deferred_adoption")
    if not pending_pin:
        return False
    need(bootstrap.get("original_checkout_lease_closed") is True
         and bootstrap.get("provider_known_closed") is True
         and bootstrap.get("stage") == "adoption_pending", "original owner/bootstrap has not closed")
    require_current_authorization(config, board, job=job)
    unit = runtime._json(runtime._wire(runner({"argv": ["systemctl", "--user", "show", config["repair_worker"]["job_unit"],
        "--property=MainPID,ActiveState"]}, cwd="/", timeout=10)))
    need(unit["returncode"] == 0 and "MainPID=0" in unit["stdout"].splitlines()
         and any(line in {"ActiveState=inactive", "ActiveState=failed"} for line in unit["stdout"].splitlines()), "original repair job is still live or unknown")
    pending = json_object(checked_descriptor(pending_pin)); invocation = Path(pending_pin["path"]).parent
    intent = invocation / "external-activation-request.json"
    if intent.exists():
        return False  # Unknown request outcomes are never replayed.
    policy = json_object(checked_descriptor(pending["policy"]))
    control = checked_control_release(policy, current_owner=False)
    argv = ["systemd-run", "--user", "--unit=ipfs-taskboard-" + REPAIR_PROFILE + "-cutover-" + job["repair_unit_token"],
        "--property=RemainAfterExit=yes", "--property=KillMode=control-group", "--property=RuntimeMaxSec=120",
        "--property=MemoryMax=256M", "--property=TasksMax=32", "--property=CPUQuota=100%",
        "--", policy["python"], "-I", "-B", str(control["owner"]), "activate",
        "--policy", pending["policy"]["path"], "--policy-sha256", pending["policy"]["sha256"], "--deferred", pending_pin["path"],
        "--deferred-sha256", pending_pin["sha256"], "--bootstrap", bootstrap_pin["path"], "--bootstrap-sha256", bootstrap_pin["sha256"]]
    new_json(intent, {"schema": DEFERRED_SCHEMA, "argv": argv, "bootstrap": bootstrap_pin, "deferred": pending_pin, "replay_authorized": False})
    result = runner({"argv": argv}, cwd="/", timeout=10)
    new_json(invocation / "external-activation-request-terminal.json", {"returncode": result["returncode"], "request": descriptor(intent)})
    return result["returncode"] == 0


def activate(args):
    pending_pin = {"path": str(args.deferred), "bytes": len(ordinary_bytes(args.deferred)), "sha256": args.deferred_sha256}
    bootstrap_pin = {"path": str(args.bootstrap), "bytes": len(ordinary_bytes(args.bootstrap)), "sha256": args.bootstrap_sha256}
    pending = json_object(checked_descriptor(pending_pin)); bootstrap = json_object(checked_descriptor(bootstrap_pin))
    need(pending["policy"] == descriptor(args.policy), "external activation owner policy differs from the original deferred grant")
    need(bootstrap.get("deferred_adoption") == pending_pin and bootstrap.get("original_checkout_lease_closed") is True
         and bootstrap.get("provider_known_closed") is True and bootstrap.get("stage") == "adoption_pending", "genuine original owner closure required")
    invocation = args.deferred.parent; plan = json_object(checked_descriptor(pending["stage"]))
    need(Path(pending["activation_path"]) == invocation / "activation-receipt.json", "activation path differs")
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime import _unit_snapshot
    from scripts.ops.agent_supervisor.install_fleet_watchdog import apply_repair_upgrade, observe_repair_upgrade
    lock_path = Path(plan["state_dir"]) / "repair-profile-cutover.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unit = "ipfs-taskboard-" + REPAIR_PROFILE + "-cutover-" + plan["source_successor"]["attempt_token"] + ".service"
        identity = proc_identity(os.getpid())
        clock = original_cutover_clock(unit, identity, command)
        deadline = clock["deadline_monotonic"]
        def bounded_runner(spec, *, cwd, timeout):
            return command(spec, cwd=cwd, timeout=min(timeout, cutover_remaining(deadline)))
        outer = _unit_snapshot(unit, {"memory_max_bytes": 256 * 1024**2,
            "cpu_quota_percent": 100, "tasks_max": 32, "runtime_max_seconds": 120}, bounded_runner)
        need(outer["invocation_id"] == clock["invocation_id"], "original cutover invocation changed")
        external_owner = {"unit": unit, "owner": identity, "invocation_id": outer["invocation_id"]}
        new_json(invocation / "activation-started.json", {"deferred": pending_pin, "bootstrap": bootstrap_pin, "replay_authorized": False})
        # The public installer rechecks the stopped original job and every CAS
        # immediately before its first write; this process is a separate unit.
        installation = apply_repair_upgrade(plan, bounded_runner, deadline=deadline)
        installation_pin = new_json(invocation / "profile-installation.json", installation)
        observed = observe_repair_upgrade(plan, bounded_runner, deadline=deadline)
        actual = {"schema": ACTIVATION_SCHEMA, "completed": True, "stage": pending["stage"],
            "installation": installation_pin, "source_commitment": pending["source_commitment"],
            "old_owner_job_closed": True, "bootstrap": bootstrap_pin, "deferred": pending_pin,
            "external_owner": external_owner, "external_owner_exit_not_yet_observed": True,
            "cutover_clock": clock, "completed_monotonic": time.monotonic(), **observed}
        # This immutable pre-exit fact is not an accepted adoption receipt.
        # The outer dispatcher observes the original manager's genuine terminal
        # before publishing any processes_closed/positive-adoption assertion.
        cutover_remaining(deadline)
        new_json(invocation / "actual-activation.json", actual)
    finally:
        os.close(fd)


def resume_source_publication(config, board, job_path):
    """Read-only existing proof + exact job CAS, never a cutover replay."""
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import lock, write_json
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import parse_authorization_epoch
    epoch = parse_authorization_epoch(Path(config["state_dir"]), board)
    if epoch is None or not epoch.value["not_before_observed_at"] <= time.time() < epoch.value["expires_at"]:
        return False
    job_pin = descriptor(job_path); job = json_object(checked_descriptor(job_pin))
    if job.get("status") != "adoption_pending":
        return False
    bootstrap_pin = job.get("candidate_runtime", {}).get("bootstrap_result")
    if not bootstrap_pin:
        return False
    bootstrap = json_object(checked_descriptor(bootstrap_pin)); pending_pin = bootstrap.get("deferred_adoption")
    if not pending_pin:
        return False
    pending = json_object(checked_descriptor(pending_pin)); invocation = Path(pending_pin["path"]).parent
    actual_path = invocation / "actual-activation.json"
    if not actual_path.exists():
        return False  # Unknown partial effects remain held, never replayed.
    actual_pin = descriptor(actual_path); actual = json_object(checked_descriptor(actual_pin))
    external_terminal = seal_external_activation_terminal(actual_pin, invocation)
    verify_actual_activation(actual, external_terminal=external_terminal, activation_pin=actual_pin)
    need(bootstrap["original_checkout_lease_closed"] is True and bootstrap["provider_known_closed"] is True
         and bootstrap["attempt_token"] == job["repair_unit_token"]
         and type(job["attempts"]) is int and bootstrap["attempts"] == job["attempts"]
         and job.get("repair_authorization_sha256") == epoch.commitment, "original epoch/job/bootstrap differs")
    stage = json_object(checked_descriptor(actual["stage"])); snapshot = stage["source_successor"]
    receipt = {"schema": ADOPTION_SCHEMA, "adopted": True,
        "freeze_sha256": pending["freeze_sha256"], "validation_sha256": pending["validation_sha256"],
        "source_commitment": pending["source_commitment"], "loaded_source_commitment": pending["source_commitment"],
        "runtime_release": stage["release"], "source_closed": True, "native_closed": True,
        "processes_closed": True, "storage_closed": True, "cleanup_errors": [],
        "original_receipt_reference": actual_pin["path"], "original_receipt_descriptor": actual_pin,
        "external_terminal": external_terminal}
    receipt_path = invocation / "activation-receipt.json"
    if receipt_path.exists():
        need(json_object(ordinary_bytes(receipt_path)) == receipt, "existing final adoption receipt differs")
    else:
        new_json(receipt_path, receipt)
    chain = job.get("repair_source_successors", [])
    link_path = invocation / "source-successor-link.json"
    if link_path.exists() and chain and chain[-1] == descriptor(link_path):
        return False
    link = {"schema": "agent-supervisor/repair-source-successor@1", "epoch_sha256": epoch.commitment,
        "attempt_number": job["attempts"], "previous_link": chain[-1] if chain else None,
        "prior_policy": snapshot["prior_policy"], "successor_policy": snapshot["successor_policy"],
        "freeze": snapshot["freeze"], "validation": snapshot["validation"],
        "activation": actual_pin, "git_snapshot": actual["stage"], "external_terminal": external_terminal}
    link_path = invocation / "source-successor-link.json"
    if link_path.exists():
        need(json_object(ordinary_bytes(link_path)) == link, "existing source link differs; no fork/replay")
        link_pin = descriptor(link_path)
    else:
        link_pin = new_json(link_path, link)
    if chain and chain[-1] == link_pin:
        return False
    successor_job = {**job, "repair_source_successors": [*chain, link_pin]}
    # Authenticate the entire proposed chain before publishing the only job
    # field that changes. Attempts, token, epoch, history and expiry are exact.
    require_current_authorization(config, board, job=successor_job)
    with lock(job_path.parent / "queue.lock") as acquired:
        need(acquired and descriptor(job_path) == job_pin, "original job CAS changed before source-link publication")
        write_json(job_path, successor_job)
    return True


def serve(args):
    """Fixed two-service entrypoint with a genuine imported-source attestation."""
    config = json_object(ordinary_bytes(args.config)); release = Path(config["runtime_release"])
    metadata = json_object(ordinary_bytes(release / "repair-source-commitment.json"))
    module_name = "fleet_watchdog" if args.service == "watchdog" else "fleet_repair"
    module = importlib.import_module("ipfs_accelerate_py.agent_supervisor.rescue." + module_name)
    source = descriptor(Path(module.__file__).absolute())
    need(source == metadata["sources"][module_name] and release in Path(source["path"]).parents,
         "actual imported module is outside the immutable repair release")
    invocation = os.environ.get("INVOCATION_ID", "")
    need(re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, "actual systemd invocation required")
    root = Path(config["state_dir"]) / "runtime-loaded"; root.mkdir(mode=0o700, exist_ok=True)
    new_json(root / (invocation + "-" + args.service + ".json"), {"schema": ACTIVATION_SCHEMA,
        "service": args.service, "owner": proc_identity(os.getpid()), "invocation_id": invocation,
        "source_commitment": metadata["source_commitment"], "loaded_module": source,
        "config": descriptor(args.config), "release": str(release)})
    return module.main((["--apply"] if args.service == "watchdog" else ["run"]) + ["--config", str(args.config)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "verify-validation", "verify-negative-validation", "verify-operational-validation", "verify-source-qualification", "adopt", "verify-adoption", "activate", "serve", "inspect-control", "verify-validation-admission"))
    for name in ("policy", "candidate-root", "freeze-path", "receipt-path", "validation-path", "invocation-root", "plan", "deferred", "bootstrap", "config"):
        parser.add_argument("--" + name, type=Path)
    for name in ("policy-sha256", "source-commitment", "plan-sha256", "deferred-sha256", "bootstrap-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--service", choices=("watchdog", "repair"))
    args = parser.parse_args(argv)
    required = {"activate": ("policy", "policy_sha256", "deferred", "deferred_sha256", "bootstrap", "bootstrap_sha256"), "serve": ("config", "service"), "inspect-control": ("policy", "policy_sha256")}
    fields = required.get(args.action, ("policy", "policy_sha256", "candidate_root", "freeze_path", "receipt_path", "source_commitment", "invocation_root"))
    if any(getattr(args, name) is None for name in fields) or args.action == "adopt" and args.validation_path is None:
        parser.error("all fixed owner binding arguments are required")
    if args.action != "serve":
        install_control_guard(args.policy, args.policy_sha256)
    dispatch = {"inspect-control": inspect_control, "validate": validate, "verify-validation": verify_validation, "verify-negative-validation": verify_negative_validation, "verify-operational-validation": verify_operational_validation, "verify-validation-admission": verify_validation_admission, "verify-source-qualification": verify_source_qualification, "adopt": adopt, "verify-adoption": verify_adoption,
        "activate": activate, "serve": serve}
    result = dispatch[args.action](args)
    return 0 if result is None else result


REVIEWED_PARENT_POLICY_SCHEMA = "reviewed-preservation-parent/v1"
REVIEWED_PARENT_WORKER_SCHEMA = "reviewed-preservation-native-worker/v1"
REVIEWED_PARENT_BINDING_SCHEMA = "reviewed-preservation-native-binding/v1"
REVIEWED_PARENT_SOURCE_SCHEMA = "reviewed-preservation-parent-source-observation/v1"
_REVIEWED_PARENT_CAPABILITIES = weakref.WeakKeyDictionary()
_REVIEWED_PARENT_ENGINE_BINDINGS = None
_REVIEWED_PARENT_SOURCE_BINDINGS = None
_REVIEWED_PARENT_PACKAGE_BINDINGS = None
# This exact maintained initializer is independently declared in the original
# Source manifest. Supporting a successor requires a new source qualification.
_REVIEWED_PARENT_ROOT_INITIALIZER_SHA256 = "5083369f823be18cd467346f6f3b955925ec65bedafe9eb28a61ca8897997c03"
_REVIEWED_PARENT_ROOT_INITIALIZER_BYTES = 40208



def _reviewed_parent_domain_v3(reviewed_version):
    # Fixed native domains, never request-selected callables or authority flags.
    need(type(reviewed_version) is int and reviewed_version in (2, 3, 4), "closed reviewed native version required")
    if reviewed_version == 2:
        return {"native_version": 2, "policy_schema": "reviewed-preservation-parent/v1",
            "worker_schema": "reviewed-preservation-native-worker/v1", "binding_schema": "reviewed-preservation-native-binding/v1",
            "fixture_schema": "reviewed-preservation-fixture-native-inputs/v1", "files": 77, "project": 34, "executed": 33, "compiled": 6,
            "success_schema": "reviewed-preservation-admitted-parent-success/v1",
            "success_profile": "original-admitted-parent-exact-reviewed-native-custody/v1"}
    if reviewed_version == 4:
        return {"native_version": 5, "policy_schema": "reviewed-definition-realization-parent/v1",
            "worker_schema": "reviewed-definition-realization-native-worker/v1",
            "binding_schema": "reviewed-definition-realization-native-binding/v1",
            "fixture_schema": "reviewed-definition-realization-fixture-native-inputs/v1",
            "success_schema": "reviewed-definition-realization-admitted-parent-success/v1",
            "success_profile": "original-admitted-parent-physical-definition-realization-custody/v1"}
    return {"native_version": 3, "policy_schema": "reviewed-preservation-parent/v3",
        "worker_schema": "reviewed-preservation-native-worker/v3", "binding_schema": "reviewed-preservation-native-binding/v3",
        "fixture_schema": "reviewed-preservation-fixture-native-inputs/v3", "files": 83, "project": 36, "executed": 35, "compiled": 8,
        "success_schema": "reviewed-preservation-admitted-parent-success/v3",
        "success_profile": "original-admitted-parent-exact-reviewed-native-custody/v3"}


def _reviewed_parent_version_v3(policy):
    need(type(policy) is dict and type(policy.get("schema")) is str
         and policy["schema"] in ("reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3", "reviewed-definition-realization-parent/v1"),
         "explicit reviewed parent native domain required")
    return {"reviewed-preservation-parent/v1": 2, "reviewed-preservation-parent/v3": 3,
        "reviewed-definition-realization-parent/v1": 4}[policy["schema"]]


def _reviewed_parent_checked_policy_v3(path, expected_policy_sha256, policy):
    return _checked_reviewed_preservation_parent_policy_core_v3(path, expected_policy_sha256,
        reviewed_version=_reviewed_parent_version_v3(policy))


def _reviewed_parent_binding_v3(policy, operation, binding, durability_files, *, attempt_directory):
    return _reviewed_parent_packet_binding_core_v3(operation, binding, durability_files,
        attempt_directory=attempt_directory, reviewed_version=_reviewed_parent_version_v3(policy))


def _reviewed_parent_policy(value):
    return type(value) is dict and type(value.get("schema")) is str and value["schema"] in ("reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3", "reviewed-definition-realization-parent/v1")


def _reviewed_parent_plan_sources(policy, policy_pin):
    v = policy["validation"]; d = Path(policy["datasets_root"])
    sources = {}
    for pin in [*policy["source_descriptors"], *v["import_sources"], *v["trusted_tests"], v["pytest_module"]]:
        need(pin["path"] not in sources or sources[pin["path"]] == pin["sha256"], "native source pin collision")
        sources[pin["path"]] = pin["sha256"]
    need(len(sources) <= 4096, "unchanged complete import manifest cap required")
    controls = {pin["path"]: pin["sha256"] for pin in policy["source_descriptors"]}
    controls.update({path: sha for path, sha in sources.items() if d in Path(path).parents})
    controls[policy_pin["path"]] = policy_pin["sha256"]
    data = policy["operation"]["native_inputs"]
    controls[data["path"]] = data["sha256"]
    need(len(controls) <= 253, "unchanged original SDK source control cap required")
    return sources, controls


def _reviewed_parent_check_plan_v1(policy, context):
    need(set(context) == {"plan", "request_pin", "invocation", "frozen"}, "closed original parent context required")
    domain = _reviewed_parent_domain_v3(_reviewed_parent_version_v3(policy))
    plan = context["plan"]; operation = policy["operation"]; invocation = Path(context["invocation"])
    fields = {"schema", "policy", "operation", "native_inputs", "candidate_root", "collection_root", "attempt_directory",
        "source_pins", "original_public_control_pins", "selected_checks", "expected_cases", "required_candidate_bodies",
        "ordinary_scratch_profile", "normal_namespace_profiles", "parent_source_observation_path",
        "parent_nonce_sha256", "parent_identity", "execution_request_path"}
    need(set(plan) == fields and plan["schema"] == domain["worker_schema"]
         and plan["operation"] == operation and plan["native_inputs"] == operation["native_inputs"]
         and plan["candidate_root"] == plan["collection_root"] == policy["source_root"]
         and plan["attempt_directory"] == str(Path(policy["validation"]["attempt_root"]) / operation["operation_id"])
         and plan["selected_checks"] == [operation["nodeid"]] and plan["expected_cases"] == 1
         and plan["ordinary_scratch_profile"] == _V5_ORDINARY_PROFILE
         and plan["normal_namespace_profiles"] == policy["validation"]["normal_namespace_profiles"]
         and plan["parent_identity"] == proc_identity(os.getpid())
         and plan["parent_source_observation_path"] == str(invocation / "parent-source-observation.json")
         and plan["execution_request_path"] == str(invocation / "execution-request.json"),
         "closed predeclared reviewed parent plan differs")
    source_pins, controls = _reviewed_parent_plan_sources(policy, plan["policy"])
    need(plan["source_pins"] == source_pins and plan["original_public_control_pins"] == controls,
         "whole reviewed parent source/data manifest differs")
    d = Path(policy["datasets_root"])
    need(plan["required_candidate_bodies"] == [str(d / "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py"),
          operation["nodeid"].split("::", 1)[0]], "exact actual native producer/test bodies required")
    plan_pin = descriptor(invocation / "validation-plan.json")
    need(json_object(checked_descriptor(plan_pin)) == plan, "original physical plan differs")
    request = json_object(checked_descriptor(context["request_pin"]))
    need(context["request_pin"]["path"] == plan["execution_request_path"]
         and request["source_sha256"] == {**controls, plan_pin["path"]: plan_pin["sha256"]}
         and request["argv"] == [policy["python"], "-I", "-B", str(Path(policy["source_root"]) /
             "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"),
             "--plan", plan_pin["path"], "--plan-sha256", plan_pin["sha256"]],
         "original reviewed parent request/source binding differs")
    frozen = context["frozen"]
    need(type(frozen) is dict and set(frozen) == {"schema", "parent_nonce"}
         and frozen["schema"] == "reviewed-preservation-original-parent-context/v1"
         and type(frozen["parent_nonce"]) is str and re.fullmatch(r"[0-9a-f]{64}", frozen["parent_nonce"])
         and hashlib.sha256(bytes.fromhex(frozen["parent_nonce"])).hexdigest() == plan["parent_nonce_sha256"],
         "original reviewed parent nonce context differs")


def _reviewed_parent_relative(value):
    need(type(value) is str and 0 < len(value) <= 512
         and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", value)
         and all(part not in (".", "..") for part in value.split("/")),
         "closed reviewed parent relative artifact path required")
    return value


def _reviewed_parent_native_descriptor(pin):
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}
         and type(pin["bytes"]) is int and 0 < pin["bytes"] <= 512 * 1024**2
         and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]),
         "bounded pinned native data descriptor required")
    path = Path(pin["path"])
    need(path.is_absolute() and ".." not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased native data required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in (0, os.getuid()) and before.st_size == pin["bytes"],
             "pinned ordinary native data differs")
        hashed = hashlib.sha256(); total = 0
        while part := os.read(fd, 1024**2):
            total += len(part); need(total <= pin["bytes"], "native data byte bound exceeded")
            hashed.update(part)
        stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        need(total == pin["bytes"] and hashed.hexdigest() == pin["sha256"]
             and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), "native data changed")
    finally:
        os.close(fd)


def _checked_reviewed_preservation_parent_policy_core_v3(path, expected_policy_sha256, *, reviewed_version):
    """Closed scientific validation policy; no repair/adoption permission."""
    domain = _reviewed_parent_domain_v3(reviewed_version)
    need(type(expected_policy_sha256) is str and _SHA.fullmatch(expected_policy_sha256),
         "independent reviewed parent policy SHA256 required")
    raw = ordinary_bytes(Path(path), 4 * 1024**2)
    need(hashlib.sha256(raw).hexdigest() == expected_policy_sha256, "reviewed parent policy differs")
    value = json_object(raw)
    need(set(value) == {"schema", "datasets_root", "source_root", "python", "source_descriptors",
                        "validation", "operation"} and value.get("schema") == domain["policy_schema"],
         "closed reviewed parent policy required")
    a, d = Path(value["source_root"]), Path(value["datasets_root"])
    need(a == Path(__file__).resolve().parents[3] and a != d
         and all(p.is_absolute() and ".." not in p.parts
                 and not any(part.is_symlink() for part in (p, *p.parents)) for p in (a, d)),
         "canonical reviewed parent source roots required")
    need(type(value["python"]) is str and Path(value["python"]).is_absolute(), "explicit parent Python required")
    sources = value["source_descriptors"]
    need(type(sources) is list and 0 < len(sources) <= 128
         and len({row["path"] for row in sources}) == len(sources), "finite original parent sources required")
    for pin in sources:
        need(pin["bytes"] <= 4 * 1024**2, "unchanged original source byte cap required")
        checked_descriptor(pin)
    v = value["validation"]
    fields = {"trusted_tests", "selected_checks", "pytest_module", "import_sources", "attempt_root",
        "cpu_slots", "memory_mb", "storage_bytes", "timeout_seconds", "child_process_slots",
        "scheduler_state_path", "expected_scheduler_policy_sha256", "roots", "resource_ledger",
        "ordinary_scratch_profile", "normal_namespace_profiles"}
    need(type(v) is dict and set(v) == fields and v["ordinary_scratch_profile"] == _V5_ORDINARY_PROFILE
         and all(type(v[name]) is int and v[name] == expected for name, expected in {
             "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 268435456 if reviewed_version == 4 else 16777216,
             "timeout_seconds": 300, "child_process_slots": 4}.items()),
         "explicit reviewed parent storage/300/1/1536/4 profile required")
    need(type(v["import_sources"]) is list and len(v["import_sources"]) <= 4096
         and type(v["trusted_tests"]) is list and len(v["trusted_tests"]) == 1,
         "finite exact reviewed parent import/test manifest required")
    all_pins = [*sources, *v["import_sources"], *v["trusted_tests"], v["pytest_module"]]
    for pin in all_pins:
        need(pin["bytes"] <= 4 * 1024**2, "unchanged normal source byte cap required")
        checked_descriptor(pin)
    # Descriptor integrity alone does not make pytest.__main__ its package.
    # Resolve metadata in the declared installation; never import or execute it.
    pytest_path = Path(v["pytest_module"]["path"])
    need(pytest_path.name == "__init__.py" and pytest_path.parent.name == "pytest",
         "reviewed parent Pytest must pin its normal package initializer")
    pytest_spec = importlib.machinery.PathFinder.find_spec(
        "pytest", [str(pytest_path.parent.parent)])
    need(pytest_spec is not None and pytest_spec.origin == str(pytest_path)
         and list(pytest_spec.submodule_search_locations or ()) == [str(pytest_path.parent)],
         "reviewed parent Pytest package metadata differs")
    pinned = {pin["path"]: pin["sha256"] for pin in all_pins}
    required = {str(Path(__file__).resolve()),
        str(a / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"),
        str(a / "ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py"),
        str(d / "scripts/ops/legal_ir/run_owned_validation.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"),
        *_v5_lifecycle_owner_paths(d)}
    need(required <= set(pinned), "complete original reviewed parent lifecycle sources required")
    direct_required = {str(Path(__file__).resolve()),
        str(a / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"),
        str(a / "ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py"),
        str(d / "scripts/ops/legal_ir/run_owned_validation.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py")}
    need(direct_required <= {pin["path"] for pin in sources}, "original process/Source/CLI direct descriptors required")
    namespace_profiles = v["normal_namespace_profiles"]
    native_namespaces_v2 = (type(namespace_profiles) is list
        and any(type(profile) is dict and type(profile.get("profile")) is str
            and profile["profile"] == "reviewed-preservation-normal-namespaces/v2" for profile in namespace_profiles))
    if native_namespaces_v2:
        expected_namespaces = (
            ("ipfs_datasets_py.logic.backends", "ipfs_datasets_py/logic/backends", "process.py"),
            ("ipfs_datasets_py.logic.families", "ipfs_datasets_py/logic/families", "namespaces.py"),
            ("ipfs_datasets_py.logic.intent_ir.formalize", "ipfs_datasets_py/logic/intent_ir/formalize", "modal_projections.py"),
        )
        need(len(namespace_profiles) == 3
             and all(type(profile) is dict and set(profile) == {
                 "profile", "fullname", "path", "device", "inode", "initializer_absent"}
                 for profile in namespace_profiles), "exact three native namespace v2 declarations required")
        for namespace, (fullname, relative, representative) in zip(namespace_profiles, expected_namespaces, strict=True):
            directory = d / relative
            need(type(namespace["profile"]) is str
                 and namespace["profile"] == "reviewed-preservation-normal-namespaces/v2"
                 and type(namespace["fullname"]) is type(namespace["path"]) is str
                 and namespace["fullname"] == fullname and namespace["path"] == str(directory)
                 and type(namespace["device"]) is int and namespace["device"] >= 0
                 and type(namespace["inode"]) is int and namespace["inode"] > 0
                 and namespace["initializer_absent"] is True
                 and str(directory / representative) in pinned,
                 "native namespace v2 name/path/identity/source declaration differs")
    else:
        need(type(v["normal_namespace_profiles"]) is list and len(v["normal_namespace_profiles"]) == 1,
             "single exact existing normal backends namespace profile required")
        namespace = v["normal_namespace_profiles"][0]
        need(type(namespace) is dict and set(namespace) == {"fullname", "path", "device", "inode", "initializer_absent"}
             and namespace["fullname"] == "ipfs_datasets_py.logic.backends"
             and namespace["path"] == str(d / "ipfs_datasets_py/logic/backends")
             and type(namespace["device"]) is int and namespace["device"] >= 0
             and type(namespace["inode"]) is int and namespace["inode"] > 0
             and namespace["initializer_absent"] is True,
             "only canonical normal backends namespace is permitted")
        namespace_profiles = [namespace]
    for namespace in namespace_profiles:
        directory = Path(namespace["path"])
        need(not any(p.is_symlink() for p in (directory, *directory.parents)), "unaliased normal namespace required")
        namespace_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(namespace_fd)
            need(stat.S_ISDIR(opened.st_mode) and opened.st_uid == os.getuid()
                 and (opened.st_dev, opened.st_ino) == (namespace["device"], namespace["inode"]),
                 "normal namespace physical identity changed")
            try:
                os.stat("__init__.py", dir_fd=namespace_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                need(False, "normal namespace initializer absence changed")
            after = directory.lstat()
            need((after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino), "normal namespace changed")
        finally:
            os.close(namespace_fd)
    need(type(v["roots"]) is list and 0 < len(v["roots"]) <= 4
         and all(type(root) is str and Path(root).is_absolute() for root in v["roots"]),
         "finite original named storage roots required")
    need(Path(v["attempt_root"]).is_absolute() and any(Path(root) in Path(v["attempt_root"]).parents
         or Path(root) == Path(v["attempt_root"]) for root in v["roots"]), "named attempt root required")
    for key in ("resource_ledger", "scheduler_state_path", "attempt_root"):
        p = Path(v[key]); need(p.is_absolute() and ".." not in p.parts
            and not any(q.is_symlink() for q in (p, *p.parents)), "unaliased existing control path required")
    need(type(v["expected_scheduler_policy_sha256"]) is str
         and _SHA.fullmatch(v["expected_scheduler_policy_sha256"]), "original scheduler policy pin required")
    op = value["operation"]
    need(type(op) is dict and set(op) == {"operation_id", "packet_root", "nodeid", "expected_source_sha256",
        "expected_graph_sha256", "expected_admission_sha256", "expected_preservation_sha256", "toolchain",
        "expected_binary_sha256", "native_argv", "native_data", "native_inputs"}
        | ({"expected_packet_paths"} if reviewed_version in (3, 4) else set())
        | (_reviewed_realization_operation_fields_v1() if reviewed_version == 4 else set()),
        "closed reviewed parent operation required")
    data_pin = op["native_inputs"]
    need(type(data_pin) is dict and set(data_pin) == {"path", "bytes", "sha256"}
         and type(data_pin["bytes"]) is int and 0 < data_pin["bytes"] <= 4 * 1024**2,
         "bounded original public native input DATA descriptor required")
    data = json_object(checked_descriptor(data_pin))
    need(set(data) == ({"schema", "fixture_only", "provenance_kind", "preparation", "graph", "admission", "args", "kwargs",
        "expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"}
        | (_reviewed_realization_data_fields_v1() if reviewed_version == 4 else set()))
        and data["schema"] == domain["fixture_schema"]
        and data["fixture_only"] is True and data["provenance_kind"] == "fictional_fixture",
        "closed public signed fictional native fixture DATA required")
    need(all(data[name] == op[name] for name in ("expected_preservation_sha256", "expected_graph_sha256",
        "expected_admission_sha256", "toolchain")), "public native fixture differs from predeclared operation pins")
    forbidden = {"private_key", "private_key_bytes", "secret_key", "signing_key", "private_bytes", "signers"}
    # Tagged dictionaries add three JSON edges per logical child. Include
    # the public envelope and terminal atom.value; the decoder still limits64.
    raw_depth_limit = 3 * 64 + 2 if reviewed_version in (3, 4) else 64
    pending, nodes = [(data, 0)], 0
    while pending:
        item, depth = pending.pop(); nodes += 1
        need(nodes <= 2_000_000 and depth <= raw_depth_limit, "bounded public native input DATA tree required")
        if type(item) is dict:
            need(not forbidden.intersection(item), "private signer or signing key DATA is forbidden")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    need(data_pin["path"] not in pinned, "native input DATA must not grant source execution permission")
    need(type(op["operation_id"]) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", op["operation_id"]),
         "exact reviewed operation ID required")
    if reviewed_version == 3:
        need(op["expected_packet_paths"] == _reviewed_parent_packet_paths_v3(data),
             "V3 opening packet paths differ from pinned complete preparation DATA")
    elif reviewed_version == 4:
        need(any(pin["path"] == str(d / "ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py")
                 for pin in sources), "physical realization requires direct predispatch coordinator Source pin")
        _reviewed_parent_realization_inputs_v1(data, op)
        need(op["expected_packet_paths"] == _reviewed_parent_realization_packet_paths_v1(data),
             "physical realization opening packet paths differ from pinned complete DATA")
    need(_reviewed_parent_relative(op["packet_root"]).startswith("exports/"), "native packet must be inside owned exports")
    need(v["selected_checks"] == [op["nodeid"]] and type(op["nodeid"]) is str and "::" in op["nodeid"]
         and op["nodeid"].split("::", 1)[0] == v["trusted_tests"][0]["path"]
         and any(a / folder in Path(v["trusted_tests"][0]["path"]).parents for folder in ("test", "tests")),
         "single exact maintained canonical native test node required")
    need(type(op["expected_source_sha256"]) is dict and op["expected_source_sha256"]
         and all(pinned.get(name) == pin for name, pin in op["expected_source_sha256"].items()),
         "exact reviewed native producer source pins required")
    for name in ("expected_graph_sha256", "expected_admission_sha256", "expected_preservation_sha256"):
        need(type(op[name]) is str and _SHA.fullmatch(op[name]), "explicit reviewed operation pin required")
    need(type(op["toolchain"]) is str and re.fullmatch(r"leanprover/lean4:v[0-9]+\.[0-9]+\.[0-9]+", op["toolchain"]),
         "version-pinned reviewed native toolchain required")
    binaries = op["expected_binary_sha256"]
    need(type(binaries) is dict and len(binaries) == 2
         and {Path(name).name for name in binaries} == {"lake", "lean"}
         and len({str(Path(name).parent) for name in binaries}) == 1
         and all(Path(name).is_absolute() and type(pin) is str and _SHA.fullmatch(pin) for name, pin in binaries.items()),
         "exact sibling reviewed native compiler pins required")
    roster = op["native_data"]
    binroot = Path(next(name for name in binaries if Path(name).name == "lake")).parent
    required_native = {"/usr/bin/prlimit", *(str(binroot / name) for name in ("lake", "lean")),
        *(str(binroot.parent / "lib/lean" / name) for name in ("libLake_shared.so", "libInit_shared.so",
            "libleanshared_1.so", "libleanshared_2.so", "libleanshared.so"))}
    need(type(roster) is list and len(roster) == 8 and {pin["path"] for pin in roster} == required_native
         and sum(pin["bytes"] for pin in roster) <= 512 * 1024**2, "exact eight native data files required")
    for pin in roster:
        _reviewed_parent_native_descriptor(pin)
        need(pin["path"] not in binaries or pin["sha256"] == binaries[pin["path"]], "native compiler roster pin differs")
    need(type(op["native_argv"]) is list and len(op["native_argv"]) == 3
         and all(type(argv) is list and argv and all(type(x) is str and x and "\x00" not in x for x in argv)
                 and argv[0] == "/usr/bin/prlimit" and "--" in argv for argv in op["native_argv"]),
         "three exact predeclared original prlimit launches required")
    wanted = [[str(binroot / "lake"), "--version"], [str(binroot / "lean"), "--version"],
              [str(binroot / "lake"), "--offline", "--no-cache", "build", "legal"]]
    need([argv[argv.index("--") + 1:] for argv in op["native_argv"]] == wanted,
         "exact original probe/probe/offline legal build roster required")
    prefix = ["/usr/bin/prlimit", "--core=0:0"]
    expected_argv = [prefix + ["--fsize=16777216:16777216", "--", *wanted[0]],
        prefix + ["--fsize=16777216:16777216", "--", *wanted[1]],
        prefix + ["--fsize=134217728:134217728", "--cpu=60:60", "--", *wanted[2]]]
    need(op["native_argv"] == expected_argv, "exact three bounded original native command limits required")
    return value

def checked_reviewed_preservation_parent_policy_v1(path, expected_policy_sha256):
    return _checked_reviewed_preservation_parent_policy_core_v3(path, expected_policy_sha256, reviewed_version=2)

def checked_reviewed_preservation_parent_policy_v3(path, expected_policy_sha256):
    return _checked_reviewed_preservation_parent_policy_core_v3(path, expected_policy_sha256, reviewed_version=3)


def _reviewed_parent_proc_stat(pid):
    need(type(pid) is int and pid > 0, "positive original parent process ID required")
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    need(fields[19].isdigit() and int(fields[19]) > 0, "original process birth required")
    return {"pid": pid, "ppid": int(fields[1]), "birth": fields[19]}


def _reviewed_parent_fd_observation(pid, fd):
    need(type(pid) is int and pid > 0 and type(fd) is int and 0 <= fd <= 1_000_000,
         "bounded original Source descriptor required")
    path = Path(f"/proc/{pid}/fd/{fd}"); info = path.stat()
    text = Path(f"/proc/{pid}/fdinfo/{fd}").read_text()
    need(len(text) <= 16384 and stat.S_ISREG(info.st_mode), "ordinary original Source FD required")
    return {"device": info.st_dev, "inode": info.st_ino, "lock_path": os.readlink(path), "fdinfo": text}


def _reviewed_parent_proc_executable(pid):
    return Path(f"/proc/{pid}/exe").resolve(strict=True)


def _reviewed_parent_kernel_read_lock(pid, row):
    observed = _reviewed_parent_fd_observation(pid, row["fd"])
    need(all(observed[name] == row[name] for name in ("device", "inode", "lock_path")),
         "original parent Source FD identity changed")
    expected = (pid, os.major(row["device"]), os.minor(row["device"]), row["inode"])
    matches = []
    for line in observed["fdinfo"].splitlines():
        match = re.fullmatch(r"lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+READ\s+(\d+)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF", line)
        if match:
            matches.append((int(match[1]), int(match[2], 16), int(match[3], 16), int(match[4])))
    need(expected in matches, "genuine parent kernel FLOCK READ is not held")



class _ReviewedParentSourceRoots:
    """Process-local Source metadata binding; never a Native success handle."""
    __slots__ = ("__weakref__",)


_REVIEWED_PARENT_SOURCE_ROOTS = weakref.WeakKeyDictionary()


def _reviewed_parent_source_root_pin(policy):
    path = str(Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py")
    return next(row for row in policy["source_descriptors"] if row["path"] == path)


def _reviewed_parent_check_source_root_binding(binding, source_pin, expected_roots):
    need(type(binding) is _ReviewedParentSourceRoots, "original canonical Source root binding required")
    entry = _REVIEWED_PARENT_SOURCE_ROOTS.get(binding)
    need(entry is not None and entry[0] == os.getpid() and entry[1] == threading.get_native_id()
         and entry[2] == canonical(source_pin)
         and entry[4] == tuple(str(Path(root)) for root in expected_roots),
         "original canonical Source root binding differs")
    module = entry[3]
    need(type(module) is types.ModuleType
         and module.__name__ == "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease"
         and sys.modules.get(module.__name__) is module
         and module.__spec__ is not None and module.__spec__.origin == source_pin["path"]
         and descriptor(Path(module.__file__)) == source_pin,
         "pinned canonical Source owner changed")
    for name, state, wrapped_state in entry[6]:
        function = vars(module).get(name)
        need(type(function) is types.FunctionType and function is state[0]
             and function.__code__ is state[1] and function.__defaults__ is state[2]
             and tuple(sorted((function.__kwdefaults__ or {}).items())) == state[3],
             "pinned canonical Source function changed: " + name)
        if wrapped_state is not None:
            wrapped = getattr(function, "__wrapped__", None)
            need(wrapped is wrapped_state[0] and wrapped.__code__ is wrapped_state[1]
                 and wrapped.__defaults__ is wrapped_state[2]
                 and tuple(sorted((wrapped.__kwdefaults__ or {}).items())) == wrapped_state[3]
                 and function.__closure__ is not None and len(function.__closure__) == 1
                 and function.__closure__[0].cell_contents is wrapped,
                 "pinned canonical Source wrapper changed: " + name)
    return entry[5]


def _reviewed_parent_prepare_source_root_binding(source_pin, expected_roots):
    """Resolve once before Native audit; subsequent checks perform no Git/lease."""
    need(type(expected_roots) in (tuple, list) and len(expected_roots) == 2,
         "two physical expected Source roots required")
    raw = checked_descriptor(source_pin)
    module = importlib.import_module("ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease")
    need(type(module) is types.ModuleType and sys.modules.get(module.__name__) is module
         and module.__file__ == module.__spec__.origin == source_pin["path"],
         "normal pinned Source42 canonicalizer owner required")
    codes = {}
    def walk(code):
        codes[code.co_qualname] = code
        for value in code.co_consts:
            if type(value) is types.CodeType: walk(value)
    walk(_ORIGINAL_COMPILE(raw, source_pin["path"], "exec", dont_inherit=True, optimize=sys.flags.optimize))
    states = []
    for name in ("_absolute", "_git", "canonical_source_root", "_directory", "source_lease"):
        function = vars(module).get(name); wrapped = getattr(function, "__wrapped__", None)
        need(type(function) is types.FunctionType, "pinned Source function required")
        if name in ("_directory", "source_lease"):
            need(type(wrapped) is types.FunctionType and wrapped.__globals__ is vars(module)
                 and wrapped.__code__ == codes.get(name)
                 and function.__code__ == contextlib.contextmanager(wrapped).__code__,
                 "pinned Source wrapper code origin differs")
        else:
            need(function.__globals__ is vars(module) and function.__code__ == codes.get(name),
                 "pinned Source canonicalizer code origin differs")
        states.append((name, _reviewed_parent_function_state(function),
                       _reviewed_parent_function_state(wrapped) if wrapped is not None else None))
    binding = object.__new__(_ReviewedParentSourceRoots)
    roots = tuple(str(Path(root)) for root in expected_roots)
    provisional = (os.getpid(), threading.get_native_id(), canonical(source_pin), module, roots, (), tuple(states))
    _REVIEWED_PARENT_SOURCE_ROOTS[binding] = provisional
    try:
        resolved = []
        for root in expected_roots:
            _reviewed_parent_check_source_root_binding(binding, source_pin, expected_roots)
            resolved.append(Path(module.canonical_source_root(root)))
            _reviewed_parent_check_source_root_binding(binding, source_pin, expected_roots)
        # Order and duplicates belong to the original two-READ protocol.
        _REVIEWED_PARENT_SOURCE_ROOTS[binding] = provisional[:5] + (tuple(resolved), provisional[6])
        _reviewed_parent_check_source_root_binding(binding, source_pin, expected_roots)
    except BaseException:
        _REVIEWED_PARENT_SOURCE_ROOTS.pop(binding, None)
        raise
    return binding


def _parent_source_observation_core(raw, *, expected_request_sha256,
        expected_parent_identity, expected_nonce_sha256, expected_sdk_argv, expected_roots,
        expected_schema, source_root_bindings=None, live=True):
    need(type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded original Source observation bytes required")
    value = json_object(raw)
    fields = {"schema", "request", "parent_identity", "nonce", "source_receipts", "source_fds", "sdk_argv"}
    if source_root_bindings is not None:
        fields.add("source_root_bindings")
        need(value.get("source_root_bindings") == source_root_bindings,
             "immutable physical/canonical Source root binding differs")
    need(type(value) is dict and set(value) == fields
         and value["schema"] == expected_schema
         and value["parent_identity"] == expected_parent_identity and value["sdk_argv"] == expected_sdk_argv
         and value["request"]["sha256"] == expected_request_sha256,
         "original parent Source observation binding differs")
    need(type(value["nonce"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["nonce"])
         and hashlib.sha256(bytes.fromhex(value["nonce"])).hexdigest() == expected_nonce_sha256,
         "original parent Source nonce differs")
    checked_descriptor(value["request"])
    need(type(expected_parent_identity) is dict and set(expected_parent_identity) == {"pid", "birth"}
         and type(expected_parent_identity["pid"]) is int and expected_parent_identity["pid"] > 0
         and type(expected_parent_identity["birth"]) is str and expected_parent_identity["birth"].isascii()
         and expected_parent_identity["birth"].isdigit() and int(expected_parent_identity["birth"]) > 0,
         "typed original parent Source identity required")
    observed_identity = value['parent_identity']
    need(type(observed_identity) is dict and set(observed_identity) == {'pid', 'birth'}
         and type(observed_identity['pid']) is int and observed_identity['pid'] > 0
         and type(observed_identity['birth']) is str and observed_identity['birth'].isascii()
         and observed_identity['birth'].isdigit() and int(observed_identity['birth']) > 0,
         'typed observed parent Source identity required')
    if live:
        parent = _reviewed_parent_proc_stat(expected_parent_identity["pid"])
        need({name: parent[name] for name in ("pid", "birth")} == expected_parent_identity,
             "original parent Source process birth differs")
    need(type(value["source_receipts"]) is list and type(value["source_fds"]) is list
         and len(value["source_receipts"]) == len(value["source_fds"]) == len(expected_roots) == 2,
         "two original parent Source READs required")
    for receipt, row, root in zip(value["source_receipts"], value["source_fds"], expected_roots, strict=True):
        need(type(receipt) is dict and set(receipt) == {"canonical_root", "mode", "lock_path", "waited_seconds", "schema_version", "cooperative"}
             and receipt["canonical_root"] == str(root) and receipt["mode"] == "read"
             and receipt["schema_version"] == "autoencoder-source-lease/v1" and receipt["cooperative"] is True
             and type(receipt["waited_seconds"]) in (int, float)
             and math.isfinite(receipt["waited_seconds"]) and receipt["waited_seconds"] >= 0
             and type(row) is dict and set(row) == {"root", "lock_path", "fd", "device", "inode"}
             and row["root"] == str(root) and row["lock_path"] == receipt["lock_path"]
             and all(type(row[key]) is int and row[key] >= 0 for key in ("fd", "device", "inode")),
             "original parent Source receipt/descriptor differs")
        lock = Path(row["lock_path"])
        need(lock == Path(root) / "workspace/control-plane/source-leases/source.lock"
             and lock.is_absolute() and not any(p.is_symlink() for p in (lock, *lock.parents)),
             "unaliased canonical Source lock path required")
        if live:
            info = lock.lstat()
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid()
                 and (info.st_dev, info.st_ino) == (row["device"], row["inode"]), "canonical Source lock identity differs")
            _reviewed_parent_kernel_read_lock(parent["pid"], row)
    return value


def _reviewed_parent_validate_source_observation(raw, *, expected_request_sha256,
        expected_parent_identity, expected_nonce_sha256, expected_sdk_argv, expected_roots):
    return _parent_source_observation_core(raw, expected_request_sha256=expected_request_sha256,
        expected_parent_identity=expected_parent_identity, expected_nonce_sha256=expected_nonce_sha256,
        expected_sdk_argv=expected_sdk_argv, expected_roots=expected_roots,
        expected_schema=REVIEWED_PARENT_SOURCE_SCHEMA)


def verify_reviewed_preservation_parent_source_observation_v1(observation_raw, *,
        expected_request_sha256, expected_parent_identity, expected_nonce_sha256, expected_sdk_argv, expected_roots):
    """Observe real parent's held Source FDs through the actual SDK child chain.

    This pure read-only check acquires no lease and mints no capability. The
    child owns no Source closure; only the original parent closes these READs.
    """
    value = _reviewed_parent_validate_source_observation(observation_raw,
        expected_request_sha256=expected_request_sha256, expected_parent_identity=expected_parent_identity,
        expected_nonce_sha256=expected_nonce_sha256, expected_sdk_argv=expected_sdk_argv, expected_roots=expected_roots)
    _parent_source_child_chain(expected_parent_identity, expected_sdk_argv)
    return {"observation_sha256": hashlib.sha256(observation_raw).hexdigest(),
        "request_sha256": expected_request_sha256, "nonce_sha256": expected_nonce_sha256,
        "parent_identity": dict(expected_parent_identity), "source_fds": [dict(row) for row in value["source_fds"]]}


def _parent_source_child_chain(expected_parent_identity, expected_sdk_argv):
    worker = _reviewed_parent_proc_stat(os.getpid())
    sdk = _reviewed_parent_proc_stat(worker["ppid"])
    need(sdk["ppid"] == expected_parent_identity["pid"], "worker is not original SDK CLI parent's child")
    need(_reviewed_parent_proc_executable(sdk["pid"]) == Path(expected_sdk_argv[0]).resolve(strict=True),
         "worker original SDK CLI executable differs")
    command = Path(f"/proc/{sdk['pid']}/cmdline").read_bytes()
    need(0 < len(command) <= 32768 and command.endswith(b"\x00")
         and command[:-1].split(b"\x00") == [os.fsencode(word) for word in expected_sdk_argv],
         "worker original SDK CLI command chain differs")


def _reviewed_parent_record_source_fd(custody, manager, held):
    _reviewed_parent_check_source_root_binding(custody.reviewed_source_root_binding,
        custody.reviewed_source_root_pin, (custody.d, custody.cwd))
    generator = getattr(manager, "gen", None); frame = getattr(generator, "gi_frame", None)
    source = Path(custody.policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"
    need(frame is not None and generator.gi_code.co_name == "source_lease"
         and generator.gi_code.co_filename == str(source) and frame.f_globals.get("__file__") == str(source),
         "original live Source42 generator required")
    need(_REVIEWED_PARENT_SOURCE_BINDINGS is not None
         and generator.gi_code is _REVIEWED_PARENT_SOURCE_BINDINGS[3][1]
         and frame.f_globals is vars(_REVIEWED_PARENT_SOURCE_BINDINGS[0]),
         "same original Source42 generator code and globals required")
    fd = frame.f_locals.get("source")
    need(type(fd) is int and fd >= 0 and frame.f_locals.get("root") == Path(held.canonical_root),
         "actual held Source42 descriptor required")
    info = os.fstat(fd)
    row = {"root": held.canonical_root, "lock_path": held.lock_path,
           "fd": fd, "device": info.st_dev, "inode": info.st_ino}
    _reviewed_parent_kernel_read_lock(os.getpid(), row)
    if not hasattr(custody, "reviewed_source_fds"):
        custody.reviewed_source_fds = []
    custody.reviewed_source_fds.append(row)


def _reviewed_parent_publish_source_observation(custody):
    plan, frozen = custody.context["plan"], custody.context["frozen"]
    need(plan["schema"] == _reviewed_parent_domain_v3(_reviewed_parent_version_v3(custody.policy))["worker_schema"] and plan["operation"] == custody.policy["operation"]
         and plan["parent_identity"] == proc_identity(os.getpid())
         and hashlib.sha256(bytes.fromhex(frozen["parent_nonce"])).hexdigest() == plan["parent_nonce_sha256"],
         "original predeclared parent Source operation differs")
    value = {"schema": REVIEWED_PARENT_SOURCE_SCHEMA, "request": custody.request_pin,
        "parent_identity": plan["parent_identity"], "nonce": frozen["parent_nonce"],
        "source_receipts": custody.receipts, "source_fds": custody.reviewed_source_fds, "sdk_argv": custody.argv}
    custody.reviewed_source_observation = new_json(Path(plan["parent_source_observation_path"]), value)
    custody.reviewed_source_raw = checked_descriptor(custody.reviewed_source_observation)
    _reviewed_parent_validate_source_observation(custody.reviewed_source_raw,
        expected_request_sha256=custody.request_pin["sha256"], expected_parent_identity=plan["parent_identity"],
        expected_nonce_sha256=plan["parent_nonce_sha256"], expected_sdk_argv=custody.argv,
        expected_roots=_reviewed_parent_check_source_root_binding(custody.reviewed_source_root_binding,
            custody.reviewed_source_root_pin, (custody.d, custody.cwd)))


def _reviewed_parent_check_worker_delegation(custody, *, require_worker_report=True):
    plan = custody.context["plan"]
    need(checked_descriptor(custody.reviewed_source_observation) == custody.reviewed_source_raw,
         "original parent Source observation changed")
    _reviewed_parent_validate_source_observation(custody.reviewed_source_raw,
        expected_request_sha256=custody.request_pin["sha256"], expected_parent_identity=plan["parent_identity"],
        expected_nonce_sha256=plan["parent_nonce_sha256"], expected_sdk_argv=custody.argv,
        expected_roots=_reviewed_parent_check_source_root_binding(custody.reviewed_source_root_binding,
            custody.reviewed_source_root_pin, (custody.d, custody.cwd)))
    if not require_worker_report:
        return
    report = json_object(ordinary_bytes(Path(plan["attempt_directory"]) / "exports/trusted-validation.json"))
    need(report["parent_source_observation"] == custody.reviewed_source_observation
         and report["parent_nonce_sha256"] == plan["parent_nonce_sha256"]
         and report["source_custody_mode"] == "delegated-original-parent-read/v1"
         and report["source_READ_closed"] is False and report["source_READ_admission_complete"] is False,
         "worker cannot replace original parent Source closure")


def _reviewed_parent_packet_binding_core_v3(operation, binding, durability_files, *, attempt_directory, reviewed_version):
    return _reviewed_parent_packet_binding(operation, binding, durability_files,
        attempt_directory=attempt_directory, reviewed_version=reviewed_version)

def _reviewed_parent_packet_binding_v1(operation, binding, durability_files, *, attempt_directory):
    return _reviewed_parent_packet_binding_core_v3(operation, binding, durability_files, attempt_directory=attempt_directory, reviewed_version=2)

def _reviewed_parent_packet_binding_v3(operation, binding, durability_files, *, attempt_directory):
    return _reviewed_parent_packet_binding_core_v3(operation, binding, durability_files, attempt_directory=attempt_directory, reviewed_version=3)


class VerifiedReviewedPreservationParentSuccess:
    """Opaque original admitted-parent engineering custody, never legal authority."""
    __slots__ = ("__weakref__",)
    __eq__ = object.__eq__
    __hash__ = object.__hash__
    def __new__(cls, *args, **kwargs):
        raise TypeError("reviewed parent success is issued only by its original successful wait")
    def __init_subclass__(cls, **kwargs):
        raise TypeError("reviewed parent success cannot be subclassed")
    def __reduce_ex__(self, protocol):
        raise TypeError("reviewed parent success cannot be copied or serialized")


def _reviewed_parent_function_state(function):
    return (function, function.__code__, function.__defaults__, tuple(sorted((function.__kwdefaults__ or {}).items())))


def _reviewed_parent_function_matches(function, state):
    return (type(function) is types.FunctionType and function is state[0] and function.__code__ is state[1]
        and function.__defaults__ is state[2] and set(function.__kwdefaults__ or {}) == {key for key, _ in state[3]}
        and all((function.__kwdefaults__ or {})[key] is value for key, value in state[3]))


def _reviewed_parent_descriptor_function(value):
    return value.__func__ if type(value) in (staticmethod, classmethod) else value.fget if type(value) is property else value


def _reviewed_parent_package_guard(package):
    """Accept only the source-pinned maintained lazy A root module class.

    Read the namespace intrinsically so this check never resolves an optional
    lazy public attribute. This observes code provenance, granting no authority.
    """
    global _REVIEWED_PARENT_PACKAGE_BINDINGS

    def require(condition, label):
        if not condition:
            raise ValueError("reviewed parent callable bindings changed: canonical source package " + label)

    def matches(function, state):
        return (type(function) is types.FunctionType and function is state[0]
            and function.__code__ is state[1] and function.__defaults__ is state[2]
            and set(function.__kwdefaults__ or {}) == {name for name, _ in state[3]}
            and all((function.__kwdefaults__ or {})[name] is value for name, value in state[3]))

    try:
        namespace = types.ModuleType.__getattribute__(package, "__dict__")
    except (TypeError, AttributeError) as error:
        raise ValueError("reviewed parent callable bindings changed: canonical source package module") from error
    filename = Path(__file__).resolve().parents[3] / "ipfs_accelerate_py/__init__.py"
    require(type(namespace) is dict and namespace.get("__name__") == "ipfs_accelerate_py"
            and sys.modules.get("ipfs_accelerate_py") is package
            and type(namespace.get("__file__")) is str
            and Path(namespace["__file__"]).resolve() == filename,
            "origin")
    spec = namespace.get("__spec__")
    require(type(spec) is importlib.machinery.ModuleSpec and spec.origin == str(filename)
            and namespace.get("ModuleType") is types.ModuleType,
            "specification")
    raw = ordinary_bytes(filename, 4 * 1024**2)
    require(len(raw) == _REVIEWED_PARENT_ROOT_INITIALIZER_BYTES
            and hashlib.sha256(raw).hexdigest() == _REVIEWED_PARENT_ROOT_INITIALIZER_SHA256,
            "initializer pin")
    owner = namespace.get("_IPFSAccelerateModule")
    require(type(owner) is type and type(package) is owner
            and owner.__module__ == "ipfs_accelerate_py" and owner.__qualname__ == "_IPFSAccelerateModule"
            and owner.__bases__ == (types.ModuleType,),
            "class identity")
    class_namespace = type.__getattribute__(owner, "__dict__")
    if _REVIEWED_PARENT_PACKAGE_BINDINGS is None:
        tree = ast.parse(raw, str(filename))
        declarations = [node for node in tree.body if type(node) is ast.ClassDef
                        and node.name == "_IPFSAccelerateModule"]
        require(len(declarations) == 1 and len(declarations[0].bases) == 1
                and type(declarations[0].bases[0]) is ast.Name
                and declarations[0].bases[0].id == "ModuleType" and not declarations[0].keywords,
                "class declaration")
        methods = [node for node in declarations[0].body if type(node) is ast.FunctionDef]
        names = {node.name for node in methods}
        require(names == {"__getattribute__", "__setattr__", "__delattr__", "__dir__"}
                and len(methods) == 4 and all(not node.decorator_list for node in methods),
                "method declaration")
        allowed = names | {"__module__", "__doc__", "__dict__", "__weakref__", "__qualname__"}
        require(set(class_namespace) <= allowed, "undeclared class namespace")
        for name in ("__dict__", "__weakref__"):
            if name in class_namespace:
                descriptor = class_namespace[name]
                require(type(descriptor) in (types.GetSetDescriptorType, types.MemberDescriptorType)
                        and descriptor.__objclass__ is owner,
                        "generated class descriptor")
        codes = {}

        def walk(code):
            codes[code.co_qualname] = code
            for item in code.co_consts:
                if type(item) is types.CodeType:
                    walk(item)

        walk(_ORIGINAL_COMPILE(raw, str(filename), "exec", dont_inherit=True, optimize=sys.flags.optimize))
        states = []
        for name in sorted(names):
            function = class_namespace.get(name)
            require(type(function) is types.FunctionType and function.__globals__ is namespace
                    and function.__module__ == "ipfs_accelerate_py"
                    and function.__qualname__ == "_IPFSAccelerateModule." + name
                    and function.__code__.co_filename == str(filename)
                    and function.__code__ == codes.get("_IPFSAccelerateModule." + name),
                    "method code origin " + name)
            states.append((name, (function, function.__code__, function.__defaults__,
                tuple(sorted((function.__kwdefaults__ or {}).items())))))
        _REVIEWED_PARENT_PACKAGE_BINDINGS = (package, namespace, owner, type(owner), owner.__bases__,
            frozenset(class_namespace), tuple(states))
    original, original_namespace, original_class, metaclass, bases, keys, states = _REVIEWED_PARENT_PACKAGE_BINDINGS
    require(package is original and namespace is original_namespace and owner is original_class
            and type(owner) is metaclass and owner.__bases__ is bases
            and frozenset(class_namespace) == keys,
            "runtime class binding")
    for name, state in states:
        function = class_namespace.get(name)
        require(matches(function, state) and function.__globals__ is namespace,
                "runtime method binding " + name)


def _reviewed_parent_guard():
    """Cooperative process integrity fence; no hostile Python memory claim."""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def matches(function, state):
        return type(function) is types.FunctionType and function is state[0] and (function.__code__ is state[1]) and (function.__defaults__ is state[2]) and (set(function.__kwdefaults__ or {}) == {key for key, _ in state[3]}) and all(((function.__kwdefaults__ or {})[key] is value for key, value in state[3]))
    name = 'ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners'
    module = sys.modules.get(name)
    require(__name__ == name and type(module) is types.ModuleType and (vars(module) is globals()) and (Path(__file__).resolve().name == 'repair_candidate_owners.py') and (module.__spec__ is not None) and (module.__spec__.origin == str(Path(__file__).resolve())) and (_ORIGINAL_COMPILE is _REVIEWED_PARENT_COMPILE) and (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == _REVIEWED_PARENT_SELF_PIN), 'reviewed parent callable bindings changed: canonical owner')
    package = sys.modules.get('ipfs_accelerate_py')
    for key, state in _REVIEWED_PARENT_FUNCTIONS:
        require(matches(globals().get(key), state), 'reviewed parent callable bindings changed: ' + key)
    _reviewed_parent_package_guard(package)
    for key, owner, metaclass, bases, keys, methods in _REVIEWED_PARENT_CLASSES:
        require(globals().get(key) is owner and type(owner) is metaclass and (owner.__bases__ is bases) and (frozenset(vars(owner)) == keys), 'reviewed parent callable bindings changed: class namespace')
        for method, state in methods:
            require(matches(_reviewed_parent_descriptor_function(vars(owner).get(method)), state), 'reviewed parent callable bindings changed: class method')
    require(isinstance(_ORIGINAL_POPEN, type) and subprocess.Popen is _ORIGINAL_POPEN and type(subprocess.Popen) is _REVIEWED_PARENT_POPEN_META and (subprocess.Popen.__bases__ is _REVIEWED_PARENT_POPEN_BASES) and (frozenset(vars(subprocess.Popen)) == _REVIEWED_PARENT_POPEN_KEYS), 'reviewed parent callable bindings changed: original Popen')
    for key, state in _REVIEWED_PARENT_POPEN_FUNCTIONS:
        require(matches(_reviewed_parent_descriptor_function(vars(subprocess.Popen).get(key)), state), 'reviewed parent callable bindings changed: original Popen method')
    if _REVIEWED_PARENT_ENGINE_BINDINGS is not None:
        module, pin, function, classes = _REVIEWED_PARENT_ENGINE_BINDINGS
        require(sys.modules.get(module.__name__) is module and descriptor(Path(module.__file__)) == pin and matches(module.run_command, function), 'reviewed parent callable bindings changed: original engine')
        for key, owner, metaclass, bases, keys, methods in classes:
            require(vars(module).get(key) is owner and type(owner) is metaclass and (owner.__bases__ is bases) and (frozenset(vars(owner)) == keys), 'reviewed parent callable bindings changed: engine class')
            for method, state in methods:
                require(matches(_reviewed_parent_descriptor_function(vars(owner).get(method)), state), 'reviewed parent callable bindings changed: engine class method')
    if _REVIEWED_PARENT_SOURCE_BINDINGS is not None:
        module, pin, wrapper, wrapped = _REVIEWED_PARENT_SOURCE_BINDINGS
        require(sys.modules.get(module.__name__) is module and descriptor(Path(module.__file__)) == pin and matches(module.source_lease, wrapper) and (getattr(module.source_lease, '__wrapped__', None) is wrapped[0]) and matches(wrapped[0], wrapped), 'reviewed parent callable bindings changed: original Source42')


def _reviewed_parent_source_guard(module, pin):
    global _REVIEWED_PARENT_SOURCE_BINDINGS
    _reviewed_parent_guard()
    need(type(module) is types.ModuleType and module.__name__ ==
         "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease"
         and sys.modules.get(module.__name__) is module, "canonical original Source42 module required")
    raw = checked_descriptor(pin); codes = {}
    def walk(code):
        codes[code.co_qualname] = code
        for item in code.co_consts:
            if type(item) is types.CodeType:
                walk(item)
    walk(_ORIGINAL_COMPILE(raw, pin["path"], "exec", dont_inherit=True, optimize=sys.flags.optimize))
    wrapper = module.source_lease; wrapped = getattr(wrapper, "__wrapped__", None)
    need(type(wrapper) is types.FunctionType and type(wrapped) is types.FunctionType
         and wrapped.__globals__ is vars(module) and wrapped.__code__ == codes.get("source_lease")
         and wrapper.__code__ == contextlib.contextmanager(wrapped).__code__
         and wrapper.__closure__ is not None and len(wrapper.__closure__) == 1
         and wrapper.__closure__[0].cell_contents is wrapped,
         "reviewed parent callable bindings changed: original Source42 code origin")
    if _REVIEWED_PARENT_SOURCE_BINDINGS is None:
        _REVIEWED_PARENT_SOURCE_BINDINGS = (module, dict(pin), _reviewed_parent_function_state(wrapper),
                                           _reviewed_parent_function_state(wrapped))
    _reviewed_parent_guard()


def _reviewed_parent_engine_guard(module, source):
    global _REVIEWED_PARENT_ENGINE_BINDINGS
    _reviewed_parent_guard()
    need(type(module) is types.ModuleType and module.__name__ == "ipfs_accelerate_py.agent_supervisor.todo_daemon.engine"
         and sys.modules.get(module.__name__) is module, "canonical original parent engine required")
    raw = checked_descriptor(source)
    codes = {}
    def walk(code):
        codes[code.co_qualname] = code
        for value in code.co_consts:
            if type(value) is types.CodeType:
                walk(value)
    walk(_ORIGINAL_COMPILE(raw, source["path"], "exec", dont_inherit=True, optimize=sys.flags.optimize))
    function = module.run_command
    need(type(function) is types.FunctionType and function.__globals__ is vars(module)
         and function.__code__ == codes.get("run_command"),
         "reviewed parent callable bindings changed: original engine source origin")
    if _REVIEWED_PARENT_ENGINE_BINDINGS is None:
        classes = []
        tree = ast.parse(raw, source["path"])
        for name in ("CommandResult", "CommandExecutionObservation"):
            owner = vars(module)[name]
            node = next(node for node in tree.body if type(node) is ast.ClassDef and node.name == name)
            declared = {node.name for node in node.body if type(node) is ast.FunctionDef}
            need(type(owner) is type and owner.__bases__ == (object,)
                 and owner.__getattribute__ is object.__getattribute__ and owner.__new__ is object.__new__
                 and not hasattr(owner, "__getattr__")
                 and not (set(vars(owner)) & {"__getattribute__", "__getattr__", "__call__"} - declared),
                 "reviewed parent callable bindings changed: engine lookup hooks")
            for method in declared:
                target = _reviewed_parent_descriptor_function(vars(owner).get(method))
                need(type(target) is types.FunctionType and target.__globals__ is vars(module)
                     and target.__code__ == codes.get(name + "." + method),
                     "reviewed parent callable bindings changed: engine method source origin")
            methods = tuple((key, _reviewed_parent_function_state(_reviewed_parent_descriptor_function(value)))
                for key, value in vars(owner).items() if type(_reviewed_parent_descriptor_function(value)) is types.FunctionType)
            classes.append((name, owner, type(owner), owner.__bases__, frozenset(vars(owner)), methods))
        classes = tuple(classes)
        _REVIEWED_PARENT_ENGINE_BINDINGS = (module, dict(source), _reviewed_parent_function_state(function), classes)
    _reviewed_parent_guard()


def _reviewed_parent_capability_entry(handle):
    if not (type(handle) is _REVIEWED_PARENT_HANDLE_CLASS
            and VerifiedReviewedPreservationParentSuccess is _REVIEWED_PARENT_HANDLE_CLASS
            and vars(_REVIEWED_PARENT_HANDLE_CLASS).get("__eq__") is object.__eq__
            and vars(_REVIEWED_PARENT_HANDLE_CLASS).get("__hash__") is object.__hash__):
        raise ValueError("reviewed parent handle was not issued by this process and native thread")
    entry = _REVIEWED_PARENT_CAPABILITIES.get(handle)
    if not (entry is not None and entry[0] == os.getpid() and entry[1] == threading.get_native_id()
            and entry[2]() is threading.current_thread() and entry[3]() is handle):
        raise ValueError("reviewed parent handle was not issued by this process and native thread")
    return entry


def _reviewed_parent_issue(custody, completed, witness):
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    _reviewed_parent_guard()
    frame = sys._getframe(1)
    need(frame.f_code is original_runner_wait.__code__ and frame.f_globals is globals()
         and frame.f_locals.get("custody") is custody and frame.f_locals.get("completed") is completed
         and frame.f_locals.get("witness") is witness and frame.f_locals.get("capture_parent_success") is True,
         "only original parent wait may issue admitted success")
    custody.clock()
    _reviewed_parent_guard()
    need(type(custody) is _V5ValidationSourceCustody and custody.closed
         and custody.context["plan"]["schema"] == _reviewed_parent_domain_v3(_reviewed_parent_version_v3(custody.policy))["worker_schema"],
         "actual original reviewed parent custody required")
    custody.checked_execution(completed, witness)
    parent = witness["parent_source_custody"]
    need(parent["closure_kind"] == "existing_verified_success" and parent["source_closed"] is True
         and parent["native_closed"] is True and parent["storage_status"] == "released"
         and completed.returncode == 0 and custody.process.returncode == 0,
         "complete original reviewed parent success required")
    verify_v5_parent_source_closeout(custody.policy, custody.request_pin, Path(custody.context["invocation"]),
        witness["parent_source_closeout"], parent["verified_checks"])
    custody.clock()
    summary = parent["verified_checks"]
    binding = json_object(checked_descriptor(summary["native_binding"]))
    durability = json_object(checked_descriptor(summary["durability"]))
    inventory = _reviewed_parent_binding_v3(custody.policy, custody.policy["operation"], binding, durability["files"],
        attempt_directory=Path(custody.context["plan"]["attempt_directory"]))
    need([list(row) for row in inventory] == summary["packet_inventory"], "original parent inventory changed after Source close")
    custody.clock()
    evidence = canonical({"operation": custody.policy["operation"], "binding": binding,
        "inventory": [list(row) for row in inventory], "request": custody.request_pin,
        "plan": descriptor(Path(custody.context["invocation"]) / "validation-plan.json"),
        "wait": custody.wait_ref, "closeout": witness["parent_source_closeout"],
        "durability": summary["durability"], "datasets_root": str(custody.d),
        "attempt_directory": custody.context["plan"]["attempt_directory"],
        "source_pins": custody.context["plan"]["source_pins"], "summary": summary,
        **({"reviewed_version": _reviewed_parent_version_v3(custody.policy),
             "expected_packet_paths": custody.policy["operation"]["expected_packet_paths"]}
           if _reviewed_parent_version_v3(custody.policy) in (3, 4) else {})})
    _reviewed_parent_guard()
    custody.clock()
    joined_binding = None
    joined_path = str(Path(custody.d) / "ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py")
    if (_reviewed_parent_version(custody.policy) in (3, 4)
            and any(pin["path"] == joined_path for pin in custody.policy["source_descriptors"])):
        joined_binding = frame.f_locals.get("joined_opening_binding")
        need(joined_binding is not None,
             "joined owner population must be authenticated before original dispatch")
        _release, current_binding = _parent_release_owner_binding_v3(json_object(evidence))
        need(_same_joined_owner_binding_v1(current_binding, joined_binding),
             "joined owner changed between opening dispatch and original parent issuance")
    handle = object.__new__(_REVIEWED_PARENT_HANDLE_CLASS)
    entry = (os.getpid(), threading.get_native_id(),
        weakref.ref(threading.current_thread()), weakref.ref(handle), evidence)
    _REVIEWED_PARENT_CAPABILITIES[handle] = entry if joined_binding is None else (*entry, joined_binding)
    try:
        custody.clock()
        _reviewed_parent_guard()
        if joined_binding is not None:
            _release, current_binding = _parent_release_owner_binding_v3(json_object(evidence))
            need(_same_joined_owner_binding_v1(current_binding, joined_binding),
                 "joined owner changed during original parent issuance")
        custody.clock()
    except BaseException:
        _REVIEWED_PARENT_CAPABILITIES.pop(handle, None)
        raise
    return handle


def _parent_native_owner(evidence, *, reviewed_version=2):
    native_version = _reviewed_parent_domain_v3(reviewed_version)["native_version"]
    name = "ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_native_v2"
    root = Path(evidence["datasets_root"])
    path = root / "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py"
    pins = evidence["source_pins"]
    need(type(pins) is dict and all(type(key) is str and type(value) is str for key, value in pins.items())
         and type(pins.get(str(path))) is str
         and _SHA.fullmatch(pins[str(path)]), "independent original Native Source pin required")
    raw = ordinary_bytes(path, 4 * 1024**2)
    need(hashlib.sha256(raw).hexdigest() == pins[str(path)], "original Native Source differs")
    native = importlib.import_module(name)
    _parent_native_bootstrap_v1(native, raw, path, pins, reviewed_version=native_version)
    owners = native._owners()
    _parent_native_owner_tuple_v1(owners, root, pins)
    _parent_native_bootstrap_v1(native, raw, path, pins, reviewed_version=native_version)
    # Invoke the actual Source-authenticated existing private binder only after
    # its code, closures, aliases and complete native snapshot schemas pass.
    native._live_success_guard(owners, reviewed_version=native_version)
    _parent_native_bootstrap_v1(native, raw, path, pins, reviewed_version=native_version)
    need(hashlib.sha256(ordinary_bytes(path, 4 * 1024**2)).hexdigest() == pins[str(path)],
         "original Native Source changed during parent replay")
    return native


def _load_parent_verified_reviewed_preservation_core_v3(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256, toolchain, reviewed_version, **kwargs):
    return _load_parent_verified_reviewed_preservation(
        handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_preservation_sha256=expected_preservation_sha256,
        expected_graph_sha256=expected_graph_sha256, expected_admission_sha256=expected_admission_sha256,
        toolchain=toolchain, reviewed_version=reviewed_version, **kwargs)

def load_parent_verified_reviewed_preservation_v1(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256, toolchain, **kwargs):
    return _load_parent_verified_reviewed_preservation_core_v3(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_preservation_sha256=expected_preservation_sha256,
        expected_graph_sha256=expected_graph_sha256, expected_admission_sha256=expected_admission_sha256,
        toolchain=toolchain, reviewed_version=2, **kwargs)

def load_parent_verified_reviewed_preservation_v3(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256, toolchain, **kwargs):
    return _load_parent_verified_reviewed_preservation_core_v3(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_preservation_sha256=expected_preservation_sha256,
        expected_graph_sha256=expected_graph_sha256, expected_admission_sha256=expected_admission_sha256,
        toolchain=toolchain, reviewed_version=3, **kwargs)


def _run_reviewed_preservation_parent_core_v3(policy_path, expected_policy_sha256, *, invocation_directory, reviewed_version):
    """Run the fixed scientific worker with existing original V5 resource owners.

    No changed repair candidate, adoption, task permission or new runner is
    created. The returned handle stays in this original parent process/thread.
    """
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    _reviewed_parent_guard()
    domain = _reviewed_parent_domain_v3(reviewed_version)
    policy = _checked_reviewed_preservation_parent_policy_core_v3(policy_path, expected_policy_sha256, reviewed_version=reviewed_version)
    v, operation = policy["validation"], policy["operation"]
    invocation = Path(invocation_directory)
    need(invocation.is_absolute() and ".." not in invocation.parts
         and not any(p.is_symlink() for p in (invocation, *invocation.parents)) and not invocation.exists(),
         "fresh original reviewed parent invocation required")
    invocation.mkdir(mode=0o700)
    attempt = Path(v["attempt_root"]) / operation["operation_id"]
    need(not attempt.exists() and not attempt.is_symlink(), "fresh named original native attempt required")
    attempt.mkdir(mode=0o700)
    nonce = os.urandom(32).hex(); parent_identity = proc_identity(os.getpid())
    policy_pin = descriptor(Path(policy_path))
    d, a = Path(policy["datasets_root"]), Path(policy["source_root"])
    source_pins, controls = _reviewed_parent_plan_sources(policy, policy_pin)
    plan = {"schema": domain["worker_schema"], "policy": policy_pin, "operation": operation,
        "native_inputs": operation["native_inputs"],
        "candidate_root": str(a), "collection_root": str(a), "attempt_directory": str(attempt),
        "source_pins": source_pins, "original_public_control_pins": controls,
        "selected_checks": [operation["nodeid"]], "expected_cases": 1,
        "required_candidate_bodies": [str(d / "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py"),
                                      operation["nodeid"].split("::", 1)[0]],
        "ordinary_scratch_profile": dict(_V5_ORDINARY_PROFILE),
        "normal_namespace_profiles": v["normal_namespace_profiles"],
        "parent_source_observation_path": str(invocation / "parent-source-observation.json"),
        "parent_nonce_sha256": hashlib.sha256(bytes.fromhex(nonce)).hexdigest(), "parent_identity": parent_identity,
        "execution_request_path": str(invocation / "execution-request.json")}
    plan_pin = new_json(invocation / "validation-plan.json", plan)
    worker = a / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"
    execution = {"schema": "owned-validation-request/v5", "argv": [policy["python"], "-I", "-B", str(worker),
        "--plan", plan_pin["path"], "--plan-sha256", plan_pin["sha256"]], "cwd": str(a),
        "selected_checks": plan["selected_checks"], "source_sha256": {**controls, plan_pin["path"]: plan_pin["sha256"]},
        **{key: v[key] for key in ("timeout_seconds", "storage_bytes", "memory_mb", "cpu_slots", "child_process_slots",
            "scheduler_state_path", "expected_scheduler_policy_sha256", "roots")},
        "attempt_directory": str(attempt), "outputs": {name: str(attempt / name) for name in ("pytest", "tmp", "cache", "db", "exports", "logs")},
        "declared_external_writable_paths": [], "process_scope": "sampled_birth_bound_descendants",
        "ordinary_scratch_profile": dict(_V5_ORDINARY_PROFILE)}
    _require_v5_lifecycle_source_pins(d, execution["source_sha256"])
    need(len(execution["source_sha256"]) <= 256, "unchanged original SDK256 source cap required")
    request_pin = new_json(invocation / "execution-request.json", execution)
    argv = [policy["python"], "-I", "-B", str(d / "scripts/ops/legal_ir/run_owned_validation.py"),
        "--request", request_pin["path"], "--expected-request-sha256", request_pin["sha256"],
        "--resource-ledger", v["resource_ledger"]]
    completed, witness, handle = original_runner_wait(policy, argv, a, 420,
        validation_context={"plan": plan, "request_pin": request_pin, "invocation": invocation,
            "frozen": {"schema": "reviewed-preservation-original-parent-context/v1", "parent_nonce": nonce}},
        capture_parent_success=True)
    need(completed.returncode == 0, "original reviewed validation did not succeed")
    return handle

def run_reviewed_preservation_parent_v1(policy_path, expected_policy_sha256, *, invocation_directory):
    return _run_reviewed_preservation_parent_core_v3(policy_path, expected_policy_sha256, invocation_directory=invocation_directory, reviewed_version=2)

def run_reviewed_preservation_parent_v3(policy_path, expected_policy_sha256, *, invocation_directory):
    return _run_reviewed_preservation_parent_core_v3(policy_path, expected_policy_sha256, invocation_directory=invocation_directory, reviewed_version=3)


SOURCE_QUALIFICATION_SCHEMA = "agent-supervisor/immutable-source-qualification-receipt@1"
_SOURCE_QUALIFICATION_PROFILE = "agent-supervisor/immutable-source-qualification@1"
_SOURCE_SNAPSHOT_SCHEMA = "agent-supervisor/immutable-source-snapshot@1"
_SOURCE_QUALIFICATION_ROLES = ("ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py",
    "ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py")


def checked_source_qualification(policy, control=None):
    """Authenticate unchanged sealed control bodies; confer no repair authority."""
    if "source_qualification" not in policy["validation"]:
        return None
    profile = policy["validation"]["source_qualification"]
    need(_v5_ordinary_validation(policy) and not {"validation_ledger_busy_retry",
         "validation_admission_retry", "external_fixture_profile"} & set(policy["validation"]),
         "immutable qualification cannot borrow repair retry or fixtures")
    fields = {"schema", "snapshot", "required_source_bodies"}
    delegated = type(profile) is dict and profile.get("schema") == IMMUTABLE_PARENT_PROFILE_SCHEMA
    if delegated:
        fields.add("source_custody_mode")
    need(type(profile) is dict and set(profile) == fields
         and (profile["schema"] == _SOURCE_QUALIFICATION_PROFILE or delegated)
         and (not delegated or profile["source_custody_mode"] == IMMUTABLE_PARENT_MODE),
         "closed immutable source qualification required")
    authenticated = checked_control_release(policy, current_owner=False)
    if control is not None:
        need(all(control.get(key) == authenticated[key] for key in ("root", "manifest", "pins")),
             "immutable source control authentication differs")
    control = authenticated
    snapshot_ref = profile["snapshot"]; snapshot = json_object(checked_control_descriptor(snapshot_ref))
    need(not Path(snapshot_ref["path"]).lstat().st_mode & 0o222
         and type(snapshot) is dict and set(snapshot) == {"schema", "binding", "source_root", "manifest", "source_commitment"}
         and snapshot["schema"] == _SOURCE_SNAPSHOT_SCHEMA, "sealed immutable source snapshot required")
    binding = snapshot["binding"]; root = control["root"]
    need(type(binding) is dict and set(binding) == {"namespace", "board_id", "attempt_id"}
         and binding["namespace"] == "ptr_lane_0" and binding["board_id"] == "legal-ir"
         and type(binding["attempt_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", binding["attempt_id"])
         and type(snapshot["source_root"]) is str and snapshot["source_root"] == str(root),
         "immutable source root or binding differs")
    manifest = snapshot["manifest"]
    need(type(manifest) is dict and 0 < len(manifest) <= 4096, "finite immutable source manifest required")
    pins = {}
    for relative, row in manifest.items():
        need(type(relative) is str and relative and str(Path(relative)) == relative
             and not Path(relative).is_absolute() and ".." not in Path(relative).parts
             and type(row) is dict and set(row) == {"bytes", "sha256", "executable"}
             and type(row["bytes"]) is int and type(row["executable"]) is bool,
             "typed relative immutable source row required")
        path = root / relative
        checked_control_descriptor(dict(path=str(path), bytes=row["bytes"], sha256=row["sha256"]))
        need(not path.lstat().st_mode & 0o222 and bool(path.lstat().st_mode & 0o111) == row["executable"],
             "immutable source seal or executable mode changed")
        pins[str(path)] = row["sha256"]
    need(pins == control["pins"] and type(snapshot["source_commitment"]) is str
         and snapshot["source_commitment"] == digest({"source_root": str(root), "manifest": manifest}),
         "immutable snapshot must equal the complete authenticated control population")
    required = profile["required_source_bodies"]
    need(type(required) is list and 0 < len(required) <= 128
         and all(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"} and type(pin["path"]) is str for pin in required)
         and [pin["path"] for pin in required] == sorted({pin["path"] for pin in required}),
         "sorted unique immutable source subjects required")
    for pin in required:
        checked_control_descriptor(pin); path = Path(pin["path"])
        need(path.suffix == ".py" and root in path.parents and pin["path"] in pins and pins[pin["path"]] == pin["sha256"],
             "immutable subject is outside the exact control snapshot")
    paths = [pin["path"] for pin in required]
    need(all(str(root / name) in paths for name in _SOURCE_QUALIFICATION_ROLES),
         "immutable qualification requires exact Main and health source bodies")
    return dict(profile=profile, snapshot=snapshot, snapshot_ref=snapshot_ref, root=root,
                required_source_bodies=paths)


def _source_qualification_plan(policy, plan, invocation):
    context = checked_source_qualification(policy)
    if context is None:
        need("source_root_bindings" not in plan and "source_qualification" not in plan and "required_source_bodies" not in plan,
             "repair plan cannot carry immutable qualification fields")
        return None
    need(_immutable_source_parent_enabled(policy) or not (_IMMUTABLE_PARENT_FIELDS & set(plan)),
         "legacy immutable plan cannot carry delegated runtime fields")
    need(canonical(plan.get("source_qualification")) == canonical(context["profile"])
         and plan.get("required_source_bodies") == context["required_source_bodies"]
         and plan.get("required_candidate_bodies") == [] and plan["candidate_root"] == str(context["root"])
         and plan["freeze"] == context["snapshot_ref"]
         and plan["source_commitment"] == context["snapshot"]["source_commitment"]
         and json_object(checked_descriptor(plan["policy"])) == policy,
         "immutable source qualification plan differs")
    expected = validation_plan(policy, plan["policy"], context["snapshot"],
        context["snapshot_ref"], context["root"], invocation,
        **({"source_parent_context": _immutable_source_parent_plan_fields(plan, invocation)}
           if _immutable_source_parent_enabled(policy) else {}))
    need(canonical(plan) == canonical(expected), "complete immutable source qualification plan differs")
    return context


def _source_qualification_report(context, plan, report):
    if context is None:
        need("source_root_bindings" not in report and "source_qualification" not in report and "required_source_bodies" not in report,
             "repair report cannot carry immutable qualification fields")
        return
    need(context["profile"]["schema"] == IMMUTABLE_PARENT_PROFILE_SCHEMA or
         not ({"parent_identity", "parent_nonce_sha256", "parent_source_observation", "source_root_bindings", "source_custody_mode"} & set(report)),
         "legacy immutable report cannot carry delegated parent fields")
    need(canonical(report.get("source_qualification")) == canonical(context["profile"])
         and report.get("required_source_bodies") == context["required_source_bodies"]
         and report.get("freeze") == context["snapshot_ref"]
         and report.get("source_commitment") == context["snapshot"]["source_commitment"]
         and report.get("semantic_negative") is False and report.get("negative_profile") is None,
         "immutable source qualification report differs")
    need(all(report["executed_source_buffers"].get(path) == plan["source_pins"][path]
             and report["raw_compiled_sources"].get(path) == plan["source_pins"][path]
             for path in context["required_source_bodies"]),
         "immutable source body lacks original raw compilation and execution")


def _source_qualification_failed_custody(policy, plan, request_pin, invocation, terminal, wait):
    """Reuse original failure provenance without producing repair diagnostics."""
    context = _source_qualification_plan(policy, plan, invocation)
    need(context is not None, "explicit source qualification failure required")
    need(type(wait["returncode"]) is int and wait["returncode"] != 0 and wait["request"] == request_pin
         and wait["original_process_owner_returned"] is True and wait["actual_Popen_returned"] is True
         and wait["actual_process_waited"] is True and wait["actual_process_returncode"] == wait["returncode"]
         and json_object(wait["stdout"].encode()) == terminal, "original failed process wait is unknown")
    checked_descriptor(wait["engine_source"])
    need(wait["engine_source"] in policy["source_descriptors"], "original failed process source is not pinned")
    need(terminal["schema"] == "owned-validation-failure-terminal/v1" and terminal["completed"] is False
         and terminal["request_sha256"] == request_pin["sha256"]
         and terminal["process_cleanup_error"] is None and terminal["owner_exit_error"] is None,
         "original post-exit validation failure is incomplete")
    reservation = terminal["reservation"]; record = reservation["record"]; native = reservation["resource_lease"]
    identity = {"pid": record["owner_pid"], "birth": record["owner_birth"]}
    need(record["validation_request_sha256"] == request_pin["sha256"] and identity == wait["runner_identity"]
         and native["lease_id"] == record["native_lease_id"] and native["owner_pid"] == identity["pid"],
         "original failed request/native/PID/birth binding differs")
    lifecycle = terminal["source_custody_lifecycle"]
    need(lifecycle["request_sha256"] == request_pin["sha256"] and lifecycle["owner"] == identity
         and lifecycle["reservation_id"] == record["reservation_id"]
         and lifecycle["events"]["owner_exit_handling_completed"] is True
         and lifecycle["source_guards"]["errors"] == []
         and all(lifecycle["source_guards"][key] is True for key in ("request_unchanged", "source_pins_unchanged", "owner_identity_unchanged")),
         "original failed owner exit/source control flow is unknown")
    producer = lifecycle["producer"]
    need(type(producer) is dict and set(producer) == {"path", "sha256"}, "typed original failed producer required")
    producer_pin = next(pin for pin in policy["source_descriptors"] if pin["path"] == producer["path"])
    checked_descriptor(producer_pin)
    need(producer_pin["sha256"] == producer["sha256"], "original failed producer identity is unpinned")
    process = reservation["process_observation"]
    report_path = Path(plan["attempt_directory"]) / "exports/trusted-validation.json"
    need(not report_path.lstat().st_mode & 0o222, "failed immutable qualification report must be sealed")
    report = json_object(ordinary_bytes(report_path))
    need(reservation["cleanup_error"] is None and native["released"] is True
         and reservation["status"] == record["status"] == "retained"
         and process["live_processes"] == 0 and process["out_of_group_live_identities"] == []
         and not process["tracking_error"] and report["worker_identity"] == record["child"]
         and report["schema"] == (IMMUTABLE_PARENT_WORKER_SCHEMA if _immutable_source_parent_enabled(policy) else WORK_SCHEMA)
         and report["passed"] is False
         and report["semantic_negative"] is False and report["negative_profile"] is None
         and report["startup"] == {"python": policy["python"], "isolated": 1, "no_bytecode": True}
         and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"]
         and report["source_READ_closed"] is (False if _immutable_source_parent_enabled(policy) else True)
         and report["cleanup_errors"] == []
         and report["freeze"] == context["snapshot_ref"]
         and report["source_commitment"] == context["snapshot"]["source_commitment"]
         and canonical(report.get("source_qualification")) == canonical(context["profile"])
         and report.get("required_source_bodies") == context["required_source_bodies"]
         and report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"]),
         "original source qualification failure custody remains unknown")
    if _immutable_source_parent_enabled(policy):
        _immutable_source_parent_report_binding(policy, plan, request_pin, report, invocation)
    return {"source_closed": True, "native_closed": True, "processes_closed": True,
            "storage_closed": False, "semantic_negative": False, "negative_profile": None}


def _source_qualification_arguments(args, context):
    need(context is not None and Path(context["snapshot_ref"]["path"]) == args.freeze_path
         and args.candidate_root == context["root"]
         and args.source_commitment == context["snapshot"]["source_commitment"],
         "immutable qualification original arguments differ")
    return context["snapshot"]


def verify_source_qualification(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    context = checked_source_qualification(policy); _source_qualification_arguments(args, context)
    need(args.receipt_path.parent == args.invocation_root and args.freeze_path.parent == args.invocation_root.parent,
         "immutable qualification receipt or invocation binding differs")
    receipt = json_object(ordinary_bytes(args.receipt_path))
    need(type(receipt) is dict and set(receipt) == {"schema", "snapshot", "source_qualification", "source_commitment",
         "passed", "source_closed", "native_closed", "processes_closed", "storage_closed", "cleanup_errors",
         "original_receipt_reference", "original_receipt_descriptor", "verified_checks", "parent_source_closeout",
         "repair_authority", "adoption_authority", "activation_authority", "completion_authority"}
         and receipt.get("schema") == _source_qualification_receipt_schema(context)
         and receipt.get("passed") is True and receipt.get("snapshot") == context["snapshot_ref"]
         and receipt.get("source_commitment") == context["snapshot"]["source_commitment"]
         and receipt.get("source_qualification") == context["profile"]
         and receipt.get("cleanup_errors") == []
         and receipt.get("original_receipt_reference") == receipt.get("original_receipt_descriptor", {}).get("path")
         and all(receipt.get(key) is True for key in ("source_closed", "native_closed", "processes_closed", "storage_closed"))
         and all(receipt.get(key) is False for key in ("repair_authority", "adoption_authority",
             "activation_authority", "completion_authority")), "positive source-only qualification required")
    plan = json_object(ordinary_bytes(args.invocation_root / "validation-plan.json"))
    _source_qualification_plan(policy, plan, args.invocation_root)
    terminal = json_object(checked_descriptor(receipt["original_receipt_descriptor"]))
    request_pin = original_execution_request(args.invocation_root)
    summary = verify_validation_evidence(policy, plan, request_pin, terminal, args.invocation_root)
    need((canonical(receipt["verified_checks"]) == canonical(summary)
          if _immutable_source_parent_enabled(policy) else receipt["verified_checks"] == summary),
         "immutable qualification verified summary differs")
    verify_v5_parent_source_closeout(policy, request_pin, args.invocation_root,
        receipt["parent_source_closeout"], summary)



IMMUTABLE_PARENT_PROFILE_SCHEMA = "agent-supervisor/immutable-source-qualification@2"
IMMUTABLE_PARENT_OBSERVATION_SCHEMA = "agent-supervisor/immutable-source-parent-observation@1"
IMMUTABLE_PARENT_WORKER_SCHEMA = "immutable-source-parent-delegated-validation@1"
IMMUTABLE_PARENT_RECEIPT_SCHEMA = "agent-supervisor/immutable-source-qualification-receipt@2"
IMMUTABLE_PARENT_MODE = "delegated-original-parent-read/v1"
_IMMUTABLE_PARENT_FIELDS = {"parent_identity", "parent_nonce_sha256", "parent_source_observation_path",
    "execution_request_path", "source_root_bindings"}


def _immutable_source_parent_enabled(policy):
    profile = policy.get("validation", {}).get("source_qualification")
    return type(profile) is dict and profile.get("schema") == IMMUTABLE_PARENT_PROFILE_SCHEMA


def _source_qualification_receipt_schema(context):
    return (IMMUTABLE_PARENT_RECEIPT_SCHEMA if context["profile"]["schema"] == IMMUTABLE_PARENT_PROFILE_SCHEMA
            else SOURCE_QUALIFICATION_SCHEMA)


def _immutable_source_parent_root_bindings(rows, physical_roots=None):
    need(type(rows) is list and len(rows) == 2, "two immutable physical/canonical Source roots required")
    for index, row in enumerate(rows):
        need(type(row) is dict and set(row) == {"physical_root", "canonical_root"}, "closed Source root mapping required")
        for key in row:
            path = Path(row[key]) if type(row[key]) is str else None
            need(path is not None and path.is_absolute() and str(path) == row[key]
                 and not any(part.is_symlink() for part in (path, *path.parents)), "unaliased Source root mapping required")
        if physical_roots is not None:
            need(row["physical_root"] == str(physical_roots[index]), "physical Source root mapping differs")
    return json_object(canonical({"rows": rows}))["rows"]


def _immutable_source_parent_plan_fields(value, invocation):
    need(type(value) is dict, "typed immutable parent plan context required")
    fields = {key: value.get(key) for key in _IMMUTABLE_PARENT_FIELDS}
    identity = fields["parent_identity"]
    need(type(identity) is dict and set(identity) == {"pid", "birth"}
         and type(identity["pid"]) is int and identity["pid"] > 0
         and type(identity["birth"]) is str and identity["birth"].isascii()
         and identity["birth"].isdigit() and int(identity["birth"]) > 0
         and type(fields["parent_nonce_sha256"]) is str and _SHA.fullmatch(fields["parent_nonce_sha256"])
         and fields["parent_source_observation_path"] == str(invocation / "parent-source-observation.json")
         and type(fields["execution_request_path"]) is str
         and Path(fields["execution_request_path"]).is_absolute()
         and Path(fields["execution_request_path"]).name == "execution-request.json",
         "closed immutable parent runtime plan fields required")
    _immutable_source_parent_root_bindings(fields["source_root_bindings"])
    return json_object(canonical(fields))


def _immutable_source_owner(policy):
    lease = public_source_lease(policy)
    source = Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"
    pin = next(row for row in policy["source_descriptors"] if row["path"] == str(source))
    raw = checked_descriptor(pin); wrapped = getattr(lease, "__wrapped__", None)
    module = sys.modules.get("ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease")
    need(type(module) is types.ModuleType and module.__file__ == str(source)
         and module.__spec__.origin == str(source) and type(wrapped) is types.FunctionType
         and wrapped.__globals__ is vars(module), "original Source42 module required")
    codes = {}
    def walk(code):
        codes[code.co_qualname] = code
        for child in code.co_consts:
            if type(child) is types.CodeType: walk(child)
    walk(_ORIGINAL_COMPILE(raw, str(source), "exec", dont_inherit=True, optimize=sys.flags.optimize))
    need(wrapped.__code__ == codes.get("source_lease") and lease.__code__ == contextlib.contextmanager(wrapped).__code__
         and lease.__closure__ is not None and len(lease.__closure__) == 1
         and lease.__closure__[0].cell_contents is wrapped, "original Source42 generator binding differs")
    resolver = module.canonical_source_root
    need(type(resolver) is types.FunctionType and resolver.__globals__ is vars(module)
         and resolver.__code__ == codes.get("canonical_source_root"), "original Source42 root resolver differs")
    return module, pin, wrapped.__code__, resolver


def _immutable_source_parent_runtime(policy, invocation):
    module, pin, code, resolver = _immutable_source_owner(policy)
    qualification = checked_source_qualification(policy)
    physical = [Path(policy["datasets_root"]), qualification["root"]]
    rows = [{"physical_root": str(root), "canonical_root": str(resolver(root))} for root in physical]
    _immutable_source_parent_root_bindings(rows, physical)
    nonce = os.urandom(32).hex()
    return {"parent_identity": proc_identity(os.getpid()), "parent_nonce": nonce,
        "parent_nonce_sha256": hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        "parent_source_observation_path": str(invocation / "parent-source-observation.json"),
        "execution_request_path": str(validation_control_directory(policy, qualification["snapshot"], invocation) / "execution-request.json"),
        "source_root_bindings": rows}


def _immutable_source_parent_check_context(custody):
    runtime = custody.context["source_parent"]; plan = custody.context["plan"]
    need(type(runtime) is dict and set(runtime) == _IMMUTABLE_PARENT_FIELDS | {"parent_nonce"}
         and _immutable_source_parent_plan_fields(runtime, Path(custody.context["invocation"]))
             == _immutable_source_parent_plan_fields(plan, Path(custody.context["invocation"]))
         and runtime["parent_identity"] == proc_identity(os.getpid())
         and type(runtime["parent_nonce"]) is str and re.fullmatch(r"[0-9a-f]{64}", runtime["parent_nonce"])
         and hashlib.sha256(bytes.fromhex(runtime["parent_nonce"])).hexdigest() == plan["parent_nonce_sha256"],
         "trusted immutable parent runtime context differs")
    need(plan["execution_request_path"] == custody.request_pin["path"], "immutable parent public request differs")
    _immutable_source_parent_root_bindings(plan["source_root_bindings"], [custody.d, custody.cwd])
    _source_qualification_plan(custody.policy, plan, Path(custody.context["invocation"]))


def _immutable_source_parent_record_fd(custody, manager, held, physical):
    module, pin, code, resolver = custody.immutable_source_binding
    checked_descriptor(pin)
    generator = getattr(manager, "gen", None); frame = getattr(generator, "gi_frame", None)
    need(frame is not None and generator.gi_code is code and frame.f_globals is vars(module)
         and module.source_lease.__wrapped__.__code__ is code
         and module.canonical_source_root is resolver, "same live original Source42 generator required")
    index = len(getattr(custody, "immutable_source_fds", []))
    binding = custody.context["plan"]["source_root_bindings"][index]
    need(binding == {"physical_root": str(physical), "canonical_root": str(resolver(physical))}
         and held.canonical_root == binding["canonical_root"], "current original Source42 mapping changed")
    fd = frame.f_locals.get("source")
    need(type(fd) is int and fd >= 0 and frame.f_locals.get("root") == Path(held.canonical_root),
         "actual held immutable parent Source descriptor required")
    info = os.fstat(fd)
    row = {"root": held.canonical_root, "lock_path": held.lock_path, "fd": fd,
        "device": info.st_dev, "inode": info.st_ino}
    _reviewed_parent_kernel_read_lock(os.getpid(), row)
    if not hasattr(custody, "immutable_source_fds"): custody.immutable_source_fds = []
    custody.immutable_source_fds.append(row)


def _immutable_source_parent_observation(raw, plan, request_pin, argv, *, live):
    rows = _immutable_source_parent_root_bindings(plan["source_root_bindings"])
    return _parent_source_observation_core(raw, expected_request_sha256=request_pin["sha256"],
        expected_parent_identity=plan["parent_identity"], expected_nonce_sha256=plan["parent_nonce_sha256"],
        expected_sdk_argv=argv, expected_roots=[Path(row["canonical_root"]) for row in rows],
        expected_schema=IMMUTABLE_PARENT_OBSERVATION_SCHEMA, source_root_bindings=rows, live=live)


def verify_immutable_source_parent_observation_v1(observation_raw, *, expected_request_sha256,
        expected_parent_identity, expected_nonce_sha256, expected_sdk_argv, expected_source_root_bindings):
    """Live Source-only delegation proof; no lease, Git, native collector or closure authority."""
    rows = _immutable_source_parent_root_bindings(expected_source_root_bindings)
    value = _parent_source_observation_core(observation_raw, expected_request_sha256=expected_request_sha256,
        expected_parent_identity=expected_parent_identity, expected_nonce_sha256=expected_nonce_sha256,
        expected_sdk_argv=expected_sdk_argv, expected_roots=[Path(row["canonical_root"]) for row in rows],
        expected_schema=IMMUTABLE_PARENT_OBSERVATION_SCHEMA, source_root_bindings=rows)
    _parent_source_child_chain(expected_parent_identity, expected_sdk_argv)
    return {"observation_sha256": hashlib.sha256(observation_raw).hexdigest(),
        "request_sha256": expected_request_sha256, "nonce_sha256": expected_nonce_sha256,
        "parent_identity": dict(expected_parent_identity), "source_fds": [dict(row) for row in value["source_fds"]],
        "source_root_bindings": rows}


def _immutable_source_parent_publish(custody):
    _immutable_source_parent_check_context(custody)
    plan = custody.context["plan"]
    value = {"schema": IMMUTABLE_PARENT_OBSERVATION_SCHEMA, "request": custody.request_pin,
        "parent_identity": plan["parent_identity"], "nonce": custody.context["source_parent"]["parent_nonce"],
        "source_receipts": custody.receipts, "source_fds": custody.immutable_source_fds,
        "sdk_argv": custody.argv, "source_root_bindings": plan["source_root_bindings"]}
    custody.immutable_source_observation = new_json(Path(plan["parent_source_observation_path"]), value)
    custody.immutable_source_raw = checked_descriptor(custody.immutable_source_observation)
    _immutable_source_parent_observation(custody.immutable_source_raw, plan, custody.request_pin, custody.argv, live=True)


def _immutable_source_parent_summary(plan, observation):
    return {"source_custody_mode": IMMUTABLE_PARENT_MODE, "parent_source_observation": observation,
        "parent_nonce_sha256": plan["parent_nonce_sha256"], "parent_identity": plan["parent_identity"],
        "source_root_bindings": plan["source_root_bindings"]}


def _immutable_source_parent_argv(policy, request_pin):
    return [policy["python"], "-I", "-B", str(Path(policy["datasets_root"]) / "scripts/ops/legal_ir/run_owned_validation.py"),
        "--request", request_pin["path"], "--expected-request-sha256", request_pin["sha256"],
        "--resource-ledger", policy["validation"]["resource_ledger"]]


def _immutable_source_parent_saved_observation(policy, plan, request_pin, invocation):
    """Archived DATA joins only: original held-Source attestation, not current Git/lock admission."""
    _immutable_source_parent_plan_fields(plan, invocation)
    _immutable_source_parent_root_bindings(plan["source_root_bindings"],
        [Path(policy["datasets_root"]), Path(plan["candidate_root"])])
    observation = descriptor(Path(plan["parent_source_observation_path"]))
    raw = checked_descriptor(observation)
    need(not Path(observation["path"]).lstat().st_mode & 0o222, "sealed immutable parent observation required")
    need(plan["execution_request_path"] == request_pin["path"], "archived public request path differs")
    observed = _immutable_source_parent_observation(raw, plan, request_pin,
        _immutable_source_parent_argv(policy, request_pin), live=False)
    need(canonical(observed["request"]) == canonical(request_pin), "archived public request descriptor differs")
    return _immutable_source_parent_summary(plan, observation)


def _immutable_source_parent_report_binding(policy, plan, request_pin, report, invocation):
    expected = _immutable_source_parent_saved_observation(policy, plan, request_pin, invocation)
    need(report["schema"] == IMMUTABLE_PARENT_WORKER_SCHEMA
         and report["source_READ_closed"] is False and report["source_READ_admission_complete"] is False
         and all(canonical(report.get(key)) == canonical(value) for key, value in expected.items()),
         "immutable delegated report parent/observation/root mapping differs")
    return expected


def _immutable_source_parent_receipts(receipts, rows, observation):
    need(canonical(receipts) == canonical(observation["source_receipts"]),
         "archived parent and observed Source receipts differ")
    need(type(receipts) is list and len(receipts) == len(rows) == 2, "two original delegated parent receipts required")
    for receipt, row in zip(receipts, rows, strict=True):
        need(receipt["canonical_root"] == row["canonical_root"] and receipt["mode"] == "read"
             and receipt["lock_path"] == str(Path(row["canonical_root"]) / "workspace/control-plane/source-leases/source.lock"),
             "original delegated parent receipt root differs")


def _immutable_source_parent_check_delegation(custody, *, require_worker_report):
    plan = custody.context["plan"]
    need(checked_descriptor(custody.immutable_source_observation) == custody.immutable_source_raw,
         "immutable parent Source observation changed")
    _immutable_source_parent_observation(custody.immutable_source_raw, plan, custody.request_pin, custody.argv, live=True)
    # Recheck the actual owner resolver/mapping while original READs remain held.
    module, pin, code, resolver = custody.immutable_source_binding
    need(_immutable_source_owner(custody.policy) == custody.immutable_source_binding,
         "original Source42 callable bindings changed")
    checked_descriptor(pin)
    need(module.canonical_source_root is resolver, "original Source42 resolver changed")
    for row in plan["source_root_bindings"]:
        need(str(resolver(Path(row["physical_root"]))) == row["canonical_root"], "held physical/canonical mapping changed")
    if require_worker_report:
        report = json_object(ordinary_bytes(Path(plan["attempt_directory"]) / "exports/trusted-validation.json"))
        _immutable_source_parent_report_binding(custody.policy, plan, custody.request_pin, report,
            Path(custody.context["invocation"]))




REVIEWED_PARENT_POLICY_SCHEMA_V3 = "reviewed-preservation-parent/v3"

REVIEWED_PARENT_WORKER_SCHEMA_V3 = "reviewed-preservation-native-worker/v3"

REVIEWED_PARENT_BINDING_SCHEMA_V3 = "reviewed-preservation-native-binding/v3"

_JOIN_RELEASE_EXPECTED_MODULES_V1 = (
    "ipfs_datasets_py.logic.autoformal.legal_release_coordinator_v1",
    "ipfs_datasets_py.logic.autoformal.corpus_contracts",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.uscode_dataset",
    "ipfs_datasets_py.logic.autoformal.uscode_ingest",
    "ipfs_datasets_py.logic.autoformal.legal_entity_index_intake",
    "ipfs_datasets_py.logic.autoformal.constitution_inventory",
    "ipfs_datasets_py.logic.autoformal.legal_parent_text_views",
    "ipfs_datasets_py.logic.autoformal.meta_ontology",
)


def _reviewed_parent_domain(reviewed_version):
    domain = _reviewed_parent_domain_v3(reviewed_version)
    return tuple(domain[name] for name in (
        "policy_schema", "worker_schema", "binding_schema", "fixture_schema"))

def _reviewed_parent_version(policy):
    return _reviewed_parent_version_v3(policy)

def _reviewed_parent_packet_paths_v3(data):
    """Derive opening paths from the independently pinned complete preparation DATA.

    This is an identity/shape check only. The original V3 native producer still
    regenerates the complete project and validates source/review/proofs.
    """
    preparation = data["preparation"]
    need(type(preparation) is dict and set(preparation) == {"manifest_json", "files"}
         and type(preparation["manifest_json"]) is str
         and len(preparation["manifest_json"].encode("utf-8")) <= 1024**2
         and type(preparation["files"]) is list and 1 <= len(preparation["files"]) <= 78,
         "complete bounded V3 preparation DATA required")
    # The current shared owner hashes exactly this ordinary string/list tree.
    need(digest(preparation) == data["expected_preservation_sha256"],
         "opening V3 preparation DATA differs from its independent complete pin")
    manifest = json_object(preparation["manifest_json"].encode("utf-8"))
    need(manifest.get("schema") == "legal-corpus-reviewed-preservation-preparation/v3"
         and manifest.get("profile") == "reviewed-multi-unit-mixed-relational-preservation/v3"
         and type(manifest.get("units")) is list and 3 <= len(manifest["units"]) <= 8,
         "explicit complete three-to-eight-unit mixed preparation required")
    names, hashes, total = [], {}, 0
    for pair in preparation["files"]:
        need(type(pair) is list and len(pair) == 2
             and all(type(value) is str for value in pair), "ordinary V3 preparation file pairs required")
        name, body = pair
        need(len(name) <= 256 and re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", name)
             and all(part not in (".", "..") for part in name.split("/")),
             "safe exact V3 preparation path required")
        raw = body.encode("utf-8"); total += len(raw)
        need(total <= 8 * 1024**2, "unchanged complete native project byte bound required")
        names.append(name); hashes[name] = hashlib.sha256(raw).hexdigest()
    need(names == sorted(set(names)) and "corpus-manifest.json" not in names
         and manifest.get("file_sha256") == hashes,
         "opening V3 file list and manifest hashes differ")
    modules = manifest.get("expected_modules")
    need(type(modules) is list and 1 <= len(modules) <= 11
         and all(type(name) is str and re.fullmatch(
             r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", name) for name in modules)
         and len(set(modules)) == len(modules),
         "exact bounded V3 compiled-module population required")
    paths = sorted(["build-receipt.json", "stdout.txt", "stderr.txt", "runner-error.txt",
        "project/corpus-manifest.json",
        *("project/" + name for name in names),
        *("executed-inputs/" + name for name in names),
        *("compiled/" + name.replace(".", "/") + ".olean" for name in modules)])
    need(len(paths) <= 172 and len(paths) == len(set(paths)),
         "complete opening V3 packet path population required")
    return paths

def _checked_reviewed_preservation_parent_policy(path, expected_policy_sha256, *, reviewed_version):
    """Compatibility name for the maintained Main parser and its depth limits."""
    return _checked_reviewed_preservation_parent_policy_core_v3(
        path, expected_policy_sha256, reviewed_version=reviewed_version)

def _reviewed_parent_packet_binding(operation, binding, durability_files, *, attempt_directory, reviewed_version):
    """Pure exact inventory join; this alone authenticates no original success."""
    domain = _reviewed_parent_domain(reviewed_version)
    fields = {"schema", "operation_id", "packet_root", "receipt_sha256", "packet_sha256",
        "preparation_sha256", "graph_sha256", "admission_sha256", "toolchain", "binary_sha256",
        "producer_pins", "execution_projection_sha256", "packet_inventory", "native_argv"}
    if reviewed_version in (3, 4):
        fields.add("expected_packet_paths")
    if reviewed_version == 4:
        fields.update({"upstream_preservation_sha256", "physical_request_sha256",
            "instance_identity_profile", "definition_application_profile"})
    need(type(binding) is dict and set(binding) == fields and binding["schema"] == domain[2],
         "closed original reviewed native packet binding required")
    wanted = {"operation_id": operation["operation_id"], "packet_root": operation["packet_root"],
        "preparation_sha256": operation["expected_realization_sha256"] if reviewed_version == 4 else operation["expected_preservation_sha256"], "graph_sha256": operation["expected_graph_sha256"],
        "admission_sha256": operation["expected_admission_sha256"], "toolchain": operation["toolchain"],
        "binary_sha256": operation["expected_binary_sha256"], "native_argv": operation["native_argv"]}
    if reviewed_version == 4:
        wanted.update(upstream_preservation_sha256=operation["expected_preservation_sha256"],
            physical_request_sha256=operation["expected_physical_request_sha256"],
            instance_identity_profile=operation["instance_identity_profile"],
            definition_application_profile=operation["definition_application_profile"])
    need(all(binding[key] == value for key, value in wanted.items()), "predeclared reviewed native operation differs")
    need(type(binding["producer_pins"]) is dict and all(binding["producer_pins"].get(name) == pin
        for name, pin in operation["expected_source_sha256"].items()), "original reviewed native source pins differ")
    for key in ("receipt_sha256", "packet_sha256", "execution_projection_sha256"):
        need(type(binding[key]) is str and _SHA.fullmatch(binding[key]), "complete original native commitment required")
    rows = binding["packet_inventory"]
    need(type(rows) is list and ((len(rows) == 77) if reviewed_version == 2 else 1 <= len(rows) <= 172),
         "complete original77-file native packet required" if reviewed_version == 2
         else "complete original V3 native packet population required")
    names = []
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and type(row["bytes"]) is int and 0 <= row["bytes"] <= (8 if reviewed_version == 4 else 48) * 1024**2
             and type(row["sha256"]) is str and _SHA.fullmatch(row["sha256"]), "exact native packet byte inventory required")
        names.append(_reviewed_parent_relative(row["path"]))
    need(names == sorted(set(names)) and sum(row["bytes"] for row in rows) <= (48 if reviewed_version == 4 else 16) * 1024**2,
         "sorted bounded complete original native packet required")
    if reviewed_version == 2:
        need(set(name for name in names if "/" not in name) == {"build-receipt.json", "stdout.txt", "stderr.txt", "runner-error.txt"}
             and sum(name.startswith("project/") for name in names) == 34
             and sum(name.startswith("executed-inputs/") for name in names) == 33
             and sum(name.startswith("compiled/") for name in names) == 6,
             "exact qualified77-file native packet partitions required")
    else:
        expected = operation["expected_packet_paths"]
        need(type(expected) is list and 1 <= len(expected) <= 172
             and all(type(name) is str for name in expected) and expected == sorted(set(expected))
             and names == expected and binding["expected_packet_paths"] == expected,
             "exact independently pinned V3 packet population required")
        need(set(name for name in names if "/" not in name)
             == {"build-receipt.json", "stdout.txt", "stderr.txt", "runner-error.txt"},
             "complete V3 top-level receipt and log population required")
    need(digest(rows) == binding["packet_sha256"]
         and next(row["sha256"] for row in rows if row["path"] == "build-receipt.json") == binding["receipt_sha256"],
         "complete original native packet/receipt commitment differs")
    prefix = _reviewed_parent_relative(operation["packet_root"]) + "/"
    need(Path(attempt_directory).is_absolute() and prefix.startswith("exports/"), "owned native export root required")
    selected = {}
    need(type(durability_files) is list, "original complete durability inventory required")
    for row in durability_files:
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256", "device", "inode"},
             "closed original durable file row required")
        _reviewed_parent_relative(row["path"])
        if row["path"].startswith(prefix):
            rel = row["path"][len(prefix):]
            need(rel not in selected, "duplicate original durable native path")
            selected[rel] = row
    need(set(selected) == set(names), "complete original native packet durability coverage differs")
    result = []
    for row in rows:
        durable = selected[row["path"]]
        need((row["bytes"], row["sha256"]) == (durable["bytes"], durable["sha256"])
             and all(type(durable[key]) is int and durable[key] >= 0 for key in ("device", "inode")),
             "original native packet byte/physical durability join differs")
        result.append((prefix + row["path"], row["bytes"], row["sha256"], durable["device"], durable["inode"]))
    return tuple(result)

def _parent_native_descriptor_function_v1(value):
    """Extract only native descriptor wrappers using identity-only dispatch."""
    kind = type(value)
    if kind is staticmethod or kind is classmethod:
        return value.__func__
    if kind is property:
        return value.fget
    return value

def _parent_native_module_metadata_v1(module, name, path):
    """Read fixed module metadata without invoking module lookup hooks."""
    need(type(module) is types.ModuleType, "ordinary canonical native owner required")
    namespace = vars(module)
    need(type(namespace) is dict and len(namespace) <= 4096
         and all(type(key) is str for key in namespace), "ordinary native namespace required")
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for key, expected in (("__name__", name), ("__file__", str(path)), ("__package__", package)):
        value = namespace.get(key)
        need(type(value) is str and value == expected, "canonical native metadata differs")
    spec = namespace.get("__spec__")
    need(type(spec) is importlib.machinery.ModuleSpec, "ordinary native specification required")
    fields = object.__getattribute__(spec, "__dict__")
    need(type(fields) is dict and all(type(key) is str for key in fields),
         "ordinary native specification fields required")
    for key, expected in (("name", name), ("origin", str(path))):
        value = fields.get(key)
        need(type(value) is str and value == expected, "canonical native specification differs")
    need(sys.modules.get(name) is module and path.resolve(strict=True) == path,
         "canonical native identity differs")
    return namespace

def _parent_native_source_plan_v1(raw, path):
    """Derive code, declarations and scalar defaults without executing Source."""
    need(type(raw) is bytes and len(raw) <= 4 * 1024**2, "bounded native Source required")
    tree = ast.parse(raw, filename=str(path))
    compiled = _ORIGINAL_COMPILE(raw, str(path), "exec",
                                 dont_inherit=True, optimize=sys.flags.optimize)
    codes, pending, declarations = {}, [compiled], {}
    while pending:
        code = pending.pop()
        need(len(codes) <= 4096, "bounded native Source code population required")
        codes[code.co_qualname] = code
        pending.extend(item for item in code.co_consts if type(item) is types.CodeType)

    def visit(nodes, prefix=""):
        for node in nodes:
            if type(node) is ast.FunctionDef:
                qualified = prefix + node.name
                need(qualified not in declarations, "unambiguous native Source function required")
                declarations[qualified] = node
                visit(node.body, qualified + ".<locals>.")
            elif type(node) is ast.ClassDef:
                visit(node.body, prefix + node.name + ".")
    visit(tree.body)
    return tree, codes, declarations

def _parent_native_literal_v1(node, values):
    """Evaluate only the closed native scalar/container constant vocabulary."""
    kind = type(node)
    if kind is ast.Name:
        need(node.id in values, "closed native Source literal reference required")
        return values[node.id]
    if kind is ast.Constant:
        _joined_literal_signature_v1(node.value)
        return node.value
    if any(kind is allowed for allowed in (ast.Tuple, ast.List, ast.Set)):
        children = tuple(_parent_native_literal_v1(child, values) for child in node.elts)
        return children if kind is ast.Tuple else list(children) if kind is ast.List else set(children)
    if kind is ast.Dict:
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                extra = _parent_native_literal_v1(value, values)
                need(type(extra) is dict, "ordinary Source dictionary expansion required")
                result.update(extra)
            else:
                result[_parent_native_literal_v1(key, values)] = _parent_native_literal_v1(value, values)
        return result
    if kind is ast.UnaryOp and any(type(node.op) is allowed for allowed in (ast.USub, ast.UAdd)):
        value = _parent_native_literal_v1(node.operand, values)
        need(type(value) is int, "native Source integer expression required")
        return -value if type(node.op) is ast.USub else value
    if kind is ast.BinOp:
        left = _parent_native_literal_v1(node.left, values)
        right = _parent_native_literal_v1(node.right, values)
        if type(left) is int and type(right) is int:
            need(max(left.bit_length(), right.bit_length()) <= 4096, "bounded Source arithmetic required")
            if type(node.op) is ast.Mult:
                result = left * right
            elif type(node.op) is ast.Add:
                result = left + right
            elif type(node.op) is ast.Sub:
                result = left - right
            else:
                raise ValueError("closed native Source arithmetic required")
            need(result.bit_length() <= 4096, "bounded Source integer required")
            return result
        if type(left) is list and type(right) is list and type(node.op) is ast.Add:
            return left + right
        if type(left) is frozenset and type(right) is frozenset:
            if type(node.op) is ast.Sub:
                return left - right
            if type(node.op) is ast.BitOr:
                return left | right
        raise ValueError("closed native Source container expression required")
    if kind is ast.Call and not node.keywords:
        if type(node.func) is ast.Name and node.func.id == "frozenset" and len(node.args) == 1:
            value = _parent_native_literal_v1(node.args[0], values)
            need(any(type(value) is allowed for allowed in (tuple, list, set, frozenset, dict)),
                 "ordinary Source frozen collection required")
            return frozenset(value)
        if (type(node.func) is ast.Attribute and type(node.func.value) is ast.Name
                and node.func.value.id == "dict" and node.func.attr == "fromkeys" and len(node.args) == 2):
            keys = _parent_native_literal_v1(node.args[0], values)
            need(type(keys) is tuple and all(type(key) is str for key in keys),
                 "ordinary Source flag names required")
            return dict.fromkeys(keys, _parent_native_literal_v1(node.args[1], values))
    raise ValueError("closed native Source literal expression required")

def _parent_native_runtime_literals_v1(namespace, tree):
    dynamic = {"_SELF_PIN", "_LIVE_SUCCESS_HANDLES", "_LIVE_NATIVE_FUNCTIONS",
        "_LIVE_HANDLE_CLASS", "_LIVE_HANDLE_METACLASS", "_LIVE_HANDLE_BASES", "_LIVE_HANDLE_KEYS",
        "_LIVE_HANDLE_FUNCTIONS", "_LIVE_POPEN_CLASS", "_LIVE_POPEN_METACLASS", "_LIVE_POPEN_BASES",
        "_LIVE_POPEN_KEYS", "_LIVE_POPEN_FUNCTIONS", "_LIVE_MODULE_BINDINGS",
        "_LIVE_REVIEW_BINDINGS", "_LIVE_MIXED_REVIEW_BINDINGS"}
    allowed = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__",
               "__cached__", "__builtins__"}
    values, caches = {}, []
    for node in tree.body:
        kind = type(node)
        if any(kind is accepted for accepted in (ast.FunctionDef, ast.ClassDef)):
            allowed.add(node.name)
        elif any(kind is accepted for accepted in (ast.Import, ast.ImportFrom)):
            if kind is ast.Import:
                imports = ((item.asname or item.name.partition(".")[0],
                    importlib.import_module(item.name if item.asname else item.name.partition(".")[0]))
                    for item in node.names)
            else:
                need(node.level == 0 and type(node.module) is str,
                     "fixed standard-library Native import required")
                owner = importlib.import_module(node.module)
                imports = ((item.asname or item.name, vars(owner).get(item.name)) for item in node.names)
            for bound, expected in imports:
                allowed.add(bound)
                need(expected is not None and namespace.get(bound) is expected,
                     "native Source runtime import alias differs")
        elif kind is ast.Assign:
            names = []
            for target in node.targets:
                if type(target) is ast.Name:
                    names.append(target.id)
                elif type(target) is ast.Tuple:
                    need(all(type(item) is ast.Name for item in target.elts),
                         "closed native Source export assignment required")
                    names.extend(item.id for item in target.elts)
                else:
                    raise ValueError("closed native Source assignment required")
            allowed.update(names)
            if len(names) != 1 or names[0] in dynamic:
                continue
            bound = names[0]
            if bound.endswith("_OPERATION_PINS") or bound == "_OPERATION_PINS":
                need(type(node.value) is ast.Constant and node.value.value is None,
                     "original lazy native pin cache declaration required")
                caches.append(bound)
            elif (type(node.value) is ast.Call and type(node.value.func) is ast.Attribute
                    and type(node.value.func.value) is ast.Name
                    and node.value.func.value.id == "re" and node.value.func.attr == "compile"):
                need(len(node.value.args) == 1 and not node.value.keywords,
                     "closed Source regular expression required")
                expected = _parent_native_literal_v1(node.value.args[0], values)
                current = namespace.get(bound)
                need(type(expected) is str and type(current) is re.Pattern
                     and type(current.pattern) is str and current.pattern == expected
                     and current.flags == re.compile(expected).flags, "native Source regular expression differs")
            else:
                values[bound] = _parent_native_literal_v1(node.value, values)
        elif kind is ast.AugAssign:
            need(type(node.target) is ast.Name and type(node.op) is ast.Add
                 and node.target.id in values, "closed native Source export append required")
            left, right = values[node.target.id], _parent_native_literal_v1(node.value, values)
            need(type(left) is list and type(right) is list, "ordinary Source export lists required")
            values[node.target.id] = left + right
    need(set(namespace) <= allowed and namespace.get("__builtins__") is __builtins__,
         "native namespace/builtin shadow bindings differ")
    for bound, expected in values.items():
        need(_joined_literal_signature_v1(namespace.get(bound)) == _joined_literal_signature_v1(expected),
             "native Source literal/resource binding differs")
    for bound in caches:
        current = namespace.get(bound)
        need(current is None or type(current) is dict and len(current) <= 4096
             and all(type(key) is str and type(value) is str and _SHA.fullmatch(value)
                     for key, value in current.items()), "ordinary native producer pin cache required")
    return values

def _parent_native_class_tables_v1(namespace, declarations, codes, check_function, state_matches):
    """Authenticate the two Source classes and trusted Popen state before iteration."""
    class_specs = (("_LiveNativeObserverRefusal", ValueError),
                   ("VerifiedReviewedPreservationNativeSuccess", object))
    handle_functions = {}
    for name, base in class_specs:
        owner = namespace.get(name)
        need(type(owner) is type, "ordinary original Native class required")
        fields = type.__getattribute__(owner, "__dict__")
        bases = type.__getattribute__(owner, "__bases__")
        need(type(bases) is tuple and len(bases) == 1 and bases[0] is base,
             "original Native class bases differ")
        module_name = type.__getattribute__(owner, "__module__")
        qualified = type.__getattribute__(owner, "__qualname__")
        need(type(module_name) is str and module_name == namespace["__name__"]
             and type(qualified) is str and qualified == name,
             "original Native class metadata differs")
        methods = {qualified[len(name) + 1:]: qualified for qualified in declarations
                   if qualified.startswith(name + ".") and ".<locals>." not in qualified}
        allowed = set(methods) | {"__module__", "__doc__", "__dict__", "__weakref__"}
        if name == "VerifiedReviewedPreservationNativeSuccess":
            allowed.update(("__slots__", "__hash__", "__eq__"))
            slots = fields.get("__slots__")
            need(type(slots) is tuple and len(slots) == 1 and type(slots[0]) is str
                 and slots[0] == "__weakref__" and fields.get("__hash__") is object.__hash__
                 and fields.get("__eq__") is object.__eq__, "original opaque Native class contract differs")
        need(all(type(key) is str for key in fields) and set(fields) <= allowed,
             "undeclared Native class namespace differs")
        for generated in ("__dict__", "__weakref__"):
            if generated in fields:
                descriptor = fields[generated]
                need(any(type(descriptor) is allowed_type for allowed_type in
                         (types.GetSetDescriptorType, types.MemberDescriptorType))
                     and descriptor.__objclass__ is owner, "ordinary generated Native descriptor required")
        for method, qualified in methods.items():
            function = _parent_native_descriptor_function_v1(fields.get(method))
            check_function(function, qualified)
            if name == "VerifiedReviewedPreservationNativeSuccess":
                handle_functions[method] = function
    handle = namespace["VerifiedReviewedPreservationNativeSuccess"]
    need(namespace.get("_LIVE_HANDLE_CLASS") is handle
         and namespace.get("_LIVE_HANDLE_METACLASS") is type
         and namespace.get("_LIVE_HANDLE_BASES") is type.__getattribute__(handle, "__bases__"),
         "original Native handle class table differs")
    handle_keys = namespace.get("_LIVE_HANDLE_KEYS")
    need(type(handle_keys) is frozenset and all(type(key) is str for key in handle_keys)
         and handle_keys == frozenset(type.__getattribute__(handle, "__dict__")),
         "original Native handle key table differs")
    tables = (("_LIVE_HANDLE_FUNCTIONS", handle_functions),)
    popen = _ORIGINAL_POPEN
    need(namespace.get("_LIVE_POPEN_CLASS") is popen
         and namespace.get("_LIVE_POPEN_METACLASS") is _REVIEWED_PARENT_POPEN_META
         and namespace.get("_LIVE_POPEN_BASES") is _REVIEWED_PARENT_POPEN_BASES,
         "original trusted Popen class table differs")
    popen_keys = namespace.get("_LIVE_POPEN_KEYS")
    need(type(popen_keys) is frozenset and all(type(key) is str for key in popen_keys)
         and popen_keys == _REVIEWED_PARENT_POPEN_KEYS, "original trusted Popen key table differs")
    popen_functions = {}
    for key, original in _REVIEWED_PARENT_POPEN_FUNCTIONS:
        current = _parent_native_descriptor_function_v1(vars(popen).get(key))
        need(type(current) is types.FunctionType and current is original[0]
             and current.__code__ is original[1] and current.__defaults__ is original[2],
             "original trusted Popen function differs")
        kwargs = current.__kwdefaults__
        need(kwargs is None or type(kwargs) is dict and all(type(name) is str for name in kwargs),
             "ordinary trusted Popen keyword defaults required")
        need(state_matches(original, current), "original trusted Popen defaults differ")
        popen_functions[key] = current
    tables += (("_LIVE_POPEN_FUNCTIONS", popen_functions),)
    for key, expected in tables:
        table = namespace.get(key)
        need(type(table) is tuple and len(table) == len(expected),
             "complete original Native class function table required")
        seen = set()
        for row in table:
            need(type(row) is tuple and len(row) == 2 and type(row[0]) is str
                 and row[0] in expected and row[0] not in seen
                 and state_matches(row[1], expected[row[0]]),
                 "original Native class function state differs")
            seen.add(row[0])

def _parent_native_trusted_metaclasses_v1():
    # These are existing standard-library infrastructure identities, not
    # caller-supplied predicates. Actual process.py uses StrEnum and Protocol.
    abc = importlib.import_module("abc")
    enum = importlib.import_module("enum")
    typing = importlib.import_module("typing")
    return (type, vars(abc)["ABCMeta"], vars(enum)["EnumType"], type(vars(typing)["Protocol"]))

def _parent_native_declared_class_v1(node, namespace, tree, prior_classes, trusted_metaclasses):
    """Bind Source base names and stdlib metaclasses without calling class hooks."""
    imports = {}
    for statement in tree.body:
        if type(statement) is ast.Import:
            for item in statement.names:
                imports[item.asname or item.name.partition(".")[0]] = (
                    item.name if item.asname else item.name.partition(".")[0], None, 0)
        elif type(statement) is ast.ImportFrom:
            for item in statement.names:
                imports[item.asname or item.name] = (statement.module or "", item.name, statement.level)
    builtins_owner = importlib.import_module("builtins")
    builtin_values = vars(builtins_owner)
    package = namespace.get("__package__")
    need(type(package) is str, "ordinary Source dependency package required")

    def reference(expression):
        if type(expression) is ast.Subscript:
            return reference(expression.value)
        if type(expression) is ast.Name:
            key = expression.id
            if key in prior_classes:
                return prior_classes[key]
            if key in imports:
                module_name, attribute, level = imports[key]
                if level:
                    module_name = importlib.util.resolve_name("." * level + module_name, package)
                if module_name in ("abc", "enum", "typing", "collections.abc", "builtins"):
                    module = importlib.import_module(module_name)
                else:
                    module = sys.modules.get(module_name if attribute is not None else module_name.partition(".")[0])
                    need(type(module) is types.ModuleType, "canonical Source class-base import required")
                fields = vars(module)
                need(type(fields) is dict and all(type(name) is str for name in fields),
                     "ordinary Source class-base import namespace required")
                expected = module if attribute is None else fields.get(attribute)
                need(expected is not None and namespace.get(key) is expected,
                     "Source class-base runtime alias differs")
                return expected
            need(key in builtin_values and key not in namespace,
                 "closed independently resolved Source class-base name required")
            return builtin_values[key]
        if type(expression) is ast.Attribute:
            owner = reference(expression.value)
            need(type(owner) is types.ModuleType, "ordinary Source class-base module alias required")
            fields = vars(owner)
            need(type(fields) is dict and all(type(key) is str for key in fields),
                 "ordinary Source class-base module namespace required")
            value = fields.get(expression.attr)
            need(value is not None, "Source class-base attribute missing")
            return value
        raise ValueError("unsupported Source class-base expression requires independent authentication")

    bases = tuple(reference(base) for base in node.bases)
    if not bases:
        bases = (object,)
    explicit = None
    for keyword in node.keywords:
        need(keyword.arg == "metaclass", "unsupported Source class keyword requires independent authentication")
        explicit = reference(keyword.value)
        need(any(explicit is trusted for trusted in trusted_metaclasses),
             "Source class metaclass must have an authenticated stdlib identity")
    expected = type if explicit is None else explicit
    for base in bases:
        candidate = type(base)
        need(any(candidate is trusted for trusted in trusted_metaclasses),
             "Source class-base metaclass must have an authenticated stdlib identity")
        if expected is candidate:
            continue
        expected_mro = type.__getattribute__(expected, "__mro__")
        candidate_mro = type.__getattribute__(candidate, "__mro__")
        if any(parent is candidate for parent in expected_mro):
            continue
        need(any(parent is expected for parent in candidate_mro),
             "compatible authenticated Source class metaclasses required")
        expected = candidate
    return expected, bases

def _parent_native_import_surface_v1(module, root, source_pins):
    """Preflight the Source-defined surface consumed by Native's existing guard."""
    need(type(module) is types.ModuleType, "ordinary Native dependency module required")
    namespace = vars(module)
    need(type(namespace) is dict and len(namespace) <= 4096
         and all(type(key) is str for key in namespace), "ordinary Native dependency namespace required")
    name = namespace.get("__name__")
    need(type(name) is str and re.fullmatch(r"ipfs_datasets_py(?:\.[A-Za-z_][A-Za-z0-9_]*)*", name),
         "fixed canonical Native dependency name required")
    stem = root.joinpath(*name.split("."))
    filename = namespace.get("__file__")
    leaf, package_path = stem.with_suffix(".py"), stem / "__init__.py"
    need(type(filename) is str and (filename == str(leaf) or filename == str(package_path)),
         "canonical Native dependency Source path required")
    path = leaf if filename == str(leaf) else package_path
    _parent_native_module_metadata_v1(module, name, path)
    expected_sha = source_pins.get(str(path))
    need(expected_sha is None or type(expected_sha) is str and _SHA.fullmatch(expected_sha),
         "ordinary optional parent dependency Source pin required")
    raw = ordinary_bytes(path, 4 * 1024**2)
    current_sha = hashlib.sha256(raw).hexdigest()
    # Preserve Native's existing current-canonical-Source provenance contract.
    # A parent commitment, when present, is additionally enforced; absence
    # grants no independently parent-custodied imported-Source claim.
    need(expected_sha is None or current_sha == expected_sha,
         "provided parent dependency Source pin differs")
    tree, codes, declarations = _parent_native_source_plan_v1(raw, path)
    need(namespace.get("__builtins__") is __builtins__, "Native dependency builtins differ")
    bound = set()
    for node in tree.body:
        if any(type(node) is allowed for allowed in (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif type(node) is ast.Import:
            bound.update(item.asname or item.name.partition(".")[0] for item in node.names)
        elif type(node) is ast.ImportFrom:
            bound.update(item.asname or item.name for item in node.names)
        elif any(type(node) is allowed for allowed in (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if type(node) is ast.Assign else (node.target,)
            for target in targets:
                bound.update(child.id for child in ast.walk(target) if type(child) is ast.Name)
    for builtin in ("type", "len", "vars", "globals", "set", "frozenset", "tuple", "all",
                    "any", "sorted", "isinstance", "compile", "ValueError", "TypeError"):
        need(builtin not in namespace or builtin in bound, "Native dependency builtin shadow differs")
    functions, classes = {}, {}
    trusted_metaclasses = _parent_native_trusted_metaclasses_v1()

    def function_surface(function, qualified):
        need(type(function) is types.FunctionType and function.__globals__ is namespace
             and type(function.__module__) is str and function.__module__ == name
             and type(function.__qualname__) is str and function.__qualname__ == qualified
             and qualified in codes, "original Native dependency Source function required")
        need(_joined_literal_signature_v1(function.__code__) == _joined_literal_signature_v1(codes[qualified]),
             "Native dependency function differs from pinned Source")
        defaults, kwargs = function.__defaults__, function.__kwdefaults__
        need(defaults is None or type(defaults) is tuple, "ordinary dependency positional defaults required")
        need(kwargs is None or type(kwargs) is dict and all(type(key) is str for key in kwargs),
             "ordinary dependency keyword defaults required")

    for node in tree.body:
        if type(node) is ast.FunctionDef:
            function = namespace.get(node.name)
            function_surface(function, node.name)
            functions[(id(None), node.name)] = function
        elif type(node) is ast.ClassDef:
            owner = namespace.get(node.name)
            expected_metaclass, expected_bases = _parent_native_declared_class_v1(
                node, namespace, tree, classes, trusted_metaclasses)
            need(type(owner) is expected_metaclass, "Source-authenticated Native dependency metaclass required")
            actual_bases = type.__getattribute__(owner, "__bases__")
            need(type(actual_bases) is tuple and len(actual_bases) == len(expected_bases)
                 and all(a is b for a, b in zip(actual_bases, expected_bases, strict=True)),
                 "Source-authenticated Native dependency bases differ")
            fields = type.__getattribute__(owner, "__dict__")
            module_name = type.__getattribute__(owner, "__module__")
            qualified = type.__getattribute__(owner, "__qualname__")
            need(type(module_name) is str and module_name == name and type(qualified) is str
                 and qualified == node.name and all(type(key) is str for key in fields),
                 "original Native dependency class metadata required")
            declared = {child.name for child in node.body if type(child) is ast.FunctionDef}
            forbidden = {"__getattribute__", "__getattr__", "__call__", "__get__", "__set__", "__delete__"}
            need(not ((set(fields) & forbidden) - declared),
                 "undeclared Native dependency lookup hook refused")
            for child in node.body:
                if type(child) is ast.FunctionDef:
                    function_surface(_parent_native_descriptor_function_v1(fields.get(child.name)),
                                     node.name + "." + child.name)
            for key, descriptor in fields.items():
                # Native\'s retained extractor uses type-membership; reject a
                # caller metaclass equality hook before its existing traversal.
                need(any(type(type(descriptor)) is trusted for trusted in trusted_metaclasses),
                     "authenticated dependency field metaclass required")
                function = _parent_native_descriptor_function_v1(descriptor)
                if type(function) is types.FunctionType:
                    defaults, kwargs = function.__defaults__, function.__kwdefaults__
                    need(defaults is None or type(defaults) is tuple,
                         "ordinary dependency class defaults required")
                    need(kwargs is None or type(kwargs) is dict and all(type(key) is str for key in kwargs),
                         "ordinary dependency class keyword defaults required")
                    functions[(id(owner), key)] = function
            classes[node.name] = owner
    return namespace, path, current_sha, classes, functions

def _parent_native_import_tables_v1(namespace, source_pins, root, state_matches, reviewed_version):
    """Validate snapshot schemas and Source completeness before Native iteration."""
    surfaces = {}

    def surface(module):
        need(type(module) is types.ModuleType, "ordinary Native snapshot module required")
        key = id(module)
        if key not in surfaces:
            surfaces[key] = _parent_native_import_surface_v1(module, root, source_pins)
        return surfaces[key]

    def snapshot(value):
        need(type(value) is tuple and len(value) == 5,
             "ordinary complete Native import snapshot required")
        module, filename, sha, classes, functions = value
        need(type(filename) is str and type(sha) is str and _SHA.fullmatch(sha)
             and type(classes) is tuple and len(classes) <= 512
             and type(functions) is tuple and len(functions) <= 4096,
             "ordinary Native import snapshot fields required")
        original, path, expected_sha, expected_classes, expected_functions = surface(module)
        need(filename == str(path) and sha == expected_sha
             and len(classes) == len(expected_classes) and len(functions) == len(expected_functions),
             "Source-complete Native import snapshot required")
        seen_classes = set()
        for row in classes:
            need(type(row) is tuple and len(row) == 5 and type(row[0]) is str
                 and row[0] in expected_classes and row[0] not in seen_classes,
                 "ordinary complete Native class snapshot required")
            key, owner, metaclass, bases, keys = row
            need(owner is expected_classes[key] and metaclass is type(owner)
                 and bases is type.__getattribute__(owner, "__bases__")
                 and type(keys) is frozenset and all(type(item) is str for item in keys)
                 and keys == frozenset(type.__getattribute__(owner, "__dict__")),
                 "original Native dependency class snapshot differs")
            seen_classes.add(key)
        seen_functions = set()
        for row in functions:
            need(type(row) is tuple and len(row) == 3 and type(row[1]) is str
                 and (row[0] is None or any(row[0] is owner for owner in expected_classes.values())),
                 "ordinary Native dependency function snapshot required")
            key = (id(row[0]), row[1])
            need(key in expected_functions and key not in seen_functions
                 and state_matches(row[2], expected_functions[key]),
                 "Source-complete Native dependency function snapshot differs")
            seen_functions.add(key)

    modules = namespace.get("_LIVE_MODULE_BINDINGS")
    if modules is not None:
        need(type(modules) is tuple and len(modules) == 5, "complete Native owner snapshots required")
        for value in modules:
            snapshot(value)
    # The existing guard reads only its selected review domain. Do not turn
    # an inactive domain cache into an undocumented evidence prerequisite.
    active = (("_LIVE_REVIEW_BINDINGS", 6) if reviewed_version == 2
              else ("_LIVE_MIXED_REVIEW_BINDINGS", 7))
    for key, primary_size in (active,):
        bindings = namespace.get(key)
        if bindings is not None:
            need(type(bindings) is tuple and len(bindings) == 2 and type(bindings[0]) is tuple
                 and len(bindings[0]) == primary_size and type(bindings[1]) is tuple
                 and len(bindings[1]) <= 128, "ordinary complete Native review snapshots required")
            for value in (*bindings[0], *bindings[1]):
                snapshot(value)
    return None

def _parent_native_owner_tuple_v1(owners, root, source_pins):
    names = ("ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_lake_v2",
        "ipfs_datasets_py.logic.autoformal.legal_corpus_lake",
        "ipfs_datasets_py.logic.autoformal.legal_pilot_lake",
        "ipfs_datasets_py.logic.backends.process",
        "ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake")
    need(type(owners) is tuple and len(owners) == len(names),
         "fixed complete Native owner tuple required")
    for module, name in zip(owners, names, strict=True):
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        _parent_native_module_metadata_v1(module, name, path)
        _parent_native_import_surface_v1(module, root, source_pins)
    return owners

def _parent_source_builtin_bindings_v1(namespace, tree):
    """Reject Source-undeclared globals before authenticating external callables."""
    need(type(namespace) is dict and len(namespace) <= 4096
         and all(type(key) is str for key in namespace), "ordinary Source owner namespace required")
    allowed = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__",
               "__cached__", "__builtins__"}
    for node in tree.body:
        if any(type(node) is accepted for accepted in (ast.FunctionDef, ast.ClassDef)):
            allowed.add(node.name)
        elif type(node) is ast.Import:
            allowed.update(item.asname or item.name.partition(".")[0] for item in node.names)
        elif type(node) is ast.ImportFrom:
            allowed.update(item.asname or item.name for item in node.names)
        elif any(type(node) is accepted for accepted in (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if type(node) is ast.Assign else (node.target,)
            for target in targets:
                allowed.update(child.id for child in ast.walk(target) if type(child) is ast.Name)
    need(set(namespace) <= allowed and namespace.get("__builtins__") is __builtins__,
         "Source owner namespace/builtin shadow bindings differ")
    return None

def _parent_native_bootstrap_v1(native, source_raw, source_path, source_pins, *, reviewed_version=2):
    """Cold-authenticate actual Native Source and its existing private binder.

    No Native predicate, owner callback, compiler/model, process or resource
    operation runs here. Process memory and stdlib retain their established
    cooperative trust scope. Mutable Native metadata is never a trust root.
    """
    need(type(reviewed_version) is int and reviewed_version in (2, 3, 5),
         "explicit existing Native review domain required")
    name = "ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_native_v2"
    namespace = _parent_native_module_metadata_v1(native, name, source_path)
    need(type(source_pins) is dict and len(source_pins) <= 4096
         and all(type(key) is str and type(value) is str for key, value in source_pins.items()),
         "ordinary independent native Source pins required")
    expected_sha = source_pins.get(str(source_path))
    need(type(expected_sha) is str and _SHA.fullmatch(expected_sha) and type(source_raw) is bytes
         and hashlib.sha256(source_raw).hexdigest() == expected_sha,
         "independent native Source pin differs")
    tree, codes, declarations = _parent_native_source_plan_v1(source_raw, source_path)
    _parent_source_builtin_bindings_v1(namespace, tree)
    literals = _parent_native_runtime_literals_v1(namespace, tree)
    need(type(namespace.get("_SELF_PIN")) is str and namespace["_SELF_PIN"] == expected_sha,
         "native independent self Source pin differs")
    checked, cells, closure_values = {}, {}, {}

    def check_function(function, qualified):
        need(type(function) is types.FunctionType and function.__globals__ is namespace
             and type(function.__name__) is str and type(function.__qualname__) is str
             and type(function.__module__) is str and function.__module__ == name
             and function.__qualname__ == qualified and qualified in declarations and qualified in codes,
             "original Source native function metadata required")
        node = declarations[qualified]
        need(function.__name__ == node.name
             and _joined_literal_signature_v1(function.__code__) == _joined_literal_signature_v1(codes[qualified]),
             "native function differs from independently pinned Source")
        wanted_defaults = (tuple(_parent_native_literal_v1(value, literals) for value in node.args.defaults)
                           if node.args.defaults else None)
        actual_defaults = function.__defaults__
        need(actual_defaults is None or type(actual_defaults) is tuple,
             "ordinary native positional defaults required")
        need(_joined_literal_signature_v1(actual_defaults) == _joined_literal_signature_v1(wanted_defaults),
             "native Source positional defaults differ")
        wanted_keywords = {argument.arg: _parent_native_literal_v1(value, literals)
            for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
            if value is not None}
        wanted_keywords = wanted_keywords if wanted_keywords else None
        actual_keywords = function.__kwdefaults__
        need(actual_keywords is None or type(actual_keywords) is dict
             and all(type(key) is str for key in actual_keywords), "ordinary native keyword defaults required")
        need(_joined_literal_signature_v1(actual_keywords) == _joined_literal_signature_v1(wanted_keywords),
             "native Source keyword defaults differ")
        previous = checked.get(qualified)
        need(previous is None or previous is function, "one original native function per Source declaration required")
        if previous is not None:
            return
        checked[qualified] = function
        actual_cells = function.__closure__
        need((actual_cells is None and not codes[qualified].co_freevars)
             or type(actual_cells) is tuple and len(actual_cells) == len(codes[qualified].co_freevars),
             "original native closure shape required")
        for cell_name, cell in zip(codes[qualified].co_freevars,
                                    () if actual_cells is None else actual_cells, strict=True):
            need(cell_name not in cells or cells[cell_name] is cell,
                 "one original shared native closure cell required")
            cells[cell_name] = cell
            try:
                value = cell.cell_contents
            except ValueError as error:
                raise ValueError("empty native integrity closure refused") from error
            closure_values[cell_name] = value
            if type(value) is types.FunctionType:
                check_function(value, value.__code__.co_qualname)

    wrapped = {"_live_success_guard": "guarded", "_live_entry_guard": "entry_guard",
        "_live_integrity_checker": "integrity",
        "build_verified_reviewed_norm_preservation_native_v2": "build_verified_reviewed_norm_preservation_native_v2",
        "load_verified_reviewed_norm_preservation_native_v2": "load_verified_reviewed_norm_preservation_native_v2",
        "build_verified_reviewed_norm_preservation_native_v3": "build_verified_reviewed_norm_preservation_native_v3",
        "load_verified_reviewed_norm_preservation_native_v3": "load_verified_reviewed_norm_preservation_native_v3"}
    for realization_name in ("build_verified_reviewed_definition_realization_native_v1",
                             "load_verified_reviewed_definition_realization_native_v1",
                             "build_verified_reviewed_physical_definition_realization_native_v2",
                             "load_verified_reviewed_physical_definition_realization_native_v2"):
        qualified_name = "_bind_live_success_independent_guard_v1.<locals>." + realization_name
        if qualified_name in declarations:
            need(qualified_name in codes, "independently compiled native realization wrapper required")
            wrapped[realization_name] = realization_name
    expected = {node.name: node.name for node in tree.body if type(node) is ast.FunctionDef}
    expected.update((key, "_bind_live_success_independent_guard_v1.<locals>." + value)
                    for key, value in wrapped.items())
    functions = {key: value for key, value in namespace.items()
        if type(value) is types.FunctionType and (value.__globals__ is namespace
            or type(value.__module__) is str and value.__module__ == name)}
    need(set(functions) == set(expected), "complete independently pinned native function population required")
    for key, qualified in expected.items():
        check_function(functions[key], qualified)

    def keywords_match(observed, function):
        current = function.__kwdefaults__
        current = {} if current is None else current
        return (type(observed) is tuple and len(observed) == len(current)
            and all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                and pair[0] == key and pair[1] is value
                for pair, (key, value) in zip(observed, sorted(current.items()), strict=True)))

    def state_matches(observed, function):
        return (type(observed) is tuple and len(observed) == 4 and observed[0] is function
            and observed[1] is function.__code__ and observed[2] is function.__defaults__
            and keywords_match(observed[3], function))

    states = namespace.get("_LIVE_NATIVE_FUNCTIONS")
    need(type(states) is tuple and len(states) == len(expected),
         "complete original native private function states required")
    seen = set()
    for row in states:
        need(type(row) is tuple and len(row) == 2 and type(row[0]) is str and row[0] in expected
             and row[0] not in seen and state_matches(row[1], functions[row[0]]),
             "original native private function state differs")
        seen.add(row[0])
    references = {"entry_guard": namespace["_live_entry_guard"], "guarded": namespace["_live_success_guard"],
        "integrity": namespace["_live_integrity_checker"],
        "original_build_core": namespace["_build_verified_reviewed_norm_preservation_native"],
        "original_load_core": namespace["_load_verified_reviewed_norm_preservation_native"],
        "original_handle_entry": namespace["_live_handle_entry"], "function_type": types.FunctionType,
        "states": states, "issued_handles": namespace.get("_LIVE_SUCCESS_HANDLES"),
        "issued_handle_class": namespace.get("VerifiedReviewedPreservationNativeSuccess"),
        "verify_native_functions": checked.get("_bind_live_success_independent_guard_v1.<locals>.verify_native_functions"),
        "original_guard": checked.get("_live_success_guard")}
    need(type(references["issued_handles"]) is weakref.WeakKeyDictionary,
         "ordinary original native handle registry required")
    for stem, function_name in (("entry", "entry_guard"), ("guard", "guarded"),
                                ("handle", "original_handle_entry"), ("original_guard", "original_guard")):
        function = references[function_name]
        need(type(function) is types.FunctionType, "complete original native closure chain required")
        references[stem + "_code"] = function.__code__
        references[stem + "_defaults"] = function.__defaults__
    need(set(closure_values) == set(references) | {
        "guard_keywords", "handle_keywords", "original_guard_keywords", "integrity_state"},
        "closed original native integrity closure population required")
    for key, value in references.items():
        need(closure_values[key] is value, "original native integrity closure reference differs")
    for key, function_name in (("guard_keywords", "guarded"), ("handle_keywords", "original_handle_entry"),
                               ("original_guard_keywords", "original_guard")):
        need(keywords_match(closure_values[key], references[function_name]),
             "original native closure keyword state differs")
    integrity_state = closure_values["integrity_state"]
    need(type(integrity_state) is tuple and len(integrity_state) == 5
         and state_matches(integrity_state[:4], references["integrity"])
         and state_matches(integrity_state[4], namespace["_live_postrun_persistence_trusted"]),
         "original native integrity/persistence state differs")
    _parent_native_class_tables_v1(namespace, declarations, codes, check_function, state_matches)
    _parent_native_import_tables_v1(namespace, source_pins, source_path.parents[3], state_matches, reviewed_version)
    return None

def _load_parent_verified_reviewed_preservation(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256, toolchain, reviewed_version, **kwargs):
    result, fence = _capture_parent_verified_reviewed_preservation(
        handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_preservation_sha256=expected_preservation_sha256,
        expected_graph_sha256=expected_graph_sha256, expected_admission_sha256=expected_admission_sha256,
        toolchain=toolchain, reviewed_version=reviewed_version, **kwargs)
    return result

def _capture_parent_verified_reviewed_preservation(handle, packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256, toolchain, reviewed_version, **kwargs):
    """Join original admitted-parent custody to exact bytes and current review.

    Runs no compiler, process, release, retry or resource acquisition. Retained
    metadata alone cannot issue or reconstruct this live parent capability.
    """
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    entry = _reviewed_parent_capability_entry(handle)
    _reviewed_parent_guard()
    evidence = json_object(entry[4]); binding = evidence["binding"]
    domain = _reviewed_parent_domain(reviewed_version)
    need(evidence.get("reviewed_version", 2) == reviewed_version and binding["schema"] == domain[2],
         "original parent capability belongs to another reviewed native domain")
    if reviewed_version == 3:
        need(evidence["expected_packet_paths"] == evidence["operation"]["expected_packet_paths"]
             == binding["expected_packet_paths"],
             "original V3 opening packet population differs")
    supplied = {"receipt_sha256": expected_receipt_sha256, "packet_sha256": expected_packet_sha256,
        "preparation_sha256": expected_preservation_sha256, "graph_sha256": expected_graph_sha256,
        "admission_sha256": expected_admission_sha256, "toolchain": toolchain,
        "binary_sha256": expected_binary_sha256}
    need(all(binding[key] == value for key, value in supplied.items()), "exact original parent native commitments required")
    supplied_raw = canonical(supplied)
    native = _parent_native_owner(evidence, reviewed_version=reviewed_version)
    contents, inventory = native._reviewed_retained_packet(packet_files)
    need(inventory == binding["packet_inventory"]
         and hashlib.sha256(native._wire(inventory, native.MAX_RECORD_BYTES)).hexdigest() == expected_packet_sha256,
         "exact original parent77-file packet required")
    original = native._live_original_inputs(preparation, graph, admission, args, kwargs)

    def fence():
        need(_reviewed_parent_capability_entry(handle) is entry, "same original parent capability required")
        _reviewed_parent_guard(); _parent_native_owner(evidence, reviewed_version=reviewed_version)
        need(native._live_original_inputs(preparation, graph, admission, args, kwargs) == original,
             "current reviewed inputs changed during admitted parent replay")
        need(canonical(supplied) == supplied_raw
             and native._reviewed_retained_packet(packet_files)[1] == binding["packet_inventory"],
             "original parent packet or expected commitments changed during replay")
        for pin in (evidence["request"], evidence["plan"], evidence["wait"], evidence["closeout"], evidence["durability"]):
            checked_descriptor(pin)
        for path, size, sha, device, inode in evidence["inventory"]:
            artifact = Path(evidence["attempt_directory"]) / path
            observed = descriptor(artifact); info = artifact.lstat()
            need((observed["bytes"], observed["sha256"], info.st_dev, info.st_ino) == (size, sha, device, inode),
                 "original parent durable native artifact changed")
    load = (native.load_reviewed_norm_preservation_native_v2 if reviewed_version == 2
            else native.load_reviewed_norm_preservation_native_v3)
    validate = (native.validate_reviewed_norm_preservation_native_v2 if reviewed_version == 2
                else native.validate_reviewed_norm_preservation_native_v3)
    replay = load(packet_files, preparation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_preservation_sha256=expected_preservation_sha256,
        expected_graph_sha256=expected_graph_sha256, expected_admission_sha256=expected_admission_sha256,
        toolchain=toolchain, **kwargs)
    fence()
    receipt = native._parse(contents["build-receipt.json"], native.MAX_RECEIPT_BYTES)
    need(native._sha(native._wire(receipt["reviewed_execution_evidence"], native.MAX_RECEIPT_BYTES))
         == binding["execution_projection_sha256"], "original parent execution projection differs")
    checked = validate(preparation, graph, admission, *args,
        expected_preservation_sha256=expected_preservation_sha256, expected_graph_sha256=expected_graph_sha256,
        expected_admission_sha256=expected_admission_sha256, toolchain=toolchain, **kwargs)
    fence()
    persisted = native._live_read_persisted_packet(Path(evidence["attempt_directory"]) / binding["packet_root"],
        checked, receipt, native._owners()[1], reviewed_version=reviewed_version)
    need(tuple(sorted(contents.items())) == persisted, "closed original durable native packet differs")
    fence()
    result = native._owners()[1].contracts.seal_record({**{key: value for key, value in replay.items()
        if key not in {"schema", "profile", "record_cid", "cid_scope", "missing_prerequisites"}},
        "schema": ("reviewed-preservation-admitted-parent-success/v1" if reviewed_version == 2
                   else "reviewed-preservation-admitted-parent-success/v3"),
        "profile": ("original-admitted-parent-exact-reviewed-native-custody/v1" if reviewed_version == 2
                    else "original-admitted-parent-exact-reviewed-native-custody/v3"),
        "original_success_authenticated": True, "native_execution_authenticated": True,
        "original_admitted_parent_custody_authenticated": True, "native_execution_performed": False,
        "olean_kernel_rechecked": False, "release_ready": False, "Lean_admitted": False,
        "training_mask": 0, "evaluation_gold_mask": 0, "authority": dict(native.AUTHORITY),
        "parent_request_sha256": evidence["request"]["sha256"],
        "parent_closeout_sha256": evidence["closeout"]["sha256"],
        "parent_durability_sha256": evidence["durability"]["sha256"],
        "missing_prerequisites": ["Reviewed release-native consumer and corpus admission remain separate.",
            "Source meaning, source edition, legal applicability and legal review remain separate."],
        "scope": "Original Source/native/process/storage parent custody for this exact packet and current reviewed replay; no new execution, legal truth, training/gold or release admission."})
    fence()
    return result, fence

def _joined_literal_signature_v1(value):
    """Bounded native scalar/code metadata only; never arbitrary equality."""
    remaining = [200000, 8 * 1024**2]

    def visit(item, depth):
        remaining[0] -= 1
        need(remaining[0] >= 0 and depth <= 64, "bounded joined identity metadata required")
        kind = type(item)
        if item is None:
            return ("none",)
        if item is Ellipsis:
            return ("ellipsis",)
        if kind is bool:
            return ("bool", item)
        if kind is int:
            need(item.bit_length() <= 4096, "bounded joined identity integer required")
            return ("int", item)
        if any(kind is allowed for allowed in (str, bytes)):
            remaining[1] -= len(item.encode("utf-8")) if kind is str else len(item)
            need(remaining[1] >= 0, "bounded joined identity literal bytes required")
            return (kind.__name__, item)
        if kind is float:
            return ("float", item.hex())
        if kind is complex:
            return ("complex", item.real.hex(), item.imag.hex())
        if any(kind is allowed for allowed in (tuple, list, set, frozenset)):
            need(len(item) <= remaining[0], "bounded joined identity population required")
            children = tuple(visit(child, depth + 1) for child in item)
            if any(kind is allowed for allowed in (set, frozenset)):
                children = tuple(sorted(children))
            return (kind.__name__, children)
        if kind is dict:
            need(len(item) <= remaining[0], "bounded joined identity mapping required")
            return ("dict", tuple((visit(key, depth + 1), visit(child, depth + 1))
                                  for key, child in item.items()))
        if kind is types.CodeType:
            fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
                "co_stacksize", "co_flags", "co_code", "co_consts", "co_names", "co_varnames",
                "co_freevars", "co_cellvars", "co_filename", "co_name", "co_qualname",
                "co_firstlineno", "co_linetable", "co_exceptiontable")
            return ("code", tuple((name, visit(getattr(item, name), depth + 1)) for name in fields))
        raise ValueError("ordinary native joined identity metadata required")

    return visit(value, 0)

def _same_joined_owner_binding_v1(current, expected):
    # Bindings contain private strong refs. Compare only bounded scalar signatures.
    if not (type(current) is tuple and len(current) == 2
            and type(expected) is tuple and len(expected) == 2):
        return False
    return (_joined_literal_signature_v1(current[0])
            == _joined_literal_signature_v1(expected[0]))

def _joined_guard_function_binding_v1(function):
    need(type(function) is types.FunctionType, "ordinary joined guard function required")
    cells, references = [], [function, function.__code__, function.__defaults__,
        function.__kwdefaults__, function.__globals__, function.__dict__]
    for cell in function.__closure__ or ():
        references.append(cell)
        try:
            value = cell.cell_contents
        except ValueError:
            cells.append((id(cell), "empty"))
        else:
            references.append(value); cells.append((id(cell), id(value)))
    kwargs = function.__kwdefaults__
    need(kwargs is None or type(kwargs) is dict and all(type(key) is str for key in kwargs),
         "ordinary joined guard keyword defaults required")
    entries = () if kwargs is None else tuple(sorted(kwargs.items()))
    references.extend(value for _key, value in entries)
    return ((id(function), id(function.__code__), id(function.__defaults__),
        id(function.__kwdefaults__), id(function.__globals__), id(function.__dict__),
        tuple((key, id(value)) for key, value in entries), tuple(cells)), tuple(references))

def _joined_contracts_primitive_binding_v1(contracts, source_raw, source_path):
    """Authenticate new helper code, defaults, aliases and registry from Source.

    Compilation obtains metadata only and executes no module. The unchanged
    Native guard is not claimed as coverage of the added primitive or registry.
    """
    import ast
    import importlib
    tree = ast.parse(source_raw, filename=str(source_path))
    _parent_source_builtin_bindings_v1(vars(contracts), tree)
    compiled = _ORIGINAL_COMPILE(source_raw, str(source_path), "exec",
                                 dont_inherit=True, optimize=sys.flags.optimize)
    pending, codes = [compiled], {}
    while pending:
        code = pending.pop()
        need(len(codes) <= 4096, "bounded primitive source code tree required")
        codes[code.co_qualname] = code
        pending.extend(value for value in code.co_consts if type(value) is types.CodeType)
    namespace = vars(contracts)
    need(all(type(name) is str for name in namespace), "ordinary primitive namespace keys required")
    names = ("_loaded_owner_shape_v1", "_loaded_owner_state_v1",
        "_loaded_owner_guard_identity_v1", "_capture_loaded_owner_guard_v1",
        "_loaded_owner_guard_binding_v1", "_loaded_owner_registry_binding_v1")
    selected = {name: value for name, value in namespace.items()
        if name.startswith(("_loaded_owner_", "_capture_loaded_owner_"))
        and type(value) is types.FunctionType}
    need(set(selected) == set(names) and "_new_loaded_owner_registry_v1" not in namespace,
         "complete original non-resettable loaded-owner primitive required")
    signatures, references, registry_objects = [], [compiled, contracts], []
    for name in names:
        function = selected[name]
        qualified = (name if name in codes
                     else "_new_loaded_owner_registry_v1.<locals>." + name)
        expected = codes.get(qualified)
        need(type(expected) is types.CodeType and function.__globals__ is namespace
             and _joined_literal_signature_v1(function.__code__) == _joined_literal_signature_v1(expected),
             "loaded-owner primitive code differs from independently pinned Source")
        nodes = [node for node in ast.walk(tree)
            if type(node) is ast.FunctionDef and node.name == name]
        need(len(nodes) == 1, "unambiguous primitive declaration required")
        node = nodes[0]
        defaults = tuple(ast.literal_eval(value) for value in node.args.defaults)
        current_defaults = () if function.__defaults__ is None else function.__defaults__
        need(type(current_defaults) is tuple
             and _joined_literal_signature_v1(current_defaults) == _joined_literal_signature_v1(defaults),
             "loaded-owner primitive positional defaults differ")
        wanted_kwargs = {argument.arg: ast.literal_eval(value)
            for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
            if value is not None}
        current_kwargs = {} if function.__kwdefaults__ is None else function.__kwdefaults__
        need(type(current_kwargs) is dict
             and _joined_literal_signature_v1(current_kwargs) == _joined_literal_signature_v1(wanted_kwargs),
             "loaded-owner primitive keyword defaults differ")
        cells = function.__closure__ or ()
        need(len(cells) == len(expected.co_freevars), "exact primitive closure shape required")
        for cell_name, cell in zip(expected.co_freevars, cells, strict=True):
            try:
                value = cell.cell_contents
            except ValueError as error:
                raise ValueError("empty original primitive closure refused") from error
            if cell_name == "registrations":
                need(type(value) is dict, "original private registry identity required")
                registry_objects.append(value)
            else:
                need(cell_name in selected and value is selected[cell_name],
                     "original primitive factory closure changed")
        signature, refs = _joined_guard_function_binding_v1(function)
        signatures.append((name, signature)); references.extend(refs)
    need(len(registry_objects) == 3 and all(value is registry_objects[0] for value in registry_objects),
         "one original private primitive registry required")
    standard = {"dataclasses", "re", "sys", "types", "importlib.machinery", "pathlib",
                "os", "json", "hashlib", "stat", "builtins", "typing"}
    aliases = []
    for node in tree.body:
        if type(node) is ast.Import:
            for item in node.names:
                bound = item.asname or item.name.partition(".")[0]
                if item.name in standard:
                    target = importlib.import_module(item.name if item.asname else bound)
                    need(namespace.get(bound) is target, "loaded-owner runtime module alias changed")
                    aliases.append((bound, id(target))); references.append(target)
        elif type(node) is ast.ImportFrom and node.module in standard:
            owner = importlib.import_module(node.module)
            for item in node.names:
                bound = item.asname or item.name
                target = vars(owner).get(item.name)
                need(target is not None and namespace.get(bound) is target,
                     "loaded-owner runtime type/function alias changed")
                aliases.append((bound, id(target))); references.append(target)
    literal_flags = []
    for node in tree.body:
        if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Name:
            name = node.targets[0].id
            if name.startswith("_LOADED_OWNER_"):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                need(_joined_literal_signature_v1(namespace.get(name)) == _joined_literal_signature_v1(value),
                     "loaded-owner primitive bound/flag changed")
                literal_flags.append((name, _joined_literal_signature_v1(value)))
    for stem, class_name in (("FIELD", "_LOADED_OWNER_FIELD_V1"), ("PARAM", "_LOADED_OWNER_PARAMS_V1")):
        owner = namespace[class_name]
        fields = type.__getattribute__(owner, "__dict__")
        wanted = tuple((name, fields[name]) for name in fields["__slots__"] if name != "__weakref__")
        observed = namespace.get("_LOADED_OWNER_" + stem + "_SLOTS_V1")
        need(type(observed) is tuple and len(observed) == len(wanted)
             and all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                 and pair[0] == name and pair[1] is descriptor
                 for pair, (name, descriptor) in zip(observed, wanted, strict=True)),
             "original native dataclass slot table changed")
        aliases.append((stem + "_SLOTS", id(observed))); references.extend((observed, *wanted))
    registry = selected["_loaded_owner_registry_binding_v1"]()
    need(type(registry) is tuple and len(registry) == 2
         and type(registry[0]) is tuple and registry[0] == (id(registry_objects[0]),)
         and type(registry[1]) is tuple and len(registry[1]) == 2
         and registry[1][0] is selected["_capture_loaded_owner_guard_v1"]
         and registry[1][1] is selected["_loaded_owner_guard_binding_v1"],
         "original private registration anchor required")
    references.extend(registry[1])
    return ((tuple(signatures), tuple(aliases), tuple(literal_flags), registry[0]), tuple(references))

def _parent_release_owner_binding_v3(evidence):
    """Resolve only the fixed R owners, independently pinned before the build."""
    import importlib
    expected_modules = (
        "ipfs_datasets_py.logic.autoformal.legal_release_coordinator_v1",
        "ipfs_datasets_py.logic.autoformal.corpus_contracts",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.uscode_dataset",
        "ipfs_datasets_py.logic.autoformal.uscode_ingest",
        "ipfs_datasets_py.logic.autoformal.legal_entity_index_intake",
        "ipfs_datasets_py.logic.autoformal.constitution_inventory",
        "ipfs_datasets_py.logic.autoformal.legal_parent_text_views",
        "ipfs_datasets_py.logic.autoformal.meta_ontology")
    need(type(_JOIN_RELEASE_EXPECTED_MODULES_V1) is tuple
         and len(_JOIN_RELEASE_EXPECTED_MODULES_V1) == len(expected_modules)
         and all(type(a) is str and a == b
                 for a, b in zip(_JOIN_RELEASE_EXPECTED_MODULES_V1, expected_modules, strict=True)),
         "fixed joined-owner source population changed")
    root = Path(evidence["datasets_root"])
    source_pins = evidence["source_pins"]
    need(type(source_pins) is dict and all(type(key) is str and type(value) is str
                                         for key, value in source_pins.items()),
         "ordinary independently pinned joined sources required")
    for name in expected_modules:
        source_path = root.joinpath(*name.split(".")).with_suffix(".py")
        pin = source_pins.get(str(source_path))
        need(type(pin) is str and _SHA.fullmatch(pin),
             "joined owner must be independently pinned before Native issuance")
    protocol = evidence.get("reviewed_version", 3)
    need(type(protocol) is int and protocol in (3, 4), "explicit joined parent domain required")
    native = _parent_native_owner(evidence, reviewed_version=protocol)
    contracts = native._owners()[1].contracts
    release = importlib.import_module(expected_modules[0])
    need(type(release) is types.ModuleType, "canonical release module required")
    release_namespace = vars(release)
    need(type(release_namespace) is dict and all(type(key) is str for key in release_namespace),
         "ordinary joined coordinator namespace keys required")
    modules = release_namespace.get("_JOIN_RELEASE_OWNER_MODULES_V1")
    need(type(modules) is tuple and len(modules) == len(expected_modules)
         and all(type(module) is types.ModuleType for module in modules),
         "fixed original joined-owner population required")
    source_rows = []
    identity_keys = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
                     "st_size", "st_mtime_ns", "st_ctime_ns")
    for module, name in zip(modules, expected_modules, strict=True):
        namespace = vars(module)
        need(type(namespace) is dict and all(type(key) is str for key in namespace),
             "ordinary joined-owner namespace keys required")
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        spec = namespace.get("__spec__")
        need(type(spec) is importlib.machinery.ModuleSpec,
             "ordinary joined-owner module spec required")
        spec_fields = object.__getattribute__(spec, "__dict__")
        need(type(spec_fields) is dict and all(type(key) is str for key in spec_fields),
             "ordinary joined-owner module spec fields required")
        need(type(namespace.get("__name__")) is str and namespace["__name__"] == name
             and sys.modules.get(name) is module
             and type(namespace.get("__file__")) is str and namespace["__file__"] == str(path)
             and type(spec_fields.get("origin")) is str and spec_fields["origin"] == str(path)
             and type(spec_fields.get("name")) is str and spec_fields["name"] == name
             and path.resolve(strict=True) == path,
             "canonical joined-owner module identity required")
        expected = evidence["source_pins"].get(str(path))
        need(type(expected) is str and _SHA.fullmatch(expected),
             "joined owner must be independently pinned before Native issuance")
        before = path.lstat()
        raw = ordinary_bytes(path, 4 * 1024**2)
        identity = tuple(getattr(before, key) for key in identity_keys)
        need(identity == tuple(getattr(path.lstat(), key) for key in identity_keys)
             and hashlib.sha256(raw).hexdigest() == expected, "joined-owner Source generation changed")
        source_rows.append((name, module, path, expected, raw, identity))
    need(modules[1] is contracts, "joined primitive is not the canonical contracts owner")
    primitive = _joined_contracts_primitive_binding_v1(contracts, source_rows[1][4], source_rows[1][2])
    signatures, references = [], [release, modules, native, native._LIVE_NATIVE_FUNCTIONS, *primitive[1]]
    for name, module, path, sha, raw, identity in source_rows:
        verifier = vars(module).get("_JOIN_LOADED_OWNER_GUARD_V1")
        binding = contracts._loaded_owner_guard_binding_v1(vars(module), verifier)
        need(type(binding) is tuple and len(binding) == 2, "original owner registration required")
        verifier()
        signatures.append((name, sha, identity, binding[0]))
        references.extend((module, verifier, *binding[1]))
    native_signatures = tuple((name, _joined_guard_function_binding_v1(function)[0])
        for name, function in ((key, vars(native).get(key)) for key, _state in native._LIVE_NATIVE_FUNCTIONS))
    return release, ((id(release), id(modules), id(native), native_signatures,
        primitive[0], tuple(signatures)), tuple(references))

def _reviewed_parent_joined_opening_v3(policy):
    operation = policy["operation"]
    root = Path(policy["datasets_root"])
    path = str(root / "ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py")
    if _reviewed_parent_version(policy) not in (3, 4):
        return None
    # Explicit existing-field opt-in: a direct coordinator descriptor selects
    # this implementation join; Native receipt producer pins keep their scope.
    sources = policy["source_descriptors"]
    if not any(pin["path"] == path for pin in sources):
        return None
    pins = {pin["path"]: pin["sha256"] for pin in (
        *policy["source_descriptors"], *policy["validation"]["import_sources"],
        *policy["validation"]["trusted_tests"], policy["validation"]["pytest_module"])}
    _release, binding = _parent_release_owner_binding_v3({
        "datasets_root": str(root), "operation": operation, "source_pins": pins,
        **({"reviewed_version": 4} if _reviewed_parent_version(policy) == 4 else {})})
    return binding

def _joined_closing_note_v1(error, message):
    # Native exception dictionary access avoids arbitrary __str__/attribute hooks.
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
        previous = dict.get(namespace, "__notes__")
        notes = (list(previous) if type(previous) is list
                 and len(previous) <= 32 and all(type(note) is str for note in previous) else [])
        notes.append(message)
        dict.__setitem__(namespace, "__notes__", notes)
    except BaseException:
        # Diagnostic decoration must never interrupt remaining closing checks.
        pass

def _joined_complete_checks_v1(checks, *, primary=None, _closing_failures=None):
    """Run independent final custody checks without hiding prior failures."""
    failures = [] if _closing_failures is None else _closing_failures
    need(type(failures) is list and len(failures) <= 32,
         "bounded private joined closing failure state required")
    for label, check in checks:
        try:
            check()
        except BaseException as error:
            if failures:
                _joined_closing_note_v1(failures[0], "additional joined closing failure in " + label)
            if len(failures) < 32:
                failures.append(error)
    if failures:
        failure = failures[0]
        if primary is not None and primary is not failure:
            _joined_closing_note_v1(failure, "prior joined replay failure retained as cause")
            raise failure from primary
        raise failure

def load_parent_verified_uscode_physical_closure_v3(handle, request_raw, *, expected_request_sha256):
    """Join one actual original parent to one current physical replay.

    The returned value is an A-owned engineering diagnostic, never a portable
    capability, legal review, corpus admission, or authorization to run tools.
    """
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    entry = _reviewed_parent_capability_entry(handle)
    _reviewed_parent_guard()
    evidence = json_object(entry[4])
    need(evidence.get("reviewed_version", 2) == 3 and len(entry) == 6 and entry[5] is not None,
         "original parent was not issued with this joined V3 owner binding")
    need(type(request_raw) is bytes and type(expected_request_sha256) is str
         and _SHA.fullmatch(expected_request_sha256)
         and hashlib.sha256(request_raw).hexdigest() == expected_request_sha256,
         "exact independently pinned physical request bytes required")
    release, observed_binding = _parent_release_owner_binding_v3(evidence)
    need(_same_joined_owner_binding_v1(observed_binding, entry[5]),
         "joined owner generation differs from original parent issuance")
    state = {}

    def observe(context):
        need(not state and type(context) is dict and context.get("reviewed_version") == 3
             and context.get("physical_request_sha256") == expected_request_sha256,
             "one exact physical V3 context required")
        selected = context["native"]
        need(type(selected) is dict and set(selected) == {
            "packet_files", "receipt_raw", "expected_packet_sha256", "expected_binary_sha256"}
             and type(selected["receipt_raw"]) is bytes,
             "joined physical replay requires the actual retained Native selection")
        context_snapshot = release._physical_replay_context_snapshot(context)
        pins = context["pins"]
        parent, fence = _capture_parent_verified_reviewed_preservation(
            handle, selected["packet_files"], context["preparation"], context["graph"],
            context["admission"], *context["args"],
            expected_receipt_sha256=hashlib.sha256(selected["receipt_raw"]).hexdigest(),
            expected_packet_sha256=selected["expected_packet_sha256"],
            expected_binary_sha256=selected["expected_binary_sha256"],
            expected_preservation_sha256=pins["preservation"], expected_graph_sha256=pins["graph"],
            expected_admission_sha256=pins["admission"], toolchain=context["toolchain"],
            reviewed_version=3, **context["kwargs"])
        state.update(parent=parent, fence=fence, context=context, context_snapshot=context_snapshot)
        need(release._physical_replay_context_snapshot(context) == context_snapshot,
             "complete physical context changed during the fixed parent observer")

    def owner_fence():
        current, current_binding = _parent_release_owner_binding_v3(evidence)
        need(current is release and _same_joined_owner_binding_v1(current_binding, entry[5]),
             "joined owner changed during physical replay")

    def context_fence():
        if "context" in state:
            owner_fence()
            need(release._physical_replay_context_snapshot(state["context"]) == state["context_snapshot"],
                 "same complete physical context required after finalized traversal")
            release._physical_postclose_fence_v1(state["context"])

    def parent_fence():
        if "fence" in state:
            state["fence"]()
        else:
            _reviewed_parent_capability_entry(handle)
            _reviewed_parent_guard()

    pending_error = None
    closing_failures = []
    try:
        physical = release._validate_uscode_release_physical_closure(
            request_raw, expected_request_sha256=expected_request_sha256,
            reviewed_version=3, _observe_checked_context=observe,
            _native_transport="direct-packet/v1")
    except BaseException as error:
        pending_error = error
        # R is the authenticated fixed traversal; its refusal stays first.
        closing_failures.append(error)
        raise
    finally:
        _joined_complete_checks_v1((
            ("loaded joined owners", owner_fence),
            ("closed physical generation", context_fence),
            ("original parent/current inputs", parent_fence)),
            primary=pending_error, _closing_failures=closing_failures)
    need(set(state) == {"parent", "fence", "context", "context_snapshot"} and type(physical) is dict
         and physical["schema"] == release.CLOSURE_SCHEMA_V3
         and physical["profile"] == release.CLOSURE_PROFILE_V3_DIRECT
         and physical["request_sha256"] == expected_request_sha256,
         "same completed physical replay result required")
    need(all(physical[name] is False for name in (
        "release_ready", "closure_complete", "admitted", "formalized", "proof_verified",
        "source_semantics_verified", "source_authenticity_verified", "source_metadata_arrow_bytes_verified",
        "original_native_parent_custody_verified", "remote_readback_verified",
        "model_or_compiler_execution_performed"))
         and all(physical[name] is True for name in ("local_replay_complete",
             "local_source_cache_entity_bindings_verified", "independent_reviewed_crosswalk_verified",
             "native_receipt_replayed"))
         and "original_admitted_native_parent_custody_not_established" in physical["blockers"]
         and all(physical[name] == evidence["binding"][field] for name, field in (
             ("graph_sha256", "graph_sha256"), ("admission_sha256", "admission_sha256"),
             ("preservation_sha256", "preparation_sha256"))),
         "retained physical report must preserve its authority limits and exact target")
    parent = state["parent"]
    native = _parent_native_owner(evidence, reviewed_version=3)
    result = native._owners()[1].contracts.seal_record({
        "schema": "legal-uscode-live-parent-physical-join/v1",
        "profile": "reviewed-v3-current-parent-and-physical-component/v1",
        "physical_request_sha256": expected_request_sha256,
        "physical_report_sha256": hashlib.sha256(release._canonical(physical)).hexdigest(),
        "retained_parent_report_sha256": digest(parent),
        "parent_request_sha256": parent["parent_request_sha256"],
        "parent_closeout_sha256": parent["parent_closeout_sha256"],
        "parent_durability_sha256": parent["parent_durability_sha256"],
        "graph_sha256": evidence["binding"]["graph_sha256"],
        "admission_sha256": evidence["binding"]["admission_sha256"],
        "preservation_sha256": evidence["binding"]["preparation_sha256"],
        "retained_packet_sha256": evidence["binding"]["packet_sha256"],
        "retained_receipt_sha256": evidence["binding"]["receipt_sha256"],
        "live_original_parent_and_current_physical_replay_joined": True,
        "wire_report_original_parent_flag_unchanged": True,
        "release_ready": False, "closure_complete": False, "admitted": False,
        "formalized": False, "proof_verified": False, "source_semantics_verified": False,
        "source_authenticity_verified": False, "remote_readback_verified": False,
        "native_execution_performed": False, "olean_kernel_rechecked": False,
        "training_mask": 0, "evaluation_gold_mask": 0, "authority": dict(native.AUTHORITY),
        "missing_prerequisites": [
            "Immutable remote Hub readback and actual statutory review remain separate.",
            "Source meaning, source edition, legal applicability and corpus admission remain separate."],
        "scope": "Same-process/thread original admitted Native parent joined to this exact current physical component; no new compiler/model execution or legal authority."})
    _joined_complete_checks_v1((
        ("loaded joined owners", owner_fence),
        ("closed physical generation", context_fence),
        ("original parent/current inputs", parent_fence)),
        _closing_failures=closing_failures)
    # No project callback or file mutation follows this final current-input fence.
    return result

def _reviewed_realization_operation_fields_v1():
    return {"expected_realization_sha256", "expected_physical_request_sha256",
        "instance_identity_profile", "definition_application_profile"}


def _reviewed_realization_data_fields_v1():
    return _reviewed_realization_operation_fields_v1() | {"preservation", "physical_request_raw"}


def _reviewed_parent_realization_inputs_v1(data, operation):
    """Authenticate public DATA commitments, never a capability or legal review."""
    need(type(data) is dict and type(operation) is dict
         and data.get("schema") == "reviewed-definition-realization-fixture-native-inputs/v1"
         and data.get("fixture_only") is True and data.get("provenance_kind") == "fictional_fixture",
         "explicit physical realization fictional DATA required")
    for key in ("expected_realization_sha256", "expected_preservation_sha256", "expected_graph_sha256",
                "expected_admission_sha256", "expected_physical_request_sha256"):
        need(type(data.get(key)) is str and _SHA.fullmatch(data[key]) and data[key] == operation.get(key),
             "physical realization complete input pin differs")
    need(data.get("toolchain") == operation.get("toolchain")
         and data.get("instance_identity_profile") == operation.get("instance_identity_profile") == "reviewed-source-use/v1"
         and data.get("definition_application_profile") == operation.get("definition_application_profile") == "whole-norm-definition-expansion/v1",
         "explicit source-use whole-norm physical profiles required")
    for name, pin in (("preparation", "expected_realization_sha256"), ("preservation", "expected_preservation_sha256")):
        document = data.get(name)
        need(type(document) is dict and set(document) == {"manifest_json", "files"}
             and type(document["manifest_json"]) is str
             and len(document["manifest_json"].encode("utf-8")) <= 1024**2
             and type(document["files"]) is list and 1 <= len(document["files"]) <= 78
             and digest(document) == data[pin], "both complete physical realization preparations must be pinned")
    encoded = data.get("physical_request_raw")
    need(type(encoded) is dict and set(encoded) == {"kind", "base64"} and encoded["kind"] == "bytes"
         and type(encoded["base64"]) is str and len(encoded["base64"]) <= 4 * 1024**2,
         "bounded original physical request byte DATA required")
    try:
        raw = base64.b64decode(encoded["base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("ordinary original physical request byte DATA required") from error
    need(0 < len(raw) <= 4 * 1024**2
         and hashlib.sha256(raw).hexdigest() == data["expected_physical_request_sha256"],
         "original physical request bytes differ from opening pin")
    return raw


def _reviewed_parent_realization_packet_paths_v1(data):
    """Opening inventory from the exact physical2 realization project, no build."""
    _reviewed_parent_realization_inputs_v1(data, data)
    prepared = data["preparation"]
    manifest = json_object(prepared["manifest_json"].encode("utf-8"))
    need(manifest.get("schema") == "legal-reviewed-physical-definition-realization-preparation/v2"
         and manifest.get("profile") == "reviewed-source-use-whole-norm-definition-physical-evidence-realization/v2"
         and type(manifest.get("units")) is list and 3 <= len(manifest["units"]) <= 8,
         "explicit complete physical source-use whole-norm realization required")
    names, hashes, total = [], {}, 0
    for pair in prepared["files"]:
        need(type(pair) is list and len(pair) == 2 and all(type(value) is str for value in pair),
             "ordinary physical realization file pairs required")
        name, body = pair
        need(len(name) <= 256 and re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", name)
             and all(part not in (".", "..") for part in name.split("/")),
             "safe physical realization project path required")
        raw = body.encode("utf-8"); total += len(raw)
        need(total <= 8 * 1024**2, "unchanged complete realization project byte bound required")
        names.append(name); hashes[name] = hashlib.sha256(raw).hexdigest()
    need(names == sorted(set(names)) and "corpus-manifest.json" not in names
         and manifest.get("file_sha256") == hashes, "complete realization project list/hash differs")
    modules = manifest.get("expected_modules")
    need(type(modules) is list and 1 <= len(modules) <= 11
         and all(type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", name)
                 for name in modules) and len(set(modules)) == len(modules),
         "bounded exact physical realization compiled-module population required")
    paths = sorted(["build-receipt.json", "stdout.txt", "stderr.txt", "runner-error.txt", "project/corpus-manifest.json",
        *("project/" + name for name in names), *("executed-inputs/" + name for name in names),
        *("compiled/" + name.replace(".", "/") + ".olean" for name in modules)])
    need(len(paths) <= 172 and len(paths) == len(set(paths)), "complete physical realization packet population required")
    return paths


def checked_reviewed_definition_realization_parent_policy_v1(path, expected_policy_sha256):
    return _checked_reviewed_preservation_parent_policy_core_v3(path, expected_policy_sha256, reviewed_version=4)


def run_reviewed_definition_realization_parent_v1(policy_path, expected_policy_sha256, *, invocation_directory):
    return _run_reviewed_preservation_parent_core_v3(policy_path, expected_policy_sha256,
        invocation_directory=invocation_directory, reviewed_version=4)


def load_parent_verified_reviewed_definition_realization_v1(handle, packet_files, preparation, preservation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_realization_sha256, expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256,
        toolchain, instance_identity_profile, definition_application_profile,
        physical_request_raw, expected_physical_request_sha256, **kwargs):
    result, _fence = _capture_parent_verified_reviewed_definition_realization_v1(
        handle, packet_files, preparation, preservation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, expected_realization_sha256=expected_realization_sha256,
        expected_preservation_sha256=expected_preservation_sha256, expected_graph_sha256=expected_graph_sha256,
        expected_admission_sha256=expected_admission_sha256, toolchain=toolchain,
        instance_identity_profile=instance_identity_profile, definition_application_profile=definition_application_profile,
        physical_request_raw=physical_request_raw, expected_physical_request_sha256=expected_physical_request_sha256, **kwargs)
    return result


def _capture_parent_verified_reviewed_definition_realization_v1(handle, packet_files, preparation, preservation, graph, admission, *args,
        expected_receipt_sha256, expected_packet_sha256, expected_binary_sha256,
        expected_realization_sha256, expected_preservation_sha256, expected_graph_sha256, expected_admission_sha256,
        toolchain, instance_identity_profile, definition_application_profile,
        physical_request_raw, expected_physical_request_sha256, **kwargs):
    """Current replay under the original A handle; retained JSON cannot issue it."""
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    entry = _reviewed_parent_capability_entry(handle)
    _reviewed_parent_guard()
    evidence = json_object(entry[4]); binding = evidence["binding"]; operation = evidence["operation"]
    need(evidence.get("reviewed_version", 2) == 4
         and binding["schema"] == "reviewed-definition-realization-native-binding/v1"
         and evidence["expected_packet_paths"] == operation["expected_packet_paths"] == binding["expected_packet_paths"],
         "original physical realization parent capability required")
    data = json_object(checked_descriptor(operation["native_inputs"]))
    opening_raw = _reviewed_parent_realization_inputs_v1(data, operation)
    need(type(physical_request_raw) is bytes and physical_request_raw == opening_raw
         and expected_physical_request_sha256 == operation["expected_physical_request_sha256"],
         "original physical realization request bytes required")
    supplied = {"receipt_sha256": expected_receipt_sha256, "packet_sha256": expected_packet_sha256,
        "preparation_sha256": expected_realization_sha256, "upstream_preservation_sha256": expected_preservation_sha256,
        "graph_sha256": expected_graph_sha256, "admission_sha256": expected_admission_sha256,
        "physical_request_sha256": expected_physical_request_sha256,
        "instance_identity_profile": instance_identity_profile, "definition_application_profile": definition_application_profile,
        "toolchain": toolchain, "binary_sha256": expected_binary_sha256}
    need(all(binding[key] == value for key, value in supplied.items()), "exact original physical realization commitments required")
    supplied_raw = canonical(supplied)
    native = _parent_native_owner(evidence, reviewed_version=4)
    contents, inventory = native._reviewed_physical_retained_packet_v2(packet_files)
    need(inventory == binding["packet_inventory"]
         and native.reviewed_physical_definition_realization_packet_sha256_v2(packet_files) == expected_packet_sha256,
         "complete original physical realization packet required")
    options = dict(expected_realization_sha256=expected_realization_sha256,
        expected_preservation_sha256=expected_preservation_sha256, expected_graph_sha256=expected_graph_sha256,
        expected_admission_sha256=expected_admission_sha256, toolchain=toolchain,
        instance_identity_profile=instance_identity_profile, definition_application_profile=definition_application_profile,
        physical_request_raw=physical_request_raw, expected_physical_request_sha256=expected_physical_request_sha256)
    original = native._live_original_inputs(preparation, graph, admission, args, kwargs,
        realization_preservation=preservation, expected_upstream_preservation_sha256=expected_preservation_sha256,
        definition_application_profile=definition_application_profile,
        physical_request_raw=physical_request_raw, expected_physical_request_sha256=expected_physical_request_sha256)

    def fence():
        need(_reviewed_parent_capability_entry(handle) is entry, "same original physical realization parent required")
        _reviewed_parent_guard(); _parent_native_owner(evidence, reviewed_version=4)
        need(native._live_original_inputs(preparation, graph, admission, args, kwargs,
            realization_preservation=preservation, expected_upstream_preservation_sha256=expected_preservation_sha256,
            definition_application_profile=definition_application_profile,
            physical_request_raw=physical_request_raw, expected_physical_request_sha256=expected_physical_request_sha256) == original,
            "current physical realization inputs changed during original parent replay")
        need(canonical(supplied) == supplied_raw and native._reviewed_physical_retained_packet_v2(packet_files)[1] == inventory,
             "physical realization packet/expected commitments changed during replay")
        need(_reviewed_parent_realization_inputs_v1(json_object(checked_descriptor(operation["native_inputs"])), operation) == opening_raw,
             "original public physical realization DATA changed")
        for pin in (evidence["request"], evidence["plan"], evidence["wait"], evidence["closeout"], evidence["durability"]):
            checked_descriptor(pin)
        for path, size, sha, device, inode in evidence["inventory"]:
            artifact = Path(evidence["attempt_directory"]) / path
            observed = descriptor(artifact); info = artifact.lstat()
            need((observed["bytes"], observed["sha256"], info.st_dev, info.st_ino) == (size, sha, device, inode),
                 "original physical realization durable artifact changed")

    replay = native.load_reviewed_physical_definition_realization_native_v2(
        packet_files, preparation, preservation, graph, admission, *args,
        expected_receipt_sha256=expected_receipt_sha256, expected_packet_sha256=expected_packet_sha256,
        expected_binary_sha256=expected_binary_sha256, **options, **kwargs)
    fence()
    receipt = native._parse(contents["build-receipt.json"], native.MAX_RECEIPT_BYTES)
    need(native._sha(native._wire(receipt["reviewed_execution_evidence"], native.MAX_RECEIPT_BYTES))
         == binding["execution_projection_sha256"], "original physical realization execution projection differs")
    checked = native.validate_reviewed_physical_definition_realization_native_v2(
        preparation, preservation, graph, admission, *args, **options, **kwargs)
    fence()
    persisted = native._live_read_persisted_packet(Path(evidence["attempt_directory"]) / binding["packet_root"],
        checked, receipt, native._owners()[1], reviewed_version=5,
        instance_identity_profile=instance_identity_profile, definition_application_profile=definition_application_profile)
    need(tuple(sorted(contents.items())) == persisted, "closed original physical realization durable packet differs")
    fence()
    result = native._owners()[1].contracts.seal_record({**{key: value for key, value in replay.items()
        if key not in {"schema", "profile", "record_cid", "cid_scope", "missing_prerequisites"}},
        "schema": "reviewed-definition-realization-admitted-parent-success/v1",
        "profile": "original-admitted-parent-physical-definition-realization-custody/v1",
        "original_success_authenticated": True, "native_execution_authenticated": True,
        "original_admitted_parent_custody_authenticated": True, "native_execution_performed": False,
        "olean_kernel_rechecked": False, "release_ready": False, "Lean_admitted": False,
        "training_mask": 0, "evaluation_gold_mask": 0, "authority": dict(native.AUTHORITY),
        "parent_request_sha256": evidence["request"]["sha256"],
        "parent_closeout_sha256": evidence["closeout"]["sha256"],
        "parent_durability_sha256": evidence["durability"]["sha256"],
        "physical_request_sha256": expected_physical_request_sha256,
        "missing_prerequisites": ["Reviewed release-native consumer and corpus admission remain separate.",
            "Source meaning, source edition, legal applicability and legal review remain separate."],
        "scope": "Original Source/native/process/storage parent custody for this exact physical realization packet and current replay; no new execution, legal truth, training/gold or release admission."})
    fence()
    return result, fence


def load_parent_verified_uscode_physical_realization_closure_v1(handle, request_raw, *, expected_request_sha256):
    """One original A4 parent joined to its immutable source-only physical V4."""
    if (_reviewed_parent_guard is not _REVIEWED_PARENT_GUARD_OBJECT
            or _reviewed_parent_guard.__code__ is not _REVIEWED_PARENT_GUARD_CODE):
        raise ValueError("reviewed parent callable bindings changed: guard")
    entry = _reviewed_parent_capability_entry(handle)
    _reviewed_parent_guard()
    evidence = json_object(entry[4]); operation = evidence["operation"]; binding = evidence["binding"]
    need(evidence.get("reviewed_version", 2) == 4 and len(entry) == 6 and entry[5] is not None,
         "original parent was not issued with this physical realization owner binding")
    data = json_object(checked_descriptor(operation["native_inputs"]))
    opening_raw = _reviewed_parent_realization_inputs_v1(data, operation)
    need(type(request_raw) is bytes and request_raw == opening_raw
         and expected_request_sha256 == operation["expected_physical_request_sha256"],
         "exact original physical realization request bytes required")
    release, observed_binding = _parent_release_owner_binding_v3(evidence)
    need(_same_joined_owner_binding_v1(observed_binding, entry[5]),
         "joined owner generation differs from original physical realization issuance")
    native = _parent_native_owner(evidence, reviewed_version=4)
    shared = native._owners()[4]
    prepared = shared.ReviewedPreservationCorpusPreparation(data["preparation"]["manifest_json"],
        tuple((name, body) for name, body in data["preparation"]["files"]))
    preservation = shared.ReviewedPreservationCorpusPreparation(data["preservation"]["manifest_json"],
        tuple((name, body) for name, body in data["preservation"]["files"]))
    state = {}

    def observe(context):
        need(not state and type(context) is dict and context.get("reviewed_version") == 4
             and context.get("physical_request_sha256") == expected_request_sha256 and context.get("native") is None
             and context.get("instance_identity_profile") == operation["instance_identity_profile"]
             and context.get("definition_application_profile") == operation["definition_application_profile"],
             "one exact source-only physical source-use whole-norm V4 context required")
        pins = context["pins"]
        need(pins["preservation"] == operation["expected_preservation_sha256"]
             and pins["graph"] == operation["expected_graph_sha256"]
             and pins["admission"] == operation["expected_admission_sha256"]
             and native._reviewed_preservation_native_document(context["preparation"]) == data["preservation"],
             "source-only physical context differs from original upstream preparation")
        snapshot = release._physical_replay_context_snapshot(context)
        packet_root = Path(evidence["attempt_directory"]) / binding["packet_root"]
        packet = tuple((row["path"], ordinary_bytes(packet_root / row["path"], 8 * 1024**2))
            for row in binding["packet_inventory"])
        parent, fence = _capture_parent_verified_reviewed_definition_realization_v1(
            handle, packet, prepared, preservation, context["graph"], context["admission"], *context["args"],
            expected_receipt_sha256=binding["receipt_sha256"], expected_packet_sha256=binding["packet_sha256"],
            expected_binary_sha256=binding["binary_sha256"], expected_realization_sha256=operation["expected_realization_sha256"],
            expected_preservation_sha256=pins["preservation"], expected_graph_sha256=pins["graph"],
            expected_admission_sha256=pins["admission"], toolchain=context["toolchain"],
            instance_identity_profile=operation["instance_identity_profile"],
            definition_application_profile=operation["definition_application_profile"],
            physical_request_raw=request_raw, expected_physical_request_sha256=expected_request_sha256, **context["kwargs"])
        state.update(parent=parent, fence=fence, context=context, snapshot=snapshot)
        need(release._physical_replay_context_snapshot(context) == snapshot,
             "complete physical realization context changed during fixed parent observer")

    def owner_fence():
        current, current_binding = _parent_release_owner_binding_v3(evidence)
        need(current is release and _same_joined_owner_binding_v1(current_binding, entry[5]),
             "joined owner changed during physical realization replay")

    def context_fence():
        if "context" in state:
            owner_fence()
            need(release._physical_replay_context_snapshot(state["context"]) == state["snapshot"],
                 "same complete physical realization context required after traversal")
            release._physical_postclose_fence_v1(state["context"])

    def parent_fence():
        if "fence" in state:
            state["fence"]()
        else:
            _reviewed_parent_capability_entry(handle); _reviewed_parent_guard()

    pending_error = None; closing_failures = []
    try:
        physical = release._validate_uscode_release_physical_closure(request_raw,
            expected_request_sha256=expected_request_sha256, reviewed_version=4,
            _observe_checked_context=observe, _native_transport="direct-packet/v1")
    except BaseException as error:
        pending_error = error; closing_failures.append(error)
        raise
    finally:
        _joined_complete_checks_v1((("loaded joined owners", owner_fence),
            ("closed physical realization generation", context_fence),
            ("original physical realization parent/current inputs", parent_fence)),
            primary=pending_error, _closing_failures=closing_failures)
    need(set(state) == {"parent", "fence", "context", "snapshot"} and type(physical) is dict
         and physical["schema"] == "legal-uscode-release-closure-report/v4"
         and physical["profile"] == "reviewed-source-use-whole-norm-physical-source-replay/v1"
         and physical["request_sha256"] == expected_request_sha256,
         "same completed physical realization replay result required")
    need(all(physical[name] is False for name in ("release_ready", "closure_complete", "admitted", "formalized",
        "proof_verified", "source_semantics_verified", "source_authenticity_verified", "source_metadata_arrow_bytes_verified",
        "original_native_parent_custody_verified", "remote_readback_verified", "model_or_compiler_execution_performed"))
         and physical["local_replay_complete"] is True
         and physical["graph_sha256"] == binding["graph_sha256"]
         and physical["admission_sha256"] == binding["admission_sha256"]
         and physical["preservation_sha256"] == binding["upstream_preservation_sha256"],
         "source-only physical report must preserve its authority limits and original target")
    parent = state["parent"]
    result = native._owners()[1].contracts.seal_record({
        "schema": "legal-uscode-live-parent-physical-realization-join/v1",
        "profile": "reviewed-physical-source-use-whole-norm-current-parent/v1",
        "physical_request_sha256": expected_request_sha256,
        "physical_report_sha256": hashlib.sha256(release._canonical(physical)).hexdigest(),
        "retained_parent_report_sha256": digest(parent),
        "parent_request_sha256": parent["parent_request_sha256"],
        "parent_closeout_sha256": parent["parent_closeout_sha256"],
        "parent_durability_sha256": parent["parent_durability_sha256"],
        "realization_sha256": binding["preparation_sha256"],
        "preservation_sha256": binding["upstream_preservation_sha256"],
        "graph_sha256": binding["graph_sha256"], "admission_sha256": binding["admission_sha256"],
        "retained_packet_sha256": binding["packet_sha256"], "retained_receipt_sha256": binding["receipt_sha256"],
        "live_original_parent_and_current_physical_replay_joined": True,
        "wire_report_original_parent_flag_unchanged": True,
        "release_ready": False, "closure_complete": False, "admitted": False, "formalized": False,
        "proof_verified": False, "source_semantics_verified": False, "source_authenticity_verified": False,
        "remote_readback_verified": False, "native_execution_performed": False, "olean_kernel_rechecked": False,
        "training_mask": 0, "evaluation_gold_mask": 0, "authority": dict(native.AUTHORITY),
        "missing_prerequisites": ["Immutable remote Hub readback and actual statutory review remain separate.",
            "Source meaning, source edition, legal applicability and corpus admission remain separate."],
        "scope": "Same-process/thread original admitted physical Native parent joined to this exact current physical component; no new compiler/model execution or legal authority."})
    _joined_complete_checks_v1((("loaded joined owners", owner_fence),
        ("closed physical realization generation", context_fence),
        ("original physical realization parent/current inputs", parent_fence)), _closing_failures=closing_failures)
    return result


# Capture originals after all definitions; legacy repair profiles never enter
# this new guard. This is cooperative integrity checking, not a Python sandbox.
_REVIEWED_PARENT_SELF_PIN = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_REVIEWED_PARENT_COMPILE = _ORIGINAL_COMPILE
_REVIEWED_PARENT_GUARD_OBJECT = _reviewed_parent_guard
_REVIEWED_PARENT_GUARD_CODE = _reviewed_parent_guard.__code__
_REVIEWED_PARENT_HANDLE_CLASS = VerifiedReviewedPreservationParentSuccess
_REVIEWED_PARENT_FUNCTIONS = tuple((name, _reviewed_parent_function_state(value))
    for name, value in tuple(globals().items()) if type(value) is types.FunctionType and value.__module__ == __name__)
_REVIEWED_PARENT_CLASSES = tuple((name, value, type(value), value.__bases__, frozenset(vars(value)),
    tuple((key, _reviewed_parent_function_state(_reviewed_parent_descriptor_function(method)))
          for key, method in vars(value).items() if type(_reviewed_parent_descriptor_function(method)) is types.FunctionType))
    for name, value in tuple(globals().items()) if isinstance(value, type) and value.__module__ == __name__)
_REVIEWED_PARENT_POPEN_META = type(_ORIGINAL_POPEN)
_REVIEWED_PARENT_POPEN_BASES = getattr(_ORIGINAL_POPEN, "__bases__", None)
_REVIEWED_PARENT_POPEN_KEYS = frozenset(vars(_ORIGINAL_POPEN))
_REVIEWED_PARENT_POPEN_FUNCTIONS = tuple((name, _reviewed_parent_function_state(_reviewed_parent_descriptor_function(value)))
    for name, value in vars(_ORIGINAL_POPEN).items() if type(_reviewed_parent_descriptor_function(value)) is types.FunctionType)

if __name__ == "__main__":
    raise SystemExit(main())

"""Trusted stdlib-only cold validation entrypoint.

No project package is imported before normal candidate search roots and raw
source guards are installed. This script is invoked by absolute path under the
public original resource owner; it performs no resource allocation itself.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
import types
import xml.etree.ElementTree as ET
_ORIGINAL_ASSERTION_ERROR = AssertionError
_ORIGINAL_COMPILE = compile
WORK_SCHEMA = "agent-supervisor/repair-trusted-validation@1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

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

class PinnedSources(importlib.abc.MetaPathFinder):
    """Normal module specifications and package readers; raw bytes precede exec."""
    def __init__(self, pins, roots, normal_namespace_profiles=None):
        self.pins, self.roots, self.executed, self.denials = pins, roots, {}, []
        self.syntax_failures = []
        self.syntax_exception_objects = []
        self.compiled_codes = {}
        profiles = [] if normal_namespace_profiles is None else normal_namespace_profiles
        need(type(profiles) is list and len(profiles) <= 1,
             "finite explicit normal namespace profiles required")
        self.normal_namespace_profiles = {}
        self.normal_namespace_origins = {}
        for profile in profiles:
            need(type(profile) is dict and set(profile) == {
                "fullname", "path", "device", "inode", "initializer_absent"}
                and profile["fullname"] == "ipfs_datasets_py.logic.backends"
                and profile["path"] == str(roots[1] / "ipfs_datasets_py/logic/backends")
                and type(profile["device"]) is type(profile["inode"]) is int
                and profile["initializer_absent"] is True
                and str(Path(profile["path"]) / "process.py") in pins,
                "only the exact declared D backends namespace is permitted")
            self.normal_namespace_profiles[profile["fullname"]] = dict(profile)
        self.verify_normal_namespaces()
        # The original interpreter supplies these boundaries before any project
        # import. PYTHONPATH or later sys.path additions grant no source scope.
        configured = sysconfig.get_paths()
        prefix = Path(sys.prefix).resolve(strict=True)
        runtime = {Path(configured[key]).resolve(strict=True) for key in ("stdlib", "platstdlib")}
        for key in ("purelib", "platlib"):
            location = Path(configured[key]).resolve(strict=True)
            if location == prefix or prefix in location.parents:
                runtime.add(location)
        self.runtime_roots = tuple(sorted(runtime))

    def verify_normal_namespaces(self):
        for profile in self.normal_namespace_profiles.values():
            path = Path(profile["path"])
            for ancestor in (path, *path.parents):
                need(stat.S_ISDIR(ancestor.lstat().st_mode), "normal namespace directory absent or aliased")
            info = path.lstat()
            need((info.st_dev, info.st_ino) == (profile["device"], profile["inode"])
                 and not os.path.lexists(path / "__init__.py"),
                 "declared normal namespace identity or initializer absence changed")
        return dict(self.normal_namespace_origins)

    def remember_code(self, code, origin):
        pending = [code]
        while pending:
            current = pending.pop()
            self.compiled_codes[current] = str(origin)
            pending.extend(value for value in current.co_consts if isinstance(value, types.CodeType))

    def refuse(self, origin, reason):
        self.denials.append({"path": str(origin), "reason": reason})
        raise ImportError(reason)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "transformers", "tensorflow", "jax", "openai", "anthropic",
                "duckdb", "huggingface_hub", "ctypes", "cffi", "multiprocessing"}:
            self.refuse(fullname, "provider/model/database/native-FFI import outside the pure validation profile")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        project_name = fullname in {"ipfs_accelerate_py", "ipfs_datasets_py"} or fullname.startswith(("ipfs_accelerate_py.", "ipfs_datasets_py."))
        if spec is not None and spec.origin is None and fullname in self.normal_namespace_profiles:
            profile = self.normal_namespace_profiles[fullname]
            try:
                self.verify_normal_namespaces()
            except (OSError, ValueError, RuntimeError):
                self.refuse(fullname, "declared normal namespace custody changed")
            if spec.loader is not None or list(spec.submodule_search_locations or ()) != [profile["path"]]:
                self.refuse(fullname, "declared normal namespace search locations changed")
            self.normal_namespace_origins[fullname] = dict(profile)
            return spec
        if spec is None or not isinstance(spec.origin, str) or spec.origin in {"built-in", "frozen"}:
            if project_name:
                self.refuse(fullname, "project package lacks a declared ordinary normal source origin")
            return spec
        origin = Path(spec.origin)
        first_party = any(origin == r or r in origin.parents for r in self.roots)
        if fullname == "ipfs_accelerate_py" or fullname.startswith("ipfs_accelerate_py."):
            if self.roots[0] not in origin.parents:
                self.refuse(origin, "candidate package resolved to the trusted release instead")
        if fullname == "ipfs_datasets_py" or fullname.startswith("ipfs_datasets_py."):
            if self.roots[1] not in origin.parents:
                self.refuse(origin, "D package resolved outside its authenticated canonical root")
        if not first_party and str(origin) not in self.pins:
            try:
                resolved = origin.resolve(strict=True)
            except (OSError, ValueError, RuntimeError):
                self.refuse(origin, "source origin could not be authenticated")
            if any(resolved == allowed or allowed in resolved.parents for allowed in self.runtime_roots):
                return spec
            self.refuse(origin, "undeclared source origin outside authenticated interpreter roots")
        if origin.suffix != ".py" or str(origin) not in self.pins:
            self.refuse(origin, "undeclared first-party source body")
        try:
            raw = ordinary_bytes(origin, 4 * 1024**2)
        except (OSError, ValueError, RuntimeError):
            self.refuse(origin, "first-party source could not be authenticated")
        if hashlib.sha256(raw).hexdigest() != self.pins[str(origin)]:
            self.refuse(origin, "raw first-party source changed")
        owner = self
        class Loader(importlib.machinery.SourceFileLoader):
            def get_code(self, name):
                try:
                    current = ordinary_bytes(origin, 4 * 1024**2)
                except (OSError, ValueError, RuntimeError):
                    owner.refuse(origin, "source could not be reauthenticated before compilation")
                if current != raw:
                    owner.refuse(origin, "source changed before compilation")
                try:
                    code = _ORIGINAL_COMPILE(current, str(origin), "exec", dont_inherit=True)
                    owner.remember_code(code, origin)
                    return code
                except SyntaxError as error:
                    owner.syntax_exception_objects.append(error)
                    owner.syntax_failures.append({"path": str(origin), "filename": error.filename,
                        "bytes": len(current), "sha256": hashlib.sha256(current).hexdigest(),
                        "exception_type": type(error).__name__, "exception_category": "SyntaxError",
                        "line": error.lineno, "offset": error.offset, "body_executed": False,
                        "message": str(error.msg)[:2048], "source_line_prefix": (error.text or "")[:2048],
                        "source_line_sha256": hashlib.sha256((error.text or "").encode()).hexdigest()})
                    raise
            def exec_module(self, module):
                code = self.get_code(module.__name__)
                exec(code, module.__dict__)
                owner.executed[str(origin)] = hashlib.sha256(raw).hexdigest()
        spec.loader = Loader(fullname, str(origin))
        return spec


def original_pytest_failure_attribution(guard, *, error, report_path, nodeid, when, selected_files,
        required_bodies, source_pins, collector_error_type, importer_code, importer_globals, wrapper_pin):
    """Bind live original failed exceptions to exact compiled source objects.

    Implicit exception context and error strings provide no attribution. The
    only collection wrapper accepted is the original pinned Pytest importer
    raising CollectError explicitly from this exact recorded SyntaxError.
    """
    result = {"nodeid": nodeid, "report_path": report_path, "when": when,
        "exception_type": type(error).__name__, "kind": "unattributed", "syntax_witness_indices": [],
        "candidate_traceback": [], "trusted_test_frame": None, "trusted_wrapper_source": None}
    if report_path not in selected_files:
        return result
    indices = [index for index, original in enumerate(guard.syntax_exception_objects) if error is original]
    if indices:
        result.update(kind="original_source_syntax_error", syntax_witness_indices=indices)
        return result
    if type(error) is collector_error_type and wrapper_pin is not None:
        indices = [index for index, original in enumerate(guard.syntax_exception_objects) if error.__cause__ is original]
        frame = error.__traceback__; trusted = False
        for _ in range(128):
            if frame is None:
                break
            if frame.tb_frame.f_code is importer_code and frame.tb_frame.f_globals is importer_globals:
                trusted = True
            frame = frame.tb_next
        if indices and trusted:
            result.update(kind="trusted_pytest_collect_wrapper", syntax_witness_indices=indices,
                trusted_wrapper_source=wrapper_pin)
            return result
    frame = error.__traceback__
    for _ in range(128):
        if frame is None:
            break
        path = guard.compiled_codes.get(frame.tb_frame.f_code)
        if path in required_bodies:
            result["candidate_traceback"].append({"path": path, "sha256": source_pins[path],
                "function": frame.tb_frame.f_code.co_name})
        if path in selected_files:
            result["trusted_test_frame"] = {"path": path, "sha256": source_pins[path]}
        frame = frame.tb_next
    if when == "call" and result["candidate_traceback"]:
        result["kind"] = "changed_candidate_runtime_error"
    elif when == "call" and type(error) is _ORIGINAL_ASSERTION_ERROR and result["trusted_test_frame"] is not None:
        result["kind"] = "trusted_test_assertion"
    return result

def validation_worker(args):
    raw = ordinary_bytes(args.plan, 4 * 1024**2)
    need(hashlib.sha256(raw).hexdigest() == args.plan_sha256, "trusted work plan changed")
    plan = json_object(raw); policy = json_object(checked_descriptor(plan["policy"]))
    need(plan["schema"] == WORK_SCHEMA, "trusted worker scope required")
    root, d, attempt = Path(plan["candidate_root"]), Path(policy["datasets_root"]), Path(plan["attempt_directory"])
    need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.")
                 or name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.") for name in sys.modules),
         "trusted worker must start before any cached project package initialization")
    need(type(plan["source_pins"]) is dict and 0 < len(plan["source_pins"]) <= 4096,
         "finite complete normal source manifest required")
    # This worker, rather than its parent, owns the only sustained source READs
    # across test execution. The resource CLI itself takes no source lease.
    namespace_profiles = plan.get("normal_namespace_profiles", [])
    need(namespace_profiles == policy["validation"].get("normal_namespace_profiles", []),
         "original declared normal namespace profiles differ")
    guard = PinnedSources(plan["source_pins"], [root, d, Path(__file__).resolve().parents[3]], namespace_profiles)
    sys.meta_path.insert(0, guard); sys.path[:0] = [str(root), str(d)]
    os.environ.update(IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease import source_lease
    import pytest
    import _pytest.python as pytest_python
    from _pytest.reports import TestReport, CollectReport
    trusted_importer = pytest_python.importtestmodule
    trusted_importer_code = trusted_importer.__code__
    trusted_importer_globals = trusted_importer.__globals__
    trusted_collector_error = pytest.Collector.CollectError
    need(descriptor(Path(pytest.__file__).absolute()) == policy["validation"]["pytest_module"],
         "actual trusted Pytest origin differs")
    need(sys.executable == policy["python"] and sys.flags.isolated == 1 and sys.dont_write_bytecode,
         "trusted original interpreter startup differs")
    phases, nodes, denied = [], [], []
    raw_compiled = {}
    failed_reports = []
    selected_files = {node.partition("::")[0] for node in plan["selected_checks"]}
    node_map = {}
    active = [False]
    os.environ.update({k: str(attempt / "tmp") for k in ("TMPDIR", "TMP", "TEMP")})
    import tempfile
    tempfile.tempdir = str(attempt / "tmp")
    def normalized(node):
        need(node in node_map, "phase for an uncollected natural case")
        return node_map[node]
    def private_target(name, directory_fd=None):
        if isinstance(name, int):
            info = os.fstat(name)
            path = Path(os.readlink("/proc/self/fd/" + str(name)))
            named = path.lstat()
            need(info.st_uid == os.getuid() and (info.st_dev, info.st_ino) == (named.st_dev, named.st_ino),
                 "owned ordinary descriptor identity required")
        else:
            path = Path(os.fsdecode(name))
            if not path.is_absolute():
                if directory_fd is None or directory_fd == -1:
                    base = Path.cwd()
                else:
                    info = os.fstat(directory_fd)
                    base = Path(os.readlink("/proc/self/fd/" + str(directory_fd)))
                    named = base.lstat()
                    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                         and (info.st_dev, info.st_ino) == (named.st_dev, named.st_ino),
                         "owned directory descriptor identity required")
                path = base / path
        need(path.is_absolute() and ".." not in path.parts
             and (path == attempt or attempt in path.parents), "target is outside the owned attempt")
        return path
    class Evidence:
        def pytest_collection_finish(self, session):
            for item in session.items:
                actual = str(Path(item.path).resolve()) + "::" + item.nodeid.partition("::")[2]
                node_map[item.nodeid] = actual
                nodes.append(actual)
            # Collection may fail on an attributed changed-candidate SyntaxError.
            # Positive/full-triplet claims are checked separately after closure.
        def pytest_runtest_logreport(self, report):
            need(type(report) is TestReport, "original natural Pytest report type required")
            row = {"nodeid": normalized(report.nodeid), "when": report.when, "outcome": report.outcome}
            if report.failed:
                text = str(report.longrepr); raw = text.encode()
                row["diagnostic"] = {"text_prefix": text[:8192], "full_text_bytes": len(raw),
                    "full_text_sha256": hashlib.sha256(raw).hexdigest()}
            phases.append(row)
        def pytest_exception_interact(self, node, call, report):
            need(type(report) in {TestReport, CollectReport} and report.failed and call.excinfo is not None,
                 "actual original Pytest failed report/exception required")
            when = "collect" if type(report) is CollectReport else report.when
            path = str(Path(node.path).resolve())
            nodeid = path + "::" + report.nodeid.partition("::")[2] if when != "collect" else str(report.nodeid)
            wrapper = descriptor(Path(pytest_python.__file__).absolute())
            if (plan["source_pins"].get(wrapper["path"]) != wrapper["sha256"]
                    or guard.executed.get(wrapper["path"]) != wrapper["sha256"]):
                wrapper = None
            failed_reports.append(original_pytest_failure_attribution(guard, error=call.excinfo.value,
                report_path=path, nodeid=nodeid, when=when, selected_files=selected_files,
                required_bodies=set(plan["required_candidate_bodies"]), source_pins=plan["source_pins"],
                collector_error_type=trusted_collector_error, importer_code=trusted_importer_code,
                importer_globals=trusted_importer_globals, wrapper_pin=wrapper))
    def audit(event, values):
        if not active[0]:
            return
        if event == "compile" and isinstance(values[1], str) and values[1] in plan["source_pins"]:
            raw_source = values[0].encode() if type(values[0]) is str else values[0]
            if type(raw_source) is not bytes or hashlib.sha256(raw_source).hexdigest() != plan["source_pins"][values[1]]:
                denied.append("changed-raw-source-compile")
                raise PermissionError("source compile lacks the exact original raw source bytes")
            raw_compiled[values[1]] = plan["source_pins"][values[1]]
        if event == "exec" and isinstance(values[0], types.CodeType) and values[0].co_filename in plan["source_pins"]:
            filename = values[0].co_filename
            if raw_compiled.get(filename) != plan["source_pins"][filename]:
                denied.append("source-exec-without-original-raw-compile")
                raise PermissionError("source execution lacks the actual raw compiler event")
            guard.remember_code(values[0], filename)
        if event in {"subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec",
                     "socket.connect", "socket.bind", "socket.getaddrinfo"}:
            denied.append(event); raise PermissionError("trusted pure validation profile forbids " + event)
        if event == "open" and isinstance(values[0], (str, bytes)):
            path, mode, flags = values; path = Path(os.fsdecode(path)).absolute()
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing and path != attempt and attempt not in path.parents:
                denied.append("outside-owned-write"); raise PermissionError("validation write outside owned attempt")
        if event in {"os.symlink", "os.link", "os.mkfifo"}:
            denied.append(event); raise PermissionError("special fixture is outside this pure validation profile")
        if event in {"os.mkdir", "os.rmdir", "os.remove", "os.unlink", "os.rename", "os.chmod"}:
            pairs = ((values[0], values[2]), (values[1], values[3])) if event == "os.rename" else (
                (values[0], values[2]),) if event in {"os.mkdir", "os.chmod"} else ((values[0], values[1]),)
            for name, directory_fd in pairs:
                try:
                    private_target(name, directory_fd)
                except (OSError, ValueError):
                    denied.append("outside-owned-mutation")
                    raise PermissionError("validation mutation outside owned attempt") from None
        if event in {"os.kill", "os.killpg"}:
            frame = sys._getframe(1)
            original = frame.f_globals.get("_owner_is_alive")
            permitted = (event == "os.kill" and type(values[0]) is type(values[1]) is int
                and values == (os.getpid(), 0) and frame.f_code is getattr(original, "__code__", None)
                and frame.f_code.co_filename == str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py"))
            if not permitted:
                denied.append("foreign-signal"); raise PermissionError("only original private self-liveness signal0 admitted")
    sys.addaudithook(audit)
    closed = False
    stable = False
    entered = 0
    code = None
    failure = None
    cleanup_errors = []
    stack = ExitStack()
    try:
        need(type(_ORIGINAL_COMPILE) is types.BuiltinFunctionType
             and _ORIGINAL_COMPILE.__module__ == "builtins" and _ORIGINAL_COMPILE.__name__ == "compile",
             "original trusted source compiler primitive required")
        stack.enter_context(source_lease(d, mode="read", timeout_seconds=policy["validation"]["timeout_seconds"]))
        entered += 1
        stack.enter_context(source_lease(root, mode="read", timeout_seconds=policy["validation"]["timeout_seconds"]))
        entered += 1
        for path, sha in plan["source_pins"].items():
            need(descriptor(Path(path))["sha256"] == sha, "source changed after READ admission")
        guard.verify_normal_namespaces()
        active[0] = True
        try:
            code = pytest.main(["-c", "/dev/null", "--noconftest", "--capture=sys", "--assert=plain", "--import-mode=importlib",
                "-p", "no:cacheprovider", "--basetemp", str(attempt / "pytest"), "--junitxml", str(attempt / "exports/test-results.xml"),
                *plan["selected_checks"]], plugins=[Evidence()])
        finally:
            for path, sha in plan["source_pins"].items():
                need(descriptor(Path(path))["sha256"] == sha, "source changed during tests")
            guard.verify_normal_namespaces()
            stable = True
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:1000]}
    finally:
        # A failed assertion/compilation never skips the original __exit__ calls.
        # An exceptional exit remains unknown rather than asserting closed=True.
        try:
            stack.close()
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:1000]})
        else:
            closed = entered == 2
        active[0] = False
    expected = Counter((node, phase) for node in plan["selected_checks"] for phase in ("setup", "call", "teardown"))
    denied.extend(guard.denials)
    complete = (nodes == plan["selected_checks"]
        and Counter((row["nodeid"], row["when"]) for row in phases) == expected)
    executed = all(guard.executed.get(path) == plan["source_pins"][path] for path in plan["required_candidate_bodies"])
    junit = {}; junit_error = None
    try:
        suites = ET.fromstring(ordinary_bytes(attempt / "exports/test-results.xml")).findall(".//testsuite")
        junit = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    except (OSError, ValueError, ET.ParseError) as error:
        junit_error = {"type": type(error).__name__, "message": str(error)[:1000]}
    common = closed and stable and failure is None and not cleanup_errors and not denied and junit_error is None
    failed_calls = [row["nodeid"] for row in phases if row["when"] == "call" and row["outcome"] == "failed"]
    call_shape = (complete and bool(failed_calls)
        and all(row["outcome"] == "passed" if row["when"] != "call" else row["outcome"] in {"passed", "failed"} for row in phases)
        and junit == {"tests": plan["expected_cases"], "failures": len(failed_calls), "errors": 0, "skipped": 0}
        and Counter(row["nodeid"] for row in failed_reports if row["when"] == "call") == Counter(failed_calls)
        and all(row["when"] == "call" for row in failed_reports))
    passed = bool(common and code == 0 and complete and executed and all(row["outcome"] == "passed" for row in phases)
        and junit == {"tests": plan["expected_cases"], "failures": 0, "errors": 0, "skipped": 0})
    call_failure = bool(common and code == 1 and call_shape and executed and failed_reports
        and all(row["kind"] in {"trusted_test_assertion", "changed_candidate_runtime_error"} for row in failed_reports))
    syntax_indices = {index for index, row in enumerate(guard.syntax_failures) if row["path"] in plan["required_candidate_bodies"]
        and row["path"] == row["filename"] and row["sha256"] == plan["source_pins"][row["path"]]
        and raw_compiled.get(row["path"]) == row["sha256"] and row["body_executed"] is False
        and row["path"] not in guard.executed and row["exception_category"] == "SyntaxError"}
    attributed_syntax = bool(failed_reports) and all(row["kind"] in {"original_source_syntax_error", "trusted_pytest_collect_wrapper"}
        and row["syntax_witness_indices"] and all(index in syntax_indices for index in row["syntax_witness_indices"])
        for row in failed_reports)
    collection_shape = (not phases and bool(failed_reports) and all(row["when"] == "collect" for row in failed_reports)
        and junit.get("errors") == len(failed_reports) and junit.get("failures") == 0 and junit.get("skipped") == 0)
    syntax_failure = bool(common and attributed_syntax and ((code == 1 and call_shape) or (code == 2 and collection_shape)))
    profile = "trusted_pytest_call_failure" if call_failure else "changed_candidate_source_syntax_error" if syntax_failure else None
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    new_json(attempt / "exports/trusted-validation.json", {"schema": WORK_SCHEMA, "passed": passed,
        "semantic_negative": profile is not None, "negative_profile": profile, "pytest_returncode": int(code) if code is not None else None,
        "source_READ_closed": closed, "source_snapshot_stable": stable, "cleanup_errors": cleanup_errors, "error": failure,
        "plan_sha256": args.plan_sha256, "freeze": plan["freeze"], "source_commitment": plan["source_commitment"],
        "phases": phases, "collected_cases": nodes, "natural_phase_counter_complete": complete,
        "failed_report_attributions": failed_reports, "JUnit": junit, "junit_error": junit_error,
        "guard_denials": denied, "executed_source_buffers": guard.executed,
        "declared_normal_namespace_profiles": namespace_profiles,
        "actual_normal_namespace_origins": guard.normal_namespace_origins,
        "raw_compiled_sources": raw_compiled, "syntax_failures": guard.syntax_failures,
        "original_compile_primitive": {"module": _ORIGINAL_COMPILE.__module__, "name": _ORIGINAL_COMPILE.__name__,
            "builtin": type(_ORIGINAL_COMPILE) is types.BuiltinFunctionType},
        "worker_identity": {"pid": os.getpid(), "birth": fields[19]},
        "full_source_manifest_sha256": hashlib.sha256(canonical(plan["source_pins"])).hexdigest(),
        "full_source_file_count": len(plan["source_pins"]),
        "pytest_module_origin": descriptor(Path(pytest.__file__).absolute()),
        "startup": {"python": sys.executable, "isolated": sys.flags.isolated,
                    "no_bytecode": sys.dont_write_bytecode}})
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    if not sys.dont_write_bytecode or not sys.flags.isolated:
        parser.error("trusted worker requires original -I -B interpreter startup")
    return validation_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

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
import signal
import socket
import subprocess
import time
import sys
import sysconfig
import types
import xml.etree.ElementTree as ET
_ORIGINAL_ASSERTION_ERROR = AssertionError
_ORIGINAL_COMPILE = compile
_ORIGINAL_TRACEBACK_DESCRIPTOR = BaseException.__dict__["__traceback__"]
_ORIGINAL_POPEN = subprocess.Popen
WORK_SCHEMA = "agent-supervisor/repair-trusted-validation@1"
V3_VALIDATION_CAPABILITY = 'agent-supervisor/repair-validation-v3@1'
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

def checked_installed_tool_descriptor(pin):
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

def expected_natural_nodes(plan):
    """Keep legacy exact nodes; v3 authenticates a finite selector expansion."""
    selectors = plan["selected_checks"]
    if "expected_nodeids" not in plan:
        need("selector_case_counts" not in plan, "partial selector expansion is forbidden")
        return selectors
    nodes, counts = plan["expected_nodeids"], plan.get("selector_case_counts")
    need(type(selectors) is list and 0 < len(selectors) <= 128
         and len(set(selectors)) == len(selectors)
         and type(nodes) is list and 0 < len(nodes) <= 1024 and len(set(nodes)) == len(nodes)
         and type(counts) is dict and set(counts) == set(selectors)
         and all(type(count) is int and count > 0 for count in counts.values())
         and sum(counts.values()) == len(nodes) == plan["expected_cases"],
         "closed finite v3 selector expansion required")
    observed = Counter()
    for node in nodes:
        need(type(node) is str and "::" in node and Path(node.partition("::")[0]).is_absolute(),
             "absolute natural case IDs required")
        matches = [selector for selector in selectors if node == selector
                   or node.startswith(selector + "::") or node.startswith(selector + "[")]
        need(len(matches) == 1, "natural case does not belong to one exact selector")
        observed[matches[0]] += 1
    need(dict(observed) == counts, "exact natural-node expansion differs from selector counts")
    return nodes


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
            need((info.st_dev, info.st_ino) == (profile["device"], profile["inode"]),
                 "declared normal namespace identity changed")
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                opened = os.fstat(descriptor)
                need((opened.st_dev, opened.st_ino) == (profile["device"], profile["inode"]),
                     "normal namespace directory changed during open")
                try:
                    os.stat("__init__.py", dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    need(False, "declared normal namespace initializer absence changed")
                after = path.lstat()
                need((after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino),
                     "normal namespace directory changed during absence check")
            finally:
                os.close(descriptor)
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
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        except (OSError, ValueError, RuntimeError) as error:
            self.refuse(fullname, "normal source specification authentication failed: "
                + type(error).__name__ + ": " + str(error)[:1000])
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
                try:
                    exec(code, module.__dict__)
                except BaseException as error:
                    # The interpreter's actual traceback distinguishes a body
                    # that began from an audit refusal before its first opcode.
                    frame = _ORIGINAL_TRACEBACK_DESCRIPTOR.__get__(error, type(error))
                    for _ in range(128):
                        if frame is None:
                            break
                        if frame.tb_frame.f_code is code:
                            owner.executed[str(origin)] = hashlib.sha256(raw).hexdigest()
                            break
                        frame = frame.tb_next
                    raise
                else:
                    owner.executed[str(origin)] = hashlib.sha256(raw).hexdigest()
        spec.loader = Loader(fullname, str(origin))
        return spec


FIXTURE_SCHEMA = 'agent-supervisor/bounded-external-fixtures@1'
CHILDREN_SCHEMA = 'agent-supervisor/bounded-validation-children@1'


def _actual_identity(path):
    path = Path(path)
    for parent in (path, *path.parents):
        need(stat.S_ISDIR(parent.lstat().st_mode), 'ordinary fixture directory ancestry required')
    info = path.lstat()
    need(info.st_uid == os.getuid(), 'fixture directory has foreign custody')
    return {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino}


def _same_selector(node, selector):
    return node == selector or node.startswith(selector + '::') or node.startswith(selector + '[')


def authenticated_fixture_mapping(plan, policy):
    """Map existing committed trusted test bytes, never install assertions."""
    if 'external_fixture_profile' not in plan:
        return None
    original = policy['validation']['external_fixture_profile']
    profile = plan['external_fixture_profile']
    need(type(original) is dict and set(original) == {'schema', 'alias_slots', 'source_slots', 'children'}
         and original['schema'] == FIXTURE_SCHEMA, 'closed original fixture definition required')
    need(type(profile) is dict and set(profile) == set(original), 'closed mapped fixture definition required')
    frozen = json_object(checked_descriptor(plan['freeze']))
    prepared = json_object(checked_descriptor(plan['trusted_test_preparation']))
    need(prepared['binding'] == frozen['binding']
         and hashlib.sha256(canonical(prepared['policy'])).hexdigest() == frozen['binding']['policy_sha256']
         and hashlib.sha256(canonical(prepared['baseline'])).hexdigest() == frozen['baseline_sha256'],
         'original prepared candidate does not bind immutable test baseline')
    root = Path(plan['candidate_root'])
    control = Path(policy['control_release']['root'])
    copies = plan['trusted_test_copies']
    tests = policy['validation']['trusted_tests']
    need(type(copies) is list and len(copies) == len(tests)
         and [row['trusted'] for row in copies] == tests, 'complete original trusted test copy population required')
    mapping = {}
    for row in copies:
        need(type(row) is dict and set(row) == {'trusted', 'actual'}, 'closed trusted test copy binding required')
        trusted, actual = row['trusted'], row['actual']
        path = Path(trusted['path'])
        need(control in path.parents, 'trusted test is outside immutable control release')
        relative = path.relative_to(control).as_posix()
        need(actual['path'] == str(root / relative) and relative in frozen['manifest']
             and relative in prepared['baseline'] and relative not in prepared['policy']['allowed_paths']
             and relative not in frozen['changed_paths']
             and frozen['manifest'][relative] == prepared['baseline'][relative]
             and actual['bytes'] == trusted['bytes'] and actual['sha256'] == trusted['sha256'],
             'executed assertion copy is not an unchanged committed trusted test')
        need(checked_descriptor(actual) == checked_descriptor(trusted), 'actual trusted test copy bytes differ')
        mapping[trusted['path']] = actual

    def selector(value):
        head, separator, suffix = value.partition('::')
        need(head in mapping, 'fixture selector does not name a trusted copied test')
        return mapping[head]['path'] + (separator + suffix if separator else '')

    def slots(values):
        result = []
        for value in values:
            row = dict(value)
            row['selector'] = selector(row['selector'])
            need(row['owner_source']['path'] in mapping
                 and row['owner_source'] in tests, 'fixture owning source is not immutable trusted test data')
            row['owner_source'] = mapping[row['owner_source']['path']]
            result.append(row)
        return result

    original_validation = policy['validation']
    mapped_selectors = [selector(node) for node in original_validation['selected_checks']]
    mapped_nodes = [selector(node) for node in original_validation['expected_nodeids']]
    mapped_counts = {selector(node): count for node, count in original_validation['selector_case_counts'].items()}
    need(plan['selected_checks'] == mapped_selectors and plan['expected_nodeids'] == mapped_nodes
         and plan['selector_case_counts'] == mapped_counts and plan['expected_cases'] == len(mapped_nodes)
         and plan['normal_namespace_profiles'] == original_validation['normal_namespace_profiles']
         and plan['source_commitment'] == frozen['source_commitment']
         and plan['required_candidate_bodies'] == [str(root / relative) for relative in frozen['changed_paths'] if relative.endswith('.py')],
         'mapped exact test/edited-body contract differs from authenticated original policy')
    expected = dict(original)
    expected['alias_slots'] = slots(original['alias_slots'])
    expected['source_slots'] = slots(original['source_slots'])
    children = dict(original['children'])
    for name in ('git_node_selectors', 'frozen_owner_node_selectors'):
        children[name] = [selector(node) for node in children[name]]
    for name in ('cold_control_slots', 'python_file_slots', 'inline_slots'):
        children[name] = slots(children[name])
    installer = dict(children['installer_child_contract'])
    installer['private_stub_node'] = selector(installer['private_stub_node'])
    installer['systemd_allowed_nodes'] = [selector(node) for node in installer['systemd_allowed_nodes']]
    children['installer_child_contract'] = installer
    expected['children'] = children
    expected_external = str(Path(plan['attempt_directory']).parent /
        (Path(plan['attempt_directory']).name + '-external-fixtures'))
    binding = frozen['binding']
    need(binding['namespace'] == 'ptr_lane_0' and binding['board_id'] == 'legal-ir'
         and re.fullmatch('[0-9a-f]{32}', binding['attempt_id']), 'fixed productive candidate binding required')
    expected_request = str(Path(policy['validation']['attempt_root']) /
        'control-requests/legal_ir_unattended_repair/legal-ir' / binding['attempt_id'] / 'execution-request.json')
    need(plan['external_fixture_directory'] == expected_external and plan['public_request_path'] == expected_request
         and profile == expected, 'mapped fixture authority differs from authenticated original definition')
    return profile


class TrustedFixtures:
    """Finite trusted call sites within one original charged EXTERNAL root."""
    def __init__(self, profile, plan, policy, source_guard, *, plan_path, plan_sha256):
        self.profile, self.plan, self.policy, self.guard = profile, plan, policy, source_guard
        self._plan_path, self._plan_sha256 = Path(plan_path), plan_sha256
        self.root = Path(plan['external_fixture_directory'])
        self.identity = _actual_identity(self.root)
        need(stat.S_IMODE(self.root.lstat().st_mode) == 0o700, 'private original fixture root required')
        self.current = None
        self.alias_events, self.source_events, self.child_events = [], [], []
        self.fixture_compiled = {}
        self.processes = []
        self.launching = False
        self.deadline_active = False
        self.children_ready = False
        self.tool_stamps = {}
        self.natural_nodes = set(expected_natural_nodes(plan))
        self.test_pins = {row['actual']['path']: row['actual'] for row in plan['trusted_test_copies']}
        raw = ordinary_bytes(Path(plan['public_request_path']), 1024 * 1024)
        self.request = json_object(raw)
        self.request_sha256 = hashlib.sha256(raw).hexdigest()
        for key, fields in [('alias_slots', {'selector', 'owner_source', 'function', 'line', 'event'}),
                            ('source_slots', {'selector', 'owner_source', 'function', 'line', 'path_suffix', 'bytes', 'sha256'})]:
            slots = profile[key]
            need(type(slots) is list and len(slots) <= 128, 'finite authenticated fixture slots required')
            for slot in slots:
                need(type(slot) is dict and set(slot) == fields
                     and slot['owner_source'] == self.test_pins.get(slot['owner_source']['path'])
                     and type(slot['function']) is str and 0 < len(slot['function']) <= 256
                     and type(slot['line']) is int and 0 < slot['line'] <= 2000000
                     and any(_same_selector(node, slot['selector']) for node in self.natural_nodes),
                     'fixture slot lacks exact immutable owning metadata')
                if key == 'alias_slots':
                    need(slot['event'] in {'os.symlink', 'os.link', 'os.mkfifo'}, 'unreviewed special fixture event')
                else:
                    need(type(slot['bytes']) is int and 0 <= slot['bytes'] <= 4 * 1024**2
                         and type(slot['sha256']) is str and _SHA.fullmatch(slot['sha256'])
                         and type(slot['path_suffix']) is str and len(slot['path_suffix']) <= 1024,
                         'finite exact fixture source payload required')
        request = self.request
        need(request['schema'] == 'owned-validation-request/v3'
             and request['attempt_directory'] == plan['attempt_directory']
             and request['selected_checks'] == plan['selected_checks']
             and request['source_sha256'].get(str(self.plan_path)) == self.plan_sha256,
             'actual original public request does not bind this immutable worker plan')
        need(request['external_fixture_profile'] == {
            'schema': 'owned-validation-external-fixtures/v1',
            'external_fixture_directory': str(self.root), 'max_bytes': 33554432}
             and request['declared_external_writable_paths'] == [str(self.root)]
             and request['storage_bytes'] == 67108864
             and request['normal_namespace_profiles'] == plan['normal_namespace_profiles']
             and request['argv'] == [policy['python'], '-I', '-B', str(Path(__file__).absolute()),
                 '--plan', str(self.plan_path), '--plan-sha256', self.plan_sha256],
             'original native fixture execution profile differs')
        attempt = Path(plan['attempt_directory'])
        charge = json_object(ordinary_bytes(attempt / 'logs/external-fixture-charge.json', 1024 * 1024))
        path_pin = charge['path_receipt']
        need(path_pin['path'] == str(attempt / 'logs/external-fixture-path.json'),
             'original path receipt is outside exact OWNED control location')
        path = json_object(checked_descriptor(path_pin))
        expected_key = 'external-fixtures-' + self.request_sha256[:32]
        need(charge['schema'] == 'owned-validation-external-original-charge/v1'
             and path['schema'] == 'owned-validation-external-original-path/v2'
             and charge['request_sha256'] == path['request_sha256'] == self.request_sha256
             and charge['reservation_id'] == path['reservation_id']
             and charge['accounting_result']['key'] == expected_key
             and type(charge['accounting_result']['bytes']) is int
             and charge['accounting_result']['bytes'] == 33554432
             and path['external_fixture_directory'] == str(self.root)
             and path['root_identity'] == self.identity and path['outside_attempt'] is True,
             'original before-child root/charge receipts differ')
        import math
        start, deadline = path['owner_started_monotonic'], path['owner_deadline_monotonic']
        need(type(start) in (int, float) and type(deadline) in (int, float)
             and math.isfinite(start) and math.isfinite(deadline) and start > 0
             and deadline == start + float(request['timeout_seconds']),
             'original native owner clock is invalid')
        self.owner_started_monotonic, self.owner_deadline_monotonic = start, deadline
        self.child_cleanup_reserve_seconds = 20
        self.child_deadline = deadline - self.child_cleanup_reserve_seconds
        self.original_owner = path['owner']
        self.control_receipts = [descriptor(attempt / 'logs/external-fixture-charge.json'), path_pin]
        self.verify_parent()

    def verify_parent(self):
        owner = self.original_owner
        need(type(owner) is dict and set(owner) == {'pid', 'birth'}
             and type(owner['pid']) is int and owner['pid'] == os.getppid(),
             'worker is not the original native owner direct child')
        fields = Path('/proc/' + str(owner['pid']) + '/stat').read_text().rsplit(')', 1)[1].split()
        need(fields[19] == owner['birth'], 'original native owner PID birth changed')
        for pin in self.control_receipts:
            checked_descriptor(pin)

    @property
    def plan_path(self):
        return self._plan_path

    @property
    def plan_sha256(self):
        return self._plan_sha256

    def refuse(self, reason, path=''):
        self.guard.denials.append({'path': str(path), 'reason': reason, 'nodeid': self.current})
        raise PermissionError(reason)

    def verify(self):
        need(_actual_identity(self.root) == self.identity, 'original external fixture root changed')
        self.verify_parent()
        need(hashlib.sha256(ordinary_bytes(Path(self.plan['public_request_path']), 1024 * 1024)).hexdigest()
             == self.request_sha256, 'actual original public request changed')
        return True

    def set_node(self, node):
        need(node is None or node in self.natural_nodes, 'fixture operation belongs to an uncollected natural case')
        self.verify()
        self.current = node
        control = self.policy['control_release']
        values = {'IPFS_ACCELERATE_VALIDATION_CONTROL_ROOT': control['root'],
            'IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_PATH': control['manifest']['path'],
            'IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_SHA256': control['manifest']['sha256']}
        os.environ.update(values)
        return values

    def verify_control_environment(self):
        control = self.policy['control_release']
        expected = {'IPFS_ACCELERATE_VALIDATION_CONTROL_ROOT': control['root'],
            'IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_PATH': control['manifest']['path'],
            'IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_SHA256': control['manifest']['sha256']}
        need(all(os.environ.get(key) == value for key, value in expected.items()),
             'authenticated immutable control fixture environment changed')

    def private(self, value, directory_fd=None, *, external_only=False):
        if type(value) is int:
            info = os.fstat(value)
            need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), 'owned ordinary regular descriptor required')
            path = Path(os.readlink('/proc/self/fd/' + str(value)))
            named = path.lstat()
            need((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino), 'regular descriptor pathname identity differs')
        else:
            path = Path(os.fsdecode(value))
            if not path.is_absolute():
                if directory_fd is None or directory_fd == -1:
                    base = Path.cwd()
                else:
                    info = os.fstat(directory_fd)
                    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), 'owned directory descriptor required')
                    base = Path(os.readlink('/proc/self/fd/' + str(directory_fd)))
                    named = base.lstat()
                    need((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino), 'directory descriptor pathname changed')
                path = base / path
        allowed = (self.root,) if external_only else (self.root, Path(self.plan['attempt_directory']))
        need(path.is_absolute() and '..' not in path.parts and any(path == root or root in path.parents for root in allowed),
             'fixture path is outside original admitted outputs')
        for parent in reversed(path.parents):
            try:
                info = parent.lstat()
            except FileNotFoundError:
                break
            need(stat.S_ISDIR(info.st_mode), 'fixture path has an aliased parent')
        return path

    def frame_slot(self, slots, *, event=None):
        if self.current is None:
            self.refuse('fixture operation has no actual current natural node')
        frame = sys._getframe(1)
        for _ in range(128):
            if frame is None:
                break
            origin = self.guard.compiled_codes.get(frame.f_code)
            for slot in slots:
                pin = slot['owner_source']
                if (origin == pin['path'] and self.test_pins.get(origin) == pin
                        and self.plan['source_pins'].get(origin) == pin['sha256']
                        and _same_selector(self.current, slot['selector'])
                        and frame.f_code.co_name == slot['function'] and frame.f_lineno == slot['line']
                        and (event is None or slot['event'] == event)):
                    return slot
            frame = frame.f_back
        self.refuse('fixture operation lacks its exact original compiled trusted call site')

    def alias(self, event, values):
        slot = self.frame_slot(self.profile['alias_slots'], event=event)
        if event == 'os.symlink':
            path = self.private(values[1], values[2], external_only=True)
        elif event == 'os.link':
            source = self.private(values[0], values[2], external_only=True)
            path = self.private(values[1], values[3], external_only=True)
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                held, named = os.fstat(fd), source.lstat()
                need(stat.S_ISREG(held.st_mode) and held.st_uid == os.getuid()
                     and (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino),
                     'fixture hardlink source lacks an original regular FD identity')
            finally:
                os.close(fd)
        else:
            path = self.private(values[0], values[2] if len(values) > 2 else None, external_only=True)
        need(len(self.alias_events) < 4096, 'finite original alias event bound exceeded')
        self.alias_events.append({'event': event, 'path': str(path), 'nodeid': self.current,
            'source': slot['owner_source'], 'function': slot['function'], 'line': slot['line']})

    def compile_source(self, filename, raw):
        path = self.private(filename, external_only=True)
        slot = self.frame_slot(self.profile['source_slots'])
        suffix = Path(slot['path_suffix']).parts
        need(suffix and not Path(slot['path_suffix']).is_absolute() and '..' not in suffix
             and path.parts[-len(suffix):] == suffix, 'fixture source path slot differs')
        data = raw.encode() if type(raw) is str else raw
        need(type(data) is bytes and len(data) == slot['bytes']
             and hashlib.sha256(data).hexdigest() == slot['sha256']
             and ordinary_bytes(path, 4 * 1024**2) == data, 'fixture raw source payload differs')
        self.fixture_compiled[str(path)] = slot['sha256']
        self.source_events.append({'event': 'compile', 'path': str(path), 'bytes': len(data),
            'sha256': slot['sha256'], 'nodeid': self.current, 'fixture_only': True})

    def exec_source(self, code):
        path = self.private(code.co_filename, external_only=True)
        digest = self.fixture_compiled.get(str(path))
        need(digest is not None and hashlib.sha256(ordinary_bytes(path, 4 * 1024**2)).hexdigest() == digest,
             'fixture source execution lacks exact prior raw source compiler witness')
        self.guard.remember_code(code, path)
        self.source_events.append({'event': 'exec', 'path': str(path), 'sha256': digest,
            'nodeid': self.current, 'fixture_only': True})

    def setup_children(self):
        children = self.profile['children']
        fields = {'schema', 'python_executable', 'installed_tool_inputs', 'installer_tool',
            'installer_child_contract', 'frozen_child_source', 'landlock_launcher_source',
            'landlock_test_source', 'git_node_selectors', 'frozen_owner_node_selectors',
            'cold_control_slots', 'python_file_slots', 'inline_slots'}
        need(type(children) is dict and set(children) == fields and children['schema'] == CHILDREN_SCHEMA
             and children['python_executable'] == self.policy['python'], 'closed original fixture child contract required')
        self.children = children
        self.tool_stamps = {}
        need(type(children['installed_tool_inputs']) is list and len(children['installed_tool_inputs']) <= 8,
             'finite installed fixture tool population required')
        for pin in children['installed_tool_inputs']:
            need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
                 and type(pin['bytes']) is int and 0 < pin['bytes'] <= 32 * 1024**2,
                 'bounded exact installed fixture tool descriptor required')
            # Installed executables retain the original owner's distinct UID0
            # or current-UID rule; ordinary source files keep their stricter rule.
            checked_installed_tool_descriptor(pin)
            info = Path(pin['path']).lstat()
            self.tool_stamps[pin['path']] = (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        for key in ('git_node_selectors', 'frozen_owner_node_selectors', 'cold_control_slots',
                    'python_file_slots', 'inline_slots'):
            need(type(children[key]) is list and len(children[key]) <= 128, 'finite source-bound child slots required')
        need(children['python_file_slots'] == [] and children['inline_slots'] == [],
             'this finite v3 revision admits no generic Python file or inline fixture slots')
        for slot in children['cold_control_slots']:
            need(type(slot) is dict and set(slot) == {'selector', 'owner_source', 'function', 'line', 'owner_sha256s'}
                 and slot['owner_source'] == self.test_pins.get(slot['owner_source']['path'])
                 and type(slot['owner_sha256s']) is list and 0 < len(slot['owner_sha256s']) <= 8
                 and all(type(value) is str and _SHA.fullmatch(value) for value in slot['owner_sha256s'])
                 and type(slot['line']) is int and slot['line'] > 0
                 and any(_same_selector(node, slot['selector']) for node in self.natural_nodes),
                 'cold control fixture lacks exact trusted source/call-site variants')
        for key in ('frozen_child_source', 'landlock_launcher_source', 'landlock_test_source'):
            need(type(children[key]) is str and len(children[key].encode()) <= 65536,
                 'fixed literal child program bound exceeded')
        relative = 'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py'
        frozen = json_object(checked_descriptor(self.plan['freeze']))
        prepared = json_object(checked_descriptor(self.plan['trusted_test_preparation']))
        path = Path(self.plan['candidate_root']) / relative
        manifest = json_object(checked_descriptor(self.policy['control_release']['manifest']))
        expected_control = str(Path(self.policy['control_release']['root']) / relative)
        rows = [row for row in manifest['files'] if row['path'] == expected_control]
        need(len(rows) == 1 and relative not in prepared['policy']['allowed_paths']
             and prepared['baseline'][relative] == frozen['manifest'][relative]
             and descriptor(path)['sha256'] == rows[0]['sha256']
             and descriptor(path)['bytes'] == rows[0]['bytes'],
             'kernel launcher source is not an unchanged trusted normal module')
        import importlib
        module = importlib.import_module('ipfs_accelerate_py.agent_supervisor.validation.validation_runtime')
        need(Path(module.__file__) == path and self.guard.executed.get(str(path)) == rows[0]['sha256'],
             'kernel boundary module lacks actual raw normal execution')
        self.kernel_source = module._validation_landlock_launcher_source()
        need(self.kernel_source == children['landlock_launcher_source'],
             'maintained original kernel launcher differs from reviewed literal')
        self.kernel_source_pin = descriptor(path)
        self.launching = False
        self.cleaning_children = False
        self.children_ready = True
        self.actual_popen = _ORIGINAL_POPEN
        self.kernel_root_identities = [_actual_identity(Path(self.plan['attempt_directory'])), self.identity]

    def verify_tools(self):
        for spelling, expected in self.tool_stamps.items():
            info = Path(spelling).lstat()
            need((info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
                  info.st_size, info.st_mtime_ns, info.st_ctime_ns) == expected,
                 'original installed fixture tool identity changed')

    def _child_kind(self, argv, kwargs):
        children = self.children
        current = self.current
        cwd = self.private(kwargs['cwd'])
        _actual_identity(cwd)
        env = kwargs.get('env')
        env = dict(os.environ) if env is None else dict(env)
        need(env.get('PATH') == '/usr/bin:/bin' and env.get('LD_PRELOAD', '') == ''
             and 'LD_LIBRARY_PATH' not in env, 'original installed child path changed')
        for name in env:
            need(not name.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) and name not in {
                'GIT_CONFIG_COUNT', 'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG', 'GIT_DIR', 'GIT_WORK_TREE',
                'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES',
                'GIT_INDEX_FILE', 'GIT_TEMPLATE_DIR', 'GIT_SSH', 'GIT_SSH_COMMAND', 'GIT_PROXY_COMMAND'},
                'fixture Git route override')
        installer = children['installer_child_contract']
        if argv[0] in {'git', '/usr/bin/git'}:
            need(any(_same_selector(current, node) for node in children['git_node_selectors']),
                 'private Git outside exact trusted selected cases')
            need(env.get('GIT_CONFIG_NOSYSTEM') == '1' and env.get('GIT_CONFIG_GLOBAL') == '/dev/null',
                 'Git user/system configuration must remain disabled')
            offset = 1
            while offset < len(argv) and argv[offset] == '-c':
                need(offset + 1 < len(argv) and argv[offset + 1] in {
                    'core.hooksPath=/dev/null', 'core.fsmonitor=false',
                    'user.name=Candidate Fixture', 'user.email=candidate@example.invalid'},
                    'unselected fixture Git configuration')
                offset += 2
            need(offset < len(argv) and argv[offset] in {
                'init', 'add', 'commit', 'rev-parse', 'status', 'worktree',
                'read-tree', 'sparse-checkout', 'checkout'}, 'unselected fixture Git verb')
            for value in argv[offset + 1:]:
                need(not value.startswith(('http:', 'https:', 'ssh:', 'git:', 'file:', 'git@')),
                     'fixture Git remote operand')
                need(value not in {'-C', '--git-dir', '--work-tree', '--exec-path', '--namespace', '--config-env'}
                     and not value.startswith(('--git-dir=', '--work-tree=', '--exec-path=', '--namespace=', '--config-env=')),
                     'fixture Git routing option')
                if value.startswith('/'):
                    self.private(value)
                else:
                    need('..' not in Path(value).parts, 'fixture Git relative path escape')
            return 'private_git', env
        if argv[0] in {'systemd-analyze', children['installer_tool']['path']}:
            need(any(_same_selector(current, node) for node in installer['systemd_allowed_nodes'])
                 and len(argv) == 5 and argv[1] == 'condition', 'unselected read-only systemd condition')
            parsed = [argument.partition('=') for argument in argv[2:]]
            need([row[0] for row in parsed] == ['ConditionPathIsDirectory', 'ConditionFileNotEmpty', 'ConditionFileNotEmpty']
                 and all(row[1] == '=' and row[2] for row in parsed), 'exact three unit predicates required')
            release, module, config = [self.private(row[2]) for row in parsed]
            need(module == release / 'ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py'
                 and kwargs.get('stdout') == subprocess.PIPE and kwargs.get('stderr') == subprocess.PIPE
                 and kwargs.get('stdin') is None and not kwargs.get('start_new_session', False),
                 'original read-only systemd predicate capture differs')
            return 'private_readonly_systemd_condition', env
        need(Path(argv[0]).resolve(strict=True) == Path(self.policy['python']).resolve(strict=True),
             'unselected fixture executable')
        if len(argv) == 7 and argv[1:5] == ['-P', '-m', 'ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog', '--apply']:
            need(_same_selector(current, installer['private_stub_node']) and argv[5] == '--config',
                 'private stub is outside exact trusted installer case')
            config = self.private(argv[6])
            body = json_object(ordinary_bytes(config))
            need(body['runtime_release'] == str(cwd), 'private stub cwd differs from original generated release')
            parts = env.get('PYTHONPATH', '').split(os.pathsep)
            need(len(parts) == 2 and parts[1] == str(cwd), 'private stub Python path differs')
            self.private(parts[0])
            need(not os.path.lexists(parts[0]), 'private stub source must remain actually deleted')
            module = cwd / 'ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py'
            need(ordinary_bytes(module).decode() == installer['private_stub_exact_selected_module_source'],
                 'private stub body is not the exact print-only fixture')
            for rel in ('ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/__init__.py',
                        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py'):
                need(ordinary_bytes(cwd / rel) == b'"""Standalone fleet watchdog runtime."""\n',
                     'private stub initializer changed')
            return 'private_installer_dummy_release', env
        if len(argv) == 8 and argv[1:3] == ['-B', '-c'] and argv[3] == children['frozen_child_source']:
            need(any(_same_selector(current, node) for node in children['frozen_owner_node_selectors']),
                 'frozen owner child outside exact trusted resume cases')
            for value in argv[4:7]:
                self.private(value)
            need(argv[7] in {'wait', 'closed', 'validated'}
                 and env.get('PYTHONPATH') == self.plan['candidate_root'] and env.get('PYTHONNOUSERSITE') == '1',
                 'frozen owner child source context differs')
            return 'private_frozen_candidate_owner', env
        if len(argv) == 13 and argv[1:3] == ['-I', '-c'] and argv[3] == children['landlock_launcher_source']:
            need(current.endswith('::test_existing_inherited_boundary_denies_active_and_shared_git_writes'),
                 'Landlock child outside exact trusted owning case')
            data = json_object(argv[4].encode())
            need(set(data) == {'expected_path_identities', 'hardlink_guard', 'writable_paths'},
                 'original Landlock child policy differs')
            for path in data['writable_paths']:
                if path != '/dev/null':
                    self.private(path)
            need(Path(argv[5]).resolve(strict=True) == Path(self.policy['python']).resolve(strict=True)
                 and argv[6:9] == ['-I', '-B', '-c'] and argv[9] == children['landlock_test_source'],
                 'Landlock inner literal differs')
            for path in argv[10:]:
                self.private(path)
            return 'private_existing_landlock_boundary', env
        if len(argv) == 9 and argv[1:3] == ['-I', '-B'] and argv[4] == 'inspect-control':
            slot = self.frame_slot(children['cold_control_slots'])
            need(argv[5] == '--policy' and argv[7] == '--policy-sha256', 'exact cold inspection arguments required')
            source = self.private(argv[3], external_only=True)
            policy_path = self.private(argv[6], external_only=True)
            need(_SHA.fullmatch(argv[8]) and hashlib.sha256(ordinary_bytes(policy_path)).hexdigest() == argv[8],
                 'cold inspection policy bytes differ')
            need(hashlib.sha256(ordinary_bytes(source, 4 * 1024**2)).hexdigest() in slot['owner_sha256s'],
                 'cold inspection owner is not an exact reviewed source variant')
            need(sum(row['nodeid'] == current and row['kind'] == 'private_cold_control' for row in self.child_events) == 0,
                 'cold owning case already consumed its one CLI call')
            return 'private_cold_control', env
        self.refuse('unselected Python fixture program', argv[0])

    def checked_popen(self, argv, *positional, **kwargs):
        try:
            need(self.current in self.natural_nodes and not positional and type(argv) in (list, tuple)
                 and argv and all(type(value) is str for value in argv), 'exact selected typed child argv required')
            need(not kwargs.get('shell') and not any(key in kwargs for key in ('executable', 'preexec_fn', 'pass_fds'))
                 and kwargs.get('process_group') is None and kwargs.get('close_fds', True) is True,
                 'fixture process override or descriptor inheritance refused')
            need(kwargs.get('cwd') is not None, 'explicit private child cwd required')
            self.verify()
            self.verify_tools()
            need(len(self.child_events) < 4096, 'finite original child count exceeded')
            need(time.monotonic() < self.child_deadline, 'original owner cleanup reserve reached')
            for name in ('stdin', 'stdout', 'stderr'):
                value = kwargs.get(name)
                if value in (subprocess.PIPE, subprocess.DEVNULL, subprocess.STDOUT, None):
                    continue
                descriptor_fd = value if type(value) is int else value.fileno()
                self.private(descriptor_fd)
            kind, env = self._child_kind(list(argv), kwargs)
            live = [process for process, row in self.processes if process.poll() is None]
            need(len(live) < 3, 'original child4 envelope includes worker plus at most3 fixture children')
            sequence = len(self.child_events)
            cache = Path(self.plan['attempt_directory']) / ('.unwritten-child-pycache-' + str(sequence))
            try:
                cache.lstat()
            except FileNotFoundError:
                pass
            else:
                need(False, 'child pycache prefix must be genuinely absent')
            env['PYTHONPYCACHEPREFIX'] = str(cache)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            kwargs['env'] = env
            policy = {'writable_paths': [self.plan['attempt_directory'], str(self.root), '/dev/null'],
                'expected_path_identities': [[row['path'], row['device'], row['inode']] for row in self.kernel_root_identities],
                'hardlink_guard': {'writable_roots': [self.plan['attempt_directory'], str(self.root)],
                    'protected_roots': [self.plan['candidate_root'], self.policy['control_release']['root']],
                    'excluded_protected_roots': []}}
            wrapped = [self.policy['python'], '-I', '-B', '-c', self.kernel_source,
                canonical(policy).decode(), *list(argv)]
            journal = Path(self.plan['attempt_directory']) / 'logs' / ('fixture-child-argv-' + str(sequence) + '.json')
            pin = new_json(journal, {'schema': 'agent-supervisor/trusted-fixture-child-argv@1',
                'nodeid': self.current, 'original_argv': list(argv), 'actual_argv': wrapped,
                'original_kernel_source': self.kernel_source_pin, 'write_root_identities': self.kernel_root_identities,
                'source_audit_hooks_inherited_across_exec': False, 'network_permission_granted': False})
            self.launching = True
            try:
                process = self.actual_popen(wrapped, **kwargs)
            finally:
                self.launching = False
            row = {'nodeid': self.current, 'kind': kind, 'pid': process.pid, 'birth': None,
                'argv': pin, 'start_new_session': kwargs.get('start_new_session', False),
                'genuine_original_Popen_returned': True, 'actual_wait_completed': False}
            self.register_child(process, row)
            return process
        except BaseException as error:
            self.guard.denials.append({'path': str(argv[0]) if isinstance(argv, (list, tuple)) and argv else '',
                'reason': 'typed child launch failed: ' + type(error).__name__, 'nodeid': self.current})
            raise

    def child_identity(self, process):
        raw = Path('/proc/' + str(process.pid) + '/stat').read_text()
        fields = raw.rsplit(')', 1)[1].split()
        need(int(raw.partition('(')[0].strip()) == process.pid and int(fields[1]) == os.getpid()
             and fields[19].isdecimal() and int(fields[19]) > 0,
             'fixture process is not this original Popen direct child')
        return {'pid': process.pid, 'birth': fields[19]}

    def register_child(self, process, row):
        # Keep the genuine process handle even if the first metadata read fails.
        self.child_events.append(row)
        self.processes.append((process, row))
        try:
            row['birth'] = self.child_identity(process)['birth']
        except BaseException as error:
            row['initial_identity_error'] = {'type': type(error).__name__, 'message': str(error)[:1000]}
            self.guard.denials.append({'path': str(process.pid),
                'reason': 'original child birth observation failed', 'nodeid': self.current})
            raise

    def close_children(self):
        self.cleaning_children = True
        errors = []
        try:
            for process, row in self.processes:
                try:
                    identity = None
                    if row['birth'] is None and process.returncode is None:
                        # This is a new observed cleanup identity, never a
                        # fabricated original birth. Original Popen ownership
                        # and PPid bind the still-unreaped direct child.
                        try:
                            identity = self.child_identity(process)
                            row['cleanup_observed_identity'] = identity
                        except FileNotFoundError:
                            pass
                    if process.poll() is None:
                        if identity is None:
                            identity = self.child_identity(process)
                        expected = row['birth'] if row['birth'] is not None else identity['birth']
                        need(identity['birth'] == expected,
                             'fixture child PID was reused before cleanup')
                        row['cleanup_observed_identity'] = identity
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    code = process.wait(timeout=1)
                    need(process.returncode == code, 'actual original child wait is unknown')
                    row.update(actual_wait_completed=True, returncode=code)
                except BaseException as error:
                    detail = {'pid': process.pid, 'type': type(error).__name__, 'message': str(error)[:1000]}
                    row['cleanup_error'] = detail
                    errors.append(detail)
            for check in ([self.verify_tools] if self.children_ready else []) + [self.verify]:
                try:
                    check()
                except BaseException as error:
                    errors.append({'type': type(error).__name__, 'message': str(error)[:1000]})
        finally:
            self.cleaning_children = False
        if errors:
            self.guard.denials.append({'path': 'fixture-child-closeout',
                'reason': 'one or more original child closeouts remain unknown', 'nodeid': self.current})
            raise RuntimeError('original fixture child cleanup incomplete: ' + json.dumps(errors, sort_keys=True))

    def signal_allowed(self, event, values):
        if event not in {'os.kill', 'os.killpg'} or values[1] not in (signal.SIGTERM, signal.SIGKILL):
            return False
        for process, row in self.processes:
            if process.pid != values[0] or (event == 'os.killpg' and not row['start_new_session']):
                continue
            try:
                fields = Path('/proc/' + str(process.pid) + '/stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                return process.poll() is not None
            identity = row.get('cleanup_observed_identity') if self.cleaning_children else None
            expected = row['birth'] if row['birth'] is not None else identity['birth'] if identity else None
            return fields[19] == expected
        return False

    def deadline_enter(self):
        remaining = self.child_deadline - time.monotonic()
        need(remaining > 0, 'original owner cleanup reserve already reached')
        def expired(number, frame):
            raise TimeoutError('original native validation owner cleanup reserve reached')
        need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
             'validation cannot replace an existing ancestor alarm')
        self.previous_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        self.deadline_active = True

    def deadline_exit(self):
        if self.deadline_active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous_alarm)
            self.deadline_active = False


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

def original_pytest_no_collectors(error, *, session, selected_checks, failed_reports,
        perform_collect_code, perform_collect_globals, usage_error_type, source_pin):
    """Observe the genuine selected-module collection UsageError, never stderr.

    Pytest clears _notfound in finally; its original traceback retains the
    exact errors list at its original raise. Other usage/selector errors are
    deliberately not semantic negative evidence.
    """
    if type(error) is not usage_error_type or perform_collect_code.co_filename != source_pin["path"]:
        return None
    if descriptor(Path(source_pin["path"])) != source_pin:
        return None
    lines = ordinary_bytes(Path(source_pin["path"])).decode("utf-8").splitlines()
    failed_files = {row["report_path"] for row in failed_reports}
    if not failed_reports or any(row["when"] != "collect" or row["kind"] != "trusted_pytest_collect_wrapper"
            or not row["syntax_witness_indices"] for row in failed_reports):
        return None
    prefix = "found no collectors for "
    frame = error.__traceback__
    while frame is not None:
        live = frame.tb_frame
        if (live.f_code is perform_collect_code and live.f_globals is perform_collect_globals
                and live.f_locals.get("self") is session and live.f_locals.get("args") == selected_checks
                and live.f_locals.get("genitems") is True and 0 < frame.tb_lineno <= len(lines)
                and lines[frame.tb_lineno - 1].strip() == "raise UsageError(*errors)"):
            errors = live.f_locals.get("errors")
            if not (type(errors) is list and errors and tuple(errors) == error.args
                    and all(type(value) is str and value.startswith(prefix) for value in errors)):
                return None
            selectors = [value[len(prefix):] for value in errors]
            expected = [node for node in selected_checks if node.partition("::")[0] in failed_files]
            if (len(selectors) != len(set(selectors)) or selectors != expected
                    or any("::" not in node for node in selectors)
                    or failed_files != {node.partition("::")[0] for node in selectors}):
                return None
            return {"schema": "agent-supervisor/original-pytest-no-collectors@1",
                "exception_type": "UsageError", "collector_source": source_pin,
                "original_raise_line": frame.tb_lineno, "no_collector_selectors": selectors}
        frame = frame.tb_next
    return None

def validation_worker(args):
    need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.")
                 or name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.") for name in sys.modules),
         "trusted worker must start before any cached project package initialization")
    raw = ordinary_bytes(args.plan, 4 * 1024**2)
    need(hashlib.sha256(raw).hexdigest() == args.plan_sha256, "trusted work plan changed")
    plan = json_object(raw); policy = json_object(checked_descriptor(plan["policy"]))
    need(plan["schema"] == WORK_SCHEMA, "trusted worker scope required")
    root, d, attempt = Path(plan["candidate_root"]), Path(policy["datasets_root"]), Path(plan["attempt_directory"])
    natural_nodes = expected_natural_nodes(plan)
    cache_prefix = attempt / ".unwritten-worker-pycache"
    try:
        cache_prefix.lstat()
    except FileNotFoundError:
        pass
    else:
        need(False, "trusted worker pycache prefix must be genuinely absent")
    sys.pycache_prefix = str(cache_prefix)
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
    import _pytest.main as pytest_main
    from _pytest.config.exceptions import UsageError
    from _pytest.reports import TestReport, CollectReport
    trusted_importer = pytest_python.importtestmodule
    trusted_importer_code = trusted_importer.__code__
    trusted_importer_globals = trusted_importer.__globals__
    trusted_collector_error = pytest.Collector.CollectError
    trusted_perform_collect = pytest_main.Session.perform_collect
    trusted_usage_source = descriptor(Path(pytest_main.__file__).absolute())
    need(descriptor(Path(pytest.__file__).absolute()) == policy["validation"]["pytest_module"],
         "actual trusted Pytest origin differs")
    need(sys.executable == policy["python"] and sys.flags.isolated == 1 and sys.dont_write_bytecode,
         "trusted original interpreter startup differs")
    phases, nodes, denied = [], [], []
    fixture_definition = authenticated_fixture_mapping(plan, policy)
    fixtures = None
    private_plugin = None
    original_popen = subprocess.Popen
    original_fifo = os.mkfifo
    def checked_popen(argv, *positional, **kwargs):
        if active[0]:
            if fixtures is None:
                denied.append("unselected-subprocess")
                raise PermissionError("pure validation profile admits no child")
            return fixtures.checked_popen(argv, *positional, **kwargs)
        return _ORIGINAL_POPEN(argv, *positional, **kwargs)
    def checked_fifo(path, mode=0o666, *, dir_fd=None):
        if active[0]:
            if fixtures is None:
                denied.append("os.mkfifo")
                raise PermissionError("pure validation profile admits no FIFO")
            try:
                fixtures.alias("os.mkfifo", (path, mode, dir_fd))
            except BaseException:
                denied.append("unselected-fifo-fixture")
                raise
        return original_fifo(path, mode, dir_fd=dir_fd)
    raw_compiled = {}
    failed_reports = []
    collection_usage = [None]
    diagnostic_bytes_remaining = [1024 * 1024]
    selected_files = {node.partition("::")[0] for node in natural_nodes}
    node_map = {}
    active = [False]
    temporary_root = Path(plan['external_fixture_directory']) / 'tmp' if fixture_definition is not None else attempt / 'tmp'
    os.environ.update({k: str(temporary_root) for k in ("TMPDIR", "TMP", "TEMP")})
    import tempfile
    tempfile.tempdir = str(temporary_root)
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
        @pytest.hookimpl(wrapper=True)
        def pytest_collection(self, session):
            try:
                return (yield)
            except BaseException as error:
                collection_usage[0] = original_pytest_no_collectors(error, session=session,
                    selected_checks=plan["selected_checks"], failed_reports=failed_reports,
                    perform_collect_code=trusted_perform_collect.__code__,
                    perform_collect_globals=trusted_perform_collect.__globals__,
                    usage_error_type=UsageError, source_pin=trusted_usage_source)
                raise
        def pytest_runtest_logstart(self, nodeid, location):
            if fixtures is not None:
                fixtures.set_node(normalized(nodeid))
                need(time.monotonic() < fixtures.child_deadline, "original native owner cleanup reserve reached")
        def pytest_runtest_logfinish(self, nodeid, location):
            if fixtures is not None:
                fixtures.verify_control_environment()
                fixtures.set_node(None)
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
                prefix = raw[:min(8192, diagnostic_bytes_remaining[0])].decode('utf-8', 'ignore')
                diagnostic_bytes_remaining[0] -= len(prefix.encode())
                row["diagnostic"] = {"text_prefix": prefix, "full_text_bytes": len(raw),
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
        if event in {"compile", "exec"}:
            filename = values[1] if event == "compile" else getattr(values[0], "co_filename", None)
            if isinstance(filename, str) and not filename.startswith("<") and filename not in plan["source_pins"]:
                try:
                    path = Path(filename)
                    physical = path.is_absolute() or path.suffix == '.py'
                    if physical:
                        resolved = path.resolve(strict=True)
                        runtime = any(resolved == allowed or allowed in resolved.parents for allowed in guard.runtime_roots)
                        if not runtime:
                            if fixtures is None:
                                raise PermissionError("undeclared physical source compile/exec origin")
                            if event == "compile":
                                fixtures.compile_source(filename, values[0])
                            else:
                                fixtures.exec_source(values[0])
                except BaseException:
                    denied.append("unclosed-physical-source-origin")
                    raise
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
        if event == "subprocess.Popen":
            if fixtures is None or not fixtures.launching:
                denied.append(event)
                raise PermissionError("only the genuine scoped original fixture Popen is permitted")
        if event in {"os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec", "socket.getaddrinfo"}:
            denied.append(event); raise PermissionError("trusted validation profile forbids " + event)
        if event == "socket.__new__" and fixtures is not None and values[1] != socket.AF_UNIX:
            denied.append("nonlocal-socket")
            raise PermissionError("fixture profile grants no network socket family")
        if event in {"socket.connect", "socket.bind", "socket.sendto"}:
            permitted = fixtures is not None and values[0].family == socket.AF_UNIX
            try:
                if permitted and len(values) > 1 and isinstance(values[1], (str, bytes)):
                    fixtures.private(values[1], external_only=True)
            except (OSError, ValueError):
                permitted = False
            if not permitted:
                denied.append(event)
                raise PermissionError("only private local fixture IPC is permitted")
        if event == "open" and isinstance(values[0], (str, bytes)):
            path, mode, flags = values; path = Path(os.fsdecode(path)).absolute()
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing:
                try:
                    if fixtures is not None:
                        fixtures.private(path)
                    elif path != attempt and attempt not in path.parents:
                        raise ValueError("outside owned attempt")
                except (OSError, ValueError):
                    denied.append("outside-owned-write")
                    raise PermissionError("validation write outside original admitted outputs") from None
        if event in {"os.symlink", "os.link", "os.mkfifo"}:
            if fixtures is None:
                denied.append(event); raise PermissionError("special fixture is outside this pure validation profile")
            try:
                fixtures.alias(event, values)
            except BaseException:
                denied.append("unselected-special-fixture")
                raise
        if event in {"os.mkdir", "os.rmdir", "os.remove", "os.unlink", "os.rename", "os.chmod"}:
            pairs = ((values[0], values[2]), (values[1], values[3])) if event == "os.rename" else (
                (values[0], values[2]),) if event in {"os.mkdir", "os.chmod"} else ((values[0], values[1]),)
            for name, directory_fd in pairs:
                try:
                    if fixtures is not None:
                        fixtures.private(name, directory_fd)
                    else:
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
            if fixtures is not None and fixtures.signal_allowed(event, values):
                permitted = True
            if not permitted:
                denied.append("foreign-signal"); raise PermissionError("only original private self-liveness or owned-child cleanup admitted")
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
        if fixture_definition is not None:
            for key in tuple(os.environ):
                if key.startswith('GIT_') or key in {'LD_LIBRARY_PATH', 'BASH_ENV', 'ENV', 'CDPATH',
                    'IPFS_ACCELERATE_META_INDEX_DUCKDB', 'IPFS_ACCELERATE_META_INDEX_DUCKLAKE'}:
                    os.environ.pop(key, None)
            os.environ.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                GIT_ALLOW_PROTOCOL='file', GIT_EXEC_PATH='/usr/lib/git-core', GIT_TERMINAL_PROMPT='0',
                PATH='/usr/bin:/bin', LD_PRELOAD='', PYTHONNOUSERSITE='1', IPFS_ACCEL_SKIP_CORE='1',
                IPFS_DATASETS_AUTO_INSTALL='0', IPFS_AUTO_INSTALL='false',
                IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(Path(plan['external_fixture_directory']) / 'orchestration'),
                IPFS_ACCELERATE_AGENT_HOME=str(Path(plan['external_fixture_directory']) / 'agent-home'))
            fixtures = TrustedFixtures(fixture_definition, plan, policy, guard,
                plan_path=args.plan, plan_sha256=args.plan_sha256)
            fixtures.setup_children()
            import importlib
            private_plugin = importlib.import_module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.validation_pytest')
            need(guard.executed.get(str(Path(private_plugin.__file__))) == plan['source_pins'].get(str(Path(private_plugin.__file__))),
                 'maintained private tmp_path plugin lacks actual raw normal execution')
            fixtures.set_node(None)
            fixtures.deadline_enter()
        subprocess.Popen = checked_popen
        os.mkfifo = checked_fifo
        active[0] = True
        try:
            base = Path(plan['external_fixture_directory']) / 'pytest' if fixtures is not None else attempt / 'pytest'
            plugins = [private_plugin, Evidence()] if private_plugin is not None else [Evidence()]
            # V3 copies live in the actual candidate; legacy selected tests
            # intentionally remain in their authenticated immutable control.
            collection_root = root if fixtures is not None else Path(policy["control_release"]["root"])
            need(all(collection_root in Path(node.partition("::")[0]).parents for node in plan["selected_checks"]),
                 "selected tests exceed the authenticated collection boundary")
            collection_boundary = ["--rootdir", str(collection_root), "--confcutdir", str(collection_root)]
            code = pytest.main(["-c", "/dev/null", "--noconftest", "--capture=sys", "--assert=plain", "--import-mode=importlib",
                "-p", "no:cacheprovider", "--basetemp", str(base), "--log-file", str(attempt / "exports/pytest.log"),
                *collection_boundary, "--junitxml", str(attempt / "exports/test-results.xml"),
                *plan["selected_checks"]], plugins=plugins)
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
        if fixtures is not None:
            try:
                fixtures.deadline_exit()
                fixtures.close_children()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:1000]})
        try:
            stack.close()
        except BaseException as error:
            cleanup_errors.append({"type": type(error).__name__, "message": str(error)[:1000]})
        else:
            closed = True
        active[0] = False
        subprocess.Popen = original_popen
        os.mkfifo = original_fifo
    expected = Counter((node, phase) for node in natural_nodes for phase in ("setup", "call", "teardown"))
    denied.extend(guard.denials)
    complete = (nodes == natural_nodes
        and Counter((row["nodeid"], row["when"]) for row in phases) == expected)
    executed = all(guard.executed.get(path) == plan["source_pins"][path] for path in plan["required_candidate_bodies"])
    junit = {}; junit_error = None
    try:
        suites = ET.fromstring(ordinary_bytes(attempt / "exports/test-results.xml")).findall(".//testsuite")
        junit = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    except (OSError, ValueError, ET.ParseError) as error:
        junit_error = {"type": type(error).__name__, "message": str(error)[:1000]}
    common = closed and entered == 2 and stable and failure is None and not cleanup_errors and not denied and junit_error is None
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
    usage_shape = (code == 4 and collection_shape and not nodes and not complete
        and junit.get("tests") == len(failed_reports) and collection_usage[0] is not None
        and plan["source_pins"].get(trusted_usage_source["path"]) == trusted_usage_source["sha256"]
        and guard.executed.get(trusted_usage_source["path"]) == trusted_usage_source["sha256"])
    syntax_failure = bool(common and attributed_syntax
        and ((code == 1 and call_shape) or (code == 2 and collection_shape) or usage_shape))
    profile = "trusted_pytest_call_failure" if call_failure else "changed_candidate_source_syntax_error" if syntax_failure else None
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    new_json(attempt / "exports/trusted-validation.json", {"schema": WORK_SCHEMA, "passed": passed,
        "semantic_negative": profile is not None, "negative_profile": profile, "pytest_returncode": int(code) if code is not None else None,
        "source_READ_closed": closed, "source_READ_admission_complete": entered == 2,
        "source_snapshot_stable": stable, "cleanup_errors": cleanup_errors, "error": failure,
        "plan_sha256": args.plan_sha256, "freeze": plan["freeze"], "source_commitment": plan["source_commitment"],
        "phases": phases, "collected_cases": nodes, "natural_phase_counter_complete": complete,
        "expected_natural_nodes": natural_nodes, "selector_case_counts": plan.get("selector_case_counts"),
        "failed_report_attributions": failed_reports, "collection_usage_attribution": collection_usage[0], "JUnit": junit, "junit_error": junit_error,
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
        "external_fixture_evidence": None if fixtures is None else {
            "schema": FIXTURE_SCHEMA, "root_identity": fixtures.identity,
            "original_request_sha256": fixtures.request_sha256,
            "original_owner": fixtures.original_owner,
            "original_owner_started_monotonic": fixtures.owner_started_monotonic,
            "original_owner_deadline_monotonic": fixtures.owner_deadline_monotonic,
            "alias_events": fixtures.alias_events, "fixture_source_events": fixtures.source_events,
            "fixture_child_events": fixtures.child_events, "test_or_task_acceptance_asserted": False,
            "ordinary_external_artifact_durability_asserted": False},
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

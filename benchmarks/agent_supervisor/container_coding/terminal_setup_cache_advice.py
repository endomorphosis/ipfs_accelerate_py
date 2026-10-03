"""Explicit, pinned Source384 setup-cache advice; no admission authority.

The policy supports the regular Source384 layout, two public aarch64 Codex
0.158.0 executables and 131 fixed extension and CPU-wheel payloads. Absent selection does nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import stat

POLICY = "source384-native-aarch64-dontneed@1"
POLICIES = (POLICY,)
ROOT = "/opt/ipfs-supervisor"
PROFILE = "source384-5cpu-12gib@1"
MAX_MANIFEST = 8 * 1024**2
MAX_HELPER = 65536
MAX_RECEIPT = 32768
MAX_RESULT = 262144
PREFIX = "source/benchmarks/agent_supervisor/container_coding/"
HELPERS = ("terminal_setup_cache_advice.py", "terminal_setup_cache_files.py",
           "terminal_setup_cache_codex.py", "terminal_setup_cache_libraries.py")
CODEX_PINS = {
    "codex": "0059c73b149a1433b634e26ad8a715e02717a9920665a9cc5676941db03c45cd",
    "codex-code-mode-host": "68237b34d0bc182e99c43ca2197a25120339c154dd3fc29f908c2a1b022efa5b",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _regular_bytes(path, maximum):
    """Read one bounded public artifact; never use this for credentials."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= maximum):
            raise ValueError("bounded singleton public artifact required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
              "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError("public artifact changed during read")
    return raw


def read_selection(path):
    value = json.loads(_regular_bytes(Path(path), MAX_RECEIPT))
    _selection_shape(value)
    return value


def _selection_shape(value):
    if (type(value) is not dict or set(value) != {"schema", "policy", "manifest_sha256"}
            or value["schema"] != "terminal-setup-cache-selection@1"
            or value["policy"] != POLICY or type(value["manifest_sha256"]) is not str
            or not re.fullmatch("[0-9a-f]{64}", value["manifest_sha256"])):
        raise ValueError("closed supported setup cache selection required")


def library_binding_for_manifest(manifest):
    """Bind exact installed paths to their immutable shipped source artifacts."""
    from .terminal_deployment import validate_torch_cpu_wheel_binding
    from .terminal_setup_cache_libraries import (ROWS, WHEEL_SHA256, TORCH_REQUIREMENT,
        SOURCE_PINS_SHA256, SOURCE_SELECTION, INSTALLER_MODE_POLICY)
    wheel = validate_torch_cpu_wheel_binding(manifest)
    if (wheel is None or wheel["sha256"] != WHEEL_SHA256
            or wheel["requirement"] != TORCH_REQUIREMENT
            or not wheel["path"].endswith("manylinux_2_28_aarch64.whl")):
        raise ValueError("native cache policy requires its exact supported aarch64 CPU wheel")
    inventory = {row["path"]: row for row in manifest["files"]}
    selected = []
    for row in ROWS:
        record = dict(row)
        if row["role"] == "installed_duckdb_extension":
            source_path = "extensions/" + row["name"]
            source = inventory.get(source_path)
            if (source is None or source["sha256"] != row["sha256"]
                    or source["bytes"] != row["expected_bytes"] or source["mode"] != row["mode"]):
                raise ValueError("native extension cache row differs from the shipped source")
            record.update(source_kind="archive_member", source_path=source_path)
        else:
            # These exact128 member hashes were extracted from WHEEL_SHA256;
            # the pinned adapter body and binding preserve that finite selection.
            prefix = "venv/lib/python3.12/site-packages/"
            if row["role"] != "installed_torch_wheel_payload" or not row["path"].startswith(prefix):
                raise ValueError("unsupported native library source")
            record.update(source_kind="wheel_member", source_path=row["path"][len(prefix):])
        selected.append(record)
    return dict(schema="terminal-native-library-cache-binding@1", wheel_sha256=WHEEL_SHA256,
                torch_requirement=TORCH_REQUIREMENT, source_pins_sha256=SOURCE_PINS_SHA256,
                source_selection=SOURCE_SELECTION, installer_mode_policy=INSTALLER_MODE_POLICY, rows=selected)


def binding_for_manifest(manifest, policy):
    """Build a closed policy using the actual archived helper byte identities."""
    if policy is None:
        return None
    if policy != POLICY or platform.machine() != "aarch64":
        raise ValueError("unsupported setup cache policy or architecture")
    from .terminal_deployment import validate_source384_binding
    from .terminal_setup_cache_files import checked_rows
    if validate_source384_binding(manifest) is None or manifest.get("codex_version") != "0.158.0":
        raise ValueError("setup cache policy requires the pinned Source384 and Codex profile")
    if manifest.get("learned_requirements"):
        raise ValueError("legacy relocated embedding layout is unsupported by setup cache policy")
    checked_rows(manifest["files"])
    rows = {row["path"]: row for row in manifest["files"]}
    pins = {}
    for name in HELPERS:
        row = rows.get(PREFIX + name)
        raw = _regular_bytes(Path(__file__).with_name(name), MAX_HELPER)
        if row is None or row["sha256"] != _sha(raw) or row["bytes"] != len(raw):
            raise ValueError("archived cache helper differs from the active canonical owner")
        pins[name] = _sha(raw)
    return dict(schema="terminal-setup-cache-binding@1", policy=POLICY,
                architecture="aarch64", codex_version="0.158.0",
                codex_sha256=dict(CODEX_PINS), helpers=pins,
                native_libraries=library_binding_for_manifest(manifest))


def validate_manifest_binding(manifest):
    binding = manifest.get("setup_cache")
    if binding is None:
        return None
    if type(binding) is not dict or set(binding) != {
            "schema", "policy", "architecture", "codex_version", "codex_sha256", "helpers", "native_libraries"}:
        raise ValueError("closed setup cache manifest binding required")
    expected = binding_for_manifest(manifest, binding.get("policy"))
    if binding != expected:
        raise ValueError("setup cache manifest binding or native pins changed")
    return binding


def select_setup_cache(archive_dir, policy):
    path = Path(archive_dir) / "manifest.json"
    raw = path.read_bytes() if policy is None else _regular_bytes(path, MAX_MANIFEST)
    manifest = json.loads(raw)
    binding = validate_manifest_binding(manifest)
    if policy is None:
        if binding is not None:
            raise ValueError("archive cache policy requires explicit matching selection")
        return None
    if policy != POLICY or binding is None or binding["policy"] != policy:
        raise ValueError("selected cache policy differs from the archive")
    return dict(schema="terminal-setup-cache-selection@1", policy=policy,
                manifest_sha256=_sha(raw))


def validate_setup_cache_selection(archive_dir, expected):
    if expected is not None:
        _selection_shape(expected)
    path = Path(archive_dir) / "manifest.json"
    raw = path.read_bytes() if expected is None else _regular_bytes(path, MAX_MANIFEST)
    if expected is not None and _sha(raw) != expected["manifest_sha256"]:
        raise ValueError("prepared cache manifest changed")
    manifest = json.loads(raw)
    binding = validate_manifest_binding(manifest)
    if (expected is None) != (binding is None):
        raise ValueError("cache policy cannot be enabled or dropped after preparation")
    return manifest, binding


def validate_setup_cache_prerequisites(expected, *, install_codex, auth_json,
                                       arm="full", resource_profile=PROFILE):
    if expected is None:
        return
    _selection_shape(expected)
    if (install_codex is not True or arm != "full" or resource_profile != PROFILE
            or platform.machine() != "aarch64"):
        raise ValueError("selected cache policy requires full Source384, aarch64 Codex and common resources")
    if auth_json is None:
        raise ValueError("selected worker boundary requires an explicit supported auth file")
    # Check only metadata. Never read, hash, serialize or include this path in a receipt.
    try:
        info = Path(auth_json).lstat()
    except OSError:
        raise ValueError("selected worker boundary auth file is unavailable") from None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= 65536:
        raise ValueError("selected worker boundary requires one bounded regular auth file")


def isolated_loader(modules, *, run, argv, receipt_bytes=None):
    """Execute exact shipped stdlib helpers without ambient Python import paths."""
    selection = dict(modules=modules, run=run, argv=argv, receipt_bytes=receipt_bytes)
    return "selection=" + repr(selection) + "\n" + r'''
import hashlib,os,stat,sys,types
def code(path,pin):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  before=os.fstat(stream.fileno())
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=65536:
   raise ValueError('bounded singleton code file required')
  raw=stream.read(65537)
  after=os.fstat(stream.fileno())
 if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')):
  raise ValueError('cache helper changed during read')
 if len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=pin:
  raise ValueError('cache helper code digest differs')
 return compile(raw,path,'exec')
compiled=[(name,path,code(path,pin)) for name,path,pin in selection['modules']]
for name,path,body in compiled:
 if name==selection['run']:
  sys.argv=[path,*selection['argv']]
  if selection['receipt_bytes'] is None:
   exec(body,{'__name__':'__main__','__file__':path})
  else:
   module=types.ModuleType(name);module.__file__=path
   exec(body,module.__dict__)
   module.protect_receipt(os.path.dirname(selection['argv'][0]),selection['receipt_bytes'])
   module.main()
 else:
  module=types.ModuleType(name);module.__file__=path
  exec(body,module.__dict__);sys.modules[name]=module
'''


async def apply_setup_cache_advice(environment, *, archive_dir, expected, boundary_output, output):
    """After worker-boundary setup, advise three closed public populations."""
    manifest, binding = validate_setup_cache_selection(archive_dir, expected)
    if binding is None:
        return None
    from .terminal_setup_cache_codex import require_receipt
    from .terminal_deployment import PYTHON
    receipt_path = Path(boundary_output) / "installation/native-codex-binary.log"
    receipt_raw = _regular_bytes(receipt_path, MAX_RECEIPT)
    require_receipt(json.loads(receipt_raw))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="terminal-setup-cache-advice@1", completed=False,
                  selection=expected, archive_sha256=manifest["archive_sha256"],
                  policy=POLICY, advice_is_best_effort=True, freed_bytes_claimed=False,
                  admission_authority=False, global_drop_caches=False, credential_contents_recorded=False)
    # Upload exact already-validated local bytes, then verify again in the isolated child.
    receipt_copy = output / "native-exposure.json"
    receipt_copy.write_bytes(receipt_raw)
    manifest_copy = output / "manifest-selection.json"
    manifest_raw = _regular_bytes(Path(archive_dir) / "manifest.json", MAX_MANIFEST)
    if _sha(manifest_raw) != expected["manifest_sha256"]:
        raise ValueError("manifest changed before cache advice")
    manifest_copy.write_bytes(manifest_raw)
    report["post_boundary_receipt_sha256"] = _sha(receipt_raw)
    modules = []
    for name in HELPERS[1:]:
        raw = _regular_bytes(Path(__file__).with_name(name), MAX_HELPER)
        if _sha(raw) != binding["helpers"][name]:
            raise ValueError("cache helper changed before transport")
        local = output / name
        local.write_bytes(raw)
        remote = ROOT + "/" + name
        await environment.upload_file(local, remote)
        modules.append((name[:-3], remote, _sha(raw)))
    await environment.upload_file(manifest_copy, ROOT + "/setup-cache-manifest.json")
    await environment.upload_file(receipt_copy, ROOT + "/codex-cache-exposure.json")

    async def invoke(label, selected_modules, run, argv, *, receipt_bytes=None):
        loader = isolated_loader(selected_modules, run=run, argv=argv, receipt_bytes=receipt_bytes)
        response = await asyncio.wait_for(environment.exec(
            command=PYTHON + " -I -S -B -c " + shlex.quote(loader), cwd="/",
            user="root", timeout_sec=60), timeout=60)
        raw = response.stdout or ""
        (output / (label + ".stderr")).write_text((response.stderr or "")[:4096])
        if len(raw.encode()) > MAX_RESULT:
            raise ValueError("cache advice response exceeds bound")
        (output / (label + ".stdout")).write_text(raw)
        if response.return_code != 0:
            raise RuntimeError("selected cache advice failed: " + label)
        return json.loads(raw)

    primary_error = False
    try:
        report["phase"] = "archive_advice"
        archive = await invoke("archive-advice", modules[:1], modules[0][0],
            [ROOT + "/setup-cache-manifest.json", expected["manifest_sha256"], manifest["archive_sha256"]])
        if (archive.get("schema") != "manifest-cache-advice@1"
                or archive.get("manifest_sha256") != expected["manifest_sha256"]
                or archive.get("archive_sha256") != manifest["archive_sha256"]
                or archive.get("selected_files") != len(manifest["files"])
                or archive.get("body_reads") != 0 or archive.get("metadata_unchanged") is not True
                or archive.get("freed_bytes_claimed") is not False):
            raise ValueError("archive advice receipt differs from selected population")
        report["archive"] = archive
        report["phase"] = "native_binary_advice"
        native = await invoke("native-advice", modules[:2], modules[1][0],
            [ROOT + "/codex-cache-exposure.json", _sha(receipt_raw)], receipt_bytes=len(receipt_raw))
        if (native.get("schema") != "pinned-native-codex-cache-advice@1"
                or native.get("post_boundary_receipt_sha256") != _sha(receipt_raw)
                or native.get("codex_version") != "0.158.0" or native.get("selected_files") != 4
                or native.get("hashed_files") != 4 or native.get("body_reads_after_advice") != 0
                or native.get("body_read_bytes") != native.get("selected_bytes")
                or type(native.get("selected_bytes")) is not int or not 0 < native["selected_bytes"] <= 2 * 1024**3
                or native.get("metadata_unchanged") is not True or native.get("freed_bytes_claimed") is not False):
            raise ValueError("native advice receipt differs from selected population")
        report["native_binaries"] = native
        report["phase"] = "native_library_advice"
        libraries = await invoke("library-advice", modules, modules[2][0], [])
        expected_rows = binding["native_libraries"]["rows"]
        expected_bytes = sum(row["expected_bytes"] for row in expected_rows)
        if (libraries.get("schema") != "pinned-native-libraries-cache-advice@1"
                or libraries.get("wheel_sha256") != binding["native_libraries"]["wheel_sha256"]
                or libraries.get("torch_requirement") != binding["native_libraries"]["torch_requirement"]
                or libraries.get("selected_files") != 131 or libraries.get("hashed_files") != 131
                or any(libraries.get(key) != binding["native_libraries"][key]
                    for key in ("source_pins_sha256", "source_selection", "installer_mode_policy"))
                or libraries.get("selected_bytes") != expected_bytes
                or libraries.get("body_read_bytes") != expected_bytes
                or libraries.get("body_reads_after_advice") != 0
                or libraries.get("metadata_unchanged") is not True or libraries.get("freed_bytes_claimed") is not False):
            raise ValueError("library advice receipt differs from the closed source binding")
        observed_rows = libraries.get("files")
        if type(observed_rows) is not list or len(observed_rows) != 131:
            raise ValueError("exact native library receipt population required")
        for actual, wanted in zip(observed_rows, expected_rows):
            if type(actual) is not dict or any(actual.get(key) != wanted[key]
                    for key in ("path", "name", "role", "uid", "mode", "expected_bytes", "sha256")):
                raise ValueError("native library receipt row changed")
        report["native_libraries"] = libraries
        report["phase"] = "resource_observation"
        from .terminal_source384_qualification import observe_resources
        report["resources"] = await observe_resources(environment, output=output, profile=PROFILE)
        report.update(completed=True, phase="complete")
        return report
    except BaseException as exc:
        primary_error = True
        report["error_type"] = type(exc).__name__
        raise
    finally:
        try:
            (output / "setup-cache-advice.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        except Exception:
            # Keep an earlier advice failure exact. A missing mandatory receipt
            # after successful advice still makes selected setup fail closed.
            if not primary_error:
                raise

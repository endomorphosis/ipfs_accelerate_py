"""Explicit isolated Grok pruning profiles; configuration evidence is not savings.

The legacy profile performs no filesystem or process access.  Explicit profiles
write only the dedicated container user's config; host config discovery and
credential inspection are deliberately outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Mapping


DEFAULT_GROK_PRUNING_PROFILE = "legacy"
GROK_PRUNING_PROFILES = ("legacy", "grok-pruning-off@1", "grok-pruning-on@1")
DECLARATION_SCHEMA = "native-grok-pruning-declaration@1"
FILE_OBSERVATION_SCHEMA = "native-grok-pruning-file-observation@1"
PREFLIGHT_SCHEMA = "native-grok-pruning-preflight@1"
EFFECTIVE_OBSERVATION_SCHEMA = "native-grok-pruning-effective-observation@1"
ISOLATED_ROOT = Path("/opt/ipfs-supervisor")
CONFIG_RELATIVE_PATH = "worker-home/.grok/config.toml"
GROK_BINARY_PATH = "/opt/ipfs-supervisor/provider-bin/grok"
GROK_BINARY_SHA256 = "45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e"
GROK_BINARY_BYTES = 142_867_512
MAX_INSPECT_STDOUT_BYTES = 256 * 1024
MAX_INSPECT_STDERR_BYTES = 64 * 1024
MAX_INSPECT_SECONDS = 30.0
_INSPECT_ROOT_KEYS = frozenset(("agents", "channel", "configSources", "cwd", "externalCompat", "grokVersion",
    "hooks", "loginPolicy", "lspServers", "marketplaces", "mcpServers", "permissions", "plugins",
    "projectInstructions", "projectRoot", "projectTrusted", "skills"))
_PRUNING_KEYS = frozenset(("pruning", "compaction", "keeplastnturns", "softtrimthreshold",
                         "softtrimhead", "softtrimtail", "hardclearageturns"))
SOURCE_PATHS = (
    "ipfs_accelerate_py/cli_runtime/grok_pruning.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/llm_router.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_comparison.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_grok_deployment.py",
)


def validate_grok_pruning_profile(value, *, provider=None, purpose="coding"):
    if type(value) is not str or value not in GROK_PRUNING_PROFILES:
        raise ValueError("unsupported Grok pruning profile")
    if value != DEFAULT_GROK_PRUNING_PROFILE and (
            type(purpose) is not str or purpose != "coding"
            or provider is not None and (type(provider) is not str or provider != "grok_cli")):
        raise ValueError("explicit pruning requires a Grok coding invocation")
    return value


def grok_pruning_selection(profile):
    profile = validate_grok_pruning_profile(profile)
    return {} if profile == DEFAULT_GROK_PRUNING_PROFILE else {"grok_pruning_profile": profile}


def render_grok_pruning_config(profile):
    profile = validate_grok_pruning_profile(profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        return None
    enabled = "true" if profile == "grok-pruning-on@1" else "false"
    return ("[compaction.pruning]\n"
            f"enabled = {enabled}\n"
            "keep_last_n_turns = 3\n"
            "soft_trim_threshold = 4000\n"
            "soft_trim_head = 1500\n"
            "soft_trim_tail = 1500\n"
            "hard_clear_age_turns = 10\n").encode("ascii")


def _non_authority():
    return {"token_savings_established": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False}


def grok_pruning_declaration(profile):
    profile = validate_grok_pruning_profile(profile)
    raw = render_grok_pruning_config(profile)
    if raw is None:
        return {}
    return {"schema": DECLARATION_SCHEMA, "profile": profile,
            "config_sha256": hashlib.sha256(raw).hexdigest(), "config_bytes": len(raw),
            "effective_settings_verified": False, **_non_authority()}


def _same_closed(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _same_closed(value[key], expected[key]) for key in expected)
    return value == expected


def _receipt_profile(value, profile):
    if profile is not None:
        return validate_grok_pruning_profile(profile)
    if type(value) is dict and not value:
        return DEFAULT_GROK_PRUNING_PROFILE
    if type(value) is not dict:
        raise ValueError("invalid Grok pruning receipt")
    return validate_grok_pruning_profile(value.get("profile"))


def validate_grok_pruning_declaration(value, *, profile=None):
    expected = grok_pruning_declaration(_receipt_profile(value, profile))
    if not _same_closed(value, expected):
        raise ValueError("invalid Grok pruning declaration")
    return expected


def _expected_file_observation(profile, expected_uid, expected_gid):
    declaration = grok_pruning_declaration(profile)
    if not declaration:
        return {}
    for owner in (expected_uid, expected_gid):
        if type(owner) is not int or owner < 0:
            raise ValueError("invalid isolated Grok config ownership")
    return {"schema": FILE_OBSERVATION_SCHEMA, "profile": profile,
            "config_sha256": declaration["config_sha256"],
            "config_bytes": declaration["config_bytes"],
            "owner_uid": expected_uid, "owner_gid": expected_gid, "mode": 0o444,
            "regular_file": True, "single_link": True, "symlink_free_path": True,
            "exact_bytes_verified": True, "checked_before_invocation": True,
            "immutable_during_invocation": False, "effective_settings_verified": False,
            **_non_authority()}


def validate_grok_pruning_file_observation(value, *, profile=None, expected_uid=0, expected_gid=0):
    expected = _expected_file_observation(_receipt_profile(value, profile), expected_uid, expected_gid)
    if not _same_closed(value, expected):
        raise ValueError("invalid deployed Grok pruning observation")
    return expected


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_fixed_config(root, expected, expected_uid, expected_gid):
    """Read via pinned directory descriptors, checking names again after reading.

    Parents may be worker-owned: this check establishes a preinvoke observation,
    not immutability while the native client subsequently runs.
    """
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("isolated Grok config root must be absolute")
    path = root / CONFIG_RELATIVE_PATH
    directories = []
    opened = []
    file_fd = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent_fd = os.open("/", flags)
        opened.append(parent_fd)
        for name in path.parts[1:-1]:
            child_fd = os.open(name, flags, dir_fd=parent_fd)
            directories.append((parent_fd, name, child_fd))
            opened.append(child_fd)
            parent_fd = child_fd
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                          dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_uid != expected_uid or before.st_gid != expected_gid
                or before.st_nlink != 1 or before.st_size != len(expected)):
            raise ValueError("deployed Grok pruning config does not match selection")
        chunks = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(file_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != expected or _file_identity(before) != _file_identity(os.fstat(file_fd)):
            raise ValueError("deployed Grok pruning config changed or mismatched")
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _file_identity(named) != _file_identity(before):
            raise ValueError("deployed Grok pruning config binding changed")
        for parent, name, child in directories:
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            held = os.fstat(child)
            if (not stat.S_ISDIR(named.st_mode)
                    or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)):
                raise ValueError("deployed Grok pruning directory binding changed")
    except OSError:
        raise ValueError("deployed Grok pruning config unavailable") from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(opened):
            os.close(descriptor)


def observe_deployed_grok_pruning(profile, *, isolated_root=ISOLATED_ROOT,
                                  expected_uid=0, expected_gid=0):
    profile = validate_grok_pruning_profile(profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        return {}
    observation = _expected_file_observation(profile, expected_uid, expected_gid)
    _read_fixed_config(isolated_root, render_grok_pruning_config(profile), expected_uid, expected_gid)
    return observation


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _inspection_receipt(profile, stdout_bytes, stdout_sha256, stderr_bytes, stderr_sha256):
    return {"schema": EFFECTIVE_OBSERVATION_SCHEMA, "profile": profile, "status": "unavailable",
            "native_command": "inspect --json", "native_inspect_commands": 1, "native_exit_code": 0,
            "cli_version": "1.0.46", "version_basis": "pinned_binary_identity",
            "native_version_scalar_verified": False,
            "binary_sha256": GROK_BINARY_SHA256, "binary_bytes": GROK_BINARY_BYTES,
            "stdout_bytes": stdout_bytes, "stdout_sha256": stdout_sha256,
            "stderr_bytes": stderr_bytes, "stderr_sha256": stderr_sha256,
            "inspect_schema_recognized": True, "config_source_status": "unreported",
            "config_source_verified": False, "pruning_values_reported": False,
            "effective_settings_verified": False, "behavior_verified": False,
            "text_generation_calls": 0, **_non_authority()}


def validate_grok_pruning_effective_observation(value, *, profile=None):
    profile = _receipt_profile(value, profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        if type(value) is not dict or value:
            raise ValueError("invalid Grok pruning inspect observation")
        return {}
    if type(value) is not dict:
        raise ValueError("invalid Grok pruning inspect observation")
    for lane, limit, minimum in (("stdout", MAX_INSPECT_STDOUT_BYTES, 1),
                                  ("stderr", MAX_INSPECT_STDERR_BYTES, 0)):
        size, digest = value.get(lane + "_bytes"), value.get(lane + "_sha256")
        if (type(size) is not int or not minimum <= size <= limit or not _digest(digest)
                or size == 0 and digest != hashlib.sha256(b"").hexdigest()):
            raise ValueError("invalid Grok pruning inspect observation")
    expected = _inspection_receipt(profile, value["stdout_bytes"], value["stdout_sha256"],
                                   value["stderr_bytes"], value["stderr_sha256"])
    if not _same_closed(value, expected):
        raise ValueError("invalid Grok pruning inspect observation")
    return expected


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native inspection member")
        result[key] = value
    return result


def _inspect_json(stdout):
    try:
        value = json.loads(stdout.decode("utf-8"), object_pairs_hook=_json_object,
                           parse_constant=lambda unused: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        pending = [(value, 0)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > 32 or nodes > 32768:
                raise ValueError("inspection structure exceeds bound")
            if type(item) is dict:
                for key, child in item.items():
                    key.encode("utf-8")
                    normalized = "".join(character for character in key.lower() if character.isascii() and character.isalnum())
                    if any(part in normalized for part in _PRUNING_KEYS):
                        # There is no qualified native pruning projection yet.
                        # New settings must never silently become 'unreported'.
                        raise ValueError("unqualified native pruning settings")
                    pending.append((child, depth + 1))
            elif type(item) is list:
                pending.extend((child, depth + 1) for child in item)
            elif type(item) is str:
                item.encode("utf-8")
            elif type(item) is float and not math.isfinite(item):
                raise ValueError("nonfinite native inspection value")
        if type(value) is not dict or set(value) != _INSPECT_ROOT_KEYS:
            raise ValueError("unqualified native inspection schema")
        for key in ("agents", "hooks", "lspServers", "marketplaces", "mcpServers", "plugins", "projectInstructions", "skills"):
            if type(value[key]) is not list:
                raise ValueError("unqualified native inspection shape")
        for key in ("configSources", "externalCompat", "loginPolicy", "permissions"):
            if type(value[key]) is not dict:
                raise ValueError("unqualified native inspection shape")
        for key, limit in (("grokVersion", 128), ("channel", 64)):
            scalar = value[key]
            if (type(scalar) is not str or not 1 <= len(scalar) <= limit
                    or any(not 32 <= ord(character) <= 126 for character in scalar)):
                raise ValueError("unqualified native inspection version shape")
        if (type(value["cwd"]) is not str or not value["cwd"] or len(value["cwd"]) > 4096
                or type(value["projectTrusted"]) is not bool
                or value["projectRoot"] is not None and type(value["projectRoot"]) is not str
                or not _same_closed(value["configSources"], {"layers": []})):
            raise ValueError("unqualified native inspection source shape")
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise ValueError("unsupported bounded native Grok inspection") from None


def observe_grok_pruning_inspection(profile, *, stdout, stderr, returncode):
    """Project a bounded native inspection without exporting its config or paths.

    The pinned client's inspected schema omits pruning values and config-file
    provenance.  Successful inspection therefore does not verify effectiveness.
    """
    profile = validate_grok_pruning_profile(profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        return {}
    if (type(stdout) is not bytes or not 0 < len(stdout) <= MAX_INSPECT_STDOUT_BYTES
            or type(stderr) is not bytes or len(stderr) > MAX_INSPECT_STDERR_BYTES
            or type(returncode) is not int or returncode != 0):
        raise ValueError("bounded native Grok inspection failed")
    _inspect_json(stdout)
    return _inspection_receipt(profile, len(stdout), hashlib.sha256(stdout).hexdigest(),
                               len(stderr), hashlib.sha256(stderr).hexdigest())


def build_grok_pruning_preflight(profile, *, file_observation, effective_observation):
    profile = validate_grok_pruning_profile(profile)
    declaration = grok_pruning_declaration(profile)
    file_observation = validate_grok_pruning_file_observation(file_observation, profile=profile)
    effective_observation = validate_grok_pruning_effective_observation(effective_observation, profile=profile)
    if not declaration:
        return {}
    return {"schema": PREFLIGHT_SCHEMA, "profile": profile, "declaration": declaration,
            "file_observation": file_observation, "effective_observation": effective_observation,
            "preflight_completed": True, "requested_profile_deployed": True, "native_inspect_completed": True,
            "effective_settings_verified": False, "config_file_consumption_verified": False,
            "behavior_verified": False, **_non_authority()}


def validate_grok_pruning_preflight(value, *, profile=None):
    profile = _receipt_profile(value, profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        if type(value) is not dict or value:
            raise ValueError("invalid Grok pruning preflight")
        return {}
    if type(value) is not dict:
        raise ValueError("invalid Grok pruning preflight")
    expected = build_grok_pruning_preflight(profile,
        file_observation=value.get("file_observation"), effective_observation=value.get("effective_observation"))
    if not _same_closed(value, expected):
        raise ValueError("invalid Grok pruning preflight")
    return expected


def _verify_native_binary(deadline):
    """Bind the inspection executable to the already qualified public build."""
    opened = []
    file_fd = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent = os.open("/", flags)
        opened.append(parent)
        directory_bindings = []
        for name in Path(GROK_BINARY_PATH).parts[1:-1]:
            child = os.open(name, flags, dir_fd=parent)
            opened.append(child)
            info = os.fstat(child)
            if info.st_uid != 0 or info.st_gid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
                raise ValueError("isolated Grok binary parent is not protected")
            directory_bindings.append((parent, name, child))
            parent = child
        name = Path(GROK_BINARY_PATH).name
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_gid != 0
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o222
                or not stat.S_IMODE(before.st_mode) & 0o111 or before.st_size != GROK_BINARY_BYTES):
            raise ValueError("isolated Grok binary does not match pinned identity")
        digest = hashlib.sha256()
        total = 0
        while True:
            if time.monotonic() >= deadline:
                raise ValueError("Grok pruning preflight deadline exceeded")
            block = os.read(file_fd, min(1024 * 1024, GROK_BINARY_BYTES - total + 1))
            if not block:
                break
            digest.update(block)
            total += len(block)
            if total > GROK_BINARY_BYTES:
                raise ValueError("isolated Grok binary exceeds pinned bounds")
        if (total != GROK_BINARY_BYTES or digest.hexdigest() != GROK_BINARY_SHA256
                or _file_identity(before) != _file_identity(os.fstat(file_fd))
                or _file_identity(before) != _file_identity(os.stat(name, dir_fd=parent, follow_symlinks=False))):
            raise ValueError("isolated Grok binary does not match pinned identity")
        for owner, component, held in directory_bindings:
            named, current = os.stat(component, dir_fd=owner, follow_symlinks=False), os.fstat(held)
            if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("isolated Grok binary binding changed")
    except OSError:
        raise ValueError("isolated Grok binary unavailable") from None
    finally:
        if file_fd is not None: os.close(file_fd)
        for descriptor in reversed(opened): os.close(descriptor)


def _run_inspect(command, env, timeout):
    """Bound both pipes while draining them; never include native text in errors."""
    process = None
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": MAX_INSPECT_STDOUT_BYTES, "stderr": MAX_INSPECT_STDERR_BYTES}
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen([command[0], "inspect", "--json"], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        for lane in ("stdout", "stderr"):
            pipe = getattr(process, lane)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, lane)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("bounded native Grok inspection timed out")
            for key, unused in selector.select(min(remaining, 0.1)):
                lane = key.data
                block = os.read(key.fileobj.fileno(), min(65536, limits[lane] - len(buffers[lane]) + 1))
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                buffers[lane].extend(block)
                if len(buffers[lane]) > limits[lane]:
                    raise ValueError("bounded native Grok inspection output exceeded")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("bounded native Grok inspection timed out")
        returncode = process.wait(timeout=remaining)
        return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode
    except (OSError, subprocess.SubprocessError):
        raise ValueError("bounded native Grok inspection failed") from None
    finally:
        selector.close()
        if process is not None:
            # A parent may exit while a descendant still holds a capture pipe.
            # The session belongs to this inspection, even after parent exit.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            try: process.wait(timeout=1)
            except (OSError, subprocess.SubprocessError): pass
            for pipe in (process.stdout, process.stderr):
                if pipe is not None: pipe.close()


def run_grok_pruning_preflight(profile, *, command, env, timeout):
    profile = validate_grok_pruning_profile(profile)
    if profile == DEFAULT_GROK_PRUNING_PROFILE:
        return {}
    if (type(command) is not list or len(command) != 1 or type(command[0]) is not str
            or command[0] != GROK_BINARY_PATH):
        raise ValueError("explicit pruning requires the pinned isolated Grok command")
    if (type(timeout) not in (float, int) or not 0 < timeout <= MAX_INSPECT_SECONDS
            or not math.isfinite(timeout)):
        raise ValueError("invalid Grok pruning preflight timeout")
    if not isinstance(env, Mapping) or len(env) > 4096:
        raise ValueError("invalid isolated Grok pruning environment")
    native_env = {}
    total = 0
    for key, value in env.items():
        if (type(key) is not str or not key or type(value) is not str
                or "\0" in key or "=" in key or "\0" in value):
            raise ValueError("invalid isolated Grok pruning environment")
        total += len(key) + len(value)
        if total > 2 * 1024 * 1024:
            raise ValueError("isolated Grok pruning environment exceeds bounds")
        native_env[key] = value
    home = str(ISOLATED_ROOT / "worker-home")
    if (native_env.get("HOME") != home or native_env.get("GROK_HOME") != home + "/.grok"
            or any(key in native_env for key in ("GROK_CONFIG", "GROK_CONFIG_PATH"))):
        raise ValueError("explicit pruning requires its isolated user config environment")
    deadline = time.monotonic() + timeout
    before = observe_deployed_grok_pruning(profile)
    _verify_native_binary(deadline)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("Grok pruning preflight deadline exceeded")
    stdout, stderr, returncode = _run_inspect(command, native_env, remaining)
    effective = observe_grok_pruning_inspection(profile, stdout=stdout, stderr=stderr, returncode=returncode)
    after = observe_deployed_grok_pruning(profile)
    if not _same_closed(before, after):
        raise ValueError("deployed Grok pruning config changed during inspection")
    return build_grok_pruning_preflight(profile, file_observation=after, effective_observation=effective)


def require_grok_pruning_archive(manifest, profile):
    if validate_grok_pruning_profile(profile) == DEFAULT_GROK_PRUNING_PROFILE:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("Grok pruning profile requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[2]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("Grok pruning profile requires a newly qualified runtime archive")


def grok_pruning_runtime_matches(report, profile):
    """Match requested deployment/preflight evidence, never infer pruning savings.

    A successful match does not assert that native effective values were exposed
    by inspection, or that any coding history was actually pruned.
    """
    profile = validate_grok_pruning_profile(profile)
    if type(report) is not dict or type(report.get("provider_invocations")) is not list:
        return None
    coding = []
    for row in report["provider_invocations"]:
        if (type(row) is not dict or row.get("purpose") not in ("planning", "coding")
                or row.get("phase") not in (None, row["purpose"])):
            return False
        if row["purpose"] == "coding":
            coding.append(row)
    if not coding:
        return None
    if report.get("unreceipted_provider_attempt"):
        return False
    for row in coding:
        policy = row.get("provider_invocation_policy")
        if row.get("provider") != "grok_cli" or type(policy) is not dict:
            return False
        try:
            validate_grok_pruning_preflight(policy.get("grok_pruning", {}), profile=profile)
        except (ValueError, TypeError):
            return False
    return True

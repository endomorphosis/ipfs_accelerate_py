"""Explicit native Grok asset and credential transport for isolated trials."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import stat
import subprocess
import tarfile
import time

GROK_VERSION = "1.0.46"
GROK_VERSION_OUTPUT = "grok 1.0.46 (2765805b9442) [stable]"
GROK_VERSION_OUTPUTS = ("grok 1.0.46 (2765805b9442)", GROK_VERSION_OUTPUT)
GROK_PROFILE = "grok-4.7-cli-1.0.46@1"
GROK_PATH = "providers/grok/grok"
GROK_SHA256 = "45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e"
GROK_BYTES = 142867512
READINESS_SCHEMA = "terminal-grok-credential-readiness@1"
READINESS_ROOT = Path("/opt/ipfs-supervisor")
READINESS_TIMEOUT = 20
READINESS_MAX_BYTES = 65536


class GrokCredentialReadinessError(RuntimeError):
    """Safe refusal with only the bounded public observation attached."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("isolated Grok credential readiness " + receipt["status"])


def _readiness_result(reason="unavailable"):
    return dict(schema=READINESS_SCHEMA, status="unavailable", reason=reason,
        provider="grok_cli", model="grok-4.7", cli_version=GROK_VERSION,
        scope="isolated_worker_credential_copy", native_command="models",
        native_catalog_commands=0, native_exit_code=None,
        stdout_bytes=0, stderr_bytes=0, models_observed=0, selected_model_observed=False,
        text_generation_calls=0, authentication_authority=False,
        proof_authority=False, execution_authority=False, completion_authority=False)


def _catalog_observation(stdout: bytes, stderr: bytes, returncode: int) -> dict:
    """Classify a bounded catalog shape; never return native text or model IDs."""
    result = _readiness_result("catalog_shape_unknown")
    result.update(native_catalog_commands=1, native_exit_code=returncode,
                  stdout_bytes=len(stdout), stderr_bytes=len(stderr))
    if len(stdout) + len(stderr) > READINESS_MAX_BYTES:
        return dict(result, reason="output_limit")
    clean = lambda raw: re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8", errors="replace"))
    out, err = clean(stdout), clean(stderr)
    text = (out + "\n" + err).lower()
    if re.search(r"\b(?:not (?:authenticated|signed in|logged in)|authentication (?:failed|required)|"
                 r"unauthorized|invalid (?:access )?token|expired (?:access )?token)\b", text):
        return dict(result, status="unauthenticated", reason="authentication_refused")
    if returncode != 0:
        return dict(result, reason="native_command_failed")
    if err.strip():
        return dict(result, reason="unexpected_diagnostic")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines or lines[0].lower() != "available models:":
        return result
    models = []
    for line in lines[1:]:
        match = re.fullmatch(r"(?:[-*] )?(grok-[a-z0-9][a-z0-9.-]{0,95})(?: \(default\))?", line)
        if match is None or match[1] in models or len(models) >= 128:
            return result
        models.append(match[1])
    if not models:
        return result
    selected = "grok-4.7" in models
    return dict(result, status="ready" if selected else "unavailable",
        reason="catalog_observed" if selected else "selected_model_unavailable",
        models_observed=len(models), selected_model_observed=selected)


def _container_grok_inputs():
    """Require the deployed nonroot boundary before inspecting its auth copy."""
    import ctypes

    root = READINESS_ROOT
    if os.getuid() != 1001 or os.geteuid() != 1001 or root.resolve() != root:
        raise ValueError("isolated worker identity required")
    from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import verify_container_worker_boundary
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
        raise ValueError("worker privilege boundary unavailable")
    boundary = root / "container-boundary.json"
    info = boundary.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
            or info.st_mode & 0o022 or not 0 < info.st_size <= 16384):
        raise ValueError("deployed boundary unavailable")
    raw = boundary.read_bytes()
    verify_container_worker_boundary(artifact=boundary, expected_sha256=hashlib.sha256(raw).hexdigest(),
        workspace=Path("/app"), purpose="validation")
    if json.loads(raw).get("provider") != "grok_cli":
        raise ValueError("Grok boundary required")
    binary = root / "provider-bin/grok"
    for path in (binary, binary.parent):
        info = path.lstat()
        if path.resolve() != path or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("protected Grok binary required")
    validate_grok_binary(binary)
    home = root / "worker-home"
    for path in (home, home / ".grok"):
        info = path.lstat()
        if (path.resolve() != path or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != 1001 or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("private worker credential directory required")
    auth = home / ".grok/auth.json"
    info = auth.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 1001 or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600 or not 0 < info.st_size <= 65536):
        raise ValueError("private copied worker credential required")
    # No credential contents, digest, mtime or inode is read into the receipt.
    return binary, home


def _bounded_catalog_command(binary, home):
    """Run one fixed metadata command; retain at most64KiB until classification."""
    environment = {"HOME": str(home), "GROK_HOME": str(home / ".grok"),
        "PATH": "/usr/bin:/bin", "NO_COLOR": "1", "TERM": "dumb",
        "GROK_CODEX_MCPS_ENABLED": "false"}
    process = subprocess.Popen([str(binary), "models"], cwd="/", env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    reason = None
    deadline = time.monotonic() + READINESS_TIMEOUT
    try:
        with selectors.DefaultSelector() as selector:
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "native_timeout"
                    break
                for key, _ in selector.select(min(.1, remaining)):
                    block = os.read(key.fd, 4096)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    room = READINESS_MAX_BYTES - sum(map(len, buffers.values()))
                    buffers[key.data].extend(block[:room])
                    if len(block) > room:
                        reason = "output_limit"
                        break
                if reason:
                    break
            if reason is None:
                try:
                    process.wait(timeout=max(.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = "native_timeout"
    finally:
        if reason is not None or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)
        process.stdout.close()
        process.stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), process.returncode, reason


def probe_container_grok_readiness() -> dict:
    """Explicit isolated-copy probe, never a host-HOME or login operation.

    The CLI may refresh or invalidate this container's copied auth.json. The
    catalog observation grants no authentication or subsequent-call authority.
    """
    try:
        binary, home = _container_grok_inputs()
    except Exception:
        return _readiness_result("isolated_boundary_unavailable")
    try:
        stdout, stderr, code, reason = _bounded_catalog_command(binary, home)
    except Exception:
        return dict(_readiness_result("native_execution_unavailable"), native_catalog_commands=1)
    result = _catalog_observation(stdout, stderr, code)
    if reason is not None:
        result.update(status="unavailable", reason=reason, models_observed=0, selected_model_observed=False)
    return result


def grok_credential_readiness_script() -> str:
    """Render the reviewed probe without requiring new code in old archives."""
    header = ("import hashlib,json,os,re,selectors,signal,stat,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "sys.path[:0]=['/opt/ipfs-supervisor/source','/opt/ipfs-supervisor/datasets','/opt/ipfs-supervisor/kit']\n")
    values = dict(READINESS_SCHEMA=READINESS_SCHEMA, GROK_VERSION=GROK_VERSION,
        GROK_SHA256=GROK_SHA256, GROK_BYTES=GROK_BYTES,
        READINESS_TIMEOUT=READINESS_TIMEOUT, READINESS_MAX_BYTES=READINESS_MAX_BYTES)
    constants = "".join(f"{key}={value!r}\n" for key, value in values.items())
    constants += "READINESS_ROOT=Path('/opt/ipfs-supervisor')\n"
    functions = (_readiness_result, _catalog_observation, _container_grok_inputs,
        validate_grok_binary, _bounded_catalog_command, probe_container_grok_readiness)
    return header + constants + "\n".join(inspect.getsource(function) for function in functions) + (
        "\nprint(json.dumps(probe_container_grok_readiness(),sort_keys=True))\n")


async def require_grok_credential_readiness(environment, *, output: Path) -> dict:
    """Opt-in deployed-worker check; persist only a closed public observation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    script = grok_credential_readiness_script()
    receipt = _readiness_result("container_transport_unavailable")
    try:
        response = await environment.exec(
            command="/opt/ipfs-supervisor/venv/bin/python -I -c " + shlex.quote(script),
            cwd="/", user="benchmarkworker", env={"HOME": "/opt/ipfs-supervisor/worker-home",
                "GROK_HOME": "/opt/ipfs-supervisor/worker-home/.grok", "PATH": "/usr/bin:/bin"},
            timeout_sec=30)
        raw = response.stdout or ""
        if response.return_code == 0 and not (response.stderr or "").strip() and len(raw.encode()) <= 4096:
            candidate = json.loads(raw)
            template = _readiness_result()
            dynamic = {"status", "reason", "native_catalog_commands", "native_exit_code",
                "stdout_bytes", "stderr_bytes", "models_observed", "selected_model_observed"}
            if (type(candidate) is dict and set(candidate) == set(template)
                    and all(type(candidate[k]) is type(v) and candidate[k] == v
                            for k, v in template.items() if k not in dynamic)
                    and candidate["status"] in {"ready", "unauthenticated", "unavailable"}
                    and candidate["reason"] in {"catalog_shape_unknown", "catalog_observed", "authentication_refused",
                        "native_command_failed", "unexpected_diagnostic", "selected_model_unavailable", "output_limit",
                        "native_timeout", "isolated_boundary_unavailable", "native_execution_unavailable"}
                    and type(candidate["native_catalog_commands"]) is int and 0 <= candidate["native_catalog_commands"] <= 1
                    and (candidate["native_exit_code"] is None or type(candidate["native_exit_code"]) is int
                         and -255 <= candidate["native_exit_code"] <= 255)
                    and all(type(candidate[k]) is int and 0 <= candidate[k] <= maximum for k, maximum in
                        (("stdout_bytes", READINESS_MAX_BYTES), ("stderr_bytes", READINESS_MAX_BYTES), ("models_observed", 128)))
                    and candidate["stdout_bytes"] + candidate["stderr_bytes"] <= READINESS_MAX_BYTES
                    and type(candidate["selected_model_observed"]) is bool
                    and (candidate["status"] != "ready" or candidate["reason"] == "catalog_observed"
                         and candidate["native_catalog_commands"] == 1 and candidate["native_exit_code"] == 0
                         and candidate["stderr_bytes"] == 0 and candidate["stdout_bytes"] > 0
                         and candidate["models_observed"] > 0 and candidate["selected_model_observed"] is True)):
                receipt = candidate
    except Exception:
        pass  # Native/transport exception strings may contain private output.
    (output / "readiness.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    if receipt["status"] != "ready":
        raise GrokCredentialReadinessError(receipt)
    return receipt


def require_grok_version_output(value: str) -> str:
    """Accept the pinned build with or without managed-install channel metadata."""
    if type(value) is not str or value.strip() not in GROK_VERSION_OUTPUTS:
        raise ValueError("native Grok version differs from the pinned build")
    return value.strip()


def grok_binding() -> dict:
    return {"path": GROK_PATH, "version": GROK_VERSION, "version_output": GROK_VERSION_OUTPUT,
            "sha256": GROK_SHA256, "bytes": GROK_BYTES, "platform": "linux-aarch64"}


def validate_grok_binding(manifest: dict, *, required: bool = False) -> dict | None:
    binding = manifest.get("grok_cli_assets")
    if binding is None and not required:
        return None
    if type(binding) is not dict or binding != grok_binding():
        raise ValueError("exact pinned Grok CLI asset binding required")
    inventory = manifest.get("files")
    if type(inventory) is not list or any(type(row) is not dict for row in inventory):
        raise ValueError("bounded Grok runtime inventory required")
    rows = [row for row in inventory if row.get("path") == GROK_PATH]
    if len(rows) != 1 or rows[0] != {
            "path": GROK_PATH, "bytes": GROK_BYTES, "sha256": GROK_SHA256, "mode": 0o755}:
        raise ValueError("Grok runtime inventory differs from its independent pin")
    return dict(binding)


def validate_grok_binary(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != GROK_BYTES:
        raise ValueError("pinned regular Grok native binary required")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != GROK_SHA256:
            raise ValueError("Grok native binary SHA-256 differs")
    if not os.access(path, os.X_OK):
        raise ValueError("executable Grok native binary required")
    return path


def validate_auth_file(path: Path) -> Path:
    path = Path(path)
    # Never return or retain credential contents or a credential fingerprint.
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 65536:
        raise ValueError("one bounded regular auth.json file required")
    return path


def verify_grok_archive(archive: Path, manifest: dict, *, required: bool = False) -> dict | None:
    binding = validate_grok_binding(manifest, required=required)
    if binding is None:
        return None
    with tarfile.open(archive) as bundle:
        members = [member for member in bundle.getmembers() if member.name == GROK_PATH]
        if len(members) != 1:
            raise ValueError("one pinned Grok archive member required")
        member = members[0]
        if not member.isfile() or member.size != GROK_BYTES or member.mode != 0o755:
            raise ValueError("Grok archive member type, size or mode differs")
        with bundle.extractfile(member) as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != GROK_SHA256:
                raise ValueError("Grok archive binary SHA-256 differs")
    return binding


def grok_exposure_script(root: str = "/opt/ipfs-supervisor") -> str:
    """Recheck the actual container binary before exposing it to either UID."""
    return f'''import hashlib,json,os,pathlib,platform,shutil,subprocess
root=pathlib.Path({root!r});source=root/{GROK_PATH!r}
if platform.system()!='Linux' or platform.machine()!='aarch64':raise SystemExit('pinned Grok platform differs')
if source.is_symlink() or not source.is_file() or source.stat().st_size!={GROK_BYTES}:raise SystemExit('pinned Grok file differs')
with source.open('rb') as stream:
 if hashlib.file_digest(stream,'sha256').hexdigest()!={GROK_SHA256!r}:raise SystemExit('pinned Grok SHA differs')
target=root/'provider-bin/grok';target.parent.mkdir(mode=0o755,exist_ok=True)
if target.exists() or target.is_symlink():raise SystemExit('Grok exposure already exists')
shutil.copyfile(source,target);target.chmod(0o555);os.chown(target,0,0)
result=subprocess.run([str(target),'--version'],capture_output=True,text=True,timeout=15,env={{'HOME':str(root/'home'),'PATH':'/usr/bin:/bin'}})
matches=result.stdout.strip() in {GROK_VERSION_OUTPUTS!r}
if result.returncode or not matches:
 print(json.dumps({{'schema':'native-grok-version-failure@1','return_code':result.returncode,'pinned_version_observed':matches,'stdout_bytes':len(result.stdout.encode()),'stderr_bytes':len(result.stderr.encode()),'provider_calls':0}}))
 raise SystemExit('native Grok version differs')
print(json.dumps({{'schema':'native-grok-runtime-bundle@1','version':{GROK_VERSION!r},'version_output_observed':result.stdout.strip(),'sha256':{GROK_SHA256!r},'bytes':{GROK_BYTES},'provider_calls':0}}))
'''

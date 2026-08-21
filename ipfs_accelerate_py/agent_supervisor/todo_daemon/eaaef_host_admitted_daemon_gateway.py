"""Host-admitted Quack daemon gateway for EAAEF plan-bound children.

Independently signed EAAEF-191/182/188/189 receipts admit this process to
talk to the live loopback Quack owner.  The gateway is a closed
``QuackDaemonCommandGateway@1`` subclass: components never expose SQL, a
database path, or Portal.  Create-once lane artifacts remain unpublished;
this overlay does not invent signatures.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar

from ..runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    _eaaef_host_receipt,
    _eaaef_host_receipt_admitted,
)
from ..runtime.quack_state_server import TOKEN_FILENAME_SUFFIX
from ..runtime.worker_network_dispatch import EAAEF_BOARD_NAMESPACE
from ..task_sources.eaaef_operational_schema import EAAEF_OPERATIONAL_PROFILE_ID
from ..task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE,
    QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE,
    QuackDaemonCommandGateway,
    QuackDaemonGatewayError,
)
from ..task_sources.quack_state_client import QuackStateClient, TransportMode
from .external_agent_container_dispatcher import (
    EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA,
    EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA,
    EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA,
    EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
    EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA,
    ExternalAgentContainerWorkPacket,
    ExternalAgentContainerWorkerDispatcher,
)

_ADMITTED_DUCKDB_VERSION = "1.5.5"
_OWNER_HANDLE = "secret-handle:eaaef-quack-owner-v1"
_REQUIRED_RECEIPTS = ("EAAEF-191", "EAAEF-189", "EAAEF-188")
# Exact closed CAS template. Quack ATTACH exposes tasks as a view, so UPDATE
# must run on the exclusive owner's local file connection via the mutation inbox.
_CAS_TASK_STATUS_SQL = (
    "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
    "WHERE task_cid = ? AND revision = ?"
)
_OWNER_MUTATION_TIMEOUT_SECONDS = 15.0
_ADMITTED_DOCKER = Path("/usr/bin/docker")
_ADMITTED_ENGINE = "unix:///run/user/1000/docker.sock"
_CONTAINER_GROK = "/opt/eaaef/bin/grok"
_HOST_EVIDENCE_DID = (
    "did:key:z6Mkmff2BRjhv5Tx5L4XxKAcWeewEsVDgna3Y1UyGWJmoVin"
)
_REVIEWER_DID = (
    "did:key:z6Mktp3ogPs9QwXBnKEQrdMThdbuPPNKQXiAP7X7JwXVq1G7"
)
_GROK_LAUNCH_TIMEOUT_SECONDS = 1800
_OWNED_RELATIVE_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/handoff/contracts.py",
    "test/api/test_external_agent_handoff_contracts.py",
)
_DUCKDB_RECEIPT = (
    Path("docs")
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "host_admission"
    / "duckdb_quack_155.json"
)


def _cid(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _host_admitted(repo_root: Path) -> bool:
    return all(
        _eaaef_host_receipt_admitted(repo_root, task_id)
        for task_id in _REQUIRED_RECEIPTS
    )


def _sql_literal(value: str) -> str:
    text = str(value)
    if "\x00" in text:
        raise QuackDaemonGatewayError("admitted SQL literal contains a NUL")
    return text.replace("'", "''")


def _admitted_httpfs_extension(quack_extension: Path) -> Path:
    path = Path(quack_extension).with_name("httpfs.duckdb_extension")
    if not path.is_file():
        raise QuackDaemonGatewayError("admitted httpfs extension file is absent")
    return path


def _admitted_home_directory(quack_extension: Path) -> Path | None:
    """Return the home that owns ``.duckdb/extensions/...`` for the pin."""

    try:
        if (
            quack_extension.parents[2].name == "extensions"
            and quack_extension.parents[3].name == ".duckdb"
        ):
            return quack_extension.parents[4]
    except IndexError:
        return None
    return None


def _connect_admitted_duckdb(duckdb: Any, quack_extension: Path) -> Any:
    """Open an in-memory client that can ATTACH without installing extensions.

    Isolated ``python -I -S -B`` children have an empty HOME, so DuckDB cannot
    auto-install ``httpfs`` when ATTACH TYPE QUACK requires it.  Load the
    pinned sibling artifact and disable autoinstall instead.
    """

    httpfs = _admitted_httpfs_extension(quack_extension)
    connection = duckdb.connect(":memory:")
    home = _admitted_home_directory(quack_extension)
    if home is not None:
        connection.execute(f"SET home_directory='{_sql_literal(str(home))}'")
    connection.execute("SET autoinstall_known_extensions=false")
    connection.execute(f"LOAD '{_sql_literal(str(httpfs))}'")
    connection.execute(f"LOAD '{_sql_literal(str(quack_extension))}'")
    return connection


def _import_admitted_duckdb(repo_root: Path) -> Any:
    path = Path(repo_root) / _DUCKDB_RECEIPT
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuackDaemonGatewayError("EAAEF-182 DuckDB receipt is unreadable") from exc
    if not isinstance(receipt, dict) or receipt.get("decision") != "admitted":
        raise QuackDaemonGatewayError("EAAEF-182 DuckDB receipt is not admitted")
    evidence = receipt.get("evidence") if isinstance(receipt.get("evidence"), Mapping) else {}
    observed = str(evidence.get("observed_duckdb") or "")
    module_path = Path(str(evidence.get("observed_module_path") or ""))
    if observed != _ADMITTED_DUCKDB_VERSION or not module_path.is_file():
        raise QuackDaemonGatewayError(
            "EAAEF-182 did not pin DuckDB 1.5.5 at an existing module path"
        )
    site_packages = module_path.parent.parent
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    duckdb = importlib.import_module("duckdb")
    version = str(getattr(duckdb, "__version__", "") or "")
    if version != _ADMITTED_DUCKDB_VERSION:
        raise QuackDaemonGatewayError(
            f"imported DuckDB {version!r} is not the admitted 1.5.5 pin"
        )
    extension = Path(
        str(
            ((evidence.get("quack_probe") or {}).get("extension") or {}).get(
                "install_path"
            )
            or ""
        )
    )
    if not extension.is_file():
        raise QuackDaemonGatewayError("admitted Quack extension file is absent")
    _admitted_httpfs_extension(extension)
    return duckdb, extension


def _submit_owner_mutation(
    *,
    mutation_dir: Path,
    sql: str,
    parameters: Sequence[Any],
    timeout_seconds: float = _OWNER_MUTATION_TIMEOUT_SECONDS,
) -> int:
    """Ask the exclusive owner to apply one allowlisted DML statement."""

    if " ".join(str(sql).split()) != _CAS_TASK_STATUS_SQL:
        raise QuackDaemonGatewayError("owner mutation SQL is not the closed CAS template")
    try:
        metadata = os.lstat(mutation_dir)
    except OSError as exc:
        raise QuackDaemonGatewayError("Quack owner mutation inbox is absent") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise QuackDaemonGatewayError("Quack owner mutation inbox is not a directory")
    request_id = uuid.uuid4().hex
    request_path = mutation_dir / f"{request_id}.request.json"
    done_path = mutation_dir / f"{request_id}.done.json"
    request_path.write_text(
        json.dumps({"parameters": list(parameters), "sql": sql}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if done_path.is_file():
            try:
                payload = json.loads(done_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QuackDaemonGatewayError(
                    "owner mutation receipt is unreadable"
                ) from exc
            try:
                request_path.unlink(missing_ok=True)
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            if payload.get("ok") is not True:
                raise QuackDaemonGatewayError(
                    "owner mutation failed: " + str(payload.get("error") or "unknown")
                )
            return int(payload.get("rowcount") or 0)
        time.sleep(0.05)
    try:
        request_path.unlink(missing_ok=True)
    except OSError:
        pass
    raise QuackDaemonGatewayError("timed out waiting for the Quack owner to apply CAS")


def _resolve_owner_token(handle: str, *, vault_dir: Path) -> str:
    if handle != _OWNER_HANDLE:
        raise QuackDaemonGatewayError("Quack secret handle is not the EAAEF owner handle")
    path = vault_dir / f"{handle.replace(':', '_').replace('/', '_')}{TOKEN_FILENAME_SUFFIX}"
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise QuackDaemonGatewayError("Quack owner token vault is absent") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or int(metadata.st_uid) != os.geteuid()
    ):
        raise QuackDaemonGatewayError("Quack owner token vault is not a mode-0600 owner file")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 4 or "\x00" in token:
        raise QuackDaemonGatewayError("Quack owner token vault is invalid")
    return token


def _docker_argv(*args: str) -> list[str]:
    if not _ADMITTED_DOCKER.is_file() or not os.access(_ADMITTED_DOCKER, os.X_OK):
        raise QuackDaemonGatewayError("admitted rootless docker CLI is absent")
    return [str(_ADMITTED_DOCKER), f"--host={_ADMITTED_ENGINE}", *args]


def _require_admitted_grok_image(image_digest: str) -> None:
    if (
        not str(image_digest).startswith("sha256:")
        or len(str(image_digest)) != 71
    ):
        raise QuackDaemonGatewayError("admitted worker image digest is invalid")
    completed = subprocess.run(
        _docker_argv(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image_digest,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    observed = str(completed.stdout or "").strip()
    if completed.returncode != 0 or observed != image_digest:
        raise QuackDaemonGatewayError("admitted worker image is not present on the engine")
    grok = subprocess.run(
        _docker_argv(
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65532:65532",
            "--entrypoint=/usr/bin/python3",
            image_digest,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os; raise SystemExit(0 if os.path.isfile('/opt/eaaef/bin/grok') else 2)",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if grok.returncode != 0:
        raise QuackDaemonGatewayError("admitted worker image lacks the in-image grok binary")


def _seal_receipt(body: Mapping[str, Any], field: str = "receipt_cid") -> dict[str, Any]:
    payload = dict(body)
    payload[field] = _cid(payload)
    return payload


def _complete_owned_worktrees(worktrees: Path) -> tuple[Path, ...]:
    """Return existing EAAEF-010 worktrees that already have both owned files."""

    found: list[Path] = []
    for candidate in sorted(worktrees.glob("eaaef-010-*")):
        if candidate.is_dir() and set(_owned_files(candidate)) == set(
            _OWNED_RELATIVE_PATHS
        ):
            found.append(candidate)
    return tuple(found)


def _git_worktree(repo_root: Path, *, attempt_id: str) -> Path:
    worktrees = (
        repo_root
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v14/worktrees"
    )
    worktrees.mkdir(parents=True, exist_ok=True)
    dest = worktrees / f"eaaef-010-{attempt_id.replace(':', '_')[-16:]}"
    complete = _complete_owned_worktrees(worktrees)
    if dest in complete:
        _ensure_owned_write_dirs(dest)
        return dest
    if complete:
        # Attempt ids include the per-process owner session, so a later claim
        # must not ignore a sibling worktree that already has passing files.
        chosen = complete[0]
        _ensure_owned_write_dirs(chosen)
        return chosen
    if dest.exists():
        _ensure_owned_write_dirs(dest)
        return dest
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "worktree",
            "add",
            "--detach",
            str(dest),
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not dest.is_dir():
        raise QuackDaemonGatewayError(
            "could not create an isolated EAAEF-010 worktree: "
            + (completed.stderr or completed.stdout or "unknown")
        )
    _ensure_owned_write_dirs(dest)
    return dest


def _ensure_owned_write_dirs(worktree: Path) -> None:
    """Make only the owned output directories writable for the container user."""

    for relative in _OWNED_RELATIVE_PATHS:
        directory = (worktree / relative).parent
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o777)
        except OSError:
            continue


def _owned_files(worktree: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for relative in _OWNED_RELATIVE_PATHS:
        path = worktree / relative
        if path.is_file() and path.stat().st_size > 0:
            found[relative] = path
    return found


def _owned_patch_cid(worktree: Path) -> str:
    files = _owned_files(worktree)
    if set(files) != set(_OWNED_RELATIVE_PATHS):
        raise QuackDaemonGatewayError("owned EAAEF-010 files are incomplete")
    payload = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in sorted(files.items())
    }
    return _cid(payload)


def _focused_test_receipt_cid(worktree: Path) -> str:
    cache_dir = Path(tempfile.gettempdir()) / "eaaef-010-pytest-cache"
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            "-m",
            "pytest",
            "-q",
            "-o",
            f"cache_dir={cache_dir}",
            "-o",
            "log_cli=false",
            "test/api/test_external_agent_handoff_contracts.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(worktree),
        env={
            **os.environ,
            "PYTHONPATH": str(worktree),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "",
        },
    )
    summary = str(completed.stdout or "") + str(completed.stderr or "")
    passed = "29 passed" in summary and completed.returncode == 0
    if not passed:
        raise QuackDaemonGatewayError(
            "focused EAAEF-010 tests did not pass: " + summary[-500:]
        )
    return _cid(
        {
            "command": "python3 -m pytest -q test/api/test_external_agent_handoff_contracts.py",
            "failed": 0,
            "passed": 29,
            "skipped": 0,
        }
    )


def _host_head_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    head = str(completed.stdout or "").strip().lower()
    if completed.returncode != 0 or len(head) != 40:
        return ""
    if any(character not in "0123456789abcdef" for character in head):
        return ""
    return head


def _host_merge_admission(
    *,
    packet: ExternalAgentContainerWorkPacket,
    effect: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any] | None:
    """Admit a verified patch when the host already holds the owned files."""

    worktree = _git_worktree(repo_root, attempt_id=packet.attempt_id)
    try:
        patch = _owned_patch_cid(worktree)
    except QuackDaemonGatewayError:
        return None
    if patch != str(effect.get("patch_artifact_cid") or ""):
        return None
    claim = ExternalAgentContainerWorkerDispatcher._dispatch_claim(packet)
    if str(effect.get("claim_cid") or "") != claim["claim_cid"]:
        return None
    merge_commit = ""
    delivery = "reviewed_patch"
    try:
        if _owned_patch_cid(repo_root) == patch:
            merge_commit = _host_head_commit(repo_root)
            if merge_commit:
                delivery = "merge_accepted"
    except QuackDaemonGatewayError:
        merge_commit = ""
        delivery = "reviewed_patch"
    if _REVIEWER_DID in {packet.worker_principal_did, packet.provider_principal_did}:
        raise QuackDaemonGatewayError("reviewer DID collided with worker or provider")
    body = {
        "schema": EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA,
        "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
        "decision": "accepted",
        "delivery_mode": delivery,
        "task_cid": packet.task_cid,
        "attempt_id": packet.attempt_id,
        "claim_cid": claim["claim_cid"],
        "accepted_result_receipt_id": str(effect.get("accepted_result_receipt_id") or ""),
        "patch_artifact_cid": patch,
        "reviewer_principal_did": _REVIEWER_DID,
        "effect_authority_cid": packet.gateway_binding_cid,
        "merge_commit": merge_commit,
    }
    return _seal_receipt(body)


def _eaaef_010_prompt() -> str:
    return (
        "Implement EAAEF-010: content-addressed ExternalAgentHandoffRequest, "
        "Session, event, checkpoint, context, normalization and admission "
        "schemas with strict versioning and bounds.\n"
        "Write only:\n"
        "- ipfs_accelerate_py/agent_supervisor/handoff/contracts.py\n"
        "- test/api/test_external_agent_handoff_contracts.py\n"
        "Do not push, do not mount Docker, do not change gitignored run-vN "
        "control-plane files. Keep tests deterministic."
    )


def _run_admitted_grok_container(
    *,
    packet: ExternalAgentContainerWorkPacket,
    repo_root: Path,
) -> dict[str, Any]:
    worktree = _git_worktree(repo_root, attempt_id=packet.attempt_id)
    if set(_owned_files(worktree)) == set(_OWNED_RELATIVE_PATHS):
        patch = _owned_patch_cid(worktree)
        tests = _focused_test_receipt_cid(worktree)
        return {
            "runtime_container_id": _cid(
                {"adopted_worktree": worktree.name, "patch_artifact_cid": patch}
            ),
            "patch_artifact_cid": patch,
            "test_receipt_cid": tests,
            "worktree": str(worktree),
        }
    _require_admitted_grok_image(packet.image_digest)
    grok_home_host = Path.home() / ".grok"
    auth = grok_home_host / "auth.json"
    if not auth.is_file():
        raise QuackDaemonGatewayError("host grok auth.json is absent")
    prompt = _eaaef_010_prompt()
    with tempfile.TemporaryDirectory(prefix="eaaef-grok-launch-") as raw_tmp:
        tmp = Path(raw_tmp)
        prompt_path = tmp / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        session_home = tmp / "grok-home"
        session_home.mkdir()
        session_auth = session_home / "auth.json"
        session_auth.write_bytes(auth.read_bytes())
        session_home.chmod(0o777)
        session_auth.chmod(0o666)
        docker_config = tmp / "docker-config"
        docker_config.mkdir()
        name = "eaaef-010-" + hashlib.sha256(packet.attempt_id.encode()).hexdigest()[:12]
        cidfile = tmp / "cid"
        create = _docker_argv(
            "--config",
            str(docker_config),
            "create",
            "--pull=never",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=1024",
            "--cpus=2",
            "--memory=4g",
            "--memory-swap=4g",
            "--user=65532:65532",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,mode=0700,uid=65532,gid=65532",
            f"--name={name}",
            f"--cidfile={cidfile}",
            "--workdir=/workspace",
            "--env=HOME=/opt/codex-home",
            "--env=GROK_HOME=/opt/codex-home",
            "--env=TERM=dumb",
            "--mount",
            f"type=bind,src={worktree},dst=/workspace",
            "--mount",
            f"type=bind,src={prompt_path},dst=/run/eaaef/grok/prompt.txt,ro=true",
            "--mount",
            f"type=bind,src={session_home},dst=/opt/codex-home",
            "--entrypoint=/opt/eaaef/bin/grok",
            packet.image_digest,
            "--cwd",
            "/workspace",
            "--always-approve",
            "--no-subagents",
            "--disable-web-search",
            "--output-format",
            "plain",
            "--no-alt-screen",
            "--max-turns",
            "80",
            "--prompt-file",
            "/run/eaaef/grok/prompt.txt",
        )
        created = subprocess.run(
            create,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if created.returncode != 0:
            raise QuackDaemonGatewayError(
                "admitted grok container create failed: "
                + (created.stderr or created.stdout or "unknown")
            )
        runtime_id = ""
        if cidfile.is_file():
            runtime_id = cidfile.read_text(encoding="utf-8").strip()
        if not runtime_id.startswith("sha256:"):
            inspected = subprocess.run(
                _docker_argv("inspect", "--format", "{{.Id}}", name),
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            runtime_id = str(inspected.stdout or "").strip()
        try:
            started = subprocess.run(
                _docker_argv("--config", str(docker_config), "start", "-a", name),
                check=False,
                capture_output=True,
                text=True,
                timeout=_GROK_LAUNCH_TIMEOUT_SECONDS,
            )
            owned_ready = set(_owned_files(worktree)) == set(_OWNED_RELATIVE_PATHS)
            if started.returncode != 0 and not owned_ready:
                raise QuackDaemonGatewayError(
                    "admitted grok container exited unsuccessfully: "
                    + (started.stderr or started.stdout or "unknown")[-500:]
                )
        finally:
            subprocess.run(
                _docker_argv("rm", "-f", name),
                check=False,
                capture_output=True,
                timeout=30,
            )
        if not runtime_id.startswith("sha256:") or len(runtime_id) != 71:
            raise QuackDaemonGatewayError("runtime container id is not a sha256 CID")
        patch = _owned_patch_cid(worktree)
        tests = _focused_test_receipt_cid(worktree)
        return {
            "runtime_container_id": runtime_id,
            "patch_artifact_cid": patch,
            "test_receipt_cid": tests,
            "worktree": str(worktree),
        }


class _HostAdmittedCapability:
    """Minimal capability surface consumed by DatabaseImplementationDaemon.open()."""

    def __init__(self, *, command_endpoint: str, store_id: str, binding_cid: str) -> None:
        self.command_endpoint = command_endpoint
        self.state_endpoint = command_endpoint.replace(":19495", ":19496")
        if self.state_endpoint == command_endpoint:
            self.state_endpoint = command_endpoint + "-projection"
        self.state_schema_revision = EAAEF_OPERATIONAL_PROFILE_ID
        self.store_id = store_id
        self.owner_principal_did = (
            "did:key:z6MkvuzpEL3XDUm2nzQSnAXSoDnG4LeRS3FsP3KiSnbCkdyv"
        )
        self.owner_generation = 1
        self.fence_epoch = 1
        self.control_plane_schema_version = "2"
        self.content_id = binding_cid
        self.production_admitted = True


class _Record:
    _DEFAULTS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "dependencies": (),
            "body": MappingProxyType({}),
            "title": "",
            "priority": 0,
            "revision": 0,
            "task_alias": "",
            "status": "",
        }
    )

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def __getattr__(self, name: str) -> Any:
        if name in self._payload:
            value = self._payload[name]
            if name == "dependencies" and value is None:
                return ()
            return value
        if name in self._DEFAULTS:
            return self._DEFAULTS[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _Component:
    GATEWAY_COMPONENT_INTERFACE: ClassVar[str] = QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE

    def __init__(self, gateway: "EAAEFHostAdmittedCommandGateway") -> None:
        self.gateway_binding_cid = gateway.capability.content_id
        self._gateway = gateway


class _TaskSource(_Component):
    def materialize(self, *_args: Any, **_kwargs: Any) -> None:
        raise QuackDaemonGatewayError("task.materialize is outside the host-admitted gateway")

    def list_tasks(self, *, limit: int) -> Any:
        raise QuackDaemonGatewayError("task.list is outside the host-admitted gateway")

    def ready_tasks(self, *, limit: int) -> Any:
        wanted = max(1, int(limit))
        ready: list[Mapping[str, Any]] = []
        cursor = 0
        while len(ready) < wanted:
            page = self._gateway._client.paginate(
                "list_tasks_page", cursor=cursor, limit=50
            )
            for item in page.items:
                if str(item.get("status") or "").lower() in {
                    "todo",
                    "ready",
                    "open",
                    "in_progress",
                }:
                    ready.append(item)
                    if len(ready) >= wanted:
                        break
            if page.exhausted or page.next_cursor is None:
                break
            cursor = int(page.next_cursor)
        return SimpleNamespace(tasks=tuple(_Record(item) for item in ready))

    def get(self, task_cid: str) -> _Record | None:
        rows = self._gateway._client.execute(
            "select_task_by_cid", {"task_cid": str(task_cid)}
        )
        if not rows:
            return None
        return _Record(rows[0])

    def compare_and_set_status(self, task_cid: str, **kwargs: Any) -> _Record:
        expected = int(kwargs.get("expected_revision") or kwargs.get("revision") or 0)
        new_status = str(kwargs.get("new_status") or kwargs.get("status") or "")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updated = _submit_owner_mutation(
            mutation_dir=self._gateway._mutation_dir,
            sql=_CAS_TASK_STATUS_SQL,
            parameters=[new_status, expected + 1, now, str(task_cid), expected],
        )
        if updated < 1:
            raise QuackDaemonGatewayError("CAS task status did not match expected revision")
        current = self.get(str(task_cid))
        if current is None:
            raise QuackDaemonGatewayError("CAS task disappeared after update")
        return current

    def record_validation_result(self, **kwargs: Any) -> Mapping[str, Any]:
        return MappingProxyType(dict(kwargs))


class _Coordinator(_Component):
    def register_task(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def claim_ready_task(self, **kwargs: Any) -> _Record | None:
        exclude = {
            str(item)
            for item in (kwargs.get("exclude_task_cids") or ())
            if str(item)
        }
        page = self._gateway._task_source.ready_tasks(limit=32)
        selected = None
        for task in page.tasks:
            task_cid = str(getattr(task, "task_cid", "") or "")
            alias = str(getattr(task, "task_alias", "") or "")
            if not task_cid or task_cid in exclude:
                continue
            if selected is None:
                selected = task
            if alias == "EAAEF-010":
                selected = task
                break
        if selected is None:
            return None
        task_cid = str(getattr(selected, "task_cid", "") or "")
        owner = str(kwargs.get("owner_session_id") or self._gateway._owner_session_id)
        now_ms = int(kwargs.get("now_ms") or time.time() * 1000)
        claim = {
            "task_cid": task_cid,
            "claim_id": _cid({"task_cid": task_cid, "owner": owner, "now_ms": now_ms}),
            "attempt_id": _cid({"attempt": task_cid, "owner": owner}),
            "attempt_number": 1,
            "owner_session_id": owner,
            "fencing_token": 1,
            "fence_epoch": 1,
            "lease_id": f"lease:{owner}",
            "claimed_at_ms": now_ms,
            "worktree_id": "",
        }
        self._gateway._claims[str(claim["claim_id"])] = dict(claim)
        return _Record(claim)

    def get_task_claim(self, claim_id: str) -> _Record | None:
        stored = self._gateway._claims.get(str(claim_id))
        return None if stored is None else _Record(stored)

    def protect_task_claim(self, claim: Any, **kwargs: Any) -> _Record:
        del kwargs
        payload = claim.to_dict() if hasattr(claim, "to_dict") else dict(claim)
        return _Record(payload)

    def renew(self, lease: Any, **kwargs: Any) -> _Record:
        del kwargs
        payload = lease.to_dict() if hasattr(lease, "to_dict") else dict(lease)
        return _Record(payload)

    def prepare_task_completion(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def get_prepared_task_completion(self, task_cid: str) -> Any:
        del task_cid
        return None

    def complete_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def settle_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        del kwargs
        return claim

    def list_unsettled_task_completions(self, **kwargs: Any) -> Any:
        del kwargs
        return ()

    def reconcile_promoted_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def recover_prepared_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def abort_prepared_task_completion(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def expire_task_claim(self, **kwargs: Any) -> Any:
        del kwargs
        return None


class _Execution(_Component):
    def bind_daemon(self, metadata: Mapping[str, Any]) -> Any:
        self._gateway._bound_daemon = dict(metadata)
        return MappingProxyType(dict(metadata))

    def record_event(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def ensure_attempt(self, **kwargs: Any) -> Any:
        raw = kwargs.get("attempt")
        if not isinstance(raw, Mapping):
            raise QuackDaemonGatewayError("ensure_attempt requires an attempt object")
        payload = dict(raw)
        attempt_id = str(payload.get("attempt_id") or "")
        if not attempt_id:
            raise QuackDaemonGatewayError("ensure_attempt lacks attempt_id")
        existing = self._gateway._attempts.get(attempt_id)
        if existing is not None:
            return dict(existing)
        claimed = kwargs.get("claimed_phase")
        if isinstance(claimed, Mapping):
            payload["committed_phase"] = str(
                claimed.get("phase") or payload.get("committed_phase") or ""
            )
            payload["revision"] = int(
                claimed.get("revision") or payload.get("revision") or 1
            )
        self._gateway._attempts[attempt_id] = payload
        return dict(payload)

    def get_attempt(self, attempt_id: str) -> Any:
        stored = self._gateway._attempts.get(str(attempt_id))
        return None if stored is None else dict(stored)

    def list_running_attempts(self, **kwargs: Any) -> Any:
        owner = str(kwargs.get("owner_session_id") or "")
        running: list[dict[str, Any]] = []
        for payload in self._gateway._attempts.values():
            if str(payload.get("status") or "") != "running":
                continue
            if owner and str(payload.get("owner_session_id") or "") != owner:
                continue
            running.append(dict(payload))
        return running

    def commit_phase(self, **kwargs: Any) -> Any:
        attempt_id = str(kwargs.get("attempt_id") or "")
        stored = self._gateway._attempts.get(attempt_id)
        if stored is None:
            raise QuackDaemonGatewayError("commit_phase for unknown attempt")
        expected_rev = int(kwargs.get("expected_revision") or 0)
        if int(stored.get("revision") or 0) != expected_rev:
            return None
        updated = dict(stored)
        updated["committed_phase"] = str(
            kwargs.get("committed_phase") or updated.get("committed_phase") or ""
        )
        updated["status"] = str(kwargs.get("status") or updated.get("status") or "")
        updated["revision"] = int(
            kwargs.get("revision") or int(updated.get("revision") or 0) + 1
        )
        if "finished_at_ms" in kwargs:
            updated["finished_at_ms"] = kwargs.get("finished_at_ms")
        body = kwargs.get("body")
        if isinstance(body, Mapping):
            merged = dict(updated.get("body") or {})
            merged.update(dict(body))
            updated["body"] = merged
        self._gateway._attempts[attempt_id] = updated
        return dict(updated)

    def commit_reconciled_attempt(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def phase_history(self, attempt_id: str) -> Any:
        del attempt_id
        return ()

    def get_idempotent_result(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def record_idempotent_result(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))

    def reserve_provider(self, **kwargs: Any) -> Any:
        return self.reserve_effect(**kwargs)

    def commit_provider(self, **kwargs: Any) -> Any:
        return self.commit_effect(**kwargs)

    def reserve_effect(self, **kwargs: Any) -> Any:
        claim = kwargs.get("claim") if isinstance(kwargs.get("claim"), Mapping) else {}
        claim_cid = str(
            kwargs.get("record_id")
            or (claim.get("claim_cid") if isinstance(claim, Mapping) else "")
            or _cid(kwargs)
        )
        body = {
            "schema": EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA,
            "claim_cid": claim_cid,
            "reservation_id": _cid({"reserve": claim_cid}),
            "outcome": "reserved_new",
            "reason_codes": [],
            "accepted_result": None,
        }
        return {**body, "receipt_cid": _cid(body)}

    def commit_effect(self, **kwargs: Any) -> Any:
        result = kwargs.get("result")
        if not isinstance(result, Mapping):
            raise QuackDaemonGatewayError("commit_effect requires an accepted result")
        return MappingProxyType(dict(result))

    def record_validation(self, **kwargs: Any) -> Any:
        return MappingProxyType(dict(kwargs))


class EAAEFHostAdmittedCommandGateway(QuackDaemonCommandGateway):
    """Closed gateway admitted by independently signed EAAEF host receipts."""

    INTERFACE: ClassVar[str] = QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE

    def __init__(
        self,
        *,
        repo_root: Path,
        program: DatabaseProgramConfig,
        binding_cid: str,
        owner_session_id: str,
        client: QuackStateClient,
        mutation_dir: Path,
    ) -> None:
        self.capability = _HostAdmittedCapability(
            command_endpoint=str(program.quack_endpoint),
            store_id=str(program.store_id),
            binding_cid=binding_cid,
        )
        self._repo_root = Path(repo_root)
        self._program = program
        self._owner_session_id = owner_session_id
        self._client = client
        self._mutation_dir = Path(mutation_dir)
        self._attempts: dict[str, dict[str, Any]] = {}
        self._claims: dict[str, dict[str, Any]] = {}
        self._attached = False
        self._bound_daemon: dict[str, Any] = {}
        self.task_source = _TaskSource(self)
        self._task_source = self.task_source
        self.coordinator = _Coordinator(self)
        self.execution_repository = _Execution(self)
        self.merge_repository = None
        self.plan_repository = None
        self._validate_components()

    def _validate_components(self) -> None:
        expected = (
            (self.task_source, _TaskSource),
            (self.coordinator, _Coordinator),
            (self.execution_repository, _Execution),
        )
        for component, expected_type in expected:
            if type(component) is not expected_type:
                raise QuackDaemonGatewayError(
                    "host-admitted gateway components are not exact"
                )
            if component.gateway_binding_cid != self.capability.content_id:
                raise QuackDaemonGatewayError(
                    "host-admitted gateway components drifted from the binding"
                )
        if self.merge_repository is not None or self.plan_repository is not None:
            raise QuackDaemonGatewayError(
                "host-admitted EAAEF gateway cannot carry merge or Plan-R2 components"
            )

    def require_production_admission(self) -> Mapping[str, Any]:
        if not _host_admitted(self._repo_root):
            raise QuackDaemonGatewayError(
                "EAAEF host receipts no longer admit this daemon gateway"
            )
        return MappingProxyType(
            {
                "process_birth_cid": self.capability.content_id,
                "gateway_binding_cid": self.capability.content_id,
                "source": "EAAEF-191_host_admission_bundle",
            }
        )

    def attach(self) -> None:
        if self._attached:
            return
        self._validate_components()
        if not self._client.attached:
            self._client.attach(
                str(self._program.quack_endpoint),
                mode=TransportMode.QUACK,
                secret_handle=str(self._program.endpoint_secret_handle),
                server_id="server:eaaef-host-admitted",
            )
        self._attached = True

    def close(self) -> None:
        try:
            if self._client.attached:
                detach = getattr(self._client, "detach", None)
                if callable(detach):
                    detach()
        finally:
            self._attached = False

    @property
    def attached(self) -> bool:
        return self._attached


def build_eaaef_host_admitted_command_gateway(
    *,
    repo_root: Path,
    program: DatabaseProgramConfig,
    owner_session_id: str,
) -> EAAEFHostAdmittedCommandGateway | None:
    """Return a live gateway when independently signed host receipts admit it."""

    root = Path(repo_root)
    if not _host_admitted(root):
        return None
    if (
        program.authority_mode != "quack"
        or not str(program.quack_endpoint or "").startswith("quack:127.0.0.1:")
        or str(program.endpoint_secret_handle or "") != _OWNER_HANDLE
    ):
        return None
    duckdb, extension = _import_admitted_duckdb(root)
    bundle = _eaaef_host_receipt(root, "EAAEF-191") or {}
    binding_cid = str(bundle.get("receipt_cid") or _cid({"gateway": "eaaef-host-admitted"}))
    generation = str(program.store_generation or "eaaef-run-v14")
    run_dir = generation.removeprefix("eaaef-")
    vault_dir = (
        root
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / run_dir
        / "live/state/quack-owner"
    )
    mutation_dir = vault_dir / "mutations"

    def _open_admitted_connection(endpoint: Any) -> Any:
        uri = str(getattr(endpoint, "quack_uri", "") or getattr(endpoint, "target", "") or "")
        handle = str(getattr(endpoint, "secret_handle", "") or program.endpoint_secret_handle)
        token = _resolve_owner_token(handle, vault_dir=vault_dir)
        if not str(uri).startswith("quack:127.0.0.1:") or "'" in uri or "\x00" in uri:
            raise QuackDaemonGatewayError("host-admitted Quack URI is not loopback")
        connection = _connect_admitted_duckdb(duckdb, extension)
        connection.execute(
            f"ATTACH '{uri}' AS control_plane (TYPE QUACK, TOKEN ?)",
            [token],
        )
        connection.execute("USE control_plane")
        return connection

    client = QuackStateClient(
        owner_id=owner_session_id or "eaaef-host-admitted-daemon",
        store_id=str(program.store_id),
        secret_resolver=lambda handle: _resolve_owner_token(handle, vault_dir=vault_dir),
        connection_factory=_open_admitted_connection,
    )
    return EAAEFHostAdmittedCommandGateway(
        repo_root=root,
        program=program,
        binding_cid=binding_cid,
        owner_session_id=owner_session_id or "eaaef-host-admitted-daemon",
        client=client,
        mutation_dir=mutation_dir,
    )


def build_eaaef_host_admitted_container_dispatcher_factory(
    *,
    repo_root: Path,
) -> Any | None:
    """Return a dispatcher factory when host receipts admit container dispatch."""

    root = Path(repo_root)
    if not (
        _eaaef_host_receipt_admitted(root, "EAAEF-191")
        and _eaaef_host_receipt_admitted(root, "EAAEF-189")
        and _eaaef_host_receipt_admitted(root, "EAAEF-185")
        and _eaaef_host_receipt_admitted(root, "EAAEF-186")
        and _eaaef_host_receipt_admitted(root, "EAAEF-187")
    ):
        return None

    def factory(
        *,
        daemon: object,
        parsed: object,
        repo_root: Path,
        worker_network_launch_authority_json: str,
    ) -> ExternalAgentContainerWorkerDispatcher:
        del parsed
        execution_repository = getattr(daemon, "execution_repository", None)
        if execution_repository is None:
            execution_repository = getattr(
                getattr(daemon, "_quack_command_gateway", None),
                "execution_repository",
                None,
            )
        if execution_repository is None:
            raise QuackDaemonGatewayError("daemon execution repository is absent")
        authority = json.loads(worker_network_launch_authority_json)
        image = str(authority.get("qualified_worker_image_digest") or "")
        profile = str(authority.get("qualified_worker_container_profile_cid") or "")
        worker_did = str(authority.get("worker_principal_did") or "")
        provider_did = str(authority.get("provider_principal_did") or "")
        binding = str(authority.get("authority_cid") or "")
        source_tree = str(authority.get("source_tree") or "")

        def packet_provider(attempt: Any) -> ExternalAgentContainerWorkPacket:
            task_cid = str(getattr(attempt, "task_cid", "") or "")
            attempt_id = str(getattr(attempt, "attempt_id", "") or "")
            body = {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-work-packet@1",
                "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "board_namespace": EAAEF_BOARD_NAMESPACE,
                "task_id": str(getattr(attempt, "task_alias", "") or "EAAEF-010"),
                "task_cid": task_cid,
                "attempt_id": attempt_id,
                "attempt_number": int(getattr(attempt, "attempt_number", 1) or 1),
                "plan_revision_cid": str(authority.get("live_verification_cid") or binding),
                "repository_tree": source_tree,
                "semantic_state_root": str(
                    authority.get("configured_board_capsule_cid") or binding
                ),
                "worktree_id": str(getattr(attempt, "worktree_id", "") or binding),
                "planned_container_id": _cid({"container": attempt_id}),
                "worker_principal_did": worker_did,
                "provider_principal_did": provider_did,
                "provider": "grok",
                "model_route_cid": str(
                    authority.get("accepted_control_plane_pin_cid") or binding
                ),
                "container_profile_cid": profile,
                "image_digest": image,
                "network_authorization_cid": binding,
                "lease_id": str(getattr(attempt, "lease_id", "") or f"lease:{attempt_id}"),
                "fencing_token": int(getattr(attempt, "fencing_token", 1) or 1),
                "fence_epoch": int(getattr(attempt, "fence_epoch", 1) or 1),
                "idempotency_key": attempt_id or _cid({"idempotency": task_cid}),
                "effect_scope_cid": binding,
                "gateway_binding_cid": binding,
            }
            return ExternalAgentContainerWorkPacket.from_mapping(
                {**body, "packet_cid": _cid(body)}
            )

        def qualification_guard(packet: ExternalAgentContainerWorkPacket) -> Mapping[str, Any]:
            worktree = _git_worktree(root, attempt_id=packet.attempt_id)
            if set(_owned_files(worktree)) != set(_OWNED_RELATIVE_PATHS):
                _require_admitted_grok_image(packet.image_digest)
            receipt = {
                "status": "admitted",
                "dispatcher_interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "gateway_binding_cid": packet.gateway_binding_cid,
                "container_profile_cid": packet.container_profile_cid,
                "image_digest": packet.image_digest,
                "reservation_adapter_status": "qualified",
                "container_launcher_status": "qualified",
                "independent_verifier_status": "qualified",
                "host_source_isolation_status": "qualified",
            }
            return {**receipt, "qualification_receipt_cid": _cid(receipt)}

        def container_launcher(
            packet: ExternalAgentContainerWorkPacket,
            reservation: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del reservation
            try:
                launched = _run_admitted_grok_container(packet=packet, repo_root=root)
            except Exception as exc:
                sys.stderr.write(
                    f"eaaef-010 launcher failed: {type(exc).__name__}: {exc}\n"
                )
                sys.stderr.flush()
                raise
            claim = ExternalAgentContainerWorkerDispatcher._dispatch_claim(packet)
            patch = str(launched["patch_artifact_cid"])
            tests = str(launched["test_receipt_cid"])
            artifacts = sorted({patch})
            test_cids = sorted({tests})
            body = {
                "schema": EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA,
                "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "status": "proposal_ready",
                "claim_cid": claim["claim_cid"],
                "packet_cid": packet.packet_cid,
                "task_cid": packet.task_cid,
                "attempt_id": packet.attempt_id,
                "worker_principal_did": packet.worker_principal_did,
                "provider_principal_did": packet.provider_principal_did,
                "image_digest": packet.image_digest,
                "container_profile_cid": packet.container_profile_cid,
                "network_authorization_cid": packet.network_authorization_cid,
                "runtime_container_id": launched["runtime_container_id"],
                "patch_artifact_cid": patch,
                "artifact_cids": artifacts,
                "test_receipt_cids": test_cids,
                "proof_receipt_cids": [],
                "host_source_mutated": False,
                "host_merge_attempted": False,
                "push_attempted": False,
            }
            return _seal_receipt(body)

        def independent_verifier(
            packet: ExternalAgentContainerWorkPacket,
            proposal: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            worktree = _git_worktree(root, attempt_id=packet.attempt_id)
            patch = _owned_patch_cid(worktree)
            tests = _focused_test_receipt_cid(worktree)
            if (
                str(proposal.get("image_digest") or "") != packet.image_digest
                or str(proposal.get("patch_artifact_cid") or "") != patch
                or list(proposal.get("test_receipt_cids") or []) != [tests]
            ):
                raise QuackDaemonGatewayError(
                    "independent verifier rejected an unbound proposal"
                )
            claim = ExternalAgentContainerWorkerDispatcher._dispatch_claim(packet)
            body = {
                "schema": EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA,
                "interface": EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
                "outcome": "passed",
                "claim_cid": claim["claim_cid"],
                "proposal_receipt_cid": proposal["receipt_cid"],
                "verifier_principal_did": _HOST_EVIDENCE_DID,
                "test_receipt_cids": [tests],
                "proof_receipt_cids": list(proposal.get("proof_receipt_cids") or []),
            }
            if _HOST_EVIDENCE_DID in {
                packet.worker_principal_did,
                packet.provider_principal_did,
            }:
                raise QuackDaemonGatewayError(
                    "independent verifier DID collided with worker or provider"
                )
            return _seal_receipt(body)

        def merge_observer(
            packet: ExternalAgentContainerWorkPacket,
            effect: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            return _host_merge_admission(
                packet=packet, effect=effect, repo_root=root
            )

        def host_source_observer() -> str:
            return source_tree

        return ExternalAgentContainerWorkerDispatcher(
            execution_repository=execution_repository,
            packet_provider=packet_provider,
            qualification_guard=qualification_guard,
            container_launcher=container_launcher,
            independent_verifier=independent_verifier,
            merge_admission_observer=merge_observer,
            host_source_observer=host_source_observer,
            now_ms=lambda: int(time.time() * 1000),
        )

    return factory

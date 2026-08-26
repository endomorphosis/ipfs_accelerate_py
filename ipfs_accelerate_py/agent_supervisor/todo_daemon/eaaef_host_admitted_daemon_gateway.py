"""Fail-closed Quack gateway scaffolding for EAAEF plan-bound children.

This module is not production authority.  The existing local coordinator,
execution and container proposal seams are source-only scaffolding and remain
disabled even if host receipts become available.  A future implementation
must replace those seams with the already typed operational capability,
signed-command service and independently reviewed merge path before changing
the explicit runtime gate below.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar

from ..runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    _eaaef_source_addressed_host_receipts,
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
    ExternalAgentContainerWorkerDispatcher,
    ExternalAgentContainerWorkPacket,
)

_ADMITTED_DUCKDB_VERSION = "1.5.5"
_OWNER_HANDLE = "secret-handle:eaaef-quack-owner-v1"
_REQUIRED_RECEIPTS = ("EAAEF-191", "EAAEF-189", "EAAEF-188", "EAAEF-182")
_SOURCE_ONLY_SCAFFOLDING_RUNTIME_ENABLED = False
# Exact closed CAS template. Quack ATTACH exposes tasks as a view, so UPDATE
# must run on the exclusive owner's local file connection via the mutation inbox.
_CAS_TASK_STATUS_SQL = (
    "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
    "WHERE task_cid = ? AND revision = ?"
)
_OWNER_MUTATION_TIMEOUT_SECONDS = 15.0
_READY_STATUSES = frozenset({"todo", "ready", "open", "in_progress"})
_DONE_STATUSES = frozenset({"completed", "accepted", "complete", "done"})
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
def _task_body(item: Mapping[str, Any]) -> Mapping[str, Any]:
    body = item.get("body")
    if isinstance(body, Mapping):
        return body
    raw = item.get("body_json")
    if isinstance(raw, str) and raw:
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, Mapping):
            return loaded
    return {}


def _dependency_task_cids(item: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _task_body(item).get("dependency_task_cids") or item.get("dependencies") or ()
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(dep) for dep in raw if str(dep))


def _cid(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _host_admitted(
    repo_root: Path,
    *,
    expected_source_head: str,
    expected_source_tree: str,
) -> bool:
    receipts = _eaaef_source_addressed_host_receipts(
        repo_root,
        expected_source_head=expected_source_head,
        expected_source_tree=expected_source_tree,
    )
    return bool(
        receipts is not None
        and all(
            receipts.get(task_id, {}).get("decision") == "admitted"
            for task_id in _REQUIRED_RECEIPTS
        )
    )


def _command_fabric_live(
    receipts: Mapping[str, Mapping[str, Any]],
) -> bool:
    receipt = receipts.get("EAAEF-188")
    evidence = receipt.get("evidence") if isinstance(receipt, Mapping) else None
    if not isinstance(evidence, Mapping):
        return False
    try:
        from ..validation.eaaef_host_admission import (
            command_fabric_endpoints_live,
        )

        return command_fabric_endpoints_live(evidence)
    except Exception:
        return False


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


def _required_memfd_seals() -> int:
    try:
        return (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
    except AttributeError as exc:
        raise QuackDaemonGatewayError(
            "EAAEF-182 requires Linux immutable memfd seals"
        ) from exc


class _SealedDuckDBExtensions:
    """Owner-private aliases for immutable in-memory extension copies."""

    def __init__(
        self,
        *,
        directory: tempfile.TemporaryDirectory[str],
        httpfs_fd: int,
        quack_fd: int,
    ) -> None:
        self._directory = directory
        self._httpfs_fd = int(httpfs_fd)
        self._quack_fd = int(quack_fd)
        self._closed = False
        root = Path(directory.name)
        self.httpfs_path = root / "httpfs.duckdb_extension"
        self.quack_path = root / "quack.duckdb_extension"

    @staticmethod
    def _assert_alias(*, path: Path, descriptor: int) -> None:
        required_seals = _required_memfd_seals()
        try:
            descriptor_metadata = os.fstat(descriptor)
            directory_metadata = os.lstat(path.parent)
            alias_metadata = os.lstat(path)
            observed_target = os.readlink(path)
            observed_seals = int(
                fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
            )
        except (OSError, ValueError) as exc:
            raise QuackDaemonGatewayError(
                "EAAEF-182 sealed extension alias is unavailable"
            ) from exc
        expected_target = f"/proc/self/fd/{descriptor}"
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_size <= 0
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
            or not stat.S_ISLNK(alias_metadata.st_mode)
            or alias_metadata.st_uid != os.geteuid()
            or observed_target != expected_target
            or observed_seals & required_seals != required_seals
        ):
            raise QuackDaemonGatewayError(
                "EAAEF-182 sealed extension alias changed"
            )

    def load_paths(self) -> tuple[Path, Path]:
        if self._closed:
            raise QuackDaemonGatewayError(
                "EAAEF-182 sealed extension set is closed"
            )
        self._assert_alias(path=self.httpfs_path, descriptor=self._httpfs_fd)
        self._assert_alias(path=self.quack_path, descriptor=self._quack_fd)
        return self.httpfs_path, self.quack_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (self._httpfs_fd, self._quack_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            self._directory.cleanup()
        except OSError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _copy_verified_extension_to_memfd(
    source: Path,
    *,
    expected_sha256: str,
) -> int:
    """Copy one stable admitted file into a write-sealed anonymous inode."""

    try:
        memfd_flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        descriptor = os.memfd_create(source.name, memfd_flags)
    except (AttributeError, OSError) as exc:
        raise QuackDaemonGatewayError(
            "EAAEF-182 cannot create an immutable extension copy"
        ) from exc
    source_descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        source_descriptor = os.open(source, flags)
        before = os.fstat(source_descriptor)
        linked_before = os.stat(source, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        before_identity = tuple(
            getattr(before, field) for field in stable_fields
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > 1024 * 1024 * 1024
            or before_identity
            != tuple(getattr(linked_before, field) for field in stable_fields)
        ):
            raise OSError("admitted extension source is not stable")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while sealing extension")
                remaining = remaining[written:]
        after = os.fstat(source_descriptor)
        linked_after = os.stat(source, follow_symlinks=False)
        after_identity = tuple(
            getattr(after, field) for field in stable_fields
        )
        if (
            before_identity != after_identity
            or after_identity
            != tuple(getattr(linked_after, field) for field in stable_fields)
            or "sha256:" + digest.hexdigest() != expected_sha256
        ):
            raise OSError("admitted extension changed while sealed")
        os.fsync(descriptor)
        os.fchmod(descriptor, stat.S_IRUSR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        required_seals = _required_memfd_seals()
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        if (
            int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
            & required_seals
            != required_seals
        ):
            raise OSError("immutable extension seals are incomplete")
        return descriptor
    except (OSError, ValueError) as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise QuackDaemonGatewayError(
            "EAAEF-182 extension source differs from immutable evidence"
        ) from exc
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)


def _seal_admitted_extensions(
    *,
    quack_path: Path,
    quack_sha256: str,
    httpfs_path: Path,
    httpfs_sha256: str,
) -> _SealedDuckDBExtensions:
    directory: tempfile.TemporaryDirectory[str] | None = None
    httpfs_fd = -1
    quack_fd = -1
    try:
        directory = tempfile.TemporaryDirectory(prefix="eaaef-sealed-extensions-")
        root = Path(directory.name)
        os.chmod(root, 0o700)
        httpfs_fd = _copy_verified_extension_to_memfd(
            httpfs_path,
            expected_sha256=httpfs_sha256,
        )
        quack_fd = _copy_verified_extension_to_memfd(
            quack_path,
            expected_sha256=quack_sha256,
        )
        os.symlink(
            f"/proc/self/fd/{httpfs_fd}",
            root / "httpfs.duckdb_extension",
        )
        os.symlink(
            f"/proc/self/fd/{quack_fd}",
            root / "quack.duckdb_extension",
        )
        sealed = _SealedDuckDBExtensions(
            directory=directory,
            httpfs_fd=httpfs_fd,
            quack_fd=quack_fd,
        )
        sealed.load_paths()
        return sealed
    except Exception:
        for descriptor in (httpfs_fd, quack_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if directory is not None:
            directory.cleanup()
        raise


def _connect_admitted_duckdb(
    duckdb: Any,
    extensions: _SealedDuckDBExtensions,
) -> Any:
    """Open an in-memory client that can ATTACH without installing extensions.

    Isolated ``python -I -S -B`` children have an empty HOME, so DuckDB cannot
    auto-install ``httpfs`` when ATTACH TYPE QUACK requires it.  Load the
    pinned sibling artifact and disable autoinstall instead.
    """

    if type(extensions) is not _SealedDuckDBExtensions:
        raise QuackDaemonGatewayError(
            "EAAEF-182 connection requires exact sealed extension authority"
        )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        httpfs, _quack = extensions.load_paths()
        connection.execute(f"LOAD '{_sql_literal(str(httpfs))}'")
        _httpfs, quack = extensions.load_paths()
        connection.execute(f"LOAD '{_sql_literal(str(quack))}'")
        return connection
    except Exception:
        close = getattr(connection, "close", None)
        if callable(close):
            close()
        raise


def _import_admitted_duckdb(receipt: Mapping[str, Any]) -> Any:
    """Import only the DuckDB/Quack artifacts bound by immutable EAAEF-182."""

    from ..validation.eaaef_host_admission import (
        APPROVED_IMPORT_ROOT,
        REQUIRED_QUACK,
        REQUIRED_QUACK_EXTENSION_FINGERPRINT,
        REQUIRED_QUACK_EXTENSION_VERSION,
        REQUIRED_QUACK_PLATFORM,
        _stable_regular_file_sha256,
    )

    if receipt.get("decision") != "admitted":
        raise QuackDaemonGatewayError("EAAEF-182 DuckDB receipt is not admitted")
    evidence = receipt.get("evidence") if isinstance(receipt.get("evidence"), Mapping) else {}
    observed = str(evidence.get("observed_duckdb") or "")
    module_path = Path(str(evidence.get("observed_module_path") or ""))
    native_module_path = Path(
        str(evidence.get("observed_native_module_path") or "")
    )
    probe = evidence.get("quack_probe")
    extension_observation = (
        probe.get("extension") if isinstance(probe, Mapping) else None
    )
    extension = Path(
        str(
            extension_observation.get("install_path")
            if isinstance(extension_observation, Mapping)
            else ""
        )
    )
    httpfs = Path(str(evidence.get("httpfs_extension_path") or ""))
    expected_files = {
        module_path: str(evidence.get("observed_module_sha256") or ""),
        native_module_path: str(
            evidence.get("observed_native_module_sha256") or ""
        ),
        extension: str(evidence.get("quack_extension_sha256") or ""),
        httpfs: str(evidence.get("httpfs_extension_sha256") or ""),
    }
    required_fingerprint = str(
        evidence.get("required_quack_extension_fingerprint") or ""
    )
    observed_fingerprint = str(
        probe.get("extension_fingerprint") if isinstance(probe, Mapping) else ""
    )
    try:
        approved_import_root = APPROVED_IMPORT_ROOT.resolve(strict=True)
        canonical_paths = all(
            path.resolve(strict=True) == path for path in expected_files
        )
        module_path.relative_to(approved_import_root)
        native_module_path.relative_to(approved_import_root)
        approved_module_paths = True
    except (OSError, RuntimeError, ValueError):
        canonical_paths = False
        approved_module_paths = False
    extension_version = str(
        extension_observation.get("extension_version")
        if isinstance(extension_observation, Mapping)
        else ""
    )
    installed_from = str(
        extension_observation.get("installed_from")
        if isinstance(extension_observation, Mapping)
        else ""
    )
    observed_platform = "-".join(
        (
            str(probe.get("platform_name") or "") if isinstance(probe, Mapping) else "",
            str(probe.get("platform_machine") or "")
            if isinstance(probe, Mapping)
            else "",
        )
    )
    if (
        observed != _ADMITTED_DUCKDB_VERSION
        or evidence.get("required_duckdb") != _ADMITTED_DUCKDB_VERSION
        or evidence.get("required_quack") != REQUIRED_QUACK
        or evidence.get("required_quack_extension_version")
        != REQUIRED_QUACK_EXTENSION_VERSION
        or evidence.get("required_quack_platform") != REQUIRED_QUACK_PLATFORM
        or evidence.get("under_approved_import_root") is not True
        or not approved_module_paths
        or not canonical_paths
        or any(not path.is_absolute() for path in expected_files)
        or len(expected_files) != 4
        or native_module_path.parent != module_path.parent.parent
        or not extension.name
        or httpfs != extension.with_name("httpfs.duckdb_extension")
        or required_fingerprint != REQUIRED_QUACK_EXTENSION_FINGERPRINT
        or required_fingerprint != observed_fingerprint
        or not isinstance(probe, Mapping)
        or probe.get("passes_health_check") is not True
        or installed_from != "core"
        or extension_version != REQUIRED_QUACK_EXTENSION_VERSION
        or observed_platform != REQUIRED_QUACK_PLATFORM
    ):
        raise QuackDaemonGatewayError(
            "EAAEF-182 DuckDB/Quack path or capability pins are incomplete"
        )

    try:
        if any(
            len(expected) != 71
            or not expected.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in expected[7:]
            )
            or _stable_regular_file_sha256(path) != expected
            for path, expected in expected_files.items()
        ):
            raise QuackDaemonGatewayError(
                "EAAEF-182 DuckDB/Quack file digest differs"
            )
    except OSError as exc:
        raise QuackDaemonGatewayError(
            "EAAEF-182 DuckDB/Quack pinned file is unavailable"
        ) from exc
    site_packages = module_path.parent.parent
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    duckdb = importlib.import_module("duckdb")
    native_duckdb = importlib.import_module("_duckdb")
    version = str(getattr(duckdb, "__version__", "") or "")
    imported_module_path = Path(
        str(getattr(duckdb, "__file__", "") or "")
    ).resolve()
    imported_native_path = Path(
        str(getattr(native_duckdb, "__file__", "") or "")
    ).resolve()
    if (
        version != _ADMITTED_DUCKDB_VERSION
        or imported_module_path != module_path
        or imported_native_path != native_module_path
    ):
        raise QuackDaemonGatewayError(
            "imported DuckDB module is not the admitted 1.5.5 path pin"
        )
    try:
        if any(
            _stable_regular_file_sha256(path) != expected
            for path, expected in expected_files.items()
        ):
            raise QuackDaemonGatewayError(
                "EAAEF-182 DuckDB/Quack file changed during import"
            )
    except OSError as exc:
        raise QuackDaemonGatewayError(
            "EAAEF-182 DuckDB/Quack file changed during import"
        ) from exc
    if _admitted_httpfs_extension(extension) != httpfs:
        raise QuackDaemonGatewayError("admitted httpfs extension path differs")
    sealed_extensions = _seal_admitted_extensions(
        quack_path=extension,
        quack_sha256=expected_files[extension],
        httpfs_path=httpfs,
        httpfs_sha256=expected_files[httpfs],
    )
    return duckdb, sealed_extensions


def _submit_owner_mutation(
    *,
    mutation_dir: Path,
    sql: str,
    parameters: Sequence[Any],
    timeout_seconds: float = _OWNER_MUTATION_TIMEOUT_SECONDS,
) -> int:
    """Reject the historical bare-SQL owner inbox until signed fabric exists."""

    del mutation_dir, sql, parameters, timeout_seconds
    raise QuackDaemonGatewayError(
        "bare owner mutation CAS is disabled; the signed command fabric is required"
    )


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


def _task_slug(task_id: str) -> str:
    raw = str(task_id or "eaaef").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in raw)
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug or "eaaef")[:32]


def _safe_owned_paths(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    owned: list[str] = []
    for item in value:
        relative = str(item or "").strip().lstrip("/")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or Path(relative).is_absolute()
        ):
            continue
        owned.append(relative)
    return tuple(dict.fromkeys(owned))


def _focused_test_paths(body: Mapping[str, Any], *, owned: Sequence[str]) -> tuple[str, ...]:
    tests = body.get("test_requirements")
    focused: object = ()
    if isinstance(tests, Mapping):
        focused = tests.get("focused") or ()
    paths: list[str] = []
    if isinstance(focused, (list, tuple, str)):
        items = (focused,) if isinstance(focused, str) else focused
        for item in items:
            for token in str(item).split():
                if token.endswith(".py") and not token.startswith("-"):
                    relative = token.lstrip("/")
                    if ".." not in Path(relative).parts:
                        paths.append(relative)
    if not paths:
        paths = [
            relative
            for relative in owned
            if relative.endswith(".py")
            and (
                relative.startswith("test/")
                or relative.startswith("tests/")
                or "/test" in relative
            )
        ]
    return tuple(dict.fromkeys(paths))


def _load_task_spec(
    repo_root: Path,
    *,
    duckdb: Any,
    extensions: _SealedDuckDBExtensions,
    task_cid: str,
    task_id: str,
) -> dict[str, Any]:
    """Load owned files and focused tests from the live Quack task row."""

    body: Mapping[str, Any] = {}
    try:
        generation = "run-v14"
        vault = (
            Path(repo_root)
            / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
            / generation
            / "live/state/quack-owner"
        )
        token = _resolve_owner_token(_OWNER_HANDLE, vault_dir=vault)
        connection = _connect_admitted_duckdb(duckdb, extensions)
        try:
            connection.execute(
                "ATTACH 'quack:127.0.0.1:19495' AS control_plane (TYPE QUACK, TOKEN ?)",
                [token],
            )
            connection.execute("USE control_plane")
            row = connection.execute(
                "SELECT task_alias, body_json FROM tasks WHERE task_cid = ? LIMIT 1",
                [str(task_cid)],
            ).fetchone()
        finally:
            connection.close()
        if row is not None:
            task_id = str(row[0] or task_id)
            loaded = json.loads(str(row[1] or "{}"))
            if isinstance(loaded, Mapping):
                body = loaded
    except Exception:
        body = {}
    owned = _safe_owned_paths(body.get("owned_files") or body.get("write_scope") or ())
    if not owned:
        raise QuackDaemonGatewayError(
            f"{task_id} has no owned files in the control plane"
        )
    tests = _focused_test_paths(body, owned=owned)
    if not tests:
        raise QuackDaemonGatewayError(f"{task_id} has no focused tests")
    return {
        "task_id": str(task_id),
        "owned": owned,
        "tests": tests,
        "objective": str(body.get("objective") or body.get("title") or task_id),
    }


def _complete_owned_worktrees(
    worktrees: Path, *, task_id: str, owned: Sequence[str]
) -> tuple[Path, ...]:
    """Return existing worktrees that already have this task's owned files."""

    slug = _task_slug(task_id)
    found: list[Path] = []
    for candidate in sorted(worktrees.glob(f"{slug}-*")):
        if candidate.is_dir() and set(_owned_files(candidate, owned)) == set(owned):
            found.append(candidate)
    return tuple(found)


def _git_worktree(
    repo_root: Path,
    *,
    attempt_id: str,
    task_id: str,
    owned: Sequence[str],
) -> Path:
    worktrees = (
        repo_root
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v14/worktrees"
    )
    worktrees.mkdir(parents=True, exist_ok=True)
    dest = worktrees / f"{_task_slug(task_id)}-{attempt_id.replace(':', '_')[-16:]}"
    complete = _complete_owned_worktrees(worktrees, task_id=task_id, owned=owned)
    if dest in complete:
        _ensure_owned_write_dirs(dest, owned)
        return dest
    if complete:
        chosen = complete[0]
        _ensure_owned_write_dirs(chosen, owned)
        return chosen
    if dest.exists():
        _ensure_owned_write_dirs(dest, owned)
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
            "could not create an isolated EAAEF worktree: "
            + (completed.stderr or completed.stdout or "unknown")
        )
    _ensure_owned_write_dirs(dest, owned)
    return dest


def _ensure_owned_write_dirs(worktree: Path, owned: Sequence[str]) -> None:
    """Make only the owned output directories writable for the container user."""

    for relative in owned:
        directory = (worktree / relative).parent
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o777)
        except OSError:
            continue


def _owned_files(worktree: Path, owned: Sequence[str] | None = None) -> dict[str, Path]:
    relatives = tuple(owned) if owned is not None else _OWNED_RELATIVE_PATHS
    found: dict[str, Path] = {}
    for relative in relatives:
        path = worktree / relative
        if path.is_file() and path.stat().st_size > 0:
            found[relative] = path
    return found


def _owned_patch_cid(worktree: Path, owned: Sequence[str] | None = None) -> str:
    relatives = tuple(owned) if owned is not None else _OWNED_RELATIVE_PATHS
    files = _owned_files(worktree, relatives)
    if set(files) != set(relatives):
        raise QuackDaemonGatewayError("owned task files are incomplete")
    payload = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in sorted(files.items())
    }
    return _cid(payload)


def _focused_test_receipt_cid(
    worktree: Path, tests: Sequence[str] | None = None
) -> str:
    test_paths = tuple(tests) if tests is not None else (
        "test/api/test_external_agent_handoff_contracts.py",
    )
    cache_dir = Path(tempfile.gettempdir()) / "eaaef-pytest-cache"
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
            *test_paths,
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
    passed_count = 0
    for token in summary.replace(",", " ").split():
        if token.isdigit() and "passed" in summary:
            # Prefer the "<n> passed" pair.
            pass
    if " passed" in summary:
        parts = summary.split(" passed", 1)[0].split()
        if parts and parts[-1].isdigit():
            passed_count = int(parts[-1])
    if completed.returncode != 0 or passed_count < 1:
        raise QuackDaemonGatewayError(
            "focused task tests did not pass: " + summary[-500:]
        )
    return _cid(
        {
            "command": "python3 -m pytest -q " + " ".join(test_paths),
            "failed": 0,
            "passed": passed_count,
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


def _install_owned_on_host(
    worktree: Path, repo_root: Path, owned: Sequence[str]
) -> None:
    for relative in owned:
        source = worktree / relative
        destination = repo_root / relative
        if not source.is_file() or source.stat().st_size <= 0:
            raise QuackDaemonGatewayError(f"worker did not write {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _host_commit_owned(
    repo_root: Path, *, owned: Sequence[str], task_id: str
) -> str:
    added = subprocess.run(
        ["git", "-C", str(repo_root), "add", "--", *owned],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if added.returncode != 0:
        raise QuackDaemonGatewayError(
            "could not stage owned files: " + (added.stderr or added.stdout or "unknown")
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--", *owned],
        check=False,
        capture_output=True,
        timeout=20,
    )
    if dirty.returncode == 0:
        return _host_head_commit(repo_root)
    committed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=EAAEF Host Merge",
            "-c",
            "user.email=eaaef-host@localhost",
            "commit",
            "-m",
            f"Implement {task_id} owned files from the admitted worker.",
            "--",
            *owned,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if committed.returncode != 0:
        raise QuackDaemonGatewayError(
            "could not commit owned files: "
            + (committed.stderr or committed.stdout or "unknown")
        )
    return _host_head_commit(repo_root)


def _host_merge_admission(
    *,
    packet: ExternalAgentContainerWorkPacket,
    effect: Mapping[str, Any],
    repo_root: Path,
    owned: Sequence[str],
) -> dict[str, Any] | None:
    """Admit a verified patch only when this task's owned files exist."""

    worktree = _git_worktree(
        repo_root,
        attempt_id=packet.attempt_id,
        task_id=packet.task_id,
        owned=owned,
    )
    try:
        patch = _owned_patch_cid(worktree, owned)
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
        if _owned_patch_cid(repo_root, owned) == patch:
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


def _task_prompt(spec: Mapping[str, Any]) -> str:
    owned_lines = "\n".join(f"- {relative}" for relative in spec["owned"])
    return (
        f"Implement {spec['task_id']}: {spec['objective']}\n"
        "Write only:\n"
        f"{owned_lines}\n"
        "Do not push, do not mount Docker, do not change gitignored run-vN "
        "control-plane files. Keep tests deterministic."
    )


def _run_admitted_grok_container(
    *,
    packet: ExternalAgentContainerWorkPacket,
    repo_root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    owned = tuple(spec["owned"])
    tests = tuple(spec["tests"])
    worktree = _git_worktree(
        repo_root,
        attempt_id=packet.attempt_id,
        task_id=str(spec["task_id"]),
        owned=owned,
    )
    if set(_owned_files(worktree, owned)) == set(owned):
        patch = _owned_patch_cid(worktree, owned)
        test_cid = _focused_test_receipt_cid(worktree, tests)
        _install_owned_on_host(worktree, repo_root, owned)
        _host_commit_owned(repo_root, owned=owned, task_id=str(spec["task_id"]))
        return {
            "runtime_container_id": _cid(
                {"adopted_worktree": worktree.name, "patch_artifact_cid": patch}
            ),
            "patch_artifact_cid": patch,
            "test_receipt_cid": test_cid,
            "worktree": str(worktree),
        }
    _require_admitted_grok_image(packet.image_digest)
    grok_home_host = Path.home() / ".grok"
    auth = grok_home_host / "auth.json"
    if not auth.is_file():
        raise QuackDaemonGatewayError("host grok auth.json is absent")
    prompt = _task_prompt(spec)
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
        name = (
            _task_slug(str(spec["task_id"]))
            + "-"
            + hashlib.sha256(packet.attempt_id.encode()).hexdigest()[:12]
        )
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
            owned_ready = set(_owned_files(worktree, owned)) == set(owned)
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
        patch = _owned_patch_cid(worktree, owned)
        test_cid = _focused_test_receipt_cid(worktree, tests)
        _install_owned_on_host(worktree, repo_root, owned)
        _host_commit_owned(repo_root, owned=owned, task_id=str(spec["task_id"]))
        return {
            "runtime_container_id": runtime_id,
            "patch_artifact_cid": patch,
            "test_receipt_cid": test_cid,
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


class _Record(Mapping):
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

    def __getitem__(self, key: str) -> Any:
        if key in self._payload:
            value = self._payload[key]
            if key == "dependencies" and value is None:
                return ()
            return value
        if key in self._DEFAULTS:
            return self._DEFAULTS[key]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

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
        catalog: list[Mapping[str, Any]] = []
        cursor = 0
        while True:
            page = self._gateway._client.paginate(
                "list_tasks_page", cursor=cursor, limit=50
            )
            catalog.extend(page.items)
            if page.exhausted or page.next_cursor is None:
                break
            cursor = int(page.next_cursor)
        status_by_cid = {
            str(item.get("task_cid") or ""): str(item.get("status") or "").lower()
            for item in catalog
            if item.get("task_cid")
        }
        ranked = sorted(
            catalog,
            key=lambda item: (
                str(item.get("task_alias") or ""),
                str(item.get("task_cid") or ""),
            ),
        )
        ready: list[Mapping[str, Any]] = []
        for item in ranked:
            cid = str(item.get("task_cid") or "")
            if not cid or status_by_cid.get(cid, "") not in _READY_STATUSES:
                continue
            detailed = self.get(cid)
            payload = detailed.to_dict() if detailed is not None else dict(item)
            deps = _dependency_task_cids(payload)
            if any(
                status_by_cid.get(dep, "") not in _DONE_STATUSES for dep in deps
            ):
                continue
            ready.append(payload)
            if len(ready) >= wanted:
                break
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
            if not task_cid or task_cid in exclude:
                continue
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
        runtime_extensions: _SealedDuckDBExtensions,
        expected_source_head: str,
        expected_source_tree: str,
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
        self._runtime_extensions = runtime_extensions
        self._expected_source_head = expected_source_head
        self._expected_source_tree = expected_source_tree
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
        if not _host_admitted(
            self._repo_root,
            expected_source_head=self._expected_source_head,
            expected_source_tree=self._expected_source_tree,
        ):
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
            self._runtime_extensions.close()

    @property
    def attached(self) -> bool:
        return self._attached


def build_eaaef_host_admitted_command_gateway(
    *,
    repo_root: Path,
    program: DatabaseProgramConfig,
    owner_session_id: str,
    expected_source_head: str,
    expected_source_tree: str,
) -> EAAEFHostAdmittedCommandGateway | None:
    """Keep source-only gateway scaffolding unreachable at runtime."""

    if not _SOURCE_ONLY_SCAFFOLDING_RUNTIME_ENABLED:
        return None

    root = Path(repo_root)
    receipts = _eaaef_source_addressed_host_receipts(
        root,
        expected_source_head=expected_source_head,
        expected_source_tree=expected_source_tree,
    )
    if receipts is None or any(
        receipts.get(task_id, {}).get("decision") != "admitted"
        for task_id in _REQUIRED_RECEIPTS
    ):
        return None
    if not _command_fabric_live(receipts):
        return None
    if (
        program.authority_mode != "quack"
        or not str(program.quack_endpoint or "").startswith("quack:127.0.0.1:")
        or str(program.endpoint_secret_handle or "") != _OWNER_HANDLE
    ):
        return None
    duckdb, extensions = _import_admitted_duckdb(
        receipts.get("EAAEF-182", {})
    )
    bundle = receipts.get("EAAEF-191", {})
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
        connection = _connect_admitted_duckdb(duckdb, extensions)
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
        runtime_extensions=extensions,
        expected_source_head=expected_source_head,
        expected_source_tree=expected_source_tree,
    )


def build_eaaef_host_admitted_container_dispatcher_factory(
    *,
    repo_root: Path,
    expected_source_head: str,
    expected_source_tree: str,
) -> Any | None:
    """Keep source-only dispatcher scaffolding unreachable at runtime."""

    if not _SOURCE_ONLY_SCAFFOLDING_RUNTIME_ENABLED:
        return None

    root = Path(repo_root)
    receipts = _eaaef_source_addressed_host_receipts(
        root,
        expected_source_head=expected_source_head,
        expected_source_tree=expected_source_tree,
    )
    required = (
        "EAAEF-191",
        "EAAEF-189",
        "EAAEF-185",
        "EAAEF-186",
        "EAAEF-187",
        "EAAEF-188",
        "EAAEF-182",
    )
    if receipts is None or any(
        receipts.get(task_id, {}).get("decision") != "admitted"
        for task_id in required
    ):
        return None
    if not _command_fabric_live(receipts):
        return None
    duckdb, extensions = _import_admitted_duckdb(
        receipts.get("EAAEF-182", {})
    )

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
            spec = _load_task_spec(
                root,
                duckdb=duckdb,
                extensions=extensions,
                task_cid=packet.task_cid,
                task_id=packet.task_id,
            )
            worktree = _git_worktree(
                root,
                attempt_id=packet.attempt_id,
                task_id=str(spec["task_id"]),
                owned=spec["owned"],
            )
            if set(_owned_files(worktree, spec["owned"])) != set(spec["owned"]):
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
                spec = _load_task_spec(
                    root,
                    duckdb=duckdb,
                    extensions=extensions,
                    task_cid=packet.task_cid,
                    task_id=packet.task_id,
                )
                launched = _run_admitted_grok_container(
                    packet=packet, repo_root=root, spec=spec
                )
            except Exception as exc:
                sys.stderr.write(
                    f"eaaef launcher failed: {type(exc).__name__}: {exc}\n"
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
            spec = _load_task_spec(
                root,
                duckdb=duckdb,
                extensions=extensions,
                task_cid=packet.task_cid,
                task_id=packet.task_id,
            )
            worktree = _git_worktree(
                root,
                attempt_id=packet.attempt_id,
                task_id=str(spec["task_id"]),
                owned=spec["owned"],
            )
            patch = _owned_patch_cid(worktree, spec["owned"])
            tests = _focused_test_receipt_cid(worktree, spec["tests"])
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
            spec = _load_task_spec(
                root,
                duckdb=duckdb,
                extensions=extensions,
                task_cid=packet.task_cid,
                task_id=packet.task_id,
            )
            return _host_merge_admission(
                packet=packet,
                effect=effect,
                repo_root=root,
                owned=spec["owned"],
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

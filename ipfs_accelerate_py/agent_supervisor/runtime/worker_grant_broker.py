"""Private, bounded credential exchange for native task-command workers."""
from __future__ import annotations

import errno
import fcntl
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import threading

from ..task_sources import typed_state_owner as typed


def sealed_worker_bootstrap() -> int:
    """Create an opaque bootstrap readable only through an inherited memfd."""
    fd = os.memfd_create("native-worker-bootstrap", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.write(fd, secrets.token_hex(32).encode("ascii"))
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_sealed_worker_bootstrap(fd: int) -> str:
    info = os.fstat(fd)
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    if (fd < 3 or not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_size != 64 or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals != seals):
        raise ValueError("worker bootstrap descriptor is not sealed")
    secret = os.pread(fd, 65, 0).decode("ascii")
    if len(secret) != 64 or any(c not in "0123456789abcdef" for c in secret):
        raise ValueError("worker bootstrap descriptor is invalid")
    return secret


class TypedStateOwnerGrantBroker:
    """Authenticate a sealed bootstrap and the actual Unix-socket peer."""

    def __init__(self, *, socket_path, bootstrap_secret, store_id, resolve_credential):
        if (type(bootstrap_secret) is not str or len(bootstrap_secret) != 64
                or any(c not in "0123456789abcdef" for c in bootstrap_secret)
                or type(store_id) is not str or not store_id or len(store_id) > 4096
                or not callable(resolve_credential)):
            raise ValueError("grant broker configuration is invalid")
        self.socket_path = Path(socket_path)
        self._secret = bootstrap_secret
        self.store_id = store_id
        self._resolve = resolve_credential
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self._inode = None
        self._lock_fd = None

    @staticmethod
    def _error(message):
        from .quack_state_server import QuackStateServerControlError
        return QuackStateServerControlError(message)

    def start(self):
        if self._listener is not None:
            raise self._error("grant broker is already started")
        parent = self.socket_path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = parent.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise self._error("grant broker requires a private same-UID directory")
        lock_path = parent / (self.socket_path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise self._error("grant broker lock is unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise self._error("grant broker already serves a live listener") from exc
            if self.socket_path.exists() or self.socket_path.is_symlink():
                previous = self.socket_path.lstat()
                if not stat.S_ISSOCK(previous.st_mode) or previous.st_uid != os.geteuid():
                    raise self._error("grant broker path is not a same-UID socket")
                # ECONNREFUSED alone also describes an active socket between
                # bind() and listen(). Preserve every kernel-bound pathname.
                with Path("/proc/net/unix").open("rb") as handle:
                    inventory = handle.read(2 * 1024 * 1024 + 1)
                if len(inventory) > 2 * 1024 * 1024:
                    raise self._error("grant broker socket inventory exceeds its bound")
                names = {os.fsencode(str(self.socket_path)), os.fsencode(str(self.socket_path.resolve()))}
                if any(len(parts := line.split(None, 7)) == 8 and parts[7] in names
                       for line in inventory.splitlines()[1:]):
                    raise self._error("grant broker already serves a live listener or bound socket")
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(0.2)
                    try:
                        probe.connect(str(self.socket_path))
                    except OSError as exc:
                        if exc.errno != errno.ECONNREFUSED:
                            raise self._error("grant broker socket liveness is unknown") from exc
                    else:
                        raise self._error("grant broker already serves a live listener")
                current = self.socket_path.lstat()
                if (previous.st_dev, previous.st_ino) != (current.st_dev, current.st_ino):
                    raise self._error("grant broker socket changed during admission")
                self.socket_path.unlink()
            listener = socket.socket(socket.AF_UNIX)
            try:
                listener.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                listener.listen(16)
                listener.settimeout(0.2)
                info = self.socket_path.lstat()
                self._inode = (info.st_dev, info.st_ino)
            except BaseException:
                listener.close()
                raise
            self._lock_fd = fd
            self._listener = listener
            self._thread = threading.Thread(target=self._serve, daemon=True, name="native-worker-grants")
            self._thread.start()
        except BaseException:
            os.close(fd)
            raise

    def alive(self):
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def _serve(self):
        while not self._stop.is_set():
            try:
                channel, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with channel:
                channel.settimeout(1)
                kind = ""
                response = {"schema": typed.TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
                            "credential_kind": kind, "ok": False, "token": "", "error_code": "grant_denied"}
                try:
                    peer_pid, peer_uid, ticks = typed._kernel_peer_identity(channel)
                    data = bytearray()
                    while b"\n" not in data:
                        part = channel.recv(4096)
                        if not part:
                            raise ValueError("incomplete request")
                        data.extend(part)
                        if len(data) > typed.MAX_GRANT_BROKER_FRAME_BYTES:
                            raise ValueError("request too large")
                    request = json.loads(bytes(data).split(b"\n", 1)[0])
                    if not isinstance(request, dict):
                        raise ValueError("invalid request")
                    kind = request.get("credential_kind", "")
                    response["credential_kind"] = kind if type(kind) is str and len(kind) < 128 else ""
                    keys = {"schema", "credential_kind", "bootstrap_secret", "client_id", "process_birth_id", "store_id"}
                    if (set(request) != keys or any(type(request[k]) is not str for k in keys)
                            or request["schema"] != typed.TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA
                            or kind not in typed.TYPED_STATE_OWNER_GRANT_BROKER_CREDENTIAL_KINDS
                            or peer_uid != os.geteuid() or ticks <= 0
                            or request["store_id"] != self.store_id
                            or not 1 <= len(request["client_id"]) <= 256
                            or request["process_birth_id"] != typed.kernel_process_birth_id(peer_pid, start_time_ticks=ticks)
                            or not hmac.compare_digest(request["bootstrap_secret"], self._secret)):
                        raise ValueError("request denied")
                    token = self._resolve(kind, request["client_id"], request["process_birth_id"], peer_pid)
                    if (not isinstance(token, str) or not 8 <= len(token) <= 256
                            or any(not c.isalnum() and c not in "_-" for c in token)):
                        raise ValueError("credential unavailable")
                    response.update(ok=True, token=token, error_code="")
                except Exception:
                    # Error text and credential material never reach diagnostics.
                    pass
                try:
                    channel.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
                except OSError:
                    pass

    def stop(self):
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            info = self.socket_path.lstat()
            if (info.st_dev, info.st_ino) == self._inode:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        self._secret = ""

"""Capture and deliver explicit nomination bytes without provider access.

The host authenticates a source by its independent digest and retains its exact
bytes. The remote protocol publishes a separate, root-owned read-only copy.
Neither operation infers capsule optionality or authenticates producer facts;
the existing owner handoff performs the task/source checks after delivery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid


MAX_NOMINATION_BYTES = 256_000
NOMINATION_SCHEMA = "supervisor-semantic-decision-nomination@1"
NOMINATION_SCHEMAS = frozenset({NOMINATION_SCHEMA, "supervisor-semantic-decision-nomination@2"})
SNAPSHOT_SCHEMA = "terminal-semantic-decision-nomination-snapshot@1"
DELIVERY_SCHEMA = "terminal-semantic-decision-nomination-delivery@1"
_POLICY_FIELDS = frozenset({
    "required_capsule_ids", "optional_capsule_ids", "selected_optional_capsule_ids",
})
_FILE_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
                "st_mode", "st_nlink", "st_uid", "st_gid")
_PATH_FIELDS = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")


def host_nomination_path(source) -> Path:
    """Validate an absolute canonical lexical path without filesystem access."""
    if not isinstance(source, (str, Path)):
        raise ValueError("literal absolute nomination path required")
    value = str(source)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("UTF-8 nomination path required") from error
    path = Path(value)
    if (not 1 <= size <= 4096 or not path.is_absolute() or str(path) != value
            or value.startswith("//") or any(part in {".", ".."} for part in value.split("/"))
            or any(char in value for char in ("\x00", "\n", "\r"))):
        raise ValueError("bounded canonical absolute nomination path required")
    return path


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("independent lowercase SHA-256 nomination pin required")
    return value


def _byte_count(value):
    if type(value) is not int or not 1 <= value <= MAX_NOMINATION_BYTES:
        raise ValueError("positive bounded nomination byte count required")
    return value


def _validate_nomination(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate nomination JSON field")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("finite nomination JSON required")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        if (type(value) is not dict or set(value) != {"schema", "policy"}
                or type(value["schema"]) is not str or value["schema"] not in NOMINATION_SCHEMAS):
            raise ValueError("exact nomination schema and fields required")
        policy = value["policy"]
        text_profile = value["schema"] == "supervisor-semantic-decision-nomination@2"
        fields = _POLICY_FIELDS | {"task_text_source_path"} if text_profile else _POLICY_FIELDS
        if type(policy) is not dict or set(policy) != fields:
            raise ValueError("exact nomination policy fields required")
        if text_profile:
            source = policy["task_text_source_path"]
            if (type(source) is not str or not source.strip() or not 1 <= len(source.encode("utf-8")) <= 4096
                    or PurePosixPath(source).is_absolute() or str(PurePosixPath(source)) != source
                    or any(part in {"", ".", ".."} for part in source.split("/"))
                    or any(char in source for char in ("\x00", "\n", "\r"))):
                raise ValueError("canonical relative literal task text source path required")
        for name in _POLICY_FIELDS:
            items = policy[name]
            if type(items) is not list or len(items) > 1024:
                raise ValueError("bounded nomination capsule lists required")
            for item in items:
                if (type(item) is not str or not item.strip() or "\x00" in item
                        or len(item.encode("utf-8")) > 4096):
                    raise ValueError("explicit bounded nomination capsule identity required")
            if len(set(items)) != len(items):
                raise ValueError("duplicate nomination capsule identity")
        required = set(policy["required_capsule_ids"])
        optional = set(policy["optional_capsule_ids"])
        if required & optional or not set(policy["selected_optional_capsule_ids"]) <= optional:
            raise ValueError("disjoint nomination partition and selected optional subset required")
    except (UnicodeError, RecursionError) as error:
        raise ValueError("bounded UTF-8 nomination JSON required") from error


def _same(left, right, fields):
    return all(getattr(left, name) == getattr(right, name) for name in fields)


class _Directories:
    """Retain every nofollow directory descriptor until its paths are checked."""

    def __init__(self, path: Path):
        self.fds = []
        self.edges = []
        try:
            self.fds.append(os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC))
            for name in path.parts[1:]:
                parent = self.fds[-1]
                descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=parent)
                self.fds.append(descriptor)
                self.edges.append((parent, name, descriptor, os.fstat(descriptor)))
            self.recheck()
        except BaseException:
            self.close()
            raise

    @property
    def fd(self):
        return self.fds[-1]

    def recheck(self):
        for parent, name, descriptor, before in self.edges:
            opened = os.fstat(descriptor)
            visible = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not _same(before, opened, _PATH_FIELDS) or not _same(opened, visible, _PATH_FIELDS):
                raise ValueError("nomination directory path changed")

    def close(self):
        for descriptor in reversed(self.fds):
            os.close(descriptor)
        self.fds.clear()


def _read_at(parent: int, name: str, expected_sha256: str, *, immutable=False):
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_NOMINATION_BYTES:
            raise ValueError("bounded regular nomination file required")
        if immutable and (before.st_uid != os.geteuid() or before.st_nlink != 1
                          or stat.S_IMODE(before.st_mode) != 0o444):
            raise ValueError("owner-controlled immutable single-link nomination snapshot required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_NOMINATION_BYTES + 1)
        after = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (not _same(before, after, _FILE_FIELDS) or not _same(after, visible, _FILE_FIELDS)
                or len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("nomination bytes differ from pin or changed during read")
        _validate_nomination(raw)
        return raw
    finally:
        os.close(descriptor)


def read_nomination_source(*, source, expected_sha256) -> bytes:
    """Read a bounded non-symlink source and authenticate exact raw bytes."""
    path = host_nomination_path(source)
    digest = _digest(expected_sha256)
    directories = _Directories(path.parent)
    try:
        raw = _read_at(directories.fd, path.name, digest)
        directories.recheck()
        return raw
    finally:
        directories.close()


def _snapshot_artifact(directory: Path, digest: str) -> Path:
    return host_nomination_path(directory / ("semantic-decision-nomination-" + digest + ".json"))


def capture_nomination_snapshot(*, source, expected_sha256, directory: Path) -> dict:
    """Publish a new immutable snapshot; existing artifacts are never reused."""
    digest = _digest(expected_sha256)
    raw = read_nomination_source(source=source, expected_sha256=digest)
    directory = host_nomination_path(directory)
    artifact = _snapshot_artifact(directory, digest)
    directories = _Directories(directory)
    temporary = ".nomination-pending-" + uuid.uuid4().hex
    pending = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directories.fd)
        pending = True
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, artifact.name, src_dir_fd=directories.fd,
                dst_dir_fd=directories.fd, follow_symlinks=False)
        os.unlink(temporary, dir_fd=directories.fd)
        pending = False
        os.fsync(directories.fd)
        observed = _read_at(directories.fd, artifact.name, digest, immutable=True)
        directories.recheck()
        if observed != raw:
            raise ValueError("nomination snapshot differs from captured source")
    finally:
        try:
            if pending:
                os.unlink(temporary, dir_fd=directories.fd)
        finally:
            directories.close()
    return {"schema": SNAPSHOT_SCHEMA, "artifact": str(artifact), "sha256": digest, "bytes": len(raw)}


def verify_nomination_snapshot(snapshot, *, directory: Path) -> bytes:
    """Reauthenticate a snapshot descriptor, its path, immutable mode and bytes."""
    if (type(snapshot) is not dict or set(snapshot) != {"schema", "artifact", "sha256", "bytes"}
            or snapshot.get("schema") != SNAPSHOT_SCHEMA or type(snapshot.get("artifact")) is not str):
        raise ValueError("exact nomination snapshot descriptor required")
    digest = _digest(snapshot["sha256"])
    expected_bytes = _byte_count(snapshot["bytes"])
    directory = host_nomination_path(directory)
    artifact = host_nomination_path(snapshot["artifact"])
    if artifact != _snapshot_artifact(directory, digest):
        raise ValueError("nomination snapshot must belong to the selected trial directory")
    directories = _Directories(directory)
    try:
        raw = _read_at(directories.fd, artifact.name, digest, immutable=True)
        directories.recheck()
        if len(raw) != expected_bytes:
            raise ValueError("nomination snapshot byte count differs")
        return raw
    finally:
        directories.close()


def nomination_delivery_arguments(*, root: str, run_id: str, expected_sha256: str,
                                  expected_bytes: int) -> dict:
    """Derive literal per-run remote paths without reading a filesystem."""
    root_path = host_nomination_path(root)
    if type(run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("independent lowercase 32-hex nomination delivery run id required")
    digest = _digest(expected_sha256)
    _byte_count(expected_bytes)
    directory = host_nomination_path(root_path / "decision-nominations" / run_id)
    temporary = host_nomination_path(directory / ".upload.json")
    artifact = host_nomination_path(directory / (digest + ".json"))
    return {"directory": str(directory), "temporary": str(temporary), "artifact": str(artifact)}


def nomination_delivery_command(*, python: str, operation: str, root: str, run_id: str,
                                expected_sha256: str, expected_bytes: int) -> list[str]:
    """Return the literal root-executed prepare/publish command."""
    if (type(python) is not str or not python or len(python.encode("utf-8")) > 4096
            or any(char in python for char in ("\x00", "\n", "\r"))):
        raise ValueError("bounded literal Python executable required")
    if type(operation) is not str or operation not in {"prepare", "publish"}:
        raise ValueError("closed nomination delivery operation required")
    nomination_delivery_arguments(root=root, run_id=run_id, expected_sha256=expected_sha256,
                                  expected_bytes=expected_bytes)
    return [python, "-P", "-c", NOMINATION_DELIVERY_CODE, operation, str(root), run_id,
            expected_sha256, str(expected_bytes)]


# This program deliberately imports only the standard library. It is supplied
# as a literal argv element, not a shell fragment or a deployed module import.
NOMINATION_DELIVERY_CODE = r'''
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid

OWNER_UID = 0
MAX_NOMINATION_BYTES = 256000
NOMINATION_SCHEMA = "supervisor-semantic-decision-nomination@1"
NOMINATION_SCHEMAS = {NOMINATION_SCHEMA, "supervisor-semantic-decision-nomination@2"}
DELIVERY_SCHEMA = "terminal-semantic-decision-nomination-delivery@1"
POLICY_FIELDS = {"required_capsule_ids", "optional_capsule_ids", "selected_optional_capsule_ids"}
FILE_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink", "st_uid", "st_gid")
PATH_FIELDS = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")

def require(value, reason):
    if not value:
        raise ValueError(reason)

def same(left, right, fields):
    return all(getattr(left, name) == getattr(right, name) for name in fields)

def path(value):
    require(type(value) is str and 1 <= len(value.encode("utf-8")) <= 4096,
            "bounded literal remote nomination path required")
    selected = Path(value)
    require(selected.is_absolute() and str(selected) == value and not value.startswith("//")
            and not any(part in {".", ".."} for part in value.split("/"))
            and not any(char in value for char in ("\x00", "\n", "\r")),
            "canonical absolute remote nomination path required")
    return selected

def validate(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate nomination JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("finite nomination JSON required")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        require(type(value) is dict and set(value) == {"schema", "policy"}
                and type(value["schema"]) is str and value["schema"] in NOMINATION_SCHEMAS,
                "exact nomination schema required")
        policy = value["policy"]
        text_profile = value["schema"] == "supervisor-semantic-decision-nomination@2"
        fields = POLICY_FIELDS | {"task_text_source_path"} if text_profile else POLICY_FIELDS
        require(type(policy) is dict and set(policy) == fields, "exact nomination policy fields required")
        if text_profile:
            source = policy["task_text_source_path"]
            require(type(source) is str and bool(source.strip()) and 1 <= len(source.encode("utf-8")) <= 4096
                    and not PurePosixPath(source).is_absolute() and str(PurePosixPath(source)) == source
                    and not any(part in {"", ".", ".."} for part in source.split("/"))
                    and not any(char in source for char in ("\x00", "\n", "\r")),
                    "canonical relative literal task text source path required")
        for name in POLICY_FIELDS:
            items = policy[name]
            require(type(items) is list and len(items) <= 1024, "bounded nomination capsule lists required")
            for item in items:
                require(type(item) is str and bool(item.strip()) and "\x00" not in item
                        and len(item.encode("utf-8")) <= 4096, "bounded explicit capsule identity required")
            require(len(set(items)) == len(items), "duplicate nomination capsule identity")
        optional = set(policy["optional_capsule_ids"])
        require(not set(policy["required_capsule_ids"]) & optional
                and set(policy["selected_optional_capsule_ids"]) <= optional,
                "disjoint nomination partition and optional subset required")
    except (UnicodeError, RecursionError) as error:
        raise ValueError("bounded UTF-8 nomination JSON required") from error

def owned_directory(descriptor, exact_mode=None):
    info = os.fstat(descriptor)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == OWNER_UID
            and not stat.S_IMODE(info.st_mode) & 0o022
            and (exact_mode is None or stat.S_IMODE(info.st_mode) == exact_mode),
            "protected owner-controlled nomination directory required")

fds = []
edges = []
def open_directory_chain(selected):
    fds.append(os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC))
    for name in selected.parts[1:]:
        parent = fds[-1]
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        fds.append(descriptor)
        edges.append((parent, name, descriptor, os.fstat(descriptor)))
    owned_directory(fds[-1])
    return fds[-1]

def child_directory(parent, name, *, exclusive=False):
    created = False
    try:
        os.mkdir(name, 0o755, dir_fd=parent)
        created = True
    except FileExistsError:
        if exclusive:
            raise ValueError("nomination delivery run directory already exists")
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    fds.append(descriptor)
    owned_directory(descriptor)
    if created:
        os.fchmod(descriptor, 0o755)
        os.fsync(descriptor)
        os.fsync(parent)
    owned_directory(descriptor, exact_mode=0o755)
    edges.append((parent, name, descriptor, os.fstat(descriptor)))
    return descriptor

def recheck():
    for parent, name, descriptor, before in edges:
        opened = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(same(before, opened, PATH_FIELDS) and same(opened, visible, PATH_FIELDS),
                "nomination directory path changed")

def read_file(parent, name, digest, size, *, immutable=False, normalize_upload=False):
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if normalize_upload:
            require(not immutable and name == ".upload.json",
                    "custody normalization is limited to the nominated upload")
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                    and before.st_size == size and 1 <= size <= MAX_NOMINATION_BYTES,
                    "exact bounded single-link regular nomination upload required")
            visible = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(same(before, visible, FILE_FIELDS), "nomination upload path changed before custody")
            # Harbor Compose may retain host UID/GID on the uploaded inode.
            # Normalize only the checked fd in this protected per-run directory;
            # never chown a path which could have been replaced after opening.
            os.fchown(descriptor, OWNER_UID, OWNER_UID)
            normalized = os.fstat(descriptor)
            require(same(before, normalized, ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_nlink"))
                    and normalized.st_uid == OWNER_UID and normalized.st_gid == OWNER_UID,
                    "nomination upload changed during custody normalization")
            before = normalized
        require(stat.S_ISREG(before.st_mode) and before.st_uid == OWNER_UID and before.st_nlink == 1
                and before.st_size == size and 1 <= size <= MAX_NOMINATION_BYTES
                and (not immutable or stat.S_IMODE(before.st_mode) == 0o444),
                "exact bounded owner-controlled single-link nomination file required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_NOMINATION_BYTES + 1)
        after = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(same(before, after, FILE_FIELDS) and same(after, visible, FILE_FIELDS)
                and len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
                "nomination differs from independent pin or changed during read")
        validate(raw)
        return raw, after
    finally:
        os.close(descriptor)

def main():
    require(os.geteuid() == OWNER_UID, "root nomination delivery authority required")
    require(len(sys.argv) == 6, "exact nomination delivery arguments required")
    operation, root_value, run_id, digest, size_value = sys.argv[1:]
    require(operation in {"prepare", "publish"}, "closed nomination delivery operation required")
    root = path(root_value)
    require(re.fullmatch(r"[0-9a-f]{32}", run_id) is not None, "32-hex independent delivery run id required")
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "independent SHA-256 nomination pin required")
    require(re.fullmatch(r"[1-9][0-9]{0,5}", size_value) is not None, "bounded nomination byte count required")
    size = int(size_value)
    require(1 <= size <= MAX_NOMINATION_BYTES, "bounded nomination byte count required")
    directory = path(str(root / "decision-nominations" / run_id))
    artifact = path(str(directory / (digest + ".json")))
    root_fd = open_directory_chain(root)
    if operation == "prepare":
        parent_fd = child_directory(root_fd, "decision-nominations")
        child_directory(parent_fd, run_id, exclusive=True)
        recheck()
    else:
        parent_fd = os.open("decision-nominations", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
        fds.append(parent_fd)
        owned_directory(parent_fd, exact_mode=0o755)
        edges.append((root_fd, "decision-nominations", parent_fd, os.fstat(parent_fd)))
        directory_fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        fds.append(directory_fd)
        owned_directory(directory_fd, exact_mode=0o755)
        edges.append((parent_fd, run_id, directory_fd, os.fstat(directory_fd)))
        raw, upload_info = read_file(directory_fd, ".upload.json", digest, size, normalize_upload=True)
        temporary = ".nomination-pending-" + uuid.uuid4().hex
        pending = False
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                 0o600, dir_fd=directory_fd)
            pending = True
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.link(temporary, artifact.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                    follow_symlinks=False)
            os.unlink(temporary, dir_fd=directory_fd)
            pending = False
            require(same(upload_info, os.stat(".upload.json", dir_fd=directory_fd, follow_symlinks=False), FILE_FIELDS),
                    "uploaded nomination changed before removal")
            os.unlink(".upload.json", dir_fd=directory_fd)
            os.fsync(directory_fd)
            observed, _ = read_file(directory_fd, artifact.name, digest, size, immutable=True)
            require(observed == raw, "published nomination differs from upload")
            recheck()
        finally:
            if pending:
                os.unlink(temporary, dir_fd=directory_fd)
    receipt = {"schema": DELIVERY_SCHEMA, "artifact": str(artifact), "sha256": digest, "bytes": size,
               "operation": operation, "published": operation == "publish",
               "provider_called": False, "completion_authority": False}
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)

try:
    main()
finally:
    for descriptor in reversed(fds):
        os.close(descriptor)
'''

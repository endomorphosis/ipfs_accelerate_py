"""Disposable native crash copy; no launch/admission/remote-effect authority.

The parent must already own the exact finite child argv. This helper creates
only a new private copy of an already closed genuine execution seed. It reuses
the published fixture's pinned installed DuckDB loader, adds the supplied raw
UNKNOWN row to its actual matching native attempt, commits/fsyncs a real WAL,
and exits without closing that connection. No original database is opened.
The resulting process witness is not a native launcher or custody receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import selectors
import stat
import subprocess
import sys
import time
import uuid

BASENAME = "crash-copy.execution.shard-0001-of-0003.duckdb"
MARKER = "crash-copy-input.json"
PROVIDER = "crash-copy-provider.json"
RESULT = "crash-copy-committed.json"
INSTALLATION = "duckdb-installation.json"
MAX_DATABASE = 16 * 1024 * 1024
MAX_SOURCE = 8 * 1024 * 1024
MAX_STREAM = 65536
CHILD_SECONDS = 20.0
ROW_FIELDS = {"invocation_id", "attempt_id", "task_cid", "idempotency_key",
              "owner_session_id", "recorded_at_ms", "result_json"}
CALLBACK_FIELDS = {"schema", "callback_state", "provider_effect_state", "task_cid",
                   "attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch"}


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate fixture JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite fixture JSON")))


def _regular(path, maximum, *, nonempty=True):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("fixture input path alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.geteuid())
                or before.st_nlink != 1 or before.st_mode & 0o002
                or not (int(nonempty) <= before.st_size <= maximum)):
            raise ValueError("fixture input regular-file custody/bound")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            if count > maximum:
                raise ValueError("fixture input byte bound")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda item: tuple(getattr(item, field) for field in fields)
        if count != before.st_size or not witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()):
            raise ValueError("fixture input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash(path, maximum=MAX_SOURCE):
    return hashlib.sha256(_regular(path, maximum)).hexdigest()


def _identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def _write(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("fixture short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _published(path, expected):
    raw = _regular(path, MAX_SOURCE)
    if path.name != "native_execution_crash_wal_fixture.py" or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("published fixture source pin differs")
    spec = importlib.util.spec_from_file_location("pinned_disposable_native_crash_fixture", path)
    module = importlib.util.module_from_spec(spec)
    # Compile these exact verified bytes; -B alone does not forbid reading an
    # existing timestamp-based .pyc beside the published source.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if _hash(path) != expected:
        raise ValueError("published fixture source changed during load")
    return module


def _validate_row(row):
    if (type(row) is not dict or set(row) != ROW_FIELDS
            or not all(type(row[key]) is str and row[key] for key in ROW_FIELDS - {"recorded_at_ms"})
            or type(row["recorded_at_ms"]) is not int or row["recorded_at_ms"] < 0
            or row["idempotency_key"] != "provider:" + row["attempt_id"]):
        raise ValueError("fixture raw UNKNOWN row is not closed")
    callback = _decode(row["result_json"])
    if (type(callback) is not dict or set(callback) != CALLBACK_FIELDS
            or callback["schema"] != "database-portal-callback-intent@1"
            or callback["callback_state"] != "started_outcome_unknown"
            or callback["provider_effect_state"] != "unknown_may_have_started"
            or callback["task_cid"] != row["task_cid"] or callback["attempt_id"] != row["attempt_id"]
            or any(type(callback[key]) is not int or callback[key] < 1 for key in ("fencing_token", "fence_epoch"))
            or any(type(callback[key]) is not str or not callback[key] for key in ("claim_id", "lease_id"))):
        raise ValueError("fixture UNKNOWN callback identity differs")
    if len(_json(row)) > 32768 or len(row["result_json"].encode()) > 16384:
        raise ValueError("fixture UNKNOWN scalar exceeds native observer bound")
    return callback


def fixture_argv(root, marker_sha):
    return (sys.executable, "-I", "-B", str(Path(__file__).resolve(strict=True)),
            "--produce", "--fixture-root", str(root), "--marker-sha256", marker_sha)


def create_crash_copy(root: Path, execution_seed: Path, provider_row: dict, producer_source: Path):
    """Create one bounded native WAL child under the caller's process group.

    This API must be invoked only under the parent's already admitted finite
    source-unit owner. Its result supplies no admission or original-store grant.
    The seed must be closed and WAL-free; its bytes are rechecked after exit.
    """
    root, execution_seed, producer_source = map(Path, (root, execution_seed, producer_source))
    if (not root.is_absolute() or root.parent.resolve(strict=True) != root.parent
            or root.exists() or root.is_symlink() or root.name in (".", "..")
            or producer_source.name != "implementation_daemon.py"):
        raise ValueError("new private fixture root/source scope differs")
    _validate_row(provider_row)
    if Path(str(execution_seed) + ".wal").exists():
        raise ValueError("execution seed is not closed/WAL-free")
    seed_raw = _regular(execution_seed, MAX_DATABASE)
    seed_hash = hashlib.sha256(seed_raw).hexdigest()
    producer_sha = _hash(producer_source)
    published_path = producer_source.parents[3] / "test/api/native_execution_crash_wal_fixture.py"
    published_sha = _hash(published_path)
    published = _published(published_path, published_sha)
    installation = published.installed_duckdb_binding()
    helper_path = Path(__file__).resolve(strict=True)
    helper_sha = _hash(helper_path)
    root.mkdir(mode=0o700)
    _write(root / BASENAME, seed_raw)
    _write(root / PROVIDER, _json(provider_row))
    _write(root / INSTALLATION, _json(installation))
    marker = dict(schema="disposable-native-three-store-crash-copy-input@1",
        fixture_only=True, original_store_authority=False, nonce=uuid.uuid4().hex,
        uid=os.geteuid(), parent_pid=os.getpid(), parent_start_time_ticks=_ticks(os.getpid()),
        owner_process_group=os.getpgrp(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        seed_path=str(execution_seed), seed_sha256=seed_hash,
        copied_seed_sha256=_hash(root / BASENAME, MAX_DATABASE),
        provider_sha256=_hash(root / PROVIDER), installation_sha256=_hash(root / INSTALLATION),
        producer_source=str(producer_source), producer_source_sha256=producer_sha,
        published_fixture_source=str(published_path), published_fixture_sha256=published_sha,
        helper_source=str(helper_path), helper_sha256=helper_sha)
    _write(root / MARKER, _json(marker))
    marker_sha = _hash(root / MARKER)
    argv = fixture_argv(root, marker_sha)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_DATASETS_AUTO_INSTALL="0", IPFS_AUTO_INSTALL="0",
        TMPDIR=str(root), TMP=str(root), TEMP=str(root))
    process = subprocess.Popen(argv, cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, start_new_session=False)
    streams, started, reaped = [bytearray(), bytearray()], time.monotonic(), False
    observed_ticks = None
    try:
        observed_ticks = _ticks(process.pid)
        if os.getpgid(process.pid) != os.getpgrp():
            raise AssertionError("fixture child escaped owning process group")
        with selectors.DefaultSelector() as selector:
            for stream, destination in zip((process.stdout, process.stderr), streams):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, destination)
            while selector.get_map():
                remaining = CHILD_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("disposable crash-copy child deadline")
                for key, _events in selector.select(min(0.2, remaining)):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    key.data.extend(block)
                    if len(key.data) > MAX_STREAM:
                        raise AssertionError("disposable crash-copy output bound")
        code = process.wait(timeout=max(0.01, CHILD_SECONDS - (time.monotonic() - started)))
        reaped = True
        if code != 0:
            raise AssertionError("disposable crash-copy producer failed; retained private diagnostics")
        result = _decode(_regular(root / RESULT, 32768))
        if (result.get("fixture_only") is not True or result.get("original_store_authority") is not False
                or result.get("committed_before_os_exit") is not True
                or result.get("child_pid") != process.pid or result.get("child_start_time_ticks") != observed_ticks
                or result.get("boot_id") != marker["boot_id"] or result.get("owner_process_group") != os.getpgrp()
                or result.get("marker_sha256") != marker_sha or result.get("provider_row") != provider_row):
            raise AssertionError("disposable crash-copy result identity differs")
        path, wal = root / BASENAME, Path(str(root / BASENAME) + ".wal")
        _regular(path, MAX_DATABASE)
        _regular(wal, MAX_DATABASE)
        if _hash(execution_seed, MAX_DATABASE) != seed_hash or Path(str(execution_seed) + ".wal").exists():
            raise AssertionError("original closed execution seed changed")
        if (_hash(root / MARKER) != marker_sha or _hash(root / PROVIDER) != marker["provider_sha256"]
                or _hash(root / INSTALLATION) != marker["installation_sha256"]
                or _hash(producer_source) != producer_sha or _hash(published_path) != published_sha
                or _hash(helper_path) != helper_sha):
            raise AssertionError("fixture immutable inputs/source pins changed")
        return dict(root=root, path=path, wal=wal, result=result, child_pid=process.pid,
            child_start_time_ticks=observed_ticks, boot_id=marker["boot_id"], child_reaped=True,
            owner_process_group=os.getpgrp(), argv=argv,
            writer=path.with_name("." + path.name + ".writer.lock"),
            policy=path.with_name("." + path.name + ".lock"),
            source_pins={str(producer_source): producer_sha, str(published_path): published_sha,
                         str(helper_path): helper_sha}, installation=installation,
            installation_pin=dict(path=str(root / INSTALLATION), sha256=marker["installation_sha256"]),
            seed_sha256=seed_hash, stdout_bytes=len(streams[0]), stderr_bytes=len(streams[1]))
    except BaseException as error:
        _write(root / "crash-copy-failure.json", _json(dict(
            schema="disposable-native-three-store-crash-copy-failure@1", fixture_only=True,
            original_store_authority=False, error_type=type(error).__name__,
            child_pid=process.pid, child_returncode=process.poll(), child_reaped_at_capture=reaped,
            stdout_text=bytes(streams[0]).decode("utf-8", "replace"),
            stderr_text=bytes(streams[1]).decode("utf-8", "replace"))))
        raise
    finally:
        if not reaped:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def produce(args):
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DATABASE, MAX_DATABASE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    root = args.fixture_root
    if (not root.is_absolute() or root.resolve(strict=True) != root or root.stat().st_uid != os.geteuid()
            or stat.S_IMODE(root.stat().st_mode) != 0o700
            or set(path.name for path in root.iterdir()) != {MARKER, PROVIDER, INSTALLATION, BASENAME}
            or _hash(root / MARKER) != args.marker_sha256):
        raise ValueError("crash-copy fixture directory/marker identity differs")
    marker = _decode(_regular(root / MARKER, 32768))
    if (marker.get("schema") != "disposable-native-three-store-crash-copy-input@1"
            or marker.get("fixture_only") is not True or marker.get("original_store_authority") is not False
            or marker.get("uid") != os.geteuid() or marker.get("parent_pid") != os.getppid()
            or marker.get("parent_start_time_ticks") != _ticks(os.getppid())
            or marker.get("owner_process_group") != os.getpgrp()
            or marker.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            or marker.get("helper_source") != str(Path(__file__).resolve(strict=True))
            or marker.get("helper_sha256") != _hash(Path(__file__).resolve(strict=True))
            or marker.get("copied_seed_sha256") != marker.get("seed_sha256")
            or marker.get("copied_seed_sha256") != _hash(root / BASENAME, MAX_DATABASE)
            or marker.get("provider_sha256") != _hash(root / PROVIDER)
            or marker.get("installation_sha256") != _hash(root / INSTALLATION)):
        raise ValueError("crash-copy fixed input/source/owner crosswalk differs")
    producer = Path(marker["producer_source"])
    if producer.name != "implementation_daemon.py" or _hash(producer) != marker["producer_source_sha256"]:
        raise ValueError("actual execution producer source changed")
    published_path = producer.parents[3] / "test/api/native_execution_crash_wal_fixture.py"
    if str(published_path) != marker["published_fixture_source"]:
        raise ValueError("published installed-backend loader source root differs")
    published = _published(published_path, marker["published_fixture_sha256"])
    row = _decode(_regular(root / PROVIDER, 32768))
    callback = _validate_row(row)
    import fcntl
    locks = []
    for suffix in (".writer.lock", ".lock"):
        path = root / ("." + BASENAME + suffix)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(descriptor)
    duckdb, installed = published._load_exact_installed_duckdb(root, marker["installation_sha256"])
    connection = duckdb.connect(str(root / BASENAME), config={"threads": "1", "memory_limit": "256MB",
        "autoinstall_known_extensions": "false", "autoload_known_extensions": "false",
        "enable_external_access": "false", "allow_unsigned_extensions": "false", "lock_configuration": "true"})
    columns = ("attempt_id", "claim_id", "task_cid", "task_alias", "attempt_number", "owner_session_id",
        "fencing_token", "fence_epoch", "lease_id", "committed_phase", "status", "started_at_ms", "finished_at_ms", "revision", "body_json")
    selected = connection.execute("SELECT " + ",".join(columns) +
        " FROM database_task_attempts WHERE attempt_id=? LIMIT 2", [row["attempt_id"]]).fetchall()
    if len(selected) != 1:
        raise ValueError("genuine seed attempt is absent/ambiguous")
    attempt = dict(zip(columns, selected[0]))
    if (attempt["status"] != "running" or attempt["committed_phase"] != "context"
            or attempt["revision"] != 2 or attempt["finished_at_ms"] is not None
            or attempt["owner_session_id"] != row["owner_session_id"]
            or any(type(attempt[key]) is not type(callback[key]) or attempt[key] != callback[key]
                   for key in ("task_cid", "attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch"))):
        raise ValueError("genuine seed native attempt/UNKNOWN identity differs")
    if connection.execute("SELECT COUNT(*) FROM provider_invocations WHERE attempt_id=?", [row["attempt_id"]]).fetchone()[0] != 0:
        raise ValueError("seed already has a provider row")
    if connection.execute("SELECT COUNT(*) FROM effect_claims WHERE attempt_id=?", [row["attempt_id"]]).fetchone()[0] != 0:
        raise ValueError("seed already has effects")
    phases = connection.execute("SELECT phase,revision FROM attempt_phases WHERE attempt_id=? ORDER BY revision LIMIT 3",
                                [row["attempt_id"]]).fetchall()
    if phases != [("claimed", 1), ("context", 2)]:
        raise ValueError("seed native phases differ")
    baseline_metadata = dict(connection.execute("SELECT key,value FROM daemon_execution_metadata").fetchall())
    if ("process_instance_id" not in baseline_metadata
            or baseline_metadata.get("execution_state_prefix") != "crash-copy"
            or baseline_metadata.get("task_shard_count") != "3"
            or baseline_metadata.get("task_shard_index") != "1"
            or baseline_metadata.get("strict_task_sharding") != "true"
            or baseline_metadata.get("logical_owner_session_id") != attempt["owner_session_id"]
            or Path(baseline_metadata.get("execution_state_dir", "")) / BASENAME != Path(marker["seed_path"])):
        raise ValueError("genuine seed strict native lane metadata differs")
    connection.execute("CHECKPOINT")
    connection.execute("BEGIN TRANSACTION")
    child_pid, child_ticks = os.getpid(), _ticks(os.getpid())
    process_instance = f"process:disposable-crash-copy:{child_pid}:{child_ticks}:{marker['boot_id']}"
    changed = connection.execute("UPDATE daemon_execution_metadata SET value=? WHERE key='process_instance_id' RETURNING key",
                                 [process_instance]).fetchall()
    if changed != [("process_instance_id",)]:
        raise ValueError("native process metadata update differs")
    # This is an explicit disposable COPY relocation. The original seed stays
    # byte-identical; copied native rows keep their real logical identities.
    # The published reader requires truthful directory metadata for this new
    # copied lane, so only that physical path and actual process ID are updated.
    relocated = connection.execute("UPDATE daemon_execution_metadata SET value=? WHERE key='execution_state_dir' RETURNING key",
                                   [str(root)]).fetchall()
    if relocated != [("execution_state_dir",)]:
        raise ValueError("native copied-lane relocation metadata differs")
    keys = ("invocation_id", "attempt_id", "task_cid", "idempotency_key", "owner_session_id", "recorded_at_ms", "result_json")
    connection.execute("INSERT INTO provider_invocations(" + ",".join(keys) + ") VALUES (?,?,?,?,?,?,?)",
                       [row[key] for key in keys])
    connection.execute("COMMIT")
    metadata = dict(connection.execute("SELECT key,value FROM daemon_execution_metadata").fetchall())
    if metadata != {**baseline_metadata, "process_instance_id": process_instance, "execution_state_dir": str(root)}:
        raise ValueError("unrelated native metadata changed")
    recorded = connection.execute("SELECT " + ",".join(keys) + " FROM provider_invocations WHERE attempt_id=? LIMIT 2",
                                  [row["attempt_id"]]).fetchall()
    if len(recorded) != 1 or dict(zip(keys, recorded[0])) != row:
        raise ValueError("raw UNKNOWN insertion readback differs")
    wal = Path(str(root / BASENAME) + ".wal")
    if not wal.is_file() or not 0 < wal.stat().st_size <= MAX_DATABASE:
        raise ValueError("real committed crash-copy WAL missing")
    descriptor = os.open(wal, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _write(root / RESULT, _json(dict(schema="disposable-native-three-store-crash-copy-result@1",
        fixture_only=True, original_store_authority=False, committed_before_os_exit=True,
        child_pid=child_pid, child_start_time_ticks=child_ticks, boot_id=marker["boot_id"],
        owner_process_group=os.getpgrp(), marker_sha256=args.marker_sha256,
        provider_row=row, metadata=metadata, copied_lane_relocated=True,
        fixture_metadata_changes=["execution_state_dir", "process_instance_id"],
        attempt_identity={key: attempt[key] for key in ("task_cid", "attempt_id", "attempt_number",
            "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch")},
        wal_bytes=wal.stat().st_size,
        duckdb_package_origin=installed["package_origin"], duckdb_native_origin=installed["native_origin"],
        duckdb_version=installed["version"])))
    # Actual committed native state remains in WAL; all real descriptors/locks
    # close only when this exact child exits. No native receipt is manufactured.
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--produce", action="store_true", required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--marker-sha256", required=True)
    args = parser.parse_args()
    try:
        produce(args)
    except Exception as error:
        sys.stderr.write(json.dumps(dict(fixture_failed=True, error_type=type(error).__name__,
            error_message=str(error).encode("utf-8", "replace")[:4096].decode("utf-8", "ignore"))) + "\n")
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()

"""Pinned inert dual-stream payload: no auth/model/network/provider imports."""
import hashlib
import json
import os
import sys
from pathlib import Path


def write_all(fd, data):
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        if count <= 0:
            raise RuntimeError("nonmodel short pipe write")
        offset += count


def write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"within_cap", "over_cap"}:
        raise SystemExit(64)
    workspace = Path("/workspace")
    write_new(workspace / "fixture-started", b"nonmodel-callable-dual-pipe\n")
    # A naive reader blocked on quiet stderr cannot let this stdout phase
    # complete: 1MiB exceeds a pipe buffer. Then alternate both actual streams.
    for _ in range(256):
        write_all(1, b"O" * 4096)
    for _ in range(32):
        write_all(2, b"E" * 1024)
    for _ in range(32):
        write_all(1, b"A" * 4096)
        write_all(2, b"S" * 1024)
    stdout = b"O" * (1024 * 1024) + b"A" * (32 * 4096)
    stderr = b"E" * (32 * 1024) + b"S" * (32 * 1024)
    summary = {"schema": "legal-ir-nonmodel-dual-pipe-payload@1", "fixture_case": sys.argv[1],
        "stdout_bytes": len(stdout), "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "models_called": 0, "provider_calls_performed": False, "canonical_tasks_accepted": 0}
    write_new(workspace / "payload-summary.json", (json.dumps(summary, sort_keys=True) + "\n").encode())
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

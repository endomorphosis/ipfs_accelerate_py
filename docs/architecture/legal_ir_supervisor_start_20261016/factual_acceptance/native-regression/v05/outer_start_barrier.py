"""Actual attached child waits for native admission, then execs exact Docker start."""
import os
from pathlib import Path
import resource
import stat
import sys
import time


def release_gate_observation(*, mode, nlink, size, data, now, deadline):
    """Pure observation contract; empty publication never grants execution."""
    if now >= deadline:
        raise SystemExit("admission barrier expired")
    if not stat.S_ISREG(mode) or nlink != 1:
        raise SystemExit("invalid admission barrier observation")
    if size == 0 and data in (b"", b"G"):
        # fstat may see zero immediately before the producer flushes G.
        # Even when read sees G, wait for a complete one-byte observation.
        return False
    if size == 1 and data == b"G":
        return True
    raise SystemExit("invalid admission barrier observation")


def main():
    if len(sys.argv) < 7 or sys.argv[2] != "--" or sys.argv[3] != "/usr/bin/docker":
        raise SystemExit("closed Docker-start barrier argv required")
    barrier = Path(sys.argv[1])
    if not barrier.is_absolute():
        raise SystemExit("absolute native admission barrier required")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            fd = os.open(barrier, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            released = release_gate_observation(mode=info.st_mode, nlink=info.st_nlink, size=info.st_size,
                data=stream.read(2), now=time.monotonic(), deadline=deadline)
        if not released:
            time.sleep(0.05)
            continue
        # Docker's Go client needs a large virtual address reservation; AS and
        # NPROC are deliberately not claimed as per-job limits. The container
        # has verified cgroup memory/pids/CPU limits; host RSS is sampled by SDK.
        for which, maximum in ((resource.RLIMIT_CPU, 120), (resource.RLIMIT_FSIZE, 2 * 1024 * 1024),
                               (resource.RLIMIT_NOFILE, 64)):
            _, hard = resource.getrlimit(which)
            value = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
            resource.setrlimit(which, (value, value))
        os.execvpe(sys.argv[3], sys.argv[3:], dict(os.environ))
    raise SystemExit("native Docker attachment barrier expired")


if __name__ == "__main__":
    main()

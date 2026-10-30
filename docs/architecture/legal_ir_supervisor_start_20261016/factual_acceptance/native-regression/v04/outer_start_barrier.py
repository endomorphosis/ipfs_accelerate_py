"""Actual attached child waits for native admission, then execs exact Docker start."""
import os
from pathlib import Path
import resource
import stat
import sys
import time


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
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 1 or stream.read(2) != b"G":
                raise SystemExit("invalid native admission barrier")
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

"""Pinned trusted child: wait for owned native attachment, then exec exact argv."""
import os
import json
from pathlib import Path
import resource
import stat
import sys
import time


def main():
    if len(sys.argv) < 7 or sys.argv[2] != "--bounds-json" or sys.argv[4] != "--":
        raise SystemExit("invalid startup barrier argv")
    barrier = Path(sys.argv[1])
    if not barrier.is_absolute() or not Path(sys.argv[5]).is_absolute() or len(sys.argv[3]) > 1024:
        raise SystemExit("absolute barrier and executable required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate runtime bound")
            result[key] = value
        return result
    bounds = json.loads(sys.argv[3], object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite bound")))
    maxima = {"cpu_seconds": 120, "memory_bytes": 512 * 1024 * 1024,
              "output_file_bytes": 2 * 1024 * 1024, "open_files": 64, "processes": 256}
    if type(bounds) is not dict or set(bounds) != set(maxima) or not all(
            type(bounds[key]) is int and 0 < bounds[key] <= maximum for key, maximum in maxima.items()):
        raise SystemExit("finite frozen fixture runtime bounds required")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            descriptor = os.open(barrier, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 1 or stream.read(2) != b"G":
                raise SystemExit("invalid startup barrier")
        # Apply only child-local hard limits immediately before productive
        # exec. NPROC is deliberately omitted: it counts the entire UID, not
        # this job. Native four-slot group enforcement remains cooperative.
        for limit_name, key in (("RLIMIT_CPU", "cpu_seconds"), ("RLIMIT_AS", "memory_bytes"),
                                ("RLIMIT_FSIZE", "output_file_bytes"), ("RLIMIT_NOFILE", "open_files")):
            which = getattr(resource, limit_name)
            _, hard = resource.getrlimit(which)
            value = bounds[key] if hard == resource.RLIM_INFINITY else min(bounds[key], hard)
            resource.setrlimit(which, (value, value))
        os.execvpe(sys.argv[5], sys.argv[5:], dict(os.environ))
    raise SystemExit("native attachment startup barrier expired")


if __name__ == "__main__":
    main()

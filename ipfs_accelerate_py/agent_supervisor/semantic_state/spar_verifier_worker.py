"""Fixed entry point executed only from the owner's sealed source archive."""
from __future__ import annotations

import fcntl
import json
import os
import resource
import sys
import types


def process_identity() -> dict:
    raw = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "parent_pid": os.getppid(), "start_ticks": int(raw[19]),
            "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip()}


def main(archive_fd: int, request_fd: int, result_fd: int) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    for fd in (archive_fd, request_fd):
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != seals:
            raise ValueError("unsealed native verifier input")
    archive = f"/proc/self/fd/{archive_fd}"
    # The verified software-contract package is storage neutral. Deliberately
    # bypass unrelated optional service/auto-install package initializers.
    for name in ("ipfs_datasets_py", "ipfs_datasets_py.logic"):
        package = types.ModuleType(name)
        package.__path__ = [archive + "/" + name.replace(".", "/")]
        sys.modules[name] = package
    sys.path.insert(0, archive)
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.logic.software_contracts.semantic_state.spar_verification import reconstruct_source
    identity = process_identity()
    print(json.dumps(identity, sort_keys=True), flush=True)
    if sys.stdin.buffer.readline(16) != b"run\n":
        raise ValueError("native parent did not admit execution")
    request = json.loads(os.pread(request_fd, os.fstat(request_fd).st_size, 0))
    try:
        result = reconstruct_source(request["repositories"])
        outcome = {"status": "verified_component", **result}
    except Exception as exc:
        outcome = {"status": "unavailable", "reason": type(exc).__name__, "detail": str(exc)[:512]}
    record = {"schema": "accelerator/spar-verifier-child-result@1",
              "request_cid": cid_for_structured(request), "process_identity": identity,
              "outcome": outcome, "completion_authority": False, "accepted_root": False}
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if len(raw) > 64 * 1024**2:
        raise ValueError("native verifier result exceeded bound")
    with os.fdopen(os.dup(result_fd), "wb") as output:
        output.write(raw)
        output.flush()
    fcntl.fcntl(result_fd, fcntl.F_ADD_SEALS, seals)


if __name__ == "__main__":
    main(*(int(value) for value in sys.argv[1:]))

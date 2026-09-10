"""Keep native board launches outside the temporary fleet repair service."""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

_REPAIR_SERVICE = re.compile(r"ipfs-taskboard-repair-job(?:-[A-Za-z0-9_.-]+)?\.service")


def in_temporary_repair_service() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("cannot determine native launch service lifetime") from exc
    return any(
        _REPAIR_SERVICE.fullmatch(part) is not None
        for line in lines
        for part in line.split(":", 2)[-1].split("/")
    )


def delegate_repair_service_launch(command: Sequence[str]) -> int | None:
    """Re-run a native launcher in its own user scope when repair-owned.

    A detached session still belongs to its parent's systemd cgroup. The
    repair service correctly kills that group when its coding job ends. A
    separate scope instead survives while the native supervisor is alive.
    The inner launcher keeps its own source, owner, credential and process
    admission checks. It sees the scope and does not delegate recursively.

    Call before retiring credentials, opening inherited admission descriptors,
    or launching workers. No cleanup policy of the repair job is relaxed.
    """
    if not in_temporary_repair_service():
        return None
    if not command or any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("native launch command is invalid")
    scope = f"ipfs-agent-runtime-{uuid.uuid4().hex}.scope"
    result = subprocess.run(
        ["/usr/bin/systemd-run", "--user", "--scope", "--collect", "--quiet",
         "--no-ask-password", f"--unit={scope}", "--", *command],
        check=False,
    )
    return int(result.returncode)

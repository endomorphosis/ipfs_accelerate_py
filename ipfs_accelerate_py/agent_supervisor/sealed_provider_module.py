"""Isolated execution of packaged provider entries from a sealed LGCVF ZIP.

The live capsule is a sealed ZIP held by descriptor.  A member path such as
``/proc/self/fd/7/.../grok_cli_runner.py`` is a valid import identity but is
not an operating-system file that Python can execute directly.  This module
defines the small, exact ``python -c`` bridge used for those two provider
entries and the parser used to forward only that capsule descriptor.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

LGCVF_SEALED_PROVIDER_MODULES: Final = frozenset(
    {
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner",
    }
)
_SEALED_MODULE_ORIGIN: Final = re.compile(
    r"/proc/self/fd/([0-9]+)/ipfs_accelerate_py/agent_supervisor/"
    r"(?:todo_daemon/implementation_daemon|provider_fallback_runner)\.py"
)
LGCVF_SEALED_PROVIDER_BOOTSTRAP: Final = r"""import fcntl,importlib.machinery,os,runpy,stat,sys
def _deny(): raise SystemExit(78)
try:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode or not sys.platform.startswith('linux'): _deny()
    fd=int(sys.argv.pop(1)); module=sys.argv.pop(1)
    if fd<3 or module not in {'ipfs_accelerate_py.agent_supervisor.grok_cli_runner','ipfs_accelerate_py.agent_supervisor.provider_fallback_runner'}: _deny()
    metadata=os.fstat(fd); required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SEAL
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size<=0 or fcntl.fcntl(fd,fcntl.F_GET_SEALS)&required!=required: _deny()
    archive='/proc/self/fd/'+str(fd); observed=os.stat(archive)
    if (observed.st_dev,observed.st_ino)!=(metadata.st_dev,metadata.st_ino): _deny()
    stdlib=[]
    for entry in sys.path:
        if type(entry) is not str or not entry or not os.path.isabs(entry): continue
        lowered=entry.casefold()
        if 'site-packages' in lowered or 'dist-packages' in lowered: continue
        stdlib.append(entry)
    sys.path[:]=[archive+'/ipfs_datasets_py',archive,*dict.fromkeys(stdlib)]
    allowed=(importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder)
    if any(finder not in allowed for finder in sys.meta_path): _deny()
    namespace=runpy.run_module(module,run_name='__main__',alter_sys=True)
    if namespace.get('__name__')!='__main__': _deny()
except SystemExit: raise
except BaseException: raise SystemExit(78)
"""


def sealed_provider_capsule_descriptor(module_file: str | Path) -> int:
    """Return the sealed archive FD named by an admitted module origin."""

    match = _SEALED_MODULE_ORIGIN.fullmatch(str(module_file))
    if match is None:
        return -1
    descriptor = int(match.group(1))
    return descriptor if descriptor >= 3 else -1


def build_sealed_provider_module_command(
    module_name: str,
    argv: Sequence[str],
    *,
    module_file: str | Path,
) -> list[str] | None:
    """Build the isolated ZIP-member command, or ``None`` off capsule."""

    module = str(module_name or "").strip()
    descriptor = sealed_provider_capsule_descriptor(module_file)
    if module not in LGCVF_SEALED_PROVIDER_MODULES or descriptor < 3:
        return None
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        LGCVF_SEALED_PROVIDER_BOOTSTRAP,
        str(descriptor),
        module,
        *(str(item) for item in argv),
    ]


def sealed_provider_module_command_descriptor(
    command: Sequence[str],
    *,
    module_name: str = "",
) -> int:
    """Return the one exact capsule FD carried by a sealed provider command."""

    tokens = tuple(str(item) for item in command)
    if len(tokens) < 8:
        return -1
    expected_module = str(module_name or "").strip()
    observed_module = tokens[7]
    if (
        tokens[1:5] != ("-I", "-S", "-B", "-c")
        or tokens[5] != LGCVF_SEALED_PROVIDER_BOOTSTRAP
        or observed_module not in LGCVF_SEALED_PROVIDER_MODULES
        or (expected_module and observed_module != expected_module)
    ):
        return -1
    try:
        if Path(tokens[0]).resolve(strict=True) != Path(sys.executable).resolve(
            strict=True
        ):
            return -1
        descriptor = int(tokens[6])
        os.fstat(descriptor)
    except (OSError, TypeError, ValueError):
        return -1
    return descriptor if descriptor >= 3 else -1


__all__ = (
    "LGCVF_SEALED_PROVIDER_BOOTSTRAP",
    "LGCVF_SEALED_PROVIDER_MODULES",
    "build_sealed_provider_module_command",
    "sealed_provider_capsule_descriptor",
    "sealed_provider_module_command_descriptor",
)

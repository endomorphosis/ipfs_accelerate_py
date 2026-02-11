import importlib
import os
import subprocess
import sys
from typing import Dict, Iterable, Tuple


def _in_venv() -> bool:
    try:
        return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    except Exception:
        return False


def _should_auto_install() -> bool:
    env = os.getenv("IPFS_ACCEL_AUTO_INSTALL")
    if env is not None:
        return env.strip() not in {"0", "false", "False", "no", "NO"}
    # Default: enable when running inside a virtual environment
    return _in_venv()


def _pip_install(package: str) -> Tuple[bool, str]:
    try:
        cmd = [sys.executable, "-m", "pip", "install", package]
        # Allow system package managers to co-exist if needed
        if os.getenv("PIP_BREAK_SYSTEM_PACKAGES") == "1":
            cmd.append("--break-system-packages")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        ok = proc.returncode == 0
        output = (proc.stdout or "") + (proc.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"pip install failed for {package}: {e}"


def ensure_packages(packages: Iterable[str] | Dict[str, str]) -> Dict[str, str]:
    """Ensure Python packages are installed.

    Accepts either a list of import names or a mapping of import_name -> pip_name.
    Returns a mapping import_name -> status string (installed|ok|failed:<msg>|skipped).
    Controlled by env var `IPFS_ACCEL_AUTO_INSTALL` (default on in venv).
    """
    mapping: Dict[str, str]
    if isinstance(packages, dict):
        mapping = dict(packages)
    else:
        mapping = {name: name for name in packages}

    results: Dict[str, str] = {}

    auto = _should_auto_install()
    for import_name, pip_name in mapping.items():
        try:
            importlib.import_module(import_name)
            results[import_name] = "ok"
            continue
        except Exception:
            if not auto:
                results[import_name] = "skipped"
                continue

        ok, out = _pip_install(pip_name)
        if not ok:
            results[import_name] = f"failed:{out.strip()[:3000]}"
            continue

        # Re-try import after installation
        try:
            importlib.invalidate_caches()
            importlib.import_module(import_name)
            results[import_name] = "installed"
        except Exception as e:
            results[import_name] = f"failed:post-import:{e}"

    return results


__all__ = ["ensure_packages"]

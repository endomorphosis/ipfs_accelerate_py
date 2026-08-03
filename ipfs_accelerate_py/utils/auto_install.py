from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import os
import re
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


_URL_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s]+)@")
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)([?&](?:[^=&]*(?:token|secret|password|passwd|api[_-]?key|signature|credential|auth)[^=&]*)=)[^&\s]*"
)


def _redact_sensitive(value: object) -> str:
    """Redact URL credentials and common secret query parameters."""

    text = str(value)
    text = _URL_USERINFO_RE.sub(r"\1***@", text)
    return _SENSITIVE_VALUE_RE.sub(r"\1***", text)


def _exception_receipt(error: BaseException) -> str:
    try:
        detail = str(error)
    except Exception:
        detail = "<unprintable>"
    payload = (
        f"{type(error).__module__}.{type(error).__qualname__}:{detail}"
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def _is_direct_reference_or_path(package: str) -> bool:
    raw = package.strip()
    return (
        " @ " in raw
        or raw.startswith((".", "/", "git+", "http://", "https://", "file:"))
    )


def _has_declarative_constraint(package: str) -> bool:
    return ";" in package or bool(re.search(r"[<>=!~]", package))


def _pip_install(package: str) -> Tuple[bool, str]:
    try:
        cmd = [sys.executable, "-m", "pip", "install", package]
        # Allow system package managers to co-exist if needed
        if os.getenv("PIP_BREAK_SYSTEM_PACKAGES") == "1":
            cmd.append("--break-system-packages")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        ok = proc.returncode == 0
        if ok:
            return True, ""
        output = ((proc.stdout or "") + (proc.stderr or "")).encode(
            "utf-8",
            errors="replace",
        )
        receipt = hashlib.sha256(output).hexdigest()[:16]
        return False, f"exit-code={proc.returncode}:receipt={receipt}"
    except Exception as e:
        error_type = type(e).__name__
        if not error_type.isidentifier():
            error_type = "Error"
        return False, f"{error_type}:receipt={_exception_receipt(e)}"


def _requirement_parser():
    """Resolve the declared or pip-vendored PEP 508 parser."""

    try:
        from packaging.requirements import Requirement

        return Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement

            return Requirement
        except ImportError:
            return None


def _requirement_applies(package: str) -> bool:
    """Return whether a PEP 508 requirement applies to this interpreter."""

    try:
        parser = _requirement_parser()
        if parser is None:
            raise ImportError("no PEP 508 parser is available")
        requirement = parser(package)
    except Exception:
        # Unknown/unsupported markers are treated as applicable so the package
        # cannot be accepted without validation. Direct references and paths
        # remain delegated to pip.
        return True

    version = tuple(sys.version_info[:3])
    version += (0,) * (3 - len(version))
    environment = {
        "python_version": f"{version[0]}.{version[1]}",
        "python_full_version": ".".join(str(part) for part in version),
    }
    return requirement.marker is None or requirement.marker.evaluate(environment)


def _installed_requirement_satisfied(package: str) -> bool:
    """Check an installed distribution against a versioned pip requirement."""

    try:
        parser = _requirement_parser()
        if parser is None:
            raise ImportError("no PEP 508 parser is available")
        requirement = parser(package)
    except Exception:
        if _is_direct_reference_or_path(package):
            return True
        # Plain unversioned imports retain legacy behavior. Versioned or marked
        # requirements fail closed when the parser is unavailable.
        return not _has_declarative_constraint(package)

    if not requirement.specifier:
        return True

    try:
        installed_version = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False

    return requirement.specifier.contains(installed_version, prereleases=True)


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
        if not _requirement_applies(pip_name):
            results[import_name] = "skipped"
            continue

        # For constrained requirements, validate distribution metadata before
        # importing. Wrong-version package code must never execute as part of a
        # repair decision.
        if _installed_requirement_satisfied(pip_name):
            try:
                importlib.import_module(import_name)
                results[import_name] = "ok"
                continue
            except Exception:
                pass

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
            if _installed_requirement_satisfied(pip_name):
                importlib.import_module(import_name)
                results[import_name] = "installed"
            else:
                results[import_name] = "failed:post-install:version-mismatch"
        except Exception as e:
            error_type = type(e).__name__
            if not error_type.isidentifier():
                error_type = "Error"
            results[import_name] = (
                f"failed:post-import:{error_type}:receipt={_exception_receipt(e)}"
            )

    return results


def ensure_distributions(
    packages: Iterable[str] | Dict[str, str],
    *,
    force: bool = False,
) -> Dict[str, str]:
    """Ensure distribution metadata without importing package code.

    This is intended for packages such as FastMCP whose import origin must be
    audited by a dedicated loader before any module code executes.
    """

    if isinstance(packages, dict):
        mapping = dict(packages)
    else:
        mapping = {name: name for name in packages}

    results: Dict[str, str] = {}
    auto = _should_auto_install()
    for name, requirement in mapping.items():
        if not _requirement_applies(requirement):
            results[name] = "skipped"
            continue
        if not force and _installed_requirement_satisfied(requirement):
            results[name] = "ok"
            continue
        if not auto:
            results[name] = "skipped"
            continue
        ok, output = _pip_install(requirement)
        if not ok:
            results[name] = f"failed:{output.strip()[:3000]}"
        elif _installed_requirement_satisfied(requirement):
            results[name] = "installed"
        else:
            results[name] = "failed:post-install:version-mismatch"
    return results


__all__ = ["ensure_distributions", "ensure_packages"]

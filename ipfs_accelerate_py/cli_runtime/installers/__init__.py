"""CLI tool installers for the shared runtime.

Importing this package is side-effect free: it does not download archives,
install tools, start processes, or probe authentication. Call
:func:`ensure_goose` only from explicit provider resolution paths.
"""

from __future__ import annotations

from .goose import (
    DEFAULT_MANIFEST_NAME,
    GOOSE_EXECUTABLE,
    GOOSE_EXECUTABLE_WINDOWS,
    LINUX_VARIANTS,
    WINDOWS_VARIANTS,
    GooseInstallResult,
    GooseReadiness,
    assess_goose_readiness,
    default_manifest_path,
    detect_linux_libc,
    detect_platform,
    discover_goose,
    ensure_goose,
    goose_auth_available,
    goose_auto_install_enabled,
    install_goose_from_manifest,
    load_release_manifest,
    managed_executable_path,
    managed_install_root,
    normalize_arch,
    normalize_os,
    select_release_asset,
    validate_platform,
)

__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "GOOSE_EXECUTABLE",
    "GOOSE_EXECUTABLE_WINDOWS",
    "LINUX_VARIANTS",
    "WINDOWS_VARIANTS",
    "GooseInstallResult",
    "GooseReadiness",
    "assess_goose_readiness",
    "default_manifest_path",
    "detect_linux_libc",
    "detect_platform",
    "discover_goose",
    "ensure_goose",
    "goose_auth_available",
    "goose_auto_install_enabled",
    "install_goose_from_manifest",
    "load_release_manifest",
    "managed_executable_path",
    "managed_install_root",
    "normalize_arch",
    "normalize_os",
    "select_release_asset",
    "validate_platform",
]

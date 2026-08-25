"""Build pinned wheels for the external-agent stack (EAAEF-160).

Does not use sibling checkouts or editable installs as release authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


STACK_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-stack-build@1"
PRIMARY_PACKAGES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
)
OPTIONAL_MCP = "Mcp-Plus-Plus"


class StackBuildError(ValueError):
    """Stack build request is not releasable."""


def plan_wheel_build(
    *,
    packages: Sequence[str],
    include_mcp: bool,
    editable: bool,
    sibling_checkout: bool,
) -> Mapping[str, Any]:
    if editable or sibling_checkout:
        raise StackBuildError("release wheels cannot depend on editable or sibling checkouts")
    selected = tuple(packages)
    unknown = [name for name in selected if name not in PRIMARY_PACKAGES and name != OPTIONAL_MCP]
    if unknown:
        raise StackBuildError(f"unknown package {unknown[0]}")
    if OPTIONAL_MCP in selected and not include_mcp:
        raise StackBuildError("MCP++ is optional and must be explicitly required")
    if any(name not in selected for name in PRIMARY_PACKAGES):
        raise StackBuildError("three primary packages are required")
    return MappingProxyType(
        {
            "schema": STACK_SCHEMA,
            "packages": list(selected),
            "include_mcp": bool(include_mcp),
            "editable": False,
            "sibling_checkout": False,
        }
    )


def default_primary_plan() -> Mapping[str, Any]:
    return plan_wheel_build(
        packages=PRIMARY_PACKAGES,
        include_mcp=False,
        editable=False,
        sibling_checkout=False,
    )

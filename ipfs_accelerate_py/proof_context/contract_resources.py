"""Data-only MCP++ contract resources. Never searches sibling trees."""

from __future__ import annotations

import importlib.resources
from typing import Any, Mapping

from ipfs_accelerate_py.proof_context.compatibility import CompatibilityError, reject_pseudo_cid

RESOURCE_PACKAGE_CANDIDATES = (
    "mcpp_contracts",
    "mcp_plus_plus_contracts",
    "mcpp.schemas",
)


class ContractResourceUnavailable(RuntimeError):
    reason = "unavailable"


def load_schema_text(resource_name: str) -> str:
    """Load a schema from an installed data-only package.

    Source checkouts of Mcp-Plus-Plus are never consulted.
    """
    if ".." in resource_name or resource_name.startswith("/"):
        raise CompatibilityError("contract resource path must be a package resource name")
    errors: list[str] = []
    for package in RESOURCE_PACKAGE_CANDIDATES:
        try:
            traversable = importlib.resources.files(package)
        except ModuleNotFoundError as exc:
            errors.append(str(exc))
            continue
        target = traversable.joinpath(resource_name)
        try:
            return target.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, AttributeError) as exc:
            errors.append(str(exc))
            continue
    raise ContractResourceUnavailable(
        "mcp-plus-plus-contracts resource is unavailable from installed packages"
    )


def admit_schema_mapping(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    marker = schema.get("schema") or schema.get("$id")
    if not isinstance(marker, str) or not marker:
        raise CompatibilityError("contract schema mapping is missing identity")
    return schema


def admit_cid(value: str) -> str:
    reject_pseudo_cid(value)
    return value

"""GitHub CLI integration with a scoped, lazy package surface."""

from __future__ import annotations

import importlib
from typing import Any


_EXPORTS = {
    "GitHubCLI": (".wrapper", "GitHubCLI"),
    "RunnerManager": (".wrapper", "RunnerManager"),
    "WorkflowQueue": (".wrapper", "WorkflowQueue"),
    # Historical alias of WorkflowQueue.
    "WorkflowManager": (".wrapper", "WorkflowQueue"),
    "GitHubAPICache": (".cache", "GitHubAPICache"),
    "get_global_cache": (".cache", "get_global_cache"),
    "configure_cache": (".cache", "configure_cache"),
    "GitHubGraphQL": (".graphql_wrapper", "GitHubGraphQL"),
}

__all__ = [
    "GitHubCLI",
    "WorkflowQueue",
    "WorkflowManager",
    "RunnerManager",
    "GitHubAPICache",
    "get_global_cache",
    "configure_cache",
    "GitHubGraphQL",
]


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = spec
    module = importlib.import_module(f"{__name__}{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    if name in {"WorkflowQueue", "WorkflowManager"}:
        globals()["WorkflowQueue"] = value
        globals()["WorkflowManager"] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))

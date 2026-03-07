"""GitHub CLI integration for IPFS Accelerate."""

from .wrapper import GitHubCLI, RunnerManager, WorkflowQueue
from .cache import GitHubAPICache, get_global_cache, configure_cache
from .graphql_wrapper import GitHubGraphQL

# Backwards compatibility: older callers imported WorkflowManager from this
# package even though the concrete implementation class is WorkflowQueue.
WorkflowManager = WorkflowQueue

__all__ = [
    "GitHubCLI",
    "WorkflowQueue",
    "WorkflowManager",
    "RunnerManager",
    "GitHubAPICache",
    "get_global_cache",
    "configure_cache",
    "GitHubGraphQL"
]

"""ASI-168 declared output marker for task proposal routing.

Canonical implementation (ASREF-landed)::

    ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router

This file is intentionally free of re-exports so it cannot dual-load landed
module classes. Package imports of the historical flat name resolve through the
landed-module alias finder to the domain package.
"""

from __future__ import annotations

ASI_168_CONSUMER_ID = "task_proposal_router"
ASI_168_REQUIREMENT_ID = "requirement:complete-provider-callsite.v1"
ASI_168_LANDED_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router"
)
ASI_168_LANDED_PATH = (
    "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py"
)
ASI_168_MIGRATED = True
ASI_168_IS_COMPLETION_EVIDENCE = False
ASI_168_IS_CORRECTNESS_EVIDENCE = False

"""DCR-031 validation surface for MCP contract obligation compilation.

Board task DCR-031 requires this module path. Implementation and exhaustive
assertions live in ``test_agent_supervisor_mcp_contract_obligations`` and the
``mcp_contract_obligations`` compiler; this module re-exports those tests so
the declared validation command stays stable.
"""

from __future__ import annotations

# Re-export every test from the shared obligations suite.
from test.api.test_agent_supervisor_mcp_contract_obligations import *  # noqa: F403

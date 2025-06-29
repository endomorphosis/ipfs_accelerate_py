# MCP Integration Checker Report

## Overview

- **Timestamp**: 2025-05-05T22:09:25.060734
- **MCP Server Status**: not_running
- **IPFS Accelerate Installed**: True
- **IPFS Accelerate Version**: 0.4.0

## System Information

- **OS**: Linux #24~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Mar 25 20:14:34 UTC 2
- **Distribution**: Linux-6.11.0-24-generic-x86_64-with-glibc2.39
- **Architecture**: x86_64
- **Python Version**: 3.12.3

## MCP Server

**Status**: not_running

**Available Endpoints**:
No endpoints available

**Available Tools**:
No tools available

## IPFS Accelerate

**Installed**: True
**Version**: 0.4.0

**Available Functions**:
No functions available

## Issues

- Port 8004 is not open
- MCP server is not running or not responding
- Error checking IPFS Accelerate: cannot access local variable 'module' where it is not associated with a value

## Recommendations

- Check if any process is using port 8004
- Start the MCP server with './restart_mcp_server.sh'

## Next Steps

1. Address the issues listed above
2. Run `fix_mcp_tool_registration.py` to register all tools
3. Run `verify_mcp_tools.py` to verify the tools are working

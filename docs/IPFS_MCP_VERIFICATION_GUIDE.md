# IPFS Accelerate MCP Server Testing and Verification Guide

This guide describes how to test, fix, and verify the IPFS Accelerate MCP server implementation, with a focus on ensuring all required tools (especially `get_hardware_info`) are properly registered and accessible.

## Overview

The IPFS Accelerate MCP server provides a standardized interface for accessing IPFS functionality through the Model Context Protocol (MCP). This implementation includes various tools for interacting with IPFS, hardware detection, and model inference.

## Available Scripts

This repository includes several scripts for testing and fixing the MCP server:

1. **`test_mcp_server_comprehensive.py`** - The main comprehensive test script
2. **`fix_and_verify_mcp_server.sh`** - Shell script that runs tests with auto-fix capabilities
3. **`run_comprehensive_diagnosis.sh`** - Diagnostic tool that provides detailed analysis and recommendations
4. **`restart_mcp_server.sh`** - Helper script to restart the MCP server after fixes

## Required Tools

The MCP server should correctly implement and expose the following tools:

- `get_hardware_info` - Get hardware acceleration information
- `health_check` - Check the health of the MCP server
- `ipfs_add_file` - Add a file to IPFS
- `ipfs_cat` - Retrieve content from IPFS by CID
- `ipfs_files_write` - Write to the IPFS Mutable File System
- `ipfs_files_read` - Read from the IPFS Mutable File System
- `list_models` - List available inference models
- `create_endpoint` - Create a model inference endpoint
- `run_inference` - Run inference using a model endpoint

## Testing Process

### 1. Run Comprehensive Diagnosis

Start by running a comprehensive diagnosis to identify any issues:

```bash
./run_comprehensive_diagnosis.sh
```

This will:
- Test connectivity to the MCP server
- Check if required tools are registered and accessible
- Provide a detailed report of the server status
- Offer recommendations for fixing any issues

### 2. Fix and Verify

If issues are detected, use the fix and verify script:

```bash
./fix_and_verify_mcp_server.sh --auto-fix
```

This script will:
- Run tests to detect issues
- Automatically fix common problems
- Restart the server if needed
- Verify that fixes were successful

### 3. Manual Verification

You can manually verify tool functionality using HTTP requests:

```bash
# Test get_hardware_info tool
curl -X POST http://127.0.0.1:8002/mcp/tool/get_hardware_info

# Test health_check tool
curl -X POST http://127.0.0.1:8002/mcp/tool/health_check
```

## Troubleshooting

If issues persist after running the auto-fix script, check the following:

1. Server logs:
   ```bash
   tail -50 final_mcp_server.log
   ```

2. Check API endpoints implementation in `mcp/server.py`
   - Ensure standard API endpoints are defined (`/tools` and `/tools/{tool_name}/invoke`)

3. Check tool registration in `final_mcp_server.py`
   - Ensure all required tools are properly defined in the `tools` list
   - Check that tool functions are correctly implemented in the `IPFSAccelerate` class

4. Restart the server to apply changes:
   ```bash
   ./restart_mcp_server.sh
   ```

## API Endpoints

The MCP server should provide the following endpoints:

1. **Standard MCP Endpoints:**
   - `/mcp/manifest` - Retrieve the manifest with tools information
   - `/mcp/tool/{tool_name}` - Direct tool access endpoint

2. **Compatibility Endpoints:**
   - `/tools` - List all available tools
   - `/tools/{tool_name}/invoke` - Tool invocation compatible with standard clients

## Understanding Test Results

The test scripts generate JSON output files with detailed information:

- `mcp_diagnostics.json` - Diagnostic test results
- `mcp_test_results.json` - Comprehensive test results
- `post_fix_mcp_test_results.json` - Test results after applying fixes

These files include:
- Server connectivity status
- List of available tools
- Status of required tools
- Hardware acceleration information
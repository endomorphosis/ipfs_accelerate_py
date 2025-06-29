# IPFS Accelerate MCP Integration Fix Guide

This guide explains how to fix the integration issues between the IPFS Accelerate Python package and its Model Context Protocol (MCP) server implementation.

## Background

The IPFS Accelerate Python package provides functionality for interacting with IPFS and related services. The MCP server exposes this functionality as tools that can be called by MCP clients. However, there are currently issues with the tool registration mechanism that prevent all tools from being properly exposed.

## Issues Identified

1. **Missing Tools in Manifest**: Some tools (`ipfs_add_file`, `ipfs_node_info`) work when called directly but don't appear in the MCP manifest.
2. **Non-functional Tools**: Some tools (`model_inference`, `list_models`) are registered but not functional.
3. **Namespace Conflicts**: Possible namespace conflicts during tool registration.
4. **Inconsistent Registration**: The tool registration process is not consistent across all tools.

## Fix Implementation

We've implemented the following fixes to address these issues:

### 1. Fixed Tool Registration Mechanism

The `fix_mcp_tool_registration.py` script provides a unified tool registration mechanism that ensures all tools are properly registered and appear in the MCP manifest. Key features:

- Imports MCP module using multiple approaches to handle different environments
- Verifies tool registration by checking the MCP manifest
- Registers resources and tools with proper categorization
- Implements missing tools with mock implementations when necessary

### 2. Updated Server Restart Mechanism

The `restart_mcp_server.sh` script has been updated to use the fixed tool registration mechanism. Key changes:

- Added option to use either the fixed or original registration mechanism
- Starts the MCP server and then applies the fixed tool registration
- Provides better error handling and reporting

### 3. Comprehensive Verification Tool

The `verify_mcp_tools.py` script provides a comprehensive way to verify that all tools are properly registered and functional. Key features:

- Checks server status and manifest
- Gets available tools and identifies missing tools
- Tests all available tools with appropriate test arguments
- Generates a detailed report of the verification results

## Usage

### Restarting the MCP Server with Fixed Registration

```bash
./restart_mcp_server.sh [--port PORT] [--host HOST] [--debug]
```

By default, the script will use the fixed tool registration mechanism. To use the original mechanism, add the `--use-original` flag:

```bash
./restart_mcp_server.sh --use-original
```

### Verifying Tool Registration and Functionality

```bash
python verify_mcp_tools.py [--host HOST] [--port PORT] [--output OUTPUT]
```

This will generate a verification report in the specified output file (default: `mcp_tools_verification_report.md`).

## Expected Tools

The following tools should be available after fixing the MCP server integration:

### Hardware Tools
- `get_hardware_info`: Get hardware information about the system

### IPFS Tools
- `ipfs_add_file`: Add a file to IPFS
- `ipfs_node_info`: Get information about the IPFS node
- `ipfs_cat`: Get the content of a file from IPFS
- `ipfs_get`: Get a file from IPFS
- `ipfs_files_write`: Write to a file in IPFS MFS
- `ipfs_files_read`: Read from a file in IPFS MFS
- `ipfs_files_ls`: List files in IPFS MFS

### Model Tools
- `model_inference`: Run inference on a model
- `list_models`: List available models
- `init_endpoints`: Initialize endpoints for models

### Virtual Filesystem Tools
- `vfs_list`: List items in the virtual filesystem
- `vfs_read`: Read from the virtual filesystem
- `vfs_write`: Write to the virtual filesystem
- `vfs_delete`: Delete from the virtual filesystem

### Storage Tools
- `create_storage`: Create a new storage volume
- `list_storage`: List available storage volumes
- `get_storage_info`: Get information about a storage volume
- `delete_storage`: Delete a storage volume

## Troubleshooting

If tools are still not appearing in the MCP manifest after applying the fixes, try the following:

1. Check the server logs for error messages
2. Verify that the MCP server is running correctly
3. Restart the MCP server with the `--debug` flag
4. Run the verification script to see which tools are missing
5. Check for namespace conflicts in the tool names

## Next Steps

After implementing these fixes, the next steps are:

1. Implement the missing tools from the expected tools list
2. Improve the mock implementations with actual functionality
3. Add proper error handling and validation to all tools
4. Add comprehensive tests for all tools
5. Update the documentation to reflect the available tools

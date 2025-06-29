# IPFS Accelerate MCP Server Tools

This package contains tools for testing, assessing, fixing, and verifying the IPFS Accelerate MCP (Model Context Protocol) server implementation.

## Overview

The IPFS Accelerate MCP server exposes IPFS functionality and related tools through a standardized API. This package contains tools to:

1. Test the coverage of tools exposed by the MCP server
2. Fix missing tool implementations 
3. Assess the overall health and functionality of the MCP server
4. Generate comprehensive verification reports
5. Automatically fix common issues

## New Comprehensive Testing and Verification Suite

We've added a comprehensive testing and verification suite to ensure the MCP server is properly configured:

### 1. Comprehensive Test Script

The `test_mcp_server_comprehensive.py` script provides:
- Connectivity testing to the MCP server
- Tool availability verification
- Hardware acceleration detection
- Auto-fixing capability for common issues

```bash
# Run basic test
python test_mcp_server_comprehensive.py

# Run with auto-fix capability
python test_mcp_server_comprehensive.py --auto-fix
```

### 2. Fix and Verify Script

The `fix_and_verify_mcp_server.sh` script automates:
- Running tests to detect issues
- Applying fixes for missing tools and endpoints
- Restarting the server if needed
- Verifying the fixes were successful

```bash
# Run with auto-fix
./fix_and_verify_mcp_server.sh --auto-fix
```

### 3. Comprehensive Diagnosis

The `run_comprehensive_diagnosis.sh` script provides:
- Detailed analysis of the MCP server
- Checking tool registration and functionality
- Recommendations for fixing issues

```bash
./run_comprehensive_diagnosis.sh
```

### 4. Verification Report Generator

The `generate_mcp_verification_report.py` script creates:
- Detailed verification reports in Markdown format
- Hardware acceleration summary
- Tool status summary
- Server health information

```bash
python generate_mcp_verification_report.py --output report.md
```

### 5. One-Step Verification

The `verify_mcp_server.sh` script combines all testing and verification in one step:
- Checks server status
- Applies fixes if needed
- Generates comprehensive report

```bash
./verify_mcp_server.sh [--start-server]
```

## Required Tools

The MCP server should correctly implement and expose these tools:

- `get_hardware_info` - Get hardware acceleration information
- `health_check` - Check the health of the MCP server
- `ipfs_add_file` - Add a file to IPFS
- `ipfs_cat` - Retrieve content from IPFS by CID
- `ipfs_files_write` - Write to the IPFS Mutable File System
- `ipfs_files_read` - Read from the IPFS Mutable File System
- `list_models` - List available inference models
- `create_endpoint` - Create a model inference endpoint
- `run_inference` - Run inference using a model endpoint

## Original MCP Server Testing Tools

Below are the original tools included in the package:

### MCP Server Testing Tool

The `test_mcp_server_coverage.py` script tests each expected tool against the MCP server. It performs the following steps:

1. Connects to the MCP server and lists available tools
2. Tests each tool with appropriate parameters
3. Reports on which tools are available and working properly
4. Calculates the coverage percentage

### MCP Server Fixing Tool

The `fix_mcp_server_tools.py` script implements and registers the missing tools:

1. `get_hardware_capabilities`: Provides detailed information about system hardware
2. `throughput_benchmark`: Benchmarks model performance with various batch sizes
3. `quantize_model`: Simulates model quantization to reduce size

## Requirements

The following Python packages are required:

- requests
- psutil
- (optional) py-cpuinfo - for better SIMD detection
- (optional) torch - for CUDA detection

You can install the dependencies with:

```bash
pip install requests psutil py-cpuinfo torch
```

## Troubleshooting

If you encounter issues with the tools, try the following:

1. Verify the MCP server is running:
   ```bash
   curl http://localhost:8001/tools
   ```

2. Check for port conflicts:
   ```bash
   lsof -i :8001
   ```

3. Restart the MCP server manually:
   ```bash
   ps aux | grep direct_mcp_server.py
   kill <PID>
   python direct_mcp_server.py --port 8001
   ```

4. Check the logs for errors:
   ```bash
   tail -f mcp_server.log
   ```

## Future Improvements

Planned improvements to the MCP server tools:

1. Implement a monitoring dashboard for MCP server status
2. Add automated regression testing for all MCP tools
3. Improve error handling and recovery mechanisms
4. Add detailed documentation for each tool's API
5. Implement tool versioning for better compatibility management

## Contributing

Contributions to improve the MCP server tools are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

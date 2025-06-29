# IPFS Accelerate MCP Testing Guide

This guide covers the comprehensive MCP testing framework for IPFS Accelerate. It explains how to use the testing tools we've developed to verify MCP server integration and diagnose issues with different versions.

## Overview

The testing framework consists of:

1. **start_mcp_server.sh** - Main script that starts/restarts the MCP server and runs comprehensive tests
2. **test_mcp_server_integration.py** - Integration test script that checks MCP server connectivity and IPFS Accelerate integration
3. **test_mcp_virtual_fs.py** - Dedicated test script for virtual filesystem functionality

These tools work together to provide detailed diagnostics about the MCP server's functionality and its integration with the IPFS Accelerate module.

## Quick Start

To run a comprehensive test of your MCP server using the recommended method:

```bash
# First start the MCP server in one terminal
python unified_mcp_server.py

# Then in another terminal, run the tests
./run_mcp_tests.sh
```

Alternatively, you can use `start_mcp_server.sh` for a combined approach:

```bash
# Start the server and run all tests
./start_mcp_server.sh --test-level comprehensive

# If you want to test a specific port
./start_mcp_server.sh --port 8003 --test-level comprehensive
```

## Using run_mcp_tests.sh

This script runs the comprehensive MCP tool tests for ipfs_accelerate_py. It activates the virtual environment, ensures the server isn't already running, and executes the test suite.

### Command Line Options

Currently, this script doesn't take command-line arguments. It runs the full test suite by default.

## Using start_mcp_server.sh

This script is an alternative entry point for the testing framework. It can start/restart the MCP server, register tools, and run various levels of tests.

### Command Line Options

```
Usage: ./start_mcp_server.sh [OPTIONS]

Options:
  --port PORT                   Set MCP server port (default: 8002)
  --host HOST                   Set MCP server host (default: 0.0.0.0)
  --debug                       Enable debug mode
  --no-tests                    Skip running tests
  --test-level LEVEL            Set test level: basic, normal, or comprehensive (default: normal)
  --output-dir DIR              Directory for test output (default: test_results)
  --timeout SECONDS             Server startup timeout (default: 10)
  --no-restart                  Don't restart existing server
  --no-integration-test         Skip enhanced integration tests
  --no-version-check            Skip MCP version compatibility check
  --no-vfs-test                 Skip virtual filesystem tests
  --help                        Show this help message
```
### Test Levels

- **basic**: Only tests server connectivity and tool registration
- **normal**: Tests basic functionality plus IPFS operations and module integration
- **comprehensive**: Tests all functionality, including virtual filesystem and additional integration tests

### Output

The script generates:

1. Log file (`test_results/mcp_test_TIMESTAMP.log`)
2. JSON test report (`test_results/mcp_test_report_TIMESTAMP.json`)
3. Additional test results when running the comprehensive tests

## Using test_mcp_server_integration.py

This script verifies that the MCP server integrates correctly with the IPFS Accelerate module.

### Command Line Options

```
Usage: python test_mcp_server_integration.py [OPTIONS]

Options:
  --host HOST       MCP server host (default: localhost)
  --port PORT       MCP server port (default: 8002)
  --protocol PROTO  Protocol to use (http/https) (default: http)
  --output-dir DIR  Directory for test output (default: test_results)
```

### Tests Performed

1. **Server Connectivity**: Verifies the MCP server is running and responsive
2. **Tool Registration**: Checks if all expected IPFS Accelerate tools are registered
3. **Hardware Detection**: Tests hardware detection capabilities using IPFS Accelerate
4. **IPFS Functionality**: Tests basic IPFS operations like adding and retrieving files
5. **Virtual Filesystem**: Tests the IPFS virtual filesystem integration

### Example

```bash
python test_mcp_server_integration.py --host localhost --port 8002 --output-dir test_results
```

## Using test_mcp_virtual_fs.py

This script specifically tests the virtual filesystem functionality provided by the IPFS Accelerate module.

### Command Line Options

```
Usage: python test_mcp_virtual_fs.py [OPTIONS]

Options:
  --host HOST       MCP server host (default: localhost)
  --port PORT       MCP server port (default: 8002)
  --protocol PROTO  Protocol to use (http/https) (default: http)
  --output FILE     Output file for JSON results
```

### Tests Performed

1. **Tool Availability**: Checks if all required VFS tools are registered
2. **Directory Operations**: Tests creating directories in the virtual filesystem
3. **Read/Write Operations**: Tests writing to and reading from the virtual filesystem
4. **Listing and Stats**: Tests listing directory contents and getting file stats
5. **Copy/Move/Remove**: Tests copying, moving, and removing files
6. **Flush**: Tests flushing changes to IPFS

### Example

```bash
python test_mcp_virtual_fs.py --host localhost --port 8002 --output vfs_test_results.json
```

## Troubleshooting

### Common Issues

1. **Missing Tools**: If the tests report missing tools, ensure the IPFS Accelerate module is correctly installed and the tools are properly registered.

2. **Server Connection Failed**: Check if the server is running on the specified port. Use `lsof -i:PORT` to see if the port is in use.

3. **Integration Test Failures**: These typically indicate issues with the IPFS Accelerate module integration. Check the test report for specific details.

4. **Virtual Filesystem Test Failures**: These indicate issues with the IPFS virtual filesystem implementation. Verify the ipfs_vfs module is correctly installed.

### Interpreting Test Results

The JSON test reports provide detailed information about each test:

- **status**: "passed", "failed", or "skipped"
- **details**: Additional information about the test result
- **summary**: Overview of test results, including counts of passed/failed tests

Look for specific error messages in the test details to understand what went wrong.

## Advanced Usage

### Testing Different Versions

To test different versions of the MCP server or IPFS Accelerate module:

1. Install the desired version
2. Run the testing framework
3. Compare the test results with previous versions

### Continuous Integration

The testing framework can be integrated into CI/CD pipelines:

```bash
./start_mcp_server.sh --test-level comprehensive --no-restart
if [ $? -ne 0 ]; then
  echo "Tests failed!"
  exit 1
fi
```

## Extending the Framework

To add new tests:

1. Identify the component to test
2. Add a new test method to the appropriate test class
3. Update the `run_all_tests` method to include your new test
4. Update the documentation to reflect the new test

## Feedback and Contribution

If you encounter issues or have suggestions for improving the testing framework, please submit an issue or pull request to the IPFS Accelerate repository.

# Worker Reconnection System for Distributed Testing Framework

This directory contains the implementation of the Worker Reconnection System, a critical component of the Distributed Testing Framework that provides reliable networking between worker nodes and the coordinator server.

## Overview

The Worker Reconnection System enables robust communication between worker nodes and the coordinator in the face of network disruptions. It provides:

- Automatic reconnection with exponential backoff and jitter
- State synchronization after reconnection
- Task resumption from checkpoints
- Message delivery reliability
- Enhanced security and performance features

## Key Components

- **Core Reconnection**
  - `worker_reconnection.py`: Base reconnection system
  - `coordinator_websocket_server.py`: Coordinator WebSocket server

- **Enhanced Features**
  - `worker_reconnection_enhancements.py`: Enhanced version with security, compression, and metrics

- **Client Applications**
  - `run_coordinator_server.py`: Standalone coordinator server
  - `run_worker_client.py`: Basic worker client
  - `run_enhanced_worker_client.py`: Enhanced worker client

- **Testing Infrastructure**
  - `run_all_reconnection_tests.sh`: All-in-one test runner
  - `run_end_to_end_reconnection_test.py`: End-to-end testing
  - `run_stress_test.py`: Stress testing

## Documentation

- [Worker Reconnection Testing Guide](WORKER_RECONNECTION_TESTING_GUIDE.md): Comprehensive guide to the testing infrastructure
- [Worker Reconnection Implementation Summary](WORKER_RECONNECTION_IMPLEMENTATION_SUMMARY.md): Summary of implementation status and features
- [Task Execution Recursion Fix](TASK_EXECUTION_RECURSION_FIX.md): Detailed explanation of the task execution recursion fix

## Quick Start

### Running the Coordinator Server

```bash
./run_coordinator_server.py --host localhost --port 8765 --log-level INFO
```

### Running a Worker Client

```bash
./run_worker_client.py --worker-id test-worker --coordinator-host localhost --coordinator-port 8765
```

### Running an Enhanced Worker Client

```bash
./run_enhanced_worker_client.py --worker-id test-worker --coordinator-host localhost --coordinator-port 8765
```

### Running Tests

```bash
# Run all tests
./run_all_reconnection_tests.sh

# Run a quick test
./run_all_reconnection_tests.sh --quick

# Run end-to-end test
./run_end_to_end_reconnection_test.py --workers 5 --duration 300

# Run stress test
./run_stress_test.py --scenario thundering_herd
```

## Implementation Status

The Worker Reconnection System is feature-complete as of March 13, 2025. Recent updates include:

- ✅ **March 13, 2025**: Fixed the task execution recursion error that previously caused tasks to fail
- ✅ **March 13, 2025**: Added comprehensive metrics tracking for performance analysis
- ✅ **March 13, 2025**: Enhanced documentation and testing infrastructure
- ✅ **March 13, 2025**: Implemented proper handling for all message types including "welcome" and "registration_ack"
- ✅ **March 13, 2025**: Fixed URL formatting issue that caused duplicated path segments in worker URLs

With these recent fixes, all major known issues have been resolved. There is one remaining enhancement opportunity (improved network disruption simulation) documented in the [Worker Reconnection Testing Guide](WORKER_RECONNECTION_TESTING_GUIDE.md), but it doesn't affect the core functionality. The system is ready for integration with the broader Distributed Testing Framework.

## License

This project is part of the IPFS Accelerate Python Framework and is covered by its licensing terms.

## Contributors

- IPFS Accelerate Development Team (2025)
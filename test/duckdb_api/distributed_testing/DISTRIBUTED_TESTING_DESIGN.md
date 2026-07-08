# Distributed Testing Framework - Design Documentation

## Overview

The Distributed Testing Framework is a comprehensive system designed to execute tests across multiple worker nodes with different hardware capabilities. It enables efficient utilization of heterogeneous computing resources for testing AI models across various hardware platforms. The system uses a coordinator-worker architecture with a central coordinator managing test distribution and workers executing tests on different platforms.

## Architecture

### Core Components

1. **Coordinator**: Central server that manages test distribution, monitors workers, and collects results
2. **Workers**: Distributed nodes that execute tests on specific hardware platforms
3. **DuckDB Integration**: Efficient database system for storing test results
4. **Template Generator**: Creates tests dynamically from templates
5. **Cross-Platform Support**: Enables workers to run on different operating systems and environments
6. **Authentication**: Secure communication between coordinator and workers
7. **Resource Manager**: Dynamic allocation of resources based on workload

### Component Relationships

```
                    ┌───────────────────┐
                    │    Coordinator    │
                    └─────────┬─────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │         DuckDB Integration           │
        └──────────────────────────────────────┘
                              │
                              ▼
┌─────────────┬─────────────────────────┬─────────────┐
│             │                         │             │
▼             ▼                         ▼             ▼
┌─────────┐ ┌─────────┐             ┌─────────┐ ┌─────────┐
│ Worker 1│ │ Worker 2│     ...     │Worker N-1│ │Worker N │
└─────────┘ └─────────┘             └─────────┘ └─────────┘
```

### Communication Flow

1. **Test Submission**: Tests are submitted to the coordinator
2. **Task Distribution**: Coordinator assigns tests to appropriate workers
3. **Test Execution**: Workers execute tests and collect results
4. **Result Submission**: Results are sent back to the coordinator
5. **Result Storage**: Coordinator stores results in DuckDB database
6. **Reporting**: Coordinator generates reports from test results

## Component Details

### 1. DuckDB Integration (Phase 1 - Completed)

The DuckDB integration provides efficient storage and retrieval of test results with features including:

- **Connection Pooling**: Efficient management of database connections
- **Batch Processing**: Bulk insertion of test results for performance
- **Transaction Support**: ACID-compliant operations with rollback capability
- **Schema Validation**: Ensures data integrity with field validation
- **Result Querying**: Flexible SQL-based result analysis

Key components:
- `DuckDBResultProcessor`: Core class for database interactions
- `WorkerDuckDBIntegration`: Worker-side result management
- `CoordinatorDuckDBIntegration`: Coordinator-side result processing

### 2. Template-Based Test Generator (Phase 2 - Completed)

The template-based test generator creates tests dynamically from templates:

- **Model Family Detection**: Automatically identifies model types
- **Template Database**: Stores and retrieves templates
- **Dependency Tracking**: Manages execution order dependencies
- **Resource Estimation**: Calculates hardware requirements for tests
- **Priority Calculation**: Determines execution priority

Key components:
- `TestGeneratorIntegration`: Main class for test generation
- `TemplateDatabase`: Manages template storage and retrieval
- `DependencyTracker`: Handles test execution dependencies

### 3. Cross-Platform Worker Support (Phase 4 - Completed)

The Cross-Platform Worker Support module enables workers to run on different operating systems and container environments, providing a unified interface for platform-specific functionality while maximizing each platform's capabilities.

#### Core Features

- **Platform Detection**: Automatically identifies the operating system or container environment
- **Hardware Detection**: Discovers available CPU, memory, GPU, and disk resources with platform-specific methods
- **Deployment Script Generation**: Creates platform-specific deployment scripts (shell scripts, batch files, docker-compose)
- **Startup Script Generation**: Produces scripts with proper environment configuration and logging
- **Path Conversion**: Ensures correct path formats across different operating systems
- **Dependency Management**: Handles installation of dependencies with platform-specific approaches

#### Key Components

- **CrossPlatformWorkerSupport**: Main class that provides a unified interface for all platform-specific operations
- **PlatformHandler**: Abstract base class defining the interface for platform-specific handlers
- **Platform-Specific Handlers**:
  - **LinuxPlatformHandler**: Linux-specific implementation with shell script generation and Linux hardware detection
  - **WindowsPlatformHandler**: Windows-specific implementation with batch file generation and Windows-specific hardware detection
  - **MacOSPlatformHandler**: macOS-specific implementation with Apple Silicon detection and macOS-specific hardware detection
  - **ContainerPlatformHandler**: Container-specific implementation with Docker/Kubernetes support and resource limit detection

#### Hardware Detection Details

The hardware detection system uses platform-specific methods to gather detailed information:

**CPU Detection**:
- **Linux**: Uses `/proc/cpuinfo` to get CPU model and core count
- **Windows**: Uses WMI or `wmic` fallback to get CPU information
- **macOS**: Uses `sysctl` to get CPU details with Apple Silicon detection
- **Container**: Detects CPU limits from container environment

**Memory Detection**:
- **Linux**: Reads from `/proc/meminfo` to get total memory
- **Windows**: Uses `psutil` for memory information
- **macOS**: Uses `sysctl hw.memsize` to get total memory
- **Container**: Detects memory limits from cgroups

**GPU Detection**:
- **NVIDIA GPUs**: Uses `nvidia-smi` across platforms for detailed GPU information
- **AMD GPUs**: Uses `rocm-smi` on Linux/Windows for AMD GPU detection
- **Intel/Integrated GPUs**: Platform-specific detection methods
- **Apple GPUs**: Uses `system_profiler` on macOS for Metal-capable GPU detection

**Disk Detection**:
- All platforms use `shutil.disk_usage()` for consistent disk information

#### Deployment Script Generation

The module generates platform-specific deployment scripts:

**Linux Deployment**:
- Shell scripts with proper executable permissions
- Package manager detection (apt, yum, dnf)
- Python dependency installation
- Proper environment variable configuration
- Error handling and logging setup

**Windows Deployment**:
- Batch files with Windows-specific syntax
- PowerShell integration for advanced scenarios
- Path handling with backslashes
- Windows-specific environment variable setup

**macOS Deployment**:
- Shell scripts with macOS-specific configurations
- Homebrew integration for package management
- Apple Silicon compatibility handling
- macOS-specific path resolution

**Container Deployment**:
- `docker-compose.yml` for multi-container orchestration
- Dockerfile generation with proper base image selection
- Volume mounting for logs and persistent data
- Container-specific environment configuration
- Resource limit configuration

#### Startup Script Generation

The module creates platform-specific worker startup scripts:

**Linux/macOS Startup**:
- Proper shebang and executable permissions
- Environment variable configuration
- Log directory creation
- Background process execution with `nohup`
- PID file generation for process management
- Error redirection and log file setup

**Windows Startup**:
- Batch file format with Windows command syntax
- Windows environment variable setup
- Background execution with `start /b`
- Windows-specific error handling
- Log directory creation with Windows path syntax

**Container Startup**:
- Container-specific entrypoint
- Container environment variable configuration
- Proper log handling for containerized processes
- Resource limit adherence

#### Path Conversion

The path conversion system ensures consistent path formatting across platforms:

- **Linux/macOS**: Forward slash path separators (`/`)
- **Windows**: Backslash path separators (`\`)
- **Relative Path Handling**: Proper conversion of relative paths
- **Drive Letter Support**: Handles Windows drive letters appropriately
- **UNC Path Support**: Handles Windows UNC paths

#### Implementation Status

The Cross-Platform Worker Support module has been fully implemented with comprehensive testing and documentation. Key milestones include:

- Initial implementation: March 18, 2025
- Unit test implementation: March 19, 2025
- Example script creation: March 19, 2025
- Documentation completion: March 20, 2025
- Integration with worker system: March 20, 2025
- Final testing and validation: March 20, 2025

#### Integration with Other Components

The Cross-Platform Worker Support module integrates with:

- **Worker Node System**: Provides platform-specific implementations for worker nodes
- **Hardware Detection**: Supplies detailed hardware information for task matching
- **Resource Monitoring**: Enables resource tracking across diverse platforms
- **Deployment System**: Facilitates worker node setup across various environments
- **CI/CD Pipeline**: Supports testing across different operating systems

#### Code Examples

**Platform Detection**:
```python
support = CrossPlatformWorkerSupport()
platform_info = support.get_platform_info()
print(f"Platform: {platform_info['platform']}")
print(f"System: {platform_info['system']}")
print(f"Architecture: {platform_info['architecture']}")
```

**Hardware Detection**:
```python
support = CrossPlatformWorkerSupport()
hardware_info = support.detect_hardware()
print(f"CPU: {hardware_info['cpu']['model']} with {hardware_info['cpu']['cores']} cores")
print(f"Memory: {hardware_info['memory']['total_gb']} GB")
print(f"GPUs: {hardware_info['gpu']['count']}")
for i, device in enumerate(hardware_info['gpu']['devices']):
    print(f"  GPU {i}: {device['name']} ({device['type']})")
```

**Deployment Script Generation**:
```python
support = CrossPlatformWorkerSupport()
config = {
    "coordinator_url": "http://coordinator.example.com:8080",
    "api_key": "your_api_key",
    "worker_id": "worker_123"
}
script_path = support.create_deployment_script(config, "deploy_worker")
```

**Startup Script Generation**:
```python
support = CrossPlatformWorkerSupport()
startup_script = support.get_startup_script(
    coordinator_url="http://coordinator.example.com:8080",
    api_key="your_api_key",
    worker_id="worker_abc123"
)
```

**Path Conversion**:
```python
support = CrossPlatformWorkerSupport()
linux_path = "/home/user/data/models.json"
windows_path = "C:\\Users\\user\\Documents\\models.json"
platform_path1 = support.convert_path_for_platform(linux_path)
platform_path2 = support.convert_path_for_platform(windows_path)
```

#### Testing

The module includes comprehensive testing:

- **Platform Detection Tests**: Verify correct platform identification
- **Hardware Detection Tests**: Validate hardware detection across platforms
- **Deployment Script Tests**: Check script generation for all platforms
- **Startup Script Tests**: Verify startup script generation
- **Path Conversion Tests**: Ensure paths are correctly converted
- **Dependency Management Tests**: Validate dependency installation
- **Integration Tests**: Verify integration with worker management system
- **Error Handling Tests**: Ensure proper handling of error conditions
- **Mocked Platform Tests**: Test cross-platform functionality in a single environment

#### Documentation

Detailed documentation is provided in `CROSS_PLATFORM_WORKER_README.md` including:

- Complete API reference
- Usage examples for all functionality
- Platform-specific considerations
- Integration guidelines
- Best practices for cross-platform deployment

Supported platforms:
- Linux (Ubuntu, CentOS, Debian, etc.)
- Windows
- macOS
- Containers (Docker, Kubernetes)

### 4. JWT Authentication (Phase 3 - Deferred)

Enhanced JWT authentication is planned to provide secure communication between the coordinator and workers:

- **Role-Based Access Control**: Specific permissions based on roles
- **Token Refresh**: Mechanism for long-running workers
- **Token Revocation**: Security incident response capability
- **Fine-Grained Permissions**: Detailed operation-level access control
- **Secure Storage**: Proper token storage on worker nodes

### 5. CI/CD Pipeline Integration (Phase 5 - Completed)

CI/CD integration automates distributed testing during development workflows, with comprehensive artifact management and test reporting:

- **CI Provider Interface**: Standardized interface for all CI providers (GitHub, GitLab, Jenkins, etc.)
- **Artifact Handling**: Standardized artifact uploading, management, and URL retrieval
- **Status Reporting**: Automated test status reporting to CI systems
- **PR/MR Comments**: Automatic commenting on Pull/Merge Requests with test results
- **Worker Provisioning**: Dynamic provisioning of worker nodes for distributed testing
- **Result Collection**: Centralized collection and storage of test results
- **Report Generation**: Comprehensive test reports with visualization

#### Key Components

- **CIProviderInterface**: Abstract base class defining standard operations for all CI providers
- **Provider Implementations**: Concrete implementations for GitHub, GitLab, Jenkins, Azure DevOps, CircleCI, TeamCity, Travis CI, and Bitbucket
- **Artifact Management**: System for storing, retrieving, and managing test artifacts
- **TestRunResult**: Standardized representation of test results across different CI systems
- **TestResultFormatter**: Formatting test results for different output formats (Markdown, HTML, JSON)
- **TestResultReporter**: Centralized reporting of test results to CI systems
- **CIProviderFactory**: Factory pattern for creating CI provider instances

#### Artifact URL Retrieval System

A key enhancement is the standardized artifact URL retrieval system that works across all CI providers:

- **Universal Interface**: Consistent `get_artifact_url` method across all providers
- **URL Caching**: Minimizes API calls by caching URLs
- **Fallback Mechanisms**: Multiple URL resolution strategies when primary methods fail
- **Provider-Specific Logic**: Each provider implements URL retrieval according to its own artifact storage mechanisms
- **Integration with Test Reports**: Artifact URLs can be included in test reports and notifications

#### Implementation Details

The CI/CD integration has been implemented with:

- Comprehensive testing of all CI providers
- Standardized error handling and logging
- Consistent artifact management across providers
- Transparent simulation mode for testing
- Integration with the DuckDB result storage system

### 6. Dynamic Resource Management (Phase 6 - Planned)

Dynamic resource management will optimize resource allocation based on workload:

- **Resource Tracking**: Fine-grained tracking of available resources
- **Adaptive Scaling**: Dynamic scaling based on workload patterns
- **Worker Reassessment**: Periodic reassessment of worker capabilities
- **Cloud Integration**: Support for ephemeral workers on cloud platforms
- **Reservation Tracking**: Resource reservation and release tracking

### 7. Auto Recovery System - High Availability (Phase 7 - Completed)

The Auto Recovery System provides high availability clustering for the coordinator component, ensuring continuous operation even when individual nodes fail.

#### Core Features

- **Coordinator Redundancy**: Multiple coordinator instances running in parallel with automatic failover
- **Leader Election**: Raft-inspired consensus algorithm for reliable leader selection
- **State Synchronization**: Efficient state replication between coordinator nodes
- **Health Monitoring**: Comprehensive system health tracking and issue detection
- **Self-Healing**: Automatic recovery from common failure conditions
- **WebNN/WebGPU Awareness**: Hardware capability detection for web platform acceleration
- **Visualization**: Real-time visualization of cluster state and health metrics

#### Architecture

The Auto Recovery System implements a state machine-based approach with the following states:

- **Leader**: Active coordinator that handles client requests and distributes tasks
- **Follower**: Standby coordinator that replicates state from the leader
- **Candidate**: Coordinator attempting to become leader during an election
- **Offline**: Coordinator that is currently unreachable

State transitions are triggered by:
- **Heartbeat Timeout**: Follower transitions to candidate when leader heartbeat is missed
- **Election Success**: Candidate transitions to leader when receiving majority votes
- **Higher Term Discovery**: Any node transitions to follower when a higher term is discovered
- **Leader Heartbeat**: Candidate or follower resets election timeout when leader heartbeat received

#### State Replication

State replication is performed through a log-based approach:

- **Log Entries**: Operations are recorded as log entries with term numbers
- **Append-Only Log**: New entries are appended to the log in order
- **Consistency Check**: Each entry includes the term number of the previous entry
- **Commit Index**: Tracks the highest log entry known to be committed
- **State Snapshots**: Periodic snapshots reduce recovery time after failures

#### Key Components

- **AutoRecoverySystem**: Extends the base AutoRecovery class with enhanced features including:
  - Health monitoring with self-healing capabilities
  - WebNN/WebGPU capability detection
  - State visualization and metric reporting
  - Message integrity verification via cryptographic hashing

- **Health Monitoring System**: 
  - Tracks CPU, memory, disk, and network metrics
  - Identifies resource constraints and performance issues
  - Implements recovery actions for common issues
  - Provides historical trend analysis

- **WebNN/WebGPU Capability Detection**:
  - Detects browser support for WebNN and WebGPU
  - Maps hardware capabilities to browser features
  - Tracks browser version-specific feature support
  - Enables hardware-aware task assignment

- **Visualization Engine**:
  - Generates cluster state visualizations
  - Creates health metrics charts and graphs
  - Visualizes leader transitions and state changes
  - Supports both HTML and text-based output formats

#### Advanced Features

The Auto Recovery System includes several advanced features:

1. **Message Integrity Verification**:
   - Hash-based verification of all inter-coordinator messages
   - Prevents message tampering and corruption
   - Ensures data integrity during coordination

2. **Adaptive Election Timeouts**:
   - Randomized timeouts to prevent election conflicts
   - Range-based timeout selection for stability
   - Configurable bounds for environment-specific tuning

3. **Self-Healing Capabilities**:
   - Memory reclamation through garbage collection
   - Disk space management with log rotation
   - Leader step-down under resource constraints
   - Error rate monitoring with threshold alerts

4. **Progressive Recovery Strategies**:
   - Graduated approach to recovery actions
   - Least-disruptive actions attempted first
   - Escalation path for persistent issues
   - Resource-aware decision making

#### Usage Examples

```python
from duckdb_api.distributed_testing.auto_recovery import AutoRecoverySystem

# Initialize the auto recovery system
auto_recovery = AutoRecoverySystem(
    coordinator_id="coordinator-1",
    coordinator_addresses=["localhost:8081", "localhost:8082"],
    db_path="./benchmark_db.duckdb",
    visualization_path="./visualizations"
)

# Start the system
auto_recovery.start()

# Register callbacks for leader transitions
def on_become_leader():
    print("This node is now the leader!")
    
def on_leader_changed(old_leader, new_leader):
    print(f"Leader changed from {old_leader} to {new_leader}")
    
auto_recovery.register_become_leader_callback(on_become_leader)
auto_recovery.register_leader_changed_callback(on_leader_changed)

# Get current health metrics
health_metrics = auto_recovery.get_health_metrics()
print(f"CPU Usage: {health_metrics['cpu_usage']}%")
print(f"Memory Usage: {health_metrics['memory_usage']}%")

# Get browser capabilities
web_capabilities = auto_recovery.get_web_capabilities()
print(f"WebNN Support: {web_capabilities['webnn_supported']}")
print(f"WebGPU Support: {web_capabilities['webgpu_supported']}")

# Create visualization
auto_recovery.create_visualization(output_dir="./visualizations")

# Stop when done
auto_recovery.stop()
```

#### Integration with Existing Components

The Auto Recovery System integrates with other components of the Distributed Testing Framework:

1. **Coordinator Integration**:
   - Registers with coordinator to receive status updates
   - Provides leader election and failover capabilities
   - Enhances coordinator with health monitoring

2. **DuckDB Integration**:
   - Uses DuckDB for state persistence
   - Synchronizes database state between coordinators
   - Maintains consistent view of test results

3. **Task Scheduler Integration**:
   - Provides WebNN/WebGPU capability information to scheduler
   - Enables hardware-aware scheduling decisions
   - Ensures task continuity during failovers

4. **Monitoring Dashboard Integration**:
   - Publishes health metrics to dashboard
   - Provides visualization components
   - Enables real-time monitoring of cluster state

The system replicates state between coordinators using a log-based approach:

- **Log Entries**: Append-only log of operations (task creation, worker registration, etc.)
- **Term Numbers**: Monotonically increasing election term numbers
- **Commit Index**: Index up to which log entries are considered committed
- **State Snapshots**: Periodic full state snapshots for efficient synchronization

#### Health Monitoring

The system includes comprehensive health monitoring:

- **CPU Usage**: Tracks CPU utilization with critical thresholds
- **Memory Usage**: Monitors memory consumption with automatic memory reclamation
- **Disk Usage**: Tracks disk space with cleanup procedures for critical levels
- **Network Latency**: Measures communication latency between coordinators
- **Error Rate**: Tracks error frequency and trends

#### WebNN/WebGPU Integration

The Auto Recovery System integrates with WebNN/WebGPU capabilities:

- **Browser Detection**: Identifies available browsers and their capabilities
- **Feature Detection**: Determines available WebNN/WebGPU features
- **Capability Visualization**: Visualizes web platform acceleration capabilities
- **Hardware-Aware Routing**: Routes tasks to workers with optimal capabilities

#### Visualization

The system generates comprehensive visualizations:

- **Cluster Status**: Network graph of coordinator nodes showing status
- **Health Metrics**: Charts of CPU, memory, disk, and error metrics
- **Leader Transitions**: Timeline of leadership changes
- **Web Capabilities**: Browser support for WebNN/WebGPU features

#### Implementation Details

The Auto Recovery System includes:

- **Base Class**: `AutoRecovery` implementing core consensus protocol
- **Enhanced Class**: `AutoRecoverySystem` with extended capabilities
- **Health Monitoring Thread**: Background thread monitoring system health
- **Visualization Thread**: Background thread generating visualizations
- **WebNN/WebGPU Detection**: Integration with hardware detection
- **Secure Messaging**: Hash-based message integrity verification

#### Code Examples

**Starting a high availability cluster:**
```python
# Create Auto Recovery System instance
auto_recovery = AutoRecoverySystem(
    coordinator_id="coordinator-1",
    coordinator_addresses=["localhost:8081", "localhost:8082"],
    db_path="./benchmark_db.duckdb",
    auto_leader_election=True,
    visualization_path="./visualizations"
)

# Start the system
auto_recovery.start()

# Check if this node is the leader
if auto_recovery.is_leader():
    # Perform leader-only operations
    pass
```

**Handling leader transitions:**
```python
# Callback when this coordinator becomes leader
auto_recovery.on_become_leader(lambda: print("I am now the leader"))

# Callback when leader changes
auto_recovery.on_leader_changed(lambda old_leader, new_leader: 
    print(f"Leader changed from {old_leader} to {new_leader}"))
```

**Health monitoring and visualization:**
```python
# Generate health metrics visualization
auto_recovery._generate_health_metrics_visualization()

# Generate cluster status visualization
auto_recovery._generate_cluster_status_visualization()
```

## Implementation Status

| Phase | Component | Status | Completion Date |
|-------|-----------|--------|-----------------|
| 1 | DuckDB Integration | ✅ Complete | March 15, 2025 |
| 2 | Template-Based Test Generator | ✅ Complete | March 16, 2025 |
| 3 | JWT Authentication | ⏸️ Deferred | - |
| 4 | Cross-Platform Worker Support | ✅ Complete | March 20, 2025 |
| 5 | CI/CD Pipeline Integration | ✅ Complete | March 27, 2025 |
| 5.1 | Artifact URL Retrieval | ✅ Complete | March 16, 2025 |
| 6 | Dynamic Resource Management | 📅 Planned | Target: April 5, 2025 |
| 7 | Auto Recovery System (High Availability) | ✅ Complete | March 16, 2025 |

## Usage Examples

### DuckDB Integration

```python
from duckdb_api.distributed_testing.worker_duckdb_integration import WorkerDuckDBIntegration

# Initialize worker-side integration
worker_db = WorkerDuckDBIntegration(db_path="./results.duckdb")

# Store test result
result = {
    "test_id": "test_123",
    "model_name": "bert-base-uncased",
    "hardware_type": "cuda",
    "success": True,
    "execution_time": 10.5,
    "memory_usage": 1.2
}
worker_db.store_result(result)

# Store batch of results
results = [result1, result2, result3]
worker_db.store_batch_results(results)
```

### Template Generator

```python
from duckdb_api.distributed_testing.test_generator_integration import TestGeneratorIntegration

# Initialize test generator
generator = TestGeneratorIntegration(template_db_path="./templates.duckdb", 
                                    coordinator_url="http://coordinator:8080")

# Generate and submit tests
results = generator.generate_and_submit_tests(
    model_name="bert-base-uncased",
    hardware_types=["cpu", "cuda", "rocm"],
    batch_sizes=[1, 4, 16]
)
```

### Cross-Platform Worker Support

```python
from duckdb_api.distributed_testing.cross_platform_worker_support import CrossPlatformWorkerSupport

# Initialize for the current platform
support = CrossPlatformWorkerSupport()

# Detect hardware capabilities
hardware_info = support.detect_hardware()
print(f"Platform: {hardware_info['platform']}")
print(f"CPU: {hardware_info['cpu']['cores']} cores")
print(f"Memory: {hardware_info['memory']['total_gb']} GB")

# Create deployment script
config = {
    "coordinator_url": "http://coordinator.example.com:8080",
    "api_key": "your_api_key",
    "worker_id": "worker_123"
}
script_path = support.create_deployment_script(config, "deploy_worker")

# Generate startup script
startup_script = support.get_startup_script(
    coordinator_url="http://coordinator.example.com:8080",
    api_key="your_api_key",
    worker_id="worker_abc123"
)
```

### CI/CD Pipeline Integration with Artifact URL Retrieval

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
ci_config = {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "your-username/your-repo"
}
ci_provider = await CIProviderFactory.create_provider("github", ci_config)

# Upload an artifact
test_run_id = "test-123"
artifact_result = await ci_provider.upload_artifact(
    test_run_id=test_run_id,
    artifact_path="./test_results.json",
    artifact_name="test_results.json"
)

# Retrieve the artifact URL for inclusion in reports or notifications
if artifact_result:
    url = await ci_provider.get_artifact_url(
        test_run_id=test_run_id,
        artifact_name="test_results.json"
    )
    if url:
        print(f"Artifact available at: {url}")
        # URL can be used in reports, notifications, or dashboards
        
        # Example: Include artifact URL in test report
        report = f"""
        # Test Run Report
        
        Test run {test_run_id} completed successfully.
        
        ## Artifacts
        
        - [Test Results JSON]({url})
        """
```

### Auto Recovery System - High Availability Cluster

```python
from duckdb_api.distributed_testing.auto_recovery import AutoRecoverySystem

# Initialize Auto Recovery System
auto_recovery = AutoRecoverySystem(
    coordinator_id="coordinator-1",
    coordinator_addresses=["localhost:8081", "localhost:8082"],
    db_path="./benchmark_db.duckdb",
    auto_leader_election=True,
    visualization_path="./visualizations"
)

# Register callbacks for leader transitions
def on_become_leader():
    print(f"Coordinator {auto_recovery.coordinator_id} became the leader")
    # Perform leader-specific initialization
    # For example, start primary load balancer

def on_leader_changed(old_leader_id, new_leader_id):
    print(f"Leader changed from {old_leader_id} to {new_leader_id}")
    # Update local state based on leader change
    # For example, redirect clients to new leader

auto_recovery.auto_recovery.on_become_leader(on_become_leader)
auto_recovery.auto_recovery.on_leader_changed(on_leader_changed)

# Start the system
auto_recovery.start()

try:
    # Register another coordinator
    auto_recovery.register_coordinator(
        coordinator_id="coordinator-2",
        address="localhost",
        port=8082,
        capabilities={"hardware_types": ["cpu", "cuda"]}
    )
    
    # Check if we're the leader
    if auto_recovery.is_leader():
        print("This coordinator is the leader")
        
        # Perform leader-only operations
        # For example, start distributing tasks
    else:
        print(f"Current leader is: {auto_recovery.get_leader_id()}")
        
        # Sync state with leader
        auto_recovery.sync_with_leader()
        
    # Get system status
    status = auto_recovery.get_status()
    print(f"Cluster size: {status['coordinators_count']} coordinators")
    print(f"Active coordinators: {status['active_coordinators']}")
    
    # Generate visualizations
    auto_recovery._generate_cluster_status_visualization()
    auto_recovery._generate_health_metrics_visualization()
    
    # Run until interrupted
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    # Gracefully shut down
    auto_recovery.stop()
    print("Auto Recovery System stopped")
```

### Running High Availability Cluster Example

```bash
# Start a 3-node high availability cluster with fault injection
./run_high_availability_cluster.sh --nodes 3 --fault-injection --runtime 300

# Start a 5-node cluster on higher ports
./run_high_availability_cluster.sh --nodes 5 --base-port 9000 --runtime 600
```

## Testing

Each component includes comprehensive testing:

1. **Unit Tests**: Verify individual component functionality
2. **Integration Tests**: Validate interaction between components
3. **End-to-End Tests**: Test complete workflows
4. **Fault Tolerance Tests**: Verify resilience during failures
5. **Load Tests**: Evaluate performance under high concurrency
6. **Cross-Platform Tests**: Verify functionality across environments

## Documentation

Detailed documentation is available for each component:

- **DISTRIBUTED_TESTING_INTEGRATION_PR.md**: Overall integration roadmap
- **CROSS_PLATFORM_WORKER_README.md**: Cross-platform support documentation
- **PHASE4_COMPLETION_SUMMARY.md**: Summary of Phase 4 completion
- **templates/*.md**: Documentation for template-based generation

## Future Enhancements

1. **Predictive Scheduling**: Use performance history to predict optimal test scheduling
2. **Advanced Load Balancing**: Intelligent load distribution based on worker capabilities
3. **Result Visualization**: Interactive dashboard for result analysis
4. **Test Prioritization**: Risk-based test prioritization for critical components
5. **Auto-Scaling**: Integration with cloud platforms for automatic worker scaling

## Conclusion

The Distributed Testing Framework provides a comprehensive solution for testing AI models across heterogeneous computing resources. With the completion of core components including DuckDB integration, template-based test generation, and cross-platform support, the framework is approaching a fully integrated state. The upcoming phases will focus on CI/CD integration and dynamic resource management to further enhance the system's capabilities.

The modular architecture ensures the system is maintainable and extensible, allowing for future enhancements. The focus on cross-platform compatibility enables the framework to utilize a wide range of computing resources efficiently, making it a versatile tool for comprehensive AI model testing.
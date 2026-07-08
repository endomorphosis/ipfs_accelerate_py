# Error Visualization Guide

This guide documents the Error Visualization system for the Distributed Testing Framework, including installation, configuration, and usage instructions.

## Overview

The Error Visualization system provides comprehensive error monitoring, analysis, and visualization capabilities for the Distributed Testing Framework. It integrates with the monitoring dashboard to provide insights into error patterns, worker status, and hardware issues, enabling rapid detection and diagnosis of problems in a distributed testing environment.

The system includes:

1. **Real-Time Error Monitoring**: WebSocket-based real-time error notifications with sound and visual alerts
2. **Error Pattern Detection**: Identification of recurring error patterns and trends for effective troubleshooting
3. **Worker Error Analysis**: Analysis of errors by worker node with status tracking and health monitoring
4. **Hardware Error Analysis**: Analysis of hardware-related errors with detailed context information
5. **Error Context Collection**: Collection of system and hardware context for comprehensive diagnostics
6. **Interactive Dashboard**: A web-based interface with real-time updates, filtering, and visualization
7. **Severity-Based Notifications**: Customizable audio and visual notifications based on error severity
8. **Accessibility Features**: ARIA attributes, keyboard navigation, and high-contrast support for inclusive usage

## Installation and Dependencies

The Error Visualization system requires the following dependencies:

```bash
# Core dependencies
numpy>=1.20.0      # For sound generation and data processing
scipy>=1.10.0      # For sound generation
aiohttp>=3.8.0     # For API and WebSocket communication
duckdb>=0.9.0      # For error data storage
websockets>=10.0   # For WebSocket communication

# Optional dependencies for enhanced visualization
plotly>=5.13.0     # For interactive visualizations (optional)
pandas>=1.5.0      # For data processing and analysis (optional)

# Testing dependencies
unittest-xml-reporting>=3.2.0  # For test reporting (optional)
html-testRunner>=1.2.1         # For HTML test reports (optional)
```

These dependencies are included in the project's `requirements.test.txt` file. To install them:

```bash
pip install -r requirements.test.txt
```

The system includes graceful fallbacks for optional dependencies, so it will still function without plotly or pandas, but with reduced visualization capabilities.

## Error Visualization Dashboard

The Error Visualization Dashboard is accessible through the monitoring dashboard at:

```
http://localhost:8080/error-visualization
```

### Dashboard Features

The dashboard includes:

- **Error Summary**: Overview of error counts, categories, and critical issues
- **Error Distribution**: Interactive visualization of errors by category
- **Error Patterns**: Analysis of recurring error patterns and trends
- **Worker Errors**: Analysis of errors by worker node with status tracking
- **Hardware Errors**: Analysis of hardware-related errors with context information
- **Real-Time Updates**: WebSocket-based live error notifications with customizable alerts
- **Auto-Refresh**: Background refresh of error data without page reload
- **Notification Controls**: Volume adjustment, mute option, and auto-refresh toggle
- **Visual Indicators**: Severity-based highlighting, animations, and badges for new errors
- **Accessibility Support**: ARIA attributes, keyboard navigation, and high-contrast mode

### Time Range Selection

You can select different time ranges for error visualization:
- Last 1 hour
- Last 6 hours
- Last 24 hours
- Last 7 days

## Real-Time Error Monitoring

The Error Visualization system features WebSocket-based real-time error monitoring. This capability allows you to see errors as they occur, without needing to refresh the page, enabling immediate responses to critical issues.

### Key Features of Real-Time Monitoring

- **Live Error Updates**: Errors appear in the dashboard in real-time with automatic insertion
- **Desktop Notifications**: Browser notifications for new errors with severity indicators (requires permission)
- **Sound Alerts**: Audio notifications with distinct sounds based on error severity
- **Volume Controls**: Adjustable notification volume with visual slider and mute toggle
- **Error Severity Detection**: Automatic categorization of errors by severity level with appropriate responses
- **Visual Highlighting**: Severity-based color highlighting for new errors (red for critical, yellow for warnings)
- **Animation Effects**: Pulsing animation for critical errors to increase visibility
- **Error Counters**: Badge showing count of new errors since last reset or page load
- **Page Title Updates**: Dynamic error counts in browser tab titles for visibility when tabbed away
- **Automatic Reconnection**: WebSocket automatically reconnects if connection is lost
- **Auto-Refresh Option**: Background data refresh at regular intervals without page reload

### Dashboard Controls

The error visualization dashboard provides several controls for customizing the experience:

#### Notification Controls

- **Volume Slider**: Adjust the volume of sound notifications with immediate feedback
- **Mute Toggle**: Quickly enable/disable all sound notifications with icon feedback
- **Clear Button**: Reset the new error counter and remove new error badges
- **Auto-Refresh Toggle**: Enable/disable automatic background data refresh
- **Time Range Selector**: Filter errors by time range (1 hour, 6 hours, 24 hours, 7 days)
- **Theme Selector**: Switch between light and dark themes for visual comfort

All controls include:
- Visual feedback on state changes
- Keyboard accessibility
- Screen reader support via ARIA attributes
- Preference persistence via localStorage

#### Notification Features

##### Sound Notifications

The system uses different sound files based on error severity to create an auditory hierarchy of alerts that intuitively communicate the importance of different errors. The carefully designed sonic characteristics make each severity level immediately distinguishable even in noisy environments.

1. **error-system-critical.mp3**: Highest-priority alert sound for system-level critical errors (1.0s duration)
   * Used for coordinator failure, database corruption, and security breaches
   * Rising frequency pattern (880Hz → 1046.5Hz → 1318.5Hz) with increasing urgency
   * Features accelerating pulse rate (4Hz to 16Hz) and harmonic richness
   * Three distinct segments that crossfade for a progressive alert
   * Subtle harmonic at 1760Hz (A6) adds richness and distinctiveness
   * Designed for immediate attention in critical system failures
   * Acoustically optimized to stand out in busy environments
   * Created with psychoacoustic principles for maximum alerting effectiveness

2. **error-critical.mp3**: High-priority alert sound for critical errors (0.7s duration)
   * Higher frequency (880Hz/440Hz) with attention-grabbing tonal pattern
   * Pulsing effect (8Hz modulation) creates urgency without being jarring
   * Amplitude modulation provides a distinctive "alert" quality
   * Automatically used for hardware availability errors, resource allocation errors, and worker crashes

3. **error-warning.mp3**: Medium-priority alert sound for warning-level errors (0.5s duration)
   * Medium frequency (660Hz/330Hz) with moderate decay
   * Dual-tone approach creates more richness than simple beeps
   * Distinctive enough to notice without being startling
   * Automatically used for network errors, resource cleanup issues, and worker timeouts

4. **error-info.mp3**: Low-priority notification sound for informational errors (0.3s duration)
   * Lower frequency (523Hz) with quick decay
   * Short, subtle tone that's non-intrusive
   * Designed to be noticeable but not distracting
   * Automatically used for test execution errors and other non-critical issues
   
5. **error-notification.mp3**: Default notification sound (used as fallback)
   * Copy of error-critical.mp3 by default
   * Used when specific severity sounds fail to load
   * Provides graceful degradation if other sound files are missing

All sound files are automatically generated and placed in the `/static/sounds/` directory. Each sound is designed with specific characteristics:
- Custom duration based on importance (0.3-1.0 seconds) to prevent notification fatigue
- Distinct tonal patterns for easier recognition by psychoacoustic principles
- Frequency differentiation for severity levels (higher = more urgent)
- Temporal design features (rising patterns, pulse rates) to communicate urgency
- Non-jarring waveforms suitable for workplace environments
- Cross-browser compatibility with standard MP3 format

##### Visual Indicators

The system provides various visual indicators for errors:

- **Critical errors**: Red highlighting with pulsing animation for sustained visibility
- **Warning errors**: Yellow highlighting to indicate medium priority issues
- **Info errors**: Blue highlighting for non-critical information
- **New error badge**: Numeric counter showing new errors since last reset
- **Page title updates**: Shows error count and critical error count in browser tab title
- **Animation effects**: Severity-based animations draw attention to new errors
- **Color-coded categories**: Error categories use consistent color schemes
- **Highlight transitions**: Smooth transitions help identify new entries

##### User Control Elements

The Error Visualization dashboard provides intuitive controls for managing notifications:

- **Volume Controls**:
  - **Volume Slider**: Precise control over notification sound volume (0-100%)
  - **Real-time Feedback**: Volume changes are immediately effective without reloading
  - **Visual Indicators**: Dynamic icon changes reflect current volume level
    * 🔊 High volume (>70%)
    * 🔉 Medium volume (30-70%)
    * 🔈 Low volume (<30%)
    * 🔇 Muted (0%)
  - **Persistence**: Volume settings are saved to localStorage and restored between sessions
  - **Accessibility**: Fully keyboard accessible with ARIA attributes for screen readers

- **Mute Toggle**:
  - **One-Click Muting**: Quickly enable/disable all sound notifications
  - **State Memory**: Remembers previous volume when unmuted
  - **Visual Indicator**: Clear icon change (🔊 → 🔇) shows muted state
  - **Intelligent Behavior**: Volume slider automatically updates when muted
  - **Keyboard Shortcut**: Accessible via keyboard navigation

- **New Error Management**:
  - **Clear Button**: Resets the new error counter and removes new error badges
  - **Counter Badge**: Shows number of new errors since last reset
  - **Critical Indicator**: Special handling of critical errors in page title
  - **Visual Persistence**: Badge remains visible until explicitly cleared

- **Refresh Controls**:
  - **Auto-refresh Toggle**: Enables automatic background data refresh
  - **Animation Feedback**: Spinning icon indicates active auto-refresh
  - **Configurable Interval**: 60-second default refresh rate
  - **AJAX Implementation**: Refreshes data without full page reload
  - **State Persistence**: Auto-refresh setting saved between sessions

All controls use consistent styling and provide immediate visual feedback on state changes to enhance usability.

### Using Real-Time Error Reporting

You can report errors in real-time using the API endpoint:

```
POST /api/report-error
```

Example using Python:

```python
import aiohttp
import json

async def report_error(dashboard_url, error_data):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{dashboard_url}/api/report-error",
            json=error_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            result = await response.json()
            return result
```

### Error Data Structure

Errors should be reported with the following data structure:

```json
{
  "timestamp": "2025-03-15T12:34:56.789",
  "worker_id": "worker-1",
  "type": "ResourceError",
  "error_category": "RESOURCE_ALLOCATION_ERROR",
  "message": "Failed to allocate GPU memory",
  "traceback": "Optional traceback string",
  "system_context": {
    "hostname": "test-node-1",
    "platform": "linux",
    "architecture": "x86_64",
    "python_version": "3.10.2",
    "metrics": {
      "cpu": {
        "percent": 75,
        "count": 16,
        "physical_count": 8,
        "frequency_mhz": 3200
      },
      "memory": {
        "used_percent": 80,
        "total_gb": 32,
        "available_gb": 6.4
      },
      "disk": {
        "used_percent": 65,
        "total_gb": 512,
        "free_gb": 179
      }
    },
    "gpu_metrics": {
      "count": 2,
      "devices": [
        {
          "index": 0,
          "name": "NVIDIA RTX 4090",
          "memory_utilization": 85,
          "temperature": 78
        },
        {
          "index": 1,
          "name": "NVIDIA RTX 4090",
          "memory_utilization": 82,
          "temperature": 75
        }
      ]
    }
  },
  "hardware_context": {
    "hardware_type": "cuda",
    "hardware_types": ["cuda", "cpu"],
    "hardware_status": {
      "overheating": false,
      "memory_pressure": true,
      "throttling": false
    }
  },
  "error_frequency": {
    "recurring": true,
    "same_type": {
      "last_1h": 3,
      "last_6h": 12,
      "last_24h": 25
    },
    "similar_message": {
      "last_1h": 2,
      "last_6h": 7,
      "last_24h": 15
    }
  }
}
```

## Testing the System

The Error Visualization system includes comprehensive testing tools to verify functionality and simulate error scenarios.

### Comprehensive Test Suite

Run the complete test suite to verify all system components:

```bash
python run_error_visualization_tests.py
```

This runs all test cases, including:
- Sound generation
- Error severity detection
- WebSocket integration
- Dashboard integration
- HTML template functionality

### Specific Component Testing

Test specific components of the system:

```bash
# Test sound generation functionality
python run_error_visualization_tests.py --type sound

# Test error severity detection
python run_error_visualization_tests.py --type severity

# Test WebSocket integration
python run_error_visualization_tests.py --type websocket

# Test dashboard integration
python run_error_visualization_tests.py --type dashboard

# Test HTML template
python run_error_visualization_tests.py --type html
```

### Generate Test Reports

Generate HTML or XML test reports:

```bash
# Generate HTML test report
python run_error_visualization_tests.py --report --report-format html

# Generate XML test report for CI/CD integration
python run_error_visualization_tests.py --report --report-format xml
```

Reports are saved to the `test_reports` directory.

### Real-Time Error Simulation

A test script is provided to simulate real-time errors and test the dashboard's notification system:

```bash
python tests/test_error_visualization_realtime.py --url http://localhost:8080 --count 10 --interval 2
```

#### Advanced Error Simulation Options

The test script supports several options for customizing error simulation:

```bash
# Generate 20 errors with 1-second interval
python tests/test_error_visualization_realtime.py --count 20 --interval 1

# Generate errors with 50% being critical
python tests/test_error_visualization_realtime.py --critical-percent 50

# Generate all critical errors
python tests/test_error_visualization_realtime.py --critical-percent 100

# Connect to a custom dashboard URL
python tests/test_error_visualization_realtime.py --url http://my-server:9000
```

#### Full Options Reference

- `--url`: URL of the dashboard server (default: http://localhost:8080)
- `--count`: Number of errors to generate (default: 10)
- `--interval`: Interval between error reports in seconds (default: 2.0)
- `--critical-percent`: Percentage of errors that should be critical (default: 20.0)

### Testing Workflow

For a complete testing workflow:

1. Start the dashboard:
   ```bash
   python run_monitoring_dashboard_with_error_visualization.py
   ```

2. Open the dashboard in your browser:
   ```
   http://localhost:8080/error-visualization
   ```

3. Run the error simulation script:
   ```bash
   python tests/test_error_visualization_realtime.py --count 15 --interval 2 --critical-percent 30
   ```

4. Observe the real-time updates in the dashboard, including:
   - Sound notifications based on error severity
   - Visual highlighting of errors
   - Error counter badge updates
   - Page title updates

5. Test the dashboard controls:
   - Adjust volume using the slider
   - Toggle mute
   - Reset error counts
   - Switch time ranges
   - Toggle auto-refresh

## Error Categories

The system recognizes the following error categories:

### System-Critical Error Categories
- `COORDINATOR_FAILURE`: Critical failure of coordinator component
- `DATABASE_CORRUPTION`: Corruption or integrity issues in database
- `SECURITY_BREACH`: Security breach or authentication failure
- `SYSTEM_CONFIG_CORRUPTION`: Critical configuration corruption

### Critical Error Categories
- `RESOURCE_ALLOCATION_ERROR`: Failed to allocate resources
- `HARDWARE_AVAILABILITY_ERROR`: Hardware availability issues
- `HARDWARE_CAPABILITY_ERROR`: Hardware capability issues
- `WORKER_CRASH_ERROR`: Worker node crashes

### Warning Error Categories
- `RESOURCE_CLEANUP_ERROR`: Failed to clean up resources
- `NETWORK_CONNECTION_ERROR`: Network connection issues
- `NETWORK_TIMEOUT_ERROR`: Network timeouts
- `HARDWARE_PERFORMANCE_ERROR`: Hardware performance degradation
- `WORKER_TIMEOUT_ERROR`: Worker node timeouts

### Info Error Categories
- `TEST_EXECUTION_ERROR`: Test execution failures
- `TEST_VALIDATION_ERROR`: Test validation failures
- `UNKNOWN_ERROR`: Unclassified errors

## Running the Dashboard

The Error Visualization system can be launched in several ways, depending on your specific needs:

### Method 1: Dedicated Dashboard Script (Recommended)

The simplest way to start the dashboard with error visualization enabled is to use the dedicated script:

```bash
# From the distributed_testing directory
python run_monitoring_dashboard_with_error_visualization.py
```

This script automatically enables error visualization without needing additional flags.

### Method 2: Standard Dashboard with Error Visualization Flag

Alternatively, you can use the standard dashboard script with the error visualization flag:

```bash
python run_monitoring_dashboard.py --enable-error-visualization
```

### Advanced Configuration Options

Both methods support additional configuration options:

```bash
# Configure host and port
python run_monitoring_dashboard_with_error_visualization.py --host 0.0.0.0 --port 8080

# Use a specific database file
python run_monitoring_dashboard_with_error_visualization.py --db-path ./benchmark_db.duckdb

# Set dashboard theme
python run_monitoring_dashboard_with_error_visualization.py --theme dark

# Use custom directories for static files and templates
python run_monitoring_dashboard_with_error_visualization.py --static-dir ./custom_static --template-dir ./custom_templates

# Enable additional integrations
python run_monitoring_dashboard_with_error_visualization.py --enable-result-aggregator --enable-e2e-test --enable-visualization
```

### Full Example with All Options

Here's a comprehensive example using all available options:

```bash
python run_monitoring_dashboard_with_error_visualization.py \
    --host 0.0.0.0 \
    --port 8080 \
    --db-path ./benchmark_db.duckdb \
    --static-dir ./static \
    --template-dir ./templates \
    --dashboard-dir ./dashboards \
    --theme dark \
    --refresh-interval 10 \
    --enable-result-aggregator \
    --result-aggregator-url http://localhost:8081 \
    --enable-e2e-test \
    --enable-visualization
```

### Running as a Service

For production environments, you may want to run the dashboard as a service. Here's a sample systemd service file:

```ini
[Unit]
Description=Distributed Testing Error Visualization Dashboard
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/distributed_testing
ExecStart=/usr/bin/python run_monitoring_dashboard_with_error_visualization.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Save this as `/etc/systemd/system/error-visualization-dashboard.service` and enable it with:

```bash
sudo systemctl enable error-visualization-dashboard.service
sudo systemctl start error-visualization-dashboard.service
```

## API Endpoints

The system provides the following API endpoints:

- `GET /error-visualization`: Web interface for error visualization
- `GET /api/errors`: Get error data in JSON format
- `POST /api/report-error`: Report a new error for real-time monitoring (NEW!)

## WebSocket Endpoints

For real-time error updates, the system provides the following WebSocket endpoint:

- `ws://localhost:8080/ws/error-visualization`: WebSocket for real-time error updates

### WebSocket Protocol

1. Client connects to the WebSocket endpoint
2. Client subscribes to error visualization updates using:
```json
{
  "type": "error_visualization_init",
  "time_range": 24
}
```
3. Client also subscribes to the general error topic:
```json
{
  "type": "subscribe",
  "topic": "error_visualization"
}
```
4. Server sends error updates as they occur:
```json
{
  "type": "error_visualization_update",
  "data": {
    "error": { /* error data */ },
    "time_range": 24
  }
}
```

## Database Integration

Error data is stored in a DuckDB database table named `worker_error_reports` with the following schema:

```sql
CREATE TABLE worker_error_reports (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    worker_id VARCHAR,
    type VARCHAR,
    error_category VARCHAR,
    message VARCHAR,
    traceback VARCHAR,
    system_context JSON,
    hardware_context JSON,
    error_frequency JSON
)
```

The database can be specified when starting the monitoring dashboard:

```bash
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-error-visualization --db-path ./benchmark_db.duckdb
```

## Error Severity Classification and Sound Selection

The Error Visualization system employs a sophisticated severity classification algorithm that intelligently analyzes incoming errors in real-time. This multi-factor analysis ensures appropriate notification intensity for each error, with a consistent response across visual, auditory, and notification channels.

### Automated Severity Detection

The system automatically determines error severity through JavaScript-based analysis of multiple factors:

```javascript
// Determine error severity for sound and notification
let errorSeverity = 'default';

if (error.is_system_critical) {
    // System-level critical errors get highest priority
    errorSeverity = 'system_critical';
} else if (error.is_critical) {
    errorSeverity = 'critical';
} else {
    // Check error category to determine severity
    const errorCategory = error.error_category || '';
    
    if (errorCategory.includes('COORDINATOR_FAILURE') || 
        errorCategory.includes('DATABASE_CORRUPTION') || 
        errorCategory.includes('SECURITY_BREACH')) {
        // Coordinator, database corruption, and security breach errors are system-critical
        errorSeverity = 'system_critical';
    } else if (errorCategory.includes('HARDWARE') || 
        errorCategory.includes('RESOURCE_ALLOCATION') || 
        errorCategory.includes('WORKER_CRASH')) {
        // Hardware, resource allocation, and worker crash errors are critical
        errorSeverity = 'critical';
    } else if (errorCategory.includes('NETWORK') || 
               errorCategory.includes('RESOURCE') ||
               errorCategory.includes('WORKER')) {
        // Network, other resource, and worker errors are warnings
        errorSeverity = 'warning';
    } else {
        // Test and other errors are info
        errorSeverity = 'info';
    }
}

// Play appropriate sound notification based on severity
playErrorNotification(errorSeverity);
```

This code is part of the `handleErrorUpdate()` function that processes incoming WebSocket error messages, ensuring consistent severity classification across all notification channels.

### System-Critical Errors

Errors are classified as **system-critical** based on these factors:

1. **Error Categories**:
   - `COORDINATOR_FAILURE` - Critical failure of the coordinator component
   - `DATABASE_CORRUPTION` - Corruption of the primary database
   - `SECURITY_BREACH` - Security breach or authentication failure

2. **System Status Indicators**:
   - Complete system unavailability
   - Data loss or integrity threats
   - Security compromises

3. **Infrastructure Impact**:
   - Multiple node failures
   - Data corruption affecting multiple components
   - Imminent system-wide failure

4. **Explicit System-Critical Flag**:
   - Errors with `is_system_critical: true` in their JSON payload

**System-Critical Error Response**:
- Highest-priority sound alerts using `error-system-critical.mp3` (rising pattern with accelerating pulse)
- Bright red highlighting with distinctive pulsing animation in dashboard
- Prominent modal dialog for immediate attention
- Desktop notifications with "SYSTEM CRITICAL:" prefix
- Error counter badge with distinctive system-critical styling
- Tab title with urgent prefix: `[SYSTEM CRITICAL] Error Visualization`
- Prioritized positioning at the top of error lists
- ARIA live region announcements with highest priority

### Critical Errors

Errors are classified as **critical** based on these factors:

1. **Error Categories**:
   - `HARDWARE_AVAILABILITY_ERROR` - Hardware device not available for operation
   - `RESOURCE_ALLOCATION_ERROR` - Failed critical resource allocation
   - `WORKER_CRASH_ERROR` - Unexpected worker node termination

2. **Hardware Status Indicators**:
   - Overheating conditions (temperature thresholds exceeded)
   - Memory pressure (high memory utilization or OOM conditions)
   - Hardware capability mismatches

3. **System Metrics**:
   - Extremely high CPU usage (>90%)
   - Critical memory utilization (>95%)
   - Disk space critically low (<5% free)

4. **Explicit Critical Flag**:
   - Errors with `is_critical: true` in their JSON payload

**Critical Error Response**:
- High-priority sound alerts using `error-critical.mp3` (880Hz/440Hz with pulsing)
- Red highlighting with persistent pulsing animation in dashboard
- Desktop notifications with "Critical Error:" prefix
- Error counter badge with visual emphasis
- Prominent positioning in error lists
- Tab title updates with critical count indicator: `(3, 2 critical) Error Visualization`
- ARIA live region announcements for screen readers

### Warning Errors

Errors are classified as **warnings** based on these factors:

1. **Error Categories**:
   - `NETWORK_CONNECTION_ERROR` - Network connectivity issues
   - `NETWORK_TIMEOUT_ERROR` - Network operation timeouts
   - `RESOURCE_CLEANUP_ERROR` - Resource cleanup failures
   - `WORKER_TIMEOUT_ERROR` - Worker response delays

2. **Hardware Status Indicators**:
   - Throttling conditions
   - Resource contention
   - Performance degradation

3. **Pattern Recognition**:
   - Recurring errors (same error multiple times)
   - Escalating error frequency
   - Related error sequences

**Warning Error Response**:
- Medium-priority sound alerts using `error-warning.mp3` (660Hz/330Hz with moderate decay)
- Yellow highlighting in dashboard using `warning-highlight 2s` animation
- Standard desktop notifications with descriptive titles
- Temporary visual highlighting
- Tab title updates with standard count: `(3) Error Visualization`
- ARIA announcements with warning context

### Info Errors

Errors are classified as **informational** based on these factors:

1. **Error Categories**:
   - `TEST_EXECUTION_ERROR` - Test process errors
   - `TEST_VALIDATION_ERROR` - Test validation failures
   - `UNKNOWN_ERROR` - Unclassified errors

2. **Impact Assessment**:
   - Low system impact
   - Non-critical components affected
   - Expected failure cases

3. **Occurrence Patterns**:
   - First-time occurrences
   - Isolated incidents
   - Non-recurring patterns

**Info Error Response**:
- Low-priority sound alerts using `error-info.mp3` (523Hz with quick decay)
- Blue or neutral highlighting in dashboard using `highlight 2s` animation
- Optional desktop notifications (shown only when tab is not focused)
- Standard entry in error logs without special emphasis
- Minimal visual highlighting without persistent effects

### Real-Time Sound Selection Logic

The system's `playErrorNotification()` function handles sound selection based on the detected severity:

```javascript
function playErrorNotification(errorType = 'default') {
    // Skip if notifications are muted
    if (notificationsMuted) {
        console.log('Notification sound muted');
        return;
    }
    
    // Create audio element if it doesn't exist
    let audio = document.getElementById('error-audio');
    if (!audio) {
        audio = document.createElement('audio');
        audio.id = 'error-audio';
        document.body.appendChild(audio);
    }
    
    // Set volume from preference
    audio.volume = notificationVolume;
    
    // Set source based on error type
    let soundFile = 'error-notification.mp3'; // Default sound
    
    if (errorType === 'system_critical') {
        soundFile = 'error-system-critical.mp3';
    } else if (errorType === 'critical') {
        soundFile = 'error-critical.mp3';
    } else if (errorType === 'warning') {
        soundFile = 'error-warning.mp3';
    } else if (errorType === 'info') {
        soundFile = 'error-info.mp3';
    }
    
    // Set the audio source
    audio.src = `/static/sounds/${soundFile}`;
    
    // Play the sound with fallback mechanism
    audio.play().catch(function(error) {
        console.log(`Error playing ${soundFile}:`, error);
        
        // Try the default notification sound as fallback
        if (soundFile !== 'error-notification.mp3') {
            audio.src = '/static/sounds/error-notification.mp3';
            audio.play().catch(function(fallbackError) {
                console.log('Error playing fallback sound:', fallbackError);
            });
        }
    });
}
```

This intelligent error classification system ensures that each error receives the appropriate level of attention through consistent visual, auditory, and notification responses. By matching notification intensity to error severity, the system prevents both alert fatigue from minor issues and ensures critical problems get immediate attention.

## Troubleshooting

This section covers common issues you might encounter when using the Error Visualization system and how to resolve them.

### Common Issues and Solutions

#### Dashboard Issues

1. **No error data available in dashboard**
   - **Problem**: Dashboard displays "No Error Data Available" message
   - **Causes**:
     - Dashboard not started with error visualization enabled
     - No errors reported yet
     - Database connection issues
   - **Solutions**:
     - Ensure you're using `run_monitoring_dashboard_with_error_visualization.py` or the `--enable-error-visualization` flag
     - Run the test error reporting script to generate some sample errors
     - Check database path and permissions

2. **WebSocket connection fails**
   - **Problem**: Error updates don't appear in real-time
   - **Causes**:
     - WebSocket server not running
     - Network/firewall blocking WebSocket connections
     - CORS issues in browser
   - **Solutions**:
     - Verify the dashboard is running (check logs)
     - Check for firewall or proxy blocking WebSocket connections
     - Check browser console for connection errors
     - Try a different browser or network connection

3. **Dashboard pages load slowly or incompletely**
   - **Problem**: Dashboard performance issues
   - **Causes**:
     - Too many errors in the database
     - Missing or slow visualization libraries
     - Browser performance issues
   - **Solutions**:
     - Specify a smaller time range (1h instead of 24h)
     - Install optional dependencies (plotly, pandas) for better performance
     - Limit the number of errors shown with filtering
     - Use Chrome or Firefox for best performance

#### Sound Notification Issues

4. **No sound notifications**
   - **Problem**: Errors appear but no sounds play
   - **Causes**:
     - Sound files missing or not generated
     - Browser audio settings (muted or disabled)
     - System is muted in dashboard
   - **Solutions**:
     - Check that sound files exist in `/static/sounds/` directory
     - Run `generate_sound_files.py` to create sound files
     - Check browser audio settings and permissions
     - Unmute notifications in the dashboard UI
     - Check volume slider in dashboard

5. **Sound file generation fails**
   - **Problem**: Error when generating sound files
   - **Causes**:
     - Missing numpy or scipy dependencies
     - Permission issues in sounds directory
   - **Solutions**:
     - Install required dependencies: `pip install numpy scipy`
     - Check permissions on the sounds directory
     - Try running with administrator/root privileges

#### Database Issues

6. **Database errors**
   - **Problem**: Errors related to database access
   - **Causes**:
     - Missing DuckDB installation
     - Invalid database path
     - Permission issues
     - Corrupt database file
   - **Solutions**:
     - Install DuckDB: `pip install duckdb`
     - Specify a valid database path with `--db-path`
     - Check file permissions
     - Create a new database file if corruption is suspected

7. **Error data not being stored**
   - **Problem**: Errors reported but not visible in dashboard
   - **Causes**:
     - Database write permissions
     - Schema mismatch
   - **Solutions**:
     - Check database file permissions
     - Verify the error schema is correct
     - Check for errors in dashboard logs

### Diagnostic Commands

#### Check Dashboard Status

```bash
# Check if dashboard process is running
ps aux | grep run_monitoring_dashboard

# Check if dashboard port is open
nc -zv localhost 8080

# Check dashboard logs
tail -f monitoring_dashboard.log
```

#### Check Database

```bash
# Verify DuckDB installation
python -c "import duckdb; print(duckdb.__version__)"

# Check database file exists
ls -la ./benchmark_db.duckdb

# Check database schema
python -c "import duckdb; conn=duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT * FROM sqlite_master WHERE type=\"table\"').fetchall())"

# Count errors in database
python -c "import duckdb; conn=duckdb.connect('./benchmark_db.duckdb'); print(conn.execute('SELECT COUNT(*) FROM worker_error_reports').fetchone())"
```

#### Check Sound Files

```bash
# Check sound files exist
ls -la ./dashboard/static/sounds/*.mp3

# Regenerate sound files
python ./dashboard/static/sounds/generate_sound_files.py

# Test sound file playback (requires ffplay)
ffplay -nodisp -autoexit ./dashboard/static/sounds/error-critical.mp3
```

### Logs and Diagnostics

For more detailed diagnostics, check the dashboard logs:

```bash
# View dashboard logs
tail -f monitoring_dashboard.log

# Increase logging verbosity (modify run_monitoring_dashboard_with_error_visualization.py)
# Change this line: logging.basicConfig(level=logging.INFO, ...)
# To: logging.basicConfig(level=logging.DEBUG, ...)

# Capture WebSocket traffic (requires tcpdump)
sudo tcpdump -i lo -n port 8080 -A | grep -i websocket
```

### Browser Console Debugging

For client-side issues, use your browser's developer tools (F12):

1. Open browser developer tools
2. Go to the Console tab
3. Look for errors or warnings
4. Check Network tab for WebSocket connection issues
5. Use Application tab to inspect localStorage settings

## Accessibility Features

The Error Visualization dashboard includes comprehensive accessibility enhancements to ensure the system is usable by all team members, including those with disabilities. These features comply with WCAG 2.1 guidelines and were implemented with input from accessibility experts.

### ARIA Implementation

- **Live Regions**: Error updates use `aria-live="polite"` regions to announce new errors to screen readers without interrupting the user's current focus
- **Descriptive Labels**: All controls have semantic `aria-label` attributes describing their function and current state
- **Error Descriptions**: Critical errors include explicit descriptive labels for accurate screen reader announcement
- **Role Attributes**: Proper semantic roles define the purpose of UI components
- **State Attributes**: `aria-checked`, `aria-expanded`, and other state attributes communicate control states
- **Focus Management**: Maintains logical focus order for keyboard navigation
- **Landmark Regions**: Dashboard sections use landmark roles for navigation

### Keyboard Accessibility

- **Full Navigation**: All interactive elements are fully keyboard accessible
- **Visible Focus**: Focus indicators are clearly visible for keyboard users
- **Logical Tab Order**: Navigation follows a logical and predictable sequence
- **Shortcut Keys**: Common actions have keyboard shortcuts (documented in help section)
- **No Keyboard Traps**: Focus can always be moved away from any element
- **Toggle Controls**: Error details, mute, and auto-refresh can all be toggled with keyboard

### High Contrast and Visual Accessibility

- **Forced Colors Mode**: The dashboard supports Windows high-contrast mode via the `forced-colors` media query
- **Enhanced Borders**: Critical UI elements have reinforced borders in high-contrast mode
- **Color Independence**: All information conveyed by color also uses other visual indicators
- **Text Contrast**: All text meets WCAG AA standard 4.5:1 contrast ratio
- **Scalable UI**: Dashboard remains usable when text is enlarged up to 200%
- **Reduced Motion**: Respects user preference for reduced motion when specified

### Screen Reader Optimization

- **Semantic HTML**: Proper heading structure and semantic elements
- **Meaningful Sequences**: Content order is logical and meaningful
- **Status Announcements**: Error status changes are announced appropriately
- **Error Severity Context**: Error severities are explicitly conveyed in announcements
- **Alternative Text**: Visual elements have appropriate text alternatives
- **Hidden Context**: Additional context is provided for screen readers via visually hidden text

### Cognitive Accessibility

- **Consistent Layout**: Dashboard maintains consistent layout and navigation
- **Error Identification**: Errors are clearly identifiable by multiple characteristics
- **Status Persistence**: Important status information remains visible
- **Simple Language**: Error messages use clear, straightforward language
- **Notification Control**: Users can control the intensity and frequency of notifications

These accessibility features ensure that the Error Visualization system is usable by all team members regardless of abilities or preferences, promoting an inclusive monitoring environment.

## Sound File Creation and Customization

The Error Visualization system includes a comprehensive sound notification system that intelligently selects appropriate sounds based on error severity. This section explains how to customize these sounds and how the included sound generation script works.

### Automatic Sound Generation

The system comes with a built-in sound generation script that creates all necessary notification sounds:

1. **Run the automated script**:
   ```bash
   cd /duckdb_api/distributed_testing/dashboard/static/sounds/
   python generate_sound_files.py
   ```

2. **Sound files created**:
   - `error-system-critical.mp3`: Highest priority alert sound (1.0s duration)
   - `error-critical.mp3`: High priority alert sound (0.7s duration)
   - `error-warning.mp3`: Medium priority alert sound (0.5s duration)
   - `error-info.mp3`: Low priority notification sound (0.3s duration)
   - `error-notification.mp3`: Default notification sound (copy of critical)

The generation script uses `numpy` and `scipy` to synthesize tones with specific acoustic properties for each severity level. It then uses `ffmpeg` to convert the WAV files to browser-compatible MP3 format, with a fallback mechanism in case `ffmpeg` is not available.

### Sound Design Principles

Each sound is carefully designed according to these principles:

1. **System-critical error sounds**:
   - Rising frequency pattern (880Hz → 1046.5Hz → 1318.5Hz) creates urgent escalation
   - Accelerating pulse rate (4Hz to 16Hz) communicates increasing urgency
   - Three distinct segments with crossfading create a progressive alert
   - 1.0 second duration ensures the sound is noticed even in noisy environments
   - Added harmonic richness creates a distinctively urgent signature
   - Designed to cut through ambient noise and command immediate attention

2. **Critical error sounds**:
   - Higher frequency base (880Hz/440Hz) creates urgency and grabs attention
   - Amplitude modulation with pulsing effect (8Hz) enhances alerting quality
   - 0.7 second duration is long enough to notice without being intrusive
   - Combined tones with different decay rates create a distinctive signature

3. **Warning error sounds**:
   - Medium frequency (660Hz/330Hz) strikes balance between alerting and comfort
   - Moderate decay profile creates "notice me" quality without startling
   - 0.5 second duration provides adequate notice for non-critical issues
   - Secondary tone at one octave lower adds richness and recognizability

4. **Info error sounds**:
   - Lower frequency (523Hz/C5) provides gentle notification
   - Quick decay (10x faster than critical) minimizes disruption
   - 0.3 second duration creates minimal interruption
   - Simple waveform for non-critical notifications

### Customizing Sound Files

You can customize the sound files in several ways:

1. **Replace the generated files**:
   - Create custom MP3 files with the same names:
     - `error-critical.mp3`
     - `error-warning.mp3`
     - `error-info.mp3`
     - `error-notification.mp3`
   - Place them in `/duckdb_api/distributed_testing/dashboard/static/sounds/`

2. **Modify the generation script**:
   - Edit `generate_sound_files.py` to adjust tone parameters:
     - Frequency values control the pitch (higher = more urgent)
     - Duration parameters control sound length
     - Decay rates control how quickly the sound fades
     - Modulation parameters control pulsing effects

3. **Source from sound libraries**:
   - Use professionally-created sounds from libraries
   - Ensure proper licensing for your environment
   - Rename files to match the expected naming convention
   - Keep sounds short (0.3-0.7 seconds) to prevent notification fatigue

### Technical Implementation

The sound generation script uses scientific computing libraries to synthesize natural-sounding notifications. Here are examples of the system-critical and critical sound generation functions:

```python
import numpy as np
from scipy.io import wavfile

def generate_system_critical_sound(filename, duration=1.0):
    """Generate the highest priority alert sound for system-level critical errors."""
    # Use a more complex sound pattern with increasing urgency
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Use multiple frequencies with a rising pattern
    frequency1 = 880  # A5
    frequency2 = 1046.5  # C6
    frequency3 = 1318.5  # E6
    
    # Create rising tones with amplitude modulation
    segment_duration = duration / 3
    segment1 = t < segment_duration
    segment2 = (t >= segment_duration) & (t < 2 * segment_duration)
    segment3 = t >= 2 * segment_duration
    
    # Create three tones that build in intensity
    tone1 = np.sin(2 * np.pi * frequency1 * t) * segment1
    tone2 = np.sin(2 * np.pi * frequency2 * t) * segment2
    tone3 = np.sin(2 * np.pi * frequency3 * t) * segment3
    
    # Combine tones with crossfade
    signal = tone1 + tone2 + tone3
    
    # Add an urgent pulsing effect that speeds up
    pulse_rate = 4 + 12 * t/duration  # Pulse rate increases from 4Hz to 16Hz
    pulse = 0.7 + 0.3 * np.sin(2 * np.pi * pulse_rate * t)
    signal = signal * pulse
    
    # Add a subtle harmonic for richness
    harmonic = 0.2 * np.sin(2 * np.pi * 2 * frequency1 * t) * np.exp(-1 * t/duration)
    signal = signal + harmonic
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)

def generate_critical_sound(filename, duration=0.7):
    """Generate a high priority alert sound for critical errors."""
    # Use a higher frequency with amplitude modulation for urgency
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Start with a higher frequency and then drop
    frequency1 = 880  # A5
    frequency2 = 440  # A4
    
    # Create two tones with amplitude modulation
    tone1 = np.sin(2 * np.pi * frequency1 * t) * np.exp(-3 * t)
    tone2 = np.sin(2 * np.pi * frequency2 * t) * (1 - np.exp(-5 * t))
    
    # Combine tones
    signal = 0.5 * tone1 + 0.5 * tone2
    
    # Add a pulsing effect
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)
    signal = signal * pulse
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)
```

The script also includes similar functions for warning and info sounds, each with parameters tuned to their specific purpose.

### Browser Integration

The JavaScript code automatically selects the appropriate sound based on error severity:

```javascript
function playErrorNotification(errorType = 'default') {
    // Skip if notifications are muted
    if (notificationsMuted) {
        console.log('Notification sound muted');
        return;
    }
    
    // Set source based on error type
    let soundFile = 'error-notification.mp3'; // Default sound
    
    if (errorType === 'critical') {
        soundFile = 'error-critical.mp3';
    } else if (errorType === 'warning') {
        soundFile = 'error-warning.mp3';
    } else if (errorType === 'info') {
        soundFile = 'error-info.mp3';
    }
    
    // Set the audio source and play
    audio.src = `/static/sounds/${soundFile}`;
    audio.volume = notificationVolume;
    audio.play();
}
```

The system provides graceful degradation by falling back to the default sound if a specific sound file cannot be loaded.

## Advanced Usage

### Custom Integration

You can integrate your own systems with the Error Visualization dashboard using the provided WebSocket and REST APIs:

```python
# Example of custom error reporting integration
import aiohttp
import json
import anyio

async def report_custom_error(dashboard_url, error_data):
    """Report a custom error to the Error Visualization system.
    
    Args:
        dashboard_url: URL of the dashboard API
        error_data: Custom error data dictionary
    """
    async with aiohttp.ClientSession() as session:
        await session.post(
            f"{dashboard_url}/api/report-error",
            json=error_data,
            headers={'Content-Type': 'application/json'}
        )

# Example of listening for error updates via WebSocket
async def listen_for_errors(dashboard_url):
    """Listen for real-time error updates via WebSocket.
    
    Args:
        dashboard_url: Base URL of the dashboard
    """
    # Convert HTTP URL to WebSocket URL
    ws_url = dashboard_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f"{ws_url}/ws/error-visualization"
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            # Subscribe to error visualization updates
            await ws.send_json({
                "type": "error_visualization_init",
                "time_range": 24
            })
            
            # Also subscribe to the general error topic
            await ws.send_json({
                "type": "subscribe",
                "topic": "error_visualization"
            })
            
            # Process incoming messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get('type') == 'error_visualization_update':
                        error = data.get('data', {}).get('error')
                        if error:
                            print(f"New error: {error.get('error_type')} - {error.get('message')}")
```

### Dashboard Customization

The dashboard appearance and behavior can be customized by modifying templates and configuration:

- **Custom Themes**: Extend the existing light and dark themes in `dashboard.css`
- **Custom Sounds**: Replace default sounds with your own audio files
- **Layout Customization**: Modify the dashboard template to reorganize sections
- **Auto-Refresh Interval**: Adjust the default 60-second refresh interval

## Future Development

Future enhancements planned for the Error Visualization system include:

1. **Advanced Analytics**:
   - Machine learning-based pattern detection and classification
   - Predictive error analysis to anticipate failures before they occur
   - Historical trend analysis with statistical significance testing

2. **Automated Resolution**:
   - Root cause correlation to automatically suggest fixes
   - Auto-remediation for common error patterns
   - Integration with CI/CD systems for automated fixes

3. **Expanded Notifications**:
   - Slack and MS Teams integration for team notifications
   - Email alerts for critical errors with configurable thresholds
   - Mobile push notifications for on-the-go monitoring
   - SMS alerts for critical infrastructure issues

4. **Enhanced Customization**:
   - User-specific notification preferences by error type and severity
   - Team-based notification rules and escalation policies
   - Custom dashboard layouts for specific monitoring needs
   - Personalized highlight colors and sound preferences

5. **Enterprise Integration**:
   - LDAP/Active Directory authentication
   - Role-based access control for dashboard features
   - Integration with external monitoring systems (Prometheus, Grafana)
   - Support for enterprise SSO solutions

## Conclusion

The Error Visualization system provides comprehensive tools for monitoring, analyzing, and diagnosing errors in the Distributed Testing Framework. The enhanced real-time monitoring capabilities deliver immediate awareness of issues as they occur, with intelligent severity classification ensuring that critical problems receive appropriate attention.

Key benefits of the system include:

- **Immediate Awareness**: Real-time error notifications via multiple channels (visual, audio, desktop)
- **Intelligent Classification**: Automatic severity assessment based on error characteristics
- **Comprehensive Analysis**: Error pattern detection, worker analysis, and hardware diagnostics
- **User-Friendly Controls**: Customizable notification preferences and dashboard options
- **Accessibility**: Full compliance with accessibility standards for inclusive team monitoring
- **Extensibility**: WebSocket and REST APIs for integration with external systems
- **Robustness**: Automatic reconnection and fault tolerance features

The Error Visualization dashboard serves as a central hub for monitoring distributed testing operations, enabling teams to quickly identify, diagnose, and resolve issues across the testing infrastructure. With its comprehensive feature set and focus on user experience, the system significantly enhances the reliability and observability of the Distributed Testing Framework.
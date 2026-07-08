# Error Visualization Implementation Guide

This guide explains how to implement and use the Error Visualization system in your distributed testing environment.

## Overview

The Error Visualization system provides real-time monitoring and analysis of errors in distributed testing environments. It includes:

1. Real-time error notifications with sound alerts
2. Error pattern detection and analysis
3. Worker status monitoring
4. Hardware error analysis
5. Interactive dashboard with WebSocket updates

## Implementing Error Reporting

### Reporting Errors from Your Code

To report errors from your code, use the API endpoint provided by the dashboard:

```python
import aiohttp
import json
from datetime import datetime

async def report_error(dashboard_url, error_data):
    """Report an error to the Error Visualization Dashboard.
    
    Args:
        dashboard_url: URL of the dashboard API (e.g., "http://localhost:8080")
        error_data: Error data dictionary
        
    Returns:
        True if error was successfully reported, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{dashboard_url}/api/report-error",
                json=error_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                result = await response.json()
                if response.status == 200 and result.get('status') == 'success':
                    print(f"Error reported successfully: {error_data.get('type')} - {error_data.get('message')}")
                    return True
                else:
                    print(f"Failed to report error: {response.status} - {result}")
                    return False
    except Exception as e:
        print(f"Exception while reporting error: {e}")
        return False

# Example usage:
error_data = {
    "timestamp": datetime.now().isoformat(),
    "worker_id": "worker-1",
    "type": "ResourceError",
    "error_category": "RESOURCE_EXHAUSTED",
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
            "overheating": False,
            "memory_pressure": True,
            "throttling": False
        }
    },
    "error_frequency": {
        "recurring": True,
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

await report_error("http://localhost:8080", error_data)
```

## Running the Dashboard

### Option 1: Run with Error Visualization Enabled

The simplest way to run the dashboard with error visualization is to use the dedicated script:

```bash
cd /path/to/duckdb_api/distributed_testing
python run_monitoring_dashboard_with_error_visualization.py
```

This will start the dashboard with error visualization enabled by default.

### Option 2: Configure the Standard Dashboard

Alternatively, you can enable error visualization in the standard monitoring dashboard:

```bash
cd /path/to/duckdb_api/distributed_testing
python run_monitoring_dashboard.py --enable-error-visualization
```

### Advanced Configuration

You can configure various aspects of the dashboard:

```bash
cd /path/to/duckdb_api/distributed_testing
python run_monitoring_dashboard_with_error_visualization.py \
    --host 0.0.0.0 \
    --port 8080 \
    --db-path ./benchmark_db.duckdb \
    --theme dark
```

## Accessing the Dashboard

Once the dashboard is running, access the error visualization page at:

```
http://localhost:8080/error-visualization
```

## Sound Notification System

The error visualization system includes a sound notification system that plays different sounds based on error severity.

### Sound Files

The system uses these sound files:

1. **error-critical.mp3**: High-priority alert sound for critical errors
2. **error-warning.mp3**: Medium-priority alert sound for warning-level errors
3. **error-info.mp3**: Low-priority notification sound for informational errors
4. **error-notification.mp3**: Default notification sound (fallback)

### Generating Sound Files

Sound files are generated automatically using the provided script:

```bash
cd /path/to/duckdb_api/distributed_testing/dashboard/static/sounds
python generate_sound_files.py
```

This will generate all necessary sound files using numpy and scipy for synthesis.

### Customizing Sounds

You can customize the sounds by:

1. Using the provided generation script with modified parameters
2. Replacing the MP3 files with your own custom sounds
3. Modifying the generate_sound_files.py script to change sound characteristics

## Dashboard Features

The error visualization dashboard provides these main features:

### Error Summary

- Overview of total errors
- Critical hardware errors
- Resource errors
- Network errors
- Recent errors list

### Error Distribution

- Chart showing distribution by category
- Error list with filtering
- Error details with system context

### Error Patterns

- Detection of recurring error patterns
- Chart showing pattern distribution by category
- List of top error patterns with frequency data

### Worker Errors

- Chart showing errors by worker
- Worker status (critical, warning, stable)
- Most common error by worker

### Hardware Errors

- Chart showing error rate by hardware type
- Hardware status cards with metrics
- Recent hardware-related errors list

### UI Controls

- Time range selection (1h, 6h, 24h, 7d)
- Sound volume control and mute option
- Error count reset button
- Auto-refresh toggle
- Theme selection (light/dark)

## Testing

The Error Visualization system includes comprehensive tests:

```bash
cd /path/to/duckdb_api/distributed_testing
python run_error_visualization_tests.py
```

Run specific test categories:

```bash
# Test sound generation
python run_error_visualization_tests.py --type sound

# Test severity detection
python run_error_visualization_tests.py --type severity

# Test WebSocket integration
python run_error_visualization_tests.py --type websocket

# Test dashboard integration
python run_error_visualization_tests.py --type dashboard

# Test HTML template
python run_error_visualization_tests.py --type html
```

Generate a test report:

```bash
python run_error_visualization_tests.py --report --report-format html
```

## Error Severity Classification

The system automatically classifies errors into three severity levels:

### Critical Errors

Detected based on:
- Error categories: HARDWARE_NOT_AVAILABLE, RESOURCE_EXHAUSTED, WORKER_CRASHED
- Hardware status: overheating, memory pressure
- System metrics: high CPU usage (>90%), high memory usage (>95%), low disk space

### Warning Errors

Detected based on:
- Error categories: NETWORK_* errors, RESOURCE_* errors (except EXHAUSTED), WORKER_* errors (except CRASHED)
- Hardware status: throttling
- System metrics: moderate resource usage
- Recurring errors: multiple occurrences within a time window

### Info Errors

Detected based on:
- Error categories: TEST_* errors and UNKNOWN_ERROR
- Low impact errors (based on system context)
- Non-recurring errors

## Integration with External Systems

The Error Visualization system is designed to be easily integrated with external monitoring, alerting, and reporting systems. This section provides implementation examples for common integration scenarios.

### Integration with Slack

You can forward critical errors to Slack channels for immediate team notification:

```python
import aiohttp
import json
import anyio
from datetime import datetime

async def report_error_with_slack_notification(dashboard_url, slack_webhook_url, error_data):
    """Report an error to the Error Visualization Dashboard and notify Slack for critical errors.
    
    Args:
        dashboard_url: URL of the dashboard API
        slack_webhook_url: Slack webhook URL for notifications
        error_data: Error data dictionary
    """
    # Report error to dashboard
    async with aiohttp.ClientSession() as session:
        # Step 1: Report to Error Visualization system
        dashboard_response = await session.post(
            f"{dashboard_url}/api/report-error",
            json=error_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check if error is critical
        is_critical = False
        
        # Check error category
        if error_data.get('error_category') in [
            'HARDWARE_NOT_AVAILABLE', 
            'RESOURCE_EXHAUSTED', 
            'WORKER_CRASH_ERROR'
        ]:
            is_critical = True
        
        # Check hardware status
        hardware_context = error_data.get('hardware_context', {})
        hardware_status = hardware_context.get('hardware_status', {})
        if hardware_status.get('overheating') or hardware_status.get('memory_pressure'):
            is_critical = True
        
        # Send to Slack if critical
        if is_critical:
            # Step 2: Send notification to Slack for critical errors
            slack_message = {
                "text": "🚨 *CRITICAL ERROR DETECTED* 🚨",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "🚨 Critical Error Detected 🚨",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Type:* {error_data.get('type')}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Category:* {error_data.get('error_category')}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Worker:* {error_data.get('worker_id')}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:* {error_data.get('timestamp')}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Message:*\n```{error_data.get('message')}```"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View in Dashboard",
                                    "emoji": True
                                },
                                "url": f"{dashboard_url}/error-visualization"
                            }
                        ]
                    }
                ]
            }
            
            await session.post(
                slack_webhook_url,
                json=slack_message,
                headers={'Content-Type': 'application/json'}
            )
```

### Integration with Prometheus for Metrics

You can export error metrics to Prometheus for long-term storage and alerting:

```python
from prometheus_client import Counter, Gauge, start_http_server
import time
import threading

# Define Prometheus metrics
error_counter = Counter('distributed_testing_errors_total', 'Total number of errors', ['category', 'worker', 'type'])
critical_error_gauge = Gauge('distributed_testing_critical_errors', 'Current number of critical errors')
error_rate = Gauge('distributed_testing_error_rate', 'Error rate per minute')

class ErrorMetricsExporter:
    def __init__(self, metrics_port=9090, poll_interval=60):
        self.metrics_port = metrics_port
        self.poll_interval = poll_interval
        self.dashboard_url = "http://localhost:8080"
        self.last_error_count = 0
        self.last_check_time = time.time()
        
    def start(self):
        # Start Prometheus HTTP server
        start_http_server(self.metrics_port)
        
        # Start metrics collection thread
        threading.Thread(target=self._collect_metrics_loop, daemon=True).start()
        
    async def _collect_metrics_loop(self):
        while True:
            try:
                await self._update_metrics()
            except Exception as e:
                print(f"Error updating metrics: {e}")
            
            # Sleep for the poll interval
            await anyio.sleep(self.poll_interval)
            
    async def _update_metrics(self):
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Fetch error data from API
            async with session.get(f"{self.dashboard_url}/api/errors?time_range=1") as response:
                if response.status != 200:
                    return
                
                data = await response.json()
                if not data.get('status') == 'success':
                    return
                
                error_data = data.get('data', {})
                
                # Update error counters
                for error in error_data.get('recent_errors', []):
                    error_counter.labels(
                        category=error.get('error_category', 'unknown'),
                        worker=error.get('worker_id', 'unknown'),
                        type=error.get('type', 'unknown')
                    ).inc()
                
                # Update critical error gauge
                critical_count = sum(1 for error in error_data.get('recent_errors', []) 
                                   if error.get('is_critical', False))
                critical_error_gauge.set(critical_count)
                
                # Calculate error rate
                current_time = time.time()
                current_count = error_data.get('summary', {}).get('total_errors', 0)
                
                time_diff = (current_time - self.last_check_time) / 60  # Convert to minutes
                count_diff = current_count - self.last_error_count
                
                if time_diff > 0:
                    error_rate.set(count_diff / time_diff)
                
                # Update last values
                self.last_check_time = current_time
                self.last_error_count = current_count

# Usage
exporter = ErrorMetricsExporter(metrics_port=9090)
exporter.start()
```

### Integration with Email Alerts

For critical errors, you may want to send email alerts to system administrators:

```python
import smtplib
from email.message import EmailMessage
import anyio

class EmailAlertSystem:
    def __init__(self, smtp_server, smtp_port, username, password, recipients):
        """Initialize email alert system.
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            recipients: List of email recipients
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.dashboard_url = "http://localhost:8080"
        
        # Start monitoring thread
        anyio.create_task_group()
    
    async def _monitor_errors(self):
        import aiohttp
        
        # Connect to WebSocket for real-time updates
        async with aiohttp.ClientSession() as session:
            ws_url = f"ws://{self.dashboard_url.replace('http://', '')}/ws/error-visualization"
            
            while True:
                try:
                    async with session.ws_connect(ws_url) as ws:
                        # Subscribe to error updates
                        await ws.send_json({
                            "type": "error_visualization_init",
                            "time_range": 24
                        })
                        
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
                                    
                                    # Check if error is critical
                                    if error and error.get('is_critical'):
                                        await self._send_email_alert(error)
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    await anyio.sleep(5)  # Wait before reconnecting
    
    async def _send_email_alert(self, error):
        """Send email alert for critical error.
        
        Args:
            error: Error data dictionary
        """
        subject = f"CRITICAL ERROR: {error.get('error_category', 'Unknown')} in {error.get('worker_id', 'Unknown')}"
        
        # Create message
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        
        # Create message body
        body = f"""
CRITICAL ERROR DETECTED

Type: {error.get('type', 'Unknown')}
Category: {error.get('error_category', 'Unknown')}
Worker: {error.get('worker_id', 'Unknown')}
Time: {error.get('timestamp', 'Unknown')}

Message:
{error.get('message', 'No message')}

Traceback:
{error.get('traceback', 'No traceback')}

View in Dashboard: {self.dashboard_url}/error-visualization
        """
        
        msg.set_content(body)
        
        # Send email
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            print(f"Email alert sent: {subject}")
        except Exception as e:
            print(f"Failed to send email alert: {e}")

# Usage
alert_system = EmailAlertSystem(
    smtp_server="smtp.example.com",
    smtp_port=587,
    username="alerts@example.com",
    password="your_password",
    recipients=["admin@example.com", "team@example.com"]
)
```

### Integration with Grafana

You can visualize error data in Grafana using the Prometheus data source:

1. Set up the Prometheus exporter as shown above
2. Add Prometheus as a data source in Grafana
3. Create a dashboard with panels like:
   - Error count by category (bar chart)
   - Critical errors over time (time series)
   - Error rate (gauge)
   - Worker status (status map)

Example Prometheus queries for Grafana:

```
# Total errors by category
sum(distributed_testing_errors_total) by (category)

# Critical errors
distributed_testing_critical_errors

# Error rate
distributed_testing_error_rate

# Errors by worker
sum(distributed_testing_errors_total) by (worker)
```

## API Reference

### Error Reporting API

- **Endpoint**: `/api/report-error`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**: Error data (see schema above)
- **Response**: `{"status": "success", "message": "Error reported successfully"}`

#### Example Request

```bash
curl -X POST http://localhost:8080/api/report-error \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-03-15T12:34:56.789",
    "worker_id": "worker-1",
    "type": "ResourceError",
    "error_category": "RESOURCE_EXHAUSTED",
    "message": "Failed to allocate GPU memory",
    "system_context": {
      "hostname": "test-node-1",
      "metrics": {
        "cpu": {"percent": 75},
        "memory": {"used_percent": 80}
      }
    },
    "hardware_context": {
      "hardware_type": "cuda",
      "hardware_status": {
        "memory_pressure": true
      }
    }
  }'
```

### Error Retrieval API

- **Endpoint**: `/api/errors`
- **Method**: GET
- **Query Parameters**: time_range (1, 6, 24, 168)
- **Response**: `{"status": "success", "data": {...}}`

#### Example Request

```bash
curl -X GET "http://localhost:8080/api/errors?time_range=24"
```

#### Example Response

```json
{
  "status": "success",
  "data": {
    "summary": {
      "total_errors": 25,
      "recurring_errors": 5,
      "resource_errors": 10,
      "network_errors": 8,
      "hardware_errors": 7,
      "critical_hardware_errors": 3,
      "time_range_hours": 24
    },
    "timestamp": "2025-03-15T14:30:45.123",
    "recent_errors": [...],
    "error_distribution": {...},
    "error_patterns": {...},
    "worker_errors": {...},
    "hardware_errors": {...}
  }
}
```

### WebSocket API

- **Endpoint**: `/ws/error-visualization`
- **Subscription Message**:
  ```json
  {
    "type": "error_visualization_init",
    "time_range": 24
  }
  ```
- **Topic Subscription Message**:
  ```json
  {
    "type": "subscribe",
    "topic": "error_visualization"
  }
  ```
- **Error Update Message**:
  ```json
  {
    "type": "error_visualization_update",
    "data": {
      "error": {...},
      "time_range": 24
    }
  }
  ```

#### WebSocket Client Example

```python
import anyio
import websockets
import json

async def monitor_errors():
    uri = "ws://localhost:8080/ws/error-visualization"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to error visualization updates
        await websocket.send(json.dumps({
            "type": "error_visualization_init",
            "time_range": 24
        }))
        
        # Subscribe to general error topic
        await websocket.send(json.dumps({
            "type": "subscribe",
            "topic": "error_visualization"
        }))
        
        # Process incoming messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "error_visualization_update":
                error = data.get("data", {}).get("error")
                if error:
                    print(f"New error: {error.get('type')} - {error.get('message')}")

# Run the WebSocket client
anyio.run(monitor_errors)
```

## Conclusion

The Error Visualization system provides a comprehensive solution for monitoring, analyzing, and visualizing errors in distributed testing environments. By following this implementation guide, you can integrate the system into your testing infrastructure and benefit from real-time error notifications, pattern detection, and detailed error analysis.

---

System Documentation: March 16, 2025
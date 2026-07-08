# Real-time Monitoring Dashboard

The Real-time Monitoring Dashboard provides a comprehensive and interactive interface for monitoring the Distributed Testing Framework's runtime health, worker nodes, task execution, hardware utilization, and network connectivity.

## Features

- **Cluster Status Overview**: Monitor critical metrics such as cluster health, active workers, total tasks, and success rate
- **Resource Usage Visualization**: Real-time charts for CPU and memory usage across the distributed testing cluster
- **Worker Node Management**: View, search, and filter worker nodes with detailed metrics and hardware capability indicators
- **Task Queue Monitoring**: Track and filter tasks by status (pending, running, completed, failed)
- **Network Connectivity Visualization**: Interactive D3.js network graph showing coordinator and worker connections
- **Hardware Availability**: Monitor available hardware types (CPU, CUDA, ROCm, WebGPU, etc.) across your worker fleet
- **WebSocket Integration**: True real-time updates via WebSocket with automatic fallback to polling if not available
- **Auto-refresh**: Configurable refresh intervals with real-time updates of all metrics
- **Interactive UI**: Search, filter, and drill down into specific information

## Screenshots

![Monitoring Dashboard Overview](../images/monitoring_dashboard_overview.png)

## Accessing the Dashboard

The Real-time Monitoring Dashboard is integrated with the existing Web Dashboard and can be accessed at:

```
http://localhost:8050/monitoring
```

Use the same credentials as the main dashboard to log in:
- Username: `admin` or `user`
- Password: `admin_password` or `user_password`

## Dashboard Components

### 1. Cluster Status Overview

The Cluster Status section provides at-a-glance metrics about your Distributed Testing Framework:

- **Cluster Health**: Overall health score with status indicator (healthy, warning, critical)
- **Active Workers**: Count of currently active worker nodes
- **Total Tasks**: Total number of tasks in the system
- **Success Rate**: Percentage of successfully completed tasks
- **Trend Indicators**: Show increase/decrease trends for key metrics

### 2. Resource Usage Charts

Interactive charts displaying critical resource usage metrics:

- **CPU Usage Chart**: Average and maximum CPU usage across workers over time
- **Memory Usage Chart**: Average and maximum memory usage across workers over time

### 3. Worker Nodes Management

Detailed view of all worker nodes in the system:

- **Search Functionality**: Search workers by ID, status, or hardware capabilities
- **Health Status Indicators**: Visual indicators for worker health (healthy, warning, critical)
- **Resource Metrics**: CPU usage, memory usage, tasks completed, and success rate
- **Hardware Capability Indicators**: Visual representation of available hardware types on each worker

### 4. Task Queue Visualization

Real-time view of the task queue:

- **Status Filtering**: Filter tasks by status (all, pending, running, completed, failed)
- **Priority Indicators**: Visual indicators for task priority
- **Task Details**: View task ID, type, status, and assigned worker

### 5. Network Connectivity Map

Interactive D3.js visualization showing the network topology:

- **Node Representation**: Coordinator and worker nodes with status indicators
- **Connection Quality**: Connection strength between nodes
- **Interactive Elements**: Drag nodes to better visualize the network structure

### 6. Hardware Availability Chart

Bar chart showing hardware availability across the worker fleet:

- **Hardware Types**: Visual breakdown of different hardware types (CPU, CUDA, ROCm, etc.)
- **Availability Metrics**: Count of available vs. total hardware resources

## API Endpoints

The Real-time Monitoring Dashboard is powered by the following API endpoints:

- `GET /api/monitoring/cluster`: Get cluster status metrics
- `GET /api/monitoring/workers`: Get worker node information
- `GET /api/monitoring/tasks`: Get task queue data with optional status filtering
- `GET /api/monitoring/resources`: Get CPU and memory usage trends
- `GET /api/monitoring/hardware`: Get hardware availability data
- `GET /api/monitoring/network`: Get network topology data

### Example API Responses

#### Cluster Status

```json
{
  "active_workers": 8,
  "total_tasks": 42,
  "success_rate": 95,
  "health": {
    "score": 98,
    "status": "healthy",
    "trend": {
      "direction": "up",
      "value": 2.3,
      "status": "up"
    }
  },
  "trends": {
    "workers": {
      "direction": "up",
      "value": 2,
      "status": "up"
    },
    "tasks": {
      "direction": "up",
      "value": 15,
      "status": "up"
    },
    "success_rate": {
      "direction": "up",
      "value": 2.5,
      "status": "up"
    }
  }
}
```

#### Workers Information

```json
[
  {
    "id": "worker-001",
    "status": "active",
    "health": "healthy",
    "cpu": 25,
    "memory": 1.2,
    "tasks_completed": 42,
    "success_rate": 95,
    "hardware": ["cpu", "cuda", "webgpu"]
  },
  {
    "id": "worker-002",
    "status": "active",
    "health": "warning",
    "cpu": 78,
    "memory": 3.5,
    "tasks_completed": 36,
    "success_rate": 87,
    "hardware": ["cpu", "cuda", "rocm"]
  }
]
```

## Frontend Technologies

The dashboard is built using the following technologies:

- **Chart.js**: For responsive, interactive charts
- **D3.js**: For the network connectivity visualization
- **Fetch API**: For data retrieval from backend endpoints
- **CSS Grid/Flexbox**: For responsive layout
- **FontAwesome**: For icons and visual indicators

## Integration with Existing Systems

The Real-time Monitoring Dashboard integrates with the Distributed Testing Coordinator and existing monitoring systems to collect and visualize data.

### Coordinator Integration

```python
from coordinator import DistributedTestingCoordinator
from monitoring import MonitoringSystem

# Initialize the coordinator
coordinator = DistributedTestingCoordinator()

# Initialize the monitoring system
monitoring_system = MonitoringSystem(coordinator=coordinator)

# Enable real-time metrics collection
monitoring_system.enable_real_time_metrics(
    collection_interval=10,  # seconds
    metrics=['cpu', 'memory', 'network', 'tasks', 'hardware']
)

# Start the monitoring system
monitoring_system.start()
```

## Customizing the Dashboard

The dashboard can be customized by modifying the following files:

- `/result_aggregator/templates/monitoring_dashboard.html`: Main template for the monitoring dashboard
- `/result_aggregator/static/css/monitoring.css`: CSS styles for the monitoring dashboard
- `/result_aggregator/web_dashboard.py`: API endpoints for monitoring data

## WebSocket Integration

The dashboard now supports WebSocket for real-time updates, providing a more efficient and responsive experience compared to polling:

```javascript
// Check if WebSocket is available
if (typeof io !== 'undefined') {
  // WebSocket is available, use it for real-time updates
  setupWebSocketConnection();
} else {
  // WebSocket not available, fall back to polling
  fetchDataWithPolling();
}

// Set up WebSocket connection for real-time updates
function setupWebSocketConnection() {
  const socket = io();
  
  socket.on('connect', function() {
    console.log('Connected to SocketIO for monitoring data');
    
    // Subscribe to monitoring updates
    socket.emit('subscribe_to_monitoring', {});
    
    // Update UI to show WebSocket is active
    document.querySelector('.auto-refresh').innerHTML = `
      <span class="websocket-indicator">
        <i class="fas fa-bolt"></i> WebSocket Connected
      </span>
    `;
  });
  
  // Handle monitoring updates from server
  socket.on('monitoring_update', function(data) {
    // Update all dashboard components with a single data package
    if (data.cluster) updateClusterOverview();
    if (data.workers) renderWorkerList(data.workers);
    if (data.tasks) renderTaskQueue(data.tasks);
    if (data.resources) updateResourceCharts();
    if (data.hardware) updateHardwareChart();
    if (data.network) updateNetworkMap(data.network);
  });
  
  // Request immediate updates when refresh button is clicked
  socket.on('request_monitoring_update', function() {
    socket.emit('request_monitoring_update', {});
  });
  
  return socket;
}
```

### WebSocket Server Implementation

On the server side, WebSocket connections are handled using Flask-SocketIO:

```python
if SOCKETIO_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle SocketIO connection."""
        logger.info(f"Client connected: {request.sid}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle SocketIO disconnect."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_to_monitoring')
    def handle_subscribe_monitoring(data):
        """Handle monitoring subscription."""
        logger.info(f"Client {request.sid} subscribed to monitoring updates")
        # Join a room for monitoring updates
        from flask_socketio import join_room
        join_room('monitoring_subscribers')
        # Send initial data
        emit_monitoring_data()
    
    @socketio.on('request_monitoring_update')
    def handle_request_monitoring_update(data):
        """Handle request for immediate monitoring data update."""
        logger.info(f"Client {request.sid} requested monitoring data update")
        # Send updated data
        emit_monitoring_data()
```

### Automatic Data Broadcasting

A background thread continuously updates connected clients:

```python
def background_monitoring_thread(interval=5):
    """Background thread for emitting monitoring data periodically."""
    logger.info(f"Starting background monitoring thread with interval {interval} seconds")
    
    while True:
        try:
            # Emit monitoring data to all subscribed clients
            emit_monitoring_data()
            # Sleep for the interval
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error in background monitoring thread: {e}")
            time.sleep(5)  # Sleep and retry after error

# Start background monitoring thread if SocketIO is available
if SOCKETIO_AVAILABLE:
    monitoring_thread = threading.Thread(
        target=background_monitoring_thread,
        args=(args.update_interval,),
        daemon=True
    )
    monitoring_thread.start()
```

### Automatic Fallback to Polling

The dashboard will automatically fall back to a polling-based approach if WebSocket is not available or disconnects:

```javascript
socket.on('disconnect', function() {
  console.log('Disconnected from SocketIO, falling back to polling');
  
  // Dispatch custom event to notify of WebSocket disconnection
  window.dispatchEvent(new CustomEvent('websocket-disconnected'));
  
  // Fall back to polling
  const autoRefreshCheckbox = document.getElementById('auto-refresh');
  if (autoRefreshCheckbox && autoRefreshCheckbox.checked) {
    fetchDataWithPolling();
  }
});
```

## Performance Considerations

For optimal performance when running the Real-time Monitoring Dashboard:

1. **Adjust Refresh Interval**: Set the auto-refresh interval based on your cluster size (larger clusters may benefit from less frequent refreshes)
2. **Limit Data Volume**: The API endpoints include pagination to limit data volume
3. **Server Resources**: Ensure the server running the dashboard has adequate CPU and memory resources
4. **Database Optimization**: Consider optimizing your database for read-heavy workloads

## Future Enhancements

Planned enhancements for the Real-time Monitoring Dashboard include:

1. ✅ **WebSocket Support**: Replace polling with WebSocket for real-time updates (Completed)
2. **Customizable Dashboard**: User-configurable dashboard layouts
3. **Historical View**: View historical performance with time range selection
4. **Alert Configuration**: Configure custom alerts based on specific metrics
5. **Mobile Optimization**: Enhanced mobile-specific views for on-the-go monitoring

## Troubleshooting

### Dashboard Not Showing Real-time Updates

- Check if WebSocket is connected (you should see "WebSocket Connected" indicator)
- If using polling fallback, check if auto-refresh is enabled
- Verify the refresh interval is appropriate
- Check browser console for JavaScript errors
- Ensure the API endpoints are responding correctly
- Verify that Flask-SocketIO is installed (`pip install flask-socketio`)

### Resource Charts Not Updating

- Verify that Chart.js is loaded correctly
- Check if resource data is being successfully retrieved from the API
- Check browser console for any errors during chart updates

### Network Map Not Rendering Properly

- Ensure D3.js is loaded correctly
- Verify the network data format matches what D3.js expects
- Check for JavaScript errors in the browser console
- Try increasing the SVG dimensions if the map appears crowded

### Performance Issues

- Increase the refresh interval to reduce server load
- Reduce the number of data points in resource charts
- Use filtering to limit the number of workers and tasks displayed
- Consider enabling database query caching

## Related Documentation

- [Web Dashboard Guide](WEB_DASHBOARD_GUIDE.md) - Main documentation for the web dashboard
- [Result Aggregation Guide](RESULT_AGGREGATION_GUIDE.md) - Documentation for the result aggregation system
- [High Availability Clustering](../README_AUTO_RECOVERY.md#high-availability-clustering-new---march-2025) - Information about the High Availability Clustering feature visualized in the dashboard
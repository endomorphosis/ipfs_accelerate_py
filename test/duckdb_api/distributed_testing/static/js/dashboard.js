/**
 * Distributed Testing Framework - Dashboard JavaScript
 * Main JavaScript file for the monitoring dashboard
 */

// Global variables
let ws;
let refreshInterval = 5000; // 5 seconds default
let autoRefresh = true;
let refreshTimer;
let pendingAlerts = 0;
let chartInstances = {};
let lastMetrics = {};
let darkMode = false;

// Dashboard initialization
document.addEventListener('DOMContentLoaded', function() {
    // Connect to WebSocket
    connectWebSocket();

    // Setup event listeners
    setupEventListeners();

    // Check for saved preferences
    loadUserPreferences();
    
    // Start auto-refresh if enabled
    if (autoRefresh) {
        startAutoRefresh();
    }

    // Initial refresh
    refreshDashboard();
});

/**
 * Connect to WebSocket server
 */
function connectWebSocket() {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
        updateConnectionStatus(true);
        
        // Request initial data
        requestInitialData();
    };
    
    ws.onmessage = function(event) {
        handleWebSocketMessage(event.data);
    };
    
    ws.onclose = function() {
        console.log('WebSocket connection closed');
        updateConnectionStatus(false);
        
        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false, 'Error');
    };
}

/**
 * Update the connection status indicator
 * @param {boolean} connected - Whether the connection is established
 * @param {string} status - Optional status text
 */
function updateConnectionStatus(connected, status) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    if (connected) {
        statusIndicator.className = 'status-indicator status-active';
        statusText.textContent = 'Connected';
    } else {
        statusIndicator.className = 'status-indicator status-inactive';
        statusText.textContent = status || 'Disconnected';
    }
}

/**
 * Request initial data from the server
 */
function requestInitialData() {
    if (ws.readyState === WebSocket.OPEN) {
        // Request metrics
        ws.send(JSON.stringify({ type: 'get_metrics' }));
        
        // Request workers
        ws.send(JSON.stringify({ type: 'get_workers' }));
        
        // Request tasks
        ws.send(JSON.stringify({ type: 'get_tasks' }));
        
        // Request alerts
        ws.send(JSON.stringify({ type: 'get_alerts' }));
        
        // Request visualizations
        ws.send(JSON.stringify({ type: 'get_visualization' }));
    }
}

/**
 * Handle WebSocket messages
 * @param {string} data - JSON data from server
 */
function handleWebSocketMessage(data) {
    const message = JSON.parse(data);
    
    switch (message.type) {
        case 'metrics_update':
            updateMetrics(message.metrics);
            break;
        
        case 'workers_update':
            updateWorkers(message.workers);
            break;
        
        case 'tasks_update':
            updateTasks(message.tasks);
            break;
        
        case 'alerts_update':
            updateAlerts(message.alerts, message.count);
            break;
        
        case 'visualization_update':
            updateVisualizations(message.data, message.visualization);
            break;
        
        case 'coordinator_message':
            // Handle messages from coordinator
            handleCoordinatorMessage(message.message);
            break;
        
        case 'error':
            console.error('Error from server:', message.message);
            showToast('Error', message.message, 'error');
            break;
    }
}

/**
 * Update dashboard metrics
 * @param {Object} metrics - Metrics data
 */
function updateMetrics(metrics) {
    lastMetrics = {...lastMetrics, ...metrics};
    
    // Update metric cards with animation
    updateMetricWithAnimation('workerCount', metrics.worker_count);
    updateMetricWithAnimation('taskCount', metrics.task_count);
    updateMetricWithAnimation('completedTaskCount', metrics.completed_task_count);
    updateMetricWithAnimation('errorCount', metrics.error_count);
    
    // Update resource utilization if available
    if (metrics.resource_utilization) {
        updateResourceUtilization(metrics.resource_utilization);
    }
}

/**
 * Update metric with animation
 * @param {string} elementId - Element ID to update
 * @param {number} newValue - New value
 */
function updateMetricWithAnimation(elementId, newValue) {
    if (newValue === undefined) return;
    
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const currentValue = parseInt(element.textContent) || 0;
    if (currentValue === newValue) return;
    
    // Add animation class
    element.classList.add('updating');
    
    // Animate the number
    const duration = 1000; // 1 second
    const steps = 20;
    const stepDuration = duration / steps;
    const increment = (newValue - currentValue) / steps;
    
    let currentStep = 0;
    
    const updateStep = () => {
        currentStep++;
        const stepValue = Math.round(currentValue + (increment * currentStep));
        element.textContent = stepValue;
        
        if (currentStep < steps) {
            setTimeout(updateStep, stepDuration);
        } else {
            element.textContent = newValue;
            // Remove animation class
            setTimeout(() => {
                element.classList.remove('updating');
            }, 300);
        }
    };
    
    updateStep();
}

/**
 * Update workers table
 * @param {Array} workers - Workers data
 */
function updateWorkers(workers) {
    const tbody = document.querySelector('#workersTable tbody');
    tbody.innerHTML = '';
    
    if (!workers || workers.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 6;
        td.className = 'text-center text-muted py-4';
        td.textContent = 'No workers available';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }
    
    workers.forEach(worker => {
        const tr = document.createElement('tr');
        
        // Worker ID
        const tdId = document.createElement('td');
        tdId.textContent = worker.entity_id;
        tr.appendChild(tdId);
        
        // Worker Name
        const tdName = document.createElement('td');
        tdName.textContent = worker.entity_name;
        tr.appendChild(tdName);
        
        // Status
        const tdStatus = document.createElement('td');
        const statusIndicator = document.createElement('span');
        statusIndicator.className = `status-indicator status-${worker.status.toLowerCase()}`;
        tdStatus.appendChild(statusIndicator);
        tdStatus.appendChild(document.createTextNode(capitalizeFirstLetter(worker.status)));
        tr.appendChild(tdStatus);
        
        // Hardware
        const tdHardware = document.createElement('td');
        if (worker.entity_data && worker.entity_data.capabilities) {
            tdHardware.textContent = worker.entity_data.capabilities.join(', ');
        } else {
            tdHardware.textContent = 'N/A';
        }
        tr.appendChild(tdHardware);
        
        // Tasks
        const tdTasks = document.createElement('td');
        if (worker.entity_data && worker.entity_data.task_count !== undefined) {
            tdTasks.textContent = worker.entity_data.task_count;
        } else {
            tdTasks.textContent = '0';
        }
        tr.appendChild(tdTasks);
        
        // Actions
        const tdActions = document.createElement('td');
        
        // View button
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-sm btn-primary me-1';
        viewBtn.innerHTML = '<i class="fa fa-eye"></i>';
        viewBtn.title = 'View Worker Details';
        viewBtn.addEventListener('click', () => viewWorkerDetails(worker));
        tdActions.appendChild(viewBtn);
        
        // Reset button (only for inactive workers)
        if (worker.status.toLowerCase() === 'inactive' || worker.status.toLowerCase() === 'error') {
            const resetBtn = document.createElement('button');
            resetBtn.className = 'btn btn-sm btn-warning me-1';
            resetBtn.innerHTML = '<i class="fa fa-refresh"></i>';
            resetBtn.title = 'Reset Worker';
            resetBtn.addEventListener('click', () => resetWorker(worker.entity_id));
            tdActions.appendChild(resetBtn);
        }
        
        // Remove button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn btn-sm btn-danger';
        removeBtn.innerHTML = '<i class="fa fa-times"></i>';
        removeBtn.title = 'Remove Worker';
        removeBtn.addEventListener('click', () => removeWorker(worker.entity_id));
        tdActions.appendChild(removeBtn);
        
        tr.appendChild(tdActions);
        
        tbody.appendChild(tr);
    });
}

/**
 * Update tasks table
 * @param {Array} tasks - Tasks data
 */
function updateTasks(tasks) {
    const tbody = document.querySelector('#tasksTable tbody');
    tbody.innerHTML = '';
    
    if (!tasks || tasks.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 7;
        td.className = 'text-center text-muted py-4';
        td.textContent = 'No tasks available';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }
    
    tasks.forEach(task => {
        const tr = document.createElement('tr');
        
        // Task ID
        const tdId = document.createElement('td');
        tdId.textContent = task.task_id;
        tr.appendChild(tdId);
        
        // Type
        const tdType = document.createElement('td');
        tdType.textContent = task.task_type;
        tr.appendChild(tdType);
        
        // Status
        const tdStatus = document.createElement('td');
        const statusIndicator = document.createElement('span');
        statusIndicator.className = `status-indicator status-${task.status.toLowerCase()}`;
        tdStatus.appendChild(statusIndicator);
        tdStatus.appendChild(document.createTextNode(capitalizeFirstLetter(task.status)));
        tr.appendChild(tdStatus);
        
        // Worker
        const tdWorker = document.createElement('td');
        tdWorker.textContent = task.worker_id || 'N/A';
        tr.appendChild(tdWorker);
        
        // Created
        const tdCreated = document.createElement('td');
        tdCreated.textContent = formatDateTime(task.created_at);
        tr.appendChild(tdCreated);
        
        // Execution Time
        const tdExecutionTime = document.createElement('td');
        if (task.execution_time) {
            tdExecutionTime.textContent = `${task.execution_time.toFixed(2)}s`;
        } else {
            tdExecutionTime.textContent = 'N/A';
        }
        tr.appendChild(tdExecutionTime);
        
        // Actions
        const tdActions = document.createElement('td');
        
        // View button
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-sm btn-primary me-1';
        viewBtn.innerHTML = '<i class="fa fa-eye"></i>';
        viewBtn.title = 'View Task Details';
        viewBtn.addEventListener('click', () => viewTaskDetails(task));
        tdActions.appendChild(viewBtn);
        
        // Cancel button (for pending or running tasks)
        if (task.status.toLowerCase() === 'pending' || task.status.toLowerCase() === 'running') {
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-sm btn-danger';
            cancelBtn.innerHTML = '<i class="fa fa-times"></i>';
            cancelBtn.title = 'Cancel Task';
            cancelBtn.addEventListener('click', () => cancelTask(task.task_id));
            tdActions.appendChild(cancelBtn);
        }
        
        // Retry button (for failed tasks)
        if (task.status.toLowerCase() === 'failed') {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'btn btn-sm btn-warning';
            retryBtn.innerHTML = '<i class="fa fa-repeat"></i>';
            retryBtn.title = 'Retry Task';
            retryBtn.addEventListener('click', () => retryTask(task.task_id));
            tdActions.appendChild(retryBtn);
        }
        
        tr.appendChild(tdActions);
        
        tbody.appendChild(tr);
    });
}

/**
 * Update alerts
 * @param {Array} alerts - Alerts data
 * @param {number} count - Number of pending alerts
 */
function updateAlerts(alerts, count) {
    // Update alerts badge
    const alertsBadge = document.getElementById('alertsBadge');
    
    if (count > 0) {
        alertsBadge.textContent = count;
        alertsBadge.style.display = 'block';
        pendingAlerts = count;
        
        // Show notification if new alerts appeared
        if (pendingAlerts > 0 && alerts && alerts.length > 0) {
            const latestAlert = alerts[0]; // Assuming alerts are sorted by timestamp
            showToast('Alert', latestAlert.alert_message, latestAlert.severity);
        }
    } else {
        alertsBadge.style.display = 'none';
        pendingAlerts = 0;
    }
    
    // Update alerts table if modal is open
    if (document.getElementById('alertsModal').classList.contains('show')) {
        loadAlerts(alerts);
    }
}

/**
 * Load alerts into alerts table
 * @param {Array} alerts - Alerts data
 */
function loadAlerts(alerts) {
    if (!alerts) {
        // Request alerts from server
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'get_alerts' }));
        }
        return;
    }
    
    const tbody = document.querySelector('#alertsTable tbody');
    tbody.innerHTML = '';
    
    if (!alerts || alerts.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 6;
        td.className = 'text-center text-muted py-4';
        td.textContent = 'No alerts available';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }
    
    alerts.forEach(alert => {
        const tr = document.createElement('tr');
        
        // Time
        const tdTime = document.createElement('td');
        tdTime.textContent = formatDateTime(alert.created_at);
        tr.appendChild(tdTime);
        
        // Type
        const tdType = document.createElement('td');
        tdType.textContent = alert.alert_type;
        tr.appendChild(tdType);
        
        // Severity
        const tdSeverity = document.createElement('td');
        let severityClass = '';
        switch (alert.severity) {
            case 'critical':
                severityClass = 'badge bg-danger';
                break;
            case 'warning':
                severityClass = 'badge bg-warning text-dark';
                break;
            case 'info':
                severityClass = 'badge bg-info text-dark';
                break;
            default:
                severityClass = 'badge bg-secondary';
        }
        const severityBadge = document.createElement('span');
        severityBadge.className = severityClass;
        severityBadge.textContent = alert.severity;
        tdSeverity.appendChild(severityBadge);
        tr.appendChild(tdSeverity);
        
        // Message
        const tdMessage = document.createElement('td');
        tdMessage.textContent = alert.alert_message;
        tr.appendChild(tdMessage);
        
        // Entity
        const tdEntity = document.createElement('td');
        if (alert.entity_id) {
            tdEntity.textContent = `${alert.entity_type || ''} ${alert.entity_id}`;
        } else {
            tdEntity.textContent = 'N/A';
        }
        tr.appendChild(tdEntity);
        
        // Actions
        const tdActions = document.createElement('td');
        
        if (alert.is_active) {
            if (!alert.is_acknowledged) {
                const ackBtn = document.createElement('button');
                ackBtn.className = 'btn btn-sm btn-primary me-1';
                ackBtn.innerHTML = '<i class="fa fa-check"></i>';
                ackBtn.title = 'Acknowledge Alert';
                ackBtn.addEventListener('click', () => acknowledgeAlert(alert.id));
                tdActions.appendChild(ackBtn);
            }
            
            const resolveBtn = document.createElement('button');
            resolveBtn.className = 'btn btn-sm btn-success';
            resolveBtn.innerHTML = '<i class="fa fa-check-circle"></i>';
            resolveBtn.title = 'Resolve Alert';
            resolveBtn.addEventListener('click', () => resolveAlert(alert.id));
            tdActions.appendChild(resolveBtn);
        } else {
            const resolvedText = document.createElement('span');
            resolvedText.className = 'badge bg-success';
            resolvedText.textContent = 'Resolved';
            tdActions.appendChild(resolvedText);
        }
        
        tr.appendChild(tdActions);
        
        tbody.appendChild(tr);
    });
}

/**
 * Acknowledge an alert
 * @param {string} alertId - Alert ID
 */
function acknowledgeAlert(alertId) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'acknowledge_alert',
            alert_id: alertId
        }));
        showToast('Alert', 'Alert acknowledged', 'info');
    }
}

/**
 * Resolve an alert
 * @param {string} alertId - Alert ID
 */
function resolveAlert(alertId) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'resolve_alert',
            alert_id: alertId
        }));
        showToast('Alert', 'Alert resolved', 'success');
    }
}

/**
 * Update visualizations
 * @param {Object} data - Visualization data
 * @param {string} vizName - Visualization name
 */
function updateVisualizations(data, vizName) {
    if (vizName === 'all') {
        // Update all visualizations
        Object.entries(data).forEach(([name, vizData]) => {
            updateVisualization(name, vizData);
        });
    } else {
        // Update specific visualization
        updateVisualization(vizName, data);
    }
}

/**
 * Update specific visualization
 * @param {string} name - Visualization name
 * @param {Object} data - Visualization data
 */
function updateVisualization(name, data) {
    if (!data) return;
    
    const containerId = getVisualizationContainerId(name);
    if (!containerId) return;
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Show loading overlay
    showLoadingOverlay(containerId);
    
    setTimeout(() => {
        if (data.type === 'none') {
            container.innerHTML = `<div class="alert alert-info">${data.message || 'No data available'}</div>`;
            hideLoadingOverlay(containerId);
            return;
        }
        
        if (data.type === 'plotly') {
            try {
                const figure = JSON.parse(data.data);
                Plotly.react(container, figure.data, figure.layout, {responsive: true})
                    .catch(err => {
                        console.error('Error updating chart:', err);
                        container.innerHTML = `<div class="alert alert-danger">Error rendering chart: ${err.message}</div>`;
                    })
                    .finally(() => {
                        hideLoadingOverlay(containerId);
                    });
            } catch (err) {
                console.error('Error parsing chart data:', err);
                container.innerHTML = `<div class="alert alert-danger">Error parsing chart data: ${err.message}</div>`;
                hideLoadingOverlay(containerId);
            }
        } else if (data.type === 'plotly_multi') {
            // Multiple charts
            try {
                const charts = data.charts || [];
                
                const promises = charts.map(chart => {
                    const chartName = chart.name;
                    const chartData = JSON.parse(chart.data);
                    const chartContainerId = getVisualizationContainerId(chartName);
                    
                    if (chartContainerId) {
                        const chartContainer = document.getElementById(chartContainerId);
                        if (chartContainer) {
                            return Plotly.react(chartContainer, chartData.data, chartData.layout, {responsive: true});
                        }
                    }
                    return Promise.resolve();
                });
                
                Promise.all(promises)
                    .catch(err => {
                        console.error('Error updating charts:', err);
                    })
                    .finally(() => {
                        hideLoadingOverlay(containerId);
                    });
            } catch (err) {
                console.error('Error parsing multi-chart data:', err);
                hideLoadingOverlay(containerId);
            }
        }
    }, 300); // Small delay for animation
}

/**
 * Get container ID for visualization
 * @param {string} name - Visualization name
 * @returns {string} Container ID
 */
function getVisualizationContainerId(name) {
    const mappings = {
        'worker_status_chart': 'workerStatusChart',
        'task_status_chart': 'taskStatusChart',
        'task_throughput_chart': 'taskThroughputChart',
        'execution_time_by_type': 'taskExecutionTimeChart',
        'execution_time_trend': 'taskExecutionTimeTrendChart',
        'resource_utilization_chart': 'resourceUtilizationChart',
        'error_distribution_chart': 'errorDistributionChart'
    };
    
    return mappings[name] || null;
}

/**
 * Handle coordinator messages
 * @param {Object} message - Coordinator message
 */
function handleCoordinatorMessage(message) {
    // Handle different message types from coordinator
    const messageType = message.type;
    
    switch (messageType) {
        case 'worker_update':
            // Refresh workers
            ws.send(JSON.stringify({ type: 'get_workers' }));
            break;
        
        case 'task_update':
            // Refresh tasks
            ws.send(JSON.stringify({ type: 'get_tasks' }));
            // Refresh metrics
            ws.send(JSON.stringify({ type: 'get_metrics' }));
            break;
        
        case 'error':
            // Refresh alerts
            ws.send(JSON.stringify({ type: 'get_alerts' }));
            // Show notification
            showToast('Error', message.message, 'error');
            break;
        
        case 'resource_usage':
            // Request resource utilization chart update
            ws.send(JSON.stringify({
                type: 'get_visualization',
                visualization: 'resource_utilization_chart'
            }));
            break;
            
        case 'notification':
            // Show notification
            showToast(message.title || 'Notification', message.message, message.level || 'info');
            break;
    }
}

/**
 * Refresh the dashboard
 */
function refreshDashboard() {
    // Request updated data from server
    if (ws.readyState === WebSocket.OPEN) {
        // Show loading indicator
        document.getElementById('refreshBtn').classList.add('rotating');
        
        // Request metrics
        ws.send(JSON.stringify({ type: 'get_metrics' }));
        
        // Request visualizations
        ws.send(JSON.stringify({ type: 'get_visualization' }));
        
        // Request specific data based on active tab
        const activeTab = document.querySelector('.nav-link.active').getAttribute('id');
        
        switch (activeTab) {
            case 'overview-tab':
                // Already covered by visualizations
                break;
            
            case 'workers-tab':
                ws.send(JSON.stringify({ type: 'get_workers' }));
                break;
            
            case 'tasks-tab':
                ws.send(JSON.stringify({ type: 'get_tasks' }));
                break;
            
            case 'performance-tab':
                ws.send(JSON.stringify({
                    type: 'get_visualization',
                    visualization: 'task_execution_time_chart'
                }));
                break;
            
            case 'resources-tab':
                ws.send(JSON.stringify({
                    type: 'get_visualization',
                    visualization: 'resource_utilization_chart'
                }));
                break;
            
            case 'events-tab':
                ws.send(JSON.stringify({
                    type: 'get_visualization',
                    visualization: 'error_distribution_chart'
                }));
                break;
        }
        
        // Hide loading indicator after a short delay
        setTimeout(() => {
            document.getElementById('refreshBtn').classList.remove('rotating');
        }, 1000);
    }
}

/**
 * Start auto-refresh timer
 */
function startAutoRefresh() {
    // Stop any existing timer
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    
    // Start new timer
    refreshTimer = setInterval(refreshDashboard, refreshInterval);
}

/**
 * Save dashboard settings
 */
function saveSettings() {
    // Get settings from form
    const newRefreshInterval = parseInt(document.getElementById('refreshInterval').value) * 1000;
    const newAutoRefresh = document.getElementById('autoRefresh').checked;
    const newCoordinatorUrl = document.getElementById('coordinatorUrl').value;
    const newDarkMode = document.getElementById('darkMode').checked;
    
    // Update settings
    refreshInterval = newRefreshInterval;
    autoRefresh = newAutoRefresh;
    
    // Update dark mode
    if (darkMode !== newDarkMode) {
        darkMode = newDarkMode;
        toggleDarkMode(darkMode);
    }
    
    // TODO: Update coordinator URL (would require server-side changes)
    
    // Restart auto-refresh if enabled
    if (autoRefresh) {
        startAutoRefresh();
    } else {
        // Stop auto-refresh
        if (refreshTimer) {
            clearInterval(refreshTimer);
            refreshTimer = null;
        }
    }
    
    // Save preferences to localStorage
    saveUserPreferences();
    
    // Close modal
    const settingsModal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
    settingsModal.hide();
    
    // Show confirmation
    showToast('Settings', 'Settings saved successfully', 'success');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        refreshDashboard();
    });

    // Alerts button
    document.getElementById('alertsBtn').addEventListener('click', function() {
        loadAlerts();
        const alertsModal = new bootstrap.Modal(document.getElementById('alertsModal'));
        alertsModal.show();
    });

    // Settings button
    document.getElementById('settingsBtn').addEventListener('click', function() {
        // Update form with current settings
        document.getElementById('refreshInterval').value = refreshInterval / 1000;
        document.getElementById('autoRefresh').checked = autoRefresh;
        document.getElementById('darkMode').checked = darkMode;
        
        const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));
        settingsModal.show();
    });

    // Save settings button
    document.getElementById('saveSettingsBtn').addEventListener('click', function() {
        saveSettings();
    });
    
    // Tab change event
    const tabs = document.querySelectorAll('.nav-link[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            // Refresh data for the active tab
            refreshDashboard();
        });
    });
}

/**
 * View worker details
 * @param {Object} worker - Worker data
 */
function viewWorkerDetails(worker) {
    // Create modal for worker details
    const modalHtml = `
        <div class="modal fade" id="workerDetailsModal" tabindex="-1" aria-labelledby="workerDetailsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="workerDetailsModalLabel">Worker Details: ${worker.entity_name}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <dl class="row">
                                    <dt class="col-sm-4">ID:</dt>
                                    <dd class="col-sm-8">${worker.entity_id}</dd>
                                    
                                    <dt class="col-sm-4">Name:</dt>
                                    <dd class="col-sm-8">${worker.entity_name}</dd>
                                    
                                    <dt class="col-sm-4">Status:</dt>
                                    <dd class="col-sm-8">
                                        <span class="status-indicator status-${worker.status.toLowerCase()}"></span>
                                        ${capitalizeFirstLetter(worker.status)}
                                    </dd>
                                    
                                    <dt class="col-sm-4">Tasks:</dt>
                                    <dd class="col-sm-8">${worker.entity_data?.task_count || 0}</dd>
                                </dl>
                            </div>
                            <div class="col-md-6">
                                <dl class="row">
                                    <dt class="col-sm-4">Host:</dt>
                                    <dd class="col-sm-8">${worker.entity_data?.host || 'N/A'}</dd>
                                    
                                    <dt class="col-sm-4">Port:</dt>
                                    <dd class="col-sm-8">${worker.entity_data?.port || 'N/A'}</dd>
                                    
                                    <dt class="col-sm-4">Last Seen:</dt>
                                    <dd class="col-sm-8">${formatDateTime(worker.entity_data?.last_seen || Date.now())}</dd>
                                    
                                    <dt class="col-sm-4">Version:</dt>
                                    <dd class="col-sm-8">${worker.entity_data?.version || 'N/A'}</dd>
                                </dl>
                            </div>
                        </div>
                        
                        <h6 class="fw-bold mb-3">Capabilities</h6>
                        <ul class="list-group mb-4">
                            ${(worker.entity_data?.capabilities || []).map(cap => `
                                <li class="list-group-item">${cap}</li>
                            `).join('')}
                            ${!(worker.entity_data?.capabilities?.length) ? '<li class="list-group-item text-muted">No capabilities reported</li>' : ''}
                        </ul>
                        
                        <h6 class="fw-bold mb-3">Resource Usage</h6>
                        <div class="progress mb-3" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" style="width: ${worker.entity_data?.resource_usage?.cpu || 0}%;" 
                                aria-valuenow="${worker.entity_data?.resource_usage?.cpu || 0}" aria-valuemin="0" aria-valuemax="100">
                                CPU: ${worker.entity_data?.resource_usage?.cpu || 0}%
                            </div>
                        </div>
                        <div class="progress mb-3" style="height: 20px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${worker.entity_data?.resource_usage?.memory || 0}%;" 
                                aria-valuenow="${worker.entity_data?.resource_usage?.memory || 0}" aria-valuemin="0" aria-valuemax="100">
                                Memory: ${worker.entity_data?.resource_usage?.memory || 0}%
                            </div>
                        </div>
                        
                        ${worker.entity_data?.errors?.length ? `
                            <h6 class="fw-bold mb-3">Recent Errors</h6>
                            <div class="alert alert-danger">
                                <ul class="mb-0">
                                    ${worker.entity_data.errors.map(err => `<li>${err.message}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        ${worker.status.toLowerCase() === 'inactive' || worker.status.toLowerCase() === 'error' ? 
                            `<button type="button" class="btn btn-warning" id="resetWorkerBtn">Reset Worker</button>` : ''}
                        <button type="button" class="btn btn-danger" id="removeWorkerBtn">Remove Worker</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to document
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer);
    
    // Initialize modal
    const modal = new bootstrap.Modal(document.getElementById('workerDetailsModal'));
    modal.show();
    
    // Add event listeners
    const resetWorkerBtn = document.getElementById('resetWorkerBtn');
    if (resetWorkerBtn) {
        resetWorkerBtn.addEventListener('click', function() {
            resetWorker(worker.entity_id);
            modal.hide();
        });
    }
    
    const removeWorkerBtn = document.getElementById('removeWorkerBtn');
    removeWorkerBtn.addEventListener('click', function() {
        removeWorker(worker.entity_id);
        modal.hide();
    });
    
    // Clean up when modal is hidden
    document.getElementById('workerDetailsModal').addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modalContainer);
    });
}

/**
 * View task details
 * @param {Object} task - Task data
 */
function viewTaskDetails(task) {
    // Create modal for task details
    const modalHtml = `
        <div class="modal fade" id="taskDetailsModal" tabindex="-1" aria-labelledby="taskDetailsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="taskDetailsModalLabel">Task Details: ${task.task_id}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <dl class="row">
                                    <dt class="col-sm-4">ID:</dt>
                                    <dd class="col-sm-8">${task.task_id}</dd>
                                    
                                    <dt class="col-sm-4">Type:</dt>
                                    <dd class="col-sm-8">${task.task_type}</dd>
                                    
                                    <dt class="col-sm-4">Status:</dt>
                                    <dd class="col-sm-8">
                                        <span class="status-indicator status-${task.status.toLowerCase()}"></span>
                                        ${capitalizeFirstLetter(task.status)}
                                    </dd>
                                    
                                    <dt class="col-sm-4">Worker:</dt>
                                    <dd class="col-sm-8">${task.worker_id || 'N/A'}</dd>
                                </dl>
                            </div>
                            <div class="col-md-6">
                                <dl class="row">
                                    <dt class="col-sm-4">Created:</dt>
                                    <dd class="col-sm-8">${formatDateTime(task.created_at)}</dd>
                                    
                                    <dt class="col-sm-4">Started:</dt>
                                    <dd class="col-sm-8">${task.started_at ? formatDateTime(task.started_at) : 'N/A'}</dd>
                                    
                                    <dt class="col-sm-4">Completed:</dt>
                                    <dd class="col-sm-8">${task.completed_at ? formatDateTime(task.completed_at) : 'N/A'}</dd>
                                    
                                    <dt class="col-sm-4">Execution Time:</dt>
                                    <dd class="col-sm-8">${task.execution_time ? `${task.execution_time.toFixed(2)}s` : 'N/A'}</dd>
                                </dl>
                            </div>
                        </div>
                        
                        <h6 class="fw-bold mb-3">Task Parameters</h6>
                        <div class="card mb-4">
                            <div class="card-body">
                                <pre class="mb-0"><code>${syntaxHighlight(JSON.stringify(task.parameters || {}, null, 2))}</code></pre>
                            </div>
                        </div>
                        
                        ${task.result ? `
                            <h6 class="fw-bold mb-3">Task Result</h6>
                            <div class="card mb-4">
                                <div class="card-body">
                                    <pre class="mb-0"><code>${syntaxHighlight(JSON.stringify(task.result, null, 2))}</code></pre>
                                </div>
                            </div>
                        ` : ''}
                        
                        ${task.error ? `
                            <h6 class="fw-bold mb-3">Task Error</h6>
                            <div class="alert alert-danger">
                                <h6 class="alert-heading">${task.error.type || 'Error'}</h6>
                                <p class="mb-0">${task.error.message || 'Unknown error'}</p>
                                ${task.error.stack ? `<pre class="mt-2 mb-0 small">${task.error.stack}</pre>` : ''}
                            </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        ${task.status.toLowerCase() === 'pending' || task.status.toLowerCase() === 'running' ? 
                            `<button type="button" class="btn btn-danger" id="cancelTaskBtn">Cancel Task</button>` : ''}
                        ${task.status.toLowerCase() === 'failed' ? 
                            `<button type="button" class="btn btn-warning" id="retryTaskBtn">Retry Task</button>` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to document
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer);
    
    // Initialize modal
    const modal = new bootstrap.Modal(document.getElementById('taskDetailsModal'));
    modal.show();
    
    // Add event listeners
    const cancelTaskBtn = document.getElementById('cancelTaskBtn');
    if (cancelTaskBtn) {
        cancelTaskBtn.addEventListener('click', function() {
            cancelTask(task.task_id);
            modal.hide();
        });
    }
    
    const retryTaskBtn = document.getElementById('retryTaskBtn');
    if (retryTaskBtn) {
        retryTaskBtn.addEventListener('click', function() {
            retryTask(task.task_id);
            modal.hide();
        });
    }
    
    // Clean up when modal is hidden
    document.getElementById('taskDetailsModal').addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modalContainer);
    });
}

/**
 * Reset a worker
 * @param {string} workerId - Worker ID
 */
function resetWorker(workerId) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'reset_worker',
            worker_id: workerId
        }));
        showToast('Worker', 'Reset request sent', 'info');
    }
}

/**
 * Remove a worker
 * @param {string} workerId - Worker ID
 */
function removeWorker(workerId) {
    if (confirm('Are you sure you want to remove this worker?')) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'remove_worker',
                worker_id: workerId
            }));
            showToast('Worker', 'Remove request sent', 'info');
        }
    }
}

/**
 * Cancel a task
 * @param {string} taskId - Task ID
 */
function cancelTask(taskId) {
    if (confirm('Are you sure you want to cancel this task?')) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'cancel_task',
                task_id: taskId
            }));
            showToast('Task', 'Cancel request sent', 'info');
        }
    }
}

/**
 * Retry a task
 * @param {string} taskId - Task ID
 */
function retryTask(taskId) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'retry_task',
            task_id: taskId
        }));
        showToast('Task', 'Retry request sent', 'info');
    }
}

/**
 * Show a toast notification
 * @param {string} title - Toast title
 * @param {string} message - Toast message
 * @param {string} type - Toast type (success, error, warning, info)
 */
function showToast(title, message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toastId = `toast-${Date.now()}`;
    const bgClass = type === 'error' ? 'bg-danger' : 
                   type === 'success' ? 'bg-success' : 
                   type === 'warning' ? 'bg-warning' : 'bg-info';
    
    const textClass = (type === 'warning' || (type === 'info' && bgClass === 'bg-info')) ? 'text-dark' : 'text-white';
    
    const toastHtml = `
        <div id="${toastId}" class="toast ${bgClass} ${textClass}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto">${title}</strong>
                <small>${new Date().toLocaleTimeString()}</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    // Add toast to container
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // Remove toast after hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Show loading overlay for a container
 * @param {string} containerId - Container ID
 */
function showLoadingOverlay(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Check if loading overlay already exists
    if (container.querySelector('.loading-overlay')) return;
    
    // Create loading overlay
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner-border" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    
    // Add overlay to container
    container.style.position = 'relative';
    container.appendChild(overlay);
}

/**
 * Hide loading overlay for a container
 * @param {string} containerId - Container ID
 */
function hideLoadingOverlay(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Remove loading overlay
    const overlay = container.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Format date/time
 * @param {string|number} timestamp - Timestamp
 * @returns {string} Formatted date/time
 */
function formatDateTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Capitalize first letter of a string
 * @param {string} str - String to capitalize
 * @returns {string} Capitalized string
 */
function capitalizeFirstLetter(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

/**
 * Toggle dark mode
 * @param {boolean} enable - Whether to enable dark mode
 */
function toggleDarkMode(enable) {
    const htmlElement = document.documentElement;
    
    if (enable) {
        htmlElement.setAttribute('data-bs-theme', 'dark');
    } else {
        htmlElement.setAttribute('data-bs-theme', 'light');
    }
}

/**
 * Save user preferences to localStorage
 */
function saveUserPreferences() {
    const preferences = {
        refreshInterval,
        autoRefresh,
        darkMode
    };
    
    localStorage.setItem('dashboard_preferences', JSON.stringify(preferences));
}

/**
 * Load user preferences from localStorage
 */
function loadUserPreferences() {
    const preferences = localStorage.getItem('dashboard_preferences');
    
    if (preferences) {
        const parsedPreferences = JSON.parse(preferences);
        
        refreshInterval = parsedPreferences.refreshInterval || 5000;
        autoRefresh = parsedPreferences.autoRefresh !== undefined ? parsedPreferences.autoRefresh : true;
        darkMode = parsedPreferences.darkMode || false;
        
        // Apply dark mode
        toggleDarkMode(darkMode);
    }
}

/**
 * Syntax highlight for JSON
 * @param {string} json - JSON string to highlight
 * @returns {string} HTML with syntax highlighting
 */
function syntaxHighlight(json) {
    if (!json) return '';
    
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}
/**
 * Reorganized MCP Dashboard
 * 
 * A completely reorganized, modular dashboard for IPFS Accelerate AI platform
 * featuring enhanced organization, portable SDK integration, and improved UX.
 * 
 * Architecture:
 * - Modular component system
 * - Portable SDK integration
 * - Enhanced hardware detection
 * - Real-time monitoring
 * - Advanced analytics
 * 
 * @version 2.0.0
 */

class ReorganizedDashboard {
    constructor() {
        // Initialize portable SDK with optimized configuration
        this.sdk = PortableMCP.createPreset('development', {
            // Use a relative endpoint so it works regardless of host/port
            endpoint: '/jsonrpc',
            enableLogging: true
        });

        // Component managers
        this.modules = new ModuleManager(this.sdk);
        this.analytics = new AnalyticsManager(this.sdk);
        this.system = new SystemManager(this.sdk);
        this.notifications = new NotificationManager();

        // Application state
        this.state = {
            currentTab: 'modules',
            theme: localStorage.getItem('dashboard-theme') || 'light',
            isConnected: false,
            serverInfo: null,
            hardwareInfo: null,
            modules: [],
            metrics: {},
            config: {
                refreshInterval: 30000,
                maxNotifications: 5,
                animationDuration: 300
            }
        };

        // Performance tracking
        this.performance = {
            startTime: Date.now(),
            pageLoads: 0,
            userInteractions: 0,
            apiCalls: 0
        };

        this.init();
    }

    // ============================================
    // INITIALIZATION
    // ============================================

    async init() {
        console.log('🚀 Initializing Reorganized MCP Dashboard v2.0');
        
        try {
            // Setup theme and UI
            this.setupTheme();
            this.setupEventListeners();
            this.setupKeyboardShortcuts();
            
            // Initialize SDK connection
            await this.initializeConnection();
            
            // Load initial data
            await this.loadInitialData();
            
            // Setup real-time updates
            this.setupRealTimeUpdates();
            
            // Initialize components
            await this.initializeComponents();
            
            // Show success notification
            this.notifications.show('✅ Dashboard initialized successfully!', 'success');
            
            console.log('✅ Dashboard initialization complete');
            
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.notifications.show(`❌ Initialization failed: ${error.message}`, 'error');
        }
    }

    async initializeConnection() {
        console.log('🔄 Establishing connection to MCP server...');
        
        // Setup SDK event listeners
        this.sdk.on('connected', () => {
            this.state.isConnected = true;
            this.updateConnectionStatus(true);
            this.notifications.show('🔗 Connected to IPFS Accelerate AI server', 'success');
        });

        this.sdk.on('disconnected', (error) => {
            this.state.isConnected = false;
            this.updateConnectionStatus(false);
            this.notifications.show('❌ Connection lost', 'error');
        });

        this.sdk.on('metricsUpdated', (metrics) => {
            this.state.metrics = metrics;
            this.updateMetricsDisplay();
        });

        // Wait for server availability
        const isAvailable = await this.sdk.waitForServer(10, 2000);
        if (!isAvailable) {
            throw new Error('MCP server is not responding');
        }

        this.state.isConnected = true;
        this.updateConnectionStatus(true);
    }

    async loadInitialData() {
        console.log('📊 Loading initial data...');
        
        try {
            // Load server information
            this.state.serverInfo = await this.sdk.getServerInfo();
            
            // Load hardware information
            this.state.hardwareInfo = await this.sdk.getHardwareInfo();
            
            // Load available modules
            this.state.modules = await this.modules.loadModules();
            
            // Update displays
            this.updateServerInfoDisplay();
            this.updateHardwareDisplay();
            
        } catch (error) {
            console.warn('⚠️ Some initial data could not be loaded:', error.message);
        }
    }

    async initializeComponents() {
        console.log('🔧 Initializing dashboard components...');
        
        // Initialize modules grid
        await this.modules.initialize();
        
        // Initialize analytics
        await this.analytics.initialize();
        
        // Initialize system monitor
        await this.system.initialize();
        
        // Setup playground
        this.setupPlayground();
        
        // Initialize Model Hub search functionality
        this.setupModelSearch();
    }

    // ============================================
    // UI SETUP & EVENT HANDLING
    // ============================================

    setupTheme() {
        document.documentElement.setAttribute('data-bs-theme', this.state.theme);
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) {
            themeIcon.className = this.state.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                this.state.currentTab = e.target.getAttribute('data-bs-target').substring(1);
                this.onTabChange(this.state.currentTab);
            });
        });

        // Window events
        window.addEventListener('beforeunload', () => {
            this.saveState();
        });

        window.addEventListener('focus', () => {
            this.refreshData();
        });

        // Performance tracking
        document.addEventListener('click', () => {
            this.performance.userInteractions++;
        });

        // Queue monitor filter event listeners
        setTimeout(() => {
            const endpointTypeFilter = document.getElementById('endpoint-type-filter');
            const statusFilter = document.getElementById('status-filter');
            
            if (endpointTypeFilter) {
                endpointTypeFilter.addEventListener('change', () => {
                    if (this.state.currentTab === 'queue-monitor') {
                        this.refreshQueueStatus();
                    }
                });
            }
            
            if (statusFilter) {
                statusFilter.addEventListener('change', () => {
                    if (this.state.currentTab === 'queue-monitor') {
                        this.refreshQueueStatus();
                    }
                });
            }
        }, 100); // Small delay to ensure elements are loaded
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.switchTab('modules');
                        break;
                    case '2':
                        e.preventDefault();
                        this.switchTab('playground');
                        break;
                    case '3':
                        e.preventDefault();
                        this.switchTab('analytics');
                        break;
                    case '4':
                        e.preventDefault();
                        this.switchTab('queue-monitor');
                        break;
                    case '5':
                        e.preventDefault();
                        this.switchTab('system');
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'd':
                        e.preventDefault();
                        this.toggleTheme();
                        break;
                }
            }
        });
    }

    setupRealTimeUpdates() {
        // Update metrics every 10 seconds
        setInterval(() => {
            this.updateMetricsDisplay();
        }, 10000);

        // Refresh data every 30 seconds
        setInterval(() => {
            if (this.state.isConnected) {
                this.refreshData();
            }
        }, this.state.config.refreshInterval);

        // Update uptime every second
        setInterval(() => {
            this.updateUptimeDisplay();
        }, 1000);
    }

    // ============================================
    // UI UPDATE METHODS
    // ============================================

    updateConnectionStatus(isConnected) {
        const statusDot = document.getElementById('connection-dot');
        const statusText = document.getElementById('connection-status');
        const serverStatus = document.getElementById('server-status');

        if (statusDot) {
            statusDot.className = `status-dot ${isConnected ? 'connected' : 'disconnected'}`;
        }
        
        if (statusText) {
            statusText.textContent = isConnected ? 'Connected' : 'Disconnected';
        }
        
        if (serverStatus) {
            serverStatus.textContent = isConnected ? 'Online' : 'Offline';
        }
    }

    updateMetricsDisplay() {
        const metrics = this.sdk.getMetrics();
        
        // Update sidebar metrics
        this.updateElement('active-requests', '0'); // Real-time active requests would need WebSocket
        this.updateElement('total-requests', metrics.totalRequests);
        this.updateElement('avg-latency', `${Math.round(metrics.averageLatency)}ms`);
        
        // Update header metrics
        this.updateElement('model-count', this.state.modules.length);
    }

    updateServerInfoDisplay() {
        if (this.state.serverInfo) {
            console.log('📊 Server info:', this.state.serverInfo);
            // Additional server info updates can be added here
        }
    }

    updateHardwareDisplay() {
        if (this.state.hardwareInfo && !this.state.hardwareInfo.error) {
            this.updateElement('cpu-info', this.state.hardwareInfo.cpu?.name || 'Unknown');
            this.updateElement('memory-info', this.formatBytes(this.state.hardwareInfo.memory?.total || 0));
            this.updateElement('gpu-info', this.state.hardwareInfo.gpu?.name || 'None');
        } else {
            this.updateElement('cpu-info', 'Detecting...');
            this.updateElement('memory-info', 'Detecting...');
            this.updateElement('gpu-info', 'Detecting...');
        }
    }

    updateUptimeDisplay() {
        const uptime = Date.now() - this.performance.startTime;
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement) {
            uptimeElement.textContent = this.formatDuration(uptime);
        }
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    // ============================================
    // TAB & NAVIGATION MANAGEMENT
    // ============================================

    switchTab(tabName) {
        const tab = document.querySelector(`[data-bs-target="#${tabName}"]`);
        if (tab) {
            const bsTab = new bootstrap.Tab(tab);
            bsTab.show();
        }
    }

    onTabChange(tabName) {
        console.log(`📄 Switched to tab: ${tabName}`);
        
        switch (tabName) {
            case 'modules':
                this.modules.refresh();
                break;
            case 'playground':
                this.setupPlayground();
                break;
            case 'analytics':
                this.analytics.refresh();
                break;
            case 'queue-monitor':
                this.refreshQueueStatus();
                break;
            case 'system':
                this.system.refresh();
                break;
        }
    }

    // ============================================
    // PLAYGROUND FUNCTIONALITY
    // ============================================

    setupPlayground() {
        const moduleSelect = document.getElementById('playground-module');
        if (moduleSelect && this.state.modules.length > 0) {
            moduleSelect.innerHTML = '<option value="">Choose a module...</option>';
            
            // Group modules by category
            const groupedModules = this.modules.getGroupedModules();
            
            Object.entries(groupedModules).forEach(([category, modules]) => {
                const optgroup = document.createElement('optgroup');
                optgroup.label = category;
                
                modules.forEach(module => {
                    const option = document.createElement('option');
                    option.value = module.id;
                    option.textContent = module.name;
                    optgroup.appendChild(option);
                });
                
                moduleSelect.appendChild(optgroup);
            });
        }
    }

    async runPlaygroundTest() {
        const moduleId = document.getElementById('playground-module').value;
        const input = document.getElementById('playground-input').value;
        
        if (!moduleId || !input.trim()) {
            this.notifications.show('Please select a module and enter input data', 'warning');
            return;
        }

        try {
            this.showLoading('playground-result');
            
            // Find the module and call its default method
            const module = this.state.modules.find(m => m.id === moduleId);
            if (!module) {
                throw new Error('Module not found');
            }

            const result = await this.modules.testModule(moduleId, input);
            this.displayResult('playground-result', result, 'Test Result');
            
        } catch (error) {
            this.displayError('playground-result', error.message);
            this.notifications.show(`Test failed: ${error.message}`, 'error');
        }
    }

    async generateSDKCode() {
        const language = document.getElementById('sdk-language').value;
        const moduleId = document.getElementById('playground-module').value;
        
        if (!moduleId) {
            this.notifications.show('Please select a module first', 'warning');
            return;
        }

        const module = this.state.modules.find(m => m.id === moduleId);
        if (!module) {
            this.notifications.show('Module not found', 'error');
            return;
        }

        const code = this.generateCodeForModule(module, language);
        
        const resultContainer = document.getElementById('sdk-code-result');
        const codeContent = document.getElementById('sdk-code-content');
        
        if (codeContent) {
            codeContent.textContent = code;
            if (window.Prism) {
                window.Prism.highlightElement(codeContent);
            }
        }
        
        if (resultContainer) {
            resultContainer.style.display = 'block';
        }
    }

    generateCodeForModule(module, language) {
        const examples = {
            javascript: `// Using Portable MCP SDK
const client = PortableMCP.create({
    endpoint: '/jsonrpc',
    timeout: 30000
});

// Call ${module.name}
try {
    const result = await client.request('${module.method}', {
        // Add your parameters here
        input: "your input data"
    });
    console.log('Result:', result);
} catch (error) {
    console.error('Error:', error.message);
}`,

            python: `# Using Python requests
import requests
import json

def call_${module.id}(input_data):
    payload = {
        "jsonrpc": "2.0",
        "method": "${module.method}",
        "params": {"input": input_data},
        "id": 1
    }
    
    response = requests.post(
        "http://localhost:8003/jsonrpc",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# Example usage
result = call_${module.id}("your input data")
print(result)`,

            curl: `# Using cURL
curl -X POST http://localhost:8003/jsonrpc \\
  -H "Content-Type: application/json" \\
  -d '{
    "jsonrpc": "2.0",
    "method": "${module.method}",
    "params": {
      "input": "your input data"
    },
    "id": 1
  }'`
        };

        return examples[language] || examples.javascript;
    }

    // ============================================
    // UTILITY METHODS
    // ============================================

    showLoading(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                </div>
            `;
            container.style.display = 'block';
        }
    }

    displayResult(containerId, result, title = 'Result') {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="result-header">
                <i class="fas fa-check-circle"></i>
                ${title}
            </div>
            <div class="result-content">
                ${this.formatResult(result)}
            </div>
        `;
        container.style.display = 'block';
    }

    displayError(containerId, message) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="result-header text-danger">
                <i class="fas fa-exclamation-circle"></i>
                Error
            </div>
            <div class="result-content">
                <div class="alert alert-danger">${message}</div>
            </div>
        `;
        container.style.display = 'block';
    }

    formatResult(result) {
        if (typeof result === 'string') {
            return `<div class="text-result">${result}</div>`;
        } else if (typeof result === 'object') {
            return `<pre class="code-block">${JSON.stringify(result, null, 2)}</pre>`;
        } else {
            return `<div class="text-result">${String(result)}</div>`;
        }
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
        }
    }

    // ============================================
    // ACTION METHODS
    // ============================================

    toggleTheme() {
        this.state.theme = this.state.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('dashboard-theme', this.state.theme);
        this.setupTheme();
        this.notifications.show(`🎨 Switched to ${this.state.theme} theme`, 'info');
    }

    async refreshData() {
        try {
            console.log('🔄 Refreshing dashboard data...');
            
            // Refresh all components
            await Promise.all([
                this.loadInitialData(),
                this.modules.refresh(),
                this.analytics.refresh(),
                this.system.refresh()
            ]);
            
            this.notifications.show('🔄 Data refreshed successfully', 'success');
            
        } catch (error) {
            console.error('❌ Data refresh failed:', error);
            this.notifications.show(`❌ Refresh failed: ${error.message}`, 'error');
        }
    }

    async exportLogs() {
        try {
            const exportData = {
                timestamp: new Date().toISOString(),
                metrics: this.sdk.getMetrics(),
                state: this.state,
                performance: this.performance,
                serverInfo: this.state.serverInfo,
                hardwareInfo: this.state.hardwareInfo
            };

            PortableMCP.downloadData(exportData, `mcp-dashboard-logs-${Date.now()}.json`);
            this.notifications.show('📄 Logs exported successfully', 'success');
            
        } catch (error) {
            this.notifications.show(`❌ Export failed: ${error.message}`, 'error');
        }
    }

    showHelp() {
        const helpContent = `
            <h5>Keyboard Shortcuts:</h5>
            <ul>
                <li><kbd>Ctrl/Cmd + 1</kbd> - Switch to Modules</li>
                <li><kbd>Ctrl/Cmd + 2</kbd> - Switch to Playground</li>
                <li><kbd>Ctrl/Cmd + 3</kbd> - Switch to Analytics</li>
                <li><kbd>Ctrl/Cmd + 4</kbd> - Switch to Queue Monitor</li>
                <li><kbd>Ctrl/Cmd + 5</kbd> - Switch to System</li>
                <li><kbd>Ctrl/Cmd + R</kbd> - Refresh Data</li>
                <li><kbd>Ctrl/Cmd + D</kbd> - Toggle Theme</li>
            </ul>
            <h5>Features:</h5>
            <ul>
                <li>Real-time metrics and monitoring</li>
                <li>Modular AI inference components</li>
                <li>Queue monitoring with endpoint breakdown</li>
                <li>Portable SDK code generation</li>
                <li>Hardware detection and system monitoring</li>
                <li>Advanced analytics and charting</li>
            </ul>
        `;
        
        this.notifications.show(helpContent, 'info', 10000);
    }

    // ============================================
    // QUEUE MONITORING METHODS
    // ============================================

    async refreshQueueStatus() {
        try {
            console.log('🔄 Refreshing queue status...');
            
            const queueStatus = await this.sdk.request('get_queue_status');
            const queueHistory = await this.sdk.request('get_queue_history');
            
            if (queueStatus.status === 'success') {
                this.updateQueueOverview(queueStatus);
                this.updateEndpointQueues(queueStatus.endpoint_queues);
            }
            
            if (queueHistory.status === 'success') {
                this.updateQueueCharts(queueHistory);
            }
            
            this.notifications.show('🔄 Queue status refreshed', 'success');
            
        } catch (error) {
            console.error('❌ Queue status refresh failed:', error);
            this.notifications.show(`❌ Queue refresh failed: ${error.message}`, 'error');
        }
    }

    updateQueueOverview(queueData) {
        const global = queueData.global_queue;
        
        document.getElementById('total-queue-size').textContent = global.total_tasks || 0;
        document.getElementById('processing-tasks').textContent = global.processing_tasks || 0;
        document.getElementById('completed-tasks').textContent = global.completed_tasks || 0;
        document.getElementById('failed-tasks').textContent = global.failed_tasks || 0;
    }

    updateEndpointQueues(endpointQueues) {
        const container = document.getElementById('endpoint-queues-container');
        if (!container) return;
        
        const typeFilter = document.getElementById('endpoint-type-filter')?.value || 'all';
        const statusFilter = document.getElementById('status-filter')?.value || 'all';
        
        let html = '';
        
        Object.entries(endpointQueues).forEach(([endpointId, endpoint]) => {
            // Apply filters
            if (typeFilter !== 'all' && endpoint.endpoint_type !== typeFilter) return;
            if (statusFilter !== 'all' && endpoint.status !== statusFilter) return;
            
            html += this.renderEndpointCard(endpointId, endpoint);
        });
        
        container.innerHTML = html || '<div class="text-center text-secondary p-4">No endpoints match the current filters</div>';
    }

    renderEndpointCard(endpointId, endpoint) {
        const statusClass = endpoint.status;
        const currentTask = endpoint.current_task;
        
        let deviceInfo = '';
        if (endpoint.endpoint_type === 'local_gpu') {
            deviceInfo = `<span class="endpoint-type">Device: ${endpoint.device}</span>`;
        } else if (endpoint.endpoint_type === 'libp2p_peer') {
            deviceInfo = `<span class="endpoint-type">Peer: ${endpoint.peer_id?.substring(0, 12)}...</span>`;
        } else if (endpoint.endpoint_type === 'api_provider') {
            deviceInfo = `<span class="endpoint-type">Provider: ${endpoint.provider} (${endpoint.key_name})</span>`;
        }
        
        let additionalMetrics = '';
        if (endpoint.network_latency !== undefined) {
            additionalMetrics += `
                <div class="endpoint-metric">
                    <div class="endpoint-metric-value">${endpoint.network_latency}ms</div>
                    <div class="endpoint-metric-label">Network Latency</div>
                </div>
            `;
        }
        if (endpoint.rate_limit_remaining !== undefined) {
            additionalMetrics += `
                <div class="endpoint-metric">
                    <div class="endpoint-metric-value">${endpoint.rate_limit_remaining}</div>
                    <div class="endpoint-metric-label">Rate Limit</div>
                </div>
            `;
        }
        
        let currentTaskHtml = '';
        if (currentTask) {
            currentTaskHtml = `
                <div class="current-task">
                    <div class="current-task-header">Current Task</div>
                    <div class="current-task-info">
                        <div><span>Task ID:</span> <strong>${currentTask.task_id}</strong></div>
                        <div><span>Model:</span> <strong>${currentTask.model}</strong></div>
                        <div><span>Type:</span> <strong>${currentTask.task_type}</strong></div>
                        <div><span>ETA:</span> <strong>${currentTask.estimated_completion}</strong></div>
                    </div>
                </div>
            `;
        }
        
        const modelTypeTags = endpoint.model_types.map(type => 
            `<span class="model-type-tag">${type}</span>`
        ).join('');
        
        return `
            <div class="endpoint-queue-card">
                <div class="endpoint-header">
                    <div>
                        <div class="endpoint-name">${endpointId}</div>
                        ${deviceInfo}
                    </div>
                    <div class="endpoint-status">
                        <span class="status-badge ${statusClass}">${endpoint.status}</span>
                    </div>
                </div>
                
                <div class="endpoint-metrics">
                    <div class="endpoint-metric">
                        <div class="endpoint-metric-value">${endpoint.queue_size}</div>
                        <div class="endpoint-metric-label">Queue Size</div>
                    </div>
                    <div class="endpoint-metric">
                        <div class="endpoint-metric-value">${endpoint.processing}</div>
                        <div class="endpoint-metric-label">Processing</div>
                    </div>
                    <div class="endpoint-metric">
                        <div class="endpoint-metric-value">${endpoint.avg_processing_time.toFixed(1)}s</div>
                        <div class="endpoint-metric-label">Avg Time</div>
                    </div>
                    ${additionalMetrics}
                </div>
                
                <div class="model-types">
                    ${modelTypeTags}
                </div>
                
                ${currentTaskHtml}
            </div>
        `;
    }

    updateQueueCharts(historyData) {
        this.updateQueueTrendChart(historyData.time_series);
        this.updateModelTypeChart(historyData.model_type_stats);
    }

    updateQueueTrendChart(timeSeries) {
        const ctx = document.getElementById('queue-trend-chart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.queueTrendChart) {
            this.queueTrendChart.destroy();
        }
        
        const labels = timeSeries.timestamps.map(ts => 
            new Date(ts * 1000).toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit' 
            })
        );
        
        this.queueTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Queue Size',
                        data: timeSeries.queue_sizes,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Processing',
                        data: timeSeries.processing_tasks,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    updateModelTypeChart(modelTypeStats) {
        const ctx = document.getElementById('model-type-chart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.modelTypeChart) {
            this.modelTypeChart.destroy();
        }
        
        const labels = Object.keys(modelTypeStats);
        const data = Object.values(modelTypeStats).map(stat => stat.total_requests);
        const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'];
        
        this.modelTypeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(label => label.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())),
                datasets: [{
                    data: data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    async updateSystemConfig() {
        // Implementation for system configuration updates
        this.notifications.show('⚙️ System configuration updated', 'success');
    }

    async exportMetrics() {
        const metrics = {
            timestamp: new Date().toISOString(),
            sdk: this.sdk.getMetrics(),
            performance: this.performance,
            system: await this.system.getMetrics()
        };

        PortableMCP.downloadData(metrics, `mcp-metrics-${Date.now()}.json`);
        this.notifications.show('📊 Metrics exported successfully', 'success');
    }

    async exportConfig() {
        const config = {
            timestamp: new Date().toISOString(),
            dashboard: this.state.config,
            sdk: this.sdk.config,
            theme: this.state.theme
        };

        PortableMCP.downloadData(config, `mcp-config-${Date.now()}.json`);
        this.notifications.show('⚙️ Configuration exported successfully', 'success');
    }

    saveState() {
        try {
            const stateToSave = {
                theme: this.state.theme,
                currentTab: this.state.currentTab,
                config: this.state.config,
                timestamp: Date.now()
            };
            
            localStorage.setItem('dashboard-state', JSON.stringify(stateToSave));
        } catch (error) {
            console.warn('Could not save dashboard state:', error);
        }
    }

    loadState() {
        try {
            const savedState = localStorage.getItem('dashboard-state');
            if (savedState) {
                const state = JSON.parse(savedState);
                this.state.theme = state.theme || 'light';
                this.state.currentTab = state.currentTab || 'modules';
                this.state.config = { ...this.state.config, ...state.config };
            }
        } catch (error) {
            console.warn('Could not load dashboard state:', error);
        }
    }

    // ============================================
    // MODEL HUB FUNCTIONALITY
    // ============================================

    async searchModels() {
        const query = document.getElementById('model-search-input').value.trim();
        const searchType = document.getElementById('search-type').value;
        
        if (!query) {
            this.notifications.show('Please enter a search query', 'warning');
            return;
        }

        this.showLoadingState();
        
        try {
            const filters = this.getCurrentFilters();
            const sortBy = document.getElementById('sort-by')?.value || 'relevance';
            
            const results = await this.sdk.request('search_huggingface_models', {
                query: query,
                search_type: searchType,
                filters: filters,
                sort_by: sortBy,
                limit: 20,
                offset: 0
            });

            this.displaySearchResults(results);
            this.updateSearchStats();
        } catch (error) {
            console.error('Model search failed:', error);
            this.notifications.show('Model search failed: ' + error.message, 'error');
        } finally {
            this.hideLoadingState();
        }
    }

    getCurrentFilters() {
        return {
            task: document.getElementById('filter-task')?.value || '',
            library: document.getElementById('filter-library')?.value || '',
            language: document.getElementById('filter-language')?.value || '',
            min_downloads: parseInt(document.getElementById('filter-downloads')?.value) || 0
        };
    }

    displaySearchResults(response) {
        const resultsContainer = document.getElementById('search-results');
        const headerContainer = document.getElementById('search-results-header');
        const noResultsMessage = document.getElementById('no-results-message');
        
        if (!response.success || response.results.length === 0) {
            headerContainer.style.display = 'none';
            noResultsMessage.style.display = 'block';
            resultsContainer.innerHTML = '';
            return;
        }

        // Hide no results message
        noResultsMessage.style.display = 'none';
        headerContainer.style.display = 'block';

        // Update search info
        document.getElementById('search-results-count').textContent = 
            `${response.total} results found`;
        document.getElementById('search-time').textContent = 
            `in ${response.search_time_ms}ms`;

        // Clear previous results
        resultsContainer.innerHTML = '';

        // Display results
        response.results.forEach(model => {
            const card = this.createModelCard(model);
            resultsContainer.appendChild(card);
        });

        // Update pagination if needed
        this.updatePagination(response);
    }

    createModelCard(model) {
        const card = document.createElement('div');
        card.className = 'model-result-card';
        
        const tagsHtml = model.tags?.slice(0, 3).map(tag => {
            let className = 'model-tag';
            if (tag === model.pipeline_tag) className += ' pipeline';
            if (tag === model.library_name) className += ' library';
            return `<span class="${className}">${tag}</span>`;
        }).join('') || '';

        // Create unique IDs for this model card
        const modelSafeId = model.full_name.replace(/[^a-zA-Z0-9]/g, '_');
        const downloadBtnId = `download-btn-${modelSafeId}`;
        const progressId = `progress-${modelSafeId}`;
        const statusId = `status-${modelSafeId}`;

        card.innerHTML = `
            <div class="model-result-header">
                <div>
                    <div class="model-title">${model.full_name}</div>
                    <div class="model-author">by ${model.author}</div>
                </div>
                <div class="model-score">
                    ${(model.search_score * 100).toFixed(0)}%
                </div>
            </div>
            <div class="model-description" onclick="dashboard.showModelDetails('${model.id}')" style="cursor: pointer;">
                ${model.description || 'No description available'}
            </div>
            <div class="model-tags">
                ${tagsHtml}
            </div>
            <div class="model-stats">
                <div class="model-stat">
                    <i class="fas fa-download"></i>
                    <span>${this.formatNumber(model.downloads)}</span>
                </div>
                <div class="model-stat">
                    <i class="fas fa-heart"></i>
                    <span>${this.formatNumber(model.likes)}</span>
                </div>
                <div class="model-stat">
                    <i class="fas fa-clock"></i>
                    <span>${this.formatDate(model.last_modified)}</span>
                </div>
            </div>
            
            <!-- Model Status and Actions -->
            <div class="model-actions mt-3">
                <div class="model-status mb-2" id="${statusId}">
                    <small class="text-muted">Checking status...</small>
                </div>
                
                <!-- Download Progress Bar (hidden by default) -->
                <div class="download-progress mb-2" id="${progressId}" style="display: none;">
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                    </div>
                    <small class="text-muted">Downloading... <span class="progress-text">0%</span></small>
                </div>
                
                <!-- Action Buttons -->
                <div class="btn-group w-100" role="group">
                    <button type="button" class="btn btn-sm btn-outline-primary" 
                            id="${downloadBtnId}" 
                            onclick="dashboard.downloadModel('${model.full_name}', '${downloadBtnId}', '${progressId}', '${statusId}')">
                        <i class="fas fa-download"></i>
                        <span class="btn-text">Download</span>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-success" 
                            onclick="dashboard.testModelInference('${model.full_name}', '${model.pipeline_tag || 'text-generation'}')"
                            disabled id="inference-btn-${modelSafeId}">
                        <i class="fas fa-play"></i>
                        Test
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-info" 
                            onclick="dashboard.showModelDetails('${model.id}')">
                        <i class="fas fa-info-circle"></i>
                        Details
                    </button>
                </div>
            </div>
        `;

        // Check model download status after creating the card
        setTimeout(() => this.checkModelStatus(model.full_name, statusId, downloadBtnId, `inference-btn-${modelSafeId}`), 100);

        return card;
    }

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    formatDate(dateStr) {
        if (!dateStr) return 'Unknown';
        const date = new Date(dateStr);
        return date.toLocaleDateString();
    }

    async showModelDetails(modelId) {
        try {
            const response = await this.sdk.call('get_huggingface_model_details', {
                model_id: modelId
            });

            if (response.success && response.model) {
                this.displayModelModal(response.model);
            } else {
                this.notifications.show('Could not load model details', 'error');
            }
        } catch (error) {
            console.error('Failed to load model details:', error);
            this.notifications.show('Failed to load model details', 'error');
        }
    }

    displayModelModal(model) {
        // Create and show a modal with detailed model information
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        
        const modelSafeId = model.full_name.replace(/[^a-zA-Z0-9]/g, '_');
        const modalStatusId = `modal-status-${modelSafeId}`;
        
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${model.full_name}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <h6>Description</h6>
                            <p>${model.description || 'No description available'}</p>
                        </div>
                        <div class="mb-3">
                            <h6>Statistics</h6>
                            <div class="row">
                                <div class="col-md-4">
                                    <strong>Downloads:</strong> ${this.formatNumber(model.downloads)}
                                </div>
                                <div class="col-md-4">
                                    <strong>Likes:</strong> ${this.formatNumber(model.likes)}
                                </div>
                                <div class="col-md-4">
                                    <strong>Task:</strong> ${model.pipeline_tag}
                                </div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <h6>Tags</h6>
                            <div class="model-tags">
                                ${model.tags?.map(tag => `<span class="model-tag">${tag}</span>`).join('') || 'No tags'}
                            </div>
                        </div>
                        
                        <!-- Model Status and Actions Section -->
                        <div class="mb-3">
                            <h6>Model Status</h6>
                            <div class="alert alert-info" id="${modalStatusId}">
                                <i class="fas fa-spinner fa-spin"></i> Checking model status...
                            </div>
                        </div>
                        
                        <!-- Download and Inference Controls -->
                        <div class="mb-3">
                            <h6>Actions</h6>
                            <div class="btn-group w-100 mb-2" role="group">
                                <button type="button" class="btn btn-outline-primary" id="modal-download-btn-${modelSafeId}"
                                        onclick="dashboard.downloadModelFromModal('${model.full_name}', '${modalStatusId}', '${modelSafeId}')">
                                    <i class="fas fa-download"></i> Download Model
                                </button>
                                <button type="button" class="btn btn-outline-success" id="modal-inference-btn-${modelSafeId}"
                                        onclick="dashboard.testModelInference('${model.full_name}', '${model.pipeline_tag || 'text-generation'}')" disabled>
                                    <i class="fas fa-play"></i> Test Inference
                                </button>
                            </div>
                            <div class="btn-group w-100" role="group">
                                <button type="button" class="btn btn-outline-info" 
                                        onclick="window.open('https://huggingface.co/${model.full_name}', '_blank')">
                                    <i class="fas fa-external-link-alt"></i> View on HuggingFace
                                </button>
                                <button type="button" class="btn btn-outline-secondary" 
                                        onclick="dashboard.showDownloadedModels()">
                                    <i class="fas fa-list"></i> Manage Downloads
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Check model status after modal is shown
        setTimeout(() => this.checkModalModelStatus(model.full_name, modalStatusId, modelSafeId), 100);

        // Clean up modal when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }


    showAdvancedFilters() {
        document.getElementById('advanced-filters').style.display = 'block';
    }

    hideAdvancedFilters() {
        document.getElementById('advanced-filters').style.display = 'none';
    }

    applyFilters() {
        this.searchModels();
        this.hideAdvancedFilters();
    }

    clearFilters() {
        document.getElementById('filter-task').value = '';
        document.getElementById('filter-library').value = '';
        document.getElementById('filter-language').value = '';
        document.getElementById('filter-downloads').value = '';
        this.hideAdvancedFilters();
    }

    sortResults() {
        this.searchModels();
    }

    async loadPopularModels() {
        this.showLoadingState();
        
        try {
            const results = await this.sdk.call('search_huggingface_models', {
                query: '',
                search_type: 'hybrid',
                sort_by: 'downloads',
                limit: 20,
                offset: 0
            });

            this.displaySearchResults(results);
        } catch (error) {
            console.error('Failed to load popular models:', error);
            this.notifications.show('Failed to load popular models', 'error');
        } finally {
            this.hideLoadingState();
        }
    }

    showLoadingState() {
        const resultsContainer = document.getElementById('search-results');
        resultsContainer.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-3">Searching models...</div>
            </div>
        `;
    }

    hideLoadingState() {
        // Loading state will be replaced by results
    }

    updatePagination(response) {
        // Implement pagination logic if needed
        const paginationContainer = document.getElementById('pagination-container');
        if (response.total > response.limit) {
            paginationContainer.style.display = 'block';
            // Add pagination buttons logic here
        } else {
            paginationContainer.style.display = 'none';
        }
    }

    async updateSearchStats() {
        try {
            const response = await this.sdk.call('get_model_search_stats');
            if (response.success) {
                this.displaySearchStats(response.stats);
            }
        } catch (error) {
            console.warn('Could not load search stats:', error);
        }
    }

    displaySearchStats(stats) {
        const statsContainer = document.getElementById('search-stats-content');
        const statsSection = document.getElementById('search-stats-section');
        
        if (!stats || stats.total_models === 0) {
            statsSection.style.display = 'none';
            return;
        }

        statsSection.style.display = 'block';
        statsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-4 text-center">
                    <h4>${this.formatNumber(stats.total_models)}</h4>
                    <p class="text-muted">Total Models</p>
                </div>
                <div class="col-md-4 text-center">
                    <h4>${stats.has_vector_index ? 'Yes' : 'No'}</h4>
                    <p class="text-muted">Vector Search</p>
                </div>
                <div class="col-md-4 text-center">
                    <h4>${stats.has_bm25_index ? 'Yes' : 'No'}</h4>
                    <p class="text-muted">Keyword Search</p>
                </div>
            </div>
            ${stats.top_tasks ? `
                <div class="mt-4">
                    <h6>Popular Tasks</h6>
                    <div class="model-tags">
                        ${Object.entries(stats.top_tasks).slice(0, 10).map(([task, count]) => 
                            `<span class="model-tag">${task} (${count})</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    // Setup search input with suggestions
    setupModelSearch() {
        const searchInput = document.getElementById('model-search-input');
        const suggestionsContainer = document.getElementById('search-suggestions');

        if (searchInput) {
            let debounceTimer;
            
            searchInput.addEventListener('input', (e) => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(async () => {
                    const query = e.target.value.trim();
                    if (query.length >= 2) {
                        await this.showSearchSuggestions(query);
                    } else {
                        suggestionsContainer.style.display = 'none';
                    }
                }, 300);
            });

            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.searchModels();
                }
            });
        }
    }

    async showSearchSuggestions(query) {
        try {
            const response = await this.sdk.call('get_model_search_suggestions', {
                query: query,
                limit: 8
            });

            if (response.success && response.suggestions.length > 0) {
                const suggestionsContainer = document.getElementById('search-suggestions');
                suggestionsContainer.innerHTML = response.suggestions.map(suggestion => 
                    `<div class="suggestion-item" onclick="dashboard.selectSuggestion('${suggestion}')">${suggestion}</div>`
                ).join('');
                suggestionsContainer.style.display = 'block';
            }
        } catch (error) {
            console.warn('Failed to get suggestions:', error);
        }
    }

    selectSuggestion(suggestion) {
        document.getElementById('model-search-input').value = suggestion;
        document.getElementById('search-suggestions').style.display = 'none';
        this.searchModels();
    }

    // ===== MODEL DOWNLOAD AND INFERENCE METHODS =====

    async checkModelStatus(modelId, statusElementId, downloadBtnId, inferenceBtnId) {
        // Check if a model is downloaded and update UI accordingly.
        try {
            const response = await this.sdk.call('get_model_download_info', {
                model_id: modelId
            });

            const statusElement = document.getElementById(statusElementId);
            const downloadBtn = document.getElementById(downloadBtnId);
            const inferenceBtn = document.getElementById(inferenceBtnId);

            if (response.success) {
                const info = response.download_info;
                
                if (info.is_downloaded) {
                    // Model is downloaded
                    statusElement.innerHTML = `<small class="text-success"><i class="fas fa-check-circle"></i> Downloaded</small>`;
                    downloadBtn.innerHTML = `<i class="fas fa-trash"></i> <span class="btn-text">Remove</span>`;
                    downloadBtn.className = 'btn btn-sm btn-outline-danger';
                    downloadBtn.onclick = () => this.removeModel(modelId, downloadBtnId, statusElementId, inferenceBtnId);
                    
                    if (inferenceBtn) {
                        inferenceBtn.disabled = false;
                        inferenceBtn.title = 'Test inference with this model';
                    }
                } else if (info.active_downloads && info.active_downloads.length > 0) {
                    // Model is being downloaded
                    const download = info.active_downloads[0];
                    statusElement.innerHTML = `<small class="text-info"><i class="fas fa-spinner fa-spin"></i> Downloading (${download.progress}%)</small>`;
                    downloadBtn.disabled = true;
                    downloadBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span class="btn-text">Downloading</span>`;
                    
                    // Start polling for download progress
                    this.pollDownloadProgress(download.download_id, modelId, statusElementId, downloadBtnId, inferenceBtnId);
                } else {
                    // Model is not downloaded
                    statusElement.innerHTML = `<small class="text-muted"><i class="fas fa-cloud"></i> Not downloaded</small>`;
                    downloadBtn.innerHTML = `<i class="fas fa-download"></i> <span class="btn-text">Download</span>`;
                    downloadBtn.className = 'btn btn-sm btn-outline-primary';
                    downloadBtn.disabled = false;
                    
                    if (inferenceBtn) {
                        inferenceBtn.disabled = true;
                        inferenceBtn.title = 'Download model first to test inference';
                    }
                }
            }
        } catch (error) {
            console.warn('Could not check model status:', error);
            const statusElement = document.getElementById(statusElementId);
            if (statusElement) {
                statusElement.innerHTML = `<small class="text-warning"><i class="fas fa-exclamation-triangle"></i> Status unknown</small>`;
            }
        }
    }

    async downloadModel(modelId, downloadBtnId, progressId, statusElementId) {
        // Download a HuggingFace model.
        try {
            this.notifications.show(`Starting download of ${modelId}...`, 'info');
            
            const downloadBtn = document.getElementById(downloadBtnId);
            const statusElement = document.getElementById(statusElementId);
            
            // Update UI to show download starting
            downloadBtn.disabled = true;
            downloadBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span class="btn-text">Starting...</span>`;
            statusElement.innerHTML = `<small class="text-info"><i class="fas fa-spinner fa-spin"></i> Starting download...</small>`;

            const response = await this.sdk.call('download_huggingface_model', {
                model_id: modelId,
                download_type: 'snapshot'
            });

            if (response.success) {
                if (response.already_downloaded) {
                    this.notifications.show(`${modelId} is already downloaded!`, 'success');
                    // Refresh status
                    this.checkModelStatus(modelId, statusElementId, downloadBtnId, `inference-btn-${modelId.replace(/[^a-zA-Z0-9]/g, '_')}`);
                } else {
                    this.notifications.show(`Download started for ${modelId}`, 'success');
                    // Start polling for progress
                    this.pollDownloadProgress(response.download_id, modelId, statusElementId, downloadBtnId, `inference-btn-${modelId.replace(/[^a-zA-Z0-9]/g, '_')}`);
                }
            } else {
                throw new Error(response.error || 'Download failed');
            }
        } catch (error) {
            console.error('Download failed:', error);
            this.notifications.show(`Download failed: ${error.message}`, 'error');
            
            // Reset download button
            const downloadBtn = document.getElementById(downloadBtnId);
            if (downloadBtn) {
                downloadBtn.disabled = false;
                downloadBtn.innerHTML = `<i class="fas fa-download"></i> <span class="btn-text">Download</span>`;
            }
        }
    }

    async pollDownloadProgress(downloadId, modelId, statusElementId, downloadBtnId, inferenceBtnId) {
        // Poll download progress and update UI.
        const pollInterval = setInterval(async () => {
            try {
                const response = await this.sdk.call('get_download_status', {
                    download_id: downloadId
                });

                if (response.success) {
                    const status = response.download_status;
                    const statusElement = document.getElementById(statusElementId);
                    const downloadBtn = document.getElementById(downloadBtnId);

                    if (status.status === 'completed') {
                        // Download completed
                        clearInterval(pollInterval);
                        this.notifications.show(`${modelId} downloaded successfully!`, 'success');
                        
                        // Refresh model status
                        this.checkModelStatus(modelId, statusElementId, downloadBtnId, inferenceBtnId);
                        
                    } else if (status.status === 'failed') {
                        // Download failed
                        clearInterval(pollInterval);
                        this.notifications.show(`Download failed: ${status.error}`, 'error');
                        
                        if (statusElement) {
                            statusElement.innerHTML = `<small class="text-danger"><i class="fas fa-exclamation-circle"></i> Download failed</small>`;
                        }
                        if (downloadBtn) {
                            downloadBtn.disabled = false;
                            downloadBtn.innerHTML = `<i class="fas fa-download"></i> <span class="btn-text">Retry</span>`;
                        }
                        
                    } else {
                        // Download in progress
                        const progress = status.progress || 0;
                        
                        if (statusElement) {
                            statusElement.innerHTML = `<small class="text-info"><i class="fas fa-spinner fa-spin"></i> Downloading (${progress}%)</small>`;
                        }
                        if (downloadBtn) {
                            downloadBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span class="btn-text">${progress}%</span>`;
                        }
                    }
                }
            } catch (error) {
                console.warn('Error polling download progress:', error);
                clearInterval(pollInterval);
            }
        }, 2000); // Poll every 2 seconds
    }

    async removeModel(modelId, downloadBtnId, statusElementId, inferenceBtnId) {
        // Remove a downloaded model.
        if (!confirm(`Are you sure you want to remove ${modelId}? This will delete all downloaded files.`)) {
            return;
        }

        try {
            this.notifications.show(`Removing ${modelId}...`, 'info');

            const response = await this.sdk.call('remove_downloaded_model', {
                model_id: modelId,
                confirm: true
            });

            if (response.success) {
                this.notifications.show(`${modelId} removed successfully`, 'success');
                
                // Refresh model status
                this.checkModelStatus(modelId, statusElementId, downloadBtnId, inferenceBtnId);
            } else {
                throw new Error(response.error || 'Failed to remove model');
            }
        } catch (error) {
            console.error('Failed to remove model:', error);
            this.notifications.show(`Failed to remove model: ${error.message}`, 'error');
        }
    }

    async testModelInference(modelId, taskType) {
        // Test inference with a downloaded model.
        try {
            // Show loading modal
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Test Inference: ${modelId}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <label class="form-label">Input Text:</label>
                                <textarea class="form-control" id="inference-input" rows="3" 
                                    placeholder="Enter your test input here...">Hello, how are you?</textarea>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Task Type:</label>
                                <select class="form-control" id="inference-task">
                                    <option value="text-generation" ${taskType === 'text-generation' ? 'selected' : ''}>Text Generation</option>
                                    <option value="text-classification" ${taskType === 'text-classification' ? 'selected' : ''}>Text Classification</option>
                                    <option value="question-answering" ${taskType === 'question-answering' ? 'selected' : ''}>Question Answering</option>
                                    <option value="summarization" ${taskType === 'summarization' ? 'selected' : ''}>Summarization</option>
                                    <option value="translation" ${taskType === 'translation' ? 'selected' : ''}>Translation</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <button class="btn btn-primary" onclick="dashboard.runInference('${modelId}')">
                                    <i class="fas fa-play"></i> Run Inference
                                </button>
                            </div>
                            <div id="inference-result" style="display: none;">
                                <hr>
                                <h6>Result:</h6>
                                <div class="alert alert-success" id="inference-output"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();

            // Clean up modal when hidden
            modal.addEventListener('hidden.bs.modal', () => {
                document.body.removeChild(modal);
            });

        } catch (error) {
            console.error('Failed to open inference test:', error);
            this.notifications.show(`Failed to open inference test: ${error.message}`, 'error');
        }
    }

    async runInference(modelId) {
        // Run inference with the model.
        try {
            const inputText = document.getElementById('inference-input').value;
            const taskType = document.getElementById('inference-task').value;

            if (!inputText.trim()) {
                this.notifications.show('Please enter some input text', 'warning');
                return;
            }

            // Show loading in result area
            const resultDiv = document.getElementById('inference-result');
            const outputDiv = document.getElementById('inference-output');
            
            resultDiv.style.display = 'block';
            outputDiv.className = 'alert alert-info';
            outputDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running inference...';

            const response = await this.sdk.call('test_model_inference', {
                model_id: modelId,
                input_text: inputText,
                task: taskType,
                max_length: 100,
                temperature: 0.7
            });

            if (response.success) {
                const result = response.inference_result;
                
                outputDiv.className = 'alert alert-success';
                outputDiv.innerHTML = `
                    <div><strong>Output:</strong> ${result.output}</div>
                    <div class="mt-2">
                        <small class="text-muted">
                            Confidence: ${(result.confidence * 100).toFixed(1)}% | 
                            Processing Time: ${result.processing_time_ms}ms | 
                            Engine: ${result.inference_engine}
                        </small>
                    </div>
                `;
                
                this.notifications.show('Inference completed successfully!', 'success');
            } else {
                throw new Error(response.error || 'Inference failed');
            }
        } catch (error) {
            console.error('Inference failed:', error);
            
            const outputDiv = document.getElementById('inference-output');
            if (outputDiv) {
                outputDiv.className = 'alert alert-danger';
                outputDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${error.message}`;
            }
            
            this.notifications.show(`Inference failed: ${error.message}`, 'error');
        }
    }

    async checkModalModelStatus(modelId, statusElementId, modelSafeId) {
        // Check model status in the modal and update UI.
        try {
            const response = await this.sdk.call('get_model_download_info', {
                model_id: modelId
            });

            const statusElement = document.getElementById(statusElementId);
            const downloadBtn = document.getElementById(`modal-download-btn-${modelSafeId}`);
            const inferenceBtn = document.getElementById(`modal-inference-btn-${modelSafeId}`);

            if (response.success) {
                const info = response.download_info;
                
                if (info.is_downloaded) {
                    // Model is downloaded
                    statusElement.className = 'alert alert-success';
                    statusElement.innerHTML = `
                        <i class="fas fa-check-circle"></i> 
                        <strong>Downloaded</strong><br>
                        <small>Size: ${(info.size_mb || 0).toFixed(1)} MB | Downloaded: ${new Date(info.downloaded_at).toLocaleDateString()}</small>
                    `;
                    
                    if (downloadBtn) {
                        downloadBtn.innerHTML = '<i class="fas fa-trash"></i> Remove Model';
                        downloadBtn.className = 'btn btn-outline-danger';
                        downloadBtn.onclick = () => this.removeModelFromModal(modelId, statusElementId, modelSafeId);
                    }
                    
                    if (inferenceBtn) {
                        inferenceBtn.disabled = false;
                    }
                } else if (info.active_downloads && info.active_downloads.length > 0) {
                    // Model is being downloaded
                    const download = info.active_downloads[0];
                    statusElement.className = 'alert alert-info';
                    statusElement.innerHTML = `
                        <i class="fas fa-spinner fa-spin"></i> 
                        <strong>Downloading (${download.progress}%)</strong><br>
                        <small>Download in progress...</small>
                    `;
                    
                    if (downloadBtn) {
                        downloadBtn.disabled = true;
                        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
                    }
                } else {
                    // Model is not downloaded
                    statusElement.className = 'alert alert-warning';
                    statusElement.innerHTML = `
                        <i class="fas fa-cloud"></i> 
                        <strong>Not Downloaded</strong><br>
                        <small>Download this model to test inference locally</small>
                    `;
                    
                    if (downloadBtn) {
                        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Model';
                        downloadBtn.className = 'btn btn-outline-primary';
                        downloadBtn.disabled = false;
                    }
                    
                    if (inferenceBtn) {
                        inferenceBtn.disabled = true;
                    }
                }
            }
        } catch (error) {
            console.warn('Could not check modal model status:', error);
            const statusElement = document.getElementById(statusElementId);
            if (statusElement) {
                statusElement.className = 'alert alert-secondary';
                statusElement.innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i> 
                    <strong>Status Unknown</strong><br>
                    <small>Could not determine model download status</small>
                `;
            }
        }
    }

    async downloadModelFromModal(modelId, statusElementId, modelSafeId) {
        // Download model from the modal interface.
        try {
            const statusElement = document.getElementById(statusElementId);
            const downloadBtn = document.getElementById(`modal-download-btn-${modelSafeId}`);
            
            // Update UI
            statusElement.className = 'alert alert-info';
            statusElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <strong>Starting download...</strong>';
            
            if (downloadBtn) {
                downloadBtn.disabled = true;
                downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            }

            const response = await this.sdk.call('download_huggingface_model', {
                model_id: modelId,
                download_type: 'snapshot'
            });

            if (response.success) {
                if (response.already_downloaded) {
                    this.notifications.show(`${modelId} is already downloaded!`, 'success');
                } else {
                    this.notifications.show(`Download started for ${modelId}`, 'success');
                }
                
                // Refresh modal status
                setTimeout(() => this.checkModalModelStatus(modelId, statusElementId, modelSafeId), 1000);
            } else {
                throw new Error(response.error || 'Download failed');
            }
        } catch (error) {
            console.error('Modal download failed:', error);
            this.notifications.show(`Download failed: ${error.message}`, 'error');
            
            // Reset status
            const statusElement = document.getElementById(statusElementId);
            if (statusElement) {
                statusElement.className = 'alert alert-danger';
                statusElement.innerHTML = `<i class="fas fa-exclamation-circle"></i> <strong>Download Failed</strong><br><small>${error.message}</small>`;
            }
        }
    }

    async removeModelFromModal(modelId, statusElementId, modelSafeId) {
        // Remove model from the modal interface.
        if (!confirm(`Are you sure you want to remove ${modelId}?`)) {
            return;
        }

        try {
            const response = await this.sdk.call('remove_downloaded_model', {
                model_id: modelId,
                confirm: true
            });

            if (response.success) {
                this.notifications.show(`${modelId} removed successfully`, 'success');
                // Refresh modal status
                this.checkModalModelStatus(modelId, statusElementId, modelSafeId);
            } else {
                throw new Error(response.error || 'Failed to remove model');
            }
        } catch (error) {
            console.error('Failed to remove model from modal:', error);
            this.notifications.show(`Failed to remove model: ${error.message}`, 'error');
        }
    }

    async showDownloadedModels() {
        // Show a modal with all downloaded models.
        try {
            const response = await this.sdk.call('list_downloaded_models', {
                include_details: true
            });

            if (!response.success) {
                throw new Error(response.error || 'Failed to load downloaded models');
            }

            const models = response.models || [];
            
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Downloaded Models (${models.length})</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${models.length === 0 ? `
                                <div class="text-center py-5">
                                    <i class="fas fa-download fa-3x text-muted mb-3"></i>
                                    <h5 class="text-muted">No models downloaded yet</h5>
                                    <p class="text-muted">Download models from the Model Hub to see them here.</p>
                                </div>
                            ` : `
                                <div class="table-responsive">
                                    <table class="table table-hover">
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Size</th>
                                                <th>Downloaded</th>
                                                <th>Status</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${models.map(model => `
                                                <tr>
                                                    <td>
                                                        <strong>${model.model_id}</strong><br>
                                                        <small class="text-muted">${model.model_name}</small>
                                                    </td>
                                                    <td>${(model.actual_size_mb || 0).toFixed(1)} MB</td>
                                                    <td>${new Date(model.downloaded_at).toLocaleDateString()}</td>
                                                    <td>
                                                        ${model.exists ? 
                                                            '<span class="badge bg-success">Available</span>' : 
                                                            '<span class="badge bg-danger">Missing</span>'
                                                        }
                                                    </td>
                                                    <td>
                                                        <div class="btn-group btn-group-sm">
                                                            <button class="btn btn-outline-success" 
                                                                    onclick="dashboard.testModelInference('${model.model_id}', 'text-generation')"
                                                                    ${!model.exists ? 'disabled' : ''}>
                                                                <i class="fas fa-play"></i> Test
                                                            </button>
                                                            <button class="btn btn-outline-danger" 
                                                                    onclick="dashboard.removeDownloadedModel('${model.model_id}')">
                                                                <i class="fas fa-trash"></i> Remove
                                                            </button>
                                                        </div>
                                                    </td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            `}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="dashboard.refreshDownloadedModels()">
                                <i class="fas fa-sync"></i> Refresh
                            </button>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();

            // Clean up modal when hidden
            modal.addEventListener('hidden.bs.modal', () => {
                document.body.removeChild(modal);
            });

        } catch (error) {
            console.error('Failed to show downloaded models:', error);
            this.notifications.show(`Failed to load downloaded models: ${error.message}`, 'error');
        }
    }

    async removeDownloadedModel(modelId) {
        // Remove a downloaded model from the management interface.
        if (!confirm(`Are you sure you want to remove ${modelId}? This will delete all downloaded files.`)) {
            return;
        }

        try {
            const response = await this.sdk.call('remove_downloaded_model', {
                model_id: modelId,
                confirm: true
            });

            if (response.success) {
                this.notifications.show(`${modelId} removed successfully`, 'success');
                // Refresh the downloaded models modal
                this.showDownloadedModels();
            } else {
                throw new Error(response.error || 'Failed to remove model');
            }
        } catch (error) {
            console.error('Failed to remove downloaded model:', error);
            this.notifications.show(`Failed to remove model: ${error.message}`, 'error');
        }
    }

    async refreshDownloadedModels() {
        // Refresh the downloaded models modal.
        this.showDownloadedModels();
    }
}

// ============================================
// MODULE MANAGER
// ============================================

class ModuleManager {
    constructor(sdk) {
        this.sdk = sdk;
        this.modules = [];
        this.categories = {
            'text': { name: 'Text Processing', icon: 'fas fa-font', color: '#3b82f6' },
            'vision': { name: 'Computer Vision', icon: 'fas fa-eye', color: '#8b5cf6' },
            'audio': { name: 'Audio Processing', icon: 'fas fa-volume-up', color: '#06b6d4' },
            'code': { name: 'Code Generation', icon: 'fas fa-code', color: '#10b981' },
            'multimodal': { name: 'Multimodal AI', icon: 'fas fa-layer-group', color: '#f59e0b' }
        };
    }

    async initialize() {
        await this.loadModules();
        this.renderModules();
    }

    async loadModules() {
        try {
            // Comprehensive set of AI modules organized by category
            this.modules = [
                // Text Processing Modules
                {
                    id: 'text-generation',
                    name: 'Text Generation',
                    category: 'text',
                    description: 'Generate human-like text from prompts',
                    method: 'generate_text',
                    params: ['prompt', 'max_length', 'temperature'],
                    placeholder: 'Write a story about...'
                },
                {
                    id: 'text-classification',
                    name: 'Text Classification',
                    category: 'text',
                    description: 'Classify text into predefined categories',
                    method: 'classify_text',
                    params: ['text', 'labels'],
                    placeholder: 'emotion'
                },
                {
                    id: 'sentiment-analysis',
                    name: 'Sentiment Analysis',
                    category: 'text',
                    description: 'Analyze sentiment and emotions in text',
                    method: 'analyze_sentiment',
                    params: ['text'],
                    placeholder: 'I love this product!'
                },
                {
                    id: 'text-embedding',
                    name: 'Text Embeddings',
                    category: 'text',
                    description: 'Generate vector embeddings for text',
                    method: 'get_text_embedding',
                    params: ['text'],
                    placeholder: 'machine learning is amazing'
                },

                // Vision Processing Modules
                {
                    id: 'image-classification',
                    name: 'Image Classification',
                    category: 'vision',
                    description: 'Classify images and identify objects',
                    method: 'classify_image',
                    params: ['image_data'],
                    placeholder: 'image_data'
                },
                {
                    id: 'object-detection',
                    name: 'Object Detection',
                    category: 'vision',
                    description: 'Detect and locate objects in images',
                    method: 'detect_objects',
                    params: ['image_data'],
                    placeholder: 'image_data'
                },
                {
                    id: 'image-generation',
                    name: 'Image Generation',
                    category: 'vision',
                    description: 'Generate images from text descriptions',
                    method: 'generate_image',
                    params: ['prompt', 'style'],
                    placeholder: 'A beautiful sunset over mountains'
                },
                {
                    id: 'image-caption',
                    name: 'Image Captioning',
                    category: 'vision',
                    description: 'Generate captions for images',
                    method: 'caption_image',
                    params: ['image_data'],
                    placeholder: 'image_data'
                },

                // Audio Processing Modules
                {
                    id: 'speech-to-text',
                    name: 'Speech to Text',
                    category: 'audio',
                    description: 'Transcribe audio to text',
                    method: 'transcribe_audio',
                    params: ['audio_data', 'language'],
                    placeholder: 'audio_data/speech'
                },
                {
                    id: 'text-to-speech',
                    name: 'Text to Speech',
                    category: 'audio',
                    description: 'Generate speech from text',
                    method: 'synthesize_speech',
                    params: ['text', 'voice'],
                    placeholder: 'Hello, this is a test message'
                },
                {
                    id: 'audio-classification',
                    name: 'Audio Classification',
                    category: 'audio',
                    description: 'Classify audio content and sounds',
                    method: 'classify_audio',
                    params: ['audio_data'],
                    placeholder: 'audio_data'
                },

                // Code Processing Modules
                {
                    id: 'code-generation',
                    name: 'Code Generation',
                    category: 'code',
                    description: 'Generate code from natural language',
                    method: 'generate_code',
                    params: ['prompt', 'language'],
                    placeholder: 'Create a Python function to sort a list'
                },
                {
                    id: 'code-completion',
                    name: 'Code Completion',
                    category: 'code',
                    description: 'Complete partial code snippets',
                    method: 'complete_code',
                    params: ['code', 'language'],
                    placeholder: 'def fibonacci(n):'
                },
                {
                    id: 'code-explanation',
                    name: 'Code Explanation',
                    category: 'code',
                    description: 'Explain what code does in natural language',
                    method: 'explain_code',
                    params: ['code', 'language'],
                    placeholder: 'for i in range(10): print(i)'
                },
                {
                    id: 'code-debugging',
                    name: 'Code Debugging',
                    category: 'code',
                    description: 'Find and fix bugs in code',
                    method: 'debug_code',
                    params: ['code', 'language'],
                    placeholder: 'def add(a, b): return a + c'
                },

                // Multimodal Processing Modules
                {
                    id: 'visual-qa',
                    name: 'Visual Q&A',
                    category: 'multimodal',
                    description: 'Answer questions about images',
                    method: 'visual_question_answering',
                    params: ['image_data', 'question'],
                    placeholder: 'What do you see in this image?'
                },
                {
                    id: 'multimodal-chat',
                    name: 'Multimodal Chat',
                    category: 'multimodal',
                    description: 'Chat with AI about images and text',
                    method: 'multimodal_chat',
                    params: ['messages', 'context'],
                    placeholder: 'Describe this image and tell me a story about it'
                },
                {
                    id: 'document-processing',
                    name: 'Document Processing',
                    category: 'multimodal',
                    description: 'Extract and analyze document content',
                    method: 'process_document',
                    params: ['document_data'],
                    placeholder: 'document_data'
                }
            ];

            return this.modules;
        } catch (error) {
            console.error('Failed to load modules:', error);
            return [];
        }
    }

    renderModules() {
        const container = document.getElementById('ai-modules');
        if (!container) return;

        container.innerHTML = '';

        // Create a comprehensive organized layout
        const wrapper = document.createElement('div');
        wrapper.className = 'modules-wrapper';
        
        // Group modules by category
        const grouped = this.getGroupedModules();

        Object.entries(grouped).forEach(([categoryId, modules]) => {
            const category = this.categories[categoryId];
            if (!category) return;

            // Create category section
            const categorySection = document.createElement('div');
            categorySection.className = 'category-section';
            
            // Category header
            const categoryHeader = document.createElement('div');
            categoryHeader.className = 'category-header';
            categoryHeader.innerHTML = `
                <div class="category-title">
                    <i class="${category.icon}" style="color: ${category.color}"></i>
                    <h3>${category.name}</h3>
                    <span class="module-count">${modules.length} modules</span>
                </div>
            `;
            categorySection.appendChild(categoryHeader);

            // Modules grid for this category
            const modulesGrid = document.createElement('div');
            modulesGrid.className = 'modules-grid';
            
            modules.forEach(module => {
                const moduleCard = this.createModuleCard(module, category);
                modulesGrid.appendChild(moduleCard);
            });
            
            categorySection.appendChild(modulesGrid);
            wrapper.appendChild(categorySection);
        });

        container.appendChild(wrapper);
    }

    createModuleCard(module, category) {
        const card = document.createElement('div');
        card.className = 'module-card';
        card.innerHTML = `
            <div class="module-header">
                <div class="module-title">
                    <i class="${category.icon} module-icon" style="color: ${category.color}"></i>
                    ${module.name}
                </div>
                <div class="module-badge" style="background-color: ${category.color}20; color: ${category.color}">
                    ${category.name}
                </div>
            </div>
            <div class="module-content">
                <p class="module-description">${module.description}</p>
                <div class="form-group">
                    <label class="form-label">Parameters:</label>
                    <div class="parameter-list">
                        ${module.params.map(param => `<span class="badge bg-light text-dark me-1">${param}</span>`).join('')}
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Test Input:</label>
                    <textarea class="form-control-modern" rows="2" 
                              id="${module.id}-input" 
                              placeholder="${module.placeholder || 'Enter test data...'}"></textarea>
                </div>
                <div class="module-actions">
                    <button class="btn-modern btn-primary-modern" 
                            onclick="dashboard.modules.testModule('${module.id}')">
                        <i class="fas fa-play"></i>
                        Test Module
                    </button>
                    <button class="btn-modern btn-secondary-modern" 
                            onclick="dashboard.modules.getModuleHelp('${module.id}')">
                        <i class="fas fa-question-circle"></i>
                        Help
                    </button>
                </div>
            </div>
            <div class="result-container" id="${module.id}-result" style="display: none;"></div>
        `;

        return card;
    }

    getGroupedModules() {
        return this.modules.reduce((acc, module) => {
            if (!acc[module.category]) {
                acc[module.category] = [];
            }
            acc[module.category].push(module);
            return acc;
        }, {});
    }

    async getModuleHelp(moduleId) {
        const module = this.modules.find(m => m.id === moduleId);
        if (!module) return;

        const helpContent = `
            <div class="help-content">
                <h5>${module.name}</h5>
                <p><strong>Description:</strong> ${module.description}</p>
                <p><strong>Method:</strong> ${module.method}</p>
                <p><strong>Parameters:</strong> ${module.params.join(', ')}</p>
                <p><strong>Example:</strong> ${module.placeholder}</p>
            </div>
        `;

        // Show help in a modal or notification
        const notification = {
            type: 'info',
            title: 'Module Help',
            message: helpContent,
            duration: 10000
        };
        
        if (window.dashboard && window.dashboard.notifications) {
            window.dashboard.notifications.show(notification.message, notification.type);
        }
    }

    async testModule(moduleId, input = null) {
        const module = this.modules.find(m => m.id === moduleId);
        if (!module) {
            throw new Error('Module not found');
        }

        const testInput = input || document.getElementById(`${moduleId}-input`)?.value;
        if (!testInput?.trim()) {
            throw new Error('Please provide test input');
        }

        const resultContainer = `${moduleId}-result`;
        
        try {
            // Show loading
            const container = document.getElementById(resultContainer);
            if (container) {
                container.innerHTML = `
                    <div class="loading-spinner">
                        <div class="spinner"></div>
                    </div>
                `;
                container.style.display = 'block';
            }

            // Prepare parameters based on module type
            let params = {};
            if (module.category === 'text') {
                params = { prompt: testInput };
            } else if (module.category === 'vision') {
                params = { image_data: testInput };
            } else if (module.category === 'audio') {
                params = { audio_data: testInput };
            } else if (module.category === 'code') {
                params = { prompt: testInput, language: 'python' };
            } else if (module.category === 'multimodal') {
                params = { image_data: '', question: testInput };
            }

            // Make the API call
            const result = await this.sdk.request(module.method, params);
            
            // Display result
            if (container) {
                container.innerHTML = `
                    <div class="result-header">
                        <i class="fas fa-check-circle"></i>
                        Result
                    </div>
                    <div class="result-content">
                        ${this.formatModuleResult(result)}
                    </div>
                `;
            }

            return result;

        } catch (error) {
            // Display error
            const container = document.getElementById(resultContainer);
            if (container) {
                container.innerHTML = `
                    <div class="result-header text-danger">
                        <i class="fas fa-exclamation-circle"></i>
                        Error
                    </div>
                    <div class="result-content">
                        <div class="alert alert-danger">${error.message}</div>
                    </div>
                `;
            }
            throw error;
        }
    }

    formatModuleResult(result) {
        if (typeof result === 'string') {
            return `<div class="text-result">${result}</div>`;
        } else if (Array.isArray(result)) {
            return `<div class="array-result">${result.map(item => `<div class="array-item">${item}</div>`).join('')}</div>`;
        } else if (typeof result === 'object') {
            return `<pre class="code-block">${JSON.stringify(result, null, 2)}</pre>`;
        } else {
            return `<div class="text-result">${String(result)}</div>`;
        }
    }

    getGroupedModules() {
        const grouped = {};
        this.modules.forEach(module => {
            if (!grouped[module.category]) {
                grouped[module.category] = [];
            }
            grouped[module.category].push(module);
        });
        return grouped;
    }

    async refresh() {
        await this.loadModules();
        this.renderModules();
    }
}

// ============================================
// ANALYTICS MANAGER
// ============================================

class AnalyticsManager {
    constructor(sdk) {
        this.sdk = sdk;
        this.charts = {};
    }

    async initialize() {
        this.setupCharts();
        await this.updateCharts();
    }

    setupCharts() {
        // Performance Chart
        const perfCtx = document.getElementById('performance-chart');
        if (perfCtx) {
            this.charts.performance = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Requests Chart
        const reqCtx = document.getElementById('requests-chart');
        if (reqCtx) {
            this.charts.requests = new Chart(reqCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Successful', 'Failed'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: ['#10b981', '#ef4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        // System Chart
        const sysCtx = document.getElementById('system-chart');
        if (sysCtx) {
            this.charts.system = new Chart(sysCtx, {
                type: 'bar',
                data: {
                    labels: ['CPU Usage', 'Memory Usage', 'Active Connections'],
                    datasets: [{
                        label: 'System Metrics',
                        data: [0, 0, 0],
                        backgroundColor: ['#8b5cf6', '#06b6d4', '#f59e0b'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }

    async updateCharts() {
        try {
            const metrics = this.sdk.getMetrics();

            // Update performance chart
            if (this.charts.performance) {
                const now = new Date().toLocaleTimeString();
                this.charts.performance.data.labels.push(now);
                this.charts.performance.data.datasets[0].data.push(metrics.averageLatency);
                
                // Keep only last 20 data points
                if (this.charts.performance.data.labels.length > 20) {
                    this.charts.performance.data.labels.shift();
                    this.charts.performance.data.datasets[0].data.shift();
                }
                
                this.charts.performance.update();
            }

            // Update requests chart
            if (this.charts.requests) {
                this.charts.requests.data.datasets[0].data = [
                    metrics.successfulRequests,
                    metrics.failedRequests
                ];
                this.charts.requests.update();
            }

            // Update system chart (mock data for demonstration)
            if (this.charts.system) {
                this.charts.system.data.datasets[0].data = [
                    Math.random() * 100,
                    Math.random() * 100,
                    Math.random() * 10
                ];
                this.charts.system.update();
            }

        } catch (error) {
            console.error('Failed to update charts:', error);
        }
    }

    async refresh() {
        await this.updateCharts();
    }
}

// ============================================
// SYSTEM MANAGER
// ============================================

class SystemManager {
    constructor(sdk) {
        this.sdk = sdk;
        this.metrics = {};
    }

    async initialize() {
        await this.loadMetrics();
    }

    async loadMetrics() {
        try {
            this.metrics = await this.sdk.getSystemMetrics();
        } catch (error) {
            console.warn('System metrics not available:', error);
            this.metrics = {};
        }
    }

    async getMetrics() {
        await this.loadMetrics();
        return this.metrics;
    }

    async refresh() {
        await this.loadMetrics();
    }
}

// ============================================
// NOTIFICATION MANAGER
// ============================================

class NotificationManager {
    constructor() {
        this.notifications = [];
        this.maxNotifications = 5;
        this.container = document.getElementById('notification-container');
    }

    show(message, type = 'info', duration = 5000) {
        const id = Date.now();
        const notification = {
            id: id,
            message: message,
            type: type,
            timestamp: new Date()
        };

        this.notifications.unshift(notification);
        
        // Limit notifications
        if (this.notifications.length > this.maxNotifications) {
            this.notifications = this.notifications.slice(0, this.maxNotifications);
        }

        this.renderNotification(notification, duration);
    }

    renderNotification(notification, duration) {
        if (!this.container) return;

        const element = document.createElement('div');
        element.className = `notification ${notification.type}`;
        element.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div class="notification-content">${notification.message}</div>
                <button type="button" class="btn-close btn-close-sm" 
                        onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        `;

        this.container.appendChild(element);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (element.parentElement) {
                    element.remove();
                }
            }, duration);
        }
    }

    clear() {
        this.notifications = [];
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// ============================================
// GLOBAL INITIALIZATION
// ============================================

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ReorganizedDashboard();
});

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReorganizedDashboard;
} else if (typeof window !== 'undefined') {
    window.ReorganizedDashboard = ReorganizedDashboard;
}
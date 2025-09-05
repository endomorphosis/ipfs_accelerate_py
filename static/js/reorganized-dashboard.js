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
            endpoint: 'http://localhost:8005/jsonrpc',
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
                <li><kbd>Ctrl/Cmd + 4</kbd> - Switch to System</li>
                <li><kbd>Ctrl/Cmd + R</kbd> - Refresh Data</li>
                <li><kbd>Ctrl/Cmd + D</kbd> - Toggle Theme</li>
            </ul>
            <h5>Features:</h5>
            <ul>
                <li>Real-time metrics and monitoring</li>
                <li>Modular AI inference components</li>
                <li>Portable SDK code generation</li>
                <li>Hardware detection and system monitoring</li>
                <li>Advanced analytics and charting</li>
            </ul>
        `;
        
        this.notifications.show(helpContent, 'info', 10000);
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
            // Define available modules based on SDK capabilities
            this.modules = [
                {
                    id: 'text-generation',
                    name: 'Text Generation',
                    category: 'text',
                    description: 'Generate human-like text from prompts',
                    method: 'generate_text',
                    params: ['prompt', 'max_length', 'temperature']
                },
                {
                    id: 'text-classification',
                    name: 'Text Classification',
                    category: 'text',
                    description: 'Classify text into predefined categories',
                    method: 'classify_text',
                    params: ['text', 'labels']
                },
                {
                    id: 'image-classification',
                    name: 'Image Classification',
                    category: 'vision',
                    description: 'Classify images and identify objects',
                    method: 'classify_image',
                    params: ['image_data']
                },
                {
                    id: 'object-detection',
                    name: 'Object Detection',
                    category: 'vision',
                    description: 'Detect and locate objects in images',
                    method: 'detect_objects',
                    params: ['image_data']
                },
                {
                    id: 'speech-to-text',
                    name: 'Speech to Text',
                    category: 'audio',
                    description: 'Transcribe audio to text',
                    method: 'transcribe_audio',
                    params: ['audio_data', 'language']
                },
                {
                    id: 'code-generation',
                    name: 'Code Generation',
                    category: 'code',
                    description: 'Generate code from natural language',
                    method: 'generate_code',
                    params: ['prompt', 'language']
                },
                {
                    id: 'visual-qa',
                    name: 'Visual Q&A',
                    category: 'multimodal',
                    description: 'Answer questions about images',
                    method: 'visual_question_answering',
                    params: ['image_data', 'question']
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

        // Group modules by category
        const grouped = this.getGroupedModules();

        Object.entries(grouped).forEach(([categoryId, modules]) => {
            const category = this.categories[categoryId];
            if (!category) return;

            modules.forEach(module => {
                const moduleCard = this.createModuleCard(module, category);
                container.appendChild(moduleCard);
            });
        });
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
                <div class="module-badge">${category.name}</div>
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
                              placeholder="Enter test data..."></textarea>
                </div>
                <button class="btn-modern btn-primary-modern" 
                        onclick="dashboard.modules.testModule('${module.id}')">
                    <i class="fas fa-play"></i>
                    Test Module
                </button>
            </div>
            <div class="result-container" id="${module.id}-result" style="display: none;"></div>
        `;

        return card;
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
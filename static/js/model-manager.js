/**
 * Model Manager JavaScript Module
 * 
 * Handles Model Manager tab functionality including:
 * - Browsing available models
 * - Searching and filtering models
 * - Viewing model details
 * - Managing model operations
 * 
 * Uses MCP JavaScript SDK for all server communication
 */

// Model Manager state
const ModelManager = {
    models: [],
    currentPage: 1,
    pageSize: 20,
    totalModels: 0,
    filters: {
        query: '',
        task: '',
        hardware: ''
    },
    stats: {},
    mcpClient: null
};

/**
 * Initialize Model Manager tab
 */
async function initializeModelManager() {
    console.log('[Model Manager] Initializing...');
    
    // Initialize MCP client if not already done
    if (!ModelManager.mcpClient) {
        // Use REST API endpoint for now since MCP SDK uses JSON-RPC
        // We'll create a wrapper that uses the SDK pattern
        ModelManager.mcpClient = {
            // Wrapper functions that call REST endpoints but follow SDK pattern
            getStats: async () => {
                const response = await fetch('/api/mcp/models/stats');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            },
            searchModels: async (query, filters) => {
                const params = new URLSearchParams({ q: query || '', limit: 20 });
                if (filters && filters.task) params.append('task', filters.task);
                if (filters && filters.hardware) params.append('hardware', filters.hardware);
                
                console.log('[Model Manager] Fetching models with params:', params.toString());
                const response = await fetch(`/api/mcp/models/search?${params}`);
                console.log('[Model Manager] Search response status:', response.status);
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                console.log('[Model Manager] Search response data:', data);
                return data;
            },
            getModelDetails: async (modelId) => {
                const response = await fetch(`/api/mcp/models/${encodeURIComponent(modelId)}/details`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            }
        };
    }
    
    // Load initial stats
    await loadModelStats();
    
    // Load models
    await loadModels();
    
    // Setup event listeners
    setupModelManagerListeners();
    
    console.log('[Model Manager] Initialization complete');
}

/**
 * Setup event listeners for Model Manager
 */
function setupModelManagerListeners() {
    // Search button
    const searchBtn = document.getElementById('mm-search-btn');
    if (searchBtn) {
        searchBtn.addEventListener('click', handleModelSearch);
    }
    
    // Search input - search on Enter key
    const searchInput = document.getElementById('mm-search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleModelSearch();
            }
        });
    }
    
    // Filter dropdowns
    const taskFilter = document.getElementById('mm-task-filter');
    if (taskFilter) {
        taskFilter.addEventListener('change', handleModelSearch);
    }
    
    const hardwareFilter = document.getElementById('mm-hardware-filter');
    if (hardwareFilter) {
        hardwareFilter.addEventListener('change', handleModelSearch);
    }
    
    // Refresh button
    const refreshBtn = document.getElementById('mm-refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadModels();
            loadModelStats();
        });
    }
}

/**
 * Load model statistics
 */
async function loadModelStats() {
    console.log('[Model Manager] Loading stats...');
    
    try {
        const data = await ModelManager.mcpClient.getStats();
        ModelManager.stats = data;
        
        // Update stats display
        updateStatsDisplay(data);
        
        console.log('[Model Manager] Stats loaded:', data);
    } catch (error) {
        console.error('[Model Manager] Error loading stats:', error);
        showModelManagerToast('Failed to load model statistics', 'error');
    }
}

/**
 * Update statistics display
 */
function updateStatsDisplay(stats) {
    // Update stat cards
    const totalModelsEl = document.getElementById('mm-total-models');
    if (totalModelsEl) {
        totalModelsEl.textContent = stats.total_cached_models || 0;
    }
    
    const perfModelsEl = document.getElementById('mm-perf-models');
    if (perfModelsEl) {
        perfModelsEl.textContent = stats.models_with_performance || 0;
    }
    
    const compatModelsEl = document.getElementById('mm-compat-models');
    if (compatModelsEl) {
        compatModelsEl.textContent = stats.models_with_compatibility || 0;
    }
    
    // Show fallback warning if applicable
    if (stats.fallback) {
        showModelManagerToast(stats.message || 'Using fallback model database', 'warning');
    }
}

/**
 * Load models from API
 */
async function loadModels(page = 1) {
    console.log('[Model Manager] Loading models...');
    
    const modelsList = document.getElementById('mm-models-list');
    if (modelsList) {
        modelsList.innerHTML = '<div class="loading-spinner"><div class="spinner"></div>Loading models...</div>';
    }
    
    try {
        // Ensure mcpClient is initialized before using it
        if (!ModelManager.mcpClient) {
            console.log('[Model Manager] mcpClient not initialized, initializing now...');
            // Initialize the client inline if not already done
            ModelManager.mcpClient = {
                getStats: async () => {
                    const response = await fetch('/api/mcp/models/stats');
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return await response.json();
                },
                searchModels: async (query, filters) => {
                    const params = new URLSearchParams({ q: query || '', limit: 20 });
                    if (filters && filters.task) params.append('task', filters.task);
                    if (filters && filters.hardware) params.append('hardware', filters.hardware);
                    
                    console.log('[Model Manager] Fetching models with params:', params.toString());
                    const response = await fetch(`/api/mcp/models/search?${params}`);
                    console.log('[Model Manager] Search response status:', response.status);
                    
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const data = await response.json();
                    console.log('[Model Manager] Search response data:', data);
                    return data;
                },
                getModelDetails: async (modelId) => {
                    const response = await fetch(`/api/mcp/models/${encodeURIComponent(modelId)}/details`);
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return await response.json();
                }
            };
        }
        
        const data = await ModelManager.mcpClient.searchModels(
            ModelManager.filters.query,
            {
                task: ModelManager.filters.task,
                hardware: ModelManager.filters.hardware
            }
        );
        
        ModelManager.models = data.results || [];
        ModelManager.totalModels = data.total || 0;
        
        // Display models
        displayModels(data.results, data.fallback);
        
        const resultsCount = data.results ? data.results.length : 0;
        console.log(`[Model Manager] Loaded ${resultsCount} models`);
    } catch (error) {
        console.error('[Model Manager] Error loading models:', error);
        if (modelsList) {
            modelsList.innerHTML = `
                <div class="alert alert-danger">
                    <h5>Failed to load models</h5>
                    <p>${error.message}</p>
                </div>
            `;
        }
        showModelManagerToast('Failed to load models', 'error');
    }
}

/**
 * Display models in the list
 */
function displayModels(models, isFallback = false) {
    const modelsList = document.getElementById('mm-models-list');
    if (!modelsList) return;
    
    // Handle undefined or null models
    if (!models || !Array.isArray(models)) {
        console.warn('[Model Manager] Invalid models data received:', models);
        modelsList.innerHTML = `
            <div class="alert alert-warning">
                <h5>No models data</h5>
                <p>The server response did not contain model information. Please try refreshing.</p>
            </div>
        `;
        return;
    }
    
    if (models.length === 0) {
        modelsList.innerHTML = `
            <div class="alert alert-info">
                <h5>No models found</h5>
                <p>Try adjusting your search criteria or filters.</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    if (isFallback) {
        html += `
            <div class="alert alert-warning mb-3">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Using fallback database</strong> - HuggingFace Hub scanner not available
            </div>
        `;
    }
    
    models.forEach(model => {
        const modelInfo = model.model_info || {};
        const performance = model.performance || {};
        const compatibility = model.compatibility || {};
        const modelId = model.model_id || 'unknown';
        
        html += `
            <div class="model-card" data-model-id="${modelId}">
                <div class="model-card-header">
                    <div class="model-title">
                        <h4>${modelInfo.model_name || modelId}</h4>
                        <span class="model-task-badge">${modelInfo.pipeline_tag || 'general'}</span>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-sm btn-info" onclick="viewModelDetails('${modelId}')">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </div>
                </div>
                <div class="model-card-body">
                    <p class="model-description">${modelInfo.description || 'No description available'}</p>
                    <div class="model-stats">
                        <span><i class="fas fa-download"></i> ${formatNumber(modelInfo.downloads || 0)} downloads</span>
                        <span><i class="fas fa-heart"></i> ${formatNumber(modelInfo.likes || 0)} likes</span>
                        ${modelInfo.architecture ? `<span><i class="fas fa-microchip"></i> ${modelInfo.architecture}</span>` : ''}
                    </div>
                    ${performance.parameters ? `
                        <div class="model-performance">
                            <span><strong>Parameters:</strong> ${performance.parameters}</span>
                            ${performance.memory_gb ? `<span><strong>Memory:</strong> ${performance.memory_gb} GB</span>` : ''}
                        </div>
                    ` : ''}
                    ${Object.keys(compatibility).length > 0 ? `
                        <div class="model-compatibility">
                            <strong>Compatible with:</strong>
                            ${compatibility.supports_cpu ? '<span class="compat-badge">CPU</span>' : ''}
                            ${compatibility.supports_gpu ? '<span class="compat-badge">GPU</span>' : ''}
                            ${compatibility.supports_mps ? '<span class="compat-badge">MPS</span>' : ''}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    modelsList.innerHTML = html;
}

/**
 * Handle model search
 */
async function handleModelSearch() {
    const searchInput = document.getElementById('mm-search-input');
    const taskFilter = document.getElementById('mm-task-filter');
    const hardwareFilter = document.getElementById('mm-hardware-filter');
    
    ModelManager.filters.query = searchInput ? searchInput.value : '';
    ModelManager.filters.task = taskFilter ? taskFilter.value : '';
    ModelManager.filters.hardware = hardwareFilter ? hardwareFilter.value : '';
    
    await loadModels();
}

/**
 * View model details
 */
async function viewModelDetails(modelId) {
    console.log(`[Model Manager] Loading details for: ${modelId}`);
    
    try {
        const data = await ModelManager.mcpClient.getModelDetails(modelId);
        showModelDetailsModal(data);
        
    } catch (error) {
        console.error('[Model Manager] Error loading model details:', error);
        showModelManagerToast('Failed to load model details', 'error');
    }
}

/**
 * Show model details in a modal
 */
function showModelDetailsModal(modelData) {
    const modalHtml = `
        <div class="modal-overlay" id="model-details-modal" onclick="closeModelDetailsModal()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3>${modelData.model_info?.model_name || modelData.model_id}</h3>
                    <button class="modal-close" onclick="closeModelDetailsModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="detail-section">
                        <h4>Model Information</h4>
                        <table class="details-table">
                            <tr><td><strong>Model ID:</strong></td><td>${modelData.model_id}</td></tr>
                            <tr><td><strong>Task:</strong></td><td>${modelData.model_info?.pipeline_tag || 'N/A'}</td></tr>
                            <tr><td><strong>Architecture:</strong></td><td>${modelData.model_info?.architecture || 'N/A'}</td></tr>
                            <tr><td><strong>Downloads:</strong></td><td>${formatNumber(modelData.model_info?.downloads || 0)}</td></tr>
                            <tr><td><strong>Likes:</strong></td><td>${formatNumber(modelData.model_info?.likes || 0)}</td></tr>
                        </table>
                    </div>
                    
                    ${modelData.performance && Object.keys(modelData.performance).length > 0 ? `
                        <div class="detail-section">
                            <h4>Performance</h4>
                            <table class="details-table">
                                ${modelData.performance.parameters ? `<tr><td><strong>Parameters:</strong></td><td>${modelData.performance.parameters}</td></tr>` : ''}
                                ${modelData.performance.memory_gb ? `<tr><td><strong>Memory:</strong></td><td>${modelData.performance.memory_gb} GB</td></tr>` : ''}
                                ${modelData.performance.latency_ms ? `<tr><td><strong>Latency:</strong></td><td>${modelData.performance.latency_ms} ms</td></tr>` : ''}
                                ${modelData.performance.throughput_tokens_per_sec ? `<tr><td><strong>Throughput:</strong></td><td>${modelData.performance.throughput_tokens_per_sec} tokens/sec</td></tr>` : ''}
                            </table>
                        </div>
                    ` : ''}
                    
                    ${modelData.compatibility && Object.keys(modelData.compatibility).length > 0 ? `
                        <div class="detail-section">
                            <h4>Compatibility</h4>
                            <table class="details-table">
                                ${modelData.compatibility.supports_cpu !== undefined ? `<tr><td><strong>CPU:</strong></td><td>${modelData.compatibility.supports_cpu ? '✓ Supported' : '✗ Not supported'}</td></tr>` : ''}
                                ${modelData.compatibility.supports_gpu !== undefined ? `<tr><td><strong>GPU:</strong></td><td>${modelData.compatibility.supports_gpu ? '✓ Supported' : '✗ Not supported'}</td></tr>` : ''}
                                ${modelData.compatibility.supports_mps !== undefined ? `<tr><td><strong>MPS:</strong></td><td>${modelData.compatibility.supports_mps ? '✓ Supported' : '✗ Not supported'}</td></tr>` : ''}
                                ${modelData.compatibility.min_ram_gb ? `<tr><td><strong>Min RAM:</strong></td><td>${modelData.compatibility.min_ram_gb} GB</td></tr>` : ''}
                                ${modelData.compatibility.recommended_hardware ? `<tr><td><strong>Recommended:</strong></td><td>${modelData.compatibility.recommended_hardware}</td></tr>` : ''}
                            </table>
                        </div>
                    ` : ''}
                    
                    <div class="detail-section">
                        <p class="model-description">${modelData.model_info?.description || 'No description available'}</p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeModelDetailsModal()">Close</button>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page
    const modalContainer = document.getElementById('modal-container') || document.body;
    const modalDiv = document.createElement('div');
    modalDiv.innerHTML = modalHtml;
    modalContainer.appendChild(modalDiv.firstElementChild);
}

/**
 * Close model details modal
 */
function closeModelDetailsModal() {
    const modal = document.getElementById('model-details-modal');
    if (modal) {
        modal.remove();
    }
}

/**
 * Show toast notification
 */
function showModelManagerToast(message, type = 'info') {
    // Try to use global toast function if available
    if (typeof showToast === 'function') {
        showToast(message, type);
        return;
    }
    
    // Fallback to console
    console.log(`[Model Manager Toast] ${type.toUpperCase()}: ${message}`);
}

/**
 * Format large numbers
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Export functions for global access
window.ModelManager = ModelManager;
window.initializeModelManager = initializeModelManager;
window.handleModelSearch = handleModelSearch;
window.viewModelDetails = viewModelDetails;
window.closeModelDetailsModal = closeModelDetailsModal;

console.log('[Model Manager] Module loaded');

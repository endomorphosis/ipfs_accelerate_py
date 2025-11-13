// IPFS Accelerate MCP Server Dashboard JavaScript

// Global state
let currentTab = 'overview';
let searchResults = [];
let compatibilityResults = [];
let autoRefreshInterval = null;

// Utility function for user notifications
function showToast(message, type = 'info', duration = 3000) {
    console.log(`[Dashboard] ${type.toUpperCase()}: ${message}`);
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Tab Management
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        if (content) {
            content.classList.remove('active');
        }
    });
    
    // Remove active class from all tab buttons (support both .tab-button and .nav-tab)
    const tabButtons = document.querySelectorAll('.tab-button, .nav-tab');
    tabButtons.forEach(button => {
        if (button) {
            button.classList.remove('active');
        }
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Add active class to selected tab button
    const selectedButton = document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (selectedButton) {
        selectedButton.classList.add('active');
    }
    
    currentTab = tabName;
    
    // Perform tab-specific initialization
    initializeTab(tabName);
}

function initializeTab(tabName) {
    switch (tabName) {
        case 'overview':
            refreshServerStatus();
            break;
        case 'ai-inference':
            updateInferenceForm();
            break;
        case 'model-manager':
            refreshModels();
            break;
        case 'model-browser':
            if (typeof initializeModelManager === 'function') {
                initializeModelManager();
            }
            // Load database stats when model browser tab opens
            loadDatabaseStats();
            break;
        case 'queue-monitor':
            refreshQueue();
            break;
        case 'github-workflows':
            // GitHub workflows tab initialization handled by github-workflows.js
            if (typeof githubManager !== 'undefined' && githubManager) {
                console.log('[Dashboard] Initializing GitHub Workflows tab');
                githubManager.initialize();
            } else {
                console.warn('[Dashboard] GitHub Workflows manager not available');
            }
            break;
        case 'mcp-tools':
            refreshTools();
            break;
        case 'coverage':
            refreshCoverageMatrix();
            break;
        case 'system-logs':
            refreshLogs();
            break;
    }
}

// AI Inference Functions
function updateInferenceForm() {
    const inferenceType = document.getElementById('inference-type')?.value || 'text-generate';
    const dynamicFields = document.getElementById('dynamic-fields');
    
    if (!dynamicFields) return;
    
    let fieldsHtml = '';
    
    switch (inferenceType) {
        case 'text-generate':
            fieldsHtml = `
                <label for="prompt">Prompt*:</label>
                <textarea id="prompt" placeholder="Enter your text generation prompt..." required></textarea>
                <label for="max-length">Max Length:</label>
                <input type="number" id="max-length" value="100" min="1" max="2048">
                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
                <label for="top-p">Top-p:</label>
                <input type="number" id="top-p" value="0.9" min="0" max="1" step="0.1">
                <label for="top-k">Top-k:</label>
                <input type="number" id="top-k" value="50" min="1" max="100">
            `;
            break;
        case 'text-classify':
            fieldsHtml = `
                <label for="text-input">Text to Classify*:</label>
                <textarea id="text-input" placeholder="Enter text to classify..." required></textarea>
                <label for="top-k-results">Top-k Results:</label>
                <input type="number" id="top-k-results" value="5" min="1" max="20">
            `;
            break;
        case 'text-embeddings':
            fieldsHtml = `
                <label for="text-input">Text for Embeddings*:</label>
                <textarea id="text-input" placeholder="Enter text to generate embeddings..." required></textarea>
            `;
            break;
        case 'text-translate':
            fieldsHtml = `
                <label for="text-input">Text to Translate*:</label>
                <textarea id="text-input" placeholder="Enter text to translate..." required></textarea>
                <label for="source-lang">Source Language:</label>
                <input type="text" id="source-lang" placeholder="auto" value="auto">
                <label for="target-lang">Target Language*:</label>
                <input type="text" id="target-lang" placeholder="es" required>
            `;
            break;
        case 'text-summarize':
            fieldsHtml = `
                <label for="text-input">Text to Summarize*:</label>
                <textarea id="text-input" placeholder="Enter long text to summarize..." required></textarea>
                <label for="max-summary-length">Max Summary Length:</label>
                <input type="number" id="max-summary-length" value="150" min="50" max="500">
            `;
            break;
        case 'text-question':
            fieldsHtml = `
                <label for="context">Context*:</label>
                <textarea id="context" placeholder="Enter context paragraph..." required></textarea>
                <label for="question">Question*:</label>
                <input type="text" id="question" placeholder="Enter your question..." required>
            `;
            break;
        case 'audio-transcribe':
            fieldsHtml = `
                <label for="audio-file">Audio File*:</label>
                <input type="file" id="audio-file" accept="audio/*" required>
                <label for="language">Language:</label>
                <input type="text" id="language" placeholder="en" value="en">
                <label for="task">Task:</label>
                <select id="task">
                    <option value="transcribe">Transcribe</option>
                    <option value="translate">Translate to English</option>
                </select>
            `;
            break;
        case 'audio-classify':
            fieldsHtml = `
                <label for="audio-file">Audio File*:</label>
                <input type="file" id="audio-file" accept="audio/*" required>
                <label for="top-k-results">Top-k Results:</label>
                <input type="number" id="top-k-results" value="5" min="1" max="20">
            `;
            break;
        case 'audio-generate':
            fieldsHtml = `
                <label for="prompt">Audio Generation Prompt*:</label>
                <textarea id="prompt" placeholder="Describe the audio you want to generate..." required></textarea>
                <label for="duration">Duration (seconds):</label>
                <input type="number" id="duration" value="10" min="1" max="60">
                <label for="sample-rate">Sample Rate:</label>
                <input type="number" id="sample-rate" value="22050" min="8000" max="48000">
            `;
            break;
        case 'audio-synthesize':
            fieldsHtml = `
                <label for="text-input">Text to Synthesize*:</label>
                <textarea id="text-input" placeholder="Enter text to convert to speech..." required></textarea>
                <label for="voice">Voice:</label>
                <select id="voice">
                    <option value="default">Default</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
                <label for="language">Language:</label>
                <input type="text" id="language" placeholder="en" value="en">
            `;
            break;
        case 'image-classify':
            fieldsHtml = `
                <label for="image-file">Image File*:</label>
                <input type="file" id="image-file" accept="image/*" required>
                <label for="top-k-results">Top-k Results:</label>
                <input type="number" id="top-k-results" value="5" min="1" max="20">
            `;
            break;
        case 'image-detect':
            fieldsHtml = `
                <label for="image-file">Image File*:</label>
                <input type="file" id="image-file" accept="image/*" required>
                <label for="confidence-threshold">Confidence Threshold:</label>
                <input type="number" id="confidence-threshold" value="0.5" min="0" max="1" step="0.1">
            `;
            break;
        case 'image-segment':
            fieldsHtml = `
                <label for="image-file">Image File*:</label>
                <input type="file" id="image-file" accept="image/*" required>
                <label for="segmentation-type">Segmentation Type:</label>
                <select id="segmentation-type">
                    <option value="semantic">Semantic</option>
                    <option value="instance">Instance</option>
                    <option value="panoptic">Panoptic</option>
                </select>
            `;
            break;
        case 'image-generate':
            fieldsHtml = `
                <label for="prompt">Image Generation Prompt*:</label>
                <textarea id="prompt" placeholder="Describe the image you want to generate..." required></textarea>
                <label for="width">Width:</label>
                <input type="number" id="width" value="512" min="128" max="1024" step="64">
                <label for="height">Height:</label>
                <input type="number" id="height" value="512" min="128" max="1024" step="64">
                <label for="inference-steps">Inference Steps:</label>
                <input type="number" id="inference-steps" value="20" min="1" max="100">
                <label for="guidance-scale">Guidance Scale:</label>
                <input type="number" id="guidance-scale" value="7.5" min="1" max="20" step="0.5">
                <label for="output-file">Output File:</label>
                <input type="text" id="output-file" placeholder="image.png">
            `;
            break;
        case 'multimodal-caption':
            fieldsHtml = `
                <label for="image-file">Image File*:</label>
                <input type="file" id="image-file" accept="image/*" required>
                <label for="max-caption-length">Max Caption Length:</label>
                <input type="number" id="max-caption-length" value="50" min="10" max="200">
            `;
            break;
        case 'multimodal-vqa':
            fieldsHtml = `
                <label for="image-file">Image File*:</label>
                <input type="file" id="image-file" accept="image/*" required>
                <label for="question">Question*:</label>
                <input type="text" id="question" placeholder="What do you see in this image?" required>
            `;
            break;
        case 'multimodal-document':
            fieldsHtml = `
                <label for="document-file">Document File*:</label>
                <input type="file" id="document-file" accept=".pdf,.doc,.docx,.txt" required>
                <label for="query">Query*:</label>
                <input type="text" id="query" placeholder="What information are you looking for?" required>
            `;
            break;
        case 'specialized-code':
            fieldsHtml = `
                <label for="prompt">Code Generation Prompt*:</label>
                <textarea id="prompt" placeholder="Describe the code you want to generate..." required></textarea>
                <label for="programming-language">Programming Language:</label>
                <select id="programming-language">
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                    <option value="go">Go</option>
                    <option value="rust">Rust</option>
                </select>
                <label for="max-length">Max Length:</label>
                <input type="number" id="max-length" value="200" min="50" max="1000">
                <label for="output-file">Output File:</label>
                <input type="text" id="output-file" placeholder="code.py">
            `;
            break;
        case 'specialized-timeseries':
            fieldsHtml = `
                <label for="data-file">Time Series Data File*:</label>
                <input type="file" id="data-file" accept=".csv,.json" required>
                <label for="forecast-horizon">Forecast Horizon:</label>
                <input type="number" id="forecast-horizon" value="10" min="1" max="100">
            `;
            break;
        case 'specialized-tabular':
            fieldsHtml = `
                <label for="data-file">Tabular Data File*:</label>
                <input type="file" id="data-file" accept=".csv,.json" required>
                <label for="task-type">Task Type:</label>
                <select id="task-type">
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                    <option value="clustering">Clustering</option>
                </select>
                <label for="target-column">Target Column:</label>
                <input type="text" id="target-column" placeholder="target">
            `;
            break;
    }
    
    dynamicFields.innerHTML = fieldsHtml;
}

function runInference() {
    const inferenceType = document.getElementById('inference-type')?.value;
    const modelId = document.getElementById('model-id')?.value;
    const resultsDiv = document.getElementById('inference-results');
    const executionTimeSpan = document.getElementById('execution-time');
    const modelUsedSpan = document.getElementById('model-used');
    
    if (!resultsDiv) return;
    
    // Show loading state
    resultsDiv.innerHTML = '<div class="spinner"></div>Loading...';
    
    // Simulate inference execution
    setTimeout(() => {
        const mockResults = generateMockInferenceResult(inferenceType);
        resultsDiv.innerHTML = mockResults.result;
        
        if (executionTimeSpan) {
            executionTimeSpan.textContent = mockResults.executionTime;
        }
        if (modelUsedSpan) {
            modelUsedSpan.textContent = mockResults.modelUsed;
        }
    }, 2000);
}

function generateMockInferenceResult(inferenceType) {
    const results = {
        'text-generate': {
            result: 'Quantum computing is a revolutionary technology that leverages quantum mechanical phenomena...',
            executionTime: '1.2s',
            modelUsed: 'gpt2-medium'
        },
        'text-classify': {
            result: 'Classification: Positive (confidence: 0.92)',
            executionTime: '0.8s',
            modelUsed: 'distilbert-base-uncased'
        },
        'text-embeddings': {
            result: 'Generated 768-dimensional embedding vector',
            executionTime: '0.5s',
            modelUsed: 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'image-classify': {
            result: 'Top predictions: 1. Golden Retriever (0.89), 2. Labrador (0.08), 3. Dog (0.03)',
            executionTime: '1.5s',
            modelUsed: 'resnet50'
        }
    };
    
    return results[inferenceType] || {
        result: `Inference completed for ${inferenceType}`,
        executionTime: '1.0s',
        modelUsed: 'auto-selected'
    };
}

function clearInferenceForm() {
    const form = document.querySelector('.inference-form');
    if (form) {
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            if (input.type === 'checkbox' || input.type === 'radio') {
                input.checked = false;
            } else {
                input.value = '';
            }
        });
    }
    
    const resultsDiv = document.getElementById('inference-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = '<p>Ready to run inference...</p>';
    }
}

// HuggingFace Model Search Functions
function searchHuggingFace() {
    const query = document.getElementById('hf-search')?.value;
    const taskFilter = document.getElementById('task-filter')?.value;
    const sizeFilter = document.getElementById('size-filter')?.value;
    const resultsDiv = document.getElementById('hf-search-results');
    
    if (!query || !resultsDiv) {
        if (resultsDiv) {
            resultsDiv.innerHTML = '<p>Please enter a search term.</p>';
        }
        return;
    }
    
    // Show loading state
    resultsDiv.innerHTML = '<div class="spinner"></div>Searching HuggingFace Hub...';
    
    // Build query parameters - use MCP API endpoint
    const params = new URLSearchParams({
        q: query,
        limit: '20'
    });
    
    if (taskFilter) {
        params.append('task', taskFilter);
    }
    
    console.log(`[Dashboard] Searching HuggingFace with query: ${query}, task: ${taskFilter}`);
    
    // Make API call to search models using MCP endpoint
    fetch(`/api/mcp/models/search?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`[Dashboard] Search results:`, data);
            displayHFResults(data);
            updateSearchStats(data);
        })
        .catch(error => {
            console.error('[Dashboard] Search error:', error);
            resultsDiv.innerHTML = `<p>Search failed: ${error.message}. Please try again.</p>`;
            showToast('Search failed', 'error');
        });
}

function displayHFResults(data) {
    const resultsDiv = document.getElementById('hf-search-results');
    
    // Handle the MCP API response structure
    const models = data.results || data.models || [];
    
    if (!resultsDiv) {
        return;
    }
    
    if (models.length === 0) {
        resultsDiv.innerHTML = '<p>No models found. Try a different search term.</p>';
        return;
    }
    
    let html = '';
    models.forEach(modelData => {
        // Handle both direct model objects and wrapped model objects
        const model = modelData.model_info || modelData;
        const modelId = modelData.model_id || model.id || model.model_id;
        const safeId = (modelId || '').replace(/[^a-zA-Z0-9]/g, '-');
        
        html += `
            <div class="model-result">
                <div class="model-header">
                    <div class="model-title">${model.model_name || model.title || modelId}</div>
                    <div class="model-actions">
                        <button class="btn btn-sm btn-warning" onclick="testModelFromHF('${modelId}')">🔧 Test</button>
                        <button class="btn btn-sm btn-success" id="download-btn-${safeId}" onclick="downloadModel('${modelId}', 'download-btn-${safeId}')">⬇️ Download</button>
                    </div>
                </div>
                <div class="model-meta">
                    <span>📊 ${model.downloads || 0} downloads</span>
                    <span>💙 ${model.likes || 0} likes</span>
                    <span>🏷️ ${model.pipeline_tag || model.task || 'General'}</span>
                </div>
                <div class="model-description">${model.description || 'No description available'}</div>
                ${model.architecture ? `<div class="model-architecture">Architecture: ${model.architecture}</div>` : ''}
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

function testModelFromHF(modelId) {
    const testModelInput = document.getElementById('test-model-id');
    if (testModelInput) {
        testModelInput.value = modelId;
    }
    
    // Switch to model manager tab and scroll to compatibility testing
    showTab('model-manager');
    setTimeout(() => {
        const compatibilitySection = document.querySelector('.compatibility-form');
        if (compatibilitySection) {
            compatibilitySection.scrollIntoView({ behavior: 'smooth' });
        }
    }, 100);
}

function downloadModel(modelId, buttonId) {
    console.log(`[Dashboard] Downloading model: ${modelId}`);
    showToast(`Initiating download for: ${modelId}`, 'info');
    
    // Update button state
    const button = buttonId ? document.getElementById(buttonId) : null;
    if (button) {
        button.disabled = true;
        button.innerHTML = '⏳ Downloading...';
        button.classList.remove('btn-success');
        button.classList.add('btn-secondary');
    }
    
    // Call the MCP API to download the model
    fetch('/api/mcp/models/download', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_id: modelId
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log(`[Dashboard] Download response:`, data);
        if (data.status === 'success') {
            showToast(`✓ Model ${modelId} downloaded successfully`, 'success');
            
            // Update button to show success
            if (button) {
                button.innerHTML = '✓ Downloaded';
                button.classList.remove('btn-secondary');
                button.classList.add('btn-info');
            }
            
            // Refresh the models list in Model Browser tab
            if (typeof loadModels === 'function') {
                loadModels();
            }
        } else {
            showToast(`Download failed: ${data.message || 'Unknown error'}`, 'error');
            
            // Reset button on failure
            if (button) {
                button.disabled = false;
                button.innerHTML = '⬇️ Download';
                button.classList.remove('btn-secondary');
                button.classList.add('btn-success');
            }
        }
    })
    .catch(error => {
        console.error('[Dashboard] Download error:', error);
        showToast(`Download failed: ${error.message}`, 'error');
        
        // Reset button on error
        if (button) {
            button.disabled = false;
            button.innerHTML = '⬇️ Download';
            button.classList.remove('btn-secondary');
            button.classList.add('btn-success');
        }
    });
}

function clearHFResults() {
    const resultsDiv = document.getElementById('hf-search-results');
    const searchInput = document.getElementById('hf-search');
    
    if (resultsDiv) {
        resultsDiv.innerHTML = '';
    }
    if (searchInput) {
        searchInput.value = '';
    }
    
    updateSearchStats({ models: [] });
}

function updateSearchStats(data) {
    const totalIndexedSpan = document.getElementById('total-indexed');
    const hfModelsSpan = document.getElementById('hf-models-count');
    const compatibleSpan = document.getElementById('compatible-models');
    const testedSpan = document.getElementById('tested-models');
    
    // Handle MCP API response structure
    const models = data.results || data.models || [];
    const total = data.total || models.length;
    
    // These stats should represent the search results, not the full database
    // We'll load database stats separately
    if (totalIndexedSpan && data.database_total !== undefined) {
        totalIndexedSpan.textContent = data.database_total;
    }
    if (hfModelsSpan && data.database_hf !== undefined) {
        hfModelsSpan.textContent = data.database_hf;
    }
    if (compatibleSpan && data.database_compatible !== undefined) {
        compatibleSpan.textContent = data.database_compatible;
    }
    if (testedSpan && data.database_tested !== undefined) {
        testedSpan.textContent = data.database_tested;
    }
}

// Load database statistics with retry logic
async function loadDatabaseStats(retries = 3) {
    console.log('[Dashboard] Loading database statistics...');
    
    for (let attempt = 0; attempt < retries; attempt++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch('/api/mcp/models/stats', {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('[Dashboard] Database stats:', data);
            
            const totalIndexedSpan = document.getElementById('total-indexed');
            const hfModelsSpan = document.getElementById('hf-models-count');
            const compatibleSpan = document.getElementById('compatible-models');
            const testedSpan = document.getElementById('tested-models');
            
            if (totalIndexedSpan) totalIndexedSpan.textContent = data.total_indexed || 0;
            if (hfModelsSpan) hfModelsSpan.textContent = data.hf_models || 0;
            if (compatibleSpan) compatibleSpan.textContent = data.compatible_models || 0;
            if (testedSpan) testedSpan.textContent = data.tested_models || 0;
            
            if (data.fallback) {
                console.warn('[Dashboard] Using fallback statistics');
            }
            
            // Success, break out of retry loop
            return;
            
        } catch (error) {
            const isNetworkError = error.toString().includes('fetch') || 
                                   error.toString().includes('NetworkError') ||
                                   error.toString().includes('ERR_NETWORK') ||
                                   error.name === 'AbortError';
            
            if (isNetworkError && attempt < retries - 1) {
                const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
                console.warn(`[Dashboard] Network error loading stats (attempt ${attempt + 1}/${retries}), retrying in ${delay}ms...`, error.message);
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                console.error('[Dashboard] Failed to load database stats after all retries:', error);
                // Keep showing zeros on error
                return;
            }
        }
    }
}

// Hardware Compatibility Testing
function testModelCompatibility() {
    const modelId = document.getElementById('test-model-id')?.value;
    const resultsDiv = document.getElementById('compatibility-results');
    
    if (!modelId) {
        showToast('Please enter a model ID to test', 'warning');
        return;
    }
    
    if (!resultsDiv) return;
    
    // Get selected platforms
    const platforms = [];
    const platformCheckboxes = [
        'test-cpu', 'test-cuda', 'test-rocm', 'test-openvino', 
        'test-mps', 'test-webgpu', 'test-directml', 'test-onnx'
    ];
    
    platformCheckboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox && checkbox.checked) {
            platforms.push(id.replace('test-', ''));
        }
    });
    
    if (platforms.length === 0) {
        showToast('Please select at least one hardware platform to test', 'warning');
        return;
    }
    
    // Get test configuration
    const batchSize = document.getElementById('batch-size')?.value || 1;
    const seqLength = document.getElementById('seq-length')?.value || 512;
    const precision = document.getElementById('precision')?.value || 'FP32';
    const iterations = document.getElementById('iterations')?.value || 10;
    
    // Show loading state
    resultsDiv.innerHTML = '<div class="spinner"></div>Testing model compatibility...';
    
    // Make API call to test compatibility
    const params = new URLSearchParams({
        model: modelId,
        platforms: platforms.join(','),
        batch_size: batchSize,
        seq_length: seqLength,
        precision: precision,
        iterations: iterations
    });
    
    fetch(`/api/models/test?${params}`)
        .then(response => response.json())
        .then(data => {
            displayCompatibilityResults(data);
            updateCompatibilityStats(data);
        })
        .catch(error => {
            console.error('Testing error:', error);
            resultsDiv.innerHTML = '<p>Testing failed. Please try again.</p>';
        });
}

function displayCompatibilityResults(data) {
    const resultsDiv = document.getElementById('compatibility-results');
    if (!resultsDiv || !data.results) return;
    
    let html = '';
    data.results.forEach(result => {
        const statusClass = result.status.toLowerCase();
        html += `
            <div class="compatibility-result ${statusClass}">
                <div class="result-header">
                    <strong>${result.platform.toUpperCase()}</strong>
                    <span class="status-badge ${statusClass}">${getStatusIcon(result.status)} ${result.status.toUpperCase()}</span>
                </div>
                <div class="result-details">
                    <div>Memory Usage: ${result.memory}</div>
                    <div>Performance: ${result.performance}</div>
                    <div>Test Time: ${result.test_time || '2.5s'}</div>
                    <div>Notes: ${result.notes}</div>
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

function getStatusIcon(status) {
    const icons = {
        'optimal': '🟢',
        'compatible': '🟡',
        'limited': '🟠',
        'unsupported': '🔴'
    };
    return icons[status.toLowerCase()] || '⚪';
}

function updateCompatibilityStats(data) {
    const compatibleSpan = document.getElementById('compatible-models');
    const testedSpan = document.getElementById('tested-models');
    
    if (data.results) {
        const compatibleCount = data.results.filter(r => r.status !== 'unsupported').length;
        if (compatibleSpan) compatibleSpan.textContent = compatibleCount;
        if (testedSpan) testedSpan.textContent = '1';
    }
}

// Smart Recommendations
function getSmartRecommendations() {
    const taskType = document.getElementById('rec-task')?.value;
    const hardware = document.getElementById('rec-hardware')?.value;
    const resultsDiv = document.getElementById('smart-recommendations');
    
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = '<div class="spinner"></div>Generating recommendations...';
    
    // Simulate API call
    setTimeout(() => {
        const mockRecommendations = generateMockRecommendations(taskType, hardware);
        resultsDiv.innerHTML = mockRecommendations;
    }, 1500);
}

function generateMockRecommendations(taskType, hardware) {
    const recommendations = {
        'text-generation': {
            cpu: [
                { name: 'DistilGPT-2', reason: 'Optimized for CPU inference, good performance/size ratio' },
                { name: 'GPT-2 Small', reason: 'Lightweight model suitable for CPU deployment' }
            ],
            cuda: [
                { name: 'GPT-2 Medium', reason: 'Good balance of quality and speed on GPU' },
                { name: 'GPT-Neo 1.3B', reason: 'Excellent GPU acceleration support' }
            ]
        },
        'text-classification': {
            cpu: [
                { name: 'DistilBERT', reason: 'Fast inference with minimal accuracy loss' },
                { name: 'MobileBERT', reason: 'Designed for mobile and edge deployment' }
            ],
            cuda: [
                { name: 'BERT Base', reason: 'Standard model with good GPU optimization' },
                { name: 'RoBERTa Base', reason: 'Improved training, excellent GPU performance' }
            ]
        }
    };
    
    const taskRecs = recommendations[taskType] || {};
    const hwRecs = taskRecs[hardware] || [{ name: 'General Purpose Model', reason: 'Suitable for most tasks' }];
    
    let html = '';
    hwRecs.forEach((rec, index) => {
        html += `
            <div class="recommendation-item">
                <div>
                    <strong>${index + 1}. ${rec.name}</strong>
                    <button class="btn btn-sm btn-primary" onclick="testModelFromHF('${rec.name.toLowerCase().replace(/\s+/g, '-')}')">🔧 Test</button>
                </div>
                <div>${rec.reason}</div>
            </div>
        `;
    });
    
    return html;
}

// Coverage Analysis Functions
function refreshCoverageMatrix() {
    const matrixBody = document.getElementById('coverage-matrix-body');
    if (!matrixBody) return;
    
    // Generate mock coverage data
    const models = ['bert-base', 'gpt2-medium', 'distilbert', 'roberta-base', 'xlnet-base'];
    const platforms = ['CPU', 'CUDA', 'ROCm', 'OpenVINO', 'MPS', 'WebGPU', 'DirectML', 'ONNX'];
    
    let html = '';
    models.forEach(model => {
        html += '<tr>';
        html += `<td><strong>${model}</strong></td>`;
        platforms.forEach(platform => {
            const status = Math.random() > 0.3 ? 'tested' : (Math.random() > 0.5 ? 'not-tested' : 'warning');
            const icon = status === 'tested' ? '✅' : (status === 'warning' ? '⚠️' : '❌');
            html += `<td><span class="status-${status}">${icon}</span></td>`;
        });
        html += '</tr>';
    });
    
    matrixBody.innerHTML = html;
}

function exportParquetData() {
    console.log('[Dashboard] Exporting parquet data'); showToast('Exporting parquet data', 'info');
}

function backupParquetData() {
    alert('Creating backup of parquet data...');
}

function analyzeTrends() {
    alert('Analyzing performance trends from historical data...');
}

function testMissingPlatforms() {
    alert('Starting automated testing for missing platform combinations...');
}

function autoFillCriticalGaps() {
    alert('Auto-filling critical coverage gaps with high-priority models...');
}

// Server Status Functions
function refreshServerStatus() {
    // Update server metrics with mock data
    const connections = document.getElementById('active-connections');
    const uptime = document.getElementById('uptime');
    const requests = document.getElementById('total-requests');
    
    if (connections) connections.textContent = Math.floor(Math.random() * 10) + 1;
    if (requests) requests.textContent = Math.floor(Math.random() * 1000) + 100;
    
    // Update uptime
    if (uptime) {
        const minutes = Math.floor(Math.random() * 60) + 1;
        uptime.textContent = `${minutes}m`;
    }
}

function refreshModels() {
    console.log('Refreshing model list...');
}

function loadModel() {
    const modelId = prompt('Enter model ID to load:');
    if (modelId) {
        alert(`Loading model: ${modelId}`);
    }
}

function viewModels() {
    showTab('model-manager');
}

function testInference() {
    showTab('ai-inference');
}

function refreshMetrics() {
    alert('Refreshing performance metrics...');
}

// Queue Functions
function refreshQueue() {
    console.log('Refreshing queue status...');
}

function clearQueue() {
    if (confirm('Are you sure you want to clear the queue?')) {
        alert('Queue cleared');
    }
}

function addWorker() {
    alert('Adding new worker to pool...');
}

function removeWorker() {
    if (confirm('Remove a worker from the pool?')) {
        alert('Worker removed');
    }
}

function exportQueueStats() {
    alert('Exporting queue statistics...');
}

// MCP Tools Functions
function refreshTools() {
    console.log('[Dashboard] Refreshing MCP tools...');
    const toolsGrid = document.querySelector('.tools-grid');
    
    if (!toolsGrid) {
        console.error('[Dashboard] Tools grid not found');
        return;
    }
    
    // Show loading state
    toolsGrid.innerHTML = '<div class="tool-tag" style="background: #e5e7eb; color: #6b7280;">Loading tools...</div>';
    
    fetch('/api/mcp/tools')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('[Dashboard] Tools loaded:', data);
            toolsGrid.innerHTML = '';
            
            // Handle different response formats
            const tools = data.tools || data.data || [];
            const total = data.total || tools.length;
            
            if (tools.length === 0) {
                toolsGrid.innerHTML = '<div class="tool-tag" style="background: #fef3c7; color: #92400e;">⚠️ No MCP tools found. Check server configuration.</div>';
                showToast('No MCP tools available', 'warning');
                return;
            }
            
            // Display each tool
            tools.forEach(tool => {
                const toolTag = document.createElement('div');
                toolTag.className = 'tool-tag';
                toolTag.textContent = tool.name || tool;
                
                // Add description as tooltip if available
                if (tool.description) {
                    toolTag.title = tool.description;
                }
                
                // Add status indicator if available
                if (tool.status === 'error' || tool.status === 'inactive') {
                    toolTag.style.background = '#fee2e2';
                    toolTag.style.color = '#991b1b';
                    toolTag.textContent += ' ⚠️';
                }
                
                toolsGrid.appendChild(toolTag);
            });
            
            console.log(`[Dashboard] Successfully loaded ${total} MCP tools`);
            showToast(`Loaded ${total} MCP tools`, 'success');
        })
        .catch(error => {
            console.error('[Dashboard] Error refreshing tools:', error);
            toolsGrid.innerHTML = `<div class="tool-tag" style="background: #fee2e2; color: #991b1b;">❌ Failed to load tools: ${error.message}</div>`;
            showToast(`Failed to load MCP tools: ${error.message}`, 'error', 5000);
        });
}

function testAPIs() {
    console.log('Testing API endpoints...');
    fetch('/api/mcp/test')
        .then(response => response.json())
        .then(data => {
            console.log('API test results:', data);
            const results = data.test_results || [];
            const operational = data.operational || 0;
            const total = data.total_tested || 0;
            
            let message = `API Test Results:\n\n`;
            message += `Total Tested: ${total}\n`;
            message += `Operational: ${operational}\n`;
            message += `Failed: ${total - operational}\n\n`;
            
            results.forEach(result => {
                const status = result.status === 'operational' ? '✓' : '✗';
                message += `${status} ${result.name}: ${result.status}\n`;
            });
            
            alert(message);
        })
        .catch(error => {
            console.error('Error testing APIs:', error);
            alert('Failed to test APIs: ' + error.message);
        });
}

function editConfig() {
    alert('Configuration editor:\n\nThis feature allows you to modify:\n- Max Queue Size\n- Request Timeout\n- Cache TTL\n- Log Level\n\nConfiguration editing will be implemented in a future update.');
}

// Logs Functions
function refreshLogs() {
    console.log('Refreshing logs...');
    const logOutput = document.getElementById('log-output');
    if (logOutput) {
        fetch('/api/mcp/logs')
            .then(response => response.json())
            .then(data => {
                console.log('Logs loaded:', data);
                if (data.logs) {
                    logOutput.innerHTML = '';
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.textContent = `${log.timestamp} - ${log.level} - ${log.message}`;
                        logOutput.appendChild(logEntry);
                    });
                    logOutput.scrollTop = logOutput.scrollHeight;
                }
            })
            .catch(error => {
                console.error('Error refreshing logs:', error);
                const errorEntry = document.createElement('div');
                errorEntry.className = 'log-entry';
                errorEntry.textContent = `${new Date().toISOString()} - ERROR - Failed to load logs: ${error.message}`;
                logOutput.appendChild(errorEntry);
            });
    }
}

function clearLogs() {
    const logOutput = document.getElementById('log-output');
    if (logOutput && confirm('Clear all logs?')) {
        logOutput.innerHTML = '';
        const clearEntry = document.createElement('div');
        clearEntry.className = 'log-entry';
        clearEntry.textContent = `${new Date().toISOString()} - INFO - Logs cleared by user`;
        logOutput.appendChild(clearEntry);
    }
}

function downloadLogs() {
    console.log('Downloading logs...');
    fetch('/api/mcp/logs')
        .then(response => response.json())
        .then(data => {
            if (data.logs) {
                const logText = data.logs.map(log => 
                    `${log.timestamp} - ${log.level} - ${log.message}`
                ).join('\n');
                
                const blob = new Blob([logText], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `mcp-logs-${new Date().toISOString().replace(/:/g, '-')}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                alert('Logs downloaded successfully!');
            }
        })
        .catch(error => {
            console.error('Error downloading logs:', error);
            alert('Failed to download logs: ' + error.message);
        });
}

// Workflow Management Functions
function refreshWorkflows() {
    console.log('Refreshing workflows...');
    fetch('/api/mcp/workflows')
        .then(response => response.json())
        .then(data => {
            console.log('Workflows loaded:', data);
            if (data.workflows) {
                displayWorkflows(data.workflows);
                updateWorkflowStats(data.workflows);
                showToast(`Loaded ${data.total} workflows`, 'success');
            }
        })
        .catch(error => {
            console.error('Error refreshing workflows:', error);
            showToast('Failed to load workflows: ' + error.message, 'error');
        });
}

function displayWorkflows(workflows) {
    const workflowsList = document.getElementById('workflows-list');
    if (!workflowsList) return;
    
    workflowsList.innerHTML = '';
    
    if (workflows.length === 0) {
        workflowsList.innerHTML = '<div class="workflow-item"><p>No workflows yet. Click "Create Workflow" to get started.</p></div>';
        return;
    }
    
    workflows.forEach(workflow => {
        const workflowItem = document.createElement('div');
        workflowItem.className = 'workflow-item';
        
        const statusClass = workflow.status === 'running' ? 'status-running' : 
                          workflow.status === 'pending' ? 'status-idle' :
                          workflow.status === 'paused' ? 'status-idle' :
                          workflow.status === 'completed' ? 'status-idle' : 'status-stopped';
        
        const percent = workflow.tasks > 0 ? Math.round((workflow.completed / workflow.tasks) * 100) : 0;
        
        workflowItem.innerHTML = `
            <div class="workflow-header">
                <h4>${workflow.name}</h4>
                <span class="workflow-status ${statusClass}">${workflow.status}</span>
            </div>
            <p class="workflow-description">${workflow.description}</p>
            <div class="workflow-progress">
                <div class="progress-info">
                    <span>Tasks: ${workflow.completed}/${workflow.tasks}</span>
                    <span>${percent}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percent}%"></div>
                </div>
            </div>
            <div class="workflow-actions">
                <button class="btn btn-sm btn-primary" onclick="viewWorkflow('${workflow.id}')">👁️ View</button>
                ${workflow.status === 'running' || workflow.status === 'pending' ? 
                    `<button class="btn btn-sm btn-warning" onclick="pauseWorkflow('${workflow.id}')">⏸️ Pause</button>` :
                    `<button class="btn btn-sm btn-success" onclick="startWorkflow('${workflow.id}')">▶️ Start</button>`
                }
                <button class="btn btn-sm btn-danger" onclick="stopWorkflow('${workflow.id}')">⏹️ Stop</button>
                <button class="btn btn-sm btn-danger" onclick="deleteWorkflow('${workflow.id}')">🗑️ Delete</button>
            </div>
        `;
        
        workflowsList.appendChild(workflowItem);
    });
}

function updateWorkflowStats(workflows) {
    const total = workflows.length;
    const running = workflows.filter(w => w.status === 'running').length;
    const completed = workflows.filter(w => w.status === 'completed').length;
    
    const totalEl = document.getElementById('total-workflows');
    const runningEl = document.getElementById('running-workflows');
    const completedEl = document.getElementById('completed-workflows');
    
    if (totalEl) totalEl.textContent = total;
    if (runningEl) runningEl.textContent = running;
    if (completedEl) completedEl.textContent = completed;
    
    // Update performance metrics (placeholder for now)
    const avgTimeEl = document.getElementById('avg-processing-time');
    const successRateEl = document.getElementById('success-rate');
    const throughputEl = document.getElementById('queue-throughput');
    const utilizationEl = document.getElementById('resource-utilization');
    
    if (avgTimeEl) avgTimeEl.textContent = '245ms';
    if (successRateEl) successRateEl.textContent = '98.5%';
    if (throughputEl) throughputEl.textContent = '150 req/min';
    if (utilizationEl) utilizationEl.textContent = '67%';
}

function createWorkflow() {
    // Ask user if they want to use a template
    const useTemplate = confirm('Would you like to create a workflow from a template?\n\nClick OK for templates, Cancel for custom workflow.');
    
    if (useTemplate) {
        createWorkflowFromTemplate();
    } else {
        openWorkflowEditor();
    }
}

// Open workflow editor modal
function openWorkflowEditor(workflowId = null) {
    const modal = document.getElementById('workflowEditorModal');
    const modeTitle = document.getElementById('editorModeTitle');
    const editingId = document.getElementById('editingWorkflowId');
    
    // Clear form
    document.getElementById('workflow-name').value = '';
    document.getElementById('workflow-description').value = '';
    document.getElementById('tasks-container').innerHTML = '';
    editingId.value = workflowId || '';
    
    if (workflowId) {
        modeTitle.textContent = 'Edit';
        // Load existing workflow
        loadWorkflowForEditing(workflowId);
    } else {
        modeTitle.textContent = 'Create';
    }
    
    updateTaskCount();
    modal.style.display = 'flex';
}

// Close workflow editor
function closeWorkflowEditor() {
    const modal = document.getElementById('workflowEditorModal');
    modal.style.display = 'none';
}

// Add task to editor
let taskCounter = 0;
function addTaskToEditor() {
    const container = document.getElementById('tasks-container');
    const taskId = 'task-' + (taskCounter++);
    
    const taskCard = document.createElement('div');
    taskCard.className = 'task-card';
    taskCard.id = taskId;
    taskCard.innerHTML = `
        <div class="task-card-header">
            <h4>Task ${taskCounter}</h4>
            <div class="task-card-actions">
                <button type="button" class="btn-danger-sm" onclick="removeTaskFromEditor('${taskId}')">Remove</button>
            </div>
        </div>
        <div class="form-group">
            <label>Task Name:</label>
            <input type="text" class="form-control task-name" placeholder="e.g., Generate Image" required>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Pipeline Type:</label>
                <select class="form-control task-type" required>
                    <option value="">Select type...</option>
                    <option value="text-generation">Text Generation</option>
                    <option value="text-to-image">Text to Image</option>
                    <option value="image-to-image">Image to Image</option>
                    <option value="image-to-video">Image to Video</option>
                    <option value="text-to-video">Text to Video</option>
                    <option value="text-to-speech">Text to Speech</option>
                    <option value="automatic-speech-recognition">Speech Recognition</option>
                    <option value="image-classification">Image Classification</option>
                    <option value="text-classification">Text Classification</option>
                    <option value="filter">Content Filter</option>
                    <option value="processing">Custom Processing</option>
                </select>
            </div>
            <div class="form-group">
                <label>Model:</label>
                <input type="text" class="form-control task-model" placeholder="e.g., stable-diffusion-xl">
            </div>
        </div>
        <div class="memory-section">
            <h5>⚠️ Memory Management (Prevent OOM)</h5>
            <div class="form-row-3">
                <label class="checkbox-label">
                    <input type="checkbox" class="task-vram-pinned">
                    Pin in VRAM
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" class="task-preemptable" checked>
                    Preemptable
                </label>
                <div class="form-group">
                    <label>Priority (1-10):</label>
                    <input type="number" class="form-control task-priority" value="5" min="1" max="10">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Max Memory (MB, 0=unlimited):</label>
                    <input type="number" class="form-control task-max-memory" value="0" min="0" max="32000">
                </div>
                <div class="form-group">
                    <label>Batch Size:</label>
                    <input type="number" class="form-control task-batch-size" value="1" min="1" max="128">
                </div>
            </div>
        </div>
        <div class="form-group">
            <label>Custom Config (JSON):</label>
            <textarea class="form-control task-config" rows="2" placeholder='{"param": "value"}'>{}</textarea>
        </div>
    `;
    
    container.appendChild(taskCard);
    updateTaskCount();
    
    // Initialize autocomplete for the model input field
    const modelInput = taskCard.querySelector('.task-model');
    if (modelInput) {
        initializeModelAutocomplete(modelInput);
    }
}

// Remove task from editor
function removeTaskFromEditor(taskId) {
    const task = document.getElementById(taskId);
    if (task && confirm('Remove this task?')) {
        task.remove();
        updateTaskCount();
    }
}

// Update task count
function updateTaskCount() {
    const count = document.querySelectorAll('.task-card').length;
    document.getElementById('task-count').textContent = `(${count})`;
}

// Save workflow
function saveWorkflow() {
    const name = document.getElementById('workflow-name').value.trim();
    const description = document.getElementById('workflow-description').value.trim();
    const workflowId = document.getElementById('editingWorkflowId').value;
    
    if (!name) {
        showToast('Please enter a workflow name', 'error');
        return;
    }
    
    // Collect tasks
    const taskCards = document.querySelectorAll('.task-card');
    if (taskCards.length === 0) {
        showToast('Please add at least one task', 'error');
        return;
    }
    
    const tasks = [];
    let isValid = true;
    
    taskCards.forEach((card, index) => {
        const taskName = card.querySelector('.task-name').value.trim();
        const taskType = card.querySelector('.task-type').value;
        const taskModel = card.querySelector('.task-model').value.trim();
        const vramPinned = card.querySelector('.task-vram-pinned').checked;
        const preemptable = card.querySelector('.task-preemptable').checked;
        const priority = parseInt(card.querySelector('.task-priority').value);
        const maxMemory = parseInt(card.querySelector('.task-max-memory').value);
        const batchSize = parseInt(card.querySelector('.task-batch-size').value);
        const configText = card.querySelector('.task-config').value.trim();
        
        if (!taskName || !taskType) {
            showToast(`Task ${index + 1}: Name and type are required`, 'error');
            isValid = false;
            return;
        }
        
        let config = {};
        try {
            if (configText) {
                config = JSON.parse(configText);
            }
        } catch (e) {
            showToast(`Task ${index + 1}: Invalid JSON in config`, 'error');
            isValid = false;
            return;
        }
        
        if (taskModel) {
            config.model = taskModel;
        }
        
        tasks.push({
            name: taskName,
            type: taskType,
            config: config,
            dependencies: [],
            input_mapping: {},
            output_keys: [],
            vram_pinned: vramPinned,
            preemptable: preemptable,
            priority: priority,
            max_memory_mb: maxMemory,
            batch_size: batchSize
        });
    });
    
    if (!isValid) return;
    
    const workflowData = {
        name: name,
        description: description,
        tasks: tasks
    };
    
    const url = workflowId ? `/api/mcp/workflows/${workflowId}` : '/api/mcp/workflows/create';
    const method = workflowId ? 'PUT' : 'POST';
    
    showToast(workflowId ? 'Updating workflow...' : 'Creating workflow...', 'info');
    
    fetch(url, {
        method: method,
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(workflowData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.workflow_id) {
            showToast(workflowId ? 'Workflow updated!' : 'Workflow created!', 'success');
            closeWorkflowEditor();
            refreshWorkflows();
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    })
    .catch(error => {
        showToast('Failed to save workflow: ' + error.message, 'error');
    });
}

// Load workflow for editing
function loadWorkflowForEditing(workflowId) {
    fetch(`/api/mcp/workflows/${workflowId}`)
        .then(response => response.json())
        .then(data => {
            if (data.workflow) {
                const wf = data.workflow;
                document.getElementById('workflow-name').value = wf.name;
                document.getElementById('workflow-description').value = wf.description || '';
                
                // Load tasks
                if (wf.tasks && wf.tasks.length > 0) {
                    wf.tasks.forEach(task => {
                        addTaskToEditor();
                        const cards = document.querySelectorAll('.task-card');
                        const card = cards[cards.length - 1];
                        
                        card.querySelector('.task-name').value = task.name;
                        card.querySelector('.task-type').value = task.type;
                        card.querySelector('.task-model').value = task.config.model || '';
                        card.querySelector('.task-vram-pinned').checked = task.vram_pinned || false;
                        card.querySelector('.task-preemptable').checked = task.preemptable !== false;
                        card.querySelector('.task-priority').value = task.priority || 5;
                        card.querySelector('.task-max-memory').value = task.max_memory_mb || 0;
                        card.querySelector('.task-batch-size').value = task.batch_size || 1;
                        card.querySelector('.task-config').value = JSON.stringify(task.config, null, 2);
                    });
                }
            }
        })
        .catch(error => {
            showToast('Failed to load workflow: ' + error.message, 'error');
        });
}

function createWorkflowFromTemplate() {
    const templates = `Available Templates:

1. Image Generation Pipeline
   - LLM prompt enhancement → image generation → upscaling
   
2. Text-to-Video Pipeline
   - Enhanced prompt → image → animated video
   
3. Safe Image Generation
   - NSFW filter → image generation → quality validation
   
4. Multimodal Content Pipeline
   - Text → Image → Audio → Video generation

Enter template number (1-4):`;
    
    const choice = prompt(templates);
    if (!choice) return;
    
    const templateMap = {
        '1': 'image_generation',
        '2': 'video_generation',
        '3': 'safe_image',
        '4': 'multimodal'
    };
    
    const templateName = templateMap[choice];
    if (!templateName) {
        showToast('Invalid template selection', 'error');
        return;
    }
    
    const customName = prompt('Enter a custom name for this workflow (optional):');
    
    showToast('Creating workflow from template...', 'info');
    
    const requestData = {
        template_name: templateName
    };
    
    if (customName) {
        requestData.custom_config = { name: customName };
    }
    
    fetch('/api/mcp/workflows/create_from_template', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(`Workflow created from template: ${data.template_used}`, 'success');
            refreshWorkflows();
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    })
    .catch(error => {
        showToast('Failed to create workflow: ' + error.message, 'error');
    });
}

function createCustomWorkflow() {
    const name = prompt('Enter workflow name:');
    if (!name) return;
    
    const description = prompt('Enter workflow description (optional):') || '';
    
    // Create a simple default workflow with one text model task
    const workflow = {
        name: name,
        description: description,
        tasks: [
            {
                name: 'Text Processing',
                type: 'text_model',
                config: {
                    model: 'gpt2',
                    inputs: {prompt: 'Hello, world!'}
                },
                dependencies: [],
                input_mapping: {},
                output_keys: ['text']
            }
        ]
    };
    
    showToast('Creating workflow...', 'info');
    
    fetch('/api/mcp/workflows/create', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(workflow)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast('Workflow created successfully!', 'success');
            refreshWorkflows();
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    })
    .catch(error => {
        showToast('Failed to create workflow: ' + error.message, 'error');
    });
}

function viewWorkflow(id) {
    // Open workflow editor for editing
    openWorkflowEditor(id);
}

function startWorkflow(id) {
    if (!confirm('Start this workflow?')) return;
    
    showToast('Starting workflow...', 'info');
    
    fetch(`/api/mcp/workflows/${id}/start`, {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Workflow started!', 'success');
                refreshWorkflows();
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            showToast('Failed to start workflow: ' + error.message, 'error');
        });
}

function pauseWorkflow(id) {
    if (!confirm('Pause this workflow?')) return;
    
    showToast('Pausing workflow...', 'info');
    
    fetch(`/api/mcp/workflows/${id}/pause`, {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Workflow paused!', 'success');
                refreshWorkflows();
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            showToast('Failed to pause workflow: ' + error.message, 'error');
        });
}

function stopWorkflow(id) {
    if (!confirm('Stop this workflow? This action cannot be undone.')) return;
    
    showToast('Stopping workflow...', 'warning');
    
    fetch(`/api/mcp/workflows/${id}/stop`, {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Workflow stopped!', 'success');
                refreshWorkflows();
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            showToast('Failed to stop workflow: ' + error.message, 'error');
        });
}

function deleteWorkflow(id) {
    if (!confirm('Delete this workflow? This action cannot be undone.')) return;
    
    showToast('Deleting workflow...', 'warning');
    
    fetch(`/api/mcp/workflows/${id}`, {method: 'DELETE'})
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Workflow deleted!', 'success');
                refreshWorkflows();
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            showToast('Failed to delete workflow: ' + error.message, 'error');
        });
}

function optimizePerformance() {
    showToast('Running performance optimization...', 'info');
    // Simulate optimization process
    setTimeout(() => {
        showToast('Performance optimization completed!\n- Reduced latency by 15%\n- Improved throughput by 10%', 'success', 5000);
    }, 2000);
}

// Model autocomplete functionality
let autocompleteTimeout = null;
let currentAutocompleteInput = null;

function initializeModelAutocomplete(inputElement) {
    // Create autocomplete container wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'autocomplete-container';
    inputElement.parentNode.insertBefore(wrapper, inputElement);
    wrapper.appendChild(inputElement);
    
    // Create autocomplete dropdown
    const dropdown = document.createElement('div');
    dropdown.className = 'autocomplete-items';
    dropdown.style.display = 'none';
    wrapper.appendChild(dropdown);
    
    let currentFocus = -1;
    
    // Handle input events
    inputElement.addEventListener('input', function(e) {
        const query = this.value.trim();
        
        // Clear existing timeout
        if (autocompleteTimeout) {
            clearTimeout(autocompleteTimeout);
        }
        
        // Hide dropdown if query is too short
        if (query.length < 2) {
            dropdown.style.display = 'none';
            return;
        }
        
        // Debounce autocomplete requests
        autocompleteTimeout = setTimeout(() => {
            fetchModelSuggestions(query, dropdown, inputElement);
        }, 300);
    });
    
    // Handle keyboard navigation
    inputElement.addEventListener('keydown', function(e) {
        const items = dropdown.getElementsByTagName('div');
        
        if (e.keyCode === 40) { // Arrow Down
            e.preventDefault();
            currentFocus++;
            addActive(items);
        } else if (e.keyCode === 38) { // Arrow Up
            e.preventDefault();
            currentFocus--;
            addActive(items);
        } else if (e.keyCode === 13) { // Enter
            e.preventDefault();
            if (currentFocus > -1 && items[currentFocus]) {
                items[currentFocus].click();
            }
        } else if (e.keyCode === 27) { // Escape
            dropdown.style.display = 'none';
            currentFocus = -1;
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (e.target !== inputElement) {
            dropdown.style.display = 'none';
            currentFocus = -1;
        }
    });
    
    function addActive(items) {
        if (!items) return false;
        removeActive(items);
        if (currentFocus >= items.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = items.length - 1;
        if (items[currentFocus]) {
            items[currentFocus].classList.add('autocomplete-active');
        }
    }
    
    function removeActive(items) {
        for (let i = 0; i < items.length; i++) {
            items[i].classList.remove('autocomplete-active');
        }
    }
}

async function fetchModelSuggestions(query, dropdown, inputElement) {
    try {
        const response = await fetch(`/api/mcp/models/autocomplete?q=${encodeURIComponent(query)}&limit=10`);
        const data = await response.json();
        
        // Clear existing suggestions
        dropdown.innerHTML = '';
        
        if (!data.suggestions || data.suggestions.length === 0) {
            dropdown.style.display = 'none';
            return;
        }
        
        // Add suggestions to dropdown
        data.suggestions.forEach(suggestion => {
            const item = document.createElement('div');
            item.innerHTML = `
                <span class="autocomplete-model-id">${suggestion.id}</span>
                <span class="autocomplete-pipeline-tag">${suggestion.pipeline_tag}</span>
                ${suggestion.downloads ? `<span class="autocomplete-downloads">${formatNumber(suggestion.downloads)} downloads</span>` : ''}
            `;
            
            // Handle click
            item.addEventListener('click', function() {
                inputElement.value = suggestion.id;
                dropdown.style.display = 'none';
                
                // Trigger change event
                inputElement.dispatchEvent(new Event('change'));
            });
            
            dropdown.appendChild(item);
        });
        
        dropdown.style.display = 'block';
        
    } catch (error) {
        console.error('Autocomplete error:', error);
        dropdown.style.display = 'none';
    }
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Auto-refresh functionality
function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        if (currentTab === 'overview') {
            refreshServerStatus();
        } else if (currentTab === 'system-logs') {
            const autoRefreshCheckbox = document.getElementById('auto-refresh');
            if (autoRefreshCheckbox && autoRefreshCheckbox.checked) {
                refreshLogs();
            }
        } else if (currentTab === 'workflow-management') {
            refreshWorkflows();
        }
    }, 5000);
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Initialize overview tab
    showTab('overview');
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Initialize inference form
    setTimeout(() => {
        updateInferenceForm();
    }, 100);
    
    // Load database statistics with delay to ensure server is ready
    setTimeout(() => {
        loadDatabaseStats();
    }, 1000);
    
    // Initialize autocomplete on test-model-id input
    const testModelInput = document.getElementById('test-model-id');
    if (testModelInput) {
        initializeModelAutocomplete(testModelInput);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
});
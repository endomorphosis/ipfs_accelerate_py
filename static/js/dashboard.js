// IPFS Accelerate MCP Server Dashboard JavaScript

// Global state
let currentTab = 'overview';
let searchResults = [];
let compatibilityResults = [];
let autoRefreshInterval = null;

// Tab Management
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        if (content) {
            content.classList.remove('active');
        }
    });
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
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
        case 'queue-monitor':
            refreshQueue();
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
    
    // Make API call to search models
    fetch(`/api/models/search?query=${encodeURIComponent(query)}&task=${taskFilter}&size=${sizeFilter}`)
        .then(response => response.json())
        .then(data => {
            displayHFResults(data);
            updateSearchStats(data);
        })
        .catch(error => {
            console.error('Search error:', error);
            resultsDiv.innerHTML = '<p>Search failed. Please try again.</p>';
        });
}

function displayHFResults(data) {
    const resultsDiv = document.getElementById('hf-search-results');
    if (!resultsDiv || !data.models || data.models.length === 0) {
        if (resultsDiv) {
            resultsDiv.innerHTML = '<p>No models found. Try a different search term.</p>';
        }
        return;
    }
    
    let html = '';
    data.models.forEach(model => {
        html += `
            <div class="model-result">
                <div class="model-header">
                    <div class="model-title">${model.title || model.id}</div>
                    <div class="model-actions">
                        <button class="btn btn-sm btn-warning" onclick="testModelFromHF('${model.id}')">🔧 Test</button>
                        <button class="btn btn-sm btn-success" onclick="downloadModel('${model.id}')">⬇️ Download</button>
                    </div>
                </div>
                <div class="model-meta">
                    <span>📊 ${model.downloads || 0} downloads</span>
                    <span>📏 ${model.size || 'Unknown'} size</span>
                    <span>🏷️ ${model.task || 'General'}</span>
                </div>
                <div class="model-description">${model.description || 'No description available'}</div>
                <div class="model-tags">
                    ${(model.tags || []).map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
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

function downloadModel(modelId) {
    alert(`Starting download for model: ${modelId}`);
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
    
    if (data.models && data.models.length > 0) {
        if (totalIndexedSpan) totalIndexedSpan.textContent = data.models.length;
        if (hfModelsSpan) hfModelsSpan.textContent = data.models.length;
    }
}

// Hardware Compatibility Testing
function testModelCompatibility() {
    const modelId = document.getElementById('test-model-id')?.value;
    const resultsDiv = document.getElementById('compatibility-results');
    
    if (!modelId) {
        alert('Please enter a model ID to test');
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
        alert('Please select at least one hardware platform to test');
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
    alert('Exporting parquet data to benchmark_results_2024-01-15.parquet');
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
    alert('Refreshing model list...');
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
    alert('Refreshing queue status...');
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
    alert('Refreshing MCP tools list...');
}

function testAPIs() {
    alert('Testing API endpoints...');
}

function editConfig() {
    alert('Opening configuration editor...');
}

// Logs Functions
function refreshLogs() {
    const logOutput = document.getElementById('log-output');
    if (logOutput) {
        const newEntry = `<div class="log-entry">${new Date().toISOString()} - INFO - Dashboard refreshed</div>`;
        logOutput.innerHTML += newEntry;
        logOutput.scrollTop = logOutput.scrollHeight;
    }
}

function clearLogs() {
    const logOutput = document.getElementById('log-output');
    if (logOutput && confirm('Clear all logs?')) {
        logOutput.innerHTML = '';
    }
}

function downloadLogs() {
    alert('Downloading system logs...');
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
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
});
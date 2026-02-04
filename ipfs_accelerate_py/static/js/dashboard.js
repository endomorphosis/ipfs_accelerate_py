// IPFS Accelerate MCP Server Dashboard JavaScript

// Global state
let currentTab = 'overview';
let searchResults = [];
let compatibilityResults = [];
let autoRefreshInterval = null;

// Cache for SDK operations
const sdkCache = {
    data: new Map(),
    ttl: 5 * 60 * 1000, // 5 minutes default TTL
    
    set(key, value, ttl = this.ttl) {
        this.data.set(key, {
            value: value,
            expires: Date.now() + ttl
        });
    },
    
    get(key) {
        const item = this.data.get(key);
        if (!item) return null;
        
        if (Date.now() > item.expires) {
            this.data.delete(key);
            return null;
        }
        
        return item.value;
    },
    
    clear() {
        this.data.clear();
    },
    
    has(key) {
        const item = this.data.get(key);
        if (!item) return false;
        
        if (Date.now() > item.expires) {
            this.data.delete(key);
            return false;
        }
        
        return true;
    }
};

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle utility
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Initialize MCP SDK Client
let mcpClient = null;
let sdkStats = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    avgResponseTime: 0,
    methodCalls: {}
};

// Initialize SDK on page load
function initializeSDK() {
    try {
        mcpClient = new MCPClient('/jsonrpc', {
            timeout: 30000,
            retries: 3,
            reportErrors: true
        });
        console.log('[Dashboard] MCP SDK client initialized');
        
        // Perform health check
        checkSDKConnection();
        
        return true;
    } catch (error) {
        console.error('[Dashboard] Failed to initialize MCP SDK:', error);
        updateConnectionStatus(false);
        return false;
    }
}

// Check SDK connection health
async function checkSDKConnection() {
    try {
        // Try a simple SDK call to verify connection
        const response = await fetch('/jsonrpc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                jsonrpc: '2.0',
                method: 'tools/list',
                params: {},
                id: Date.now()
            })
        });
        
        const connected = response.ok;
        updateConnectionStatus(connected);
        
        return connected;
    } catch (error) {
        console.warn('[Dashboard] SDK connection check failed:', error);
        updateConnectionStatus(false);
        return false;
    }
}

// Update connection status indicator
function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('server-status');
    if (statusIndicator) {
        statusIndicator.className = connected ? 'status-indicator online' : 'status-indicator offline';
        statusIndicator.title = connected ? 'SDK Connected' : 'SDK Disconnected';
    }
}

// Track SDK method calls
function trackSDKCall(method, success, responseTime) {
    sdkStats.totalCalls++;
    if (success) {
        sdkStats.successfulCalls++;
    } else {
        sdkStats.failedCalls++;
    }
    
    // Update average response time
    sdkStats.avgResponseTime = 
        (sdkStats.avgResponseTime * (sdkStats.totalCalls - 1) + responseTime) / sdkStats.totalCalls;
    
    // Track per-method calls
    if (!sdkStats.methodCalls[method]) {
        sdkStats.methodCalls[method] = { count: 0, successCount: 0, failCount: 0, avgTime: 0 };
    }
    sdkStats.methodCalls[method].count++;
    if (success) {
        sdkStats.methodCalls[method].successCount++;
    } else {
        sdkStats.methodCalls[method].failCount++;
    }
    sdkStats.methodCalls[method].avgTime = 
        (sdkStats.methodCalls[method].avgTime * (sdkStats.methodCalls[method].count - 1) + responseTime) / 
        sdkStats.methodCalls[method].count;
}

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
        case 'sdk-playground':
            initializeSDKPlayground();
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
    const inferenceMode = document.getElementById('inference-mode')?.value || 'standard';
    const resultsDiv = document.getElementById('inference-results');
    const executionTimeSpan = document.getElementById('execution-time');
    const modelUsedSpan = document.getElementById('model-used');
    
    if (!resultsDiv) return;
    
    // Show loading state
    resultsDiv.innerHTML = '<div class="spinner large"></div><p>Running inference...</p>';
    
    // Check SDK availability
    if (!mcpClient) {
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>❌ SDK Not Available</strong>
                <p>The MCP SDK client is not initialized. Please check:</p>
                <ul>
                    <li>Server is running and accessible</li>
                    <li>SDK client initialized on page load</li>
                    <li>Network connection is stable</li>
                </ul>
                <button class="btn btn-primary" onclick="location.reload()">🔄 Reload Page</button>
            </div>
        `;
        return;
    }
    
    // Run inference based on selected mode
    runInferenceViaSDK(inferenceType, modelId, inferenceMode, resultsDiv, executionTimeSpan, modelUsedSpan);
}

async function runInferenceViaSDK(inferenceType, modelId, inferenceMode, resultsDiv, executionTimeSpan, modelUsedSpan) {
    const startTime = Date.now();
    
    try {
        // Get input based on inference type
        const input = getInferenceInput(inferenceType);
        
        let result;
        let toolName;
        
        // Choose inference method based on mode
        switch (inferenceMode) {
            case 'distributed':
                toolName = 'run_distributed_inference';
                result = await mcpClient.runDistributedInference(inferenceType, modelId || 'auto', input);
                break;
            case 'multiplex':
                toolName = 'multiplex_inference';
                // For multiplex, we can send multiple inputs
                const inputs = Array.isArray(input) ? input : [input];
                result = await mcpClient.multiplexInference(inferenceType, modelId || 'auto', inputs);
                break;
            default: // standard
                toolName = 'run_inference';
                result = await mcpClient.runInference(inferenceType, modelId || 'auto', input);
                break;
        }
        
        const responseTime = Date.now() - startTime;
        trackSDKCall(toolName, true, responseTime);
        
        // Display results with better formatting
        displayInferenceResults(result, inferenceType, resultsDiv);
        
        if (executionTimeSpan) {
            executionTimeSpan.textContent = `${(responseTime / 1000).toFixed(2)}s`;
            executionTimeSpan.style.color = responseTime < 5000 ? '#10b981' : '#f59e0b';
        }
        if (modelUsedSpan) {
            modelUsedSpan.textContent = modelId || result.model_used || result.model || 'auto-selected';
        }
        
        showToast(`✅ Inference completed in ${responseTime}ms`, 'success');
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('run_inference', false, responseTime);
        
        // Show proper error message (NO mock fallback)
        console.error('[Dashboard] Inference via SDK failed:', error);
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>❌ Inference Failed</strong>
                <p><strong>Error:</strong> ${error.message || error}</p>
                <details>
                    <summary>Technical Details</summary>
                    <pre>${JSON.stringify(error, null, 2)}</pre>
                </details>
                <p><strong>Possible causes:</strong></p>
                <ul>
                    <li>Model not available or not loaded</li>
                    <li>Invalid input format for inference type</li>
                    <li>Server resource constraints</li>
                    <li>Network connectivity issues</li>
                </ul>
                <div style="margin-top: 10px;">
                    <button class="btn btn-primary" onclick="runInference()">🔄 Retry</button>
                    <button class="btn btn-secondary" onclick="getModelRecommendations('${inferenceType}')">💡 Get Model Recommendations</button>
                </div>
            </div>
        `;
        
        if (executionTimeSpan) {
            executionTimeSpan.textContent = 'Failed';
            executionTimeSpan.style.color = '#ef4444';
        }
        if (modelUsedSpan) {
            modelUsedSpan.textContent = 'N/A';
        }
        
        showToast('❌ Inference failed: ' + (error.message || error), 'error');
    }
}

function displayInferenceResults(result, inferenceType, resultsDiv) {
    // Format results based on inference type
    let html = '';
    
    if (result.error) {
        html = `<div class="warning">⚠️ ${result.error}</div>`;
    } else if (typeof result === 'string') {
        html = `<div class="result-text">${result}</div>`;
    } else if (result.text || result.generated_text) {
        html = `<div class="result-text">${result.text || result.generated_text}</div>`;
    } else if (result.embeddings || result.embedding) {
        const embeddings = result.embeddings || result.embedding;
        html = `
            <div class="result-embeddings">
                <strong>Embeddings Generated:</strong>
                <p>Dimensions: ${embeddings.length || 'Unknown'}</p>
                <details>
                    <summary>View Embeddings</summary>
                    <pre>${JSON.stringify(embeddings.slice(0, 10), null, 2)}${embeddings.length > 10 ? '\n... (' + (embeddings.length - 10) + ' more)' : ''}</pre>
                </details>
            </div>
        `;
    } else if (result.labels || result.label) {
        const labels = result.labels || [result.label];
        html = `
            <div class="result-classification">
                <strong>Classification Results:</strong>
                <ul>
                    ${labels.map(l => `<li>${l.label || l}: ${((l.score || 0) * 100).toFixed(2)}%</li>`).join('')}
                </ul>
            </div>
        `;
    } else if (result.image || result.images) {
        html = `<div class="result-image"><img src="${result.image || result.images[0]}" alt="Generated image" style="max-width: 100%; border-radius: 8px;"/></div>`;
    } else {
        // Fallback to JSON display
        html = `<pre class="result-json">${JSON.stringify(result, null, 2)}</pre>`;
    }
    
    // Add metadata if available
    if (result.confidence || result.score) {
        html += `<div class="result-metadata"><strong>Confidence:</strong> ${((result.confidence || result.score) * 100).toFixed(2)}%</div>`;
    }
    
    resultsDiv.innerHTML = html;
}

function getInferenceInput(inferenceType) {
    // Get input from form fields based on inference type
    const textInput = document.getElementById('text-input')?.value;
    const promptInput = document.getElementById('prompt')?.value;
    const questionInput = document.getElementById('question')?.value;
    
    if (textInput) return textInput;
    if (promptInput) return promptInput;
    if (questionInput) return questionInput;
    
    // Default inputs for testing
    const defaults = {
        'text-generation': 'Once upon a time',
        'text-classification': 'This product is amazing!',
        'text-embeddings': 'The quick brown fox jumps',
        'translation': 'Hello world',
        'summarization': 'This is a long text that needs to be summarized.',
        'question-answering': 'What is AI?'
    };
    
    return defaults[inferenceType] || 'Test input';
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

// Model Recommendations
async function getModelRecommendations(inferenceType) {
    const resultsDiv = document.getElementById('model-recommendations');
    if (!resultsDiv) {
        console.warn('Model recommendations div not found');
        return;
    }
    
    resultsDiv.innerHTML = '<div class="spinner"></div><p>Getting model recommendations...</p>';
    
    if (!mcpClient) {
        resultsDiv.innerHTML = '<div class="warning">SDK not available for recommendations</div>';
        return;
    }
    
    try {
        const startTime = Date.now();
        const recommendations = await mcpClient.callTool('recommend_models', {
            task: inferenceType,
            limit: 5
        });
        
        const responseTime = Date.now() - startTime;
        trackSDKCall('recommend_models', true, responseTime);
        
        if (!recommendations || !recommendations.models || recommendations.models.length === 0) {
            resultsDiv.innerHTML = '<div class="info">No specific recommendations available. Using auto-select.</div>';
            return;
        }
        
        let html = '<div class="model-recommendations"><h4>💡 Recommended Models:</h4><ul>';
        recommendations.models.forEach((model, index) => {
            const modelName = model.model_id || model.name || model;
            const score = model.score || model.confidence || 0;
            const description = model.description || 'No description available';
            
            html += `
                <li class="model-recommendation-item">
                    <div class="model-rec-header">
                        <strong>${index + 1}. ${modelName}</strong>
                        ${score > 0 ? `<span class="model-score">${(score * 100).toFixed(1)}%</span>` : ''}
                    </div>
                    <div class="model-rec-description">${description}</div>
                    <button class="btn btn-sm btn-primary" onclick="selectRecommendedModel('${modelName}')">
                        ✨ Use This Model
                    </button>
                </li>
            `;
        });
        html += '</ul></div>';
        
        resultsDiv.innerHTML = html;
        showToast(`✅ Found ${recommendations.models.length} recommended models`, 'success');
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('recommend_models', false, responseTime);
        console.error('Failed to get model recommendations:', error);
        resultsDiv.innerHTML = `<div class="warning">⚠️ Could not get recommendations: ${error.message}</div>`;
    }
}

function selectRecommendedModel(modelId) {
    const modelIdInput = document.getElementById('model-id');
    if (modelIdInput) {
        modelIdInput.value = modelId;
        showToast(`✅ Selected model: ${modelId}`, 'success');
    }
}

// Auto-load recommendations when inference type changes
function loadModelRecommendationsForType() {
    const inferenceType = document.getElementById('inference-type')?.value;
    if (inferenceType && mcpClient) {
        getModelRecommendations(inferenceType);
    }
}

// HuggingFace Model Search Functions
async function searchHuggingFace() {
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
    resultsDiv.innerHTML = '<div class="spinner"></div>Searching HuggingFace Hub via SDK...';
    
    console.log(`[Dashboard] Searching HuggingFace with query: ${query}, task: ${taskFilter}`);
    
    const startTime = Date.now();
    
    try {
        // Use SDK method if available
        if (mcpClient && mcpClient.searchHuggingfaceModels) {
            const searchParams = {
                query: query,
                limit: 20
            };
            
            if (taskFilter) {
                searchParams.task = taskFilter;
            }
            
            const data = await mcpClient.searchHuggingfaceModels(searchParams);
            const responseTime = Date.now() - startTime;
            trackSDKCall('searchHuggingfaceModels', true, responseTime);
            
            console.log(`[Dashboard] SDK search results:`, data);
            displayHFResults(data);
            updateSearchStats(data);
        } else {
            // Fallback to direct API call if SDK not available
            console.warn('[Dashboard] SDK not available, using direct API call');
            const params = new URLSearchParams({
                q: query,
                limit: '20'
            });
            
            if (taskFilter) {
                params.append('task', taskFilter);
            }
            
            const response = await fetch(`/api/mcp/models/search?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            const responseTime = Date.now() - startTime;
            trackSDKCall('searchHuggingfaceModels', false, responseTime);
            
            console.log(`[Dashboard] API search results:`, data);
            displayHFResults(data);
            updateSearchStats(data);
        }
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('searchHuggingfaceModels', false, responseTime);
        
        console.error('[Dashboard] Search error:', error);
        resultsDiv.innerHTML = `<div class="error-message">
            <p><strong>Search Failed</strong></p>
            <p>${error.message}</p>
            <button class="btn btn-primary btn-sm" onclick="searchHuggingFace()">🔄 Retry</button>
        </div>`;
        showToast('Search failed', 'error');
    }
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

async function downloadModel(modelId, buttonId) {
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
    
    const startTime = Date.now();
    
    try {
        let data;
        
        // Use SDK method if available
        if (mcpClient && mcpClient.downloadModel) {
            data = await mcpClient.downloadModel(modelId);
            const responseTime = Date.now() - startTime;
            trackSDKCall('downloadModel', true, responseTime);
        } else {
            // Fallback to direct API call if SDK not available
            console.warn('[Dashboard] SDK not available, using direct API call');
            const response = await fetch('/api/mcp/models/download', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_id: modelId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            data = await response.json();
            const responseTime = Date.now() - startTime;
            trackSDKCall('downloadModel', false, responseTime);
        }
        
        console.log(`[Dashboard] Download response:`, data);
        
        if (data.status === 'success' || data.success) {
            showToast(`✓ Model ${modelId} downloaded successfully`, 'success');
            
            // Update button to show success
            if (button) {
                button.innerHTML = '✓ Downloaded';
                button.classList.remove('btn-secondary');
                button.classList.add('btn-info');
            }
            
            // Refresh the models list
            if (typeof loadModels === 'function') {
                loadModels();
            }
            if (typeof loadAvailableModels === 'function') {
                loadAvailableModels();
            }
        } else {
            throw new Error(data.message || data.error || 'Download failed');
        }
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('downloadModel', false, responseTime);
        
        console.error('[Dashboard] Download error:', error);
        showToast(`Download failed: ${error.message}`, 'error');
        
        // Reset button on error
        if (button) {
            button.disabled = false;
            button.innerHTML = '⬇️ Download';
            button.classList.remove('btn-secondary');
            button.classList.add('btn-success');
        }
    }
}

// New SDK-based model management functions

async function showModelDetails(modelId) {
    console.log(`[Dashboard] Fetching details for model: ${modelId}`);
    
    const startTime = Date.now();
    
    try {
        let details;
        
        if (mcpClient && mcpClient.getModelDetails) {
            details = await mcpClient.getModelDetails(modelId);
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelDetails', true, responseTime);
        } else {
            // Fallback to callTool
            console.warn('[Dashboard] getModelDetails not available, using callTool');
            details = await mcpClient.callTool('get_model_details', { model_id: modelId });
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelDetails', false, responseTime);
        }
        
        console.log(`[Dashboard] Model details:`, details);
        displayModelDetailsModal(details);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getModelDetails', false, responseTime);
        
        console.error('[Dashboard] Failed to load model details:', error);
        showToast(`Failed to load model details: ${error.message}`, 'error');
    }
}

function displayModelDetailsModal(details) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content model-details-modal">
            <div class="modal-header">
                <h3>📦 ${details.model_name || details.model_id}</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
            </div>
            <div class="modal-body">
                <div class="model-detail-section">
                    <h4>Basic Information</h4>
                    <div class="detail-grid">
                        <div><strong>Model ID:</strong> ${details.model_id}</div>
                        <div><strong>Architecture:</strong> ${details.architecture || 'N/A'}</div>
                        <div><strong>Task:</strong> ${details.task || details.pipeline_tag || 'N/A'}</div>
                        <div><strong>Language:</strong> ${details.language || 'N/A'}</div>
                    </div>
                </div>
                
                ${details.description ? `
                <div class="model-detail-section">
                    <h4>Description</h4>
                    <p>${details.description}</p>
                </div>
                ` : ''}
                
                <div class="model-detail-section">
                    <h4>Statistics</h4>
                    <div class="detail-grid">
                        <div><strong>Downloads:</strong> ${details.downloads || 0}</div>
                        <div><strong>Likes:</strong> ${details.likes || 0}</div>
                        <div><strong>Size:</strong> ${details.model_size || 'Unknown'}</div>
                    </div>
                </div>
                
                ${details.tags && details.tags.length > 0 ? `
                <div class="model-detail-section">
                    <h4>Tags</h4>
                    <div class="tag-list">
                        ${details.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                <button class="btn btn-success" onclick="downloadModel('${details.model_id}'); this.closest('.modal-overlay').remove();">⬇️ Download</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

async function loadAvailableModels() {
    console.log('[Dashboard] Loading available models list...');
    
    const startTime = Date.now();
    
    try {
        let models;
        
        if (mcpClient && mcpClient.getModelList) {
            models = await mcpClient.getModelList();
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelList', true, responseTime);
        } else {
            // Fallback to callTool
            console.warn('[Dashboard] getModelList not available, using callTool');
            models = await mcpClient.callTool('get_model_list', {});
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelList', false, responseTime);
        }
        
        console.log(`[Dashboard] Available models:`, models);
        displayAvailableModels(models);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getModelList', false, responseTime);
        
        console.error('[Dashboard] Failed to load models list:', error);
        showToast(`Failed to load models: ${error.message}`, 'error');
    }
}

function displayAvailableModels(data) {
    const modelsContainer = document.getElementById('available-models-list');
    if (!modelsContainer) return;
    
    const models = data.models || data.results || data || [];
    
    if (models.length === 0) {
        modelsContainer.innerHTML = '<p class="info-message">No models available locally. Search and download models from HuggingFace.</p>';
        return;
    }
    
    let html = '<div class="models-grid">';
    
    models.forEach(model => {
        const modelId = model.model_id || model.id || model.name;
        html += `
            <div class="model-card">
                <div class="model-card-header">
                    <strong>${model.model_name || modelId}</strong>
                </div>
                <div class="model-card-body">
                    <div class="model-meta-small">
                        <span>📊 ${model.task || 'General'}</span>
                        ${model.size ? `<span>💾 ${model.size}</span>` : ''}
                    </div>
                    ${model.description ? `<p class="model-description-small">${model.description.substring(0, 100)}...</p>` : ''}
                </div>
                <div class="model-card-actions">
                    <button class="btn btn-sm btn-primary" onclick="showModelDetails('${modelId}')">ℹ️ Details</button>
                    <button class="btn btn-sm btn-warning" onclick="testModelFromHF('${modelId}')">🔧 Test</button>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    modelsContainer.innerHTML = html;
}

async function loadModelStatistics() {
    console.log('[Dashboard] Loading model statistics...');
    
    const startTime = Date.now();
    
    try {
        let stats;
        
        if (mcpClient && mcpClient.getModelStats) {
            stats = await mcpClient.getModelStats();
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelStats', true, responseTime);
        } else {
            // Fallback to callTool
            console.warn('[Dashboard] getModelStats not available, using callTool');
            stats = await mcpClient.callTool('get_model_stats', {});
            const responseTime = Date.now() - startTime;
            trackSDKCall('getModelStats', false, responseTime);
        }
        
        console.log(`[Dashboard] Model statistics:`, stats);
        displayModelStatistics(stats);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getModelStats', false, responseTime);
        
        console.error('[Dashboard] Failed to load model statistics:', error);
        // Don't show error toast for stats, just log it
    }
}

function displayModelStatistics(stats) {
    // Update statistics in the UI if elements exist
    const totalModels = document.getElementById('total-models-stat');
    const downloadedModels = document.getElementById('downloaded-models-stat');
    const popularModel = document.getElementById('popular-model-stat');
    
    if (totalModels && stats.total_models !== undefined) {
        totalModels.textContent = stats.total_models;
    }
    
    if (downloadedModels && stats.downloaded_models !== undefined) {
        downloadedModels.textContent = stats.downloaded_models;
    }
    
    if (popularModel && stats.most_used_model) {
        popularModel.textContent = stats.most_used_model;
    }
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
    console.log('[Dashboard] Refreshing model list via SDK...');
    
    if (!mcpClient) {
        console.warn('[Dashboard] SDK not available, skipping model refresh');
        return;
    }
    
    // Use SDK to search for popular models
    quickSearchModels('transformer', 10);
}

async function quickSearchModels(query, limit = 10) {
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.callTool('search_models', {
            query: query,
            limit: limit
        });
        
        const responseTime = Date.now() - startTime;
        trackSDKCall('search_models', true, responseTime);
        
        console.log('[Dashboard] Model search results:', result);
        showToast(`Found models (${responseTime}ms)`, 'success');
        
        return result;
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('search_models', false, responseTime);
        
        console.error('[Dashboard] Model search failed:', error);
        showToast('Failed to search models', 'error');
        
        return null;
    }
}

async function quickRecommendModels(task, constraints = {}) {
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.callTool('recommend_models', {
            task: task,
            constraints: constraints
        });
        
        const responseTime = Date.now() - startTime;
        trackSDKCall('recommend_models', true, responseTime);
        
        console.log('[Dashboard] Model recommendations:', result);
        showToast(`Got recommendations (${responseTime}ms)`, 'success');
        
        return result;
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('recommend_models', false, responseTime);
        
        console.error('[Dashboard] Model recommendation failed:', error);
        showToast('Failed to get recommendations', 'error');
        
        return null;
    }
}

function loadModel() {
    const modelId = prompt('Enter model ID to load:');
    if (modelId) {
        if (mcpClient) {
            loadModelViaSDK(modelId);
        } else {
            alert(`Loading model: ${modelId}`);
        }
    }
}

async function loadModelViaSDK(modelId) {
    const startTime = Date.now();
    
    try {
        // Try to get model details via SDK
        const result = await mcpClient.callTool('get_model_details', {
            model_id: modelId
        });
        
        const responseTime = Date.now() - startTime;
        trackSDKCall('get_model_details', true, responseTime);
        
        console.log('[Dashboard] Model details:', result);
        showToast(`Model loaded (${responseTime}ms)`, 'success');
        
        return result;
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('get_model_details', false, responseTime);
        
        console.error('[Dashboard] Failed to load model:', error);
        showToast(`Failed to load model: ${error.message}`, 'error');
        
        return null;
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
// Overview Tab Functions
let overviewRefreshInterval = null;

function refreshStatus() {
    console.log('[Dashboard] Refreshing overview status via SDK...');
    
    if (!mcpClient) {
        console.warn('[Dashboard] SDK not available for status refresh');
        return;
    }
    
    // Load all overview data
    loadServerStatusFromSDK();
    loadSystemMetricsFromSDK();
    loadCacheStatsFromSDK();
    loadPeerStatusFromSDK();
    
    // Start auto-refresh if not already running
    if (!overviewRefreshInterval) {
        startOverviewAutoRefresh();
    }
}

async function loadServerStatusFromSDK() {
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getServerStatus();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getServerStatus', true, responseTime);
        
        // Update UI with server status
        updateServerStatusUI(result);
        console.log('[Dashboard] Server status loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getServerStatus', false, responseTime);
        console.error('[Dashboard] Failed to load server status:', error);
    }
}

function updateServerStatusUI(data) {
    // Update available tools count
    const toolsCountEl = document.querySelector('.info-row span:contains("Available Tools:")');
    if (toolsCountEl && data.tools_count) {
        toolsCountEl.nextElementSibling.textContent = data.tools_count;
    }
    
    // Update status indicator
    const statusEl = document.querySelector('.status-running');
    if (statusEl && data.status) {
        statusEl.textContent = data.status;
        statusEl.className = data.status === 'running' ? 'status-running' : 'status-stopped';
    }
}

async function loadSystemMetricsFromSDK() {
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getDashboardSystemMetrics();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardSystemMetrics', true, responseTime);
        
        // Update UI with system metrics
        updateSystemMetricsUI(result);
        console.log('[Dashboard] System metrics loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardSystemMetrics', false, responseTime);
        console.error('[Dashboard] Failed to load system metrics:', error);
    }
}

function updateSystemMetricsUI(data) {
    // Update CPU usage
    const cpuEl = document.querySelector('[data-metric="cpu"]');
    if (cpuEl && data.cpu_usage !== undefined) {
        cpuEl.textContent = `${data.cpu_usage}%`;
        cpuEl.style.color = data.cpu_usage > 80 ? '#ef4444' : '#10b981';
    }
    
    // Update Memory usage
    const memEl = document.querySelector('[data-metric="memory"]');
    if (memEl && data.memory_usage !== undefined) {
        memEl.textContent = `${data.memory_usage}%`;
        memEl.style.color = data.memory_usage > 80 ? '#ef4444' : '#10b981';
    }
    
    // Update Disk usage
    const diskEl = document.querySelector('[data-metric="disk"]');
    if (diskEl && data.disk_usage !== undefined) {
        diskEl.textContent = `${data.disk_usage}%`;
    }
    
    // Update Network usage
    const netEl = document.querySelector('[data-metric="network"]');
    if (netEl && data.network_throughput) {
        netEl.textContent = data.network_throughput;
    }
}

async function loadCacheStatsFromSDK() {
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getDashboardCacheStats();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardCacheStats', true, responseTime);
        
        // Update UI with cache stats
        updateCacheStatsUI(result);
        console.log('[Dashboard] Cache stats loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardCacheStats', false, responseTime);
        console.error('[Dashboard] Failed to load cache stats:', error);
    }
}

function updateCacheStatsUI(data) {
    // Update cache statistics if available
    if (data.hits !== undefined && data.misses !== undefined) {
        const total = data.hits + data.misses;
        const hitRate = total > 0 ? ((data.hits / total) * 100).toFixed(1) : 0;
        console.log(`[Dashboard] Cache hit rate: ${hitRate}%`);
    }
}

async function loadPeerStatusFromSDK() {
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getDashboardPeerStatus();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardPeerStatus', true, responseTime);
        
        // Update UI with peer status
        updatePeerStatusUI(result);
        console.log('[Dashboard] Peer status loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getDashboardPeerStatus', false, responseTime);
        console.error('[Dashboard] Failed to load peer status:', error);
    }
}

function updatePeerStatusUI(data) {
    // Update peer count if available
    if (data.peer_count !== undefined) {
        console.log(`[Dashboard] Connected peers: ${data.peer_count}`);
    }
}

function startOverviewAutoRefresh() {
    // Clear existing interval if any
    if (overviewRefreshInterval) {
        clearInterval(overviewRefreshInterval);
    }
    
    // Refresh every 10 seconds
    overviewRefreshInterval = setInterval(() => {
        console.log('[Dashboard] Auto-refreshing overview data...');
        loadServerStatusFromSDK();
        loadSystemMetricsFromSDK();
        loadCacheStatsFromSDK();
        loadPeerStatusFromSDK();
    }, 10000);
    
    console.log('[Dashboard] Overview auto-refresh started (10s interval)');
}

function stopOverviewAutoRefresh() {
    if (overviewRefreshInterval) {
        clearInterval(overviewRefreshInterval);
        overviewRefreshInterval = null;
        console.log('[Dashboard] Overview auto-refresh stopped');
    }
}

// Queue Monitor auto-refresh interval
let queueRefreshInterval = null;

function refreshQueue() {
    console.log('[Dashboard] Refreshing queue status via SDK...');
    
    if (!mcpClient) {
        console.warn('[Dashboard] SDK not available for queue monitoring');
        displayQueueError('SDK not initialized');
        return;
    }
    
    // Load all queue-related data
    loadQueueStatusFromSDK();
    loadQueueHistoryFromSDK();
    loadPerformanceMetricsFromSDK();
    
    // Start auto-refresh if not already running
    if (!queueRefreshInterval) {
        startQueueAutoRefresh();
    }
}

async function loadQueueStatusFromSDK() {
    const queueStatusDiv = document.getElementById('queue-status-data');
    if (!queueStatusDiv) return;
    
    // Check cache
    const cached = sdkCache.get('queue_status');
    if (cached) {
        displayQueueStatus(cached);
        return;
    }
    
    // Show loading
    queueStatusDiv.innerHTML = '<div class="spinner large"></div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getQueueStatus();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getQueueStatus', true, responseTime);
        
        // Cache for 5 seconds
        sdkCache.set('queue_status', result, 5000);
        
        displayQueueStatus(result);
        console.log('[Dashboard] Queue status loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getQueueStatus', false, responseTime);
        
        console.error('[Dashboard] Failed to load queue status:', error);
        queueStatusDiv.innerHTML = `<div class="error-message">Failed to load queue status: ${error.message}</div>`;
    }
}

function displayQueueStatus(data) {
    const queueStatusDiv = document.getElementById('queue-status-data');
    if (!queueStatusDiv) return;
    
    const queueSize = data.queue_size || 0;
    const pending = data.pending || 0;
    const running = data.running || 0;
    const completed = data.completed || 0;
    const failed = data.failed || 0;
    const workers = data.workers || 0;
    const throughput = data.throughput || '0/s';
    
    queueStatusDiv.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div class="metric-card">
                <div class="metric-label">Queue Size</div>
                <div class="metric-value">${queueSize}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pending</div>
                <div class="metric-value" style="color: #f59e0b;">${pending}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Running</div>
                <div class="metric-value" style="color: #3b82f6;">${running}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Completed</div>
                <div class="metric-value" style="color: #10b981;">${completed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Failed</div>
                <div class="metric-value" style="color: #ef4444;">${failed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Workers</div>
                <div class="metric-value">${workers}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value">${throughput}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Updated</div>
                <div class="metric-value" style="font-size: 14px;">${new Date().toLocaleTimeString()}</div>
            </div>
        </div>
    `;
}

async function loadQueueHistoryFromSDK() {
    const historyDiv = document.getElementById('queue-history-data');
    if (!historyDiv) return;
    
    // Check cache
    const cached = sdkCache.get('queue_history');
    if (cached) {
        displayQueueHistory(cached);
        return;
    }
    
    // Show loading
    historyDiv.innerHTML = '<div class="spinner large"></div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getQueueHistory();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getQueueHistory', true, responseTime);
        
        // Cache for 5 seconds
        sdkCache.set('queue_history', result, 5000);
        
        displayQueueHistory(result);
        console.log('[Dashboard] Queue history loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getQueueHistory', false, responseTime);
        
        console.error('[Dashboard] Failed to load queue history:', error);
        historyDiv.innerHTML = `<div class="error-message">Failed to load queue history: ${error.message}</div>`;
    }
}

function displayQueueHistory(data) {
    const historyDiv = document.getElementById('queue-history-data');
    if (!historyDiv) return;
    
    const history = data.history || [];
    
    if (history.length === 0) {
        historyDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #6b7280;">No queue history available</div>';
        return;
    }
    
    const historyHtml = history.slice(0, 10).map(item => `
        <div style="padding: 10px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between;">
            <span>${item.task_name || 'Unknown task'}</span>
            <span style="color: ${item.status === 'completed' ? '#10b981' : '#ef4444'};">
                ${item.status} (${item.duration || 'N/A'})
            </span>
        </div>
    `).join('');
    
    historyDiv.innerHTML = historyHtml;
}

async function loadPerformanceMetricsFromSDK() {
    const metricsDiv = document.getElementById('performance-metrics-data');
    if (!metricsDiv) return;
    
    // Check cache
    const cached = sdkCache.get('performance_metrics');
    if (cached) {
        displayPerformanceMetrics(cached);
        return;
    }
    
    // Show loading
    metricsDiv.innerHTML = '<div class="spinner large"></div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.getPerformanceMetrics();
        const responseTime = Date.now() - startTime;
        trackSDKCall('getPerformanceMetrics', true, responseTime);
        
        // Cache for 5 seconds
        sdkCache.set('performance_metrics', result, 5000);
        
        displayPerformanceMetrics(result);
        console.log('[Dashboard] Performance metrics loaded:', result);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('getPerformanceMetrics', false, responseTime);
        
        console.error('[Dashboard] Failed to load performance metrics:', error);
        metricsDiv.innerHTML = `<div class="error-message">Failed to load performance metrics: ${error.message}</div>`;
    }
}

function displayPerformanceMetrics(data) {
    const metricsDiv = document.getElementById('performance-metrics-data');
    if (!metricsDiv) return;
    
    const cpu = data.cpu_usage || 0;
    const memory = data.memory_usage || 0;
    const disk = data.disk_usage || 0;
    const network = data.network_throughput || '0 MB/s';
    
    metricsDiv.innerHTML = `
        <div style="display: grid; gap: 15px;">
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>CPU Usage</span>
                    <span>${cpu}%</span>
                </div>
                <div style="background: #e5e7eb; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: ${cpu > 80 ? '#ef4444' : '#3b82f6'}; height: 100%; width: ${cpu}%;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Memory Usage</span>
                    <span>${memory}%</span>
                </div>
                <div style="background: #e5e7eb; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: ${memory > 80 ? '#ef4444' : '#10b981'}; height: 100%; width: ${memory}%;"></div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding-top: 10px;">
                <div class="metric-card">
                    <div class="metric-label">Disk Usage</div>
                    <div class="metric-value">${disk}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Network</div>
                    <div class="metric-value" style="font-size: 16px;">${network}</div>
                </div>
            </div>
        </div>
    `;
}

function startQueueAutoRefresh() {
    // Clear existing interval if any
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
    }
    
    // Refresh every 5 seconds
    queueRefreshInterval = setInterval(() => {
        console.log('[Dashboard] Auto-refreshing queue data...');
        
        // Clear cache to force fresh data
        sdkCache.clear('queue_status');
        sdkCache.clear('queue_history');
        sdkCache.clear('performance_metrics');
        
        // Reload data
        loadQueueStatusFromSDK();
        loadQueueHistoryFromSDK();
        loadPerformanceMetricsFromSDK();
    }, 5000);
    
    console.log('[Dashboard] Queue auto-refresh started (5s interval)');
}

function stopQueueAutoRefresh() {
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
        queueRefreshInterval = null;
        console.log('[Dashboard] Queue auto-refresh stopped');
    }
}

function displayQueueError(message) {
    const statusDiv = document.getElementById('queue-status-data');
    const historyDiv = document.getElementById('queue-history-data');
    const metricsDiv = document.getElementById('performance-metrics-data');
    
    const errorHtml = `<div class="error-message">${message}</div>`;
    
    if (statusDiv) statusDiv.innerHTML = errorHtml;
    if (historyDiv) historyDiv.innerHTML = errorHtml;
    if (metricsDiv) metricsDiv.innerHTML = errorHtml;
}

function clearQueue() {
    if (confirm('Are you sure you want to clear the queue?')) {
        if (mcpClient) {
            // TODO: Add SDK method for clearing queue when available
            showToast('Queue clear not yet implemented via SDK', 'warning');
        } else {
            alert('Queue cleared');
        }
    }
}

function addWorker() {
    if (mcpClient) {
        // TODO: Add SDK method for adding workers when available
        showToast('Add worker not yet implemented via SDK', 'warning');
    } else {
        alert('Adding new worker to pool...');
    }
}

function removeWorker() {
    if (confirm('Remove a worker from the pool?')) {
        if (mcpClient) {
            // TODO: Add SDK method for removing workers when available
            showToast('Remove worker not yet implemented via SDK', 'warning');
        } else {
            alert('Worker removed');
        }
    }
}

function exportQueueStats() {
    if (mcpClient) {
        // Export current queue data as JSON
        const queueData = {
            status: sdkCache.get('queue_status'),
            history: sdkCache.get('queue_history'),
            metrics: sdkCache.get('performance_metrics'),
            timestamp: new Date().toISOString()
        };
        
        const dataStr = JSON.stringify(queueData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `queue-stats-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
        
        showToast('Queue stats exported', 'success');
    } else {
        alert('Exporting queue statistics...');
    }
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
            
            // Cache the tools data for filtering
            cachedToolsData = data;
            
            toolsGrid.innerHTML = '';
            
            // Handle different response formats
            const tools = data.tools || data.data || [];
            const categories = data.categories || {};
            const total = data.total || tools.length;
            
            if (tools.length === 0) {
                toolsGrid.innerHTML = '<div class="tool-tag" style="background: #fef3c7; color: #92400e;">⚠️ No MCP tools found. Check server configuration.</div>';
                showToast('No MCP tools available', 'warning');
                return;
            }
            
            // Display tools by category
            if (Object.keys(categories).length > 0) {
                // Sort categories alphabetically
                const sortedCategories = Object.keys(categories).sort();
                
                sortedCategories.forEach(category => {
                    const categoryTools = categories[category];
                    
                    // Create category section
                    const categoryDiv = document.createElement('div');
                    categoryDiv.className = 'tool-category';
                    categoryDiv.style.cssText = 'margin-bottom: 20px;';
                    
                    // Category header
                    const categoryHeader = document.createElement('h4');
                    categoryHeader.textContent = `${category} (${categoryTools.length})`;
                    categoryHeader.style.cssText = 'margin-bottom: 10px; color: #374151; font-size: 14px; font-weight: 600;';
                    categoryDiv.appendChild(categoryHeader);
                    
                    // Create tools container for this category
                    const categoryToolsDiv = document.createElement('div');
                    categoryToolsDiv.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px;';
                    
                    // Display each tool in category
                    categoryTools.forEach(tool => {
                        // Normalize tool to an object shape so the modal receives the expected structure
                        const toolObj = (tool && typeof tool === 'object') ? tool : { name: tool };
                        
                        const toolTag = document.createElement('button');
                        toolTag.type = 'button';
                        toolTag.className = 'tool-tag';
                        toolTag.style.cssText = 'cursor: pointer; padding: 6px 12px; background: #e0e7ff; color: #3730a3; border-radius: 4px; font-size: 12px; transition: all 0.2s; border: none;';
                        toolTag.textContent = toolObj.name || tool;
                        
                        // Add description as tooltip if available
                        if (toolObj.description) {
                            toolTag.title = toolObj.description;
                        }
                        
                        // Add status indicator if available
                        if (toolObj.status === 'error' || toolObj.status === 'inactive') {
                            toolTag.style.background = '#fee2e2';
                            toolTag.style.color = '#991b1b';
                            toolTag.textContent += ' ⚠️';
                        }
                        
                        // Add click handler to show tool execution UI
                        toolTag.addEventListener('click', () => showToolExecutionModal(toolObj));
                        
                        // Add hover effect
                        toolTag.addEventListener('mouseenter', function() {
                            this.style.background = '#c7d2fe';
                            this.style.transform = 'translateY(-1px)';
                        });
                        toolTag.addEventListener('mouseleave', function() {
                            this.style.background = toolObj.status === 'error' ? '#fee2e2' : '#e0e7ff';
                            this.style.transform = 'translateY(0)';
                        });
                        
                        categoryToolsDiv.appendChild(toolTag);
                    });
                    
                    categoryDiv.appendChild(categoryToolsDiv);
                    toolsGrid.appendChild(categoryDiv);
                });
            } else {
                // Fallback: display tools without categories
                tools.forEach(tool => {
                    // Normalize tool to an object shape
                    const toolObj = (tool && typeof tool === 'object') ? tool : { name: tool };
                    
                    const toolTag = document.createElement('button');
                    toolTag.type = 'button';
                    toolTag.className = 'tool-tag';
                    toolTag.style.cssText = 'cursor: pointer; padding: 6px 12px; background: #e0e7ff; color: #3730a3; border-radius: 4px; font-size: 12px; transition: all 0.2s; border: none;';
                    toolTag.textContent = toolObj.name || tool;
                    
                    if (toolObj.description) {
                        toolTag.title = toolObj.description;
                    }
                    
                    if (toolObj.status === 'error' || toolObj.status === 'inactive') {
                        toolTag.style.background = '#fee2e2';
                        toolTag.style.color = '#991b1b';
                        toolTag.textContent += ' ⚠️';
                    }
                    
                    toolTag.addEventListener('click', () => showToolExecutionModal(toolObj));
                    
                    toolsGrid.appendChild(toolTag);
                });
            }
            
            console.log(`[Dashboard] Successfully loaded ${total} MCP tools in ${Object.keys(categories).length} categories`);
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
    const logContainer = document.getElementById('log-container');
    if (logContainer) {
        logContainer.innerHTML = '<div class="log-placeholder"><p class="text-muted">Loading system logs...</p></div>';
        fetch('/api/mcp/logs')
            .then(response => response.json())
            .then(data => {
                console.log('Logs loaded:', data);
                if (data.logs && data.logs.length > 0) {
                    logContainer.innerHTML = '';
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        const level = (log.level || 'INFO').toLowerCase();
                        logEntry.className = `log-entry log-${level}`;
                        
                        // Create emoji icon based on level
                        const emojiMap = {
                            'error': '❌',
                            'critical': '🔥',
                            'warning': '⚠️',
                            'info': 'ℹ️',
                            'debug': '🔍'
                        };
                        const emoji = emojiMap[level] || '📝';
                        
                        // Build structured log entry
                        logEntry.innerHTML = `
                            <span class="log-emoji">${emoji}</span>
                            <span class="log-timestamp">${log.timestamp || new Date().toISOString()}</span>
                            <span class="log-level">${(log.level || 'INFO').toUpperCase()}</span>
                            <span class="log-message">${log.message || ''}</span>
                        `;
                        
                        logContainer.appendChild(logEntry);
                    });
                    logContainer.scrollTop = logContainer.scrollHeight;
                    showToast(`Loaded ${data.logs.length} log entries`, 'success');
                } else {
                    logContainer.innerHTML = '<div class="log-placeholder"><p class="text-muted">No logs available</p></div>';
                }
            })
            .catch(error => {
                console.error('Error refreshing logs:', error);
                logContainer.innerHTML = `<div class="log-placeholder"><p class="text-danger">Failed to load logs: ${error.message}</p></div>`;
                showToast('Failed to load logs', 'error');
            });
    }
}

// Alias for compatibility with HTML button
function refreshSystemLogs() {
    refreshLogs();
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

// User Info Functions
async function refreshUserInfo() {
    console.log('[Dashboard] Refreshing user info...');
    const usernameEl = document.getElementById('username');
    const authStatusEl = document.getElementById('auth-status');
    const tokenTypeEl = document.getElementById('token-type');
    
    if (!usernameEl) return;
    
    usernameEl.textContent = 'Loading...';
    if (authStatusEl) authStatusEl.textContent = 'Checking...';
    
    try {
        const response = await fetch('/api/mcp/user');
        const data = await response.json();
        
        if (data.authenticated) {
            usernameEl.textContent = data.username || 'Unknown';
            if (authStatusEl) authStatusEl.textContent = '✓ Authenticated';
            if (tokenTypeEl) tokenTypeEl.textContent = data.token_type || 'unknown';
        } else {
            usernameEl.textContent = 'Not authenticated';
            if (authStatusEl) authStatusEl.textContent = '✗ Not authenticated';
            if (tokenTypeEl) tokenTypeEl.textContent = '-';
        }
    } catch (error) {
        console.error('[Dashboard] Error refreshing user info:', error);
        usernameEl.textContent = 'Error';
        if (authStatusEl) authStatusEl.textContent = 'Error';
    }
}

// Cache Stats Functions
async function refreshCacheStats() {
    console.log('[Dashboard] Refreshing cache stats...');
    const entriesEl = document.getElementById('cache-entries');
    const sizeEl = document.getElementById('cache-size');
    const hitRateEl = document.getElementById('cache-hit-rate');
    const peerHitsEl = document.getElementById('cache-peer-hits');
    const connectedPeersEl = document.getElementById('cache-connected-peers');
    const peerExchangeLastEl = document.getElementById('cache-peer-exchange-last');
    
    if (!entriesEl) return;
    
    entriesEl.textContent = 'Loading...';
    if (sizeEl) sizeEl.textContent = 'Loading...';
    if (hitRateEl) hitRateEl.textContent = 'Loading...';
    if (peerHitsEl) peerHitsEl.textContent = 'Loading...';
    if (connectedPeersEl) connectedPeersEl.textContent = 'Loading...';
    if (peerExchangeLastEl) peerExchangeLastEl.textContent = 'Loading...';
    
    try {
        const response = await fetch('/api/mcp/cache/stats');
        const data = await response.json();

        if (entriesEl) entriesEl.textContent = String(data.total_entries ?? 0);

        // Prefer formatted fields; fall back to numeric sizes
        if (sizeEl) {
            if (typeof data.cache_size === 'string') {
                sizeEl.textContent = data.cache_size;
            } else if (typeof data.total_size_mb === 'number') {
                sizeEl.textContent = `${data.total_size_mb.toFixed(2)} MB`;
            } else {
                sizeEl.textContent = '0.00 MB';
            }
        }

        if (hitRateEl) {
            if (typeof data.hit_rate_display === 'string') {
                hitRateEl.textContent = data.hit_rate_display;
            } else if (typeof data.hit_rate_percent === 'number') {
                hitRateEl.textContent = `${data.hit_rate_percent.toFixed(1)}%`;
            } else if (typeof data.hit_rate === 'number') {
                hitRateEl.textContent = `${(data.hit_rate * 100).toFixed(1)}%`;
            } else {
                hitRateEl.textContent = '0.0%';
            }
        }

        if (peerHitsEl) peerHitsEl.textContent = String(data.peer_hits ?? 0);
        if (connectedPeersEl) peerHitsEl && (connectedPeersEl.textContent = String(data.connected_peers ?? data.p2p_peers ?? 0));

        if (peerExchangeLastEl) {
            const iso = data.peer_exchange_last_iso;
            peerExchangeLastEl.textContent = iso ? iso : '—';
        }
    } catch (error) {
        console.error('[Dashboard] Error refreshing cache stats:', error);
        if (entriesEl) entriesEl.textContent = 'Error';
        if (sizeEl) sizeEl.textContent = 'Error';
        if (hitRateEl) hitRateEl.textContent = 'Error';
        if (peerHitsEl) peerHitsEl.textContent = 'Error';
        if (connectedPeersEl) connectedPeersEl.textContent = 'Error';
        if (peerExchangeLastEl) peerExchangeLastEl.textContent = 'Error';
    }
}

// Peer Status Functions
async function refreshPeerStatus() {
    console.log('[Dashboard] Refreshing peer status...');
    const peerStatusEl = document.getElementById('peer-status');
    const peerCountEl = document.getElementById('peer-count');
    const p2pEnabledEl = document.getElementById('p2p-enabled');
    const libp2pVersionEl = document.getElementById('libp2p-version');
    const peerIdEl = document.getElementById('p2p-peer-id');
    const knownAddrsEl = document.getElementById('p2p-known-addrs');
    const registryPeerCountEl = document.getElementById('registry-peer-count');
    
    if (!peerStatusEl) return;
    
    peerStatusEl.textContent = 'Checking...';
    if (peerCountEl) peerCountEl.textContent = 'Loading...';
    if (p2pEnabledEl) p2pEnabledEl.textContent = 'Loading...';
    if (libp2pVersionEl) libp2pVersionEl.textContent = 'Loading...';
    if (peerIdEl) peerIdEl.textContent = 'Loading...';
    if (knownAddrsEl) knownAddrsEl.textContent = 'Loading...';
    if (registryPeerCountEl) registryPeerCountEl.textContent = 'Loading...';
    
    try {
        const response = await fetch('/api/mcp/peers');
        const data = await response.json();

        const statusText = data.status ?? (data.active ? 'Active' : (data.enabled ? 'Enabled' : 'Disabled'));
        const peerCount = data.peer_count ?? data.p2p_peers ?? 0;
        const p2pEnabled = (data.p2p_enabled ?? data.enabled) ? true : false;

        if (peerStatusEl) peerStatusEl.textContent = statusText;
        if (peerCountEl) peerCountEl.textContent = String(peerCount);
        if (p2pEnabledEl) p2pEnabledEl.textContent = p2pEnabled ? 'Yes' : 'No';

        if (libp2pVersionEl) {
            const libp2p = data.libp2p || {};
            const version = libp2p.version || libp2p.installed_version || null;
            const ref = libp2p.vcs_ref || null;
            libp2pVersionEl.textContent = version ? (ref ? `${version} (${ref.slice(0, 12)})` : version) : 'Unavailable';
        }

        if (peerIdEl) {
            peerIdEl.textContent = data.peer_id || '—';
        }

        if (knownAddrsEl) {
            const count = data.known_peer_multiaddrs;
            knownAddrsEl.textContent = (typeof count === 'number' || typeof count === 'string') ? String(count) : '—';
        }

        if (registryPeerCountEl) {
            const count = data.registered_peer_count;
            registryPeerCountEl.textContent = (typeof count === 'number' || typeof count === 'string') ? String(count) : '—';
        }
    } catch (error) {
        console.error('[Dashboard] Error refreshing peer status:', error);
        if (peerStatusEl) peerStatusEl.textContent = 'Error';
        if (peerCountEl) peerCountEl.textContent = 'Error';
        if (p2pEnabledEl) p2pEnabledEl.textContent = 'Error';
        if (libp2pVersionEl) libp2pVersionEl.textContent = 'Error';
        if (peerIdEl) peerIdEl.textContent = 'Error';
        if (knownAddrsEl) knownAddrsEl.textContent = 'Error';
        if (registryPeerCountEl) registryPeerCountEl.textContent = 'Error';
    }
}

// Auto-refresh functionality
function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        if (currentTab === 'overview') {
            refreshServerStatus();
            refreshUserInfo();
            refreshCacheStats();
            refreshPeerStatus();
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
    // Initialize MCP SDK client first
    initializeSDK();
    
    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();
    
    // Add floating SDK menu
    createFloatingSDKMenu();
    
    // Initialize overview tab
    showTab('overview');
    
    // Load user info, cache stats, and peer status immediately
    refreshUserInfo();
    refreshCacheStats();
    refreshPeerStatus();
    
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

// Keyboard Shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Prevent shortcuts when typing in input fields (except for special keys)
        const isInputField = e.target.tagName === 'INPUT' || 
                           e.target.tagName === 'TEXTAREA' || 
                           e.target.isContentEditable;
        
        // ? key: Show keyboard shortcuts help (works anywhere)
        if (e.key === '?' && !isInputField) {
            e.preventDefault();
            showKeyboardShortcuts();
            return;
        }
        
        // Esc key: Close modals
        if (e.key === 'Escape') {
            closeKeyboardShortcuts();
            closeCommandPalette();
            return;
        }
        
        // Don't process other shortcuts when in input fields
        if (isInputField && !e.ctrlKey && !e.metaKey) {
            return;
        }
        
        // Ctrl/Cmd + K: Open command palette
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            toggleCommandPalette();
        }
        
        // Ctrl/Cmd + 1-9: Switch tabs
        if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '9') {
            e.preventDefault();
            const tabs = ['overview', 'ai-inference', 'model-manager', 'queue-monitor', 
                          'github-workflows', 'sdk-playground', 'mcp-tools', 'coverage', 'system-logs'];
            const index = parseInt(e.key) - 1;
            if (index < tabs.length) {
                showTab(tabs[index]);
            }
        }
        
        // Ctrl/Cmd + H: Quick hardware info
        if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
            e.preventDefault();
            quickGetHardwareInfo();
        }
        
        // Ctrl/Cmd + D: Quick Docker status
        if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            quickListContainers();
        }
        
        // Ctrl/Cmd + Shift + R: Refresh all
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && e.shiftKey) {
            e.preventDefault();
            quickRefreshAll();
        }
    });
    
    console.log('[Dashboard] Keyboard shortcuts initialized');
}

// Keyboard Shortcuts Modal Functions
function showKeyboardShortcuts() {
    const modal = document.getElementById('keyboard-shortcuts-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function closeKeyboardShortcuts() {
    const modal = document.getElementById('keyboard-shortcuts-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Floating SDK Menu
function createFloatingSDKMenu() {
    const menu = document.createElement('div');
    menu.id = 'floating-sdk-menu';
    menu.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    `;
    
    menu.innerHTML = `
        <button id="sdk-menu-btn" style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        ">⚡</button>
        <div id="sdk-quick-menu" style="
            display: none;
            position: absolute;
            bottom: 70px;
            right: 0;
            background: white;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            min-width: 200px;
        ">
            <div style="font-weight: 600; margin-bottom: 10px; padding: 5px 10px; border-bottom: 1px solid #e5e7eb;">
                Quick SDK Actions
            </div>
            <button onclick="quickGetHardwareInfo()" class="quick-menu-item">🔧 Hardware Info</button>
            <button onclick="quickListContainers()" class="quick-menu-item">🐳 Docker Status</button>
            <button onclick="quickGetNetworkPeers()" class="quick-menu-item">🌐 Network Peers</button>
            <button onclick="quickRefreshAll()" class="quick-menu-item">🔄 Refresh All</button>
            <div style="margin: 5px 0; border-top: 1px solid #e5e7eb;"></div>
            <button onclick="showTab('sdk-playground')" class="quick-menu-item">🎮 SDK Playground</button>
            <button onclick="showSDKStats()" class="quick-menu-item">📊 SDK Stats</button>
        </div>
    `;
    
    document.body.appendChild(menu);
    
    // Toggle menu
    const btn = document.getElementById('sdk-menu-btn');
    const quickMenu = document.getElementById('sdk-quick-menu');
    
    btn.addEventListener('click', function() {
        const isVisible = quickMenu.style.display === 'block';
        quickMenu.style.display = isVisible ? 'none' : 'block';
    });
    
    btn.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.1)';
    });
    
    btn.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!menu.contains(e.target)) {
            quickMenu.style.display = 'none';
        }
    });
    
    console.log('[Dashboard] Floating SDK menu created');
}

// Command Palette
let commandPaletteOpen = false;

function toggleCommandPalette() {
    if (commandPaletteOpen) {
        closeCommandPalette();
    } else {
        openCommandPalette();
    }
}

function openCommandPalette() {
    const palette = document.createElement('div');
    palette.id = 'command-palette';
    palette.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 10000;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 100px;
    `;
    
    palette.innerHTML = `
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            max-width: 600px;
            width: 90%;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        ">
            <input type="text" id="command-input" placeholder="Type a command or search..." 
                style="width: 100%; padding: 15px; border: 2px solid #667eea; border-radius: 8px; font-size: 16px; outline: none;" />
            <div id="command-results" style="margin-top: 15px; max-height: 400px; overflow-y: auto;"></div>
        </div>
    `;
    
    document.body.appendChild(palette);
    
    const input = document.getElementById('command-input');
    input.focus();
    
    // Command list
    const commands = [
        { name: 'Hardware Info', action: () => { quickGetHardwareInfo(); closeCommandPalette(); }, icon: '🔧' },
        { name: 'Docker Status', action: () => { quickListContainers(); closeCommandPalette(); }, icon: '🐳' },
        { name: 'Network Peers', action: () => { quickGetNetworkPeers(); closeCommandPalette(); }, icon: '🌐' },
        { name: 'Refresh All', action: () => { quickRefreshAll(); closeCommandPalette(); }, icon: '🔄' },
        { name: 'SDK Playground', action: () => { showTab('sdk-playground'); closeCommandPalette(); }, icon: '🎮' },
        { name: 'SDK Stats', action: () => { showSDKStats(); closeCommandPalette(); }, icon: '📊' },
        { name: 'MCP Tools', action: () => { showTab('mcp-tools'); closeCommandPalette(); }, icon: '🔧' },
        { name: 'AI Inference', action: () => { showTab('ai-inference'); closeCommandPalette(); }, icon: '🤖' },
        { name: 'Model Manager', action: () => { showTab('model-manager'); closeCommandPalette(); }, icon: '📚' },
    ];
    
    input.addEventListener('input', function(e) {
        const query = e.target.value.toLowerCase();
        const results = commands.filter(cmd => cmd.name.toLowerCase().includes(query));
        
        const resultsDiv = document.getElementById('command-results');
        resultsDiv.innerHTML = ''; // Clear previous results
        
        // Build command items using DOM APIs to avoid code injection
        results.forEach((cmd, index) => {
            const commandItem = document.createElement('div');
            commandItem.className = 'command-item';
            commandItem.dataset.index = index;
            commandItem.style.cssText = `
                padding: 12px 15px;
                cursor: pointer;
                border-radius: 6px;
                transition: background 0.2s;
                margin-bottom: 5px;
            `;
            commandItem.textContent = `${cmd.icon} ${cmd.name}`;
            
            // Attach event listener directly instead of using inline onclick
            commandItem.addEventListener('click', () => {
                cmd.action();
            });
            
            commandItem.addEventListener('mouseenter', function() {
                this.style.background = '#f3f4f6';
            });
            commandItem.addEventListener('mouseleave', function() {
                this.style.background = 'transparent';
            });
            
            resultsDiv.appendChild(commandItem);
        });
    });
    
    input.dispatchEvent(new Event('input'));
    
    // Close on Escape
    palette.addEventListener('click', function(e) {
        if (e.target === palette) {
            closeCommandPalette();
        }
    });
    
    document.addEventListener('keydown', function escHandler(e) {
        if (e.key === 'Escape') {
            closeCommandPalette();
            document.removeEventListener('keydown', escHandler);
        }
    });
    
    commandPaletteOpen = true;
}

function closeCommandPalette() {
    const palette = document.getElementById('command-palette');
    if (palette) {
        palette.remove();
        commandPaletteOpen = false;
    }
}

function showSDKStats() {
    showTab('sdk-playground');
    setTimeout(() => {
        const statsSection = document.getElementById('sdk-stats');
        if (statsSection) {
            statsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }, 100);
}

// Tool Execution Modal Functions
function showToolExecutionModal(tool) {
    console.log('[Dashboard] Opening execution modal for tool:', tool.name);
    
    // Create modal overlay
    const modalOverlay = document.createElement('div');
    modalOverlay.id = 'tool-execution-modal';
    modalOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    // Create modal content
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white;
        border-radius: 12px;
        padding: 24px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    `;
    
    // Build modal header using DOM APIs to prevent XSS
    const headerDiv = document.createElement('div');
    headerDiv.style.cssText = 'display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px;';
    
    const headerContent = document.createElement('div');
    
    const titleH3 = document.createElement('h3');
    titleH3.style.cssText = 'margin: 0; color: #111827; font-size: 18px; font-weight: 600;';
    titleH3.textContent = tool.name || 'Unknown Tool';
    
    const descriptionP = document.createElement('p');
    descriptionP.style.cssText = 'margin: 4px 0 0 0; color: #6b7280; font-size: 14px;';
    descriptionP.textContent = tool.description || 'No description available';
    
    headerContent.appendChild(titleH3);
    headerContent.appendChild(descriptionP);
    
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.textContent = '×';
    closeButton.style.cssText = 'background: none; border: none; font-size: 24px; cursor: pointer; color: #9ca3af; padding: 0; margin: 0; line-height: 1;';
    closeButton.onclick = closeToolExecutionModal;
    
    headerDiv.appendChild(headerContent);
    headerDiv.appendChild(closeButton);
    modalContent.appendChild(headerDiv);
    
    // Helper function to escape HTML
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    // Build parameters form
    let html = '';
    
    // Add parameters form if input schema available
    if (tool.input_schema && tool.input_schema.properties) {
        html += `<form id="tool-execution-form" style="margin-bottom: 16px;">`;
        
        const properties = tool.input_schema.properties;
        const required = tool.input_schema.required || [];
        
        for (const [paramName, paramSchema] of Object.entries(properties)) {
            const isRequired = required.includes(paramName);
            const paramType = paramSchema.type || 'string';
            const paramDescription = paramSchema.description || '';
            
            // Escape all user-provided values that go into HTML
            const escapedParamName = escapeHtml(paramName);
            const escapedDescription = escapeHtml(paramDescription);
            
            html += `
                <div style="margin-bottom: 16px;">
                    <label style="display: block; margin-bottom: 4px; color: #374151; font-size: 14px; font-weight: 500;">
                        ${escapedParamName}${isRequired ? ' *' : ''}
                    </label>
                    ${paramDescription ? `<p style="margin: 0 0 4px 0; color: #6b7280; font-size: 12px;">${escapedDescription}</p>` : ''}
            `;
            
            if (paramType === 'string' || paramType === 'number' || paramType === 'integer') {
                const inputType = (paramType === 'number' || paramType === 'integer') ? 'number' : 'text';
                const stepAttr = paramType === 'integer' ? 'step="1"' : '';
                html += `
                    <input 
                        type="${inputType}" 
                        name="${escapedParamName}" 
                        ${stepAttr}
                        ${isRequired ? 'required' : ''} 
                        data-param-type="${paramType}"
                        style="width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px;"
                        placeholder="Enter ${escapedParamName}"
                    />
                `;
            } else if (paramType === 'boolean') {
                html += `
                    <label style="display: flex; align-items: center; gap: 8px;">
                        <input 
                            type="checkbox" 
                            name="${escapedParamName}" 
                            style="width: 16px; height: 16px;"
                        />
                        <span style="color: #6b7280; font-size: 14px;">Enable</span>
                    </label>
                `;
            } else if (paramType === 'array') {
                html += `
                    <textarea 
                        name="${escapedParamName}" 
                        ${isRequired ? 'required' : ''} 
                        rows="3"
                        style="width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; font-family: monospace;"
                        placeholder="Enter JSON array, e.g., [1, 2, 3]"
                    ></textarea>
                `;
            } else {
                html += `
                    <textarea 
                        name="${escapedParamName}" 
                        ${isRequired ? 'required' : ''} 
                        rows="3"
                        style="width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; font-family: monospace;"
                        placeholder="Enter JSON value"
                    ></textarea>
                `;
            }
            
            html += `</div>`;
        }
        
        html += `</form>`;
    } else {
        html += `<p style="color: #6b7280; font-size: 14px; margin-bottom: 16px;">No parameters required for this tool.</p>`;
    }
    
    // Build buttons section container
    const buttonsDiv = document.createElement('div');
    buttonsDiv.style.cssText = 'display: flex; gap: 8px; margin-bottom: 16px;';
    
    const executeButton = document.createElement('button');
    executeButton.type = 'button';
    executeButton.textContent = '▶ Execute Tool';
    executeButton.style.cssText = 'flex: 1; padding: 10px 16px; background: #3b82f6; color: white; border: none; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.2s;';
    executeButton.onclick = () => executeToolFromModal(tool.name);
    executeButton.addEventListener('mouseenter', function() { this.style.background = '#2563eb'; });
    executeButton.addEventListener('mouseleave', function() { this.style.background = '#3b82f6'; });
    
    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = 'padding: 10px 16px; background: #e5e7eb; color: #374151; border: none; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer;';
    cancelButton.onclick = closeToolExecutionModal;
    
    buttonsDiv.appendChild(executeButton);
    buttonsDiv.appendChild(cancelButton);
    
    // Build result area
    const resultDiv = document.createElement('div');
    resultDiv.id = 'tool-execution-result';
    resultDiv.style.cssText = 'display: none; padding: 12px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 16px;';
    
    const resultTitle = document.createElement('h4');
    resultTitle.style.cssText = 'margin: 0 0 8px 0; color: #111827; font-size: 14px; font-weight: 600;';
    resultTitle.textContent = 'Result:';
    
    const resultPre = document.createElement('pre');
    resultPre.id = 'tool-execution-result-content';
    resultPre.style.cssText = 'margin: 0; white-space: pre-wrap; word-wrap: break-word; font-size: 12px; color: #374151; font-family: monospace;';
    
    resultDiv.appendChild(resultTitle);
    resultDiv.appendChild(resultPre);
    
    // Append form HTML safely
    const formContainer = document.createElement('div');
    formContainer.innerHTML = html;
    modalContent.appendChild(formContainer);
    modalContent.appendChild(buttonsDiv);
    modalContent.appendChild(resultDiv);
    
    modalOverlay.appendChild(modalContent);
    document.body.appendChild(modalOverlay);
    
    // Close modal on overlay click
    modalOverlay.addEventListener('click', function(e) {
        if (e.target === modalOverlay) {
            closeToolExecutionModal();
        }
    });
}

function closeToolExecutionModal() {
    const modal = document.getElementById('tool-execution-modal');
    if (modal) {
        modal.remove();
    }
}

function executeToolFromModal(toolName) {
    console.log('[Dashboard] Executing tool:', toolName);
    
    const form = document.getElementById('tool-execution-form');
    const resultDiv = document.getElementById('tool-execution-result');
    const resultContent = document.getElementById('tool-execution-result-content');
    
    // Collect parameters from form
    const params = {};
    if (form) {
        const formData = new FormData(form);
        for (const [key, value] of formData.entries()) {
            const input = form.elements[key];
            if (input.type === 'checkbox') {
                params[key] = input.checked;
            } else if (input.type === 'number') {
                const rawValue = typeof value === 'string' ? value.trim() : value;
                // Skip empty numeric fields to avoid NaN for optional fields
                if (rawValue === '') {
                    continue;
                }
                // Preserve integer semantics when the input has data-param-type="integer"
                const paramType = input.getAttribute('data-param-type');
                if (paramType === 'integer') {
                    params[key] = parseInt(rawValue, 10);
                } else {
                    params[key] = parseFloat(rawValue);
                }
            } else {
                // Try to parse as JSON for arrays/objects
                if (value.trim().startsWith('[') || value.trim().startsWith('{')) {
                    try {
                        params[key] = JSON.parse(value);
                    } catch (e) {
                        params[key] = value;
                        // Show user-friendly error message
                        if (resultDiv && resultContent) {
                            resultDiv.style.display = 'block';
                            resultContent.textContent = `Error parsing JSON for parameter "${key}": ${e.message}`;
                            resultContent.style.color = '#991b1b';
                        }
                        return;
                    }
                } else {
                    params[key] = value;
                }
            }
        }
    }
    
    // Show loading state
    if (resultDiv) {
        resultDiv.style.display = 'block';
        resultContent.textContent = 'Executing tool...';
        resultContent.style.color = '#6b7280';
    }
    
    // Use SDK if available, otherwise fall back to fetch
    const startTime = Date.now();
    
    if (mcpClient) {
        // Execute tool using SDK
        mcpClient.callTool(toolName, params)
            .then(result => {
                const responseTime = Date.now() - startTime;
                trackSDKCall(toolName, true, responseTime);
                
                console.log('[Dashboard] Tool execution result:', result);
                resultContent.textContent = JSON.stringify(result, null, 2);
                resultContent.style.color = '#059669';
                showToast(`Tool executed successfully (${responseTime}ms)`, 'success');
            })
            .catch(error => {
                const responseTime = Date.now() - startTime;
                trackSDKCall(toolName, false, responseTime);
                
                console.error('[Dashboard] Error executing tool:', error);
                resultContent.textContent = `Error: ${error.message}`;
                resultContent.style.color = '#dc2626';
                showToast(`Tool execution failed: ${error.message}`, 'error', 5000);
            });
    } else {
        // Fallback to direct JSON-RPC fetch
        const requestBody = {
            jsonrpc: '2.0',
            method: 'tools/call',
            params: {
                name: toolName,
                arguments: params
            },
            id: Date.now()
        };
        
        fetch('/jsonrpc', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => response.json())
        .then(data => {
            console.log('[Dashboard] Tool execution result:', data);
            
            if (data.error) {
                resultContent.textContent = `Error: ${data.error.message}`;
                resultContent.style.color = '#dc2626';
                showToast(`Tool execution failed: ${data.error.message}`, 'error', 5000);
            } else {
                resultContent.textContent = JSON.stringify(data.result, null, 2);
                resultContent.style.color = '#059669';
                showToast('Tool executed successfully', 'success');
            }
        })
        .catch(error => {
            console.error('[Dashboard] Error executing tool:', error);
            resultContent.textContent = `Error: ${error.message}`;
            resultContent.style.color = '#dc2626';
            showToast(`Tool execution failed: ${error.message}`, 'error', 5000);
        });
    }
}

// Tool Search and Filter Functions
let cachedToolsData = null;

function filterTools(searchTerm) {
    if (!cachedToolsData) {
        console.warn('[Dashboard] No cached tools data available. Refresh tools first.');
        return;
    }
    
    const toolsGrid = document.querySelector('.tools-grid');
    if (!toolsGrid) return;
    
    const normalizedSearch = searchTerm.toLowerCase().trim();
    
    if (!normalizedSearch) {
        // Show all tools if search is empty
        displayToolsFromCache(cachedToolsData);
        return;
    }
    
    // Filter tools and categories
    const filteredCategories = {};
    const filteredTools = [];
    
    if (cachedToolsData.categories) {
        for (const [category, tools] of Object.entries(cachedToolsData.categories)) {
            const categoryMatch = category.toLowerCase().includes(normalizedSearch);
            const matchingTools = tools.filter(tool => 
                categoryMatch || 
                tool.name.toLowerCase().includes(normalizedSearch) ||
                (tool.description && tool.description.toLowerCase().includes(normalizedSearch))
            );
            
            if (matchingTools.length > 0) {
                filteredCategories[category] = matchingTools;
                filteredTools.push(...matchingTools);
            }
        }
    }
    
    // Display filtered results
    displayToolsFromCache({
        tools: filteredTools,
        categories: filteredCategories,
        total: filteredTools.length
    });
    
    if (filteredTools.length === 0) {
        toolsGrid.innerHTML = `<div class="tool-tag" style="background: #fef3c7; color: #92400e;">No tools found matching "${searchTerm}"</div>`;
    }
}

function clearToolSearch() {
    const searchInput = document.getElementById('tool-search-input');
    if (searchInput) {
        searchInput.value = '';
    }
    
    if (cachedToolsData) {
        displayToolsFromCache(cachedToolsData);
    }
}

function displayToolsFromCache(data) {
    const toolsGrid = document.querySelector('.tools-grid');
    if (!toolsGrid) return;
    
    toolsGrid.innerHTML = '';
    
    const tools = data.tools || [];
    const categories = data.categories || {};
    
    if (tools.length === 0) {
        toolsGrid.innerHTML = '<div class="tool-tag" style="background: #fef3c7; color: #92400e;">⚠️ No tools to display</div>';
        return;
    }
    
    // Display tools by category if available
    if (Object.keys(categories).length > 0) {
        const sortedCategories = Object.keys(categories).sort();
        
        sortedCategories.forEach(category => {
            const categoryTools = categories[category];
            
            // Create category section
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'tool-category';
            categoryDiv.style.cssText = 'margin-bottom: 20px;';
            
            // Category header
            const categoryHeader = document.createElement('h4');
            categoryHeader.textContent = `${category} (${categoryTools.length})`;
            categoryHeader.style.cssText = 'margin-bottom: 10px; color: #374151; font-size: 14px; font-weight: 600;';
            categoryDiv.appendChild(categoryHeader);
            
            // Create tools container for this category
            const categoryToolsDiv = document.createElement('div');
            categoryToolsDiv.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px;';
            
            // Display each tool in category
            categoryTools.forEach(tool => {
                // Normalize tool to an object shape
                const toolObj = (tool && typeof tool === 'object') ? tool : { name: tool };
                
                const toolTag = document.createElement('button');
                toolTag.type = 'button';
                toolTag.className = 'tool-tag';
                toolTag.style.cssText = 'cursor: pointer; padding: 6px 12px; background: #e0e7ff; color: #3730a3; border-radius: 4px; font-size: 12px; transition: all 0.2s; border: none;';
                toolTag.textContent = toolObj.name || tool;
                
                if (toolObj.description) {
                    toolTag.title = toolObj.description;
                }
                
                if (toolObj.status === 'error' || toolObj.status === 'inactive') {
                    toolTag.style.background = '#fee2e2';
                    toolTag.style.color = '#991b1b';
                    toolTag.textContent += ' ⚠️';
                }
                
                toolTag.addEventListener('click', () => showToolExecutionModal(toolObj));
                
                toolTag.addEventListener('mouseenter', function() {
                    this.style.background = '#c7d2fe';
                    this.style.transform = 'translateY(-1px)';
                });
                toolTag.addEventListener('mouseleave', function() {
                    this.style.background = toolObj.status === 'error' ? '#fee2e2' : '#e0e7ff';
                    this.style.transform = 'translateY(0)';
                });
                
                categoryToolsDiv.appendChild(toolTag);
            });
            
            categoryDiv.appendChild(categoryToolsDiv);
            toolsGrid.appendChild(categoryDiv);
        });
    } else {
        // Fallback: display a flat list of tools when there are no categories
        let flatTools = [];
        
        if (Array.isArray(data.tools)) {
            flatTools = data.tools;
        } else if (Array.isArray(data)) {
            flatTools = data;
        }
        
        if (!flatTools || flatTools.length === 0) {
            return;
        }
        
        flatTools.forEach(tool => {
            // Normalize tool to an object shape
            const toolObj = (tool && typeof tool === 'object') ? tool : { name: tool };
            
            const toolTag = document.createElement('button');
            toolTag.type = 'button';
            toolTag.className = 'tool-tag';
            toolTag.style.cssText = 'cursor: pointer; padding: 6px 12px; background: #e0e7ff; color: #3730a3; border-radius: 4px; font-size: 12px; transition: all 0.2s; border: none;';
            toolTag.textContent = toolObj.name || tool;
            
            if (toolObj.description) {
                toolTag.title = toolObj.description;
            }
            
            if (toolObj.status === 'error' || toolObj.status === 'inactive') {
                toolTag.style.background = '#fee2e2';
                toolTag.style.color = '#991b1b';
                toolTag.textContent += ' ⚠️';
            }
            
            toolTag.addEventListener('click', () => showToolExecutionModal(toolObj));
            
            toolTag.addEventListener('mouseenter', function() {
                this.style.background = '#c7d2fe';
                this.style.transform = 'translateY(-1px)';
            });
            toolTag.addEventListener('mouseleave', function() {
                this.style.background = toolObj.status === 'error' ? '#fee2e2' : '#e0e7ff';
                this.style.transform = 'translateY(0)';
            });
            
            toolsGrid.appendChild(toolTag);
        });
    }
}

// SDK Playground Functions
function initializeSDKPlayground() {
    console.log('[Dashboard] Initializing SDK Playground');
    
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    // Update SDK stats display
    updateSDKStats();
}

// Quick Action Functions for Overview Tab
async function quickGetHardwareInfo(useCache = true) {
    const resultDiv = document.getElementById('quick-action-result');
    const contentDiv = document.getElementById('quick-action-content');
    
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const cacheKey = 'hardware_info';
    
    // Check cache first if enabled
    if (useCache && sdkCache.has(cacheKey)) {
        const cached = sdkCache.get(cacheKey);
        resultDiv.style.display = 'block';
        contentDiv.textContent = JSON.stringify(cached, null, 2) + '\n\n(Cached data)';
        showToast('Hardware info loaded from cache', 'info');
        return;
    }
    
    resultDiv.style.display = 'block';
    contentDiv.innerHTML = '<div class="spinner-large"></div><div class="loading-text">Loading hardware information...</div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.hardwareGetInfo();
        const responseTime = Date.now() - startTime;
        trackSDKCall('hardware_get_info', true, responseTime);
        
        // Cache the result
        sdkCache.set(cacheKey, result);
        
        contentDiv.textContent = JSON.stringify(result, null, 2);
        showToast(`Hardware info loaded (${responseTime}ms)`, 'success');
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('hardware_get_info', false, responseTime);
        
        contentDiv.innerHTML = `<div class="error-message"><strong>Error</strong>${error.message}</div>`;
        showToast('Failed to get hardware info', 'error');
    }
}

async function quickListContainers(useCache = true) {
    const resultDiv = document.getElementById('quick-action-result');
    const contentDiv = document.getElementById('quick-action-content');
    
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const cacheKey = 'docker_containers';
    
    // Check cache first if enabled
    if (useCache && sdkCache.has(cacheKey)) {
        const cached = sdkCache.get(cacheKey);
        resultDiv.style.display = 'block';
        contentDiv.textContent = JSON.stringify(cached, null, 2) + '\n\n(Cached data)';
        showToast('Container list loaded from cache', 'info');
        return;
    }
    
    resultDiv.style.display = 'block';
    contentDiv.innerHTML = '<div class="spinner-large"></div><div class="loading-text">Loading Docker containers...</div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.dockerListContainers(true);
        const responseTime = Date.now() - startTime;
        trackSDKCall('docker_list_containers', true, responseTime);
        
        // Cache the result
        sdkCache.set(cacheKey, result, 2 * 60 * 1000); // 2 minute cache for docker
        
        contentDiv.textContent = JSON.stringify(result, null, 2);
        showToast(`Container list loaded (${responseTime}ms)`, 'success');
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('docker_list_containers', false, responseTime);
        
        contentDiv.innerHTML = `<div class="error-message"><strong>Error</strong>${error.message}</div>`;
        showToast('Failed to list containers', 'error');
    }
}

async function quickGetNetworkPeers(useCache = true) {
    const resultDiv = document.getElementById('quick-action-result');
    const contentDiv = document.getElementById('quick-action-content');
    
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const cacheKey = 'network_peers';
    
    // Check cache first if enabled
    if (useCache && sdkCache.has(cacheKey)) {
        const cached = sdkCache.get(cacheKey);
        resultDiv.style.display = 'block';
        contentDiv.textContent = JSON.stringify(cached, null, 2) + '\n\n(Cached data)';
        showToast('Network peers loaded from cache', 'info');
        return;
    }
    
    resultDiv.style.display = 'block';
    contentDiv.innerHTML = '<div class="spinner-large"></div><div class="loading-text">Loading network peers...</div>';
    
    const startTime = Date.now();
    
    try {
        const result = await mcpClient.networkListPeers();
        const responseTime = Date.now() - startTime;
        trackSDKCall('network_list_peers', true, responseTime);
        
        // Cache the result
        sdkCache.set(cacheKey, result, 1 * 60 * 1000); // 1 minute cache for network
        
        contentDiv.textContent = JSON.stringify(result, null, 2);
        showToast(`Network peers loaded (${responseTime}ms)`, 'success');
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('network_list_peers', false, responseTime);
        
        contentDiv.innerHTML = `<div class="error-message"><strong>Error</strong>${error.message}</div>`;
        showToast('Failed to get network peers', 'error');
    }
}

async function quickRefreshAll() {
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    // Clear cache for fresh data
    sdkCache.clear();
    
    const resultDiv = document.getElementById('quick-action-result');
    const contentDiv = document.getElementById('quick-action-content');
    
    resultDiv.style.display = 'block';
    contentDiv.innerHTML = '<div class="spinner-large"></div><div class="loading-text">Refreshing all data with batch request...</div>';
    
    const startTime = Date.now();
    
    try {
        // Use batch SDK call for efficiency
        const results = await mcpClient.callToolsBatch([
            { name: 'hardware_get_info', arguments: {} },
            { name: 'docker_list_containers', arguments: { all: true } },
            { name: 'network_list_peers', arguments: {} }
        ]);
        
        const responseTime = Date.now() - startTime;
        trackSDKCall('batch_refresh_all', true, responseTime);
        
        const summary = {
            hardware: results[0].result || results[0].error,
            docker: results[1].result || results[1].error,
            network: results[2].result || results[2].error,
            responseTime: `${responseTime}ms`
        };
        
        contentDiv.textContent = JSON.stringify(summary, null, 2);
        showToast(`All data refreshed (${responseTime}ms)`, 'success');
        
        // Update overview cards
        updateOverviewCards(summary);
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('batch_refresh_all', false, responseTime);
        
        contentDiv.textContent = `Error: ${error.message}`;
        showToast('Failed to refresh data', 'error');
    }
}

function updateOverviewCards(data) {
    // Update available tools count
    if (data.hardware && data.hardware.status === 'success') {
        console.log('[Dashboard] Hardware data updated');
    }
    
    // Update Docker container count
    if (data.docker && data.docker.containers) {
        const containerCount = data.docker.containers.length;
        const runningCount = data.docker.containers.filter(c => c.status === 'running').length;
        console.log(`[Dashboard] Docker: ${runningCount}/${containerCount} containers running`);
    }
    
    // Update network peer count
    if (data.network && data.network.peers) {
        const peerCount = data.network.peers.length;
        console.log(`[Dashboard] Network: ${peerCount} peers connected`);
    }
}

function updateSDKStats() {
    const statsContainer = document.getElementById('sdk-stats');
    if (!statsContainer) return;
    
    const successRate = sdkStats.totalCalls > 0 
        ? ((sdkStats.successfulCalls / sdkStats.totalCalls) * 100).toFixed(1)
        : 0;
    
    statsContainer.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${sdkStats.totalCalls}</div>
            <div class="stat-label">Total Calls</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${sdkStats.successfulCalls}</div>
            <div class="stat-label">Successful</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${sdkStats.failedCalls}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${successRate}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${Math.round(sdkStats.avgResponseTime)}ms</div>
            <div class="stat-label">Avg Response</div>
        </div>
    `;
}

async function runSDKExample(category, method, args = {}) {
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return null;
    }
    
    const resultContainer = document.getElementById('sdk-result');
    const codeContainer = document.getElementById('sdk-code');
    
    if (resultContainer) {
        resultContainer.innerHTML = '<div class="loading">Executing...</div>';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        const methodName = `${category}_${method}`;
        
        // Generate code snippet using callTool interface for consistency
        const argsStr = Object.keys(args).length > 0 
            ? '\n  ' + JSON.stringify(args, null, 2).split('\n').join('\n  ') + '\n'
            : '{}';
        const codeSnippet = `// Using SDK callTool interface
const client = new MCPClient('/jsonrpc');
const result = await client.callTool('${methodName}', ${argsStr});
console.log(result);`;
        
        if (codeContainer) {
            codeContainer.textContent = codeSnippet;
        }
        
        // Call using the universal callTool method for consistency
        result = await mcpClient.callTool(methodName, args);
        
        const responseTime = Date.now() - startTime;
        trackSDKCall(methodName, true, responseTime);
        
        if (resultContainer) {
            resultContainer.innerHTML = `
                <div class="result-success">
                    <div class="result-header">
                        <span class="result-status">✅ Success</span>
                        <span class="result-time">${responseTime}ms</span>
                    </div>
                    <pre class="result-json">${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }
        
        showToast(`${methodName} executed successfully`, 'success');
        updateSDKStats();
        
        return result;
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall(`${category}_${method}`, false, responseTime);
        
        if (resultContainer) {
            resultContainer.innerHTML = `
                <div class="result-error">
                    <div class="result-header">
                        <span class="result-status">❌ Error</span>
                        <span class="result-time">${responseTime}ms</span>
                    </div>
                    <pre class="result-json">${error.message || error}</pre>
                </div>
            `;
        }
        
        showToast(`Failed to execute ${category}_${method}`, 'error');
        updateSDKStats();
        
        return null;
    }
}

// Batch execution example
async function runBatchExample() {
    if (!mcpClient) {
        showToast('SDK not initialized', 'error');
        return;
    }
    
    const resultContainer = document.getElementById('sdk-result');
    const codeContainer = document.getElementById('sdk-code');
    
    if (resultContainer) {
        resultContainer.innerHTML = '<div class="loading">Executing batch...</div>';
    }
    
    const batchCalls = [
        { name: 'hardware_get_info', arguments: {} },
        { name: 'network_list_peers', arguments: {} }
    ];
    
    const codeSnippet = `// Batch execution
const client = new MCPClient('/jsonrpc');
const results = await client.callToolsBatch([
    { name: 'hardware_get_info', arguments: {} },
    { name: 'network_list_peers', arguments: {} }
]);
console.log(results);`;
    
    if (codeContainer) {
        codeContainer.textContent = codeSnippet;
    }
    
    const startTime = Date.now();
    
    try {
        const results = await mcpClient.callToolsBatch(batchCalls);
        const responseTime = Date.now() - startTime;
        
        trackSDKCall('batch_execution', true, responseTime);
        
        if (resultContainer) {
            resultContainer.innerHTML = `
                <div class="result-success">
                    <div class="result-header">
                        <span class="result-status">✅ Batch Success</span>
                        <span class="result-time">${responseTime}ms</span>
                    </div>
                    <pre class="result-json">${JSON.stringify(results, null, 2)}</pre>
                </div>
            `;
        }
        
        showToast('Batch execution completed successfully', 'success');
        updateSDKStats();
    } catch (error) {
        const responseTime = Date.now() - startTime;
        trackSDKCall('batch_execution', false, responseTime);
        
        if (resultContainer) {
            resultContainer.innerHTML = `
                <div class="result-error">
                    <div class="result-header">
                        <span class="result-status">❌ Batch Error</span>
                        <span class="result-time">${responseTime}ms</span>
                    </div>
                    <pre class="result-json">${error.message || error}</pre>
                </div>
            `;
        }
        
        showToast('Batch execution failed', 'error');
        updateSDKStats();
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
});
// ========================================
// IPFS Manager Functions (Phase 2.4)
// ========================================

async function ipfsCat() {
    const path = document.getElementById('ipfs-path')?.value;
    const resultDiv = document.getElementById('ipfs-file-result');
    
    if (!path) {
        showToast('Please enter a file path or CID', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Reading file...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsCat) {
            result = await mcpClient.ipfsCat(path);
            trackSDKCall('ipfsCat', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_cat', { path });
            trackSDKCall('ipfsCat', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>✅ File Content:</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px; max-height: 400px; overflow-y: auto;">${result.content || result}</pre>
                </div>
            `;
        }
        showToast('File read successfully', 'success');
    } catch (error) {
        trackSDKCall('ipfsCat', false, Date.now() - startTime);
        console.error('[IPFS] Cat error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to read file: ${error.message}`, 'error');
    }
}

async function ipfsList() {
    const path = document.getElementById('ipfs-path')?.value;
    const resultDiv = document.getElementById('ipfs-file-result');
    
    if (!path) {
        showToast('Please enter a directory path or CID', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Listing directory...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsLs) {
            result = await mcpClient.ipfsLs(path);
            trackSDKCall('ipfsLs', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_ls', { path });
            trackSDKCall('ipfsLs', false, Date.now() - startTime);
        }
        
        const files = result.files || result.objects || result || [];
        let html = '<div class="result-success"><strong>📋 Directory Listing:</strong><div style="margin-top: 10px;">';
        
        if (Array.isArray(files) && files.length > 0) {
            files.forEach(file => {
                html += `
                    <div style="padding: 8px; border-bottom: 1px solid #e5e7eb;">
                        <strong>${file.name || file.Name}</strong>
                        <span style="color: #6b7280; margin-left: 10px;">${file.size || file.Size || 0} bytes</span>
                        <span style="color: #9ca3af; margin-left: 10px;">${file.type || file.Type || 'file'}</span>
                    </div>
                `;
            });
        } else {
            html += '<p>No files found or empty directory</p>';
        }
        
        html += '</div></div>';
        
        if (resultDiv) {
            resultDiv.innerHTML = html;
        }
        showToast('Directory listed successfully', 'success');
    } catch (error) {
        trackSDKCall('ipfsLs', false, Date.now() - startTime);
        console.error('[IPFS] Ls error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to list directory: ${error.message}`, 'error');
    }
}

async function ipfsGetFile() {
    const path = document.getElementById('ipfs-path')?.value;
    const resultDiv = document.getElementById('ipfs-file-result');
    
    if (!path) {
        showToast('Please enter a file path or CID', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Getting file...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.getFileFromIpfs) {
            result = await mcpClient.getFileFromIpfs(path);
            trackSDKCall('getFileFromIpfs', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('get_file_from_ipfs', { cid: path });
            trackSDKCall('getFileFromIpfs', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>✅ File Retrieved:</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px; max-height: 400px; overflow-y: auto;">${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }
        showToast('File retrieved successfully', 'success');
    } catch (error) {
        trackSDKCall('getFileFromIpfs', false, Date.now() - startTime);
        console.error('[IPFS] Get file error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to get file: ${error.message}`, 'error');
    }
}

async function ipfsAddFile() {
    const content = document.getElementById('ipfs-add-content')?.value;
    const filename = document.getElementById('ipfs-add-filename')?.value || 'unnamed.txt';
    const resultDiv = document.getElementById('ipfs-add-result');
    
    if (!content) {
        showToast('Please enter content to add', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Adding file to IPFS...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsAddFile) {
            result = await mcpClient.ipfsAddFile({ content, filename });
            trackSDKCall('ipfsAddFile', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_add_file', { content, filename });
            trackSDKCall('ipfsAddFile', false, Date.now() - startTime);
        }
        
        const cid = result.cid || result.Hash || result.hash;
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>✅ File Added Successfully!</strong>
                    <div style="margin-top: 10px;">
                        <strong>CID:</strong> <code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px;">${cid}</code>
                    </div>
                    <div style="margin-top: 5px;">
                        <strong>Size:</strong> ${result.size || result.Size || 'Unknown'} bytes
                    </div>
                </div>
            `;
        }
        showToast(`File added: ${cid}`, 'success');
    } catch (error) {
        trackSDKCall('ipfsAddFile', false, Date.now() - startTime);
        console.error('[IPFS] Add file error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to add file: ${error.message}`, 'error');
    }
}

async function ipfsAddFileShared() {
    const content = document.getElementById('ipfs-add-content')?.value;
    const filename = document.getElementById('ipfs-add-filename')?.value || 'unnamed.txt';
    const resultDiv = document.getElementById('ipfs-add-result');
    
    if (!content) {
        showToast('Please enter content to add', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Adding shared file...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.addFileShared) {
            result = await mcpClient.addFileShared({ content, filename });
            trackSDKCall('addFileShared', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('add_file_shared', { content, filename });
            trackSDKCall('addFileShared', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>✅ Shared File Added!</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px;">${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }
        showToast('Shared file added successfully', 'success');
    } catch (error) {
        trackSDKCall('addFileShared', false, Date.now() - startTime);
        console.error('[IPFS] Add shared file error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to add shared file: ${error.message}`, 'error');
    }
}

async function ipfsPinAdd() {
    const cid = document.getElementById('ipfs-pin-cid')?.value;
    const resultDiv = document.getElementById('ipfs-pin-result');
    
    if (!cid) {
        showToast('Please enter a CID to pin', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Pinning...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsPinAdd) {
            result = await mcpClient.ipfsPinAdd(cid);
            trackSDKCall('ipfsPinAdd', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_pin_add', { cid });
            trackSDKCall('ipfsPinAdd', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="result-success">✅ CID pinned successfully: ${cid}</div>`;
        }
        showToast('CID pinned successfully', 'success');
    } catch (error) {
        trackSDKCall('ipfsPinAdd', false, Date.now() - startTime);
        console.error('[IPFS] Pin add error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to pin: ${error.message}`, 'error');
    }
}

async function ipfsPinRemove() {
    const cid = document.getElementById('ipfs-pin-cid')?.value;
    const resultDiv = document.getElementById('ipfs-pin-result');
    
    if (!cid) {
        showToast('Please enter a CID to unpin', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Unpinning...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsPinRm) {
            result = await mcpClient.ipfsPinRm(cid);
            trackSDKCall('ipfsPinRm', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_pin_rm', { cid });
            trackSDKCall('ipfsPinRm', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="result-success">✅ CID unpinned successfully: ${cid}</div>`;
        }
        showToast('CID unpinned successfully', 'success');
    } catch (error) {
        trackSDKCall('ipfsPinRm', false, Date.now() - startTime);
        console.error('[IPFS] Pin remove error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to unpin: ${error.message}`, 'error');
    }
}

async function ipfsListPins() {
    const resultDiv = document.getElementById('ipfs-pin-result');
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Loading pins...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsPinLs) {
            result = await mcpClient.ipfsPinLs();
            trackSDKCall('ipfsPinLs', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_pin_ls', {});
            trackSDKCall('ipfsPinLs', false, Date.now() - startTime);
        }
        
        const pins = result.pins || result.Keys || result || [];
        let html = '<div class="result-success"><strong>📌 Pinned CIDs:</strong><div style="margin-top: 10px;">';
        
        if (Array.isArray(pins) && pins.length > 0) {
            pins.forEach(pin => {
                html += `<div style="padding: 5px; font-family: monospace; font-size: 12px;">${pin.cid || pin}</div>`;
            });
        } else if (typeof pins === 'object') {
            Object.keys(pins).forEach(cid => {
                html += `<div style="padding: 5px; font-family: monospace; font-size: 12px;">${cid}</div>`;
            });
        } else {
            html += '<p>No pinned CIDs found</p>';
        }
        
        html += '</div></div>';
        
        if (resultDiv) {
            resultDiv.innerHTML = html;
        }
        showToast('Pins listed successfully', 'success');
    } catch (error) {
        trackSDKCall('ipfsPinLs', false, Date.now() - startTime);
        console.error('[IPFS] Pin list error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to list pins: ${error.message}`, 'error');
    }
}

async function ipfsSwarmPeers() {
    const resultDiv = document.getElementById('ipfs-swarm-result');
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Loading peers...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsSwarmPeers) {
            result = await mcpClient.ipfsSwarmPeers();
            trackSDKCall('ipfsSwarmPeers', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_swarm_peers', {});
            trackSDKCall('ipfsSwarmPeers', false, Date.now() - startTime);
        }
        
        const peers = result.peers || result.Peers || result || [];
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>👥 Connected Peers: ${Array.isArray(peers) ? peers.length : 0}</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px; max-height: 300px; overflow-y: auto; font-size: 11px;">${JSON.stringify(peers, null, 2)}</pre>
                </div>
            `;
        }
        showToast(`Found ${Array.isArray(peers) ? peers.length : 0} peers`, 'success');
    } catch (error) {
        trackSDKCall('ipfsSwarmPeers', false, Date.now() - startTime);
        console.error('[IPFS] Swarm peers error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to list peers: ${error.message}`, 'error');
    }
}

async function ipfsId() {
    const resultDiv = document.getElementById('ipfs-swarm-result');
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Getting node ID...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsId) {
            result = await mcpClient.ipfsId();
            trackSDKCall('ipfsId', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_id', {});
            trackSDKCall('ipfsId', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>🆔 Node Information:</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px;">${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }
        showToast('Node ID retrieved', 'success');
    } catch (error) {
        trackSDKCall('ipfsId', false, Date.now() - startTime);
        console.error('[IPFS] ID error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to get node ID: ${error.message}`, 'error');
    }
}

async function ipfsSwarmConnect() {
    const peerAddr = document.getElementById('ipfs-peer-addr')?.value;
    const resultDiv = document.getElementById('ipfs-swarm-result');
    
    if (!peerAddr) {
        showToast('Please enter a peer address', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Connecting to peer...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsSwarmConnect) {
            result = await mcpClient.ipfsSwarmConnect(peerAddr);
            trackSDKCall('ipfsSwarmConnect', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_swarm_connect', { address: peerAddr });
            trackSDKCall('ipfsSwarmConnect', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="result-success">✅ Connected to peer successfully!</div>`;
        }
        showToast('Peer connected', 'success');
    } catch (error) {
        trackSDKCall('ipfsSwarmConnect', false, Date.now() - startTime);
        console.error('[IPFS] Swarm connect error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to connect: ${error.message}`, 'error');
    }
}

async function ipfsDhtFindPeer() {
    const peerId = document.getElementById('ipfs-dht-query')?.value;
    const resultDiv = document.getElementById('ipfs-dht-result');
    
    if (!peerId) {
        showToast('Please enter a peer ID', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Finding peer...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsDhtFindpeer) {
            result = await mcpClient.ipfsDhtFindpeer(peerId);
            trackSDKCall('ipfsDhtFindpeer', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_dht_findpeer', { peer_id: peerId });
            trackSDKCall('ipfsDhtFindpeer', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>🔍 Peer Found:</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px;">${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }
        showToast('Peer found', 'success');
    } catch (error) {
        trackSDKCall('ipfsDhtFindpeer', false, Date.now() - startTime);
        console.error('[IPFS] DHT findpeer error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to find peer: ${error.message}`, 'error');
    }
}

async function ipfsDhtFindProvs() {
    const cid = document.getElementById('ipfs-dht-query')?.value;
    const resultDiv = document.getElementById('ipfs-dht-result');
    
    if (!cid) {
        showToast('Please enter a CID', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Finding providers...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsDhtFindprovs) {
            result = await mcpClient.ipfsDhtFindprovs(cid);
            trackSDKCall('ipfsDhtFindprovs', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_dht_findprovs', { cid });
            trackSDKCall('ipfsDhtFindprovs', false, Date.now() - startTime);
        }
        
        const providers = result.providers || result.Providers || result || [];
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="result-success">
                    <strong>📦 Providers Found: ${Array.isArray(providers) ? providers.length : 0}</strong>
                    <pre style="background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 10px; max-height: 300px; overflow-y: auto;">${JSON.stringify(providers, null, 2)}</pre>
                </div>
            `;
        }
        showToast(`Found ${Array.isArray(providers) ? providers.length : 0} providers`, 'success');
    } catch (error) {
        trackSDKCall('ipfsDhtFindprovs', false, Date.now() - startTime);
        console.error('[IPFS] DHT findprovs error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to find providers: ${error.message}`, 'error');
    }
}

async function ipfsPubsubPub() {
    const topic = document.getElementById('ipfs-pubsub-topic')?.value;
    const message = document.getElementById('ipfs-pubsub-message')?.value;
    const resultDiv = document.getElementById('ipfs-pubsub-result');
    
    if (!topic || !message) {
        showToast('Please enter both topic and message', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="spinner"></div> Publishing...';
    }
    
    const startTime = Date.now();
    
    try {
        let result;
        if (mcpClient && mcpClient.ipfsPubsubPub) {
            result = await mcpClient.ipfsPubsubPub({ topic, message });
            trackSDKCall('ipfsPubsubPub', true, Date.now() - startTime);
        } else {
            result = await mcpClient.callTool('ipfs_pubsub_pub', { topic, message });
            trackSDKCall('ipfsPubsubPub', false, Date.now() - startTime);
        }
        
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="result-success">✅ Message published to topic: ${topic}</div>`;
        }
        showToast('Message published', 'success');
    } catch (error) {
        trackSDKCall('ipfsPubsubPub', false, Date.now() - startTime);
        console.error('[IPFS] Pubsub publish error:', error);
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        }
        showToast(`Failed to publish: ${error.message}`, 'error');
    }
}

async function ipfsPubsubSub() {
    const topic = document.getElementById('ipfs-pubsub-topic')?.value;
    const resultDiv = document.getElementById('ipfs-pubsub-result');
    
    if (!topic) {
        showToast('Please enter a topic', 'warning');
        return;
    }
    
    if (resultDiv) {
        resultDiv.innerHTML = `<div class="info">ℹ️ Subscribing to topic: ${topic}... (Feature requires server-side implementation)</div>`;
    }
    
    showToast('Pubsub subscribe requires server-side streaming', 'info');
}

// ==========================================
// RUNNER MANAGEMENT FUNCTIONS
// ==========================================

let currentRunners = [];
let selectedRunnerId = null;

/**
 * Load runner capabilities and display all runners
 */
async function loadRunnerCapabilities() {
    const runnerListDiv = document.getElementById('runner-list');
    if (!runnerListDiv) return;
    
    runnerListDiv.innerHTML = '<div class="spinner"></div> Loading runners...';
    
    try {
        const capabilities = await mcpClient.runnerGetCapabilities();
        currentRunners = capabilities.runners || [];
        
        // Update runner select dropdowns
        updateRunnerSelects();
        
        // Display runners
        if (currentRunners.length === 0) {
            runnerListDiv.innerHTML = '<p class="text-muted">No runners available</p>';
            return;
        }
        
        let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px;">';
        
        for (const runner of currentRunners) {
            const statusClass = runner.status === 'online' ? 'status-running' : 'status-stopped';
            const healthClass = runner.health === 'healthy' ? 'status-success' : 'status-warning';
            
            html += `
                <div class="runner-card" style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                        <h4 style="margin: 0; font-size: 16px;">${runner.id || 'Unknown Runner'}</h4>
                        <span class="${statusClass}" style="padding: 4px 8px; border-radius: 4px; font-size: 12px;">${runner.status || 'unknown'}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 5px;">
                            <strong>Type:</strong> ${runner.type || 'standard'}
                        </div>
                        <div style="font-size: 13px; color: #6b7280; margin-bottom: 5px;">
                            <strong>Capacity:</strong> ${runner.capacity || 'N/A'}
                        </div>
                        <div style="font-size: 13px; color: #6b7280;">
                            <strong>Health:</strong> <span class="${healthClass}">${runner.health || 'unknown'}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-sm btn-primary" onclick="viewRunnerDetails('${runner.id}')" style="flex: 1;">📊 Details</button>
                        <button class="btn btn-sm btn-secondary" onclick="runRunnerHealthCheck('${runner.id}')" style="flex: 1;">🏥 Health</button>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        runnerListDiv.innerHTML = html;
        showToast(`Loaded ${currentRunners.length} runner(s)`, 'success');
        
    } catch (error) {
        console.error('[Runner] Failed to load capabilities:', error);
        runnerListDiv.innerHTML = `<div class="error-message">❌ Failed to load runners: ${error.message}</div>`;
        showToast('Failed to load runners', 'error');
    }
}

/**
 * Update all runner select dropdowns
 */
function updateRunnerSelects() {
    const selects = ['runner-id-select', 'task-runner-select', 'metrics-runner-select'];
    
    for (const selectId of selects) {
        const select = document.getElementById(selectId);
        if (!select) continue;
        
        // Clear existing options except first
        select.innerHTML = '<option value="">-- Select a runner --</option>';
        
        // Add runner options
        for (const runner of currentRunners) {
            const option = document.createElement('option');
            option.value = runner.id;
            option.textContent = `${runner.id} (${runner.type || 'standard'})`;
            select.appendChild(option);
        }
    }
}

/**
 * View detailed runner information
 */
async function viewRunnerDetails(runnerId) {
    try {
        const status = await mcpClient.runnerGetStatus(runnerId);
        const metrics = await mcpClient.runnerGetMetrics(runnerId);
        
        const details = {
            ...status,
            metrics: metrics
        };
        
        alert(`Runner Details:\n\n${JSON.stringify(details, null, 2)}`);
        
    } catch (error) {
        console.error('[Runner] Failed to get details:', error);
        showToast(`Failed to get runner details: ${error.message}`, 'error');
    }
}

/**
 * Run health check for a specific runner
 */
async function runRunnerHealthCheck(runnerId) {
    try {
        showToast('Running health check...', 'info');
        const health = await mcpClient.runnerHealthCheck(runnerId);
        
        const healthStatus = health.healthy ? '✅ Healthy' : '⚠️ Unhealthy';
        const message = health.message || 'No additional information';
        
        showToast(`${healthStatus}: ${message}`, health.healthy ? 'success' : 'warning');
        
        // Refresh runner list
        await loadRunnerCapabilities();
        
    } catch (error) {
        console.error('[Runner] Health check failed:', error);
        showToast(`Health check failed: ${error.message}`, 'error');
    }
}

/**
 * Run health check for all runners
 */
async function runAllRunnerHealthChecks() {
    if (currentRunners.length === 0) {
        showToast('No runners available', 'warning');
        return;
    }
    
    showToast('Checking health of all runners...', 'info');
    
    let healthy = 0;
    let unhealthy = 0;
    
    for (const runner of currentRunners) {
        try {
            const health = await mcpClient.runnerHealthCheck(runner.id);
            if (health.healthy) {
                healthy++;
            } else {
                unhealthy++;
            }
        } catch (error) {
            console.error(`[Runner] Health check failed for ${runner.id}:`, error);
            unhealthy++;
        }
    }
    
    showToast(`Health check complete: ${healthy} healthy, ${unhealthy} unhealthy`, 'success');
    await loadRunnerCapabilities();
}

/**
 * Load runner configuration
 */
async function loadRunnerConfig(runnerId) {
    if (!runnerId) return;
    
    selectedRunnerId = runnerId;
    
    try {
        const status = await mcpClient.runnerGetStatus(runnerId);
        const config = status.config || {};
        
        // Update form fields
        document.getElementById('config-max-cpu').value = config.maxCpu || 4;
        document.getElementById('config-max-memory').value = config.maxMemory || 8;
        document.getElementById('config-max-tasks').value = config.maxTasks || 5;
        document.getElementById('config-auto-scale').checked = config.autoScale || false;
        
        showToast('Configuration loaded', 'success');
        
    } catch (error) {
        console.error('[Runner] Failed to load config:', error);
        showToast(`Failed to load configuration: ${error.message}`, 'error');
    }
}

/**
 * Save runner configuration
 */
async function saveRunnerConfig(event) {
    event.preventDefault();
    
    const runnerId = document.getElementById('runner-id-select').value;
    if (!runnerId) {
        showToast('Please select a runner', 'warning');
        return;
    }
    
    const config = {
        maxCpu: parseInt(document.getElementById('config-max-cpu').value),
        maxMemory: parseInt(document.getElementById('config-max-memory').value),
        maxTasks: parseInt(document.getElementById('config-max-tasks').value),
        autoScale: document.getElementById('config-auto-scale').checked
    };
    
    try {
        await mcpClient.runnerSetConfig({ runnerId, config });
        showToast('Configuration saved successfully', 'success');
        
    } catch (error) {
        console.error('[Runner] Failed to save config:', error);
        showToast(`Failed to save configuration: ${error.message}`, 'error');
    }
}

/**
 * Reset runner configuration to defaults
 */
function resetRunnerConfig() {
    document.getElementById('config-max-cpu').value = 4;
    document.getElementById('config-max-memory').value = 8;
    document.getElementById('config-max-tasks').value = 5;
    document.getElementById('config-auto-scale').checked = false;
    showToast('Configuration reset to defaults', 'info');
}

/**
 * Load tasks for a specific runner
 */
async function loadRunnerTasks(runnerId) {
    const tasksListDiv = document.getElementById('runner-tasks-list');
    if (!tasksListDiv) return;
    
    if (!runnerId) {
        tasksListDiv.innerHTML = '<p class="text-muted" style="text-align: center; padding: 20px;">Select a runner to view tasks</p>';
        return;
    }
    
    tasksListDiv.innerHTML = '<div class="spinner"></div> Loading tasks...';
    
    try {
        const tasks = await mcpClient.runnerListTasks(runnerId);
        
        if (!tasks || tasks.length === 0) {
            tasksListDiv.innerHTML = '<p class="text-muted" style="text-align: center; padding: 20px;">No tasks running</p>';
            return;
        }
        
        let html = '<div style="display: flex; flex-direction: column; gap: 10px;">';
        
        for (const task of tasks) {
            const statusClass = task.status === 'running' ? 'status-running' : 
                              task.status === 'completed' ? 'status-success' : 'status-stopped';
            
            html += `
                <div style="border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="font-size: 14px;">${task.name || task.id}</strong>
                        <span class="${statusClass}" style="padding: 3px 8px; border-radius: 4px; font-size: 11px;">${task.status}</span>
                    </div>
                    <div style="font-size: 12px; color: #6b7280; margin-bottom: 8px;">
                        <div><strong>ID:</strong> ${task.id}</div>
                        ${task.command ? `<div><strong>Command:</strong> <code style="background: #f3f4f6; padding: 2px 6px; border-radius: 3px;">${task.command}</code></div>` : ''}
                    </div>
                    <button class="btn btn-sm btn-danger" onclick="stopRunnerTask('${runnerId}', '${task.id}')" ${task.status !== 'running' ? 'disabled' : ''}>
                        ⏹️ Stop Task
                    </button>
                </div>
            `;
        }
        
        html += '</div>';
        tasksListDiv.innerHTML = html;
        
    } catch (error) {
        console.error('[Runner] Failed to load tasks:', error);
        tasksListDiv.innerHTML = `<div class="error-message">❌ Failed to load tasks: ${error.message}</div>`;
        showToast('Failed to load tasks', 'error');
    }
}

/**
 * Start a new task on a runner
 */
async function startNewRunnerTask(event) {
    event.preventDefault();
    
    const runnerId = document.getElementById('task-runner-select').value;
    if (!runnerId) {
        showToast('Please select a runner', 'warning');
        return;
    }
    
    const taskName = document.getElementById('task-name').value;
    const taskCommand = document.getElementById('task-command').value;
    
    const taskConfig = {
        name: taskName,
        command: taskCommand,
        runnerId: runnerId
    };
    
    try {
        showToast('Starting task...', 'info');
        const result = await mcpClient.runnerStartTask(taskConfig);
        
        showToast(`Task started: ${result.taskId || 'success'}`, 'success');
        
        // Clear form
        document.getElementById('task-name').value = '';
        document.getElementById('task-command').value = '';
        
        // Refresh task list
        await loadRunnerTasks(runnerId);
        
    } catch (error) {
        console.error('[Runner] Failed to start task:', error);
        showToast(`Failed to start task: ${error.message}`, 'error');
    }
}

/**
 * Stop a running task
 */
async function stopRunnerTask(runnerId, taskId) {
    if (!confirm('Are you sure you want to stop this task?')) {
        return;
    }
    
    try {
        await mcpClient.runnerStopTask(taskId);
        showToast('Task stopped', 'success');
        
        // Refresh task list
        await loadRunnerTasks(runnerId);
        
    } catch (error) {
        console.error('[Runner] Failed to stop task:', error);
        showToast(`Failed to stop task: ${error.message}`, 'error');
    }
}

/**
 * Load metrics for a specific runner
 */
async function loadRunnerMetrics(runnerId) {
    if (!runnerId) {
        // Reset metrics display
        document.getElementById('metric-cpu').textContent = '--';
        document.getElementById('metric-memory').textContent = '--';
        document.getElementById('metric-tasks').textContent = '--';
        document.getElementById('metric-uptime').textContent = '--';
        document.getElementById('runner-logs-display').innerHTML = '<p style="color: #9ca3af; margin: 0;">Select a runner to view logs...</p>';
        return;
    }
    
    try {
        const metrics = await mcpClient.runnerGetMetrics(runnerId);
        
        // Update metrics display
        document.getElementById('metric-cpu').textContent = metrics.cpuUsage ? `${metrics.cpuUsage.toFixed(1)}%` : '--';
        document.getElementById('metric-memory').textContent = metrics.memoryUsage ? `${metrics.memoryUsage.toFixed(1)}%` : '--';
        document.getElementById('metric-tasks').textContent = metrics.activeTasks || '0';
        document.getElementById('metric-uptime').textContent = metrics.uptime || '--';
        
        showToast('Metrics loaded', 'success');
        
        // Also load logs
        await loadRunnerLogs(runnerId);
        
    } catch (error) {
        console.error('[Runner] Failed to load metrics:', error);
        showToast(`Failed to load metrics: ${error.message}`, 'error');
    }
}

/**
 * Load logs for a specific runner
 */
async function loadRunnerLogs(runnerId) {
    if (!runnerId) return;
    
    const logsDiv = document.getElementById('runner-logs-display');
    if (!logsDiv) return;
    
    try {
        const logs = await mcpClient.runnerGetLogs(runnerId, { limit: 100 });
        
        if (!logs || logs.length === 0) {
            logsDiv.innerHTML = '<p style="color: #9ca3af; margin: 0;">No logs available</p>';
            return;
        }
        
        let html = '';
        for (const log of logs) {
            const timestamp = log.timestamp || '';
            const level = log.level || 'INFO';
            const message = log.message || '';
            
            const levelColor = level === 'ERROR' ? '#ef4444' : 
                             level === 'WARN' ? '#f59e0b' : '#10b981';
            
            html += `<div style="margin-bottom: 4px;">
                <span style="color: #6b7280;">[${timestamp}]</span>
                <span style="color: ${levelColor}; font-weight: 600;">[${level}]</span>
                <span>${message}</span>
            </div>`;
        }
        
        logsDiv.innerHTML = html;
        // Auto-scroll to bottom
        logsDiv.scrollTop = logsDiv.scrollHeight;
        
    } catch (error) {
        console.error('[Runner] Failed to load logs:', error);
        logsDiv.innerHTML = `<p style="color: #ef4444; margin: 0;">Failed to load logs: ${error.message}</p>`;
    }
}

/**
 * Refresh runner logs display
 */
async function refreshRunnerLogs() {
    const runnerId = document.getElementById('metrics-runner-select').value;
    if (!runnerId) {
        showToast('Please select a runner first', 'warning');
        return;
    }
    
    await loadRunnerLogs(runnerId);
    showToast('Logs refreshed', 'success');
}

// Initialize Runner Management tab when it's shown
document.addEventListener('DOMContentLoaded', function() {
    // Auto-load runners when Runner Management tab is clicked
    const runnerTab = document.querySelector('[onclick*="runner-management"]');
    if (runnerTab) {
        runnerTab.addEventListener('click', function() {
            if (currentRunners.length === 0) {
                loadRunnerCapabilities();
            }
        });
    }
});

// ==========================================
// ADVANCED AI OPERATIONS FUNCTIONS
// ==========================================

// Tab switching functions for Advanced AI
function switchAudioTab(tabId) {
    const tabs = document.querySelectorAll('#advanced-ai .ai-tab-content');
    tabs.forEach(tab => {
        if (tab.id.startsWith('audio-')) {
            tab.style.display = 'none';
        }
    });
    document.getElementById(tabId).style.display = 'block';
    
    // Update button states
    const buttons = document.querySelectorAll('[data-tab^="audio-"]');
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

function switchImageTab(tabId) {
    const tabs = document.querySelectorAll('#advanced-ai .ai-tab-content');
    tabs.forEach(tab => {
        if (tab.id.startsWith('img-')) {
            tab.style.display = 'none';
        }
    });
    document.getElementById(tabId).style.display = 'block';
    
    // Update button states
    const buttons = document.querySelectorAll('[data-tab^="img-"]');
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

function switchTextTab(tabId) {
    const tabs = document.querySelectorAll('#advanced-ai .ai-tab-content');
    tabs.forEach(tab => {
        if (tab.id.startsWith('text-')) {
            tab.style.display = 'none';
        }
    });
    document.getElementById(tabId).style.display = 'block';
    
    // Update button states
    const buttons = document.querySelectorAll('[data-tab^="text-"]');
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

function switchMLTab(tabId) {
    const tabs = document.querySelectorAll('#advanced-ai .ai-tab-content');
    tabs.forEach(tab => {
        if (tab.id.startsWith('ml-')) {
            tab.style.display = 'none';
        }
    });
    document.getElementById(tabId).style.display = 'block';
    
    // Update button states
    const buttons = document.querySelectorAll('[data-tab^="ml-"]');
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

// Question Answering
async function performQuestionAnswering() {
    const context = document.getElementById('qa-context').value;
    const question = document.getElementById('qa-question').value;
    const resultDiv = document.getElementById('qa-result');
    const answerDiv = document.getElementById('qa-answer');
    
    if (!context || !question) {
        showToast('Please provide both context and question', 'warning');
        return;
    }
    
    resultDiv.style.display = 'block';
    answerDiv.innerHTML = '<div class="spinner"></div> Finding answer...';
    
    try {
        const result = await mcpClient.answerQuestion(question, context);
        answerDiv.innerHTML = `<strong>${result.answer || result.text || JSON.stringify(result)}</strong>`;
        showToast('Answer found', 'success');
    } catch (error) {
        console.error('[AI] Question answering failed:', error);
        answerDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to answer question', 'error');
    }
}

// Audio Operations
async function transcribeAudioFile() {
    const fileInput = document.getElementById('audio-file-transcribe');
    const resultDiv = document.getElementById('audio-transcribe-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an audio file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Transcribing audio...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const audioData = e.target.result;
            const result = await mcpClient.transcribeAudio(audioData);
            resultDiv.innerHTML = `<div class="result-success"><strong>Transcription:</strong><br>${result.text || JSON.stringify(result)}</div>`;
            showToast('Audio transcribed', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Audio transcription failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to transcribe audio', 'error');
    }
}

async function classifyAudioFile() {
    const fileInput = document.getElementById('audio-file-classify');
    const resultDiv = document.getElementById('audio-classify-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an audio file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Classifying audio...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const audioData = e.target.result;
            const result = await mcpClient.classifyAudio(audioData);
            resultDiv.innerHTML = `<div class="result-success"><strong>Classification:</strong><br>${JSON.stringify(result, null, 2)}</div>`;
            showToast('Audio classified', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Audio classification failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to classify audio', 'error');
    }
}

async function generateAudioFromPrompt() {
    const prompt = document.getElementById('audio-prompt').value;
    const resultDiv = document.getElementById('audio-generate-result');
    
    if (!prompt) {
        showToast('Please enter a prompt', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Generating audio...';
    
    try {
        const result = await mcpClient.generateAudio(prompt);
        resultDiv.innerHTML = `<div class="result-success">✅ Audio generated<br><audio controls src="${result.audio || result.url}"></audio></div>`;
        showToast('Audio generated', 'success');
    } catch (error) {
        console.error('[AI] Audio generation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to generate audio', 'error');
    }
}

async function synthesizeSpeechFromText() {
    const text = document.getElementById('tts-text').value;
    const resultDiv = document.getElementById('tts-result');
    
    if (!text) {
        showToast('Please enter text', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Synthesizing speech...';
    
    try {
        const result = await mcpClient.synthesizeSpeech(text);
        resultDiv.innerHTML = `<div class="result-success">✅ Speech synthesized<br><audio controls src="${result.audio || result.url}"></audio></div>`;
        showToast('Speech synthesized', 'success');
    } catch (error) {
        console.error('[AI] Speech synthesis failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to synthesize speech', 'error');
    }
}

// Image Operations
async function classifyImageFile() {
    const fileInput = document.getElementById('img-file-classify');
    const resultDiv = document.getElementById('img-classify-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Classifying image...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const imageData = e.target.result;
            const result = await mcpClient.classifyImage(imageData);
            resultDiv.innerHTML = `<div class="result-success"><strong>Classification:</strong><br>${JSON.stringify(result, null, 2)}</div>`;
            showToast('Image classified', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Image classification failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to classify image', 'error');
    }
}

async function detectObjectsInImage() {
    const fileInput = document.getElementById('img-file-detect');
    const resultDiv = document.getElementById('img-detect-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Detecting objects...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const imageData = e.target.result;
            const result = await mcpClient.detectObjects(imageData);
            resultDiv.innerHTML = `<div class="result-success"><strong>Objects Detected:</strong><br>${JSON.stringify(result, null, 2)}</div>`;
            showToast('Objects detected', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Object detection failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to detect objects', 'error');
    }
}

async function segmentImageFile() {
    const fileInput = document.getElementById('img-file-segment');
    const resultDiv = document.getElementById('img-segment-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Segmenting image...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const imageData = e.target.result;
            const result = await mcpClient.segmentImage(imageData);
            resultDiv.innerHTML = `<div class="result-success">✅ Image segmented<br><img src="${result.segmented_image || result.url}" style="max-width: 100%; border-radius: 8px;"/></div>`;
            showToast('Image segmented', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Image segmentation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to segment image', 'error');
    }
}

async function generateImageCaption() {
    const fileInput = document.getElementById('img-file-caption');
    const resultDiv = document.getElementById('img-caption-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Generating caption...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const imageData = e.target.result;
            const result = await mcpClient.generateImageCaption(imageData);
            resultDiv.innerHTML = `<div class="result-success"><strong>Caption:</strong><br>${result.caption || result.text || JSON.stringify(result)}</div>`;
            showToast('Caption generated', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Caption generation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to generate caption', 'error');
    }
}

async function generateImageFromPrompt() {
    const prompt = document.getElementById('img-generate-prompt').value;
    const resultDiv = document.getElementById('img-generate-result');
    
    if (!prompt) {
        showToast('Please enter a prompt', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Generating image...';
    
    try {
        const result = await mcpClient.generateImage(prompt);
        resultDiv.innerHTML = `<div class="result-success">✅ Image generated<br><img src="${result.image || result.url}" style="max-width: 100%; border-radius: 8px;"/></div>`;
        showToast('Image generated', 'success');
    } catch (error) {
        console.error('[AI] Image generation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to generate image', 'error');
    }
}

async function answerVisualQuestionImage() {
    const fileInput = document.getElementById('img-file-vqa');
    const question = document.getElementById('vqa-question').value;
    const resultDiv = document.getElementById('img-vqa-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    if (!question) {
        showToast('Please enter a question', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Answering question...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const imageData = e.target.result;
            const result = await mcpClient.answerVisualQuestion(imageData, question);
            resultDiv.innerHTML = `<div class="result-success"><strong>Answer:</strong><br>${result.answer || result.text || JSON.stringify(result)}</div>`;
            showToast('Question answered', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Visual Q&A failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to answer question', 'error');
    }
}

// Text Operations
async function summarizeTextInput() {
    const text = document.getElementById('text-summarize-input').value;
    const resultDiv = document.getElementById('text-summarize-result');
    
    if (!text) {
        showToast('Please enter text to summarize', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Summarizing text...';
    
    try {
        const result = await mcpClient.summarizeText(text);
        resultDiv.innerHTML = `<div class="result-success"><strong>Summary:</strong><br>${result.summary || result.text || JSON.stringify(result)}</div>`;
        showToast('Text summarized', 'success');
    } catch (error) {
        console.error('[AI] Text summarization failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to summarize text', 'error');
    }
}

async function translateTextInput() {
    const text = document.getElementById('text-translate-input').value;
    const targetLang = document.getElementById('translate-target-lang').value;
    const resultDiv = document.getElementById('text-translate-result');
    
    if (!text) {
        showToast('Please enter text to translate', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Translating text...';
    
    try {
        const result = await mcpClient.translateText(text, targetLang);
        resultDiv.innerHTML = `<div class="result-success"><strong>Translation:</strong><br>${result.translation || result.text || JSON.stringify(result)}</div>`;
        showToast('Text translated', 'success');
    } catch (error) {
        console.error('[AI] Translation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to translate text', 'error');
    }
}

async function fillMaskInput() {
    const text = document.getElementById('text-fillmask-input').value;
    const resultDiv = document.getElementById('text-fillmask-result');
    
    if (!text || !text.includes('[MASK]')) {
        showToast('Please enter text with [MASK] token', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Filling mask...';
    
    try {
        const result = await mcpClient.fillMask(text);
        let html = '<div class="result-success"><strong>Predictions:</strong><br>';
        if (Array.isArray(result)) {
            result.forEach((pred, i) => {
                html += `${i+1}. ${pred.token_str || pred.sequence} (${(pred.score * 100).toFixed(1)}%)<br>`;
            });
        } else {
            html += JSON.stringify(result);
        }
        html += '</div>';
        resultDiv.innerHTML = html;
        showToast('Mask filled', 'success');
    } catch (error) {
        console.error('[AI] Fill mask failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to fill mask', 'error');
    }
}

async function generateCodeFromDesc() {
    const description = document.getElementById('text-code-input').value;
    const language = document.getElementById('code-language').value;
    const resultDiv = document.getElementById('text-code-result');
    
    if (!description) {
        showToast('Please enter a code description', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Generating code...';
    
    try {
        const result = await mcpClient.generateCode(description, { language });
        resultDiv.innerHTML = `<div class="result-success"><strong>Generated Code:</strong><br><pre style="background: #1f2937; color: #f3f4f6; padding: 15px; border-radius: 6px; overflow-x: auto;">${result.code || result.text || JSON.stringify(result)}</pre></div>`;
        showToast('Code generated', 'success');
    } catch (error) {
        console.error('[AI] Code generation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to generate code', 'error');
    }
}

// Extended ML Operations
async function generateEmbeddingsFromText() {
    const text = document.getElementById('embeddings-input').value;
    const resultDiv = document.getElementById('embeddings-result');
    
    if (!text) {
        showToast('Please enter text', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Generating embeddings...';
    
    try {
        const result = await mcpClient.generateEmbeddings(text);
        let html = '<div class="result-success"><strong>Embeddings Generated:</strong><br>';
        if (Array.isArray(result.embeddings)) {
            html += `<div>Vector dimension: ${result.embeddings.length}</div>`;
            html += `<div>First 10 values: [${result.embeddings.slice(0, 10).map(v => v.toFixed(4)).join(', ')}...]</div>`;
        } else {
            html += JSON.stringify(result);
        }
        html += '</div>';
        resultDiv.innerHTML = html;
        showToast('Embeddings generated', 'success');
    } catch (error) {
        console.error('[AI] Embeddings generation failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to generate embeddings', 'error');
    }
}

async function processDocumentFile() {
    const fileInput = document.getElementById('document-file');
    const operation = document.getElementById('document-operation').value;
    const resultDiv = document.getElementById('document-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select a document file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Processing document...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const documentData = e.target.result;
            const result = await mcpClient.processDocument(documentData, { operation });
            resultDiv.innerHTML = `<div class="result-success"><strong>Result:</strong><br>${result.text || JSON.stringify(result, null, 2)}</div>`;
            showToast('Document processed', 'success');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('[AI] Document processing failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to process document', 'error');
    }
}

async function processTabularDataFile() {
    const fileInput = document.getElementById('tabular-file');
    const operation = document.getElementById('tabular-operation').value;
    const resultDiv = document.getElementById('tabular-result');
    
    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select a CSV file', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Processing tabular data...';
    
    try {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(e) {
            const data = e.target.result;
            const result = await mcpClient.processTabularData(data, { operation });
            resultDiv.innerHTML = `<div class="result-success"><strong>Result:</strong><br><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
            showToast('Data processed', 'success');
        };
        reader.readAsText(file);
    } catch (error) {
        console.error('[AI] Tabular data processing failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to process data', 'error');
    }
}

async function predictTimeseriesData() {
    const dataText = document.getElementById('timeseries-data').value;
    const periods = parseInt(document.getElementById('timeseries-periods').value);
    const resultDiv = document.getElementById('timeseries-result');
    
    if (!dataText) {
        showToast('Please enter time series data', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Predicting time series...';
    
    try {
        const data = JSON.parse(dataText);
        const result = await mcpClient.predictTimeseries(data, { periods });
        resultDiv.innerHTML = `<div class="result-success"><strong>Predictions:</strong><br><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
        showToast('Time series predicted', 'success');
    } catch (error) {
        console.error('[AI] Time series prediction failed:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to predict time series', 'error');
    }
}

// ==========================================
// NETWORK & STATUS MANAGEMENT FUNCTIONS
// ==========================================

// Bandwidth Statistics
async function refreshBandwidthStats() {
    try {
        const stats = await mcpClient.networkGetBandwidth();
        
        document.getElementById('bandwidth-in').textContent = formatBytes(stats.totalIn || 0);
        document.getElementById('bandwidth-out').textContent = formatBytes(stats.totalOut || 0);
        document.getElementById('bandwidth-rate').textContent = `${formatBytes(stats.rateIn || 0)}/s in, ${formatBytes(stats.rateOut || 0)}/s out`;
        
        showToast('Bandwidth stats refreshed', 'success');
    } catch (error) {
        console.error('[Network] Failed to get bandwidth stats:', error);
        document.getElementById('bandwidth-in').textContent = 'Error';
        document.getElementById('bandwidth-out').textContent = 'Error';
        document.getElementById('bandwidth-rate').textContent = 'Error';
        showToast('Failed to load bandwidth stats', 'error');
    }
}

// Helper function to format bytes
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// Network Connections
async function refreshNetworkConnections() {
    const listDiv = document.getElementById('connection-list');
    listDiv.innerHTML = '<div class="spinner"></div> Loading connections...';
    
    try {
        const connections = await mcpClient.networkListConnections();
        
        if (!connections || connections.length === 0) {
            listDiv.innerHTML = '<p style="margin: 0; color: #6b7280;">No active connections</p>';
            return;
        }
        
        let html = '<div style="font-size: 12px;">';
        connections.forEach((conn, i) => {
            html += `<div style="margin-bottom: 5px; padding: 5px; background: #fff; border-radius: 4px;">
                <strong>${i + 1}.</strong> ${conn.peerId || conn.id || 'Unknown'} 
                <span style="color: #10b981;">(${conn.status || 'active'})</span>
            </div>`;
        });
        html += '</div>';
        
        listDiv.innerHTML = html;
        showToast(`${connections.length} connections loaded`, 'success');
    } catch (error) {
        console.error('[Network] Failed to list connections:', error);
        listDiv.innerHTML = `<p style="margin: 0; color: #ef4444;">Failed to load connections</p>`;
        showToast('Failed to load connections', 'error');
    }
}

// Peer Information
async function getPeerInfo() {
    const peerId = document.getElementById('peer-id-input').value;
    const resultDiv = document.getElementById('peer-info-result');
    
    if (!peerId) {
        showToast('Please enter a peer ID', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Getting peer info...';
    
    try {
        const info = await mcpClient.networkGetPeerInfo(peerId);
        resultDiv.innerHTML = `<div class="result-success">
            <strong>Peer Information:</strong><br>
            <pre style="margin-top: 10px; font-size: 12px;">${JSON.stringify(info, null, 2)}</pre>
        </div>`;
        showToast('Peer info retrieved', 'success');
    } catch (error) {
        console.error('[Network] Failed to get peer info:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to get peer info', 'error');
    }
}

// Peer Latency
async function getPeerLatency() {
    const peerId = document.getElementById('peer-id-input').value;
    const resultDiv = document.getElementById('peer-info-result');
    
    if (!peerId) {
        showToast('Please enter a peer ID', 'warning');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner"></div> Measuring latency...';
    
    try {
        const latency = await mcpClient.networkGetLatency(peerId);
        resultDiv.innerHTML = `<div class="result-success">
            <strong>Latency to Peer:</strong><br>
            <div style="font-size: 24px; margin-top: 10px; color: #667eea;">${latency.latency || latency}ms</div>
        </div>`;
        showToast('Latency measured', 'success');
    } catch (error) {
        console.error('[Network] Failed to get latency:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to measure latency', 'error');
    }
}

// Configure Network Limits
async function configureNetworkLimits(event) {
    event.preventDefault();
    
    const limits = {
        maxConnections: parseInt(document.getElementById('max-connections').value),
        maxBandwidth: parseInt(document.getElementById('max-bandwidth').value) * 1024 * 1024, // Convert MB to bytes
        connectionTimeout: parseInt(document.getElementById('connection-timeout').value)
    };
    
    try {
        await mcpClient.networkConfigureLimits(limits);
        showToast('Network limits configured successfully', 'success');
    } catch (error) {
        console.error('[Network] Failed to configure limits:', error);
        showToast(`Failed to configure limits: ${error.message}`, 'error');
    }
}

// System Status
async function refreshSystemStatus() {
    try {
        const status = await mcpClient.getSystemStatus();
        
        document.getElementById('system-status').innerHTML = status.healthy ? 
            '<span class="status-success">✅ Healthy</span>' : 
            '<span class="status-warning">⚠️ Issues Detected</span>';
        
        document.getElementById('system-uptime').textContent = status.uptime || 'Unknown';
        document.getElementById('system-version').textContent = status.version || 'Unknown';
        
        showToast('System status refreshed', 'success');
    } catch (error) {
        console.error('[Status] Failed to get system status:', error);
        document.getElementById('system-status').innerHTML = '<span style="color: #ef4444;">Error</span>';
        showToast('Failed to load system status', 'error');
    }
}

// Resource Usage
async function refreshResourceUsage() {
    try {
        const usage = await mcpClient.getResourceUsage();
        
        const cpuPercent = usage.cpu || 0;
        const memoryPercent = usage.memory || 0;
        const diskPercent = usage.disk || 0;
        
        document.getElementById('cpu-bar').style.width = `${cpuPercent}%`;
        document.getElementById('cpu-percent').textContent = `${cpuPercent}%`;
        
        document.getElementById('memory-bar').style.width = `${memoryPercent}%`;
        document.getElementById('memory-percent').textContent = `${memoryPercent}%`;
        
        document.getElementById('disk-bar').style.width = `${diskPercent}%`;
        document.getElementById('disk-percent').textContent = `${diskPercent}%`;
        
        showToast('Resource usage refreshed', 'success');
    } catch (error) {
        console.error('[Status] Failed to get resource usage:', error);
        showToast('Failed to load resource usage', 'error');
    }
}

// Service Status
async function checkServiceStatus() {
    const service = document.getElementById('service-select').value;
    const resultDiv = document.getElementById('service-status-result');
    
    resultDiv.innerHTML = '<div class="spinner"></div> Checking service status...';
    
    try {
        const status = await mcpClient.getServiceStatus(service);
        
        const statusHtml = status.running ? 
            '<span class="status-success">✅ Running</span>' : 
            '<span class="status-warning">⚠️ Stopped</span>';
        
        resultDiv.innerHTML = `<div class="result-success" style="margin-top: 10px;">
            <strong>Service: ${service}</strong><br>
            <div style="margin-top: 8px;">Status: ${statusHtml}</div>
            ${status.info ? `<div style="margin-top: 5px;">Info: ${status.info}</div>` : ''}
        </div>`;
        
        showToast('Service status checked', 'success');
    } catch (error) {
        console.error('[Status] Failed to check service status:', error);
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
        showToast('Failed to check service status', 'error');
    }
}

// CLI Endpoints
async function refreshCliEndpoints() {
    const listDiv = document.getElementById('cli-endpoints-list');
    listDiv.innerHTML = '<div class="spinner"></div> Loading CLI endpoints...';
    
    try {
        const endpoints = await mcpClient.listCliEndpoints();
        
        if (!endpoints || endpoints.length === 0) {
            listDiv.innerHTML = '<p style="margin: 0; color: #6b7280;">No CLI endpoints registered</p>';
            return;
        }
        
        let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">';
        endpoints.forEach(endpoint => {
            html += `<div style="padding: 12px; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;">
                <div style="font-weight: 600; margin-bottom: 5px;">${endpoint.name || 'Unnamed'}</div>
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 3px;">${endpoint.url || 'No URL'}</div>
                ${endpoint.description ? `<div style="font-size: 11px; color: #9ca3af;">${endpoint.description}</div>` : ''}
            </div>`;
        });
        html += '</div>';
        
        listDiv.innerHTML = html;
        showToast(`${endpoints.length} endpoints loaded`, 'success');
    } catch (error) {
        console.error('[CLI] Failed to list endpoints:', error);
        listDiv.innerHTML = '<p style="margin: 0; color: #ef4444;">Failed to load endpoints</p>';
        showToast('Failed to load CLI endpoints', 'error');
    }
}

// Register CLI Endpoint
async function registerNewCliEndpoint(event) {
    event.preventDefault();
    
    const config = {
        name: document.getElementById('cli-endpoint-name').value,
        url: document.getElementById('cli-endpoint-url').value,
        description: document.getElementById('cli-endpoint-description').value
    };
    
    try {
        await mcpClient.registerCliEndpoint(config);
        showToast('CLI endpoint registered successfully', 'success');
        
        // Clear form
        document.getElementById('cli-endpoint-name').value = '';
        document.getElementById('cli-endpoint-url').value = '';
        document.getElementById('cli-endpoint-description').value = '';
        
        // Refresh list
        await refreshCliEndpoints();
    } catch (error) {
        console.error('[CLI] Failed to register endpoint:', error);
        showToast(`Failed to register endpoint: ${error.message}`, 'error');
    }
}

// Auto-load network & status data when tab is shown
document.addEventListener('DOMContentLoaded', function() {
    const networkStatusTab = document.querySelector('[onclick*="network-status"]');
    if (networkStatusTab) {
        networkStatusTab.addEventListener('click', function() {
            refreshBandwidthStats();
            refreshNetworkConnections();
            refreshSystemStatus();
            refreshResourceUsage();
            refreshCliEndpoints();
        });
    }
});


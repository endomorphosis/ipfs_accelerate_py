/**
 * Kitchen Sink AI Model Testing Interface - SDK Edition
 * Refactored to use only the MCP JavaScript SDK with JSON-RPC
 */

class KitchenSinkSDKApp {
    constructor() {
        this.mcpClient = new MCPClient('/jsonrpc');
        this.currentResults = {};
        this.models = [];
        this.notifications = [];
        this.serverInfo = null;
        this.init();
    }

    async init() {
        console.log('Initializing Kitchen Sink AI Testing Interface (SDK Edition)...');
        
        // Setup UI components
        this.setupNotificationSystem();
        this.setupKeyboardShortcuts();
        
        // Check server connection
        await this.checkServerConnection();
        
        // Initialize components that depend on server
        if (this.serverInfo) {
            this.setupAutocomplete();
            this.setupFormHandlers();
            this.setupRangeInputs();
            this.setupModelManager();
            this.setupHuggingFaceBrowser();
            
            // Load initial data
            await this.loadModels();
            await this.loadServerInfo();
        }
        
        console.log('Kitchen Sink SDK App initialized');
    }

    // ============================================
    // SERVER CONNECTION & STATUS
    // ============================================

    async checkServerConnection() {
        try {
            this.showNotification('Connecting to MCP server...', 'info');
            
            const isAvailable = await this.mcpClient.waitForServer(5, 2000);
            if (isAvailable) {
                this.serverInfo = await this.mcpClient.getServerInfo();
                this.showNotification('Connected to MCP server successfully!', 'success');
                this.updateServerStatus(true);
            } else {
                throw new Error('Server not available');
            }
        } catch (error) {
            console.error('Failed to connect to MCP server:', error);
            this.showNotification('Failed to connect to MCP server. Please check if the server is running.', 'error');
            this.updateServerStatus(false);
        }
    }

    updateServerStatus(connected) {
        const statusElements = document.querySelectorAll('.server-status');
        statusElements.forEach(element => {
            element.className = `server-status ${connected ? 'connected' : 'disconnected'}`;
            element.textContent = connected ? 'Connected' : 'Disconnected';
        });
    }

    // ============================================
    // NOTIFICATION SYSTEM
    // ============================================

    setupNotificationSystem() {
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                max-width: 400px;
            `;
            document.body.appendChild(container);
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} fade-in`;
        notification.style.cssText = `
            margin-bottom: 10px;
            animation: slideIn 0.3s ease-out;
        `;
        
        const iconClass = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        }[type] || 'fas fa-info-circle';
        
        notification.innerHTML = `
            <i class="${iconClass}"></i>
            <span>${message}</span>
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;
        
        document.getElementById('notification-container').appendChild(notification);
        
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
    }

    // ============================================
    // MODEL MANAGEMENT VIA SDK
    // ============================================

    async loadModels() {
        try {
            const response = await this.mcpClient.listModels();
            this.models = response.models || [];
            console.log(`Loaded ${this.models.length} models from MCP server`);
            this.updateModelCounts();
        } catch (error) {
            console.error('Error loading models:', error);
            this.showNotification('Error loading models: ' + error.message, 'error');
        }
    }

    async loadServerInfo() {
        try {
            this.serverInfo = await this.mcpClient.getServerInfo();
            this.updateServerInfoDisplay();
        } catch (error) {
            console.error('Error loading server info:', error);
        }
    }

    updateServerInfoDisplay() {
        const serverInfoElement = document.getElementById('server-info');
        if (serverInfoElement && this.serverInfo) {
            serverInfoElement.innerHTML = `
                <strong>Server:</strong> ${this.serverInfo.name}<br>
                <strong>Version:</strong> ${this.serverInfo.version}<br>
                <strong>Methods:</strong> ${this.serverInfo.methods_count}<br>
                <strong>Status:</strong> <span class="server-status connected">Connected</span>
            `;
        }
    }

    updateModelCounts() {
        const countElements = document.querySelectorAll('.model-count');
        countElements.forEach(element => {
            element.textContent = this.models.length;
        });
    }

    // ============================================
    // AUTOCOMPLETE SETUP
    // ============================================

    setupAutocomplete() {
        const modelInputs = document.querySelectorAll('.model-autocomplete input');
        modelInputs.forEach(input => {
            this.setupSingleAutocomplete(input);
        });
    }

    setupSingleAutocomplete(input) {
        // Remove existing autocomplete if any
        if (input.autocompleteList) {
            input.autocompleteList.remove();
        }

        let currentFocus = -1;
        
        input.addEventListener('input', (e) => {
            const value = e.target.value;
            this.closeAllLists();
            
            if (!value) return;
            
            currentFocus = -1;
            const listContainer = document.createElement('div');
            listContainer.className = 'autocomplete-items';
            listContainer.style.cssText = `
                position: absolute;
                border: 1px solid #d4edda;
                border-bottom: none;
                border-top: none;
                z-index: 99;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                max-height: 200px;
                overflow-y: auto;
            `;
            
            input.parentNode.appendChild(listContainer);
            input.autocompleteList = listContainer;
            
            // Filter models based on input
            const filteredModels = this.models.filter(model => {
                const modelId = model.model_id || model.id || 'unknown';
                return modelId.toLowerCase().includes(value.toLowerCase());
            }).slice(0, 10); // Limit to 10 results
            
            filteredModels.forEach((model, index) => {
                const modelId = model.model_id || model.id || 'unknown';
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                item.style.cssText = `
                    padding: 10px;
                    cursor: pointer;
                    background-color: #fff;
                    border-bottom: 1px solid #d4edda;
                `;
                item.innerHTML = `<strong>${modelId}</strong>`;
                
                item.addEventListener('click', () => {
                    input.value = modelId;
                    this.closeAllLists();
                });
                
                listContainer.appendChild(item);
            });
        });
        
        input.addEventListener('keydown', (e) => {
            let items = input.autocompleteList?.getElementsByClassName('autocomplete-item');
            if (!items) return;
            
            if (e.keyCode === 40) { // Down arrow
                currentFocus++;
                this.addActive(items, currentFocus);
            } else if (e.keyCode === 38) { // Up arrow
                currentFocus--;
                this.addActive(items, currentFocus);
            } else if (e.keyCode === 13) { // Enter
                e.preventDefault();
                if (currentFocus > -1 && items[currentFocus]) {
                    items[currentFocus].click();
                }
            }
        });
    }

    addActive(items, currentFocus) {
        if (!items) return;
        this.removeActive(items);
        if (currentFocus >= items.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = items.length - 1;
        items[currentFocus].classList.add('autocomplete-active');
        items[currentFocus].style.backgroundColor = '#e9ecef';
    }

    removeActive(items) {
        Array.from(items).forEach(item => {
            item.classList.remove('autocomplete-active');
            item.style.backgroundColor = '#fff';
        });
    }

    closeAllLists() {
        document.querySelectorAll('.autocomplete-items').forEach(list => list.remove());
    }

    // ============================================
    // FORM HANDLERS FOR ALL INFERENCE TYPES
    // ============================================

    setupFormHandlers() {
        // Text Generation
        this.setupFormHandler('text-generation-form', this.handleTextGeneration.bind(this));
        
        // Text Classification
        this.setupFormHandler('text-classification-form', this.handleTextClassification.bind(this));
        
        // Text Embeddings
        this.setupFormHandler('text-embeddings-form', this.handleTextEmbeddings.bind(this));
        
        // Model Recommendations
        this.setupFormHandler('model-recommendations-form', this.handleModelRecommendations.bind(this));
        
        // Model Manager
        this.setupFormHandler('model-search-form', this.handleModelSearch.bind(this));
        
        // Additional forms for new inference types
        this.setupFormHandler('audio-transcription-form', this.handleAudioTranscription.bind(this));
        this.setupFormHandler('image-classification-form', this.handleImageClassification.bind(this));
        this.setupFormHandler('image-generation-form', this.handleImageGeneration.bind(this));
        this.setupFormHandler('code-generation-form', this.handleCodeGeneration.bind(this));
    }

    setupFormHandler(formId, handler) {
        const form = document.getElementById(formId);
        if (form) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const submitBtn = form.querySelector('button[type="submit"]');
                const originalText = submitBtn.textContent;
                
                try {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                    
                    await handler(new FormData(form));
                    
                } catch (error) {
                    console.error('Form submission error:', error);
                    this.showNotification('Error: ' + error.message, 'error');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalText;
                }
            });
        }
    }

    // ============================================
    // INFERENCE HANDLERS USING SDK
    // ============================================

    async handleTextGeneration(formData) {
        const prompt = formData.get('prompt');
        const modelId = formData.get('model') || null;
        const maxLength = parseInt(formData.get('max_length')) || 100;
        const temperature = parseFloat(formData.get('temperature')) || 0.7;
        
        if (!prompt) {
            this.showNotification('Please enter a prompt', 'warning');
            return;
        }
        
        try {
            const result = await this.mcpClient.generateText(prompt, {
                modelId,
                maxLength,
                temperature
            });
            
            this.displayResult('text-generation-result', {
                generated_text: result.generated_text,
                model_used: result.model_used,
                parameters: result.parameters
            });
            
            this.showNotification('Text generated successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Text generation failed: ${error.message}`);
        }
    }

    async handleTextClassification(formData) {
        const text = formData.get('text');
        const modelId = formData.get('model') || null;
        
        if (!text) {
            this.showNotification('Please enter text to classify', 'warning');
            return;
        }
        
        try {
            const result = await this.mcpClient.classifyText(text, { modelId });
            
            this.displayClassificationResult('text-classification-result', result);
            this.showNotification('Text classified successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Text classification failed: ${error.message}`);
        }
    }

    async handleTextEmbeddings(formData) {
        const text = formData.get('text');
        const modelId = formData.get('model') || null;
        
        if (!text) {
            this.showNotification('Please enter text to embed', 'warning');
            return;
        }
        
        try {
            const result = await this.mcpClient.generateEmbeddings(text, { modelId });
            
            this.displayEmbeddingsResult('text-embeddings-result', result);
            this.showNotification('Embeddings generated successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Embedding generation failed: ${error.message}`);
        }
    }

    async handleModelRecommendations(formData) {
        const taskType = formData.get('task_type') || 'text_generation';
        const inputType = formData.get('input_type') || 'text';
        
        try {
            const result = await this.mcpClient.getModelRecommendations(taskType, inputType);
            
            this.displayRecommendationsResult('model-recommendations-result', result);
            this.showNotification('Model recommendations generated!', 'success');
            
        } catch (error) {
            throw new Error(`Model recommendations failed: ${error.message}`);
        }
    }

    async handleModelSearch(formData) {
        const query = formData.get('search_query') || '';
        const limit = parseInt(formData.get('limit')) || 10;
        
        try {
            const result = await this.mcpClient.searchModels(query, limit);
            
            this.displayModelSearchResult('model-search-result', result);
            this.showNotification(`Found ${result.total} models`, 'success');
            
        } catch (error) {
            throw new Error(`Model search failed: ${error.message}`);
        }
    }

    // New inference handlers
    async handleAudioTranscription(formData) {
        const audioFile = formData.get('audio_file');
        const modelId = formData.get('model') || null;
        
        if (!audioFile || audioFile.size === 0) {
            this.showNotification('Please select an audio file', 'warning');
            return;
        }
        
        try {
            // Convert file to base64 or handle file upload
            const audioData = await this.fileToBase64(audioFile);
            const result = await this.mcpClient.transcribeAudio(audioData, { modelId });
            
            this.displayResult('audio-transcription-result', result);
            this.showNotification('Audio transcribed successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Audio transcription failed: ${error.message}`);
        }
    }

    async handleImageClassification(formData) {
        const imageFile = formData.get('image_file');
        const modelId = formData.get('model') || null;
        
        if (!imageFile || imageFile.size === 0) {
            this.showNotification('Please select an image file', 'warning');
            return;
        }
        
        try {
            const imageData = await this.fileToBase64(imageFile);
            const result = await this.mcpClient.classifyImage(imageData, { modelId });
            
            this.displayResult('image-classification-result', result);
            this.showNotification('Image classified successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Image classification failed: ${error.message}`);
        }
    }

    async handleImageGeneration(formData) {
        const prompt = formData.get('prompt');
        const modelId = formData.get('model') || null;
        const width = parseInt(formData.get('width')) || 512;
        const height = parseInt(formData.get('height')) || 512;
        
        if (!prompt) {
            this.showNotification('Please enter a prompt', 'warning');
            return;
        }
        
        try {
            const result = await this.mcpClient.generateImage(prompt, {
                modelId,
                width,
                height
            });
            
            this.displayResult('image-generation-result', result);
            this.showNotification('Image generated successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Image generation failed: ${error.message}`);
        }
    }

    async handleCodeGeneration(formData) {
        const description = formData.get('description');
        const language = formData.get('language') || 'python';
        const modelId = formData.get('model') || null;
        
        if (!description) {
            this.showNotification('Please enter a code description', 'warning');
            return;
        }
        
        try {
            const result = await this.mcpClient.generateCode(description, {
                modelId,
                language
            });
            
            this.displayCodeResult('code-generation-result', result);
            this.showNotification('Code generated successfully!', 'success');
            
        } catch (error) {
            throw new Error(`Code generation failed: ${error.message}`);
        }
    }

    // ============================================
    // RESULT DISPLAY METHODS
    // ============================================

    displayResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="result-item">
                    <pre><code>${JSON.stringify(result, null, 2)}</code></pre>
                </div>
            `;
        }
    }

    displayClassificationResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element && result.classification) {
            const classification = result.classification;
            element.innerHTML = `
                <div class="classification-result">
                    <h5>Classification Result</h5>
                    <div class="result-item">
                        <strong>Label:</strong> ${classification.label}<br>
                        <strong>Confidence:</strong> ${(classification.confidence * 100).toFixed(2)}%<br>
                        <strong>Model:</strong> ${result.model_used}
                    </div>
                    ${classification.all_scores ? `
                        <div class="all-scores">
                            <h6>All Scores:</h6>
                            ${classification.all_scores.map(score => `
                                <div class="score-item">
                                    <span>${score.label}</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${score.score * 100}%"></div>
                                    </div>
                                    <span>${(score.score * 100).toFixed(2)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }
    }

    displayEmbeddingsResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element && result.embeddings) {
            const embeddings = result.embeddings.slice(0, 20); // Show first 20 dimensions
            element.innerHTML = `
                <div class="embeddings-result">
                    <h5>Text Embeddings</h5>
                    <div class="result-item">
                        <strong>Dimensions:</strong> ${result.dimension}<br>
                        <strong>Model:</strong> ${result.model_used}
                    </div>
                    <div class="embedding-visualization">
                        ${embeddings.map((val, idx) => `
                            <span class="embedding-value" title="Dimension ${idx}: ${val}">
                                ${val.toFixed(3)}
                            </span>
                        `).join('')}
                        ${result.embeddings.length > 20 ? `<span class="embedding-more">... and ${result.embeddings.length - 20} more</span>` : ''}
                    </div>
                    <button class="btn btn-sm btn-outline-primary mt-2" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                        Show All Values
                    </button>
                    <pre style="display: none;"><code>${JSON.stringify(result.embeddings, null, 2)}</code></pre>
                </div>
            `;
        }
    }

    displayCodeResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element && result.code) {
            element.innerHTML = `
                <div class="code-result">
                    <h5>Generated Code</h5>
                    <div class="result-item">
                        <strong>Model:</strong> ${result.model_used}
                    </div>
                    <div class="code-block">
                        <pre><code>${result.code}</code></pre>
                    </div>
                    <button class="btn btn-sm btn-outline-primary" onclick="navigator.clipboard.writeText(\`${result.code.replace(/`/g, '\\`')}\`)">
                        Copy Code
                    </button>
                </div>
            `;
        }
    }

    displayRecommendationsResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="recommendations-result">
                    <h5>Model Recommendations</h5>
                    <div class="result-item">
                        <strong>Algorithm:</strong> ${result.algorithm}<br>
                        <strong>Task Type:</strong> ${result.context?.task_type}<br>
                        <strong>Input Type:</strong> ${result.context?.input_type}
                    </div>
                    ${result.recommendations.length > 0 ? `
                        <div class="recommendations-list">
                            ${result.recommendations.map(rec => `
                                <div class="recommendation-item">
                                    <strong>${rec.model_id || rec}</strong>
                                    ${rec.confidence ? `<span class="confidence">(${(rec.confidence * 100).toFixed(2)}%)</span>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : '<p>No recommendations available</p>'}
                </div>
            `;
        }
    }

    displayModelSearchResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="model-search-result">
                    <h5>Model Search Results</h5>
                    <div class="result-item">
                        <strong>Query:</strong> "${result.query}"<br>
                        <strong>Total Results:</strong> ${result.total}
                    </div>
                    <div class="models-list">
                        ${result.models.map(model => {
                            const modelId = model.model_id || model.id || 'unknown';
                            return `
                                <div class="model-item">
                                    <strong>${modelId}</strong>
                                    ${model.type ? `<span class="model-type">${model.type}</span>` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }
    }

    // ============================================
    // UTILITY METHODS
    // ============================================

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    setupRangeInputs() {
        const rangeInputs = document.querySelectorAll('input[type="range"]');
        rangeInputs.forEach(input => {
            const output = input.nextElementSibling;
            if (output && output.tagName === 'OUTPUT') {
                input.addEventListener('input', () => {
                    output.value = input.value;
                });
            }
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit active form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeForm = document.querySelector('form:hover, form:focus-within');
                if (activeForm) {
                    e.preventDefault();
                    activeForm.dispatchEvent(new Event('submit'));
                }
            }
            
            // Escape to close autocomplete
            if (e.key === 'Escape') {
                this.closeAllLists();
            }
        });
    }

    setupModelManager() {
        // Model manager specific setup if needed
        console.log('Model manager setup complete');
    }

    setupHuggingFaceBrowser() {
        // HuggingFace browser specific setup if needed
        console.log('HuggingFace browser setup complete');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.kitchenSinkApp = new KitchenSinkSDKApp();
});

// CSS styles for the app
const styles = `
<style>
.notification-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    max-width: 400px;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.fade-in {
    animation: slideIn 0.3s ease-out;
}

.autocomplete-items {
    position: absolute;
    border: 1px solid #d4edda;
    border-bottom: none;
    border-top: none;
    z-index: 99;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    max-height: 200px;
    overflow-y: auto;
}

.autocomplete-item {
    padding: 10px;
    cursor: pointer;
    background-color: #fff;
    border-bottom: 1px solid #d4edda;
}

.autocomplete-item:hover,
.autocomplete-active {
    background-color: #e9ecef !important;
}

.server-status.connected {
    color: #28a745;
}

.server-status.disconnected {
    color: #dc3545;
}

.embedding-visualization {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 10px 0;
}

.embedding-value {
    background: linear-gradient(45deg, #007bff, #6c757d);
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.8em;
}

.confidence-bar {
    height: 10px;
    background-color: #e9ecef;
    border-radius: 5px;
    overflow: hidden;
    display: inline-block;
    width: 100px;
    margin: 0 10px;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(45deg, #28a745, #20c997);
    transition: width 0.5s ease;
}

.code-block {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
    overflow-x: auto;
}

.result-item {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', styles);
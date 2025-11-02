/**
 * Enhanced IPFS Accelerate AI Dashboard
 * 
 * A comprehensive dashboard for AI model inference with enhanced UI,
 * real-time metrics, and deep integration with ipfs_accelerate_py.
 */

class EnhancedAIDashboard {
    constructor() {
        this.mcpClient = new MCPClient('/jsonrpc', {
            timeout: 60000,
            retries: 3
        });
        
        // State management
        this.state = {
            connected: false,
            models: [],
            serverInfo: null,
            requestCount: 0,
            startTime: Date.now(),
            theme: localStorage.getItem('theme') || 'light',
            hardwareInfo: null,
            performanceMetrics: {
                latency: [],
                throughput: [],
                timestamps: []
            }
        };
        
        // Charts
        this.charts = {
            performance: null,
            systemMetrics: null
        };
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Enhanced AI Dashboard...');
        
        // Apply saved theme
        this.applyTheme();
        
        // Setup UI components
        this.setupNotificationSystem();
        this.setupEventListeners();
        this.setupFormHandlers();
        this.setupCharts();
        
        // Initialize connection
        await this.initializeConnection();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        console.log('✅ Enhanced AI Dashboard initialized successfully');
    }

    // ============================================
    // CONNECTION & SERVER MANAGEMENT
    // ============================================

    async initializeConnection() {
        try {
            this.showNotification('🔄 Connecting to IPFS Accelerate AI server...', 'info');
            
            // Check server availability
            const isAvailable = await this.mcpClient.waitForServer(10, 2000);
            if (!isAvailable) {
                throw new Error('Server not responding');
            }
            
            // Get server information
            this.state.serverInfo = await this.mcpClient.getServerInfo();
            this.state.connected = true;
            
            // Load initial data
            await Promise.all([
                this.loadModels(),
                this.loadHardwareInfo(),
                this.loadSystemMetrics()
            ]);
            
            this.updateConnectionStatus(true);
            this.showNotification('✅ Connected to IPFS Accelerate AI successfully!', 'success');
            
        } catch (error) {
            console.error('❌ Connection failed:', error);
            this.updateConnectionStatus(false);
            this.showNotification(`❌ Connection failed: ${error.message}`, 'error');
        }
    }

    updateConnectionStatus(connected) {
        this.state.connected = connected;
        
        const statusElement = document.getElementById('connection-status');
        const statusDot = statusElement.querySelector('.status-dot');
        const statusText = statusElement.querySelector('span');
        
        if (connected) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected to IPFS Accelerate AI';
        } else {
            statusDot.className = 'status-dot disconnected';
            statusText.textContent = 'Disconnected';
        }
        
        this.updateMetricsDisplay();
    }

    async loadModels() {
        try {
            const response = await this.mcpClient.listModels();
            this.state.models = response.models || [];
            console.log(`📚 Loaded ${this.state.models.length} models`);
            this.populateModelSelectors();
        } catch (error) {
            console.error('Error loading models:', error);
            this.showNotification('⚠️ Failed to load models', 'warning');
        }
    }

    async loadHardwareInfo() {
        try {
            // Try to get hardware information from ipfs_accelerate_py
            const hardwareInfo = await this.mcpClient.request('get_hardware_info');
            this.state.hardwareInfo = hardwareInfo;
            this.updateHardwareDisplay();
        } catch (error) {
            console.log('Hardware info not available from server');
            // Fallback to browser-based hardware detection
            this.detectBrowserHardware();
        }
    }

    async loadSystemMetrics() {
        try {
            const metrics = await this.mcpClient.request('get_system_metrics');
            this.updateSystemMetricsChart(metrics);
        } catch (error) {
            console.log('System metrics not available');
        }
    }

    // ============================================
    // UI SETUP AND EVENT HANDLERS
    // ============================================

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('[data-bs-toggle="pill"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('data-bs-target');
                this.onTabChange(target);
            });
        });

        // Model search
        const modelSearch = document.getElementById('model-search');
        if (modelSearch) {
            modelSearch.addEventListener('input', this.debounce(() => {
                this.searchModels();
            }, 300));
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeForm = document.querySelector('.tab-pane.active form');
                if (activeForm) {
                    e.preventDefault();
                    activeForm.dispatchEvent(new Event('submit'));
                }
            }
        });
    }

    setupFormHandlers() {
        const forms = [
            'text-generation-form',
            'text-analysis-form', 
            'embeddings-form',
            'audio-transcription-form',
            'speech-synthesis-form',
            'image-analysis-form',
            'image-generation-form',
            'multimodal-form',
            'code-generation-form'
        ];

        forms.forEach(formId => {
            const form = document.getElementById(formId);
            if (form) {
                form.addEventListener('submit', (e) => this.handleFormSubmit(e, formId));
            }
        });
    }

    async handleFormSubmit(e, formId) {
        e.preventDefault();
        
        const form = e.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalContent = submitBtn.innerHTML;
        
        try {
            // Show loading state
            this.showLoadingState(form, true);
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            submitBtn.disabled = true;
            
            const startTime = Date.now();
            const formData = new FormData(form);
            
            // Route to appropriate handler
            let result;
            switch (formId) {
                case 'text-generation-form':
                    result = await this.handleTextGeneration(formData);
                    break;
                case 'text-analysis-form':
                    result = await this.handleTextAnalysis(formData);
                    break;
                case 'embeddings-form':
                    result = await this.handleEmbeddings(formData);
                    break;
                case 'audio-transcription-form':
                    result = await this.handleAudioTranscription(formData);
                    break;
                case 'speech-synthesis-form':
                    result = await this.handleSpeechSynthesis(formData);
                    break;
                case 'image-analysis-form':
                    result = await this.handleImageAnalysis(formData);
                    break;
                case 'image-generation-form':
                    result = await this.handleImageGeneration(formData);
                    break;
                case 'multimodal-form':
                    result = await this.handleMultimodal(formData);
                    break;
                case 'code-generation-form':
                    result = await this.handleCodeGeneration(formData);
                    break;
                default:
                    throw new Error('Unknown form type');
            }
            
            // Update performance metrics
            const latency = Date.now() - startTime;
            this.updatePerformanceMetrics(latency);
            
            // Display result
            const resultElementId = formId.replace('-form', '-result');
            this.displayResult(resultElementId, result);
            
            this.showNotification('✅ Request completed successfully!', 'success');
            
        } catch (error) {
            console.error('Form submission error:', error);
            this.showNotification(`❌ Error: ${error.message}`, 'error');
        } finally {
            // Reset form state
            this.showLoadingState(form, false);
            submitBtn.innerHTML = originalContent;
            submitBtn.disabled = false;
            this.state.requestCount++;
            this.updateMetricsDisplay();
        }
    }

    // ============================================
    // INFERENCE HANDLERS
    // ============================================

    async handleTextGeneration(formData) {
        const prompt = formData.get('prompt');
        const modelId = formData.get('model_id') || null;
        const maxLength = parseInt(formData.get('max_length')) || 100;
        const temperature = parseFloat(formData.get('temperature')) || 0.7;
        
        return await this.mcpClient.generateText(prompt, {
            modelId,
            maxLength,
            temperature
        });
    }

    async handleTextAnalysis(formData) {
        const text = formData.get('text');
        const analysisType = formData.get('analysis_type');
        const modelId = formData.get('model_id') || null;
        
        switch (analysisType) {
            case 'sentiment':
            case 'classification':
                return await this.mcpClient.classifyText(text, { modelId });
            case 'emotion':
                return await this.mcpClient.request('analyze_emotion', { text, model_id: modelId });
            case 'topic':
                return await this.mcpClient.request('extract_topics', { text, model_id: modelId });
            default:
                return await this.mcpClient.classifyText(text, { modelId });
        }
    }

    async handleEmbeddings(formData) {
        const text = formData.get('text');
        const modelId = formData.get('model_id') || null;
        const normalize = formData.get('normalize') === 'on';
        
        return await this.mcpClient.generateEmbeddings(text, { modelId, normalize });
    }

    async handleAudioTranscription(formData) {
        const audioFile = formData.get('audio_file');
        if (!audioFile || audioFile.size === 0) {
            throw new Error('Please select an audio file');
        }
        
        const audioData = await this.fileToBase64(audioFile);
        return await this.mcpClient.transcribeAudio(audioData);
    }

    async handleSpeechSynthesis(formData) {
        const text = formData.get('text');
        const modelId = formData.get('model_id') || null;
        
        return await this.mcpClient.synthesizeSpeech(text, { modelId });
    }

    async handleImageAnalysis(formData) {
        const imageFile = formData.get('image_file');
        const task = formData.get('task');
        
        if (!imageFile || imageFile.size === 0) {
            throw new Error('Please select an image file');
        }
        
        const imageData = await this.fileToBase64(imageFile);
        
        switch (task) {
            case 'classification':
                return await this.mcpClient.classifyImage(imageData);
            case 'object-detection':
                return await this.mcpClient.detectObjects(imageData);
            case 'segmentation':
                return await this.mcpClient.segmentImage(imageData);
            case 'ocr':
                return await this.mcpClient.request('extract_text_from_image', { image: imageData });
            default:
                return await this.mcpClient.classifyImage(imageData);
        }
    }

    async handleImageGeneration(formData) {
        const prompt = formData.get('prompt');
        const width = parseInt(formData.get('width')) || 512;
        const height = parseInt(formData.get('height')) || 512;
        
        return await this.mcpClient.generateImage(prompt, { width, height });
    }

    async handleMultimodal(formData) {
        const imageFile = formData.get('image_file');
        const task = formData.get('task');
        const question = formData.get('question');
        
        if (!imageFile || imageFile.size === 0) {
            throw new Error('Please select an image file');
        }
        
        const imageData = await this.fileToBase64(imageFile);
        
        switch (task) {
            case 'caption':
                return await this.mcpClient.generateImageCaption(imageData);
            case 'vqa':
                if (!question) throw new Error('Please provide a question for VQA');
                return await this.mcpClient.answerVisualQuestion(imageData, question);
            case 'ocr':
                return await this.mcpClient.processDocument(imageData);
            default:
                return await this.mcpClient.generateImageCaption(imageData);
        }
    }

    async handleCodeGeneration(formData) {
        const description = formData.get('description');
        const language = formData.get('language');
        const style = formData.get('style');
        
        return await this.mcpClient.generateCode(description, { 
            language,
            style 
        });
    }

    // ============================================
    // RESULT DISPLAY
    // ============================================

    displayResult(elementId, result) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.style.display = 'block';
        
        // Create result header
        const header = `
            <div class="result-header">
                <i class="fas fa-check-circle"></i>
                <span>Result</span>
                <div class="ms-auto">
                    <small class="text-muted">${new Date().toLocaleTimeString()}</small>
                </div>
            </div>
        `;
        
        // Format result based on type
        let content = '';
        
        if (result.generated_text) {
            content = this.formatTextResult(result);
        } else if (result.classification) {
            content = this.formatClassificationResult(result);
        } else if (result.embeddings) {
            content = this.formatEmbeddingsResult(result);
        } else if (result.transcription) {
            content = this.formatAudioResult(result);
        } else if (result.image_url || result.generated_image) {
            content = this.formatImageResult(result);
        } else if (result.code) {
            content = this.formatCodeResult(result);
        } else {
            content = this.formatGenericResult(result);
        }
        
        element.innerHTML = header + content;
    }

    formatTextResult(result) {
        return `
            <div class="text-result">
                <div class="generated-text p-3 border rounded bg-light">
                    ${result.generated_text}
                </div>
                <div class="result-meta mt-2">
                    <span class="badge bg-primary">Model: ${result.model_used}</span>
                    ${result.parameters ? `<span class="badge bg-secondary">Length: ${result.parameters.max_length}</span>` : ''}
                    ${result.parameters ? `<span class="badge bg-secondary">Temperature: ${result.parameters.temperature}</span>` : ''}
                </div>
            </div>
        `;
    }

    formatClassificationResult(result) {
        const classification = result.classification;
        const confidence = (classification.confidence * 100).toFixed(1);
        
        let scoresList = '';
        if (classification.all_scores) {
            scoresList = `
                <div class="all-scores mt-3">
                    <h6>All Classifications:</h6>
                    ${classification.all_scores.map(score => `
                        <div class="score-item d-flex align-items-center mb-2">
                            <span class="flex-shrink-0" style="width: 120px;">${score.label}</span>
                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                <div class="progress-bar" style="width: ${score.score * 100}%"></div>
                            </div>
                            <span class="flex-shrink-0">${(score.score * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        return `
            <div class="classification-result">
                <div class="main-result p-3 border rounded">
                    <h5 class="text-success">
                        <i class="fas fa-tag"></i> ${classification.label}
                    </h5>
                    <div class="confidence-display">
                        <span class="fs-4 fw-bold text-primary">${confidence}%</span>
                        <small class="text-muted"> confidence</small>
                    </div>
                </div>
                ${scoresList}
                <div class="result-meta mt-2">
                    <span class="badge bg-primary">Model: ${result.model_used}</span>
                </div>
            </div>
        `;
    }

    formatEmbeddingsResult(result) {
        const embeddings = result.embeddings;
        const displayCount = Math.min(20, embeddings.length);
        const displayEmbeddings = embeddings.slice(0, displayCount);
        
        return `
            <div class="embeddings-result">
                <div class="embedding-stats p-3 border rounded bg-light mb-3">
                    <div class="row text-center">
                        <div class="col-md-4">
                            <div class="metric-value">${result.dimension}</div>
                            <div class="metric-label">Dimensions</div>
                        </div>
                        <div class="col-md-4">
                            <div class="metric-value">${embeddings.length}</div>
                            <div class="metric-label">Vector Length</div>
                        </div>
                        <div class="col-md-4">
                            <div class="metric-value">${result.model_used.split('/').pop()}</div>
                            <div class="metric-label">Model</div>
                        </div>
                    </div>
                </div>
                <div class="embedding-visualization">
                    <h6>Vector Visualization (first ${displayCount} dimensions):</h6>
                    <div class="embedding-viz">
                        ${displayEmbeddings.map((val, idx) => `
                            <div class="embedding-value" title="Dimension ${idx}: ${val}">
                                ${val.toFixed(3)}
                            </div>
                        `).join('')}
                    </div>
                    ${embeddings.length > displayCount ? `
                        <p class="text-muted mt-2">... and ${embeddings.length - displayCount} more dimensions</p>
                    ` : ''}
                </div>
                <button class="btn btn-outline-primary btn-sm mt-2" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                    Show Raw Data
                </button>
                <pre class="mt-2" style="display: none; font-size: 0.8rem;"><code>${JSON.stringify(embeddings, null, 2)}</code></pre>
            </div>
        `;
    }

    formatAudioResult(result) {
        return `
            <div class="audio-result">
                <div class="transcription p-3 border rounded bg-light">
                    <h6><i class="fas fa-quote-left"></i> Transcription:</h6>
                    <p class="mb-0">${result.transcription || result.text}</p>
                </div>
                <div class="result-meta mt-2">
                    <span class="badge bg-primary">Model: ${result.model_used}</span>
                    ${result.language ? `<span class="badge bg-secondary">Language: ${result.language}</span>` : ''}
                </div>
            </div>
        `;
    }

    formatImageResult(result) {
        const imageUrl = result.image_url || result.generated_image;
        return `
            <div class="image-result">
                <div class="generated-image text-center p-3 border rounded">
                    <img src="${imageUrl}" alt="Generated Image" class="img-fluid rounded" style="max-height: 400px;">
                </div>
                <div class="result-meta mt-2">
                    <span class="badge bg-primary">Model: ${result.model_used}</span>
                    ${result.width ? `<span class="badge bg-secondary">${result.width}x${result.height}</span>` : ''}
                </div>
            </div>
        `;
    }

    formatCodeResult(result) {
        return `
            <div class="code-result">
                <div class="code-header d-flex justify-content-between align-items-center mb-2">
                    <h6><i class="fas fa-code"></i> Generated Code</h6>
                    <button class="btn btn-outline-primary btn-sm" onclick="navigator.clipboard.writeText(\`${result.code.replace(/`/g, '\\`')}\`)">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
                <pre class="language-${result.language || 'python'}"><code>${result.code}</code></pre>
                <div class="result-meta mt-2">
                    <span class="badge bg-primary">Model: ${result.model_used}</span>
                    ${result.language ? `<span class="badge bg-secondary">Language: ${result.language}</span>` : ''}
                </div>
            </div>
        `;
    }

    formatGenericResult(result) {
        return `
            <div class="generic-result">
                <pre class="bg-light p-3 border rounded"><code>${JSON.stringify(result, null, 2)}</code></pre>
            </div>
        `;
    }

    // ============================================
    // CHARTS AND VISUALIZATIONS
    // ============================================

    setupCharts() {
        this.setupPerformanceChart();
        this.setupSystemMetricsChart();
    }

    setupPerformanceChart() {
        const ctx = document.getElementById('performance-chart');
        if (!ctx) return;

        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                }
            }
        });
    }

    setupSystemMetricsChart() {
        const ctx = document.getElementById('system-metrics-chart');
        if (!ctx) return;

        this.charts.systemMetrics = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'GPU'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgb(99, 102, 241)',
                        'rgb(139, 92, 246)',
                        'rgb(6, 182, 212)'
                    ]
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

    updatePerformanceMetrics(latency) {
        const now = new Date();
        this.state.performanceMetrics.latency.push(latency);
        this.state.performanceMetrics.timestamps.push(now.toLocaleTimeString());
        
        // Keep only last 20 data points
        if (this.state.performanceMetrics.latency.length > 20) {
            this.state.performanceMetrics.latency.shift();
            this.state.performanceMetrics.timestamps.shift();
        }
        
        if (this.charts.performance) {
            this.charts.performance.data.labels = this.state.performanceMetrics.timestamps;
            this.charts.performance.data.datasets[0].data = this.state.performanceMetrics.latency;
            this.charts.performance.update('none');
        }
    }

    updateSystemMetricsChart(metrics) {
        if (this.charts.systemMetrics && metrics) {
            this.charts.systemMetrics.data.datasets[0].data = [
                metrics.cpu_usage || 0,
                metrics.memory_usage || 0,
                metrics.gpu_usage || 0
            ];
            this.charts.systemMetrics.update();
        }
    }

    // ============================================
    // MODEL MANAGEMENT
    // ============================================

    populateModelSelectors() {
        const selectors = document.querySelectorAll('select[name="model_id"]');
        
        selectors.forEach(select => {
            // Clear existing options except first
            const firstOption = select.querySelector('option');
            select.innerHTML = '';
            if (firstOption) {
                select.appendChild(firstOption);
            }
            
            // Add model options
            this.state.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id || model.id || model.name;
                option.textContent = model.model_id || model.id || model.name;
                select.appendChild(option);
            });
        });
    }

    async searchModels() {
        const query = document.getElementById('model-search')?.value || '';
        
        try {
            const response = await this.mcpClient.searchModels(query, 50);
            this.displayModelList(response.models || []);
        } catch (error) {
            console.error('Model search error:', error);
            this.showNotification('Failed to search models', 'error');
        }
    }

    displayModelList(models) {
        const container = document.getElementById('model-list');
        if (!container) return;
        
        if (models.length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No models found</div>';
            return;
        }
        
        container.innerHTML = models.map(model => {
            const modelId = model.model_id || model.id || model.name || 'Unknown';
            const modelType = model.type || 'Unknown';
            const description = model.description || 'No description available';
            
            return `
                <div class="model-card">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${modelId}</h6>
                            <p class="text-muted mb-2">${description}</p>
                        </div>
                        <span class="model-badge">${modelType}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    // ============================================
    // HARDWARE AND SYSTEM INFO
    // ============================================

    detectBrowserHardware() {
        const hardwareInfo = {
            platform: navigator.platform,
            userAgent: navigator.userAgent,
            memory: navigator.deviceMemory || 'Unknown',
            cores: navigator.hardwareConcurrency || 'Unknown',
            gpu: 'Detecting...'
        };
        
        // Try to detect GPU
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                hardwareInfo.gpu = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            } else {
                hardwareInfo.gpu = 'WebGL Available';
            }
        } else {
            hardwareInfo.gpu = 'WebGL Not Available';
        }
        
        this.state.hardwareInfo = hardwareInfo;
        this.updateHardwareDisplay();
    }

    updateHardwareDisplay() {
        const container = document.getElementById('hardware-info');
        if (!container || !this.state.hardwareInfo) return;
        
        const info = this.state.hardwareInfo;
        container.innerHTML = `
            <div class="hardware-item mb-2">
                <i class="fas fa-microchip text-primary"></i>
                <strong>CPU Cores:</strong> ${info.cores}
            </div>
            <div class="hardware-item mb-2">
                <i class="fas fa-memory text-success"></i>
                <strong>Memory:</strong> ${info.memory} GB
            </div>
            <div class="hardware-item mb-2">
                <i class="fas fa-display text-info"></i>
                <strong>GPU:</strong> ${info.gpu}
            </div>
            <div class="hardware-item mb-2">
                <i class="fas fa-desktop text-warning"></i>
                <strong>Platform:</strong> ${info.platform}
            </div>
        `;
    }

    // ============================================
    // UI UTILITIES
    // ============================================

    showLoadingState(form, loading) {
        let overlay = form.querySelector('.loading-overlay');
        
        if (loading) {
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'loading-overlay';
                overlay.innerHTML = '<div class="spinner"></div>';
                form.style.position = 'relative';
                form.appendChild(overlay);
            }
        } else {
            if (overlay) {
                overlay.remove();
            }
        }
    }

    setupNotificationSystem() {
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
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
        const container = document.getElementById('notification-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `alert alert-${this.getBootstrapAlertClass(type)} notification-toast`;
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="${this.getNotificationIcon(type)} me-2"></i>
                <span class="flex-grow-1">${message}</span>
                <button type="button" class="btn-close ms-2" onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        `;
        
        container.appendChild(notification);
        
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
    }

    getBootstrapAlertClass(type) {
        const mapping = {
            success: 'success',
            error: 'danger',
            warning: 'warning',
            info: 'info'
        };
        return mapping[type] || 'info';
    }

    getNotificationIcon(type) {
        const mapping = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return mapping[type] || 'fas fa-info-circle';
    }

    // ============================================
    // THEME MANAGEMENT
    // ============================================

    applyTheme() {
        document.documentElement.setAttribute('data-bs-theme', this.state.theme);
        const icon = document.getElementById('theme-icon');
        if (icon) {
            icon.className = this.state.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    // ============================================
    // PERIODIC UPDATES
    // ============================================

    startPeriodicUpdates() {
        // Update uptime every second
        setInterval(() => {
            this.updateUptime();
        }, 1000);
        
        // Update system metrics every 30 seconds
        setInterval(() => {
            if (this.state.connected) {
                this.loadSystemMetrics();
            }
        }, 30000);
    }

    updateUptime() {
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement) {
            const uptime = Date.now() - this.state.startTime;
            const minutes = Math.floor(uptime / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            uptimeElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    updateMetricsDisplay() {
        // Update model count
        const modelsCountElement = document.getElementById('models-count');
        if (modelsCountElement) {
            modelsCountElement.textContent = this.state.models.length;
        }
        
        // Update methods count
        const methodsCountElement = document.getElementById('methods-count');
        if (methodsCountElement) {
            methodsCountElement.textContent = this.state.serverInfo?.methods_count || 0;
        }
        
        // Update request count
        const requestsCountElement = document.getElementById('requests-count');
        if (requestsCountElement) {
            requestsCountElement.textContent = this.state.requestCount;
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

    debounce(func, wait) {
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

    onTabChange(target) {
        console.log(`Switched to tab: ${target}`);
        // Trigger any tab-specific initialization if needed
    }
}

// Global functions for HTML event handlers
function toggleTheme() {
    if (window.dashboard) {
        window.dashboard.state.theme = window.dashboard.state.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', window.dashboard.state.theme);
        window.dashboard.applyTheme();
    }
}

function searchModels() {
    if (window.dashboard) {
        window.dashboard.searchModels();
    }
}

// Initialize the dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new EnhancedAIDashboard();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedAIDashboard;
}
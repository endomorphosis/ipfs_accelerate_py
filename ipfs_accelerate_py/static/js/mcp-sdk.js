/**
 * MCP JSON-RPC JavaScript SDK
 * 
 * A comprehensive JavaScript SDK for communicating with the MCP server
 * using JSON-RPC 2.0 protocol.
 */

class MCPClient {
    constructor(endpoint = '/jsonrpc', options = {}) {
        this.endpoint = endpoint;
        this.options = {
            timeout: 30000,
            retries: 3,
            reportErrors: false,  // Enable error reporting to server
            errorReportEndpoint: '/report-error',
            ...options
        };
        this.requestId = 0;
        this.pendingRequests = new Map();
    }

    /**
     * Generate a unique request ID
     */
    generateRequestId() {
        return ++this.requestId;
    }

    /**
     * Make a JSON-RPC 2.0 request
     */
    async request(method, params = {}) {
        const id = this.generateRequestId();
        const requestBody = {
            jsonrpc: "2.0",
            method: method,
            params: params,
            id: id
        };

        try {
            const response = await this._makeHttpRequest(requestBody);
            
            if (response.error) {
                const mcpError = new MCPError(response.error.code, response.error.message, response.error.data);
                // Report error if enabled
                if (this.options.reportErrors) {
                    this._reportError(mcpError, { method, params });
                }
                throw mcpError;
            }
            
            return response.result;
        } catch (error) {
            if (error instanceof MCPError) {
                throw error;
            }
            const internalError = new MCPError(-32603, "Internal error", error.message);
            // Report error if enabled
            if (this.options.reportErrors) {
                this._reportError(internalError, { method, params, originalError: error });
            }
            throw internalError;
        }
    }

    /**
     * Make a notification (fire-and-forget)
     */
    async notify(method, params = {}) {
        const requestBody = {
            jsonrpc: "2.0",
            method: method,
            params: params
        };

        await this._makeHttpRequest(requestBody);
    }

    /**
     * Make batch requests
     */
    async batch(requests) {
        const requestBodies = requests.map(({ method, params }) => ({
            jsonrpc: "2.0",
            method: method,
            params: params || {},
            id: this.generateRequestId()
        }));

        const responses = await this._makeHttpRequest(requestBodies);
        
        return responses.map(response => {
            if (response.error) {
                return { error: new MCPError(response.error.code, response.error.message, response.error.data) };
            }
            return { result: response.result };
        });
    }

    /**
     * Check if an error is a transient network error that should be retried
     */
    _isRetriableError(error) {
        // Network errors that are typically transient
        const retriableErrors = [
            'NetworkError',
            'Failed to fetch',
            'Network request failed',
            'ERR_NETWORK_CHANGED',
            'ERR_CONNECTION_REFUSED',
            'ERR_CONNECTION_RESET',
            'ECONNREFUSED',
            'ECONNRESET',
            'ETIMEDOUT'
        ];
        
        const errorStr = error.toString();
        return retriableErrors.some(pattern => errorStr.includes(pattern)) ||
               error.name === 'AbortError' ||
               error.name === 'NetworkError' ||
               error.name === 'TypeError' && errorStr.includes('fetch');
    }

    /**
     * Make HTTP request with timeout and retry logic
     */
    async _makeHttpRequest(body) {
        let lastError;
        
        for (let attempt = 0; attempt < this.options.retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.options.timeout);
                
                const response = await fetch(this.endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                // Attempt to parse JSON robustly; log if empty/truncated
                const text = await response.text();
                try {
                    return JSON.parse(text);
                } catch (e) {
                    console.error('Failed to parse JSON response:', e, 'Raw:', text?.slice(0, 200));
                    throw e;
                }
                
            } catch (error) {
                lastError = error;
                
                // Log network errors for debugging
                if (this._isRetriableError(error)) {
                    console.warn(`[MCP SDK] Network error (attempt ${attempt + 1}/${this.options.retries}):`, error.message);
                }
                
                // Retry on retriable errors or if we have retries left
                if (attempt < this.options.retries - 1 && this._isRetriableError(error)) {
                    const delay = Math.pow(2, attempt) * 1000;
                    console.log(`[MCP SDK] Retrying in ${delay}ms...`);
                    await this._delay(delay); // Exponential backoff
                } else if (attempt < this.options.retries - 1) {
                    // For non-retriable errors, use shorter delay
                    await this._delay(500);
                }
            }
        }
        
        console.error('[MCP SDK] All retry attempts failed:', lastError);
        throw lastError;
    }

    /**
     * Report an error to the server for auto-healing
     */
    async _reportError(error, context = {}) {
        // Don't report if error reporting is disabled
        if (!this.options.reportErrors) {
            return;
        }

        try {
            const errorData = {
                error_type: error.name || error.constructor.name || 'Error',
                error_message: error.message || String(error),
                stack_trace: error.stack || new Error().stack,
                context: {
                    timestamp: new Date().toISOString(),
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    ...context
                }
            };

            // Send error report to server (fire-and-forget, don't await)
            fetch(this.options.errorReportEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(errorData)
            }).catch(err => {
                // Silently ignore reporting failures to avoid infinite loops
                console.debug('[MCP SDK] Failed to report error:', err);
            });
        } catch (e) {
            // Silently ignore any errors in error reporting
            console.debug('[MCP SDK] Error in _reportError:', e);
        }
    }

    /**
     * Delay utility for retries
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ============================================
    // MODEL MANAGEMENT METHODS
    // ============================================

    /**
     * List all available models
     */
    async listModels() {
        return await this.request('list_models');
    }

    /**
     * Get specific model information
     */
    async getModel(modelId) {
        return await this.request('get_model', { model_id: modelId });
    }

    /**
     * Search models
     */
    async searchModels(query, limit = 10) {
        return await this.request('search_models', { query, limit });
    }

    /**
     * Get model recommendations using bandit algorithms
     */
    async getModelRecommendations(taskType, inputType = 'text') {
        return await this.request('get_model_recommendations', { 
            task_type: taskType, 
            input_type: inputType 
        });
    }

    /**
     * Add a new model
     */
    async addModel(modelId, metadata = {}) {
        return await this.request('add_model', { model_id: modelId, ...metadata });
    }

    // ============================================
    // TEXT PROCESSING METHODS
    // ============================================

    /**
     * Generate text using causal language modeling
     */
    async generateText(prompt, options = {}) {
        return await this.request('generate_text', {
            prompt,
            model_id: options.modelId,
            max_length: options.maxLength || 100,
            temperature: options.temperature || 0.7,
            top_p: options.topP,
            top_k: options.topK
        });
    }

    /**
     * Classify text (sentiment analysis, etc.)
     */
    async classifyText(text, options = {}) {
        return await this.request('classify_text', {
            text,
            model_id: options.modelId,
            return_all_scores: options.returnAllScores || false
        });
    }

    /**
     * Generate text embeddings
     */
    async generateEmbeddings(text, options = {}) {
        return await this.request('generate_embeddings', {
            text,
            model_id: options.modelId,
            normalize: options.normalize || false
        });
    }

    /**
     * Fill masked text
     */
    async fillMask(text, options = {}) {
        return await this.request('fill_mask', {
            text,
            model_id: options.modelId,
            top_k: options.topK || 5
        });
    }

    /**
     * Translate text
     */
    async translateText(text, targetLanguage, options = {}) {
        return await this.request('translate_text', {
            text,
            target_language: targetLanguage,
            source_language: options.sourceLanguage,
            model_id: options.modelId
        });
    }

    /**
     * Summarize text
     */
    async summarizeText(text, options = {}) {
        return await this.request('summarize_text', {
            text,
            model_id: options.modelId,
            max_length: options.maxLength || 150,
            min_length: options.minLength || 30
        });
    }

    /**
     * Answer questions
     */
    async answerQuestion(question, context, options = {}) {
        return await this.request('answer_question', {
            question,
            context,
            model_id: options.modelId
        });
    }

    // ============================================
    // AUDIO PROCESSING METHODS
    // ============================================

    /**
     * Transcribe audio to text
     */
    async transcribeAudio(audioData, options = {}) {
        return await this.request('transcribe_audio', {
            audio: audioData,
            model_id: options.modelId,
            language: options.language
        });
    }

    /**
     * Classify audio content
     */
    async classifyAudio(audioData, options = {}) {
        return await this.request('classify_audio', {
            audio: audioData,
            model_id: options.modelId
        });
    }

    /**
     * Synthesize speech from text
     */
    async synthesizeSpeech(text, options = {}) {
        return await this.request('synthesize_speech', {
            text,
            model_id: options.modelId,
            voice: options.voice,
            speed: options.speed || 1.0
        });
    }

    /**
     * Generate audio
     */
    async generateAudio(prompt, options = {}) {
        return await this.request('generate_audio', {
            prompt,
            model_id: options.modelId,
            duration: options.duration || 10
        });
    }

    // ============================================
    // VISION PROCESSING METHODS
    // ============================================

    /**
     * Classify images
     */
    async classifyImage(imageData, options = {}) {
        return await this.request('classify_image', {
            image: imageData,
            model_id: options.modelId,
            top_k: options.topK || 5
        });
    }

    /**
     * Detect objects in images
     */
    async detectObjects(imageData, options = {}) {
        return await this.request('detect_objects', {
            image: imageData,
            model_id: options.modelId,
            threshold: options.threshold || 0.5
        });
    }

    /**
     * Segment images
     */
    async segmentImage(imageData, options = {}) {
        return await this.request('segment_image', {
            image: imageData,
            model_id: options.modelId
        });
    }

    /**
     * Generate images
     */
    async generateImage(prompt, options = {}) {
        return await this.request('generate_image', {
            prompt,
            model_id: options.modelId,
            width: options.width || 512,
            height: options.height || 512,
            num_inference_steps: options.steps || 50,
            guidance_scale: options.guidanceScale || 7.5
        });
    }

    // ============================================
    // MULTIMODAL PROCESSING METHODS
    // ============================================

    /**
     * Generate image captions
     */
    async generateImageCaption(imageData, options = {}) {
        return await this.request('generate_image_caption', {
            image: imageData,
            model_id: options.modelId,
            max_length: options.maxLength || 50
        });
    }

    /**
     * Answer questions about images
     */
    async answerVisualQuestion(imageData, question, options = {}) {
        return await this.request('answer_visual_question', {
            image: imageData,
            question,
            model_id: options.modelId
        });
    }

    /**
     * Process documents
     */
    async processDocument(documentData, options = {}) {
        return await this.request('process_document', {
            document: documentData,
            model_id: options.modelId,
            task: options.task || 'extraction'
        });
    }

    // ============================================
    // SPECIALIZED METHODS
    // ============================================

    /**
     * Predict time series
     */
    async predictTimeseries(data, options = {}) {
        return await this.request('predict_timeseries', {
            data,
            model_id: options.modelId,
            prediction_length: options.predictionLength || 10
        });
    }

    /**
     * Generate code
     */
    async generateCode(description, options = {}) {
        return await this.request('generate_code', {
            description,
            model_id: options.modelId,
            language: options.language || 'python',
            max_length: options.maxLength || 200
        });
    }

    /**
     * Process tabular data
     */
    async processTabularData(data, options = {}) {
        return await this.request('process_tabular_data', {
            data,
            model_id: options.modelId,
            task: options.task || 'classification'
        });
    }

    // ============================================
    // SYSTEM METHODS
    // ============================================

    /**
     * List all available methods
     */
    async listMethods() {
        return await this.request('list_methods');
    }

    /**
     * Get server information
     */
    async getServerInfo() {
        return await this.request('get_server_info');
    }

    // ============================================
    // UTILITY METHODS
    // ============================================

    /**
     * Check if the server is available
     */
    async ping() {
        try {
            await this.getServerInfo();
            return true;
        } catch (error) {
            return false;
        }
    }

    /**
     * Wait for server to be available
     */
    async waitForServer(maxAttempts = 10, delay = 1000) {
        for (let i = 0; i < maxAttempts; i++) {
            if (await this.ping()) {
                return true;
            }
            await this._delay(delay);
        }
        return false;
    }
}

/**
 * MCP Error class for handling JSON-RPC errors
 */
class MCPError extends Error {
    constructor(code, message, data = null) {
        super(message);
        this.name = 'MCPError';
        this.code = code;
        this.data = data;
    }

    static fromJSONRPC(error) {
        return new MCPError(error.code, error.message, error.data);
    }
}

/**
 * MCP Client Factory for easy initialization
 */
class MCPClientFactory {
    static create(endpoint = '/jsonrpc', options = {}) {
        return new MCPClient(endpoint, options);
    }

    static createWithAuth(endpoint = '/jsonrpc', apiKey, options = {}) {
        const authOptions = {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${apiKey}`
            }
        };
        return new MCPClient(endpoint, authOptions);
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MCPClient, MCPError, MCPClientFactory };
} else if (typeof window !== 'undefined') {
    window.MCPClient = MCPClient;
    window.MCPError = MCPError;
    window.MCPClientFactory = MCPClientFactory;
}
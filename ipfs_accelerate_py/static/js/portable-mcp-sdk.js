/**
 * Portable MCP SDK
 * 
 * A highly portable, framework-agnostic JavaScript SDK for communicating with 
 * MCP (Model Context Protocol) servers. Designed for maximum reusability across
 * different environments and applications.
 * 
 * Features:
 * - Zero dependencies
 * - TypeScript-like type safety with JSDoc
 * - Built-in retry logic and error handling
 * - Modular architecture
 * - Support for batch operations
 * - Real-time event system
 * - Hardware detection integration
 * - IPFS Accelerate Py library integration
 * 
 * @version 2.0.0
 * @author IPFS Accelerate AI Team
 */

(function(global) {
    'use strict';

    /**
     * @typedef {Object} MCPConfig
     * @property {string} endpoint - The JSON-RPC endpoint URL
     * @property {number} timeout - Request timeout in milliseconds
     * @property {number} retries - Number of retry attempts
     * @property {boolean} enableLogging - Enable debug logging
     * @property {Object} headers - Additional headers to send with requests
     */

    /**
     * @typedef {Object} MCPRequest
     * @property {string} jsonrpc - JSON-RPC version (always "2.0")
     * @property {string} method - Method name to call
     * @property {Object} params - Method parameters
     * @property {number|string} id - Request identifier
     */

    /**
     * @typedef {Object} MCPResponse
     * @property {string} jsonrpc - JSON-RPC version
     * @property {*} result - Success result
     * @property {Object} error - Error object (if any)
     * @property {number|string} id - Request identifier
     */

    /**
     * @typedef {Object} MCPError
     * @property {number} code - Error code
     * @property {string} message - Error message
     * @property {*} data - Additional error data
     */

    /**
     * Custom MCP Error class
     */
    class MCPError extends Error {
        /**
         * @param {number} code - Error code
         * @param {string} message - Error message
         * @param {*} data - Additional error data
         */
        constructor(code, message, data = null) {
            super(message);
            this.name = 'MCPError';
            this.code = code;
            this.data = data;
        }
    }

    /**
     * Event Emitter for real-time communication
     */
    class EventEmitter {
        constructor() {
            this.events = {};
        }

        /**
         * @param {string} event 
         * @param {Function} listener 
         */
        on(event, listener) {
            if (!this.events[event]) {
                this.events[event] = [];
            }
            this.events[event].push(listener);
        }

        /**
         * @param {string} event 
         * @param {Function} listener 
         */
        off(event, listener) {
            if (!this.events[event]) return;
            this.events[event] = this.events[event].filter(l => l !== listener);
        }

        /**
         * @param {string} event 
         * @param {...*} args 
         */
        emit(event, ...args) {
            if (!this.events[event]) return;
            this.events[event].forEach(listener => {
                try {
                    listener.apply(this, args);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Portable MCP Client SDK
     */
    class PortableMCPSDK extends EventEmitter {
        /**
         * @param {MCPConfig} config - Configuration options
         */
        constructor(config = {}) {
            super();
            
            // Default configuration
            this.config = {
                endpoint: '/jsonrpc',
                timeout: 30000,
                retries: 3,
                enableLogging: false,
                headers: {
                    'Content-Type': 'application/json'
                },
                ...config
            };

            // Internal state
            this.requestId = 0;
            this.pendingRequests = new Map();
            this.connectionState = 'disconnected';
            this.serverInfo = null;
            this.hardwareInfo = null;
            this.metrics = {
                totalRequests: 0,
                successfulRequests: 0,
                failedRequests: 0,
                averageLatency: 0,
                lastRequestTime: null
            };

            // Start connection monitoring
            this._startConnectionMonitoring();
        }

        /**
         * Generate a unique request ID
         * @returns {number}
         */
        _generateRequestId() {
            return ++this.requestId;
        }

        /**
         * Log debug messages if logging is enabled
         * @param {...*} args 
         */
        _log(...args) {
            if (this.config.enableLogging) {
                console.log('[PortableMCPSDK]', ...args);
            }
        }

        /**
         * Make an HTTP request with retry logic
         * @param {MCPRequest|MCPRequest[]} body 
         * @returns {Promise<MCPResponse|MCPResponse[]>}
         */
        async _makeHttpRequest(body) {
            const startTime = Date.now();
            let lastError;

            for (let attempt = 0; attempt < this.config.retries; attempt++) {
                try {
                    this._log(`Attempt ${attempt + 1}/${this.config.retries}:`, body);
                    
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

                    const response = await fetch(this.config.endpoint, {
                        method: 'POST',
                        headers: this.config.headers,
                        body: JSON.stringify(body),
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    // Parse JSON more defensively to surface truncated responses
                    const text = await response.text();
                    let result;
                    try {
                        result = JSON.parse(text);
                    } catch (e) {
                        console.error('[PortableMCPSDK] Failed to parse JSON response:', e, 'Raw:', text?.slice(0, 200));
                        throw e;
                    }
                    
                    // Update metrics
                    const latency = Date.now() - startTime;
                    this._updateMetrics(true, latency);
                    
                    this._log('Response:', result);
                    return result;

                } catch (error) {
                    lastError = error;
                    this._log(`Attempt ${attempt + 1} failed:`, error.message);
                    
                    if (attempt === this.config.retries - 1) {
                        this._updateMetrics(false, Date.now() - startTime);
                    }
                    
                    // Wait before retrying (exponential backoff)
                    if (attempt < this.config.retries - 1) {
                        await this._delay(Math.pow(2, attempt) * 1000);
                    }
                }
            }

            throw new MCPError(-32603, `Request failed after ${this.config.retries} attempts: ${lastError.message}`, lastError);
        }

        /**
         * Delay execution
         * @param {number} ms 
         * @returns {Promise<void>}
         */
        _delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        /**
         * Update request metrics
         * @param {boolean} success 
         * @param {number} latency 
         */
        _updateMetrics(success, latency) {
            this.metrics.totalRequests++;
            this.metrics.lastRequestTime = Date.now();
            
            if (success) {
                this.metrics.successfulRequests++;
            } else {
                this.metrics.failedRequests++;
            }

            // Update average latency (moving average)
            this.metrics.averageLatency = (this.metrics.averageLatency + latency) / 2;
            
            this.emit('metricsUpdated', this.metrics);
        }

        /**
         * Start connection monitoring
         */
        _startConnectionMonitoring() {
            const checkConnection = async () => {
                try {
                    await this.ping();
                    if (this.connectionState !== 'connected') {
                        this.connectionState = 'connected';
                        this.emit('connected');
                    }
                } catch (error) {
                    if (this.connectionState !== 'disconnected') {
                        this.connectionState = 'disconnected';
                        this.emit('disconnected', error);
                    }
                }
            };

            // Initial check
            checkConnection();
            
            // Periodic checks
            setInterval(checkConnection, 30000); // Every 30 seconds
        }

        // ============================================
        // CORE API METHODS
        // ============================================

        /**
         * Make a JSON-RPC request
         * @param {string} method 
         * @param {Object} params 
         * @returns {Promise<*>}
         */
        async request(method, params = {}) {
            const id = this._generateRequestId();
            const requestBody = {
                jsonrpc: "2.0",
                method: method,
                params: params,
                id: id
            };

            try {
                const response = await this._makeHttpRequest(requestBody);
                
                if (response.error) {
                    throw new MCPError(response.error.code, response.error.message, response.error.data);
                }
                
                this.emit('requestSuccess', { method, params, result: response.result });
                return response.result;

            } catch (error) {
                this.emit('requestError', { method, params, error });
                throw error;
            }
        }

        /**
         * Make a notification (fire-and-forget)
         * @param {string} method 
         * @param {Object} params 
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
         * @param {Array<{method: string, params?: Object}>} requests 
         * @returns {Promise<Array<{result?: *, error?: MCPError}>>}
         */
        async batch(requests) {
            const requestBodies = requests.map(({ method, params }) => ({
                jsonrpc: "2.0",
                method: method,
                params: params || {},
                id: this._generateRequestId()
            }));

            const responses = await this._makeHttpRequest(requestBodies);
            
            return responses.map(response => {
                if (response.error) {
                    return { error: new MCPError(response.error.code, response.error.message, response.error.data) };
                }
                return { result: response.result };
            });
        }

        // ============================================
        // SERVER MANAGEMENT
        // ============================================

        /**
         * Ping the server
         * @returns {Promise<string>}
         */
        async ping() {
            return await this.request('ping');
        }

        /**
         * Get server information
         * @returns {Promise<Object>}
         */
        async getServerInfo() {
            if (!this.serverInfo) {
                this.serverInfo = await this.request('get_server_info');
            }
            return this.serverInfo;
        }

        /**
         * Get available methods
         * @returns {Promise<Array<string>>}
         */
        async getAvailableMethods() {
            return await this.request('get_available_methods');
        }

        /**
         * Wait for server to be available
         * @param {number} maxAttempts 
         * @param {number} interval 
         * @returns {Promise<boolean>}
         */
        async waitForServer(maxAttempts = 10, interval = 2000) {
            for (let i = 0; i < maxAttempts; i++) {
                try {
                    await this.ping();
                    return true;
                } catch (error) {
                    if (i === maxAttempts - 1) {
                        return false;
                    }
                    await this._delay(interval);
                }
            }
            return false;
        }

        // ============================================
        // HARDWARE & SYSTEM INTEGRATION
        // ============================================

        /**
         * Get hardware information (IPFS Accelerate Py integration)
         * @returns {Promise<Object>}
         */
        async getHardwareInfo() {
            if (!this.hardwareInfo) {
                try {
                    this.hardwareInfo = await this.request('get_hardware_info');
                } catch (error) {
                    this._log('Hardware info not available:', error.message);
                    this.hardwareInfo = { error: 'Hardware detection not available' };
                }
            }
            return this.hardwareInfo;
        }

        /**
         * Get system metrics
         * @returns {Promise<Object>}
         */
        async getSystemMetrics() {
            try {
                return await this.request('get_system_metrics');
            } catch (error) {
                this._log('System metrics not available:', error.message);
                return { error: 'System metrics not available' };
            }
        }

        /**
         * Get model list
         * @returns {Promise<Array>}
         */
        async getModels() {
            try {
                return await this.request('get_models');
            } catch (error) {
                this._log('Model list not available:', error.message);
                return [];
            }
        }

        // ============================================
        // AI INFERENCE MODULES
        // ============================================

        /**
         * Text Generation Module
         */
        get text() {
            return {
                generate: (prompt, options = {}) => this.request('generate_text', { prompt, ...options }),
                classify: (text, labels = []) => this.request('classify_text', { text, labels }),
                summarize: (text, maxLength = 150) => this.request('summarize_text', { text, max_length: maxLength }),
                translate: (text, targetLang, sourceLang = 'auto') => this.request('translate_text', { text, target_language: targetLang, source_language: sourceLang }),
                embed: (text) => this.request('get_text_embedding', { text }),
                sentiment: (text) => this.request('analyze_sentiment', { text }),
                extract: (text, entityTypes = []) => this.request('extract_entities', { text, entity_types: entityTypes })
            };
        }

        /**
         * Computer Vision Module
         */
        get vision() {
            return {
                classify: (imageData) => this.request('classify_image', { image_data: imageData }),
                detect: (imageData) => this.request('detect_objects', { image_data: imageData }),
                generate: (prompt, options = {}) => this.request('generate_image', { prompt, ...options }),
                caption: (imageData) => this.request('caption_image', { image_data: imageData }),
                ocr: (imageData) => this.request('extract_text_from_image', { image_data: imageData }),
                segment: (imageData) => this.request('segment_image', { image_data: imageData }),
                enhance: (imageData, options = {}) => this.request('enhance_image', { image_data: imageData, ...options })
            };
        }

        /**
         * Audio Processing Module
         */
        get audio() {
            return {
                transcribe: (audioData, language = 'auto') => this.request('transcribe_audio', { audio_data: audioData, language }),
                synthesize: (text, voice = 'default') => this.request('synthesize_speech', { text, voice }),
                classify: (audioData) => this.request('classify_audio', { audio_data: audioData }),
                generate: (prompt, options = {}) => this.request('generate_audio', { prompt, ...options }),
                denoise: (audioData) => this.request('denoise_audio', { audio_data: audioData })
            };
        }

        /**
         * Code Generation Module
         */
        get code() {
            return {
                generate: (prompt, language = 'python') => this.request('generate_code', { prompt, language }),
                complete: (code, language = 'python') => this.request('complete_code', { code, language }),
                explain: (code, language = 'python') => this.request('explain_code', { code, language }),
                debug: (code, error, language = 'python') => this.request('debug_code', { code, error, language }),
                optimize: (code, language = 'python') => this.request('optimize_code', { code, language })
            };
        }

        /**
         * Multimodal AI Module
         */
        get multimodal() {
            return {
                vqa: (imageData, question) => this.request('visual_question_answering', { image_data: imageData, question }),
                chat: (messages, imageData = null) => this.request('multimodal_chat', { messages, image_data: imageData }),
                analyze: (data, dataType = 'auto') => this.request('multimodal_analysis', { data, data_type: dataType })
            };
        }

        // ============================================
        // UTILITY METHODS
        // ============================================

        /**
         * Get current metrics
         * @returns {Object}
         */
        getMetrics() {
            return { ...this.metrics };
        }

        /**
         * Reset metrics
         */
        resetMetrics() {
            this.metrics = {
                totalRequests: 0,
                successfulRequests: 0,
                failedRequests: 0,
                averageLatency: 0,
                lastRequestTime: null
            };
            this.emit('metricsReset');
        }

        /**
         * Get connection state
         * @returns {string}
         */
        getConnectionState() {
            return this.connectionState;
        }

        /**
         * Update configuration
         * @param {Partial<MCPConfig>} newConfig 
         */
        updateConfig(newConfig) {
            this.config = { ...this.config, ...newConfig };
            this.emit('configUpdated', this.config);
        }

        /**
         * Export client state and metrics
         * @returns {Object}
         */
        exportState() {
            return {
                config: this.config,
                metrics: this.metrics,
                connectionState: this.connectionState,
                serverInfo: this.serverInfo,
                hardwareInfo: this.hardwareInfo,
                timestamp: Date.now()
            };
        }

        /**
         * Create a specialized client for a specific module
         * @param {string} moduleName 
         * @returns {Object}
         */
        createModuleClient(moduleName) {
            const baseClient = this;
            
            return {
                async call(method, params = {}) {
                    return await baseClient.request(`${moduleName}_${method}`, params);
                },
                
                async batch(calls) {
                    const requests = calls.map(({ method, params }) => ({
                        method: `${moduleName}_${method}`,
                        params: params || {}
                    }));
                    return await baseClient.batch(requests);
                },
                
                get metrics() {
                    return baseClient.getMetrics();
                },
                
                on: baseClient.on.bind(baseClient),
                off: baseClient.off.bind(baseClient)
            };
        }
    }

    // ============================================
    // FACTORY FUNCTIONS
    // ============================================

    /**
     * Create a new MCP SDK instance
     * @param {MCPConfig} config 
     * @returns {PortableMCPSDK}
     */
    function createMCPClient(config = {}) {
        return new PortableMCPSDK(config);
    }

    /**
     * Create a client optimized for specific use cases
     * @param {string} preset - 'development', 'production', 'embedded'
     * @param {Partial<MCPConfig>} customConfig 
     * @returns {PortableMCPSDK}
     */
    function createPresetClient(preset = 'development', customConfig = {}) {
        const presets = {
            development: {
                enableLogging: true,
                timeout: 60000,
                retries: 1
            },
            production: {
                enableLogging: false,
                timeout: 30000,
                retries: 3
            },
            embedded: {
                enableLogging: false,
                timeout: 10000,
                retries: 1
            }
        };

        const config = { ...presets[preset], ...customConfig };
        return new PortableMCPSDK(config);
    }

    // ============================================
    // BROWSER COMPATIBILITY LAYER
    // ============================================

    /**
     * File upload helper for browser environments
     * @param {File} file 
     * @returns {Promise<string>}
     */
    async function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    /**
     * Download helper for browser environments
     * @param {*} data 
     * @param {string} filename 
     * @param {string} type 
     */
    function downloadData(data, filename, type = 'application/json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    // ============================================
    // EXPORT/GLOBAL REGISTRATION
    // ============================================

    // Create the main SDK object
    const PortableMCP = {
        // Classes
        SDK: PortableMCPSDK,
        Error: MCPError,
        EventEmitter: EventEmitter,
        
        // Factory functions
        create: createMCPClient,
        createPreset: createPresetClient,
        
        // Utilities
        fileToBase64: fileToBase64,
        downloadData: downloadData,
        
        // Version
        version: '2.0.0'
    };

    // Register globally
    if (typeof module !== 'undefined' && module.exports) {
        // Node.js environment
        module.exports = PortableMCP;
    } else if (typeof define === 'function' && define.amd) {
        // AMD environment
        define([], function() {
            return PortableMCP;
        });
    } else {
        // Browser environment
        global.PortableMCP = PortableMCP;
        
        // Legacy compatibility
        global.MCPClient = PortableMCPSDK;
        global.createMCPClient = createMCPClient;
    }

})(typeof window !== 'undefined' ? window : global);
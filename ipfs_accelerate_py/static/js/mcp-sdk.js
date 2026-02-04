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
    // MCP TOOL METHODS
    // ============================================

    /**
     * Call any MCP tool by name
     */
    async callTool(toolName, args = {}) {
        return await this.request('tools/call', {
            name: toolName,
            arguments: args
        });
    }

    /**
     * Call multiple MCP tools in a batch
     * @param {Array<{name: string, arguments: Object}>} tools - Array of tool calls
     * @returns {Promise<Array<{result?: any, error?: MCPError}>>}
     * 
     * @example
     * const results = await client.callToolsBatch([
     *   { name: 'github_list_repos', arguments: { owner: 'octocat' } },
     *   { name: 'docker_list_containers', arguments: { all: true } }
     * ]);
     */
    async callToolsBatch(tools) {
        const requests = tools.map(tool => ({
            method: 'tools/call',
            params: {
                name: tool.name,
                arguments: tool.arguments || {}
            }
        }));
        
        return await this.batch(requests);
    }

    // GitHub Tools
    async githubListRepos(owner = null, limit = 30) {
        return await this.callTool('github_list_repos', { owner, limit });
    }

    async githubGetRepo(owner, repo) {
        return await this.callTool('github_get_repo', { owner, repo });
    }

    async githubListPrs(owner, repo, state = 'open') {
        return await this.callTool('github_list_prs', { owner, repo, state });
    }

    async githubGetPr(owner, repo, pr_number) {
        return await this.callTool('github_get_pr', { owner, repo, pr_number });
    }

    async githubListIssues(owner, repo, state = 'open') {
        return await this.callTool('github_list_issues', { owner, repo, state });
    }

    async githubGetIssue(owner, repo, issue_number) {
        return await this.callTool('github_get_issue', { owner, repo, issue_number });
    }

    // Docker Tools
    async dockerRunContainer(image, command = null, env = null) {
        return await this.callTool('docker_run_container', { image, command, env });
    }

    async dockerListContainers(all = false) {
        return await this.callTool('docker_list_containers', { all });
    }

    async dockerStopContainer(container_id) {
        return await this.callTool('docker_stop_container', { container_id });
    }

    async dockerPullImage(image) {
        return await this.callTool('docker_pull_image', { image });
    }

    // Hardware Tools
    async hardwareGetInfo() {
        return await this.callTool('hardware_get_info', {});
    }

    async hardwareTest() {
        return await this.callTool('hardware_test', {});
    }

    async hardwareRecommend(task_type) {
        return await this.callTool('hardware_recommend', { task_type });
    }

    // Runner Tools
    async runnerStartAutoscaler(owner, interval = 60) {
        return await this.callTool('runner_start_autoscaler', { owner, interval });
    }

    async runnerStopAutoscaler() {
        return await this.callTool('runner_stop_autoscaler', {});
    }

    async runnerGetStatus() {
        return await this.callTool('runner_get_status', {});
    }

    async runnerListWorkflows(owner, repo) {
        return await this.callTool('runner_list_workflows', { owner, repo });
    }

    async runnerProvisionForWorkflow(workflow_id, runner_type = 'ubuntu-latest') {
        return await this.callTool('runner_provision_for_workflow', { workflow_id, runner_type });
    }

    async runnerListContainers() {
        return await this.callTool('runner_list_containers', {});
    }

    async runnerStopContainer(container_id) {
        return await this.callTool('runner_stop_container', { container_id });
    }

    // IPFS Files Tools
    async ipfsFilesAdd(path, content) {
        return await this.callTool('ipfs_files_add', { path, content });
    }

    async ipfsFilesGet(cid, output_path = null) {
        return await this.callTool('ipfs_files_get', { cid, output_path });
    }

    async ipfsFilesCat(cid) {
        return await this.callTool('ipfs_files_cat', { cid });
    }

    async ipfsFilesPin(cid) {
        return await this.callTool('ipfs_files_pin', { cid });
    }

    async ipfsFilesUnpin(cid) {
        return await this.callTool('ipfs_files_unpin', { cid });
    }

    async ipfsFilesList(path = '/') {
        return await this.callTool('ipfs_files_list', { path });
    }

    async ipfsFilesValidateCid(cid) {
        return await this.callTool('ipfs_files_validate_cid', { cid });
    }

    // Network Tools
    async networkListPeers() {
        return await this.callTool('network_list_peers', {});
    }

    async networkConnectPeer(peer_address) {
        return await this.callTool('network_connect_peer', { peer_address });
    }

    async networkDisconnectPeer(peer_id) {
        return await this.callTool('network_disconnect_peer', { peer_id });
    }

    async networkDhtPut(key, value) {
        return await this.callTool('network_dht_put', { key, value });
    }

    async networkDhtGet(key) {
        return await this.callTool('network_dht_get', { key });
    }

    async networkGetSwarmInfo() {
        return await this.callTool('network_get_swarm_info', {});
    }

    async networkGetBandwidth() {
        return await this.callTool('network_get_bandwidth', {});
    }

    async networkPingPeer(peer_id) {
        return await this.callTool('network_ping_peer', { peer_id });
    }

    async checkNetworkStatus() {
        return await this.callTool('check_network_status', {});
    }

    async getNetworkStatus() {
        return await this.callTool('get_network_status', {});
    }

    async getConnectedPeers() {
        return await this.callTool('get_connected_peers', {});
    }

    // Advanced IPFS Operations
    async ipfsCat(cid) {
        return await this.callTool('ipfs_cat', { cid });
    }

    async ipfsLs(path = '/') {
        return await this.callTool('ipfs_ls', { path });
    }

    async ipfsMkdir(path) {
        return await this.callTool('ipfs_mkdir', { path });
    }

    async ipfsAddFile(path, content) {
        return await this.callTool('ipfs_add_file', { path, content });
    }

    async ipfsPinAdd(cid) {
        return await this.callTool('ipfs_pin_add', { cid });
    }

    async ipfsPinRm(cid) {
        return await this.callTool('ipfs_pin_rm', { cid });
    }

    async ipfsSwarmPeers() {
        return await this.callTool('ipfs_swarm_peers', {});
    }

    async ipfsSwarmConnect(address) {
        return await this.callTool('ipfs_swarm_connect', { address });
    }

    async ipfsId() {
        return await this.callTool('ipfs_id', {});
    }

    async ipfsDhtFindpeer(peer_id) {
        return await this.callTool('ipfs_dht_findpeer', { peer_id });
    }

    async ipfsDhtFindprovs(cid) {
        return await this.callTool('ipfs_dht_findprovs', { cid });
    }

    async ipfsPubsubPub(topic, message) {
        return await this.callTool('ipfs_pubsub_pub', { topic, message });
    }

    async ipfsFilesRead(path) {
        return await this.callTool('ipfs_files_read', { path });
    }

    async ipfsFilesWrite(path, content) {
        return await this.callTool('ipfs_files_write', { path, content });
    }

    async addFile(path, content, options = {}) {
        return await this.callTool('add_file', { path, content, ...options });
    }

    async addFileShared(path, content, options = {}) {
        return await this.callTool('add_file_shared', { path, content, ...options });
    }

    async addFileToIpfs(content, options = {}) {
        return await this.callTool('add_file_to_ipfs', { content, ...options });
    }

    async getFileFromIpfs(cid, output_path = null) {
        return await this.callTool('get_file_from_ipfs', { cid, output_path });
    }

    // Endpoint Management Tools
    async getEndpoint(endpoint_id) {
        return await this.callTool('get_endpoint', { endpoint_id });
    }

    async getEndpoints() {
        return await this.callTool('get_endpoints', {});
    }

    async addEndpoint(config) {
        return await this.callTool('add_endpoint', config);
    }

    async getEndpointDetails(endpoint_id) {
        return await this.callTool('get_endpoint_details', { endpoint_id });
    }

    async getEndpointStatus(endpoint_id) {
        return await this.callTool('get_endpoint_status', { endpoint_id });
    }

    async getEndpointHandlersByModel(model_id) {
        return await this.callTool('get_endpoint_handlers_by_model', { model_id });
    }

    async configureApiProvider(provider, config) {
        return await this.callTool('configure_api_provider', { provider, config });
    }

    async listCliEndpoints() {
        return await this.callTool('list_cli_endpoints_tool', {});
    }

    async listCliEndpointsTool() {
        return await this.callTool('list_cli_endpoints_tool', {});
    }

    // Status and Health Tools
    async getServerStatus() {
        return await this.callTool('get_server_status', {});
    }

    async getSystemStatus() {
        return await this.callTool('get_system_status', {});
    }

    async getQueueStatus() {
        return await this.callTool('get_queue_status', {});
    }

    async getQueueHistory(limit = 100) {
        return await this.callTool('get_queue_history', { limit });
    }

    async getPerformanceMetrics() {
        return await this.callTool('get_performance_metrics', {});
    }

    // Dashboard Data Tools
    async getDashboardCacheStats() {
        return await this.callTool('get_dashboard_cache_stats', {});
    }

    async getDashboardPeerStatus() {
        return await this.callTool('get_dashboard_peer_status', {});
    }

    async getDashboardSystemMetrics() {
        return await this.callTool('get_dashboard_system_metrics', {});
    }

    async getDashboardUserInfo() {
        return await this.callTool('get_dashboard_user_info', {});
    }

    // Workflow Management Tools
    async createWorkflow(config) {
        return await this.callTool('create_workflow', config);
    }

    async deleteWorkflow(workflow_id) {
        return await this.callTool('delete_workflow', { workflow_id });
    }

    async getWorkflow(workflow_id) {
        return await this.callTool('get_workflow', { workflow_id });
    }

    async listWorkflows() {
        return await this.callTool('list_workflows', {});
    }

    async getWorkflowTemplates() {
        return await this.callTool('get_workflow_templates', {});
    }

    async createWorkflowFromTemplate(template_id, params = {}) {
        return await this.callTool('create_workflow_from_template', { template_id, ...params });
    }

    // GitHub Workflows (Advanced)
    async ghGetAuthStatus() {
        return await this.callTool('gh_get_auth_status', {});
    }

    async ghListRunners(owner, repo) {
        return await this.callTool('gh_list_runners', { owner, repo });
    }

    async ghGetRunnerLabels(owner, repo) {
        return await this.callTool('gh_get_runner_labels', { owner, repo });
    }

    async ghListWorkflowRuns(owner, repo, workflow_id = null) {
        return await this.callTool('gh_list_workflow_runs', { owner, repo, workflow_id });
    }

    async ghCreateWorkflowQueues(owner, repo, config = {}) {
        return await this.callTool('gh_create_workflow_queues', { owner, repo, ...config });
    }

    async ghGetCacheStats(owner, repo) {
        return await this.callTool('gh_get_cache_stats', { owner, repo });
    }

    // Model Management (Extended)
    async getModelDetails(model_id) {
        return await this.callTool('get_model_details', { model_id });
    }

    async getModelList(filter = {}) {
        return await this.callTool('get_model_list', filter);
    }

    async getModelStats(model_id = null) {
        return await this.callTool('get_model_stats', { model_id });
    }

    async getModelQueues() {
        return await this.callTool('get_model_queues', {});
    }

    async downloadModel(model_id, options = {}) {
        return await this.callTool('download_model', { model_id, ...options });
    }

    async listAvailableModels(filter = {}) {
        return await this.callTool('list_available_models', filter);
    }

    async ipfsAccelerateModel(model_id, config = {}) {
        return await this.callTool('ipfs_accelerate_model', { model_id, ...config });
    }

    async ipfsBenchmarkModel(model_id, config = {}) {
        return await this.callTool('ipfs_benchmark_model', { model_id, ...config });
    }

    async ipfsModelStatus(model_id) {
        return await this.callTool('ipfs_model_status', { model_id });
    }

    async ipfsGetHardwareInfo() {
        return await this.callTool('ipfs_get_hardware_info', {});
    }

    // CLI and Configuration Tools
    async getCliCapabilities() {
        return await this.callTool('get_cli_capabilities', {});
    }

    async getCliConfig() {
        return await this.callTool('get_cli_config', {});
    }

    async getCliInstall() {
        return await this.callTool('get_cli_install', {});
    }

    async getCliProviders() {
        return await this.callTool('get_cli_providers', {});
    }

    async checkCliVersion() {
        return await this.callTool('check_cli_version', {});
    }

    async getDistributedCapabilities() {
        return await this.callTool('get_distributed_capabilities', {});
    }

    // Copilot SDK Tools
    async copilotSdkCreateSession(config = {}) {
        return await this.callTool('copilot_sdk_create_session', config);
    }

    async copilotSdkDestroySession(session_id) {
        return await this.callTool('copilot_sdk_destroy_session', { session_id });
    }

    async copilotSdkListSessions() {
        return await this.callTool('copilot_sdk_list_sessions', {});
    }

    async copilotSdkGetTools() {
        return await this.callTool('copilot_sdk_get_tools', {});
    }

    async copilotSdkSendMessage(session_id, message) {
        return await this.callTool('copilot_sdk_send_message', { session_id, message });
    }

    async copilotSdkStreamMessage(session_id, message) {
        return await this.callTool('copilot_sdk_stream_message', { session_id, message });
    }

    async copilotSuggestCommand(description, context = {}) {
        return await this.callTool('copilot_suggest_command', { description, ...context });
    }

    async copilotSuggestGitCommand(description, context = {}) {
        return await this.callTool('copilot_suggest_git_command', { description, ...context });
    }

    async copilotExplainCommand(command) {
        return await this.callTool('copilot_explain_command', { command });
    }

    // Session Management
    async getSession(session_id) {
        return await this.callTool('get_session', { session_id });
    }

    async endSession(session_id) {
        return await this.callTool('end_session', { session_id });
    }

    // Logging and Operations
    async logOperation(operation, data = {}) {
        return await this.callTool('log_operation', { operation, ...data });
    }

    // Inference Tools (Extended)
    async cliInference(model, input, options = {}) {
        return await this.callTool('cli_inference', { model, input, ...options });
    }

    async runInference(model, inputs, options = {}) {
        return await this.callTool('run_inference', { model, inputs, ...options });
    }

    async runDistributedInference(model, inputs, options = {}) {
        return await this.callTool('run_distributed_inference', { model, inputs, ...options });
    }

    async multiplexInference(requests) {
        return await this.callTool('multiplex_inference', { requests });
    }

    async runModelTest(model_id, test_config = {}) {
        return await this.callTool('run_model_test', { model_id, ...test_config });
    }

    // Model Search and Recommendations
    async recommendModels(task_type, input_type = 'text', limit = 10) {
        return await this.callTool('recommend_models', { task_type, input_type, limit });
    }

    async searchHuggingfaceModels(query, filter = {}) {
        return await this.callTool('search_huggingface_models', { query, ...filter });
    }

    // Endpoint Management (Extended)
    async registerEndpoint(endpoint_config) {
        return await this.callTool('register_endpoint', endpoint_config);
    }

    async updateEndpoint(endpoint_id, updates) {
        return await this.callTool('update_endpoint', { endpoint_id, ...updates });
    }

    async removeEndpoint(endpoint_id) {
        return await this.callTool('remove_endpoint', { endpoint_id });
    }

    async registerCliEndpoint(config) {
        return await this.callTool('register_cli_endpoint_tool', config);
    }

    async registerCliEndpointTool(config) {
        return await this.callTool('register_cli_endpoint_tool', config);
    }

    async validateCliConfig(config) {
        return await this.callTool('validate_cli_config', config);
    }

    // Workflow Control
    async startWorkflow(workflow_id, params = {}) {
        return await this.callTool('start_workflow', { workflow_id, ...params });
    }

    async stopWorkflow(workflow_id) {
        return await this.callTool('stop_workflow', { workflow_id });
    }

    async pauseWorkflow(workflow_id) {
        return await this.callTool('pause_workflow', { workflow_id });
    }

    async updateWorkflow(workflow_id, updates) {
        return await this.callTool('update_workflow', { workflow_id, ...updates });
    }

    // P2P Workflow Tools
    async p2pSubmitTask(task_config) {
        return await this.callTool('p2p_submit_task', task_config);
    }

    async p2pGetNextTask(worker_id, capabilities = {}) {
        return await this.callTool('p2p_get_next_task', { worker_id, ...capabilities });
    }

    async p2pMarkTaskComplete(task_id, result) {
        return await this.callTool('p2p_mark_task_complete', { task_id, result });
    }

    async p2pUpdatePeerState(peer_id, state) {
        return await this.callTool('p2p_update_peer_state', { peer_id, state });
    }

    async p2pSchedulerStatus() {
        return await this.callTool('p2p_scheduler_status', {});
    }

    async p2pCheckWorkflowTags(tags) {
        return await this.callTool('p2p_check_workflow_tags', { tags });
    }

    async p2pGetMerkleClock() {
        return await this.callTool('p2p_get_merkle_clock', {});
    }

    // Session Management (Extended)
    async startSession(config = {}) {
        return await this.callTool('start_session', config);
    }

    // Logging
    async logRequest(request_data) {
        return await this.callTool('log_request', request_data);
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
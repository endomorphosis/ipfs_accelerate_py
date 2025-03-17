import { VLLM } from '../vllm/vllm';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import {
  VllmRequest,
  VllmUnifiedResponse,
  VllmBatchResponse,
  VllmStreamChunk,
  VllmModelInfo,
  VllmModelStatistics,
  VllmLoraAdapter,
  VllmLoadLoraResponse,
  VllmQuantizationConfig,
  VllmQuantizationResponse,
  VllmContainerConfig,
  VllmContainerStatus
} from './types';
import * as child_process from 'child_process';
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';

/**
 * VLLM Unified API backend with container management
 * Extends the base VLLM API backend with Docker container management capabilities
 */
export class VllmUnified extends VLLM {
  // Docker container management
  private containerEnabled: boolean = false;
  private containerId: string | null = null;
  private containerConfig: VllmContainerConfig | null = null;
  private containerImageDefault: string = 'vllm/vllm-openai:latest';
  private containerStatus: VllmContainerStatus = 'stopped';
  private containerStartupTimeoutMs: number = 60000; // 60 seconds
  private containerHealthCheckIntervalMs: number = 5000; // 5 seconds
  private containerLogs: string[] = [];
  private containerMaxLogEntries: number = 100;
  private containerVolumeMounts: string[] = [];
  private containerEnvVars: Record<string, string> = {};
  private containerPorts: Record<string, string> = { '8000': '8000' };
  private containerProcess: any = null;
  private containerHealthCheckTimer: any = null;
  private containerStartupPromise: Promise<boolean> | null = null;
  private containerApiUrl: string = 'http://localhost:8000';
  private containerConfigPath: string = '';
  private containerModelsPath: string = '';
  private containerAutomaticRetryEnabled: boolean = true;
  private containerRetryCount: number = 0;
  private containerMaxRetries: number = 3;

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);

    // Parse container configuration from metadata
    this.containerEnabled = metadata.vllm_container_enabled === true || 
                           metadata.vllm_container === true || 
                           process.env.VLLM_CONTAINER_ENABLED === 'true';

    // Set container related properties
    this.containerImageDefault = metadata.vllm_container_image || 
                               process.env.VLLM_CONTAINER_IMAGE || 
                               this.containerImageDefault;

    this.containerApiUrl = metadata.vllm_api_url || 
                          metadata.vllmApiUrl || 
                          process.env.VLLM_API_URL || 
                          this.containerApiUrl;

    this.containerConfigPath = metadata.vllm_config_path || 
                              process.env.VLLM_CONFIG_PATH || 
                              path.join(os.homedir(), '.vllm');

    this.containerModelsPath = metadata.vllm_models_path || 
                              process.env.VLLM_MODELS_PATH || 
                              path.join(os.homedir(), '.vllm/models');

    // Configure container if enabled
    if (this.containerEnabled) {
      this.configureContainer(metadata);
    }
  }

  /**
   * Configure Docker container settings
   */
  private configureContainer(metadata: ApiMetadata) {
    // Check that we're in a Node.js environment
    if (typeof process === 'undefined' || typeof child_process === 'undefined') {
      console.warn('Container mode not supported in this environment');
      this.containerEnabled = false;
      return;
    }

    // Configure the container
    this.containerConfig = {
      image: metadata.vllm_container_image || process.env.VLLM_CONTAINER_IMAGE || this.containerImageDefault,
      gpu: metadata.vllm_container_gpu === true || process.env.VLLM_CONTAINER_GPU === 'true',
      models_path: metadata.vllm_models_path || process.env.VLLM_MODELS_PATH || this.containerModelsPath,
      config_path: metadata.vllm_config_path || process.env.VLLM_CONFIG_PATH || this.containerConfigPath,
      api_port: parseInt(metadata.vllm_api_port || process.env.VLLM_API_PORT || '8000', 10),
      tensor_parallel_size: parseInt(metadata.vllm_tensor_parallel_size || process.env.VLLM_TENSOR_PARALLEL_SIZE || '1', 10),
      max_model_len: parseInt(metadata.vllm_max_model_len || process.env.VLLM_MAX_MODEL_LEN || '0', 10),
      gpu_memory_utilization: parseFloat(metadata.vllm_gpu_memory_utilization || process.env.VLLM_GPU_MEMORY_UTILIZATION || '0.9'),
      quantization: metadata.vllm_quantization || process.env.VLLM_QUANTIZATION || null,
      trust_remote_code: metadata.vllm_trust_remote_code === true || process.env.VLLM_TRUST_REMOTE_CODE === 'true',
      custom_args: metadata.vllm_custom_args || process.env.VLLM_CUSTOM_ARGS || ''
    };

    // Configure Docker volume mounts for the container
    this.containerVolumeMounts = [
      `${this.containerConfig.models_path}:/models`,
      `${this.containerConfig.config_path}:/root/.vllm`
    ];

    // Add additional volume mounts if specified
    if (metadata.vllm_extra_mounts) {
      const extraMounts = Array.isArray(metadata.vllm_extra_mounts) 
        ? metadata.vllm_extra_mounts 
        : [metadata.vllm_extra_mounts];
      
      this.containerVolumeMounts = [...this.containerVolumeMounts, ...extraMounts];
    }

    // Configure container ports
    this.containerPorts = {
      [`${this.containerConfig.api_port}`]: '8000'
    };

    // Configure container environment variables
    this.containerEnvVars = {
      'VLLM_MODEL_PATH': metadata.vllm_model || 
                         process.env.VLLM_MODEL || 
                         this.defaultModel,
      'VLLM_TENSOR_PARALLEL_SIZE': this.containerConfig.tensor_parallel_size.toString(),
      'VLLM_GPU_MEMORY_UTILIZATION': this.containerConfig.gpu_memory_utilization.toString(),
      'VLLM_ENFORCE_EAGER': 'true' // Ensure eager mode for better stability
    };

    // Add quantization if specified
    if (this.containerConfig.quantization) {
      this.containerEnvVars['VLLM_QUANTIZATION'] = this.containerConfig.quantization;
    }

    // Set max model length if specified
    if (this.containerConfig.max_model_len > 0) {
      this.containerEnvVars['VLLM_MAX_MODEL_LEN'] = this.containerConfig.max_model_len.toString();
    }

    // Set trust remote code if specified
    if (this.containerConfig.trust_remote_code) {
      this.containerEnvVars['VLLM_TRUST_REMOTE_CODE'] = 'true';
    }

    // Ensure configuration directory exists
    try {
      if (!fs.existsSync(this.containerConfig.config_path)) {
        fs.mkdirSync(this.containerConfig.config_path, { recursive: true });
      }
      if (!fs.existsSync(this.containerConfig.models_path)) {
        fs.mkdirSync(this.containerConfig.models_path, { recursive: true });
      }
    } catch (error) {
      console.error('Failed to create container configuration directories:', error);
    }

    // Create a VLLM configuration file
    this.writeContainerConfigFile();
  }

  /**
   * Write container configuration file
   */
  private writeContainerConfigFile() {
    if (!this.containerConfig) return;

    try {
      const configFilePath = path.join(this.containerConfig.config_path, 'config.json');
      const config = {
        model: this.containerEnvVars['VLLM_MODEL_PATH'],
        tensor_parallel_size: this.containerConfig.tensor_parallel_size,
        gpu_memory_utilization: this.containerConfig.gpu_memory_utilization,
        trust_remote_code: this.containerConfig.trust_remote_code,
        max_model_len: this.containerConfig.max_model_len > 0 ? this.containerConfig.max_model_len : undefined,
        quantization: this.containerConfig.quantization || undefined
      };

      fs.writeFileSync(configFilePath, JSON.stringify(config, null, 2));
    } catch (error) {
      console.error('Failed to write container configuration file:', error);
    }
  }

  /**
   * Start the VLLM container
   */
  async startContainer(): Promise<boolean> {
    // Avoid duplicate startups
    if (this.containerStartupPromise) {
      return this.containerStartupPromise;
    }

    // Return early if container is already running
    if (this.containerStatus === 'running' && this.containerId) {
      return true;
    }

    // Create a promise for the startup process
    this.containerStartupPromise = new Promise<boolean>(async (resolve) => {
      try {
        // Check if Docker is available
        try {
          child_process.execSync('docker --version', { stdio: 'pipe' });
        } catch (error) {
          console.error('Docker is not available:', error);
          this.containerStatus = 'error';
          resolve(false);
          return;
        }

        // Update container status
        this.containerStatus = 'starting';
        this.logToContainer('Starting VLLM container...');

        // Check if the image exists, pull if not
        try {
          this.logToContainer(`Checking for container image: ${this.containerConfig?.image}`);
          const imageExists = child_process.execSync(`docker image inspect ${this.containerConfig?.image} 2>/dev/null`);
        } catch (error) {
          // Image doesn't exist, pull it
          this.logToContainer(`Pulling container image: ${this.containerConfig?.image}`);
          
          try {
            child_process.execSync(`docker pull ${this.containerConfig?.image}`, { stdio: 'pipe' });
          } catch (pullError) {
            console.error('Failed to pull container image:', pullError);
            this.containerStatus = 'error';
            resolve(false);
            return;
          }
        }

        // Prepare run command for Docker
        let runCommand = 'docker run -d --rm ';
        
        // Add GPU support if enabled
        if (this.containerConfig?.gpu) {
          runCommand += '--gpus all ';
        }
        
        // Add volume mounts
        for (const mount of this.containerVolumeMounts) {
          runCommand += `-v ${mount} `;
        }
        
        // Add port mappings
        for (const [host, container] of Object.entries(this.containerPorts)) {
          runCommand += `-p ${host}:${container} `;
        }
        
        // Add environment variables
        for (const [key, value] of Object.entries(this.containerEnvVars)) {
          runCommand += `-e ${key}="${value}" `;
        }
        
        // Add the image and any custom arguments
        runCommand += this.containerConfig?.image || this.containerImageDefault;
        
        if (this.containerConfig?.custom_args) {
          runCommand += ` ${this.containerConfig.custom_args}`;
        }

        // Start the container
        this.logToContainer(`Starting container with command: ${runCommand}`);
        const containerId = child_process.execSync(runCommand).toString().trim();
        this.containerId = containerId;
        
        // Start health checks
        this.startContainerHealthChecks();
        
        // Wait for the service to be ready
        this.logToContainer(`Container started with ID: ${containerId}`);
        this.logToContainer('Waiting for API to be ready...');

        // Wait for the container to be ready
        const startTime = Date.now();
        let isReady = false;
        
        while (Date.now() - startTime < this.containerStartupTimeoutMs) {
          try {
            // Check if the API is ready
            const response = await fetch(`${this.containerApiUrl}/v1/models`);
            if (response.ok) {
              isReady = true;
              break;
            }
          } catch (error) {
            // API not ready yet, wait and try again
            await new Promise(r => setTimeout(r, 1000));
          }
        }

        if (isReady) {
          this.containerStatus = 'running';
          this.logToContainer('Container API is ready');
          resolve(true);
        } else {
          this.containerStatus = 'error';
          this.logToContainer('Container startup timed out');
          
          // Kill the container if it's still running
          try {
            if (this.containerId) {
              child_process.execSync(`docker kill ${this.containerId}`);
            }
          } catch (error) {
            // Ignore errors during cleanup
          }
          
          resolve(false);
        }
      } catch (error) {
        console.error('Failed to start container:', error);
        this.containerStatus = 'error';
        resolve(false);
      } finally {
        // Clear the startup promise
        this.containerStartupPromise = null;
      }
    });

    return this.containerStartupPromise;
  }

  /**
   * Stop the VLLM container
   */
  async stopContainer(): Promise<boolean> {
    if (!this.containerId || this.containerStatus !== 'running') {
      return true;
    }

    try {
      this.containerStatus = 'stopping';
      this.logToContainer('Stopping container...');
      
      // Stop the container
      child_process.execSync(`docker stop ${this.containerId}`);
      
      // Clear container ID and status
      this.containerId = null;
      this.containerStatus = 'stopped';
      this.logToContainer('Container stopped');
      
      // Stop health checks
      this.stopContainerHealthChecks();
      
      return true;
    } catch (error) {
      console.error('Failed to stop container:', error);
      this.containerStatus = 'error';
      return false;
    }
  }

  /**
   * Restart the VLLM container
   */
  async restartContainer(): Promise<boolean> {
    await this.stopContainer();
    return this.startContainer();
  }

  /**
   * Get container status
   */
  getContainerStatus(): VllmContainerStatus {
    return this.containerStatus;
  }

  /**
   * Start container health checks
   */
  private startContainerHealthChecks() {
    if (this.containerHealthCheckTimer) {
      clearInterval(this.containerHealthCheckTimer);
    }
    
    this.containerHealthCheckTimer = setInterval(() => {
      this.checkContainerHealth();
    }, this.containerHealthCheckIntervalMs);
  }

  /**
   * Stop container health checks
   */
  private stopContainerHealthChecks() {
    if (this.containerHealthCheckTimer) {
      clearInterval(this.containerHealthCheckTimer);
      this.containerHealthCheckTimer = null;
    }
  }

  /**
   * Check container health
   */
  private async checkContainerHealth() {
    if (!this.containerId || this.containerStatus !== 'running') {
      return;
    }

    try {
      // Check if container is still running
      const containerInfo = child_process.execSync(`docker inspect ${this.containerId} --format '{{.State.Running}}'`).toString().trim();
      
      if (containerInfo !== 'true') {
        this.logToContainer('Container is not running');
        this.containerStatus = 'stopped';
        this.containerId = null;
        this.stopContainerHealthChecks();
        
        // Auto restart if enabled
        if (this.containerAutomaticRetryEnabled && this.containerRetryCount < this.containerMaxRetries) {
          this.containerRetryCount++;
          this.logToContainer(`Auto-restarting container (attempt ${this.containerRetryCount}/${this.containerMaxRetries})...`);
          this.startContainer();
        }
        
        return;
      }
      
      // Check if API is still responsive
      try {
        const response = await fetch(`${this.containerApiUrl}/v1/models`);
        if (!response.ok) {
          this.logToContainer('API is not responsive');
          // Do not immediately restart, wait for the container check to handle it
        }
      } catch (error) {
        this.logToContainer(`API check failed: ${error.message}`);
        // Do not immediately restart, wait for the container check to handle it
      }
    } catch (error) {
      console.error('Failed to check container health:', error);
    }
  }

  /**
   * Log message to container logs
   */
  private logToContainer(message: string) {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${message}`;
    
    // Add to logs with limit
    this.containerLogs.push(logEntry);
    if (this.containerLogs.length > this.containerMaxLogEntries) {
      this.containerLogs.shift();
    }
    
    // Also log to console
    console.log(`VLLM Container: ${message}`);
  }

  /**
   * Get container logs
   */
  getContainerLogs(): string[] {
    return [...this.containerLogs];
  }

  /**
   * Check if API is ready (overrides base method)
   */
  async testEndpoint(): Promise<boolean> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          return false;
        }
      }
    }
    
    // Call the parent method
    return super.testEndpoint();
  }

  /**
   * Make a request to the VLLM API (overrides base method)
   */
  async makeRequest(
    endpointUrl: string, 
    data: any, 
    model?: string, 
    options?: ApiRequestOptions
  ): Promise<any> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    // Call the parent method
    return super.makeRequest(endpointUrl, data, model, options);
  }

  /**
   * Make a streaming request to the VLLM API (overrides base method)
   */
  async *makeStreamRequestVllm(
    endpointUrl: string,
    data: any,
    options?: ApiRequestOptions
  ): AsyncGenerator<VllmStreamChunk> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    // Call the parent method
    yield* super.makeStreamRequestVllm(endpointUrl, data, options);
  }

  /**
   * Get the container configuration
   */
  getContainerConfig(): VllmContainerConfig | null {
    return this.containerConfig;
  }

  /**
   * Set the container configuration
   */
  setContainerConfig(config: Partial<VllmContainerConfig>): void {
    if (!this.containerConfig) {
      this.containerConfig = {
        image: this.containerImageDefault,
        gpu: false,
        models_path: this.containerModelsPath,
        config_path: this.containerConfigPath,
        api_port: 8000,
        tensor_parallel_size: 1,
        max_model_len: 0,
        gpu_memory_utilization: 0.9,
        quantization: null,
        trust_remote_code: false,
        custom_args: ''
      };
    }
    
    // Update configuration
    this.containerConfig = {
      ...this.containerConfig,
      ...config
    };
    
    // Update container environment variables
    if (config.tensor_parallel_size) {
      this.containerEnvVars['VLLM_TENSOR_PARALLEL_SIZE'] = config.tensor_parallel_size.toString();
    }
    
    if (config.gpu_memory_utilization) {
      this.containerEnvVars['VLLM_GPU_MEMORY_UTILIZATION'] = config.gpu_memory_utilization.toString();
    }
    
    if (config.max_model_len && config.max_model_len > 0) {
      this.containerEnvVars['VLLM_MAX_MODEL_LEN'] = config.max_model_len.toString();
    }
    
    if (config.quantization) {
      this.containerEnvVars['VLLM_QUANTIZATION'] = config.quantization;
    }
    
    if (config.trust_remote_code !== undefined) {
      this.containerEnvVars['VLLM_TRUST_REMOTE_CODE'] = config.trust_remote_code ? 'true' : 'false';
    }
    
    // Write updated configuration to file
    this.writeContainerConfigFile();
  }

  /**
   * Enable container mode
   */
  enableContainerMode(config?: Partial<VllmContainerConfig>): void {
    this.containerEnabled = true;
    
    if (config) {
      this.setContainerConfig(config);
    }
    
    if (this.containerStatus !== 'running') {
      this.startContainer().catch(error => {
        console.error('Failed to start container:', error);
      });
    }
  }

  /**
   * Disable container mode
   */
  async disableContainerMode(): Promise<void> {
    this.containerEnabled = false;
    
    if (this.containerStatus === 'running') {
      await this.stopContainer();
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(
    model: string = this.getDefaultModel()
  ): Promise<VllmModelInfo> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    const url = `${this.containerApiUrl}/v1/models/${model}`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model info: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  }

  /**
   * Get model statistics
   */
  async getModelStatistics(
    model: string = this.getDefaultModel()
  ): Promise<VllmModelStatistics> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    const url = `${this.containerApiUrl}/v1/models/${model}/statistics`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model statistics: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model statistics:', error);
      throw error;
    }
  }

  /**
   * Process a batch with detailed metrics (overriding parent implementation)
   */
  async processBatchWithMetrics(
    batchData: string[],
    model?: string,
    parameters?: Record<string, any>
  ): Promise<[string[], Record<string, any>]> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    // Use the parent implementation for the container API endpoint
    const endpointUrl = `${this.containerApiUrl}/v1/completions`;
    
    // Start timing
    const startTime = Date.now();
    
    // Prepare batch request
    const modelName = model || this.getDefaultModel();
    const requestData = {
      prompts: batchData,
      model: modelName,
      ...parameters
    };
    
    try {
      // Make the request
      const response = await this.makePostRequestVllm(`${endpointUrl}/batch`, requestData) as VllmBatchResponse;
      
      // Calculate metrics
      const endTime = Date.now();
      const totalDuration = endTime - startTime;
      
      // Extract metrics
      const metrics = {
        model: response.metadata?.model || modelName,
        batch_size: batchData.length,
        successful_items: (response.texts?.length || 0),
        total_time_ms: totalDuration,
        average_time_per_item_ms: totalDuration / batchData.length,
        finish_reasons: response.metadata?.finish_reasons || [],
        usage: response.metadata?.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      // Return the results and metrics
      return [response.texts || [], metrics];
    } catch (error) {
      console.error('Batch processing failed:', error);
      throw error;
    }
  }

  /**
   * Stream text generation with container support
   */
  async *streamGeneration(
    prompt: string,
    model?: string,
    parameters?: Record<string, any>
  ): AsyncGenerator<string> {
    // If container is enabled, ensure it's running
    if (this.containerEnabled) {
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw this.createApiError('VLLM container failed to start', 500, 'container_error');
        }
      }
    }
    
    const endpointUrl = `${this.containerApiUrl}/v1/completions`;
    const modelName = model || this.getDefaultModel();
    
    const requestData: VllmRequest = {
      prompt,
      model: modelName,
      stream: true,
      ...parameters
    };
    
    const stream = await this.makeStreamRequestVllm(endpointUrl, requestData);
    
    for await (const chunk of stream) {
      if (chunk.text) {
        yield chunk.text;
      }
    }
  }

  /**
   * Check container port availability
   */
  private async checkPortAvailability(port: number): Promise<boolean> {
    if (typeof process === 'undefined' || typeof child_process === 'undefined') {
      return true; // Can't check in non-Node environment
    }
    
    try {
      // Different check commands based on platform
      if (os.platform() === 'win32') {
        const output = child_process.execSync(`netstat -ano | findstr :${port}`).toString();
        return output.trim() === '';
      } else {
        const output = child_process.execSync(`lsof -i:${port}`).toString();
        return output.trim() === '';
      }
    } catch (error) {
      // If command fails, port is likely available
      return true;
    }
  }

  /**
   * Load models into the container
   */
  async loadModels(modelPaths: string[]): Promise<boolean> {
    if (!this.containerEnabled || !this.containerConfig) {
      throw new Error('Container mode is not enabled');
    }
    
    try {
      // Ensure container is running
      if (this.containerStatus !== 'running') {
        const started = await this.startContainer();
        if (!started) {
          throw new Error('Failed to start container');
        }
      }
      
      let success = true;
      
      // Copy each model to the models directory
      for (const modelPath of modelPaths) {
        try {
          const modelName = path.basename(modelPath);
          const targetPath = path.join(this.containerConfig.models_path, modelName);
          
          // Create directory if needed
          if (!fs.existsSync(targetPath)) {
            fs.mkdirSync(targetPath, { recursive: true });
          }
          
          // Copy files (this is simplified and would need to be more robust for production)
          if (fs.statSync(modelPath).isDirectory()) {
            fs.readdirSync(modelPath).forEach(file => {
              const srcFile = path.join(modelPath, file);
              const dstFile = path.join(targetPath, file);
              fs.copyFileSync(srcFile, dstFile);
            });
          } else {
            fs.copyFileSync(modelPath, targetPath);
          }
          
          this.logToContainer(`Model loaded: ${modelName}`);
        } catch (error) {
          this.logToContainer(`Failed to load model ${modelPath}: ${error.message}`);
          success = false;
        }
      }
      
      return success;
    } catch (error) {
      this.logToContainer(`Error loading models: ${error.message}`);
      return false;
    }
  }

  /**
   * Validate container capabilities
   */
  async validateContainerCapabilities(): Promise<Record<string, boolean>> {
    if (!this.containerEnabled) {
      throw new Error('Container mode is not enabled');
    }
    
    const capabilities: Record<string, boolean> = {
      docker_available: false,
      gpu_support: false,
      port_available: false,
      api_accessible: false,
      model_loaded: false
    };
    
    try {
      // Check if Docker is available
      try {
        child_process.execSync('docker --version', { stdio: 'pipe' });
        capabilities.docker_available = true;
      } catch (error) {
        capabilities.docker_available = false;
      }
      
      // Check if GPU is available (if enabled)
      if (this.containerConfig?.gpu) {
        try {
          const gpuInfo = child_process.execSync('nvidia-smi', { stdio: 'pipe' }).toString();
          capabilities.gpu_support = gpuInfo.includes('NVIDIA-SMI');
        } catch (error) {
          capabilities.gpu_support = false;
        }
      } else {
        capabilities.gpu_support = true; // Not needed if not enabled
      }
      
      // Check if port is available
      const port = this.containerConfig?.api_port || 8000;
      capabilities.port_available = await this.checkPortAvailability(port);
      
      // Check if API is accessible (only if container is running)
      if (this.containerStatus === 'running') {
        try {
          const response = await fetch(`${this.containerApiUrl}/v1/models`);
          capabilities.api_accessible = response.ok;
        } catch (error) {
          capabilities.api_accessible = false;
        }
        
        // Check if model is loaded
        try {
          const modelResponse = await fetch(`${this.containerApiUrl}/v1/models/${this.getDefaultModel()}`);
          capabilities.model_loaded = modelResponse.ok;
        } catch (error) {
          capabilities.model_loaded = false;
        }
      }
      
      return capabilities;
    } catch (error) {
      console.error('Failed to validate container capabilities:', error);
      return capabilities;
    }
  }
}
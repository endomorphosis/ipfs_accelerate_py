/**
 * OpenVINO Model Server (OVMS) API Backend
 */

import { 
  ApiResources, 
  ApiMetadata, 
  ApiRequestOptions,
  Message,
  ChatCompletionResponse,
  StreamChunk
} from '../types';
import { BaseApiBackend } from '../base';
import { 
  OVMSRequestData, 
  OVMSResponse, 
  OVMSRequestOptions,
  OVMSModelMetadata,
  OVMSModelConfig,
  OVMSServerStatistics,
  OVMSModelStatistics,
  OVMSQuantizationConfig
} from './types';

export class OVMS extends BaseApiBackend {
  protected apiUrl: string;
  protected modelName: string;
  protected modelVersion: string;
  protected precision: string;
  
  constructor(resources: ApiResources = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
    
    // Initialize OVMS-specific properties
    this.apiUrl = metadata.ovms_api_url || process.env.OVMS_API_URL || 'http://localhost:9000';
    this.modelName = metadata.ovms_model || process.env.OVMS_MODEL || 'model';
    this.modelVersion = metadata.ovms_version || process.env.OVMS_VERSION || 'latest';
    this.precision = metadata.ovms_precision || process.env.OVMS_PRECISION || 'FP32';
    
    // Set timeout for OVMS requests
    this.timeout = metadata.timeout || parseInt(process.env.OVMS_TIMEOUT || '30') * 1000;
  }
  
  /**
   * Get API key - OVMS doesn't typically use API keys, but implementation
   * supports them for potential authentication systems
   */
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.ovms_api_key || process.env.OVMS_API_KEY || '';
  }
  
  /**
   * Get the default model
   */
  protected getDefaultModel(): string {
    return 'model';
  }
  
  /**
   * Create an endpoint handler for making OVMS requests
   */
  createEndpointHandler(
    endpointUrl: string = `${this.apiUrl}/v1/models/${this.modelName}:predict`,
    model: string = this.modelName
  ): (data: OVMSRequestData) => Promise<OVMSResponse> {
    return async (data: OVMSRequestData): Promise<OVMSResponse> => {
      return this.makePostRequestOVMS(endpointUrl, data);
    };
  }
  
  /**
   * Test the OVMS endpoint
   */
  async testEndpoint(
    endpointUrl: string = `${this.apiUrl}/v1/models/${this.modelName}:predict`,
    model: string = this.modelName
  ): Promise<boolean> {
    try {
      // Create a simple test instance with basic data
      const testData: OVMSRequestData = {
        instances: [{
          data: [0, 1, 2, 3]
        }]
      };
      
      const response = await this.makePostRequestOVMS(endpointUrl, testData);
      return !!response && !!response.predictions;
    } catch (error) {
      console.error('OVMS endpoint test failed:', error);
      return false;
    }
  }
  
  /**
   * Make a POST request to the OVMS server
   */
  async makePostRequestOVMS(
    endpointUrl: string,
    data: OVMSRequestData,
    options: OVMSRequestOptions = {}
  ): Promise<OVMSResponse> {
    const requestId = options.requestId || `ovms_${Date.now()}`;
    const apiKey = options.apiKey || this.apiKey;
    
    // Handle request options specific to OVMS
    const modelVersion = options.version || this.modelVersion;
    
    // Add version to URL if not 'latest'
    let url = endpointUrl;
    if (modelVersion && modelVersion !== 'latest' && !url.includes('/versions/')) {
      url = url.replace(':predict', `/versions/${modelVersion}:predict`);
    }
    
    // Set headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    
    // Add API key if available
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Add request tracking if enabled
    if (this.requestTracking) {
      headers['X-Request-ID'] = requestId;
    }
    
    try {
      return await this.retryableRequest(async () => {
        const response = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(data),
          signal: options.signal || AbortSignal.timeout(options.timeout || this.timeout)
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw this.createApiError(
            `OVMS API error: ${errorText}`,
            response.status,
            'ovms_error'
          );
        }
        
        return await response.json();
      });
    } catch (error) {
      // Track the error if request tracking is enabled
      this.trackRequestResult(false, requestId, error as Error);
      
      // Re-throw the error
      throw error;
    }
  }
  
  /**
   * Format a request for OVMS
   */
  formatRequest(
    handler: (data: OVMSRequestData) => Promise<OVMSResponse>,
    input: any
  ): Promise<OVMSResponse> {
    // Handle different input types
    let formattedData: OVMSRequestData;
    
    if (Array.isArray(input)) {
      // Input is an array
      if (input.length > 0 && typeof input[0] === 'object') {
        // Batch of inputs
        formattedData = {
          instances: input
        };
      } else {
        // Single array input
        formattedData = {
          instances: [{
            data: input
          }]
        };
      }
    } else if (typeof input === 'object') {
      // Input is already an object, possibly already formatted
      if (input.instances || input.inputs) {
        // Already in OVMS format
        formattedData = input;
      } else if (input.data) {
        // Object with data field
        formattedData = {
          instances: [input]
        };
      } else {
        // General object, wrap as instance
        formattedData = {
          instances: [{ data: input }]
        };
      }
    } else {
      // Other scalar values, convert to array
      formattedData = {
        instances: [{
          data: [input]
        }]
      };
    }
    
    return handler(formattedData);
  }
  
  /**
   * Get model information
   */
  async getModelInfo(model: string = this.modelName): Promise<OVMSModelMetadata> {
    const url = `${this.apiUrl}/v1/models/${model}`;
    
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
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  }
  
  /**
   * Get model versions
   */
  async getModelVersions(model: string = this.modelName): Promise<string[]> {
    try {
      const modelInfo = await this.getModelInfo(model);
      return modelInfo.versions || [];
    } catch (error) {
      console.error('Failed to get model versions:', error);
      throw error;
    }
  }
  
  /**
   * Run inference with OVMS
   */
  async infer(
    model: string = this.modelName,
    data: any,
    options: OVMSRequestOptions = {}
  ): Promise<any> {
    const endpointUrl = `${this.apiUrl}/v1/models/${model}:predict`;
    const handler = this.createEndpointHandler(endpointUrl, model);
    
    const response = await this.formatRequest(handler, data);
    
    // Return predictions array if available, otherwise return full response
    return response.predictions || response;
  }
  
  /**
   * Run batch inference with OVMS
   */
  async batchInfer(
    model: string = this.modelName,
    dataBatch: any[],
    options: OVMSRequestOptions = {}
  ): Promise<any[]> {
    const endpointUrl = `${this.apiUrl}/v1/models/${model}:predict`;
    
    // Format as a batch request
    const batchRequest: OVMSRequestData = {
      instances: dataBatch.map(item => {
        if (typeof item === 'object' && !Array.isArray(item)) {
          return item;
        } else {
          return { data: item };
        }
      })
    };
    
    const response = await this.makePostRequestOVMS(endpointUrl, batchRequest, options);
    
    // Return the predictions array
    return response.predictions || [];
  }
  
  /**
   * Set model configuration
   */
  async setModelConfig(
    model: string = this.modelName,
    config: OVMSModelConfig
  ): Promise<any> {
    const configUrl = `${this.apiUrl}/v1/models/${model}/config`;
    
    try {
      const response = await fetch(configUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to set model config: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to set model config:', error);
      throw error;
    }
  }
  
  /**
   * Set execution mode (latency or throughput)
   */
  async setExecutionMode(
    model: string = this.modelName,
    mode: 'latency' | 'throughput'
  ): Promise<any> {
    return this.setModelConfig(model, {
      execution_mode: mode
    });
  }
  
  /**
   * Reload model
   */
  async reloadModel(model: string = this.modelName): Promise<any> {
    const reloadUrl = `${this.apiUrl}/v1/models/${model}/reload`;
    
    try {
      const response = await fetch(reloadUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to reload model: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to reload model:', error);
      throw error;
    }
  }
  
  /**
   * Get model status
   */
  async getModelStatus(model: string = this.modelName): Promise<any> {
    const statusUrl = `${this.apiUrl}/v1/models/${model}/status`;
    
    try {
      const response = await fetch(statusUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model status: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model status:', error);
      throw error;
    }
  }
  
  /**
   * Get server statistics
   */
  async getServerStatistics(): Promise<OVMSServerStatistics> {
    const statsUrl = `${this.apiUrl}/v1/statistics`;
    
    try {
      const response = await fetch(statsUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get server statistics: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get server statistics:', error);
      throw error;
    }
  }
  
  /**
   * Infer with specific version
   */
  async inferWithVersion(
    model: string = this.modelName,
    version: string,
    data: any,
    options: OVMSRequestOptions = {}
  ): Promise<any> {
    // Add version to options
    return this.infer(model, data, {
      ...options,
      version
    });
  }
  
  /**
   * Explain prediction
   */
  async explainPrediction(
    model: string = this.modelName,
    data: any,
    options: OVMSRequestOptions = {}
  ): Promise<any> {
    const explainUrl = `${this.apiUrl}/v1/models/${model}:explain`;
    
    // Create handler for explain endpoint
    const handler = (requestData: OVMSRequestData) => {
      return this.makePostRequestOVMS(explainUrl, requestData, options);
    };
    
    // Format and send the request
    return this.formatRequest(handler, data);
  }
  
  /**
   * Get model metadata with shapes
   */
  async getModelMetadataWithShapes(model: string = this.modelName): Promise<OVMSModelMetadata> {
    const metadataUrl = `${this.apiUrl}/v1/models/${model}/metadata`;
    
    try {
      const response = await fetch(metadataUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model metadata: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model metadata:', error);
      throw error;
    }
  }
  
  /**
   * Set quantization configuration
   */
  async setQuantization(
    model: string = this.modelName,
    config: OVMSQuantizationConfig
  ): Promise<any> {
    const quantUrl = `${this.apiUrl}/v1/models/${model}/quantization`;
    
    try {
      const response = await fetch(quantUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to set quantization: ${response.statusText}`,
          response.status,
          'ovms_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to set quantization:', error);
      throw error;
    }
  }
  
  /**
   * Check if a model is compatible with this API
   */
  isCompatibleModel(model: string): boolean {
    // OVMS can work with any model that has been deployed to the server
    // This implementation checks if the model exists in the server's deployed models
    if (!model) return false;
    
    try {
      // Attempt to get model info without throwing
      this.getModelInfo(model)
        .then(info => true)
        .catch(err => false);
        
      return true;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Required abstract method implementations
   */
  
  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    // Convert to OVMS format and call makePostRequestOVMS
    const endpointUrl = options?.endpoint || `${this.apiUrl}/v1/models/${this.modelName}:predict`;
    return this.makePostRequestOVMS(endpointUrl, data, { 
      ...options, 
      apiKey
    });
  }

  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // OVMS doesn't natively support streaming, so we'll simulate it
    // by returning the entire response in a single chunk
    const response = await this.makePostRequest(data, undefined, options);
    
    // Yield the response as a single chunk
    yield {
      content: JSON.stringify(response.predictions || response),
      type: 'result',
      done: true
    };
  }

  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    // OVMS doesn't support chat directly, so we'll convert messages to a format
    // that OVMS can understand and run inference
    
    // Extract the content from the last user message
    const lastUserMessage = messages.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) {
      throw new Error('No user message found');
    }
    
    const content = lastUserMessage.content;
    let input: any;
    
    if (typeof content === 'string') {
      // Simple text input
      input = content;
    } else if (Array.isArray(content)) {
      // Array input - could be multimodal or just array of values
      input = content;
    } else {
      // Object input
      input = content;
    }
    
    // Run inference
    const result = await this.infer(
      options?.model || this.modelName,
      input,
      options
    );
    
    // Convert to ChatCompletionResponse format
    return {
      content: JSON.stringify(result),
      role: 'assistant',
      model: options?.model || this.modelName
    };
  }

  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Similar to makeStreamRequest, we'll simulate streaming
    const response = await this.chat(messages, options);
    
    // Yield the response as a single chunk
    yield {
      content: response.content,
      role: 'assistant',
      type: 'result',
      done: true
    };
  }
}

// Default export
export default OVMS;
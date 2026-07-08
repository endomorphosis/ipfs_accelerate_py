/**
 * Tests for the Groq API Backend
 */

import { GroqBackend } from '../../src/api_backends/groq';
import { ApiMetadata, Message } from '../../src/api_backends/types';

// Mock fetch for testing
global.fetch = jest.fn();

describe('GroqBackend', () => {
  let groqBackend: GroqBackend;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset all mocks
    jest.resetAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      json: jest.fn().mockResolvedValue({
        id: 'test-id',
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'Hello, I am an AI assistant.'
            },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      groq_api_key: 'test-api-key',
      groq_api_version: 'v1',
      user_agent: 'IPFS-Accelerate-Test/1.0'
    };
    
    // Create backend instance
    groqBackend = new GroqBackend({}, mockMetadata);
  });
  
  test('should initialize with proper configuration', () => {
    expect(groqBackend).toBeDefined();
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = groqBackend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
  });
  
  test('should use API key from environment if not in metadata', () => {
    // Create instance without API key to test environment variable
    const originalEnv = process.env.GROQ_API_KEY;
    try {
      process.env.GROQ_API_KEY = 'env-api-key';
      const emptyMetadata = {};
      // @ts-ignore - Testing protected method
      const apiKey = groqBackend.getApiKey(emptyMetadata);
      expect(apiKey).toBe('env-api-key');
    } finally {
      process.env.GROQ_API_KEY = originalEnv;
    }
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = groqBackend.getDefaultModel();
    expect(model).toBeDefined();
    expect(model).toBe('llama3-8b-8192');
  });
  
  test('should create endpoint handler', () => {
    const handler = groqBackend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
  });
  
  test('should test API endpoint', async () => {
    const result = await groqBackend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      'https://api.groq.com/openai/v1/models',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          Authorization: 'Bearer test-api-key'
        })
      })
    );
  });
  
  test('should handle test endpoint errors', async () => {
    // Mock an error response
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));
    
    const result = await groqBackend.testEndpoint();
    expect(result).toBe(false);
  });
  
  test('should make a POST request', async () => {
    const data = {
      model: 'llama3-8b-8192',
      messages: [{ role: 'user', content: 'Hello' }]
    };
    
    const result = await groqBackend.makePostRequest(data);
    expect(result).toBeDefined();
    expect(result.id).toBe('test-id');
    expect(result.choices[0].message.content).toBe('Hello, I am an AI assistant.');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(data)
      })
    );
  });
  
  test('should generate a chat completion', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await groqBackend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.id).toBe('test-id');
    expect(response.model).toBe('test-model');
    expect(response.content).toBe('Hello, I am an AI assistant.');
    expect(response.usage).toEqual({
      inputTokens: 10,
      outputTokens: 20,
      totalTokens: 30
    });
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('Hello')
      })
    );
  });
  
  test('should handle API errors', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: jest.fn().mockResolvedValue({
        error: {
          message: 'Invalid API key',
          type: 'authentication_error'
        }
      })
    });
    
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    await expect(groqBackend.chat(messages)).rejects.toThrow();
  });
  
  test('should check if a model is compatible', () => {
    expect(groqBackend.isCompatibleModel('llama3-8b-8192')).toBe(true);
    expect(groqBackend.isCompatibleModel('nonexistent-model')).toBe(false);
  });
  
  test('should get available models', () => {
    const models = groqBackend.getAvailableModels();
    expect(models).toContain('llama3-8b-8192');
    expect(models).toContain('gemma2-9b-it');
  });
  
  test('should get model info', () => {
    const modelInfo = groqBackend.getModelInfo('llama3-8b-8192');
    expect(modelInfo).toBeDefined();
    expect(modelInfo.context_window).toBe(8192);
  });
  
  test('should get usage stats', () => {
    const stats = groqBackend.getUsageStats();
    expect(stats).toBeDefined();
    expect(stats.total_requests).toBe(0);
    expect(stats.total_tokens).toBe(0);
  });
  
  test('should reset usage stats', () => {
    // Mock existing usage stats
    (groqBackend as any).usageStats = {
      startedAt: new Date(),
      totalTokens: 100,
      totalRequests: 5,
      totalCost: 0.0001,
      requestHistory: [/* some items */]
    };
    
    groqBackend.resetUsageStats();
    
    const stats = groqBackend.getUsageStats();
    expect(stats.total_tokens).toBe(0);
    expect(stats.total_requests).toBe(0);
  });
});

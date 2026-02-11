// Tests for Gemini API Backend
import { Gemini } from '../src/api_backends/gemini';
import { ApiMetadata, Message } from '../src/api_backends/types';

// Mock fetch for testing
global.fetch = jest.fn();

describe('Gemini API Backend', () => {
  let backend: Gemini;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
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
      gemini_api_key: 'test-api-key'
    };
    
    // Create backend instance
    backend = new Gemini({}, mockMetadata);
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBeDefined();
    expect(typeof model).toBe('string');
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
  });
  
  test('should test endpoint', async () => {
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should handle chat completion', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.id).toBe('test-id');
    expect(response.model).toBe('test-model');
    expect(response.content).toBe('Hello, I am an AI assistant.');
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should handle API errors', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValue({
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
    
    await expect(backend.chat(messages)).rejects.toThrow();
  });
  
  test('should check model compatibility', () => {
    // Test with compatible model
    const compatibleModel = backend.isCompatibleModel('sample-model');
    expect(compatibleModel).toBeDefined();
    
    // Type of result should be boolean
    expect(typeof compatibleModel).toBe('boolean');
  });
});

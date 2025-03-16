/**
 * Comprehensive test file for HF TGI Unified API Backend
 * This test covers all aspects of the HF TGI Unified backend functionality
 */
import { HfTgiUnified } from '../../src/api_backends/hf_tgi_unified/hf_tgi_unified';
import { HfTgiUnifiedRequest, HfTgiUnifiedResponse } from '../../src/api_backends/hf_tgi_unified/types';
import { ApiMetadata, ApiRequestOptions, Message, PriorityLevel } from '../../src/api_backends/types';

// Mock child_process for container tests
jest.mock('child_process', () => ({
  execSync: jest.fn().mockReturnValue(Buffer.from('container-id')),
  spawn: jest.fn().mockImplementation(() => ({
    stdout: {
      on: jest.fn()
    },
    stderr: {
      on: jest.fn()
    },
    on: jest.fn()
  }))
}));

// Mock fs for file operations
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('{}'))
}));

// Mock fetch for API calls
global.fetch = jest.fn();

// Mock AbortSignal for timeout
global.AbortSignal = {
  timeout: jest.fn().mockReturnValue({
    aborted: false
  })
} as any;

// Mock TextEncoder and TextDecoder for streaming response tests
global.TextEncoder = require('util').TextEncoder;
global.TextDecoder = require('util').TextDecoder;

// Mock ReadableStream for testing streaming
const mockReadableStream = () => {
  const chunks = [
    '{"token":{"id":"1048","text":" Hello"},"generated_text":null,"details":null}',
    '{"token":{"id":"764","text":", how"},"generated_text":null,"details":null}',
    '{"token":{"id":"423","text":" can"},"generated_text":null,"details":null}',
    '{"token":{"id":"358","text":" I"},"generated_text":null,"details":null}',
    '{"token":{"id":"631","text":" help"},"generated_text":null,"details":null}',
    '{"token":{"id":"443","text":" you"},"generated_text":null,"details":null}',
    '{"token":{"id":"29889","text":"?"},"generated_text":null,"details":null}',
    '{"token":{"id":null,"text":null},"generated_text":" Hello, how can I help you?","details":{"finish_reason":"eos_token","generated_tokens":7,"seed":null,"prefill":[{"id":1,"text":"<s>"},{"id":2,"text":"User:"},{"id":2277,"text":" Hello"},{"id":29871,"text":"\\n\\n"},{"id":3,"text":"Assistant:"}],"tokens":[{"id":1048,"text":" Hello","logprob":-1.1079101562,"special":false},{"id":764,"text":", how","logprob":-2.0419921875,"special":false},{"id":423,"text":" can","logprob":-0.2104492188,"special":false},{"id":358,"text":" I","logprob":-0.0783691406,"special":false},{"id":631,"text":" help","logprob":-0.6293945312,"special":false},{"id":443,"text":" you","logprob":-0.1171875,"special":false},{"id":29889,"text":"?","logprob":-0.1245117188,"special":false}]}}'
  ];
  
  let currentIndex = 0;
  
  return {
    getReader: () => ({
      read: () => {
        if (currentIndex < chunks.length) {
          const chunk = chunks[currentIndex];
          currentIndex++;
          return Promise.resolve({
            done: false,
            value: new TextEncoder().encode(chunk + '\n')
          });
        } else {
          return Promise.resolve({ done: true, value: undefined });
        }
      },
      cancel: jest.fn(),
      releaseLock: jest.fn()
    })
  };
};

describe('HF TGI Unified API Backend', () => {
  let backend: HfTgiUnified;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        generated_text: "Hello, how can I help you?",
        details: {
          finish_reason: "eos_token",
          generated_tokens: 7
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      hf_tgi_api_key: 'test-api-key',
      hf_tgi_api_url: 'https://api-inference.huggingface.co/models/gpt2',
      hf_tgi_model: 'gpt2'
    };
    
    // Create backend instance
    backend = new HfTgiUnified({}, mockMetadata);
  });
  
  // Core initialization and configuration tests
  
  describe('Core initialization and configuration', () => {
    test('should initialize correctly', () => {
      expect(backend).toBeDefined();
      
      // Test with custom API URL
      const customBackend = new HfTgiUnified({}, { 
        hf_tgi_api_url: 'https://custom-tgi-url.com',
        hf_tgi_model: 'custom-model'
      });
      expect(customBackend).toBeDefined();
      
      // @ts-ignore - Accessing protected property for testing
      expect(customBackend.apiUrl).toBe('https://custom-tgi-url.com');
      // @ts-ignore - Accessing protected property for testing
      expect(customBackend.defaultModel).toBe('custom-model');
    });
    
    test('should get API key from metadata', () => {
      // @ts-ignore - Testing protected method
      const apiKey = backend.getApiKey(mockMetadata);
      expect(apiKey).toBe('test-api-key');
      
      // Test environment variable fallback
      const originalEnv = process.env.HF_API_KEY;
      process.env.HF_API_KEY = 'env-api-key';
      // @ts-ignore - Testing protected method
      const envApiKey = backend.getApiKey({});
      expect(envApiKey).toBe('env-api-key');
      
      // Restore original environment
      process.env.HF_API_KEY = originalEnv;
    });
    
    test('should get default model', () => {
      // @ts-ignore - Testing protected method
      const model = backend.getDefaultModel();
      expect(model).toBeDefined();
      expect(typeof model).toBe('string');
      expect(model).toBe('gpt2');
      
      // Test with different model in metadata
      const customBackend = new HfTgiUnified({}, { 
        hf_tgi_model: 'mistralai/Mistral-7B-Instruct-v0.1'
      });
      // @ts-ignore - Testing protected method
      expect(customBackend.getDefaultModel()).toBe('mistralai/Mistral-7B-Instruct-v0.1');
      
      // Test fallback to default model
      const fallbackBackend = new HfTgiUnified({}, {});
      // @ts-ignore - Testing protected method
      expect(fallbackBackend.getDefaultModel()).toBe('TinyLlama/TinyLlama-1.1B-Chat-v0.1');
    });
    
    test('should extract container configuration from metadata', () => {
      // Test with custom container config
      const customBackend = new HfTgiUnified({}, { 
        hf_tgi_container_image: 'ghcr.io/huggingface/text-generation-inference:latest',
        hf_tgi_container_ports: '8080:80'
      });
      // @ts-ignore - Testing protected property
      expect(customBackend.containerImage).toBe('ghcr.io/huggingface/text-generation-inference:latest');
      // @ts-ignore - Testing protected property
      expect(customBackend.containerPorts).toBe('8080:80');
    });
  });
  
  // API endpoint and request handling tests
  
  describe('API endpoint and request handling', () => {
    test('should create endpoint handler', () => {
      const handler = backend.createEndpointHandler();
      expect(handler).toBeDefined();
      expect(typeof handler).toBe('function');
      
      // Test with custom URL
      const customHandler = backend.createEndpointHandler('https://custom-endpoint.com');
      expect(customHandler).toBeDefined();
      expect(typeof customHandler).toBe('function');
    });
    
    test('should test endpoint', async () => {
      const result = await backend.testEndpoint();
      expect(result).toBe(true);
      expect(global.fetch).toHaveBeenCalled();
      
      // Test with custom URL and model
      const customResult = await backend.testEndpoint('https://custom-endpoint.com', 'custom-model');
      expect(customResult).toBe(true);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
    
    test('should handle endpoint failures', async () => {
      // Mock fetch failure
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Connection refused'));
      
      const result = await backend.testEndpoint();
      expect(result).toBe(false);
      
      // Mock HTTP error
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });
      
      const result2 = await backend.testEndpoint();
      expect(result2).toBe(false);
    });
    
    test('should make POST request', async () => {
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello, can you help me?',
        parameters: {
          max_new_tokens: 20,
          temperature: 0.7,
          top_p: 0.9
        }
      };
      
      const response = await backend.makePostRequest(data);
      
      expect(response).toBeDefined();
      expect(global.fetch).toHaveBeenCalled();
      expect(global.fetch).toHaveBeenCalledWith(
        'https://api-inference.huggingface.co/models/gpt2',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-api-key'
          }),
          body: expect.stringContaining('"inputs":"Hello, can you help me?"')
        })
      );
      
      // Test with options
      const options: ApiRequestOptions = {
        endpoint: 'https://custom-endpoint.com',
        model: 'custom-model',
        priority: PriorityLevel.HIGH,
        timeoutMs: 5000
      };
      
      const customResponse = await backend.makePostRequest(data, undefined, options);
      
      expect(customResponse).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(global.fetch).toHaveBeenLastCalledWith(
        'https://custom-endpoint.com',
        expect.any(Object)
      );
    });
    
    test('should handle API errors', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: jest.fn().mockResolvedValue({
          error: 'Invalid input format'
        })
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          max_new_tokens: 20
        }
      };
      
      await expect(backend.makePostRequest(data)).rejects.toThrow('HF TGI API error: Bad Request');
    });
    
    test('should handle network errors', async () => {
      // Mock network error
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          max_new_tokens: 20
        }
      };
      
      await expect(backend.makePostRequest(data)).rejects.toThrow('Network error');
    });
    
    test('should make streaming request', async () => {
      // Mock streaming response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        body: mockReadableStream(),
        status: 200
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello, can you help me?',
        parameters: {
          max_new_tokens: 20,
          temperature: 0.7,
          top_p: 0.9,
          stream: true
        }
      };
      
      const stream = await backend.makeStreamRequest(data);
      
      expect(stream).toBeDefined();
      
      // Collect items from the stream
      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBeGreaterThan(0);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"stream":true')
        })
      );
    });
    
    test('should handle streaming errors', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          stream: true
        }
      };
      
      await expect(backend.makeStreamRequest(data)).rejects.toThrow('HF TGI API streaming error: Internal Server Error');
    });
    
    test('should handle null body in streaming response', async () => {
      // Mock response with null body
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        body: null,
        status: 200
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          stream: true
        }
      };
      
      await expect(backend.makeStreamRequest(data)).rejects.toThrow('Response body is null');
    });
  });
  
  // Model compatibility tests
  
  describe('Model compatibility', () => {
    test('should check model compatibility', () => {
      // Test with compatible models
      expect(backend.isCompatibleModel('gpt2')).toBe(true);
      expect(backend.isCompatibleModel('TinyLlama/TinyLlama-1.1B-Chat-v0.1')).toBe(true);
      expect(backend.isCompatibleModel('mistralai/Mistral-7B-Instruct-v0.1')).toBe(true);
      expect(backend.isCompatibleModel('llama')).toBe(true);
      
      // Test with incompatible models
      expect(backend.isCompatibleModel('BAAI/bge-small-en-v1.5')).toBe(false);
      expect(backend.isCompatibleModel('clip-vit-base-patch32')).toBe(false);
      expect(backend.isCompatibleModel('facebook/bart-base')).toBe(false);
    });
  });
  
  // Chat interface tests
  
  describe('Chat interface', () => {
    test('should implement chat method', async () => {
      const messages: Message[] = [
        { role: 'user', content: 'Hello, can you help me?' }
      ];
      
      const response = await backend.chat(messages);
      
      expect(response).toBeDefined();
      expect(response.content).toBe('Hello, how can I help you?');
      expect(response.model).toBe('gpt2');
      expect(global.fetch).toHaveBeenCalledWith(
        'https://api-inference.huggingface.co/models/gpt2',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"inputs"')
        })
      );
      
      // Test with options
      const optionsResponse = await backend.chat(messages, {
        model: 'custom-model',
        temperature: 0.7,
        maxTokens: 100
      });
      
      expect(optionsResponse).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      // The second call should use the custom model
      expect(global.fetch).toHaveBeenLastCalledWith(
        expect.stringContaining('custom-model'),
        expect.any(Object)
      );
      // Should include the temperature parameter
      expect(global.fetch).toHaveBeenLastCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"temperature":0.7')
        })
      );
    });
    
    test('should handle multiple messages in chat', async () => {
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Hello, how are you?' },
        { role: 'assistant', content: 'I am doing well, thank you for asking.' },
        { role: 'user', content: 'Can you help me with a question?' }
      ];
      
      const response = await backend.chat(messages);
      
      expect(response).toBeDefined();
      expect(response.content).toBe('Hello, how can I help you?');
      // Should format the chat history correctly
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('You are a helpful assistant')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('Hello, how are you?')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('I am doing well, thank you for asking.')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('Can you help me with a question?')
        })
      );
    });
    
    test('should handle errors in chat', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hello' }
      ];
      
      await expect(backend.chat(messages)).rejects.toThrow('HF TGI API error: Internal Server Error');
    });
    
    test('should implement streamChat method', async () => {
      // Mock streaming response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        body: mockReadableStream(),
        status: 200
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hello, can you help me?' }
      ];
      
      const stream = backend.streamChat(messages);
      
      expect(stream).toBeDefined();
      
      // Collect items from the stream
      const chunks = [];
      for await (const chunk of await stream) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBeGreaterThan(0);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"stream":true')
        })
      );
    });
    
    test('should handle options in streamChat', async () => {
      // Mock streaming response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        body: mockReadableStream(),
        status: 200
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hello, can you help me?' }
      ];
      
      const options = {
        model: 'custom-model',
        temperature: 0.8,
        maxTokens: 100,
        topP: 0.95
      };
      
      const stream = backend.streamChat(messages, options);
      
      expect(stream).toBeDefined();
      await stream; // Resolve the promise
      
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('custom-model'),
        expect.objectContaining({
          body: expect.stringContaining('"temperature":0.8')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"max_new_tokens":100')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"top_p":0.95')
        })
      );
    });
    
    test('should handle errors in streamChat', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hello' }
      ];
      
      await expect(backend.streamChat(messages)).rejects.toThrow('HF TGI API streaming error: Internal Server Error');
    });
  });
  
  // Container management tests
  
  describe('Container management', () => {
    test('should start container', async () => {
      const child_process = require('child_process');
      
      // @ts-ignore - Testing method not exposed in interface
      const containerId = await backend.startContainer();
      
      expect(containerId).toBe('container-id');
      expect(child_process.execSync).toHaveBeenCalled();
      expect(child_process.execSync).toHaveBeenCalledWith(
        expect.stringContaining('docker run'),
        expect.any(Object)
      );
    });
    
    test('should start container with custom model', async () => {
      const child_process = require('child_process');
      
      // @ts-ignore - Testing method not exposed in interface
      const containerId = await backend.startContainer('custom-model', '/models/custom-model');
      
      expect(containerId).toBe('container-id');
      expect(child_process.execSync).toHaveBeenCalled();
      expect(child_process.execSync).toHaveBeenCalledWith(
        expect.stringContaining('custom-model'),
        expect.any(Object)
      );
    });
    
    test('should stop container', async () => {
      const child_process = require('child_process');
      
      // @ts-ignore - Testing method not exposed in interface
      await backend.stopContainer('container-id');
      
      expect(child_process.execSync).toHaveBeenCalled();
      expect(child_process.execSync).toHaveBeenCalledWith(
        expect.stringContaining('docker stop container-id'),
        expect.any(Object)
      );
    });
    
    test('should handle container errors', async () => {
      const child_process = require('child_process');
      
      // Mock command error
      child_process.execSync.mockImplementationOnce(() => {
        throw new Error('Container error');
      });
      
      // @ts-ignore - Testing method not exposed in interface
      await expect(backend.startContainer()).rejects.toThrow('Failed to start TGI container: Container error');
    });
  });
  
  // Advanced inference parameters tests
  
  describe('Advanced inference parameters', () => {
    test('should configure advanced parameters', async () => {
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          max_new_tokens: 100,
          temperature: 0.7,
          top_p: 0.9,
          top_k: 50,
          repetition_penalty: 1.2,
          do_sample: true,
          seed: 42,
          stop: ['User:', 'END'],
          watermark: true,
          details: true
        }
      };
      
      await backend.makePostRequest(data);
      
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"max_new_tokens":100')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"temperature":0.7')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"top_p":0.9')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"top_k":50')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"repetition_penalty":1.2')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"do_sample":true')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"seed":42')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"stop":["User:","END"]')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"watermark":true')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"details":true')
        })
      );
    });
    
    test('should configure LoRA and quantization parameters', async () => {
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello',
        parameters: {
          adapter_id: 'custom-lora',
          quantize: 'bitsandbytes-4bit'
        }
      };
      
      await backend.makePostRequest(data);
      
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"adapter_id":"custom-lora"')
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"quantize":"bitsandbytes-4bit"')
        })
      );
    });
  });
  
  // Error handling and recovery tests
  
  describe('Error handling and recovery', () => {
    test('should implement circuit breaker pattern', async () => {
      // Create backend with circuit breaker config
      const circuitBackend = new HfTgiUnified({}, {
        ...mockMetadata,
        circuit_breaker_threshold: 3,
        circuit_breaker_timeout_ms: 5000
      });
      
      // Mock repeated failures to trigger circuit breaker
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Server error 1'))
        .mockRejectedValueOnce(new Error('Server error 2'))
        .mockRejectedValueOnce(new Error('Server error 3'));
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello'
      };
      
      // First failure
      try {
        await circuitBackend.makePostRequest(data);
      } catch (error) {
        expect(error.message).toContain('Server error 1');
      }
      
      // Second failure
      try {
        await circuitBackend.makePostRequest(data);
      } catch (error) {
        expect(error.message).toContain('Server error 2');
      }
      
      // Third failure should trigger circuit breaker
      try {
        await circuitBackend.makePostRequest(data);
      } catch (error) {
        expect(error.message).toContain('Server error 3');
      }
      
      // Fourth attempt should fail fast with circuit breaker error
      try {
        await circuitBackend.makePostRequest(data);
      } catch (error) {
        expect(error.message).toContain('Circuit breaker is open');
      }
      
      // Verify only 3 API calls were made (not 4)
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
    
    test('should implement request queue with priority', async () => {
      // Create backend with queue config
      const queueBackend = new HfTgiUnified({}, {
        ...mockMetadata,
        queue_size: 5
      });
      
      // Mock successful response after delay
      (global.fetch as jest.Mock).mockImplementation(() => {
        return new Promise(resolve => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: () => Promise.resolve({
                generated_text: "Hello, how can I help you?",
                details: {
                  finish_reason: "eos_token",
                  generated_tokens: 7
                }
              })
            });
          }, 100);
        });
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello'
      };
      
      // Send multiple requests with different priorities
      const highPriorityPromise = queueBackend.makePostRequest(data, undefined, { priority: PriorityLevel.HIGH });
      const normalPriorityPromise = queueBackend.makePostRequest(data, undefined, { priority: PriorityLevel.NORMAL });
      const lowPriorityPromise = queueBackend.makePostRequest(data, undefined, { priority: PriorityLevel.LOW });
      
      // All should complete successfully
      const results = await Promise.all([highPriorityPromise, normalPriorityPromise, lowPriorityPromise]);
      
      expect(results).toBeDefined();
      expect(results.length).toBe(3);
      expect(results[0]).toBeDefined();
      expect(results[1]).toBeDefined();
      expect(results[2]).toBeDefined();
      
      // Verify 3 API calls were made
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
    
    test('should implement retries for transient errors', async () => {
      // Create backend with retry config
      const retryBackend = new HfTgiUnified({}, {
        ...mockMetadata,
        retry_count: 2,
        retry_delay_ms: 100
      });
      
      // Mock failure then success
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Transient error'))
        .mockResolvedValueOnce({
          ok: true,
          json: jest.fn().mockResolvedValue({
            generated_text: "Hello, how can I help you?",
            details: {
              finish_reason: "eos_token",
              generated_tokens: 7
            }
          })
        });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello'
      };
      
      // Should retry and eventually succeed
      const result = await retryBackend.makePostRequest(data);
      
      expect(result).toBeDefined();
      expect(result.generated_text).toBe('Hello, how can I help you?');
      
      // Verify 2 API calls were made (1 failure, 1 success)
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
    
    test('should handle API rate limiting', async () => {
      // Mock rate limit response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 429,
        statusText: 'Too Many Requests',
        headers: {
          get: jest.fn().mockImplementation(name => {
            if (name === 'retry-after') return '2';
            return null;
          })
        }
      });
      
      const data: HfTgiUnifiedRequest = {
        inputs: 'Hello'
      };
      
      // Rate limiting should trigger backoff
      await expect(backend.makePostRequest(data)).rejects.toThrow('HF TGI API error: Too Many Requests');
      
      // Verify the retry-after header was checked
      const mockResponse = (global.fetch as jest.Mock).mock.results[0].value;
      expect(mockResponse.headers.get).toHaveBeenCalledWith('retry-after');
    });
  });
  
  // Model card and metadata tests
  
  describe('Model card and metadata', () => {
    test('should get model card info', async () => {
      // Mock model info response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          id: 'gpt2',
          pipeline_tag: 'text-generation',
          tags: ['text-generation', 'pytorch'],
          downloads: 1000000,
          likes: 500,
          author: 'OpenAI',
          license: 'mit',
          model_size: '124M',
          revision: 'main'
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const modelInfo = await backend.getModelInfo('gpt2');
      
      expect(modelInfo).toBeDefined();
      expect(modelInfo.id).toBe('gpt2');
      expect(modelInfo.pipeline_tag).toBe('text-generation');
      expect(modelInfo.tags).toContain('text-generation');
      expect(modelInfo.author).toBe('OpenAI');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('https://huggingface.co/api/models/gpt2'),
        expect.any(Object)
      );
    });
    
    test('should get model token count', async () => {
      // Mock token count response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          token_count: 12,
          tokens: ['Hello', ',', ' how', ' can', ' I', ' help', ' you', '?']
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const tokenCount = await backend.getTokenCount('gpt2', 'Hello, how can I help you?');
      
      expect(tokenCount).toBeDefined();
      expect(tokenCount.token_count).toBe(12);
      expect(tokenCount.tokens).toContain(' help');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/tokenize'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Hello, how can I help you?')
        })
      );
    });
  });
  
  // Additional API features tests
  
  describe('Additional API features', () => {
    test('should format prompts for chat models', () => {
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Hello, how are you?' },
        { role: 'assistant', content: 'I am doing well, thank you for asking.' },
        { role: 'user', content: 'Can you help me with a question?' }
      ];
      
      // @ts-ignore - Testing protected method
      const formattedPrompt = backend.formatMessagesForModel(messages, 'mistralai/Mistral-7B-Instruct-v0.1');
      
      expect(formattedPrompt).toBeDefined();
      expect(formattedPrompt).toContain('You are a helpful assistant.');
      expect(formattedPrompt).toContain('Hello, how are you?');
      expect(formattedPrompt).toContain('I am doing well, thank you for asking.');
      expect(formattedPrompt).toContain('Can you help me with a question?');
      
      // Different models have different formats
      // @ts-ignore - Testing protected method
      const llama2Format = backend.formatMessagesForModel(messages, 'meta-llama/Llama-2-7b-chat-hf');
      expect(llama2Format).toBeDefined();
      expect(llama2Format).toContain('[INST]');
      
      // @ts-ignore - Testing protected method
      const mistralFormat = backend.formatMessagesForModel(messages, 'mistralai/Mistral-7B-Instruct-v0.1');
      expect(mistralFormat).toBeDefined();
      expect(mistralFormat).toContain('[INST]');
      
      // @ts-ignore - Testing protected method
      const tinyLlamaFormat = backend.formatMessagesForModel(messages, 'TinyLlama/TinyLlama-1.1B-Chat-v0.1');
      expect(tinyLlamaFormat).toBeDefined();
      expect(tinyLlamaFormat).toContain('<s>');
    });
    
    test('should get model generation parameters', () => {
      // @ts-ignore - Testing protected method
      const defaultParams = backend.getGenerationParameters();
      
      expect(defaultParams).toBeDefined();
      expect(defaultParams.max_new_tokens).toBeDefined();
      expect(defaultParams.temperature).toBeDefined();
      expect(defaultParams.top_p).toBeDefined();
      
      // Test with custom options
      const options = {
        temperature: 0.8,
        maxTokens: 100,
        topP: 0.95,
        topK: 40,
      };
      
      // @ts-ignore - Testing protected method
      const customParams = backend.getGenerationParameters(options);
      
      expect(customParams).toBeDefined();
      expect(customParams.temperature).toBe(0.8);
      expect(customParams.max_new_tokens).toBe(100);
      expect(customParams.top_p).toBe(0.95);
      expect(customParams.top_k).toBe(40);
    });
  });
});

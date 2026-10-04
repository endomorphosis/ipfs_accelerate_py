// Tests for OpenAI Mini API Backend
import { OpenAiMini } from '../../src/api_backends/openai_mini';
import { ChatMessage } from '../../src/api_backends/types';
import { CircuitBreaker } from '../../src/api_backends/utils/circuit_breaker';

// Mock dependencies
jest.mock('../../src/api_backends/utils/circuit_breaker');

// Mock global fetch
global.fetch = jest.fn();
const mockFetch = global.fetch as jest.Mock;

// Mock fs and path modules
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('mock audio content'))
}));

jest.mock('path', () => ({
  basename: jest.fn().mockReturnValue('test.mp3')
}));

describe('OpenAI Mini API Backend', () => {
  let backend: OpenAiMini;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    mockFetch.mockReset();
    
    // Default mock response
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: 1679000000,
        model: 'gpt-3.5-turbo',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: 'Hello, I am a helpful assistant.'
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18
        }
      })
    });
    
    // Set up api key in environment
    process.env.OPENAI_API_KEY = 'test-api-key';
    
    // Create test instance
    backend = new OpenAiMini();
  });
  
  afterEach(() => {
    delete process.env.OPENAI_API_KEY;
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    expect(CircuitBreaker).toHaveBeenCalled();
  });
  
  test('should initialize with custom options', () => {
    const customBackend = new OpenAiMini({
      apiUrl: 'https://custom-openai-url.com/v1',
      maxRetries: 5,
      requestTimeout: 60000,
      useRequestQueue: false,
      debug: true
    });
    
    expect(customBackend).toBeDefined();
    
    // @ts-ignore - Access private property for testing
    expect(customBackend.baseUrl).toBe('https://custom-openai-url.com/v1');
    
    // @ts-ignore - Access private property for testing
    expect(customBackend.options.maxRetries).toBe(5);
    
    // @ts-ignore - Access private property for testing
    expect(customBackend.options.requestTimeout).toBe(60000);
    
    // @ts-ignore - Access private property for testing
    expect(customBackend.options.useRequestQueue).toBe(false);
    
    // @ts-ignore - Access private property for testing
    expect(customBackend.options.debug).toBe(true);
  });
  
  test('should get API key from environment', () => {
    // @ts-ignore - Access private property for testing
    expect(backend.apiKey).toBe('test-api-key');
  });
  
  test('should get API key from metadata', () => {
    const metadataBackend = new OpenAiMini({}, {
      openai_mini_api_key: 'metadata-api-key'
    });
    
    // @ts-ignore - Access private property for testing
    expect(metadataBackend.apiKey).toBe('metadata-api-key');
  });
  
  test('should return correct default model', () => {
    expect(backend.getDefaultModel()).toBe('gpt-3.5-turbo');
  });
  
  test('should check model compatibility correctly', () => {
    // Compatible models
    expect(backend.isCompatibleModel('gpt-3.5-turbo')).toBe(true);
    expect(backend.isCompatibleModel('gpt-4')).toBe(true);
    expect(backend.isCompatibleModel('gpt-4-turbo')).toBe(true);
    expect(backend.isCompatibleModel('text-embedding-ada-002')).toBe(true);
    expect(backend.isCompatibleModel('whisper-1')).toBe(true);
    expect(backend.isCompatibleModel('tts-1')).toBe(true);
    expect(backend.isCompatibleModel('tts-1-hd')).toBe(true);
    expect(backend.isCompatibleModel('dall-e-2')).toBe(true);
    expect(backend.isCompatibleModel('dall-e-3')).toBe(true);
    
    // Compatible model prefixes
    expect(backend.isCompatibleModel('gpt-3.5-something-new')).toBe(true);
    expect(backend.isCompatibleModel('gpt-4-something-new')).toBe(true);
    expect(backend.isCompatibleModel('text-embedding-new-model')).toBe(true);
    
    // Incompatible models
    expect(backend.isCompatibleModel('llama-7b')).toBe(false);
    expect(backend.isCompatibleModel('claude-2')).toBe(false);
    expect(backend.isCompatibleModel('falcon-7b')).toBe(false);
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(handler.makeRequest).toBeDefined();
    expect(typeof handler.makeRequest).toBe('function');
  });
  
  test('should test endpoint successfully', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: [
          { id: 'model-1', object: 'model' },
          { id: 'model-2', object: 'model' }
        ]
      })
    });
    
    const result = await backend.testEndpoint();
    
    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/models',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });
  
  test('should return false for failed endpoint test', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: 'Unauthorized'
    });
    
    const result = await backend.testEndpoint();
    
    expect(result).toBe(false);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
  
  test('should make post request successfully', async () => {
    const data = { test: 'data' };
    
    const result = await backend.makePostRequest('/test-endpoint', data);
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/test-endpoint',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: JSON.stringify(data)
      })
    );
    
    expect(result).toEqual({
      id: 'chatcmpl-123',
      object: 'chat.completion',
      created: 1679000000,
      model: 'gpt-3.5-turbo',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: 'Hello, I am a helpful assistant.'
        },
        finish_reason: 'stop'
      }],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 8,
        total_tokens: 18
      }
    });
  });
  
  test('should handle errors in post request', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ error: 'Invalid request parameters' })
    });
    
    await expect(backend.makePostRequest('/test-endpoint', {}))
      .rejects.toThrow('OpenAI Mini API error (400)');
  });
  
  test('should handle chat completion', async () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello, how are you?' }
    ];
    
    const result = await backend.chat(messages);
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"model":"gpt-3.5-turbo"')
      })
    );
    
    // Verify request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'gpt-3.5-turbo',
      messages
    });
    
    // Verify response
    expect(result.content).toBe('Hello, I am a helpful assistant.');
    expect(result.done).toBe(true);
  });
  
  test('should handle chat with custom options', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    await backend.chat(messages, {
      model: 'gpt-4',
      temperature: 0.8,
      top_p: 0.9,
      max_tokens: 100,
      stop: ['END'],
      priority: 'HIGH'
    });
    
    // Verify request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'gpt-4',
      messages,
      temperature: 0.8,
      top_p: 0.9,
      max_tokens: 100,
      stop: ['END']
    });
  });
  
  test('should handle streaming chat', async () => {
    // Create a mock stream
    const mockStream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        
        controller.enqueue(encoder.encode('data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1679000000,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'));
        controller.enqueue(encoder.encode('data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1679000000,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'));
        controller.enqueue(encoder.encode('data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1679000000,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'));
        controller.enqueue(encoder.encode('data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1679000000,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      }
    });
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: mockStream
    });
    
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const generator = backend.streamChat(messages);
    
    // Verify stream is returned
    expect(generator).toBeDefined();
    
    // Collect chunks
    const chunks = [];
    for await (const chunk of generator) {
      chunks.push(chunk);
    }
    
    // Verify fetch was called correctly
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][1].body).toContain('"stream":true');
    
    // Verify chunks
    expect(chunks.length).toBe(3);
    expect(chunks[0].content).toBe('');
    expect(chunks[1].content).toBe('Hello');
    expect(chunks[2].content).toBe(' world');
    expect(chunks[2].done).toBe(true);
  });
  
  test('should upload file', async () => {
    const fs = require('fs');
    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(Buffer.from('test file content'));
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'file-123',
        object: 'file',
        created_at: 1679000000,
        filename: 'test.jsonl',
        purpose: 'fine-tune',
        bytes: 100,
        status: 'uploaded',
        status_details: null
      })
    });
    
    const result = await backend.uploadFile('/path/to/file.jsonl', {
      purpose: 'fine-tune',
      fileName: 'custom-name.jsonl'
    });
    
    expect(fs.existsSync).toHaveBeenCalledWith('/path/to/file.jsonl');
    expect(fs.readFileSync).toHaveBeenCalledWith('/path/to/file.jsonl');
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe('https://api.openai.com/v1/files');
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    
    // Verify response
    expect(result.id).toBe('file-123');
    expect(result.purpose).toBe('fine-tune');
    expect(result.filename).toBe('test.jsonl');
  });
  
  test('should handle file not found', async () => {
    const fs = require('fs');
    fs.existsSync.mockReturnValue(false);
    
    await expect(backend.uploadFile('/path/to/nonexistent.jsonl'))
      .rejects.toThrow('File not found');
  });
  
  test('should transcribe audio', async () => {
    const fs = require('fs');
    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(Buffer.from('audio content'));
    
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        text: 'This is a transcription of the audio.'
      })
    });
    
    const result = await backend.transcribeAudio('/path/to/audio.mp3', {
      model: 'whisper-1',
      language: 'en',
      prompt: 'Transcribe the following audio:',
      response_format: 'text',
      temperature: 0.3
    });
    
    expect(fs.existsSync).toHaveBeenCalledWith('/path/to/audio.mp3');
    expect(fs.readFileSync).toHaveBeenCalledWith('/path/to/audio.mp3');
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe('https://api.openai.com/v1/audio/transcriptions');
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    
    // Verify response
    expect(result).toBe('This is a transcription of the audio.');
  });
  
  test('should handle text to speech', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer
    });
    
    const result = await backend.textToSpeech('Convert this text to speech', {
      model: 'tts-1',
      voice: 'alloy',
      speed: 1.2,
      response_format: 'mp3'
    });
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe('https://api.openai.com/v1/audio/speech');
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    
    // Verify request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'tts-1',
      input: 'Convert this text to speech',
      voice: 'alloy',
      speed: 1.2,
      response_format: 'mp3'
    });
    
    // Verify response is Buffer
    expect(Buffer.isBuffer(result)).toBe(true);
  });
  
  test('should generate image', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        created: 1679000000,
        data: [{
          url: 'https://example.com/image.png',
          revised_prompt: 'A detailed image of a mountain landscape'
        }]
      })
    });
    
    const result = await backend.generateImage('A mountain landscape', {
      model: 'dall-e-3',
      size: '1024x1024',
      quality: 'hd',
      n: 1,
      response_format: 'url',
      style: 'vivid'
    });
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe('https://api.openai.com/v1/images/generations');
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
    
    // Verify request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'dall-e-3',
      prompt: 'A mountain landscape',
      size: '1024x1024',
      quality: 'hd',
      n: 1,
      response_format: 'url',
      style: 'vivid'
    });
    
    // Verify response
    expect(result.data[0].url).toBe('https://example.com/image.png');
    expect(result.data[0].revised_prompt).toBe('A detailed image of a mountain landscape');
  });
  
  test('should throw error when API key is missing', async () => {
    delete process.env.OPENAI_API_KEY;
    
    // Create a backend without API key
    const noKeyBackend = new OpenAiMini();
    
    // @ts-ignore - Access private property for testing
    expect(noKeyBackend.apiKey).toBeNull();
    
    // Try to make a request
    await expect(noKeyBackend.chat([{ role: 'user', content: 'Hello' }]))
      .rejects.toThrow('API key not found');
  });
});
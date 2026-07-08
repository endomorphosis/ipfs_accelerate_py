import { Groq } from '../src/api_backends/groq';

describe('Groq API Backend', () => {
  let groq: Groq;
  
  beforeEach(() => {
    // Create a new instance for each test
    groq = new Groq({}, {
      groq_api_key: 'test-api-key'
    });
    
    // Mock fetch to avoid actual API calls
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          id: "chatcmpl-123456789",
          object: "chat.completion",
          created: Date.now(),
          model: "llama2-70b-4096",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "This is a test response from Groq API"
              },
              finish_reason: "stop"
            }
          ],
          usage: {
            prompt_tokens: 15,
            completion_tokens: 20,
            total_tokens: 35
          }
        })
      }) as any
    );
  });
  
  test('should initialize with provided API key', () => {
    expect(groq).toBeDefined();
    expect((groq as any).apiKey).toBe('test-api-key');
  });
  
  test('should create an endpoint handler', () => {
    const handler = groq.createEndpointHandler();
    expect(typeof handler).toBe('function');
  });
  
  test('should test an endpoint', async () => {
    const result = await groq.testEndpoint();
    expect(result).toBe(true);
    expect(fetch).toHaveBeenCalled();
    
    // Verify the request
    expect(fetch).toHaveBeenCalledWith(
      'https://api.groq.com/openai/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: expect.stringContaining('"model":"llama2-70b-4096"')
      })
    );
  });
  
  test('should check model compatibility correctly', () => {
    // Should be compatible
    expect(groq.isCompatibleModel('llama2-70b-4096')).toBe(true);
    expect(groq.isCompatibleModel('llama3-8b-8192')).toBe(true);
    expect(groq.isCompatibleModel('mistral-7b-instruct')).toBe(true);
    expect(groq.isCompatibleModel('mixtral-8x7b')).toBe(true);
    expect(groq.isCompatibleModel('groq-model')).toBe(true);
    
    // Should not be compatible
    expect(groq.isCompatibleModel('')).toBe(false);
    expect(groq.isCompatibleModel('gpt-4')).toBe(false);
    expect(groq.isCompatibleModel('claude-2')).toBe(false);
  });
  
  test('should implement chat method', async () => {
    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello, how are you?' }
    ];
    
    const response = await groq.chat(messages, {
      model: 'llama2-70b-4096',
      temperature: 0.7,
      maxTokens: 100
    });
    
    // Verify response format
    expect(response).toHaveProperty('content');
    expect(response.content).toBe('This is a test response from Groq API');
    expect(response.model).toBe('llama2-70b-4096');
    expect(response.usage).toBeDefined();
    expect(response.usage.total_tokens).toBe(35);
    
    // Verify the request
    expect(fetch).toHaveBeenCalledWith(
      'https://api.groq.com/openai/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"model":"llama2-70b-4096"')
      })
    );
    
    // Check that request body contains all the expected parameters
    const callArgs = (fetch as jest.Mock).mock.calls[0];
    const requestBody = JSON.parse(callArgs[1].body);
    
    expect(requestBody.model).toBe('llama2-70b-4096');
    expect(requestBody.temperature).toBe(0.7);
    expect(requestBody.max_tokens).toBe(100);
    expect(requestBody.messages).toEqual(messages);
  });
  
  test('should handle errors properly', async () => {
    // Mock a failed API response
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: {
            message: "Invalid API key",
            type: "authentication_error"
          }
        })
      }) as any
    );
    
    try {
      await groq.chat([{ role: 'user', content: 'Hello' }]);
      fail('Expected an error to be thrown');
    } catch (error: any) {
      expect(error.message).toBe('Invalid API key');
      expect(error.status).toBe(401);
      expect(error.type).toBe('authentication_error');
    }
  });
  
  test('should handle request timeouts', async () => {
    // Mock an aborted request
    const mockAbortError = new Error('The operation was aborted');
    mockAbortError.name = 'AbortError';
    
    global.fetch = jest.fn(() => {
      throw mockAbortError;
    });
    
    try {
      await groq.chat([{ role: 'user', content: 'Hello' }], { timeout: 1000 });
      fail('Expected an error to be thrown');
    } catch (error: any) {
      expect(error.message).toContain('timed out');
      expect(error.type).toBe('timeout_error');
      expect(error.status).toBe(408);
    }
  });
  
  test('should implement streaming chat', async () => {
    // Mock a streaming response
    const mockReadable = {
      getReader: jest.fn().mockReturnValue({
        read: jest.fn()
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"!"},"index":0,"finish_reason":"stop"}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: [DONE]\n\n')
          })
          .mockResolvedValueOnce({
            done: true,
            value: undefined
          })
      })
    };
    
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: true,
        body: mockReadable,
        status: 200
      } as any)
    );
    
    const messages = [
      { role: 'user', content: 'Say hello' }
    ];
    
    const stream = groq.streamChat(messages, {
      model: 'llama2-70b-4096',
      temperature: 0.8
    });
    
    // Collect all chunks
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    // Verify the chunks
    expect(chunks.length).toBe(3);
    expect(chunks[0].content).toBe('Hello');
    expect(chunks[1].content).toBe(' world');
    expect(chunks[2].content).toBe('!');
    
    // Verify the stream request
    expect(fetch).toHaveBeenCalledWith(
      'https://api.groq.com/openai/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
    
    // Check request body contains all parameters
    const callArgs = (fetch as jest.Mock).mock.calls[0];
    const requestBody = JSON.parse(callArgs[1].body);
    
    expect(requestBody.model).toBe('llama2-70b-4096');
    expect(requestBody.temperature).toBe(0.8);
    expect(requestBody.stream).toBe(true);
    expect(requestBody.messages).toEqual(messages);
  });
});
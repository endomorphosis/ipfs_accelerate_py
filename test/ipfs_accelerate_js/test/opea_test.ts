import { OPEA } from '../src/api_backends/opea';

describe('OPEA API Backend', () => {
  let opea: OPEA;
  
  beforeEach(() => {
    // Create a new instance for each test
    opea = new OPEA({}, {
      opea_api_url: 'http://localhost:8000',
      opea_api_key: 'test-api-key',
      opea_model: 'gpt-3.5-turbo'
    });
    
    // Mock fetch to avoid actual API calls
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          choices: [
            {
              message: {
                content: "This is a test response from OPEA API",
                role: "assistant"
              },
              finish_reason: "stop"
            }
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 15,
            total_tokens: 25
          },
          id: "test-response-id",
          created: Date.now(),
          model: "gpt-3.5-turbo"
        })
      }) as any
    );
  });
  
  test('should initialize with provided settings', () => {
    expect(opea).toBeDefined();
    expect((opea as any).apiUrl).toBe('http://localhost:8000');
    expect((opea as any).defaultModel).toBe('gpt-3.5-turbo');
  });
  
  test('should create an endpoint handler', () => {
    const handler = opea.createEndpointHandler();
    expect(typeof handler).toBe('function');
  });
  
  test('should create a custom endpoint handler for a specific URL', () => {
    const handler = opea.createEndpointHandler('http://custom-api.com/v1/chat/completions');
    expect(typeof handler).toBe('function');
  });
  
  test('should test an endpoint', async () => {
    const result = await opea.testEndpoint();
    expect(result).toBe(true);
    expect(fetch).toHaveBeenCalled();
  });
  
  test('should make a POST request to the OPEA API', async () => {
    const response = await (opea as any).makePostRequestOPEA(
      'http://localhost:8000/v1/chat/completions',
      { 
        messages: [{ role: 'user', content: 'Hello' }],
        model: 'gpt-3.5-turbo'
      }
    );
    
    expect(response).toHaveProperty('choices');
    expect(response.choices?.[0]?.message?.content).toBe('This is a test response from OPEA API');
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: expect.any(String)
      })
    );
  });
  
  test('should implement chat method', async () => {
    const messages = [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await opea.chat(messages);
    
    expect(response).toHaveProperty('content');
    expect(response.content).toBe('This is a test response from OPEA API');
    expect(response.role).toBe('assistant');
    expect(response.model).toBe('gpt-3.5-turbo');
    expect(response.usage).toBeDefined();
    expect(fetch).toHaveBeenCalled();
  });
  
  test('should properly handle request options in chat method', async () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    
    const options = {
      model: 'gpt-4',
      temperature: 0.7,
      max_tokens: 100,
      top_p: 0.9,
      stop: ['END']
    };
    
    await opea.chat(messages, options);
    
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/chat/completions',
      expect.objectContaining({
        body: expect.stringContaining('"model":"gpt-4"'),
      })
    );
    
    // Verify that body contains all the options
    const callArgs = (fetch as jest.Mock).mock.calls[0];
    const requestBody = JSON.parse(callArgs[1].body);
    
    expect(requestBody.temperature).toBe(0.7);
    expect(requestBody.max_tokens).toBe(100);
    expect(requestBody.top_p).toBe(0.9);
    expect(requestBody.stop).toEqual(['END']);
  });
  
  test('should handle errors correctly', async () => {
    // Mock a failed API response
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: false,
        status: 401,
        text: () => Promise.resolve('Unauthorized')
      }) as any
    );
    
    try {
      await opea.chat([{ role: 'user', content: 'Hello' }]);
      fail('Expected an error to be thrown');
    } catch (error: any) {
      expect(error.message).toContain('OPEA API error');
      expect(error.status).toBe(401);
    }
  });
  
  test('should check compatibility of models', () => {
    // Should be compatible
    expect(opea.isCompatibleModel('gpt-3.5-turbo')).toBe(true);
    expect(opea.isCompatibleModel('gpt-4')).toBe(true);
    expect(opea.isCompatibleModel('text-embedding-ada-002')).toBe(true);
    expect(opea.isCompatibleModel('claude-2')).toBe(true);
    expect(opea.isCompatibleModel('llama-7b')).toBe(true);
    expect(opea.isCompatibleModel('mistral-7b-instruct')).toBe(true);
    
    // Should not be compatible
    expect(opea.isCompatibleModel('')).toBe(false);
    expect(opea.isCompatibleModel('unknown-model')).toBe(false);
  });
  
  test('should implement streaming chat method', async () => {
    // Mock streaming response
    const mockReadable = {
      getReader: jest.fn().mockReturnValue({
        read: jest.fn()
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"This "},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"is "},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"streaming"},"index":0}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode('data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n')
          })
          .mockResolvedValueOnce({
            done: true,
            value: undefined
          }),
        cancel: jest.fn(),
        releaseLock: jest.fn()
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
      { role: 'user', content: 'Stream a response' }
    ];
    
    const stream = opea.streamChat(messages);
    
    // Collect all chunks
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    // Verify that we got all the expected chunks
    expect(chunks.length).toBe(6); // 5 delta chunks + final complete chunk
    
    // Check the roles and content
    expect(chunks[0].role).toBe('assistant');
    expect(chunks[0].content).toBe('');
    
    expect(chunks[1].content).toBe('This ');
    expect(chunks[2].content).toBe('is ');
    expect(chunks[3].content).toBe('streaming');
    
    // Check the final "done" chunk
    expect(chunks[4].done).toBe(true);
    
    // Check the complete response
    expect(chunks[5].type).toBe('complete');
    expect(chunks[5].content).toBe('This is streaming');
    expect(chunks[5].role).toBe('assistant');
    expect(chunks[5].done).toBe(true);
    
    // Verify request parameters
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
});
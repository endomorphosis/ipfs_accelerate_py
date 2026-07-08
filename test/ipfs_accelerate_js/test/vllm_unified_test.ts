import { VllmUnified } from '../src/api_backends/vllm_unified';

describe('VLLM Unified API Backend', () => {
  let vllmUnified: VllmUnified;
  
  beforeEach(() => {
    // Create a new instance for each test
    vllmUnified = new VllmUnified();
    
    // Mock fetch to avoid actual API calls
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          text: "This is a test response from VLLM API",
          metadata: {
            finish_reason: "length",
            model: "llama-7b",
            usage: {
              prompt_tokens: 10,
              completion_tokens: 20,
              total_tokens: 30
            }
          }
        })
      }) as any
    );
  });
  
  test('should initialize with default settings', () => {
    expect(vllmUnified).toBeDefined();
  });
  
  test('should create an endpoint handler', () => {
    const handler = vllmUnified.createEndpointHandler();
    expect(typeof handler).toBe('function');
  });
  
  test('should create a VLLM endpoint handler for a specific URL and model', () => {
    const handler = vllmUnified.createVllmEndpointHandler('http://localhost:8000', 'llama-7b');
    expect(typeof handler).toBe('function');
  });
  
  test('should test an endpoint', async () => {
    const result = await vllmUnified.testEndpoint();
    expect(result).toBe(true);
    expect(fetch).toHaveBeenCalled();
  });
  
  test('should make a POST request to the VLLM API', async () => {
    const response = await vllmUnified.makePostRequestVllm(
      'http://localhost:8000',
      { prompt: 'Hello', model: 'llama-7b' }
    );
    
    expect(response).toHaveProperty('text');
    expect(response.text).toBe('This is a test response from VLLM API');
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: expect.any(String)
      })
    );
  });
  
  test('should implement chat method', async () => {
    const messages = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await vllmUnified.chat(messages);
    
    expect(response).toHaveProperty('content');
    expect(response.model).toBeDefined();
    expect(response.usage).toBeDefined();
    expect(fetch).toHaveBeenCalled();
  });
  
  test('should properly format different types of inputs', async () => {
    // Create a mock handler function
    const mockHandler = jest.fn(() => Promise.resolve({ text: 'Formatted response' }));
    
    // Test with string input
    await vllmUnified.formatRequest(mockHandler, 'Hello world');
    expect(mockHandler).toHaveBeenCalledWith({ prompt: 'Hello world' });
    
    // Test with array input
    await vllmUnified.formatRequest(mockHandler, ['Hello', 'World']);
    expect(mockHandler).toHaveBeenCalledWith({ prompts: ['Hello', 'World'] });
    
    // Test with object input containing prompt
    await vllmUnified.formatRequest(mockHandler, { prompt: 'Hello', temperature: 0.7 });
    expect(mockHandler).toHaveBeenCalledWith({ prompt: 'Hello', temperature: 0.7 });
  });
});
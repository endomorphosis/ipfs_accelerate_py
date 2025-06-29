import { Groq } from '../../src/api_backends/groq';
import fetch from 'node-fetch';

// Mock the fetch function
jest.mock('node-fetch', () => {
  return jest.fn();
});

describe('Groq API Backend', () => {
  let groq: Groq;
  const mockApiKey = 'test-api-key';
  const mockMessages = [{ role: 'user', content: 'Hello' }];
  const mockResponse = {
    id: 'response-id',
    object: 'chat.completion',
    created: Date.now(),
    model: 'llama2-70b-4096',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: 'Hello there\! How can I help you today?'
        },
        finish_reason: 'stop'
      }
    ],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 9,
      total_tokens: 19
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
    groq = new Groq(
      {},
      { groq_api_key: mockApiKey }
    );
    
    // Mock successful response
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });
  });

  describe('chat', () => {
    it('should return a chat completion response', async () => {
      const result = await groq.chat(mockMessages);
      
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual({
        id: mockResponse.id,
        model: mockResponse.model,
        content: mockResponse.choices[0].message.content,
        created: mockResponse.created,
        usage: mockResponse.usage
      });
    });

    it('should use the provided model when specified', async () => {
      const customModel = 'mixtral-8x7b';
      
      await groq.chat(mockMessages, { model: customModel });
      
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);
      
      expect(requestBody.model).toBe(customModel);
    });

    it('should use the specified temperature', async () => {
      const temperature = 0.7;
      
      await groq.chat(mockMessages, { temperature });
      
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);
      
      expect(requestBody.temperature).toBe(temperature);
    });

    it('should apply the max tokens limit', async () => {
      const maxTokens = 100;
      
      await groq.chat(mockMessages, { maxTokens });
      
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);
      
      expect(requestBody.max_tokens).toBe(maxTokens);
    });
  });

  describe('streamChat', () => {
    beforeEach(() => {
      // Mock the ReadableStream for streaming responses
      const mockStream = {
        getReader: jest.fn().mockReturnValue({
          read: jest.fn()
            .mockResolvedValueOnce({ 
              done: false, 
              value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n')
            })
            .mockResolvedValueOnce({ 
              done: false, 
              value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":" there"}}]}\n\n')
            })
            .mockResolvedValueOnce({ 
              done: false, 
              value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"\!"}}]}\n\n')
            })
            .mockResolvedValueOnce({ 
              done: false, 
              value: new TextEncoder().encode('data: [DONE]\n\n')
            })
            .mockResolvedValueOnce({ done: true })
        })
      };
      
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        body: mockStream
      });
    });

    it('should yield stream chunks', async () => {
      const stream = groq.streamChat(mockMessages);
      
      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      
      expect(chunks).toHaveLength(3);
      expect(chunks[0].content).toBe('Hello');
      expect(chunks[1].content).toBe(' there');
      expect(chunks[2].content).toBe('\!');
    });
  });

  describe('error handling', () => {
    it('should throw an error when the API returns an error', async () => {
      const errorMessage = 'API error';
      
      (fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ error: { message: errorMessage } }),
      });
      
      await expect(groq.chat(mockMessages)).rejects.toThrow();
    });

    it('should throw an error when no API key is provided', async () => {
      groq = new Groq({}, {});
      
      await expect(groq.chat(mockMessages)).rejects.toThrow();
    });

    it('should handle timeout errors', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('AbortError'));
      
      await expect(groq.chat(mockMessages)).rejects.toThrow();
    });
  });

  describe('isCompatibleModel', () => {
    it('should return true for compatible models', () => {
      const compatibleModels = [
        'llama2-70b-4096',
        'mixtral-8x7b',
        'gemma-7b',
        'llama-2-7b-chat'
      ];
      
      compatibleModels.forEach(model => {
        expect(groq.isCompatibleModel(model)).toBe(true);
      });
    });

    it('should return false for incompatible models', () => {
      const incompatibleModels = [
        'gpt-4',
        'claude-3-opus',
        'anthropic-claude-2'
      ];
      
      incompatibleModels.forEach(model => {
        expect(groq.isCompatibleModel(model)).toBe(false);
      });
    });
  });

  describe('testEndpoint', () => {
    it('should return true when the endpoint works', async () => {
      const result = await groq.testEndpoint();
      
      expect(result).toBe(true);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should return false when the endpoint fails', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('Connection error'));
      
      const result = await groq.testEndpoint();
      
      expect(result).toBe(false);
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });
});

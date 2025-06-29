import { HfTeiUnified } from '../../src/api_backends/hf_tei_unified';
import fetch from 'node-fetch';

// Mock the fetch function
jest.mock('node-fetch', () => {
  return jest.fn();
});

describe('HfTeiUnified API Backend', () => {
  let hfTeiUnified: HfTeiUnified;
  const mockApiKey = 'test-api-key';
  const mockEmbedding = [0.1, 0.2, 0.3, 0.4, 0.5];
  const mockResponse = {
    embeddings: [mockEmbedding],
    model: 'test-model',
    dimensions: mockEmbedding.length
  };

  beforeEach(() => {
    jest.clearAllMocks();
    hfTeiUnified = new HfTeiUnified(
      {},
      { hf_tei_api_key: mockApiKey }
    );
    
    // Mock successful response
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });
  });

  describe('getEmbedding', () => {
    it('should return embeddings for a single text input', async () => {
      const text = 'test text';
      const result = await hfTeiUnified.getEmbedding(text);
      
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockEmbedding);
    });

    it('should return embeddings for multiple text inputs', async () => {
      const texts = ['test text 1', 'test text 2'];
      
      // Mock response for multiple inputs
      (fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          embeddings: [mockEmbedding, mockEmbedding],
          model: 'test-model',
          dimensions: mockEmbedding.length
        }),
      });
      
      const result = await hfTeiUnified.getEmbedding(texts);
      
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual([mockEmbedding, mockEmbedding]);
    });

    it('should use the specified model when provided', async () => {
      const text = 'test text';
      const model = 'custom-model';
      
      await hfTeiUnified.getEmbedding(text, { model });
      
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);
      
      expect(requestBody.model).toBe(model);
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
      
      await expect(hfTeiUnified.getEmbedding('test')).rejects.toThrow();
    });

    it('should throw an error when no API key is provided', async () => {
      hfTeiUnified = new HfTeiUnified({}, {});
      
      await expect(hfTeiUnified.getEmbedding('test')).rejects.toThrow();
    });
  });

  describe('isCompatibleModel', () => {
    it('should return true for compatible models', () => {
      const compatibleModels = [
        'sentence-transformers/all-MiniLM-L6-v2',
        'BAAI/bge-small-en-v1.5',
        'thenlper/gte-base',
        'openai/text-embedding-ada-002',
        'text-embedding'
      ];
      
      compatibleModels.forEach(model => {
        expect(hfTeiUnified.isCompatibleModel(model)).toBe(true);
      });
    });

    it('should return false for incompatible models', () => {
      const incompatibleModels = [
        'gpt-4',
        'mistral-7b',
        'llava-v1.5',
        'whisper-small'
      ];
      
      incompatibleModels.forEach(model => {
        expect(hfTeiUnified.isCompatibleModel(model)).toBe(false);
      });
    });
  });

  describe('testEndpoint', () => {
    it('should return true when the endpoint works', async () => {
      const result = await hfTeiUnified.testEndpoint();
      
      expect(result).toBe(true);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should return false when the endpoint fails', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('Connection error'));
      
      const result = await hfTeiUnified.testEndpoint();
      
      expect(result).toBe(false);
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });
});

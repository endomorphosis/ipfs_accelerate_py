/**
 * Tests for WebNN Standalone Implementation
 */

import {
  isWebNNSupported,
  getWebNNDeviceInfo,
  getWebNNBrowserRecommendations,
  runWebNNExample,
  isRecommendedBrowserForWebNN,
  getWebNNPerformanceTier
} from './webnn_standalone';

describe('WebNN Standalone', () => {
  describe('browser support detection', () => {
    it('should check if WebNN is supported', async () => {
      const isSupported = await isWebNNSupported();
      expect(typeof isSupported).toBe('boolean');
    });
    
    it('should get browser recommendations', () => {
      const recommendations = getWebNNBrowserRecommendations();
      
      expect(recommendations).toBeDefined();
      expect(recommendations.bestBrowser).toBe('edge');
      expect(recommendations.recommendation).toBeDefined();
      expect(recommendations.browserRanking).toBeDefined();
      expect(recommendations.currentBrowser).toBeDefined();
      expect(typeof recommendations.isUsingRecommendedBrowser).toBe('boolean');
    });
    
    it('should check if current browser is recommended', () => {
      const isRecommended = isRecommendedBrowserForWebNN();
      expect(typeof isRecommended).toBe('boolean');
    });
  });
  
  describe('device information', () => {
    it('should get WebNN device information', async () => {
      const deviceInfo = await getWebNNDeviceInfo();
      
      // In our test environment with mock WebNN, we should get valid info
      expect(deviceInfo).toBeDefined();
      
      if (deviceInfo) {
        expect(deviceInfo.deviceType).toBeDefined();
        expect(deviceInfo.operations).toBeDefined();
        expect(Array.isArray(deviceInfo.operations)).toBe(true);
      }
    });
    
    it('should detect performance tier', async () => {
      const tier = await getWebNNPerformanceTier();
      
      expect(['high', 'medium', 'low', 'unsupported']).toContain(tier);
    });
  });
  
  describe('example runs', () => {
    it('should run ReLU example', async () => {
      const result = await runWebNNExample('relu');
      
      expect(result).toBeDefined();
      expect(result.supported).toBeDefined();
      
      if (result.supported && result.initialized && !result.error) {
        expect(result.result).toBeDefined();
        expect(Array.isArray(result.result)).toBe(true);
        expect(result.performance).toBeDefined();
        expect(result.performance?.totalTime).toBeGreaterThan(0);
      }
    });
    
    it('should run matrix multiplication example', async () => {
      const result = await runWebNNExample('matmul');
      
      expect(result).toBeDefined();
      expect(result.supported).toBeDefined();
      
      if (result.supported && result.initialized && !result.error) {
        expect(result.result).toBeDefined();
        expect(Array.isArray(result.result)).toBe(true);
        expect(result.result?.length).toBe(4); // 2x2 matrix result
      }
    });
    
    it('should run sigmoid example', async () => {
      const result = await runWebNNExample('sigmoid');
      
      expect(result).toBeDefined();
      
      if (result.supported && result.initialized && !result.error) {
        expect(result.result).toBeDefined();
        expect(Array.isArray(result.result)).toBe(true);
      }
    });
    
    it('should run tanh example', async () => {
      const result = await runWebNNExample('tanh');
      
      expect(result).toBeDefined();
      
      if (result.supported && result.initialized && !result.error) {
        expect(result.result).toBeDefined();
        expect(Array.isArray(result.result)).toBe(true);
      }
    });
    
    it('should run softmax example', async () => {
      const result = await runWebNNExample('softmax');
      
      expect(result).toBeDefined();
      
      if (result.supported && result.initialized && !result.error) {
        expect(result.result).toBeDefined();
        expect(Array.isArray(result.result)).toBe(true);
      }
    });
  });
  
  describe('error handling', () => {
    it('should handle when WebNN is not supported', async () => {
      // Temporarily mock navigator.ml to be undefined
      const originalMl = navigator.ml;
      (navigator as any).ml = undefined;
      
      const result = await runWebNNExample();
      
      expect(result.supported).toBe(false);
      expect(result.initialized).toBe(false);
      expect(result.error).toBeDefined();
      
      // Restore navigator.ml
      (navigator as any).ml = originalMl;
    });
    
    it('should handle initialization errors', async () => {
      // Temporarily mock navigator.ml.createContext to throw
      const originalMl = navigator.ml;
      (navigator as any).ml = {
        async createContext() {
          throw new Error('Simulated error');
        }
      };
      
      const result = await runWebNNExample();
      
      expect(result.supported).toBe(true);
      expect(result.initialized).toBe(false);
      expect(result.error).toBeDefined();
      
      // Restore navigator.ml
      (navigator as any).ml = originalMl;
    });
    
    it('should handle operation errors', async () => {
      const originalMl = navigator.ml;
      
      // Set up WebNN support but make execution fail
      (navigator as any).ml = {
        async createContext() {
          return {
            createOperand: () => ({}),
          };
        },
        MLGraphBuilder: class MockBuilder {
          constructor() {}
          
          relu() {
            throw new Error('Simulated operation error');
          }
          
          build() {
            throw new Error('Simulated operation error');
          }
        }
      };
      
      const result = await runWebNNExample();
      
      expect(result.error).toBeDefined();
      
      // Restore navigator.ml
      (navigator as any).ml = originalMl;
    });
  });
});
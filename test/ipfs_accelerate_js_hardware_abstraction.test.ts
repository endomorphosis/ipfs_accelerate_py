/**
 * Hardware Abstraction Layer Tests
 * Tests for the hardware abstraction layer implementation
 */

import { HardwareAbstraction, createHardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';

describe('HardwareAbstraction', () => {
  let hal: HardwareAbstraction;

  beforeEach(() => {
    hal = new HardwareAbstraction();
  });

  afterEach(() => {
    hal.dispose();
  });

  describe('initialization', () => {
    it('should initialize successfully', async () => {
      const result = await hal.initialize();
      expect(result).toBe(true);
    });

    it('should detect available backends', async () => {
      await hal.initialize();
      const availableBackends = hal.getAvailableBackends();
      // At minimum, CPU should always be available
      expect(availableBackends).toContain('cpu');
    });

    it('should return hardware capabilities', async () => {
      await hal.initialize();
      const capabilities = hal.getCapabilities();
      expect(capabilities).not.toBeNull();
    });
  });

  describe('backend selection', () => {
    it('should select the best backend for a model type', async () => {
      await hal.initialize();
      
      // Test with various model types
      const visionBackend = hal.getBestBackend('vision');
      expect(visionBackend).not.toBeNull();
      
      const textBackend = hal.getBestBackend('text');
      expect(textBackend).not.toBeNull();
      
      const audioBackend = hal.getBestBackend('audio');
      expect(audioBackend).not.toBeNull();
    });

    it('should respect model preferences in options', async () => {
      // Create HAL with model preferences
      const customHal = new HardwareAbstraction({
        modelPreferences: {
          'vision': 'cpu',
          'text': 'cpu'
        }
      });
      
      await customHal.initialize();
      
      const visionBackend = customHal.getBestBackend('vision');
      expect(visionBackend).not.toBeNull();
      expect(visionBackend?.type).toBe('cpu');
      
      const textBackend = customHal.getBestBackend('text');
      expect(textBackend).not.toBeNull();
      expect(textBackend?.type).toBe('cpu');
      
      customHal.dispose();
    });
  });

  describe('execution', () => {
    it('should execute operations on the best available backend', async () => {
      await hal.initialize();
      
      // Create test tensors (simple 2x2 matrices)
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([5, 6, 7, 8]);
      const shape = [2, 2];
      
      // Execute matmul operation
      const result = await hal.execute('matmul', {
        a: { data: a, shape },
        b: { data: b, shape }
      }, {
        modelType: 'generic'
      });
      
      // Expected result:
      // [1, 2] * [5, 6] = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
      // [3, 4] * [7, 8] = [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
      
      expect(result).toBeDefined();
      expect(result.data).toBeDefined();
      expect(result.shape).toEqual([2, 2]);
      
      // Check if the result is approximately correct
      // Note: Different backends might have slight numerical differences
      const resultData = new Float32Array(result.data);
      expect(Math.abs(resultData[0] - 19)).toBeLessThan(0.1);
      expect(Math.abs(resultData[1] - 22)).toBeLessThan(0.1);
      expect(Math.abs(resultData[2] - 43)).toBeLessThan(0.1);
      expect(Math.abs(resultData[3] - 50)).toBeLessThan(0.1);
    });
    
    it('should use the preferred backend when specified', async () => {
      await hal.initialize();
      
      // Only proceed if CPU backend is available
      if (!hal.hasBackend('cpu')) {
        console.warn('CPU backend not available, skipping test');
        return;
      }
      
      // Create test tensors
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([5, 6, 7, 8]);
      const shape = [2, 2];
      
      // Execute with specified backend preference
      const result = await hal.execute('matmul', {
        a: { data: a, shape },
        b: { data: b, shape }
      }, {
        modelType: 'generic',
        preferredBackend: 'cpu'
      });
      
      expect(result).toBeDefined();
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('factory function', () => {
    it('should create and initialize HAL with factory function', async () => {
      const hal = await createHardwareAbstraction();
      
      expect(hal).toBeInstanceOf(HardwareAbstraction);
      expect(hal.getAvailableBackends().length).toBeGreaterThan(0);
      
      hal.dispose();
    });
  });
});
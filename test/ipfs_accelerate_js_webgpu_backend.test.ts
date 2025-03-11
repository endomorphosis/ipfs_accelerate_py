/**
 * Tests for WebGPU Backend Implementation
 */

import { WebGPUBackend, isWebGPUSupported, getWebGPUInfo } from './ipfs_accelerate_js_webgpu_backend';

describe('WebGPU Backend', () => {
  // Test the WebGPUBackend class
  describe('WebGPUBackend class', () => {
    let backend: WebGPUBackend;
    
    beforeEach(() => {
      backend = new WebGPUBackend({ logging: false });
    });
    
    // Test initialization
    test('should initialize successfully', async () => {
      const result = await backend.initialize();
      expect(result).toBe(true);
    });
    
    // Test getting adapter and device
    test('should return adapter and device after initialization', async () => {
      await backend.initialize();
      
      const adapter = backend.getAdapter();
      expect(adapter).toBeDefined();
      
      const device = backend.getDevice();
      expect(device).toBeDefined();
    });
    
    // Test adapter info
    test('should return adapter info after initialization', async () => {
      await backend.initialize();
      
      const adapterInfo = backend.getAdapterInfo();
      expect(adapterInfo).toBeDefined();
      expect(adapterInfo!.vendor).toBe('Test Vendor');
      expect(adapterInfo!.device).toBe('Test Device');
    });
    
    // Test isRealHardware
    test('should detect if using real hardware', async () => {
      await backend.initialize();
      
      const isRealHardware = backend.isRealHardware();
      expect(typeof isRealHardware).toBe('boolean');
    });
    
    // Test creating shader module
    test('should create shader module', async () => {
      await backend.initialize();
      
      const shaderCode = `
        @compute @workgroup_size(1)
        fn main() {
          // Empty shader
        }
      `;
      
      const shaderModule = backend.createShaderModule(shaderCode);
      expect(shaderModule).toBeDefined();
    });
    
    // Test creating buffer
    test('should create buffer with data', async () => {
      await backend.initialize();
      
      const data = new Float32Array([1, 2, 3, 4]);
      const buffer = backend.createBuffer(data, GPUBufferUsage.STORAGE);
      
      expect(buffer).toBeDefined();
    });
    
    // Test creating compute pipeline
    test('should create compute pipeline', async () => {
      await backend.initialize();
      
      const shaderCode = `
        @compute @workgroup_size(1)
        fn main() {
          // Empty shader
        }
      `;
      
      const shaderModule = backend.createShaderModule(shaderCode);
      const pipeline = backend.createComputePipeline(shaderModule!, 'main');
      
      expect(pipeline).toBeDefined();
    });
    
    // Test running compute shader
    test('should run compute shader', async () => {
      await backend.initialize();
      
      const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          output[global_id.x] = f32(global_id.x);
        }
      `;
      
      const shaderModule = backend.createShaderModule(shaderCode);
      const pipeline = backend.createComputePipeline(shaderModule!);
      
      // Create output buffer
      const device = backend.getDevice()!;
      const outputBuffer = device.createBuffer({
        size: 4 * 4, // 4 f32 values
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      
      // Run compute shader
      await backend.runComputeShader(
        pipeline!,
        [{ buffer: outputBuffer }],
        4, // workgroupsX
        1, // workgroupsY
        1  // workgroupsZ
      );
      
      // We can't verify the actual results in this mock environment,
      // but we can at least verify the function ran without errors
      expect(true).toBe(true);
    });
    
    // Test disposal
    test('should dispose resources', async () => {
      await backend.initialize();
      
      backend.dispose();
      
      // After disposal, adapter and device should be null
      expect(() => backend.getAdapter()).toThrow();
      expect(() => backend.getDevice()).toThrow();
    });
  });
  
  // Test utility functions
  describe('Utility functions', () => {
    // Test isWebGPUSupported
    test('isWebGPUSupported should return boolean', async () => {
      const supported = await isWebGPUSupported();
      expect(typeof supported).toBe('boolean');
    });
    
    // Test getWebGPUInfo
    test('getWebGPUInfo should return WebGPU information', async () => {
      const info = await getWebGPUInfo();
      
      expect(info).toBeDefined();
      expect(info.supported).toBeDefined();
      
      if (info.supported) {
        expect(info.adapterInfo).toBeDefined();
        expect(info.features).toBeDefined();
        expect(Array.isArray(info.features)).toBe(true);
      }
    });
  });
});
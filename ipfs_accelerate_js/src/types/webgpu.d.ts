/**
 * Enhanced TypeScript definitions for WebGPU
 */

// WebGPU Buffer
interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

interface GPUBuffer {
  readonly size: number;
  readonly usage: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

// WebGPU Texture
interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: number;
}

type GPUTextureDimension = '1d' | '2d' | '3d';
type GPUTextureFormat = 'rgba8unorm' | 'rgba16float' | 'rgba32float' | 'r8unorm' | 'r16float' | 'r32float' | string;

interface GPUExtent3D {
  width: number;
  height?: number;
  depthOrArrayLayers?: number;
}

interface GPUTexture {
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface GPUTextureViewDescriptor {
  format?: GPUTextureFormat;
  dimension?: GPUTextureViewDimension;
  aspect?: GPUTextureAspect;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
}

type GPUTextureViewDimension = '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only';

interface GPUTextureView {
  // Empty interface for type checking
}

// WebGPU Shader
interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
  sourceMap?: object;
}

interface GPUShaderModule {
  // Empty interface for type checking
}

// WebGPU Pipeline
interface GPUComputePipelineDescriptor {
  label?: string;
  layout?: GPUPipelineLayout | 'auto';
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
  };
}

interface GPUComputePipeline {
  // Empty interface for type checking
}

// WebGPU Pass
interface GPUComputePassDescriptor {
  label?: string;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

// WebGPU Commands
interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPUCommandBuffer {
  // Empty interface for type checking
}

// WebGPU Queue
interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number
  ): void;
}

// WebGPU Device
interface GPUDevice {
  readonly queue: GPUQueue;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  destroy(): void;
}

// WebGPU Adapter
interface GPUAdapter {
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

// WebGPU Bind Group
interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: any;
  sampler?: any;
  texture?: any;
  storageTexture?: any;
}

interface GPUBindGroupLayout {
  // Empty interface for type checking
}

interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: GPUBindGroupLayoutEntry[];
}

interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

interface GPUBindGroup {
  // Empty interface for type checking
}

// Navigator interface
interface NavigatorGPU {
  gpu: {
    requestAdapter(): Promise<GPUAdapter>;
  };
}

interface Navigator extends NavigatorGPU {}

// Type for our code
export type WebGPUBackendType = 'webgpu';

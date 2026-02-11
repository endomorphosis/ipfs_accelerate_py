/**
 * TypeScript definitions for WebGPU
 * Based on the WebGPU specification: https://gpuweb.github.io/gpuweb/
 */

/**
 * GPUBufferUsageFlags - Flags for controlling buffer usage
 */
declare type GPUBufferUsageFlags = number;

/**
 * GPUDevice - The primary interface for WebGPU
 */
interface GPUDevice {
  // Resource creation
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
  
  // Pipeline creation
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  
  // Binding
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  
  // Command submission
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  
  // Queue access
  queue: GPUQueue;
  
  // Device information
  adapter: GPUAdapter;
  features: Set<string>;
  limits: Record<string, number>;
  
  // Event handling
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
  
  // Device loss
  lost: Promise<GPUDeviceLostInfo>;
  
  // Debugging
  pushErrorScope(filter: GPUErrorFilter): void;
  popErrorScope(): Promise<GPUError | null>;
  
  // Experimental
  createQuerySet?(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
}

/**
 * GPUBuffer - Represents a block of memory that can be used in GPU operations
 */
interface GPUBuffer {
  // Properties
  size: number;
  usage: GPUBufferUsageFlags;
  
  // Methods
  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

/**
 * GPUBufferDescriptor - Description of a buffer to be created
 */
interface GPUBufferDescriptor {
  size: number;
  usage: GPUBufferUsageFlags;
  mappedAtCreation?: boolean;
  label?: string;
}

/**
 * GPUShaderModuleDescriptor - Description of a shader module to be created
 */
interface GPUShaderModuleDescriptor {
  code: string;
  label?: string;
}

/**
 * GPUComputePipelineDescriptor - Description of a compute pipeline to be created
 */
interface GPUComputePipelineDescriptor {
  layout: GPUPipelineLayout | 'auto';
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
  };
  label?: string;
}

/**
 * GPUPipelineLayout - Layout of a pipeline
 */
interface GPUPipelineLayout {
  // This interface is mostly used as a marker
}

/**
 * GPUPipelineLayoutDescriptor - Description of a pipeline layout to be created
 */
interface GPUPipelineLayoutDescriptor {
  bindGroupLayouts: GPUBindGroupLayout[];
  label?: string;
}

/**
 * GPUBindGroupLayout - Layout of a bind group
 */
interface GPUBindGroupLayout {
  // This interface is mostly used as a marker
}

/**
 * GPUBindGroupLayoutDescriptor - Description of a bind group layout to be created
 */
interface GPUBindGroupLayoutDescriptor {
  entries: GPUBindGroupLayoutEntry[];
  label?: string;
}

/**
 * GPUBindGroupLayoutEntry - Entry in a bind group layout
 */
interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: GPUShaderStageFlags;
  buffer?: GPUBufferBindingLayout;
  sampler?: GPUSamplerBindingLayout;
  texture?: GPUTextureBindingLayout;
  storageTexture?: GPUStorageTextureBindingLayout;
}

/**
 * GPUBufferBindingLayout - Layout for a buffer binding
 */
interface GPUBufferBindingLayout {
  type?: 'uniform' | 'storage' | 'read-only-storage';
  hasDynamicOffset?: boolean;
  minBindingSize?: number;
}

/**
 * GPUSamplerBindingLayout - Layout for a sampler binding
 */
interface GPUSamplerBindingLayout {
  type?: 'filtering' | 'non-filtering' | 'comparison';
}

/**
 * GPUTextureBindingLayout - Layout for a texture binding
 */
interface GPUTextureBindingLayout {
  sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';
  viewDimension?: GPUTextureViewDimension;
  multisampled?: boolean;
}

/**
 * GPUStorageTextureBindingLayout - Layout for a storage texture binding
 */
interface GPUStorageTextureBindingLayout {
  access: 'write-only';
  format: GPUTextureFormat;
  viewDimension?: GPUTextureViewDimension;
}

/**
 * GPUTextureFormat - Format of a texture
 */
type GPUTextureFormat = string;

/**
 * GPUTextureViewDimension - Dimension of a texture view
 */
type GPUTextureViewDimension = '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';

/**
 * GPUShaderStageFlags - Flags for shader stages
 */
type GPUShaderStageFlags = number;

/**
 * GPUBindGroup - Group of resource bindings
 */
interface GPUBindGroup {
  // This interface is mostly used as a marker
}

/**
 * GPUBindGroupDescriptor - Description of a bind group to be created
 */
interface GPUBindGroupDescriptor {
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
  label?: string;
}

/**
 * GPUBindGroupEntry - Entry in a bind group
 */
interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

/**
 * GPUBindingResource - Resource to be bound
 * This is a union type in practice
 */
type GPUBindingResource = GPUSampler | GPUTextureView | GPUBufferBinding;

/**
 * GPUBufferBinding - Binding information for a buffer
 */
interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

/**
 * GPUTexture - Represents an image that can be used in GPU operations
 */
interface GPUTexture {
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

/**
 * GPUTextureDescriptor - Description of a texture to be created
 */
interface GPUTextureDescriptor {
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
  label?: string;
}

/**
 * GPUExtent3D - Size of a 3D object
 */
type GPUExtent3D = number[] | { width: number; height?: number; depthOrArrayLayers?: number };

/**
 * GPUTextureDimension - Dimension of a texture
 */
type GPUTextureDimension = '1d' | '2d' | '3d';

/**
 * GPUTextureUsageFlags - Flags for controlling texture usage
 */
type GPUTextureUsageFlags = number;

/**
 * GPUTextureView - View of a texture
 */
interface GPUTextureView {
  // This interface is mostly used as a marker
}

/**
 * GPUTextureViewDescriptor - Description of a texture view to be created
 */
interface GPUTextureViewDescriptor {
  format?: GPUTextureFormat;
  dimension?: GPUTextureViewDimension;
  aspect?: GPUTextureAspect;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
  label?: string;
}

/**
 * GPUTextureAspect - Aspect of a texture
 */
type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only';

/**
 * GPUSampler - Controls how textures are sampled in shaders
 */
interface GPUSampler {
  // This interface is mostly used as a marker
}

/**
 * GPUSamplerDescriptor - Description of a sampler to be created
 */
interface GPUSamplerDescriptor {
  addressModeU?: GPUAddressMode;
  addressModeV?: GPUAddressMode;
  addressModeW?: GPUAddressMode;
  magFilter?: GPUFilterMode;
  minFilter?: GPUFilterMode;
  mipmapFilter?: GPUFilterMode;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: GPUCompareFunction;
  maxAnisotropy?: number;
  label?: string;
}

/**
 * GPUAddressMode - Addressing mode for texture coordinates
 */
type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat';

/**
 * GPUFilterMode - Filtering mode for texture sampling
 */
type GPUFilterMode = 'nearest' | 'linear';

/**
 * GPUCompareFunction - Comparison function
 */
type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';

/**
 * GPUCommandEncoder - Builds a sequence of commands
 */
interface GPUCommandEncoder {
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number
  ): void;
  copyBufferToTexture(
    source: GPUImageCopyBuffer,
    destination: GPUImageCopyTexture,
    copySize: GPUExtent3D
  ): void;
  copyTextureToBuffer(
    source: GPUImageCopyTexture,
    destination: GPUImageCopyBuffer,
    copySize: GPUExtent3D
  ): void;
  copyTextureToTexture(
    source: GPUImageCopyTexture,
    destination: GPUImageCopyTexture,
    copySize: GPUExtent3D
  ): void;
  resolveQuerySet?(
    querySet: GPUQuerySet,
    firstQuery: number,
    queryCount: number,
    destination: GPUBuffer,
    destinationOffset: number
  ): void;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

/**
 * GPUComputePassEncoder - Records commands in a compute pass
 */
interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(
    index: number,
    bindGroup: GPUBindGroup,
    dynamicOffsets?: number[]
  ): void;
  dispatchWorkgroups(
    workgroupCountX: number,
    workgroupCountY?: number,
    workgroupCountZ?: number
  ): void;
  dispatchWorkgroupsIndirect(
    indirectBuffer: GPUBuffer,
    indirectOffset: number
  ): void;
  end(): void;
}

/**
 * GPUComputePassDescriptor - Description of a compute pass
 */
interface GPUComputePassDescriptor {
  label?: string;
}

/**
 * GPUComputePipeline - Pipeline for compute operations
 */
interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

/**
 * GPURenderPassEncoder - Records commands in a render pass
 */
interface GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(
    index: number,
    bindGroup: GPUBindGroup,
    dynamicOffsets?: number[]
  ): void;
  setViewport(
    x: number,
    y: number,
    width: number,
    height: number,
    minDepth: number,
    maxDepth: number
  ): void;
  setScissorRect(x: number, y: number, width: number, height: number): void;
  setBlendConstant(color: GPUColor): void;
  setStencilReference(reference: number): void;
  beginOcclusionQuery(queryIndex: number): void;
  endOcclusionQuery(): void;
  executeBundles(bundles: GPURenderBundle[]): void;
  end(): void;

  // Draw calls
  draw(
    vertexCount: number,
    instanceCount?: number,
    firstVertex?: number,
    firstInstance?: number
  ): void;
  drawIndirect(
    indirectBuffer: GPUBuffer,
    indirectOffset: number
  ): void;
  drawIndexed(
    indexCount: number,
    instanceCount?: number,
    firstIndex?: number,
    baseVertex?: number,
    firstInstance?: number
  ): void;
  drawIndexedIndirect(
    indirectBuffer: GPUBuffer,
    indirectOffset: number
  ): void;
}

/**
 * GPURenderPassDescriptor - Description of a render pass
 */
interface GPURenderPassDescriptor {
  colorAttachments: (GPURenderPassColorAttachment | null)[];
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  occlusionQuerySet?: GPUQuerySet;
  label?: string;
}

/**
 * GPURenderPassColorAttachment - Color attachment for a render pass
 */
interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  resolveTarget?: GPUTextureView;
  clearValue?: GPUColor;
  loadOp: GPULoadOp;
  storeOp: GPUStoreOp;
}

/**
 * GPUColor - RGBA color
 */
type GPUColor = [number, number, number, number] | { r: number; g: number; b: number; a: number };

/**
 * GPULoadOp - Load operation for attachments
 */
type GPULoadOp = 'load' | 'clear';

/**
 * GPUStoreOp - Store operation for attachments
 */
type GPUStoreOp = 'store' | 'discard';

/**
 * GPURenderPassDepthStencilAttachment - Depth/stencil attachment for a render pass
 */
interface GPURenderPassDepthStencilAttachment {
  view: GPUTextureView;
  depthClearValue?: number;
  depthLoadOp?: GPULoadOp;
  depthStoreOp?: GPUStoreOp;
  depthReadOnly?: boolean;
  stencilClearValue?: number;
  stencilLoadOp?: GPULoadOp;
  stencilStoreOp?: GPUStoreOp;
  stencilReadOnly?: boolean;
}

/**
 * GPURenderPipeline - Pipeline for render operations
 */
interface GPURenderPipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

/**
 * GPURenderPipelineDescriptor - Description of a render pipeline to be created
 */
interface GPURenderPipelineDescriptor {
  layout: GPUPipelineLayout | 'auto';
  vertex: {
    module: GPUShaderModule;
    entryPoint: string;
    buffers?: GPUVertexBufferLayout[];
  };
  primitive?: {
    topology?: GPUPrimitiveTopology;
    stripIndexFormat?: GPUIndexFormat;
    frontFace?: GPUFrontFace;
    cullMode?: GPUCullMode;
  };
  depthStencil?: GPUDepthStencilState;
  multisample?: {
    count?: number;
    mask?: number;
    alphaToCoverageEnabled?: boolean;
  };
  fragment?: {
    module: GPUShaderModule;
    entryPoint: string;
    targets: GPUColorTargetState[];
  };
  label?: string;
}

/**
 * GPUVertexBufferLayout - Layout of a vertex buffer
 */
interface GPUVertexBufferLayout {
  arrayStride: number;
  attributes: GPUVertexAttribute[];
  stepMode?: GPUVertexStepMode;
}

/**
 * GPUVertexAttribute - Attribute in a vertex buffer
 */
interface GPUVertexAttribute {
  format: GPUVertexFormat;
  offset: number;
  shaderLocation: number;
}

/**
 * GPUVertexFormat - Format of a vertex attribute
 */
type GPUVertexFormat = string;

/**
 * GPUVertexStepMode - Step mode for vertex attributes
 */
type GPUVertexStepMode = 'vertex' | 'instance';

/**
 * GPUPrimitiveTopology - Topology of primitives
 */
type GPUPrimitiveTopology = 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';

/**
 * GPUIndexFormat - Format of indices
 */
type GPUIndexFormat = 'uint16' | 'uint32';

/**
 * GPUFrontFace - Winding order for front face
 */
type GPUFrontFace = 'ccw' | 'cw';

/**
 * GPUCullMode - Culling mode
 */
type GPUCullMode = 'none' | 'front' | 'back';

/**
 * GPUDepthStencilState - State for depth and stencil operations
 */
interface GPUDepthStencilState {
  format: GPUTextureFormat;
  depthWriteEnabled: boolean;
  depthCompare: GPUCompareFunction;
  stencilFront?: GPUStencilFaceState;
  stencilBack?: GPUStencilFaceState;
  stencilReadMask?: number;
  stencilWriteMask?: number;
  depthBias?: number;
  depthBiasSlopeScale?: number;
  depthBiasClamp?: number;
}

/**
 * GPUStencilFaceState - State for stencil operations on a face
 */
interface GPUStencilFaceState {
  compare?: GPUCompareFunction;
  failOp?: GPUStencilOperation;
  depthFailOp?: GPUStencilOperation;
  passOp?: GPUStencilOperation;
}

/**
 * GPUStencilOperation - Operation for stencil
 */
type GPUStencilOperation = 'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap';

/**
 * GPUColorTargetState - State for a color target
 */
interface GPUColorTargetState {
  format: GPUTextureFormat;
  blend?: GPUBlendState;
  writeMask?: GPUColorWriteFlags;
}

/**
 * GPUBlendState - State for blending
 */
interface GPUBlendState {
  color: GPUBlendComponent;
  alpha: GPUBlendComponent;
}

/**
 * GPUBlendComponent - Component of blend state
 */
interface GPUBlendComponent {
  srcFactor?: GPUBlendFactor;
  dstFactor?: GPUBlendFactor;
  operation?: GPUBlendOperation;
}

/**
 * GPUBlendFactor - Factor for blending
 */
type GPUBlendFactor = string;

/**
 * GPUBlendOperation - Operation for blending
 */
type GPUBlendOperation = 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max';

/**
 * GPUColorWriteFlags - Flags for color write mask
 */
type GPUColorWriteFlags = number;

/**
 * GPUImageCopyBuffer - Buffer for image copy
 */
interface GPUImageCopyBuffer {
  buffer: GPUBuffer;
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

/**
 * GPUImageCopyTexture - Texture for image copy
 */
interface GPUImageCopyTexture {
  texture: GPUTexture;
  mipLevel?: number;
  origin?: GPUOrigin3D;
  aspect?: GPUTextureAspect;
}

/**
 * GPUOrigin3D - Origin of a 3D object
 */
type GPUOrigin3D = number[] | { x?: number; y?: number; z?: number };

/**
 * GPUCommandBuffer - Buffer of commands for the GPU
 */
interface GPUCommandBuffer {
  // Properties
  label?: string;
  
  // Methods
  executionTime?: Promise<number>;
}

/**
 * GPUCommandBufferDescriptor - Description of a command buffer
 */
interface GPUCommandBufferDescriptor {
  label?: string;
}

/**
 * GPUCommandEncoderDescriptor - Description of a command encoder
 */
interface GPUCommandEncoderDescriptor {
  label?: string;
}

/**
 * GPUShaderModule - Module containing shader code
 */
interface GPUShaderModule {
  // This interface is mostly used as a marker
}

/**
 * GPUQueue - Queue for submitting commands
 */
interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  onSubmittedWorkDone(): Promise<void>;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number
  ): void;
  writeTexture(
    destination: GPUImageCopyTexture,
    data: BufferSource,
    dataLayout: GPUImageDataLayout,
    size: GPUExtent3D
  ): void;
}

/**
 * GPUImageDataLayout - Layout of image data
 */
interface GPUImageDataLayout {
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

/**
 * GPUQuerySet - Set of queries
 */
interface GPUQuerySet {
  destroy(): void;
}

/**
 * GPUQuerySetDescriptor - Description of a query set
 */
interface GPUQuerySetDescriptor {
  type: GPUQueryType;
  count: number;
  label?: string;
}

/**
 * GPUQueryType - Type of query
 */
type GPUQueryType = 'occlusion' | 'timestamp';

/**
 * GPUDeviceLostInfo - Information about a lost device
 */
interface GPUDeviceLostInfo {
  reason?: string;
  message: string;
}

/**
 * GPUError - Error from the GPU
 */
interface GPUError {
  message: string;
}

/**
 * GPUErrorFilter - Filter for errors
 */
type GPUErrorFilter = 'out-of-memory' | 'validation';

/**
 * GPURenderBundle - Bundle of render commands
 */
interface GPURenderBundle {
  // This interface is mostly used as a marker
}

/**
 * GPUAdapter - Represents a physical device
 */
interface GPUAdapter {
  // Properties
  name: string;
  features: Set<string>;
  limits: Record<string, number>;
  
  // Methods
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

/**
 * GPUDeviceDescriptor - Description of a device to be requested
 */
interface GPUDeviceDescriptor {
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
  label?: string;
}

/**
 * GPUMapModeFlags - Flags for buffer mapping
 */
type GPUMapModeFlags = number;

/**
 * Constants for WebGPU
 */
declare namespace GPUBufferUsage {
  const MAP_READ: number;
  const MAP_WRITE: number;
  const COPY_SRC: number;
  const COPY_DST: number;
  const INDEX: number;
  const VERTEX: number;
  const UNIFORM: number;
  const STORAGE: number;
  const INDIRECT: number;
  const QUERY_RESOLVE: number;
}

declare namespace GPUShaderStage {
  const VERTEX: number;
  const FRAGMENT: number;
  const COMPUTE: number;
}

declare namespace GPUColorWrite {
  const RED: number;
  const GREEN: number;
  const BLUE: number;
  const ALPHA: number;
  const ALL: number;
}

declare namespace GPUTextureUsage {
  const COPY_SRC: number;
  const COPY_DST: number;
  const TEXTURE_BINDING: number;
  const STORAGE_BINDING: number;
  const RENDER_ATTACHMENT: number;
}

declare namespace GPUMapMode {
  const READ: number;
  const WRITE: number;
}

/**
 * NavigatorGPU - Extension of Navigator with GPU support
 */
interface NavigatorGPU {
  gpu: GPU;
}

/**
 * GPU - Entry point for WebGPU
 */
interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

/**
 * GPURequestAdapterOptions - Options for adapter request
 */
interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

/**
 * Extend Navigator to include GPU
 */
interface Navigator extends NavigatorGPU {}

/**
 * Extend Window for WorkerNavigator
 */
interface WorkerNavigator extends NavigatorGPU {}
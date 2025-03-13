/**
 * WebGPU type definitions
 * 
 * This file provides TypeScript type definitions for WebGPU API.
 * These are needed for compatibility with different TypeScript versions
 * or when working with custom implementations.
 */

// WebGPU API types
declare global {
  interface Navigator {
    readonly gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPURequestAdapterOptions {
    powerPreference?: 'high-performance' | 'low-power';
    forceFallbackAdapter?: boolean;
  }

  interface GPUAdapter {
    readonly features: Set<GPUFeatureName>;
    readonly limits: GPUSupportedLimits;
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    requestAdapterInfo(unmaskHints?: string[]): Promise<GPUAdapterInfo>;
  }

  interface GPUAdapterInfo {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  }

  interface GPUDevice {
    readonly features: Set<GPUFeatureName>;
    readonly limits: GPUSupportedLimits;
    readonly queue: GPUQueue;
    destroy(): void;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
    createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    addEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions): void;
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions): void;
  }

  interface GPUSupportedLimits {
    readonly maxTextureDimension1D?: number;
    readonly maxTextureDimension2D?: number;
    readonly maxTextureDimension3D?: number;
    readonly maxTextureArrayLayers?: number;
    readonly maxBindGroups?: number;
    readonly maxBindingsPerBindGroup?: number;
    readonly maxDynamicUniformBuffersPerPipelineLayout?: number;
    readonly maxDynamicStorageBuffersPerPipelineLayout?: number;
    readonly maxSampledTexturesPerShaderStage?: number;
    readonly maxSamplersPerShaderStage?: number;
    readonly maxStorageBuffersPerShaderStage?: number;
    readonly maxStorageTexturesPerShaderStage?: number;
    readonly maxUniformBuffersPerShaderStage?: number;
    readonly maxUniformBufferBindingSize?: number;
    readonly maxStorageBufferBindingSize?: number;
    readonly minUniformBufferOffsetAlignment?: number;
    readonly minStorageBufferOffsetAlignment?: number;
    readonly maxVertexBuffers?: number;
    readonly maxVertexAttributes?: number;
    readonly maxVertexBufferArrayStride?: number;
    readonly maxInterStageShaderComponents?: number;
    readonly maxComputeWorkgroupStorageSize?: number;
    readonly maxComputeInvocationsPerWorkgroup?: number;
    readonly maxComputeWorkgroupSizeX?: number;
    readonly maxComputeWorkgroupSizeY?: number;
    readonly maxComputeWorkgroupSizeZ?: number;
    readonly maxComputeWorkgroupsPerDimension?: number;
  }

  interface GPUQueue {
    submit(commandBuffers: Iterable<GPUCommandBuffer>): void;
    onSubmittedWorkDone(): Promise<void>;
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource, dataOffset?: number, size?: number): void;
    writeTexture(destination: GPUImageCopyTexture, data: BufferSource, dataLayout: GPUImageDataLayout, size: GPUExtent3D): void;
  }

  interface GPUShaderModuleDescriptor {
    code: string;
    sourceMap?: any;
  }

  interface GPUShaderModule {
    readonly compilationInfo: Promise<GPUCompilationInfo>;
  }

  interface GPUCompilationInfo {
    readonly messages: ReadonlyArray<GPUCompilationMessage>;
  }

  interface GPUCompilationMessage {
    readonly message: string;
    readonly type: GPUCompilationMessageType;
    readonly lineNum: number;
    readonly linePos: number;
    readonly offset: number;
    readonly length: number;
  }

  type GPUCompilationMessageType = 'error' | 'warning' | 'info';

  interface GPUBufferDescriptor {
    size: number;
    usage: GPUBufferUsageFlags;
    mappedAtCreation?: boolean;
  }

  interface GPUBuffer {
    readonly size: number;
    readonly usage: GPUBufferUsageFlags;
    mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  type GPUBufferUsageFlags = number;
  type GPUMapModeFlags = number;

  interface GPUTextureDescriptor {
    size: GPUExtent3D;
    mipLevelCount?: number;
    sampleCount?: number;
    dimension?: GPUTextureDimension;
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
  }

  interface GPUExtent3D {
    width: number;
    height?: number;
    depthOrArrayLayers?: number;
  }

  type GPUTextureDimension = '1d' | '2d' | '3d';
  type GPUTextureFormat = string;
  type GPUTextureUsageFlags = number;

  interface GPUTexture {
    readonly width: number;
    readonly height: number;
    readonly depthOrArrayLayers: number;
    readonly mipLevelCount: number;
    readonly sampleCount: number;
    readonly dimension: GPUTextureDimension;
    readonly format: GPUTextureFormat;
    readonly usage: GPUTextureUsageFlags;
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
  }

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
  }

  type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat';
  type GPUFilterMode = 'nearest' | 'linear';
  type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';

  interface GPUSampler {
  }

  interface GPUBindGroupLayoutDescriptor {
    entries: GPUBindGroupLayoutEntry[];
  }

  interface GPUBindGroupLayoutEntry {
    binding: number;
    visibility: GPUShaderStageFlags;
    buffer?: GPUBufferBindingLayout;
    sampler?: GPUSamplerBindingLayout;
    texture?: GPUTextureBindingLayout;
    storageTexture?: GPUStorageTextureBindingLayout;
  }

  type GPUShaderStageFlags = number;

  interface GPUBufferBindingLayout {
    type?: GPUBufferBindingType;
    hasDynamicOffset?: boolean;
    minBindingSize?: number;
  }

  type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';

  interface GPUSamplerBindingLayout {
    type?: GPUSamplerBindingType;
  }

  type GPUSamplerBindingType = 'filtering' | 'non-filtering' | 'comparison';

  interface GPUTextureBindingLayout {
    sampleType?: GPUTextureSampleType;
    viewDimension?: GPUTextureViewDimension;
    multisampled?: boolean;
  }

  type GPUTextureSampleType = 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';

  interface GPUStorageTextureBindingLayout {
    access?: GPUStorageTextureAccess;
    format: GPUTextureFormat;
    viewDimension?: GPUTextureViewDimension;
  }

  type GPUStorageTextureAccess = 'write-only';

  interface GPUBindGroupLayout {
  }

  interface GPUPipelineLayoutDescriptor {
    bindGroupLayouts: GPUBindGroupLayout[];
  }

  interface GPUPipelineLayout {
  }

  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }

  interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBindingResource;
  }

  type GPUBindingResource = GPUSampler | GPUTextureView | GPUBufferBinding | GPUExternalTexture;

  interface GPUBufferBinding {
    buffer: GPUBuffer;
    offset?: number;
    size?: number;
  }

  interface GPUExternalTexture {
  }

  interface GPUBindGroup {
  }

  interface GPUPipelineDescriptorBase {
    layout?: GPUPipelineLayout | 'auto';
  }

  interface GPUComputePipelineDescriptor extends GPUPipelineDescriptorBase {
    compute: GPUProgrammableStage;
  }

  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
    constants?: Record<string, number>;
  }

  interface GPUComputePipeline {
    readonly layout: GPUPipelineLayout;
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPURenderPipelineDescriptor extends GPUPipelineDescriptorBase {
    vertex: GPUVertexState;
    primitive?: GPUPrimitiveState;
    depthStencil?: GPUDepthStencilState;
    multisample?: GPUMultisampleState;
    fragment?: GPUFragmentState;
  }

  interface GPUVertexState extends GPUProgrammableStage {
    buffers?: GPUVertexBufferLayout[];
  }

  interface GPUVertexBufferLayout {
    arrayStride: number;
    stepMode?: GPUVertexStepMode;
    attributes: GPUVertexAttribute[];
  }

  type GPUVertexStepMode = 'vertex' | 'instance';

  interface GPUVertexAttribute {
    format: GPUVertexFormat;
    offset: number;
    shaderLocation: number;
  }

  type GPUVertexFormat = string;

  interface GPUPrimitiveState {
    topology?: GPUPrimitiveTopology;
    stripIndexFormat?: GPUIndexFormat;
    frontFace?: GPUFrontFace;
    cullMode?: GPUCullMode;
    unclippedDepth?: boolean;
  }

  type GPUPrimitiveTopology = 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';
  type GPUIndexFormat = 'uint16' | 'uint32';
  type GPUFrontFace = 'ccw' | 'cw';
  type GPUCullMode = 'none' | 'front' | 'back';

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

  interface GPUStencilFaceState {
    compare?: GPUCompareFunction;
    failOp?: GPUStencilOperation;
    depthFailOp?: GPUStencilOperation;
    passOp?: GPUStencilOperation;
  }

  type GPUStencilOperation = 'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap';

  interface GPUMultisampleState {
    count?: number;
    mask?: number;
    alphaToCoverageEnabled?: boolean;
  }

  interface GPUFragmentState extends GPUProgrammableStage {
    targets: GPUColorTargetState[];
  }

  interface GPUColorTargetState {
    format: GPUTextureFormat;
    blend?: GPUBlendState;
    writeMask?: GPUColorWriteFlags;
  }

  interface GPUBlendState {
    color: GPUBlendComponent;
    alpha: GPUBlendComponent;
  }

  interface GPUBlendComponent {
    operation?: GPUBlendOperation;
    srcFactor?: GPUBlendFactor;
    dstFactor?: GPUBlendFactor;
  }

  type GPUBlendOperation = 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max';
  type GPUBlendFactor = 'zero' | 'one' | 'src' | 'one-minus-src' | 'src-alpha' | 'one-minus-src-alpha' | 'dst' | 'one-minus-dst' | 'dst-alpha' | 'one-minus-dst-alpha' | 'src-alpha-saturated' | 'constant' | 'one-minus-constant';
  type GPUColorWriteFlags = number;

  interface GPURenderPipeline {
    readonly layout: GPUPipelineLayout;
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPUCommandEncoderDescriptor {
    label?: string;
  }

  interface GPUCommandEncoder {
    beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
    beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
    copyBufferToTexture(source: GPUImageCopyBuffer, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
    copyTextureToBuffer(source: GPUImageCopyTexture, destination: GPUImageCopyBuffer, copySize: GPUExtent3D): void;
    copyTextureToTexture(source: GPUImageCopyTexture, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
    clearBuffer(buffer: GPUBuffer, offset?: number, size?: number): void;
    finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
  }

  interface GPURenderPassDescriptor {
    colorAttachments: GPURenderPassColorAttachment[];
    depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
    occlusionQuerySet?: GPUQuerySet;
    timestampWrites?: GPURenderPassTimestampWrites;
  }

  interface GPURenderPassColorAttachment {
    view: GPUTextureView;
    resolveTarget?: GPUTextureView;
    clearValue?: GPUColor;
    loadOp: GPULoadOp;
    storeOp: GPUStoreOp;
  }

  type GPUColor = [number, number, number, number] | { r: number; g: number; b: number; a: number };
  type GPULoadOp = 'load' | 'clear';
  type GPUStoreOp = 'store' | 'discard';

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

  interface GPUQuerySet {
    readonly type: GPUQueryType;
    readonly count: number;
    destroy(): void;
  }

  type GPUQueryType = 'occlusion' | 'timestamp';

  interface GPURenderPassTimestampWrites {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex?: number;
    endOfPassWriteIndex?: number;
  }

  interface GPURenderPassEncoder {
    setPipeline(pipeline: GPURenderPipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
    setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number): void;
    setIndexBuffer(buffer: GPUBuffer, indexFormat: GPUIndexFormat, offset?: number, size?: number): void;
    draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
    drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
    drawIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
    drawIndexedIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
    setViewport(x: number, y: number, width: number, height: number, minDepth: number, maxDepth: number): void;
    setScissorRect(x: number, y: number, width: number, height: number): void;
    setBlendConstant(color: GPUColor): void;
    setStencilReference(reference: number): void;
    beginOcclusionQuery(queryIndex: number): void;
    endOcclusionQuery(): void;
    executeBundles(bundles: GPURenderBundle[]): void;
    end(): void;
  }

  interface GPURenderBundle {
  }

  interface GPUComputePassDescriptor {
    timestampWrites?: GPUComputePassTimestampWrites;
  }

  interface GPUComputePassTimestampWrites {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex?: number;
    endOfPassWriteIndex?: number;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    dispatchWorkgroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
    end(): void;
  }

  interface GPUImageCopyBuffer {
    buffer: GPUBuffer;
    layout: GPUImageDataLayout;
  }

  interface GPUImageDataLayout {
    offset?: number;
    bytesPerRow?: number;
    rowsPerImage?: number;
  }

  interface GPUImageCopyTexture {
    texture: GPUTexture;
    mipLevel?: number;
    origin?: GPUOrigin3D;
    aspect?: GPUTextureAspect;
  }

  type GPUOrigin3D = [number, number, number] | { x?: number; y?: number; z?: number };

  interface GPUCommandBufferDescriptor {
    label?: string;
  }

  interface GPUCommandBuffer {
  }

  // Constants
  const GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
  };

  const GPUMapMode: {
    READ: number;
    WRITE: number;
  };

  const GPUTextureUsage: {
    COPY_SRC: number;
    COPY_DST: number;
    TEXTURE_BINDING: number;
    STORAGE_BINDING: number;
    RENDER_ATTACHMENT: number;
  };

  const GPUShaderStage: {
    VERTEX: number;
    FRAGMENT: number;
    COMPUTE: number;
  };

  const GPUColorWrite: {
    RED: number;
    GREEN: number;
    BLUE: number;
    ALPHA: number;
    ALL: number;
  };

  type GPUFeatureName = string;
}
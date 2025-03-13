/**
 * WebGPU TypeScript definitions
 * 
 * These are simplified type definitions for WebGPU to use in our SDK.
 * For complete definitions, see:
 * https://github.com/gpuweb/types/blob/main/dist/index.d.ts
 */

declare global {
  /**
   * The entry point to WebGPU functionality in the browser
   */
  interface Navigator {
    readonly gpu?: GPU;
  }

  /**
   * Main GPU interface provided by the browser
   */
  interface GPU {
    /**
     * Request an adapter based on the provided options
     */
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  /**
   * Options for requesting a GPU adapter
   */
  interface GPURequestAdapterOptions {
    /**
     * Power preference for the adapter
     */
    powerPreference?: 'high-performance' | 'low-power';
    
    /**
     * Force fallback adapter (software implementation)
     */
    forceFallbackAdapter?: boolean;
  }

  /**
   * GPU Adapter interface representing the physical device
   */
  interface GPUAdapter {
    /**
     * Adapter info containing vendor and architecture information
     */
    readonly name: string;
    
    /**
     * Request a device based on the provided options
     */
    requestDevice(options?: GPUDeviceDescriptor): Promise<GPUDevice>;
    
    /**
     * Check if the adapter supports certain features
     */
    readonly features: GPUSupportedFeatures;
    
    /**
     * Adapter limits such as maximum texture sizes, etc.
     */
    readonly limits: GPUSupportedLimits;
    
    /**
     * Adapter info
     */
    readonly isFallbackAdapter?: boolean;
    
    /**
     * Adapter info
     */
    readonly adapterInfo: GPUAdapterInfo;
  }

  /**
   * Adapter info containing vendor and architecture information
   */
  interface GPUAdapterInfo {
    /**
     * Adapter vendor (e.g., "Google")
     */
    vendor: string;
    
    /**
     * Adapter architecture (e.g., "Metal")
     */
    architecture: string;
    
    /**
     * Adapter device (e.g., "Apple M1")
     */
    device: string;
    
    /**
     * Adapter description (e.g., "Apple M1 Pro")
     */
    description: string;
  }

  /**
   * Set of supported features
   */
  interface GPUSupportedFeatures {
    /**
     * Check if feature is supported
     */
    has(feature: string): boolean;
    
    /**
     * List of supported features
     */
    [Symbol.iterator](): IterableIterator<string>;
  }

  /**
   * Supported device limits
   */
  interface GPUSupportedLimits {
    /**
     * Maximum allowed binding groups
     */
    readonly maxBindGroups: number;
    
    /**
     * Maximum allowed compute workgroup size (X dimension)
     */
    readonly maxComputeWorkgroupSizeX: number;
    
    /**
     * Maximum allowed compute workgroup size (Y dimension)
     */
    readonly maxComputeWorkgroupSizeY: number;
    
    /**
     * Maximum allowed compute workgroup size (Z dimension)
     */
    readonly maxComputeWorkgroupSizeZ: number;
    
    /**
     * Maximum allowed compute workgroups per dimension (X)
     */
    readonly maxComputeWorkgroupsPerDimension: number;
    
    /**
     * Maximum allowed storage buffer binding size
     */
    readonly maxStorageBufferBindingSize: number;
    
    /**
     * Maximum allowed uniform buffer binding size
     */
    readonly maxUniformBufferBindingSize: number;
  }

  /**
   * Device descriptor for requesting a device
   */
  interface GPUDeviceDescriptor {
    /**
     * Required features for the device
     */
    requiredFeatures?: string[];
    
    /**
     * Required limits for the device
     */
    requiredLimits?: Record<string, number>;
    
    /**
     * Default queue descriptor
     */
    defaultQueue?: GPUQueueDescriptor;
  }

  /**
   * Queue descriptor
   */
  interface GPUQueueDescriptor {
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * GPU Device interface representing a logical device
   */
  interface GPUDevice {
    /**
     * Features supported by the device
     */
    readonly features: GPUSupportedFeatures;
    
    /**
     * Limits of the device
     */
    readonly limits: GPUSupportedLimits;
    
    /**
     * Default queue for the device
     */
    readonly queue: GPUQueue;
    
    /**
     * Label for debugging
     */
    label: string;
    
    /**
     * Create a buffer
     */
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    
    /**
     * Create a compute pipeline
     */
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    
    /**
     * Create a bind group layout
     */
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    
    /**
     * Create a pipeline layout
     */
    createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
    
    /**
     * Create a bind group
     */
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    
    /**
     * Create a shader module
     */
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    
    /**
     * Create a command encoder
     */
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    
    /**
     * Create a texture
     */
    createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
    
    /**
     * Create a sampler
     */
    createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
    
    /**
     * Check if the device is lost
     */
    readonly lost: Promise<GPUDeviceLostInfo>;
    
    /**
     * Push error scope
     */
    pushErrorScope(filter: GPUErrorFilter): void;
    
    /**
     * Pop error scope
     */
    popErrorScope(): Promise<GPUError | null>;
    
    /**
     * Error callback
     */
    onuncapturederror: ((event: GPUUncapturedErrorEvent) => void) | null;
    
    /**
     * Destroy the device
     */
    destroy(): void;
  }

  /**
   * Buffer descriptor
   */
  interface GPUBufferDescriptor {
    /**
     * Size of the buffer in bytes
     */
    size: number;
    
    /**
     * Usage flags for the buffer
     */
    usage: number;
    
    /**
     * Whether buffer is mappable
     */
    mappedAtCreation?: boolean;
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Buffer interface
   */
  interface GPUBuffer {
    /**
     * Size of the buffer in bytes
     */
    readonly size: number;
    
    /**
     * Usage flags for the buffer
     */
    readonly usage: number;
    
    /**
     * Map the buffer for reading
     */
    mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
    
    /**
     * Get mapped range
     */
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    
    /**
     * Unmap the buffer
     */
    unmap(): void;
    
    /**
     * Destroy the buffer
     */
    destroy(): void;
  }

  /**
   * Buffer usage flags
   */
  const enum GPUBufferUsage {
    MAP_READ = 0x0001,
    MAP_WRITE = 0x0002,
    COPY_SRC = 0x0004,
    COPY_DST = 0x0008,
    INDEX = 0x0010,
    VERTEX = 0x0020,
    UNIFORM = 0x0040,
    STORAGE = 0x0080,
    INDIRECT = 0x0100,
    QUERY_RESOLVE = 0x0200,
  }

  /**
   * Shader module descriptor
   */
  interface GPUShaderModuleDescriptor {
    /**
     * WGSL code
     */
    code: string;
    
    /**
     * Source map URL
     */
    sourceMap?: object;
    
    /**
     * Hints for compilation
     */
    hints?: Record<string, any>;
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Shader module interface
   */
  interface GPUShaderModule {
    /**
     * Compilation info
     */
    readonly compilationInfo: Promise<GPUCompilationInfo>;
  }

  /**
   * Compilation info
   */
  interface GPUCompilationInfo {
    /**
     * Compilation messages
     */
    readonly messages: ReadonlyArray<GPUCompilationMessage>;
  }

  /**
   * Compilation message
   */
  interface GPUCompilationMessage {
    /**
     * Message text
     */
    readonly message: string;
    
    /**
     * Message type
     */
    readonly type: GPUCompilationMessageType;
    
    /**
     * Line number
     */
    readonly lineNum: number;
    
    /**
     * Column number
     */
    readonly linePos: number;
    
    /**
     * Offset
     */
    readonly offset: number;
    
    /**
     * Length
     */
    readonly length: number;
  }

  /**
   * Compilation message type
   */
  type GPUCompilationMessageType = 'error' | 'warning' | 'info';

  /**
   * Compute pipeline descriptor
   */
  interface GPUComputePipelineDescriptor {
    /**
     * Pipeline layout
     */
    layout: GPUPipelineLayout | 'auto';
    
    /**
     * Compute stage
     */
    compute: GPUProgrammableStage;
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Programmable stage
   */
  interface GPUProgrammableStage {
    /**
     * Shader module
     */
    module: GPUShaderModule;
    
    /**
     * Entry point
     */
    entryPoint: string;
    
    /**
     * Constants
     */
    constants?: Record<string, number>;
  }

  /**
   * Pipeline layout descriptor
   */
  interface GPUPipelineLayoutDescriptor {
    /**
     * Bind group layouts
     */
    bindGroupLayouts: GPUBindGroupLayout[];
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Pipeline layout interface
   */
  interface GPUPipelineLayout {
    /**
     * Label for debugging
     */
    label: string;
  }

  /**
   * Bind group layout descriptor
   */
  interface GPUBindGroupLayoutDescriptor {
    /**
     * Entries
     */
    entries: GPUBindGroupLayoutEntry[];
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Bind group layout entry
   */
  interface GPUBindGroupLayoutEntry {
    /**
     * Binding index
     */
    binding: number;
    
    /**
     * Visibility
     */
    visibility: number;
    
    /**
     * Buffer type
     */
    buffer?: GPUBufferBindingLayout;
    
    /**
     * Sampler type
     */
    sampler?: GPUSamplerBindingLayout;
    
    /**
     * Texture type
     */
    texture?: GPUTextureBindingLayout;
    
    /**
     * Storage texture type
     */
    storageTexture?: GPUStorageTextureBindingLayout;
    
    /**
     * External texture type
     */
    externalTexture?: GPUExternalTextureBindingLayout;
  }

  /**
   * Buffer binding layout
   */
  interface GPUBufferBindingLayout {
    /**
     * Buffer type
     */
    type?: GPUBufferBindingType;
    
    /**
     * Whether buffer has dynamic offset
     */
    hasDynamicOffset?: boolean;
    
    /**
     * Minimum buffer binding size
     */
    minBindingSize?: number;
  }

  /**
   * Buffer binding type
   */
  type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';

  /**
   * Sampler binding layout
   */
  interface GPUSamplerBindingLayout {
    /**
     * Sampler type
     */
    type?: GPUSamplerBindingType;
  }

  /**
   * Sampler binding type
   */
  type GPUSamplerBindingType = 'filtering' | 'non-filtering' | 'comparison';

  /**
   * Texture binding layout
   */
  interface GPUTextureBindingLayout {
    /**
     * Sample type
     */
    sampleType?: GPUTextureSampleType;
    
    /**
     * View dimension
     */
    viewDimension?: GPUTextureViewDimension;
    
    /**
     * Whether texture is multisampled
     */
    multisampled?: boolean;
  }

  /**
   * Texture sample type
   */
  type GPUTextureSampleType = 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';

  /**
   * Texture view dimension
   */
  type GPUTextureViewDimension = '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';

  /**
   * Storage texture binding layout
   */
  interface GPUStorageTextureBindingLayout {
    /**
     * Access mode
     */
    access?: GPUStorageTextureAccess;
    
    /**
     * Format
     */
    format: GPUTextureFormat;
    
    /**
     * View dimension
     */
    viewDimension?: GPUTextureViewDimension;
  }

  /**
   * Storage texture access
   */
  type GPUStorageTextureAccess = 'write-only' | 'read-only' | 'read-write';

  /**
   * External texture binding layout
   */
  interface GPUExternalTextureBindingLayout {
  }

  /**
   * Bind group layout interface
   */
  interface GPUBindGroupLayout {
    /**
     * Label for debugging
     */
    label: string;
  }

  /**
   * Bind group descriptor
   */
  interface GPUBindGroupDescriptor {
    /**
     * Layout
     */
    layout: GPUBindGroupLayout;
    
    /**
     * Entries
     */
    entries: GPUBindGroupEntry[];
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Bind group entry
   */
  interface GPUBindGroupEntry {
    /**
     * Binding index
     */
    binding: number;
    
    /**
     * Resource
     */
    resource: GPUBindingResource;
  }

  /**
   * Binding resource
   */
  type GPUBindingResource = 
    | GPUSampler 
    | GPUTextureView 
    | GPUBufferBinding 
    | GPUExternalTexture;

  /**
   * Buffer binding
   */
  interface GPUBufferBinding {
    /**
     * Buffer
     */
    buffer: GPUBuffer;
    
    /**
     * Offset
     */
    offset?: number;
    
    /**
     * Size
     */
    size?: number;
  }

  /**
   * Bind group interface
   */
  interface GPUBindGroup {
    /**
     * Label for debugging
     */
    label: string;
  }

  /**
   * Compute pipeline interface
   */
  interface GPUComputePipeline {
    /**
     * Label for debugging
     */
    label: string;
    
    /**
     * Get bind group layout
     */
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  /**
   * Command encoder descriptor
   */
  interface GPUCommandEncoderDescriptor {
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Command encoder interface
   */
  interface GPUCommandEncoder {
    /**
     * Begin compute pass
     */
    beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
    
    /**
     * Copy buffer to buffer
     */
    copyBufferToBuffer(
      source: GPUBuffer,
      sourceOffset: number,
      destination: GPUBuffer,
      destinationOffset: number,
      size: number
    ): void;
    
    /**
     * Copy buffer to texture
     */
    copyBufferToTexture(
      source: GPUImageCopyBuffer,
      destination: GPUImageCopyTexture,
      copySize: GPUExtent3D
    ): void;
    
    /**
     * Copy texture to buffer
     */
    copyTextureToBuffer(
      source: GPUImageCopyTexture,
      destination: GPUImageCopyBuffer,
      copySize: GPUExtent3D
    ): void;
    
    /**
     * Copy texture to texture
     */
    copyTextureToTexture(
      source: GPUImageCopyTexture,
      destination: GPUImageCopyTexture,
      copySize: GPUExtent3D
    ): void;
    
    /**
     * Finish encoding
     */
    finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
    
    /**
     * Push debug group
     */
    pushDebugGroup(groupLabel: string): void;
    
    /**
     * Pop debug group
     */
    popDebugGroup(): void;
    
    /**
     * Insert debug marker
     */
    insertDebugMarker(markerLabel: string): void;
  }

  /**
   * Compute pass descriptor
   */
  interface GPUComputePassDescriptor {
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Compute pass encoder interface
   */
  interface GPUComputePassEncoder {
    /**
     * Set pipeline
     */
    setPipeline(pipeline: GPUComputePipeline): void;
    
    /**
     * Set bind group
     */
    setBindGroup(
      index: number,
      bindGroup: GPUBindGroup,
      dynamicOffsets?: number[] | Uint32Array
    ): void;
    
    /**
     * Dispatch workgroups
     */
    dispatchWorkgroups(
      workgroupCountX: number,
      workgroupCountY?: number,
      workgroupCountZ?: number
    ): void;
    
    /**
     * End pass
     */
    end(): void;
    
    /**
     * Push debug group
     */
    pushDebugGroup(groupLabel: string): void;
    
    /**
     * Pop debug group
     */
    popDebugGroup(): void;
    
    /**
     * Insert debug marker
     */
    insertDebugMarker(markerLabel: string): void;
  }

  /**
   * Command buffer descriptor
   */
  interface GPUCommandBufferDescriptor {
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Command buffer interface
   */
  interface GPUCommandBuffer {
    /**
     * Label for debugging
     */
    readonly label: string;
  }

  /**
   * Queue interface
   */
  interface GPUQueue {
    /**
     * Label for debugging
     */
    label: string;
    
    /**
     * Submit command buffers
     */
    submit(commandBuffers: GPUCommandBuffer[]): void;
    
    /**
     * Write buffer
     */
    writeBuffer(
      buffer: GPUBuffer,
      bufferOffset: number,
      data: BufferSource,
      dataOffset?: number,
      size?: number
    ): void;
    
    /**
     * Write texture
     */
    writeTexture(
      destination: GPUImageCopyTexture,
      data: BufferSource,
      dataLayout: GPUImageDataLayout,
      size: GPUExtent3D
    ): void;
    
    /**
     * On submitted work done
     */
    onSubmittedWorkDone(): Promise<void>;
  }

  /**
   * Texture descriptor
   */
  interface GPUTextureDescriptor {
    /**
     * Size
     */
    size: GPUExtent3D;
    
    /**
     * Mip level count
     */
    mipLevelCount?: number;
    
    /**
     * Sample count
     */
    sampleCount?: number;
    
    /**
     * Dimension
     */
    dimension?: GPUTextureDimension;
    
    /**
     * Format
     */
    format: GPUTextureFormat;
    
    /**
     * Usage
     */
    usage: number;
    
    /**
     * View formats
     */
    viewFormats?: GPUTextureFormat[];
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Texture format
   */
  type GPUTextureFormat = string;

  /**
   * Texture dimension
   */
  type GPUTextureDimension = '1d' | '2d' | '3d';

  /**
   * Texture usage flags
   */
  const enum GPUTextureUsage {
    COPY_SRC = 0x01,
    COPY_DST = 0x02,
    TEXTURE_BINDING = 0x04,
    STORAGE_BINDING = 0x08,
    RENDER_ATTACHMENT = 0x10,
  }

  /**
   * Texture interface
   */
  interface GPUTexture {
    /**
     * Create view
     */
    createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
    
    /**
     * Destroy
     */
    destroy(): void;
    
    /**
     * Width
     */
    readonly width: number;
    
    /**
     * Height
     */
    readonly height: number;
    
    /**
     * Depth or array layers
     */
    readonly depthOrArrayLayers: number;
    
    /**
     * Mip level count
     */
    readonly mipLevelCount: number;
    
    /**
     * Sample count
     */
    readonly sampleCount: number;
    
    /**
     * Dimension
     */
    readonly dimension: GPUTextureDimension;
    
    /**
     * Format
     */
    readonly format: GPUTextureFormat;
    
    /**
     * Usage
     */
    readonly usage: number;
  }

  /**
   * Texture view descriptor
   */
  interface GPUTextureViewDescriptor {
    /**
     * Format
     */
    format?: GPUTextureFormat;
    
    /**
     * Dimension
     */
    dimension?: GPUTextureViewDimension;
    
    /**
     * Aspect
     */
    aspect?: GPUTextureAspect;
    
    /**
     * Base mip level
     */
    baseMipLevel?: number;
    
    /**
     * Mip level count
     */
    mipLevelCount?: number;
    
    /**
     * Base array layer
     */
    baseArrayLayer?: number;
    
    /**
     * Array layer count
     */
    arrayLayerCount?: number;
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Texture aspect
   */
  type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only';

  /**
   * Texture view interface
   */
  interface GPUTextureView {
    /**
     * Label for debugging
     */
    readonly label: string;
  }

  /**
   * Sampler descriptor
   */
  interface GPUSamplerDescriptor {
    /**
     * Address mode u
     */
    addressModeU?: GPUAddressMode;
    
    /**
     * Address mode v
     */
    addressModeV?: GPUAddressMode;
    
    /**
     * Address mode w
     */
    addressModeW?: GPUAddressMode;
    
    /**
     * Mag filter
     */
    magFilter?: GPUFilterMode;
    
    /**
     * Min filter
     */
    minFilter?: GPUFilterMode;
    
    /**
     * Mipmap filter
     */
    mipmapFilter?: GPUFilterMode;
    
    /**
     * Lod min clamp
     */
    lodMinClamp?: number;
    
    /**
     * Lod max clamp
     */
    lodMaxClamp?: number;
    
    /**
     * Compare function
     */
    compare?: GPUCompareFunction;
    
    /**
     * Max anisotropy
     */
    maxAnisotropy?: number;
    
    /**
     * Label for debugging
     */
    label?: string;
  }

  /**
   * Address mode
   */
  type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat';

  /**
   * Filter mode
   */
  type GPUFilterMode = 'nearest' | 'linear';

  /**
   * Compare function
   */
  type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';

  /**
   * Sampler interface
   */
  interface GPUSampler {
    /**
     * Label for debugging
     */
    readonly label: string;
  }

  /**
   * Image copy buffer
   */
  interface GPUImageCopyBuffer {
    /**
     * Buffer
     */
    buffer: GPUBuffer;
    
    /**
     * Offset
     */
    offset?: number;
    
    /**
     * Bytes per row
     */
    bytesPerRow?: number;
    
    /**
     * Rows per image
     */
    rowsPerImage?: number;
  }

  /**
   * Image copy texture
   */
  interface GPUImageCopyTexture {
    /**
     * Texture
     */
    texture: GPUTexture;
    
    /**
     * Mip level
     */
    mipLevel?: number;
    
    /**
     * Origin
     */
    origin?: GPUOrigin3D;
    
    /**
     * Aspect
     */
    aspect?: GPUTextureAspect;
  }

  /**
   * Origin 3D
   */
  type GPUOrigin3D = GPUOrigin3DDict | number[];

  /**
   * Origin 3D dict
   */
  interface GPUOrigin3DDict {
    /**
     * X
     */
    x?: number;
    
    /**
     * Y
     */
    y?: number;
    
    /**
     * Z
     */
    z?: number;
  }

  /**
   * Extent 3D
   */
  type GPUExtent3D = GPUExtent3DDict | number[];

  /**
   * Extent 3D dict
   */
  interface GPUExtent3DDict {
    /**
     * Width
     */
    width: number;
    
    /**
     * Height
     */
    height?: number;
    
    /**
     * Depth or array layers
     */
    depthOrArrayLayers?: number;
  }

  /**
   * Image data layout
   */
  interface GPUImageDataLayout {
    /**
     * Offset
     */
    offset?: number;
    
    /**
     * Bytes per row
     */
    bytesPerRow?: number;
    
    /**
     * Rows per image
     */
    rowsPerImage?: number;
  }

  /**
   * Device lost info
   */
  interface GPUDeviceLostInfo {
    /**
     * Reason
     */
    readonly reason: GPUDeviceLostReason;
    
    /**
     * Message
     */
    readonly message: string;
  }

  /**
   * Device lost reason
   */
  type GPUDeviceLostReason = 'unknown' | 'destroyed';

  /**
   * Error filter
   */
  type GPUErrorFilter = 'out-of-memory' | 'validation';

  /**
   * Error
   */
  interface GPUError {
    /**
     * Message
     */
    readonly message: string;
  }

  /**
   * Uncaptured error event
   */
  interface GPUUncapturedErrorEvent extends Event {
    /**
     * Error
     */
    readonly error: GPUError;
  }

  /**
   * External texture
   */
  interface GPUExternalTexture {
    /**
     * Label for debugging
     */
    readonly label: string;
  }
}

// This is needed to make this a module
export {};
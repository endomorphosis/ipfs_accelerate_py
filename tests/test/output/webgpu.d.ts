
interface GPUDevice {
  createBuffer(descriptor: any): GPUBuffer;
  createShaderModule(descriptor: any): GPUShaderModule;
  createComputePipeline(descriptor: any): GPUComputePipeline;
  createBindGroup(descriptor: any): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  queue: GPUQueue;
}

interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
  features: Set<string>;
  limits: any;
  get_preferred_format(): string;
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  write_buffer(buffer: GPUBuffer, offset: number, data: any): void;
  on_submitted_work_done(): Promise<void>;
}

interface GPUBuffer {
  map_async(mode: number): Promise<void>;
  get_mapped_range(): ArrayBuffer;
  unmap(): void;
}

interface GPUShaderModule {}

interface GPUComputePipeline {}

interface GPUBindGroup {}

interface GPUCommandEncoder {
  begin_compute_pass(): GPUComputePassEncoder;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  set_pipeline(pipeline: GPUComputePipeline): void;
  set_bind_group(index: number, bindGroup: GPUBindGroup): void;
  dispatch_workgroups(...args: number[]): void;
  end(): void;
}

interface GPUCommandBuffer {}

interface NavigatorGPU {
  request_adapter(): Promise<GPUAdapter>;
  requestAdapter(): Promise<GPUAdapter>;
}

interface Navigator {
  gpu: NavigatorGPU;
}

declare var navigator: Navigator;

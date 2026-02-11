#!/usr/bin/env python3
# sample_webgpu_backend.py
# Sample WebGPU backend implementation for testing the Python to TypeScript converter

from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import re

class WebGPUBackend:
    """
    WebGPU backend implementation for hardware acceleration in web browsers.
    Provides an interface to the WebGPU API for compute operations.
    """
    
    def __init__(self, options: Dict[str, Any] = None):
        """
        Initialize WebGPU backend with optional configuration.
        
        Args:
            options: Configuration options for the WebGPU backend
        """
        self.device: Optional[Any] = None
        self.adapter: Optional[Any] = None
        self.initialized: bool = False
        self.features: List[str] = []
        self.limits: Dict[str, int] = {}
        self.pipeline_cache: Dict[str, Any] = {}
        self.buffer_cache: Dict[str, Any] = {}
        self.options = options or {}
        self.logger = logging.getLogger("WebGPUBackend")
    
    async def initialize(self) -> bool:
        """
        Initialize the WebGPU backend by requesting an adapter and device.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Request adapter from navigator.gpu
            self.adapter = await navigator.gpu.request_adapter()
            
            if not self.adapter:
                self.logger.error("WebGPU not supported or disabled")
                return False
            
            # Request device from adapter
            self.device = await self.adapter.request_device()
            
            if not self.device:
                self.logger.error("Failed to get WebGPU device")
                return False
            
            # Extract supported features
            self.features = list(self.adapter.features)
            
            # Extract limits
            self.limits = {
                "maxBindGroups": self.adapter.limits.maxBindGroups,
                "maxComputeWorkgroupSizeX": self.adapter.limits.maxComputeWorkgroupSizeX,
                "maxComputeWorkgroupSizeY": self.adapter.limits.maxComputeWorkgroupSizeY,
                "maxComputeWorkgroupSizeZ": self.adapter.limits.maxComputeWorkgroupSizeZ,
                "maxBufferSize": self.adapter.limits.maxBufferSize
            }
            
            self.initialized = True
            self.logger.info(f"WebGPU initialized with {len(self.features)} features")
            return True
        except Exception as e:
            self.logger.error(f"WebGPU initialization error: {e}")
            return False
    
    def create_buffer(self, size: int, usage: int, label: Optional[str] = None) -> Optional[Any]:
        """
        Create a GPU buffer with the specified size and usage.
        
        Args:
            size: Size of the buffer in bytes
            usage: Buffer usage flags (e.g., STORAGE, UNIFORM, COPY_SRC, COPY_DST)
            label: Optional debug label for the buffer
            
        Returns:
            GPUBuffer object or None if creation failed
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return None
        
        try:
            buffer = self.device.create_buffer({
                "size": size,
                "usage": usage,
                "mappedAtCreation": False,
                "label": label
            })
            
            # Cache buffer by label if provided
            if label:
                self.buffer_cache[label] = buffer
                
            return buffer
        except Exception as e:
            self.logger.error(f"Error creating WebGPU buffer: {e}")
            return None
    
    def write_buffer(self, buffer: Any, data: Union[List[float], List[int], bytes], offset: int = 0) -> bool:
        """
        Write data to a GPU buffer.
        
        Args:
            buffer: The GPU buffer to write to
            data: Data to write to the buffer
            offset: Offset in bytes to start writing at
            
        Returns:
            True if write was successful, False otherwise
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return False
        
        try:
            self.device.queue.write_buffer(buffer, offset, data)
            return True
        except Exception as e:
            self.logger.error(f"Error writing to WebGPU buffer: {e}")
            return False
    
    async def read_buffer(self, buffer: Any, size: int) -> Optional[bytes]:
        """
        Read data from a GPU buffer.
        
        Args:
            buffer: The GPU buffer to read from
            size: Number of bytes to read
            
        Returns:
            Buffer data as bytes, or None if read failed
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return None
        
        try:
            # Create a mapping for reading
            await buffer.map_async(1)  # GPUMapMode.READ = 1
            
            # Get a mapped range of the buffer
            mapped_range = buffer.get_mapped_range()
            
            # Copy the data
            result = bytes(mapped_range)
            
            # Unmap the buffer
            buffer.unmap()
            
            return result
        except Exception as e:
            self.logger.error(f"Error reading from WebGPU buffer: {e}")
            return None
    
    async def create_compute_pipeline(self, shader: str, entry_point: str = "main") -> Optional[Any]:
        """
        Create a compute pipeline using the provided shader code.
        
        Args:
            shader: WGSL shader code
            entry_point: Entry point function name in the shader
            
        Returns:
            GPUComputePipeline object or None if creation failed
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return None
        
        try:
            # Create shader module
            shader_module = self.device.create_shader_module({
                "code": shader
            })
            
            # Create pipeline
            pipeline = await self.device.create_compute_pipeline({
                "layout": "auto",
                "compute": {
                    "module": shader_module,
                    "entryPoint": entry_point
                }
            })
            
            # Cache pipeline using a hash of the shader code
            shader_hash = str(hash(shader))
            self.pipeline_cache[shader_hash] = pipeline
            
            return pipeline
        except Exception as e:
            self.logger.error(f"Error creating compute pipeline: {e}")
            return None
    
    def create_bind_group(self, layout: Any, entries: List[Dict[str, Any]]) -> Optional[Any]:
        """
        Create a bind group for a compute pipeline.
        
        Args:
            layout: GPUBindGroupLayout object
            entries: List of binding entries
            
        Returns:
            GPUBindGroup object or None if creation failed
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return None
        
        try:
            bind_group = self.device.create_bind_group({
                "layout": layout,
                "entries": entries
            })
            
            return bind_group
        except Exception as e:
            self.logger.error(f"Error creating bind group: {e}")
            return None
    
    async def run_compute(self, pipeline: Any, bind_groups: List[Any], 
                         workgroups: Tuple[int, int, int] = (1, 1, 1)) -> bool:
        """
        Run a compute operation using the provided pipeline and bind groups.
        
        Args:
            pipeline: GPUComputePipeline to use
            bind_groups: List of GPUBindGroup objects to bind
            workgroups: Tuple of (x, y, z) workgroup dimensions
            
        Returns:
            True if compute operation was successful, False otherwise
        """
        if not self.device:
            self.logger.error("WebGPU device not initialized")
            return False
        
        try:
            # Create command encoder
            encoder = self.device.create_command_encoder()
            
            # Begin compute pass
            pass_encoder = encoder.begin_compute_pass()
            
            # Set pipeline
            pass_encoder.set_pipeline(pipeline)
            
            # Set bind groups
            for index, bind_group in enumerate(bind_groups):
                pass_encoder.set_bind_group(index, bind_group)
            
            # Dispatch workgroups
            pass_encoder.dispatch_workgroups(*workgroups)
            
            # End pass
            pass_encoder.end()
            
            # Finish encoding and submit commands
            command_buffer = encoder.finish()
            self.device.queue.submit([command_buffer])
            
            # Wait for GPU to complete
            await self.device.queue.on_submitted_work_done()
            
            return True
        except Exception as e:
            self.logger.error(f"Error running compute operation: {e}")
            return False
    
    def destroy(self) -> None:
        """
        Clean up WebGPU resources.
        """
        # Clear caches
        self.pipeline_cache = {}
        self.buffer_cache = {}
        
        # Set device and adapter to None to release references
        self.device = None
        self.adapter = None
        self.initialized = False
        
        self.logger.info("WebGPU resources destroyed")
    
    @property
    def is_initialized(self) -> bool:
        """
        Check if the WebGPU backend is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self.initialized and self.device is not None
    
    def get_preferred_format(self) -> str:
        """
        Get the preferred swap chain format.
        
        Returns:
            Preferred format as string
        """
        if not self.adapter:
            return "bgra8unorm"
        
        return self.adapter.get_preferred_format()
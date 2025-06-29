
from .base_test import BaseTest

class HardwareTest(BaseTest):
    """Base class for hardware tests."""
    
    def setUp(self):
        super().setUp()
        self.detect_hardware()
        
    def detect_hardware(self):
        """Detect available hardware."""
        self.has_webgpu = self._check_webgpu()
        self.has_webnn = self._check_webnn()
        
    def _check_webgpu(self):
        """Check if WebGPU is available."""
        # Placeholder for hardware detection
        return False
        
    def _check_webnn(self):
        """Check if WebNN is available."""
        # Placeholder for hardware detection
        return False
    
    def skip_if_no_webgpu(self):
        """Skip test if WebGPU is not available."""
        if not self.has_webgpu:
            self.skipTest("WebGPU not available")
    
    def skip_if_no_webnn(self):
        """Skip test if WebNN is not available."""
        if not self.has_webnn:
            self.skipTest("WebNN not available")

"""
Hardware Detection and Selection

Detect available hardware and select optimal hardware for models
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class HardwareCapability:
    """Hardware capability information"""
    name: str
    available: bool
    device_count: int = 0
    memory_total_mb: float = 0
    memory_available_mb: float = 0
    compute_capability: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HardwareDetector:
    """
    Detect available hardware capabilities
    
    Supports:
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs)
    - MPS (Apple Silicon)
    - OpenVINO (Intel)
    - QNN (Qualcomm)
    - CPU (always available)
    """
    
    def __init__(self):
        self.capabilities: Dict[str, HardwareCapability] = {}
        self._detect_all()
    
    def _detect_all(self):
        """Detect all hardware"""
        self.capabilities = {
            "cuda": self._detect_cuda(),
            "rocm": self._detect_rocm(),
            "mps": self._detect_mps(),
            "openvino": self._detect_openvino(),
            "qnn": self._detect_qnn(),
            "cpu": self._detect_cpu(),
        }
        
        available = [name for name, cap in self.capabilities.items() if cap.available]
        logger.info(f"Available hardware: {', '.join(available)}")
    
    def _detect_cuda(self) -> HardwareCapability:
        """Detect CUDA/NVIDIA GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device = torch.cuda.current_device()
                memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                memory_available = memory_total - memory_allocated
                compute_capability = f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}"
                
                return HardwareCapability(
                    name="cuda",
                    available=True,
                    device_count=device_count,
                    memory_total_mb=memory_total,
                    memory_available_mb=memory_available,
                    compute_capability=compute_capability,
                    metadata={"torch_version": torch.__version__}
                )
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")
        
        return HardwareCapability(name="cuda", available=False)
    
    def _detect_rocm(self) -> HardwareCapability:
        """Detect ROCm/AMD GPU"""
        try:
            import torch
            if hasattr(torch, 'hip') and torch.hip.is_available():
                device_count = torch.hip.device_count()
                return HardwareCapability(
                    name="rocm",
                    available=True,
                    device_count=device_count,
                    metadata={"torch_version": torch.__version__}
                )
        except Exception as e:
            logger.debug(f"ROCm not available: {e}")
        
        return HardwareCapability(name="rocm", available=False)
    
    def _detect_mps(self) -> HardwareCapability:
        """Detect Apple Metal Performance Shaders"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return HardwareCapability(
                    name="mps",
                    available=True,
                    device_count=1,
                    metadata={"torch_version": torch.__version__}
                )
        except Exception as e:
            logger.debug(f"MPS not available: {e}")
        
        return HardwareCapability(name="mps", available=False)
    
    def _detect_openvino(self) -> HardwareCapability:
        """Detect OpenVINO"""
        try:
            import openvino
            return HardwareCapability(
                name="openvino",
                available=True,
                device_count=1,
                metadata={"version": openvino.__version__}
            )
        except Exception as e:
            logger.debug(f"OpenVINO not available: {e}")
        
        return HardwareCapability(name="openvino", available=False)
    
    def _detect_qnn(self) -> HardwareCapability:
        """Detect Qualcomm Neural Network SDK"""
        try:
            # QNN detection would go here
            # For now, assume not available
            pass
        except Exception as e:
            logger.debug(f"QNN not available: {e}")
        
        return HardwareCapability(name="qnn", available=False)
    
    def _detect_cpu(self) -> HardwareCapability:
        """Detect CPU (always available)"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return HardwareCapability(
                name="cpu",
                available=True,
                device_count=1,
                memory_total_mb=memory.total / (1024**2),
                memory_available_mb=memory.available / (1024**2),
            )
        except Exception as e:
            # CPU should always be available even if psutil fails
            return HardwareCapability(name="cpu", available=True, device_count=1)
    
    def get_capability(self, hardware: str) -> Optional[HardwareCapability]:
        """Get capability for specific hardware"""
        return self.capabilities.get(hardware)
    
    def is_available(self, hardware: str) -> bool:
        """Check if hardware is available"""
        cap = self.capabilities.get(hardware)
        return cap.available if cap else False
    
    def get_available_hardware(self) -> List[str]:
        """Get list of available hardware"""
        return [name for name, cap in self.capabilities.items() if cap.available]
    
    def get_best_hardware(
        self,
        supported_hardware: List[str],
        preferred_order: List[str] = None
    ) -> Optional[str]:
        """
        Select best available hardware
        
        Args:
            supported_hardware: Hardware supported by the model
            preferred_order: Preferred hardware order
            
        Returns:
            Best hardware name or None
        """
        if preferred_order is None:
            preferred_order = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]
        
        # Find first preferred hardware that is both available and supported
        for hardware in preferred_order:
            if hardware in supported_hardware and self.is_available(hardware):
                return hardware
        
        return None


class HardwareSelector:
    """
    Intelligent hardware selection for models
    
    Considers:
    - Hardware availability
    - Model support
    - Current load
    - Memory constraints
    """
    
    def __init__(self, detector: HardwareDetector):
        self.detector = detector
        self.load_tracker: Dict[str, int] = {}  # Track active models per hardware
    
    def select_hardware(
        self,
        model_info: Dict,
        preferred_order: List[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Select optimal hardware for a model
        
        Args:
            model_info: Model information including supported_hardware
            preferred_order: Preferred hardware order
            
        Returns:
            Tuple of (hardware_name, reason)
        """
        supported = model_info.get("supported_hardware", ["cpu"])
        
        # Get best available hardware
        hardware = self.detector.get_best_hardware(supported, preferred_order)
        
        if hardware:
            reason = f"Selected {hardware} (available and supported)"
            return hardware, reason
        else:
            reason = "No suitable hardware available, falling back to CPU"
            return "cpu", reason
    
    def track_load(self, hardware: str, delta: int = 1):
        """Track hardware load (increment/decrement)"""
        current = self.load_tracker.get(hardware, 0)
        self.load_tracker[hardware] = max(0, current + delta)
    
    def get_load(self, hardware: str) -> int:
        """Get current load for hardware"""
        return self.load_tracker.get(hardware, 0)

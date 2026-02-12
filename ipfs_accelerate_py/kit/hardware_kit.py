"""
Hardware Kit - Core Hardware Detection and Management

This module provides core hardware detection and management operations.
It can be used by both the unified CLI and MCP server.
"""

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_HARDWARE_KIT: Optional["HardwareKit"] = None


def _get_default_hardware_kit() -> "HardwareKit":
    global _DEFAULT_HARDWARE_KIT
    if _DEFAULT_HARDWARE_KIT is None:
        _DEFAULT_HARDWARE_KIT = HardwareKit()
    return _DEFAULT_HARDWARE_KIT


@dataclass
class HardwareInfo:
    """Hardware information."""
    cpu: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    accelerators: Dict[str, Any] = field(default_factory=dict)
    platform_info: Dict[str, Any] = field(default_factory=dict)


class HardwareKit:
    """
    Core hardware operations module.
    
    Provides hardware detection and management functionality that can be
    used by CLI, MCP tools, or directly in Python code.
    """
    
    def __init__(self):
        """Initialize Hardware Kit."""
        pass
    
    def get_platform_info(self) -> Dict[str, Any]:
        """
        Get platform information.
        
        Returns:
            Dictionary with platform info
        """
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
        except Exception as e:
            logger.warning(f"Failed to get platform info: {e}")
            return {}
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            Dictionary with CPU info
        """
        import os
        
        cpu_info = {
            "count": os.cpu_count() or 0,
            "architecture": platform.machine(),
        }
        
        # Try to get more detailed CPU info
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    # Extract model name
                    for line in cpuinfo.split('\n'):
                        if "model name" in line:
                            cpu_info["model"] = line.split(':')[1].strip()
                            break
        except Exception as e:
            logger.debug(f"Could not get detailed CPU info: {e}")
        
        return cpu_info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information.
        
        Returns:
            Dictionary with memory info
        """
        memory_info = {}
        
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if "MemTotal" in line:
                            # Extract total memory in KB and convert to GB
                            total_kb = int(line.split()[1])
                            memory_info["total_gb"] = round(total_kb / (1024 * 1024), 2)
                        elif "MemAvailable" in line:
                            available_kb = int(line.split()[1])
                            memory_info["available_gb"] = round(available_kb / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")
        
        return memory_info
    
    def detect_cuda(self) -> Dict[str, Any]:
        """
        Detect CUDA GPUs.
        
        Returns:
            Dictionary with CUDA info
        """
        cuda_info = {
            "available": False,
            "devices": [],
        }
        
        try:
            # Try using nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                cuda_info["available"] = True
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            cuda_info["devices"].append({
                                "name": parts[0],
                                "memory": parts[1] if len(parts) > 1 else "Unknown",
                                "driver": parts[2] if len(parts) > 2 else "Unknown"
                            })
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"CUDA not detected: {e}")
        
        # Try PyTorch CUDA detection
        try:
            import torch
            if torch.cuda.is_available():
                cuda_info["available"] = True
                cuda_info["pytorch_version"] = torch.__version__
                cuda_info["cuda_version"] = torch.version.cuda
                if not cuda_info["devices"]:
                    for i in range(torch.cuda.device_count()):
                        cuda_info["devices"].append({
                            "name": torch.cuda.get_device_name(i),
                            "index": i
                        })
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch CUDA detection failed: {e}")
        
        return cuda_info
    
    def detect_rocm(self) -> Dict[str, Any]:
        """
        Detect ROCm GPUs (AMD).
        
        Returns:
            Dictionary with ROCm info
        """
        rocm_info = {
            "available": False,
            "devices": [],
        }
        
        try:
            # Try using rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                rocm_info["available"] = True
                # Parse output for device names
                for line in result.stdout.strip().split('\n'):
                    if "GPU" in line:
                        rocm_info["devices"].append({"info": line.strip()})
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"ROCm not detected: {e}")
        
        return rocm_info
    
    def detect_metal(self) -> Dict[str, Any]:
        """
        Detect Metal support (Apple Silicon).
        
        Returns:
            Dictionary with Metal info
        """
        metal_info = {
            "available": False,
        }
        
        # Check if on macOS
        if platform.system() == "Darwin":
            metal_info["available"] = True
            metal_info["platform"] = "macOS"
            metal_info["machine"] = platform.machine()
            
            # Check for Apple Silicon
            if platform.machine() == "arm64":
                metal_info["apple_silicon"] = True
        
        return metal_info
    
    def detect_webgpu(self) -> Dict[str, Any]:
        """
        Detect WebGPU support.
        
        Returns:
            Dictionary with WebGPU info
        """
        webgpu_info = {
            "available": False,
            "note": "WebGPU requires browser environment"
        }
        
        # WebGPU is primarily browser-based
        # Check if we're in a web environment
        try:
            # This would need to be implemented with actual WebGPU detection
            # For now, just return basic info
            pass
        except Exception as e:
            logger.debug(f"WebGPU detection: {e}")
        
        return webgpu_info
    
    def detect_webnn(self) -> Dict[str, Any]:
        """
        Detect WebNN support.
        
        Returns:
            Dictionary with WebNN info
        """
        webnn_info = {
            "available": False,
            "note": "WebNN requires browser environment"
        }
        
        # WebNN is primarily browser-based
        # Check if we're in a web environment
        try:
            # This would need to be implemented with actual WebNN detection
            # For now, just return basic info
            pass
        except Exception as e:
            logger.debug(f"WebNN detection: {e}")
        
        return webnn_info
    
    def get_hardware_info(self, include_detailed: bool = False) -> HardwareInfo:
        """
        Get comprehensive hardware information.
        
        Args:
            include_detailed: Include detailed hardware information
            
        Returns:
            HardwareInfo object
        """
        hardware_info = HardwareInfo()
        
        # Get platform info
        hardware_info.platform_info = self.get_platform_info()
        
        # Get CPU info
        hardware_info.cpu = self.get_cpu_info()
        
        # Get memory info
        hardware_info.memory = self.get_memory_info()
        
        # Detect accelerators
        accelerators = {}
        
        # CUDA
        cuda_info = self.detect_cuda()
        if cuda_info["available"]:
            accelerators["cuda"] = cuda_info
        
        # ROCm
        rocm_info = self.detect_rocm()
        if rocm_info["available"]:
            accelerators["rocm"] = rocm_info
        
        # Metal
        metal_info = self.detect_metal()
        if metal_info["available"]:
            accelerators["metal"] = metal_info
        
        # WebGPU (if applicable)
        if include_detailed:
            accelerators["webgpu"] = self.detect_webgpu()
            accelerators["webnn"] = self.detect_webnn()
        
        hardware_info.accelerators = accelerators
        
        return hardware_info
    
    def test_hardware(
        self,
        accelerator: str = "all",
        test_level: str = "basic"
    ) -> Dict[str, Any]:
        """
        Test hardware accelerators.
        
        Args:
            accelerator: Accelerator to test (cuda, cpu, webgpu, webnn, all)
            test_level: Level of testing (basic, comprehensive)
            
        Returns:
            Dictionary with test results
        """
        results = {
            "tested": accelerator,
            "level": test_level,
            "tests": {}
        }
        
        if accelerator in ["cuda", "all"]:
            results["tests"]["cuda"] = self._test_cuda(test_level)
        
        if accelerator in ["cpu", "all"]:
            results["tests"]["cpu"] = self._test_cpu(test_level)
        
        return results
    
    def _test_cuda(self, test_level: str) -> Dict[str, Any]:
        """Test CUDA functionality."""
        result = {"available": False, "tests_passed": False}
        
        try:
            import torch
            if torch.cuda.is_available():
                result["available"] = True
                
                if test_level == "basic":
                    # Basic test: create a tensor
                    x = torch.ones(10, device='cuda')
                    result["tests_passed"] = True
                    result["device_count"] = torch.cuda.device_count()
                elif test_level == "comprehensive":
                    # More comprehensive test
                    x = torch.randn(1000, 1000, device='cuda')
                    y = torch.matmul(x, x)
                    result["tests_passed"] = True
                    result["device_count"] = torch.cuda.device_count()
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _test_cpu(self, test_level: str) -> Dict[str, Any]:
        """Test CPU functionality."""
        result = {"available": True, "tests_passed": False}
        
        try:
            if test_level == "basic":
                # Basic test
                import numpy as np
                x = np.ones(10)
                result["tests_passed"] = True
            elif test_level == "comprehensive":
                # More comprehensive test
                import numpy as np
                x = np.random.randn(1000, 1000)
                y = np.dot(x, x)
                result["tests_passed"] = True
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def recommend_hardware(
        self,
        model_name: str,
        task: str = "inference",
        consider_available_only: bool = True
    ) -> Dict[str, Any]:
        """
        Get hardware recommendations for a model.
        
        Args:
            model_name: Model name
            task: Task type (inference, training, fine-tuning)
            consider_available_only: Only consider available hardware
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "model": model_name,
            "task": task,
            "recommendations": []
        }
        
        # Get available hardware
        hardware = self.get_hardware_info()
        
        # Simple recommendations based on model size and task
        if "large" in model_name.lower() or "xl" in model_name.lower():
            if hardware.accelerators.get("cuda", {}).get("available"):
                recommendations["recommendations"].append({
                    "accelerator": "cuda",
                    "reason": "Large model benefits from GPU acceleration",
                    "priority": 1
                })
            elif not consider_available_only:
                recommendations["recommendations"].append({
                    "accelerator": "cuda",
                    "reason": "Large model requires GPU for reasonable performance",
                    "priority": 1,
                    "available": False
                })
        
        # Always recommend CPU as fallback
        recommendations["recommendations"].append({
            "accelerator": "cpu",
            "reason": "Fallback option, available on all systems",
            "priority": 10,
            "available": True
        })
        
        return recommendations


# Convenience functions

def get_hardware_kit() -> HardwareKit:
    """
    Get a HardwareKit instance.
    
    Returns:
        HardwareKit instance
    """
    return HardwareKit()


def get_info() -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry."""

    info = _get_default_hardware_kit().get_hardware_info(include_detailed=False)
    return {
        "cpu": info.cpu,
        "gpu": info.gpu,
        "memory": info.memory,
        "accelerators": info.accelerators,
        "platform_info": info.platform_info,
    }


def test() -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry."""

    return _get_default_hardware_kit().test_hardware(accelerator="all", test_level="basic")


def recommend(task_type: str) -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry.

    The registry schema currently provides `task_type` (inference/training/etc).
    We translate that into a basic accelerator recommendation using detected
    hardware.
    """

    task = (task_type or "").strip().lower() or "inference"
    hw = _get_default_hardware_kit().get_hardware_info(include_detailed=False)
    has_cuda = bool(hw.accelerators.get("cuda", {}).get("available"))

    recs: List[Dict[str, Any]] = []
    if has_cuda and task in {"training", "fine-tuning", "finetuning", "inference"}:
        recs.append(
            {
                "accelerator": "cuda",
                "reason": "CUDA GPU available",
                "priority": 1,
                "available": True,
            }
        )

    recs.append(
        {
            "accelerator": "cpu",
            "reason": "Fallback option, available on all systems",
            "priority": 10,
            "available": True,
        }
    )

    return {"task_type": task_type, "recommendations": recs, "hardware": {"accelerators": hw.accelerators}}


__all__ = [
    'HardwareKit',
    'HardwareInfo',
    'get_hardware_kit',
    'get_info',
    'test',
    'recommend',
]

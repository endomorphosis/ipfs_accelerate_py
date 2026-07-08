"""
Hardware detection utilities for model conversion.
"""

import logging
import os
import platform
import subprocess
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class HardwareDetector:
    """
    Utilities for hardware detection.
    """
    
    @staticmethod
    def detect_cuda() -> Tuple[bool, Dict[str, Any]]:
        """
        Detect CUDA availability and get information.
        
        Returns:
            Tuple of (available, info)
        """
        try:
            # Try to import torch for CUDA detection
            import torch
            
            if torch.cuda.is_available():
                info = {
                    'version': torch.version.cuda,
                    'device_count': torch.cuda.device_count(),
                    'devices': []
                }
                
                # Get information for each device
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    info['devices'].append({
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'total_memory': device_props.total_memory,
                        'compute_capability': f"{device_props.major}.{device_props.minor}"
                    })
                    
                return True, info
            else:
                return False, {}
                
        except ImportError:
            # If torch is not available, try command line tools
            try:
                if platform.system() == 'Windows':
                    # On Windows, try nvidia-smi
                    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
                    if result.returncode == 0 and 'GPU' in result.stdout:
                        return True, {'command_line': True, 'output': result.stdout}
                    else:
                        return False, {}
                else:
                    # On Linux, try nvidia-smi or nvcc
                    for cmd in [['nvidia-smi', '-L'], ['nvcc', '--version']]:
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                return True, {'command_line': True, 'output': result.stdout}
                        except:
                            pass
                    return False, {}
            except:
                return False, {}
                
    @staticmethod
    def detect_rocm() -> Tuple[bool, Dict[str, Any]]:
        """
        Detect ROCm (AMD GPU) availability and get information.
        
        Returns:
            Tuple of (available, info)
        """
        try:
            # Try to import torch for ROCm detection
            import torch
            
            if torch.cuda.is_available():
                # Check if this is an AMD GPU
                device_name = torch.cuda.get_device_name(0).lower()
                if any(x in device_name for x in ["amd", "radeon"]):
                    info = {
                        'device_count': torch.cuda.device_count(),
                        'devices': []
                    }
                    
                    # Get information for each device
                    for i in range(torch.cuda.device_count()):
                        device_props = torch.cuda.get_device_properties(i)
                        info['devices'].append({
                            'index': i,
                            'name': torch.cuda.get_device_name(i),
                            'total_memory': device_props.total_memory
                        })
                        
                    return True, info
                else:
                    return False, {}
            else:
                return False, {}
                
        except ImportError:
            # If torch is not available, try command line tools
            try:
                if platform.system() == 'Linux':
                    # On Linux, try rocm-smi
                    result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                    if result.returncode == 0 and 'GPU' in result.stdout:
                        return True, {'command_line': True, 'output': result.stdout}
                    else:
                        return False, {}
                else:
                    return False, {}
            except:
                return False, {}
                
    @staticmethod
    def detect_openvino() -> Tuple[bool, Dict[str, Any]]:
        """
        Detect OpenVINO availability and get information.
        
        Returns:
            Tuple of (available, info)
        """
        try:
            # Try newer OpenVINO API
            try:
                from openvino.runtime import Core, get_version
                
                core = Core()
                info = {
                    'version': get_version(),
                    'devices': core.available_devices
                }
                return True, info
            except ImportError:
                # Try older OpenVINO API
                try:
                    from openvino.inference_engine import IECore
                    
                    ie = IECore()
                    info = {
                        'devices': ie.available_devices
                    }
                    return True, info
                except ImportError:
                    return False, {}
        except Exception:
            return False, {}
            
    @staticmethod
    def detect_mps() -> Tuple[bool, Dict[str, Any]]:
        """
        Detect Apple Metal Performance Shaders availability.
        
        Returns:
            Tuple of (available, info)
        """
        try:
            # Check if we're on Apple platform
            if platform.system() != 'Darwin':
                return False, {}
                
            # Try to import torch for MPS detection
            import torch
            
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info = {
                    'available': True,
                    'built': torch.backends.mps.is_built(),
                    'system': platform.system(),
                    'machine': platform.machine()
                }
                return True, info
            else:
                return False, {}
        except ImportError:
            # If torch is not available, check system info
            if platform.system() == 'Darwin' and platform.machine() in ['arm64', 'x86_64']:
                # macOS with Apple Silicon or Intel (modern enough for MPS)
                return True, {
                    'system': platform.system(),
                    'machine': platform.machine()
                }
            else:
                return False, {}
                
    @staticmethod
    def detect_webgpu() -> Tuple[bool, Dict[str, Any]]:
        """
        Check if system is capable of running WebGPU workloads.
        
        Returns:
            Tuple of (capable, info)
        """
        # WebGPU is a web standard, not directly detectable in Python
        # We can only provide information about compatible hardware
        
        # Check if we have CUDA or Metal, which are likely to support WebGPU
        cuda_available, cuda_info = HardwareDetector.detect_cuda()
        rocm_available, rocm_info = HardwareDetector.detect_rocm()
        mps_available, mps_info = HardwareDetector.detect_mps()
        
        if cuda_available or rocm_available or mps_available:
            return True, {
                'potential_support': True,
                'cuda': cuda_available,
                'rocm': rocm_available,
                'mps': mps_available,
                'browser_check_required': True,
                'note': 'WebGPU capabilities depend on browser support and drivers'
            }
        else:
            return False, {
                'potential_support': False,
                'note': 'No compatible GPU hardware detected'
            }
            
    @staticmethod
    def detect_webnn() -> Tuple[bool, Dict[str, Any]]:
        """
        Check if system is capable of running WebNN workloads.
        
        Returns:
            Tuple of (capable, info)
        """
        # WebNN is a web standard, not directly detectable in Python
        # We can provide information about compatible hardware
        
        # Check for CPUs with good neural network capabilities
        cpu_info = HardwareDetector.get_cpu_info()
        
        # Check if we have CUDA, ROCm, or Metal
        cuda_available, _ = HardwareDetector.detect_cuda()
        rocm_available, _ = HardwareDetector.detect_rocm()
        mps_available, _ = HardwareDetector.detect_mps()
        openvino_available, openvino_info = HardwareDetector.detect_openvino()
        
        has_capable_cpu = (
            cpu_info.get('vendor') in ['GenuineIntel', 'AuthenticAMD'] and
            cpu_info.get('features') and
            any(f in cpu_info['features'] for f in ['avx2', 'avx512', 'vnni'])
        )
        
        if has_capable_cpu or cuda_available or rocm_available or mps_available or openvino_available:
            return True, {
                'potential_support': True,
                'cpu_capable': has_capable_cpu,
                'cuda': cuda_available,
                'rocm': rocm_available,
                'mps': mps_available,
                'openvino': openvino_available,
                'browser_check_required': True,
                'note': 'WebNN capabilities depend on browser support and hardware'
            }
        else:
            return False, {
                'potential_support': False,
                'note': 'No compatible hardware detected for WebNN'
            }
    
    @staticmethod
    def detect_qualcomm() -> Tuple[bool, Dict[str, Any]]:
        """
        Detect Qualcomm AI Engine capabilities.
        
        Returns:
            Tuple of (available, info)
        """
        # Check for Qualcomm libraries
        try:
            # Try to find Qualcomm Neural Processing SDK
            qnn_paths = [
                '/opt/qcom/aistack',
                'C:\\Program Files\\Qualcomm\\AI Engine',
                os.path.expanduser('~/Library/Qualcomm/AI_Engine')
            ]
            
            for path in qnn_paths:
                if os.path.exists(path):
                    return True, {
                        'sdk_path': path,
                        'platform': platform.system()
                    }
            
            # On Android, check if we're running on a Qualcomm device
            if platform.system() == 'Linux' and 'android' in platform.version().lower():
                try:
                    # Try to check device information
                    result = subprocess.run(['getprop', 'ro.board.platform'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0 and any(x in result.stdout.lower() for x in ['snapdragon', 'msm', 'sm']):
                        return True, {
                            'platform': 'Android',
                            'board': result.stdout.strip()
                        }
                except:
                    pass
            
            return False, {}
        except:
            return False, {}
            
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get detailed CPU information.
        
        Returns:
            Dictionary with CPU details
        """
        info = {
            'processor': platform.processor(),
            'cores': os.cpu_count(),
            'system': platform.system(),
            'architecture': platform.machine()
        }
        
        # Try to get more detailed CPU info
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    
                # Extract vendor
                for line in cpuinfo.split('\n'):
                    if 'vendor_id' in line:
                        info['vendor'] = line.split(':')[1].strip()
                        break
                        
                # Extract model name
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['model'] = line.split(':')[1].strip()
                        break
                        
                # Extract features
                for line in cpuinfo.split('\n'):
                    if 'flags' in line or 'Features' in line:
                        info['features'] = line.split(':')[1].strip().split()
                        break
            except:
                pass
        elif platform.system() == 'Darwin':
            try:
                result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'machdep.cpu.brand_string' in line:
                            info['model'] = line.split(':')[1].strip()
                        elif 'machdep.cpu.vendor' in line:
                            info['vendor'] = line.split(':')[1].strip()
                        elif 'machdep.cpu.features' in line:
                            info['features'] = line.split(':')[1].strip().split()
            except:
                pass
        elif platform.system() == 'Windows':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                  r'HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0')
                info['model'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
                info['vendor'] = winreg.QueryValueEx(key, 'VendorIdentifier')[0]
                winreg.CloseKey(key)
            except:
                pass
                
        return info
        
    @staticmethod
    def get_available_hardware() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available hardware.
        
        Returns:
            Dictionary mapping hardware names to information
        """
        hardware_info = {}
        
        # CPU is always available
        hardware_info['cpu'] = {
            'available': True,
            'info': HardwareDetector.get_cpu_info()
        }
        
        # Check CUDA (NVIDIA GPU)
        cuda_available, cuda_info = HardwareDetector.detect_cuda()
        hardware_info['cuda'] = {
            'available': cuda_available,
            'info': cuda_info
        }
        
        # Check ROCm (AMD GPU)
        rocm_available, rocm_info = HardwareDetector.detect_rocm()
        hardware_info['rocm'] = {
            'available': rocm_available,
            'info': rocm_info
        }
        
        # Check OpenVINO
        openvino_available, openvino_info = HardwareDetector.detect_openvino()
        hardware_info['openvino'] = {
            'available': openvino_available,
            'info': openvino_info
        }
        
        # Check MPS (Apple Metal)
        mps_available, mps_info = HardwareDetector.detect_mps()
        hardware_info['mps'] = {
            'available': mps_available,
            'info': mps_info
        }
        
        # Check Qualcomm
        qualcomm_available, qualcomm_info = HardwareDetector.detect_qualcomm()
        hardware_info['qnn'] = {
            'available': qualcomm_available,
            'info': qualcomm_info
        }
        
        # Check WebGPU and WebNN
        webgpu_capable, webgpu_info = HardwareDetector.detect_webgpu()
        hardware_info['webgpu'] = {
            'available': webgpu_capable,
            'info': webgpu_info
        }
        
        webnn_capable, webnn_info = HardwareDetector.detect_webnn()
        hardware_info['webnn'] = {
            'available': webnn_capable,
            'info': webnn_info
        }
        
        return hardware_info
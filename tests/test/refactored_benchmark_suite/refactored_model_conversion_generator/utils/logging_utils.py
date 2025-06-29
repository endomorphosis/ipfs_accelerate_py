"""
Logging utilities for model conversion.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

def setup_logger(name: str = 'model_converter', 
               level: int = logging.INFO,
               log_file: Optional[str] = None,
               log_format: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with formatting and handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (if None, only console logging is used)
        log_format: Log format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Create formatter
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
    """
    Log system information.
    
    Args:
        logger: Logger to use
        
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    import os
    import datetime
    
    # Collect system information
    sys_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'timestamp': datetime.datetime.now().isoformat(),
        'user': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
    }
    
    # Try to get more detailed information
    try:
        # Get RAM information
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        sys_info['memory'] = line.strip()
                        break
        elif platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split(':')[1].strip())
                sys_info['memory'] = f"MemTotal: {mem_bytes / (1024*1024):.2f} MB"
    except:
        pass
    
    # Try to get GPU information
    try:
        from .hardware_detection import HardwareDetector
        sys_info['hardware'] = {}
        
        # Get CPU info
        sys_info['hardware']['cpu'] = HardwareDetector.get_cpu_info()
        
        # Get CUDA info
        cuda_available, cuda_info = HardwareDetector.detect_cuda()
        sys_info['hardware']['cuda'] = {
            'available': cuda_available,
            'info': cuda_info
        }
        
        # Get ROCm info
        rocm_available, rocm_info = HardwareDetector.detect_rocm()
        sys_info['hardware']['rocm'] = {
            'available': rocm_available,
            'info': rocm_info
        }
    except:
        pass
    
    # Log information
    logger.info("System Information:")
    logger.info(f"  Platform: {sys_info['platform']}")
    logger.info(f"  Processor: {sys_info['processor']}")
    logger.info(f"  Python: {sys_info['python_version']}")
    
    if 'memory' in sys_info:
        logger.info(f"  Memory: {sys_info['memory']}")
        
    if 'hardware' in sys_info:
        if sys_info['hardware'].get('cuda', {}).get('available'):
            logger.info("  CUDA: Available")
            if 'version' in sys_info['hardware']['cuda'].get('info', {}):
                logger.info(f"    Version: {sys_info['hardware']['cuda']['info']['version']}")
            if 'device_count' in sys_info['hardware']['cuda'].get('info', {}):
                logger.info(f"    GPUs: {sys_info['hardware']['cuda']['info']['device_count']}")
        else:
            logger.info("  CUDA: Not available")
            
        if sys_info['hardware'].get('rocm', {}).get('available'):
            logger.info("  ROCm: Available")
            if 'device_count' in sys_info['hardware']['rocm'].get('info', {}):
                logger.info(f"    GPUs: {sys_info['hardware']['rocm']['info']['device_count']}")
        else:
            logger.info("  ROCm: Not available")
    
    return sys_info
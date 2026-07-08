"""
Logging utilities for the benchmark suite.
"""

import os
import logging
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file provided
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create automatic log file based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auto_log_file = f"benchmark_{timestamp}.log"
    auto_file_handler = logging.FileHandler(auto_log_file)
    auto_file_handler.setLevel(level)
    auto_file_handler.setFormatter(formatter)
    logger.addHandler(auto_file_handler)
    
    return logger
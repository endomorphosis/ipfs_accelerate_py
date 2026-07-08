#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for the benchmark suite.

This module provides functionality for loading and saving benchmark configurations
from/to files in various formats (YAML, JSON).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("benchmark.config")

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load benchmark configuration from file.
    
    Supports YAML and JSON formats based on file extension.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Dictionary with benchmark configuration
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                logger.error("PyYAML is required for YAML configuration files. Install with 'pip install PyYAML'")
                return {}
                
        elif file_ext == '.json':
            with open(file_path, 'r') as f:
                config = json.load(f)
                
        else:
            logger.error(f"Unsupported configuration file format: {file_ext}")
            return {}
        
        # Validate configuration
        if not isinstance(config, dict):
            logger.error(f"Invalid configuration format: expected dictionary, got {type(config)}")
            return {}
        
        # Convert environment variables if needed
        config = _process_env_vars(config)
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration file {file_path}: {e}")
        return {}

def save_config_to_file(config: Dict[str, Any], file_path: str) -> bool:
    """
    Save benchmark configuration to file.
    
    Args:
        config: Benchmark configuration dictionary
        file_path: Path to output file
        
    Returns:
        True if successful, False otherwise
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        if file_ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                logger.error("PyYAML is required for YAML configuration files. Install with 'pip install PyYAML'")
                return False
                
        elif file_ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        else:
            logger.error(f"Unsupported configuration file format: {file_ext}")
            return False
        
        logger.info(f"Configuration saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {e}")
        return False

def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process environment variables in configuration.
    
    Replaces ${ENV_VAR} with the value of the environment variable.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Processed configuration dictionary
    """
    if isinstance(config, dict):
        return {k: _process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_process_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract environment variable name
        env_var = config[2:-1]
        env_value = os.environ.get(env_var)
        if env_value is None:
            logger.warning(f"Environment variable {env_var} not found")
            return config
        return env_value
    else:
        return config

def create_benchmark_configs_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Create benchmark configurations from a configuration file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        List of benchmark configuration dictionaries
    """
    config = load_config_from_file(file_path)
    
    if not config:
        return []
    
    # Handle different configuration formats
    if "models" in config and isinstance(config["models"], list):
        # Multiple models in a single config
        result = []
        
        for model_config in config["models"]:
            # Create a configuration for each model
            if not isinstance(model_config, dict) or "id" not in model_config:
                logger.warning(f"Invalid model configuration: {model_config}")
                continue
            
            # Start with common configuration
            single_config = {
                "model_id": model_config["id"],
            }
            
            # Add model-specific configuration
            for key, value in model_config.items():
                if key == "id":
                    continue
                single_config[key] = value
            
            # Add global configuration
            for key, value in config.items():
                if key == "models":
                    continue
                if key not in single_config:
                    single_config[key] = value
            
            result.append(single_config)
        
        return result
        
    elif "model_id" in config:
        # Single model configuration
        return [config]
        
    else:
        logger.error("Invalid configuration: missing 'models' or 'model_id'")
        return []
#!/usr/bin/env python3
"""
Configuration Management for Bridge Components

This module provides configuration management for the integration bridges
between various components of the IPFS Accelerate Framework, including
the Benchmark to Predictive Performance Bridge.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bridge_config")

# Default configuration locations
DEFAULT_CONFIG_LOCATIONS = [
    # Current directory
    "bridge_config.json",
    # User's home directory
    os.path.expanduser("~/.ipfs_accelerate/bridge_config.json"),
    # System-wide configuration
    "/etc/ipfs_accelerate/bridge_config.json",
]

# Default configuration values
DEFAULT_CONFIG = {
    "benchmark_predictive_performance": {
        "enabled": True,
        "benchmark_db_path": "benchmark_db.duckdb",
        "predictive_api_url": "http://localhost:8080",
        "api_key": None,
        "sync_interval_minutes": 60,
        "auto_sync_enabled": False,
        "sync_limit": 100,
        "sync_days_lookback": 30,
        "high_priority_models": [
            "bert-base-uncased",
            "gpt2",
            "t5-base",
            "vit-base-patch16-224"
        ],
        "report_output_dir": "reports"
    },
    "logging": {
        "level": "INFO",
        "file": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}

def find_config_file() -> Optional[str]:
    """
    Find the configuration file from the default locations.
    
    Returns:
        Path to the configuration file, or None if not found
    """
    for location in DEFAULT_CONFIG_LOCATIONS:
        if os.path.exists(location):
            return location
    
    return None

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Optional path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Find config file if not specified
    if not config_path:
        config_path = find_config_file()
    
    # Load from file if found
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
                # Update configuration
                for section, values in file_config.items():
                    if section in config:
                        config[section].update(values)
                    else:
                        config[section] = values
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    else:
        logger.info("Using default configuration")
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False

def get_bridge_config(config: Dict[str, Any], bridge_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific bridge.
    
    Args:
        config: Configuration dictionary
        bridge_name: Name of the bridge
        
    Returns:
        Bridge configuration dictionary
    """
    if bridge_name in config:
        return config[bridge_name].copy()
    else:
        # Return default if available
        return DEFAULT_CONFIG.get(bridge_name, {}).copy()

def update_bridge_config(config: Dict[str, Any], bridge_name: str, bridge_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific bridge.
    
    Args:
        config: Configuration dictionary
        bridge_name: Name of the bridge
        bridge_config: Bridge configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()
    
    if bridge_name in updated_config:
        updated_config[bridge_name].update(bridge_config)
    else:
        updated_config[bridge_name] = bridge_config
    
    return updated_config

def create_default_config(config_path: str) -> bool:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    return save_config(DEFAULT_CONFIG, config_path)

def main():
    """
    Main entry point for command line usage.
    
    Usage:
        python bridge_config.py create [--path CONFIG_PATH]
        python bridge_config.py show [--path CONFIG_PATH]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Bridge Configuration Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create default configuration file")
    create_parser.add_argument("--path", type=str, default="bridge_config.json", help="Path to save configuration file")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show current configuration")
    show_parser.add_argument("--path", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "create":
        if create_default_config(args.path):
            print(f"Created default configuration at {args.path}")
        else:
            print(f"Failed to create configuration at {args.path}")
    elif args.command == "show":
        config = load_config(args.path)
        print(json.dumps(config, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
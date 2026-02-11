"""
IPFS Configuration Management

This module handles configuration for IPFS integration with the DuckDB benchmark database.
It provides a centralized way to manage IPFS settings, including endpoints, storage options,
and distributed computing parameters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class IPFSConfig:
    """Configuration for IPFS integration."""
    
    # IPFS daemon settings
    ipfs_api_url: str = "http://127.0.0.1:5001"
    ipfs_gateway_url: str = "http://127.0.0.1:8080"
    ipfs_timeout: int = 30
    
    # Storage settings
    enable_ipfs_storage: bool = False  # Default off for backward compatibility
    local_cache_dir: str = str(Path.home() / ".ipfs_benchmarks" / "cache")
    max_cache_size_gb: float = 10.0
    auto_pin: bool = True  # Pin content to IPFS by default
    
    # Distributed operations
    enable_distributed: bool = False
    distributed_workers: int = 1
    p2p_enabled: bool = False
    
    # Knowledge graph settings
    enable_knowledge_graph: bool = False
    knowledge_graph_backend: str = "memory"  # Options: memory, ipfs, external
    
    # Performance settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    prefetch_enabled: bool = True
    
    # ipfs_datasets_py integration
    use_ipfs_datasets: bool = True
    ipfs_datasets_config: Dict[str, Any] = field(default_factory=dict)
    
    # ipfs_kit_py integration
    use_ipfs_kit: bool = True
    ipfs_kit_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Ensure cache directory exists
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Set default ipfs_datasets config if not provided
        if not self.ipfs_datasets_config:
            self.ipfs_datasets_config = {
                'use_accelerate': True,
                'enable_distributed': self.enable_distributed
            }
        
        # Set default ipfs_kit config if not provided
        if not self.ipfs_kit_config:
            self.ipfs_kit_config = {
                'api_url': self.ipfs_api_url,
                'timeout': self.ipfs_timeout,
                'auto_start_daemons': False  # Don't auto-start by default
            }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'IPFSConfig':
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            IPFSConfig instance
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> 'IPFSConfig':
        """Load configuration from environment variables.
        
        Environment variables:
            IPFS_API_URL: IPFS API endpoint
            IPFS_GATEWAY_URL: IPFS gateway endpoint
            IPFS_ENABLE_STORAGE: Enable IPFS storage (true/false)
            IPFS_ENABLE_DISTRIBUTED: Enable distributed operations (true/false)
            IPFS_CACHE_DIR: Local cache directory
            
        Returns:
            IPFSConfig instance
        """
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            'IPFS_API_URL': 'ipfs_api_url',
            'IPFS_GATEWAY_URL': 'ipfs_gateway_url',
            'IPFS_TIMEOUT': ('ipfs_timeout', int),
            'IPFS_ENABLE_STORAGE': ('enable_ipfs_storage', lambda x: x.lower() == 'true'),
            'IPFS_ENABLE_DISTRIBUTED': ('enable_distributed', lambda x: x.lower() == 'true'),
            'IPFS_CACHE_DIR': 'local_cache_dir',
            'IPFS_MAX_CACHE_SIZE_GB': ('max_cache_size_gb', float),
            'IPFS_ENABLE_CACHE': ('enable_cache', lambda x: x.lower() == 'true'),
            'IPFS_ENABLE_KNOWLEDGE_GRAPH': ('enable_knowledge_graph', lambda x: x.lower() == 'true'),
        }
        
        for env_var, config_field in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if isinstance(config_field, tuple):
                    field_name, converter = config_field
                    try:
                        config_dict[field_name] = converter(value)
                    except Exception as e:
                        logger.warning(f"Failed to convert {env_var}={value}: {e}")
                else:
                    config_dict[config_field] = value
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration
        """
        try:
            config_dict = asdict(self)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return asdict(self)
    
    def update(self, **kwargs) -> None:
        """Update configuration fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration field: {key}")
    
    def is_ipfs_enabled(self) -> bool:
        """Check if IPFS storage is enabled.
        
        Returns:
            True if IPFS storage is enabled
        """
        return self.enable_ipfs_storage and (self.use_ipfs_datasets or self.use_ipfs_kit)
    
    def is_distributed_enabled(self) -> bool:
        """Check if distributed operations are enabled.
        
        Returns:
            True if distributed operations are enabled
        """
        return self.enable_distributed and self.use_ipfs_datasets
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate cache size
        if self.max_cache_size_gb <= 0:
            errors.append("max_cache_size_gb must be positive")
        
        # Validate timeouts
        if self.ipfs_timeout <= 0:
            errors.append("ipfs_timeout must be positive")
        
        # Validate worker count
        if self.distributed_workers < 1:
            errors.append("distributed_workers must be at least 1")
        
        # Validate URLs
        if not self.ipfs_api_url.startswith(('http://', 'https://')):
            errors.append("ipfs_api_url must be a valid HTTP(S) URL")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True


# Global configuration instance
_global_config: Optional[IPFSConfig] = None


def get_ipfs_config() -> IPFSConfig:
    """Get the global IPFS configuration instance.
    
    Returns:
        Global IPFSConfig instance
    """
    global _global_config
    if _global_config is None:
        # Try to load from environment first
        _global_config = IPFSConfig.from_env()
        
        # Try to load from config file if it exists
        config_path = os.path.join(Path.home(), ".ipfs_benchmarks", "config.json")
        if os.path.exists(config_path):
            _global_config = IPFSConfig.from_file(config_path)
    
    return _global_config


def set_ipfs_config(config: IPFSConfig) -> None:
    """Set the global IPFS configuration instance.
    
    Args:
        config: IPFSConfig instance to set as global
    """
    global _global_config
    _global_config = config


def reset_ipfs_config() -> None:
    """Reset the global IPFS configuration to defaults."""
    global _global_config
    _global_config = None

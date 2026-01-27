"""
Secrets Manager with Encrypted Credential Storage

Provides secure storage for API keys and credentials using encryption.
Supports environment variables, encrypted files, and in-memory storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Secure credential storage with encryption support.
    
    Features:
    - Encrypted credential storage using Fernet (symmetric encryption)
    - Environment variable fallback
    - In-memory credential caching
    - Secure file permissions
    """
    
    def __init__(
        self,
        secrets_file: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        use_encryption: bool = True
    ):
        """
        Initialize secrets manager.
        
        Args:
            secrets_file: Path to encrypted secrets file (default: ~/.ipfs_accelerate/secrets.enc)
            encryption_key: Encryption key (auto-generated if None and use_encryption=True)
            use_encryption: Whether to encrypt secrets file (default: True)
        """
        self.use_encryption = use_encryption
        self._credentials: Dict[str, str] = {}
        
        # Set default secrets file location
        if secrets_file is None:
            secrets_dir = Path.home() / ".ipfs_accelerate"
            secrets_dir.mkdir(exist_ok=True, mode=0o700)  # Secure permissions
            secrets_file = str(secrets_dir / "secrets.enc")
        
        self.secrets_file = secrets_file
        
        # Initialize encryption
        if use_encryption:
            self._init_encryption(encryption_key)
        else:
            self._cipher = None
        
        # Load existing secrets
        self._load_secrets()
    
    def _init_encryption(self, encryption_key: Optional[bytes] = None):
        """Initialize encryption with Fernet."""
        try:
            from cryptography.fernet import Fernet
            
            if encryption_key is None:
                # Try to load key from environment or key file
                key_file = Path(self.secrets_file).parent / "secrets.key"
                
                if key_file.exists():
                    with open(key_file, 'rb') as f:
                        encryption_key = f.read()
                elif 'IPFS_ACCELERATE_SECRETS_KEY' in os.environ:
                    encryption_key = os.environ['IPFS_ACCELERATE_SECRETS_KEY'].encode()
                else:
                    # Generate new key
                    encryption_key = Fernet.generate_key()
                    # Save key securely
                    with open(key_file, 'wb') as f:
                        f.write(encryption_key)
                    os.chmod(key_file, 0o600)  # Secure permissions
                    logger.info(f"Generated new encryption key at {key_file}")
            
            self._cipher = Fernet(encryption_key)
            
        except ImportError:
            logger.warning(
                "cryptography library not installed. "
                "Install with: pip install cryptography. "
                "Falling back to unencrypted storage."
            )
            self.use_encryption = False
            self._cipher = None
    
    def _load_secrets(self):
        """Load secrets from file."""
        if not os.path.exists(self.secrets_file):
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                data = f.read()
            
            if self.use_encryption and self._cipher:
                # Decrypt data
                decrypted = self._cipher.decrypt(data)
                self._credentials = json.loads(decrypted.decode())
            else:
                # Load as plain JSON
                self._credentials = json.loads(data.decode())
            
            logger.info(f"Loaded {len(self._credentials)} secrets from {self.secrets_file}")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            self._credentials = {}
    
    def _save_secrets(self):
        """Save secrets to file."""
        try:
            # Serialize credentials
            data = json.dumps(self._credentials).encode()
            
            if self.use_encryption and self._cipher:
                # Encrypt data
                encrypted = self._cipher.encrypt(data)
                write_data = encrypted
            else:
                write_data = data
            
            # Write to file with secure permissions
            with open(self.secrets_file, 'wb') as f:
                f.write(write_data)
            os.chmod(self.secrets_file, 0o600)  # Secure permissions
            
            logger.info(f"Saved {len(self._credentials)} secrets to {self.secrets_file}")
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a credential by key.
        
        Priority:
        1. In-memory cache
        2. Environment variable (key converted to UPPER_CASE)
        3. Default value
        
        Args:
            key: Credential key (e.g., 'openai_api_key')
            default: Default value if not found
            
        Returns:
            Credential value or default
        """
        # Check in-memory cache first
        if key in self._credentials:
            return self._credentials[key]
        
        # Check environment variables (convert key to env var format)
        env_key = key.upper()
        if env_key in os.environ:
            return os.environ[env_key]
        
        # Check alternative environment variable formats
        # e.g., 'openai_api_key' -> 'OPENAI_API_KEY'
        alt_formats = [
            key.upper().replace('-', '_'),
            key.upper().replace('.', '_'),
        ]
        
        for env_key in alt_formats:
            if env_key in os.environ:
                return os.environ[env_key]
        
        return default
    
    def set_credential(self, key: str, value: str, persist: bool = True):
        """
        Set a credential.
        
        Args:
            key: Credential key
            value: Credential value
            persist: Whether to save to disk (default: True)
        """
        self._credentials[key] = value
        
        if persist:
            self._save_secrets()
    
    def delete_credential(self, key: str, persist: bool = True):
        """
        Delete a credential.
        
        Args:
            key: Credential key
            persist: Whether to save to disk (default: True)
        """
        if key in self._credentials:
            del self._credentials[key]
            
            if persist:
                self._save_secrets()
    
    def list_credential_keys(self) -> list:
        """
        List all credential keys (not values).
        
        Returns:
            List of credential keys
        """
        return list(self._credentials.keys())
    
    def clear_all(self, persist: bool = True):
        """
        Clear all credentials.
        
        Args:
            persist: Whether to save to disk (default: True)
        """
        self._credentials.clear()
        
        if persist:
            self._save_secrets()


# Global instance
_global_secrets_manager: Optional[SecretsManager] = None


def get_global_secrets_manager(
    secrets_file: Optional[str] = None,
    encryption_key: Optional[bytes] = None,
    use_encryption: bool = True
) -> SecretsManager:
    """
    Get or create the global secrets manager instance.
    
    Args:
        secrets_file: Path to secrets file (only used on first call)
        encryption_key: Encryption key (only used on first call)
        use_encryption: Whether to use encryption (only used on first call)
        
    Returns:
        Global SecretsManager instance
    """
    global _global_secrets_manager
    
    if _global_secrets_manager is None:
        _global_secrets_manager = SecretsManager(
            secrets_file=secrets_file,
            encryption_key=encryption_key,
            use_encryption=use_encryption
        )
    
    return _global_secrets_manager

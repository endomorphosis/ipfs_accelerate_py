#!/usr/bin/env python3
"""
Distributed Testing Framework - Security Module

This module implements security features for the distributed testing framework.
It provides authentication, authorization, and secure communication between
coordinator and worker nodes.

Usage:
    Import this module in coordinator.py and worker.py to enable security features.
    """

    import base64
    import hashlib
    import hmac
    import json
    import logging
    import os
    import secrets
    import time
    from datetime import datetime, timedelta
    from typing import Dict, List, Optional, Any, Tuple

    import jwt

# Configure logging
    logging.basicConfig()))))
    level=logging.INFO,
    format='%()))))asctime)s - %()))))name)s - %()))))levelname)s - %()))))message)s'
    )
    logger = logging.getLogger()))))__name__)

# Default validity period for tokens ()))))1 hour)
    DEFAULT_TOKEN_EXPIRY = 3600

class SecurityManager:
    """Security manager for distributed testing framework."""
    
    def __init__()))))
    self,
    secret_key: Optional[str] = None,
    token_expiry: int = DEFAULT_TOKEN_EXPIRY,
    required_roles: List[str] = None,
    ):
        """
        Initialize the security manager.
        
        Args:
            secret_key: Secret key for token signing ()))))default: randomly generated)
            token_expiry: Token expiry time in seconds ()))))default: 1 hour)
            required_roles: List of roles required for API access ()))))default: ["worker"]),
            """
        # Set secret key ()))))generate if not provided)
            self.secret_key = secret_key or self._generate_secret_key())))))
        
        # Set token expiry time
            self.token_expiry = token_expiry
        
        # Set required roles
            self.required_roles = required_roles or ["worker"]
            ,,
        # Store API keys and roles:
            self.api_keys: Dict[str, Dict[str, Any]] = {}}}}}
            ,,
        # Store worker tokens
            self.worker_tokens: Dict[str, Dict[str, Any]] = {}}}}}
            ,,
            logger.info()))))"Security manager initialized")
    
    def _generate_secret_key()))))self) -> str:
        """
        Generate a random secret key.
        
        Returns:
            Random secret key as a string
            """
        return secrets.token_hex()))))32)
    
        def generate_api_key()))))self, name: str, roles: List[str] = None) -> str:,
        """
        Generate a new API key.
        
        Args:
            name: Name for the API key
            roles: List of roles for the API key ()))))default: ["worker"]),
            
        Returns:
            API key as a string
            """
        # Generate random API key
            api_key = secrets.token_hex()))))16)
        
        # Set roles ()))))default to worker if none provided):
        if roles is None:
            roles = ["worker"]
            ,,
        # Store API key metadata
            self.api_keys[api_key] = {}}}},
            "name": name,
            "roles": roles,
            "created": datetime.now()))))).isoformat()))))),
            }
        
            logger.info()))))f"Generated API key for {}}}}name} with roles {}}}}roles}")
        
            return api_key
    
            def validate_api_key()))))self, api_key: str) -> Tuple[bool, List[str]]:,
            """
            Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of ()))))is_valid, roles)
            """
        if api_key in self.api_keys:
            return True, self.api_keys[api_key]["roles"],
        else:
            return False, []
            ,,
            def has_required_role()))))self, roles: List[str]) -> bool:,
            """
            Check if the roles include at least one required role.
        :
        Args:
            roles: List of roles to check
            
        Returns:
            True if at least one role matches, False otherwise
        """:
        return any()))))role in self.required_roles for role in roles):
            def generate_worker_token()))))self, worker_id: str, api_key: str) -> Optional[str]:,
            """
            Generate a JWT token for a worker.
        
        Args:
            worker_id: Worker ID
            api_key: API key used for authentication
            
        Returns:
            JWT token as a string, or None if API key is invalid
            """
        # Validate API key
        is_valid, roles = self.validate_api_key()))))api_key):
        if not is_valid:
            logger.warning()))))f"Invalid API key used for worker {}}}}worker_id}")
            return None
        
        # Check if has required role::
        if not self.has_required_role()))))roles):
            logger.warning()))))f"API key for worker {}}}}worker_id} lacks required role")
            return None
        
        # Generate token
            now = datetime.utcnow())))))
            expiry = now + timedelta()))))seconds=self.token_expiry)
        
            payload = {}}}}
            "sub": worker_id,
            "roles": roles,
            "iat": now.timestamp()))))),
            "exp": expiry.timestamp()))))),
            }
        
            token = jwt.encode()))))payload, self.secret_key, algorithm="HS256")
        
        # Store token
            self.worker_tokens[worker_id] = {}}}},
            "token": token,
            "expiry": expiry.isoformat()))))),
            "roles": roles,
            }
        
            logger.info()))))f"Generated token for worker {}}}}worker_id}")
        
            return token
    
            def validate_worker_token()))))self, token: str) -> Tuple[bool, Optional[str], List[str]]:,
            """
            Validate a worker token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Tuple of ()))))is_valid, worker_id, roles)
            """
        try:
            # Decode and verify token
            payload = jwt.decode()))))token, self.secret_key, algorithms=["HS256"])
            ,
            worker_id = payload.get()))))"sub")
            roles = payload.get()))))"roles", [])
            ,
            # Check if worker token is stored:
            if worker_id not in self.worker_tokens:
                logger.warning()))))f"Token for unknown worker {}}}}worker_id}")
            return False, None, []
            ,,
            # Check if has required role::
            if not self.has_required_role()))))roles):
                logger.warning()))))f"Token for worker {}}}}worker_id} lacks required role")
            return False, None, []
            ,,
            return True, worker_id, roles
            
        except jwt.ExpiredSignatureError:
            logger.warning()))))"Token has expired")
            return False, None, []
            ,,
        except jwt.InvalidTokenError:
            logger.warning()))))"Invalid token")
            return False, None, []
            ,,
    def generate_hmac()))))self, message: str) -> str:
        """
        Generate HMAC for a message.
        
        Args:
            message: Message to sign
            
        Returns:
            HMAC signature as a base64-encoded string
            """
            h = hmac.new()))))
            self.secret_key.encode()))))), 
            message.encode()))))), 
            hashlib.sha256
            )
            return base64.b64encode()))))h.digest())))))).decode())))))
    
    def verify_hmac()))))self, message: str, signature: str) -> bool:
        """
        Verify HMAC signature for a message.
        
        Args:
            message: Message to verify
            signature: HMAC signature as a base64-encoded string
            
        Returns:
            True if valid, False otherwise
            """
            expected = self.generate_hmac()))))message)
            return hmac.compare_digest()))))expected, signature)
    :
        def sign_message()))))self, message: Dict[str, Any]) -> Dict[str, Any]:,
        """
        Sign a message with a timestamp and HMAC.
        
        Args:
            message: Message to sign
            
        Returns:
            Message with added timestamp and signature
            """
        # Add timestamp
            signed_message = message.copy())))))
            signed_message["timestamp"] = int()))))time.time()))))))
            ,
        # Convert to string for signing
            message_str = json.dumps()))))signed_message, sort_keys=True)
        
        # Generate signature
            signature = self.generate_hmac()))))message_str)
        
        # Add signature
            signed_message["signature"] = signature
            ,
            return signed_message
    
            def verify_message()))))self, message: Dict[str, Any], max_age: int = 60) -> bool:,
            """
            Verify a signed message.
        
        Args:
            message: Signed message to verify
            max_age: Maximum message age in seconds ()))))default: 60)
            
        Returns:
            True if valid, False otherwise
            """
        # Extract signature:
        if "signature" not in message:
            logger.warning()))))"Message has no signature")
            return False
        
            signature = message.pop()))))"signature")
        
        # Extract timestamp
        if "timestamp" not in message:
            logger.warning()))))"Message has no timestamp")
            return False
        
            timestamp = message.get()))))"timestamp")
        
        # Check timestamp freshness
            now = int()))))time.time()))))))
        if now - timestamp > max_age:
            logger.warning()))))f"Message too old ())))){}}}}now - timestamp} seconds)")
            return False
        
        # Convert to string for verification
            message_str = json.dumps()))))message, sort_keys=True)
        
        # Verify signature
            return self.verify_hmac()))))message_str, signature)
    
    def save_config()))))self, file_path: str) -> None:
        """
        Save security configuration to a file.
        
        Args:
            file_path: Path to configuration file
            """
            config = {}}}}
            "secret_key": self.secret_key,
            "token_expiry": self.token_expiry,
            "required_roles": self.required_roles,
            "api_keys": self.api_keys,
            # Don't save worker tokens as they are temporary
            }
        
        with open()))))file_path, "w") as f:
            json.dump()))))config, f)
            
            logger.info()))))f"Security configuration saved to {}}}}file_path}")
    
            @classmethod
    def load_config()))))cls, file_path: str) -> "SecurityManager":
        """
        Load security configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            SecurityManager instance
            """
        with open()))))file_path, "r") as f:
            config = json.load()))))f)
        
        # Create security manager with loaded configuration
            manager = cls()))))
            secret_key=config.get()))))"secret_key"),
            token_expiry=config.get()))))"token_expiry", DEFAULT_TOKEN_EXPIRY),
            required_roles=config.get()))))"required_roles"),
            )
        
        # Load API keys
            manager.api_keys = config.get()))))"api_keys", {}}}}})
        
            logger.info()))))f"Security configuration loaded from {}}}}file_path}")
        
            return manager


# Helper middleware for aiohttp to check authentication
async def auth_middleware()))))app, handler):
    """Middleware to check authentication for aiohttp requests."""
    async def middleware_handler()))))request):
        # Skip authentication for specific routes
        if request.path in ["/", "/status", "/docs"]:,
    return await handler()))))request)
        
        # Get security manager from app
    security_manager = app.get()))))"security_manager")
        if not security_manager:
            # No security manager, skip authentication
    return await handler()))))request)
        
        # Check for API key in header
    api_key = request.headers.get()))))"X-API-Key")
        if api_key:
            # Validate API key
            is_valid, roles = security_manager.validate_api_key()))))api_key)
            if is_valid and security_manager.has_required_role()))))roles):
                # Set roles in request
                request["roles"] = roles,,
            return await handler()))))request)
        
        # Check for JWT token in header
            auth_header = request.headers.get()))))"Authorization")
        if auth_header and auth_header.startswith()))))"Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            ,
            # Validate token
            is_valid, worker_id, roles = security_manager.validate_worker_token()))))token)
            if is_valid:
                # Set worker_id and roles in request
                request["worker_id"] = worker_id,
                request["roles"] = roles,,
            return await handler()))))request)
        
        # Authentication failed
            return web.json_response()))))
            {}}}}"error": "Authentication required"},
            status=401
            )
    
    return middleware_handler


# Example usage
if __name__ == "__main__":
    # Create security manager
    security_manager = SecurityManager())))))
    
    # Generate API key
    api_key = security_manager.generate_api_key()))))"test-worker", ["worker", "admin"]),
    print()))))f"API Key: {}}}}api_key}")
    
    # Validate API key
    is_valid, roles = security_manager.validate_api_key()))))api_key)
    print()))))f"API Key valid: {}}}}is_valid}, roles: {}}}}roles}")
    
    # Generate worker token
    worker_id = "worker-001"
    token = security_manager.generate_worker_token()))))worker_id, api_key)
    print()))))f"Worker token: {}}}}token}")
    
    # Validate worker token
    is_valid, worker_id, roles = security_manager.validate_worker_token()))))token)
    print()))))f"Token valid: {}}}}is_valid}, worker_id: {}}}}worker_id}, roles: {}}}}roles}")
    
    # Example message signing
    message = {}}}}"type": "heartbeat", "worker_id": worker_id}
    signed_message = security_manager.sign_message()))))message)
    print()))))f"Signed message: {}}}}signed_message}")
    
    # Verify message
    is_valid = security_manager.verify_message()))))signed_message)
    print()))))f"Message valid: {}}}}is_valid}")
    
    # Save configuration
    security_manager.save_config()))))"security_config.json")
    
    # Load configuration
    loaded_manager = SecurityManager.load_config()))))"security_config.json")
    print()))))f"Loaded manager secret key: {}}}}loaded_manager.secret_key}")
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
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default validity period for tokens (1 hour)
DEFAULT_TOKEN_EXPIRY = 3600

class SecurityManager:
    """Security manager for distributed testing framework."""
    
    def __init__(
        self,
        db=None,
        config_path: Optional[str] = None,
        secret_key: Optional[str] = None,
        token_expiry: int = DEFAULT_TOKEN_EXPIRY,
        required_roles: List[str] = None
    ):
        """
        Initialize the security manager.
        
        Args:
            db: Database connection (optional)
            config_path: Path to configuration file (optional)
            secret_key: Secret key for token signing (default: randomly generated)
            token_expiry: Token expiry time in seconds (default: 1 hour)
            required_roles: List of roles required for API access (default: ["worker"])
        """
        # Try to load config from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Use values from config file unless explicitly provided by the caller.
                secret_key = secret_key or config.get('secret_key')

                # token_expiry has a non-None default; treat the default as "not explicitly provided"
                # so config files can override it.
                if token_expiry == DEFAULT_TOKEN_EXPIRY:
                    token_expiry = config.get('token_expiry', DEFAULT_TOKEN_EXPIRY)

                required_roles = required_roles or config.get('required_roles')
                
                # Load API keys
                self.api_keys = config.get('api_keys', {})
                
                logger.info(f"Loaded security configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading security configuration: {e}")
                self.api_keys = {}
        else:
            self.api_keys = {}
        
        # Set secret key (generate if not provided)
        self.secret_key = secret_key or self._generate_secret_key()
        
        # Set token expiry time
        self.token_expiry = token_expiry
        
        # Set required roles
        self.required_roles = required_roles or ["worker"]
        
        # Store worker tokens
        self.worker_tokens = {}
        
        # Store database connection
        self.db = db
        
        # Initialize database tables if provided
        if self.db:
            self._init_database_tables()
        
        logger.info("Security manager initialized")
    
    def _init_database_tables(self):
        """Initialize database tables for security."""
        try:
            # API keys table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id VARCHAR PRIMARY KEY,
                api_key VARCHAR,
                name VARCHAR,
                roles JSON,
                created TIMESTAMP
            )
            """)
            
            # Worker tokens table (optional, could be in-memory only)
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_tokens (
                worker_id VARCHAR PRIMARY KEY,
                token VARCHAR,
                roles JSON,
                expiry TIMESTAMP
            )
            """)
            
            # Load API keys from database
            result = self.db.execute("SELECT key_id, api_key, name, roles, created FROM api_keys").fetchall()
            for row in result:
                key_id, api_key, name, roles_json, created = row
                roles = json.loads(roles_json)
                self.api_keys[api_key] = {
                    "key_id": key_id,
                    "name": name,
                    "roles": roles,
                    "created": created
                }
            
            logger.info(f"Loaded {len(self.api_keys)} API keys from database")
            
        except Exception as e:
            logger.error(f"Error initializing security tables: {e}")
    
    def _generate_secret_key(self) -> str:
        """
        Generate a random secret key.
        
        Returns:
            Random secret key as a string
        """
        return secrets.token_hex(32)
    
    def generate_api_key(self, name: str, roles: List[str] = None) -> Dict[str, Any]:
        """
        Generate a new API key.
        
        Args:
            name: Name for the API key
            roles: List of roles for the API key (default: ["worker"])
            
        Returns:
            Dictionary with API key information
        """
        # Generate random API key and key ID
        api_key = secrets.token_hex(16)
        key_id = secrets.token_hex(8)
        
        # Set roles (default to worker if none provided)
        if roles is None:
            roles = ["worker"]
        
        # Create key info
        now = datetime.now()
        key_info = {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "roles": roles,
            "created": now.isoformat()
        }
        
        # Store API key metadata
        self.api_keys[api_key] = {
            "key_id": key_id,
            "name": name,
            "roles": roles,
            "created": now.isoformat()
        }
        
        # Store in database if available
        if self.db:
            try:
                self.db.execute(
                    """
                    INSERT INTO api_keys (key_id, api_key, name, roles, created)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key_id, api_key, name, json.dumps(roles), now)
                )
            except Exception as e:
                logger.error(f"Error storing API key in database: {e}")
        
        logger.info(f"Generated API key for {name} with roles {roles}")
        
        return key_info
    
    def verify_api_key(self, api_key: str, required_role: str = None) -> bool:
        """
        Verify an API key against a required role.
        
        Args:
            api_key: API key to verify
            required_role: Required role (optional)
            
        Returns:
            True if valid and has required role, False otherwise
        """
        if not api_key:
            return False
            
        is_valid, roles = self.validate_api_key(api_key)
        if not is_valid:
            return False
            
        if required_role:
            return required_role in roles
            
        return True
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, List[str]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, roles)
        """
        if api_key in self.api_keys:
            return True, self.api_keys[api_key]["roles"]
        else:
            return False, []
            
    def has_required_role(self, roles: List[str]) -> bool:
        """
        Check if the roles include at least one required role.
        
        Args:
            roles: List of roles to check
            
        Returns:
            True if at least one role matches, False otherwise
        """
        return any(role in self.required_roles for role in roles)
            
    def generate_token(self, subject: str, role: str) -> str:
        """
        Generate a JWT token.
        
        Args:
            subject: Token subject (e.g., worker_id)
            role: Token role
            
        Returns:
            JWT token as a string
        """
        # Use epoch seconds for iat/exp to avoid timezone-naive datetime.timestamp()
        # behavior (which can yield iat in the future on some systems).
        issued_at = int(time.time())
        expiry = issued_at + int(self.token_expiry)

        payload = {
            "sub": subject,
            "role": role,
            "iat": issued_at,
            "exp": expiry,
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # If token is returned as bytes (depends on jwt version), convert to string
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        logger.info(f"Generated token for {subject} with role {role}")
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            # Decode and verify token (small leeway helps avoid edge-case clock drift)
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"], leeway=1)
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
            
    def generate_worker_token(self, worker_id: str, api_key: str) -> Optional[str]:
        """
        Generate a JWT token for a worker.
        
        Args:
            worker_id: Worker ID
            api_key: API key used for authentication
            
        Returns:
            JWT token as a string, or None if API key is invalid
        """
        # Validate API key
        is_valid, roles = self.validate_api_key(api_key)
        if not is_valid:
            logger.warning(f"Invalid API key used for worker {worker_id}")
            return None
        
        # Check if has required role
        if not self.has_required_role(roles):
            logger.warning(f"API key for worker {worker_id} lacks required role")
            return None
        
        # Generate token
        issued_at = int(time.time())
        expiry = issued_at + int(self.token_expiry)

        payload = {
            "sub": worker_id,
            "roles": roles,
            "iat": issued_at,
            "exp": expiry,
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # If token is returned as bytes (depends on jwt version), convert to string
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        # Store token
        self.worker_tokens[worker_id] = {
            "token": token,
            "expiry": datetime.fromtimestamp(expiry).isoformat(),
            "roles": roles,
        }
        
        # Store in database if available
        if self.db:
            try:
                self.db.execute(
                    """
                    INSERT INTO worker_tokens (worker_id, token, roles, expiry)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(worker_id) DO UPDATE SET
                    token = excluded.token,
                    roles = excluded.roles,
                    expiry = excluded.expiry
                    """,
                    (worker_id, token, json.dumps(roles), expiry)
                )
            except Exception as e:
                logger.error(f"Error storing worker token in database: {e}")
        
        logger.info(f"Generated token for worker {worker_id}")
        
        return token
    
    def validate_worker_token(self, token: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate a worker token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Tuple of (is_valid, worker_id, roles)
        """
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"], leeway=1)
            
            worker_id = payload.get("sub")
            roles = payload.get("roles", [])
            
            # Check if worker token is stored
            if worker_id not in self.worker_tokens:
                logger.warning(f"Token for unknown worker {worker_id}")
                return False, None, []
            
            # Check if has required role
            if not self.has_required_role(roles):
                logger.warning(f"Token for worker {worker_id} lacks required role")
                return False, None, []
            
            return True, worker_id, roles
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return False, None, []
            
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return False, None, []
            
    def generate_hmac(self, message: str) -> str:
        """
        Generate HMAC for a message.
        
        Args:
            message: Message to sign
            
        Returns:
            HMAC signature as a base64-encoded string
        """
        h = hmac.new(
            self.secret_key.encode(), 
            message.encode(), 
            hashlib.sha256
        )
        return base64.b64encode(h.digest()).decode()
    
    def verify_hmac(self, message: str, signature: str) -> bool:
        """
        Verify HMAC signature for a message.
        
        Args:
            message: Message to verify
            signature: HMAC signature as a base64-encoded string
            
        Returns:
            True if valid, False otherwise
        """
        expected = self.generate_hmac(message)
        return hmac.compare_digest(expected, signature)
    def sign_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a message with a timestamp and HMAC.
        
        Args:
            message: Message to sign
            
        Returns:
            Message with added timestamp and signature
        """
        # Add timestamp
        signed_message = message.copy()
        signed_message["timestamp"] = int(time.time())
        
        # Convert to string for signing
        message_str = json.dumps(signed_message, sort_keys=True)
        
        # Generate signature
        signature = self.generate_hmac(message_str)
        
        # Add signature
        signed_message["signature"] = signature
        
        return signed_message
    
    def verify_message(self, message: Dict[str, Any], max_age: int = 60) -> bool:
        """
        Verify a signed message.
        
        Args:
            message: Signed message to verify
            max_age: Maximum message age in seconds (default: 60)
            
        Returns:
            True if valid, False otherwise
        """
        # Work on a copy; callers may reuse the message.
        message_to_verify = message.copy()

        # Extract signature
        if "signature" not in message_to_verify:
            logger.warning("Message has no signature")
            return False

        signature = message_to_verify.pop("signature")
        
        # Extract timestamp
        if "timestamp" not in message_to_verify:
            logger.warning("Message has no timestamp")
            return False

        timestamp = message_to_verify.get("timestamp")
        
        # Check timestamp freshness
        now = int(time.time())
        if now - timestamp > max_age:
            logger.warning(f"Message too old ({now - timestamp} seconds)")
            return False
        
        # Convert to string for verification
        message_str = json.dumps(message_to_verify, sort_keys=True)
        
        # Verify signature
        return self.verify_hmac(message_str, signature)
    
    def save_config(self, file_path: str) -> None:
        """
        Save security configuration to a file.
        
        Args:
            file_path: Path to configuration file
        """
        config = {
            "secret_key": self.secret_key,
            "token_expiry": self.token_expiry,
            "required_roles": self.required_roles,
            "api_keys": self.api_keys,
            # Don't save worker tokens as they are temporary
        }
        
        with open(file_path, "w") as f:
            json.dump(config, f)
            
        logger.info(f"Security configuration saved to {file_path}")
    
    @classmethod
    def load_config(cls, file_path: str) -> "SecurityManager":
        """
        Load security configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            SecurityManager instance
        """
        with open(file_path, "r") as f:
            config = json.load(f)
        
        # Create security manager with loaded configuration
        manager = cls(
            secret_key=config.get("secret_key"),
            token_expiry=config.get("token_expiry", DEFAULT_TOKEN_EXPIRY),
            required_roles=config.get("required_roles"),
        )
        
        # Load API keys
        manager.api_keys = config.get("api_keys", {})
        
        logger.info(f"Security configuration loaded from {file_path}")
        
        return manager


# Helper middleware for aiohttp to check authentication
async def auth_middleware(app, handler):
    """Middleware to check authentication for aiohttp requests."""
    async def middleware_handler(request):
        # Skip authentication for specific routes
        if request.path in ["/", "/status", "/docs"]:
            return await handler(request)
        
        # Get security manager from app
        security_manager = app["security_manager"] if "security_manager" in app else None
        if not security_manager:
            # No security manager, skip authentication
            return await handler(request)
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Validate API key
            is_valid, roles = security_manager.validate_api_key(api_key)
            if is_valid and security_manager.has_required_role(roles):
                # Set roles in request
                request["roles"] = roles
                return await handler(request)
        
        # Check for JWT token in header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Validate token
            is_valid, worker_id, roles = security_manager.validate_worker_token(token)
            if is_valid:
                # Set worker_id and roles in request
                request["worker_id"] = worker_id
                request["roles"] = roles
                return await handler(request)
        
        # Authentication failed
        return web.json_response(
            {"error": "Authentication required"},
            status=401
        )
    
    return middleware_handler


# Example usage
if __name__ == "__main__":
    # Create security manager
    security_manager = SecurityManager()
    
    # Generate API key
    key_info = security_manager.generate_api_key("test-worker", ["worker", "admin"])
    api_key = key_info["api_key"]
    print(f"API Key: {api_key}")
    
    # Validate API key
    is_valid, roles = security_manager.validate_api_key(api_key)
    print(f"API Key valid: {is_valid}, roles: {roles}")
    
    # Generate worker token
    worker_id = "worker-001"
    token = security_manager.generate_worker_token(worker_id, api_key)
    print(f"Worker token: {token}")
    
    # Validate worker token
    is_valid, worker_id, roles = security_manager.validate_worker_token(token)
    print(f"Token valid: {is_valid}, worker_id: {worker_id}, roles: {roles}")
    
    # Example message signing
    message = {"type": "heartbeat", "worker_id": worker_id}
    signed_message = security_manager.sign_message(message)
    print(f"Signed message: {signed_message}")
    
    # Verify message
    is_valid = security_manager.verify_message(signed_message)
    print(f"Message valid: {is_valid}")
    
    # Save configuration
    security_manager.save_config("security_config.json")
    
    # Load configuration
    loaded_manager = SecurityManager.load_config("security_config.json")
    print(f"Loaded manager secret key: {loaded_manager.secret_key}")
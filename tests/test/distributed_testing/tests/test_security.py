#!/usr/bin/env python3
"""
Tests for the Distributed Testing Framework Security Module

This module tests the security features of the distributed testing framework,
including API key management, token generation and validation, and message signing.
"""

import json
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from security import SecurityManager, auth_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSecurityManager(unittest.TestCase):
    """Test cases for the SecurityManager class."""

    def setUp(self):
        """Set up test environment."""
        self.secret_key = "test_secret_key"
        self.security = SecurityManager(
            secret_key=self.secret_key,
            token_expiry=60
        )

    def test_init(self):
        """Test security manager initialization."""
        # Test basic initialization
        self.assertEqual(self.security.secret_key, self.secret_key)
        self.assertEqual(self.security.token_expiry, 60)
        self.assertEqual(self.security.required_roles, ["worker"])
        self.assertEqual(len(self.security.api_keys), 0)
        self.assertEqual(len(self.security.worker_tokens), 0)

    def test_generate_api_key(self):
        """Test API key generation."""
        # Generate API key
        key_info = self.security.generate_api_key("test_key", ["worker", "admin"])
        
        # Verify key info structure
        self.assertIn("key_id", key_info)
        self.assertIn("api_key", key_info)
        self.assertIn("name", key_info)
        self.assertIn("roles", key_info)
        self.assertIn("created", key_info)
        
        # Verify key values
        self.assertEqual(key_info["name"], "test_key")
        self.assertEqual(key_info["roles"], ["worker", "admin"])
        
        # Verify API key is stored
        api_key = key_info["api_key"]
        self.assertIn(api_key, self.security.api_keys)
        self.assertEqual(self.security.api_keys[api_key]["name"], "test_key")
        self.assertEqual(self.security.api_keys[api_key]["roles"], ["worker", "admin"])

    def test_verify_api_key(self):
        """Test API key verification."""
        # Generate test key
        key_info = self.security.generate_api_key("test_key", ["worker"])
        api_key = key_info["api_key"]
        
        # Test valid key
        self.assertTrue(self.security.verify_api_key(api_key))
        
        # Test valid key with role
        self.assertTrue(self.security.verify_api_key(api_key, "worker"))
        
        # Test invalid role
        self.assertFalse(self.security.verify_api_key(api_key, "admin"))
        
        # Test invalid key
        self.assertFalse(self.security.verify_api_key("invalid_key"))
        
        # Test empty key
        self.assertFalse(self.security.verify_api_key(""))
        self.assertFalse(self.security.verify_api_key(None))

    def test_validate_api_key(self):
        """Test API key validation."""
        # Generate test key
        key_info = self.security.generate_api_key("test_key", ["worker", "admin"])
        api_key = key_info["api_key"]
        
        # Test valid key
        is_valid, roles = self.security.validate_api_key(api_key)
        self.assertTrue(is_valid)
        self.assertEqual(roles, ["worker", "admin"])
        
        # Test invalid key
        is_valid, roles = self.security.validate_api_key("invalid_key")
        self.assertFalse(is_valid)
        self.assertEqual(roles, [])

    def test_has_required_role(self):
        """Test role checking."""
        # Default required roles: ["worker"]
        
        # Test with matching role
        self.assertTrue(self.security.has_required_role(["worker"]))
        self.assertTrue(self.security.has_required_role(["admin", "worker"]))
        
        # Test with non-matching role
        self.assertFalse(self.security.has_required_role(["admin"]))
        self.assertFalse(self.security.has_required_role([]))
        
        # Change required roles
        self.security.required_roles = ["admin"]
        
        # Test with matching role
        self.assertTrue(self.security.has_required_role(["admin"]))
        self.assertTrue(self.security.has_required_role(["admin", "worker"]))
        
        # Test with non-matching role
        self.assertFalse(self.security.has_required_role(["worker"]))

    def test_generate_token(self):
        """Test token generation."""
        # Generate token
        token = self.security.generate_token("worker_123", "worker")
        
        # Decode token to verify contents
        payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
        
        # Verify payload
        self.assertEqual(payload["sub"], "worker_123")
        self.assertEqual(payload["role"], "worker")
        self.assertIn("iat", payload)
        self.assertIn("exp", payload)
        
        # Verify expiry (should be 60 seconds from now, with some margin)
        now = time.time()
        self.assertLess(payload["exp"] - now, 61)
        self.assertGreater(payload["exp"] - now, 59)

    def test_verify_token(self):
        """Test token verification."""
        # Generate token
        token = self.security.generate_token("worker_123", "worker")
        
        # Verify valid token
        payload = self.security.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "worker_123")
        self.assertEqual(payload["role"], "worker")
        
        # Verify invalid token
        self.assertIsNone(self.security.verify_token("invalid_token"))
        
        # Test expired token
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
            self.assertIsNone(self.security.verify_token(token))
        
        # Test invalid token
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.InvalidTokenError("Invalid token")
            self.assertIsNone(self.security.verify_token(token))

    def test_generate_worker_token(self):
        """Test worker token generation."""
        # Create test API key
        key_info = self.security.generate_api_key("test_key", ["worker"])
        api_key = key_info["api_key"]
        
        # Generate worker token
        token = self.security.generate_worker_token("worker_123", api_key)
        self.assertIsNotNone(token)
        
        # Verify token is stored
        self.assertIn("worker_123", self.security.worker_tokens)
        self.assertEqual(self.security.worker_tokens["worker_123"]["token"], token)
        
        # Test with invalid API key
        token = self.security.generate_worker_token("worker_456", "invalid_key")
        self.assertIsNone(token)
        
        # Test with API key lacking required role
        key_info = self.security.generate_api_key("test_key2", ["viewer"])
        api_key = key_info["api_key"]
        token = self.security.generate_worker_token("worker_789", api_key)
        self.assertIsNone(token)

    def test_validate_worker_token(self):
        """Test worker token validation."""
        # Create test API key and token
        key_info = self.security.generate_api_key("test_key", ["worker"])
        api_key = key_info["api_key"]
        token = self.security.generate_worker_token("worker_123", api_key)
        
        # Validate valid token
        is_valid, worker_id, roles = self.security.validate_worker_token(token)
        self.assertTrue(is_valid)
        self.assertEqual(worker_id, "worker_123")
        self.assertEqual(roles, ["worker"])
        
        # Test expired token
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
            is_valid, worker_id, roles = self.security.validate_worker_token(token)
            self.assertFalse(is_valid)
            self.assertIsNone(worker_id)
            self.assertEqual(roles, [])
        
        # Test invalid token
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.InvalidTokenError("Invalid token")
            is_valid, worker_id, roles = self.security.validate_worker_token(token)
            self.assertFalse(is_valid)
            self.assertIsNone(worker_id)
            self.assertEqual(roles, [])
        
        # Test unknown worker
        # First, decode the token to manipulate it
        payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
        payload["sub"] = "unknown_worker"
        modified_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # If token is bytes, convert to string
        if isinstance(modified_token, bytes):
            modified_token = modified_token.decode('utf-8')
        
        # Validate token for unknown worker
        is_valid, worker_id, roles = self.security.validate_worker_token(modified_token)
        self.assertFalse(is_valid)
        self.assertIsNone(worker_id)
        self.assertEqual(roles, [])

    def test_sign_and_verify_message(self):
        """Test message signing and verification."""
        # Create a test message
        message = {
            "type": "test",
            "id": "123",
            "data": {
                "field1": "value1",
                "field2": 42
            }
        }
        
        # Sign the message
        signed_message = self.security.sign_message(message)
        
        # Verify signed message has signature and timestamp
        self.assertIn("signature", signed_message)
        self.assertIn("timestamp", signed_message)
        
        # Verify original data is preserved
        self.assertEqual(signed_message["type"], "test")
        self.assertEqual(signed_message["id"], "123")
        self.assertEqual(signed_message["data"]["field1"], "value1")
        self.assertEqual(signed_message["data"]["field2"], 42)
        
        # Verify message
        self.assertTrue(self.security.verify_message(signed_message.copy()))
        
        # Test with modified message
        modified_message = signed_message.copy()
        modified_message["data"]["field1"] = "modified"
        self.assertFalse(self.security.verify_message(modified_message))
        
        # Test with expired message
        expired_message = signed_message.copy()
        expired_message["timestamp"] = int(time.time()) - 120  # 2 minutes ago
        self.assertFalse(self.security.verify_message(expired_message))
        
        # Test with missing signature
        no_signature_message = signed_message.copy()
        del no_signature_message["signature"]
        self.assertFalse(self.security.verify_message(no_signature_message))
        
        # Test with missing timestamp
        no_timestamp_message = signed_message.copy()
        del no_timestamp_message["timestamp"]
        self.assertFalse(self.security.verify_message(no_timestamp_message))

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create a security manager with sample API keys
        security = SecurityManager(
            secret_key="test_save_key",
            token_expiry=300,
            required_roles=["worker", "admin"]
        )
        
        # Add some API keys
        security.generate_api_key("key1", ["worker"])
        security.generate_api_key("key2", ["worker", "admin"])
        
        # Create a temporary file for config
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save configuration
            security.save_config(temp_path)
            
            # Check that file exists and contains data
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r") as f:
                config_data = json.load(f)
                self.assertEqual(config_data["secret_key"], "test_save_key")
                self.assertEqual(config_data["token_expiry"], 300)
                self.assertEqual(config_data["required_roles"], ["worker", "admin"])
                self.assertEqual(len(config_data["api_keys"]), 2)
            
            # Load configuration into a new manager
            loaded_security = SecurityManager.load_config(temp_path)
            
            # Verify loaded configuration
            self.assertEqual(loaded_security.secret_key, "test_save_key")
            self.assertEqual(loaded_security.token_expiry, 300)
            self.assertEqual(loaded_security.required_roles, ["worker", "admin"])
            self.assertEqual(len(loaded_security.api_keys), 2)
            
            # Verify API keys were loaded
            api_keys = list(loaded_security.api_keys.keys())
            for api_key in api_keys:
                if loaded_security.api_keys[api_key]["name"] == "key1":
                    self.assertEqual(loaded_security.api_keys[api_key]["roles"], ["worker"])
                elif loaded_security.api_keys[api_key]["name"] == "key2":
                    self.assertEqual(loaded_security.api_keys[api_key]["roles"], ["worker", "admin"])
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_init_with_db(self):
        """Test initialization with database."""
        # Create mock database
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [
            ("key_id_1", "api_key_1", "test_key_1", json.dumps(["worker"]), "2023-01-01T00:00:00"),
            ("key_id_2", "api_key_2", "test_key_2", json.dumps(["worker", "admin"]), "2023-01-02T00:00:00")
        ]
        
        # Create security manager with mock database
        security = SecurityManager(db=mock_db)
        
        # Verify database tables were created
        calls = mock_db.execute.call_args_list
        self.assertEqual(len(calls), 3)  # Two CREATE TABLE and one SELECT
        
        # Verify API keys were loaded
        self.assertEqual(len(security.api_keys), 2)
        self.assertIn("api_key_1", security.api_keys)
        self.assertIn("api_key_2", security.api_keys)
        self.assertEqual(security.api_keys["api_key_1"]["name"], "test_key_1")
        self.assertEqual(security.api_keys["api_key_2"]["name"], "test_key_2")
        self.assertEqual(security.api_keys["api_key_1"]["roles"], ["worker"])
        self.assertEqual(security.api_keys["api_key_2"]["roles"], ["worker", "admin"])

    def test_init_with_config_file(self):
        """Test initialization with configuration file."""
        # Create a temporary config file
        config = {
            "secret_key": "file_secret_key",
            "token_expiry": 600,
            "required_roles": ["worker", "admin"],
            "api_keys": {
                "key_from_file": {
                    "key_id": "file_key_id",
                    "name": "file_key",
                    "roles": ["worker"],
                    "created": "2023-01-01T00:00:00"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(config, temp_file)
        
        try:
            # Create security manager with config file
            security = SecurityManager(config_path=temp_path)
            
            # Verify config was loaded
            self.assertEqual(security.secret_key, "file_secret_key")
            self.assertEqual(security.token_expiry, 600)
            self.assertEqual(security.required_roles, ["worker", "admin"])
            
            # Verify API keys were loaded
            self.assertEqual(len(security.api_keys), 1)
            self.assertIn("key_from_file", security.api_keys)
            self.assertEqual(security.api_keys["key_from_file"]["name"], "file_key")
            self.assertEqual(security.api_keys["key_from_file"]["roles"], ["worker"])
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestAuthMiddleware:
    """Test cases for the auth_middleware function."""
    
    @pytest.fixture
    def setup_app(self):
        """Set up web app with security manager for testing."""
        app = {}
        
        # Create security manager
        security = SecurityManager(secret_key="middleware_test_key")
        app["security_manager"] = security
        
        # Add test API key
        key_info = security.generate_api_key("test_key", ["worker"])
        api_key = key_info["api_key"]
        
        # Add test worker token
        token = security.generate_worker_token("test_worker", api_key)
        
        return app, api_key, token
    
    @pytest.mark.asyncio
    async def test_auth_middleware_public_route(self, setup_app):
        """Test middleware with public route."""
        app, _, _ = setup_app
        
        # Create mock handler
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Create middleware handler
        middleware_handler = await auth_middleware(app, mock_handler)
        
        # Test public route
        for route in ["/", "/status", "/docs"]:
            request = make_mocked_request("GET", route)
            response = await middleware_handler(request)
            
            # Verify request was handled without authentication
            assert response.status == 200
            assert await response.text() == "Success"
    
    @pytest.mark.asyncio
    async def test_auth_middleware_api_key(self, setup_app):
        """Test middleware with API key."""
        app, api_key, _ = setup_app
        
        # Create mock handler
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Create middleware handler
        middleware_handler = await auth_middleware(app, mock_handler)
        
        # Test with valid API key
        request = make_mocked_request(
            "GET", "/protected",
            headers={"X-API-Key": api_key}
        )
        response = await middleware_handler(request)
        
        # Verify request was authenticated
        assert response.status == 200
        assert await response.text() == "Success"
        
        # Test with invalid API key
        request = make_mocked_request(
            "GET", "/protected",
            headers={"X-API-Key": "invalid_key"}
        )
        response = await middleware_handler(request)
        
        # Verify authentication failed
        assert response.status == 401
        
        # Get response body
        body = await response.json()
        assert "error" in body
        assert body["error"] == "Authentication required"
    
    @pytest.mark.asyncio
    async def test_auth_middleware_token(self, setup_app):
        """Test middleware with token."""
        app, _, token = setup_app
        
        # Create mock handler
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Create middleware handler
        middleware_handler = await auth_middleware(app, mock_handler)
        
        # Test with valid token
        request = make_mocked_request(
            "GET", "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        response = await middleware_handler(request)
        
        # Verify request was authenticated
        assert response.status == 200
        assert await response.text() == "Success"
        
        # Test with invalid token
        request = make_mocked_request(
            "GET", "/protected",
            headers={"Authorization": "Bearer invalid_token"}
        )
        response = await middleware_handler(request)
        
        # Verify authentication failed
        assert response.status == 401
    
    @pytest.mark.asyncio
    async def test_auth_middleware_no_auth(self, setup_app):
        """Test middleware with no authentication."""
        app, _, _ = setup_app
        
        # Create mock handler
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Create middleware handler
        middleware_handler = await auth_middleware(app, mock_handler)
        
        # Test with no authentication
        request = make_mocked_request("GET", "/protected")
        response = await middleware_handler(request)
        
        # Verify authentication failed
        assert response.status == 401
    
    @pytest.mark.asyncio
    async def test_auth_middleware_no_security_manager(self):
        """Test middleware with no security manager."""
        # Create app without security manager
        app = {}
        
        # Create mock handler
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Create middleware handler
        middleware_handler = await auth_middleware(app, mock_handler)
        
        # Test with no security manager
        request = make_mocked_request("GET", "/protected")
        response = await middleware_handler(request)
        
        # Verify request was allowed without authentication
        assert response.status == 200
        assert await response.text() == "Success"


if __name__ == '__main__':
    unittest.main()
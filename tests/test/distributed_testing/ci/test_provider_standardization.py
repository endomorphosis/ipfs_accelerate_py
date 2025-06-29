#!/usr/bin/env python3
"""
Test script for standardized CI provider interfaces.

This script tests that all CI providers correctly implement the CIProviderInterface.
"""

import asyncio
import inspect
import logging
from typing import Dict, Any, List

from distributed_testing.ci import (
    CIProviderInterface,
    CIProviderFactory,
    GitHubClient,
    GitLabClient,
    JenkinsClient,
    AzureDevOpsClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_provider_interface_implementation():
    """Test that all providers correctly implement the interface."""
    providers = [
        ("github", GitHubClient),
        ("gitlab", GitLabClient),
        ("jenkins", JenkinsClient),
        ("azure", AzureDevOpsClient)
    ]
    
    # Get abstract methods from the interface
    abstract_methods = set()
    for name, method in inspect.getmembers(CIProviderInterface, inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            abstract_methods.add(name)
    
    logger.info(f"Found {len(abstract_methods)} abstract methods in CIProviderInterface:")
    for method in sorted(abstract_methods):
        logger.info(f"  - {method}")
    
    # Check that all providers implement all abstract methods
    success = True
    for provider_name, provider_class in providers:
        logger.info(f"\nChecking {provider_name} provider...")
        
        # Check if the class inherits from the interface
        if not issubclass(provider_class, CIProviderInterface):
            logger.error(f"  - FAILED: {provider_name} does not inherit from CIProviderInterface")
            success = False
            continue
        
        # Check if all abstract methods are implemented
        not_implemented = []
        for method_name in abstract_methods:
            provider_method = getattr(provider_class, method_name, None)
            
            if provider_method is None:
                not_implemented.append(method_name)
                continue
            
            # Check if the method is still abstract in the provider
            if getattr(provider_method, "__isabstractmethod__", False):
                not_implemented.append(method_name)
                continue
        
        if not_implemented:
            logger.error(f"  - FAILED: {provider_name} does not implement methods: {', '.join(not_implemented)}")
            success = False
        else:
            logger.info(f"  - SUCCESS: {provider_name} correctly implements all abstract methods")
    
    return success

async def test_artifact_handling_standardization():
    """Test that all providers support standardized artifact handling."""
    from distributed_testing.ci.artifact_handler import get_artifact_handler
    
    providers = [
        ("github", GitHubClient),
        ("gitlab", GitLabClient),
        ("jenkins", JenkinsClient),
        ("azure", AzureDevOpsClient)
    ]
    
    logger.info("\nChecking artifact handling standardization...")
    
    # Check that all providers implement upload_artifact method
    success = True
    for provider_name, provider_class in providers:
        logger.info(f"Checking {provider_name} provider artifact handling...")
        
        # Verify method exists
        if not hasattr(provider_class, "upload_artifact"):
            logger.error(f"  - FAILED: {provider_name} does not implement upload_artifact method")
            success = False
            continue
        
        # Create an instance with mock config
        provider = provider_class()
        mock_config = {
            "token": "test-token",
            "repository": "test/repo",
            "organization": "test-org",
            "project": "test-project"
        }
        
        try:
            # Initialize with mock config
            await provider.initialize(mock_config)
            
            # Get method reference
            upload_method = getattr(provider, "upload_artifact")
            
            # Check method signature
            sig = inspect.signature(upload_method)
            
            # Check required parameters
            required_params = ["test_run_id", "artifact_path", "artifact_name"]
            missing_params = [param for param in required_params if param not in sig.parameters]
            
            if missing_params:
                logger.error(f"  - FAILED: {provider_name} upload_artifact method is missing parameters: {', '.join(missing_params)}")
                success = False
                continue
            
            # Check return type annotation
            return_annotation = sig.return_annotation
            if return_annotation != bool and return_annotation != inspect.Signature.empty:
                logger.warning(f"  - WARNING: {provider_name} upload_artifact method has unexpected return type annotation: {return_annotation}")
            
            # Check if method can be integrated with artifact handler
            handler = get_artifact_handler()
            try:
                handler.register_provider(provider_name, provider)
                logger.info(f"  - SUCCESS: {provider_name} can be registered with artifact handler")
            except Exception as e:
                logger.error(f"  - FAILED: {provider_name} cannot be registered with artifact handler: {str(e)}")
                success = False
                continue
            
            logger.info(f"  - SUCCESS: {provider_name} implements standardized artifact handling")
        
        except Exception as e:
            logger.error(f"  - ERROR testing {provider_name}: {str(e)}")
            success = False
            continue
        
        finally:
            # Clean up
            try:
                await provider.close()
            except:
                pass
    
    return success

async def test_provider_factory():
    """Test that the factory correctly registers and creates providers."""
    # Get available providers
    available_providers = CIProviderFactory.get_available_providers()
    
    logger.info(f"Available providers: {', '.join(available_providers)}")
    
    # Check that all providers are registered
    expected_providers = ["github", "gitlab", "jenkins", "azure"]
    all_registered = True
    
    for provider in expected_providers:
        if provider not in available_providers:
            logger.error(f"Provider {provider} is not registered with the factory")
            all_registered = False
    
    return all_registered

async def main():
    """Run all tests."""
    logger.info("Testing standardized CI provider interfaces...")
    
    # Test provider interface implementation
    interface_test_result = await test_provider_interface_implementation()
    
    # Test provider factory
    factory_test_result = await test_provider_factory()
    
    # Test artifact handling standardization
    artifact_test_result = await test_artifact_handling_standardization()
    
    # Print overall result
    if interface_test_result and factory_test_result and artifact_test_result:
        logger.info("\nAll tests PASSED! CI providers are correctly standardized.")
    else:
        logger.error("\nTests FAILED! Some CI providers are not correctly standardized.")

if __name__ == "__main__":
    asyncio.run(main())
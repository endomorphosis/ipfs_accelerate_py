#!/usr/bin/env python3
"""Tests for standardized CI provider interfaces.

These tests are intentionally offline/unit-level:
- verify concrete providers implement the abstract interface
- verify required artifact-related methods exist with sane signatures
- verify the provider factory registers expected providers
"""

import inspect
import logging

import pytest

from distributed_testing.ci import (
    AzureDevOpsClient,
    CIProviderFactory,
    CIProviderInterface,
    GitHubClient,
    GitLabClient,
    JenkinsClient,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_provider_interface_implementation():
    """All providers inherit from the interface and implement abstract methods."""
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
    
    assert success

def test_artifact_handling_standardization():
    """Providers expose standardized artifact handling methods (signature-level)."""
    
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
        
        try:
            upload_method = getattr(provider_class, "upload_artifact")
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
            
            logger.info(f"  - SUCCESS: {provider_name} implements standardized artifact handling")

        except Exception as e:
            logger.error(f"  - ERROR testing {provider_name}: {str(e)}")
            success = False
            continue

    assert success

def test_provider_factory():
    """The factory reports expected providers."""
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
    
    assert all_registered
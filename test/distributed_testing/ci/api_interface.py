#!/usr/bin/env python3
"""
Standardized API Interface for CI/CD Integrations

This module defines standardized interfaces for CI/CD system integrations,
ensuring consistent behavior across different CI providers (GitHub, GitLab, Jenkins, etc.)
"""

import abc
from typing import Dict, List, Any, Optional, Union


class CIProviderInterface(abc.ABC):
    """
    Abstract base class defining the standard interface for all CI providers.
    
    This interface ensures that all CI providers implement a consistent set of methods,
    making it easier to switch between providers or create new implementations.
    """
    
    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the CI provider with configuration.
        
        Args:
            config: Configuration dictionary containing provider-specific settings
            
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abc.abstractmethod
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run in the CI system.
        
        Args:
            test_run_data: Test run data including name, build_id, etc.
            
        Returns:
            Dictionary with test run information
        """
        pass
    
    @abc.abstractmethod
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a test run in the CI system.
        
        Args:
            test_run_id: Test run ID
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        pass
    
    @abc.abstractmethod
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """
        Add a comment to a pull request.
        
        Args:
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            True if comment was added successfully
        """
        pass
    
    @abc.abstractmethod
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload an artifact for a test run.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        pass
    
    @abc.abstractmethod
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """
        Get the URL for a test run artifact.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Name of artifact
            
        Returns:
            URL to the artifact or None if not found
        """
        pass
    
    @abc.abstractmethod
    async def get_test_run_status(self, test_run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run.
        
        Args:
            test_run_id: Test run ID
            
        Returns:
            Dictionary with test run status information
        """
        pass
    
    @abc.abstractmethod
    async def set_build_status(self, status: str, description: str) -> bool:
        """
        Set the build status in the CI system.
        
        Args:
            status: Status to set (success, failure, pending)
            description: Status description
            
        Returns:
            True if status was set successfully
        """
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the CI provider and clean up resources.
        
        Returns:
            None
        """
        pass


class TestRunResult:
    """
    Standardized representation of test run results across different CI systems.
    
    This class provides a common structure for test results, making it easier to
    generate reports and visualizations regardless of the CI system used.
    """

    __test__ = False
    
    def __init__(
        self,
        test_run_id: str,
        status: str,
        total_tests: int,
        passed_tests: int,
        failed_tests: int,
        skipped_tests: int,
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a test run result.
        
        Args:
            test_run_id: Test run ID
            status: Overall status (success, failure, running, etc.)
            total_tests: Total number of tests
            passed_tests: Number of passed tests
            failed_tests: Number of failed tests
            skipped_tests: Number of skipped tests
            duration_seconds: Duration in seconds
            metadata: Additional metadata
        """
        self.test_run_id = test_run_id
        self.status = status
        self.total_tests = total_tests
        self.passed_tests = passed_tests
        self.failed_tests = failed_tests
        self.skipped_tests = skipped_tests
        self.duration_seconds = duration_seconds
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "test_run_id": self.test_run_id,
            "status": self.status,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestRunResult':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TestRunResult instance
        """
        return cls(
            test_run_id=data["test_run_id"],
            status=data["status"],
            total_tests=data["total_tests"],
            passed_tests=data["passed_tests"],
            failed_tests=data["failed_tests"],
            skipped_tests=data["skipped_tests"],
            duration_seconds=data["duration_seconds"],
            metadata=data.get("metadata", {})
        )


class CIProviderFactory:
    """
    Factory for creating CI provider instances.
    
    This factory makes it easy to create the appropriate CI provider based on
    the environment or configuration, abstracting away the implementation details.
    """
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type) -> None:
        """
        Register a CI provider class.
        
        Args:
            provider_type: CI provider type identifier
            provider_class: CI provider class
        """
        cls._providers[provider_type] = provider_class
    
    @classmethod
    async def create_provider(cls, provider_type: str, config: Optional[Dict[str, Any]] = None) -> CIProviderInterface:
        """
        Create a CI provider instance.
        
        Args:
            provider_type: CI provider type identifier
            config: Configuration for the provider
            
        Returns:
            CI provider instance
        
        Raises:
            ValueError: If provider type is not registered
        """
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown CI provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        provider = provider_class()
        await provider.initialize(config or {})
        
        return provider
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available provider types.
        
        Returns:
            List of provider type identifiers
        """
        return list(cls._providers.keys())
#!/usr/bin/env python3
"""
Standardized Artifact Handling for CI/CD Providers

This module provides a standardized way to handle artifacts across different CI/CD providers.
It includes classes and utilities for artifact management, ensuring consistent behavior
regardless of the underlying CI/CD system.
"""

import anyio
import logging
import os
import json
import time
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from test.tests.distributed.distributed_testing.ci.api_interface import CIProviderInterface
from test.tests.distributed.distributed_testing.ci.artifact_metadata import ArtifactMetadata, ArtifactDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ArtifactMetadata is now imported from artifact_metadata.py

class ArtifactStorage:
    """
    Local storage for artifacts.
    
    This class manages local storage for artifacts, including organizing them
    by test run, handling metadata, and supporting backup/restore operations.
    """
    
    def __init__(self, storage_dir: str = "./artifacts"):
        """
        Initialize artifact storage.
        
        Args:
            storage_dir: Directory for artifact storage
        """
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.artifacts_by_test_run: Dict[str, List[ArtifactMetadata]] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing metadata if available
        if os.path.exists(self.metadata_file):
            self._load_metadata()
    
    def _load_metadata(self):
        """Load artifact metadata from file."""
        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
            
            # Organize artifacts by test run
            self.artifacts_by_test_run = {}
            
            for test_run_id, artifacts_data in data.items():
                self.artifacts_by_test_run[test_run_id] = [
                    ArtifactMetadata.from_dict(artifact_data)
                    for artifact_data in artifacts_data
                ]
            
            logger.info(f"Loaded metadata for {len(self.artifacts_by_test_run)} test runs")
        
        except Exception as e:
            logger.error(f"Error loading artifact metadata: {str(e)}")
            self.artifacts_by_test_run = {}
    
    def _save_metadata(self):
        """Save artifact metadata to file."""
        try:
            # Organize metadata by test run
            data = {}
            
            for test_run_id, artifacts in self.artifacts_by_test_run.items():
                data[test_run_id] = [
                    artifact.to_dict() for artifact in artifacts
                ]
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved metadata for {len(self.artifacts_by_test_run)} test runs")
        
        except Exception as e:
            logger.error(f"Error saving artifact metadata: {str(e)}")
    
    def store_artifact(
        self,
        source_path: str,
        artifact_name: str,
        artifact_type: str,
        test_run_id: str,
        provider_name: str,
        provider_specific_id: Optional[str] = None
    ) -> Optional[ArtifactMetadata]:
        """
        Store an artifact in local storage.
        
        Args:
            source_path: Path to source file
            artifact_name: Name of the artifact
            artifact_type: Type of artifact
            test_run_id: ID of the test run
            provider_name: Name of the CI provider
            provider_specific_id: Provider-specific artifact ID (optional)
            
        Returns:
            ArtifactMetadata if successful, None otherwise
        """
        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return None
            
            # Create test run directory if it doesn't exist
            test_run_dir = os.path.join(self.storage_dir, test_run_id)
            os.makedirs(test_run_dir, exist_ok=True)
            
            # Generate a storage filename
            storage_filename = f"{int(time.time())}_{artifact_name}"
            storage_path = os.path.join(test_run_dir, storage_filename)
            
            # Copy the file to storage
            shutil.copy2(source_path, storage_path)
            
            # Create artifact metadata
            metadata = ArtifactMetadata(
                artifact_name=artifact_name,
                artifact_path=storage_path,
                artifact_type=artifact_type,
                test_run_id=test_run_id,
                provider_name=provider_name,
                provider_specific_id=provider_specific_id
            )
            
            # Add to artifacts by test run
            if test_run_id not in self.artifacts_by_test_run:
                self.artifacts_by_test_run[test_run_id] = []
            
            self.artifacts_by_test_run[test_run_id].append(metadata)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Stored artifact {artifact_name} for test run {test_run_id}")
            return metadata
        
        except Exception as e:
            logger.error(f"Error storing artifact: {str(e)}")
            return None
    
    def get_artifacts_for_test_run(self, test_run_id: str) -> List[ArtifactMetadata]:
        """
        Get all artifacts for a test run.
        
        Args:
            test_run_id: ID of the test run
            
        Returns:
            List of artifact metadata
        """
        return self.artifacts_by_test_run.get(test_run_id, [])
    
    def get_artifact_by_name(self, test_run_id: str, artifact_name: str) -> Optional[ArtifactMetadata]:
        """
        Get artifact by name for a test run.
        
        Args:
            test_run_id: ID of the test run
            artifact_name: Name of the artifact
            
        Returns:
            Artifact metadata if found, None otherwise
        """
        for artifact in self.artifacts_by_test_run.get(test_run_id, []):
            if artifact.artifact_name == artifact_name:
                return artifact
        
        return None
    
    def purge_artifacts_for_test_run(self, test_run_id: str) -> bool:
        """
        Purge all artifacts for a test run.
        
        Args:
            test_run_id: ID of the test run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we have artifacts for this test run
            if test_run_id not in self.artifacts_by_test_run:
                logger.warning(f"No artifacts found for test run {test_run_id}")
                return True
            
            # Get the test run directory
            test_run_dir = os.path.join(self.storage_dir, test_run_id)
            
            # Delete the directory and its contents
            if os.path.exists(test_run_dir):
                shutil.rmtree(test_run_dir)
            
            # Remove from artifacts by test run
            self.artifacts_by_test_run.pop(test_run_id)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Purged artifacts for test run {test_run_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error purging artifacts for test run {test_run_id}: {str(e)}")
            return False

class ArtifactHandler:
    """
    Handler for managing artifacts across CI providers.
    
    This class provides a standardized way to handle artifacts across different
    CI providers, ensuring consistent behavior and centralized management.
    """
    
    def __init__(self, storage_dir: str = "./artifacts"):
        """
        Initialize artifact handler.
        
        Args:
            storage_dir: Directory for artifact storage
        """
        self.storage = ArtifactStorage(storage_dir)
        self.provider_handlers: Dict[str, CIProviderInterface] = {}
    
    def register_provider(self, provider_name: str, provider: CIProviderInterface):
        """
        Register a CI provider for artifact handling.
        
        Args:
            provider_name: Name of the provider
            provider: CI provider instance
        """
        self.provider_handlers[provider_name] = provider
        logger.info(f"Registered provider {provider_name} for artifact handling")
    
    async def upload_artifact(
        self,
        source_path: str,
        artifact_name: str,
        artifact_type: str,
        test_run_id: str,
        provider_name: str,
        store_locally: bool = True
    ) -> Tuple[bool, Optional[ArtifactMetadata]]:
        """
        Upload an artifact to a CI provider and optionally store locally.
        
        Args:
            source_path: Path to source file
            artifact_name: Name of the artifact
            artifact_type: Type of artifact
            test_run_id: ID of the test run
            provider_name: Name of the CI provider
            store_locally: Whether to store the artifact locally
            
        Returns:
            Tuple of (success, metadata)
        """
        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return False, None
            
            # Check if provider is registered
            if provider_name not in self.provider_handlers:
                logger.warning(f"Provider {provider_name} not registered for artifact handling")
                # We can still store locally
                if store_locally:
                    metadata = self.storage.store_artifact(
                        source_path=source_path,
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                        test_run_id=test_run_id,
                        provider_name=provider_name
                    )
                    return True, metadata
                return False, None
            
            # Get provider
            provider = self.provider_handlers[provider_name]
            
            # Upload to provider
            upload_success = await provider.upload_artifact(
                test_run_id=test_run_id,
                artifact_path=source_path,
                artifact_name=artifact_name
            )
            
            if not upload_success:
                logger.warning(f"Failed to upload artifact {artifact_name} to provider {provider_name}")
                
                # Store locally even if upload failed
                if store_locally:
                    metadata = self.storage.store_artifact(
                        source_path=source_path,
                        artifact_name=artifact_name,
                        artifact_type=artifact_type,
                        test_run_id=test_run_id,
                        provider_name=provider_name
                    )
                    return True, metadata
                
                return False, None
            
            # Store locally if requested
            if store_locally:
                metadata = self.storage.store_artifact(
                    source_path=source_path,
                    artifact_name=artifact_name,
                    artifact_type=artifact_type,
                    test_run_id=test_run_id,
                    provider_name=provider_name
                )
            else:
                # Create metadata without storing
                metadata = ArtifactMetadata(
                    artifact_name=artifact_name,
                    artifact_path=source_path,
                    artifact_type=artifact_type,
                    test_run_id=test_run_id,
                    provider_name=provider_name
                )
            
            logger.info(f"Uploaded artifact {artifact_name} for test run {test_run_id} to provider {provider_name}")
            return True, metadata
        
        except Exception as e:
            logger.error(f"Error uploading artifact: {str(e)}")
            return False, None
    
    async def upload_artifacts_batch(
        self,
        artifacts: List[Dict[str, Any]],
        provider_name: str
    ) -> Dict[str, Tuple[bool, Optional[ArtifactMetadata]]]:
        """
        Upload multiple artifacts in a batch.
        
        Args:
            artifacts: List of artifacts with source_path, artifact_name, artifact_type, test_run_id
            provider_name: Name of the CI provider
            
        Returns:
            Dictionary mapping artifact names to (success, metadata) tuples
        """
        results = {}
        
        for artifact in artifacts:
            source_path = artifact.get("source_path")
            artifact_name = artifact.get("artifact_name")
            artifact_type = artifact.get("artifact_type")
            test_run_id = artifact.get("test_run_id")
            store_locally = artifact.get("store_locally", True)
            
            if not source_path or not artifact_name or not artifact_type or not test_run_id:
                logger.error(f"Missing required fields for artifact: {artifact}")
                results[artifact_name] = (False, None)
                continue
            
            success, metadata = await self.upload_artifact(
                source_path=source_path,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                test_run_id=test_run_id,
                provider_name=provider_name,
                store_locally=store_locally
            )
            
            results[artifact_name] = (success, metadata)
        
        return results
    
    def get_artifacts_for_test_run(self, test_run_id: str) -> List[ArtifactMetadata]:
        """
        Get all artifacts for a test run.
        
        Args:
            test_run_id: ID of the test run
            
        Returns:
            List of artifact metadata
        """
        return self.storage.get_artifacts_for_test_run(test_run_id)
    
    def get_artifact_by_name(self, test_run_id: str, artifact_name: str) -> Optional[ArtifactMetadata]:
        """
        Get artifact by name for a test run.
        
        Args:
            test_run_id: ID of the test run
            artifact_name: Name of the artifact
            
        Returns:
            Artifact metadata if found, None otherwise
        """
        return self.storage.get_artifact_by_name(test_run_id, artifact_name)
    
    async def purge_artifacts_for_test_run(self, test_run_id: str) -> bool:
        """
        Purge all artifacts for a test run.
        
        Args:
            test_run_id: ID of the test run
            
        Returns:
            True if successful, False otherwise
        """
        return self.storage.purge_artifacts_for_test_run(test_run_id)

# Singleton instance for global use
_artifact_handler = None

def get_artifact_handler(storage_dir: str = "./artifacts") -> ArtifactHandler:
    """
    Get the global artifact handler instance.
    
    Args:
        storage_dir: Directory for artifact storage
        
    Returns:
        ArtifactHandler instance
    """
    global _artifact_handler
    
    if _artifact_handler is None:
        _artifact_handler = ArtifactHandler(storage_dir)
    
    return _artifact_handler
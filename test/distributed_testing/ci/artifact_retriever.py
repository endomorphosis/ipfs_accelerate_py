#!/usr/bin/env python3
"""
Artifact Retrieval System for CI/CD Integrations

This module provides a comprehensive system for discovering and retrieving artifacts
from various CI/CD providers, with caching for efficient retrieval, trend analysis
for performance metrics, and tools for comparing artifacts.

Key features:
- Intelligent artifact retrieval from multiple CI/CD providers
- Local caching system for efficient artifact access
- Caching limits based on size and age with automatic cleanup
- Support for downloading both binary and text artifacts
- Parallel batch retrieval for improved performance
- Content hash verification for cached artifacts
- Trend analysis for metrics across multiple artifacts
- Artifact comparison for detecting changes between versions
- Graceful fallbacks when artifacts cannot be retrieved

Example usage:

```python
# Create artifact retriever with custom settings
retriever = ArtifactRetriever(
    cache_dir="./artifact_cache",
    max_cache_size_mb=1024,
    max_cache_age_days=7
)

# Register CI providers
retriever.register_provider("github", github_client)
retriever.register_provider("gitlab", gitlab_client)

# Retrieve an artifact with caching
artifact_path, artifact_metadata = await retriever.retrieve_artifact(
    test_run_id="test-123",
    artifact_name="performance_report.json",
    provider_name="github",
    use_cache=True
)

# Process the artifact
with open(artifact_path, "r") as f:
    data = json.load(f)
    
# Batch retrieve multiple artifacts in parallel
artifacts_to_retrieve = [
    {"test_run_id": "test-123", "artifact_name": "logs.txt", "provider_name": "github"},
    {"test_run_id": "test-123", "artifact_name": "metrics.json", "provider_name": "github"},
    {"test_run_id": "test-456", "artifact_name": "report.json", "provider_name": "gitlab"}
]

results = await retriever.retrieve_artifacts_batch(artifacts_to_retrieve)

# Analyze performance metrics trend
trend = await retriever.analyze_metrics_trend(
    provider_name="github",
    artifact_type="performance_report",
    metric_name="throughput",
    days=30
)

# Compare two versions of an artifact
comparison = await retriever.compare_artifacts(
    artifact1={"test_run_id": "test-123", "artifact_name": "report.json", "provider_name": "github"},
    artifact2={"test_run_id": "test-456", "artifact_name": "report.json", "provider_name": "github"}
)
```

This module works in conjunction with the artifact_metadata and artifact_handler
modules to provide a complete solution for artifact management in the distributed
testing framework's CI/CD integration system.
"""

import anyio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Tuple

import aiohttp
import hashlib

from distributed_testing.ci.api_interface import CIProviderInterface
from distributed_testing.ci.artifact_metadata import ArtifactMetadata, ArtifactDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArtifactRetriever:
    """
    Enhanced artifact retrieval system with caching and analysis.
    
    This class provides tools for retrieving artifacts from CI providers,
    caching them locally, and analyzing trends in artifact metrics.
    """
    
    def __init__(
        self,
        cache_dir: str = "./artifact_cache",
        max_cache_size_mb: int = 1024,
        max_cache_age_days: int = 7,
        providers: Optional[Dict[str, CIProviderInterface]] = None
    ):
        """
        Initialize the artifact retriever.
        
        Args:
            cache_dir: Directory for caching artifacts
            max_cache_size_mb: Maximum cache size in MB
            max_cache_age_days: Maximum age of cached artifacts in days
            providers: Dictionary mapping provider names to provider instances
        """
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cache_age_days = max_cache_age_days
        self.providers = providers or {}
        self.session = None
        self.cache_metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cache metadata if available
        if os.path.exists(self.cache_metadata_file):
            self._load_cache_metadata()
    
    def register_provider(self, name: str, provider: CIProviderInterface) -> None:
        """
        Register a CI provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
        logger.info(f"Registered provider {name} for artifact retrieval")
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    def _load_cache_metadata(self) -> None:
        """Load cache metadata from file."""
        try:
            with open(self.cache_metadata_file, "r") as f:
                self.cache_metadata = json.load(f)
            logger.info(f"Loaded cache metadata for {len(self.cache_metadata)} artifacts")
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            self.cache_metadata = {}
    
    def _save_cache_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.cache_metadata_file, "w") as f:
                json.dump(self.cache_metadata, f, indent=2)
            logger.info(f"Saved cache metadata for {len(self.cache_metadata)} artifacts")
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _get_cache_size(self) -> int:
        """
        Get total size of artifact cache in bytes.
        
        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            # Skip metadata file from size calculation
            for file in files:
                if file == "cache_metadata.json":
                    continue
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size
    
    async def _cleanup_cache(self) -> None:
        """Clean up artifact cache to respect size and age limits."""
        # Check if cleanup is needed
        cache_size_bytes = self._get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        if cache_size_mb < self.max_cache_size_mb:
            logger.debug(f"Cache size {cache_size_mb:.2f} MB is below limit of {self.max_cache_size_mb} MB")
            return
        
        logger.info(f"Cache size {cache_size_mb:.2f} MB exceeds limit of {self.max_cache_size_mb} MB, cleaning up")
        
        # Get all cached artifacts with access time
        cached_artifacts = []
        for cache_key, metadata in self.cache_metadata.items():
            cached_artifacts.append({
                "key": cache_key,
                "path": metadata.get("cache_path"),
                "last_access": metadata.get("last_access_time", 0),
                "size": metadata.get("size", 0)
            })
        
        # Sort by last access time (oldest first)
        cached_artifacts.sort(key=lambda x: x["last_access"])
        
        # Remove oldest artifacts until we're under the size limit
        current_size_mb = cache_size_mb
        removed = 0
        
        for artifact in cached_artifacts:
            if current_size_mb <= self.max_cache_size_mb * 0.8:  # Clean up to 80% of limit
                break
            
            if os.path.exists(artifact["path"]):
                file_size = os.path.getsize(artifact["path"]) / (1024 * 1024)
                try:
                    os.remove(artifact["path"])
                    current_size_mb -= file_size
                    self.cache_metadata.pop(artifact["key"], None)
                    removed += 1
                    logger.debug(f"Removed cached artifact: {artifact['path']}, saved {file_size:.2f} MB")
                except Exception as e:
                    logger.warning(f"Failed to remove cached artifact {artifact['path']}: {str(e)}")
        
        # Save updated cache metadata
        self._save_cache_metadata()
        logger.info(f"Removed {removed} cached artifacts, new cache size: {current_size_mb:.2f} MB")
    
    def _get_cache_path_for_artifact(
        self,
        test_run_id: str,
        artifact_name: str,
        provider_name: str
    ) -> str:
        """
        Get cache path for an artifact.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Artifact name
            provider_name: Provider name
            
        Returns:
            Cache path
        """
        # Create a cache key that uniquely identifies the artifact
        cache_key = f"{provider_name}_{test_run_id}_{artifact_name}"
        
        # Create a directory structure based on provider and test run
        provider_dir = os.path.join(self.cache_dir, provider_name)
        test_run_dir = os.path.join(provider_dir, test_run_id)
        os.makedirs(test_run_dir, exist_ok=True)
        
        # Create a cache file path
        cache_path = os.path.join(test_run_dir, artifact_name)
        
        return cache_path, cache_key
    
    async def retrieve_artifact(
        self,
        test_run_id: str,
        artifact_name: str,
        provider_name: str,
        use_cache: bool = True,
        verify_hash: bool = True
    ) -> Optional[Tuple[str, ArtifactMetadata]]:
        """
        Retrieve an artifact from a CI provider, with caching.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Artifact name
            provider_name: Provider name
            use_cache: Whether to use the cache
            verify_hash: Whether to verify the hash of the cached artifact
            
        Returns:
            Tuple of (artifact path, artifact metadata) or None if retrieval fails
        """
        await self._ensure_session()
        
        # Get cache path and key
        cache_path, cache_key = self._get_cache_path_for_artifact(
            test_run_id=test_run_id,
            artifact_name=artifact_name,
            provider_name=provider_name
        )
        
        # Check if we can use the cached version
        if use_cache and cache_key in self.cache_metadata and os.path.exists(cache_path):
            logger.info(f"Using cached artifact: {cache_path}")
            
            # Update last access time
            self.cache_metadata[cache_key]["last_access_time"] = time.time()
            self._save_cache_metadata()
            
            # Create artifact metadata
            cached_metadata = self.cache_metadata[cache_key].get("metadata", {})
            artifact_metadata = ArtifactMetadata(
                artifact_name=artifact_name,
                artifact_path=cache_path,
                artifact_type=cached_metadata.get("artifact_type"),
                test_run_id=test_run_id,
                provider_name=provider_name,
                provider_specific_id=cached_metadata.get("provider_specific_id"),
                labels=cached_metadata.get("labels", []),
                metadata=cached_metadata.get("additional_metadata", {})
            )
            
            # Verify hash if requested
            if verify_hash and "content_hash" in self.cache_metadata[cache_key]:
                stored_hash = self.cache_metadata[cache_key]["content_hash"]
                if artifact_metadata.content_hash != stored_hash:
                    logger.warning(f"Hash mismatch for cached artifact {cache_path}, retrieving fresh copy")
                    # Continue with fresh retrieval
                else:
                    return cache_path, artifact_metadata
            else:
                return cache_path, artifact_metadata
        
        # Retrieve from provider
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not registered for artifact retrieval")
            return None
        
        provider = self.providers[provider_name]
        
        # Get artifact URL from provider
        artifact_url = await provider.get_artifact_url(test_run_id, artifact_name)
        
        if not artifact_url:
            logger.error(f"Failed to get URL for artifact {artifact_name} from provider {provider_name}")
            return None
        
        logger.info(f"Retrieving artifact from URL: {artifact_url}")
        
        try:
            # Download the artifact
            async with self.session.get(artifact_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download artifact: {response.status} - {await response.text()}")
                    return None
                
                # Check content type to determine if binary
                content_type = response.headers.get("Content-Type", "")
                is_binary = not content_type.startswith(("text/", "application/json", "application/xml"))
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                
                # Write to cache file
                if is_binary:
                    # Binary mode for binary files
                    with open(cache_path, "wb") as f:
                        f.write(await response.read())
                else:
                    # Text mode for text files
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(await response.text())
            
            # Create artifact metadata
            artifact_metadata = ArtifactMetadata(
                artifact_name=artifact_name,
                artifact_path=cache_path,
                test_run_id=test_run_id,
                provider_name=provider_name
            )
            
            # Update cache metadata
            self.cache_metadata[cache_key] = {
                "cache_path": cache_path,
                "download_time": time.time(),
                "last_access_time": time.time(),
                "url": artifact_url,
                "size": os.path.getsize(cache_path),
                "content_hash": artifact_metadata.content_hash,
                "metadata": {
                    "artifact_type": artifact_metadata.artifact_type,
                    "provider_specific_id": artifact_metadata.provider_specific_id,
                    "labels": artifact_metadata.labels,
                    "additional_metadata": artifact_metadata.metadata
                }
            }
            
            self._save_cache_metadata()
            
            # Clean up cache if needed
            await self._cleanup_cache()
            
            logger.info(f"Successfully retrieved and cached artifact: {cache_path}")
            return cache_path, artifact_metadata
        
        except Exception as e:
            logger.error(f"Error retrieving artifact {artifact_name}: {str(e)}")
            return None
    
    async def retrieve_artifacts_batch(
        self,
        artifacts: List[Dict[str, str]],
        use_cache: bool = True
    ) -> Dict[str, Tuple[Optional[str], Optional[ArtifactMetadata]]]:
        """
        Retrieve multiple artifacts in parallel.
        
        Args:
            artifacts: List of dictionaries with test_run_id, artifact_name, provider_name
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary mapping artifact names to (path, metadata) tuples
        """
        # Keep this implementation simple and anyio-friendly: run sequentially.
        # (Parallelization can be reintroduced later via an anyio TaskGroup.)
        results: Dict[str, Tuple[Optional[str], Optional[ArtifactMetadata]]] = {}

        for artifact in artifacts:
            test_run_id = artifact.get("test_run_id")
            artifact_name = artifact.get("artifact_name")
            provider_name = artifact.get("provider_name")

            if not test_run_id or not artifact_name or not provider_name:
                logger.warning(f"Skipping artifact with missing required fields: {artifact}")
                continue

            try:
                result = await self.retrieve_artifact(
                    test_run_id=test_run_id,
                    artifact_name=artifact_name,
                    provider_name=provider_name,
                    use_cache=use_cache,
                )
                results[artifact_name] = result if result else (None, None)
            except Exception as e:
                logger.error(f"Error retrieving artifact {artifact_name}: {str(e)}")
                results[artifact_name] = (None, None)

        return results
    
    async def analyze_metrics_trend(
        self,
        provider_name: str,
        artifact_type: str,
        metric_name: str,
        days: int = 7,
        max_artifacts: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze trend of a metric across multiple artifacts over time.
        
        Args:
            provider_name: Provider name
            artifact_type: Type of artifacts to analyze
            metric_name: Name of the metric to analyze
            days: Number of days to look back
            max_artifacts: Maximum number of artifacts to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not registered for artifact retrieval")
            return {"error": f"Provider {provider_name} not registered"}
        
        provider = self.providers[provider_name]
        
        # Get all test runs from the provider (implementation would depend on provider API)
        # For now, we'll just use the cache metadata to find relevant artifacts
        relevant_artifacts = []
        
        for cache_key, metadata in self.cache_metadata.items():
            if cache_key.startswith(f"{provider_name}_"):
                artifact_metadata = metadata.get("metadata", {})
                if artifact_metadata.get("artifact_type") == artifact_type:
                    cache_path = metadata.get("cache_path")
                    if os.path.exists(cache_path):
                        # Load the artifact
                        artifact = ArtifactMetadata(
                            artifact_name=os.path.basename(cache_path),
                            artifact_path=cache_path,
                            artifact_type=artifact_type,
                            provider_name=provider_name
                        )
                        
                        # Check if we have a creation time and it's within the date range
                        creation_time = metadata.get("download_time")
                        if creation_time:
                            creation_date = datetime.fromtimestamp(creation_time)
                            if (datetime.now() - creation_date).days <= days:
                                relevant_artifacts.append(artifact)
        
        # Sort artifacts by creation time
        relevant_artifacts.sort(key=lambda a: a.creation_time, reverse=True)
        
        # Limit to max_artifacts
        relevant_artifacts = relevant_artifacts[:max_artifacts]
        
        # Extract metrics
        metrics = ArtifactDiscovery.extract_metrics_from_artifacts(
            artifacts=relevant_artifacts,
            metric_names=[metric_name]
        )
        
        values = metrics.get(metric_name, [])
        
        # Calculate trend statistics
        result = {
            "metric": metric_name,
            "artifact_type": artifact_type,
            "provider": provider_name,
            "days_analyzed": days,
            "artifacts_analyzed": len(relevant_artifacts),
            "values": values,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "avg": sum(values) / len(values) if values else None,
            "trend": None
        }
        
        # Calculate trend (simple linear regression)
        if len(values) >= 2:
            # Simple trend calculation
            if values[0] > values[-1]:
                result["trend"] = "increasing"
            elif values[0] < values[-1]:
                result["trend"] = "decreasing"
            else:
                result["trend"] = "stable"
            
            # Calculate percentage change
            first_value = values[-1]  # oldest value
            last_value = values[0]    # newest value
            
            if first_value != 0:
                percent_change = ((last_value - first_value) / abs(first_value)) * 100
                result["percent_change"] = percent_change
        
        return result
    
    async def compare_artifacts(
        self,
        artifact1: Dict[str, str],
        artifact2: Dict[str, str],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two artifacts.
        
        Args:
            artifact1: Dictionary with test_run_id, artifact_name, provider_name
            artifact2: Dictionary with test_run_id, artifact_name, provider_name
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary with comparison results
        """
        # Retrieve artifacts
        result1 = await self.retrieve_artifact(
            test_run_id=artifact1.get("test_run_id", ""),
            artifact_name=artifact1.get("artifact_name", ""),
            provider_name=artifact1.get("provider_name", ""),
            use_cache=use_cache
        )
        
        result2 = await self.retrieve_artifact(
            test_run_id=artifact2.get("test_run_id", ""),
            artifact_name=artifact2.get("artifact_name", ""),
            provider_name=artifact2.get("provider_name", ""),
            use_cache=use_cache
        )
        
        if not result1 or not result2:
            logger.error("Failed to retrieve one or both artifacts for comparison")
            return {"error": "Failed to retrieve one or both artifacts"}
        
        path1, metadata1 = result1
        path2, metadata2 = result2
        
        # Compare basic metadata
        comparison = {
            "artifact1": {
                "name": metadata1.artifact_name,
                "type": metadata1.artifact_type,
                "size": metadata1.file_size,
                "provider": metadata1.provider_name,
                "mimetype": metadata1.mimetype
            },
            "artifact2": {
                "name": metadata2.artifact_name,
                "type": metadata2.artifact_type,
                "size": metadata2.file_size,
                "provider": metadata2.provider_name,
                "mimetype": metadata2.mimetype
            },
            "identical": metadata1.content_hash == metadata2.content_hash,
            "content_differences": {}
        }
        
        # If content is identical, no need for further comparison
        if comparison["identical"]:
            logger.info("Artifacts are identical based on content hash")
            return comparison
        
        # If both are not binary, compare content
        if not metadata1.is_binary and not metadata2.is_binary:
            # Compare line by line for text files
            if metadata1.mimetype and metadata1.mimetype.startswith("text/") and \
               metadata2.mimetype and metadata2.mimetype.startswith("text/"):
                
                try:
                    with open(path1, "r", encoding="utf-8", errors="replace") as f1, \
                         open(path2, "r", encoding="utf-8", errors="replace") as f2:
                        lines1 = f1.readlines()
                        lines2 = f2.readlines()
                    
                    # Calculate differences
                    if len(lines1) != len(lines2):
                        comparison["content_differences"]["line_count"] = {
                            "artifact1": len(lines1),
                            "artifact2": len(lines2)
                        }
                    
                    # Find first few differing lines
                    differing_lines = []
                    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                        if line1 != line2:
                            differing_lines.append({
                                "line_number": i + 1,
                                "artifact1": line1.strip(),
                                "artifact2": line2.strip()
                            })
                            if len(differing_lines) >= 5:  # Limit to 5 differences
                                break
                    
                    if differing_lines:
                        comparison["content_differences"]["differing_lines"] = differing_lines
                
                except Exception as e:
                    logger.error(f"Error comparing text files: {str(e)}")
                    comparison["content_differences"]["error"] = f"Error comparing text files: {str(e)}"
            
            # Compare JSON files
            elif path1.endswith(".json") and path2.endswith(".json"):
                try:
                    with open(path1, "r", encoding="utf-8") as f1, \
                         open(path2, "r", encoding="utf-8") as f2:
                        data1 = json.load(f1)
                        data2 = json.load(f2)
                    
                    # Compare top-level keys
                    keys1 = set(data1.keys())
                    keys2 = set(data2.keys())
                    
                    if keys1 != keys2:
                        comparison["content_differences"]["different_keys"] = {
                            "only_in_artifact1": list(keys1 - keys2),
                            "only_in_artifact2": list(keys2 - keys1),
                            "common": list(keys1 & keys2)
                        }
                    
                    # Compare common metrics
                    metrics_diff = {}
                    for key in ["tests", "passed", "failed", "skipped", "duration", "execution_time", 
                               "throughput", "latency", "memory_usage", "cpu_usage"]:
                        if key in data1 and key in data2 and data1[key] != data2[key]:
                            metrics_diff[key] = {
                                "artifact1": data1[key],
                                "artifact2": data2[key],
                                "difference": self._calculate_difference(data1[key], data2[key])
                            }
                    
                    if metrics_diff:
                        comparison["content_differences"]["metrics"] = metrics_diff
                
                except Exception as e:
                    logger.error(f"Error comparing JSON files: {str(e)}")
                    comparison["content_differences"]["error"] = f"Error comparing JSON files: {str(e)}"
        else:
            # For binary files, we can only compare size and hash
            comparison["content_differences"]["binary_comparison"] = {
                "size_difference_bytes": metadata2.file_size - metadata1.file_size,
                "size_difference_percent": (
                    ((metadata2.file_size - metadata1.file_size) / metadata1.file_size) * 100
                    if metadata1.file_size > 0 else None
                )
            }
        
        return comparison
    
    def _calculate_difference(self, value1: Any, value2: Any) -> Any:
        """
        Calculate the difference between two values.
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            Difference calculation
        """
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # Calculate absolute and percentage difference
            abs_diff = value2 - value1
            percent_diff = (abs_diff / value1) * 100 if value1 != 0 else None
            
            return {
                "absolute": abs_diff,
                "percent": percent_diff
            }
        else:
            # For non-numeric values, return a simple "different" indicator
            return "different"
    
    async def close(self) -> None:
        """Close the retriever and clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("ArtifactRetriever closed")
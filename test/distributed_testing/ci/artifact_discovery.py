#!/usr/bin/env python3
"""
Artifact Discovery and Retrieval System

This module provides tools for discovering, retrieving, and analyzing artifacts
across different CI/CD systems. It integrates with the artifact metadata system
and provides utilities for searching, retrieving, and comparing artifacts.
"""

import anyio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from .ci.artifact_metadata import ArtifactMetadata, ArtifactDiscovery
from .ci.artifact_handler import ArtifactHandler, get_artifact_handler
from .ci.api_interface import CIProviderInterface, CIProviderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArtifactRetriever:
    """
    Retrieves artifacts from CI/CD systems and local storage.
    
    This class provides utilities for retrieving artifacts from different CI/CD
    systems and local storage, including discovery, download, and caching.
    """
    
    def __init__(
        self,
        artifact_handler: Optional[ArtifactHandler] = None,
        cache_dir: str = "./.artifact_cache",
        max_cache_size_mb: int = 1024
    ):
        """
        Initialize artifact retriever.
        
        Args:
            artifact_handler: Artifact handler instance
            cache_dir: Directory for caching retrieved artifacts
            max_cache_size_mb: Maximum cache size in MB
        """
        self.artifact_handler = artifact_handler or get_artifact_handler()
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"ArtifactRetriever initialized with cache_dir={cache_dir}, max_cache_size_mb={max_cache_size_mb}")
    
    async def retrieve_artifact(
        self,
        provider_name: str,
        test_run_id: str,
        artifact_name: str,
        local_path: Optional[str] = None,
        force_download: bool = False
    ) -> Optional[ArtifactMetadata]:
        """
        Retrieve an artifact from a CI provider.
        
        Args:
            provider_name: Name of the CI provider
            test_run_id: Test run ID
            artifact_name: Name of the artifact
            local_path: Local path to save the artifact to (optional)
            force_download: Force download even if in cache
            
        Returns:
            Artifact metadata if successful, None otherwise
        """
        # Check if we have a registered provider
        if provider_name not in self.artifact_handler.provider_handlers:
            logger.error(f"Provider {provider_name} not registered")
            return None
        
        provider = self.artifact_handler.provider_handlers[provider_name]
        
        # Check if we have a cached version
        cache_key = f"{provider_name}_{test_run_id}_{artifact_name}"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path) and not force_download:
            logger.info(f"Using cached artifact: {cache_path}")
            
            # Create metadata for cached artifact
            metadata = ArtifactMetadata(
                artifact_name=artifact_name,
                artifact_path=cache_path,
                test_run_id=test_run_id,
                provider_name=provider_name,
                labels=["cached"]
            )
            
            return metadata
        
        # Download artifact
        destination_path = local_path or cache_path
        
        try:
            # Get artifact URL from provider
            artifact_url = await provider.get_artifact_url(test_run_id, artifact_name)
            
            if not artifact_url:
                logger.error(f"Failed to get artifact URL for {artifact_name} from {provider_name}")
                return None
            
            # Download artifact
            success = await self._download_artifact(artifact_url, destination_path)
            
            if not success:
                logger.error(f"Failed to download artifact {artifact_name} from {artifact_url}")
                return None
            
            # Create metadata for downloaded artifact
            metadata = ArtifactMetadata(
                artifact_name=artifact_name,
                artifact_path=destination_path,
                test_run_id=test_run_id,
                provider_name=provider_name,
                labels=["downloaded"]
            )
            
            logger.info(f"Downloaded artifact {artifact_name} to {destination_path}")
            
            # Clean cache if needed
            await self._clean_cache_if_needed()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error retrieving artifact: {str(e)}")
            return None
    
    async def _download_artifact(self, url: str, destination: str) -> bool:
        """
        Download an artifact from a URL.
        
        Args:
            url: URL to download from
            destination: Local path to save to
            
        Returns:
            True if successful
        """
        import aiohttp
        import aiofiles
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download artifact: {response.status}")
                        return False
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    
                    # Download file
                    async with aiofiles.open(destination, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break
                            await f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading artifact: {str(e)}")
            return False
    
    async def _clean_cache_if_needed(self) -> None:
        """
        Clean cache if it exceeds the maximum size.
        """
        try:
            # Get current cache size
            total_size = 0
            
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            # Convert to MB
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > self.max_cache_size_mb:
                logger.info(f"Cache size ({total_size_mb:.2f} MB) exceeds limit ({self.max_cache_size_mb} MB). Cleaning...")
                
                # Get cache files sorted by access time (oldest first)
                cache_files = []
                
                for root, _, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        atime = os.path.getatime(file_path)
                        size = os.path.getsize(file_path)
                        cache_files.append((file_path, atime, size))
                
                cache_files.sort(key=lambda x: x[1])
                
                # Delete oldest files until we're under the limit
                current_size_mb = total_size_mb
                
                for file_path, _, size in cache_files:
                    if current_size_mb <= self.max_cache_size_mb * 0.8:  # Aim for 80% of limit
                        break
                    
                    try:
                        os.remove(file_path)
                        size_mb = size / (1024 * 1024)
                        current_size_mb -= size_mb
                        logger.info(f"Deleted cache file: {file_path} ({size_mb:.2f} MB)")
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file_path}: {str(e)}")
                
                logger.info(f"Cache cleaned. New size: {current_size_mb:.2f} MB")
        
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
    
    async def find_artifacts_by_query(
        self,
        provider_name: Optional[str] = None,
        test_run_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        days_back: int = 7,
        max_results: int = 100
    ) -> List[ArtifactMetadata]:
        """
        Find artifacts matching a query.
        
        Args:
            provider_name: Name of the CI provider
            test_run_id: Test run ID
            artifact_type: Type of artifact
            name_pattern: Pattern to match in artifact name
            days_back: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of matching artifact metadata
        """
        # Get artifacts from storage
        all_artifacts = []
        
        for test_run in self.artifact_handler.storage.artifacts_by_test_run.values():
            all_artifacts.extend(test_run)
        
        # Filter by age
        if days_back > 0:
            cutoff_time = (datetime.now() - timedelta(days=days_back)).isoformat()
            all_artifacts = [a for a in all_artifacts if a.creation_time >= cutoff_time]
        
        # Use discovery tools to find matching artifacts
        matching_artifacts = ArtifactDiscovery.discover_artifacts(
            artifacts=all_artifacts,
            artifact_type=artifact_type,
            test_run_id=test_run_id,
            provider_name=provider_name,
            name_pattern=name_pattern,
            sort_by="creation_time",
            max_results=max_results
        )
        
        return matching_artifacts
    
    async def retrieve_latest_artifact_by_type(
        self,
        artifact_type: str,
        provider_name: Optional[str] = None,
        test_run_id: Optional[str] = None,
        days_back: int = 7,
        force_download: bool = False
    ) -> Optional[ArtifactMetadata]:
        """
        Retrieve the latest artifact of a specific type.
        
        Args:
            artifact_type: Type of artifact
            provider_name: Name of the CI provider
            test_run_id: Test run ID
            days_back: Number of days to look back
            force_download: Force download even if in cache
            
        Returns:
            Latest artifact metadata if found, None otherwise
        """
        # Find matching artifacts
        artifacts = await self.find_artifacts_by_query(
            provider_name=provider_name,
            test_run_id=test_run_id,
            artifact_type=artifact_type,
            days_back=days_back,
            max_results=100
        )
        
        # Find latest artifact
        latest_artifact = ArtifactDiscovery.find_latest_artifact(
            artifacts=artifacts,
            artifact_type=artifact_type,
            test_run_id=test_run_id
        )
        
        if not latest_artifact:
            logger.warning(f"No {artifact_type} artifact found")
            return None
        
        # Check if we need to download it
        if not os.path.exists(latest_artifact.artifact_path) or force_download:
            # Try to retrieve from provider
            if latest_artifact.provider_name and latest_artifact.test_run_id:
                return await self.retrieve_artifact(
                    provider_name=latest_artifact.provider_name,
                    test_run_id=latest_artifact.test_run_id,
                    artifact_name=latest_artifact.artifact_name,
                    local_path=latest_artifact.artifact_path,
                    force_download=force_download
                )
        
        return latest_artifact
    
    async def analyze_artifacts_for_trends(
        self,
        artifact_type: str,
        metric_names: List[str],
        provider_name: Optional[str] = None,
        days_back: int = 30,
        window_size: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze artifacts for trends in specified metrics.
        
        Args:
            artifact_type: Type of artifact to analyze
            metric_names: List of metric names to analyze
            provider_name: Name of the CI provider to filter by
            days_back: Number of days to look back
            window_size: Window size for moving average
            
        Returns:
            Dictionary with trend analysis by metric
        """
        # Find matching artifacts
        artifacts = await self.find_artifacts_by_query(
            provider_name=provider_name,
            artifact_type=artifact_type,
            days_back=days_back,
            max_results=1000
        )
        
        if not artifacts:
            logger.warning(f"No {artifact_type} artifacts found")
            return {}
        
        # Extract metrics from artifacts
        metrics_data = ArtifactDiscovery.extract_metrics_from_artifacts(artifacts, metric_names)
        
        # Calculate trends
        trends = {}
        
        for metric_name, values in metrics_data.items():
            if not values:
                continue
            
            # Sort artifacts by creation time
            sorted_artifacts = sorted(
                [a for a in artifacts if "metrics" in a.content_metadata and metric_name in a.content_metadata["metrics"]],
                key=lambda a: a.creation_time
            )
            
            # Get timestamps and values
            timestamps = [datetime.fromisoformat(a.creation_time) for a in sorted_artifacts]
            values = [a.content_metadata["metrics"][metric_name] for a in sorted_artifacts]
            
            # Calculate moving average if we have enough data
            if len(values) >= window_size:
                moving_avg = []
                
                for i in range(len(values) - window_size + 1):
                    window = values[i:i+window_size]
                    avg = sum(window) / len(window)
                    moving_avg.append({
                        "timestamp": timestamps[i+window_size-1].isoformat(),
                        "value": avg
                    })
                
                # Calculate trend
                if len(moving_avg) >= 2:
                    first_avg = moving_avg[0]["value"]
                    last_avg = moving_avg[-1]["value"]
                    
                    percent_change = ((last_avg - first_avg) / first_avg) * 100 if first_avg else 0
                    
                    # Determine trend direction
                    if percent_change > 5:
                        trend = "increasing"
                    elif percent_change < -5:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                    
                    trends[metric_name] = {
                        "data_points": len(values),
                        "moving_average": moving_avg,
                        "percent_change": percent_change,
                        "trend": trend,
                        "first_value": values[0],
                        "last_value": values[-1],
                        "min_value": min(values),
                        "max_value": max(values)
                    }
        
        return trends
    
    async def compare_artifact_contents(
        self,
        artifact1: ArtifactMetadata,
        artifact2: ArtifactMetadata
    ) -> Dict[str, Any]:
        """
        Compare the contents of two artifacts.
        
        Args:
            artifact1: First artifact
            artifact2: Second artifact
            
        Returns:
            Dictionary with comparison results
        """
        import difflib
        
        result = {
            "artifact1": artifact1.artifact_name,
            "artifact2": artifact2.artifact_name,
            "differences": {}
        }
        
        # Skip binary files
        if artifact1.is_binary or artifact2.is_binary:
            result["differences"]["error"] = "Cannot compare binary files"
            return result
        
        try:
            # Read file contents
            with open(artifact1.artifact_path, "r", encoding="utf-8", errors="replace") as f1:
                content1 = f1.readlines()
            
            with open(artifact2.artifact_path, "r", encoding="utf-8", errors="replace") as f2:
                content2 = f2.readlines()
            
            # Calculate diff
            differ = difflib.Differ()
            diff = list(differ.compare(content1, content2))
            
            # Extract changes
            additions = [line[2:] for line in diff if line.startswith('+ ')]
            deletions = [line[2:] for line in diff if line.startswith('- ')]
            changes = [line[2:] for line in diff if line.startswith('? ')]
            
            # Calculate similarity
            matcher = difflib.SequenceMatcher(None, ''.join(content1), ''.join(content2))
            similarity = matcher.ratio() * 100
            
            result["differences"] = {
                "additions": len(additions),
                "deletions": len(deletions),
                "changes": len(changes),
                "similarity_percent": similarity,
                "total_changes": len(additions) + len(deletions) + len(changes),
                "diff_summary": diff[:20]  # First 20 lines of diff
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing artifacts: {str(e)}")
            result["differences"]["error"] = str(e)
            return result


async def list_artifact_types():
    """
    List all available artifact types.
    """
    print("Available artifact types:")
    for artifact_type, patterns in ArtifactMetadata.ARTIFACT_TYPES.items():
        print(f"- {artifact_type}: {', '.join(patterns)}")


async def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Artifact Discovery and Retrieval")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List types action
    list_types_parser = subparsers.add_parser("list-types", help="List available artifact types")
    
    # Find artifacts action
    find_parser = subparsers.add_parser("find", help="Find artifacts")
    find_parser.add_argument("--provider", help="CI provider name")
    find_parser.add_argument("--test-run", help="Test run ID")
    find_parser.add_argument("--type", help="Artifact type")
    find_parser.add_argument("--name", help="Name pattern")
    find_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    find_parser.add_argument("--max", type=int, default=10, help="Maximum results")
    
    # Retrieve action
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve artifact")
    retrieve_parser.add_argument("--provider", required=True, help="CI provider name")
    retrieve_parser.add_argument("--test-run", required=True, help="Test run ID")
    retrieve_parser.add_argument("--name", required=True, help="Artifact name")
    retrieve_parser.add_argument("--output", help="Output path")
    retrieve_parser.add_argument("--force", action="store_true", help="Force download")
    
    # Latest action
    latest_parser = subparsers.add_parser("latest", help="Get latest artifact of type")
    latest_parser.add_argument("--type", required=True, help="Artifact type")
    latest_parser.add_argument("--provider", help="CI provider name")
    latest_parser.add_argument("--test-run", help="Test run ID")
    latest_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    latest_parser.add_argument("--force", action="store_true", help="Force download")
    
    # Trends action
    trends_parser = subparsers.add_parser("trends", help="Analyze trends")
    trends_parser.add_argument("--type", required=True, help="Artifact type")
    trends_parser.add_argument("--metrics", required=True, help="Comma-separated metrics")
    trends_parser.add_argument("--provider", help="CI provider name")
    trends_parser.add_argument("--days", type=int, default=30, help="Days to look back")
    trends_parser.add_argument("--window", type=int, default=7, help="Window size")
    
    # Compare action
    compare_parser = subparsers.add_parser("compare", help="Compare artifacts")
    compare_parser.add_argument("--artifact1", required=True, help="Path to first artifact")
    compare_parser.add_argument("--artifact2", required=True, help="Path to second artifact")
    
    args = parser.parse_args()
    
    if args.action == "list-types":
        await list_artifact_types()
    
    elif args.action == "find":
        handler = get_artifact_handler()
        retriever = ArtifactRetriever(handler)
        
        artifacts = await retriever.find_artifacts_by_query(
            provider_name=args.provider,
            test_run_id=args.test_run,
            artifact_type=args.type,
            name_pattern=args.name,
            days_back=args.days,
            max_results=args.max
        )
        
        if not artifacts:
            print("No matching artifacts found")
        else:
            print(f"Found {len(artifacts)} matching artifacts:")
            for i, artifact in enumerate(artifacts):
                print(f"{i+1}. {artifact.artifact_name} ({artifact.artifact_type}) - {artifact.creation_time}")
    
    elif args.action == "retrieve":
        handler = get_artifact_handler()
        retriever = ArtifactRetriever(handler)
        
        artifact = await retriever.retrieve_artifact(
            provider_name=args.provider,
            test_run_id=args.test_run,
            artifact_name=args.name,
            local_path=args.output,
            force_download=args.force
        )
        
        if artifact:
            print(f"Retrieved artifact: {artifact.artifact_path}")
        else:
            print("Failed to retrieve artifact")
    
    elif args.action == "latest":
        handler = get_artifact_handler()
        retriever = ArtifactRetriever(handler)
        
        artifact = await retriever.retrieve_latest_artifact_by_type(
            artifact_type=args.type,
            provider_name=args.provider,
            test_run_id=args.test_run,
            days_back=args.days,
            force_download=args.force
        )
        
        if artifact:
            print(f"Latest {args.type} artifact: {artifact.artifact_name}")
            print(f"Created: {artifact.creation_time}")
            print(f"Path: {artifact.artifact_path}")
        else:
            print(f"No {args.type} artifact found")
    
    elif args.action == "trends":
        handler = get_artifact_handler()
        retriever = ArtifactRetriever(handler)
        
        metrics = [m.strip() for m in args.metrics.split(",")]
        
        trends = await retriever.analyze_artifacts_for_trends(
            artifact_type=args.type,
            metric_names=metrics,
            provider_name=args.provider,
            days_back=args.days,
            window_size=args.window
        )
        
        if not trends:
            print("No trend data available")
        else:
            print(f"Trend analysis for {args.type} artifacts:")
            for metric, trend_data in trends.items():
                print(f"\nMetric: {metric}")
                print(f"Trend: {trend_data['trend']}")
                print(f"Percent change: {trend_data['percent_change']:.2f}%")
                print(f"Data points: {trend_data['data_points']}")
                print(f"Current value: {trend_data['last_value']}")
                print(f"Range: {trend_data['min_value']} - {trend_data['max_value']}")
    
    elif args.action == "compare":
        # Create artifacts
        artifact1 = ArtifactMetadata(
            artifact_name=os.path.basename(args.artifact1),
            artifact_path=args.artifact1
        )
        
        artifact2 = ArtifactMetadata(
            artifact_name=os.path.basename(args.artifact2),
            artifact_path=args.artifact2
        )
        
        retriever = ArtifactRetriever()
        
        result = await retriever.compare_artifact_contents(artifact1, artifact2)
        
        print(f"Comparison of {result['artifact1']} and {result['artifact2']}:")
        for key, value in result["differences"].items():
            if key == "diff_summary":
                print("\nDiff summary (first 20 lines):")
                for line in value:
                    print(f"  {line}")
            else:
                print(f"{key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    anyio.run(main())
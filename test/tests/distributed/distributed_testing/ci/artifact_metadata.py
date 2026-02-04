#!/usr/bin/env python3
"""
Comprehensive Artifact Metadata System for CI/CD Integrations

This module provides an enhanced system for artifact metadata management, 
content classification, discovery, and analysis across different CI/CD providers.
It enables standardized handling of artifacts regardless of the underlying CI/CD system.

Key features:
- Automatic artifact type detection based on file extension and content
- MIME type detection and binary/text classification
- Content analysis and extraction of key metrics from JSON, XML, and text files
- Comprehensive metadata extraction for various artifact types
- Content validation through hash verification
- Labeling system for custom categorization
- Powerful discovery capabilities with filtering by various criteria
- Content-based discovery to find artifacts based on their contents
- Metrics extraction and aggregation for trend analysis
- Grouping utilities for organizing artifacts by type or other criteria

Example usage:

```python
# Create artifact metadata with automatic type detection
metadata = ArtifactMetadata(
    artifact_name="test_report.json",
    artifact_path="/path/to/report.json",
    test_run_id="test-123",
    provider_name="github"
)

# Add custom labels
metadata.add_label("performance")
metadata.add_label("regression-test")

# Add custom metadata
metadata.add_metadata("version", "1.0")
metadata.add_metadata("platform", "linux")

# Discover artifacts matching criteria
matching_artifacts = ArtifactDiscovery.discover_artifacts(
    artifacts=all_artifacts,
    artifact_type="performance_report",
    labels=["regression-test"],
    metadata_query={"platform": "linux"},
    content_query={"metrics.throughput": 1250.5}
)

# Group artifacts by type
grouped_artifacts = ArtifactDiscovery.group_artifacts_by_type(all_artifacts)

# Find latest artifact of a specific type
latest_perf_report = ArtifactDiscovery.find_latest_artifact(
    artifacts=all_artifacts,
    artifact_type="performance_report"
)

# Extract metrics from multiple artifacts
metrics = ArtifactDiscovery.extract_metrics_from_artifacts(
    artifacts=perf_artifacts,
    metric_names=["throughput", "latency", "memory_usage"]
)
```

This module is part of the distributed testing framework's CI/CD integration
system and works in conjunction with the artifact_handler and artifact_retriever
modules to provide a complete solution for artifact management.
"""

import hashlib
import json
import logging
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArtifactMetadata:
    """
    Enhanced artifact metadata with comprehensive content information.
    
    This class extends the basic artifact metadata with content classification,
    validation tools, and comprehensive metadata extraction capabilities.
    """
    
    # Known artifact types and their potential file extensions
    ARTIFACT_TYPES = {
        "test_log": [".log", ".txt", "_log.txt"],
        "test_report": [".xml", ".json", "_report.json", "_report.xml", "_results.xml"],
        "performance_report": ["_perf.json", "_performance.json", "_benchmark.json"],
        "coverage_report": ["_coverage.xml", "_coverage.json", "coverage.xml", "coverage.json"],
        "test_summary": ["_summary.json", "_summary.txt", "_summary.md"],
        "image": [".png", ".jpg", ".jpeg", ".gif", ".svg"],
        "archive": [".zip", ".tar", ".tar.gz", ".tgz"],
        "binary": [".bin", ".exe", ".dll", ".so"],
        "model": [".onnx", ".pt", ".h5", ".pb"],
        "metrics": ["_metrics.json", "_metrics.csv"],
        "trace": ["_trace.json", ".trace"],
        "profiling": ["_profile.json", "_profiling.json"],
        "crash_dump": [".dmp", ".dump", "_crash.txt"]
    }
    
    def __init__(
        self,
        artifact_name: str,
        artifact_path: str,
        artifact_type: Optional[str] = None,
        test_run_id: Optional[str] = None,
        provider_name: Optional[str] = None,
        provider_specific_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize artifact metadata with enhanced fields.
        
        Args:
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (auto-detected if None)
            test_run_id: ID of the test run this artifact belongs to
            provider_name: Name of the CI provider
            provider_specific_id: Provider-specific artifact ID
            labels: List of labels to categorize the artifact
            metadata: Additional metadata as key-value pairs
        """
        self.artifact_name = artifact_name
        self.artifact_path = artifact_path
        self.test_run_id = test_run_id
        self.provider_name = provider_name
        self.provider_specific_id = provider_specific_id
        self.labels = labels or []
        self.metadata = metadata or {}
        self.creation_time = datetime.now().isoformat()
        
        # Auto-detect artifact type if not provided
        self.artifact_type = artifact_type or self._detect_artifact_type(artifact_name, artifact_path)
        
        # Extract mimetype
        self.mimetype = self._get_mimetype(artifact_path)
        
        # Calculate file metadata if file exists
        if os.path.exists(artifact_path):
            self.file_size = os.path.getsize(artifact_path)
            self.content_hash = self._calculate_file_hash(artifact_path)
            self.last_modified = datetime.fromtimestamp(os.path.getmtime(artifact_path)).isoformat()
            self.is_binary = self._is_binary_file(artifact_path)
            self.file_extension = os.path.splitext(artifact_path)[1].lower()
            
            # Extract content metadata based on type
            self.content_metadata = self._extract_content_metadata()
        else:
            self.file_size = 0
            self.content_hash = None
            self.last_modified = self.creation_time
            self.is_binary = False
            self.file_extension = os.path.splitext(artifact_path)[1].lower()
            self.content_metadata = {}
    
    def _detect_artifact_type(self, name: str, path: str) -> str:
        """
        Auto-detect artifact type based on name and extension.
        
        Args:
            name: Artifact name
            path: Artifact file path
            
        Returns:
            Detected artifact type
        """
        if not path:
            return "unknown"
        
        # Check for matches in known types
        file_path = Path(path)
        filename = file_path.name.lower()
        
        for artifact_type, patterns in self.ARTIFACT_TYPES.items():
            for pattern in patterns:
                if filename.endswith(pattern):
                    return artifact_type
        
        # Fall back to extension-based detection
        extension = file_path.suffix.lower()
        
        if extension in [".log", ".txt"]:
            return "log"
        elif extension in [".xml", ".json", ".html", ".md"]:
            return "report"
        elif extension in [".png", ".jpg", ".jpeg", ".gif", ".svg"]:
            return "image"
        elif extension in [".zip", ".tar", ".tar.gz", ".tgz"]:
            return "archive"
        elif extension in [".exe", ".dll", ".so", ".bin"]:
            return "binary"
        
        # Default case
        return "generic"
    
    def _get_mimetype(self, path: str) -> str:
        """
        Get mimetype for a file.
        
        Args:
            path: Path to file
        
        Returns:
            Mimetype string
        """
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type or "application/octet-stream"
    
    def _is_binary_file(self, path: str) -> bool:
        """
        Check if a file is binary.
        
        Args:
            path: Path to file
            
        Returns:
            True if binary, False otherwise
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read as text
                return False
        except UnicodeDecodeError:
            return True
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash of file content
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _extract_content_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from file content based on file type.
        
        Returns:
            Dictionary of content metadata
        """
        if not os.path.exists(self.artifact_path):
            return {}
        
        if self.is_binary:
            return {"binary": True}
        
        # Extract metadata based on file type
        metadata = {}
        
        try:
            # Handle JSON files
            if self.artifact_path.endswith(".json"):
                with open(self.artifact_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract key metrics if it's a test report
                if self.artifact_type == "test_report" or self.artifact_type == "performance_report":
                    metrics = {}
                    
                    # Look for common metrics in test reports
                    for key in ["tests", "passed", "failed", "skipped", "duration", "execution_time", 
                               "throughput", "latency", "memory_usage", "cpu_usage"]:
                        if key in data:
                            metrics[key] = data[key]
                    
                    metadata["metrics"] = metrics
                    
                    # Get summary data if available
                    if "summary" in data:
                        metadata["summary"] = data["summary"]
                
                # Safe extract top-level metadata (limit depth to avoid huge metadata)
                # Add first-level keys to help with artifact discovery
                keys = set()
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)) and len(keys) < 10:
                        keys.add(key)
                
                if keys:
                    metadata["top_level_keys"] = list(keys)
            
            # Handle XML files
            elif self.artifact_path.endswith(".xml"):
                # Basic XML metadata - detailed parsing would require specific parsing logic
                # based on expected XML schema (test report, coverage report, etc.)
                metadata["xml"] = True
            
            # Handle text and log files
            elif self.mimetype and self.mimetype.startswith("text/"):
                # Get line count and extract first few lines for preview
                line_count = 0
                preview_lines = []
                
                with open(self.artifact_path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        line_count += 1
                        if i < 5:  # Get first 5 lines as preview
                            preview_lines.append(line.strip())
                
                metadata["line_count"] = line_count
                metadata["preview"] = preview_lines
        
        except Exception as e:
            logger.warning(f"Error extracting content metadata for {self.artifact_path}: {str(e)}")
        
        return metadata
    
    def validate(self) -> bool:
        """
        Validate artifact metadata and file integrity.
        
        Returns:
            True if valid
        """
        # Check if file exists
        if not os.path.exists(self.artifact_path):
            logger.warning(f"Artifact file does not exist: {self.artifact_path}")
            return False
        
        # Check file size
        if self.file_size != os.path.getsize(self.artifact_path):
            logger.warning(f"Artifact file size has changed: {self.artifact_path}")
            return False
        
        # Check file hash
        current_hash = self._calculate_file_hash(self.artifact_path)
        if self.content_hash != current_hash:
            logger.warning(f"Artifact file hash has changed: {self.artifact_path}")
            return False
        
        return True
    
    def add_label(self, label: str) -> None:
        """
        Add a label to the artifact.
        
        Args:
            label: Label to add
        """
        if label not in self.labels:
            self.labels.append(label)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add a metadata field.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "artifact_name": self.artifact_name,
            "artifact_path": self.artifact_path,
            "artifact_type": self.artifact_type,
            "test_run_id": self.test_run_id,
            "provider_name": self.provider_name,
            "provider_specific_id": self.provider_specific_id,
            "creation_time": self.creation_time,
            "file_size": self.file_size,
            "content_hash": self.content_hash,
            "last_modified": self.last_modified,
            "mimetype": self.mimetype,
            "is_binary": self.is_binary,
            "file_extension": self.file_extension,
            "labels": self.labels,
            "metadata": self.metadata,
            "content_metadata": self.content_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactMetadata':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ArtifactMetadata instance
        """
        metadata = cls(
            artifact_name=data["artifact_name"],
            artifact_path=data["artifact_path"],
            artifact_type=data.get("artifact_type"),
            test_run_id=data.get("test_run_id"),
            provider_name=data.get("provider_name"),
            provider_specific_id=data.get("provider_specific_id"),
            labels=data.get("labels", []),
            metadata=data.get("metadata", {})
        )
        
        metadata.creation_time = data.get("creation_time", metadata.creation_time)
        metadata.file_size = data.get("file_size", 0)
        metadata.content_hash = data.get("content_hash")
        metadata.last_modified = data.get("last_modified", metadata.creation_time)
        metadata.mimetype = data.get("mimetype")
        metadata.is_binary = data.get("is_binary", False)
        metadata.file_extension = data.get("file_extension", "")
        metadata.content_metadata = data.get("content_metadata", {})
        
        return metadata


class ArtifactDiscovery:
    """
    Artifact discovery and retrieval utilities.
    
    This class provides tools for discovering artifacts based on criteria,
    retrieving artifacts, and analyzing artifact metadata.
    """
    
    @staticmethod
    def discover_artifacts(
        artifacts: List[ArtifactMetadata],
        artifact_type: Optional[str] = None,
        test_run_id: Optional[str] = None,
        provider_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        metadata_query: Optional[Dict[str, Any]] = None,
        content_query: Optional[Dict[str, Any]] = None,
        sort_by: str = "creation_time",
        max_results: int = 100
    ) -> List[ArtifactMetadata]:
        """
        Discover artifacts based on search criteria.
        
        Args:
            artifacts: List of all artifacts to search
            artifact_type: Type of artifact to filter by
            test_run_id: Test run ID to filter by
            provider_name: Provider name to filter by
            labels: List of labels that must all be present
            name_pattern: Pattern to match in artifact name
            metadata_query: Key-value pairs to match in metadata
            content_query: Key-value pairs to match in content_metadata 
            sort_by: Field to sort by
            max_results: Maximum number of results to return
            
        Returns:
            Filtered list of artifacts
        """
        import re
        
        # Filter artifacts based on criteria
        filtered_artifacts = []
        
        for artifact in artifacts:
            # Match artifact type if specified
            if artifact_type and artifact.artifact_type != artifact_type:
                continue
            
            # Match test run ID if specified
            if test_run_id and artifact.test_run_id != test_run_id:
                continue
            
            # Match provider name if specified
            if provider_name and artifact.provider_name != provider_name:
                continue
            
            # Match all required labels if specified
            if labels and not all(label in artifact.labels for label in labels):
                continue
            
            # Match name pattern if specified
            if name_pattern and not re.search(name_pattern, artifact.artifact_name, re.IGNORECASE):
                continue
            
            # Match metadata query if specified
            if metadata_query:
                match = True
                for key, value in metadata_query.items():
                    if key not in artifact.metadata or artifact.metadata[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Match content query if specified
            if content_query:
                match = True
                for key, value in content_query.items():
                    # Handle nested path with dot notation (e.g., "metrics.throughput")
                    keys = key.split(".")
                    target = artifact.content_metadata
                    
                    # Navigate through nested keys
                    for k in keys[:-1]:
                        if k in target and isinstance(target[k], dict):
                            target = target[k]
                        else:
                            match = False
                            break
                    
                    # Check final key
                    if match and (keys[-1] not in target or target[keys[-1]] != value):
                        match = False
                
                if not match:
                    continue
            
            # All criteria matched, add to filtered list
            filtered_artifacts.append(artifact)
        
        # Sort artifacts
        if sort_by == "creation_time":
            filtered_artifacts.sort(key=lambda a: a.creation_time, reverse=True)
        elif sort_by == "file_size":
            filtered_artifacts.sort(key=lambda a: a.file_size, reverse=True)
        elif sort_by == "name":
            filtered_artifacts.sort(key=lambda a: a.artifact_name)
        
        # Limit results
        return filtered_artifacts[:max_results]
    
    @staticmethod
    def group_artifacts_by_type(artifacts: List[ArtifactMetadata]) -> Dict[str, List[ArtifactMetadata]]:
        """
        Group artifacts by type.
        
        Args:
            artifacts: List of artifacts
            
        Returns:
            Dictionary mapping artifact type to list of artifacts
        """
        grouped = {}
        
        for artifact in artifacts:
            artifact_type = artifact.artifact_type or "unknown"
            
            if artifact_type not in grouped:
                grouped[artifact_type] = []
            
            grouped[artifact_type].append(artifact)
        
        return grouped
    
    @staticmethod
    def find_latest_artifact(
        artifacts: List[ArtifactMetadata],
        artifact_type: str,
        test_run_id: Optional[str] = None
    ) -> Optional[ArtifactMetadata]:
        """
        Find the latest artifact of a specific type.
        
        Args:
            artifacts: List of artifacts
            artifact_type: Type of artifact to find
            test_run_id: Test run ID to filter by
            
        Returns:
            Latest artifact or None if not found
        """
        matching = [
            a for a in artifacts 
            if a.artifact_type == artifact_type and (test_run_id is None or a.test_run_id == test_run_id)
        ]
        
        if not matching:
            return None
        
        # Return the most recent artifact
        return max(matching, key=lambda a: a.creation_time)
    
    @staticmethod
    def extract_metrics_from_artifacts(
        artifacts: List[ArtifactMetadata],
        metric_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Extract specific metrics from artifacts for analysis.
        
        Args:
            artifacts: List of artifacts
            metric_names: List of metric names to extract
            
        Returns:
            Dictionary mapping metric name to list of values
        """
        metrics = {name: [] for name in metric_names}
        
        for artifact in artifacts:
            # Check if artifact has content_metadata with metrics
            if "metrics" in artifact.content_metadata:
                artifact_metrics = artifact.content_metadata["metrics"]
                
                for name in metric_names:
                    if name in artifact_metrics:
                        metrics[name].append(artifact_metrics[name])
        
        return metrics
# Enhanced Artifact Handling System

## Overview

The Enhanced Artifact Handling System is a comprehensive solution for managing, discovering, and analyzing CI/CD artifacts across different providers. It provides standardized metadata extraction, content classification, efficient retrieval with caching, trend analysis for metrics, and comparison tools.

## Key Components

The system consists of three main components:

1. **Artifact Metadata** - Extracts and manages comprehensive metadata about artifacts
2. **Artifact Discovery** - Provides powerful search and analysis capabilities for artifacts
3. **Artifact Retriever** - Efficiently retrieves and caches artifacts from CI providers

## Feature Highlights

- Automatic artifact type detection based on file extension and content
- MIME type detection and binary/text classification
- Content analysis and extraction of key metrics from various file types
- Content validation through hash verification
- Custom labeling and metadata system
- Powerful discovery with filtering by multiple criteria
- Content-based discovery to find artifacts based on their contents
- Metrics extraction and aggregation for trend analysis
- Intelligent caching for efficient artifact access
- Parallel batch retrieval for improved performance
- Trend analysis for metrics across multiple artifacts
- Artifact comparison for detecting changes between versions
- Graceful fallbacks when artifacts cannot be retrieved

## Architecture

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   CI Provider API   │      │   Artifact Handler   │      │  Artifact Metadata  │
│   (GitHub, GitLab,  │◄────►│   (Upload, Store)    │◄────►│  (Extract, Analyze) │
│    Jenkins, etc.)   │      │                     │      │                     │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                                      ▲                             ▲
                                      │                             │
                                      ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│                     │      │  Artifact Retriever  │      │ Artifact Discovery  │
│   Local Storage     │◄────►│   (Cache, Retrieve)  │◄────►│  (Search, Filter)   │
│                     │      │                     │      │                     │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                                      ▲
                                      │
                                      ▼
                               ┌─────────────────────┐
                               │    Trend Analysis   │
                               │    & Comparison     │
                               │                     │
                               └─────────────────────┘
```

## Integration with CI Providers

The system provides standardized interfaces for CI provider implementations:

- `CIProviderInterface` abstract base class defines the standard API
- Each provider (GitHub, GitLab, Jenkins, etc.) implements this interface
- `CIProviderFactory` creates provider instances based on configuration
- Providers implement artifact uploading and URL retrieval

## Using the Artifact Metadata System

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

# Check if metadata is valid
if metadata.validate():
    print("Artifact is valid!")
```

## Using the Artifact Discovery System

```python
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

## Using the Artifact Retriever

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

## Initializing All Components Together

The system provides utility functions to initialize all components together:

```python
from distributed_testing.ci.register_providers import initialize_artifact_systems

# Define provider configurations
provider_configs = {
    "github": {
        "token": "github_token",
        "repository": "owner/repo",
        "commit_sha": "abcdef123456"
    },
    "gitlab": {
        "token": "gitlab_token",
        "project": "group/project"
    }
}

# Initialize all systems with providers
providers = await initialize_artifact_systems(
    provider_configs=provider_configs,
    artifact_handler_storage_dir="./artifacts",
    artifact_retriever_cache_dir="./artifact_cache"
)

# Now you can use the providers
github_provider = providers["github"]
gitlab_provider = providers["gitlab"]

# And artifact systems are already set up with these providers
artifact_handler = get_artifact_handler()
artifact_retriever = get_artifact_retriever()
```

## CI System-Specific Implementations

### GitHub

- Uses GitHub Gists API for text artifacts
- Uses GitHub Releases API for binary artifacts
- Intelligent detection of binary vs. text files
- Caching of artifact URLs for efficient retrieval

### GitLab

- Uses GitLab Repository Files API
- Creates dedicated branch for test artifacts
- Stores artifacts organized by test run
- Supports both text and binary artifacts with appropriate encoding

## Running Tests

The system includes comprehensive tests to verify functionality:

```bash
# Run the artifact handling tests
python run_test_artifact_discovery.py

# Run specific test functions
python -c "import anyio; from run_test_artifact_discovery import test_artifact_metadata_extraction; anyio.run(test_artifact_metadata_extraction)"
```

## Future Enhancements

- Add support for more CI providers (Circle CI, Travis CI, Azure DevOps)
- Implement artifact compression for more efficient storage
- Add advanced visualization for trend analysis results
- Enhance comparison with visual diff capabilities
- Implement notification system for significant metric changes
- Add support for artifact expiration policies
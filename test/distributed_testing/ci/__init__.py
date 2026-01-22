"""
CI Client modules for Distributed Testing Framework

This package provides clients for interacting with various CI/CD systems:
- GitHub Actions
- GitLab CI
- Jenkins
- Azure DevOps
- CircleCI
- Travis CI
- Bitbucket Pipelines
- TeamCity

These clients enable the distributed testing framework to report test results,
update build status, add PR comments, and upload artifacts to CI/CD systems.

The package features a standardized API interface to ensure consistent
behavior across different CI providers and make it easy to switch between them.
"""

# Import standardized interface
from .api_interface import (
    CIProviderInterface, 
    TestRunResult,
    CIProviderFactory
)

# Import implementation classes
from .github_client import GitHubClient
from .gitlab_client import GitLabClient
from .jenkins_client import JenkinsClient
from .azure_client import AzureDevOpsClient
from .circleci_client import CircleCIClient
from .bitbucket_client import BitbucketClient
from .teamcity_client import TeamCityClient
from .travis_client import TravisClient

# Import provider registration module
from .register_providers import register_all_providers

# Register all providers with factory
register_all_providers()

# Export key classes for easy import
__all__ = [
    "CIProviderInterface",
    "TestRunResult",
    "CIProviderFactory",
    "GitHubClient",
    "GitLabClient",
    "JenkinsClient",
    "AzureDevOpsClient",
    "CircleCIClient",
    "BitbucketClient",
    "TeamCityClient",
    "TravisClient",
    "register_all_providers"
]
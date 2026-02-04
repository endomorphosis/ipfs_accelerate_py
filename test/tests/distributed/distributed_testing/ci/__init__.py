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

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import standardized interface
from test.tests.distributed.distributed_testing.ci.api_interface import (
    CIProviderInterface, 
    TestRunResult,
    CIProviderFactory
)

# Import implementation classes (optional: may require extra deps like aiohttp)
try:
    from test.tests.distributed.distributed_testing.ci.github_client import GitHubClient
except Exception as e:  # pragma: no cover
    GitHubClient = None  # type: ignore[assignment]
    logger.debug("GitHubClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.gitlab_client import GitLabClient
except Exception as e:  # pragma: no cover
    GitLabClient = None  # type: ignore[assignment]
    logger.debug("GitLabClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.jenkins_client import JenkinsClient
except Exception as e:  # pragma: no cover
    JenkinsClient = None  # type: ignore[assignment]
    logger.debug("JenkinsClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.azure_client import AzureDevOpsClient
except Exception as e:  # pragma: no cover
    AzureDevOpsClient = None  # type: ignore[assignment]
    logger.debug("AzureDevOpsClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.circleci_client import CircleCIClient
except Exception as e:  # pragma: no cover
    CircleCIClient = None  # type: ignore[assignment]
    logger.debug("CircleCIClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.bitbucket_client import BitbucketClient
except Exception as e:  # pragma: no cover
    BitbucketClient = None  # type: ignore[assignment]
    logger.debug("BitbucketClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.teamcity_client import TeamCityClient
except Exception as e:  # pragma: no cover
    TeamCityClient = None  # type: ignore[assignment]
    logger.debug("TeamCityClient unavailable: %s", e)

try:
    from test.tests.distributed.distributed_testing.ci.travis_client import TravisClient
except Exception as e:  # pragma: no cover
    TravisClient = None  # type: ignore[assignment]
    logger.debug("TravisClient unavailable: %s", e)

# Import provider registration module (optional)
try:
    from test.tests.distributed.distributed_testing.ci.register_providers import register_all_providers

    # Register all providers with factory.
    # This may fail if optional client deps are missing, so keep it best-effort.
    try:
        register_all_providers()
    except Exception as e:  # pragma: no cover
        logger.debug("CI provider registration skipped: %s", e)
except Exception as e:  # pragma: no cover
    register_all_providers = None  # type: ignore[assignment]
    logger.debug("register_all_providers unavailable: %s", e)

# Export key classes for easy import
AzureClient = AzureDevOpsClient

__all__ = [
    "CIProviderInterface",
    "TestRunResult",
    "CIProviderFactory",
    "GitHubClient",
    "GitLabClient",
    "JenkinsClient",
    "AzureDevOpsClient",
    "AzureClient",
    "CircleCIClient",
    "BitbucketClient",
    "TeamCityClient",
    "TravisClient",
    "register_all_providers"
]
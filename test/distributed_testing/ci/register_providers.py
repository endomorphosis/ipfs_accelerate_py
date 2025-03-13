#!/usr/bin/env python3
"""
CI Provider Registration Module

This module registers all CI provider implementations with the CIProviderFactory.
This allows the factory to create the appropriate client based on the provider type.
"""

import logging

from distributed_testing.ci.api_interface import CIProviderFactory
from distributed_testing.ci.github_client import GitHubClient
from distributed_testing.ci.jenkins_client import JenkinsClient
from distributed_testing.ci.gitlab_client import GitLabClient
from distributed_testing.ci.azure_client import AzureDevOpsClient
from distributed_testing.ci.circleci_client import CircleCIClient
from distributed_testing.ci.bitbucket_client import BitbucketClient
from distributed_testing.ci.teamcity_client import TeamCityClient
from distributed_testing.ci.travis_client import TravisClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Register all CI providers with the factory
def register_all_providers():
    """Register all CI provider implementations with the factory."""
    try:
        # Register GitHub provider
        CIProviderFactory.register_provider("github", GitHubClient)
        logger.info("Registered GitHubClient as 'github' provider")
        
        # Register Jenkins provider
        CIProviderFactory.register_provider("jenkins", JenkinsClient)
        logger.info("Registered JenkinsClient as 'jenkins' provider")
        
        # Register GitLab provider
        CIProviderFactory.register_provider("gitlab", GitLabClient)
        logger.info("Registered GitLabClient as 'gitlab' provider")
        
        # Register Azure DevOps provider
        CIProviderFactory.register_provider("azure", AzureDevOpsClient)
        logger.info("Registered AzureDevOpsClient as 'azure' provider")
        
        # Register CircleCI provider
        CIProviderFactory.register_provider("circleci", CircleCIClient)
        logger.info("Registered CircleCIClient as 'circleci' provider")
        
        # Register Bitbucket Pipelines provider
        CIProviderFactory.register_provider("bitbucket", BitbucketClient)
        logger.info("Registered BitbucketClient as 'bitbucket' provider")
        
        # Register TeamCity provider
        CIProviderFactory.register_provider("teamcity", TeamCityClient)
        logger.info("Registered TeamCityClient as 'teamcity' provider")
        
        # Register Travis CI provider
        CIProviderFactory.register_provider("travis", TravisClient)
        logger.info("Registered TravisClient as 'travis' provider")
        
        logger.info("All CI providers registered successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error registering CI providers: {str(e)}")
        return False
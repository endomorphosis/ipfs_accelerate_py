#!/usr/bin/env python3
"""
CI Provider Registration Module

This module registers all CI provider implementations with the CIProviderFactory
and integrates with the artifact handling system. This allows for CI provider creation
and automatic registration with the artifact handling system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from distributed_testing.ci.api_interface import CIProviderFactory, CIProviderInterface
from distributed_testing.ci.github_client import GitHubClient
from distributed_testing.ci.jenkins_client import JenkinsClient
from distributed_testing.ci.gitlab_client import GitLabClient
from distributed_testing.ci.azure_client import AzureDevOpsClient
from distributed_testing.ci.circleci_client import CircleCIClient
from distributed_testing.ci.bitbucket_client import BitbucketClient
from distributed_testing.ci.teamcity_client import TeamCityClient
from distributed_testing.ci.travis_client import TravisClient
from distributed_testing.ci.artifact_handler import get_artifact_handler
from distributed_testing.ci.artifact_retriever import ArtifactRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global artifact retriever instance
_artifact_retriever = None

def get_artifact_retriever(cache_dir: str = "./artifact_cache") -> ArtifactRetriever:
    """
    Get the global artifact retriever instance.
    
    Args:
        cache_dir: Directory for artifact cache
    
    Returns:
        ArtifactRetriever instance
    """
    global _artifact_retriever
    
    if _artifact_retriever is None:
        _artifact_retriever = ArtifactRetriever(cache_dir=cache_dir)
    
    return _artifact_retriever

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

async def create_and_register_provider(
    provider_type: str,
    config: Dict[str, Any],
    artifact_handler_storage_dir: Optional[str] = None,
    artifact_retriever_cache_dir: Optional[str] = None
) -> Optional[CIProviderInterface]:
    """
    Create a CI provider and register it with artifact handling systems.
    
    This function creates a CI provider using the factory and automatically
    registers it with both the artifact handler and artifact retriever.
    
    Args:
        provider_type: Type of CI provider to create
        config: Configuration for the provider
        artifact_handler_storage_dir: Directory for artifact storage (optional)
        artifact_retriever_cache_dir: Directory for artifact cache (optional)
        
    Returns:
        Initialized CI provider or None if initialization failed
    """
    try:
        # Create the provider using the factory
        provider = await CIProviderFactory.create_provider(provider_type, config)
        
        if not provider:
            logger.error(f"Failed to create provider of type {provider_type}")
            return None
        
        # Register with artifact handler
        if artifact_handler_storage_dir:
            artifact_handler = get_artifact_handler(artifact_handler_storage_dir)
            artifact_handler.register_provider(provider_type, provider)
            logger.info(f"Registered {provider_type} with artifact handler")
        
        # Register with artifact retriever
        if artifact_retriever_cache_dir:
            artifact_retriever = get_artifact_retriever(artifact_retriever_cache_dir)
            artifact_retriever.register_provider(provider_type, provider)
            logger.info(f"Registered {provider_type} with artifact retriever")
        
        return provider
    
    except Exception as e:
        logger.error(f"Error creating and registering {provider_type} provider: {str(e)}")
        return None

async def initialize_artifact_systems(
    provider_configs: Dict[str, Dict[str, Any]],
    artifact_handler_storage_dir: str = "./artifacts",
    artifact_retriever_cache_dir: str = "./artifact_cache"
) -> Dict[str, CIProviderInterface]:
    """
    Initialize all artifact systems with providers.
    
    This function initializes the artifact handler and retriever with
    all the specified providers.
    
    Args:
        provider_configs: Dictionary mapping provider names to configurations
        artifact_handler_storage_dir: Directory for artifact storage
        artifact_retriever_cache_dir: Directory for artifact cache
        
    Returns:
        Dictionary mapping provider names to initialized providers
    """
    providers = {}
    
    # Register all provider types with the factory
    register_all_providers()
    
    # Create and register each provider
    for provider_type, config in provider_configs.items():
        provider = await create_and_register_provider(
            provider_type=provider_type,
            config=config,
            artifact_handler_storage_dir=artifact_handler_storage_dir,
            artifact_retriever_cache_dir=artifact_retriever_cache_dir
        )
        
        if provider:
            providers[provider_type] = provider
    
    logger.info(f"Initialized {len(providers)} providers with artifact systems")
    return providers
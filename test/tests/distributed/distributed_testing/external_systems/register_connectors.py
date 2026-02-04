#!/usr/bin/env python3
"""
External System Connectors Registration Module

This module imports and registers all available external system connectors
to ensure they are available through the ExternalSystemFactory.
"""

import logging
import importlib

from test.tests.distributed.distributed_testing.external_systems.api_interface import ExternalSystemFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of all connector modules
CONNECTOR_MODULES = [
    "distributed_testing.external_systems.jira_connector",
    "distributed_testing.external_systems.slack_connector",
    "distributed_testing.external_systems.testrail_connector",
    "distributed_testing.external_systems.prometheus_connector",
    "distributed_testing.external_systems.email_connector",
    "distributed_testing.external_systems.msteams_connector"
]

def register_all_connectors():
    """
    Dynamically import and register all external system connectors.
    
    This function should be called during framework initialization to ensure 
    all connectors are properly registered with the ExternalSystemFactory.
    
    Returns:
        Dict[str, type]: A dictionary of registered connector types.
    """
    registered_connectors = {}
    
    for module_name in CONNECTOR_MODULES:
        try:
            # Import the module
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported connector module: {module_name}")
            
            # Module is already imported, which triggers the registration
            # Get the connector name from the module name
            connector_name = module_name.split(".")[-1].replace("_connector", "")
            
            # Add to our registry for caller's reference
            connector_class = ExternalSystemFactory.get_connector_class(connector_name)
            if connector_class:
                registered_connectors[connector_name] = connector_class
                logger.info(f"Registered external system connector: {connector_name}")
            else:
                logger.warning(f"Failed to register connector: {connector_name} (not found in factory)")
                
        except ImportError as e:
            logger.warning(f"Failed to import connector module {module_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error registering connector {module_name}: {str(e)}")
    
    return registered_connectors

def get_available_connectors():
    """
    Get a list of available connector types.
    
    Returns:
        List[str]: List of available connector type identifiers.
    """
    return ExternalSystemFactory.get_available_connectors()

def create_connector(system_type, config):
    """
    Create a connector instance of the specified type with the given configuration.
    
    This is a convenience wrapper around ExternalSystemFactory.create_connector.
    
    Args:
        system_type (str): The type of external system connector to create
        config (Dict[str, Any]): Configuration dictionary for the connector
        
    Returns:
        ExternalSystemInterface: An initialized connector instance
        
    Raises:
        ValueError: If the system type is not registered
    """
    return ExternalSystemFactory.create_connector(system_type, config)

# Automatically register all connectors when the module is imported
registered_connectors = register_all_connectors()
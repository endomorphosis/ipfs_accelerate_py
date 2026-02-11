#!/usr/bin/env python3
"""
Standardized API Interface for External System Integrations

This module defines standardized interfaces for external system integrations,
ensuring consistent behavior across different systems (issue trackers, test management
systems, notification systems, etc.)
"""

import abc
from typing import Dict, List, Any, Optional, Union

class ExternalSystemInterface(abc.ABC):
    """
    Abstract base class defining the standard interface for all external system connectors.
    
    This interface ensures that all external system connectors implement a consistent set of methods,
    making it easier to switch between systems or create new implementations.
    """
    
    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the external system connector with configuration.
        
        Args:
            config: Configuration dictionary containing provider-specific settings
            
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abc.abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the external system.
        
        Returns:
            True if connection succeeded
        """
        pass
    
    @abc.abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to the external system.
        
        Returns:
            True if connected
        """
        pass
    
    @abc.abstractmethod
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on the external system.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        pass
    
    @abc.abstractmethod
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the external system for data.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        pass
    
    @abc.abstractmethod
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the external system.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        pass
    
    @abc.abstractmethod
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an item in the external system.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        pass
    
    @abc.abstractmethod
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete an item from the external system.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        pass
    
    @abc.abstractmethod
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get an item from the external system.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        pass
    
    @abc.abstractmethod
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about the external system.
        
        Returns:
            Dictionary with system information
        """
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the connection to the external system and clean up resources.
        
        Returns:
            None
        """
        pass


class ConnectorCapabilities:
    """
    Class representing the capabilities of an external system connector.
    
    This class provides a standardized way to describe the features supported
    by a particular external system connector.
    """
    
    def __init__(
        self,
        supports_create: bool = True,
        supports_update: bool = True,
        supports_delete: bool = True,
        supports_query: bool = True,
        supports_batch_operations: bool = False,
        supports_attachments: bool = False,
        supports_comments: bool = False,
        supports_custom_fields: bool = False,
        supports_relationships: bool = False,
        supports_history: bool = False,
        item_types: List[str] = None,
        query_operators: List[str] = None,
        max_batch_size: int = 0,
        rate_limit: int = 0,
        **additional_capabilities
    ):
        """
        Initialize connector capabilities.
        
        Args:
            supports_create: Whether the connector supports creating items
            supports_update: Whether the connector supports updating items
            supports_delete: Whether the connector supports deleting items
            supports_query: Whether the connector supports querying
            supports_batch_operations: Whether the connector supports batch operations
            supports_attachments: Whether the connector supports attachments
            supports_comments: Whether the connector supports comments
            supports_custom_fields: Whether the connector supports custom fields
            supports_relationships: Whether the connector supports relationships
            supports_history: Whether the connector supports history tracking
            item_types: List of supported item types
            query_operators: List of supported query operators
            max_batch_size: Maximum batch size for batch operations
            rate_limit: Rate limit for API calls (requests per minute)
            **additional_capabilities: Additional capability flags
        """
        self.supports_create = supports_create
        self.supports_update = supports_update
        self.supports_delete = supports_delete
        self.supports_query = supports_query
        self.supports_batch_operations = supports_batch_operations
        self.supports_attachments = supports_attachments
        self.supports_comments = supports_comments
        self.supports_custom_fields = supports_custom_fields
        self.supports_relationships = supports_relationships
        self.supports_history = supports_history
        self.item_types = item_types or []
        self.query_operators = query_operators or ["=", "!=", "<", "<=", ">", ">=", "IN", "NOT IN", "LIKE"]
        self.max_batch_size = max_batch_size
        self.rate_limit = rate_limit
        
        # Add additional capabilities
        for key, value in additional_capabilities.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "supports_create": self.supports_create,
            "supports_update": self.supports_update,
            "supports_delete": self.supports_delete,
            "supports_query": self.supports_query,
            "supports_batch_operations": self.supports_batch_operations,
            "supports_attachments": self.supports_attachments,
            "supports_comments": self.supports_comments,
            "supports_custom_fields": self.supports_custom_fields,
            "supports_relationships": self.supports_relationships,
            "supports_history": self.supports_history,
            "item_types": self.item_types,
            "query_operators": self.query_operators,
            "max_batch_size": self.max_batch_size,
            "rate_limit": self.rate_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectorCapabilities':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ConnectorCapabilities instance
        """
        return cls(**data)


class ExternalSystemResult:
    """
    Standardized representation of operation results from external systems.
    
    This class provides a common structure for operation results, making it easier to
    process and handle results regardless of the external system used.
    """
    
    def __init__(
        self,
        success: bool,
        operation: str,
        result_data: Optional[Any] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an operation result.
        
        Args:
            success: Whether the operation succeeded
            operation: The operation that was performed
            result_data: The result data from the operation
            error_message: Error message if the operation failed
            error_code: Error code if the operation failed
            metadata: Additional metadata
        """
        self.success = success
        self.operation = operation
        self.result_data = result_data
        self.error_message = error_message
        self.error_code = error_code
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "success": self.success,
            "operation": self.operation,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExternalSystemResult':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ExternalSystemResult instance
        """
        return cls(
            success=data.get("success", False),
            operation=data.get("operation", ""),
            result_data=data.get("result_data"),
            error_message=data.get("error_message"),
            error_code=data.get("error_code"),
            metadata=data.get("metadata", {})
        )


class ExternalSystemFactory:
    """
    Factory for creating external system connector instances.
    
    This factory makes it easy to create the appropriate connector based on
    the system type, abstracting away the implementation details.
    """
    
    _connectors = {}
    
    @classmethod
    def register_connector(cls, system_type: str, connector_class: type) -> None:
        """
        Register an external system connector class.
        
        Args:
            system_type: External system type identifier
            connector_class: Connector class
        """
        cls._connectors[system_type] = connector_class
    
    @classmethod
    async def create_connector(cls, system_type: str, config: Dict[str, Any]) -> ExternalSystemInterface:
        """
        Create an external system connector instance.
        
        Args:
            system_type: External system type identifier
            config: Configuration for the connector
            
        Returns:
            External system connector instance
        
        Raises:
            ValueError: If system type is not registered
        """
        if system_type not in cls._connectors:
            raise ValueError(f"Unknown external system type: {system_type}")
        
        connector_class = cls._connectors[system_type]
        connector = connector_class()
        await connector.initialize(config)
        
        return connector
    
    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """
        Get list of available connector types.
        
        Returns:
            List of connector type identifiers
        """
        return list(cls._connectors.keys())
    
    @classmethod
    def get_connector_class(cls, system_type: str) -> Optional[type]:
        """
        Get the connector class for a system type.
        
        Args:
            system_type: External system type identifier
            
        Returns:
            Connector class or None if not found
        """
        return cls._connectors.get(system_type)
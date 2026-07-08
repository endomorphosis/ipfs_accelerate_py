"""
Unified Tool Registry for MCP Server

This module provides a unified interface for registering all MCP tools,
consolidating both legacy decorator-based tools and new kit-wrapper tools
into a single registry that can be discovered by the JavaScript SDK.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a registered MCP tool."""
    name: str
    description: str
    category: str
    function: Callable
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'examples': self.examples,
            'tags': self.tags,
            'status': 'active'
        }


class UnifiedToolRegistry:
    """
    Unified registry for all MCP tools.
    
    This class consolidates tool registration from multiple sources:
    - Kit-based tools (GitHub, Docker, Hardware, Runner, IPFS, Network)
    - Legacy decorator-based tools (Inference, Endpoints, Status, Workflows)
    - Model management tools
    
    It provides a single interface for tool discovery and execution.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str = "",
        category: str = "Other",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool in the unified registry.
        
        Args:
            name: Unique tool name
            function: Callable function for the tool
            description: Tool description
            category: Tool category (GitHub, Docker, Hardware, etc.)
            input_schema: JSON schema for input parameters
            output_schema: JSON schema for output
            examples: List of usage examples
            tags: Additional tags for categorization
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered. Overwriting...")
        
        # Auto-categorize if not specified
        if category == "Other":
            category = self._auto_categorize(name)
        
        # Create default schema if not provided
        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            function=function,
            input_schema=input_schema,
            output_schema=output_schema,
            examples=examples or [],
            tags=tags or []
        )
        
        self.tools[name] = metadata
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        logger.debug(f"Registered tool: {name} (category: {category})")
    
    def _auto_categorize(self, tool_name: str) -> str:
        """Auto-categorize tool by name prefix."""
        if tool_name.startswith('github_'):
            return 'GitHub'
        elif tool_name.startswith('docker_'):
            return 'Docker'
        elif tool_name.startswith('hardware_'):
            return 'Hardware'
        elif tool_name.startswith('runner_'):
            return 'Runner'
        elif tool_name.startswith('ipfs_files_'):
            return 'IPFS Files'
        elif tool_name.startswith('network_'):
            return 'Network'
        elif 'model' in tool_name.lower() or tool_name.startswith('search_') or tool_name.startswith('recommend_'):
            return 'Models'
        elif 'inference' in tool_name.lower() or 'generate' in tool_name.lower():
            return 'Inference'
        elif 'workflow' in tool_name.lower():
            return 'Workflows'
        elif 'dashboard' in tool_name.lower():
            return 'Dashboard'
        elif 'endpoint' in tool_name.lower():
            return 'Endpoints'
        elif 'status' in tool_name.lower() or 'health' in tool_name.lower():
            return 'Status'
        else:
            return 'Other'
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[ToolMetadata]:
        """List all registered tools."""
        return list(self.tools.values())
    
    def list_tool_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get tools organized by category."""
        return self._categories.copy()
    
    def get_tools_by_category(self, category: str) -> List[ToolMetadata]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            KeyError: If tool not found
            Exception: If tool execution fails
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        
        tool = self.tools[name]
        
        try:
            return tool.function(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            raise
    
    def to_api_response(self) -> Dict[str, Any]:
        """
        Convert registry to API response format.
        
        Returns a dictionary suitable for the /api/mcp/tools endpoint.
        """
        tools_list = [tool.to_dict() for tool in self.tools.values()]
        
        categories = {}
        for category, tool_names in self._categories.items():
            categories[category] = [
                self.tools[name].to_dict() 
                for name in tool_names 
                if name in self.tools
            ]
        
        return {
            'tools': tools_list,
            'categories': categories,
            'total': len(tools_list),
            'category_count': len(categories)
        }


# Global registry instance
_global_registry: Optional[UnifiedToolRegistry] = None


def get_global_registry() -> UnifiedToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = UnifiedToolRegistry()
    return _global_registry


def register_tool_with_mcp(mcp: Any, registry: UnifiedToolRegistry, tool_name: str) -> None:
    """
    Register a tool from the unified registry with an MCP server instance.
    
    Args:
        mcp: MCP server instance (FastMCP or StandaloneMCP)
        registry: UnifiedToolRegistry instance
        tool_name: Name of the tool to register
    """
    tool = registry.get_tool(tool_name)
    if not tool:
        logger.warning(f"Tool '{tool_name}' not found in registry")
        return
    
    # Register with MCP server
    try:
        mcp.register_tool(
            name=tool.name,
            function=tool.function,
            description=tool.description,
            input_schema=tool.input_schema
        )
        logger.debug(f"Registered tool '{tool_name}' with MCP server")
    except Exception as e:
        logger.error(f"Failed to register tool '{tool_name}' with MCP: {e}")


def register_all_tools_with_mcp(mcp: Any, registry: Optional[UnifiedToolRegistry] = None) -> None:
    """
    Register all tools from the unified registry with an MCP server.
    
    Args:
        mcp: MCP server instance
        registry: Optional registry (uses global if not provided)
    """
    if registry is None:
        registry = get_global_registry()
    
    for tool_name in registry.list_tool_names():
        register_tool_with_mcp(mcp, registry, tool_name)
    
    logger.info(f"Registered {len(registry.list_tool_names())} tools with MCP server")

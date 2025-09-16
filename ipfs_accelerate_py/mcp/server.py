"""
IPFS Accelerate MCP Server

This module provides the MCP server for IPFS Accelerate.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional, List, Union, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.server")

class StandaloneMCP:
    """
    Standalone MCP Implementation
    
    This class provides a standalone implementation of the Model Context Protocol
    when FastMCP is not available.
    """
    
    def __init__(self, name: str):
        """
        Initialize the Standalone MCP
        
        Args:
            name: Name of the server
        """
        self.name = name
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        
        logger.info(f"Using standalone MCP implementation: {name}")
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        input_schema: Dict[str, Any]
    ) -> None:
        """
        Register a tool with the MCP server
        
        Args:
            name: Name of the tool
            function: Function to be called when the tool is used
            description: Description of the tool
            input_schema: JSON schema for the tool's input
        """
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema
        }
        
        logger.debug(f"Registered tool: {name}")
    
    def register_resource(
        self,
        uri: str,
        function: Callable,
        description: str
    ) -> None:
        """
        Register a resource with the MCP server
        
        Args:
            uri: URI of the resource
            function: Function to be called when the resource is accessed
            description: Description of the resource
        """
        self.resources[uri] = {
            "function": function,
            "description": description
        }
        
        logger.debug(f"Registered resource: {uri}")
    
    def register_prompt(
        self,
        name: str,
        template: str,
        description: str,
        input_schema: Dict[str, Any]
    ) -> None:
        """
        Register a prompt with the MCP server
        
        Args:
            name: Name of the prompt
            template: Template for the prompt
            description: Description of the prompt
            input_schema: JSON schema for the prompt's input
        """
        self.prompts[name] = {
            "template": template,
            "description": description,
            "input_schema": input_schema
        }
        
        logger.debug(f"Registered prompt: {name}")
    
    def create_fastapi_app(
        self,
        title: str,
        description: str,
        version: str,
        docs_url: str,
        redoc_url: str,
        mount_path: str
    ) -> Any:
        """
        Create a FastAPI app for the MCP server
        
        Args:
            title: Title of the API
            description: Description of the API
            version: Version of the API
            docs_url: URL for the API documentation
            redoc_url: URL for the API redoc documentation
            mount_path: Path to mount the API at
            
        Returns:
            FastAPI app
        """
        logger.debug(f"Creating FastAPI app for standalone MCP: {title}")
        
        try:
            from fastapi import FastAPI, APIRouter, Body, Depends
            from pydantic import BaseModel, Field, create_model
            from functools import partial
            
            app = FastAPI(
                title=title,
                description=description,
                version=version,
                docs_url=docs_url,
                redoc_url=redoc_url
            )
            
            router = APIRouter()
            
            # Create a single endpoint for all tools that dynamically dispatches based on the tool name
            from fastapi import HTTPException, Path, Body
            
            @router.post("/tool/{tool_name}", summary="Generic tool endpoint")
            async def generic_tool_endpoint(tool_name: str = Path(..., description="The name of the tool to execute"), 
                                           data: dict = Body({}, description="Tool input data")):
                if tool_name not in self.tools:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                try:
                    # Get the tool function
                    tool = self.tools[tool_name]
                    tool_function = tool["function"]
                    
                    # Execute the tool function
                    result = tool_function(**data)
                    return result
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Log all registered tools
            for name, tool in self.tools.items():
                logger.debug(f"Registered tool: {name} (accessible at POST /tool/{name})")
            
            # Create a single endpoint for all resources that dynamically dispatches based on the resource URI
            @router.get("/resource/{resource_uri:path}", summary="Generic resource endpoint")
            async def generic_resource_endpoint(resource_uri: str = Path(..., description="The URI of the resource to access")):
                if resource_uri not in self.resources:
                    raise HTTPException(status_code=404, detail=f"Resource '{resource_uri}' not found")
                
                try:
                    # Get the resource function
                    resource = self.resources[resource_uri]
                    resource_function = resource["function"]
                    
                    # Execute the resource function
                    result = resource_function()
                    return result
                except Exception as e:
                    logger.error(f"Error accessing resource {resource_uri}: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Log all registered resources
            for uri, resource in self.resources.items():
                logger.debug(f"Registered resource: {uri} (accessible at GET /resource/{uri})")
            
            # Mount the router
            app.include_router(router, prefix=mount_path)
            
            # Debug: Print all registered routes
            logger.debug(f"FastAPI app created for standalone MCP with routes:")
            for route in app.routes:
                logger.debug(f"Route: {route.path} {route.methods if hasattr(route, 'methods') else ''}")
            
            return app
        
        except ImportError:
            logger.error("Failed to create FastAPI app: FastAPI not installed")
            raise
        
        except Exception as e:
            logger.error(f"Failed to create FastAPI app: {e}")
            raise
    
    def _create_pydantic_model(self, name: str, schema: Dict[str, Any]) -> Any:
        """
        Create a Pydantic model from a JSON schema
        
        Args:
            name: Name of the model
            schema: JSON schema for the model
            
        Returns:
            Pydantic model
        """
        from pydantic import create_model, Field
        
        if "properties" not in schema:
            return create_model(name, __base__=BaseModel)
        
        required = schema.get("required", [])
        fields = {}
        
        for prop_name, prop_schema in schema["properties"].items():
            field_type = self._get_field_type(prop_schema)
            default = None if prop_name in required else prop_schema.get("default", ...)
            description = prop_schema.get("description", "")
            
            fields[prop_name] = (field_type, Field(default=default, description=description))
        
        return create_model(name, **fields, __base__=BaseModel)
    
    def _get_field_type(self, schema: Dict[str, Any]) -> Any:
        """
        Get the Python type for a JSON schema type
        
        Args:
            schema: JSON schema
            
        Returns:
            Python type
        """
        if "type" not in schema:
            return Any
        
        schema_type = schema["type"]
        
        if schema_type == "string":
            return str
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return float
        elif schema_type == "boolean":
            return bool
        elif schema_type == "array":
            return List[self._get_field_type(schema.get("items", {}))]
        elif schema_type == "object":
            return Dict[str, Any]
        else:
            return Any

class IPFSAccelerateMCPServer:
    """
    IPFS Accelerate MCP Server
    
    This class provides a Model Context Protocol server for IPFS Accelerate.
    """
    
    def __init__(
        self,
        name: str = "ipfs-accelerate",
        host: str = "localhost",
        port: int = 8000,
        mount_path: str = "/mcp",
        debug: bool = False
    ):
        """
        Initialize the IPFS Accelerate MCP Server
        
        Args:
            name: Name of the server
            host: Host to bind the server to
            port: Port to bind the server to
            mount_path: Path to mount the server at
            debug: Enable debug logging
        """
        self.name = name
        self.host = host
        self.port = port
        self.mount_path = mount_path
        self.debug = debug
        self._using_fastmcp = False
        
        # Configure logging
        if debug:
            logging.getLogger("ipfs_accelerate_mcp").setLevel(logging.DEBUG)
        
        # Set up server attributes
        self.mcp = None
        self.fastapi_app = None
        self.server_url = f"http://{host}:{port}{mount_path}"
        
        logger.debug(f"Initialized IPFS Accelerate MCP Server: {self.server_url}")
    
    def setup(self) -> None:
        """
        Set up the MCP server
        
        This function sets up the MCP server with all tools and resources.
        """
        logger.info(f"Setting up IPFS Accelerate MCP Server: {self.name}")
        
        try:
            # Try to import FastMCP
            try:
                from fastmcp import FastMCP
                
                # Create FastMCP instance
                self.mcp = FastMCP(name=self.name)
                
                # Create FastAPI app
                self.fastapi_app = self.mcp.create_fastapi_app(
                    title="IPFS Accelerate MCP API",
                    description="API for the IPFS Accelerate MCP Server",
                    version="0.1.0",
                    docs_url="/docs",
                    redoc_url="/redoc",
                    mount_path=self.mount_path
                )
                
                # Use FastMCP implementation
                logger.info("Using FastMCP implementation")
                self._using_fastmcp = True
            
            except ImportError:
                # Use standalone implementation
                logger.warning("FastMCP not available, using standalone implementation")
                self.mcp = StandaloneMCP(name=self.name)
                
                # Create FastAPI app
                self.fastapi_app = self.mcp.create_fastapi_app(
                    title="IPFS Accelerate MCP API",
                    description="API for the IPFS Accelerate MCP Server",
                    version="0.1.0",
                    docs_url="/docs",
                    redoc_url="/redoc",
                    mount_path=self.mount_path
                )
                
                self._using_fastmcp = False
            
            # Enable CORS for external API consumers (configurable via MCP_CORS_ORIGINS)
            try:
                from fastapi.middleware.cors import CORSMiddleware

                allowed = os.getenv("MCP_CORS_ORIGINS", "*")
                allow_origins = [o.strip() for o in allowed.split(",") if o.strip()] or ["*"]

                self.fastapi_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=allow_origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
                logger.info(f"CORS enabled for MCP API (origins: {allow_origins})")
            except Exception as e:
                logger.warning(f"CORS not enabled (missing dependency or error): {e}")

            # Register tools
            self._register_tools()
            
            # Register resources
            self._register_resources()
            
            # Register prompts
            self._register_prompts()
            
            logger.info(f"IPFS Accelerate MCP Server set up: {self.server_url}")
        
        except Exception as e:
            logger.error(f"Error setting up MCP server: {e}")
            raise
    
    def run(self) -> None:
        """
        Run the MCP server
        
        This function runs the MCP server using uvicorn.
        """
        if self.fastapi_app is None:
            self.setup()
        
        logger.info(f"Running IPFS Accelerate MCP Server at {self.server_url}")
        
        try:
            import uvicorn
            
            # Run the server
            uvicorn.run(
                self.fastapi_app,
                host=self.host,
                port=self.port,
                log_level="debug" if self.debug else "info"
            )
        
        except ImportError:
            logger.error("Failed to import uvicorn. Please install with 'pip install uvicorn'.")
            raise
        
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise
    
    def _register_tools(self) -> None:
        """
        Register tools with the MCP server
        
        This function registers all tools with the MCP server.
        """
        logger.debug("Registering tools with MCP server")
        
        try:
            # Import tools
            from ipfs_accelerate_py.mcp.tools import register_all_tools
            
            # Register tools
            register_all_tools(self.mcp)
            
            logger.debug("Tools registered with MCP server")
        
        except Exception as e:
            logger.error(f"Error registering tools with MCP server: {e}")
            raise
    
    def _register_resources(self) -> None:
        """
        Register resources with the MCP server
        
        This function registers all resources with the MCP server.
        """
        logger.debug("Registering resources with MCP server")
        
        try:
            # Import resources
            from ipfs_accelerate_py.mcp.resources import register_all_resources
            
            # Register resources
            register_all_resources(self.mcp)
            
            logger.debug("Resources registered with MCP server")
        
        except Exception as e:
            logger.error(f"Error registering resources with MCP server: {e}")
            raise
    
    def _register_prompts(self) -> None:
        """
        Register prompts with the MCP server
        
        This function registers all prompts with the MCP server.
        """
        logger.debug("Registering prompts with MCP server")
        
        try:
            # Define default help prompt
            self.mcp.register_prompt(
                name="ipfs_help",
                template="""
                # IPFS Accelerate Help
                
                IPFS Accelerate provides tools and resources for working with IPFS and accelerating AI models.
                
                ## Available Tools
                
                {% for tool_name, tool in server.tools.items() %}
                - **{{ tool_name }}**: {{ tool.description }}
                {% endfor %}
                
                ## Available Resources
                
                {% for uri, resource in server.resources.items() %}
                - **{{ uri }}**: {{ resource.description }}
                {% endfor %}
                """,
                description="Get help with IPFS Accelerate",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
            
            logger.debug("Prompts registered with MCP server")
        
        except Exception as e:
            logger.error(f"Error registering prompts with MCP server: {e}")
            # Don't raise here, as prompts are optional
            pass

def main() -> None:
    """
    Main entry point for the IPFS Accelerate MCP Server
    
    This function parses command-line arguments and runs the server.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Server")
    
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the server")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--mount-path", default="/mcp", help="Path to mount the server at")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create server
    server = IPFSAccelerateMCPServer(
        name=args.name,
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        debug=args.debug
    )
    
    # Run server
    try:
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
    
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

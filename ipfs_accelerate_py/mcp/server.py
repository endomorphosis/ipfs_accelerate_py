"""
IPFS Accelerate MCP Server

This module provides the MCP server for IPFS Accelerate.
"""

from __future__ import annotations

import os
import sys
import json
import importlib
import importlib.metadata
import importlib.util
import hashlib
import logging
import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.server")

FASTMCP_VERSION = "2.14.7"
FASTMCP_REQUIREMENT = (
    f"fastmcp=={FASTMCP_VERSION}; python_version >= '3.10'"
)
FASTAPI_REQUIREMENT = "fastapi>=0.110.0,<1.0.0"
UVICORN_LEGACY_REQUIREMENT = (
    "uvicorn>=0.27.0,<0.35.0; python_version < '3.10'"
)
UVICORN_REQUIREMENT = (
    "uvicorn>=0.35.0,<1.0.0; python_version >= '3.10'"
)
WEBSOCKETS_LEGACY_REQUIREMENT = (
    "websockets==10.4; python_version < '3.10'"
)
WEBSOCKETS_REQUIREMENT = "websockets>=15.0.1; python_version >= '3.10'"


def _exception_receipt(error: BaseException) -> str:
    """Return a secret-safe diagnostic receipt for an exception."""

    try:
        detail = str(error)
    except Exception:
        detail = "<unprintable>"
    payload = (
        f"{type(error).__module__}.{type(error).__qualname__}:{detail}"
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_exception_summary(context: str, error: BaseException) -> str:
    """Describe an exception without exposing its potentially secret text."""

    error_type = type(error).__name__
    if not error_type.isidentifier():
        error_type = "Error"
    return f"{context} ({error_type}; receipt={_exception_receipt(error)})"


def _ensure_mcp_runtime_dependencies() -> Dict[str, str]:
    """Lazily ensure optional server dependencies on explicit first use."""

    from ..utils.auto_install import ensure_distributions, ensure_packages

    packages = {
        "fastapi": FASTAPI_REQUIREMENT,
        "uvicorn": (
            UVICORN_REQUIREMENT
            if sys.version_info >= (3, 10)
            else UVICORN_LEGACY_REQUIREMENT
        ),
        "websockets": (
            WEBSOCKETS_REQUIREMENT
            if sys.version_info >= (3, 10)
            else WEBSOCKETS_LEGACY_REQUIREMENT
        ),
    }
    if sys.version_info >= (3, 10):
        fastmcp_status = ensure_distributions(
            {"fastmcp": FASTMCP_REQUIREMENT}
        )
    else:
        fastmcp_status = {}
    return {**ensure_packages(packages), **fastmcp_status}


def _import_fastmcp_v2() -> Tuple[Optional[Any], str]:
    """Load the supported FastMCP distribution without mutating import state."""

    if sys.version_info < (3, 10):
        return None, "FastMCP 2.14.7 requires Python 3.10 or newer"

    try:
        distribution = importlib.metadata.distribution("fastmcp")
    except importlib.metadata.PackageNotFoundError:
        return None, "FastMCP is not installed"
    except Exception as error:
        return None, _safe_exception_summary(
            "FastMCP metadata could not be read",
            error,
        )

    installed_version = str(getattr(distribution, "version", "") or "")

    if installed_version != FASTMCP_VERSION:
        return (
            None,
            f"unsupported FastMCP version {installed_version}; "
            f"required version is {FASTMCP_VERSION}",
        )

    distribution_files = getattr(distribution, "files", None)
    if not distribution_files:
        return None, "FastMCP distribution does not expose an auditable file list"
    try:
        owned_files = {
            os.path.realpath(os.fspath(distribution.locate_file(path)))
            for path in distribution_files
        }
    except Exception as error:
        return None, _safe_exception_summary(
            "FastMCP distribution files could not be audited",
            error,
        )

    try:
        import_spec = importlib.util.find_spec("fastmcp")
    except Exception as error:
        return None, _safe_exception_summary(
            "FastMCP import origin could not be resolved",
            error,
        )
    pre_import_origin = getattr(import_spec, "origin", None)
    if not pre_import_origin:
        return None, "FastMCP import origin is unavailable"
    resolved_pre_import_origin = os.path.realpath(os.fspath(pre_import_origin))
    if resolved_pre_import_origin not in owned_files:
        return None, "FastMCP import origin is not owned by the audited distribution"

    try:
        module = importlib.import_module("fastmcp")
    except Exception as error:
        return None, _safe_exception_summary(
            f"FastMCP {FASTMCP_VERSION} could not be imported",
            error,
        )

    module_version = str(getattr(module, "__version__", "") or "")
    if module_version != FASTMCP_VERSION:
        return (
            None,
            "FastMCP module/distribution version mismatch "
            f"({module_version or 'missing'} != {FASTMCP_VERSION})",
        )

    post_import_spec = getattr(module, "__spec__", None)
    post_import_origin = getattr(post_import_spec, "origin", None)
    module_file = getattr(module, "__file__", None)
    if not post_import_origin or not module_file:
        return None, "FastMCP module origin is unavailable after import"
    resolved_post_import_origin = os.path.realpath(os.fspath(post_import_origin))
    resolved_module_file = os.path.realpath(os.fspath(module_file))
    if (
        resolved_post_import_origin != resolved_pre_import_origin
        or resolved_module_file != resolved_pre_import_origin
        or resolved_post_import_origin not in owned_files
    ):
        return None, "FastMCP module origin changed during import"
    return module, ""


def _import_fastmcp_v2_with_repair() -> Tuple[Optional[Any], str]:
    """Retry an audited import after an explicitly authorized dependency repair."""

    module, reason = _import_fastmcp_v2()
    if module is not None or not reason.startswith(
        f"FastMCP {FASTMCP_VERSION} could not be imported"
    ):
        return module, reason

    from ..utils.auto_install import ensure_distributions

    repair_status = ensure_distributions(
        {"fastmcp": FASTMCP_REQUIREMENT},
        force=True,
    ).get("fastmcp")
    if repair_status != "installed":
        return module, reason
    return _import_fastmcp_v2()


def _repair_fastmcp_v2_runtime() -> Tuple[Optional[Any], str]:
    """Make one authorized exact-pin repair attempt after a wiring failure."""

    from ..utils.auto_install import ensure_distributions

    repair_status = ensure_distributions(
        {"fastmcp": FASTMCP_REQUIREMENT},
        force=True,
    ).get("fastmcp")
    if repair_status != "installed":
        return None, "FastMCP runtime repair was not authorized or did not succeed"
    return _import_fastmcp_v2()

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
        self._error_handler = None
        
        logger.info(f"Using standalone MCP implementation: {name}")
        
        # Initialize error handler if available
        self._init_error_handler()
    
    def _init_error_handler(self):
        """Initialize the error handler for auto-healing."""
        try:
            import os
            from ipfs_accelerate_py.error_handler import CLIErrorHandler
            
            # Check if auto-healing is enabled
            enable_auto_issue = os.environ.get('IPFS_AUTO_ISSUE', '').lower() in ('1', 'true', 'yes')
            enable_auto_pr = os.environ.get('IPFS_AUTO_PR', '').lower() in ('1', 'true', 'yes')
            enable_auto_heal = os.environ.get('IPFS_AUTO_HEAL', '').lower() in ('1', 'true', 'yes')
            repo = os.environ.get('IPFS_REPO', 'endomorphosis/ipfs_accelerate_py')
            
            if enable_auto_issue or enable_auto_pr or enable_auto_heal:
                self._error_handler = CLIErrorHandler(
                    repo=repo,
                    enable_auto_issue=enable_auto_issue,
                    enable_auto_pr=enable_auto_pr,
                    enable_auto_heal=enable_auto_heal,
                    log_context_lines=50
                )
                logger.info(f"MCP auto-healing enabled: issue={enable_auto_issue}, pr={enable_auto_pr}, heal={enable_auto_heal}")
        except ImportError as e:
            logger.debug(f"Error handler not available for MCP: {e}")
        except Exception as e:
            logger.debug(f"Failed to initialize MCP error handler: {e}")
    
    def _report_tool_error(self, tool_name: str, exception: Exception, params: dict):
        """Report a tool execution error to the auto-healing system."""
        if not self._error_handler:
            return
        
        try:
            context = {
                'mcp_server': self.name,
                'tool_name': tool_name,
                'tool_params': str(params),
                'error_source': 'mcp_tool'
            }
            
            # Capture the error
            self._error_handler.capture_error(exception, context=context)
            
            # Create issue if enabled
            if self._error_handler.enable_auto_issue:
                self._error_handler.create_issue_from_error(exception, context=context)
        except Exception as e:
            logger.debug(f"Failed to report tool error: {e}")
    
    def _report_resource_error(self, resource_uri: str, exception: Exception):
        """Report a resource access error to the auto-healing system."""
        if not self._error_handler:
            return
        
        try:
            context = {
                'mcp_server': self.name,
                'resource_uri': resource_uri,
                'error_source': 'mcp_resource'
            }
            
            # Capture the error
            self._error_handler.capture_error(exception, context=context)
            
            # Create issue if enabled
            if self._error_handler.enable_auto_issue:
                self._error_handler.create_issue_from_error(exception, context=context)
        except Exception as e:
            logger.debug(f"Failed to report resource error: {e}")
    
    def _report_client_error(self, error_data: dict):
        """
        Report a client-side (JavaScript SDK) error to the auto-healing system.
        
        Args:
            error_data: Dictionary containing error details from the client
        """
        if not self._error_handler:
            logger.debug("Error handler not available, skipping client error report")
            return
        
        try:
            # Create a synthetic exception from the client error data
            error_type = error_data.get('error_type', 'ClientError')
            error_message = error_data.get('error_message', 'Unknown client error')
            stack_trace = error_data.get('stack_trace', '')
            client_context = error_data.get('context', {})
            
            # Build context
            context = {
                'mcp_server': self.name,
                'error_source': 'mcp_javascript_sdk',
                'client_context': client_context,
                'client_stack_trace': stack_trace,
            }
            
            # Create a RuntimeError with the client's error message
            exception = RuntimeError(f"[JavaScript SDK] {error_type}: {error_message}")
            
            # Capture the error
            self._error_handler.capture_error(exception, context=context)
            
            # Create issue if enabled
            if self._error_handler.enable_auto_issue:
                self._error_handler.create_issue_from_error(exception, context=context)
            
            logger.info(f"Reported client error to auto-healing system: {error_type}")
        except Exception as e:
            logger.error(f"Failed to report client error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def register_tool(
        self,
        name: str,
        function: Callable | None = None,
        description: str = "",
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        func: Callable | None = None,
        category: str | None = None,
        runtime: str | None = None,
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        """
        Register a tool with the MCP server
        
        Args:
            name: Name of the tool
            function: Function to be called when the tool is used
            description: Description of the tool
            input_schema: JSON schema for the tool's input
        """
        tool_function = function or func
        if tool_function is None:
            raise ValueError(f"Tool {name!r} must provide a function")

        ctx = str(execution_context or runtime or "").strip().lower()
        if ctx not in {"", "server", "worker"}:
            ctx = ""

        from .fastmcp_compat import canonical_function_input_schema

        canonical_schema = canonical_function_input_schema(tool_function)
        published_schema = (
            canonical_schema
            if canonical_schema is not None
            else input_schema
            or {"type": "object", "properties": {}, "required": []}
        )

        self.tools[name] = {
            "function": tool_function,
            "description": description,
            "input_schema": published_schema,
            "category": category,
            "runtime": runtime,
            # Tool routing metadata used by p2p call_tool.
            # - 'server': safe/control-plane, can run inline in the MCP+p2p process
            # - 'worker': must run in thin executor workers
            "execution_context": ctx or None,
            "tags": [str(t) for t in (tags or []) if str(t).strip()],
            "metadata": metadata,
        }
        
        logger.debug(f"Registered tool: {name}")
    
    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        tags: list[str] | None = None,
    ):
        """
        Decorator for registering tools (FastMCP compatibility)
        
        This decorator allows tools to be registered using the @mcp.tool() syntax
        compatible with FastMCP, but internally uses register_tool.
        
        Returns:
            Decorator function
        """
        def decorator(func):
            # Extract function name and docstring
            tool_name = str(name or func.__name__)
            tool_desc = description if description is not None else (func.__doc__ or "No description")
            
            # Register the tool
            self.register_tool(
                name=tool_name,
                function=func,
                description=str(tool_desc),
                input_schema=input_schema,
                execution_context=execution_context,
                tags=tags,
            )
            
            return func
        
        return decorator
    
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

    def access_resource(self, uri: str, **kwargs) -> Any:
        """
        Access a registered resource by URI.

        This helper mirrors FastMCP's access_resource to provide a common API
        for tools that query resources. It calls the registered function with
        any provided keyword arguments.
        """
        resource = self.resources.get(uri)
        if not resource:
            return None
        try:
            func = resource.get("function")
            if callable(func):
                return func(**kwargs) if kwargs else func()
        except Exception as e:
            logger.error(f"Error accessing resource {uri}: {e}")
            logger.debug(traceback.format_exc())
        return None
    
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

            def _server_info() -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "description": description,
                    "version": version,
                }

            def _health_info() -> Dict[str, Any]:
                return {
                    "status": "ok",
                    "tools_count": len(self.tools or {}),
                    "resources_count": len(self.resources or {}),
                    "prompts_count": len(self.prompts or {}),
                }

            @router.get("/", summary="Server info")
            async def server_info_endpoint():
                return _server_info()

            @router.get("/health", summary="Health check")
            async def health_endpoint():
                return _health_info()

            @router.get("/tools", summary="List tools")
            async def list_tools_endpoint():
                return sorted(list((self.tools or {}).keys()))

            @router.get("/resources", summary="List resources")
            async def list_resources_endpoint():
                return sorted(list((self.resources or {}).keys()))
            
            # Create a single endpoint for all tools that dynamically dispatches based on the tool name
            from fastapi import HTTPException, Path, Body
            
            async def _execute_tool(tool_name: str, data: dict):
                if tool_name not in (self.tools or {}):
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

                try:
                    tool = self.tools[tool_name]
                    tool_function = tool["function"]
                    result = tool_function(**(data or {}))
                    try:
                        import inspect

                        if inspect.isawaitable(result):
                            result = await result
                    except Exception:
                        # Best-effort: if inspect isn't available or something
                        # odd happens, just return the raw result.
                        pass
                    return result
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")

                    try:
                        self._report_tool_error(tool_name, e, data or {})
                    except Exception as report_error:
                        logger.debug(f"Failed to report error to auto-healing system: {report_error}")

                    raise HTTPException(status_code=500, detail=str(e))

            # Back-compat endpoint (singular)
            @router.post("/tool/{tool_name}", summary="Generic tool endpoint")
            async def generic_tool_endpoint(
                tool_name: str = Path(..., description="The name of the tool to execute"),
                data: dict = Body({}, description="Tool input data"),
            ):
                return await _execute_tool(tool_name, data)

            # Preferred endpoint (plural)
            @router.post("/tools/{tool_name}", summary="Tool endpoint")
            async def tool_endpoint(
                tool_name: str = Path(..., description="The name of the tool to execute"),
                data: dict = Body({}, description="Tool input data"),
            ):
                return await _execute_tool(tool_name, data)
            
            # Log all registered tools
            for name, tool in self.tools.items():
                logger.debug(f"Registered tool: {name} (accessible at POST /tool/{name})")
            
            # Create a single endpoint for all resources that dynamically dispatches based on the resource URI
            async def _get_resource(resource_uri: str):
                if resource_uri not in (self.resources or {}):
                    raise HTTPException(status_code=404, detail=f"Resource '{resource_uri}' not found")

                try:
                    resource = self.resources[resource_uri]
                    resource_function = resource["function"]
                    result = resource_function()
                    return result
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error accessing resource {resource_uri}: {e}")

                    try:
                        self._report_resource_error(resource_uri, e)
                    except Exception as report_error:
                        logger.debug(f"Failed to report error to auto-healing system: {report_error}")

                    raise HTTPException(status_code=500, detail=str(e))

            # Back-compat endpoint (singular)
            @router.get("/resource/{resource_uri:path}", summary="Generic resource endpoint")
            async def generic_resource_endpoint(
                resource_uri: str = Path(..., description="The URI of the resource to access"),
            ):
                return await _get_resource(resource_uri)

            # Preferred endpoint (plural)
            @router.get("/resources/{resource_uri:path}", summary="Resource endpoint")
            async def resource_endpoint(
                resource_uri: str = Path(..., description="The URI of the resource to access"),
            ):
                return await _get_resource(resource_uri)
            
            # Log all registered resources
            for uri, resource in self.resources.items():
                logger.debug(f"Registered resource: {uri} (accessible at GET /resource/{uri})")
            
            # Add error reporting endpoint for JavaScript SDK
            @router.post("/report-error", summary="Report client-side error")
            async def report_error_endpoint(error_data: dict = Body(..., description="Error details from client")):
                """
                Endpoint for JavaScript SDK to report client-side errors.
                
                Expected error_data format:
                {
                    "error_type": "string",
                    "error_message": "string", 
                    "stack_trace": "string",
                    "context": {...}
                }
                
                Security: This endpoint validates and sanitizes input before processing.
                """
                try:
                    # Validate required fields
                    if not isinstance(error_data, dict):
                        raise HTTPException(status_code=400, detail="Invalid error data format")
                    
                    required_fields = ["error_type", "error_message"]
                    for field in required_fields:
                        if field not in error_data:
                            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                    
                    # Sanitize and limit field sizes to prevent abuse
                    max_message_length = 10000
                    max_stack_trace_length = 50000
                    
                    error_type = str(error_data.get("error_type", ""))[:500]
                    error_message = str(error_data.get("error_message", ""))[:max_message_length]
                    stack_trace = str(error_data.get("stack_trace", ""))[:max_stack_trace_length]
                    context = error_data.get("context", {})
                    
                    # Validate context is dict and limit size
                    if not isinstance(context, dict):
                        context = {}
                    
                    # Create sanitized error data
                    sanitized_error_data = {
                        "error_type": error_type,
                        "error_message": error_message,
                        "stack_trace": stack_trace,
                        "context": context
                    }
                    
                    # Report the error to the auto-healing system
                    self._report_client_error(sanitized_error_data)
                    return {"status": "ok", "message": "Error reported successfully"}
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Failed to report client error: {e}")
                    return {"status": "error", "message": "Internal server error"}
            
            # Mount the router (mount_path may be "" when used as a sub-app)
            app.include_router(router, prefix=mount_path)

            # Also register a no-trailing-slash route for direct serving at mount_path
            # (e.g. GET /mcp) to avoid redirects.
            normalized = (mount_path or "").rstrip("/")
            if normalized and normalized != "/":
                app.add_api_route(normalized, lambda: _server_info(), methods=["GET"], include_in_schema=False)
            
            # Debug: Print all registered routes
            logger.debug("FastAPI app created for standalone MCP with routes:")
            for route in app.routes:
                route_path = getattr(
                    route,
                    "path",
                    getattr(route, "prefix", type(route).__name__),
                )
                route_methods = getattr(route, "methods", "")
                logger.debug("Route: %s %s", route_path, route_methods)
            
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
        from pydantic import BaseModel, create_model, Field
        
        if "properties" not in schema:
            return create_model(name, __base__=BaseModel)
        
        required = schema.get("required", [])
        fields = {}
        
        for prop_name, prop_schema in schema["properties"].items():
            field_type = self._get_field_type(prop_schema)
            if prop_name in required:
                default = ...
            elif "default" in prop_schema:
                default = prop_schema["default"]
            else:
                default = None
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
        if isinstance(schema.get("anyOf"), list):
            option_types = [
                self._get_field_type(option)
                for option in schema["anyOf"]
                if isinstance(option, dict)
            ]
            if option_types:
                return Union[tuple(option_types)]
            return Any

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
            value_schema = schema.get("additionalProperties")
            if isinstance(value_schema, dict):
                return Dict[str, self._get_field_type(value_schema)]
            return Dict[str, Any]
        elif schema_type == "null":
            return type(None)
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
    host: str = "0.0.0.0",
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

        fastmcp_ready = False
        try:
            dependency_reason = ""
            try:
                dependency_status = _ensure_mcp_runtime_dependencies()
                logger.debug("MCP first-use dependency status: %s", dependency_status)
                unavailable = sorted(
                    name
                    for name, status in dependency_status.items()
                    if status not in {"ok", "installed"}
                )
                if unavailable:
                    dependency_reason = (
                        "MCP runtime dependency contract is unsatisfied: "
                        + ", ".join(unavailable)
                    )
            except Exception as dependency_error:
                dependency_reason = _safe_exception_summary(
                    "MCP first-use dependency check failed",
                    dependency_error,
                )
                logger.warning(
                    "%s; fallbacks remain available",
                    dependency_reason,
                )

            if dependency_reason:
                fastmcp_module, import_reason = None, dependency_reason
            else:
                fastmcp_module, import_reason = _import_fastmcp_v2_with_repair()
            fastmcp_ready, wiring_reason = self._try_setup_fastmcp_v2(
                fastmcp_module
            )
            if not fastmcp_ready and fastmcp_module is not None:
                repaired_module, repair_reason = _repair_fastmcp_v2_runtime()
                if repaired_module is not None:
                    fastmcp_ready, wiring_reason = self._try_setup_fastmcp_v2(
                        repaired_module
                    )
                elif repair_reason:
                    logger.debug("FastMCP wiring repair unavailable: %s", repair_reason)
            if not fastmcp_ready:
                self._setup_standalone(import_reason or wiring_reason)

            if fastmcp_ready:
                fallback_reason = ""
                try:
                    self._register_components()
                except Exception as registration_error:
                    fallback_reason = (
                        "FastMCP compatibility registration raised "
                        f"{type(registration_error).__name__}"
                    )
                else:
                    from .fastmcp_compat import get_registration_failures

                    failures = get_registration_failures(self.mcp)
                    if failures:
                        sample = ", ".join(
                            f"{failure['kind']}:{failure['name_sha256'][:12]} "
                            f"({failure['error_type']})"
                            for failure in failures[:5]
                        )
                        fallback_reason = (
                            "FastMCP rejected compatibility registrations: "
                            + sample
                        )

                if fallback_reason:
                    self._setup_standalone(fallback_reason)
                    fastmcp_ready = False
                    self._register_components()
            else:
                self._register_components()

            self._enable_cors()

            if fastmcp_ready:
                # Do not advertise the FastMCP transport until its ASGI app and
                # every required compatibility registration have succeeded.
                self._using_fastmcp = True
                logger.info("Using FastMCP %s implementation", FASTMCP_VERSION)

            logger.info(f"IPFS Accelerate MCP Server set up: {self.server_url}")

        except Exception as e:
            if fastmcp_ready:
                self._using_fastmcp = False
                self.mcp = None
                self.fastapi_app = None
            logger.error("%s", _safe_exception_summary("Error setting up MCP server", e))
            raise

    def _register_components(self) -> None:
        """Register all required legacy compatibility components."""

        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _enable_cors(self) -> None:
        """Enable the optional CORS middleware on the selected ASGI app."""

        try:
            from fastapi.middleware.cors import CORSMiddleware

            allowed = os.getenv("MCP_CORS_ORIGINS", "*")
            allow_origins = [
                origin.strip()
                for origin in allowed.split(",")
                if origin.strip()
            ] or ["*"]
            self.fastapi_app.add_middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.info("CORS enabled for MCP API (origins: %s)", allow_origins)
        except Exception as error:
            logger.warning(
                "%s",
                _safe_exception_summary("CORS not enabled", error),
            )

    def _try_setup_fastmcp_v2(self, fastmcp_module: Any) -> Tuple[bool, str]:
        """Wire a FastMCP 2 ASGI app, returning an explicit fallback reason."""

        if fastmcp_module is None:
            return False, "FastMCP is unavailable"

        fastmcp_class = getattr(fastmcp_module, "FastMCP", None)
        if fastmcp_class is None:
            return False, "FastMCP module does not expose FastMCP"

        reported_version = str(getattr(fastmcp_module, "__version__", "") or "")
        if reported_version and reported_version != FASTMCP_VERSION:
            return False, f"unsupported FastMCP version {reported_version}"

        try:
            candidate = fastmcp_class(name=self.name)

            # The legacy registries use register_* methods. Apply all adapters
            # before any tool, resource, or prompt registration can occur.
            from .fastmcp_compat import (
                ensure_register_prompt_compat,
                ensure_register_resource_compat,
                ensure_register_tool_compat,
            )

            for compatibility_shim in (
                ensure_register_tool_compat,
                ensure_register_resource_compat,
                ensure_register_prompt_compat,
            ):
                candidate = compatibility_shim(candidate)

            missing_methods = [
                method_name
                for method_name in (
                    "register_tool",
                    "register_resource",
                    "register_prompt",
                )
                if not callable(getattr(candidate, method_name, None))
            ]
            if missing_methods:
                return (
                    False,
                    "FastMCP compatibility shims did not provide "
                    + ", ".join(missing_methods),
                )

            http_app_factory = getattr(candidate, "http_app", None)
            if not callable(http_app_factory):
                return False, "FastMCP does not expose the v2 http_app API"

            app = http_app_factory(path=self.mount_path or "/")
            if app is None or not callable(app):
                return False, "FastMCP http_app did not return an ASGI application"
        except Exception as error:
            return False, _safe_exception_summary(
                "FastMCP v2 wiring failed",
                error,
            )

        # Stage the FastMCP objects for registration. setup() makes the public
        # transport claim only after all required registration calls succeed.
        self.mcp = candidate
        self.fastapi_app = app
        return True, ""

    def _setup_standalone(self, reason: str) -> None:
        """Create the explicit standalone fallback application."""

        logger.warning("%s; using standalone MCP implementation", reason)
        standalone = StandaloneMCP(name=self.name)
        app = standalone.create_fastapi_app(
            title="IPFS Accelerate MCP API",
            description="API for the IPFS Accelerate MCP Server",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            mount_path=self.mount_path,
        )
        self.mcp = standalone
        self.fastapi_app = app
        self._using_fastmcp = False
    
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
            raise

def main() -> None:
    """
    Main entry point for the IPFS Accelerate MCP Server
    
    This function parses command-line arguments and runs the server.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Server")
    
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
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


@dataclass
class MCPToolInfo:
    name: str
    function: Callable
    description: str
    input_schema: Dict[str, Any]


@dataclass
class MCPResourceInfo:
    path: str
    function: Callable
    description: str


@dataclass
class MCPPromptInfo:
    name: str
    template: str
    description: str
    input_schema: Dict[str, Any]


class MCPServerWrapper:
    """Compatibility wrapper for legacy/tests/CLI usage."""

    def __init__(
        self,
        name: str,
        description: str,
        accelerate_instance: Any,
        host: str = "0.0.0.0",
        port: int = 9000,
        mount_path: str = "/mcp",
        debug: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.host = host
        self.port = port
        self.mount_path = mount_path
        self.debug = debug
        self.state = SimpleNamespace(accelerate=accelerate_instance)

        self.mcp = StandaloneMCP(name=self.name)
        self.mcp.state = SimpleNamespace(accelerate=accelerate_instance)
        self.app = self.mcp.create_fastapi_app(
            title="IPFS Accelerate MCP API",
            description=self.description or "IPFS Accelerate MCP",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            mount_path=self.mount_path,
        )

        from ipfs_accelerate_py.mcp.tools import register_all_tools
        from ipfs_accelerate_py.mcp.resources import register_all_resources

        register_all_tools(self.mcp)
        register_all_resources(self.mcp)

        # Compatibility aliases expected by legacy tests
        try:
            from ipfs_accelerate_py.mcp.tools.hardware import get_hardware_info, recommend_hardware

            if "detect_hardware" not in self.mcp.tools:
                self.mcp.register_tool(
                    name="detect_hardware",
                    function=get_hardware_info,
                    description="Detect available hardware",
                    input_schema={"type": "object", "properties": {}, "required": []},
                    execution_context="server",
                )

            if "get_optimal_hardware" not in self.mcp.tools:
                self.mcp.register_tool(
                    name="get_optimal_hardware",
                    function=recommend_hardware,
                    description="Get optimal hardware for a model",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string", "description": "Model name"},
                            "task": {
                                "type": "string",
                                "description": "Task type",
                                "enum": ["inference", "training", "fine-tuning"],
                                "default": "inference",
                            },
                            "consider_available_only": {
                                "type": "boolean",
                                "description": "Only consider available hardware",
                                "default": True,
                            },
                        },
                        "required": ["model_name"],
                    },
                    execution_context="server",
                )
        except Exception:
            pass

        try:
            import platform

            if "system://info" not in self.mcp.resources:
                self.mcp.register_resource(
                    uri="system://info",
                    function=lambda: {
                        "platform": platform.platform(),
                        "python_version": platform.python_version(),
                    },
                    description="Basic system information",
                )

            if "system://capabilities" not in self.mcp.resources:
                self.mcp.register_resource(
                    uri="system://capabilities",
                    function=lambda: {"accelerators": {}, "features": {}},
                    description="System capabilities",
                )

            if "models://available" not in self.mcp.resources:
                from ipfs_accelerate_py.mcp.resources.model_info import get_default_supported_models

                def _available_models():
                    data = get_default_supported_models()
                    models = []
                    for category in data.get("categories", {}).values():
                        models.extend(category.get("models", []))
                    return models

                self.mcp.register_resource(
                    uri="models://available",
                    function=_available_models,
                    description="Available models",
                )
        except Exception:
            pass

        # Register prompts (best-effort)
        try:
            self.mcp.register_prompt(
                name="ipfs_help",
                template="IPFS Accelerate MCP help",
                description="Get help with IPFS Accelerate",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass

    @property
    def tools(self) -> List[MCPToolInfo]:
        return [
            MCPToolInfo(
                name=name,
                function=tool.get("function"),
                description=tool.get("description", ""),
                input_schema=tool.get("input_schema", {}),
            )
            for name, tool in (self.mcp.tools or {}).items()
        ]

    @property
    def resources(self) -> List[MCPResourceInfo]:
        return [
            MCPResourceInfo(
                path=uri,
                function=resource.get("function"),
                description=resource.get("description", ""),
            )
            for uri, resource in (self.mcp.resources or {}).items()
        ]

    @property
    def prompts(self) -> List[MCPPromptInfo]:
        return [
            MCPPromptInfo(
                name=name,
                template=prompt.get("template", ""),
                description=prompt.get("description", ""),
                input_schema=prompt.get("input_schema", {}),
            )
            for name, prompt in (self.mcp.prompts or {}).items()
        ]

    def run(self, host: Optional[str] = None, port: Optional[int] = None, reload: bool = False) -> None:
        """Run the MCP server via uvicorn."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=host or self.host,
            port=port or self.port,
            log_level="debug" if self.debug else "info",
            reload=reload,
        )


_MCP_SERVER_INSTANCE: Optional[MCPServerWrapper] = None
_MCP_LIKE_INSTANCE: Optional[Any] = None
_MCP_FACADE_USAGE_TELEMETRY = {
    "facade_calls": 0,
    "legacy_wrapper_calls": 0,
    "unified_bridge_calls": 0,
    "dry_run_calls": 0,
    "rollback_calls": 0,
    "bridge_disable_ignored_calls": 0,
    "bridge_failure_calls": 0,
    "warning_emissions": 0,
    "reason_counts": {},
}
_MCP_FACADE_WARNED_REASONS: set[str] = set()


def _env_flag_enabled(name: str) -> bool:
    try:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False


def _record_mcp_facade_usage(telemetry: dict) -> None:
    """Track compatibility-facade usage for cutover/deprecation telemetry."""
    _MCP_FACADE_USAGE_TELEMETRY["facade_calls"] += 1
    reason = str(telemetry.get("reason") or "legacy_fallback")
    reason_counts = _MCP_FACADE_USAGE_TELEMETRY["reason_counts"]
    reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    if telemetry.get("used_legacy_wrapper"):
        _MCP_FACADE_USAGE_TELEMETRY["legacy_wrapper_calls"] += 1
    if telemetry.get("bridge_active"):
        _MCP_FACADE_USAGE_TELEMETRY["unified_bridge_calls"] += 1
    if telemetry.get("cutover_dry_run"):
        _MCP_FACADE_USAGE_TELEMETRY["dry_run_calls"] += 1
    if telemetry.get("force_legacy_rollback"):
        _MCP_FACADE_USAGE_TELEMETRY["rollback_calls"] += 1
    if telemetry.get("bridge_disable_ignored"):
        _MCP_FACADE_USAGE_TELEMETRY["bridge_disable_ignored_calls"] += 1
    if telemetry.get("bridge_error"):
        _MCP_FACADE_USAGE_TELEMETRY["bridge_failure_calls"] += 1
    if telemetry.get("deprecation_warning_emitted"):
        _MCP_FACADE_USAGE_TELEMETRY["warning_emissions"] += 1


def _warn_legacy_facade_usage(reason: str) -> bool:
    """Emit the D2 opt-in-only deprecation notice once per reason."""
    normalized_reason = str(reason or "legacy_fallback")
    if normalized_reason in _MCP_FACADE_WARNED_REASONS:
        return False

    logger.warning(
        "Legacy MCP facade runtime path is deprecated (D2 opt-in only); reason=%s. "
        "Canonical mcp_server startup is now the default and legacy routing should be reserved for explicit rollback/testing only.",
        normalized_reason,
    )
    _MCP_FACADE_WARNED_REASONS.add(normalized_reason)
    return True


def get_mcp_facade_telemetry() -> dict:
    """Return a snapshot of compatibility-facade usage telemetry."""
    snapshot = dict(_MCP_FACADE_USAGE_TELEMETRY)
    snapshot["reason_counts"] = dict(_MCP_FACADE_USAGE_TELEMETRY["reason_counts"])
    return snapshot


def _reset_mcp_facade_telemetry() -> None:
    """Reset compatibility-facade usage telemetry for deterministic tests."""
    for key in _MCP_FACADE_USAGE_TELEMETRY:
        _MCP_FACADE_USAGE_TELEMETRY[key] = {} if key == "reason_counts" else 0
    _MCP_FACADE_WARNED_REASONS.clear()


def set_mcp_like_instance(mcp_like: Any) -> None:
    """Set a global MCP-like instance for in-process tool invocation.

    This is used in deployments that do not create an `MCPServerWrapper`
    (e.g. the Flask dashboard), but still need a tool registry for libp2p
    `op=call_tool`.
    """

    global _MCP_LIKE_INSTANCE
    _MCP_LIKE_INSTANCE = mcp_like


def create_mcp_server(
    name: str = "ipfs-accelerate",
    description: str = "",
    accelerate_instance: Optional[Any] = None,
    host: str = "0.0.0.0",
    port: int = 9000,
    mount_path: str = "/mcp",
    debug: bool = False,
    _skip_unified_bridge: bool = False,
) -> MCPServerWrapper:
    """Create a compatibility MCP server wrapper."""
    global _MCP_SERVER_INSTANCE
    cutover_dry_run_enabled = _env_flag_enabled("IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN")
    force_legacy_rollback = _env_flag_enabled("IPFS_MCP_FORCE_LEGACY_ROLLBACK")
    bridge_env_value = os.environ.get("IPFS_MCP_ENABLE_UNIFIED_BRIDGE")
    bridge_requested = _env_flag_enabled("IPFS_MCP_ENABLE_UNIFIED_BRIDGE")
    bridge_explicit = bridge_env_value is not None
    cutover_dry_run_status = {
        "enabled": bool(cutover_dry_run_enabled),
        "ok": False,
        "error": "",
    }
    facade_telemetry = {
        "facade": "ipfs_accelerate_py.mcp.server.create_mcp_server",
        "bridge_requested": bool(bridge_requested),
        "bridge_defaulted": not bool(bridge_explicit),
        "bridge_disable_ignored": bool(bridge_explicit and not bridge_requested),
        "bridge_active": False,
        "used_legacy_wrapper": False,
        "force_legacy_rollback": bool(force_legacy_rollback),
        "cutover_dry_run": bool(cutover_dry_run_enabled),
        "dry_run_ok": False,
        "deprecation_phase": "D2_opt_in_only",
        "deprecation_warning_emitted": False,
        "selected_runtime": "legacy",
        "reason": "legacy_fallback",
        "bridge_error": "",
    }

    # Route creation through the unified canonical package by default.
    # The private skip flag prevents recursion.
    if not _skip_unified_bridge:
        try:
            # Phase D2 keeps unified startup as the only supported facade default;
            # the legacy runtime now requires explicit rollback opt-in.
            bridge_enabled = True
            if force_legacy_rollback:
                bridge_enabled = False
                facade_telemetry["bridge_disable_ignored"] = False
                facade_telemetry["reason"] = "force_legacy_rollback"
            if bridge_enabled:
                from ipfs_accelerate_py.mcp_server.server import create_server as create_unified_server

                if cutover_dry_run_enabled:
                    try:
                        create_unified_server(
                            name=name,
                            description=description,
                            accelerate_instance=accelerate_instance,
                            host=host,
                            port=port,
                            mount_path=mount_path,
                            debug=debug,
                        )
                        cutover_dry_run_status["ok"] = True
                        facade_telemetry["dry_run_ok"] = True
                        facade_telemetry["reason"] = "dry_run_legacy_fallback"
                        logger.info("Unified cutover dry-run validation succeeded; continuing on legacy path")
                    except Exception as dry_run_exc:
                        cutover_dry_run_status["error"] = str(dry_run_exc)
                        facade_telemetry["bridge_error"] = str(dry_run_exc)
                        facade_telemetry["reason"] = "dry_run_failure_fallback"
                        logger.warning(
                            "Unified cutover dry-run validation failed, continuing on legacy path: %s",
                            dry_run_exc,
                        )
                else:
                    server = create_unified_server(
                        name=name,
                        description=description,
                        accelerate_instance=accelerate_instance,
                        host=host,
                        port=port,
                        mount_path=mount_path,
                        debug=debug,
                    )
                    facade_telemetry["bridge_active"] = True
                    facade_telemetry["selected_runtime"] = "unified"
                    facade_telemetry["reason"] = "unified_default" if not bridge_explicit else "unified_bridge"
                    _MCP_SERVER_INSTANCE = server
                    try:
                        set_mcp_like_instance(getattr(server, "mcp", None) or server)
                    except Exception:
                        pass
                    try:
                        setattr(server, "_mcp_facade_telemetry", dict(facade_telemetry))
                    except Exception:
                        pass
                    _record_mcp_facade_usage(facade_telemetry)
                    return server
        except Exception as e:
            facade_telemetry["bridge_error"] = str(e)
            facade_telemetry["reason"] = "bridge_error_fallback"
            logger.warning(f"Unified MCP bridge unavailable, falling back to legacy wrapper: {e}")

    server = MCPServerWrapper(
        name=name,
        description=description,
        accelerate_instance=accelerate_instance,
        host=host,
        port=port,
        mount_path=mount_path,
        debug=debug,
    )
    facade_telemetry["used_legacy_wrapper"] = True
    _MCP_SERVER_INSTANCE = server
    try:
        # Also expose the underlying MCP registry for callers that only
        # need a tool registry (e.g. libp2p tool bridge).
        set_mcp_like_instance(getattr(server, "mcp", None) or server)
    except Exception:
        pass
    facade_telemetry["deprecation_warning_emitted"] = _warn_legacy_facade_usage(
        str(facade_telemetry.get("reason") or "legacy_fallback")
    )
    if cutover_dry_run_enabled:
        try:
            setattr(server, "_unified_cutover_dry_run", dict(cutover_dry_run_status))
        except Exception:
            pass
    try:
        setattr(server, "_mcp_facade_telemetry", dict(facade_telemetry))
    except Exception:
        pass
    _record_mcp_facade_usage(facade_telemetry)
    return server


def get_mcp_server_instance() -> Optional[Any]:
    """Return the last created MCP server instance or MCP-like registry, if any."""
    return _MCP_SERVER_INSTANCE or _MCP_LIKE_INSTANCE

if __name__ == "__main__":
    main()

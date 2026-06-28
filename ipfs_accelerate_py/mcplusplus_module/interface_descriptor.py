"""Profile A: MCP-IDL Interface Descriptors for ipfs_accelerate_py.

Implements the MCP++ Profile A specification:
- CID-addressed interface contracts (InterfaceDescriptor)
- InterfaceRepository for storing and querying descriptors
- Auto-generation of descriptors from registered MCP tools
- Schema validation for method inputs/outputs

Module: ipfs_accelerate_py.mcplusplus_module.interface_descriptor
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cid_ucan import compute_cid


@dataclass
class MethodDescriptor:
    """Describes a single method in an interface."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    idempotent: bool = False
    cacheable: bool = False
    timeout_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "idempotent": self.idempotent,
            "cacheable": self.cacheable,
            "timeout_ms": self.timeout_ms,
        }


@dataclass
class InterfaceDescriptor:
    """A CID-addressed interface contract (Profile A).

    Describes a set of methods that a service provides, including
    input/output schemas, versioning, and metadata.
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    methods: List[MethodDescriptor] = field(default_factory=list)
    author: str = ""
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            # Include full method schemas to avoid CID collisions between
            # interfaces with same method names but different signatures
            self.cid = compute_cid({
                "type": "interface_descriptor",
                "name": self.name,
                "version": self.version,
                "methods": [
                    {"name": m.name, "input_schema": m.input_schema, "output_schema": m.output_schema}
                    for m in self.methods
                ],
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "methods": [m.to_dict() for m in self.methods],
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "method_count": len(self.methods),
        }


class InterfaceRepository:
    """Repository for storing and querying interface descriptors."""

    def __init__(self):
        self._interfaces: Dict[str, InterfaceDescriptor] = {}
        self._by_name: Dict[str, str] = {}  # name → cid

    def register(self, interface: InterfaceDescriptor) -> str:
        """Register an interface. Returns its CID."""
        self._interfaces[interface.cid] = interface
        self._by_name[interface.name] = interface.cid
        return interface.cid

    def get(self, cid: str) -> Optional[InterfaceDescriptor]:
        return self._interfaces.get(cid)

    def get_by_name(self, name: str) -> Optional[InterfaceDescriptor]:
        cid = self._by_name.get(name)
        if cid:
            return self._interfaces.get(cid)
        return None

    def list_all(self) -> List[InterfaceDescriptor]:
        return list(self._interfaces.values())

    def search(self, query: str) -> List[InterfaceDescriptor]:
        """Search interfaces by name or tag."""
        query_lower = query.lower()
        results = []
        for iface in self._interfaces.values():
            if query_lower in iface.name.lower():
                results.append(iface)
            elif any(query_lower in tag.lower() for tag in iface.tags):
                results.append(iface)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self._interfaces),
            "interfaces": [i.to_dict() for i in self._interfaces.values()],
        }


# ---------------------------------------------------------------------------
# Auto-generation from MCP tools
# ---------------------------------------------------------------------------

def generate_from_mcp_tools(mcp_server) -> InterfaceDescriptor:
    """Auto-generate an InterfaceDescriptor from registered MCP tools.

    Introspects the MCP server's tool registry and creates a complete
    interface descriptor with schemas derived from tool definitions.
    """
    methods = []

    if hasattr(mcp_server, 'tools'):
        for tool_name, tool_fn in mcp_server.tools.items():
            # Extract schema from tool metadata if available
            input_schema = {}
            description = ""

            if hasattr(tool_fn, '__doc__') and tool_fn.__doc__:
                description = tool_fn.__doc__.strip().split('\n')[0]

            if hasattr(tool_fn, '_input_schema'):
                input_schema = tool_fn._input_schema
            elif hasattr(tool_fn, '__annotations__'):
                # Build schema from type annotations
                for param, ptype in tool_fn.__annotations__.items():
                    if param == 'return':
                        continue
                    type_name = getattr(ptype, '__name__', str(ptype))
                    input_schema[param] = {"type": type_name}

            methods.append(MethodDescriptor(
                name=tool_name,
                description=description,
                input_schema={"type": "object", "properties": input_schema},
            ))

    return InterfaceDescriptor(
        name="ipfs-accelerate",
        version="1.0.0",
        description="IPFS Accelerate MCP++ server — hardware-accelerated ML inference and P2P task distribution",
        methods=methods,
        author="endomorphosis",
        tags=["ipfs", "accelerate", "ml", "inference", "p2p", "hardware"],
    )


# ---------------------------------------------------------------------------
# Pre-built interface descriptors
# ---------------------------------------------------------------------------

# P2P TaskQueue Interface
P2P_TASKQUEUE_INTERFACE = InterfaceDescriptor(
    name="ipfs-accelerate-p2p-taskqueue",
    version="1.0.0",
    description="Distributed task queue over libp2p for ML workloads",
    methods=[
        MethodDescriptor(name="p2p_taskqueue_status", description="Get task queue status"),
        MethodDescriptor(name="p2p_taskqueue_submit", description="Submit a task to the queue",
                         input_schema={"type": "object", "properties": {"task_type": {"type": "string"}, "payload": {"type": "object"}}}),
        MethodDescriptor(name="p2p_taskqueue_claim_next", description="Claim next available task"),
        MethodDescriptor(name="p2p_taskqueue_complete", description="Mark a task as complete",
                         input_schema={"type": "object", "properties": {"task_id": {"type": "string"}, "result": {"type": "object"}}}),
        MethodDescriptor(name="p2p_taskqueue_list", description="List tasks with optional filters"),
        MethodDescriptor(name="p2p_taskqueue_get", description="Get task details by ID"),
        MethodDescriptor(name="p2p_taskqueue_heartbeat", description="Send worker heartbeat"),
    ],
    tags=["p2p", "taskqueue", "distributed"],
)

# P2P Workflow Interface
P2P_WORKFLOW_INTERFACE = InterfaceDescriptor(
    name="ipfs-accelerate-p2p-workflow",
    version="1.0.0",
    description="Workflow scheduling and execution over libp2p",
    methods=[
        MethodDescriptor(name="p2p_scheduler_status", description="Get scheduler status"),
        MethodDescriptor(name="p2p_submit_task", description="Submit workflow task"),
        MethodDescriptor(name="p2p_get_next_task", description="Get next scheduled task"),
        MethodDescriptor(name="p2p_mark_task_complete", description="Complete a workflow task"),
        MethodDescriptor(name="p2p_list_peers", description="List connected workflow peers"),
        MethodDescriptor(name="p2p_cache_get", description="Get value from distributed cache"),
        MethodDescriptor(name="p2p_cache_set", description="Set value in distributed cache"),
    ],
    tags=["p2p", "workflow", "scheduler"],
)

# Hardware Accelerate Interface
HARDWARE_ACCELERATE_INTERFACE = InterfaceDescriptor(
    name="ipfs-accelerate-hardware",
    version="1.0.0",
    description="Hardware-accelerated ML model inference",
    methods=[
        MethodDescriptor(name="run_model", description="Run inference on a model",
                         input_schema={"type": "object", "properties": {"model": {"type": "string"}, "input": {"type": "object"}}}),
        MethodDescriptor(name="list_models", description="List available models", cacheable=True),
        MethodDescriptor(name="hardware_profile", description="Get hardware acceleration profile", cacheable=True),
        MethodDescriptor(name="capabilities", description="List server capabilities", idempotent=True, cacheable=True),
        MethodDescriptor(name="benchmark", description="Run hardware benchmark"),
        MethodDescriptor(name="model_info", description="Get detailed model information"),
    ],
    tags=["hardware", "ml", "inference", "gpu", "accelerate"],
)


# ---------------------------------------------------------------------------
# Global singleton (thread-safe)
# ---------------------------------------------------------------------------

_REPOSITORY: Optional[InterfaceRepository] = None
_REPO_LOCK = threading.Lock()


def get_interface_repository() -> InterfaceRepository:
    """Get the global interface repository, pre-loaded with built-in descriptors."""
    global _REPOSITORY
    if _REPOSITORY is None:
        with _REPO_LOCK:
            if _REPOSITORY is None:
                _REPOSITORY = InterfaceRepository()
                _REPOSITORY.register(P2P_TASKQUEUE_INTERFACE)
                _REPOSITORY.register(P2P_WORKFLOW_INTERFACE)
                _REPOSITORY.register(HARDWARE_ACCELERATE_INTERFACE)
    return _REPOSITORY


# ---------------------------------------------------------------------------
# Schema validation (lightweight, no jsonschema dependency)
# ---------------------------------------------------------------------------

def validate_params(method_name: str, params: Dict[str, Any],
                    repository: Optional[InterfaceRepository] = None) -> Optional[str]:
    """Validate params against the method's declared input_schema.

    Returns None if valid (or no schema available), or an error string if invalid.
    This is a lightweight type-only validator — it checks 'required' fields and
    top-level property types without a full JSON Schema library.
    """
    repo = repository or get_interface_repository()

    # Find the method descriptor across all interfaces
    method_desc = None
    for iface in repo.list_all():
        for m in iface.methods:
            if m.name == method_name:
                method_desc = m
                break
        if method_desc:
            break

    if not method_desc or not method_desc.input_schema:
        return None  # No schema = no validation

    schema = method_desc.input_schema
    if schema.get("type") != "object":
        return None  # Only validate object schemas

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for field_name in required:
        if field_name not in params:
            return f"Missing required parameter: '{field_name}'"

    # Check type compatibility for provided params
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for param_name, param_value in params.items():
        if param_name.startswith("_"):
            continue  # Skip internal params
        if param_name not in properties:
            continue  # Extra params are allowed (additionalProperties: true by default)
        expected_type = properties[param_name].get("type")
        if expected_type and expected_type in _TYPE_MAP:
            py_type = _TYPE_MAP[expected_type]
            if param_value is not None and not isinstance(param_value, py_type):
                return (
                    f"Parameter '{param_name}' has type {type(param_value).__name__}, "
                    f"expected {expected_type}"
                )

    return None  # Valid

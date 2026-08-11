"""FastMCP compatibility helpers.

This module provides a small adapter layer so code written against the older
`StandaloneMCP.register_tool(...)` API can also operate with `fastmcp.FastMCP`.

FastMCP registers tools via the `mcp.tool(...)` decorator and does not expose a
`register_tool` method.
"""

from __future__ import annotations

import logging
import inspect
import copy
import hashlib
import re
import types
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger("ipfs_accelerate_mcp.fastmcp_compat")

_FAILURE_ATTRIBUTE = "_fastmcp_compat_registration_failures"
_SCHEMA_DRIFT_ATTRIBUTE = "_fastmcp_compat_schema_drifts"
_MAX_FAILURE_RECEIPTS = 100


def resolve_fastmcp_types() -> tuple[Any, Any]:
    """Resolve FastMCP/Context through the audited loader or safe mocks."""

    from .server import _import_fastmcp_v2

    module, _ = _import_fastmcp_v2()
    fastmcp_class = getattr(module, "FastMCP", None)
    context_class = getattr(module, "Context", None)
    if callable(fastmcp_class) and context_class is not None:
        return fastmcp_class, context_class

    from .mock_mcp import Context, FastMCP

    return FastMCP, Context


def _ensure_dict_view(mcp: Any, attribute: str) -> dict[str, Any]:
    """Return a mutable legacy registry view or fail explicitly."""

    existing = getattr(mcp, attribute, None)
    if existing is None:
        try:
            setattr(mcp, attribute, {})
        except Exception as error:
            raise RuntimeError(
                f"FastMCP compatibility cannot expose .{attribute}"
            ) from error
        existing = getattr(mcp, attribute, None)
    if not isinstance(existing, dict):
        raise TypeError(
            f"FastMCP compatibility requires .{attribute} to be a dict, "
            f"not {type(existing).__name__}"
        )
    return existing


def _attach_method(mcp: Any, name: str, function: Callable[..., Any]) -> None:
    """Attach a compatibility method without silently accepting failure."""

    try:
        setattr(mcp, name, function)
    except Exception as error:
        raise RuntimeError(f"FastMCP compatibility cannot attach {name}") from error
    if not callable(getattr(mcp, name, None)):
        raise RuntimeError(f"FastMCP compatibility did not attach {name}")


def _record_failure(mcp: Any, kind: str, name: str, error: Exception) -> None:
    """Record a bounded failure receipt before allowing callers to handle it."""

    failures = getattr(mcp, _FAILURE_ATTRIBUTE, None)
    if failures is None:
        failures = []
        try:
            setattr(mcp, _FAILURE_ATTRIBUTE, failures)
        except Exception as attach_error:
            raise RuntimeError(
                "FastMCP compatibility cannot record registration failures"
            ) from attach_error
    if not isinstance(failures, list):
        raise TypeError("FastMCP compatibility failure ledger is not a list")
    kind_value = str(kind)
    safe_kind = (
        kind_value
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", kind_value)
        else "component"
    )
    error_type = type(error).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type):
        error_type = "RegistrationError"
    name_digest = hashlib.sha256(
        str(name).encode("utf-8", errors="replace")
    ).hexdigest()
    receipt = {
        "kind": safe_kind,
        "name_sha256": name_digest,
        "error_type": error_type,
    }
    if len(failures) < _MAX_FAILURE_RECEIPTS:
        failures.append(receipt)


def get_registration_failures(mcp: Any) -> tuple[dict[str, str], ...]:
    """Return immutable copies of compatibility registration failures."""

    failures = getattr(mcp, _FAILURE_ATTRIBUTE, [])
    if not isinstance(failures, list):
        return (
            {
                "kind": "adapter",
                "name_sha256": hashlib.sha256(b"failure-ledger").hexdigest(),
                "error_type": "InvalidLedger",
            },
        )
    normalized: list[dict[str, str]] = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        name_sha256 = item.get("name_sha256")
        error_type = item.get("error_type")
        if not (
            isinstance(kind, str)
            and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", kind)
            and isinstance(name_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", name_sha256)
            and isinstance(error_type, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type)
        ):
            normalized.append(
                {
                    "kind": "adapter",
                    "name_sha256": hashlib.sha256(b"failure-ledger-entry").hexdigest(),
                    "error_type": "InvalidLedger",
                }
            )
            continue
        normalized.append(
            {
                "kind": kind,
                "name_sha256": name_sha256,
                "error_type": error_type,
            }
        )
    return tuple(normalized)


def _strip_schema_titles(value: Any, *, root: bool = False) -> Any:
    """Normalize Pydantic's callable schema to FastMCP's public parameters."""

    if isinstance(value, list):
        return [_strip_schema_titles(item) for item in value]
    if not isinstance(value, dict):
        return value
    # FastMCP drops presentation titles from typed schemas, but retains the
    # title Pydantic emits for ``Any`` parameters (including ``Any = None``),
    # because that is the only remaining indication of the parameter name.
    # Match that distinction instead of removing every title from mappings
    # that also happen to contain a default.
    keep_untyped_title = "title" in value and set(value).issubset({"title", "default"})
    normalized = {
        key: _strip_schema_titles(item)
        for key, item in value.items()
        if key != "title" or keep_untyped_title
    }
    if root and normalized.get("additionalProperties") is False:
        normalized.pop("additionalProperties", None)
    return normalized


def canonical_function_input_schema(
    function: Callable[..., Any],
) -> Optional[dict[str, Any]]:
    """Derive the callable contract, returning ``None`` if it is not inspectable."""

    unwrapped = inspect.unwrap(function)
    try:
        parameters = inspect.signature(unwrapped).parameters.values()
    except Exception:
        return None

    # FastMCP 2.14.7 derives its tool parameters from Pydantic's callable
    # TypeAdapter, then removes presentation-only titles. Reuse that stable
    # mechanism when Pydantic 2 is present so Standalone and FastMCP publish
    # the same nullable/default/type contract.
    try:
        from pydantic import TypeAdapter

        schema = TypeAdapter(unwrapped).json_schema()
        if isinstance(schema, dict):
            normalized = _strip_schema_titles(schema, root=True)
            if (
                normalized.get("type") == "object"
                and isinstance(normalized.get("properties"), dict)
            ):
                return normalized
    except Exception:
        pass

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    try:
        resolved_hints = get_type_hints(unwrapped)
    except Exception:
        resolved_hints = {}
    for parameter in parameters:
        if parameter.name in {"self", "cls"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = resolved_hints.get(parameter.name, parameter.annotation)
        property_schema = _annotation_schema(annotation)
        if parameter.default is not inspect.Parameter.empty:
            property_schema["default"] = parameter.default
        properties[parameter.name] = property_schema
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _function_schema(function: Callable[..., Any]) -> dict[str, Any]:
    """Build the stable parameter surface for a callable."""

    return canonical_function_input_schema(function) or {
        "type": "object",
        "properties": {},
    }


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """Map an annotation to a JSON schema, retaining nullable contracts."""

    origin = get_origin(annotation)
    union_origins = {Union}
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_origins.add(union_type)
    if origin in union_origins:
        options = []
        for item in get_args(annotation):
            if item is type(None):
                options.append({"type": "null"})
            else:
                options.append(_annotation_schema(item))
        return {"anyOf": options} if options else {"type": "string"}
    if origin is Literal:
        values = list(get_args(annotation))
        value_type = type(values[0]) if values else str
        schema = _annotation_schema(value_type)
        schema["enum"] = values
        return schema
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or origin is list:
        args = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _annotation_schema(args[0])
        return schema
    if annotation is dict or origin is dict:
        schema = {"type": "object"}
        args = get_args(annotation)
        if len(args) == 2:
            schema["additionalProperties"] = _annotation_schema(args[1])
        return schema
    return {"type": "string"}


def _annotation_schema_type(annotation: Any) -> str:
    """Map a resolved annotation to its top-level JSON schema type."""

    origin = get_origin(annotation)
    union_origins = {Union}
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_origins.add(union_type)
    if origin in union_origins:
        non_null = [item for item in get_args(annotation) if item is not type(None)]
        if len(non_null) == 1:
            return _annotation_schema_type(non_null[0])
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is list or origin is list:
        return "array"
    if annotation is dict or origin is dict:
        return "object"
    return "string"


def function_input_schema(function: Callable[..., Any]) -> dict[str, Any]:
    """Return the signature-derived schema used by legacy registries."""

    return _function_schema(function)


def _registered_tool_schema(
    registered: Any,
    function: Callable[..., Any],
) -> dict[str, Any]:
    """Use FastMCP's protocol schema when available."""

    parameters = getattr(registered, "parameters", None)
    if isinstance(parameters, dict):
        schema = copy.deepcopy(parameters)
        if (
            schema.get("type") != "object"
            or not isinstance(schema.get("properties"), dict)
            or not isinstance(schema.get("required", []), list)
        ):
            raise TypeError("FastMCP returned an invalid tool parameter schema")
        return schema
    return _function_schema(function)


def _material_schema(schema: dict[str, Any]) -> tuple[Any, ...]:
    """Project a JSON schema onto call-safety-relevant fields."""

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    property_types = []
    for name, value in properties.items():
        if not isinstance(value, dict):
            property_types.append((str(name), None, None))
            continue
        schema_type = value.get("type")
        enum_values = value.get("enum")
        if schema_type is None and isinstance(value.get("anyOf"), list):
            non_null_options = [
                option
                for option in value["anyOf"]
                if isinstance(option, dict)
                and option.get("type") not in {None, "null"}
            ]
            non_null_types = sorted(
                str(option.get("type")) for option in non_null_options
            )
            # Optional[T] is materially the same callable contract as T with a
            # default. Preserve tuples only for genuine multi-type unions.
            schema_type = (
                non_null_types[0]
                if len(non_null_types) == 1
                else tuple(non_null_types)
            )
            if len(non_null_options) == 1:
                enum_values = non_null_options[0].get("enum")
        normalized_enum = (
            tuple(sorted(repr(item) for item in enum_values))
            if isinstance(enum_values, list)
            else None
        )
        property_types.append((str(name), schema_type, normalized_enum))
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    return tuple(sorted(property_types)), tuple(sorted(str(item) for item in required))


def _record_schema_drift(mcp: Any, name: str) -> None:
    """Record bounded, secret-safe evidence of ignored legacy schema drift."""

    drifts = getattr(mcp, _SCHEMA_DRIFT_ATTRIBUTE, None)
    if drifts is None:
        drifts = []
        setattr(mcp, _SCHEMA_DRIFT_ATTRIBUTE, drifts)
    if not isinstance(drifts, list):
        raise TypeError("FastMCP compatibility schema-drift ledger is not a list")
    if len(drifts) < _MAX_FAILURE_RECEIPTS:
        drifts.append(
            {
                "kind": "tool_schema",
                "name_sha256": hashlib.sha256(
                    str(name).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
        )


def get_schema_drifts(mcp: Any) -> tuple[dict[str, str], ...]:
    """Return immutable copies of nonfatal legacy schema drift receipts."""

    drifts = getattr(mcp, _SCHEMA_DRIFT_ATTRIBUTE, [])
    if not isinstance(drifts, list):
        return ({"kind": "invalid_ledger", "name_sha256": ""},)
    return tuple(dict(item) for item in drifts if isinstance(item, dict))


def _validate_explicit_schema(
    mcp: Any,
    name: str,
    explicit_schema: Optional[dict[str, Any]],
    native_schema: dict[str, Any],
) -> None:
    """Ignore contradictory legacy schemas in favor of the callable contract."""

    if explicit_schema is None:
        return
    if _material_schema(explicit_schema) == _material_schema(native_schema):
        return
    _record_schema_drift(mcp, name)
    logger.warning(
        "Ignored contradictory legacy schema for tool receipt=%s",
        hashlib.sha256(str(name).encode("utf-8", errors="replace")).hexdigest()[:12],
    )


def _tool_view_record(
    function: Callable[..., Any],
    *,
    name: str,
    description: Optional[str] = None,
    input_schema: Optional[dict[str, Any]] = None,
    category: Optional[str] = None,
    execution_context: Optional[str] = None,
    runtime: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "function": function,
        "description": description or inspect.getdoc(function) or "",
        "input_schema": input_schema or _function_schema(function),
        "category": category,
        "execution_context": execution_context,
        "runtime": runtime,
        "tags": [str(tag) for tag in (tags or []) if str(tag).strip()],
        "metadata": dict(metadata or {}),
    }


def ensure_register_tool_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_tool(...)` exists.

    If the provided MCP instance already has `register_tool`, this is a no-op.
    If it looks like a FastMCP instance (has `tool`), a `register_tool` method
    is attached that delegates to `mcp.tool(...)`.

    Note: FastMCP does not currently accept an `input_schema` parameter. When
    adapting, schemas are ignored and FastMCP will infer inputs from function
    type hints.

    Returns:
        The same MCP instance, potentially patched.
    """

    tools = _ensure_dict_view(mcp, "tools")

    if callable(getattr(mcp, "register_tool", None)):
        return mcp

    native_tool = getattr(mcp, "tool", None)
    if not callable(native_tool):
        return mcp

    def _native_tool_name(
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        configured = kwargs.get("name")
        if configured is None and args and isinstance(args[0], str):
            configured = args[0]
        return str(configured or getattr(function, "__name__", "tool"))

    def _store_native_tool(
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        registered: Any,
    ) -> None:
        name = _native_tool_name(function, args, kwargs)
        raw_tags = kwargs.get("tags")
        tags = list(raw_tags) if isinstance(raw_tags, (list, set, tuple)) else []
        tools[name] = _tool_view_record(
            function,
            name=name,
            description=kwargs.get("description"),
            input_schema=_registered_tool_schema(registered, function),
            tags=tags,
        )

    def _tool_with_view(*args: Any, **kwargs: Any) -> Any:
        direct_function = args[0] if args and callable(args[0]) else None
        failure_name = str(
            kwargs.get("name")
            or (args[0] if args and isinstance(args[0], str) else "tool")
        )
        try:
            result = native_tool(*args, **kwargs)
        except Exception as error:
            _record_failure(mcp, "tool", failure_name, error)
            raise

        if direct_function is not None:
            _store_native_tool(direct_function, args, kwargs, result)
            return result
        if not callable(result):
            return result

        def _decorator(function: Callable[..., Any]) -> Any:
            name = _native_tool_name(function, args, kwargs)
            try:
                registered = result(function)
            except Exception as error:
                _record_failure(mcp, "tool", name, error)
                raise
            _store_native_tool(function, args, kwargs, registered)
            return registered

        return _decorator

    def _register_tool(
        *,
        name: str,
        function: Optional[Callable[..., Any]] = None,
        func: Optional[Callable[..., Any]] = None,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        category: Optional[str] = None,
        execution_context: Optional[str] = None,
        runtime: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **metadata: Any,
    ) -> Any:
        function = function or func
        if function is None:
            raise TypeError("register_tool requires 'function' (or 'func')")
        if input_schema is not None:
            logger.debug("Ignoring input_schema for FastMCP tool '%s'", name)

        # Register with FastMCP first. A compatibility view must never claim a
        # component which the protocol server rejected.
        try:
            decorator = native_tool(name=name, description=description)
            registered = decorator(function)
        except Exception as error:
            _record_failure(mcp, "tool", name, error)
            raise
        native_schema = _registered_tool_schema(registered, function)
        _validate_explicit_schema(mcp, name, input_schema, native_schema)
        tools[name] = _tool_view_record(
            function,
            name=name,
            description=description,
            input_schema=native_schema,
            category=category,
            execution_context=execution_context,
            runtime=runtime,
            tags=tags,
            metadata=metadata,
        )
        return registered

    _attach_method(mcp, "tool", _tool_with_view)
    _attach_method(mcp, "register_tool", _register_tool)
    logger.info("Attached register_tool compatibility shim to FastMCP")

    return mcp


def ensure_register_resource_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_resource(...)` exists.

    StandaloneMCP uses `register_resource(uri=..., function=..., description=...)`.
    FastMCP registers resources via the `mcp.resource(uri, ...)` decorator and
    requires URL-like URIs (e.g. `mcp://server_config`).

    For compatibility, bare URIs are automatically prefixed with `mcp://`.
    """

    resources = _ensure_dict_view(mcp, "resources")

    if callable(getattr(mcp, "register_resource", None)):
        return mcp

    native_resource = getattr(mcp, "resource", None)
    if not callable(native_resource):
        return mcp

    def _resource_with_view(*args: Any, **kwargs: Any) -> Any:
        uri = str(args[0] if args else kwargs.get("uri", "resource"))
        try:
            decorator = native_resource(*args, **kwargs)
        except Exception as error:
            _record_failure(mcp, "resource", uri, error)
            raise
        if not callable(decorator):
            return decorator

        def _decorate(function: Callable[..., Any]) -> Any:
            try:
                registered = decorator(function)
            except Exception as error:
                _record_failure(mcp, "resource", uri, error)
                raise
            resources[uri] = {
                "function": function,
                "description": kwargs.get("description")
                or inspect.getdoc(function)
                or "",
                "normalized_uri": uri,
                "metadata": {},
            }
            return registered

        return _decorate

    def _normalize_uri(uri: str) -> str:
        if "://" in uri:
            return uri
        return f"mcp://{uri.lstrip('/')}"

    def _register_resource(
        *,
        uri: str,
        function: Callable[..., Any],
        description: Optional[str] = None,
        **metadata: Any,
    ) -> Any:
        normalized = _normalize_uri(uri)

        # FastMCP requires URI templates with placeholders when the resource
        # function accepts parameters.
        try:
            sig = inspect.signature(inspect.unwrap(function))
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except Exception:
            params = []

        if params and "{" not in normalized:
            for p in params:
                normalized = normalized.rstrip("/") + f"/{{{p.name}}}"
        try:
            decorator = native_resource(normalized, description=description)
            registered = decorator(function)
        except Exception as error:
            _record_failure(mcp, "resource", uri, error)
            raise
        resources[uri] = {
            "function": function,
            "description": description or "",
            "normalized_uri": normalized,
            "metadata": metadata,
        }
        return registered

    _attach_method(mcp, "resource", _resource_with_view)
    _attach_method(mcp, "register_resource", _register_resource)
    logger.info("Attached register_resource compatibility shim to FastMCP")

    return mcp


def ensure_register_prompt_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_prompt(...)` exists.

    StandaloneMCP exposes `register_prompt(name=..., template=..., description=..., input_schema=...)`.
    FastMCP registers prompts via the `mcp.prompt(...)` decorator and infers input
    schemas from function signatures.

    For compatibility, `template` is returned verbatim by a zero-arg function and
    `input_schema` is ignored.
    """

    prompts = _ensure_dict_view(mcp, "prompts")

    if callable(getattr(mcp, "register_prompt", None)):
        return mcp

    native_prompt = getattr(mcp, "prompt", None)
    if not callable(native_prompt):
        return mcp

    def _native_prompt_name(
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        configured = kwargs.get("name")
        if configured is None and args and isinstance(args[0], str):
            configured = args[0]
        return str(configured or getattr(function, "__name__", "prompt"))

    def _store_native_prompt(
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        name = _native_prompt_name(function, args, kwargs)
        prompts[name] = {
            "template": "",
            "function": function,
            "description": kwargs.get("description")
            or inspect.getdoc(function)
            or "",
            "input_schema": _function_schema(function),
            "metadata": {},
        }

    def _prompt_with_view(*args: Any, **kwargs: Any) -> Any:
        direct_function = args[0] if args and callable(args[0]) else None
        failure_name = str(
            kwargs.get("name")
            or (args[0] if args and isinstance(args[0], str) else "prompt")
        )
        try:
            result = native_prompt(*args, **kwargs)
        except Exception as error:
            _record_failure(mcp, "prompt", failure_name, error)
            raise
        if direct_function is not None:
            _store_native_prompt(direct_function, args, kwargs)
            return result
        if not callable(result):
            return result

        def _decorator(function: Callable[..., Any]) -> Any:
            name = _native_prompt_name(function, args, kwargs)
            try:
                registered = result(function)
            except Exception as error:
                _record_failure(mcp, "prompt", name, error)
                raise
            _store_native_prompt(function, args, kwargs)
            return registered

        return _decorator

    def _register_prompt(
        *,
        name: str,
        template: str,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        **metadata: Any,
    ) -> Any:
        if input_schema is not None:
            logger.debug("Ignoring input_schema for FastMCP prompt '%s'", name)

        def _prompt_fn() -> str:
            return template

        try:
            decorator = native_prompt(name=name, description=description)
            registered = decorator(_prompt_fn)
        except Exception as error:
            _record_failure(mcp, "prompt", name, error)
            raise
        prompts[name] = {
            "template": template,
            "description": description or "",
            "input_schema": input_schema
            or {"type": "object", "properties": {}, "required": []},
            "metadata": metadata,
        }
        return registered

    _attach_method(mcp, "prompt", _prompt_with_view)
    _attach_method(mcp, "register_prompt", _register_prompt)
    logger.info("Attached register_prompt compatibility shim to FastMCP")

    return mcp

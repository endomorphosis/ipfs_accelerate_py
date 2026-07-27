"""Conformance tests for the bounded ``ai.catalog.v1`` MCP++ IDL."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    AI_CATALOG_INVOKE_AUTHORITY,
    AI_CATALOG_READ_AUTHORITY,
    AI_CATALOG_REFRESH_AUTHORITY,
    AI_CATALOG_SCHEMA_REVISION,
    AI_CATALOG_VERSION,
    IDLValidationError,
    InterfaceDescriptorRegistry,
    InterfaceUpgradeRequired,
    ai_catalog_v1_input_schemas,
    authorize_ai_catalog_operation,
    build_ai_catalog_v1_descriptor,
    build_descriptor,
    canonicalize_descriptor,
    compute_interface_cid,
    resolve_ai_catalog_operation,
    validate_ai_catalog_payload,
)


EXPECTED_OPERATIONS = {
    "model_catalog_list_services",
    "model_catalog_list_models",
    "model_catalog_get",
    "model_catalog_resolve",
    "model_catalog_health",
    "model_catalog_refresh",
    "llm_generate",
    "embeddings_generate",
    "multimodal_generate",
    "voice_transcribe",
    "voice_synthesize",
}
EXPECTED_INTERFACE_CID = (
    "cidv1-sha256-13e0f0a7b9d8cae9b5d0ca0d5d4c1c0e"
    "ea392e2225b5e5e3f05aa272bbf7315d"
)


class _SchemaCollector:
    def __init__(self) -> None:
        self.schemas: dict[str, dict] = {}

    def register_tool(self, **registration: object) -> None:
        self.schemas[str(registration["name"])] = copy.deepcopy(
            registration["input_schema"]  # type: ignore[arg-type]
        )


def _local_mcp_input_schemas() -> dict[str, dict]:
    from ipfs_accelerate_py.mcp_server.tools.ai_router_tools.text_embedding import (
        register_native_ai_router_tools as register_text_embedding,
    )
    from ipfs_accelerate_py.mcp_server.tools.ai_router_tools.vision_voice import (
        register_native_vision_voice_tools,
    )
    from ipfs_accelerate_py.mcp_server.tools.model_tools.native_model_tools import (
        register_native_model_tools,
    )

    collector = _SchemaCollector()
    register_native_model_tools(collector)
    register_text_embedding(collector)
    register_native_vision_voice_tools(collector)
    return {
        name: collector.schemas[name]
        for name in EXPECTED_OPERATIONS
    }


def _methods() -> dict[str, dict]:
    descriptor = build_ai_catalog_v1_descriptor()
    return {
        method["operation"]: method
        for method in descriptor["methods"]
    }


def _walk_schemas(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_schemas(child)


def test_descriptor_covers_local_mcp_operations_and_schemas_exactly() -> None:
    methods = _methods()
    input_schemas = ai_catalog_v1_input_schemas()

    assert set(methods) == EXPECTED_OPERATIONS
    assert input_schemas == _local_mcp_input_schemas()
    assert all(
        method["input_schema"] == input_schemas[operation]
        for operation, method in methods.items()
    )


def test_authorities_are_separate_and_fail_closed() -> None:
    methods = _methods()
    grouped = {
        authority: {
            name
            for name, method in methods.items()
            if method["required_authority"] == authority
        }
        for authority in (
            AI_CATALOG_READ_AUTHORITY,
            AI_CATALOG_REFRESH_AUTHORITY,
            AI_CATALOG_INVOKE_AUTHORITY,
        )
    }
    assert grouped[AI_CATALOG_READ_AUTHORITY] == {
        "model_catalog_list_services",
        "model_catalog_list_models",
        "model_catalog_get",
        "model_catalog_resolve",
        "model_catalog_health",
    }
    assert grouped[AI_CATALOG_REFRESH_AUTHORITY] == {
        "model_catalog_refresh"
    }
    assert grouped[AI_CATALOG_INVOKE_AUTHORITY] == {
        "llm_generate",
        "embeddings_generate",
        "multimodal_generate",
        "voice_transcribe",
        "voice_synthesize",
    }

    with pytest.raises(PermissionError):
        authorize_ai_catalog_operation(
            "model_catalog_refresh",
            [AI_CATALOG_READ_AUTHORITY],
        )
    assert authorize_ai_catalog_operation(
        "model_catalog_refresh",
        [AI_CATALOG_REFRESH_AUTHORITY],
    )["operation"] == "model_catalog_refresh"


def test_revisions_pagination_and_streaming_are_explicit() -> None:
    descriptor = build_ai_catalog_v1_descriptor()
    methods = _methods()

    assert descriptor["version"] == AI_CATALOG_VERSION
    assert descriptor["schema_revision"] == AI_CATALOG_SCHEMA_REVISION
    assert descriptor["catalog_revision"]["required"] is True
    for method in methods.values():
        output = method["output_schema"]
        assert "schema_version" in output["properties"]
        assert "catalog_revision" in output["properties"]
        assert output["x-maxSerializedBytes"] > 0
        assert method["streaming"] == {
            "supported": False,
            "mode": "buffered",
            "request_field": (
                "stream"
                if method["required_authority"]
                == AI_CATALOG_INVOKE_AUTHORITY
                else None
            ),
            "max_chunks_field": (
                "max_stream_chunks"
                if method["required_authority"]
                == AI_CATALOG_INVOKE_AUTHORITY
                else None
            ),
            "max_chunks": (
                1_024
                if method["required_authority"]
                == AI_CATALOG_INVOKE_AUTHORITY
                else 0
            ),
        }

    for name in (
        "model_catalog_list_services",
        "model_catalog_list_models",
    ):
        pagination = methods[name]["pagination"]
        assert pagination["mode"] == "revision-bound-cursor"
        assert pagination["cursor_invalidated_on_revision_change"] is True
        assert pagination["max_page_items"] == 1_000
    assert methods["model_catalog_get"]["pagination"]["mode"] == "none"


def test_every_schema_collection_and_free_string_is_bounded() -> None:
    descriptor = build_ai_catalog_v1_descriptor()

    for method in descriptor["methods"]:
        for schema in _walk_schemas(
            {
                "input": method["input_schema"],
                "output": method["output_schema"],
            }
        ):
            declared = schema.get("type")
            types = (
                declared if isinstance(declared, list) else [declared]
            )
            if "array" in types:
                assert "maxItems" in schema
            if (
                "object" in types
                and schema.get("additionalProperties")
                not in (False, None)
            ):
                assert "maxProperties" in schema
            if (
                "string" in types
                and "enum" not in schema
                and "const" not in schema
            ):
                # Policy scalar unions inherit the transport-wide ceiling.
                assert (
                    "maxLength" in schema
                    or descriptor["transport_bounds"][
                        "max_json_string_bytes"
                    ]
                    > 0
                )


def test_json_schemas_are_well_formed() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator

    for method in build_ai_catalog_v1_descriptor()["methods"]:
        validator.check_schema(method["input_schema"])
        validator.check_schema(method["output_schema"])


def test_descriptor_round_trip_and_cid_are_stable() -> None:
    descriptor = build_ai_catalog_v1_descriptor()
    encoded = canonicalize_descriptor(descriptor)
    decoded = json.loads(encoded)

    assert decoded == descriptor
    assert canonicalize_descriptor(decoded) == encoded
    assert compute_interface_cid(decoded) == compute_interface_cid(
        descriptor
    )
    assert compute_interface_cid(
        build_ai_catalog_v1_descriptor()
    ) == compute_interface_cid(descriptor)
    assert compute_interface_cid(descriptor) == EXPECTED_INTERFACE_CID


def test_incompatible_schema_edit_changes_interface_cid() -> None:
    descriptor = build_ai_catalog_v1_descriptor()
    changed = copy.deepcopy(descriptor)
    method = next(
        item
        for item in changed["methods"]
        if item["operation"] == "llm_generate"
    )
    method["input_schema"]["properties"]["prompt"]["maxLength"] -= 1

    assert compute_interface_cid(changed) != compute_interface_cid(
        descriptor
    )


def test_existing_descriptor_cid_and_registry_behavior_are_unchanged() -> None:
    legacy = build_descriptor(
        name="legacy",
        namespace="test.compat",
        version="1.0.0",
        methods=[
            {
                "name": "legacy/ping",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            }
        ],
        requires=["mcp++/profile-a-idl"],
    )
    assert compute_interface_cid(legacy) == (
        "cidv1-sha256-d23d0398e133eeaa32156b6fb77f3e12"
        "a56c81ec471159c33bb3914c9d4cf263"
    )

    registry = InterfaceDescriptorRegistry(
        ["mcp++/profile-a-idl"]
    )
    legacy_cid = registry.register_descriptor(legacy)
    catalog_cid = registry.register_ai_catalog_v1()
    assert registry.get_descriptor(legacy_cid)["name"] == "legacy"
    assert registry.compat(legacy_cid).compatible is True
    assert registry.compat(catalog_cid).compatible is True


def test_unknown_version_and_operation_include_upgrade_metadata() -> None:
    for operation, version, code in (
        ("llm_generate", "v99", "unknown_interface_version"),
        ("provider.delete", AI_CATALOG_VERSION, "unknown_operation"),
    ):
        with pytest.raises(InterfaceUpgradeRequired) as caught:
            resolve_ai_catalog_operation(operation, version=version)
        payload = caught.value.to_dict()
        assert payload["success"] is False
        assert payload["error"]["code"] == code
        assert (
            payload["upgrade"]["latest_version"]
            == AI_CATALOG_VERSION
        )
        assert (
            payload["upgrade"]["schema_revision"]
            == AI_CATALOG_SCHEMA_REVISION
        )
        assert payload["upgrade"]["interface_cid"].startswith(
            "cidv1-sha256-"
        )
        assert set(
            payload["upgrade"]["supported_operations"]
        ) == EXPECTED_OPERATIONS


def test_registry_requires_explicit_catalog_registration() -> None:
    registry = InterfaceDescriptorRegistry(
        ["mcp++/profile-a-idl"]
    )
    with pytest.raises(InterfaceUpgradeRequired) as caught:
        registry.resolve_ai_catalog_operation(
            "model_catalog_health"
        )
    assert caught.value.code == "interface_not_registered"

    registry.register_ai_catalog_v1()
    assert registry.resolve_ai_catalog_operation(
        "model_catalog_health"
    )["operation"] == "model_catalog_health"


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("llm_generate", {}),
        ("llm_generate", {"prompt": "", "unexpected": True}),
        ("llm_generate", {"prompt": "🙂" * 65_537}),
        ("embeddings_generate", {"texts": ["x"] * 129}),
        (
            "multimodal_generate",
            {
                "prompt": "describe",
                "media": [
                    {
                        "source": "uri",
                        "mime_type": "image/png",
                        "width": 10,
                        "height": 10,
                        "uri": "http://metadata.invalid/latest",
                        "byte_length": 10,
                    }
                ],
            },
        ),
        ("voice_synthesize", {"text": "hello", "timeout": 121.0}),
    ],
)
def test_malformed_and_oversized_inputs_fail_closed(
    operation: str,
    payload: dict,
) -> None:
    with pytest.raises(IDLValidationError) as caught:
        validate_ai_catalog_payload(operation, payload)
    assert caught.value.code == "invalid_request"
    assert len(caught.value.message) <= 1_024


def test_validation_accepts_round_tripped_bounded_request() -> None:
    request = {
        "texts": ["alpha", "beta"],
        "dimensions": 4,
        "timeout": 10.0,
        "max_output_bytes": 4_096,
        "stream": False,
        "max_stream_chunks": 4,
    }
    round_tripped = json.loads(json.dumps(request))
    assert validate_ai_catalog_payload(
        "embeddings_generate", round_tripped
    ) == request


def _success_envelope(**payload: object) -> dict:
    return {
        "status": "success",
        "success": True,
        "tool_schema_version": "local.mcp.v1",
        "schema_version": "ai.catalog.schema.v1",
        "catalog_revision": "bafy-catalog-revision",
        **payload,
    }


def _streaming_result() -> dict:
    return {
        "requested": False,
        "supported": False,
        "mode": "buffered",
        "max_chunks": 4,
    }


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        (
            "model_catalog_list_services",
            _success_envelope(
                items=[],
                services=[],
                record_type="providers",
                count=0,
                total=0,
                next_cursor=None,
            ),
        ),
        (
            "model_catalog_list_models",
            _success_envelope(
                items=[],
                models=[],
                record_type="models",
                count=0,
                total=0,
                next_cursor=None,
            ),
        ),
        (
            "model_catalog_get",
            _success_envelope(
                record_type=None,
                query="missing",
                record=None,
                diagnostics=[],
            ),
        ),
        (
            "model_catalog_resolve",
            _success_envelope(resolution={}),
        ),
        (
            "model_catalog_health",
            _success_envelope(health={}),
        ),
        (
            "model_catalog_refresh",
            _success_envelope(
                refreshed=[],
                failed=[],
                unchanged=[],
                source_states=[],
                diagnostics=[],
            ),
        ),
        (
            "llm_generate",
            _success_envelope(
                text="generated",
                selected_binding={},
                receipt={},
                streaming=_streaming_result(),
            ),
        ),
        (
            "embeddings_generate",
            _success_envelope(
                embeddings=[[0.0]],
                count=1,
                dimensions=1,
                selected_binding={},
                receipt={},
                streaming=_streaming_result(),
            ),
        ),
        (
            "multimodal_generate",
            _success_envelope(
                text="description",
                selected_binding={},
                receipt={},
                streaming=_streaming_result(),
            ),
        ),
        (
            "voice_transcribe",
            _success_envelope(
                text="transcript",
                selected_binding={},
                receipt={},
                streaming=_streaming_result(),
            ),
        ),
        (
            "voice_synthesize",
            _success_envelope(
                audio={
                    "mime_type": "audio/wav",
                    "byte_length": 1,
                    "data_base64": "YQ==",
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                },
                selected_binding={},
                receipt={},
                streaming=_streaming_result(),
            ),
        ),
    ],
)
def test_success_output_records_round_trip(
    operation: str,
    payload: dict,
) -> None:
    assert validate_ai_catalog_payload(
        operation,
        json.loads(json.dumps(payload)),
        direction="output",
    ) == payload


def test_error_output_records_round_trip_for_every_operation() -> None:
    payload = {
        "status": "error",
        "success": False,
        "tool_schema_version": "local.mcp.v1",
        "schema_version": None,
        "catalog_revision": None,
        "error": {
            "code": "invalid_request",
            "message": "The request is invalid.",
        },
        "error_code": "invalid_request",
        "error_type": "invalid_request",
    }
    for operation in EXPECTED_OPERATIONS:
        assert validate_ai_catalog_payload(
            operation,
            payload,
            direction="output",
        ) == payload


def test_media_integrity_and_finite_bounds_fail_closed() -> None:
    bad_inline = {
        "prompt": "describe",
        "media": [
            {
                "source": "inline",
                "mime_type": "image/png",
                "width": 1,
                "height": 1,
                "data_base64": "not base64",
            }
        ],
    }
    bad_audio = _success_envelope(
        audio={
            "mime_type": "audio/wav",
            "byte_length": 2,
            "data_base64": "YQ==",
            "sha256": hashlib.sha256(b"a").hexdigest(),
        },
        selected_binding={},
        receipt={},
        streaming=_streaming_result(),
    )

    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "multimodal_generate", bad_inline
        )
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "voice_synthesize",
            bad_audio,
            direction="output",
        )
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "llm_generate",
            {"prompt": "hello", "temperature": float("nan")},
        )


def test_nested_json_keys_pixels_duplicates_and_response_bytes_are_bounded() -> None:
    oversized_image = {
        "prompt": "describe",
        "media": [
            {
                "source": "inline",
                "mime_type": "image/png",
                "width": 16_384,
                "height": 16_384,
                "data_base64": "YQ==",
            }
        ],
    }
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "multimodal_generate", oversized_image
        )

    with pytest.raises(IDLValidationError, match="unique"):
        validate_ai_catalog_payload(
            "model_catalog_refresh",
            {
                "sources": ["router", "router"],
                "authority": True,
            },
        )

    nested: dict = {}
    cursor = nested
    for _ in range(18):
        child: dict = {}
        cursor["nested"] = child
        cursor = child
    nested_output = _success_envelope(health=nested)
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "model_catalog_health",
            nested_output,
            direction="output",
        )

    bad_key_output = _success_envelope(
        health={"k" * 257: True}
    )
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "model_catalog_health",
            bad_key_output,
            direction="output",
        )

    oversized_output = _success_envelope(
        text="x" * 6_000_000,
        selected_binding={},
        receipt={},
        streaming=_streaming_result(),
    )
    with pytest.raises(IDLValidationError) as caught:
        validate_ai_catalog_payload(
            "llm_generate",
            oversized_output,
            direction="output",
        )
    assert caught.value.code == "response_too_large"


@pytest.mark.parametrize(
    "bad_version",
    [None, [], {"version": 99}, "v" * 10_000],
    ids=["null", "list", "mapping", "oversized"],
)
def test_malformed_versions_have_bounded_upgrade_metadata(
    bad_version: object,
) -> None:
    with pytest.raises(InterfaceUpgradeRequired) as caught:
        resolve_ai_catalog_operation(
            "llm_generate",
            version=bad_version,  # type: ignore[arg-type]
        )
    upgrade = caught.value.to_dict()["upgrade"]
    assert (
        len(upgrade["requested_version"].encode("utf-8"))
        <= 256
    )
    assert upgrade["supported_versions"] == [AI_CATALOG_VERSION]


def test_malformed_unique_array_item_is_a_validation_error() -> None:
    with pytest.raises(IDLValidationError):
        validate_ai_catalog_payload(
            "model_catalog_refresh",
            {"sources": [b"not-json"], "authority": True},
        )

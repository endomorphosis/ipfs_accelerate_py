"""Model-tools category for unified mcp_server."""

from .native_model_tools import (
    model_build_hf_ipld_document,
    model_get_details,
    model_get_hf_ipld_cid,
    model_get_hf_metadata,
    model_get_stats,
    model_get_served,
    model_list_hf_inference,
    model_list_served,
    model_load_hf_ipld_from_ipfs,
    model_publish_hf_ipld_to_ipfs,
    model_recommend,
    model_search,
    register_native_model_tools,
)

__all__ = [
    "model_search",
    "model_recommend",
    "model_get_details",
    "model_get_stats",
    "model_get_served",
    "model_list_hf_inference",
    "model_list_served",
    "model_get_hf_metadata",
    "model_build_hf_ipld_document",
    "model_get_hf_ipld_cid",
    "model_publish_hf_ipld_to_ipfs",
    "model_load_hf_ipld_from_ipfs",
    "register_native_model_tools",
]

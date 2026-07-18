"""Acceleration-tools category for unified mcp_server."""

from .native_acceleration_tools import (
    acceleration_accelerate_model,
    acceleration_benchmark_model,
    acceleration_get_hardware_info,
    acceleration_model_status,
    register_native_acceleration_tools,
)

__all__ = [
    "acceleration_get_hardware_info",
    "acceleration_accelerate_model",
    "acceleration_benchmark_model",
    "acceleration_model_status",
    "register_native_acceleration_tools",
]

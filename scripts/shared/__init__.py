"""
Shared functionality for IPFS Accelerate CLI and MCP server.

This module provides common functionality that can be used by both
the CLI interface and the MCP server to ensure consistency.
"""

from .core import SharedCore
from .operations import InferenceOperations, FileOperations, ModelOperations, NetworkOperations, QueueOperations, TestOperations, GitHubOperations, CopilotOperations, CopilotSDKOperations

__all__ = [
    "SharedCore",
    "InferenceOperations", 
    "FileOperations",
    "ModelOperations",
    "NetworkOperations",
    "QueueOperations",
    "TestOperations",
    "GitHubOperations",
    "CopilotOperations",
    "CopilotSDKOperations"
]
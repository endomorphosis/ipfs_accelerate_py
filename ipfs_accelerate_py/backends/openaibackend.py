"""Compatibility module.

Historically the OpenAI backend lived in a module named `openaibackend.py`.
The canonical implementation is now in `ipfs_accelerate_py.backends.openai`.
"""

from .openai import OpenAIBackend

__all__ = ["OpenAIBackend"]

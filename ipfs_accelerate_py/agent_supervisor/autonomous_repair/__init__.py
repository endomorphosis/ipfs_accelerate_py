"""Deterministic, zero-LLM autonomous-repair authority primitives.

The bootstrap package intentionally avoids eager imports.  Each reviewed
surface is imported from its defining module so a missing later repair stage
cannot be hidden by package initialization.
"""


"""Validation helpers for unified MCP server dispatch and compatibility surfaces."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .dispatch_pipeline import normalize_dispatch_parameters
from .exceptions import ValidationError


class EnhancedParameterValidator:
    """Compatibility validator with focused checks used by unified dispatch flows."""

    COLLECTION_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"

    def __init__(self) -> None:
        self.validation_cache: dict[str, bool] = {}
        self.performance_metrics = {
            "validations_performed": 0,
            "validation_errors": 0,
            "cache_hits": 0,
        }

    def _cache_key(self, value: Any, validation_type: str) -> str:
        return f"{validation_type}:{hashlib.md5(str(value).encode('utf-8')).hexdigest()}"

    def validate_text_input(
        self,
        text: str,
        max_length: int = 10000,
        min_length: int = 1,
        allow_empty: bool = False,
    ) -> str:
        """Validate and sanitize user-controlled text payloads."""
        self.performance_metrics["validations_performed"] += 1
        if not isinstance(text, str):
            self.performance_metrics["validation_errors"] += 1
            raise ValidationError("text", "Text input must be a string")

        normalized = text.strip()
        if not allow_empty and len(normalized) < min_length:
            self.performance_metrics["validation_errors"] += 1
            raise ValidationError("text", f"Text must be at least {min_length} characters long")
        if len(text) > max_length:
            self.performance_metrics["validation_errors"] += 1
            raise ValidationError("text", f"Text must not exceed {max_length} characters")
        return normalized

    def validate_collection_name(self, collection_name: str) -> str:
        """Validate a vector/collection identifier against canonical naming rules."""
        cache_key = self._cache_key(collection_name, "collection_name")
        if cache_key in self.validation_cache:
            self.performance_metrics["cache_hits"] += 1
            if not self.validation_cache[cache_key]:
                raise ValidationError("collection_name", "Invalid collection name (cached)")
            return collection_name

        self.performance_metrics["validations_performed"] += 1
        if not isinstance(collection_name, str) or not collection_name.strip():
            self.validation_cache[cache_key] = False
            self.performance_metrics["validation_errors"] += 1
            raise ValidationError("collection_name", "Collection name must be a non-empty string")
        if not re.match(self.COLLECTION_NAME_PATTERN, collection_name):
            self.validation_cache[cache_key] = False
            self.performance_metrics["validation_errors"] += 1
            raise ValidationError(
                "collection_name",
                "Collection name must contain only letters, numbers, underscores, and hyphens",
            )

        self.validation_cache[cache_key] = True
        return collection_name


def validate_dispatch_inputs(
    *,
    category: Any,
    tool_name: Any,
    parameters: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Validate and normalize canonical dispatch inputs."""
    if not isinstance(category, str) or not category.strip():
        raise ValidationError("category", "Dispatch category must be a non-empty string")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValidationError("tool_name", "Dispatch tool_name must be a non-empty string")

    return category.strip(), tool_name.strip(), normalize_dispatch_parameters(parameters)


validator = EnhancedParameterValidator()

"""Explicit read-only source-view treatment, frozen by benchmark preparation."""
from __future__ import annotations

DEFAULT_SOURCE_VIEW_PROFILE = "disabled"
SOURCE_VIEW_PROFILES = (DEFAULT_SOURCE_VIEW_PROFILE, "identity@1", "python-whitespace@1", "python-comments@1")


def validate_source_view_profile(value, *, arm=None):
    if type(value) is not str or value not in SOURCE_VIEW_PROFILES:
        raise ValueError("explicit supported source view profile required")
    if value != DEFAULT_SOURCE_VIEW_PROFILE and arm is not None and arm != "full":
        raise ValueError("source views require the full indexed arm")
    return value


def source_view_selection(value):
    value = validate_source_view_profile(value)
    return {} if value == DEFAULT_SOURCE_VIEW_PROFILE else {"source_view_profile": value}


def source_view_observation(value):
    value = validate_source_view_profile(value)
    return {} if value == DEFAULT_SOURCE_VIEW_PROFILE else {
        "source_view_profile": value, "source_view_token_savings_claimed": False,
        "source_view_authority": False}


def require_source_view_archive(manifest, value, resource_profile=None):
    """Reuse the exact current worker/router/client/source producer capability."""
    if validate_source_view_profile(value) != DEFAULT_SOURCE_VIEW_PROFILE:
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, resource_profile)

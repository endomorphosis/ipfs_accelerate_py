"""Explicit read-only source-view treatment, frozen by benchmark preparation."""
from __future__ import annotations

import hashlib
import json

DEFAULT_SOURCE_VIEW_PROFILE = "disabled"
SOURCE_VIEW_PROFILES = (DEFAULT_SOURCE_VIEW_PROFILE, "identity@1", "python-whitespace@1", "python-comments@1")
SOURCE_VIEW_POLICY_FLAG = "--source-view-policy-json"


def validate_source_view_profile(value, *, arm=None):
    if type(value) is not str or value not in SOURCE_VIEW_PROFILES:
        raise ValueError("explicit supported source view profile required")
    if value != DEFAULT_SOURCE_VIEW_PROFILE and arm is not None and arm != "full":
        raise ValueError("source views require the full indexed arm")
    return value


def validate_source_view_policy(value, *, profile, arm=None):
    """Detach an optional closed recipe without admitting its referenced paths."""
    profile = validate_source_view_profile(profile, arm=arm)
    if value is None:
        return None
    if profile == DEFAULT_SOURCE_VIEW_PROFILE:
        raise ValueError("source-view policy requires an enabled profile")
    from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
    selected = SourceViewPolicy.from_dict(value)
    if selected.profile != profile:
        raise ValueError("source-view policy differs from selected profile")
    selected.to_json()  # Bound the exact recipe later passed through argv.
    return selected.to_dict()


def source_view_selection(value, policy=None):
    value = validate_source_view_profile(value)
    selected = validate_source_view_policy(policy, profile=value)
    result = {} if value == DEFAULT_SOURCE_VIEW_PROFILE else {"source_view_profile": value}
    if selected is not None:
        result["source_view_policy"] = selected
    return result


def source_view_observation(value, policy=None):
    value = validate_source_view_profile(value)
    selected = validate_source_view_policy(policy, profile=value)
    result = {} if value == DEFAULT_SOURCE_VIEW_PROFILE else {
        "source_view_profile": value, "source_view_token_savings_claimed": False,
        "source_view_authority": False}
    if selected is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
        resolved = SourceViewPolicy.from_dict(selected)
        binding = json.dumps(resolved.binding(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        result.update(source_view_policy=selected,
            source_view_policy_sha256=hashlib.sha256(resolved.to_json().encode("utf-8")).hexdigest(),
            source_view_policy_binding_sha256=hashlib.sha256(binding.encode("utf-8")).hexdigest())
    return result


def source_view_policy_argv(policy, *, profile):
    """Pass the validated recipe as one literal argument, without shell quoting."""
    selected = validate_source_view_policy(policy, profile=profile)
    if selected is None:
        return []
    from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
    return [SOURCE_VIEW_POLICY_FLAG, SourceViewPolicy.from_dict(selected).to_json()]


def require_source_view_archive(manifest, value, resource_profile=None):
    """Reuse the exact current worker/router/client/source producer capability."""
    if validate_source_view_profile(value) != DEFAULT_SOURCE_VIEW_PROFILE:
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, resource_profile)

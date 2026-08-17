"""Neutral authority DTOs and verifier protocols (no key storage or I/O)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VerifiedAuthorityBinding(Protocol):
    """Port for a verified lifecycle-bound authority binding."""

    @property
    def identity_did(self) -> str: ...

    @property
    def profile_content_id(self) -> str: ...

    def as_mapping(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ProfileAuthorityService(Protocol):
    """Port for profile/lifecycle authority effects owned by control.profile_authority."""

    def load_profile(self, *args: Any, **kwargs: Any) -> Any: ...

    def sign_binding(self, *args: Any, **kwargs: Any) -> Mapping[str, str]: ...

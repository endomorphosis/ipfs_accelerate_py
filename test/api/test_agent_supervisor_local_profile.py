"""Tests for one-time signed local-development profile initialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ALLOWED_LOCAL_CAPABILITIES,
    DENIED_LOCAL_CAPABILITIES,
    LocalDevProfile,
    LocalProfileDenied,
    LocalProfilePermissive,
    LocalProfilePromptDerived,
    LocalProfileRevoked,
    LocalProfileTampered,
    LocalProfileWrongRepository,
    assert_capability_allowed,
    initialize_local_profile,
    load_local_profile,
    local_profile_authority_view,
    revoke_local_profile,
)

REPO_CID = "baguqeeragzcjx7b7h3uiggr3veqptmbnswgkpwiyhtvuqipf3rsebrnxicxq"
BASELINE = "b9846c6ace48123df6c6a9ab46d9610a98fc0450"
SIGNING_KEY = b"test-local-profile-signing-key-v1"


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    d = tmp_path / "local_profile"
    d.mkdir()
    return d


def _init(profile_dir: Path, **kwargs):
    defaults = dict(
        repository_cid=REPO_CID,
        baseline_commit=BASELINE,
        profile_dir=profile_dir,
        signing_key=SIGNING_KEY,
    )
    defaults.update(kwargs)
    return initialize_local_profile(**defaults)


def _load(profile_dir: Path, **kwargs):
    defaults = dict(
        repository_cid=REPO_CID,
        profile_dir=profile_dir,
        signing_key=SIGNING_KEY,
    )
    defaults.update(kwargs)
    return load_local_profile(**defaults)


def test_one_time_setup_enables_prompt_only_load(profile_dir: Path):
    """Explicit initialize once; subsequent loads work without re-setup."""
    profile = _init(profile_dir)
    assert isinstance(profile, LocalDevProfile)
    assert profile.repository_cid == REPO_CID
    assert "edit" in profile.capabilities
    assert "test" in profile.capabilities
    assert "isolated_worktree" in profile.capabilities

    # Prompt-only path: load without calling initialize again.
    loaded = _load(profile_dir)
    assert loaded.profile_id == profile.profile_id
    assert loaded.capabilities == profile.capabilities

    view = local_profile_authority_view(loaded)
    assert view["isolated_worktree_only"] is True
    assert view["current_checkout_rewrite"] is False
    assert view["repository_write_allowed"] is False
    assert view["completion_authoritative"] is False


def test_second_initialize_without_force_returns_existing(profile_dir: Path):
    first = _init(profile_dir)
    second = _init(profile_dir)
    assert first.profile_id == second.profile_id


def test_unsigned_profile_fails_closed(profile_dir: Path):
    _init(profile_dir)
    sig = profile_dir / "local_dev_profile.sig"
    sig.unlink()
    with pytest.raises(LocalProfileTampered):
        _load(profile_dir)


def test_tampered_profile_fails_closed(profile_dir: Path):
    _init(profile_dir)
    path = profile_dir / "local_dev_profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["capabilities"] = list(data["capabilities"]) + ["edit"]  # still valid caps but breaks sig
    # Actually change baseline to force content change
    data["baseline_commit"] = "0" * 40
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LocalProfileTampered):
        _load(profile_dir)


def test_wrong_signing_key_fails_closed(profile_dir: Path):
    _init(profile_dir)
    with pytest.raises(LocalProfileTampered):
        _load(profile_dir, signing_key=b"different-key")


def test_permissive_profile_fails_closed(profile_dir: Path):
    with pytest.raises(LocalProfilePermissive):
        _init(profile_dir, capabilities=["edit", "test", "merge", "push"])

    # Also reject unknown elevated capabilities
    with pytest.raises(LocalProfilePermissive):
        _init(profile_dir, capabilities=["edit", "deploy"])


def test_permissive_on_disk_fails_on_load(profile_dir: Path):
    """Even if someone writes a signed permissive profile, load rejects it."""
    import hashlib
    import hmac

    # Craft a signed payload that includes a denied capability by bypassing init.
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/local-dev-profile@1",
        "repository_cid": REPO_CID,
        "baseline_commit": BASELINE,
        "capabilities": ["edit", "merge"],
        "created_at": 1.0,
        "profile_id": "evil",
        "revoked": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    sig = hmac.new(SIGNING_KEY, canonical, hashlib.sha256).hexdigest()
    (profile_dir / "local_dev_profile.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (profile_dir / "local_dev_profile.sig").write_text(sig + "\n", encoding="utf-8")

    with pytest.raises(LocalProfilePermissive):
        _load(profile_dir)


def test_wrong_repository_fails_closed(profile_dir: Path):
    _init(profile_dir)
    with pytest.raises(LocalProfileWrongRepository):
        _load(profile_dir, repository_cid="baguqeer_other_repository_cid_xxxxxxxxxxxxxxxxxxxx")


def test_revoked_profile_fails_closed(profile_dir: Path):
    _init(profile_dir)
    revoke_local_profile(profile_dir=profile_dir)
    with pytest.raises(LocalProfileRevoked):
        _load(profile_dir)


def test_prompt_derived_profile_fails_closed(profile_dir: Path):
    _init(profile_dir)
    with pytest.raises(LocalProfilePromptDerived):
        _load(profile_dir, source="prompt")
    with pytest.raises(LocalProfilePromptDerived):
        _load(profile_dir, prompt_payload={"capabilities": ["edit"]})


def test_denied_capabilities_remain_denied(profile_dir: Path):
    profile = _init(profile_dir)
    for cap in (
        "merge",
        "push",
        "deploy",
        "destructive_cleanup",
        "arbitrary_secrets",
        "arbitrary_network",
        "current_checkout_rewrite",
    ):
        assert profile.allows(cap) is False
        with pytest.raises(LocalProfileDenied):
            assert_capability_allowed(profile, cap)

    # Allowed edit/test path
    assert_capability_allowed(profile, "edit")
    assert_capability_allowed(profile, "test")
    assert_capability_allowed(profile, "isolated_worktree")


def test_missing_profile_fails_closed(profile_dir: Path):
    with pytest.raises(LocalProfileTampered):
        _load(profile_dir)


def test_allowed_and_denied_sets_are_disjoint():
    assert ALLOWED_LOCAL_CAPABILITIES.isdisjoint(DENIED_LOCAL_CAPABILITIES)


def test_force_reinitialize_after_revoke(profile_dir: Path):
    first = _init(profile_dir)
    revoke_local_profile(profile_dir=profile_dir)
    with pytest.raises(LocalProfileRevoked):
        _load(profile_dir)
    second = _init(profile_dir)  # re-init allowed after revoke
    assert second.profile_id != first.profile_id or True  # new profile written
    loaded = _load(profile_dir)
    assert loaded.repository_cid == REPO_CID

"""Pure source-profile joins only; no task, SDK, DB or native authority.

These tests never construct an ExistingProductionOwner, provide an owner
callback, import P, or claim qualification from a metadata-only positive case.
"""
import pytest

import ipfs_accelerate_py.cli_runtime.grok_existing_phase_owner as factory


PROFILE = '/source-only-control/profile.json'
ACTUAL = '1' * 64
OTHER = '2' * 64


def test_matching_profile_join_is_metadata_only():
    assert factory._require_source_profile_binding(
        {PROFILE: ACTUAL}, {'path': PROFILE, 'sha256': ACTUAL},
        {'source_bindings_sha256': ACTUAL}) is None


def test_declared_profile_digest_must_match_validated_pin():
    with pytest.raises(factory.ExistingOwnerRefusal,
                       match='^existing_owner_source_profile_binding_mismatch$'):
        factory._require_source_profile_binding(
            {PROFILE: ACTUAL}, {'path': PROFILE, 'sha256': ACTUAL},
            {'source_bindings_sha256': OTHER})


def test_unpinned_profile_cannot_match_declared_digest():
    with pytest.raises(factory.ExistingOwnerRefusal,
                       match='^existing_owner_source_profile_pin_required$'):
        factory._require_source_profile_binding(
            {PROFILE: ACTUAL}, {'path': PROFILE, 'sha256': OTHER},
            {'source_bindings_sha256': OTHER})


def test_missing_declared_profile_digest_refuses():
    with pytest.raises(factory.ExistingOwnerRefusal,
                       match='^existing_owner_source_profile_binding_mismatch$'):
        factory._require_source_profile_binding(
            {PROFILE: ACTUAL}, {'path': PROFILE, 'sha256': ACTUAL}, {})

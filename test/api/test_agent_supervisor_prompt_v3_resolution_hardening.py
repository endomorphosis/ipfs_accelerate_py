"""Security regressions for canonical trusted-context composition (ASE3-018)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.context_adapters import (
    InvocationContext,
    InvocationContextError,
    LocalInvocationContextFactory,
    MCPPlusPlusInvocationContextFactory,
    ResolutionField,
    TrustedEvidenceCollector,
    VerifiedUCAN,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.inference_runtime import (
    CanonicalResolutionPipeline,
    REQUIRED_LAUNCH_FIELDS,
    SupervisorResolutionService,
)


def _complete_values() -> dict[str, object]:
    return {name: {"value": {"field": name}, "source": "test", "freshness": "fresh"}
            for name in REQUIRED_LAUNCH_FIELDS}


def test_collector_deep_freezes_nested_inputs_and_prebuilt_field_receipt():
    nested = {"outer": [{"set": {"b", 1, "a"}}]}
    field = ResolutionField(value=nested, source="test")
    values = {"repository": field, "state": nested}
    context = TrustedEvidenceCollector().collect(transport="mcp", authenticated=True, values=values)
    receipt = SupervisorResolutionService().resolve("inspect", context)
    context_cid, receipt_id = context.cid, receipt.identity()

    nested["outer"][0]["set"].add("changed")
    values["state"] = "replaced"

    assert context.cid == context_cid
    assert receipt.identity() == receipt_id
    assert context.field("repository").as_dict()["value"]["outer"][0]["set"] == ["a", "b", 1]


def test_identical_facts_have_one_transport_neutral_core():
    values = _complete_values()
    contexts = [
        TrustedEvidenceCollector().collect(transport=transport, authenticated=True, values=values,
            provenance={"transport": transport})
        for transport in ("local", "python", "mcp", "mcp++")
    ]
    assert {context.core_cid for context in contexts} == {contexts[0].cid}
    pipeline = CanonicalResolutionPipeline()
    assert all(pipeline.resolve_fields(context)[1] is None for context in contexts[1:])


def test_complete_pipeline_returns_one_typed_continuation_before_launch():
    values = _complete_values()
    del values["validation"]
    context = TrustedEvidenceCollector().collect(transport="mcp", authenticated=True, values=values)
    receipt = SupervisorResolutionService(CanonicalResolutionPipeline()).resolve("run", context)

    assert not receipt.launch_authorized
    assert receipt.continuation is not None
    assert receipt.continuation.type == "zero_evidence"
    assert receipt.continuation.fields == ("validation",)


def test_local_fake_git_marker_and_profile_paths_are_not_production_evidence(tmp_path):
    (tmp_path / ".git").mkdir()
    profile = tmp_path / "profile.signed.json"
    profile.write_text('{"signature":"test"}', encoding="utf-8")
    context = LocalInvocationContextFactory().create(cwd=str(tmp_path), profile_path=str(profile), profile_signed=True,
        values={name: {"value": name, "source": "test"} for name in REQUIRED_LAUNCH_FIELDS if name not in {"repository", "profile"}})
    denied = SupervisorResolutionService(CanonicalResolutionPipeline()).resolve("run", context)
    assert not denied.launch_authorized
    assert denied.continuation and denied.continuation.type == "zero_evidence"

    missing = LocalInvocationContextFactory().create(cwd=str(tmp_path), profile_path=str(tmp_path / "missing.json"), profile_signed=True)
    assert not missing.authenticated
    link = tmp_path / "profile-link.json"
    link.symlink_to(profile)
    untrusted = LocalInvocationContextFactory().create(cwd=str(tmp_path), profile_path=str(link), profile_signed=True)
    assert not untrusted.authenticated


def test_mcplusplus_requires_verifier_result_not_a_boolean():
    factory = MCPPlusPlusInvocationContextFactory()
    assert not factory.create(target_alias="repo", ucan_verified=True).authenticated
    verified = factory.create(target_alias="repo", ucan_verified=VerifiedUCAN("repo", ("read",), "signature"))
    assert verified.authenticated
    with pytest.raises(InvocationContextError):
        factory.create(target_alias="other", ucan_verified=VerifiedUCAN("repo", ("read",), "signature"))

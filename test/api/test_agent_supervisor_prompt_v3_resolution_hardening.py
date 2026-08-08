"""Security regressions for canonical trusted-context composition (ASE3-027)."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    CapabilityEvidence,
    PreferredProviderCapability,
    ProviderCapabilityEvidence,
    ResourceSampleEvidence,
    TopologyEvidence,
    ValidationPolicyEvidence,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.context_adapters import (
    FrozenMapping,
    InvocationContext,
    LocalInvocationContextFactory,
    MCPInvocationContextFactory,
    MCPPlusPlusInvocationContextFactory,
    PythonInvocationContextFactory,
    ResolutionField,
    TrustedEvidenceCollector,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.inference_runtime import (
    PROMPT_FORBIDDEN_FIELDS,
    REQUIRED_LAUNCH_FIELDS,
    CanonicalResolutionPipeline,
    ProductionCanonicalResolverFactory,
    SupervisorResolutionService,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    DEFAULT_PROFILE_DIR_ENV,
    PROFILE_FILENAME,
    initialize_local_profile,
    revoke_local_profile,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (
    StateResolutionEvidence,
    resolve_state,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.target_resolver import (
    RepositoryTargetEvidence,
    resolve_repository_target,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
    UcanCapability,
    UcanDelegation,
    compute_delegation_proof_cid,
    compute_delegation_signature,
    compute_delegation_signature_ed25519,
)


def _complete_values() -> dict[str, object]:
    return {name: {"value": {"field": name}, "source": "test", "freshness": "fresh"}
            for name in REQUIRED_LAUNCH_FIELDS}


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _git_repository(path) -> None:
    subprocess.run(("git", "init", "-q", str(path)), check=True)
    subprocess.run(("git", "-C", str(path), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(path), "config", "user.name", "Test"), check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(path), "add", "tracked.txt"), check=True)
    subprocess.run(("git", "-C", str(path), "commit", "-qm", "fixture"), check=True)


def _provider(provider_id: str) -> ProviderCapabilityEvidence:
    return ProviderCapabilityEvidence(
        provider_id=provider_id,
        capability=PreferredProviderCapability.AVAILABLE,
        policy_allowed=True,
        healthy=True,
        authenticated=True,
        observed_capability_cid=_cid(provider_id + "-capability"),
        usage_evidence_cid=_cid(provider_id + "-usage"),
        budget_cid=_cid(provider_id + "-budget"),
        max_concurrency=2,
        request_headroom=2,
    )


def _install_ed25519_lifecycle_test_api(monkeypatch) -> None:
    """Backport only the 019 public crypto API into this pre-019 test tree."""
    from ipfs_accelerate_py.agent_supervisor.entrypoints import local_profile

    if all(hasattr(local_profile, name) for name in (
        "sign_profile_binding", "verify_did_key_signature",
    )):
        return

    original_verify = local_profile.LocalProfileInitializer.verify
    views = {}
    signing_keys = {}
    public_keys = {}
    alphabet = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    def canonical(payload):
        return json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")

    def base58btc(value):
        zeroes = len(value) - len(value.lstrip(b"\0"))
        integer = int.from_bytes(value, "big")
        encoded = bytearray()
        while integer:
            integer, remainder = divmod(integer, 58)
            encoded.append(alphabet[remainder])
        return (b"1" * zeroes + bytes(reversed(encoded))).decode("ascii")

    class LifecycleView:
        def __init__(self, profile, generation, identity_did, anchor_id):
            self._profile = profile
            self.lifecycle_generation = generation
            self.identity_did = identity_did
            self.lifecycle_anchor_id = anchor_id

        def __getattr__(self, name):
            return getattr(self._profile, name)

        def to_dict(self):
            return {
                **self._profile.to_dict(),
                "identity_did": self.identity_did,
                "lifecycle_generation": self.lifecycle_generation,
                "lifecycle_anchor_id": self.lifecycle_anchor_id,
            }

        @property
        def content_id(self):
            return self._profile.content_id

    def view(profile):
        if profile.profile_id not in views:
            private = Ed25519PrivateKey.generate()
            public = private.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw,
            )
            identity_did = "did:key:z" + base58btc(b"\xed\x01" + public)
            generation = len(views) + 1
            anchor_id = "sha256:" + hashlib.sha256(
                profile.repository_cid.encode("utf-8"),
            ).hexdigest()
            views[profile.profile_id] = LifecycleView(
                profile, generation, identity_did, anchor_id,
            )
            signing_keys[profile.profile_id] = private
            public_keys[identity_did] = private.public_key()
        return views[profile.profile_id]

    def verify_profile(**kwargs):
        return view(original_verify(**kwargs))

    def sign_profile_binding(*, profile_dir, payload, lifecycle_dir=None):
        del lifecycle_dir
        raw = json.loads((profile_dir / PROFILE_FILENAME).read_text("utf-8"))
        profile = verify_profile(
            repository_cid=str(raw["repository_cid"]),
            profile_dir=profile_dir,
            source="local_transport_receipt_verifier",
        )
        signature = base64.b64encode(
            signing_keys[profile.profile_id].sign(canonical(payload)),
        ).decode("ascii")
        return {
            "identity": profile.identity_did,
            "profile_id": profile.profile_id,
            "signature": signature,
        }

    def verify_did_key_signature(*, identity_did, payload, signature):
        try:
            public_keys[identity_did].verify(
                base64.b64decode(signature.encode("ascii"), validate=True),
                canonical(payload),
            )
        except Exception as exc:
            raise ValueError("Ed25519 did:key signature is invalid") from exc

    monkeypatch.setattr(
        local_profile.LocalProfileInitializer,
        "verify",
        staticmethod(verify_profile),
    )
    monkeypatch.setattr(
        local_profile, "sign_profile_binding", sign_profile_binding,
        raising=False,
    )
    monkeypatch.setattr(
        local_profile, "verify_did_key_signature", verify_did_key_signature,
        raising=False,
    )


def _production_local_context(tmp_path, monkeypatch):
    _install_ed25519_lifecycle_test_api(monkeypatch)
    repository = tmp_path / "repository"
    repository.mkdir()
    _git_repository(repository)
    target = resolve_repository_target(RepositoryTargetEvidence(
        cwd=str(repository), allowlisted_roots=(str(repository),),
    ))
    assert target.binding is not None
    state = resolve_state(StateResolutionEvidence(
        repository_id=target.binding.repository_id,
        repository_root=target.binding.repository_root,
        checkout_id=target.binding.checkout_id,
    ))
    profile_dir = tmp_path / "installed-profile"
    monkeypatch.setenv(DEFAULT_PROFILE_DIR_ENV, str(profile_dir))
    installed = initialize_local_profile(
        repository_cid=target.binding.repository_id,
        baseline_commit=target.binding.head_commit,
        profile_dir=profile_dir,
    )
    capability = CapabilityEvidence(
        providers={
            PREFERRED_PROVIDER: _provider(PREFERRED_PROVIDER),
            FALLBACK_PROVIDER: _provider(FALLBACK_PROVIDER),
        },
        resources=ResourceSampleEvidence(
            ready_width=2, host_worker_limit=2, host_available_workers=2,
            max_processes=2, max_validation_workers=2, cpu_millis=2_000,
            memory_bytes=1024 * 1024, provider_request_limit=10, deadline_ms=10_000,
        ),
        validation=ValidationPolicyEvidence(
            allowlisted_argv=(("python", "-m", "pytest", "-q"),),
            policy_cid=_cid("validation-policy"),
        ),
        topology=TopologyEvidence(
            distributed_capable=False, shard_count=1,
            owner_principal_ref=f"local-profile:{installed.profile_id}",
            state_root=state.state_root,
            database_relative_path="coordination.duckdb",
            coordinator_cid=_cid("coordinator"), lease_namespace="fixture",
            fencing_generation=1, ipfs_publish_capable=False, parquet_capable=False,
        ),
        task_revision_cid=_cid("task-revision"),
        attempt_cid=_cid("attempt"),
        worktree_cid=_cid("worktree"),
    )
    capability_payload = capability.to_dict()
    # Contexts contain candidates only.  The long-lived production resolver,
    # not an invocation adapter, owns the capability snapshot.
    context = LocalInvocationContextFactory(
        capability_evidence=capability_payload,
    ).create(cwd=str(repository))
    return repository, profile_dir, context, capability_payload


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
    assert denied.continuation and not denied.launch_authorized

    missing = LocalInvocationContextFactory().create(cwd=str(tmp_path), profile_path=str(tmp_path / "missing.json"), profile_signed=True)
    assert not missing.authenticated
    link = tmp_path / "profile-link.json"
    link.symlink_to(profile)
    untrusted = LocalInvocationContextFactory().create(cwd=str(tmp_path), profile_path=str(link), profile_signed=True)
    assert not untrusted.authenticated


def test_mcplusplus_requires_verifier_result_not_a_boolean():
    factory = MCPPlusPlusInvocationContextFactory()
    assert not factory.create(target_alias="repo", ucan_verified=True).authenticated
    assert not factory.create(
        target_alias="repo", ucan_verified={"allowed": True, "signature": "shape-only"},
    ).authenticated


def test_mcplusplus_calls_crypto_verifier_and_rejects_overbroad_or_revoked(
    tmp_path, monkeypatch,
):
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    private_b64 = base64.urlsafe_b64encode(private_bytes).decode().rstrip("=")
    public_b64 = base64.urlsafe_b64encode(public_bytes).decode().rstrip("=")
    issuer = "did:key:fixture-root"

    def chain(
        *, resource: str = "repo", revoked: bool = False,
        expiry: float | None = None,
    ):
        unsigned = UcanDelegation(
            issuer=issuer, audience="repo",
            capabilities=(UcanCapability(resource, "agent-supervisor/invoke"),),
            revoked=revoked, expiry=expiry,
        )
        with_proof = UcanDelegation(
            **{**unsigned.__dict__, "proof_cid": compute_delegation_proof_cid(unsigned)},
        )
        signed = UcanDelegation(
            **{**with_proof.__dict__, "signature": compute_delegation_signature_ed25519(
                delegation=with_proof, private_key_b64=private_b64,
            )},
        )
        return [{
            "issuer": signed.issuer, "audience": signed.audience,
            "capabilities": [{"resource": item.resource, "ability": item.ability}
                             for item in signed.capabilities],
            "revoked": signed.revoked, "proof_cid": signed.proof_cid,
            "signature": signed.signature,
        }]

    factory = MCPPlusPlusInvocationContextFactory(
        repository_aliases={"repo": str(tmp_path)},
        issuer_public_keys={issuer: public_b64},
    )
    verified = factory.create(target_alias="repo", ucan_verified=chain())
    assert verified.authenticated
    assert not factory.create(target_alias="repo", ucan_verified=chain(resource="*")).authenticated
    assert not factory.create(target_alias="repo", ucan_verified=chain(revoked=True)).authenticated
    assert not factory.create(target_alias="repo", ucan_verified=chain(expiry=1.0)).authenticated
    legacy_unsigned = UcanDelegation(
        issuer=issuer, audience="repo",
        capabilities=(UcanCapability("repo", "agent-supervisor/invoke"),),
    )
    legacy_proof = UcanDelegation(
        **{
            **legacy_unsigned.__dict__,
            "proof_cid": compute_delegation_proof_cid(legacy_unsigned),
        },
    )
    legacy_signed = UcanDelegation(
        **{
            **legacy_proof.__dict__,
            # The shared verifier retains a non-cryptographic compatibility
            # signature.  Effect-bearing MCP++ admission must reject it even
            # when a caller recomputes that token from the public key hint.
            "signature": compute_delegation_signature(
                delegation=legacy_proof, issuer_key_hint=public_b64,
            ),
        },
    )
    legacy_chain = [{
        "issuer": legacy_signed.issuer,
        "audience": legacy_signed.audience,
        "capabilities": [{
            "resource": "repo", "ability": "agent-supervisor/invoke",
        }],
        "proof_cid": legacy_signed.proof_cid,
        "signature": legacy_signed.signature,
    }]
    assert not factory.create(
        target_alias="repo", ucan_verified=legacy_chain,
    ).authenticated

    # Adapter-owned/self-chosen keys can produce a diagnostic preview but are
    # not authority.  The production resolver must independently own and use
    # the alias map, key anchors, and revocation snapshot.
    from ipfs_accelerate_py.agent_supervisor.entrypoints import authority_resolver

    captured = []
    original_resolve_authority = authority_resolver.resolve_authority

    def capture_authority(request):
        captured.append(request)
        return original_resolve_authority(request)

    monkeypatch.setattr(authority_resolver, "resolve_authority", capture_authority)
    unconfigured_authority, _ = ProductionCanonicalResolverFactory()._authority(
        verified, repository_id="repository-fixture",
    )
    assert not unconfigured_authority.authorized
    assert captured[-1].authenticated_principal is None

    production = ProductionCanonicalResolverFactory(
        mcp_repository_aliases={"repo": str(tmp_path)},
        mcpplusplus_issuer_public_keys={issuer: public_b64},
    )
    verified_authority, _ = production._authority(
        verified, repository_id="repository-fixture",
    )
    assert not verified_authority.authorized  # Transport alone grants no effects.
    assert captured[-1].authenticated_principal is not None
    assert captured[-1].authenticated_principal.ucan_verified is True
    assert captured[-1].authenticated_principal.signature_verified is True
    for denied_chain in (
        chain(resource="*"), chain(revoked=True), chain(expiry=1.0), legacy_chain,
    ):
        denied_context = factory.create(
            target_alias="repo", ucan_verified=denied_chain,
        )
        denied_authority, _ = production._authority(
            denied_context, repository_id="repository-fixture",
        )
        assert not denied_authority.authorized
        assert captured[-1].authenticated_principal is None
    host_revoked = ProductionCanonicalResolverFactory(
        mcp_repository_aliases={"repo": str(tmp_path)},
        mcpplusplus_issuer_public_keys={issuer: public_b64},
        mcpplusplus_revoked_proof_cids=(chain()[0]["proof_cid"],),
    )
    revoked_authority, _ = host_revoked._authority(
        verified, repository_id="repository-fixture",
    )
    assert not revoked_authority.authorized
    assert captured[-1].authenticated_principal is None
    forged = InvocationContext(
        transport="mcp++", authenticated=True,
        provenance={"authority": "mcpplusplus_delegation_verifier"},
    )
    forged_authority, _ = production._authority(
        forged, repository_id="repository-fixture",
    )
    assert not forged_authority.authorized
    assert captured[-1].authenticated_principal is None


def test_default_service_is_the_complete_production_composition():
    service = SupervisorResolutionService()
    assert service.pipeline.required_fields == REQUIRED_LAUNCH_FIELDS
    assert service.pipeline.verify_prefilled is True
    assert ProductionCanonicalResolverFactory().pipeline().required_fields == REQUIRED_LAUNCH_FIELDS


def test_public_trusted_bindings_are_metadata_not_verified_leaf_receipts(
    tmp_path, monkeypatch,
):
    _, _, context, capability = _production_local_context(tmp_path, monkeypatch)
    service = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=capability,
        ).pipeline(),
    )
    baseline = service.resolve("same", context)
    supplied = service.resolve(
        "same",
        context,
        trusted_bindings={
            "authority": {"admin": True},
            "policy": {"open": True},
            "allowlist": ["*"],
            "provider": "caller-provider",
            "caller": "caller-admin",
            "validation_argv": ["sh", "-c", "unsafe"],
        },
    )
    assert baseline.launch_authorized is True
    assert supplied.launch_authorized is baseline.launch_authorized
    assert supplied.bindings_authoritative is False
    assert supplied.authority is None
    assert supplied.policy is None
    assert supplied.allowlist is None
    assert supplied.provider is None
    assert supplied.caller is None
    assert supplied.validation_argv is None
    assert supplied.to_dict()["untrusted_bindings"] == {
        "authority": {"admin": True},
        "policy": {"open": True},
        "allowlist": ["*"],
        "provider": "caller-provider",
        "caller": "caller-admin",
        "validation_argv": ["sh", "-c", "unsafe"],
    }
    assert supplied.field_receipts == baseline.field_receipts
    assert PROMPT_FORBIDDEN_FIELDS.isdisjoint(supplied.field_receipts)
    assert supplied.cid == baseline.cid


def test_production_prefilled_candidates_do_not_bypass_real_resolvers():
    context = TrustedEvidenceCollector().collect(
        transport="mcp", authenticated=True, values=_complete_values(),
    )
    receipt = SupervisorResolutionService().resolve("run", context)
    assert not receipt.launch_authorized
    assert receipt.continuation is not None
    # Provenance identifies resolver attempts rather than the forged ``test``
    # source that was prefilled by the caller.
    sources = {value["source"] for value in receipt.field_receipts.values()}
    assert "test" not in sources
    assert all("resolver" in source for source in sources)


def test_direct_context_cannot_self_allowlist_or_inject_capability(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git_repository(repository)
    injected_capability = {"schema": "shape-only", "content_id": "caller-recomputed"}
    forged = InvocationContext(
        transport="mcp++",
        authenticated=True,
        fields={
            "repository": ResolutionField(str(repository), source="caller"),
            "repository_allowlist": ResolutionField([str(repository)], source="caller"),
            "capability_evidence": ResolutionField(injected_capability, source="caller"),
        },
        provenance={"authority": "mcpplusplus_delegation_verifier"},
    )

    receipt = SupervisorResolutionService().resolve("run", forged)
    assert not receipt.launch_authorized
    assert receipt.field_receipts["repository"]["value"] is None
    assert receipt.field_receipts["repository"]["source"].endswith(
        "resolver_owned_repository_scope_required",
    )
    evidence, verified = ProductionCanonicalResolverFactory()._capability_evidence(
        forged, state_root=str(repository / ".git" / "agent-supervisor"),
    )
    assert not verified
    assert evidence.providers[PREFERRED_PROVIDER].authenticated is False


def test_plain_caller_local_context_cannot_borrow_host_profile(
    tmp_path, monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.entrypoints import (
        inference_runtime,
        local_profile,
    )

    repository, _, signed_context, capability = _production_local_context(
        tmp_path, monkeypatch,
    )
    service = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=capability,
        ).pipeline(),
    )
    raw = InvocationContext(
        transport="local",
        authenticated=False,
        fields={
            "repository": ResolutionField(
                str(repository), source="remote-caller",
            ),
        },
    )
    raw_receipt = service.resolve("run", raw)
    assert not raw_receipt.launch_authorized
    assert raw_receipt.field_receipts["repository"]["value"] is None
    assert raw_receipt.field_receipts["repository"]["source"].endswith(
        "verified_local_adapter_receipt_required",
    )
    prefilled = InvocationContext(
        transport="local",
        authenticated=True,
        fields=signed_context.fields,
        provenance={"authority": "caller-local"},
    )
    assert prefilled.core_cid == signed_context.core_cid
    assert not service.resolve("run", prefilled).launch_authorized

    # A real receipt is bound to the exact immutable core and cannot be moved
    # onto the caller-created core.  The signed envelope itself never changes
    # the transport-neutral core identity.
    copied = InvocationContext(
        transport="local",
        authenticated=True,
        fields=raw.fields,
        adapter_receipt=signed_context.adapter_receipt,
    )
    assert copied.core_cid == raw.core_cid
    assert copied.core_cid != signed_context.core_cid
    assert not service.resolve("run", copied).launch_authorized
    assert InvocationContext(
        transport="mcp",
        authenticated=False,
        fields=signed_context.fields,
    ).core_cid == signed_context.core_cid

    forged_receipt = signed_context.envelope.to_dict()["adapter_receipt"]
    assert isinstance(forged_receipt, dict)
    signed_body = {
        key: value for key, value in forged_receipt.items()
        if key != "signature"
    }
    local_profile.verify_did_key_signature(
        identity_did=forged_receipt["identity_did"],
        payload=signed_body,
        signature=forged_receipt["signature"],
    )
    assert forged_receipt["core_cid"] == signed_context.core_cid
    assert forged_receipt["repository_root"] == str(repository)
    assert len(forged_receipt["nonce"]) == 32
    forged_receipt["profile_id"] = "caller-selected-profile"
    forged = InvocationContext(
        transport="local",
        authenticated=True,
        fields=signed_context.fields,
        adapter_receipt=forged_receipt,
    )
    assert not service.resolve("run", forged).launch_authorized

    expires_at_ns = signed_body["expires_at_ns"]
    monkeypatch.setattr(
        inference_runtime.time, "time_ns", lambda: expires_at_ns + 1,
    )
    assert not service.resolve("run", signed_context).launch_authorized


def test_local_adapter_receipt_rotation_and_revocation_fail_closed(
    tmp_path, monkeypatch,
):
    repository, profile_dir, context, capability = _production_local_context(
        tmp_path, monkeypatch,
    )
    service = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=capability,
        ).pipeline(),
    )
    assert context.authenticated
    assert context.adapter_receipt is not None
    assert service.resolve("run", context).launch_authorized

    target = resolve_repository_target(RepositoryTargetEvidence(
        cwd=str(repository), allowlisted_roots=(str(repository),),
    ))
    assert target.binding is not None
    initialize_local_profile(
        repository_cid=target.binding.repository_id,
        baseline_commit=target.binding.head_commit,
        profile_dir=profile_dir,
        force=True,
    )
    assert not service.resolve("run", context).launch_authorized

    rotated = LocalInvocationContextFactory().create(cwd=str(repository))
    assert rotated.authenticated
    assert rotated.core_cid != context.core_cid
    # The prior capability receipt is also principal-bound and must not cross
    # rotation.  A fresh resolver-owned snapshot for the new principal restores
    # the usable path.
    assert not service.resolve("run", rotated).launch_authorized
    rotated_binding = rotated.envelope.to_dict()["adapter_receipt"]
    assert isinstance(rotated_binding, dict)
    rotated_capability = json.loads(json.dumps(capability))
    rotated_capability["topology"]["owner_principal_ref"] = (
        f"local-profile:{rotated_binding['profile_id']}"
    )
    rotated_capability.pop("content_id", None)
    rotated_service = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=rotated_capability,
        ).pipeline(),
    )
    assert rotated_service.resolve("run", rotated).launch_authorized
    revoke_local_profile(profile_dir=profile_dir)
    assert not rotated_service.resolve("run", rotated).launch_authorized


def test_no_importable_adapter_issuer_or_object_attribute_can_mint_trust(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.entrypoints import context_adapters

    repository = tmp_path / "repository"
    repository.mkdir()
    _git_repository(repository)
    assert not hasattr(context_adapters, "_ADAPTER_SEAL_KEY")
    assert not hasattr(context_adapters, "_attest_adapter_context")

    forged = InvocationContext(
        transport="mcp++",
        authenticated=True,
        fields={
            "repository": ResolutionField(str(repository), source="caller"),
            "repository_alias": ResolutionField("repo", source="caller"),
            "repository_allowlist": ResolutionField(
                (str(repository),), source="caller",
            ),
            "ucan_delegation_chain": ResolutionField(
                ({"allowed": True, "signature": "caller"},), source="caller",
            ),
        },
        provenance={"authority": "caller"},
    )
    # Even object.__setattr__ cannot recreate the removed admission path.
    object.__setattr__(forged, "_adapter_receipt", "v1:all:caller")

    receipt = SupervisorResolutionService().resolve("run", forged)
    assert not receipt.launch_authorized
    assert receipt.field_receipts["repository"]["value"] is None
    assert receipt.field_receipts["repository"]["source"].endswith(
        "resolver_owned_repository_scope_required",
    )
    authority, _ = ProductionCanonicalResolverFactory()._authority(
        forged, repository_id="caller-repository",
    )
    assert not authority.authorized
    assert authority.principal.principal_ref == ""


def test_client_path_and_symlink_cannot_enter_python_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = tmp_path / "allowed-link"
    link.symlink_to(allowed, target_is_directory=True)
    factory = PythonInvocationContextFactory()

    for roots, repository in (
        ((str(allowed),), str(outside)),
        ((str(link),), str(link)),
    ):
        try:
            factory.create(allowlisted_roots=roots, repository=repository)
        except ValueError:
            pass
        else:
            raise AssertionError("client path/symlink must not acquire an allowlist receipt")


def test_per_invocation_capability_mapping_does_not_acquire_adapter_trust(
    tmp_path, monkeypatch,
):
    repository, _, _, raw_capability = _production_local_context(
        tmp_path, monkeypatch,
    )
    injected = LocalInvocationContextFactory().create(
        cwd=str(repository), values={"capability_evidence": raw_capability},
    )

    receipt = SupervisorResolutionService().resolve("run", injected)
    assert not receipt.launch_authorized
    for name in ("resources", "validation", "topology"):
        assert receipt.field_receipts[name]["value"] is None
        assert receipt.field_receipts[name]["source"].endswith(
            "verified_capability_evidence_required",
        )

    wrong_state = json.loads(json.dumps(raw_capability))
    wrong_state["topology"]["state_root"] = str(repository / "foreign-state")
    wrong_state.pop("content_id", None)
    state_mismatch = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=wrong_state,
        ).pipeline(),
    ).resolve("run", injected)
    assert not state_mismatch.launch_authorized
    for name in ("resources", "validation", "topology"):
        assert state_mismatch.field_receipts[name]["value"] is None


def test_python_and_mcp_booleans_never_supply_effect_authority(
    tmp_path, monkeypatch,
):
    repository, _, _, raw_capability = _production_local_context(
        tmp_path, monkeypatch,
    )
    contexts = (
        PythonInvocationContextFactory(
            capability_evidence=raw_capability,
        ).create(
            allowlisted_roots=(str(repository),), authenticated=True,
        ),
        PythonInvocationContextFactory().create(
            allowlisted_roots=(str(repository),), authenticated=True,
            values={"capability_evidence": raw_capability},
        ),
        MCPInvocationContextFactory(
            repository_aliases={"repo": str(repository)},
            capability_evidence=raw_capability,
        ).create(target_alias="repo", authenticated=True),
        MCPInvocationContextFactory(
            repository_aliases={"repo": str(repository)},
        ).create(
            target_alias="repo", authenticated=True,
            values={"capability_evidence": raw_capability},
        ),
    )

    for context in contexts:
        receipt = SupervisorResolutionService().resolve("run", context)
        assert not receipt.launch_authorized
        assert receipt.field_receipts["repository"]["value"] is None
        assert receipt.field_receipts["profile"]["value"] is None
        assert "resolver" in receipt.field_receipts["profile"]["source"]


def test_production_composition_calls_all_public_leaf_resolvers(tmp_path, monkeypatch):
    _, _, context, capability = _production_local_context(tmp_path, monkeypatch)
    calls: dict[str, int] = {}
    targets = (
        ("target", "ipfs_accelerate_py.agent_supervisor.entrypoints.target_resolver", "resolve_repository_target"),
        ("state", "ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver", "resolve_state"),
        ("run", "ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver", "resolve_run_candidates"),
        ("objective_task", "ipfs_accelerate_py.agent_supervisor.entrypoints.objective_resolver", "resolve_objectives"),
        ("capability", "ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver", "resolve_capabilities"),
        ("authority", "ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver", "resolve_authority"),
        ("profile", "ipfs_accelerate_py.agent_supervisor.entrypoints.profile_resolver", "resolve_supervisor_profile"),
    )
    for label, module_name, attribute in targets:
        module = __import__(module_name, fromlist=[attribute])
        original = getattr(module, attribute)

        def wrapper(*args, _label=label, _original=original, **kwargs):
            calls[_label] = calls.get(_label, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, attribute, wrapper)

    service = SupervisorResolutionService(
        ProductionCanonicalResolverFactory(
            capability_evidence=capability,
        ).pipeline(),
    )
    receipt = service.resolve("improve safely", context)
    assert receipt.launch_authorized
    assert calls == {name: 1 for name, _, _ in targets}
    sources = {name: value["source"] for name, value in receipt.field_receipts.items()}
    assert sources["repository"].startswith("target_resolver:")
    assert sources["state"].startswith("state_resolver:")
    assert sources["run"].startswith("run_resolver:")
    assert sources["objective"].startswith("objective_resolver:")
    assert sources["task_source"].startswith("task_source_resolver:")
    assert sources["profile"].startswith("profile_resolver:")
    for name in ("resources", "validation", "topology"):
        assert sources[name].startswith("capability_resolver:")


def test_installed_profile_tamper_revocation_and_symlink_fail_closed(tmp_path, monkeypatch):
    repository, profile_dir, context, _ = _production_local_context(
        tmp_path, monkeypatch,
    )
    assert context.authenticated
    profile_path = profile_dir / PROFILE_FILENAME
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["profile_id"] = "substituted"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    assert not LocalInvocationContextFactory().create(cwd=str(repository)).authenticated

    target = resolve_repository_target(RepositoryTargetEvidence(
        cwd=str(repository), allowlisted_roots=(str(repository),),
    ))
    assert target.binding is not None
    initialize_local_profile(
        repository_cid=target.binding.repository_id,
        baseline_commit=target.binding.head_commit,
        profile_dir=profile_dir, force=True,
    )
    revoke_local_profile(profile_dir=profile_dir)
    assert not LocalInvocationContextFactory().create(cwd=str(repository)).authenticated

    replacement = tmp_path / "replacement-profile"
    initialize_local_profile(
        repository_cid=target.binding.repository_id,
        baseline_commit=target.binding.head_commit,
        profile_dir=replacement,
    )
    link = tmp_path / "profile-link"
    link.symlink_to(replacement, target_is_directory=True)
    monkeypatch.setenv(DEFAULT_PROFILE_DIR_ENV, str(link))
    assert not LocalInvocationContextFactory().create(cwd=str(repository)).authenticated


def test_mixed_mapping_keys_and_sets_have_stable_context_and_receipt_identity():
    left = FrozenMapping(((True, {"b", 2, "a"}), (1, "integer"), ("1", "text")))
    right = FrozenMapping((("1", "text"), (1, "integer"), (True, {2, "a", "b"})))
    assert len(left) == 3
    assert left[True] == right[True]
    values_left = {"repository": ResolutionField(value=left, source="fixture")}
    values_right = {"repository": ResolutionField(value=right, source="fixture")}
    first = TrustedEvidenceCollector().collect(
        transport="mcp", authenticated=True, values=values_left,
    )
    second = TrustedEvidenceCollector().collect(
        transport="mcp", authenticated=True, values=values_right,
    )
    assert first.cid == second.cid
    pipeline = CanonicalResolutionPipeline(required_fields=("repository",))
    first_receipt = SupervisorResolutionService(pipeline).resolve("same", first)
    second_receipt = SupervisorResolutionService(pipeline).resolve("same", second)
    assert first_receipt.identity() == second_receipt.identity()

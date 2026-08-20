"""FACP-044: information-flow lattice, declassification, and two-run suites."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.information_flow_assurance import (
    ASSURER_VERSION,
    BUNDLE,
    CANARY_PREFIX,
    EVIDENCE_SUBSET,
    FORBIDDEN_DECLASSIFICATION_DESTINATIONS,
    GOAL_ID,
    HOST_PATH_REDACTED,
    PUBLIC_CHANNEL_CEILING,
    REDACTED_VALUE,
    SCHEMA,
    TASK_ID,
    DeclassificationPermit,
    InformationFlowAssurer,
    InformationFlowError,
    InformationFlowReport,
    LabeledValue,
    LeakKind,
    PublicChannel,
    SecurityLabel,
    TaintStore,
    TwoRunPropertyKind,
    TwoRunTrace,
    TwoRunVerdict,
    assert_public_channels_clean,
    audit_public_payload,
    build_critical_two_run_pair,
    compare_two_runs,
    declassify,
    default_two_run_models,
    join_labels,
    key_looks_protected,
    label_for_key,
    labeled,
    lattice_labels,
    looks_like_host_path,
    mint_canary,
    model_for,
    project_public_value,
    require_declassification_bindings,
    run_critical_two_run_suites,
    simulate_browser_host_kernel,
    simulate_credential_kernel,
)


def test_facp_envelope_constants() -> None:
    assert TASK_ID == "FACP-044"
    assert GOAL_ID == "FACP-G420"
    assert BUNDLE == "facp/static/information-flow"
    assert SCHEMA == "facp/information-flow@1"
    assert ASSURER_VERSION.startswith("ifa/")
    assert "Public" in EVIDENCE_SUBSET
    assert "WitnessSecret" in EVIDENCE_SUBSET
    assert "browser-host" in EVIDENCE_SUBSET
    assert "witness" in EVIDENCE_SUBSET


def test_security_lattice_order_and_join() -> None:
    labels = lattice_labels()
    assert [item.value for item in labels] == [
        "Public",
        "Internal",
        "RepositoryPrivate",
        "TenantPrivate",
        "MatterConfidential",
        "Credential",
        "CryptographicSecret",
        "WitnessSecret",
    ]
    assert SecurityLabel.PUBLIC.flows_to(SecurityLabel.WITNESS_SECRET)
    assert not SecurityLabel.WITNESS_SECRET.flows_to(SecurityLabel.PUBLIC)
    assert (
        join_labels(SecurityLabel.CREDENTIAL, SecurityLabel.TENANT_PRIVATE)
        is SecurityLabel.CREDENTIAL
    )
    assert SecurityLabel.INTERNAL.meet(SecurityLabel.CREDENTIAL) is SecurityLabel.INTERNAL
    assert PUBLIC_CHANNEL_CEILING is SecurityLabel.INTERNAL


def test_labeled_value_redacts_high_material_from_public_dict() -> None:
    secret = labeled(
        "raw-credential-value",
        SecurityLabel.CREDENTIAL,
        source_id="credential.secret",
        with_canary=True,
    )
    public = secret.to_public_dict()
    assert public["redacted"] is True
    assert "value" not in public
    assert public["digest"].startswith("sha256:")
    assert secret.canary.startswith(CANARY_PREFIX)
    assert "raw-credential-value" not in json.dumps(public)

    low = labeled("ok", SecurityLabel.PUBLIC, source_id="status")
    low_public = low.to_public_dict()
    assert low_public["redacted"] is False
    assert low_public["value"] == "ok"


def test_taint_propagation_joins_labels() -> None:
    store = TaintStore()
    store.write(
        "a",
        labeled("one", SecurityLabel.TENANT_PRIVATE, source_id="a", with_canary=True),
    )
    store.write(
        "b",
        labeled("two", SecurityLabel.CREDENTIAL, source_id="b", with_canary=True),
    )
    joined = store.propagate(("a", "b"), "c")
    assert joined.label is SecurityLabel.CREDENTIAL
    assert store.label_of("c") is SecurityLabel.CREDENTIAL
    snapshot = store.snapshot()
    assert snapshot["c"]["redacted"] is True
    assert "one" not in json.dumps(snapshot)
    assert "two" not in json.dumps(snapshot)


def test_declassification_requires_complete_bindings() -> None:
    with pytest.raises(InformationFlowError, match="empty"):
        DeclassificationPermit(
            policy_id="",
            actor="actor:operator",
            destination="host.evidence_store",
            source="credential.secret",
            purpose="issue opaque receipt digest",
            from_label=SecurityLabel.CREDENTIAL,
            to_label=SecurityLabel.INTERNAL,
        )

    with pytest.raises(InformationFlowError, match="browser|public"):
        DeclassificationPermit(
            policy_id="policy:ifa-1",
            actor="actor:operator",
            destination="browser",
            source="credential.secret",
            purpose="display secret",
            from_label=SecurityLabel.CREDENTIAL,
            to_label=SecurityLabel.PUBLIC,
        )

    permit = DeclassificationPermit(
        policy_id="policy:ifa-credential-digest",
        actor="actor:effect_admission_kernel",
        destination="host.receipt_projector",
        source="credential.secret",
        purpose="emit opaque credential presence digest",
        from_label=SecurityLabel.CREDENTIAL,
        to_label=SecurityLabel.INTERNAL,
    )
    require_declassification_bindings(permit)
    payload = permit.to_dict()
    assert payload["bindings_complete"] is True
    for key in ("policy_id", "actor", "destination", "source", "purpose"):
        assert payload[key]
    assert permit.permit_id


def test_declassify_lowers_only_with_exact_source_match() -> None:
    value = labeled(
        "cred-raw",
        SecurityLabel.CREDENTIAL,
        source_id="credential.secret",
    )
    permit = DeclassificationPermit(
        policy_id="policy:ifa-credential-digest",
        actor="actor:effect_admission_kernel",
        destination="host.receipt_projector",
        source="credential.secret",
        purpose="emit opaque credential presence digest",
        from_label=SecurityLabel.CREDENTIAL,
        to_label=SecurityLabel.INTERNAL,
    )
    lowered = declassify(value, permit)
    assert lowered.label is SecurityLabel.INTERNAL
    assert isinstance(lowered.value, dict)
    assert lowered.value["declassified"] is True
    assert lowered.value["digest"] == value.digest
    assert "cred-raw" not in json.dumps(lowered.to_public_dict())

    wrong_source = DeclassificationPermit(
        policy_id="policy:ifa-credential-digest",
        actor="actor:effect_admission_kernel",
        destination="host.receipt_projector",
        source="other.source",
        purpose="emit opaque credential presence digest",
        from_label=SecurityLabel.CREDENTIAL,
        to_label=SecurityLabel.INTERNAL,
    )
    with pytest.raises(InformationFlowError, match="source"):
        declassify(value, wrong_source)


@pytest.mark.parametrize(
    "destination",
    sorted(FORBIDDEN_DECLASSIFICATION_DESTINATIONS),
)
def test_browser_and_public_destinations_cannot_declassify(destination: str) -> None:
    with pytest.raises(InformationFlowError):
        DeclassificationPermit(
            policy_id="policy:bad",
            actor="actor:x",
            destination=destination,
            source="credential.secret",
            purpose="leak",
            from_label=SecurityLabel.CREDENTIAL,
            to_label=SecurityLabel.PUBLIC,
        )


def test_host_path_and_key_heuristics() -> None:
    assert looks_like_host_path("/home/barberb/secret.key")
    assert looks_like_host_path("~/lift_coding/repo")
    assert looks_like_host_path("C:\\Users\\x\\file")
    assert not looks_like_host_path("relative/path.py")
    assert key_looks_protected("api_key")
    assert key_looks_protected("host_path")
    assert key_looks_protected("private_witness")
    assert label_for_key("api_key") is SecurityLabel.CREDENTIAL
    assert label_for_key("proof_witness") is SecurityLabel.WITNESS_SECRET
    assert label_for_key("status") is SecurityLabel.PUBLIC


def test_public_projection_redacts_secrets_and_host_paths() -> None:
    payload = {
        "status": "ok",
        # Proposal-gate-approved test sentinel (not a concrete credential).
        "api_key": "sk-live-not-a-real-key",
        "workdir": "/home/barberb/lift_coding/private",
        "note": "relative/ok.py",
        "nested": {"password": "hunter2", "count": 2},
    }
    projected = project_public_value(payload, channel=PublicChannel.LOG)
    assert projected["status"] == "ok"
    assert projected["api_key"] == REDACTED_VALUE
    assert projected["workdir"] == HOST_PATH_REDACTED
    assert projected["note"] == "relative/ok.py"
    assert projected["nested"]["password"] == REDACTED_VALUE
    assert projected["nested"]["count"] == 2
    text = json.dumps(projected)
    assert "sk-live-not-a-real-key" not in text
    assert "/home/barberb" not in text
    assert "hunter2" not in text


def test_public_audit_detects_leaks_and_canaries() -> None:
    canary = mint_canary("credential.secret", kind="Credential")
    dirty = {
        "message": "failed",
        "detail": canary,
        "path": "/var/lib/ipfs/private",
        "api_key": "raw-key",
    }
    audit = audit_public_payload(
        dirty,
        channel=PublicChannel.RECEIPT,
        canaries=(canary,),
        known_protected=("raw-key",),
    )
    assert audit.clean is False
    kinds = {item.kind for item in audit.findings}
    assert LeakKind.CANARY in kinds
    assert LeakKind.HOST_PATH in kinds
    assert LeakKind.FORBIDDEN_KEY in kinds
    projected_text = json.dumps(dict(audit.projected))
    assert canary not in projected_text
    assert "raw-key" not in projected_text
    assert "/var/lib/ipfs/private" not in projected_text
    assert audit.projected["path"] == HOST_PATH_REDACTED
    assert audit.projected["api_key"] == REDACTED_VALUE
    assert audit.projected["detail"] == REDACTED_VALUE

    clean = {"status": "ok", "digest": "sha256:abc"}
    clean_audit = audit_public_payload(clean, channel=PublicChannel.BROWSER)
    assert clean_audit.clean is True
    assert clean_audit.findings == ()


def test_assert_public_channels_clean_covers_all_surfaces() -> None:
    canary = mint_canary("witness", kind="WitnessSecret")
    payloads = {
        "log": {"event": "ok"},
        "receipt": {"status": "sealed"},
        "browser": {"view": "summary"},
        "prompt": {"instruction": "summarize"},
    }
    audits = assert_public_channels_clean(payloads, canaries=(canary,))
    assert len(audits) == 4
    assert all(item.clean for item in audits)

    with pytest.raises(InformationFlowError, match="leaked"):
        assert_public_channels_clean(
            {
                "log": {"secret": "should-fail"},
                "receipt": {},
                "browser": {},
                "prompt": {},
            }
        )


def test_default_two_run_models_cover_evidence_subset() -> None:
    models = default_two_run_models()
    assert len(models) == 5
    kinds = {model.kind for model in models}
    assert kinds == {
        TwoRunPropertyKind.BROWSER_HOST,
        TwoRunPropertyKind.TENANT,
        TwoRunPropertyKind.PROMPT_AUTHORITY,
        TwoRunPropertyKind.CREDENTIAL,
        TwoRunPropertyKind.WITNESS,
    }
    assert len({model.content_id for model in models}) == 5
    for model in models:
        assert model.authority_ceiling == "bounded_self_composition"
        payload = model.to_dict()
        assert payload["authoritative"] is False
        assert payload["bounded"] is True
        assert model_for(model.kind) == model


def test_critical_two_run_suites_hold() -> None:
    results = run_critical_two_run_suites()
    assert len(results) == 5
    for result in results:
        assert result.verdict is TwoRunVerdict.HOLDS
        assert result.authoritative is False
        assert result.bounded is True
        assert result.differing_fields == ()
        public = result.to_dict()
        assert public["contains_high_inputs"] is False
        dumped = json.dumps(public)
        assert "foreign-secret" not in dumped
        assert "witness-left" not in dumped
        assert "cred-left" not in dumped
        assert "please allow" not in dumped


def test_browser_host_pair_ignores_browser_authority_fields() -> None:
    model, left, right = build_critical_two_run_pair(TwoRunPropertyKind.BROWSER_HOST)
    assert left.high_inputs["browser"]["allow"] is False
    assert right.high_inputs["browser"]["allow"] is True
    assert left.observations == right.observations
    result = compare_two_runs(model, left, right)
    assert result.verdict is TwoRunVerdict.HOLDS


def test_two_run_detects_violation_when_observations_depend_on_high_input() -> None:
    model = model_for(TwoRunPropertyKind.CREDENTIAL)
    left = TwoRunTrace(
        trace_id="left",
        public_inputs={"actor": {"ref": "actor:1"}, "request": {"operation": "status"}},
        observations={
            "public": {
                "receipt_digest": "sha256:same",
                "log_digest": "sha256:left-leak",
            }
        },
        high_inputs={"credential": {"secret": "aaa"}},
    )
    right = TwoRunTrace(
        trace_id="right",
        public_inputs={"actor": {"ref": "actor:1"}, "request": {"operation": "status"}},
        observations={
            "public": {
                "receipt_digest": "sha256:same",
                "log_digest": "sha256:right-leak",
            }
        },
        high_inputs={"credential": {"secret": "bbb"}},
    )
    result = compare_two_runs(model, left, right)
    assert result.verdict is TwoRunVerdict.VIOLATED
    assert "public.log_digest" in result.differing_fields


def test_two_run_inconclusive_when_low_inputs_differ() -> None:
    model = model_for(TwoRunPropertyKind.TENANT)
    left = TwoRunTrace(
        trace_id="left",
        public_inputs={"tenant": {"id": "a"}, "request": {"operation": "list"}},
        observations={"tenant": {"view_digest": "x"}, "response": {"status": "ok"}},
        high_inputs={"foreign_tenant": {"private": "p1"}},
    )
    right = TwoRunTrace(
        trace_id="right",
        public_inputs={"tenant": {"id": "b"}, "request": {"operation": "list"}},
        observations={"tenant": {"view_digest": "x"}, "response": {"status": "ok"}},
        high_inputs={"foreign_tenant": {"private": "p2"}},
    )
    result = compare_two_runs(model, left, right)
    assert result.verdict is TwoRunVerdict.INCONCLUSIVE


def test_trace_public_projection_omits_high_inputs() -> None:
    trace = simulate_credential_kernel(
        actor_ref="actor:1",
        operation="status",
        credential_secret="super-secret-credential",
    )
    public = trace.to_public_dict()
    assert public["high_inputs_redacted"] is True
    assert "super-secret-credential" not in json.dumps(public)
    assert "super-secret-credential" not in repr(trace)


def test_assurer_end_to_end_report() -> None:
    assurer = InformationFlowAssurer()
    assurer.label_and_store(
        "credential.secret",
        "synthetic-credential",
        SecurityLabel.CREDENTIAL,
        with_canary=True,
    )
    assurer.label_and_store(
        "proof.witness",
        "synthetic-witness",
        SecurityLabel.WITNESS_SECRET,
        with_canary=True,
    )
    permit = DeclassificationPermit(
        policy_id="policy:ifa-credential-digest",
        actor="actor:effect_admission_kernel",
        destination="host.receipt_projector",
        source="credential.secret",
        purpose="emit opaque credential presence digest",
        from_label=SecurityLabel.CREDENTIAL,
        to_label=SecurityLabel.INTERNAL,
    )
    assurer.register_permit(permit)
    lowered = assurer.declassify_path("credential.secret", permit)
    assert lowered.label is SecurityLabel.INTERNAL

    report = assurer.run_assurance(
        public_payloads={
            "log": {"event": "assurance_ok", "digest": "sha256:1"},
            "receipt": {"status": "sealed", "digest": "sha256:2"},
            "browser": {"summary": "nonauthority"},
            "prompt": {"instruction": "describe status only"},
        },
        known_protected=("synthetic-credential", "synthetic-witness"),
    )
    assert isinstance(report, InformationFlowReport)
    assert report.all_suites_hold
    assert len(report.two_run_results) == 5
    assert len(report.public_audits) == 4
    assert len(report.declassification_permits) == 1
    payload = report.to_dict()
    assert payload["task_id"] == "FACP-044"
    assert payload["authoritative"] is False
    dumped = json.dumps(payload)
    assert "synthetic-credential" not in dumped
    assert "synthetic-witness" not in dumped
    assert "/home/" not in dumped


def test_browser_host_kernel_is_stable_under_authority_flips() -> None:
    base = dict(
        method="tools/call",
        resource="accelerate.inference",
        argument_cid="bafyargument0001",
    )
    a = simulate_browser_host_kernel(
        **base, browser_allow=False, browser_consent=False, browser_dry_run=True
    )
    b = simulate_browser_host_kernel(
        **base, browser_allow=True, browser_consent=True, browser_dry_run=False
    )
    assert a.observations == b.observations
    assert a.public_inputs == b.public_inputs

"""PTR-166: proving-key possession cannot fabricate a pass.

The suite:

* asserts the manifest-pinned native provider identity;
* performs one real ephemeral test-only setup / prove / verify through the
  production V5 route;
* shows a genuine proof over a fabricated unsigned receipt may satisfy raw
  proof math yet fails authority before candidate publication;
* accepts one correctly signed real-backend vector;
* forces every signature, key, trust-policy, CID, nonce, epoch, revocation and
  downgrade mutation to return RUN;
* uses body-oracle evidence (not a skip counter) to detect false admissions.

Zero pytest.skip / xfail / conditional backend bypasses: missing real backend
material hard-fails fixture construction.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    verify_runner_pass_attestation_with_key,
)
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_V5_INTERFACE,
    TestPassStatementV5,
)
from ipfs_datasets_py.logic.zkp.test_certificate_assurance import (
    is_locally_verified_runner_assurance,
    verify_local_runner_attestation_v5,
)
from ipfs_datasets_py.logic.zkp.test_execution_certificate import (
    CertificateVerificationStatus,
    verify_test_execution_certificate_v5,
)
from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
    NATIVE_GROTH16_V5_INTERFACE,
    NativeGroth16V5Proof,
    NativeGroth16V5Status,
    is_native_groth16_v5_provider,
)


def _load_sibling_fixture_module() -> ModuleType:
    """Load the co-located fixture helper without relying on import path layout.

    Pytest rootdir is ``external/ipfs_accelerate/test`` (see pytest.ini), so a
    bare ``from proof_reuse_authenticated_real_backend_fixture import ...``
    raises ModuleNotFoundError under the hermetic validation PYTHONPATH.
    """

    module_name = "proof_reuse_authenticated_real_backend_fixture"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    fixture_path = Path(__file__).resolve().parent / f"{module_name}.py"
    if not fixture_path.is_file():
        raise ImportError(
            f"authenticated real-backend fixture missing at {fixture_path}"
        )
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fixture module from {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_fixture = _load_sibling_fixture_module()
AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE = (
    _fixture.AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE
)
AuthorityBodyObservation = _fixture.AuthorityBodyObservation
AuthorityBodyOracle = _fixture.AuthorityBodyOracle
MUTATION_KINDS = _fixture.MUTATION_KINDS
NOW = _fixture.NOW
AuthenticatedRealBackendFixture = _fixture.AuthenticatedRealBackendFixture
downgrade_payloads = _fixture.downgrade_payloads
get_shared_fixture = _fixture.get_shared_fixture

# This module must never introduce skip/xfail markers or conditional bypasses.
pytestmark = []


def _observation_from_result(
    *,
    case_id: str,
    result: object,
    raw_proof_math_ok: bool = False,
) -> AuthorityBodyObservation:
    status = getattr(result, "status", "")
    status_text = str(status.value if hasattr(status, "value") else status)
    return AuthorityBodyObservation(
        case_id=case_id,
        test_action=str(getattr(result, "test_action", "run")),
        can_authorize_skip=bool(getattr(result, "can_authorize_skip", False)),
        status=status_text,
        reason=str(
            getattr(result, "detail", "") or getattr(result, "reason", "") or ""
        ),
        raw_proof_math_ok=raw_proof_math_ok,
    )


@pytest.fixture(scope="module")
def fixture() -> AuthenticatedRealBackendFixture:
    """Shared ephemeral real-backend fixture (one setup per module)."""

    return get_shared_fixture()


@pytest.fixture(scope="module")
def body_oracle() -> AuthorityBodyOracle:
    return AuthorityBodyOracle()


# ---------------------------------------------------------------------------
# Manifest-pinned identity + ephemeral production V5 path
# ---------------------------------------------------------------------------


def test_manifest_pinned_native_provider_identity(
    fixture: AuthenticatedRealBackendFixture,
) -> None:
    identity = fixture.assert_manifest_pinned_identity()
    assert identity["provider_interface"] == NATIVE_GROTH16_V5_INTERFACE
    assert (
        identity["conformance_interface"]
        == AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE
    )
    assert identity["statement_interface"] == TEST_PASS_STATEMENT_V5_INTERFACE
    assert identity["v5_profile_id"]
    assert Path(identity["proving_key_path"]).is_file()
    assert Path(identity["verifying_key_path"]).is_file()
    assert fixture.provider is not None
    assert fixture.provider.interface == NATIVE_GROTH16_V5_INTERFACE
    assert is_native_groth16_v5_provider(fixture.provider)


def test_ephemeral_setup_prove_verify_production_v5_route(
    fixture: AuthenticatedRealBackendFixture,
    body_oracle: AuthorityBodyOracle,
) -> None:
    """One actual ephemeral setup + prove + verify through production V5."""

    assert fixture.artifacts_root.is_dir()
    assert (fixture.artifacts_root / "v5" / "proving_key.bin").is_file()
    assert (fixture.artifacts_root / "v5" / "verifying_key.bin").is_file()
    assert isinstance(fixture.positive_statement, TestPassStatementV5)
    assert isinstance(fixture.positive_proof, NativeGroth16V5Proof)
    assert fixture.provider is not None
    assert fixture.positive_witness is not None
    assert fixture.trust is not None

    native = fixture.provider.verify(
        fixture.positive_statement,
        fixture.positive_proof,
    )
    assert native.status is NativeGroth16V5Status.READY

    result = fixture.positive_authority_result()
    observation = body_oracle.record(
        _observation_from_result(
            case_id="positive:signed-real-backend",
            result=result,
            raw_proof_math_ok=True,
        )
    )
    assert observation.test_action == "skip"
    assert observation.can_authorize_skip is True
    assert result.status is CertificateVerificationStatus.VERIFIED
    assert result.can_authorize_skip is True
    assert result.test_action == "skip"

    assurance = verify_local_runner_attestation_v5(
        fixture.positive_statement,
        fixture.positive_witness,
        policy_bytes=fixture.trust.policy_bytes,
        pinned_policy_cid=fixture.trust.policy.cid,
        pinned_public_key_material=fixture.trust.public_key_material,
        now=NOW,
    )
    assert is_locally_verified_runner_assurance(assurance)


def test_correctly_signed_real_backend_vector_succeeds(
    fixture: AuthenticatedRealBackendFixture,
) -> None:
    """One correctly signed real-backend vector authorizes skip exactly once."""

    assert fixture.signed_full is not None
    signed = fixture.signed_full
    verified = verify_runner_pass_attestation_with_key(
        signed.attestation,
        receipt=signed.receipt,
        policy=signed.trust.policy,
        pinned_policy_cid=signed.trust.policy.cid,
        current_execution_key_cid=signed.receipt.execution_key_cid,
        current_candidate_context_cid=signed.candidate_context_cid,
        pinned_public_key_material=signed.trust.public_key_material,
        now=NOW,
    )
    assert verified.valid is True
    assert verified.signed_receipt is not None

    result = fixture.positive_authority_result()
    assert result.status is CertificateVerificationStatus.VERIFIED
    assert result.can_authorize_skip is True
    assert result.test_action == "skip"


# ---------------------------------------------------------------------------
# Proving-key-only forgery boundary
# ---------------------------------------------------------------------------


def test_fabricated_unsigned_proof_math_ok_but_authority_run(
    fixture: AuthenticatedRealBackendFixture,
    body_oracle: AuthorityBodyOracle,
) -> None:
    """Genuine proof over fabricated unsigned receipt fails authority.

    Possession of the ephemeral proving key is enough for raw proof math and
    insufficient for pass authority / candidate publication.
    """

    observation = fixture.unsigned_fabricated_fails_before_publication(
        oracle=body_oracle
    )
    assert observation.raw_proof_math_ok is True
    assert observation.test_action == "run"
    assert observation.can_authorize_skip is False
    assert observation.status != CertificateVerificationStatus.VERIFIED.value
    assert fixture.full_type_signature_is_rejected() is True


def test_every_signature_key_policy_cid_nonce_epoch_revocation_downgrade_returns_run(
    fixture: AuthenticatedRealBackendFixture,
    body_oracle: AuthorityBodyOracle,
) -> None:
    """Closed forgery population: every required mutation returns RUN."""

    population = fixture.forgery_population()
    kinds_seen = {case.kind for case in population}
    assert kinds_seen == set(MUTATION_KINDS)

    for case in population:
        observation = fixture.evaluate_forgery_case(case, oracle=body_oracle)
        assert observation.test_action == "run", (
            f"{case.case_id} expected RUN, got {observation.test_action}: "
            f"{observation.reason}"
        )
        assert observation.can_authorize_skip is False, case.case_id
        assert observation.status != CertificateVerificationStatus.VERIFIED.value, (
            case.case_id
        )

    assert fixture.positive_statement is not None
    assert fixture.positive_witness is not None
    assert fixture.trust is not None
    for bad in downgrade_payloads():
        result = verify_test_execution_certificate_v5(
            fixture.positive_statement,
            fixture.positive_witness,
            bad,
            fixture.provider,
            policy_bytes=fixture.trust.policy_bytes,
            pinned_policy_cid=fixture.trust.policy.cid,
            pinned_public_key_material=fixture.trust.public_key_material,
            now=NOW,
        )
        assert result.can_authorize_skip is False
        assert result.test_action == "run"
        body_oracle.record(
            _observation_from_result(
                case_id=f"downgrade-payload:{type(bad).__name__}",
                result=result,
                raw_proof_math_ok=False,
            )
        )


def test_body_oracle_not_skip_counter_determines_false_admissions(
    fixture: AuthenticatedRealBackendFixture,
    body_oracle: AuthorityBodyOracle,
) -> None:
    """False admissions are body-oracle evidence, not a skip tally."""

    if body_oracle.skip_count == 0:
        result = fixture.positive_authority_result()
        body_oracle.record(
            _observation_from_result(
                case_id="oracle:positive",
                result=result,
                raw_proof_math_ok=True,
            )
        )
    if body_oracle.run_count == 0:
        fixture.unsigned_fabricated_fails_before_publication(oracle=body_oracle)

    summary = body_oracle.summary()
    assert summary["observations"] >= 2
    assert body_oracle.false_admissions == ()
    for item in body_oracle.observations:
        if item.can_authorize_skip:
            assert item.status == CertificateVerificationStatus.VERIFIED.value
            assert item.test_action == "skip"
        else:
            assert item.test_action == "run"
            assert item.status != CertificateVerificationStatus.VERIFIED.value


def test_module_declares_no_skip_xfail_or_backend_bypass() -> None:
    """Static assurance: this suite does not call skip/xfail bypass APIs."""

    source = Path(__file__).read_text(encoding="utf-8")
    # Build tokens dynamically so this meta-test does not contain the literals.
    py = "pytest"
    mark = "mark"
    call_skip = py + ".skip("
    call_xfail = py + ".xfail("
    mark_skip = "@" + py + "." + mark + ".skip"
    mark_xfail = "@" + py + "." + mark + ".xfail"
    mark_skipif = "@" + py + "." + mark + ".skipif"
    importorskip = "importorskip("

    # Only count non-docstring, non-meta-test call sites: strip this function body.
    lines = source.splitlines()
    body_start = next(
        i for i, line in enumerate(lines)
        if line.startswith("def test_module_declares_no_skip_xfail_or_backend_bypass")
    )
    scanned = "\n".join(lines[:body_start])
    # Also drop the module docstring.
    if scanned.lstrip().startswith('"""'):
        end = scanned.find('"""', 3)
        if end != -1:
            scanned = scanned[end + 3 :]

    assert call_skip not in scanned
    assert call_xfail not in scanned
    assert mark_skip not in scanned
    assert mark_xfail not in scanned
    assert mark_skipif not in scanned
    assert importorskip not in scanned

    create_src = inspect.getsource(AuthenticatedRealBackendFixture.create)
    assert (py + ".skip") not in create_src
    assert "skipif" not in create_src

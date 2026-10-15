"""Qualification of the shared native owner fixture, with no provider calls."""
from test.api.public_validation_fixture import public_doctor_service


def test_native_fixture_captures_signed_public_and_doctor_requirements(tmp_path, monkeypatch):
    with public_doctor_service(tmp_path, monkeypatch) as case:
        original = case.original.original_requirements
        assert original.metadata["doctor_requirements"] == original._doctor_requirements.metadata
        assert original._public_request == (str(case.public_request["artifact"]),
            case.public_request["expected_sha256"], case.public_request["task_cid"])
        assert case.planning["provider_calls"] == case.dispatch["provider_calls"] == 0
        assert len(case.prepared["spec"]["validations"]) == 3

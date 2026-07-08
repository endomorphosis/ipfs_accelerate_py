from ipfs_accelerate_py.api_backends.hf_tgi import hf_tgi


def test_hf_tgi_run_inference_normalizes_generated_text(monkeypatch):
    backend = hf_tgi.__new__(hf_tgi)

    def fake_generate_text(model_id, inputs, parameters=None, api_token=None, request_id=None, endpoint_id=None):
        return {
            "generated_text": f"reply:{inputs}",
            "request_id": request_id,
        }

    monkeypatch.setattr(backend, "generate_text", fake_generate_text)

    result = backend.run_inference(
        model_id="demo-model",
        inputs="hello world",
        parameters={"temperature": 0.1},
        request_id="req-1",
        endpoint_id="endpoint-1",
    )

    assert result["model"] == "demo-model"
    assert result["task"] == "text-generation"
    assert result["outputs"] == ["reply:hello world"]
    assert result["backend"] == "hf_tgi"
    assert result["implementation_type"] == "(REAL)"
    assert result["request_id"] == "req-1"
    assert result["endpoint_id"] == "endpoint-1"
    assert result["raw_response"]["generated_text"] == "reply:hello world"
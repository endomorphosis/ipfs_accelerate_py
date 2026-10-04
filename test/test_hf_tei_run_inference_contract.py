from ipfs_accelerate_py.api_backends.hf_tei import hf_tei


def test_hf_tei_run_inference_normalizes_batch_embeddings(monkeypatch):
    backend = hf_tei.__new__(hf_tei)

    def fake_batch_embed(model_id, texts, api_token=None, request_id=None, endpoint_id=None):
        return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(backend, "batch_embed", fake_batch_embed)

    result = backend.run_inference(
        model_id="embed-model",
        texts=["hello", "world"],
        request_id="req-embed-1",
        endpoint_id="endpoint-embed-1",
    )

    assert result["model"] == "embed-model"
    assert result["task"] == "text-embedding"
    assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert result["outputs"] == [[0.1, 0.2], [0.3, 0.4]]
    assert result["backend"] == "hf_tei"
    assert result["implementation_type"] == "(REAL)"
    assert result["request_id"] == "req-embed-1"
    assert result["endpoint_id"] == "endpoint-embed-1"

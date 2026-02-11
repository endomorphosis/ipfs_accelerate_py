from unittest.mock import patch, MagicMock


def test_openai_backend_uses_cache_to_avoid_network_calls(monkeypatch):
	from ipfs_accelerate_py.backends.openai import OpenAIBackend

	dummy_cache = MagicMock()
	dummy_cache.get_completion.return_value = "cached response"

	with patch("ipfs_accelerate_py.backends.openai.get_llm_cache", return_value=dummy_cache), patch(
		"ipfs_accelerate_py.backends.openai.requests.post"
	) as mock_post:
		backend = OpenAIBackend(id="openai", api_key="test-key", engine="test-engine", use_cache=True)
		response = backend("test prompt")
		assert response == "cached response"
		mock_post.assert_not_called()


def test_huggingface_backend_uses_cache_to_avoid_network_calls(monkeypatch):
	from ipfs_accelerate_py.backends.huggingface import HuggingFaceBackend

	dummy_cache = MagicMock()
	dummy_cache.get_completion.return_value = "cached hf response"

	fake_response = MagicMock()
	fake_response.json.return_value = {"generated_text": "remote"}
	fake_response.raise_for_status.return_value = None

	with patch("ipfs_accelerate_py.backends.huggingface.get_llm_cache", return_value=dummy_cache), patch(
		"ipfs_accelerate_py.backends.huggingface.requests.request", return_value=fake_response
	) as mock_req:
		backend = HuggingFaceBackend(id="huggingface", api_key="test-token", engine="test-engine", use_cache=True)
		response = backend({"inputs": "test prompt"})
		assert response == "cached hf response"
		mock_req.assert_not_called()

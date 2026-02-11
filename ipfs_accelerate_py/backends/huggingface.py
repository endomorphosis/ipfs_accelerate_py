import json
import os
import re

import requests


try:
	from ipfs_accelerate_py.common.llm_cache import get_llm_cache
except Exception:  # pragma: no cover
	get_llm_cache = None


_ENV_REF_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _resolve_api_key(value):
	if not value:
		return (
			os.getenv("HF_TOKEN")
			or os.getenv("HUGGINGFACE_HUB_TOKEN")
			or os.getenv("HUGGINGFACEHUB_API_TOKEN")
		)
	if isinstance(value, str):
		match = _ENV_REF_RE.match(value.strip())
		if match:
			return os.getenv(match.group(1))
		if value.startswith("env:"):
			return os.getenv(value[4:])
	return value


class HuggingFaceBackend:
	def __init__(self, id, api_key, engine, **config):
		self.id = id
		self.api_key = _resolve_api_key(api_key)
		if not self.api_key:
			raise ValueError(
				"HuggingFace API token is required (set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or pass api_key)"
			)
		self.engine = engine
		self.config = config

		self.API_URL = "https://api-inference.huggingface.co/models/" + self.engine
		self.headers = {"Authorization": f"Bearer {self.api_key}"}

		self.cache = None
		if get_llm_cache is not None:
			try:
				self.cache = get_llm_cache("huggingface", api_key=self.api_key)
			except Exception:
				self.cache = None

	def __call__(self, payload):
		use_cache = bool(self.config.get("use_cache", True))
		payload = payload or {}
		try:
			payload_json = json.dumps(payload, sort_keys=True, default=str)
		except Exception:
			payload_json = str(payload)

		inputs = payload.get("inputs", "")
		if isinstance(inputs, str):
			prompt = inputs
		else:
			try:
				prompt = json.dumps(inputs, sort_keys=True, default=str)
			except Exception:
				prompt = str(inputs)

		parameters = payload.get("parameters")
		if not isinstance(parameters, dict):
			parameters = {}
		temperature = float(parameters.get("temperature", 0.0) or 0.0)
		max_tokens = parameters.get("max_new_tokens", parameters.get("max_tokens"))

		if use_cache and self.cache is not None:
			try:
				cached = self.cache.get_completion(
					prompt=prompt,
					model=self.engine,
					temperature=temperature,
					max_tokens=max_tokens,
					hf_payload=payload_json,
				)
				if cached is not None:
					return cached
			except Exception:
				pass

		data = json.dumps(payload)
		response = requests.request("POST", self.API_URL, headers=self.headers, data=data)
		result = json.loads(response.content.decode("utf-8"))
		if use_cache and self.cache is not None:
			try:
				self.cache.cache_completion(
					prompt=prompt,
					response=result,
					model=self.engine,
					temperature=temperature,
					max_tokens=max_tokens,
					hf_payload=payload_json,
				)
			except Exception:
				pass
		return result

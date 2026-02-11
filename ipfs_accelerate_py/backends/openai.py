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
		return os.getenv("OPENAI_API_KEY")
	if isinstance(value, str):
		match = _ENV_REF_RE.match(value.strip())
		if match:
			return os.getenv(match.group(1))
		if value.startswith("env:"):
			return os.getenv(value[4:])
	return value


class OpenAIBackend:
	def __init__(self, id, api_key, engine, **config):
		self.id = id
		self.api_key = _resolve_api_key(api_key)
		if not self.api_key:
			raise ValueError("OpenAI API key is required (set OPENAI_API_KEY or pass api_key)")
		self.engine = engine
		self.config = config

		self.cache = None
		if get_llm_cache is not None:
			try:
				self.cache = get_llm_cache("openai", api_key=self.api_key)
			except Exception:
				self.cache = None

	def __call__(self, text):
		temperature = float(self.config.get("temperature", 0.0) or 0.0)
		max_tokens = self.config.get("max_tokens")
		top_p = self.config.get("top_p")
		frequency_penalty = self.config.get("frequency_penalty")
		presence_penalty = self.config.get("presence_penalty")
		stop = self.config.get("stop")
		use_cache = bool(self.config.get("use_cache", True))

		if use_cache and self.cache is not None:
			try:
				cached = self.cache.get_completion(
					prompt=text,
					model=self.engine,
					temperature=temperature,
					max_tokens=max_tokens,
					top_p=top_p,
					frequency_penalty=frequency_penalty,
					presence_penalty=presence_penalty,
					stop=stop,
				)
				if cached is not None:
					if isinstance(cached, str):
						return cached
					if isinstance(cached, dict):
						try:
							return cached["choices"][0]["text"].strip()
						except Exception:
							return cached
					return cached
			except Exception:
				pass

		r = requests.post(
			"https://api.openai.com/v1/engines/" + self.engine + "/completions",
			headers={
				"Content-Type": "application/json",
				"Authorization": "Bearer %s" % self.api_key,
			},
			data=json.dumps({"prompt": text, **self.config}),
		)

		data = json.loads(r.text)

		try:
			result = data["choices"][0]["text"].strip()
			if use_cache and self.cache is not None:
				try:
					self.cache.cache_completion(
						prompt=text,
						response=result,
						model=self.engine,
						temperature=temperature,
						max_tokens=max_tokens,
						top_p=top_p,
						frequency_penalty=frequency_penalty,
						presence_penalty=presence_penalty,
						stop=stop,
					)
				except Exception:
					pass
			return result
		except Exception:
			raise Exception("empty-response")

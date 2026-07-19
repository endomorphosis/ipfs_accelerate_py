"""Compatibility skillset wrapper for llama.cpp.

Older worker code expected a ``llama_cpp_kit`` class with chat/completion
methods.  The runtime implementation now delegates to the centralized
``ipfs_accelerate_py.llm_router`` llama.cpp provider so install, update,
server probing, and HTTP request behavior do not drift across call sites.
"""

from __future__ import annotations

import re
import os
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.utils.llama_cpp import (
    config_from_env,
    ensure_llama_cpp_server,
)


class _TaskAbortion(RuntimeError):
    pass


class llama_cpp_kit:
    def __init__(self, resources: Optional[Mapping[str, Any]] = None, meta: Optional[Mapping[str, Any]] = None):
        self.resources = dict(resources or {})
        self.meta = dict(meta or {})
        self.chat_template = str(self.meta.get("chat_template") or self.resources.get("chat_template") or "openai")
        self.model_name = str(
            self.resources.get("model")
            or self.resources.get("model_name")
            or self.resources.get("checkpoint")
            or ""
        ).replace("@gguf", "")
        self.provider_name = self._configured_provider_name()
        self.TaskAbortion = _TaskAbortion
        self.should_abort = lambda: False

    def __call__(self, method: str, **kwargs: Any):
        if method in {"llm_chat", "llama_cpp", "llama_cpp_chat"}:
            return self.chat(**kwargs)
        if method == "llm_chat_logits":
            return self.chat_logits(**kwargs)
        if method in {"llm_complete", "text_complete"}:
            return self.llm_complete(**kwargs)
        if method == "llm_logits":
            return self.logits(**kwargs)
        if method == "instruct":
            return self.instruct(**kwargs)
        raise ValueError(f"unknown method: {method}")

    def ensure_server(self, *, autostart: bool = False, auto_install: bool = False, auto_update: bool = False):
        config = config_from_env(
            **{
                key: value
                for key, value in {
                    "model_ref": self.resources.get("model_ref"),
                    "host": self.resources.get("host"),
                    "port": self.resources.get("port"),
                    "context_size": self.resources.get("context_size") or self.resources.get("contextSize"),
                    "threads": self.resources.get("threads"),
                    "gpu_layers": self.resources.get("gpu_layers"),
                    "extra_args": self.resources.get("extra_args"),
                    "log_dir": self.resources.get("log_dir"),
                }.items()
                if value is not None
            }
        )
        return ensure_llama_cpp_server(
            config,
            autostart=autostart,
            auto_install=auto_install,
            auto_update=auto_update,
        )

    def chat(self, messages: Sequence[Mapping[str, Any]], system: Optional[str] = None, **kwargs: Any):
        normalized = self._normalize_messages(messages, system=system)
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in normalized)
        return self.llm_complete(prompt=prompt, **kwargs)

    def llama_cpp_chat(self, messages: Sequence[Mapping[str, Any]], system: Optional[str] = None, **kwargs: Any):
        return self.chat(messages, system=system, **kwargs)

    def chat_logits(
        self,
        messages: Sequence[Mapping[str, Any]],
        logits_for: Sequence[str],
        chat_template: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        prompt = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self._normalize_messages(messages, system=system)
        )
        return self.logits(prompt=prompt, logits_for=logits_for, **kwargs)

    def llm_complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        stopping_regex: Optional[str] = None,
        logit_bias: Optional[Mapping[str, float]] = None,
        stop: Optional[Sequence[str] | str] = None,
        **kwargs: Any,
    ):
        text = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            logit_bias=logit_bias,
            **kwargs,
        )
        if stopping_regex:
            match = re.search(stopping_regex, text)
            if match:
                text = text[: match.start()]
        if stream:
            yield {"text": text, "done": False}
            yield {"text": "", "done": True}
        else:
            yield {"text": text, "done": True}

    def text_complete(self, prompt: str, **kwargs: Any):
        return self.llm_complete(prompt=prompt, **kwargs)

    def logits(self, prompt: str, logits_for: Sequence[str], **kwargs: Any) -> dict[str, dict[str, float]]:
        from ipfs_accelerate_py import llm_router

        top_logprobs = max(1, min(len(tuple(logits_for or ())), 20))
        response = llm_router.chat_completions_create(
            messages=[{"role": "user", "content": prompt}],
            provider=self.provider_name,
            model=self.model_name or None,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=1,
            temperature=0.0,
            **kwargs,
        )
        observed: dict[str, float] = {}
        try:
            top = response.choices[0].logprobs.content[0].top_logprobs
            observed = {entry.token: float(entry.logprob) for entry in top}
        except Exception:
            observed = {}
        return {"logits": {str(token): float(observed.get(str(token), float("-inf"))) for token in logits_for}}

    def instruct(self, context: Sequence[Mapping[str, Any]], instruction: str, **kwargs: Any):
        context_text = "\n\n".join(
            str(item.get("data") or item.get("text") or item.get("content") or "")
            for item in context or ()
            if isinstance(item, Mapping)
        )
        prompt = f"{context_text}\n\nInstruction:\n{instruction}".strip()
        return self.llm_complete(prompt=prompt, **kwargs)

    def unload(self) -> None:
        return None

    def test(self, **kwargs: Any):
        if self.provider_name != "llama_cpp":
            from ipfs_accelerate_py import llm_router

            provider = llm_router._builtin_provider_by_name(self.provider_name)
            return {"provider": self.provider_name, "available": provider is not None}
        return self.ensure_server(**kwargs).to_dict()

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        from ipfs_accelerate_py import llm_router

        return llm_router.generate_text(
            str(prompt or ""),
            provider=self.provider_name,
            model_name=self.model_name or None,
            **kwargs,
        )

    def _configured_provider_name(self) -> str:
        provider = str(
            self.meta.get("provider")
            or self.meta.get("provider_name")
            or self.resources.get("provider")
            or self.resources.get("provider_name")
            or os.getenv("IPFS_ACCELERATE_LLAMA_CPP_KIT_PROVIDER")
            or os.getenv("IPFS_ACCELERATE_LLAMA_CPP_PROVIDER")
            or "llama_cpp"
        ).strip().lower()
        if os.getenv("IPFS_ACCELERATE_LLAMA_CPP_USE_NATIVE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return "llama_cpp_native"
        return provider or "llama_cpp"

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, Any]],
        *,
        system: Optional[str] = None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        if system:
            normalized.append({"role": "system", "content": str(system)})
        for message in messages or ():
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "user")
            content = str(message.get("content") or message.get("text") or "")
            normalized.append({"role": role, "content": content})
        return normalized


export = llama_cpp_kit

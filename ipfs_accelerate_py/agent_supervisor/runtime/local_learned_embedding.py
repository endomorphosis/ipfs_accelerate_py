"""Shared actual offline safetensors provider for native learned retrieval.

Loading is local-only, uses installed built-in model classes, and disables
remote code. Runtime callers additionally require owner-controlled assets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _model_manifest(snapshot: Path) -> dict:
    """Pin local bytes, including HF cache symlink targets, without loading code."""
    snapshot = snapshot.resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("model snapshot must be an existing directory")
    inventory = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("model directory symlinks are unsupported")
            continue
        if not path.is_file():
            raise ValueError("model snapshot contains a missing or non-regular file")
        size = path.stat().st_size
        if size > 256_000_000:
            raise ValueError("model file exceeds local qualification bound")
        name = str(path.relative_to(snapshot))
        if path.suffix in {".py", ".bin", ".pt", ".pth", ".pkl", ".pickle"}:
            raise ValueError("only local safetensors model weights are supported")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory[name] = {"sha256": digest.hexdigest(), "bytes": size}
    required = {"model.safetensors", "config.json", "modules.json", "tokenizer.json"}
    if not required.issubset(inventory):
        raise ValueError("local snapshot lacks required safetensors/tokenizer/config files")
    modules = json.loads((snapshot / "modules.json").read_text())
    approved = {
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    }
    for module in modules:
        if module.get("type") not in approved:
            raise ValueError("snapshot requests an unsupported model module")
        relative = Path(module.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model module path escapes snapshot")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "bert" or config.get("auto_map"):
        raise ValueError("qualification supports local built-in BERT models only")
    return {"schema": "local-embedding-model-manifest@1", "files": inventory}


class _LocalRouterModel:
    """A real router provider with explicit offline, safetensors-only loading."""

    router_provider_name = "huggingface-local-pinned"

    def __init__(self, snapshot: Path, artifact_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.artifact_id = artifact_id
        self.calls = 0
        self.texts = 0
        self.model = SentenceTransformer(
            str(snapshot),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            model_kwargs={"use_safetensors": True, "local_files_only": True},
            config_kwargs={"local_files_only": True},
        )
        self.model.eval()

    def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
        if model_name != self.artifact_id or device != "cpu":
            raise ValueError("router invocation drifted from pinned local model")
        self.calls += 1
        self.texts += len(texts)
        return self.model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()


class _PinnedRouterBackend:
    kind = "local-safetensors-via-embeddings-router"

    def __init__(self, policy, model: _LocalRouterModel) -> None:
        from ipfs_accelerate_py.router_deps import RouterDeps
        self.policy = policy
        self.model = model
        self.traces: list[dict] = []
        # Private, initially empty local cache: no ambient/distributed values
        # can substitute for pinned model output or trigger network access.
        self.deps = RouterDeps()

    def embed(self, texts):
        from ipfs_accelerate_py import embeddings_router

        vectors = embeddings_router.embed_texts(
            texts,
            provider="huggingface",
            provider_instance=self.model,
            deps=self.deps,
            model_name=self.policy.model_artifact_id,
            device="cpu",
            normalize_embeddings=True,
            pinned_policy_id=self.policy.policy_id,
        )
        self.traces.append(embeddings_router.get_last_embedding_trace())
        if self.traces[-1].get("fallback_used") is not False:
            raise ValueError("pinned local embedding route attempted a fallback")
        return vectors

"""Worker loop for accelerate task delegation.

Designed to be run inside long-lived services (e.g. systemd). It can optionally
start the libp2p TaskQueue RPC service in-process.

Task handlers are intentionally minimal and gated:
- Always enabled: ``text-generation``
- Enabled when an accelerate instance provides ``call_tool``: ``tool.call``
- Opt-in via env: ``shell`` (disabled by default)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import importlib
import importlib.util
from .task_queue import QueuedTask, TaskQueue


def _transformers_spec_origin() -> str:
    """Best-effort module origin for diagnosing import issues."""

    try:
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            return ""
        origin = getattr(spec, "origin", None)
        if origin:
            return str(origin)
        locs = getattr(spec, "submodule_search_locations", None)
        if locs:
            locs_list = list(locs)
            if locs_list:
                return str(locs_list[0])
    except Exception:
        return ""
    return ""


def _transformers_import_ok() -> tuple[bool, str]:
    """Return (ok, detail) for importing transformers.

    We use this to avoid claiming mesh tasks on workers that will fail at
    runtime due to a broken/recursive transformers import (common when a local
    `transformers.py` shadows the package).
    """

    origin = _transformers_spec_origin()
    try:
        importlib.import_module("transformers")
        return (True, origin)
    except Exception as exc:
        if origin:
            detail = f"{origin} ({type(exc).__name__}: {exc})"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        return (False, detail)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_batch_cap(raw: object | None, *, default: int = 1) -> int:
    """Parse an env-style batch cap.

    Conventions:
      - unset/empty -> default
      - integer > 0 -> hard cap
      - 0 / "auto" -> request auto estimation (returns 0)
    """

    if raw is None:
        return int(default)
    s = str(raw).strip().lower()
    if not s:
        return int(default)
    if s in {"auto"}:
        return 0
    try:
        v = int(float(s))
        return v
    except Exception:
        return int(default)


def _minimal_hf_enabled() -> bool:
    # Reuse existing gating used by the drain harness.
    if _truthy(os.environ.get("IPFS_ACCEL_SKIP_CORE")):
        return True
    if _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM")):
        return True
    # General-purpose minimal HF inference for non-textgen tasks.
    if _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_HF")):
        return True
    return False


_HF_TEXTGEN_LOCK = threading.RLock()
_HF_TEXTGEN_PIPELINE: object | None = None
_HF_TEXTGEN_MODEL_ID: str | None = None
_HF_TEXTGEN_MODEL_BYTES: int | None = None
_HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH: int | None = None

_HF_TEXT2TEXT_LOCK = threading.RLock()
_HF_TEXT2TEXT_PIPELINE: object | None = None
_HF_TEXT2TEXT_MODEL_ID: str | None = None
_HF_TEXT2TEXT_MODEL_BYTES: int | None = None
_HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH: int | None = None

_HF_TEXTCLS_LOCK = threading.RLock()
_HF_TEXTCLS_PIPELINE: object | None = None
_HF_TEXTCLS_MODEL_ID: str | None = None
_HF_TEXTCLS_MODEL_BYTES: int | None = None
_HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH: int | None = None

_HF_EMBED_LOCK = threading.RLock()
_HF_EMBED_MODEL: object | None = None
_HF_EMBED_TOKENIZER: object | None = None
_HF_EMBED_MODEL_ID: str | None = None
_HF_EMBED_MODEL_BYTES: int | None = None
_HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH: int | None = None


def _available_ram_bytes() -> int:
    """Best-effort available system RAM bytes (Linux-friendly)."""

    # Prefer psutil if installed.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(getattr(vm, "available", 0) or 0)
    except Exception:
        pass

    # Linux fallback.
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return int(kb) * 1024
    except Exception:
        pass
    return 0


def _available_vram_bytes() -> int:
    """Best-effort free VRAM bytes for CUDA device 0 (if available).

    Deprecated: prefer `_cuda_free_bytes(device_index=...)` or
    `_available_vram_bytes_for_model(model)`.
    """

    return _cuda_free_bytes(device_index=0)


def _cuda_device_count() -> int:
    try:
        import torch  # type: ignore

        if not bool(getattr(torch, "cuda", None)):
            return 0
        if not torch.cuda.is_available():
            return 0
        return max(0, int(torch.cuda.device_count()))
    except Exception:
        return 0


def _cuda_free_bytes(*, device_index: int) -> int:
    """Best-effort free VRAM bytes for a CUDA device."""

    try:
        import torch  # type: ignore

        if not bool(getattr(torch, "cuda", None)):
            return 0
        if not torch.cuda.is_available():
            return 0
        idx = int(device_index)
        if idx < 0 or idx >= int(torch.cuda.device_count()):
            return 0
        free_b, _total_b = torch.cuda.mem_get_info(idx)
        return int(free_b)
    except Exception:
        return 0


def _cuda_free_bytes_all() -> int:
    """Best-effort sum of free VRAM across all CUDA devices."""

    total = 0
    for i in range(_cuda_device_count()):
        total += int(_cuda_free_bytes(device_index=i))
    return int(total)


def _cuda_best_device_index() -> int | None:
    """Return the CUDA device index with the most free VRAM."""

    best_i: int | None = None
    best_free = -1
    for i in range(_cuda_device_count()):
        free_b = int(_cuda_free_bytes(device_index=i))
        if free_b > best_free:
            best_free = free_b
            best_i = int(i)
    return best_i


def _hf_device_pref() -> str:
    """Parse desired HF device preference.

    Returns a normalized string:
      - 'auto'
      - 'cpu'
      - 'cuda:{i}'
    """

    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_DEVICE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_HF_DEVICE")
        or "auto"
    )
    s = str(raw).strip().lower()
    if not s or s in {"auto"}:
        return "auto"
    if s in {"cpu", "-1"}:
        return "cpu"
    if s in {"cuda", "gpu"}:
        return "cuda:0"
    if s.startswith("cuda:"):
        tail = s.split(":", 1)[1].strip()
        try:
            return f"cuda:{int(float(tail))}"
        except Exception:
            return "auto"
    try:
        return f"cuda:{int(float(s))}"
    except Exception:
        return "auto"


def _hf_cuda_max_memory_map() -> dict[int, int]:
    """Compute a conservative `max_memory` map for Transformers device_map.

    Uses per-device free VRAM at runtime, reserving some slack.
    """

    try:
        frac_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_GPU_MAX_MEMORY_FRACTION")
        frac = float(frac_raw) if frac_raw is not None else 0.92
    except Exception:
        frac = 0.92
    frac = max(0.1, min(0.99, float(frac)))

    try:
        reserve_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_GPU_MEMORY_RESERVE_MB")
        reserve_mb = int(float(reserve_raw)) if reserve_raw is not None else 768
    except Exception:
        reserve_mb = 768
    reserve_b = max(0, int(reserve_mb)) * 1024 * 1024

    mm: dict[int, int] = {}
    for i in range(_cuda_device_count()):
        free_b = int(_cuda_free_bytes(device_index=i))
        allow = int(free_b * frac) - int(reserve_b)
        # Transformers expects positive values; skip unusable devices.
        if allow > 256 * 1024 * 1024:
            mm[int(i)] = int(allow)
    return mm


def _available_vram_bytes_for_model(model: object) -> int:
    """Best-effort free VRAM budget for the model placement.

    - If the model is sharded (`hf_device_map`), use sum free VRAM.
    - If the model is on a single CUDA device, use free VRAM of that device.
    - Otherwise return 0.
    """

    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            used: set[int] = set()
            for v in device_map.values():
                s = str(v)
                if s.startswith("cuda:"):
                    try:
                        used.add(int(s.split(":", 1)[1]))
                    except Exception:
                        continue
            # Conservative: treat usable budget as the minimum free VRAM among
            # devices participating in the shard.
            if used:
                mins: list[int] = [int(_cuda_free_bytes(device_index=i)) for i in sorted(used)]
                mins = [m for m in mins if m > 0]
                return int(min(mins)) if mins else 0
            return int(_cuda_free_bytes_all())
    except Exception:
        pass

    try:
        import torch  # type: ignore

        for p in getattr(model, "parameters", lambda: [])():
            dev = getattr(p, "device", None)
            if dev is not None and str(getattr(dev, "type", "")) == "cuda":
                idx = getattr(dev, "index", None)
                if idx is None:
                    # If index is unknown, treat as aggregate.
                    return int(_cuda_free_bytes_all())
                return int(_cuda_free_bytes(device_index=int(idx)))
        return 0
    except Exception:
        return 0


def _hf_pipeline_device() -> int:
    """Return HF pipeline `device` arg.

    Convention (HF transformers):
      -1 => CPU
      >=0 => CUDA device index

    Override via env:
      IPFS_ACCELERATE_PY_TASK_WORKER_HF_DEVICE=auto|cpu|cuda|0|1|cuda:0|cuda:1
    """

    # NOTE: this helper returns a *single* pipeline `device` value.
    # Multi-GPU sharding is handled separately via `device_map='auto'`.
    pref = _hf_device_pref()
    if pref == "cpu":
        return -1
    if pref.startswith("cuda:"):
        try:
            return int(float(pref.split(":", 1)[1]))
        except Exception:
            return 0

    best = _cuda_best_device_index()
    if best is not None:
        return int(best)
    return -1


def _hf_should_use_device_map_auto() -> bool:
    """True if we should attempt Transformers `device_map='auto'` sharding."""

    if _hf_device_pref() != "auto":
        return False
    if _cuda_device_count() <= 0:
        return False
    # We *attempt* device_map=auto even on 1 GPU because it can still
    # automatically choose dtype/placement. If accelerate isn't installed, we'll
    # fall back to single-device.
    return True


def _hf_pipeline_device_kind(model: object) -> str:
    try:
        import torch  # type: ignore

        for p in getattr(model, "parameters", lambda: [])():
            dev = getattr(p, "device", None)
            if dev is not None:
                return "cuda" if str(getattr(dev, "type", "")) == "cuda" else "cpu"
        return "cpu"
    except Exception:
        return "cpu"


def _hf_model_primary_input_device(model: object) -> str:
    """Return a best-effort device string for placing input tensors.

    For sharded models, we choose the first CUDA device in the hf_device_map.
    """

    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            cuda_idxs: list[int] = []
            for v in device_map.values():
                s = str(v)
                if s.startswith("cuda:"):
                    try:
                        cuda_idxs.append(int(s.split(":", 1)[1]))
                    except Exception:
                        continue
            if cuda_idxs:
                return f"cuda:{min(cuda_idxs)}"
    except Exception:
        pass

    try:
        for p in getattr(model, "parameters", lambda: [])():
            dev = getattr(p, "device", None)
            if dev is not None:
                t = str(getattr(dev, "type", "cpu"))
                idx = getattr(dev, "index", None)
                if t == "cuda" and idx is not None:
                    return f"cuda:{int(idx)}"
                return str(t)
    except Exception:
        pass
    return "cpu"


def _hf_estimate_max_batch_size(
    *,
    model: object,
    max_seq_len_tokens: int,
) -> int:
    """Estimate a safe max batch size for causal LM generation.

    Uses a simple budget model:
      available_memory * fraction - model_bytes >= batch * seq_len * kv_bytes_per_token_per_batch

    KV cache approximation (per batch, per token):
      num_layers * 2(K/V) * hidden_size * bytes_per_elem

    This is intentionally conservative and meant to prevent OOM.
    """

    global _HF_TEXTGEN_MODEL_BYTES, _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH

    max_seq_len = max(1, int(max_seq_len_tokens or 1))

    # Determine memory kind based on where the model lives.
    kind = _hf_pipeline_device_kind(model)
    avail = _available_vram_bytes_for_model(model) if kind == "cuda" else _available_ram_bytes()
    if avail <= 0:
        return 1

    # Fraction of free memory we're willing to consume for generation.
    # Default: a bit more conservative on GPU.
    try:
        frac_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TEXTGEN_BATCH_MEM_FRACTION")
        frac = float(frac_raw) if frac_raw is not None else (0.55 if kind == "cuda" else 0.65)
    except Exception:
        frac = 0.55 if kind == "cuda" else 0.65
    frac = max(0.1, min(frac, 0.95))

    # Reserve some slack for Python/runtime overhead.
    try:
        reserve_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TEXTGEN_BATCH_RESERVE_MB")
        reserve_bytes = int(float(reserve_raw) * 1024 * 1024) if reserve_raw is not None else (512 * 1024 * 1024)
    except Exception:
        reserve_bytes = 512 * 1024 * 1024
    reserve_bytes = max(0, reserve_bytes)

    budget = int(avail * frac) - int(reserve_bytes)
    if budget <= 0:
        return 1

    # Cache model bytes and KV bytes-per-token-per-batch for this loaded model.
    if _HF_TEXTGEN_MODEL_BYTES is None:
        try:
            total = 0
            for p in getattr(model, "parameters", lambda: [])():
                try:
                    total += int(p.numel()) * int(p.element_size())
                except Exception:
                    continue
            _HF_TEXTGEN_MODEL_BYTES = int(total)
        except Exception:
            _HF_TEXTGEN_MODEL_BYTES = 0

    if _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH is None:
        try:
            cfg = getattr(model, "config", None)
            n_layers = int(
                getattr(cfg, "n_layer", None)
                or getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "n_layers", None)
                or 0
            )
            hidden = int(
                getattr(cfg, "n_embd", None)
                or getattr(cfg, "hidden_size", None)
                or getattr(cfg, "d_model", None)
                or 0
            )

            elem_bytes = 4
            try:
                # Use the first parameter dtype.
                for p in getattr(model, "parameters", lambda: [])():
                    elem_bytes = int(p.element_size())
                    break
            except Exception:
                elem_bytes = 4

            # KV cache: K and V per layer.
            kv = int(max(1, n_layers)) * 2 * int(max(1, hidden)) * int(max(1, elem_bytes))
            # Apply a safety multiplier for attention/activation overhead.
            kv = int(kv * 1.35)
            _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH = int(kv)
        except Exception:
            _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH = 0

    model_bytes = int(_HF_TEXTGEN_MODEL_BYTES or 0)
    kv_per_token = int(_HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH or 0)
    if kv_per_token <= 0:
        return 1

    remaining = budget - model_bytes
    if remaining <= 0:
        return 1

    per_batch = int(kv_per_token) * int(max_seq_len)
    if per_batch <= 0:
        return 1

    max_b = int(remaining // per_batch)
    return max(1, min(max_b, 128))


def _hf_estimate_max_batch_size_encoder(
    *,
    model: object,
    max_seq_len_tokens: int,
    cache_act_bytes_ref: list[int | None],
    cache_model_bytes_ref: list[int | None],
) -> int:
    """Conservative estimator for encoder-style inference (embeddings/classification).

    Budgets (weights + per-token activation scratch). This is intentionally rough
    and biased toward avoiding OOM.
    """

    max_seq_len = max(1, int(max_seq_len_tokens or 1))
    kind = _hf_pipeline_device_kind(model)
    avail = _available_vram_bytes_for_model(model) if kind == "cuda" else _available_ram_bytes()
    if avail <= 0:
        return 1

    try:
        frac_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_MEM_FRACTION")
        frac = float(frac_raw) if frac_raw is not None else (0.75 if kind == "cuda" else 0.60)
    except Exception:
        frac = 0.75 if kind == "cuda" else 0.60
    frac = max(0.05, min(0.95, float(frac)))

    try:
        reserve_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_MEM_RESERVE_MB")
        reserve_mb = int(float(reserve_raw)) if reserve_raw is not None else (1024 if kind == "cuda" else 2048)
    except Exception:
        reserve_mb = 1024 if kind == "cuda" else 2048
    reserve_bytes = max(0, int(reserve_mb)) * 1024 * 1024

    budget = int(avail * frac) - int(reserve_bytes)
    if budget <= 0:
        return 1

    if cache_model_bytes_ref and cache_model_bytes_ref[0] is None:
        try:
            total = 0
            for p in getattr(model, "parameters", lambda: [])():
                try:
                    total += int(p.numel()) * int(p.element_size())
                except Exception:
                    continue
            cache_model_bytes_ref[0] = int(total)
        except Exception:
            cache_model_bytes_ref[0] = 0

    model_bytes = int((cache_model_bytes_ref[0] or 0) if cache_model_bytes_ref else 0)
    remaining = budget - model_bytes
    if remaining <= 0:
        return 1

    if cache_act_bytes_ref and cache_act_bytes_ref[0] is None:
        act = 0
        try:
            cfg = getattr(model, "config", None)
            hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or 0)
            layers = int(
                getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "num_layers", None)
                or getattr(cfg, "n_layer", None)
                or getattr(cfg, "encoder_layers", None)
                or 0
            )
            if hidden <= 0:
                hidden = 768
            if layers <= 0:
                layers = 12

            elem_bytes = 2
            try:
                for p in getattr(model, "parameters", lambda: [])():
                    elem_bytes = int(p.element_size())
                    break
            except Exception:
                elem_bytes = 2

            try:
                mult_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_ACT_MULT")
                mult = float(mult_raw) if mult_raw is not None else 8.0
            except Exception:
                mult = 8.0
            mult = max(2.0, min(32.0, float(mult)))

            act = int(layers) * int(hidden) * int(max(1, elem_bytes))
            act = int(act * mult)
        except Exception:
            act = 0
        cache_act_bytes_ref[0] = int(act)

    act_per_token = int((cache_act_bytes_ref[0] or 0) if cache_act_bytes_ref else 0)
    if act_per_token <= 0:
        return 1

    per_batch = int(act_per_token) * int(max_seq_len)
    if per_batch <= 0:
        return 1
    max_b = int(remaining // per_batch)
    return max(1, min(int(max_b), 128))


def _hf_get_textgen_pipeline(*, requested_model: str) -> object:
    """Get or create the cached HF text-generation pipeline."""

    global _HF_TEXTGEN_PIPELINE, _HF_TEXTGEN_MODEL_ID, _HF_TEXTGEN_MODEL_BYTES, _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    with _HF_TEXTGEN_LOCK:
        if _HF_TEXTGEN_PIPELINE is None or _HF_TEXTGEN_MODEL_ID != requested_model:
            tokenizer = AutoTokenizer.from_pretrained(requested_model)
            # Automatic placement:
            # - If HF_DEVICE is pinned (cpu/cuda:i), load normally and let pipeline move.
            # - If HF_DEVICE=auto and CUDA is available, prefer device_map='auto'
            #   with a max_memory budget based on free VRAM.
            pref = _hf_device_pref()
            device_arg: int = -1
            model_kwargs: dict[str, Any] = {}

            if pref.startswith("cuda:"):
                try:
                    device_arg = int(float(pref.split(":", 1)[1]))
                except Exception:
                    device_arg = _hf_pipeline_device()
                model = AutoModelForCausalLM.from_pretrained(requested_model)
            elif pref == "cpu":
                device_arg = -1
                model = AutoModelForCausalLM.from_pretrained(requested_model)
            else:
                # auto
                model = None
                if _hf_should_use_device_map_auto():
                    try:
                        mm = _hf_cuda_max_memory_map()
                        model_kwargs = {
                            "device_map": "auto",
                            "max_memory": mm if mm else None,
                            "torch_dtype": "auto",
                            "low_cpu_mem_usage": True,
                        }
                        # Remove None entries for older transformers.
                        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                        model = AutoModelForCausalLM.from_pretrained(requested_model, **model_kwargs)
                        # When sharded, do not let pipeline relocate the model.
                        device_arg = -1
                    except Exception:
                        model = None

                if model is None:
                    # Fallback: pick the best single GPU (most free VRAM) and
                    # let pipeline move it.
                    best = _cuda_best_device_index()
                    device_arg = int(best) if best is not None else -1
                    model = AutoModelForCausalLM.from_pretrained(requested_model)
            if (
                getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token_id = tokenizer.eos_token_id

            _HF_TEXTGEN_PIPELINE = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=int(device_arg),
            )
            _HF_TEXTGEN_MODEL_ID = requested_model
            _HF_TEXTGEN_MODEL_BYTES = None
            _HF_TEXTGEN_KV_BYTES_PER_TOKEN_PER_BATCH = None

        return _HF_TEXTGEN_PIPELINE


def _hf_get_text2text_pipeline(*, requested_model: str) -> object:
    """Get or create the cached HF text2text-generation pipeline."""

    global _HF_TEXT2TEXT_PIPELINE, _HF_TEXT2TEXT_MODEL_ID, _HF_TEXT2TEXT_MODEL_BYTES, _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline  # type: ignore

    with _HF_TEXT2TEXT_LOCK:
        if _HF_TEXT2TEXT_PIPELINE is None or _HF_TEXT2TEXT_MODEL_ID != requested_model:
            tokenizer = AutoTokenizer.from_pretrained(requested_model)
            pref = _hf_device_pref()
            device_arg: int = -1

            if pref.startswith("cuda:"):
                try:
                    device_arg = int(float(pref.split(":", 1)[1]))
                except Exception:
                    device_arg = _hf_pipeline_device()
                model = AutoModelForSeq2SeqLM.from_pretrained(requested_model)
            elif pref == "cpu":
                device_arg = -1
                model = AutoModelForSeq2SeqLM.from_pretrained(requested_model)
            else:
                model = None
                if _hf_should_use_device_map_auto():
                    try:
                        mm = _hf_cuda_max_memory_map()
                        kwargs: dict[str, Any] = {
                            "device_map": "auto",
                            "max_memory": mm if mm else None,
                            "torch_dtype": "auto",
                            "low_cpu_mem_usage": True,
                        }
                        kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        model = AutoModelForSeq2SeqLM.from_pretrained(requested_model, **kwargs)
                        device_arg = -1
                    except Exception:
                        model = None
                if model is None:
                    best = _cuda_best_device_index()
                    device_arg = int(best) if best is not None else -1
                    model = AutoModelForSeq2SeqLM.from_pretrained(requested_model)
            if (
                getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token_id = tokenizer.eos_token_id
            _HF_TEXT2TEXT_PIPELINE = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=int(device_arg),
            )
            _HF_TEXT2TEXT_MODEL_ID = requested_model
            _HF_TEXT2TEXT_MODEL_BYTES = None
            _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH = None
        return _HF_TEXT2TEXT_PIPELINE


def _hf_get_textcls_pipeline(*, requested_model: str) -> object:
    """Get or create the cached HF text-classification pipeline."""

    global _HF_TEXTCLS_PIPELINE, _HF_TEXTCLS_MODEL_ID, _HF_TEXTCLS_MODEL_BYTES, _HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # type: ignore

    with _HF_TEXTCLS_LOCK:
        if _HF_TEXTCLS_PIPELINE is None or _HF_TEXTCLS_MODEL_ID != requested_model:
            tokenizer = AutoTokenizer.from_pretrained(requested_model)
            pref = _hf_device_pref()
            device_arg: int = -1

            if pref.startswith("cuda:"):
                try:
                    device_arg = int(float(pref.split(":", 1)[1]))
                except Exception:
                    device_arg = _hf_pipeline_device()
                model = AutoModelForSequenceClassification.from_pretrained(requested_model)
            elif pref == "cpu":
                device_arg = -1
                model = AutoModelForSequenceClassification.from_pretrained(requested_model)
            else:
                model = None
                if _hf_should_use_device_map_auto():
                    try:
                        mm = _hf_cuda_max_memory_map()
                        kwargs: dict[str, Any] = {
                            "device_map": "auto",
                            "max_memory": mm if mm else None,
                            "torch_dtype": "auto",
                            "low_cpu_mem_usage": True,
                        }
                        kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        model = AutoModelForSequenceClassification.from_pretrained(requested_model, **kwargs)
                        device_arg = -1
                    except Exception:
                        model = None
                if model is None:
                    best = _cuda_best_device_index()
                    device_arg = int(best) if best is not None else -1
                    model = AutoModelForSequenceClassification.from_pretrained(requested_model)
            _HF_TEXTCLS_PIPELINE = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=int(device_arg),
            )
            _HF_TEXTCLS_MODEL_ID = requested_model
            _HF_TEXTCLS_MODEL_BYTES = None
            _HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH = None
        return _HF_TEXTCLS_PIPELINE


def _hf_get_embed_components(*, requested_model: str) -> tuple[object, object]:
    """Get or create cached (tokenizer, model) for embeddings."""

    global _HF_EMBED_MODEL, _HF_EMBED_TOKENIZER, _HF_EMBED_MODEL_ID, _HF_EMBED_MODEL_BYTES, _HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH

    from transformers import AutoModel, AutoTokenizer  # type: ignore

    with _HF_EMBED_LOCK:
        if _HF_EMBED_MODEL is None or _HF_EMBED_TOKENIZER is None or _HF_EMBED_MODEL_ID != requested_model:
            tok = AutoTokenizer.from_pretrained(requested_model)
            pref = _hf_device_pref()
            if pref.startswith("cuda:"):
                model = AutoModel.from_pretrained(requested_model)
                # Move the model explicitly for embedding forward() path.
                try:
                    dev = pref
                    model = model.to(dev)
                except Exception:
                    pass
            elif pref == "cpu":
                model = AutoModel.from_pretrained(requested_model)
            else:
                model = None
                if _hf_should_use_device_map_auto():
                    try:
                        mm = _hf_cuda_max_memory_map()
                        kwargs: dict[str, Any] = {
                            "device_map": "auto",
                            "max_memory": mm if mm else None,
                            "torch_dtype": "auto",
                            "low_cpu_mem_usage": True,
                        }
                        kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        model = AutoModel.from_pretrained(requested_model, **kwargs)
                    except Exception:
                        model = None
                if model is None:
                    model = AutoModel.from_pretrained(requested_model)
                    best = _cuda_best_device_index()
                    if best is not None:
                        try:
                            model = model.to(f"cuda:{int(best)}")
                        except Exception:
                            pass
            _HF_EMBED_TOKENIZER = tok
            _HF_EMBED_MODEL = model
            _HF_EMBED_MODEL_ID = requested_model
            _HF_EMBED_MODEL_BYTES = None
            _HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH = None
        return (_HF_EMBED_TOKENIZER, _HF_EMBED_MODEL)


def _hf_textgen(prompt: str, *, model_name: str | None, max_new_tokens: int, temperature: float) -> str:
    """Minimal local text-generation without importing ipfs_accelerate_py core."""

    global _HF_TEXTGEN_PIPELINE, _HF_TEXTGEN_MODEL_ID

    requested_model = str(model_name or os.environ.get("IPFS_ACCELERATE_PY_LLM_MODEL") or "gpt2").strip() or "gpt2"
    safe_max_new = max(1, min(int(max_new_tokens or 128), 1024))
    temp = float(temperature) if temperature is not None else 0.2

    try:
        gen = _hf_get_textgen_pipeline(requested_model=requested_model)
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "transformers is required for minimal text-generation; install it (and ensure it is importable)"
        ) from exc
    except RecursionError as exc:
        origin = _transformers_spec_origin()
        hint = (
            f" (origin={origin})" if origin else ""
        )
        raise RuntimeError(
            "failed to import transformers (RecursionError) for minimal text-generation"
            + hint
            + "; this often means a local 'transformers.py' is shadowing the package"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"minimal text-generation failed: {type(exc).__name__}: {exc}") from exc

    # Pipeline calls are not guaranteed thread-safe; guard the call.
    with _HF_TEXTGEN_LOCK:
        out = gen(
            str(prompt or ""),
            max_new_tokens=safe_max_new,
            do_sample=temp > 0,
            temperature=max(temp, 1e-6),
            pad_token_id=getattr(getattr(gen, "tokenizer", None), "pad_token_id", None),
        )

    if isinstance(out, list) and out and isinstance(out[0], dict):
        text = out[0].get("generated_text")
        if isinstance(text, str):
            return text
    return str(out)


def _hf_textgen_batch(
    prompts: list[str],
    *,
    model_name: str | None,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Minimal local batched text-generation.

    Uses the same cached HF pipeline as `_hf_textgen`, but submits a list of
    prompts in one call to reduce per-request overhead.
    """

    if not prompts:
        return []

    requested_model = str(model_name or os.environ.get("IPFS_ACCELERATE_PY_LLM_MODEL") or "gpt2").strip() or "gpt2"
    safe_max_new = max(1, min(int(max_new_tokens or 128), 1024))
    temp = float(temperature) if temperature is not None else 0.2

    try:
        gen = _hf_get_textgen_pipeline(requested_model=requested_model)
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "transformers is required for minimal text-generation; install it (and ensure it is importable)"
        ) from exc
    except RecursionError as exc:
        origin = _transformers_spec_origin()
        hint = (
            f" (origin={origin})" if origin else ""
        )
        raise RuntimeError(
            "failed to import transformers (RecursionError) for minimal text-generation"
            + hint
            + "; this often means a local 'transformers.py' is shadowing the package"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"minimal text-generation failed: {type(exc).__name__}: {exc}") from exc

    with _HF_TEXTGEN_LOCK:
        out = gen(
            [str(p or "") for p in prompts],
            max_new_tokens=safe_max_new,
            do_sample=temp > 0,
            temperature=max(temp, 1e-6),
            pad_token_id=getattr(getattr(gen, "tokenizer", None), "pad_token_id", None),
        )

    texts: list[str] = []
    if isinstance(out, list):
        for item in out:
            if isinstance(item, list) and item and isinstance(item[0], dict):
                v = item[0].get("generated_text")
                texts.append(str(v) if v is not None else str(item))
            elif isinstance(item, dict):
                v = item.get("generated_text")
                texts.append(str(v) if v is not None else str(item))
            else:
                texts.append(str(item))
        return texts

    # Unexpected shape; fall back to per-prompt.
    return [_hf_textgen(p, model_name=requested_model, max_new_tokens=safe_max_new, temperature=temp) for p in prompts]


def _hf_textgen_batch_auto(
    prompts: list[str],
    *,
    model_name: str | None,
    max_new_tokens: int,
    temperature: float,
    requested_batch_max: int,
) -> tuple[list[str], int]:
    """Generate for a batch, auto-capping size based on memory."""

    if not prompts:
        return ([], 0)

    requested_model = str(model_name or os.environ.get("IPFS_ACCELERATE_PY_LLM_MODEL") or "gpt2").strip() or "gpt2"
    safe_max_new = max(1, min(int(max_new_tokens or 128), 1024))
    temp = float(temperature) if temperature is not None else 0.2

    gen = _hf_get_textgen_pipeline(requested_model=requested_model)
    tokenizer = getattr(gen, "tokenizer", None)
    model = getattr(gen, "model", None)

    # Estimate max seq len in tokens for this batch.
    max_prompt_tokens = 0
    if tokenizer is not None:
        try:
            for p in prompts:
                ids = getattr(tokenizer, "encode", None)
                if callable(ids):
                    n = len(tokenizer.encode(str(p or "")))
                else:
                    n = len(tokenizer(str(p or ""), add_special_tokens=False).get("input_ids") or [])
                if n > max_prompt_tokens:
                    max_prompt_tokens = n
        except Exception:
            max_prompt_tokens = 0

    max_seq_len = int(max_prompt_tokens) + int(safe_max_new)

    auto_max = 1
    if model is not None:
        try:
            auto_max = _hf_estimate_max_batch_size(model=model, max_seq_len_tokens=max_seq_len)
        except Exception:
            auto_max = 1

    cap = int(requested_batch_max) if int(requested_batch_max) > 0 else int(auto_max)
    cap = max(1, min(cap, int(auto_max), 128, len(prompts)))
    texts = _hf_textgen_batch(
        prompts[:cap],
        model_name=requested_model,
        max_new_tokens=safe_max_new,
        temperature=temp,
    )
    return (texts, cap)


def _extract_hf_input_text(task_payload: Any) -> str:
    if isinstance(task_payload, dict):
        for k in ("prompt", "text", "input", "inputs", "sentence", "query"):
            v = task_payload.get(k)
            if isinstance(v, str):
                return v
    if isinstance(task_payload, str):
        return task_payload
    return str(task_payload or "")


def _hf_text2text_batch_auto(
    prompts: list[str],
    *,
    model_name: str | None,
    max_new_tokens: int,
    temperature: float,
    requested_batch_max: int,
) -> tuple[list[str], int]:
    """Seq2seq generation batch with optional auto batch-size estimation."""

    if not prompts:
        return ([], 0)

    requested_model = str(model_name or os.environ.get("IPFS_ACCELERATE_PY_LLM_MODEL") or "t5-small").strip() or "t5-small"
    safe_max_new = max(1, min(int(max_new_tokens or 128), 1024))
    temp = float(temperature) if temperature is not None else 0.2

    try:
        gen = _hf_get_text2text_pipeline(requested_model=requested_model)
    except Exception as exc:
        raise RuntimeError(f"transformers is required for minimal text2text-generation: {exc}")

    tokenizer = getattr(gen, "tokenizer", None)
    model = getattr(gen, "model", None)

    max_prompt_tokens = 0
    if tokenizer is not None:
        try:
            for p in prompts:
                ids = getattr(tokenizer, "encode", None)
                if callable(ids):
                    n = len(tokenizer.encode(str(p or "")))
                else:
                    n = len(tokenizer(str(p or ""), add_special_tokens=False).get("input_ids") or [])
                if n > max_prompt_tokens:
                    max_prompt_tokens = n
        except Exception:
            max_prompt_tokens = 0

    # Conservative proxy: input + output tokens.
    max_seq_len = int(max_prompt_tokens) + int(safe_max_new)

    auto_max = 1
    if model is not None:
        # Cache model bytes + kv per token for this model.
        global _HF_TEXT2TEXT_MODEL_BYTES, _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH

        if _HF_TEXT2TEXT_MODEL_BYTES is None:
            try:
                total = 0
                for p in getattr(model, "parameters", lambda: [])():
                    try:
                        total += int(p.numel()) * int(p.element_size())
                    except Exception:
                        continue
                _HF_TEXT2TEXT_MODEL_BYTES = int(total)
            except Exception:
                _HF_TEXT2TEXT_MODEL_BYTES = 0

        if _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH is None:
            try:
                cfg = getattr(model, "config", None)
                layers = int(
                    getattr(cfg, "num_decoder_layers", None)
                    or getattr(cfg, "decoder_layers", None)
                    or getattr(cfg, "num_layers", None)
                    or getattr(cfg, "n_layer", None)
                    or 0
                )
                hidden = int(getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", None) or 0)
                if layers <= 0:
                    layers = 12
                if hidden <= 0:
                    hidden = 768
                elem_bytes = 2
                try:
                    for p in getattr(model, "parameters", lambda: [])():
                        elem_bytes = int(p.element_size())
                        break
                except Exception:
                    elem_bytes = 2
                kv = int(layers) * 2 * int(hidden) * int(max(1, elem_bytes))
                kv = int(kv * 1.35)
                _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH = int(kv)
            except Exception:
                _HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH = 0

        kind = _hf_pipeline_device_kind(model)
        avail = _available_vram_bytes() if kind == "cuda" else _available_ram_bytes()
        if avail > 0:
            try:
                frac_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_MEM_FRACTION")
                frac = float(frac_raw) if frac_raw is not None else (0.75 if kind == "cuda" else 0.60)
            except Exception:
                frac = 0.75 if kind == "cuda" else 0.60
            frac = max(0.05, min(0.95, float(frac)))

            try:
                reserve_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_HF_MEM_RESERVE_MB")
                reserve_mb = int(float(reserve_raw)) if reserve_raw is not None else (1024 if kind == "cuda" else 2048)
            except Exception:
                reserve_mb = 1024 if kind == "cuda" else 2048
            reserve_bytes = max(0, int(reserve_mb)) * 1024 * 1024

            budget = int(avail * frac) - int(reserve_bytes)
            remaining = int(budget) - int(_HF_TEXT2TEXT_MODEL_BYTES or 0)
            denom = int(_HF_TEXT2TEXT_KV_BYTES_PER_TOKEN_PER_BATCH or 0) * int(max_seq_len)
            if remaining > 0 and denom > 0:
                auto_max = int(remaining // denom)
                auto_max = max(1, min(int(auto_max), 128))

    cap = int(requested_batch_max) if int(requested_batch_max) > 0 else int(auto_max)
    cap = max(1, min(cap, int(auto_max), 128, len(prompts)))

    with _HF_TEXT2TEXT_LOCK:
        out = gen(
            [str(p or "") for p in prompts[:cap]],
            max_new_tokens=int(safe_max_new),
            do_sample=temp > 0,
            temperature=max(float(temp), 1e-6),
            pad_token_id=getattr(getattr(gen, "tokenizer", None), "pad_token_id", None),
        )

    texts: list[str] = []
    if isinstance(out, list):
        for item in out:
            if isinstance(item, dict):
                texts.append(str(item.get("generated_text") or item.get("text") or item.get("output") or ""))
            else:
                texts.append(str(item))
    else:
        texts = [str(out)]
    return (texts, int(cap))


def _hf_textcls_batch_auto(
    texts: list[str],
    *,
    model_name: str | None,
    requested_batch_max: int,
) -> tuple[list[Any], int]:
    if not texts:
        return ([], 0)

    requested_model = str(model_name or "distilbert-base-uncased-finetuned-sst-2-english").strip() or "distilbert-base-uncased-finetuned-sst-2-english"

    try:
        clf = _hf_get_textcls_pipeline(requested_model=requested_model)
    except Exception as exc:
        raise RuntimeError(f"transformers is required for minimal text-classification: {exc}")

    tokenizer = getattr(clf, "tokenizer", None)
    model = getattr(clf, "model", None)

    max_seq_len = 1
    if tokenizer is not None:
        try:
            for t in texts:
                ids = getattr(tokenizer, "encode", None)
                if callable(ids):
                    n = len(tokenizer.encode(str(t or "")))
                else:
                    n = len(tokenizer(str(t or ""), add_special_tokens=False).get("input_ids") or [])
                if n > max_seq_len:
                    max_seq_len = n
        except Exception:
            max_seq_len = 1

    auto_max = 1
    if model is not None:
        global _HF_TEXTCLS_MODEL_BYTES, _HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH
        act_ref = [_HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH]
        model_ref = [_HF_TEXTCLS_MODEL_BYTES]
        auto_max = _hf_estimate_max_batch_size_encoder(
            model=model,
            max_seq_len_tokens=int(max_seq_len),
            cache_act_bytes_ref=act_ref,
            cache_model_bytes_ref=model_ref,
        )
        _HF_TEXTCLS_ACT_BYTES_PER_TOKEN_PER_BATCH = act_ref[0]
        _HF_TEXTCLS_MODEL_BYTES = model_ref[0]

    cap = int(requested_batch_max) if int(requested_batch_max) > 0 else int(auto_max)
    cap = max(1, min(cap, int(auto_max), 128, len(texts)))

    with _HF_TEXTCLS_LOCK:
        out = clf([str(t or "") for t in texts[:cap]], truncation=True)
    if isinstance(out, list):
        return (out, int(cap))
    return ([out], int(cap))


def _hf_embed_batch_auto(
    texts: list[str],
    *,
    model_name: str | None,
    requested_batch_max: int,
) -> tuple[list[list[float]], int]:
    if not texts:
        return ([], 0)

    requested_model = str(model_name or "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2"

    try:
        tok, model = _hf_get_embed_components(requested_model=requested_model)
    except Exception as exc:
        raise RuntimeError(f"transformers is required for minimal embedding: {exc}")

    max_seq_len = 1
    try:
        for t in texts:
            enc = tok(str(t or ""), add_special_tokens=True, truncation=False)
            ids = enc.get("input_ids") if isinstance(enc, dict) else None
            if isinstance(ids, list):
                max_seq_len = max(max_seq_len, len(ids))
    except Exception:
        max_seq_len = 1

    global _HF_EMBED_MODEL_BYTES, _HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH
    act_ref = [_HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH]
    model_ref = [_HF_EMBED_MODEL_BYTES]
    auto_max = _hf_estimate_max_batch_size_encoder(
        model=model,
        max_seq_len_tokens=int(max_seq_len),
        cache_act_bytes_ref=act_ref,
        cache_model_bytes_ref=model_ref,
    )
    _HF_EMBED_ACT_BYTES_PER_TOKEN_PER_BATCH = act_ref[0]
    _HF_EMBED_MODEL_BYTES = model_ref[0]

    cap = int(requested_batch_max) if int(requested_batch_max) > 0 else int(auto_max)
    cap = max(1, min(cap, int(auto_max), 128, len(texts)))

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"torch is required for minimal embedding: {exc}")

    device = _hf_model_primary_input_device(model)

    batch_texts = [str(t or "") for t in texts[:cap]]
    with torch.no_grad():
        encoded = tok(batch_texts, padding=True, truncation=True, return_tensors="pt")
        try:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        except Exception:
            pass
        outputs = model(**encoded)
        last = getattr(outputs, "last_hidden_state", None)
        if last is None and isinstance(outputs, (tuple, list)) and outputs:
            last = outputs[0]
        if last is None:
            raise RuntimeError("embedding model returned no last_hidden_state")
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = last.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).type_as(last)
            summed = (last * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom
        try:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        except Exception:
            pass
        vecs = pooled.detach().cpu().tolist()
    return ([list(map(float, v)) for v in (vecs if isinstance(vecs, list) else [])], int(cap))


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "generated_text", "output"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        if "infer" in value:
            return _extract_text(value.get("infer"))
    return str(value)


def _run_text_generation(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}
    prompt = payload.get("prompt") if isinstance(payload, dict) else payload

    # Prefer dispatching through the main ipfs_accelerate_py instance if provided.
    # This allows the worker to reuse the same endpoint handlers / queues / model
    # management configuration as the MCP service.
    if accelerate_instance is not None and isinstance(payload, dict):
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")

                data = {
                    "prompt": str(prompt or ""),
                    "max_new_tokens": int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128),
                    "temperature": float(payload.get("temperature") or 0.2),
                }

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                # Some implementations return an Exception object instead of
                # raising it. Treat that as an error so we fall back to router-
                # based generation below.
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"text": _extract_text(accel_result)}
        except Exception:
            # Fall back to router-based generation.
            pass

    max_new_tokens = int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128)
    temperature = float(payload.get("temperature") or 0.2)

    # Minimal path for deterministic/local-first testing: avoid importing the
    # full ipfs_accelerate_py package (which can enable optional integrations).
    if _truthy(os.environ.get("IPFS_ACCEL_SKIP_CORE")) or _truthy(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM")
    ):
        text = _hf_textgen(
            str(prompt or ""),
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}

    # Default fallback: use the accelerate llm_router, which can still integrate
    # with InferenceBackendManager multiplexing when enabled via env vars.
    from ipfs_accelerate_py import llm_router

    try:
        text = llm_router.generate_text(
            str(prompt or ""),
            provider=None,
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}
    except Exception:
        # If optional providers (e.g. Codex/Copilot CLIs) are importable but not
        # actually functional on this machine, fall back to a minimal local HF
        # path so LAN workers can still run text-generation workloads.
        text = _hf_textgen(
            str(prompt or ""),
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}


def _run_text2text_generation(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}
    prompt = _extract_hf_input_text(payload)

    # Prefer dispatching through accelerate_instance if available.
    if accelerate_instance is not None and isinstance(payload, dict):
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")
                data = dict(payload)
                data.setdefault("prompt", str(prompt or ""))

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"text": _extract_text(accel_result)}
        except Exception:
            pass

    max_new_tokens = int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128) if isinstance(payload, dict) else 128
    temperature = float(payload.get("temperature") or 0.2) if isinstance(payload, dict) else 0.2

    if not _minimal_hf_enabled():
        raise RuntimeError("text2text-generation requires accelerate_instance or minimal HF enabled")

    texts, _used = _hf_text2text_batch_auto(
        [str(prompt or "")],
        model_name=(model_name or None),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        requested_batch_max=1,
    )
    return {"text": str(texts[0] if texts else "")}


def _run_embedding(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}

    if accelerate_instance is not None and isinstance(payload, dict):
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")
                data = dict(payload)
                data.setdefault("text", _extract_hf_input_text(payload))

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"result": accel_result}
        except Exception:
            pass

    if not _minimal_hf_enabled():
        raise RuntimeError("embedding requires accelerate_instance or minimal HF enabled")

    text = _extract_hf_input_text(payload)
    vecs, _used = _hf_embed_batch_auto([str(text or "")], model_name=(model_name or None), requested_batch_max=1)
    emb = vecs[0] if vecs else []
    return {"embedding": emb, "dim": int(len(emb))}


def _run_text_classification(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}
    text = _extract_hf_input_text(payload)

    if accelerate_instance is not None and isinstance(payload, dict):
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")
                data = dict(payload)
                data.setdefault("text", str(text or ""))

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"result": accel_result}
        except Exception:
            pass

    if not _minimal_hf_enabled():
        raise RuntimeError("text-classification requires accelerate_instance or minimal HF enabled")

    out, _used = _hf_textcls_batch_auto([str(text or "")], model_name=(model_name or None), requested_batch_max=1)
    return {"result": out[0] if isinstance(out, list) and out else out}


def _run_hf_pipeline(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("hf.pipeline payload must be a dict")
    pipeline_task = str(payload.get("pipeline_task") or payload.get("task") or "").strip()
    if not pipeline_task:
        raise ValueError("hf.pipeline requires payload.pipeline_task")

    if accelerate_instance is not None:
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")
                data = dict(payload)
                data.setdefault("pipeline_task", pipeline_task)

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"result": accel_result}
        except Exception:
            pass

    if not _minimal_hf_enabled():
        raise RuntimeError("hf.pipeline requires accelerate_instance or minimal HF enabled")

    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required for hf.pipeline: {exc}")

    kwargs = dict(payload)
    kwargs.pop("pipeline_task", None)
    kwargs.pop("task", None)
    inputs = kwargs.pop("inputs", None)
    # Allow explicit device override via payload.device; otherwise pick GPU when available.
    device_arg = kwargs.pop("device", None)
    if device_arg is None:
        device_arg = _hf_pipeline_device()
    p = pipeline(pipeline_task, model=(model_name or None), device=device_arg)
    out = p(inputs, **kwargs)  # type: ignore[misc]
    return {"result": out}


def _run_tool_call(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("tool.call payload must be a dict")

    tool_name = str(payload.get("tool") or payload.get("tool_name") or payload.get("name") or "").strip()
    if not tool_name:
        raise ValueError("tool.call missing tool name")
    args = payload.get("args") or payload.get("arguments") or payload.get("params") or {}
    if not isinstance(args, dict):
        args = {"value": args}

    if accelerate_instance is None:
        raise RuntimeError("tool.call requires accelerate_instance")
    fn = getattr(accelerate_instance, "call_tool", None)
    if not callable(fn):
        raise RuntimeError("accelerate_instance does not implement call_tool")

    try:
        import anyio
        import inspect

        async def _do() -> Any:
            result = fn(tool_name, args)
            if inspect.isawaitable(result):
                return await result
            return result

        out = anyio.run(_do, backend="trio")
    except Exception as exc:
        raise RuntimeError(f"tool.call failed: {exc}")
    return {"tool": tool_name, "result": out}


def _run_shell(task: Dict[str, Any]) -> Dict[str, Any]:
    allow = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow:
        raise RuntimeError("shell task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL=1)")

    if not _docker_tasks_enabled():
        raise RuntimeError("shell requires docker (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)")

    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("shell payload must be a dict")

    argv = payload.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("shell payload.argv must be a non-empty list[str]")

    timeout_s = payload.get("timeout_s")
    try:
        timeout_v = float(timeout_s) if timeout_s is not None else None
    except Exception:
        timeout_v = None

    image = (
        str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_SHELL_IMAGE") or "ubuntu:22.04").strip()
        or "ubuntu:22.04"
    )
    cmd_str = shlex.join([str(x) for x in argv])
    command = ["/bin/sh", "-lc", cmd_str]

    from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

    res = execute_docker_hub_container(
        image=image,
        command=command,
        timeout=int(timeout_v) if timeout_v is not None else 300,
        network_mode="none",
        no_new_privileges=True,
        stream_output=bool(payload.get("stream_output")),
    )

    return {
        "returncode": int(getattr(res, "exit_code", -1)),
        "stdout": str(getattr(res, "stdout", "") or ""),
        "stderr": str(getattr(res, "stderr", "") or ""),
        "success": bool(getattr(res, "success", False)),
        "exit_code": int(getattr(res, "exit_code", -1)),
    }


def _docker_tasks_enabled() -> bool:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    # Auto-enable if daemon is accessible.
    return _docker_daemon_available()


def _docker_daemon_available() -> bool:
    try:
        from ipfs_accelerate_py.docker_executor import docker_daemon_available

        return bool(docker_daemon_available())
    except Exception:
        return False


def _maybe_split_argv(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x for x in value if x]
    if isinstance(value, str):
        parts = [p for p in value.split() if p]
        return parts if parts else None
    return None


def _build_docker_run_cmd(*, payload: Dict[str, Any]) -> Tuple[list[str], float | None]:
    image = str(payload.get("image") or "").strip()
    if not image:
        raise ValueError("docker task missing payload.image")

    cmd: list[str] = ["docker", "run", "--rm"]

    # GPU support (optional): docker run --gpus <value>
    # Common values: "all", "device=0".
    gpus = payload.get("gpus")
    if gpus is None:
        gpus = payload.get("gpu")
    if gpus is True:
        gpus = "all"
    if isinstance(gpus, str) and gpus.strip():
        cmd.extend(["--gpus", str(gpus).strip()])

    memory_limit = payload.get("memory_limit")
    if memory_limit is not None:
        cmd.extend(
            [
                "--memory",
                str(memory_limit),
            ]
        )
    cpu_limit = payload.get("cpu_limit")
    if cpu_limit is not None:
        try:
            cmd.extend(["--cpus", str(float(cpu_limit))])
        except Exception:
            pass

    if bool(payload.get("read_only")):
        cmd.append("--read-only")
    if payload.get("no_new_privileges") is None or bool(payload.get("no_new_privileges")):
        cmd.append("--security-opt=no-new-privileges")
    user = payload.get("user")
    if user is not None and str(user).strip():
        cmd.extend(["--user", str(user).strip()])

    network_mode = payload.get("network_mode")
    if network_mode is not None and str(network_mode).strip():
        cmd.extend(["--network", str(network_mode).strip()])
    else:
        cmd.extend(["--network", "none"])

    env = payload.get("environment") or payload.get("env") or {}
    if isinstance(env, dict):
        for k, v in env.items():
            cmd.extend(["-e", f"{str(k)}={str(v)}"])

    vols = payload.get("volumes") or {}
    if isinstance(vols, dict):
        for host_path, container_path in vols.items():
            cmd.extend(["-v", f"{str(host_path)}:{str(container_path)}"])

    working_dir = payload.get("working_dir")
    if working_dir is not None and str(working_dir).strip():
        cmd.extend(["-w", str(working_dir).strip()])

    entrypoint = _maybe_split_argv(payload.get("entrypoint"))
    command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
    if entrypoint:
        cmd.extend(["--entrypoint", entrypoint[0]])

    cmd.append(image)

    if entrypoint and len(entrypoint) > 1:
        cmd.extend(entrypoint[1:])
        if command:
            cmd.extend(command)
    elif command:
        cmd.extend(command)

    timeout_s = payload.get("timeout")
    timeout_v: float | None
    if timeout_s is None:
        timeout_v = None
    else:
        try:
            timeout_v = float(timeout_s)
        except Exception:
            timeout_v = None
    return cmd, timeout_v


def _stream_subprocess(
    *,
    argv: list[str],
    task_id: str,
    queue: TaskQueue,
    timeout_s: float | None,
    heartbeat_interval_s: float = 0.5,
    flush_interval_s: float = 0.25,
) -> Dict[str, Any]:
    start = time.time()
    deadline = (start + timeout_s) if timeout_s and timeout_s > 0 else None

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []

    last_flush = 0.0
    last_heartbeat = 0.0

    def _safe_update(**kwargs: Any) -> None:
        try:
            queue.update(**kwargs)
        except Exception:
            pass

    def _pump(stream, *, stream_name: str) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            if stream_name == "stdout":
                stdout_buf.append(line)
            else:
                stderr_buf.append(line)
        try:
            stream.close()
        except Exception:
            pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout,), kwargs={"stream_name": "stdout"}, daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr,), kwargs={"stream_name": "stderr"}, daemon=True)
    t_out.start()
    t_err.start()

    try:
        while True:
            now = time.time()

            if deadline is not None and now >= deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                _safe_update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "timeout", "ts": now}},
                    append_log=f"[worker] timeout after {timeout_s}s; killed process",
                    log_stream="stderr",
                )
                break

            rc = proc.poll()

            if now - last_heartbeat >= heartbeat_interval_s:
                _safe_update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "running", "heartbeat_ts": now}},
                )
                last_heartbeat = now

            if now - last_flush >= flush_interval_s:
                # Flush buffered stdout/stderr lines to the queue.
                while stdout_buf:
                    line = stdout_buf.pop(0)
                    _safe_update(
                        task_id=task_id,
                        status="running",
                        append_log=line.rstrip("\n"),
                        log_stream="stdout",
                    )
                while stderr_buf:
                    line = stderr_buf.pop(0)
                    _safe_update(
                        task_id=task_id,
                        status="running",
                        append_log=line.rstrip("\n"),
                        log_stream="stderr",
                    )
                last_flush = now

            if rc is not None:
                break

            time.sleep(0.05)

        # Final flush
        while stdout_buf:
            line = stdout_buf.pop(0)
            _safe_update(
                task_id=task_id,
                status="running",
                append_log=line.rstrip("\n"),
                log_stream="stdout",
            )
        while stderr_buf:
            line = stderr_buf.pop(0)
            _safe_update(
                task_id=task_id,
                status="running",
                append_log=line.rstrip("\n"),
                log_stream="stderr",
            )
    finally:
        try:
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
        except Exception:
            pass

    end = time.time()
    rc_final = proc.returncode if proc.returncode is not None else -1
    return {
        "success": bool(rc_final == 0),
        "exit_code": int(rc_final),
        "execution_time": float(end - start),
    }


def _run_docker_hub(task: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("internal: _run_docker_hub is replaced by a queue-aware handler")


def _run_docker_github(task: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("internal: _run_docker_github is replaced by a queue-aware handler")


def _supported_task_types_from_env(default: list[str]) -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
        or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES")
        or ""
    )
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if parts:
        return parts

    base = list(default)
    if _docker_tasks_enabled():
        base.extend(
            [
                "docker.execute",
                "docker.run",
                "docker.hub",
                "docker.execute_docker_container",
                "docker.github",
                "docker.github_repo",
                "docker.build_and_execute_github_repo",
            ]
        )
    # Shell tasks are docker-backed; only enable when explicitly allowed.
    enable_shell_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or ""
    enable_shell = str(enable_shell_raw).strip().lower() in {"1", "true", "yes", "on"}
    if enable_shell and _docker_tasks_enabled():
        base.append("shell")
    return base


def _task_types_overridden_via_env() -> bool:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
        or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES")
        or ""
    )
    return bool([p.strip() for p in str(raw).split(",") if p.strip()])


def _accelerate_supports_tool_call(accelerate_instance: object | None) -> bool:
    fn = getattr(accelerate_instance, "call_tool", None)
    return bool(callable(fn))


def _compute_supported_task_types(
    *,
    supported_task_types: Optional[list[str]],
    accelerate_instance: object | None,
) -> list[str]:
    # If explicitly provided, respect it.
    if isinstance(supported_task_types, list) and supported_task_types:
        out = [str(x).strip() for x in supported_task_types if str(x).strip()]
        return out

    # Default: include handler aliases we can run locally.
    # NOTE: text-generation requires either an accelerate instance (preferred)
    # or a working `transformers` import for the minimal fallback.
    base_defaults: list[str] = []
    can_textgen = False
    if accelerate_instance is not None and callable(getattr(accelerate_instance, "infer", None)):
        can_textgen = True
    else:
        ok, _detail = _transformers_import_ok()
        can_textgen = bool(ok)
    if can_textgen:
        base_defaults.extend(["text-generation", "text_generation", "generation"])
    if _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_HF")):
        base_defaults.extend(
            [
                "text2text-generation",
                "text2text_generation",
                "embedding",
                "embeddings",
                "text-embedding",
                "text_embedding",
                "text-classification",
                "text_classification",
                "hf.pipeline",
                "hf_pipeline",
            ]
        )

    # Mesh-targeted LLM execution (e.g., copilot_cli). Only advertise this task
    # type when explicitly enabled, since it may rely on external tooling.
    if _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI")):
        base_defaults.extend(["llm.generate", "llm_generate"])
    out = _supported_task_types_from_env(base_defaults)

    # Add tool.call only when we can actually execute it, and only when the
    # task types weren't explicitly overridden via env (where the user likely
    # wants an exact allowlist).
    if (not _task_types_overridden_via_env()) and _accelerate_supports_tool_call(accelerate_instance):
        out.extend(["tool.call", "tool"])

    # Deduplicate while keeping order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        tt = str(t or "").strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        deduped.append(tt)
    return deduped


def _worker_mesh_enabled() -> bool:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH") or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_MESH")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _worker_mesh_refresh_s() -> float:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_REFRESH_S")
    try:
        return max(0.5, float(raw)) if raw is not None else 5.0
    except Exception:
        return 5.0


def _worker_mesh_claim_interval_s() -> float:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S")
    try:
        return max(0.1, float(raw)) if raw is not None else 0.5
    except Exception:
        return 0.5


def _worker_mesh_max_peers() -> int:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MAX_PEERS")
    try:
        return max(1, min(100, int(raw))) if raw is not None else 10
    except Exception:
        return 10


def _expected_session_tag() -> str:
    return str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or "").strip()


def _task_required_session(task_payload: object) -> str:
    """Extract an optional session affinity tag from a task payload."""

    if not isinstance(task_payload, dict):
        return ""
    for k in ("session_id", "session", "p2p_session"):
        v = task_payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _session_allows_task(*, task_payload: object, local_session: str) -> bool:
    required = _task_required_session(task_payload)
    if not required:
        return True
    if not str(local_session or "").strip():
        # If a task requires a session but this worker has none configured,
        # treat it as not eligible.
        return False
    return required == str(local_session).strip()


def _copilot_session_controls_allowed(
    *,
    payload: dict,
    local_session: str,
    assigned_worker_id: str,
) -> None:
    """Enforce safety policy for Copilot session controls.

    Rationale:
    - `--resume` and `--continue` can cause the worker to access prior chat state
      associated with its local Copilot account.
    - In a pooled/mesh setup, we only want peers within the same P2P session
      boundary to be able to request session continuity.

    Policy:
    - If the worker has a configured P2P session (`local_session`), then any
      task requesting session continuity must include a matching session id.
    - Disallow `continue_session` without an explicit `resume_session_id`
      unless the worker opts in via env.
    """

    if not isinstance(payload, dict):
        return

    resume_session_id = payload.get("resume_session_id")
    continue_session = bool(payload.get("continue_session", False))

    wants_resume = isinstance(resume_session_id, str) and bool(resume_session_id.strip())
    wants_continue = bool(continue_session)

    if not (wants_resume or wants_continue):
        return

    # Session ownership: only the worker that started/owns the session may
    # attempt `--resume`/`--continue`. Enforce via sticky pin.
    sticky = payload.get("sticky_worker_id")
    sticky_text = str(sticky or "").strip() if isinstance(sticky, (str, int, float)) else ""
    assigned = str(assigned_worker_id or "").strip()
    if not sticky_text:
        raise RuntimeError("copilot session continuity requires sticky_worker_id")
    if assigned and sticky_text != assigned:
        raise RuntimeError("copilot session continuity requires sticky_worker_id to match assigned_worker")

    local = str(local_session or "").strip()
    if local:
        required = _task_required_session(payload)
        if not required:
            raise RuntimeError("copilot session continuity requires a session_id")
        if required != local:
            raise RuntimeError("copilot session continuity requires matching session_id")

    if wants_continue and not wants_resume:
        allow = (
            str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_CONTINUE_WITHOUT_RESUME") or "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if not allow:
            raise RuntimeError(
                "copilot_cli disallows continue_session without resume_session_id "
                "(set IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_CONTINUE_WITHOUT_RESUME=1 to override)"
            )


def _allowed_llm_providers() -> set[str]:
    """Return allowed providers for llm.generate.

    This prevents remote peers from selecting unexpected providers on a worker.

    Env:
      - IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS
      - IPFS_DATASETS_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS (compat)
    """

    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS")
    if raw is None:
        raw = os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS")
    text = str(raw or "").strip()
    if not text:
        return {"copilot_cli"}

    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        return {"copilot_cli"}

    if "*" in parts or "all" in parts:
        # Conservative built-in allowlist. (Exclude 'mock' and other test-only providers.)
        return {
            "copilot_cli",
            "copilot_sdk",
            "codex_cli",
            "gemini_cli",
            "gemini_py",
            "claude_code",
            "claude_py",
            "openrouter",
        }

    return set(parts)


def _run_llm_generate(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run an LLM provider via llm_router (intended for copilot_cli mesh)."""

    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("llm.generate payload must be a dict")

    # Decrypt encrypted prompt fields (if present) before extracting values.
    if "prompt" not in payload and isinstance(payload.get("prompt_enc"), dict):
        try:
            from ipfs_accelerate_py.p2p_tasks.protocol import decrypt_text

            pt = decrypt_text(payload.get("prompt_enc"))
            if isinstance(pt, str):
                payload["prompt"] = pt
        except Exception:
            pass

    chat_session_id = payload.get("chat_session_id")
    resume_session_id = payload.get("resume_session_id")

    prompt = payload.get("prompt")
    if prompt is None:
        prompt = payload.get("text")
    if prompt is None:
        prompt = payload.get("input")

    provider = str(payload.get("provider") or "copilot_cli").strip().lower() or "copilot_cli"
    allowed = _allowed_llm_providers()
    if provider not in allowed:
        raise RuntimeError(f"llm.generate provider not allowed: {provider}")

    # Session continuity is only supported for copilot_cli today.
    if provider == "copilot_cli":
        _copilot_session_controls_allowed(
            payload=payload,
            local_session=_expected_session_tag(),
            assigned_worker_id=str(task.get("assigned_worker") or "").strip(),
        )
    else:
        if (isinstance(payload.get("resume_session_id"), str) and str(payload.get("resume_session_id") or "").strip()) or bool(
            payload.get("continue_session", False)
        ):
            raise RuntimeError("resume_session_id/continue_session only supported for provider='copilot_cli'")



    if provider == "copilot_cli":
        allow = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not allow:
            raise RuntimeError("copilot_cli tasks disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI=1)")

    model_name = str(task.get("model_name") or payload.get("model") or payload.get("model_name") or "").strip() or None

    # Forward known safe flags.
    kwargs: Dict[str, Any] = {}
    allow_paths = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_LLM_PATH_ARGS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_PATH_ARGS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    forwarded_keys = ["timeout", "trace", "trace_jsonl_path", "trace_dir"]
    if provider == "copilot_cli":
        forwarded_keys.extend(
            [
                "copilot_config_dir",
                "copilot_log_dir",
                "resume_session_id",
                "continue_session",
            ]
        )

    for k in forwarded_keys:
        if k in payload:
            # Path-like args can alter account/config state or write files.
            # Require an explicit opt-in on the worker.
            if not allow_paths and k in {"trace_jsonl_path", "trace_dir", "copilot_config_dir", "copilot_log_dir"}:
                raise RuntimeError(
                    f"llm.generate disallows '{k}' unless IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_LLM_PATH_ARGS=1"
                )
            kwargs[k] = payload.get(k)

    from ipfs_accelerate_py import llm_router

    text = llm_router.generate_text(str(prompt or ""), model_name=model_name, provider=provider, **kwargs)
    session_id = _expected_session_tag()

    out: Dict[str, Any] = {
        "text": str(text),
        "provider": provider,
        "session_id": session_id,
        "executor_worker_id": str(task.get("assigned_worker") or "").strip(),
    }
    try:
        out["executor_peer_id"] = _read_local_announce_peer_id()
        out["executor_multiaddr"] = _read_local_announce_multiaddr()
    except Exception:
        pass
    if isinstance(chat_session_id, str) and chat_session_id.strip():
        out["chat_session_id"] = chat_session_id.strip()
    if isinstance(resume_session_id, str) and resume_session_id.strip():
        out["resume_session_id"] = resume_session_id.strip()
    return out


def _read_local_announce_peer_id() -> str:
    """Best-effort local peer id from announce JSON.

    Used only for mesh bookkeeping/logging; worker mesh scheduling uses a stable
    `peer_id` hint derived from worker_id.
    """

    path = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE") or "").strip()
    if not path or path.lower() in {"0", "false", "no", "off"}:
        return ""
    try:
        if not os.path.exists(path):
            return ""
        data = json.loads(open(path, "r", encoding="utf-8").read() or "{}")
        if isinstance(data, dict):
            return str(data.get("peer_id") or "").strip()
    except Exception:
        return ""
    return ""


def _read_local_announce_multiaddr() -> str:
    path = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE") or "").strip()
    if not path or path.lower() in {"0", "false", "no", "off"}:
        return ""
    try:
        if not os.path.exists(path):
            return ""
        data = json.loads(open(path, "r", encoding="utf-8").read() or "{}")
        if isinstance(data, dict):
            return str(data.get("multiaddr") or "").strip()
    except Exception:
        return ""
    return ""


def _mesh_safe_task_types(supported: list[str]) -> list[str]:
    """Return supported task types for mesh claiming.

    Historically, mesh mode used an allowlist. We now allow whatever task types
    this worker is configured to support.
    """

    out: list[str] = []
    for t in list(supported or []):
        tt = str(t or "").strip()
        if tt:
            out.append(tt)
    return out


def _start_mesh_discovery_thread(
    *,
    stop: threading.Event,
    worker_id: str,
    peers_out: dict[str, object],
    peers_lock: threading.RLock,
    max_peers: int,
    refresh_s: float,
) -> threading.Thread:
    """Background mDNS discovery thread.

    Writes a list[RemoteQueue] into peers_out["peers"].
    """

    expected_session = _expected_session_tag()

    def _loop() -> None:
        # Import lazily: libp2p is optional in some environments.
        try:
            from ipfs_accelerate_py.p2p_tasks.client import (
                RemoteQueue,
                discover_peers_via_mdns_sync,
                request_status_sync,
            )
        except Exception:
            return

        # If our own peer id is known, help avoid selecting ourselves in
        # environments where exclude_self can't resolve it.
        local_peer_id = _read_local_announce_peer_id()

        while not stop.is_set():
            try:
                discovered = discover_peers_via_mdns_sync(timeout_s=1.0, limit=int(max_peers), exclude_self=True)
            except Exception:
                discovered = []

            keep: list[RemoteQueue] = []
            for rq in list(discovered or []):
                try:
                    pid = str(getattr(rq, "peer_id", "") or "").strip()
                    ma = str(getattr(rq, "multiaddr", "") or "").strip()
                except Exception:
                    continue
                if not pid or not ma:
                    continue
                if local_peer_id and pid == local_peer_id:
                    continue

                if expected_session:
                    try:
                        resp = request_status_sync(remote=rq, timeout_s=3.0, detail=False)
                        if not (isinstance(resp, dict) and resp.get("ok")):
                            continue
                        if str(resp.get("session") or "").strip() != expected_session:
                            continue
                    except Exception:
                        continue

                keep.append(rq)

            with peers_lock:
                peers_out["peers"] = keep

            stop.wait(float(refresh_s))

    t = threading.Thread(target=_loop, name=f"task_worker_mesh_mdns[{worker_id}]", daemon=True)
    t.start()
    return t


def _worker_mesh_static_peers() -> list[object]:
    """Return RemoteQueue targets to use for mesh claims, if configured.

    Env:
      IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS

    Format:
      - Comma-separated list of multiaddrs including /p2p/<peer_id>
        Example: /ip4/192.168.0.54/tcp/9101/p2p/12D3KooW...

    When provided and valid, the worker will skip mDNS discovery and only
    claim tasks from these peers.
    """

    raw = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS") or "").strip()
    if not raw:
        return []

    # Lazy import: this is only used in mesh mode.
    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
    except Exception:
        return []

    out: list[object] = []
    for part in [p.strip() for p in raw.split(",")]:
        if not part:
            continue

        multiaddr = str(part).strip()
        peer_id = ""
        if "/p2p/" in multiaddr:
            try:
                peer_id = multiaddr.split("/p2p/", 1)[1].split("/", 1)[0].strip()
            except Exception:
                peer_id = ""
        if not peer_id:
            # Require peer id to avoid ambiguous dialing.
            continue

        try:
            out.append(RemoteQueue(peer_id=str(peer_id), multiaddr=str(multiaddr)))
        except Exception:
            continue

    return out


def run_worker(
    *,
    queue_path: str,
    worker_id: str,
    poll_interval_s: float = 0.5,
    once: bool = False,
    p2p_service: bool = False,
    p2p_listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
    supported_task_types: Optional[list[str]] = None,
    mesh: Optional[bool] = None,
    mesh_refresh_s: Optional[float] = None,
    mesh_claim_interval_s: Optional[float] = None,
    mesh_max_peers: Optional[int] = None,
    stop_event: threading.Event | None = None,
) -> int:
    if p2p_service:
        # Run the libp2p service in a background thread so the worker loop can
        # remain simple and blocking.
        def _run_service() -> None:
            try:
                import anyio

                service_module = importlib.import_module("ipfs_accelerate_py.p2p_tasks.service")
                a_serve_task_queue = getattr(service_module, "serve_task_queue")

                async def _main() -> None:
                    await a_serve_task_queue(
                        queue_path=queue_path,
                        listen_port=p2p_listen_port,
                        accelerate_instance=accelerate_instance,
                    )

                anyio.run(_main, backend="trio")
            except Exception as exc:
                import sys
                import traceback

                print(f"ipfs_accelerate_py worker: failed to start p2p task service: {exc}", file=sys.stderr)
                traceback.print_exc()

        t = threading.Thread(
            target=_run_service,
            name=f"ipfs_accelerate_p2p_task_service[{worker_id}]",
            daemon=True,
        )

        t.start()

    queue = TaskQueue(queue_path)

    local_session = _expected_session_tag()

    handlers: dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def _wrap(fn: Callable[..., Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def _runner(task_dict: Dict[str, Any]) -> Dict[str, Any]:
            return fn(task_dict, accelerate_instance=accelerate_instance)  # type: ignore[misc]

        return _runner

    handlers["text-generation"] = _wrap(_run_text_generation)
    handlers["text_generation"] = _wrap(_run_text_generation)
    handlers["generation"] = _wrap(_run_text_generation)

    handlers["text2text-generation"] = _wrap(_run_text2text_generation)
    handlers["text2text_generation"] = _wrap(_run_text2text_generation)

    handlers["embedding"] = _wrap(_run_embedding)
    handlers["embeddings"] = _wrap(_run_embedding)
    handlers["text-embedding"] = _wrap(_run_embedding)
    handlers["text_embedding"] = _wrap(_run_embedding)

    handlers["text-classification"] = _wrap(_run_text_classification)
    handlers["text_classification"] = _wrap(_run_text_classification)

    handlers["hf.pipeline"] = _wrap(_run_hf_pipeline)
    handlers["hf_pipeline"] = _wrap(_run_hf_pipeline)

    handlers["tool.call"] = _wrap(_run_tool_call)
    handlers["tool"] = _wrap(_run_tool_call)

    # Mesh-targeted LLM execution (e.g., copilot_cli). This handler intentionally
    # does not take accelerate_instance.
    handlers["llm.generate"] = _run_llm_generate
    handlers["llm_generate"] = _run_llm_generate

    def _shell_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("shell payload must be a dict")

        enable_shell_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or ""
        allow = str(enable_shell_raw).strip().lower() in {"1", "true", "yes", "on"}
        if not allow:
            raise RuntimeError("shell task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL=1)")

        if not _docker_tasks_enabled():
            raise RuntimeError("shell requires docker (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)")

        task_id = str(task_dict.get("task_id") or "")
        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("shell payload.argv must be a non-empty list[str]")

        timeout_s = payload.get("timeout_s")
        try:
            timeout_v = float(timeout_s) if timeout_s is not None else None
        except Exception:
            timeout_v = None

        image = (
            str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_SHELL_IMAGE") or "ubuntu:22.04").strip()
            or "ubuntu:22.04"
        )
        cmd_str = shlex.join([str(x) for x in argv])
        command = ["/bin/sh", "-lc", cmd_str]

        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "starting", "ts": time.time(), "image": image}},
            )
        except Exception:
            pass

        from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

        res = execute_docker_hub_container(
            image=image,
            command=command,
            timeout=int(timeout_v) if timeout_v is not None else 300,
            network_mode="none",
            no_new_privileges=True,
            stream_output=bool(payload.get("stream_output")),
        )

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass

        return {
            "returncode": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
        }

    handlers["shell"] = _shell_handler

    def _docker_hub_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not _docker_tasks_enabled():
            raise RuntimeError(
                "docker task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)"
            )

        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("docker payload must be a dict")
        task_id = str(task_dict.get("task_id") or "")
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "starting", "ts": time.time()}},
            )
        except Exception:
            pass

        image = str(payload.get("image") or "").strip()
        if not image:
            raise ValueError("docker task missing payload.image")

        # Optional true streaming mode (uses docker CLI directly). Keep disabled
        # by default to preserve testability (tests monkeypatch docker_executor).
        stream_mode = bool(payload.get("stream_output")) or str(
            os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_DOCKER_STREAM") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        if stream_mode:
            argv, timeout_v = _build_docker_run_cmd(payload=payload)
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=f"[worker] exec: {' '.join(argv)}",
                    log_stream="stderr",
                )
            except Exception:
                pass
            result = _stream_subprocess(argv=argv, task_id=task_id, queue=queue, timeout_s=timeout_v)
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "exited", "ts": time.time()}},
                )
            except Exception:
                pass
            return result

        from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

        command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
        entrypoint = _maybe_split_argv(payload.get("entrypoint"))

        environment = payload.get("environment") or payload.get("env") or {}
        if not isinstance(environment, dict):
            environment = {}
        environment = {str(k): str(v) for k, v in environment.items()}

        volumes = payload.get("volumes") or {}
        if not isinstance(volumes, dict):
            volumes = {}
        volumes = {str(k): str(v) for k, v in volumes.items()}

        kwargs: Dict[str, Any] = {}
        # GPU support (optional): passed through to docker_executor.
        if payload.get("gpus") is not None:
            kwargs["gpus"] = "all" if payload.get("gpus") is True else str(payload.get("gpus"))
        elif payload.get("gpu") is not None:
            kwargs["gpus"] = "all" if payload.get("gpu") is True else str(payload.get("gpu"))
        if payload.get("memory_limit") is not None:
            kwargs["memory_limit"] = str(payload.get("memory_limit"))
        if payload.get("cpu_limit") is not None:
            try:
                kwargs["cpu_limit"] = float(payload.get("cpu_limit"))
            except Exception:
                pass
        if payload.get("timeout") is not None:
            try:
                kwargs["timeout"] = int(payload.get("timeout"))
            except Exception:
                pass
        if payload.get("network_mode") is not None:
            kwargs["network_mode"] = str(payload.get("network_mode"))
        if payload.get("working_dir") is not None:
            kwargs["working_dir"] = str(payload.get("working_dir"))
        if payload.get("read_only") is not None:
            kwargs["read_only"] = bool(payload.get("read_only"))
        if payload.get("no_new_privileges") is not None:
            kwargs["no_new_privileges"] = bool(payload.get("no_new_privileges"))
        if payload.get("user") is not None:
            kwargs["user"] = str(payload.get("user"))

        stop = threading.Event()

        def _hb() -> None:
            while not stop.is_set():
                try:
                    queue.update(
                        task_id=task_id,
                        status="running",
                        result_patch={"progress": {"phase": "running", "heartbeat_ts": time.time(), "image": image}},
                    )
                except Exception:
                    pass
                stop.wait(0.5)

        t = threading.Thread(target=_hb, name=f"docker_hb[{task_id}]", daemon=True)
        t.start()
        try:
            res = execute_docker_hub_container(
                image=image,
                command=command,
                entrypoint=entrypoint,
                environment=environment,
                volumes=volumes,
                **kwargs,
            )
        finally:
            stop.set()
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass
        if getattr(res, "error_message", None):
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=str(getattr(res, "error_message")),
                    log_stream="stderr",
                )
            except Exception:
                pass
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "exited", "ts": time.time(), "image": image}},
            )
        except Exception:
            pass

        return {
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": float(getattr(res, "execution_time", 0.0) or 0.0),
            "error_message": getattr(res, "error_message", None),
        }

    def _docker_github_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not _docker_tasks_enabled():
            raise RuntimeError(
                "docker task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)"
            )

        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("docker.github payload must be a dict")
        task_id = str(task_dict.get("task_id") or "")

        repo_url = str(payload.get("repo_url") or payload.get("repo") or "").strip()
        if not repo_url:
            raise ValueError("docker.github task missing payload.repo_url")

        branch = str(payload.get("branch") or "main")
        dockerfile_path = str(payload.get("dockerfile_path") or payload.get("dockerfile") or "Dockerfile")
        context_path = str(payload.get("context_path") or payload.get("context") or ".")

        command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
        entrypoint = _maybe_split_argv(payload.get("entrypoint"))

        environment = payload.get("environment") or payload.get("env") or {}
        if not isinstance(environment, dict):
            environment = {}
        environment = {str(k): str(v) for k, v in environment.items()}

        build_args = payload.get("build_args") or {}
        if not isinstance(build_args, dict):
            build_args = {}
        build_args = {str(k): str(v) for k, v in build_args.items()}

        kwargs: Dict[str, Any] = {}
        if payload.get("memory_limit") is not None:
            kwargs["memory_limit"] = str(payload.get("memory_limit"))
        if payload.get("cpu_limit") is not None:
            try:
                kwargs["cpu_limit"] = float(payload.get("cpu_limit"))
            except Exception:
                pass
        if payload.get("timeout") is not None:
            try:
                kwargs["timeout"] = int(payload.get("timeout"))
            except Exception:
                pass
        if payload.get("network_mode") is not None:
            kwargs["network_mode"] = str(payload.get("network_mode"))
        if payload.get("working_dir") is not None:
            kwargs["working_dir"] = str(payload.get("working_dir"))
        if payload.get("volumes") is not None and isinstance(payload.get("volumes"), dict):
            kwargs["volumes"] = {str(k): str(v) for k, v in payload.get("volumes").items()}
        if payload.get("read_only") is not None:
            kwargs["read_only"] = bool(payload.get("read_only"))
        if payload.get("no_new_privileges") is not None:
            kwargs["no_new_privileges"] = bool(payload.get("no_new_privileges"))
        if payload.get("user") is not None:
            kwargs["user"] = str(payload.get("user"))

        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "building", "ts": time.time(), "repo_url": repo_url}},
            )
        except Exception:
            pass

        from ipfs_accelerate_py.docker_executor import build_and_execute_from_github

        stop = threading.Event()

        def _hb() -> None:
            while not stop.is_set():
                try:
                    queue.update(
                        task_id=task_id,
                        status="running",
                        result_patch={
                            "progress": {
                                "phase": "building_and_running",
                                "heartbeat_ts": time.time(),
                                "repo_url": repo_url,
                                "branch": branch,
                            }
                        },
                    )
                except Exception:
                    pass
                stop.wait(0.5)

        t = threading.Thread(target=_hb, name=f"docker_github_hb[{task_id}]", daemon=True)
        t.start()
        try:
            res = build_and_execute_from_github(
                repo_url=repo_url,
                branch=branch,
                dockerfile_path=dockerfile_path,
                context_path=context_path,
                command=command,
                entrypoint=entrypoint,
                environment=environment,
                build_args=build_args,
                **kwargs,
            )
        finally:
            stop.set()
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass
        if getattr(res, "error_message", None):
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=str(getattr(res, "error_message")),
                    log_stream="stderr",
                )
            except Exception:
                pass
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "exited", "ts": time.time(), "repo_url": repo_url}},
            )
        except Exception:
            pass

        return {
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": float(getattr(res, "execution_time", 0.0) or 0.0),
            "error_message": getattr(res, "error_message", None),
        }

    handlers["docker.execute"] = _docker_hub_handler
    handlers["docker.run"] = _docker_hub_handler
    handlers["docker.execute_docker_container"] = _docker_hub_handler
    handlers["docker.hub"] = _docker_hub_handler
    handlers["docker.github"] = _docker_github_handler
    handlers["docker.github_repo"] = _docker_github_handler
    handlers["docker.build_and_execute_github_repo"] = _docker_github_handler

    supported = _compute_supported_task_types(
        supported_task_types=supported_task_types,
        accelerate_instance=accelerate_instance,
    )

    mesh_enabled = bool(_worker_mesh_enabled()) if mesh is None else bool(mesh)
    mesh_refresh = float(_worker_mesh_refresh_s()) if mesh_refresh_s is None else float(mesh_refresh_s)
    mesh_claim_interval = (
        float(_worker_mesh_claim_interval_s())
        if mesh_claim_interval_s is None
        else float(mesh_claim_interval_s)
    )
    mesh_peers_limit = int(_worker_mesh_max_peers()) if mesh_max_peers is None else int(mesh_max_peers)

    # Mesh discovery state.
    mesh_stop = threading.Event()
    mesh_peers_lock = threading.RLock()
    mesh_peers_state: dict[str, object] = {"peers": []}
    mesh_rr = 0
    last_mesh_claim = 0.0

    mesh_thread: threading.Thread | None = None
    if mesh_enabled:
        static_peers = _worker_mesh_static_peers()
        if static_peers:
            with mesh_peers_lock:
                mesh_peers_state["peers"] = list(static_peers)
        else:
            mesh_thread = _start_mesh_discovery_thread(
                stop=mesh_stop,
                worker_id=str(worker_id),
                peers_out=mesh_peers_state,
                peers_lock=mesh_peers_lock,
                max_peers=int(mesh_peers_limit),
                refresh_s=float(mesh_refresh),
            )

    def _snapshot_mesh_peers() -> list[object]:
        with mesh_peers_lock:
            peers = mesh_peers_state.get("peers")
            return list(peers) if isinstance(peers, list) else []

    def _maybe_claim_from_mesh() -> Optional[tuple[object, Dict[str, Any]]]:
        nonlocal mesh_rr, last_mesh_claim
        if not mesh_enabled:
            return None
        now = time.time()
        if (now - last_mesh_claim) < float(mesh_claim_interval):
            return None

        peers = _snapshot_mesh_peers()
        if not peers:
            last_mesh_claim = now
            return None

        # Avoid tight loops when there are peers but all are empty/unreachable.
        last_mesh_claim = now

        # Lazy import to avoid requiring libp2p in non-mesh deployments.
        try:
            from ipfs_accelerate_py.p2p_tasks.client import claim_many_sync, claim_next_sync
        except Exception:
            return None

        # Restrict mesh to safe task types.
        mesh_supported = _mesh_safe_task_types(list(supported or []))
        if not mesh_supported:
            return None

        # Try a small number of peers per poll (fanout) so one worker can help
        # drain multiple remote queues promptly when several peers are backed up.
        try:
            fanout_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEER_FANOUT")
            fanout_n = int(float(fanout_raw)) if fanout_raw is not None else 1
        except Exception:
            fanout_n = 1
        fanout_n = max(1, min(int(fanout_n), 16))

        idx0 = int(mesh_rr) % max(1, len(peers))
        mesh_rr = (idx0 + fanout_n) % max(1, len(peers))

        # Batch-claim when enabled (reduces RPC overhead; enables micro-batching).
        try:
            batch_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_BATCH")
            batch_n = int(float(batch_raw)) if batch_raw is not None else 1
        except Exception:
            batch_n = 1
        batch_n = max(1, min(int(batch_n), 64))

        for i in range(min(int(fanout_n), len(peers))):
            remote = peers[(idx0 + i) % max(1, len(peers))]

            # Use any prefetched tasks from prior batch-claims first.
            try:
                buf = getattr(remote, "_mesh_prefetched", None)
            except Exception:
                buf = None
            if isinstance(buf, list) and buf:
                t0 = buf[0] if isinstance(buf[0], dict) else None
                try:
                    setattr(remote, "_mesh_prefetched", list(buf[1:]))
                except Exception:
                    pass
                if isinstance(t0, dict):
                    return (remote, t0)

            if batch_n > 1:
                try:
                    tasks = claim_many_sync(
                        remote=remote,
                        worker_id=str(worker_id),
                        supported_task_types=list(mesh_supported),
                        max_tasks=int(batch_n),
                        same_task_type=True,
                        session_id=local_session or None,
                        peer_id=str(worker_id),
                        clock=None,
                    )
                    if tasks and isinstance(tasks[0], dict):
                        # Return the first; stash the rest on the remote object.
                        try:
                            setattr(remote, "_mesh_prefetched", list(tasks[1:]))
                        except Exception:
                            pass
                        return (remote, tasks[0])
                except Exception:
                    # Fall back to single-claim.
                    pass

            try:
                task = claim_next_sync(
                    remote=remote,  # RemoteQueue
                    worker_id=str(worker_id),
                    supported_task_types=list(mesh_supported),
                    session_id=local_session or None,
                    peer_id=str(worker_id),
                    clock=None,
                )
                if task is None:
                    continue
                if isinstance(task, dict):
                    return (remote, task)
            except Exception:
                continue

        return None

    def _pop_prefetched(remote: object) -> list[Dict[str, Any]]:
        try:
            buf = getattr(remote, "_mesh_prefetched", None)
        except Exception:
            buf = None
        if not isinstance(buf, list) or not buf:
            return []
        out = [x for x in buf if isinstance(x, dict)]
        try:
            setattr(remote, "_mesh_prefetched", [])
        except Exception:
            pass
        return out

    def _complete_mesh_task(
        *,
        remote: object,
        task_id: str,
        ok: bool,
        result: Dict[str, Any] | None,
        error: str | None,
    ) -> None:
        try:
            from ipfs_accelerate_py.p2p_tasks.client import complete_task_sync

            complete_task_sync(
                remote=remote,  # RemoteQueue
                task_id=str(task_id),
                status="completed" if ok else "failed",
                result=(result if ok else None),
                error=(None if ok else str(error or "unknown error")),
            )
        except Exception:
            return

    def _stamp_result_meta(*, result: Dict[str, Any] | None, assigned_worker: str | None = None) -> Dict[str, Any] | None:
        if not isinstance(result, dict):
            return result
        out = dict(result)
        try:
            wid = str(assigned_worker or "").strip() or str(worker_id)
        except Exception:
            wid = str(worker_id)
        try:
            sid = str(local_session or "").strip()
        except Exception:
            sid = ""

        if wid and not str(out.get("executor_worker_id") or "").strip():
            out["executor_worker_id"] = wid
        if sid and not str(out.get("session_id") or "").strip():
            out["session_id"] = sid
        return out

    def _complete_local_task(*, task_id: str, ok: bool, result: Dict[str, Any] | None, error: str | None) -> None:
        try:
            if ok:
                queue.complete(task_id=str(task_id), status="completed", result=(result or {}))
            else:
                queue.complete(task_id=str(task_id), status="failed", error=str(error or "unknown error"))
        except Exception:
            return

    def _microbatch_and_complete(
        *,
        batch_tasks: list[Dict[str, Any]],
        mesh: bool,
        remote: object | None,
    ) -> list[Dict[str, Any]]:
        """Best-effort micro-batching for homogeneous HF tasks.

        Completes any tasks that are successfully micro-batched and returns the
        remaining tasks that should be executed sequentially.
        """

        if not batch_tasks:
            return []

        try:
            ttype0 = str(batch_tasks[0].get("task_type") or "").strip().lower()
        except Exception:
            ttype0 = ""

        def _complete_one(*, tid: str, ok: bool, res: Dict[str, Any] | None, err: str | None) -> None:
            if mesh:
                if remote is None:
                    return
                if ok and isinstance(res, dict):
                    res = dict(res)
                    res = _stamp_result_meta(result=res)
                    progress = res.get("progress")
                    if not isinstance(progress, dict):
                        progress = {}
                    if not progress.get("worker_id"):
                        progress = dict(progress)
                        progress["worker_id"] = str(worker_id)
                        progress["task_type"] = str(ttype0)
                        progress["mesh"] = True
                    res["progress"] = progress
                _complete_mesh_task(remote=remote, task_id=tid, ok=ok, result=res, error=err)
            else:
                if ok and isinstance(res, dict):
                    res = dict(res)
                    res = _stamp_result_meta(result=res)
                    progress = res.get("progress")
                    if not isinstance(progress, dict):
                        progress = {}
                    if not progress.get("worker_id"):
                        progress = dict(progress)
                        progress["worker_id"] = str(worker_id)
                        progress["task_type"] = str(ttype0)
                    res["progress"] = progress
                _complete_local_task(task_id=tid, ok=ok, result=res, error=err)

        # text-generation batching (minimal HF only)
        tb_cap = _parse_batch_cap(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TEXTGEN_BATCH_MAX"), default=1)
        tb_limit = 64 if tb_cap <= 0 else max(1, min(int(tb_cap), 64))
        can_textgen_batch = (
            ttype0 in {"text-generation", "text_generation", "generation"}
            and tb_limit > 1
            and accelerate_instance is None
            and (
                _truthy(os.environ.get("IPFS_ACCEL_SKIP_CORE"))
                or _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM"))
            )
        )

        if can_textgen_batch:
            text_tasks: list[Dict[str, Any]] = []
            for t in batch_tasks[:tb_limit]:
                if str(t.get("task_type") or "").strip().lower() not in {"text-generation", "text_generation", "generation"}:
                    break
                text_tasks.append(t)

            def _params(t: Dict[str, Any]) -> tuple[str, int, float]:
                payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                model = str(t.get("model_name") or "")
                mx = int((payload or {}).get("max_new_tokens") or (payload or {}).get("max_tokens") or 128)
                temp = float((payload or {}).get("temperature") or 0.2)
                return (model, mx, temp)

            if text_tasks:
                base = _params(text_tasks[0])
                if all(_params(t) == base for t in text_tasks):
                    prompts = []
                    for t in text_tasks:
                        payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                        prompts.append(str(_extract_hf_input_text(payload) or ""))

                    texts, used = _hf_textgen_batch_auto(
                        prompts,
                        model_name=(base[0] or None),
                        max_new_tokens=int(base[1]),
                        temperature=float(base[2]),
                        requested_batch_max=int(tb_cap),
                    )

                    for t, text in zip(text_tasks[:used], texts):
                        tid = str(t.get("task_id") or "").strip()
                        if not tid:
                            continue
                        _complete_one(tid=tid, ok=True, res={"text": str(text)}, err=None)

                    return list(batch_tasks[used:])

        # text2text-generation batching (minimal HF only)
        t2t_cap = _parse_batch_cap(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TEXT2TEXT_BATCH_MAX"), default=1)
        t2t_limit = 64 if t2t_cap <= 0 else max(1, min(int(t2t_cap), 64))
        can_t2t_batch = (
            ttype0 in {"text2text-generation", "text2text_generation"}
            and t2t_limit > 1
            and accelerate_instance is None
            and _minimal_hf_enabled()
        )
        if can_t2t_batch:
            t2t_tasks: list[Dict[str, Any]] = []
            for t in batch_tasks[:t2t_limit]:
                if str(t.get("task_type") or "").strip().lower() not in {"text2text-generation", "text2text_generation"}:
                    break
                t2t_tasks.append(t)

            def _t2t_params(t: Dict[str, Any]) -> tuple[str, int, float]:
                payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                model = str(t.get("model_name") or "")
                mx = int((payload or {}).get("max_new_tokens") or (payload or {}).get("max_tokens") or 128)
                temp = float((payload or {}).get("temperature") or 0.2)
                return (model, mx, temp)

            if t2t_tasks:
                base = _t2t_params(t2t_tasks[0])
                if all(_t2t_params(t) == base for t in t2t_tasks):
                    prompts = []
                    for t in t2t_tasks:
                        payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                        prompts.append(str(_extract_hf_input_text(payload) or ""))

                    texts, used = _hf_text2text_batch_auto(
                        prompts,
                        model_name=(base[0] or None),
                        max_new_tokens=int(base[1]),
                        temperature=float(base[2]),
                        requested_batch_max=int(t2t_cap),
                    )

                    for t, text in zip(t2t_tasks[:used], texts):
                        tid = str(t.get("task_id") or "").strip()
                        if not tid:
                            continue
                        _complete_one(tid=tid, ok=True, res={"text": str(text)}, err=None)

                    return list(batch_tasks[used:])

        # text-classification batching (minimal HF only)
        cls_cap = _parse_batch_cap(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TEXTCLS_BATCH_MAX"), default=1)
        cls_limit = 64 if cls_cap <= 0 else max(1, min(int(cls_cap), 64))
        can_cls_batch = (
            ttype0 in {"text-classification", "text_classification"}
            and cls_limit > 1
            and accelerate_instance is None
            and _minimal_hf_enabled()
        )
        if can_cls_batch:
            cls_tasks: list[Dict[str, Any]] = []
            for t in batch_tasks[:cls_limit]:
                if str(t.get("task_type") or "").strip().lower() not in {"text-classification", "text_classification"}:
                    break
                cls_tasks.append(t)

            def _cls_params(t: Dict[str, Any]) -> tuple[str]:
                return (str(t.get("model_name") or ""),)

            if cls_tasks:
                base = _cls_params(cls_tasks[0])
                if all(_cls_params(t) == base for t in cls_tasks):
                    texts_in = []
                    for t in cls_tasks:
                        payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                        texts_in.append(str(_extract_hf_input_text(payload) or ""))

                    outs, used = _hf_textcls_batch_auto(
                        texts_in,
                        model_name=(base[0] or None),
                        requested_batch_max=int(cls_cap),
                    )

                    for t, out in zip(cls_tasks[:used], outs):
                        tid = str(t.get("task_id") or "").strip()
                        if not tid:
                            continue
                        _complete_one(tid=tid, ok=True, res={"result": out}, err=None)

                    return list(batch_tasks[used:])

        # embedding batching (minimal HF only)
        emb_cap = _parse_batch_cap(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_EMBED_BATCH_MAX"), default=1)
        emb_limit = 64 if emb_cap <= 0 else max(1, min(int(emb_cap), 64))
        can_emb_batch = (
            ttype0 in {"embedding", "embeddings", "text-embedding", "text_embedding"}
            and emb_limit > 1
            and accelerate_instance is None
            and _minimal_hf_enabled()
        )
        if can_emb_batch:
            emb_types = {"embedding", "embeddings", "text-embedding", "text_embedding"}
            emb_tasks: list[Dict[str, Any]] = []
            for t in batch_tasks[:emb_limit]:
                if str(t.get("task_type") or "").strip().lower() not in emb_types:
                    break
                emb_tasks.append(t)

            def _emb_params(t: Dict[str, Any]) -> tuple[str]:
                return (str(t.get("model_name") or ""),)

            if emb_tasks:
                base = _emb_params(emb_tasks[0])
                if all(_emb_params(t) == base for t in emb_tasks):
                    texts_in = []
                    for t in emb_tasks:
                        payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                        texts_in.append(str(_extract_hf_input_text(payload) or ""))

                    vecs, used = _hf_embed_batch_auto(
                        texts_in,
                        model_name=(base[0] or None),
                        requested_batch_max=int(emb_cap),
                    )

                    for t, vec in zip(emb_tasks[:used], vecs):
                        tid = str(t.get("task_id") or "").strip()
                        if not tid:
                            continue
                        emb = list(vec) if isinstance(vec, list) else []
                        _complete_one(tid=tid, ok=True, res={"embedding": emb, "dim": int(len(emb))}, err=None)

                    return list(batch_tasks[used:])

        return list(batch_tasks)

    # Local batch-claim (homogeneous) to enable micro-batching.
    local_claim_cap = _parse_batch_cap(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_LOCAL_CLAIM_BATCH"), default=1)
    try:
        local_claim_n = int(local_claim_cap) if int(local_claim_cap) > 0 else 1
    except Exception:
        local_claim_n = 1
    local_claim_n = max(1, min(int(local_claim_n), 128))

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return 0

            claimed_local: list[QueuedTask] = []
            if local_claim_n > 1:
                try:
                    claimed_local = queue.claim_next_many(
                        worker_id=worker_id,
                        supported_task_types=supported,
                        max_tasks=int(local_claim_n),
                        same_task_type=True,
                        session_id=local_session or None,
                    )
                except Exception:
                    claimed_local = []
            else:
                try:
                    one = queue.claim_next(
                        worker_id=worker_id,
                        supported_task_types=supported,
                        session_id=local_session or None,
                    )
                except Exception:
                    # Treat claim errors as transient; avoid crashing the worker loop.
                    time.sleep(max(0.05, float(poll_interval_s)))
                    one = None
                claimed_local = [one] if one is not None else []

            if not claimed_local:
                # No local work; optionally help the mesh by claiming from peers.
                mesh_claim = _maybe_claim_from_mesh()
                if mesh_claim is not None:
                    remote, remote_task = mesh_claim
                    task_id = str(remote_task.get("task_id") or "").strip()
                    if not task_id:
                        # Nothing we can do.
                        continue

                    if not _session_allows_task(task_payload=remote_task.get("payload"), local_session=local_session):
                        # Best-effort: release and skip.
                        try:
                            from ipfs_accelerate_py.p2p_tasks.client import release_task_sync

                            release_task_sync(
                                remote=remote,  # RemoteQueue
                                task_id=str(task_id),
                                worker_id=str(worker_id),
                                reason="session_mismatch",
                            )
                        except Exception:
                            pass
                        continue

                    # Opportunistically include any prefetched tasks from a
                    # prior batch claim.
                    batch_tasks = [remote_task] + _pop_prefetched(remote)

                    # Micro-batch a homogeneous prefix when safe and enabled.
                    batch_tasks = _microbatch_and_complete(batch_tasks=batch_tasks, mesh=True, remote=remote)

                    for t in batch_tasks:
                        tid = str(t.get("task_id") or "").strip()
                        if not tid:
                            continue
                        result: Dict[str, Any] | None = None
                        error: str | None = None
                        ok = False
                        try:
                            ttype = str(t.get("task_type") or "").strip().lower()
                            handler = handlers.get(ttype)
                            if handler is None:
                                raise RuntimeError(f"Unsupported task_type: {t.get('task_type')}")
                            result = handler(
                                {
                                    "task_id": tid,
                                    "task_type": t.get("task_type"),
                                    "model_name": t.get("model_name"),
                                    "payload": t.get("payload"),
                                    "assigned_worker": str(t.get("assigned_worker") or worker_id).strip(),
                                }
                            )
                            if isinstance(result, dict):
                                # Ensure mesh executions are attributable.
                                result = dict(result)
                                result = _stamp_result_meta(result=result, assigned_worker=str(t.get("assigned_worker") or worker_id).strip())
                                progress = result.get("progress")
                                if not isinstance(progress, dict):
                                    progress = {}
                                if not progress.get("worker_id"):
                                    progress = dict(progress)
                                    progress["worker_id"] = str(worker_id)
                                    progress["task_type"] = str(t.get("task_type") or "")
                                    progress["mesh"] = True
                                result["progress"] = progress
                            ok = True
                        except Exception as exc:
                            ok = False
                            error = str(exc)
                        _complete_mesh_task(remote=remote, task_id=tid, ok=ok, result=result, error=error)
                    if once:
                        return 0
                    continue

                if once:
                    return 0
                sleep_s = max(0.05, float(poll_interval_s))
                if stop_event is None:
                    time.sleep(sleep_s)
                else:
                    stop_event.wait(sleep_s)
                continue

            # Batch path: complete a homogeneous set locally (no per-task heartbeat).
            if len(claimed_local) > 1:
                batch_tasks_local: list[Dict[str, Any]] = []
                for t in claimed_local:
                    if t is None:
                        continue
                    if not _session_allows_task(task_payload=t.payload, local_session=local_session):
                        try:
                            queue.release(task_id=str(t.task_id), worker_id=str(worker_id), reason="session_mismatch")
                        except Exception:
                            pass
                        continue
                    batch_tasks_local.append(
                        {
                            "task_id": t.task_id,
                            "task_type": t.task_type,
                            "model_name": t.model_name,
                            "payload": t.payload,
                            "assigned_worker": str(t.assigned_worker or worker_id).strip(),
                        }
                    )

                batch_tasks_local = _microbatch_and_complete(batch_tasks=batch_tasks_local, mesh=False, remote=None)

                for t in batch_tasks_local:
                    tid = str(t.get("task_id") or "").strip()
                    if not tid:
                        continue
                    result: Dict[str, Any] | None = None
                    error: str | None = None
                    ok = False
                    try:
                        ttype = str(t.get("task_type") or "").strip().lower()
                        handler = handlers.get(ttype)
                        if handler is None:
                            raise RuntimeError(f"Unsupported task_type: {t.get('task_type')}")
                        result = handler(
                            {
                                "task_id": tid,
                                "task_type": t.get("task_type"),
                                "model_name": t.get("model_name"),
                                "payload": t.get("payload"),
                                "assigned_worker": str(t.get("assigned_worker") or worker_id).strip(),
                            }
                        )
                        if isinstance(result, dict):
                            result = dict(result)
                            result = _stamp_result_meta(result=result, assigned_worker=str(t.get("assigned_worker") or worker_id).strip())
                            progress = result.get("progress")
                            if not isinstance(progress, dict):
                                progress = {}
                            if not progress.get("worker_id"):
                                progress = dict(progress)
                                progress["worker_id"] = str(worker_id)
                                progress["task_type"] = str(t.get("task_type") or "")
                            result["progress"] = progress
                        ok = True
                    except Exception as exc:
                        ok = False
                        error = str(exc)
                    _complete_local_task(task_id=tid, ok=ok, result=result, error=error)

                if once:
                    return 0
                continue

            task = claimed_local[0]

            if task is not None and not _session_allows_task(task_payload=task.payload, local_session=local_session):
                try:
                    queue.release(task_id=str(task.task_id), worker_id=str(worker_id), reason="session_mismatch")
                except Exception:
                    pass
                if once:
                    return 0
                continue

            # Generic heartbeat/progress so peers can observe long-running tasks via
            # RPC get/wait polling even for non-docker handlers.
            stop_hb = threading.Event()

            def _task_hb() -> None:
                while not stop_hb.is_set():
                    try:
                        queue.update(
                            task_id=task.task_id,
                            status="running",
                            result_patch={
                                "progress": {
                                    "heartbeat_ts": time.time(),
                                    "worker_id": str(worker_id),
                                    "task_type": str(task.task_type),
                                }
                            },
                        )
                    except Exception:
                        pass
                    stop_hb.wait(0.5)

            # Also mark initial start.
            try:
                queue.update(
                    task_id=task.task_id,
                    status="running",
                    result_patch={
                        "progress": {
                            "phase": "starting",
                            "ts": time.time(),
                            "worker_id": str(worker_id),
                            "task_type": str(task.task_type),
                        }
                    },
                )
            except Exception:
                pass

            hb_thread = threading.Thread(target=_task_hb, name=f"task_hb[{task.task_id}]", daemon=True)
            hb_thread.start()

            result: Dict[str, Any] | None = None
            error: str | None = None
            status: str = "failed"
            try:
                ttype = str(task.task_type or "").strip().lower()
                handler = handlers.get(ttype)
                if handler is None:
                    status = "failed"
                    error = f"Unsupported task_type: {task.task_type}"
                else:
                    result = handler(
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "model_name": task.model_name,
                            "payload": task.payload,
                            "assigned_worker": str(task.assigned_worker or worker_id).strip(),
                        }
                    )
                    result = _stamp_result_meta(result=result, assigned_worker=str(task.assigned_worker or worker_id).strip())
                    status = "completed"
            except Exception as exc:
                status = "failed"
                error = str(exc)
            finally:
                # Stop heartbeat before completing to avoid DuckDB write conflicts.
                stop_hb.set()
                try:
                    hb_thread.join(timeout=0.5)
                except Exception:
                    pass

            if status == "completed":
                queue.complete(task_id=task.task_id, status="completed", result=result or {})
            else:
                queue.complete(task_id=task.task_id, status="failed", error=error or "unknown error")

            if once:
                return 0
    finally:
        try:
            queue.close()
        except Exception:
            pass
        if mesh_enabled:
            mesh_stop.set()
            if mesh_thread is not None:
                try:
                    mesh_thread.join(timeout=0.5)
                except Exception:
                    pass


def run_autoscaled_workers(
    *,
    queue_path: str,
    base_worker_id: str,
    min_workers: int = 1,
    max_workers: int = 4,
    scale_poll_s: float = 2.0,
    scale_down_idle_s: float = 30.0,
    poll_interval_s: float = 0.25,
    once: bool = False,
    p2p_service: bool = False,
    p2p_listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
    supported_task_types: Optional[list[str]] = None,
    mesh: Optional[bool] = None,
    mesh_refresh_s: float | None = None,
    mesh_claim_interval_s: float | None = None,
    mesh_max_peers: int | None = None,
    mesh_children: Optional[bool] = False,
    autoscale_remote: bool = False,
    remote_refresh_s: float = 5.0,
    remote_max_peers: int = 10,
    use_processes: bool = False,
    stop_event: threading.Event | None = None,
) -> int:
    """Autoscale workers based on local and (optionally) remote backlog.

    This manager starts between `min_workers` and `max_workers` workers.
    Each worker runs `run_worker(...)` with a unique worker_id.

    By default workers run as threads.
    When `use_processes=True`, workers are spawned as child Python processes and
    terminated when scaled down.

        Notes:
        - By default, scale decisions are based on the *local* DuckDB queue backlog.
        - When `autoscale_remote=True`, the manager also polls discovered peers
            (via TaskQueue status(detail=True)) and scales up when remote queues have
            queued tasks of types this node supports.
    - Workers stop cooperatively after they finish their current task.
    """

    import uuid
    import subprocess
    import sys

    min_w = max(0, int(min_workers))
    max_w = max(min_w, int(max_workers))
    poll_s = max(0.2, float(scale_poll_s))
    idle_s = max(0.0, float(scale_down_idle_s))

    if bool(use_processes) and accelerate_instance is not None:
        # In-process accelerate instances can't be shared across processes.
        # Fall back to threads in that case.
        use_processes = False

    if once:
        # Autoscale manager is intended for long-running services.
        # For one-shot runs, just run a single worker.
        return run_worker(
            queue_path=queue_path,
            worker_id=str(base_worker_id),
            poll_interval_s=float(poll_interval_s),
            once=True,
            p2p_service=bool(p2p_service),
            p2p_listen_port=p2p_listen_port,
            accelerate_instance=accelerate_instance,
            supported_task_types=supported_task_types,
            mesh=mesh,
            stop_event=stop_event,
        )

    # Determine supported task types for counting and for child workers.
    # If not explicitly provided, mirror run_worker() defaults.
    supported_for_workers = _compute_supported_task_types(
        supported_task_types=supported_task_types,
        accelerate_instance=accelerate_instance,
    )

    # Lightweight queue reader used only for counts.
    q = TaskQueue(queue_path)

    workers_lock = threading.RLock()
    workers: list[tuple[str, threading.Thread, threading.Event]] = []
    procs: list[tuple[str, subprocess.Popen[object]]] = []
    last_nonzero_ts = 0.0

    # Remote backlog aggregation state.
    remote_lock = threading.RLock()
    remote_state: dict[str, object] = {
        "queued": 0,
        "queued_by_type": {},
        "peers": [],
        "ts": 0.0,
    }
    remote_stop = threading.Event()
    remote_thread: threading.Thread | None = None

    def _make_id(idx: int) -> str:
        return f"{str(base_worker_id)}-w{int(idx)}-{uuid.uuid4().hex[:6]}"

    def _start_one(*, idx: int, start_service: bool) -> None:
        wid = _make_id(idx)

        mesh_for_worker = mesh if start_service else (mesh_children if mesh_children is not None else mesh)
        if bool(use_processes):
            cmd: list[str] = [
                sys.executable,
                "-m",
                "ipfs_accelerate_py.p2p_tasks.worker",
                "--queue",
                str(queue_path),
                "--worker-id",
                str(wid),
                "--poll-interval-s",
                str(float(poll_interval_s)),
            ]
            if bool(start_service):
                cmd.append("--p2p-service")
                if p2p_listen_port is not None:
                    cmd.extend(["--p2p-listen-port", str(int(p2p_listen_port))])
            if bool(mesh_for_worker):
                cmd.append("--mesh")
                if mesh_refresh_s is not None:
                    cmd.extend(["--mesh-refresh-s", str(float(mesh_refresh_s))])
                if mesh_claim_interval_s is not None:
                    cmd.extend(["--mesh-claim-interval-s", str(float(mesh_claim_interval_s))])
                if mesh_max_peers is not None:
                    cmd.extend(["--mesh-max-peers", str(int(mesh_max_peers))])

            env = dict(os.environ)
            if supported_for_workers:
                joined = ",".join(list(supported_for_workers))
                env.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES", joined)
                env.setdefault("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES", joined)

            proc = subprocess.Popen(cmd, env=env)
            with workers_lock:
                procs.append((wid, proc))
            return

        ev = threading.Event()

        def _run() -> None:
            run_worker(
                queue_path=queue_path,
                worker_id=wid,
                poll_interval_s=float(poll_interval_s),
                once=False,
                p2p_service=bool(start_service),
                p2p_listen_port=p2p_listen_port,
                accelerate_instance=accelerate_instance,
                supported_task_types=list(supported_for_workers or []),
                mesh=mesh_for_worker,
                mesh_refresh_s=mesh_refresh_s,
                mesh_claim_interval_s=mesh_claim_interval_s,
                mesh_max_peers=mesh_max_peers,
                stop_event=ev,
            )

        t = threading.Thread(target=_run, name=f"task_worker[{wid}]", daemon=True)
        t.start()
        with workers_lock:
            workers.append((wid, t, ev))

    def _stop_extra(desired: int) -> None:
        with workers_lock:
            while len(procs) > desired:
                wid, proc = procs.pop()
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        pass
            while len(workers) > desired:
                wid, t, ev = workers.pop()
                try:
                    ev.set()
                except Exception:
                    pass
                try:
                    t.join(timeout=0.2)
                except Exception:
                    pass

    # Start initial pool.
    initial = min_w if min_w > 0 else 1
    initial = min(initial, max_w)
    for i in range(initial):
        _start_one(idx=i, start_service=(bool(p2p_service) and i == 0))

    # Optional: remote backlog polling thread.
    if bool(autoscale_remote):
        try:
            from ipfs_accelerate_py.p2p_tasks.client import (
                RemoteQueue,
                discover_peers_via_mdns_sync,
                request_status_sync,
            )

            expected_session = _expected_session_tag()

            def _remote_loop() -> None:
                refresh_s = max(0.5, float(remote_refresh_s))
                limit = max(1, min(100, int(remote_max_peers)))

                while not remote_stop.is_set() and (stop_event is None or not stop_event.is_set()):
                    peers: list[RemoteQueue] = []
                    try:
                        peers = discover_peers_via_mdns_sync(timeout_s=1.0, limit=limit, exclude_self=True)
                    except Exception:
                        peers = []

                    queued_total = 0
                    queued_by_type: dict[str, int] = {}
                    keep: list[RemoteQueue] = []
                    for rq in list(peers or []):
                        try:
                            resp = request_status_sync(remote=rq, timeout_s=3.0, detail=True)
                        except Exception:
                            continue
                        if not (isinstance(resp, dict) and resp.get("ok")):
                            continue
                        if expected_session and str(resp.get("session") or "").strip() != expected_session:
                            continue

                        queue_info = resp.get("queue")
                        if not isinstance(queue_info, dict):
                            continue
                        qb = queue_info.get("queued_by_type")
                        if not isinstance(qb, dict):
                            qb = {}

                        keep.append(rq)
                        for ttype, count in qb.items():
                            tt = str(ttype or "").strip()
                            if not tt:
                                continue
                            if supported_for_workers and tt not in supported_for_workers:
                                continue
                            try:
                                n = int(count)
                            except Exception:
                                n = 0
                            if n <= 0:
                                continue
                            queued_total += n
                            queued_by_type[tt] = int(queued_by_type.get(tt, 0) + n)

                    with remote_lock:
                        remote_state["queued"] = int(queued_total)
                        remote_state["queued_by_type"] = dict(queued_by_type)
                        remote_state["peers"] = list(keep)
                        remote_state["ts"] = float(time.time())

                    remote_stop.wait(refresh_s)

            remote_thread = threading.Thread(
                target=_remote_loop,
                name=f"task_worker_autoscale_remote[{base_worker_id}]",
                daemon=True,
            )
            remote_thread.start()
        except Exception:
            remote_thread = None

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            try:
                pending_local = int(q.count(status="queued", task_types=list(supported_for_workers or [])))
            except Exception:
                pending_local = 0

            pending_remote = 0
            if bool(autoscale_remote):
                with remote_lock:
                    try:
                        pending_remote = int(remote_state.get("queued") or 0)
                    except Exception:
                        pending_remote = 0

            pending = max(0, int(pending_local + pending_remote))

            now = time.time()
            if pending > 0:
                last_nonzero_ts = now

            # Simple heuristic: scale workers up to min(max_w, max(min_w, pending)).
            desired = min_w
            if pending > 0:
                desired = max(min_w, min(max_w, pending))

            # Scale down only after being idle for a while.
            with workers_lock:
                current = (len(procs) if bool(use_processes) else 0) + len(workers)
            if desired < current:
                if idle_s <= 0.0 or (last_nonzero_ts and (now - last_nonzero_ts) >= idle_s) or pending == 0:
                    _stop_extra(desired)
            elif desired > current:
                for i in range(current, desired):
                    _start_one(idx=i, start_service=False)

            if stop_event is None:
                time.sleep(poll_s)
            else:
                stop_event.wait(poll_s)

    finally:
        remote_stop.set()
        if remote_thread is not None:
            try:
                remote_thread.join(timeout=0.5)
            except Exception:
                pass
        # Stop all child workers.
        with workers_lock:
            for _wid, _proc in procs:
                try:
                    _proc.terminate()
                except Exception:
                    pass
            for _wid, _proc in procs:
                try:
                    _proc.wait(timeout=1.0)
                except Exception:
                    try:
                        _proc.kill()
                    except Exception:
                        pass
            for _wid, _t, _ev in workers:
                try:
                    _ev.set()
                except Exception:
                    pass
            for _wid, _t, _ev in workers:
                try:
                    _t.join(timeout=0.5)
                except Exception:
                    pass
            workers.clear()
            procs.clear()
        try:
            q.close()
        except Exception:
            pass

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ipfs_accelerate_py task worker")
    from .task_queue import default_queue_path

    parser.add_argument(
        "--queue",
        dest="queue_path",
        default=default_queue_path(),
        help="Path to task queue DuckDB file (default: env or shared cache path)",
    )
    parser.add_argument("--worker-id", dest="worker_id", required=True, help="Worker identifier")
    parser.add_argument("--poll-interval-s", dest="poll_interval_s", type=float, default=0.5)
    parser.add_argument("--once", action="store_true", help="Process at most one task")
    parser.add_argument("--p2p-service", action="store_true", help="Also start a local libp2p TaskQueue RPC service")
    parser.add_argument(
        "--p2p-listen-port",
        type=int,
        default=None,
        help="TCP port for libp2p service (default: env or 9710)",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Enable mDNS mesh mode: discover peers and claim tasks from them (opt-in).",
    )
    parser.add_argument(
        "--mesh-refresh-s",
        type=float,
        default=None,
        help="How often to refresh mDNS peer list (default: env or 5s)",
    )
    parser.add_argument(
        "--mesh-claim-interval-s",
        type=float,
        default=None,
        help="Minimum seconds between mesh claim attempts when idle (default: env or 0.5s)",
    )
    parser.add_argument(
        "--mesh-max-peers",
        type=int,
        default=None,
        help="Max peers to keep from mDNS discovery (default: env or 10)",
    )

    parser.add_argument(
        "--autoscale",
        action="store_true",
        help="Enable autoscaling worker threads (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE)",
    )
    parser.add_argument(
        "--autoscale-min",
        type=int,
        default=None,
        help="Min autoscaled workers (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN or 1)",
    )
    parser.add_argument(
        "--autoscale-max",
        type=int,
        default=None,
        help="Max autoscaled workers (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX or 4)",
    )
    parser.add_argument(
        "--autoscale-poll-s",
        type=float,
        default=None,
        help="Autoscale poll seconds (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_POLL_S or 2)",
    )
    parser.add_argument(
        "--autoscale-idle-s",
        type=float,
        default=None,
        help="Seconds idle before scaling down (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_IDLE_S or 30)",
    )
    parser.add_argument(
        "--autoscale-remote",
        action="store_true",
        help="Also scale based on discovered remote backlog (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE)",
    )
    parser.add_argument(
        "--autoscale-remote-refresh-s",
        type=float,
        default=None,
        help="Remote backlog refresh seconds (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_REFRESH_S or 5)",
    )
    parser.add_argument(
        "--autoscale-remote-max-peers",
        type=int,
        default=None,
        help="Max remote peers for backlog polling (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_MAX_PEERS or 10)",
    )
    parser.add_argument(
        "--autoscale-mesh-children",
        action="store_true",
        help="Enable mesh mode for autoscaled child workers (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MESH_CHILDREN)",
    )
    parser.add_argument(
        "--autoscale-processes",
        action="store_true",
        help="Spawn autoscaled workers as child processes (default: env IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_PROCESSES)",
    )

    args = parser.parse_args(argv)

    env_autoscale = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE")) or _truthy(
        os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_AUTOSCALE")
    )
    autoscale_enabled = bool(args.autoscale) or bool(env_autoscale)

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        try:
            return int(float(str(raw).strip()))
        except Exception:
            return int(default)

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        try:
            return float(str(raw).strip())
        except Exception:
            return float(default)

    if autoscale_enabled:
        min_w = int(args.autoscale_min) if args.autoscale_min is not None else _env_int(
            "IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN", 1
        )
        max_w = int(args.autoscale_max) if args.autoscale_max is not None else _env_int(
            "IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX", 4
        )
        poll_s = float(args.autoscale_poll_s) if args.autoscale_poll_s is not None else _env_float(
            "IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_POLL_S", 2.0
        )
        idle_s = float(args.autoscale_idle_s) if args.autoscale_idle_s is not None else _env_float(
            "IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_IDLE_S", 30.0
        )
        remote_default = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE"))
        remote_refresh_s = (
            float(args.autoscale_remote_refresh_s)
            if args.autoscale_remote_refresh_s is not None
            else _env_float("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_REFRESH_S", 5.0)
        )
        remote_max_peers = (
            int(args.autoscale_remote_max_peers)
            if args.autoscale_remote_max_peers is not None
            else _env_int("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_MAX_PEERS", 10)
        )
        mesh_children_default = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MESH_CHILDREN"))
        proc_default = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_PROCESSES"))

        return run_autoscaled_workers(
            queue_path=args.queue_path,
            base_worker_id=str(args.worker_id),
            min_workers=max(1, int(min_w)),
            max_workers=max(1, int(max_w)),
            scale_poll_s=max(0.1, float(poll_s)),
            scale_down_idle_s=max(0.0, float(idle_s)),
            poll_interval_s=float(args.poll_interval_s),
            once=bool(args.once),
            p2p_service=bool(args.p2p_service),
            p2p_listen_port=args.p2p_listen_port,
            mesh=(bool(args.mesh) if bool(args.mesh) else None),
            mesh_refresh_s=args.mesh_refresh_s,
            mesh_claim_interval_s=args.mesh_claim_interval_s,
            mesh_max_peers=args.mesh_max_peers,
            mesh_children=bool(args.autoscale_mesh_children) or bool(mesh_children_default),
            autoscale_remote=bool(args.autoscale_remote) or bool(remote_default),
            remote_refresh_s=max(0.5, float(remote_refresh_s)),
            remote_max_peers=max(1, int(remote_max_peers)),
            use_processes=bool(args.autoscale_processes) or bool(proc_default),
        )

    return run_worker(
        queue_path=args.queue_path,
        worker_id=args.worker_id,
        poll_interval_s=float(args.poll_interval_s),
        once=bool(args.once),
        p2p_service=bool(args.p2p_service),
        p2p_listen_port=args.p2p_listen_port,
        mesh=bool(args.mesh) if bool(args.mesh) else None,
        mesh_refresh_s=args.mesh_refresh_s,
        mesh_claim_interval_s=args.mesh_claim_interval_s,
        mesh_max_peers=args.mesh_max_peers,
    )


if __name__ == "__main__":
    raise SystemExit(main())

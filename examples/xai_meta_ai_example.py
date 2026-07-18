#!/usr/bin/env python3
"""
Example: Using xAI Grok and Meta AI backends with IPFS Accelerate

Demonstrates:
- Direct use of the xai and meta_ai API backends
- Text generation via the LLM router (provider="xai" / provider="meta_ai")
- Embeddings via the embeddings router
- Vision / multimodal inference via the multimodal router
- The xAI Grok CLI integration (plan mode, subagents, web search)
- The Meta AI CLI integration (creative mode, vision chat, model selector)

Required environment variables
--------------------------------
xAI:
  XAI_API_KEY   or   ipfs_accelerate_py_XAI_API_KEY

Meta AI:
  META_AI_API_KEY   or   ipfs_accelerate_py_META_AI_API_KEY
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Direct API backend usage
# ---------------------------------------------------------------------------

def example_xai_direct():
    """Use the xai backend class directly."""
    print("\n" + "=" * 70)
    print("Example 1a: xAI Grok – direct backend")
    print("=" * 70)

    from ipfs_accelerate_py.api_backends.xai import xai

    client = xai(metadata={"api_key": os.environ.get("XAI_API_KEY", "")})
    print("Available models:", client.list_models())
    info = client.get_model_info("grok-3")
    print("grok-3 context window:", info.get("context_window") if info else "N/A")

    if not client.api_key:
        print("(skipping live call – XAI_API_KEY not set)")
        return

    text = client.generate("What is IPFS in one sentence?")
    print("Response:", text)


def example_meta_ai_direct():
    """Use the meta_ai backend class directly."""
    print("\n" + "=" * 70)
    print("Example 1b: Meta AI – direct backend")
    print("=" * 70)

    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai

    client = meta_ai(metadata={"api_key": os.environ.get("META_AI_API_KEY", "")})
    print("Available models:", client.list_models()[:3], "...")

    if not client.api_key:
        print("(skipping live call – META_AI_API_KEY not set)")
        return

    text = client.generate("What is libp2p in one sentence?")
    print("Response:", text)


# ---------------------------------------------------------------------------
# 2. LLM router
# ---------------------------------------------------------------------------

def example_llm_router():
    """Route text generation requests through the LLM router."""
    print("\n" + "=" * 70)
    print("Example 2: LLM router (generate_text)")
    print("=" * 70)

    try:
        from ipfs_accelerate_py import generate_text
    except ImportError as e:
        print(f"LLM router not available: {e}")
        return

    if os.environ.get("XAI_API_KEY") or os.environ.get("ipfs_accelerate_py_XAI_API_KEY"):
        result = generate_text("Say hello from xAI Grok.", provider="xai", model_name="grok-3")
        print("xAI response:", result)
    else:
        print("(skipping xAI call – XAI_API_KEY not set)")

    if os.environ.get("META_AI_API_KEY") or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY"):
        result = generate_text(
            "Say hello from Meta Llama.",
            provider="meta_ai",
            model_name="meta-llama/Llama-3.3-70B-Instruct",
        )
        print("Meta AI response:", result)
    else:
        print("(skipping Meta AI call – META_AI_API_KEY not set)")

    # Provider aliases also work
    # generate_text("Hello", provider="grok")        # xAI
    # generate_text("Hello", provider="spark")       # Meta Spark
    # generate_text("Hello", provider="meta_llama")  # Meta Llama


# ---------------------------------------------------------------------------
# 3. Embeddings router
# ---------------------------------------------------------------------------

def example_embeddings_router():
    """Generate embeddings via the embeddings router."""
    print("\n" + "=" * 70)
    print("Example 3: Embeddings router (embed_texts)")
    print("=" * 70)

    try:
        from ipfs_accelerate_py import embed_texts
    except ImportError as e:
        print(f"Embeddings router not available: {e}")
        return

    texts = ["IPFS is a distributed file system.", "libp2p is a modular networking stack."]

    if os.environ.get("XAI_API_KEY") or os.environ.get("ipfs_accelerate_py_XAI_API_KEY"):
        vecs = embed_texts(texts, provider="xai")
        print(f"xAI embeddings: {len(vecs)} vectors, dim={len(vecs[0])}")
    else:
        print("(skipping xAI embeddings – XAI_API_KEY not set)")

    if os.environ.get("META_AI_API_KEY") or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY"):
        vecs = embed_texts(texts, provider="meta_ai")
        print(f"Meta AI embeddings: {len(vecs)} vectors, dim={len(vecs[0])}")
    else:
        print("(skipping Meta AI embeddings – META_AI_API_KEY not set)")


# ---------------------------------------------------------------------------
# 4. Multimodal router (vision)
# ---------------------------------------------------------------------------

def example_multimodal_router():
    """Run vision inference via the multimodal router."""
    print("\n" + "=" * 70)
    print("Example 4: Multimodal router (generate_multimodal)")
    print("=" * 70)

    try:
        from ipfs_accelerate_py import generate_multimodal
    except ImportError as e:
        print(f"Multimodal router not available: {e}")
        return

    sample_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    if os.environ.get("XAI_API_KEY") or os.environ.get("ipfs_accelerate_py_XAI_API_KEY"):
        result = generate_multimodal(
            "Describe this image briefly.",
            image=sample_image,
            provider="xai",
            model_name="grok-2-vision-1212",
        )
        print("xAI vision response:", result)
    else:
        print("(skipping xAI vision – XAI_API_KEY not set)")

    if os.environ.get("META_AI_API_KEY") or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY"):
        result = generate_multimodal(
            "Describe this image briefly.",
            image=sample_image,
            provider="meta_ai",
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
        )
        print("Meta AI vision response:", result)
    else:
        print("(skipping Meta AI vision – META_AI_API_KEY not set)")


# ---------------------------------------------------------------------------
# 5. CLI integrations
# ---------------------------------------------------------------------------

def example_xai_cli_integration():
    """Demonstrate the xAI Grok CLI integration (plan mode, subagents)."""
    print("\n" + "=" * 70)
    print("Example 5a: xAI Grok CLI integration")
    print("=" * 70)

    try:
        from ipfs_accelerate_py.cli_integrations import XAIGrokCLIIntegration
    except ImportError as e:
        print(f"xAI CLI integration not available: {e}")
        return

    grok = XAIGrokCLIIntegration(headless=True)
    print("Tool name:", grok.get_tool_name())
    print("Available models:", grok.list_models()[:3])

    if not (os.environ.get("XAI_API_KEY") or os.environ.get("ipfs_accelerate_py_XAI_API_KEY")):
        print("(skipping live calls – XAI_API_KEY not set)")
        return

    # Plan mode
    result = grok.plan_mode("Write a Python function to compute Fibonacci numbers.")
    print("Plan mode approved:", result.get("approved"))
    if result.get("plan"):
        print("Plan (first 200 chars):", result["plan"][:200])

    # Subagents
    tasks = ["What is IPFS?", "What is libp2p?"]
    results = grok.spawn_subagents(tasks)
    for task, res in zip(tasks, results):
        print(f"Subagent [{task[:30]}]: {res.get('response', '')[:80]}")


def example_meta_ai_cli_integration():
    """Demonstrate the Meta AI CLI integration (creative mode, vision)."""
    print("\n" + "=" * 70)
    print("Example 5b: Meta AI CLI integration")
    print("=" * 70)

    try:
        from ipfs_accelerate_py.cli_integrations import MetaAICLIIntegration
    except ImportError as e:
        print(f"Meta AI CLI integration not available: {e}")
        return

    meta = MetaAICLIIntegration(headless=True)
    print("Tool name:", meta.get_tool_name())
    print("Available models:", meta.list_models()[:3])

    if not (os.environ.get("META_AI_API_KEY") or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY")):
        print("(skipping live calls – META_AI_API_KEY not set)")
        return

    # Model selector
    creative_model = meta.suggest_model(task_hint="creative")
    vision_model = meta.suggest_model(task_hint="vision")
    print("Suggested model for creative tasks:", creative_model)
    print("Suggested model for vision tasks:", vision_model)

    # Creative mode (headless → auto-approved)
    result = meta.creative_mode("Write a three-line haiku about distributed systems.")
    print("Creative mode approved:", result.get("approved"))
    if result.get("content"):
        print("Content:", result["content"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    example_xai_direct()
    example_meta_ai_direct()
    example_llm_router()
    example_embeddings_router()
    example_multimodal_router()
    example_xai_cli_integration()
    example_meta_ai_cli_integration()
    print("\nAll examples completed.")


if __name__ == "__main__":
    main()

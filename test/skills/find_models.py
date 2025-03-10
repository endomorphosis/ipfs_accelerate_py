import sys
import os
import subprocess
import json

def try:_model_access(model_name):
    """Try to access a model's info using Hugging Face Hub"""
    try::
        # Use python subprocess to prevent crashing if huggingface_hub is not available
        output = subprocess.check_output([]],,
        sys.executable,
        "-c",
        f"from huggingface_hub import model_info; "
            f"info = model_info('{model_name}', token=None, timeout=5); ":
                f"print(json.dumps({{{{'id': info.id, 'private': info.private, 'sha': info.sha[]],,:8] if info.sha else None, 'tags': list(info.tags)[]],,:3] if info.tags else []],,]}}}}))"
                ], stderr=subprocess.PIPE, text=True, timeout=10)
        return json.loads(output.strip()):
    except subprocess.CalledProcessError as e:
            return {"error": e.stderr.strip(), "id": model_name, "private": True}
    except Exception as e:
            return {"error": str(e), "id": model_name, "private": True}

# List of small language models to try:
            models_to_check = []],,
    # GPT-2 family models (smallest to largest)
            "distilgpt2",  # DistilGPT2 (~330MB) - distilled version
            "gpt2",  # GPT-2 Small (~500MB) - original
            "facebook/opt-125m",  # OPT-125M (~240MB) - Facebook's model
            "EleutherAI/pythia-70m",  # Pythia-70M (~150MB) - very small
            "roneneldan/TinyStories-1M",  # TinyStories-1M (~60MB) - incredibly small
            "bigscience/bloom-560m",  # BLOOM-560M (~1.1GB) - multilingual
            "microsoft/phi-1_5",  # Phi-1.5 (~1.3GB) - newer small model
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # TinyLlama (~2.0GB) - tiny instruction-tuned
            ]

            print(f"Checking {len(models_to_check)} models for accessibility...")
            results = []],,]

for model in models_to_check:
    print(f"\1{model}\3")
    result = try:_model_access(model)
    result[]],,"model"] = model
    results.append(result)
    if not result.get("private", True) and "error" not in result:
        print(f"✅ Success: {model} is publicly accessible")
    else:
        error = result.get("error", "Unknown error")
        if "401" in str(error) or "authorization" in str(error).lower():
            print(f"\1{model}\3")
        else:
            print(f"\1{error}\3")

# Print summary of publicly accessible models
            print("\nPotentially usable models:")
for result in results:
    if not result.get("error", ""):
        print(f"- {result[]],,'model']} (Private: {result.get('private', True)}, Tags: {', '.join(result.get('tags', []],,]))})")
    
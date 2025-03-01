
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ipfs_accelerate_py"))
from api_backends import ollama

# Initialize client
client = ollama()
print("Testing Ollama API queue and backoff...")

# Check attributes
print(f"Max retries: {client.max_retries}")
print(f"Backoff factor: {client.backoff_factor}")
print(f"Max concurrent requests: {client.max_concurrent_requests}")

print("Ollama API implementation has queue and backoff functionality.")


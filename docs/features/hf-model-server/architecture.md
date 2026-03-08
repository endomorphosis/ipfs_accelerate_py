# HuggingFace Model Server - Architecture Diagrams

> Visual representations of the current system and proposed unified server architecture

---

## Current System Architecture

### Generator to Skills Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Skill Generation System                          │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Model Input: "bert-base-uncased"                            │   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                            ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Architecture Detection                                      │   │
│  │  • Query HuggingFace metadata                               │   │
│  │  • Classify: encoder-only | decoder-only | vision | etc.    │   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                            ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Template Selection                                          │   │
│  │  • Base: hf_reference_template.py                           │   │
│  │  • Architecture-specific templates                          │   │
│  │  • Task-specific logic                                      │   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                            ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Variable Substitution (Jinja2)                             │   │
│  │  • {model_type} → "bert"                                    │   │
│  │  • {task_type} → "text_embedding"                           │   │
│  │  • {hidden_size}, {num_layers}, etc.                        │   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                            ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Hardware Template Composition                               │   │
│  │  • init_cuda(), init_rocm(), init_cpu(), etc.               │   │
│  │  • Hardware-specific endpoint handlers                      │   │
│  │  • Fallback logic                                           │   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                            ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Generated Skill: hf_bert.py                                │   │
│  │                                                              │   │
│  │  class hf_bert:                                             │   │
│  │      def __init__(...)                                      │   │
│  │      def init_cpu(...)                                      │   │
│  │      def init_cuda(...)                                     │   │
│  │      def init_rocm(...)                                     │   │
│  │      def create_cpu_text_embedding_endpoint_handler(...)    │   │
│  │      def create_cuda_text_embedding_endpoint_handler(...)   │   │
│  │      ...                                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                     Generated Skills Storage                       │
│                                                                     │
│  ipfs_accelerate_py/worker/skillset/                              │
│  ├── hf_bert.py          ✅ Production ready                      │
│  ├── hf_gpt2.py          ✅ Production ready                      │
│  ├── hf_llama.py         ✅ Production ready                      │
│  ├── hf_clip.py          ✅ Production ready                      │
│  ├── hf_whisper.py       ✅ Production ready                      │
│  ├── hf_t5.py            ✅ Production ready                      │
│  └── ... (15+ skills)                                             │
│                                                                     │
│  test/ipfs_accelerate_py/worker/skillset/                         │
│  ├── hf_albert.py        🧪 Test/dev                             │
│  ├── hf_bart.py          🧪 Test/dev                             │
│  ├── hf_deberta.py       🧪 Test/dev                             │
│  └── ... (200+ skills)                                            │
└───────────────────────────────────────────────────────────────────┘

⚠️  PROBLEM: No unified way to serve these skills!
    Each skill must be manually imported and used.
```

---

## Hardware Abstraction Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Application/Model Layer                           │
│                 (hf_bert, hf_llama, hf_clip, etc.)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 Hardware Abstraction Interface                       │
│                                                                       │
│  • detect_available_hardware() → {platform: bool}                   │
│  • select_optimal_hardware(model_name) → "cuda" | "rocm" | ...     │
│  • get_device(hardware) → torch.device                              │
│                                                                       │
└─────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────────┘
      ↓         ↓         ↓         ↓         ↓         ↓
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│   CUDA   │   ROCm   │   MPS    │ OpenVINO │   QNN    │   CPU    │
│ (NVIDIA) │  (AMD)   │ (Apple)  │ (Intel)  │(Qualcomm)│(Fallback)│
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│          │          │          │          │          │          │
│ Features │ Features │ Features │ Features │ Features │ Features │
│ ────────│ ──────── │ ──────── │ ──────── │ ──────── │ ────────│
│ • FP16   │ • FP16   │ • FP16   │ • INT8   │ • INT8   │ • FP32   │
│ • Tensor │ • HIP API│ • Unified│ • Model  │ • Hexagon│ • MKL    │
│   Cores  │ • MIOpen │   Memory │   Convert│   DSP    │ • OpenMP │
│ • Flash  │ • Limited│ • Limited│ • CPU/GPU│ • Edge   │ • ONNX   │
│   Attn   │   Quant  │   Quant  │ • VPU    │   Focus  │   Runtime│
│ • 8/4bit │          │          │          │          │          │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
    ↓           ↓           ↓           ↓           ↓           ↓
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ CUDA     │ ROCm     │ Metal    │ OpenVINO │ QNN      │ CPU      │
│ Runtime  │ Runtime  │ Runtime  │ Runtime  │ Runtime  │ Native   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Hardware Selection Priority (Auto-detect):
1. CUDA (if available + compatible)
2. ROCm (if CUDA unavailable + AMD GPU present)
3. MPS (if Apple Silicon)
4. OpenVINO (if Intel hardware + model converted)
5. QNN (if Qualcomm + edge deployment)
6. CPU (ultimate fallback, always available)
```

---

## Proposed Unified Model Server Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                │
│                                                                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │   REST API    │  │     gRPC      │  │   WebSocket   │           │
│  │   (FastAPI)   │  │   (optional)  │  │   (optional)  │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Python SDK (for direct integration)                       │     │
│  └───────────────────────────────────────────────────────────┘     │
└────────────────────────────┬──────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY LAYER                              │
│                                                                       │
│  ┌───────────────┬───────────────┬───────────────┬──────────────┐  │
│  │    OpenAI     │  HuggingFace  │    Custom     │     MCP      │  │
│  │  Compatible   │  Compatible   │   Endpoints   │   Protocol   │  │
│  ├───────────────┼───────────────┼───────────────┼──────────────┤  │
│  │ /v1/          │ /v1/generate  │ /v1/infer/    │ (via         │  │
│  │ completions   │ /v1/embed     │ {model}       │  FastMCP)    │  │
│  │ /v1/          │ /v1/classify  │ /v1/hardware  │              │  │
│  │ embeddings    │ /v1/detect    │ /health       │              │  │
│  │ /v1/models    │               │ /metrics      │              │  │
│  └───────────────┴───────────────┴───────────────┴──────────────┘  │
│                                                                       │
│  Middleware: Auth, CORS, Rate Limiting, Request Validation           │
└────────────────────────────┬──────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    REQUEST MANAGEMENT LAYER                          │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Request Router & Queue Manager                              │   │
│  │  • Priority Queue (High/Normal/Low)                          │   │
│  │  • Circuit Breaker (fault tolerance)                         │   │
│  │  • Rate Limiter (per-user, per-model)                        │   │
│  │  • Request Batching (efficiency)                             │   │
│  │  • Load Balancer (distribute load)                           │   │
│  │  • Response Cache (reduce redundancy)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬──────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL MANAGEMENT LAYER                         │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Model Registry                                               │  │
│  │  • discover_models() - Scan for hf_*.py files               │  │
│  │  • register_model() - Add model to registry                 │  │
│  │  • list_models() - Get all available models                 │  │
│  │  • get_model() - Retrieve specific model info               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Hardware Manager                                             │  │
│  │  • detect_hardware() - Auto-detect available platforms       │  │
│  │  • select_hardware() - Choose optimal for model              │  │
│  │  • benchmark_model() - Performance profiling                 │  │
│  │  • get_device() - Get PyTorch device                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Model Loader                                                 │  │
│  │  • load_model() - Load and initialize skill                  │  │
│  │  • unload_model() - Free resources                           │  │
│  │  • Loaded model cache                                        │  │
│  │  • Memory management                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Bandit Recommender (Optional)                               │  │
│  │  • Intelligent model selection                               │  │
│  │  • Performance-based routing                                 │  │
│  │  • A/B testing support                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       EXECUTION ENGINE                               │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Model Executor                                               │  │
│  │  • execute() - Run inference on loaded model                 │  │
│  │  • Thread pool for concurrent execution                      │  │
│  │  • Async/await support                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Hardware Abstraction Layer                                   │  │
│  │  ┌────────┬────────┬────────┬────────┬────────┬────────┐    │  │
│  │  │  CUDA  │  ROCm  │  MPS   │OpenVINO│  QNN   │  CPU   │    │  │
│  │  └────────┴────────┴────────┴────────┴────────┴────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        HF SKILLS LAYER                               │
│                                                                       │
│  Generated Skills (automatically discovered and loaded):             │
│                                                                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐         │
│  │  hf_bert.py │ hf_gpt2.py  │ hf_llama.py │ hf_clip.py  │         │
│  │  • Text     │ • Text Gen  │ • LLM       │ • Vision+   │         │
│  │  Embedding  │             │             │ Text        │         │
│  └─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                                                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐         │
│  │hf_whisper.py│  hf_t5.py   │  hf_vit.py  │ hf_detr.py  │         │
│  │ • Speech    │ • Seq2Seq   │ • Image     │ • Object    │         │
│  │   to Text   │ Translation │   Classify  │   Detection │         │
│  └─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                                                       │
│  ... and 200+ more skills ...                                        │
└─────────────────────────────────────────────────────────────────────┘

Data Flow Example (Text Generation):
1. Client sends POST to /v1/completions
2. API Gateway validates & routes request
3. Request Queue prioritizes & batches
4. Model Registry finds hf_gpt2
5. Hardware Manager selects CUDA (if available)
6. Model Loader loads hf_gpt2 with init_cuda()
7. Executor calls create_cuda_text_generation_endpoint_handler()
8. Model generates text on CUDA
9. Response returned to client
```

---

## Component Interaction Diagram

### Model Loading Flow

```
┌─────────┐
│ Client  │ POST /v1/completions {model: "hf_gpt2", prompt: "..."}
└────┬────┘
     │
     ↓
┌────────────────────┐
│  API Gateway       │ Validate request, extract model name
└─────────┬──────────┘
          │
          ↓
┌─────────────────────────┐
│  Model Registry         │ Query: Is "hf_gpt2" registered?
│  └→ YES: Return         │ → ModelInfo(name="hf_gpt2", 
│     model_info          │            arch="decoder-only",
└─────────┬───────────────┘            tasks=["text_generation"])
          │
          ↓
┌─────────────────────────┐
│  Hardware Manager       │ select_hardware("hf_gpt2")
│  └→ detect_hardware()   │ → Available: {cuda: True, rocm: False, ...}
│  └→ priority: CUDA      │ → Selected: "cuda"
└─────────┬───────────────┘
          │
          ↓
┌─────────────────────────┐
│  Model Loader           │ load_model("hf_gpt2", "cuda")
│  └→ Check cache         │ → Not in cache, need to load
│  └→ Import hf_gpt2.py   │ → import ipfs_accelerate_py.worker.skillset.hf_gpt2
│  └→ Instantiate         │ → skill = hf_gpt2()
│  └→ Initialize          │ → skill.init_cuda()
│  └→ Cache               │ → cache["hf_gpt2_cuda"] = skill
└─────────┬───────────────┘
          │
          ↓ (skill instance)
┌─────────────────────────┐
│  Model Executor         │ execute(skill, "text_generation", prompt)
│  └→ Get handler         │ → handler = skill.create_cuda_text_generation_endpoint_handler()
│  └→ Run inference       │ → result = handler(prompt)
└─────────┬───────────────┘
          │
          ↓ (result)
┌─────────────────────────┐
│  API Gateway            │ Format response (OpenAI-compatible)
└─────────┬───────────────┘
          │
          ↓
┌─────────────────────────┐
│  Client                 │ Receive: {"choices": [{"text": "..."}], ...}
└─────────────────────────┘
```

---

## Deployment Architecture

### Single-Node Deployment

```
┌───────────────────────────────────────────────────────────┐
│                        Server Host                         │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐│
│  │           Docker Container                             ││
│  │                                                         ││
│  │  ┌─────────────────────────────────────────────────┐  ││
│  │  │  HF Model Server (FastAPI)                      │  ││
│  │  │  Port: 8000                                     │  ││
│  │  │  • Model Registry (auto-discover)               │  ││
│  │  │  • Hardware Manager (detect CUDA/ROCm/CPU)      │  ││
│  │  │  • Model Loader (LRU cache, 16GB limit)         │  ││
│  │  │  • Executor (4 worker threads)                  │  ││
│  │  └─────────────────────────────────────────────────┘  ││
│  │                                                         ││
│  │  ┌─────────────────────────────────────────────────┐  ││
│  │  │  PyTorch + Transformers                         │  ││
│  │  └─────────────────────────────────────────────────┘  ││
│  │                                                         ││
│  │  ┌─────────────────────────────────────────────────┐  ││
│  │  │  CUDA Runtime                                   │  ││
│  │  └─────────────────────────────────────────────────┘  ││
│  └───────────────────────────────────────────────────────┘│
│                                                             │
│  Hardware:                                                  │
│  • CPU: 8+ cores                                           │
│  • RAM: 32GB+                                              │
│  • GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)               │
└───────────────────────────────────────────────────────────┘
```

### Multi-Node Kubernetes Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                            │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Load Balancer (Ingress)                                      │  │
│  │  • SSL Termination                                            │  │
│  │  • Path routing                                               │  │
│  └────────────────────┬──────────────────────────────────────────┘  │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Service: hf-model-server (ClusterIP)                        │   │
│  │  • Port: 80 → 8000                                           │   │
│  └────────┬──────────────────┬──────────────────┬───────────────┘   │
│           ↓                  ↓                  ↓                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │  Pod 1          │ │  Pod 2          │ │  Pod 3          │       │
│  │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │       │
│  │  │   Server  │  │ │  │   Server  │  │ │  │   Server  │  │       │
│  │  │   App     │  │ │  │   App     │  │ │  │   App     │  │       │
│  │  └─────┬─────┘  │ │  └─────┬─────┘  │ │  └─────┬─────┘  │       │
│  │        ↓        │ │        ↓        │ │        ↓        │       │
│  │  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │       │
│  │  │ GPU: CUDA │  │ │  │ GPU: CUDA │  │ │  │ GPU: ROCm │  │       │
│  │  │ 24GB VRAM │  │ │  │ 24GB VRAM │  │ │  │ 16GB VRAM │  │       │
│  │  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │       │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘       │
│                                                                       │
│  Horizontal Pod Autoscaler:                                          │
│  • Min: 2 pods                                                       │
│  • Max: 10 pods                                                      │
│  • Target: 70% GPU utilization                                       │
│                                                                       │
│  Persistent Volumes:                                                 │
│  • Model cache: 100GB (shared NFS)                                  │
│  • Logs: 10GB per pod                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Monitoring Stack                                 │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Prometheus                                                  │   │
│  │  • Scrapes /metrics endpoint                                │   │
│  │  • Time-series storage                                      │   │
│  │  • Alerting rules                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             ↓                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Grafana Dashboards                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │  Model Metrics  │  │ Hardware Metrics│                  │   │
│  │  │  • Requests/s   │  │ • GPU utilization│                 │   │
│  │  │  • Latency      │  │ • VRAM usage    │                  │   │
│  │  │  • Error rate   │  │ • Temperature   │                  │   │
│  │  └─────────────────┘  └─────────────────┘                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │  Queue Metrics  │  │  System Metrics │                  │   │
│  │  │  • Queue depth  │  │  • CPU usage    │                  │   │
│  │  │  • Wait time    │  │  • RAM usage    │                  │   │
│  │  │  • Throughput   │  │  • Network I/O  │                  │   │
│  │  └─────────────────┘  └─────────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Logging (ELK Stack / Loki)                                 │   │
│  │  • Application logs                                         │   │
│  │  • Error tracking                                           │   │
│  │  • Request tracing                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

Key Metrics Collected:
┌────────────────────────────────────────────────────────────┐
│ • model_requests_total{model, hardware, status}            │
│ • model_request_duration_seconds{model, hardware}          │
│ • model_tokens_processed_total{model}                      │
│ • hardware_utilization_percent{hardware}                   │
│ • hardware_memory_used_bytes{hardware}                     │
│ • queue_depth{priority}                                    │
│ • circuit_breaker_state{model}                            │
│ • cache_hit_ratio                                          │
└────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Security Layers                               │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. Network Security                                         │   │
│  │  • HTTPS/TLS 1.3                                            │   │
│  │  • Firewall rules                                           │   │
│  │  • DDoS protection                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             ↓                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  2. Authentication & Authorization                           │   │
│  │  • API Keys (X-API-Key header)                              │   │
│  │  • JWT tokens                                               │   │
│  │  • Rate limiting per user                                   │   │
│  │  • Role-based access control (RBAC)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             ↓                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  3. Input Validation                                         │   │
│  │  • Pydantic models                                          │   │
│  │  • Schema validation                                        │   │
│  │  • Sanitization (prevent injection)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             ↓                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  4. Resource Limits                                          │   │
│  │  • Max tokens per request                                   │   │
│  │  • Memory limits                                            │   │
│  │  • Timeout limits                                           │   │
│  │  • Concurrent request limits                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             ↓                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  5. Audit & Logging                                          │   │
│  │  • Request logging                                          │   │
│  │  • Error tracking                                           │   │
│  │  • Security event logging                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary

These diagrams illustrate:

1. **Current System:** Generator creates 200+ skills, but no unified server
2. **Hardware Layer:** 6-platform support with automatic detection
3. **Proposed Server:** Complete architecture with all components
4. **Component Flow:** How requests flow through the system
5. **Deployment:** Single-node and multi-node options
6. **Monitoring:** Observability stack
7. **Security:** Multi-layer protection

**Next:** Review the full technical document [`review.md`](./review.md) for implementation details.

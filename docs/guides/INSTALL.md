# Installation Guide

This project now uses pyproject.toml as the single source of truth for dependencies. Create a virtual environment and choose one of the install profiles below.

## 1) Core install (default)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```
Notes:
- We pin urllib3<2 to maintain compatibility with ipfshttpclient 0.8.x.

## 2) Minimal runtime
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[minimal]
```

## 3) Full runtime (models, api server, etc.)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[full]
```

## 4) MCP server extras
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[mcp]
```

## 5) WebNN/WebGPU related
Use the separate requirements file if you need browser automation and web backends:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_webnn_webgpu.txt
```

## 6) Enhanced model scraper
This set is heavy and optional. Install only if you are using the enhanced scraping pipeline:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_enhanced_scraper.txt
```

## Troubleshooting
- Externally managed environment error (PEP 668): create a venv first as shown above.
- If you already have urllib3>=2, pip will downgrade to <2 for ipfshttpclient compatibility.
- If you run into a solver conflict, try a clean venv: `rm -rf .venv && python3 -m venv .venv`.

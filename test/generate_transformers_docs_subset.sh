#!/bin/bash

# Generate a subset of the documentation (for faster testing)
echo "Generating subset of Transformers documentation..."
python build_transformers_docs.py --output-dir ./transformers_docs_build --language en --subset --max-files 20

echo "Documentation subset generated in ./transformers_docs_build/en/"
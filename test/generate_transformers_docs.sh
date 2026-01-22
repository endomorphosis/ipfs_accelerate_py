#!/bin/bash

# Generate full documentation
echo "Generating full Transformers documentation..."
python build_transformers_docs.py --output-dir ./transformers_docs_build --language en

echo "Documentation generated in ./transformers_docs_build/en/"
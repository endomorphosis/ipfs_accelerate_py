#!/bin/bash

echo "Installing dependencies..."
source ipfs_env/bin/activate
pip install flask flask-cors requests

echo "Testing minimal server..."
python3 test_minimal_server.py

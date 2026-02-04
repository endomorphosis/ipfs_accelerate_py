#!/bin/bash

# Build and run examples
echo "Building examples..."
npm run build:examples

# Copy HTML files to dist directory
echo "Copying HTML files..."
mkdir -p dist/examples
cp src/examples/*.html dist/examples/

echo "Starting server..."
npm run serve
#!/bin/bash
# Script to fix the key model tests for all hardware platforms

echo "Starting to fix key model tests across hardware platforms..."

# First list all models that need fixing
echo "Models that need fixing:"
python fix_key_model_tests.py --list-models

# Fix high priority models first
echo -e "\nFixing high priority models:"
python fix_key_model_tests.py --high-priority

# Fix remaining models
echo -e "\nFixing medium priority models:"
python fix_key_model_tests.py

echo -e "\nDone fixing model tests!"
echo "Remember to verify the implementations by running the tests."
echo "You can use the commands found in CLAUDE.md to test the models on different hardware platforms."
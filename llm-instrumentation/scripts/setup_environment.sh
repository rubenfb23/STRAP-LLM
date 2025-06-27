#!/bin/bash

# This script sets up the environment for the LLM Instrumentation Framework.

# Create a virtual environment
python3 -m venv ../.venv

# Activate the virtual environment
source ../.venv/bin/activate

# Install the required packages
python3 -m pip install -r ../requirements.txt

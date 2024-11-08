#!/usr/bin/env bash

# Update the package list and install system dependencies
apt-get update
apt-get install -y swig libfaiss-dev build-essential

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
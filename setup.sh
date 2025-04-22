#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
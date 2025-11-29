#!/bin/bash
# RunPod Setup Script for Reasoning Direction Experiments
# Run this after starting your RunPod instance

set -e

echo "=================================="
echo "RunPod Setup for R1D1 Project"
echo "=================================="

# 1. Install uv (fast Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Clone or upload the project
# Option A: If you have it in a git repo
# git clone <your-repo-url> r1d1
# cd r1d1

# Option B: If uploading (you'll do this manually)
# Just cd into the uploaded directory
echo "Make sure you've uploaded the r1d1 directory to this pod"
cd /workspace/r1d1 || cd ~/r1d1 || echo "Please cd to your project directory"

# 3. Install dependencies
echo "Installing dependencies with uv..."
uv sync

# 4. Check GPU availability
echo "Checking GPU..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

# 5. Create results directory
mkdir -p results

echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Quick commands:"
echo "  Test: uv run python examples/quick_test.py"
echo "  Full: uv run python run_experiment.py"
echo ""
echo "Don't forget to download results/ before stopping the pod!"


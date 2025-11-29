# Usage Guide

## Installation

```bash
# Clone/navigate to the project
cd r1d1

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Simple Test Run

For a quick test with minimal compute:

```bash
uv run python examples/quick_test.py
```

This uses:
- Toy problems (simple arithmetic)
- Just 3 layers (0, 1, 2)
- Only 5 samples for activation extraction
- Pre-attention points only
- Tests only the reasoning model

### 2. Full Experiment

Run the complete experiment pipeline:

```bash
uv run python run_experiment.py
```

This will:
1. Load both models (Llama-3 8B and DeepSeek R1)
2. Extract activations from GSM8K problems (100 samples)
3. Calculate reasoning directions
4. Run interventions on the reasoning model
5. Save all results to the `results/` directory

### 3. Interactive Jupyter Notebook

Launch Jupyter to experiment interactively:

```bash
uv run jupyter lab
# Then open examples/interactive_demo.ipynb
```

The notebook walks through:
- Loading models
- Extracting activations
- Calculating directions
- Testing interventions step-by-step
- Visualizing results

## Command Line Options

### Dataset Selection

```bash
# Use GSM8K (default, 100 samples)
uv run python run_experiment.py --dataset gsm8k --num-samples 100

# Use toy problems for quick testing
uv run python run_experiment.py --dataset toy
```

### Layer Selection

```bash
# Test specific layers
uv run python run_experiment.py --layers 0 1 2 3

# Test a range of layers
uv run python run_experiment.py --layers 0 5 10 15 20 25 30

# Test all layers (0-30 for 8B models)
uv run python run_experiment.py --layers $(seq 0 30)
```

### Intervention Strengths

```bash
# Default strengths
uv run python run_experiment.py --strengths -0.1 -0.05 0.05 0.1

# Test stronger interventions
uv run python run_experiment.py --strengths -0.2 -0.1 0.1 0.2

# Fine-grained sweep
uv run python run_experiment.py --strengths -0.15 -0.1 -0.05 0.05 0.1 0.15
```

### Activation Points

```bash
# Test all points (pre, mid, post attention)
uv run python run_experiment.py --points pre mid post

# Test only pre-attention (fastest)
uv run python run_experiment.py --points pre

# Test post-attention only
uv run python run_experiment.py --points post
```

### Direction Types

```bash
# Test difference direction (recommended)
uv run python run_experiment.py --direction-types difference

# Test all direction types
uv run python run_experiment.py --direction-types difference original reasoning random

# Compare reasoning vs original
uv run python run_experiment.py --direction-types reasoning original
```

### Target Models

```bash
# Test on reasoning model only (default)
uv run python run_experiment.py --test-reasoning

# Test on non-reasoning model only
uv run python run_experiment.py --test-non-reasoning

# Test on both models
uv run python run_experiment.py --test-both
```

### Model Selection

```bash
# Use different models
uv run python run_experiment.py \
    --reasoning-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --non-reasoning-model meta-llama/Meta-Llama-3-8B-Instruct
```

## Common Workflows

### Quick Testing (Fast)

```bash
# 5-10 minutes on single GPU
uv run python run_experiment.py \
    --dataset toy \
    --layers 0 1 2 \
    --points pre \
    --strengths -0.1 0.1 \
    --direction-types difference
```

### Standard Experiment (Medium)

```bash
# 30-60 minutes on single GPU
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 50 \
    --layers 0 1 2 3 10 15 20 \
    --points pre post \
    --strengths -0.1 -0.05 0.05 0.1 \
    --test-reasoning
```

### Comprehensive Sweep (Slow)

```bash
# Several hours on single GPU
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 100 \
    --layers 0 1 2 3 5 7 10 12 15 17 20 22 25 27 30 \
    --points pre mid post \
    --strengths -0.2 -0.15 -0.1 -0.05 0.05 0.1 0.15 0.2 \
    --direction-types difference original reasoning random \
    --test-both
```

### Layer 0 Focus (Reproduce Key Finding)

Based on the paper's finding that layer 0 shows strong effects:

```bash
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 100 \
    --layers 0 \
    --points pre mid post \
    --strengths -0.2 -0.15 -0.1 -0.05 0.05 0.1 0.15 0.2 \
    --test-reasoning
```

## Analyzing Results

### View Results Files

Results are saved in the `results/` directory with timestamps:

```bash
# List all results
ls -lh results/

# View latest intervention results
cat results/interventions_reasoning_*.json | tail -100
```

### Use the Analysis Script

```bash
# Analyze token counts and reasoning suppression
uv run python examples/analyze_results.py results/interventions_reasoning_20241129_143022.json
```

The analysis script shows:
- Baseline token counts
- Top 10 most impactful interventions
- Cases where reasoning was suppressed
- Configuration patterns

### Load Results in Python

```python
import json
import torch

# Load intervention results
with open("results/interventions_reasoning_20241129_143022.json") as f:
    data = json.load(f)
    results = data["results"]
    config = data["config"]

# Load activations
activations = torch.load("results/activations_20241129_142800.pt")
r_activations = activations["reasoning"]
nr_activations = activations["non_reasoning"]

# Load directions
directions = torch.load("results/directions_20241129_142900.pt")
direction_vectors = directions["directions"]
```

## Expected Outputs

### Activation Extraction

```
Extracting Activations
================================================================================
Loading data...
Extracting from non-reasoning model...
Extracting activations: 100%|████████████| 100/100
Extracting from reasoning model...
Extracting activations: 100%|████████████| 100/100
✓ Activations extracted successfully
✓ Activations saved to results/activations_YYYYMMDD_HHMMSS.pt
```

### Direction Calculation

```
Calculating Directions
================================================================================
✓ Calculated 27 direction vectors
✓ Directions saved to results/directions_YYYYMMDD_HHMMSS.pt
```

### Interventions

```
Running Interventions on reasoning model
================================================================================
Testing on 10 prompts
✓ Completed 360 intervention runs
✓ Results saved to results/interventions_reasoning_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### Out of Memory

If you run out of GPU memory:

```bash
# Reduce batch size by using fewer samples
uv run python run_experiment.py --num-samples 20

# Test fewer layers
uv run python run_experiment.py --layers 0 1 2

# Use CPU (slow but works)
uv run python run_experiment.py --device cpu
```

### Model Download Issues

Models are downloaded automatically from HuggingFace. If you have issues:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/drive/huggingface

# Login if using gated models
huggingface-cli login
```

### Import Errors

Make sure you've installed the package:

```bash
# Reinstall in development mode
uv sync

# Or
pip install -e .
```

## Next Steps

1. **Reproduce key findings**: Start with layer 0 interventions on the reasoning model
2. **Explore new layers**: Test which layers show the strongest effects
3. **Try different strengths**: Find the threshold where reasoning completely breaks down
4. **Test on non-reasoning model**: See if you can induce reasoning behavior
5. **Analyze patterns**: Use the analysis script to identify interesting configurations
6. **Extend the code**: Add new direction calculation methods or intervention strategies

## Performance Tips

- Use `--points pre` only for 3x speedup
- Reduce `--num-samples` for activation extraction (20-50 is often sufficient)
- Test on toy problems first before full GSM8K
- Use fewer layers initially, expand once you find interesting ones
- Save intermediate results (activations, directions) to avoid recomputation


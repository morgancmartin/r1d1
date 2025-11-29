# Reasoning Direction Experiments

This project reproduces experiments exploring the existence of a "reasoning direction" in the activation space of large language models, inspired by the methodology from "Refusal in Large Language Models as Mediated by a Single Direction" (Arditi et al.).

## Overview

The experiments investigate whether reasoning behavior in LLMs can be controlled by manipulating specific directions in activation space, similar to how refusal behavior was shown to be controllable. The key findings suggest that:

1. **Reasoning models** (like DeepSeek R1) show bidirectional control when interventions are applied - reasoning can be both enhanced (more verbose, uncertain) and suppressed (no `<think>` tags)
2. **Non-reasoning models** (like Llama-3 8B) don't readily adopt explicit reasoning behavior from these interventions
3. **Layer specificity** matters - early layers (0-3) show the most pronounced effects

## Models Tested

- **Non-reasoning**: Llama-3 8B Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`)
- **Reasoning**: DeepSeek R1 Distill Llama-8B (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)

## Installation

This project uses `uv` for package management. Install dependencies:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

Alternatively, with pip:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

Run the full experiment pipeline with default settings:

```bash
uv run python run_experiment.py
```

### Custom Configuration

```bash
# Use toy problems for quick testing
uv run python run_experiment.py --dataset toy

# Test specific layers and strengths
uv run python run_experiment.py \
    --layers 0 1 2 3 \
    --strengths -0.2 -0.1 0.1 0.2

# Test on both models
uv run python run_experiment.py --test-both

# Use specific activation points
uv run python run_experiment.py --points pre post

# Test different direction types
uv run python run_experiment.py \
    --direction-types difference random
```

## Project Structure

```
r1d1/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses
│   ├── model_loader.py        # Model loading utilities
│   ├── data_loader.py         # Dataset loading (GSM8K, toy problems)
│   ├── activation_extractor.py # Extract activations from models
│   ├── direction_calculator.py # Calculate reasoning directions
│   ├── intervention.py        # Apply interventions during generation
│   └── experiment.py          # Main experiment orchestration
├── run_experiment.py          # CLI entry point
├── pyproject.toml            # Project dependencies
└── README.md
```

## Methodology

### 1. Activation Extraction

Activations are extracted from the residual stream at three points in each layer:
- **Pre-attention** (`hook_resid_pre`): Before attention block
- **Mid-attention** (`hook_resid_mid`): After attention, before MLP
- **Post-attention** (`hook_resid_post`): After MLP

### 2. Direction Calculation

Four types of directions are computed:

1. **Difference**: `normalize(mean(reasoning_activations) - mean(non_reasoning_activations))`
2. **Original**: `normalize(mean(non_reasoning_activations))`
3. **Reasoning**: `normalize(mean(reasoning_activations))`
4. **Random**: Randomly generated vectors (control)

### 3. Intervention

During generation, directions are added to activations at specified layers:

```python
def reasoning_enhancement_hook(activation, hook):
    return activation + (strength * direction.unsqueeze(0).unsqueeze(0))
```

Parameters varied:
- **Layers**: 0-30 (default focuses on 0, 1, 2, 3, 10, 15, 20, 25, 30)
- **Strengths**: -0.1, -0.05, 0.05, 0.1 (negative suppresses, positive enhances)
- **Direction types**: difference, original, reasoning, random

## Expected Results

### On Reasoning Model

- **Positive strengths** (0.05-0.1): Enhanced reasoning with increased verbosity, hesitation, backtracking
- **Very high strengths** (0.1-0.2): Output degrades to repetitive `<think>` tokens
- **Negative strengths** (-0.1 to -0.05): Suppressed reasoning, may skip `<think>` sections entirely
- **Early layers** (0-3): Most pronounced effects
- **Later layers** (19+): Diminishing effects

### On Non-Reasoning Model

- Generally **maintains typical response patterns** regardless of intervention
- Suggests reasoning requires more than single-direction modification
- Base model may lack necessary architectural structures

## Command Line Options

```
--non-reasoning-model  HuggingFace model name for non-reasoning model
--reasoning-model      HuggingFace model name for reasoning model
--dataset             Dataset to use (gsm8k or toy)
--num-samples         Number of samples for activation extraction
--layers              Layer indices to analyze
--strengths           Intervention strengths to test
--points              Activation points (pre, mid, post)
--direction-types     Direction types to calculate
--max-tokens          Maximum tokens to generate
--temperature         Sampling temperature
--test-reasoning      Test interventions on reasoning model
--test-non-reasoning  Test interventions on non-reasoning model
--test-both          Test interventions on both models
--device             Device to run on (cuda or cpu)
```

## Output

Results are saved in the `results/` directory with timestamps:

- `activations_YYYYMMDD_HHMMSS.pt`: Extracted activations
- `directions_YYYYMMDD_HHMMSS.pt`: Calculated direction vectors
- `interventions_[model]_YYYYMMDD_HHMMSS.json`: Intervention results with generated outputs

## Hardware Requirements

- **GPU**: Recommended (models are 8B parameters)
- **RAM**: 32GB+ recommended
- **VRAM**: 24GB+ recommended for loading both models simultaneously
- **Storage**: ~50GB for model downloads

## Datasets

### GSM8K
Mathematical reasoning problems automatically loaded via HuggingFace `datasets` library.

### Toy Problems
10 simple arithmetic problems for quick testing:
- "What is 2+2?"
- "What is 5+7?"
- etc.

## Citation

This work is inspired by:
```
Arditi, A., et al. (2024). Refusal in Large Language Models as Mediated by a Single Direction.
```

## License

MIT

## Contributing

This is a research reproduction project. Feel free to extend the experiments or improve the code!


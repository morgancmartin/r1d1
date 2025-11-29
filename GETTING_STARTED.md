# Getting Started with Reasoning Direction Experiments

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /mnt/data/workspace/r1d1
uv sync
```

This installs all required packages including PyTorch, TransformerLens, and HuggingFace libraries.

### 2. Run Your First Experiment

```bash
# Quick test with toy problems (5-10 minutes)
uv run python examples/quick_test.py
```

This will:
- Load Llama-3 8B and DeepSeek R1 models
- Extract activations from 5 simple math problems
- Calculate reasoning directions
- Test interventions on layers 0, 1, 2
- Save results to `results/` directory

### 3. View Results

```bash
# List generated results
ls -lh results/

# Analyze the latest intervention results
uv run python examples/analyze_results.py results/interventions_reasoning_*.json
```

## 📚 What You Can Do

### Option A: Command Line (Recommended for batch experiments)

```bash
# Run full GSM8K experiment
uv run python run_experiment.py

# Customize parameters
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 50 \
    --layers 0 1 2 3 \
    --strengths -0.1 -0.05 0.05 0.1 \
    --test-reasoning

# See all options
uv run python run_experiment.py --help
```

### Option B: Interactive Notebook (Recommended for exploration)

```bash
# Launch Jupyter Lab
uv run jupyter lab

# Open: examples/interactive_demo.ipynb
```

The notebook walks through:
1. Loading both models
2. Extracting activations from toy problems
3. Calculating reasoning directions
4. Testing interventions with different strengths
5. Analyzing the results interactively

### Option C: Python API (Recommended for custom experiments)

```python
from src.config import ExperimentConfig, DataConfig
from src.experiment import ReasoningDirectionExperiment

# Create custom configuration
exp_config = ExperimentConfig(
    layers=[0, 1, 2],
    strengths=[-0.1, 0.1],
    activation_points=["pre"],
)

data_config = DataConfig(
    use_toy_problems=True,
    num_samples=5,
)

# Run experiment
experiment = ReasoningDirectionExperiment(exp_config)
experiment.run_full_experiment(
    data_config=data_config,
    test_on_reasoning_model=True,
)
```

## 🎯 Key Concepts

### 1. Models
- **Non-reasoning**: Llama-3 8B Instruct (baseline)
- **Reasoning**: DeepSeek R1 Distill Llama-8B (uses `<think>` tags)

### 2. Activation Points
- **Pre**: Before attention block (purest residual stream)
- **Mid**: After attention, before MLP
- **Post**: After full transformer block

### 3. Direction Types
- **Difference**: reasoning - non_reasoning (main direction)
- **Reasoning**: Reasoning model activations only
- **Original**: Non-reasoning model activations only
- **Random**: Random vectors (control)

### 4. Intervention Strengths
- **Negative** (e.g., -0.1): Suppress reasoning, reduce verbosity
- **Zero**: No intervention (baseline)
- **Positive** (e.g., +0.1): Enhance reasoning, increase verbosity

### 5. Layers
- **Early** (0-3): Strongest effects observed
- **Middle** (10-15): Moderate effects
- **Late** (20-30): Diminishing effects

## 🔍 What to Look For

### Expected Results on Reasoning Model

**Baseline (no intervention)**:
```
<think>
Let me solve 2+2.
2 + 2 = 4
</think>
The answer is 4.
```

**Enhanced reasoning (strength +0.1, layer 0)**:
```
<think>
I need to solve 2+2. Let me think carefully about this.
First, I have 2. Then I'm adding 2 more.
Actually, let me reconsider...
2 + 2 = 4. Yes, I'm confident.
Or wait, should I double-check? ...
</think>
The answer is 4.
```
→ Much longer, more hesitant, backtracking

**Suppressed reasoning (strength -0.1, layer 0)**:
```
The answer is 4.
```
→ No `<think>` section at all!

## 📊 Example Workflows

### Reproduce Key Paper Finding
Focus on layer 0 with various strengths:

```bash
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 100 \
    --layers 0 \
    --strengths -0.2 -0.1 -0.05 0.05 0.1 0.2 \
    --test-reasoning
```

### Find Best Layer for Control
Sweep across many layers:

```bash
uv run python run_experiment.py \
    --layers 0 1 2 3 5 7 10 12 15 17 20 22 25 27 30 \
    --strengths -0.1 0.1
```

### Compare Direction Types
Test different direction calculation methods:

```bash
uv run python run_experiment.py \
    --direction-types difference reasoning original random \
    --layers 0 1 2
```

### Test Both Models
See if non-reasoning model can be "taught" to reason:

```bash
uv run python run_experiment.py \
    --test-both \
    --layers 0 1 2 3
```

## 🐛 Troubleshooting

### Out of GPU Memory
```bash
# Use fewer samples
uv run python run_experiment.py --num-samples 20

# Test fewer layers
uv run python run_experiment.py --layers 0 1 2

# Use CPU (slow but works)
uv run python run_experiment.py --device cpu
```

### Models Not Downloading
```bash
# Check internet connection
ping huggingface.co

# Set larger cache
export HF_HOME=/path/to/large/storage

# Login if needed (for gated models)
huggingface-cli login
```

### Import Errors
```bash
# Reinstall package
uv sync --reinstall

# Or use pip
pip install -e .
```

## 📖 Documentation

- **README.md**: Project background and methodology
- **USAGE.md**: Comprehensive usage guide with all options
- **PROJECT_OVERVIEW.md**: Architecture and design decisions
- **GETTING_STARTED.md** (this file): Quick start guide

## 💡 Tips

1. **Start small**: Use toy problems and 2-3 layers first
2. **Layer 0 is special**: Most pronounced effects are here
3. **Save intermediates**: Activations and directions are reusable
4. **Iterate**: Find interesting layers, then sweep strengths
5. **Compare**: Always include baseline (no intervention) results

## 🎓 Learning Path

1. **Day 1**: Run quick_test.py, understand the pipeline
2. **Day 2**: Work through interactive_demo.ipynb
3. **Day 3**: Run full GSM8K experiment, analyze results
4. **Day 4**: Focus experiments on interesting findings
5. **Day 5**: Modify code for custom experiments

## 📈 Next Steps

Once you've reproduced the basic results:

1. **Analyze patterns**: Which layers show strongest effects?
2. **Test thresholds**: At what strength does reasoning break down?
3. **Compare directions**: Is difference better than reasoning-only?
4. **Extend methodology**: Can you find better directions?
5. **Try new tasks**: Test on code generation, logic problems

## 🤝 Need Help?

- Check USAGE.md for detailed command line options
- Check PROJECT_OVERVIEW.md for architecture details
- Review the notebook for step-by-step examples
- Read the source code docstrings for API documentation

## ✅ Verification

After installation, verify everything works:

```bash
# Check Python version (should be 3.10+)
uv run python --version

# Check package installation
uv run python -c "from src.experiment import ReasoningDirectionExperiment; print('OK')"

# Check GPU (if available)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎉 You're Ready!

Your reasoning direction experiment framework is ready to use. Start with `quick_test.py` and explore from there!

```bash
uv run python examples/quick_test.py
```

Happy experimenting! 🔬


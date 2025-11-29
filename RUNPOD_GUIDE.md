# RunPod Setup Guide for Reasoning Direction Experiments

## Why RunPod?

✅ **Cost-effective**: $0.30-1.20/hour vs AWS $3-8/hour  
✅ **Flexible**: Pay-per-second, no commitments  
✅ **Easy**: Pre-configured PyTorch templates  
✅ **Powerful**: Access to A40, A6000, H100 GPUs

## Step-by-Step Setup

### 1. Choose Your GPU

**Budget Option** ($0.30-0.50/hr):
- RTX 3090 (24GB) - Tight but works
- RTX 4090 (24GB) - Better headroom

**Recommended** ($0.60-0.80/hr):
- A40 (48GB) - Plenty of VRAM, can load both models comfortably
- A5000 (24GB) - Good balance

**Premium** ($0.80-1.20/hr):
- A6000 (48GB) - Fastest, most VRAM
- A100 (40/80GB) - Overkill but fast

💡 **Pro tip**: Use **Secure Cloud** with **On-Demand** for reliability, or **Community Cloud** with **Spot** for 50-70% savings (may be interrupted).

### 2. Create RunPod Instance

1. Go to [runpod.io](https://runpod.io) and sign up
2. Click "Deploy" → "GPU Instance"
3. Choose template: **PyTorch 2.x** (has CUDA pre-installed)
4. Select GPU based on budget above
5. Set disk space: **50GB+** (for model downloads)
6. Deploy!

### 3. Upload Your Code

**Option A: Direct Upload** (easiest)
```bash
# On your local machine, create a tarball
cd /mnt/data/workspace
tar -czf r1d1.tar.gz r1d1/

# In RunPod web terminal, upload the file (use web interface)
# Then extract:
cd /workspace
tar -xzf r1d1.tar.gz
cd r1d1
```

**Option B: Git** (if you have a repo)
```bash
# In RunPod terminal
cd /workspace
git clone <your-repo-url> r1d1
cd r1d1
```

**Option C: RunPod Volume** (for repeated use)
- Create a persistent volume
- Upload once, reuse across pods

### 4. Run Setup Script

```bash
# Make executable and run
chmod +x runpod_setup.sh
./runpod_setup.sh
```

This will:
- Install uv package manager
- Install all Python dependencies
- Verify GPU access
- Create results directory

### 5. Run Experiments

```bash
# Quick test (5-10 min, $0.05-0.10)
uv run python examples/quick_test.py

# Standard experiment (30-60 min, $0.30-1.00)
uv run python run_experiment.py

# Layer 0 focus (20-30 min, $0.20-0.60)
uv run python run_experiment.py \
    --layers 0 \
    --strengths -0.2 -0.15 -0.1 -0.05 0.05 0.1 0.15 0.2
```

### 6. Download Results

**Option A: Web Interface**
- Navigate to `results/` in RunPod file browser
- Download files individually

**Option B: SCP** (faster for many files)
```bash
# On your local machine
scp -P <runpod-ssh-port> root@<runpod-ip>:/workspace/r1d1/results/* ./results/
```

**Option C: Tar and Download**
```bash
# In RunPod terminal
cd /workspace/r1d1
tar -czf results.tar.gz results/

# Then download via web interface
```

### 7. Stop Pod

**IMPORTANT**: Don't forget to stop your pod when done! RunPod charges by the second.

## 💰 Cost Optimization Tips

### 1. Use Spot Instances
Save 50-70% with spot instances (Community Cloud). Risk: may be interrupted.

```bash
# Always save intermediate results
uv run python run_experiment.py  # Auto-saves activations, directions
```

### 2. Sequential Model Loading
If VRAM is tight, modify code to load models one at a time:

**Create** `examples/memory_efficient_test.py`:
```python
from src.config import ExperimentConfig, DataConfig
from src.model_loader import ModelLoader
from src.data_loader import DataLoader
from src.activation_extractor import ActivationExtractor
from src.direction_calculator import DirectionCalculator
import torch

# Config
exp_config = ExperimentConfig(layers=[0, 1, 2], strengths=[-0.1, 0.1])
data_config = DataConfig(use_toy_problems=True, num_samples=5)

# Load and extract from non-reasoning model
print("=== Non-Reasoning Model ===")
loader = ModelLoader()
nr_model = loader.load_model("llama3-8b", device="cuda")
data_loader = DataLoader(data_config)
nr_problems = data_loader.load_data(nr_model)
nr_extractor = ActivationExtractor(nr_model)
nr_activations = nr_extractor.extract_activations(
    [p["prompt"] for p in nr_problems], 
    exp_config.layers, 
    ["pre"]
)
# Free memory
del nr_model
torch.cuda.empty_cache()

# Load and extract from reasoning model
print("=== Reasoning Model ===")
r_model = loader.load_model("deepseek-r1-llama-8b", device="cuda")
r_problems = data_loader.load_data(r_model)
r_extractor = ActivationExtractor(r_model)
r_activations = r_extractor.extract_activations(
    [p["prompt"] for p in r_problems],
    exp_config.layers,
    ["pre"]
)

# Calculate directions (both models unloaded now for extraction)
calculator = DirectionCalculator(device="cuda")
directions = calculator.calculate_all_directions(
    r_activations, nr_activations, ["difference"], normalize=True
)

# Run interventions (only reasoning model needed)
from src.intervention import InterventionEngine
engine = InterventionEngine(r_model)
results = engine.run_intervention_sweep(
    prompts=[p["prompt"] for p in r_problems],
    directions=directions,
    layers=exp_config.layers,
    points=["pre"],
    direction_types=["difference"],
    strengths=exp_config.strengths,
    include_baseline=True,
)

print(f"Completed {len(results)} interventions")
```

Run with:
```bash
uv run python examples/memory_efficient_test.py
```

This uses ~12-14GB instead of ~22GB!

### 3. Batch Your Experiments

Instead of multiple short sessions, plan one longer session:

```bash
# Run multiple experiments in sequence
uv run python run_experiment.py --layers 0 1 2 --dataset toy
uv run python run_experiment.py --layers 0 --strengths -0.2 -0.1 0.1 0.2
uv run python run_experiment.py --layers 3 5 7 --strengths -0.1 0.1
# Total: ~2 hours, but only pay for 2 hours vs 3 separate sessions
```

### 4. Use Toy Problems First

GSM8K downloads and processes 100 samples. Start with toy problems:

```bash
# Toy problems (fast, $0.05-0.10)
uv run python run_experiment.py --dataset toy --layers 0 1 2

# If results look good, then run full GSM8K
uv run python run_experiment.py --dataset gsm8k --num-samples 100
```

## 🔧 Troubleshooting

### Out of VRAM Error

```bash
# Solution 1: Use memory-efficient script (see above)
uv run python examples/memory_efficient_test.py

# Solution 2: Reduce samples
uv run python run_experiment.py --num-samples 20

# Solution 3: Fewer layers at once
uv run python run_experiment.py --layers 0 1 2
```

### Models Download Slowly

RunPod has fast internet, but HuggingFace can be slow. First run will take 10-15 minutes to download models (~16GB total).

```bash
# Pre-download models
uv run python -c "from src.model_loader import ModelLoader; l=ModelLoader(); l.load_model('llama3-8b'); l.load_model('deepseek-r1-llama-8b')"
```

### Connection Drops

Use `tmux` or `screen` so experiments continue if you disconnect:

```bash
# Install tmux (usually pre-installed)
apt-get update && apt-get install -y tmux

# Start session
tmux new -s experiment

# Run your experiment
uv run python run_experiment.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t experiment
```

## 📊 Example Session Timeline

**Total Time**: ~1.5 hours  
**Estimated Cost**: $0.45-1.80 (depending on GPU)

1. **Setup** (10 min): Upload code, install dependencies
2. **Test** (5 min): Run quick_test.py to verify
3. **Experiment 1** (30 min): Layer 0 sweep
4. **Experiment 2** (30 min): Multi-layer comparison
5. **Experiment 3** (15 min): Direction type comparison
6. **Download** (10 min): Package and download results

## 🎯 Recommended First Session

Try this for your first RunPod session:

```bash
# 1. Setup (included in script)
./runpod_setup.sh

# 2. Quick verification
uv run python examples/quick_test.py

# 3. Key finding reproduction (layer 0)
uv run python run_experiment.py \
    --dataset gsm8k \
    --num-samples 50 \
    --layers 0 \
    --strengths -0.15 -0.1 -0.05 0.05 0.1 0.15 \
    --test-reasoning

# 4. Download results
cd /workspace/r1d1
tar -czf results_$(date +%Y%m%d).tar.gz results/
# Download via web interface

# 5. Stop pod!
```

**Total time**: ~45 minutes  
**Total cost**: $0.23-0.90

## 🔄 Alternative: Vast.ai

Similar to RunPod, often cheaper:
- Even lower prices (Community GPUs)
- More variability in reliability
- Similar setup process

## 🏠 Alternative: Local + Smaller Models

If you want to experiment locally first, consider:
- Using smaller models (1.5B, 3B versions)
- Running inference on CPU (very slow but free)
- Google Colab Pro ($10/month, T4 GPU)

## ✅ Pre-Flight Checklist

Before starting RunPod session:
- [ ] Code is ready and tested locally (if possible)
- [ ] You know which experiments to run
- [ ] You have a download plan for results
- [ ] You've budgeted time and cost
- [ ] You have the setup script ready

## 📞 Support

If issues arise:
- RunPod Discord: Very active community
- RunPod docs: docs.runpod.io
- This project: Check USAGE.md and GETTING_STARTED.md

---

**Bottom line**: RunPod is perfect for this project. Budget $0.50-2.00 for a productive session. Use spot instances for savings, and don't forget to stop your pod! 🚀


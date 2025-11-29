#!/usr/bin/env python3
"""Memory-efficient experiment that loads models sequentially.

This script is useful when running on GPUs with limited VRAM (e.g., 16-24GB).
It loads models one at a time instead of simultaneously, reducing peak memory usage
from ~22GB to ~14GB.
"""

import torch
from pathlib import Path

from src.config import ExperimentConfig, DataConfig
from src.model_loader import ModelLoader
from src.data_loader import DataLoader
from src.activation_extractor import ActivationExtractor
from src.direction_calculator import DirectionCalculator
from src.intervention import InterventionEngine


def main():
    """Run memory-efficient experiment."""
    
    print("=" * 80)
    print("MEMORY-EFFICIENT REASONING DIRECTION EXPERIMENT")
    print("=" * 80)
    print("\nThis version loads models sequentially to reduce VRAM usage.")
    print("Peak VRAM: ~14GB (vs ~22GB for standard version)")
    print()
    
    # Configuration
    exp_config = ExperimentConfig(
        layers=[0, 1, 2],  # Just a few layers
        strengths=[-0.1, -0.05, 0.05, 0.1],
        activation_points=["pre"],  # Just pre-attention
        direction_types=["difference"],
    )
    
    data_config = DataConfig(
        use_toy_problems=True,  # Fast testing
        num_samples=5,
    )
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Phase 1: Extract from non-reasoning model
    print("PHASE 1: Non-Reasoning Model (Llama-3 8B)")
    print("=" * 80)
    
    loader = ModelLoader()
    print("\nLoading non-reasoning model...")
    nr_model = loader.load_model("llama3-8b", device="cuda")
    
    print("Loading and formatting data...")
    data_loader = DataLoader(data_config)
    nr_problems = data_loader.load_data(nr_model)
    nr_prompts = [p["prompt"] for p in nr_problems]
    
    print("Extracting activations...")
    nr_extractor = ActivationExtractor(nr_model)
    nr_activations = nr_extractor.extract_activations(
        prompts=nr_prompts,
        layers=exp_config.layers,
        points=exp_config.activation_points,
    )
    
    print(f"✓ Extracted activations from {len(nr_prompts)} prompts")
    
    # Free GPU memory
    print("\nFreeing GPU memory...")
    del nr_model
    del nr_extractor
    torch.cuda.empty_cache()
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after cleanup: {allocated:.2f} GB")
    
    # Phase 2: Extract from reasoning model
    print("\n" + "=" * 80)
    print("PHASE 2: Reasoning Model (DeepSeek R1)")
    print("=" * 80)
    
    print("\nLoading reasoning model...")
    r_model = loader.load_model("deepseek-r1-llama-8b", device="cuda")
    
    print("Loading and formatting data...")
    r_problems = data_loader.load_data(r_model)
    r_prompts = [p["prompt"] for p in r_problems]
    
    print("Extracting activations...")
    r_extractor = ActivationExtractor(r_model)
    r_activations = r_extractor.extract_activations(
        prompts=r_prompts,
        layers=exp_config.layers,
        points=exp_config.activation_points,
    )
    
    print(f"✓ Extracted activations from {len(r_prompts)} prompts")
    
    # Phase 3: Calculate directions
    print("\n" + "=" * 80)
    print("PHASE 3: Calculate Directions")
    print("=" * 80)
    
    calculator = DirectionCalculator(device="cuda")
    directions = calculator.calculate_all_directions(
        reasoning_activations=r_activations,
        non_reasoning_activations=nr_activations,
        direction_types=exp_config.direction_types,
        normalize=True,
    )
    
    print(f"✓ Calculated {len(directions)} direction vectors")
    
    # Phase 4: Run interventions (reasoning model still loaded)
    print("\n" + "=" * 80)
    print("PHASE 4: Run Interventions")
    print("=" * 80)
    
    engine = InterventionEngine(r_model)
    
    results = engine.run_intervention_sweep(
        prompts=r_prompts,
        directions=directions,
        layers=exp_config.layers,
        points=exp_config.activation_points,
        direction_types=exp_config.direction_types,
        strengths=exp_config.strengths,
        max_new_tokens=256,
        temperature=0.7,
        include_baseline=True,
    )
    
    print(f"\n✓ Completed {len(results)} intervention runs")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = results_dir / f"interventions_memory_efficient_{timestamp}.json"
    
    results_dict = []
    for r in results:
        results_dict.append({
            "prompt": r.prompt,
            "layer": r.layer,
            "point": r.point,
            "direction_type": r.direction_type,
            "strength": r.strength,
            "output": r.output,
            "num_tokens": r.num_tokens,
        })
    
    with open(save_path, "w") as f:
        json.dump({
            "results": results_dict,
            "config": {
                "layers": exp_config.layers,
                "strengths": exp_config.strengths,
                "direction_types": exp_config.direction_types,
                "activation_points": exp_config.activation_points,
            }
        }, f, indent=2)
    
    print(f"✓ Results saved to {save_path}")
    
    # Memory stats
    print("\n" + "=" * 80)
    print("Memory Statistics")
    print("=" * 80)
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Current GPU memory allocated: {allocated:.2f} GB")
    print(f"Current GPU memory reserved: {reserved:.2f} GB")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {results_dir}")
    print("\nAnalyze with:")
    print(f"  python examples/analyze_results.py {save_path}")


if __name__ == "__main__":
    main()


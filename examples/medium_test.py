#!/usr/bin/env python3
"""Medium test - GSM8K problems but optimized for speed.

This gives better signal than toy problems but runs faster than full experiment.
Should take ~20-30 minutes instead of 1-2 hours.
"""

from src.config import ExperimentConfig, DataConfig
from src.experiment import ReasoningDirectionExperiment


def main():
    """Run a medium-sized test with GSM8K."""
    
    # Optimized config for good signal + reasonable time
    exp_config = ExperimentConfig(
        layers=[0],  # Just layer 0 (most important from paper)
        strengths=[-0.15, -0.1, 0.1, 0.15],  # Test a range including stronger values
        activation_points=["pre"],  # Just pre-attention
        direction_types=["difference"],  # Just the main direction
        max_new_tokens=256,  # Reasonable length
    )
    
    data_config = DataConfig(
        use_toy_problems=False,  # Use real GSM8K!
        num_samples=25,  # 25 problems for extraction (sweet spot)
        dataset_name="gsm8k",
    )
    
    print("\n" + "="*80)
    print("MEDIUM TEST: GSM8K problems, Layer 0 focus")
    print("="*80)
    print(f"- Extracting from {data_config.num_samples} GSM8K problems")
    print(f"- Testing layer {exp_config.layers} only")
    print(f"- Strengths: {exp_config.strengths}")
    print(f"- Expected runtime: ~20-30 minutes")
    print("="*80 + "\n")
    
    # Run experiment
    experiment = ReasoningDirectionExperiment(exp_config)
    experiment.run_full_experiment(
        data_config=data_config,
        test_on_reasoning_model=True,
        test_on_non_reasoning_model=False,
    )
    
    print("\n" + "="*80)
    print("DONE! Check results/ directory")
    print("Analyze with:")
    print("  python examples/analyze_results.py results/interventions_reasoning_*.json")
    print("="*80)


if __name__ == "__main__":
    main()


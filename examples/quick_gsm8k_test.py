#!/usr/bin/env python3
"""Quick GSM8K test - fast but with real problems.

Uses actual GSM8K problems but minimal samples for speed.
Should take ~10-15 minutes.
"""

from src.config import ExperimentConfig, DataConfig
from src.experiment import ReasoningDirectionExperiment


def main():
    """Run a quick test with GSM8K problems."""
    
    # Minimal config but with real problems
    exp_config = ExperimentConfig(
        layers=[0],  # Just layer 0 (strongest effects)
        strengths=[-0.1, 0.1],  # Just two strengths (suppress vs enhance)
        activation_points=["pre"],  # Just pre-attention
        direction_types=["difference"],  # Just the main direction
        max_new_tokens=512,  # Enough for complex reasoning + answer
    )
    
    data_config = DataConfig(
        use_toy_problems=False,  # Use real GSM8K!
        num_samples=10,  # Small but enough for signal
        dataset_name="gsm8k",
    )
    
    # Note: DataLoader will use "complex" sampling by default
    # This selects harder problems from the top 50% by question length
    
    print("\n" + "="*80)
    print("QUICK GSM8K TEST: Real problems, fast runtime")
    print("="*80)
    print(f"- Extracting from {data_config.num_samples} GSM8K problems")
    print(f"- Testing layer {exp_config.layers} only")
    print(f"- Strengths: {exp_config.strengths}")
    print(f"- Expected runtime: ~10-15 minutes")
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
    print("="*80)
    print("\nQuick check for effects:")
    print("  # Look for token count differences")
    print("  cat results/interventions_reasoning_*.json | grep -E '(num_tokens|strength)' | tail -50")


if __name__ == "__main__":
    main()


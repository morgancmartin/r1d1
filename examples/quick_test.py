#!/usr/bin/env python3
"""Quick test script for rapid iteration."""

from src.config import ExperimentConfig, DataConfig
from src.experiment import ReasoningDirectionExperiment


def main():
    """Run a quick test with minimal parameters."""
    
    # Use toy problems and minimal layers for fast testing
    exp_config = ExperimentConfig(
        layers=[0, 1, 2],  # Just test a few early layers
        strengths=[-0.1, 0.1],  # Two strengths
        activation_points=["pre"],  # Just pre-attention
        direction_types=["difference"],  # Just the main direction type
        max_new_tokens=256,  # Shorter generations
    )
    
    data_config = DataConfig(
        use_toy_problems=True,  # Use simple toy problems
        num_samples=5,  # Just 5 samples for activation extraction
    )
    
    # Run experiment
    experiment = ReasoningDirectionExperiment(exp_config)
    experiment.run_full_experiment(
        data_config=data_config,
        test_on_reasoning_model=True,
        test_on_non_reasoning_model=False,  # Skip non-reasoning to save time
    )


if __name__ == "__main__":
    main()


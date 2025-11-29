#!/usr/bin/env python3
"""Main script to run reasoning direction experiments."""

import argparse
from src.config import ExperimentConfig, DataConfig
from src.experiment import ReasoningDirectionExperiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run reasoning direction experiments on LLMs"
    )
    
    # Model selection
    parser.add_argument(
        "--non-reasoning-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name for non-reasoning model"
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model name for reasoning model"
    )
    
    # Data selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "toy"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to extract activations from"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 10, 15, 20, 25, 30],
        help="Layers to analyze"
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[-0.1, -0.05, 0.05, 0.1],
        help="Intervention strengths to test"
    )
    parser.add_argument(
        "--points",
        type=str,
        nargs="+",
        default=["pre", "mid", "post"],
        choices=["pre", "mid", "post"],
        help="Activation points to hook"
    )
    parser.add_argument(
        "--direction-types",
        type=str,
        nargs="+",
        default=["difference", "original", "reasoning"],
        choices=["difference", "original", "reasoning", "random"],
        help="Types of directions to calculate"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    # Target models for intervention
    parser.add_argument(
        "--test-reasoning",
        action="store_true",
        default=True,
        help="Test interventions on reasoning model"
    )
    parser.add_argument(
        "--test-non-reasoning",
        action="store_true",
        default=False,
        help="Test interventions on non-reasoning model"
    )
    parser.add_argument(
        "--test-both",
        action="store_true",
        help="Test interventions on both models"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    
    return parser.parse_args()


def main():
    """Main experiment runner."""
    args = parse_args()
    
    # Create experiment config
    exp_config = ExperimentConfig(
        non_reasoning_model=args.non_reasoning_model,
        reasoning_model=args.reasoning_model,
        activation_points=args.points,
        layers=args.layers,
        strengths=args.strengths,
        direction_types=args.direction_types,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    # Create data config
    data_config = DataConfig(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        use_toy_problems=(args.dataset == "toy"),
    )
    
    # Determine which models to test
    test_reasoning = args.test_reasoning or args.test_both
    test_non_reasoning = args.test_non_reasoning or args.test_both
    
    # Run experiment
    experiment = ReasoningDirectionExperiment(exp_config)
    experiment.run_full_experiment(
        data_config=data_config,
        test_on_reasoning_model=test_reasoning,
        test_on_non_reasoning_model=test_non_reasoning,
    )


if __name__ == "__main__":
    main()


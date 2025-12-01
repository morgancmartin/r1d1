"""Main experiment orchestration."""

import torch
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from tqdm import tqdm

from .config import ExperimentConfig, DataConfig, MODELS
from .model_loader import ModelLoader
from .data_loader import DataLoader
from .activation_extractor import ActivationExtractor
from .direction_calculator import DirectionCalculator
from .intervention import InterventionEngine, InterventionResult


class ReasoningDirectionExperiment:
    """Main experiment class for reasoning direction research."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components (lazy loading)
        self.reasoning_model = None
        self.non_reasoning_model = None
        self.data_loader = None
        
    def load_models(self):
        """Load both reasoning and non-reasoning models."""
        print("=" * 80)
        print("Loading Models")
        print("=" * 80)
        
        loader = ModelLoader()
        
        print("\n1. Loading non-reasoning model...")
        self.non_reasoning_model = loader.load_model(
            self.config.non_reasoning_model,
            device=self.config.device,
        )
        
        print("\n2. Loading reasoning model...")
        self.reasoning_model = loader.load_model(
            self.config.reasoning_model,
            device=self.config.device,
        )
        
        print("\n✓ Models loaded successfully")
    
    def extract_activations(
        self,
        data_config: DataConfig,
        save: bool = True,
    ):
        """Extract activations from both models on the dataset.
        
        Args:
            data_config: Data configuration
            save: Whether to save activations to disk
            
        Returns:
            Tuple of (reasoning_activations, non_reasoning_activations)
        """
        print("\n" + "=" * 80)
        print("Extracting Activations")
        print("=" * 80)
        
        # Load data
        self.data_loader = DataLoader(data_config)
        
        print("\nLoading data for non-reasoning model...")
        nr_problems = self.data_loader.load_data(self.non_reasoning_model)
        nr_prompts = [p["prompt"] for p in nr_problems]
        
        print("Loading data for reasoning model...")
        r_problems = self.data_loader.load_data(self.reasoning_model)
        r_prompts = [p["prompt"] for p in r_problems]
        
        # Extract from non-reasoning model
        print("\nExtracting from non-reasoning model...")
        nr_extractor = ActivationExtractor(self.non_reasoning_model)
        nr_activations = nr_extractor.extract_activations(
            prompts=nr_prompts,
            layers=self.config.layers,
            points=self.config.activation_points,
        )
        
        # Extract from reasoning model
        print("\nExtracting from reasoning model...")
        r_extractor = ActivationExtractor(self.reasoning_model)
        r_activations = r_extractor.extract_activations(
            prompts=r_prompts,
            layers=self.config.layers,
            points=self.config.activation_points,
        )
        
        print("\n✓ Activations extracted successfully")
        
        # Optionally save
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.results_dir / f"activations_{timestamp}.pt"
            torch.save({
                "reasoning": r_activations,
                "non_reasoning": nr_activations,
                "config": self.config,
            }, save_path)
            print(f"✓ Activations saved to {save_path}")
        
        return r_activations, nr_activations
    
    def calculate_directions(
        self,
        reasoning_activations,
        non_reasoning_activations,
        save: bool = True,
    ):
        """Calculate reasoning directions from activations.
        
        Args:
            reasoning_activations: Activations from reasoning model
            non_reasoning_activations: Activations from non-reasoning model
            save: Whether to save directions to disk
            
        Returns:
            Dictionary of direction vectors
        """
        print("\n" + "=" * 80)
        print("Calculating Directions")
        print("=" * 80)
        
        calculator = DirectionCalculator(device=self.config.device)
        
        directions = calculator.calculate_all_directions(
            reasoning_activations=reasoning_activations,
            non_reasoning_activations=non_reasoning_activations,
            direction_types=self.config.direction_types,
            normalize=True,
        )
        
        print(f"\n✓ Calculated {len(directions)} direction vectors")
        
        # Optionally save
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.results_dir / f"directions_{timestamp}.pt"
            torch.save({
                "directions": directions,
                "config": self.config,
            }, save_path)
            print(f"✓ Directions saved to {save_path}")
        
        return directions
    
    def run_interventions(
        self,
        directions,
        test_data_config: Optional[DataConfig] = None,
        target_model: str = "reasoning",
        save: bool = True,
    ):
        """Run intervention experiments.
        
        Args:
            directions: Dictionary of direction vectors
            test_data_config: Data config for test set (if None, uses toy problems)
            target_model: Which model to intervene on ("reasoning" or "non_reasoning")
            save: Whether to save results to disk
            
        Returns:
            List of intervention results
        """
        print("\n" + "=" * 80)
        print(f"Running Interventions on {target_model} model")
        print("=" * 80)
        
        # Select target model
        if target_model == "reasoning":
            model = self.reasoning_model
        else:
            model = self.non_reasoning_model
        
        # Load test data
        # Default to toy problems if not specified (backward compatibility)
        if test_data_config is None:
            test_data_config = DataConfig(use_toy_problems=True)
        
        data_loader = DataLoader(test_data_config)
        test_problems = data_loader.load_data(model)
        test_prompts = [p["prompt"] for p in test_problems]
        
        print(f"\nTesting on {len(test_prompts)} prompts")
        
        # Run interventions
        engine = InterventionEngine(model)
        
        results = engine.run_intervention_sweep(
            prompts=test_prompts,
            directions=directions,
            layers=self.config.layers,
            points=self.config.activation_points,
            direction_types=self.config.direction_types,
            strengths=self.config.strengths,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            include_baseline=True,
        )
        
        print(f"\n✓ Completed {len(results)} intervention runs")
        
        # Save results
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.results_dir / f"interventions_{target_model}_{timestamp}.json"
            
            # Convert results to JSON-serializable format
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
                        "layers": self.config.layers,
                        "strengths": self.config.strengths,
                        "direction_types": self.config.direction_types,
                        "activation_points": self.config.activation_points,
                    }
                }, f, indent=2)
            
            print(f"✓ Results saved to {save_path}")
        
        return results
    
    def run_full_experiment(
        self,
        data_config: DataConfig,
        test_on_reasoning_model: bool = True,
        test_on_non_reasoning_model: bool = True,
    ):
        """Run the complete experiment pipeline.
        
        Args:
            data_config: Configuration for data loading
            test_on_reasoning_model: Whether to test interventions on reasoning model
            test_on_non_reasoning_model: Whether to test on non-reasoning model
        """
        print("\n" + "=" * 80)
        print("REASONING DIRECTION EXPERIMENT")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load models
        self.load_models()
        
        # Step 2: Extract activations
        r_activations, nr_activations = self.extract_activations(data_config)
        
        # Step 3: Calculate directions
        directions = self.calculate_directions(r_activations, nr_activations)
        
        # Step 4: Run interventions
        # Use the same data_config for testing as we used for extraction
        if test_on_reasoning_model:
            print("\n" + "=" * 80)
            print("Testing on Reasoning Model")
            print("=" * 80)
            self.run_interventions(directions, test_data_config=data_config, target_model="reasoning")
        
        if test_on_non_reasoning_model:
            print("\n" + "=" * 80)
            print("Testing on Non-Reasoning Model")
            print("=" * 80)
            self.run_interventions(directions, test_data_config=data_config, target_model="non_reasoning")
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved in: {self.results_dir}")


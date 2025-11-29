"""Intervention utilities for modifying model activations."""

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .activation_extractor import ActivationPoint
from .direction_calculator import DirectionType


@dataclass
class InterventionResult:
    """Result of an intervention experiment."""
    
    prompt: str
    layer: int
    point: ActivationPoint
    direction_type: DirectionType
    strength: float
    output: str
    num_tokens: int


class InterventionEngine:
    """Handles interventions on model activations."""
    
    def __init__(self, model: HookedTransformer):
        """Initialize the intervention engine.
        
        Args:
            model: The HookedTransformer model to intervene on
        """
        self.model = model
    
    def get_hook_name(self, layer: int, point: ActivationPoint) -> str:
        """Get the hook name for a specific layer and activation point.
        
        Args:
            layer: Layer index
            point: Activation point (pre/mid/post)
            
        Returns:
            Hook name string
        """
        if point == "pre":
            return f"blocks.{layer}.hook_resid_pre"
        elif point == "mid":
            return f"blocks.{layer}.hook_resid_mid"
        elif point == "post":
            return f"blocks.{layer}.hook_resid_post"
        else:
            raise ValueError(f"Unknown activation point: {point}")
    
    def create_enhancement_hook(
        self,
        direction: Tensor,
        strength: float = 1.0,
    ):
        """Create a hook function that adds a direction to activations.
        
        Args:
            direction: Direction vector to add (d_model,)
            strength: Strength multiplier for the direction
            
        Returns:
            Hook function
        """
        def reasoning_enhancement_hook(
            activation: Tensor,
            hook: HookPoint,
        ):
            """Hook that adds direction * strength to activations.
            
            Args:
                activation: Current activation (batch, seq_len, d_model)
                hook: Hook point
                
            Returns:
                Modified activation
            """
            # Move direction to same device as activation
            dir_device = direction.to(activation.device)
            
            # Add direction (broadcasting across batch and sequence)
            # direction shape: (d_model,) -> (1, 1, d_model)
            modified = activation + (strength * dir_device.unsqueeze(0).unsqueeze(0))
            
            return modified
        
        return reasoning_enhancement_hook
    
    def generate_with_intervention(
        self,
        prompt: str,
        layer: int,
        point: ActivationPoint,
        direction: Tensor,
        strength: float,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **generation_kwargs,
    ) -> Tuple[str, int]:
        """Generate text with an intervention applied.
        
        Args:
            prompt: Input prompt
            layer: Layer to intervene on
            point: Activation point to intervene at
            direction: Direction vector to add
            strength: Strength of intervention
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Tuple of (generated_text, num_tokens)
        """
        # Tokenize input
        tokens = self.model.to_tokens(prompt)
        
        # Create hook
        hook_fn = self.create_enhancement_hook(direction, strength)
        hook_name = self.get_hook_name(layer, point)
        
        # Generate with hook
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                fwd_hooks=[(hook_name, hook_fn)],
                **generation_kwargs,
            )
        
        # Decode output
        generated_text = self.model.to_string(output[0])
        num_tokens = output.shape[1]
        
        return generated_text, num_tokens
    
    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **generation_kwargs,
    ) -> Tuple[str, int]:
        """Generate text without intervention (baseline).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Tuple of (generated_text, num_tokens)
        """
        tokens = self.model.to_tokens(prompt)
        
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **generation_kwargs,
            )
        
        generated_text = self.model.to_string(output[0])
        num_tokens = output.shape[1]
        
        return generated_text, num_tokens
    
    def run_intervention_sweep(
        self,
        prompts: List[str],
        directions: Dict[Tuple[int, ActivationPoint, DirectionType], Tensor],
        layers: List[int],
        points: List[ActivationPoint],
        direction_types: List[DirectionType],
        strengths: List[float],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        include_baseline: bool = True,
    ) -> List[InterventionResult]:
        """Run a sweep of interventions across parameters.
        
        Args:
            prompts: List of input prompts
            directions: Dictionary of direction vectors
            layers: Layers to test
            points: Activation points to test
            direction_types: Direction types to test
            strengths: Intervention strengths to test
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            include_baseline: Whether to include baseline (no intervention) results
            
        Returns:
            List of intervention results
        """
        results = []
        
        for prompt in prompts:
            # Optionally add baseline
            if include_baseline:
                output, num_tokens = self.generate_baseline(
                    prompt, max_new_tokens, temperature
                )
                results.append(InterventionResult(
                    prompt=prompt,
                    layer=-1,  # -1 indicates baseline
                    point="pre",
                    direction_type="difference",
                    strength=0.0,
                    output=output,
                    num_tokens=num_tokens,
                ))
            
            # Run interventions
            for layer in layers:
                for point in points:
                    for direction_type in direction_types:
                        direction = directions.get((layer, point, direction_type))
                        if direction is None:
                            continue
                        
                        for strength in strengths:
                            output, num_tokens = self.generate_with_intervention(
                                prompt=prompt,
                                layer=layer,
                                point=point,
                                direction=direction,
                                strength=strength,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                            )
                            
                            results.append(InterventionResult(
                                prompt=prompt,
                                layer=layer,
                                point=point,
                                direction_type=direction_type,
                                strength=strength,
                                output=output,
                                num_tokens=num_tokens,
                            ))
        
        return results


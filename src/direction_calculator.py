"""Direction calculation utilities."""

import torch
from torch import Tensor
from typing import Dict, Tuple, Literal
from .activation_extractor import ActivationPoint


DirectionType = Literal["difference", "original", "reasoning", "random"]


class DirectionCalculator:
    """Calculates reasoning directions from activations."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize the direction calculator.
        
        Args:
            device: Device to perform calculations on
        """
        self.device = device
    
    def normalize_direction(self, direction: Tensor) -> Tensor:
        """Normalize a direction vector to unit length.
        
        Args:
            direction: Direction vector of shape (d_model,)
            
        Returns:
            Normalized direction vector
        """
        norm = torch.norm(direction)
        if norm > 0:
            return direction / norm
        return direction
    
    def calculate_difference_direction(
        self,
        reasoning_activations: Tensor,
        non_reasoning_activations: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """Calculate the difference direction between reasoning and non-reasoning.
        
        Args:
            reasoning_activations: Activations from reasoning model (batch, seq, d_model)
            non_reasoning_activations: Activations from non-reasoning model
            normalize: Whether to normalize to unit vector
            
        Returns:
            Direction vector of shape (d_model,)
        """
        # Calculate mean activations
        reasoning_mean = reasoning_activations.mean(dim=[0, 1])
        non_reasoning_mean = non_reasoning_activations.mean(dim=[0, 1])
        
        # Calculate difference
        direction = reasoning_mean - non_reasoning_mean
        
        # Normalize if requested
        if normalize:
            direction = self.normalize_direction(direction)
        
        return direction
    
    def calculate_original_direction(
        self,
        non_reasoning_activations: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """Calculate direction from non-reasoning model only.
        
        Args:
            non_reasoning_activations: Activations from non-reasoning model
            normalize: Whether to normalize to unit vector
            
        Returns:
            Direction vector of shape (d_model,)
        """
        direction = non_reasoning_activations.mean(dim=[0, 1])
        
        if normalize:
            direction = self.normalize_direction(direction)
        
        return direction
    
    def calculate_reasoning_direction(
        self,
        reasoning_activations: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """Calculate direction from reasoning model only.
        
        Args:
            reasoning_activations: Activations from reasoning model
            normalize: Whether to normalize to unit vector
            
        Returns:
            Direction vector of shape (d_model,)
        """
        direction = reasoning_activations.mean(dim=[0, 1])
        
        if normalize:
            direction = self.normalize_direction(direction)
        
        return direction
    
    def generate_random_direction(
        self,
        d_model: int,
        mean: float = 0.0,
        std: float = 1.0,
        normalize: bool = True,
    ) -> Tensor:
        """Generate a random direction vector.
        
        Args:
            d_model: Dimension of the model
            mean: Mean of random distribution
            std: Standard deviation of random distribution
            normalize: Whether to normalize to unit vector
            
        Returns:
            Random direction vector of shape (d_model,)
        """
        direction = torch.randn(d_model, device=self.device) * std + mean
        
        if normalize:
            direction = self.normalize_direction(direction)
        
        return direction
    
    def calculate_all_directions(
        self,
        reasoning_activations: Dict[Tuple[int, ActivationPoint], Tensor],
        non_reasoning_activations: Dict[Tuple[int, ActivationPoint], Tensor],
        direction_types: list[DirectionType],
        normalize: bool = True,
    ) -> Dict[Tuple[int, ActivationPoint, DirectionType], Tensor]:
        """Calculate all specified direction types for all layers and points.
        
        Args:
            reasoning_activations: Activations from reasoning model
            non_reasoning_activations: Activations from non-reasoning model
            direction_types: Types of directions to calculate
            normalize: Whether to normalize directions
            
        Returns:
            Dictionary mapping (layer, point, type) to direction vectors
        """
        directions = {}
        
        # Get keys (layer, point combinations)
        keys = list(reasoning_activations.keys())
        
        for layer, point in keys:
            r_act = reasoning_activations[(layer, point)]
            nr_act = non_reasoning_activations[(layer, point)]
            
            for dtype in direction_types:
                if dtype == "difference":
                    direction = self.calculate_difference_direction(
                        r_act, nr_act, normalize
                    )
                elif dtype == "original":
                    direction = self.calculate_original_direction(
                        nr_act, normalize
                    )
                elif dtype == "reasoning":
                    direction = self.calculate_reasoning_direction(
                        r_act, normalize
                    )
                elif dtype == "random":
                    # Get d_model from activation shape
                    d_model = r_act.shape[-1]
                    direction = self.generate_random_direction(
                        d_model, normalize=normalize
                    )
                else:
                    raise ValueError(f"Unknown direction type: {dtype}")
                
                directions[(layer, point, dtype)] = direction
        
        return directions


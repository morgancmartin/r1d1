"""Activation extraction utilities."""

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from typing import Dict, List, Literal, Tuple
from tqdm import tqdm


ActivationPoint = Literal["pre", "mid", "post"]


class ActivationExtractor:
    """Extracts activations from model layers."""
    
    def __init__(self, model: HookedTransformer):
        """Initialize the activation extractor.
        
        Args:
            model: The HookedTransformer model to extract from
        """
        self.model = model
        self.n_layers = model.cfg.n_layers
    
    def get_hook_name(self, layer: int, point: ActivationPoint) -> str:
        """Get the hook name for a specific layer and activation point.
        
        Args:
            layer: Layer index
            point: Activation point (pre/mid/post)
            
        Returns:
            Hook name string
        """
        if point == "pre":
            # Pre-attention: residual stream before attention
            return f"blocks.{layer}.hook_resid_pre"
        elif point == "mid":
            # Mid-attention: residual stream after attention, before MLP
            return f"blocks.{layer}.hook_resid_mid"
        elif point == "post":
            # Post-attention: residual stream after MLP
            return f"blocks.{layer}.hook_resid_post"
        else:
            raise ValueError(f"Unknown activation point: {point}")
    
    def extract_activations(
        self,
        prompts: List[str],
        layers: List[int],
        points: List[ActivationPoint],
        batch_size: int = 1,
    ) -> Dict[Tuple[int, ActivationPoint], Tensor]:
        """Extract activations from specified layers and points.
        
        Args:
            prompts: List of prompts to run through the model
            layers: List of layer indices to extract from
            points: List of activation points to extract
            batch_size: Batch size for processing (default 1 for simplicity)
            
        Returns:
            Dictionary mapping (layer, point) to activation tensors
            Shape of each tensor: (num_prompts, seq_len, d_model)
        """
        activations = {}
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize
            tokens = self.model.to_tokens(batch_prompts)
            
            # Storage for this batch
            batch_activations = {(layer, point): [] for layer in layers for point in points}
            
            # Define hooks to capture activations
            def make_hook(layer_idx: int, point: ActivationPoint):
                def hook(activation: Tensor, hook):
                    batch_activations[(layer_idx, point)].append(activation.detach().cpu())
                return hook
            
            # Add hooks
            hooks = []
            for layer in layers:
                for point in points:
                    hook_name = self.get_hook_name(layer, point)
                    hooks.append((hook_name, make_hook(layer, point)))
            
            # Run model with hooks
            with torch.no_grad():
                self.model.run_with_hooks(
                    tokens,
                    fwd_hooks=hooks,
                )
            
            # Aggregate activations
            for key in batch_activations:
                if key not in activations:
                    activations[key] = []
                # Concatenate batch dimension
                activations[key].extend(batch_activations[key])
        
        # Stack all activations - handle variable sequence lengths by padding
        for key in activations:
            if len(activations[key]) == 0:
                continue
            
            # Find max sequence length
            max_seq_len = max(act.shape[1] for act in activations[key])
            
            # Pad all activations to max length
            padded_acts = []
            for act in activations[key]:
                if act.shape[1] < max_seq_len:
                    # Pad on the sequence dimension
                    pad_size = max_seq_len - act.shape[1]
                    padding = torch.zeros(
                        act.shape[0], pad_size, act.shape[2],
                        dtype=act.dtype, device=act.device
                    )
                    act = torch.cat([act, padding], dim=1)
                padded_acts.append(act)
            
            # Now concatenate
            activations[key] = torch.cat(padded_acts, dim=0)
        
        return activations
    
    def get_mean_activation(
        self,
        activations: Tensor,
        aggregate_seq: bool = True,
    ) -> Tensor:
        """Calculate mean activation across samples (and optionally sequence).
        
        Args:
            activations: Activation tensor of shape (batch, seq_len, d_model)
            aggregate_seq: Whether to average across sequence dimension
            
        Returns:
            Mean activation tensor
        """
        if aggregate_seq:
            # Average across batch and sequence dimensions
            # Shape: (d_model,)
            return activations.mean(dim=[0, 1])
        else:
            # Average only across batch dimension
            # Shape: (seq_len, d_model)
            return activations.mean(dim=0)
    
    def get_last_token_activation(
        self,
        activations: Tensor,
    ) -> Tensor:
        """Get activation at the last token position for each sample.
        
        Args:
            activations: Activation tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Activations at last token, shape (batch, d_model)
        """
        return activations[:, -1, :]


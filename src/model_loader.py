"""Model loading utilities."""

import torch
from transformer_lens import HookedTransformer
from typing import Optional

from .config import ModelConfig, MODELS


class ModelLoader:
    """Handles loading of HookedTransformer models."""
    
    @staticmethod
    def load_model(
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> HookedTransformer:
        """Load a model using HookedTransformer.
        
        Args:
            model_name: Name of the model (either key in MODELS or HF model path)
            device: Device to load model on
            dtype: Data type for model weights
            
        Returns:
            Loaded HookedTransformer model
        """
        # Check if this is a predefined model config
        if model_name in MODELS:
            config = MODELS[model_name]
            hf_name = config.hf_model_name
        else:
            hf_name = model_name
        
        print(f"Loading model: {hf_name}")
        
        # Convert dtype string to torch dtype
        torch_dtype = getattr(torch, dtype)
        
        # Load model with HookedTransformer
        model = HookedTransformer.from_pretrained(
            hf_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        
        # Set to evaluation mode
        model.eval()
        
        print(f"Model loaded successfully. Number of layers: {model.cfg.n_layers}")
        
        return model
    
    @staticmethod
    def get_model_config(model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a predefined model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig if found, None otherwise
        """
        return MODELS.get(model_name)


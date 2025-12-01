"""Model loading utilities."""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .config import ModelConfig, MODELS
from .hooked_hf_model import load_hooked_hf_model


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
        
        # Try loading - if model not in whitelist, use custom HF wrapper
        try:
            model = HookedTransformer.from_pretrained(
                hf_name,
                device=device,
                torch_dtype=torch_dtype,
            )
        except (ValueError, KeyError) as e:
            if "not found" in str(e) or "Valid official model names" in str(e):
                print(f"Model not in TransformerLens whitelist.")
                print(f"Loading as HookedHFModel (custom wrapper with manual hooks)...")
                
                # Use our custom HuggingFace wrapper
                model = load_hooked_hf_model(
                    hf_name,
                    device=device,
                    torch_dtype=torch_dtype,
                )
                print(f"✓ Successfully loaded {hf_name} with custom hooks")
            else:
                raise
        
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


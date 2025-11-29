"""Model loading utilities."""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        
        # Try loading - if model not in whitelist, load from HuggingFace directly
        try:
            model = HookedTransformer.from_pretrained(
                hf_name,
                device=device,
                torch_dtype=torch_dtype,
            )
        except (ValueError, KeyError) as e:
            if "not found" in str(e) or "Valid official model names" in str(e):
                print(f"Model not in TransformerLens whitelist. Loading directly from HuggingFace...")
                # Load using HuggingFace transformers, then wrap in HookedTransformer
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_name,
                    torch_dtype=torch_dtype,
                    device_map=device,
                )
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
                
                # Convert to HookedTransformer
                model = HookedTransformer.from_pretrained(
                    hf_name,
                    hf_model=hf_model,
                    tokenizer=tokenizer,
                    device=device,
                    torch_dtype=torch_dtype,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                )
                print(f"Successfully loaded {hf_name} via HuggingFace")
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


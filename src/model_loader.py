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
        
        # Try loading - if model not in whitelist, use alternative approach
        try:
            model = HookedTransformer.from_pretrained(
                hf_name,
                device=device,
                torch_dtype=torch_dtype,
            )
        except (ValueError, KeyError) as e:
            if "not found" in str(e) or "Valid official model names" in str(e):
                print(f"Model not in TransformerLens whitelist. Loading via HuggingFace then converting...")
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
                
                # Use from_pretrained_no_processing - the proper way to load unsupported models
                print("Using from_pretrained_no_processing...")
                try:
                    model = HookedTransformer.from_pretrained_no_processing(
                        hf_name,
                        device=device,
                        dtype=torch_dtype,
                        tokenizer=tokenizer,
                        fold_ln=False,
                        center_writing_weights=False,
                        center_unembed=False,
                    )
                    print(f"Successfully loaded {hf_name} as HookedTransformer")
                except Exception as e2:
                    print(f"from_pretrained_no_processing failed: {e2}")
                    print("This model may not be compatible with TransformerLens")
                    raise RuntimeError(
                        f"Unable to load {hf_name} with TransformerLens. "
                        f"The model architecture may not be supported. "
                        f"Original error: {e2}"
                    )
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


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
                print(f"Model not in TransformerLens whitelist. Trying with trust_remote_code...")
                # Try with trust_remote_code for newer models
                try:
                    model = HookedTransformer.from_pretrained(
                        hf_name,
                        device=device,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )
                    print(f"Successfully loaded {hf_name} with trust_remote_code")
                except Exception as e2:
                    print(f"Failed with trust_remote_code. Trying direct HuggingFace load...")
                    # Load the model and tokenizer separately using HuggingFace
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hf_name,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        device_map="auto",
                    )
                    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
                    
                    # Wrap in HookedTransformer using from_pretrained_no_processing
                    print("Wrapping HuggingFace model in HookedTransformer...")
                    model = HookedTransformer(
                        hf_model.config,
                        tokenizer=tokenizer,
                        move_to_device=False,  # Already on device
                    )
                    model.model = hf_model
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


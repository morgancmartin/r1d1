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
                    print(f"Failed with trust_remote_code. Trying HookedTransformer.from_pretrained with hf_model...")
                    # Load the model and tokenizer separately using HuggingFace
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hf_name,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        device_map="auto",
                    )
                    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
                    
                    # Try passing the loaded model to from_pretrained
                    print("Wrapping HuggingFace model in HookedTransformer...")
                    try:
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
                        print(f"Successfully loaded {hf_name} via HuggingFace with HookedTransformer wrapper")
                    except Exception as e3:
                        print(f"TransformerLens wrapping also failed: {e3}")
                        print(f"Returning raw HuggingFace model (hooks may not work perfectly)")
                        # Return the HF model directly - we'll need to handle hooks differently
                        model = hf_model
                        model.tokenizer = tokenizer
                        # Add a fake cfg for compatibility
                        model.cfg = type('obj', (object,), {
                            'n_layers': len(hf_model.model.layers) if hasattr(hf_model, 'model') else hf_model.config.num_hidden_layers,
                            'd_model': hf_model.config.hidden_size,
                        })()
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


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
                print(f"Model not in TransformerLens whitelist. Trying to load as Llama architecture...")
                
                # Load HF model and tokenizer first
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
                
                # Use from_pretrained_no_processing which bypasses the name check
                print("Using from_pretrained_no_processing to bypass whitelist...")
                try:
                    from transformer_lens.loading_from_pretrained import get_pretrained_model_config
                    
                    # Pretend it's a Llama model for config purposes
                    cfg = get_pretrained_model_config(
                        "meta-llama/Meta-Llama-3-8B-Instruct",
                        hf_model=hf_model,
                        checkpoint_index=None,
                        checkpoint_value=None,
                        fold_ln=False,
                        center_writing_weights=False,
                        center_unembed=False,
                        dtype=torch_dtype,
                    )
                    
                    # Now create HookedTransformer with the config
                    model = HookedTransformer(cfg, tokenizer=tokenizer, move_to_device=False)
                    model.load_state_dict(hf_model.state_dict(), strict=False)
                    model.to(device)
                    
                    print(f"Successfully loaded {hf_name} as HookedTransformer using Llama architecture")
                except Exception as e2:
                    print(f"Architecture loading failed: {e2}")
                    print("Falling back to raw HuggingFace model...")
                    # Return the HF model directly
                    model = hf_model
                    model.tokenizer = tokenizer
                    model.to(device)
                    # Add minimal compatibility layer
                    model.cfg = type('obj', (object,), {
                        'n_layers': hf_model.config.num_hidden_layers,
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


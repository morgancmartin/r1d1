"""Custom HookedTransformer-compatible wrapper for HuggingFace models.

This module provides a way to use HuggingFace models with the same interface
as TransformerLens HookedTransformer, when the model isn't officially supported.
"""

import torch
from torch import nn, Tensor
from typing import Optional, List, Tuple, Callable, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class HookedHFModel:
    """Wrapper that makes HuggingFace models compatible with HookedTransformer interface."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        """Initialize the hooked HuggingFace model.
        
        Args:
            model: Pre-loaded HuggingFace model
            tokenizer: Pre-loaded tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Create a config object compatible with HookedTransformer
        self.cfg = type('Config', (), {
            'n_layers': model.config.num_hidden_layers,
            'd_model': model.config.hidden_size,
            'n_heads': model.config.num_attention_heads,
            'd_vocab': model.config.vocab_size,
        })()
        
        # Store activation hooks
        self._hook_handles = []
        self._activation_cache = {}
        
    def to_tokens(
        self,
        text: str | List[str],
        prepend_bos: bool = True,
        padding_side: Optional[str] = None,
    ) -> Tensor:
        """Tokenize text (HookedTransformer-compatible interface).
        
        Args:
            text: Text or list of texts to tokenize
            prepend_bos: Whether to prepend BOS token (handled by tokenizer)
            padding_side: Which side to pad on
            
        Returns:
            Token tensor of shape (batch, seq_len)
        """
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return encoded['input_ids'].to(self.device)
    
    def to_string(self, tokens: Tensor) -> str | List[str]:
        """Convert tokens back to string.
        
        Args:
            tokens: Token tensor of shape (batch, seq_len) or (seq_len,)
            
        Returns:
            Decoded string or list of strings
        """
        if tokens.dim() == 1:
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        else:
            return [self.tokenizer.decode(t, skip_special_tokens=False) for t in tokens]
    
    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Forward pass returning logits.
        
        Args:
            input_ids: Token IDs
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        outputs = self.model(input_ids=input_ids, **kwargs)
        return outputs.logits
    
    def run_with_hooks(
        self,
        input_ids: Tensor,
        fwd_hooks: List[Tuple[str, Callable]] = None,
        **kwargs
    ) -> Tensor:
        """Run forward pass with activation hooks.
        
        Args:
            input_ids: Token IDs
            fwd_hooks: List of (hook_name, hook_fn) tuples
            
        Returns:
            Model outputs
        """
        # Clear previous hooks
        self._clear_hooks()
        self._activation_cache = {}
        
        if fwd_hooks:
            # Register hooks
            for hook_name, hook_fn in fwd_hooks:
                self._register_hook(hook_name, hook_fn)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, **kwargs)
        
        # Clear hooks after use
        self._clear_hooks()
        
        return outputs.logits
    
    def _register_hook(self, hook_name: str, hook_fn: Callable):
        """Register a forward hook on a specific module.
        
        Args:
            hook_name: Name like "blocks.0.hook_resid_pre"
            hook_fn: Hook function to call
        """
        # Parse hook name to find the right module
        # Format: blocks.{layer}.hook_resid_{pre|mid|post}
        parts = hook_name.split('.')
        
        if len(parts) >= 3 and parts[0] == 'blocks':
            layer_idx = int(parts[1])
            hook_type = parts[2]  # hook_resid_pre, hook_resid_mid, hook_resid_post
            
            # Get the layer module
            layer = self.model.model.layers[layer_idx]
            
            # Determine which module to hook based on type
            if 'pre' in hook_type:
                # Pre-attention: hook the input to the layer
                target_module = layer
                position = 'pre'
            elif 'mid' in hook_type:
                # Mid: after attention, before MLP
                target_module = layer.post_attention_layernorm
                position = 'post'
            elif 'post' in hook_type:
                # Post: after MLP (end of block)
                target_module = layer
                position = 'post'
            else:
                raise ValueError(f"Unknown hook type: {hook_type}")
            
            # Create wrapper that captures activations
            def make_hook(hook_fn, position):
                def hook_wrapper(module, input, output):
                    # For pre hooks, use input; for post hooks, use output
                    if position == 'pre':
                        activation = input[0] if isinstance(input, tuple) else input
                    else:
                        activation = output[0] if isinstance(output, tuple) else output
                    
                    # Call the user's hook function
                    # Create a minimal hook point object
                    hook_point = type('HookPoint', (), {'name': hook_name})()
                    result = hook_fn(activation, hook_point)
                    
                    return result if result is not None else (output if position == 'post' else None)
                
                return hook_wrapper
            
            # Register the hook
            if position == 'pre':
                handle = target_module.register_forward_pre_hook(make_hook(hook_fn, position))
            else:
                handle = target_module.register_forward_hook(make_hook(hook_fn, position))
            
            self._hook_handles.append(handle)
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        fwd_hooks: List[Tuple[str, Callable]] = None,
        **kwargs
    ) -> Tensor:
        """Generate text with optional hooks.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            fwd_hooks: List of (hook_name, hook_fn) tuples
            
        Returns:
            Generated token IDs
        """
        # Register hooks if provided
        if fwd_hooks:
            for hook_name, hook_fn in fwd_hooks:
                self._register_hook(hook_name, hook_fn)
        
        try:
            # Use HuggingFace's generate
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p if top_p is not None else 1.0,
                    top_k=top_k if top_k is not None else 50,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **kwargs
                )
        finally:
            # Clear hooks
            self._clear_hooks()
        
        return output
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device: str):
        """Move model to device."""
        self.model.to(device)
        self.device = device
        return self


def load_hooked_hf_model(
    model_name: str,
    device: str = "cuda",
    torch_dtype = torch.float16,
) -> HookedHFModel:
    """Load a HuggingFace model as a HookedHFModel.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load on
        torch_dtype: Data type for model weights
        
    Returns:
        HookedHFModel instance
    """
    print(f"Loading {model_name} as HookedHFModel...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create wrapper
    hooked_model = HookedHFModel(model, tokenizer, device)
    hooked_model.eval()
    
    print(f"✓ Loaded {model_name} with {hooked_model.cfg.n_layers} layers")
    
    return hooked_model


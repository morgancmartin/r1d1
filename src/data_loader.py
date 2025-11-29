"""Data loading utilities for reasoning direction experiments."""

from typing import List, Dict, Optional
from datasets import load_dataset
from transformer_lens import HookedTransformer

from .config import DataConfig, TOY_PROBLEMS


class DataLoader:
    """Handles loading and formatting of datasets."""
    
    def __init__(self, config: DataConfig):
        """Initialize the data loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.dataset = None
        
    def load_gsm8k(self, num_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load GSM8K math problems.
        
        Args:
            num_samples: Number of samples to load. If None, uses config value.
            
        Returns:
            List of problem dictionaries with 'question' and 'answer' keys
        """
        if num_samples is None:
            num_samples = self.config.num_samples
            
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split=self.config.dataset_split)
        
        # Sample and format
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            problems.append({
                "question": item["question"],
                "answer": item["answer"],
            })
        
        return problems
    
    def load_toy_problems(self) -> List[Dict[str, str]]:
        """Load simple toy math problems.
        
        Returns:
            List of problem dictionaries with 'question' key
        """
        return [{"question": q} for q in TOY_PROBLEMS]
    
    def format_prompt(
        self, 
        question: str, 
        model: HookedTransformer,
        system_prompt: Optional[str] = None
    ) -> str:
        """Format a question using the model's chat template.
        
        Args:
            question: The question to format
            model: The model to use for formatting
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": question
        })
        
        # Use the model's tokenizer to apply chat template
        if hasattr(model.tokenizer, "apply_chat_template"):
            prompt = model.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            if system_prompt:
                prompt = f"System: {system_prompt}\n\nUser: {question}\n\nAssistant:"
            else:
                prompt = f"User: {question}\n\nAssistant:"
        
        return prompt
    
    def load_data(
        self, 
        model: Optional[HookedTransformer] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load and optionally format data.
        
        Args:
            model: Optional model for prompt formatting
            system_prompt: Optional system prompt for formatting
            
        Returns:
            List of problem dictionaries
        """
        if self.config.use_toy_problems:
            problems = self.load_toy_problems()
        else:
            problems = self.load_gsm8k()
        
        # Format prompts if model is provided
        if model is not None:
            for problem in problems:
                problem["prompt"] = self.format_prompt(
                    problem["question"], 
                    model,
                    system_prompt
                )
        
        return problems


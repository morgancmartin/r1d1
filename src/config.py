"""Configuration for reasoning direction experiments."""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    name: str
    hf_model_name: str
    is_reasoning_model: bool
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    
    dataset_name: str = "gsm8k"
    dataset_split: str = "test"
    num_samples: int = 100
    use_toy_problems: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Models
    non_reasoning_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    reasoning_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    # Activation points to hook
    activation_points: List[Literal["pre", "mid", "post"]] = None
    
    # Layers to analyze
    layers: List[int] = None
    
    # Intervention strengths
    strengths: List[float] = None
    
    # Direction types
    direction_types: List[Literal["difference", "original", "reasoning", "random"]] = None
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Set defaults for list fields."""
        if self.activation_points is None:
            self.activation_points = ["pre", "mid", "post"]
        
        if self.layers is None:
            # Default to a subset of layers for testing
            self.layers = [0, 1, 2, 3, 10, 15, 20, 25, 30]
        
        if self.strengths is None:
            self.strengths = [-0.1, -0.05, 0.05, 0.1]
        
        if self.direction_types is None:
            self.direction_types = ["difference", "original", "reasoning"]


# Predefined model configurations
MODELS = {
    "llama3-8b": ModelConfig(
        name="llama3-8b",
        hf_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        is_reasoning_model=False,
    ),
    "deepseek-r1-llama-8b": ModelConfig(
        name="deepseek-r1-llama-8b",
        hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        is_reasoning_model=True,
    ),
}

# Toy math problems for testing
TOY_PROBLEMS = [
    "What is 2+2?",
    "What is 5+7?",
    "What is 10-3?",
    "What is 6*4?",
    "What is 15/3?",
    "What is 8+5-2?",
    "What is 3*3*2?",
    "What is 20/4+1?",
    "What is 7-2+5?",
    "What is 9+6-3?",
]


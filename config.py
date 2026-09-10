"""
Configuration file for Sanskrit LLM training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""
    vocab_size: int = 16000  # Sanskrit-specific vocabulary size
    n_embd: int = 768        # Embedding dimension
    n_head: int = 12         # Number of attention heads
    n_layer: int = 12        # Number of transformer layers
    block_size: int = 1024   # Maximum context length
    dropout: float = 0.1
    bias: bool = True        # Use bias in LayerNorm/Linear

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_iters: int = 50000
    eval_interval: int = 500
    eval_iters: int = 200
    warmup_iters: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    log_interval: int = 100

@dataclass
class DataConfig:
    """Data preprocessing settings"""
    data_path: str = "data/SB.txt"
    train_split: float = 0.9
    min_text_length: int = 100  # Minimum characters per text
    max_text_length: int = 50000  # Maximum characters per text
    remove_metadata: bool = True
    normalize_devanagari: bool = True
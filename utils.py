"""
Utility functions for Sanskrit LLM
"""

import torch
import numpy as np
from typing import List, Optional
import random
import os

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_model_summary(model: torch.nn.Module):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def get_optimizer_state(optimizer):
    """Get optimizer state for logging"""
    return {
        'lr': optimizer.param_groups[0]['lr'],
        'weight_decay': optimizer.param_groups[0]['weight_decay']
    }

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss"""
    return np.exp(loss)

def save_metrics(metrics: dict, filename: str):
    """Save training metrics to JSON"""
    import json
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filename: str) -> dict:
    """Load training metrics from JSON"""
    import json
    with open(filename, 'r') as f:
        return json.load(f)
"""
Utility functions for N-gram Language Modeling
=============================================

Common utility functions used across the project.
"""

import numpy as np
import math
from typing import List, Dict, Any
import random


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def safe_log(x: float) -> float:
    """Safe logarithm that handles zero and very small values."""
    if x <= 0:
        return -float('inf')
    return math.log(x)


def safe_exp(x: float) -> float:
    """Safe exponential that handles very large negative values."""
    if x == -float('inf'):
        return 0.0
    if x > 700:  # Prevent overflow
        return float('inf')
    return math.exp(x)


def calculate_perplexity(log_probabilities: List[float], num_tokens: int) -> float:
    """Calculate perplexity from log probabilities."""
    if num_tokens == 0:
        return float('inf')
    
    avg_log_prob = sum(log_probabilities) / num_tokens
    return safe_exp(-avg_log_prob)


def normalize_probabilities(probs: List[float]) -> List[float]:
    """Normalize a list of probabilities to sum to 1."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def sample_from_distribution(items: List[Any], probabilities: List[float]) -> Any:
    """Sample an item from a probability distribution."""
    if not items or not probabilities:
        raise ValueError("Items and probabilities must not be empty")
    
    if len(items) != len(probabilities):
        raise ValueError("Items and probabilities must have the same length")
    
    # Normalize probabilities
    normalized_probs = normalize_probabilities(probabilities)
    
    # Sample using numpy
    return np.random.choice(items, p=normalized_probs)


def format_perplexity(perplexity: float) -> str:
    """Format perplexity for display."""
    if perplexity == float('inf'):
        return "INF"
    elif perplexity > 1e6:
        return f"{perplexity:.2e}"
    else:
        return f"{perplexity:.2f}"


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                      length: int = 50, fill: str = '█'):
    """Print a progress bar."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

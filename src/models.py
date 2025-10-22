"""
N-gram Language Models Implementation
====================================

Implements various N-gram language models including:
- Maximum Likelihood Estimation (MLE)
- Add-1 (Laplace) Smoothing
- Linear Interpolation
- Stupid Backoff
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .ngram_counts import NGramCounter
from .utils import safe_log, safe_exp, sample_from_distribution


class BaseNGramModel:
    """Base class for N-gram language models."""
    
    def __init__(self, n: int):
        self.n = n
        self.counter = NGramCounter(n)
        self.vocab = set()
        self.vocab_size = 0
    
    def train(self, sentences: List[List[str]]) -> None:
        """Train the model on sentences."""
        self.counter.count_ngrams(sentences)
        self.vocab = self.counter.get_vocabulary()
        self.vocab_size = len(self.vocab)
        self._calculate_probabilities()
    
    def _calculate_probabilities(self) -> None:
        """Calculate model-specific probabilities. Override in subclasses."""
        pass  # Default implementation does nothing
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get probability of an N-gram. Override in subclasses."""
        raise NotImplementedError
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """Calculate perplexity on sentences."""
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            # Add sentence boundaries
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            total_tokens += len(tokens) - (self.n - 1)  # Exclude context tokens
            
            # Calculate log probabilities for each N-gram
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                prob = self.get_probability(ngram)
                
                if prob == 0:
                    return float('inf')  # Return infinity for zero probability
                
                total_log_prob += safe_log(prob)
        
        if total_tokens == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_tokens
        return safe_exp(-avg_log_prob)


class MLEModel(BaseNGramModel):
    """Maximum Likelihood Estimation N-gram model."""
    
    def __init__(self, n: int):
        super().__init__(n)
        self.probabilities = {}
    
    def _calculate_probabilities(self) -> None:
        """Calculate maximum likelihood probabilities."""
        if self.n == 1:
            # Unigram model
            total_count = self.counter.get_total_ngrams()
            for ngram, count in self.counter.ngram_counts.items():
                self.probabilities[ngram] = count / total_count
        else:
            # Higher-order N-gram models
            for ngram, count in self.counter.ngram_counts.items():
                context = ngram[:-1]
                context_count = self.counter.get_context_count(context)
                if context_count > 0:
                    self.probabilities[ngram] = count / context_count
                else:
                    self.probabilities[ngram] = 0.0
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get probability of an N-gram."""
        return self.probabilities.get(ngram, 0.0)


class AddOneModel(BaseNGramModel):
    """Add-1 (Laplace) smoothing model."""
    
    def __init__(self, n: int):
        super().__init__(n)
        self.probabilities = {}
    
    def _calculate_probabilities(self) -> None:
        """Calculate Add-1 smoothed probabilities."""
        if self.n == 1:
            # Unigram with Add-1
            total_count = self.counter.get_total_ngrams()
            for ngram, count in self.counter.ngram_counts.items():
                self.probabilities[ngram] = (count + 1) / (total_count + self.vocab_size)
        else:
            # Higher-order N-gram with Add-1
            for ngram, count in self.counter.ngram_counts.items():
                context = ngram[:-1]
                context_count = self.counter.get_context_count(context)
                self.probabilities[ngram] = (count + 1) / (context_count + self.vocab_size)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get Add-1 smoothed probability."""
        if ngram in self.probabilities:
            return self.probabilities[ngram]
        
        if self.n == 1:
            # Unigram: return 1/(total_count + vocab_size)
            total_count = self.counter.get_total_ngrams()
            return 1.0 / (total_count + self.vocab_size)
        else:
            # Higher-order: return 1/(context_count + vocab_size)
            context = ngram[:-1]
            context_count = self.counter.get_context_count(context)
            return 1.0 / (context_count + self.vocab_size)


class LinearInterpolationModel(BaseNGramModel):
    """Linear interpolation model combining unigram, bigram, and trigram."""
    
    def __init__(self, lambdas: List[float]):
        super().__init__(3)  # Trigram model
        self.lambdas = lambdas
        self.unigram_model = None
        self.bigram_model = None
        self.trigram_model = None
        
        # Validate lambda weights
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError("Lambda weights must sum to 1.0")
    
    def train(self, sentences: List[List[str]]) -> None:
        """Train all component models."""
        super().train(sentences)
        
        # Train unigram model
        self.unigram_model = MLEModel(1)
        self.unigram_model.train(sentences)
        
        # Train bigram model
        self.bigram_model = MLEModel(2)
        self.bigram_model.train(sentences)
        
        # Train trigram model
        self.trigram_model = MLEModel(3)
        self.trigram_model.train(sentences)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get interpolated probability."""
        if len(ngram) != 3:
            raise ValueError("Linear interpolation model expects trigrams")
        
        # Get probabilities from each model
        unigram_prob = self.unigram_model.get_probability((ngram[2],))
        bigram_prob = self.bigram_model.get_probability(ngram[1:])
        trigram_prob = self.trigram_model.get_probability(ngram)
        
        # Linear interpolation
        interpolated_prob = (self.lambdas[0] * unigram_prob + 
                           self.lambdas[1] * bigram_prob + 
                           self.lambdas[2] * trigram_prob)
        
        # Ensure non-zero probability
        return max(interpolated_prob, 1e-10)


class StupidBackoffModel(BaseNGramModel):
    """Stupid backoff model."""
    
    def __init__(self, alpha: float = 0.4):
        super().__init__(3)  # Trigram model
        self.alpha = alpha
        self.unigram_model = None
        self.bigram_model = None
        self.trigram_model = None
    
    def train(self, sentences: List[List[str]]) -> None:
        """Train all component models."""
        super().train(sentences)
        
        # Train unigram model
        self.unigram_model = MLEModel(1)
        self.unigram_model.train(sentences)
        
        # Train bigram model
        self.bigram_model = MLEModel(2)
        self.bigram_model.train(sentences)
        
        # Train trigram model
        self.trigram_model = MLEModel(3)
        self.trigram_model.train(sentences)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get backoff probability."""
        if len(ngram) != 3:
            raise ValueError("Stupid backoff model expects trigrams")
        
        # Try trigram first
        trigram_prob = self.trigram_model.get_probability(ngram)
        if trigram_prob > 0:
            return trigram_prob
        
        # Backoff to bigram
        bigram_prob = self.bigram_model.get_probability(ngram[1:])
        if bigram_prob > 0:
            return self.alpha * bigram_prob
        
        # Backoff to unigram
        unigram_prob = self.unigram_model.get_probability((ngram[2],))
        if unigram_prob > 0:
            return self.alpha * self.alpha * unigram_prob
        
        # If all probabilities are zero, return small positive value
        return 1e-10


def create_model(model_type: str, n: int = 3, **kwargs) -> BaseNGramModel:
    """Factory function to create N-gram models."""
    if model_type == "mle":
        return MLEModel(n)
    elif model_type == "add1":
        return AddOneModel(n)
    elif model_type == "linear_interp":
        lambdas = kwargs.get('lambdas', [0.1, 0.3, 0.6])
        return LinearInterpolationModel(lambdas)
    elif model_type == "stupid_backoff":
        alpha = kwargs.get('alpha', 0.4)
        return StupidBackoffModel(alpha)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

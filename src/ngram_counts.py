"""
N-gram Counting for Language Modeling
====================================

Handles counting and statistics for N-grams of different orders.
"""

from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict, Counter
import math


class NGramCounter:
    """Counts and manages N-gram statistics."""
    
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.total_ngrams = 0
    
    def count_ngrams(self, sentences: List[List[str]]) -> None:
        """Count N-grams from tokenized sentences."""
        for sentence in sentences:
            # Add sentence boundaries
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            # Count N-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.ngram_counts[ngram] += 1
                self.total_ngrams += 1
                
                # Update vocabulary
                for token in ngram:
                    self.vocab.add(token)
                
                # Count context (for conditional probabilities)
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
    
    def get_ngram_count(self, ngram: Tuple[str, ...]) -> int:
        """Get count for a specific N-gram."""
        return self.ngram_counts.get(ngram, 0)
    
    def get_context_count(self, context: Tuple[str, ...]) -> int:
        """Get count for a context (N-1 gram)."""
        return self.context_counts.get(context, 0)
    
    def get_vocabulary(self) -> Set[str]:
        """Get vocabulary from counted N-grams."""
        return self.vocab.copy()
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_total_ngrams(self) -> int:
        """Get total number of N-grams counted."""
        return self.total_ngrams
    
    def get_unique_ngrams(self) -> int:
        """Get number of unique N-grams."""
        return len(self.ngram_counts)
    
    def get_ngram_frequency_distribution(self) -> Dict[int, int]:
        """Get frequency distribution of N-grams."""
        freq_dist = Counter()
        for count in self.ngram_counts.values():
            freq_dist[count] += 1
        return dict(freq_dist)
    
    def get_most_frequent_ngrams(self, n: int = 20) -> List[Tuple[Tuple[str, ...], int]]:
        """Get most frequent N-grams."""
        return self.ngram_counts.most_common(n)
    
    def get_rare_ngrams(self, max_freq: int = 1) -> List[Tuple[str, ...]]:
        """Get N-grams with frequency <= max_freq."""
        return [ngram for ngram, count in self.ngram_counts.items() 
                if count <= max_freq]
    
    def calculate_sparsity(self) -> float:
        """Calculate sparsity of N-gram counts."""
        if self.n == 1:
            # For unigrams, sparsity is 0 (all words are counted)
            return 0.0
        
        # Calculate theoretical maximum N-grams
        vocab_size = len(self.vocab)
        max_ngrams = vocab_size ** self.n
        
        # Calculate actual unique N-grams
        unique_ngrams = len(self.ngram_counts)
        
        # Sparsity = 1 - (unique_ngrams / max_ngrams)
        sparsity = 1 - (unique_ngrams / max_ngrams) if max_ngrams > 0 else 1.0
        return sparsity
    
    def get_ngram_statistics(self) -> Dict[str, Any]:
        """Get comprehensive N-gram statistics."""
        freq_dist = self.get_ngram_frequency_distribution()
        
        return {
            'n': self.n,
            'total_ngrams': self.total_ngrams,
            'unique_ngrams': self.get_unique_ngrams(),
            'vocab_size': self.get_vocabulary_size(),
            'sparsity': self.calculate_sparsity(),
            'avg_frequency': self.total_ngrams / self.get_unique_ngrams() if self.get_unique_ngrams() > 0 else 0,
            'singleton_ngrams': freq_dist.get(1, 0),
            'singleton_ratio': freq_dist.get(1, 0) / self.get_unique_ngrams() if self.get_unique_ngrams() > 0 else 0,
            'max_frequency': max(self.ngram_counts.values()) if self.ngram_counts else 0,
            'min_frequency': min(self.ngram_counts.values()) if self.ngram_counts else 0
        }
    
    def print_statistics(self) -> None:
        """Print N-gram statistics."""
        stats = self.get_ngram_statistics()
        print(f"N-gram Statistics (N={self.n}):")
        print(f"  Total N-grams: {stats['total_ngrams']:,}")
        print(f"  Unique N-grams: {stats['unique_ngrams']:,}")
        print(f"  Vocabulary size: {stats['vocab_size']:,}")
        print(f"  Sparsity: {stats['sparsity']:.4f}")
        print(f"  Average frequency: {stats['avg_frequency']:.2f}")
        print(f"  Singleton N-grams: {stats['singleton_ngrams']:,} ({stats['singleton_ratio']:.2%})")
        print(f"  Max frequency: {stats['max_frequency']}")
        print(f"  Min frequency: {stats['min_frequency']}")
    
    def get_ngrams_by_context(self, context: Tuple[str, ...]) -> List[Tuple[Tuple[str, ...], int]]:
        """Get all N-grams that start with the given context."""
        ngrams = []
        for ngram, count in self.ngram_counts.items():
            if ngram[:-1] == context:
                ngrams.append((ngram, count))
        return sorted(ngrams, key=lambda x: x[1], reverse=True)
    
    def get_contexts(self) -> Set[Tuple[str, ...]]:
        """Get all contexts (N-1 grams)."""
        return set(self.context_counts.keys())
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about contexts."""
        context_counts = list(self.context_counts.values())
        
        return {
            'unique_contexts': len(self.context_counts),
            'avg_context_count': sum(context_counts) / len(context_counts) if context_counts else 0,
            'max_context_count': max(context_counts) if context_counts else 0,
            'min_context_count': min(context_counts) if context_counts else 0
        }

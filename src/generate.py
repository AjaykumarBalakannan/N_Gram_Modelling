"""
Text Generation for N-gram Language Models
==========================================

Handles text generation using trained N-gram models.
"""

import random
import numpy as np
from typing import List, Tuple, Any
from collections import defaultdict

from .models import BaseNGramModel
from .utils import sample_from_distribution, set_random_seed


class TextGenerator:
    """Generates text using N-gram language models."""
    
    def __init__(self, model: BaseNGramModel, seed: int = 42):
        self.model = model
        self.vocab = list(model.vocab)
        set_random_seed(seed)
    
    def generate_text(self, max_length: int = 20, start_tokens: List[str] = None) -> List[str]:
        """Generate text using the model."""
        if start_tokens is None:
            start_tokens = ['<s>'] * (self.model.n - 1)
        
        generated = start_tokens.copy()
        
        for _ in range(max_length):
            # Get context for next word
            if self.model.n == 1:
                context = ()
            else:
                context = tuple(generated[-(self.model.n-1):])
            
            # Get possible next words and their probabilities
            next_word_probs = self._get_next_word_probabilities(context)
            
            if not next_word_probs:
                break
            
            # Sample next word
            words, probs = zip(*next_word_probs)
            next_word = sample_from_distribution(list(words), list(probs))
            generated.append(next_word)
            
            # Stop at sentence end
            if next_word == '</s>':
                break
        
        return generated
    
    def _get_next_word_probabilities(self, context: Tuple[str, ...]) -> List[Tuple[str, float]]:
        """Get probabilities for next words given context."""
        next_word_probs = []
        
        for word in self.vocab:
            if self.model.n == 1:
                ngram = (word,)
            else:
                ngram = context + (word,)
            
            prob = self.model.get_probability(ngram)
            if prob > 0:
                next_word_probs.append((word, prob))
        
        return next_word_probs
    
    def generate_multiple_samples(self, num_samples: int = 5, max_length: int = 20) -> List[List[str]]:
        """Generate multiple text samples."""
        samples = []
        for _ in range(num_samples):
            generated = self.generate_text(max_length)
            # Remove sentence markers for display
            clean_tokens = [token for token in generated if token not in ['<s>', '</s>']]
            samples.append(clean_tokens)
        return samples
    
    def generate_with_temperature(self, max_length: int = 20, temperature: float = 1.0) -> List[str]:
        """Generate text with temperature sampling."""
        if start_tokens is None:
            start_tokens = ['<s>'] * (self.model.n - 1)
        
        generated = start_tokens.copy()
        
        for _ in range(max_length):
            # Get context for next word
            if self.model.n == 1:
                context = ()
            else:
                context = tuple(generated[-(self.model.n-1):])
            
            # Get probabilities
            next_word_probs = self._get_next_word_probabilities(context)
            
            if not next_word_probs:
                break
            
            # Apply temperature
            words, probs = zip(*next_word_probs)
            probs = np.array(probs)
            
            # Apply temperature scaling
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)  # Renormalize
            
            next_word = sample_from_distribution(list(words), list(probs))
            generated.append(next_word)
            
            # Stop at sentence end
            if next_word == '</s>':
                break
        
        return generated
    
    def generate_beam_search(self, max_length: int = 20, beam_size: int = 5) -> List[List[str]]:
        """Generate text using beam search."""
        # Initialize beam with start tokens
        start_tokens = ['<s>'] * (self.model.n - 1)
        beam = [(start_tokens, 0.0)]  # (sequence, log_probability)
        
        for _ in range(max_length):
            new_beam = []
            
            for sequence, log_prob in beam:
                # Get context
                if self.model.n == 1:
                    context = ()
                else:
                    context = tuple(sequence[-(self.model.n-1):])
                
                # Get next word probabilities
                next_word_probs = self._get_next_word_probabilities(context)
                
                if not next_word_probs:
                    new_beam.append((sequence, log_prob))
                    continue
                
                # Add each possible next word to beam
                for word, prob in next_word_probs:
                    new_sequence = sequence + [word]
                    new_log_prob = log_prob + np.log(prob)
                    new_beam.append((new_sequence, new_log_prob))
                    
                    # Stop if we hit end token
                    if word == '</s>':
                        break
            
            # Keep top beam_size candidates
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Check if all sequences end with </s>
            if all(seq[-1] == '</s>' for seq, _ in beam):
                break
        
        # Return clean sequences
        results = []
        for sequence, _ in beam:
            clean_tokens = [token for token in sequence if token not in ['<s>', '</s>']]
            results.append(clean_tokens)
        
        return results
    
    def calculate_perplexity_of_generated(self, generated_text: List[str]) -> float:
        """Calculate perplexity of generated text."""
        # Add sentence boundaries
        tokens = ['<s>'] * (self.model.n - 1) + generated_text + ['</s>']
        
        total_log_prob = 0.0
        total_ngrams = 0
        
        for i in range(len(tokens) - self.model.n + 1):
            ngram = tuple(tokens[i:i + self.model.n])
            prob = self.model.get_probability(ngram)
            
            if prob == 0:
                return float('inf')
            
            total_log_prob += np.log(prob)
            total_ngrams += 1
        
        if total_ngrams == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_ngrams
        return np.exp(-avg_log_prob)
    
    def analyze_generated_text(self, samples: List[List[str]]) -> dict:
        """Analyze quality of generated text samples."""
        if not samples:
            return {}
        
        # Basic statistics
        sample_lengths = [len(sample) for sample in samples]
        avg_length = np.mean(sample_lengths)
        
        # Vocabulary diversity
        all_tokens = []
        for sample in samples:
            all_tokens.extend(sample)
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        diversity_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Repetition analysis
        repetitions = 0
        for sample in samples:
            for i in range(len(sample) - 1):
                if sample[i] == sample[i+1]:
                    repetitions += 1
        
        repetition_rate = repetitions / total_tokens if total_tokens > 0 else 0
        
        # Perplexity analysis
        perplexities = []
        for sample in samples:
            if sample:  # Only calculate if sample is not empty
                perplexity = self.calculate_perplexity_of_generated(sample)
                if perplexity != float('inf'):
                    perplexities.append(perplexity)
        
        avg_perplexity = np.mean(perplexities) if perplexities else float('inf')
        
        return {
            'num_samples': len(samples),
            'avg_length': avg_length,
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'diversity_ratio': diversity_ratio,
            'repetition_rate': repetition_rate,
            'avg_perplexity': avg_perplexity,
            'samples': samples
        }
    
    def print_generated_samples(self, samples: List[List[str]], title: str = "Generated Text Samples") -> None:
        """Print generated text samples in a formatted way."""
        print(f"\n{title}:")
        print("-" * 40)
        
        for i, sample in enumerate(samples, 1):
            text = ' '.join(sample)
            print(f"Sample {i}: {text}")
        
        print("-" * 40)


def generate_text_with_model(model: BaseNGramModel, num_samples: int = 5, 
                           max_length: int = 20, method: str = "random") -> List[List[str]]:
    """Convenience function to generate text with a model."""
    generator = TextGenerator(model)
    
    if method == "random":
        return generator.generate_multiple_samples(num_samples, max_length)
    elif method == "beam":
        return generator.generate_beam_search(max_length, beam_size=num_samples)
    else:
        raise ValueError(f"Unknown generation method: {method}")

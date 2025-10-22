"""
Vocabulary Management for N-gram Language Modeling
=================================================

Handles vocabulary creation, word-to-index mapping, and vocabulary statistics.
"""

from typing import Dict, Set, List, Any
from collections import Counter
import pickle
import os


class Vocabulary:
    """Manages vocabulary for N-gram models."""
    
    def __init__(self, unk_token: str = '<unk>', start_token: str = '<s>', 
                 end_token: str = '</s>', pad_token: str = '<pad>'):
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.vocab_size = 0
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self) -> None:
        """Add special tokens to vocabulary."""
        special_tokens = [self.unk_token, self.start_token, self.end_token, self.pad_token]
        for token in special_tokens:
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
    
    def add_word(self, word: str) -> int:
        """Add a word to vocabulary and return its index."""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        return self.word_to_idx[word]
    
    def add_words(self, words: List[str]) -> None:
        """Add multiple words to vocabulary."""
        for word in words:
            self.add_word(word)
    
    def get_index(self, word: str) -> int:
        """Get index of a word, return UNK index if not found."""
        return self.word_to_idx.get(word, self.word_to_idx[self.unk_token])
    
    def get_word(self, idx: int) -> str:
        """Get word from index."""
        return self.idx_to_word.get(idx, self.unk_token)
    
    def build_from_sentences(self, sentences: List[List[str]], 
                           min_freq: int = 1) -> None:
        """Build vocabulary from sentences with frequency threshold."""
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
        
        # Add words that meet frequency threshold
        for word, count in self.word_counts.items():
            if count >= min_freq:
                self.add_word(word)
        
        self.vocab_size = len(self.word_to_idx)
    
    def filter_vocabulary(self, min_freq: int = 1) -> None:
        """Filter vocabulary by minimum frequency."""
        # Keep only words that meet frequency threshold
        filtered_words = {word for word, count in self.word_counts.items() 
                         if count >= min_freq}
        
        # Rebuild vocabulary
        self.word_to_idx.clear()
        self.idx_to_word.clear()
        self._add_special_tokens()
        
        for word in filtered_words:
            self.add_word(word)
        
        self.vocab_size = len(self.word_to_idx)
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        total_words = sum(self.word_counts.values())
        unique_words = len(self.word_counts)
        
        return {
            'vocab_size': self.vocab_size,
            'total_words': total_words,
            'unique_words': unique_words,
            'unk_count': self.word_counts.get(self.unk_token, 0),
            'unk_ratio': self.word_counts.get(self.unk_token, 0) / total_words if total_words > 0 else 0,
            'avg_word_freq': total_words / unique_words if unique_words > 0 else 0,
            'min_freq': min(self.word_counts.values()) if self.word_counts else 0,
            'max_freq': max(self.word_counts.values()) if self.word_counts else 0
        }
    
    def print_vocabulary_stats(self) -> None:
        """Print vocabulary statistics."""
        stats = self.get_vocabulary_stats()
        print("Vocabulary Statistics:")
        print(f"  Vocabulary size: {stats['vocab_size']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Unique words: {stats['unique_words']:,}")
        print(f"  UNK count: {stats['unk_count']:,}")
        print(f"  UNK ratio: {stats['unk_ratio']:.4f}")
        print(f"  Average word frequency: {stats['avg_word_freq']:.2f}")
        print(f"  Min frequency: {stats['min_freq']}")
        print(f"  Max frequency: {stats['max_freq']}")
    
    def get_most_frequent_words(self, n: int = 20) -> List[tuple]:
        """Get most frequent words."""
        return self.word_counts.most_common(n)
    
    def get_rare_words(self, max_freq: int = 5) -> List[str]:
        """Get words with frequency <= max_freq."""
        return [word for word, count in self.word_counts.items() 
                if count <= max_freq]
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_counts': self.word_counts,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'unk_token': self.unk_token,
                'start_token': self.start_token,
                'end_token': self.end_token,
                'pad_token': self.pad_token
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load(self, filepath: str) -> None:
        """Load vocabulary from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.word_counts = vocab_data['word_counts']
        self.vocab_size = vocab_data['vocab_size']
        
        special_tokens = vocab_data['special_tokens']
        self.unk_token = special_tokens['unk_token']
        self.start_token = special_tokens['start_token']
        self.end_token = special_tokens['end_token']
        self.pad_token = special_tokens['pad_token']

"""
Text Preprocessing for N-gram Language Modeling
==============================================

Handles text preprocessing including tokenization, sentence boundaries,
and vocabulary management.
"""

import re
from typing import List, Set, Dict, Any
from collections import Counter


class TextPreprocessor:
    """Handles text preprocessing for N-gram models."""
    
    def __init__(self, min_word_freq: int = 2, unk_token: str = '<unk>',
                 start_token: str = '<s>', end_token: str = '</s>'):
        self.min_word_freq = min_word_freq
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = set()
        self.word_counts = Counter()
        self.vocab_size = 0
    
    def load_data(self, file_path: str) -> List[List[str]]:
        """Load and preprocess data from file."""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by whitespace (already tokenized in PTB)
                    tokens = line.split()
                    sentences.append(tokens)
        return sentences
    
    def build_vocabulary(self, sentences: List[List[str]]) -> None:
        """Build vocabulary from training sentences."""
        # Count word frequencies
        for sentence in sentences:
            for token in sentence:
                self.word_counts[token] += 1
        
        # Create vocabulary with frequency threshold
        self.vocab = {word for word, count in self.word_counts.items() 
                     if count >= self.min_word_freq}
        
        # Add special tokens
        self.vocab.add(self.unk_token)
        self.vocab.add(self.start_token)
        self.vocab.add(self.end_token)
        
        self.vocab_size = len(self.vocab)
    
    def preprocess_sentence(self, sentence: List[str], n: int = 1) -> List[str]:
        """Preprocess a single sentence for N-gram modeling."""
        # Replace rare words with UNK
        processed = []
        for token in sentence:
            if token in self.vocab:
                processed.append(token)
            else:
                processed.append(self.unk_token)
        
        # Add sentence boundaries
        return [self.start_token] * (n - 1) + processed + [self.end_token]
    
    def preprocess_sentences(self, sentences: List[List[str]], n: int = 1) -> List[List[str]]:
        """Preprocess multiple sentences."""
        return [self.preprocess_sentence(sentence, n) for sentence in sentences]
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            'vocab_size': self.vocab_size,
            'total_words': sum(self.word_counts.values()),
            'unique_words': len(self.word_counts),
            'unk_ratio': self.word_counts.get(self.unk_token, 0) / sum(self.word_counts.values()),
            'min_freq': self.min_word_freq
        }
    
    def print_vocabulary_stats(self) -> None:
        """Print vocabulary statistics."""
        stats = self.get_vocabulary_stats()
        print("Vocabulary Statistics:")
        print(f"  Vocabulary size: {stats['vocab_size']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Unique words: {stats['unique_words']:,}")
        print(f"  UNK ratio: {stats['unk_ratio']:.4f}")
        print(f"  Min frequency threshold: {stats['min_freq']}")


def load_ptb_data(train_path: str, valid_path: str, test_path: str, 
                  min_word_freq: int = 2) -> tuple:
    """Load and preprocess Penn Treebank data."""
    preprocessor = TextPreprocessor(min_word_freq=min_word_freq)
    
    # Load raw data
    train_sentences = preprocessor.load_data(train_path)
    valid_sentences = preprocessor.load_data(valid_path)
    test_sentences = preprocessor.load_data(test_path)
    
    # Build vocabulary from training data
    preprocessor.build_vocabulary(train_sentences)
    
    print(f"Loaded {len(train_sentences)} training sentences")
    print(f"Loaded {len(valid_sentences)} validation sentences")
    print(f"Loaded {len(test_sentences)} test sentences")
    preprocessor.print_vocabulary_stats()
    
    return preprocessor, train_sentences, valid_sentences, test_sentences

#!/usr/bin/env python3
"""
Main execution script for N-gram Language Modeling
=================================================

Trains/evaluates all models and prints a results table.

Usage:
    python run_train_eval.py

This script:
1. Loads Penn Treebank dataset
2. Trains all N-gram models (MLE, Add-1, Linear Interpolation, Stupid Backoff)
3. Tunes hyperparameters using validation data
4. Evaluates all models on test data
5. Generates text samples using the best model
6. Prints comprehensive results table
"""

import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import load_ptb_data
from src.eval import ModelEvaluator
from src.generate import TextGenerator


def print_banner():
    """Print project banner."""
    print("=" * 80)
    print("N-GRAM LANGUAGE MODELING AND EVALUATION")
    print("=" * 80)
    print("Penn Treebank Dataset Analysis")
    print("Implementing MLE, Add-1, Linear Interpolation, and Stupid Backoff")
    print("=" * 80)
    print()


def print_section_header(section_name: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f" {section_name}")
    print("=" * 60)


def main():
    """Main execution pipeline."""
    start_time = time.time()
    
    # Print banner
    print_banner()
    
    # Check if dataset exists
    data_dir = "data"
    required_files = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]
    
    print("Checking dataset availability...")
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"ERROR: Required file {file_path} not found!")
            print("Please ensure the Penn Treebank dataset is in the data/ directory.")
            return 1
        else:
            print(f"✓ Found {file}")
    
    print("\nDataset verification complete!")
    
    try:
        # Phase 1: Data Loading and Preprocessing
        print_section_header("PHASE 1: DATA LOADING AND PREPROCESSING")
        print("Loading and preprocessing Penn Treebank dataset...")
        
        # Load data
        preprocessor, train_sentences, valid_sentences, test_sentences = load_ptb_data(
            "data/ptb.train.txt", "data/ptb.valid.txt", "data/ptb.test.txt", min_word_freq=2
        )
        
        # Preprocess sentences for different N-gram orders
        print("\nPreprocessing sentences for N-gram modeling...")
        train_processed = preprocessor.preprocess_sentences(train_sentences, n=3)
        valid_processed = preprocessor.preprocess_sentences(valid_sentences, n=3)
        test_processed = preprocessor.preprocess_sentences(test_sentences, n=3)
        
        print(f"Preprocessed {len(train_processed)} training sentences")
        print(f"Preprocessed {len(valid_processed)} validation sentences")
        print(f"Preprocessed {len(test_processed)} test sentences")
        
        # Phase 2: Model Training and Evaluation
        print_section_header("PHASE 2: MODEL TRAINING AND EVALUATION")
        print("Training and evaluating all N-gram models...")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(train_processed, valid_processed, test_processed)
        
        # Run complete evaluation
        evaluation_results = evaluator.evaluate_all_models()
        
        # Phase 3: Text Generation
        print_section_header("PHASE 3: TEXT GENERATION")
        print("Generating text samples using the best model...")
        
        # Get best model and generate text
        best_model_name, best_model = evaluator.get_best_model()
        generator = TextGenerator(best_model)
        
        # Generate samples
        samples = generator.generate_multiple_samples(num_samples=5, max_length=20)
        generator.print_generated_samples(samples, f"Generated Text Samples ({best_model_name})")
        
        # Analyze generated text
        analysis = generator.analyze_generated_text(samples)
        print(f"\nText Quality Analysis:")
        print(f"  Average length: {analysis['avg_length']:.1f} tokens")
        print(f"  Vocabulary diversity: {analysis['diversity_ratio']:.3f}")
        print(f"  Repetition rate: {analysis['repetition_rate']:.3f}")
        print(f"  Average perplexity: {analysis['avg_perplexity']:.2f}")
        
        # Phase 4: Results Summary
        print_section_header("PHASE 4: RESULTS SUMMARY")
        
        # Print final results
        evaluator.print_results_table()
        
        # Save results
        evaluator.save_results("results.csv")
        
        # Note about visualizations
        print("\nVisualizations generated:")
        print("  - perplexity_comparison.png: Bar charts comparing model performance")
        print("  - ngram_order_analysis.png: Line plot showing N-gram order impact")
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nEXECUTION TIME: {execution_time:.2f} seconds")
        
        # Print next steps
        print("\nNEXT STEPS:")
        print("-" * 40)
        print("1. Review the results table above")
        print("2. Check 'results.csv' for detailed numerical results")
        print("3. Examine generated text samples")
        print("4. Use individual modules for further experimentation")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        return 1
        
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print("Please check the error message and try again.")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

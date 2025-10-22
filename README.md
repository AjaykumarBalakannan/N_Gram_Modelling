# N-Gram Language Modeling and Evaluation

This project implements and evaluates various N-gram Language Models (LMs) to understand the trade-offs between different N-gram orders and the critical role of smoothing and backoff techniques in handling data sparsity.

## Dataset

- **Source**: Penn Treebank dataset from Kaggle
- **Files**: 
  - `data/train.txt` - Training data
  - `data/valid.txt` - Validation data (for hyperparameter tuning)
  - `data/test.txt` - Test data (for final evaluation)

## Project Structure

```
N_Gram_Modelling/
├── data/                      # Dataset directory
│   ├── train.txt             # Training data
│   ├── valid.txt             # Validation data
│   └── test.txt              # Test data
├── src/                      # Source code directory
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Utility functions
│   ├── preprocessing.py      # Text preprocessing
│   ├── vocabulary.py         # Vocabulary management
│   ├── ngram_counts.py       # N-gram counting
│   ├── models.py             # N-gram model implementations
│   ├── eval.py               # Model evaluation
│   └── generate.py           # Text generation
├── run_train_eval.py         # Main execution script
├── report_template.md        # Report template for submission
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features Implemented

### 1. Maximum Likelihood Estimation (MLE) Models
- Unigram (N=1), Bigram (N=2), Trigram (N=3), 4-gram (N=4) models
- Handles zero probabilities by reporting INF perplexity
- Demonstrates the necessity of smoothing

### 2. Smoothing and Backoff Strategies
- **Add-1 (Laplace) Smoothing**: For trigram models
- **Linear Interpolation**: Combines unigram, bigram, and trigram probabilities
- **Stupid Backoff**: Implements the Stupid Backoff algorithm
- Hyperparameter tuning using validation data

### 3. Evaluation Metrics
- **Perplexity**: Primary evaluation metric (lower is better)
- Comprehensive comparison across all models
- Statistical analysis of results

### 4. Text Generation
- Probabilistic text generation using best model
- Multiple generation methods (random, beam search, temperature sampling)
- Quality analysis of generated text

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd /Users/aj/Documents/N_Gram_Modelling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset is in place**
   ```bash
   ls data/
   # Should show: train.txt, valid.txt, test.txt
   ```

## Usage

### Quick Start
Run the complete evaluation and analysis pipeline:

```bash
python run_train_eval.py
```

This will:
1. Load and preprocess Penn Treebank dataset
2. Train all N-gram models (MLE, Add-1, Linear Interpolation, Stupid Backoff)
3. Tune hyperparameters using validation data
4. Evaluate all models on test data
5. Generate text samples using the best model
6. Print comprehensive results table

### Individual Module Usage

#### Using Preprocessing
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(min_word_freq=2)
sentences = preprocessor.load_data("data/train.txt")
preprocessor.build_vocabulary(sentences)
processed = preprocessor.preprocess_sentences(sentences, n=3)
```

#### Using Models
```python
from src.models import MLEModel, AddOneModel, LinearInterpolationModel, StupidBackoffModel

# Train a model
model = MLEModel(3)  # Trigram model
model.train(sentences)

# Calculate perplexity
perplexity = model.perplexity(test_sentences)
print(f"Perplexity: {perplexity:.2f}")
```

#### Using Text Generation
```python
from src.generate import TextGenerator

generator = TextGenerator(model)
samples = generator.generate_multiple_samples(num_samples=5, max_length=20)
for sample in samples:
    print(' '.join(sample))
```

#### Using Evaluation
```python
from src.eval import ModelEvaluator

evaluator = ModelEvaluator(train_data, dev_data, test_data)
results = evaluator.evaluate_all_models()
evaluator.print_results_table()
```

## Model Implementations

### MLEModel
```python
model = MLEModel(n=3)  # Trigram model
model.train(sentences)
perplexity = model.perplexity(test_sentences)
```

### AddOneModel (Laplace Smoothing)
```python
model = AddOneModel(n=3)
model.train(sentences)
perplexity = model.perplexity(test_sentences)
```

### LinearInterpolationModel
```python
lambdas = [0.1, 0.3, 0.6]  # Must sum to 1.0
model = LinearInterpolationModel(lambdas)
model.train(sentences)
perplexity = model.perplexity(test_sentences)
```

### StupidBackoffModel
```python
model = StupidBackoffModel(alpha=0.4)
model.train(sentences)
perplexity = model.perplexity(test_sentences)
```

## Output Files

After running the complete pipeline, you'll get:

1. **`results.csv`** - Detailed perplexity results for all models
2. **Console output** - Comprehensive results table and analysis
3. **Generated text samples** - 5+ samples from the best model

## Key Results Expected

### Perplexity Trends
- MLE models: Higher order → better performance until data sparsity kicks in
- Smoothed models: Significantly better than unsmoothed MLE
- Best performance: Usually linear interpolation or stupid backoff

### Generated Text Quality
- Coherent word sequences
- Vocabulary diversity analysis
- Repetition rate assessment

## Technical Details

### Preprocessing
- **Tokenization**: Space-separated (pre-processed dataset)
- **Sentence Boundaries**: Added `<s>` and `</s>` markers
- **Unknown Words**: Handled with `<unk>` token
- **Numbers**: Replaced with `N` token
- **Vocabulary Filtering**: Minimum frequency threshold

### Numerical Stability
- All probability calculations use log-space to avoid underflow
- Perplexity calculated as exp(-average_log_probability)
- Safe handling of zero probabilities

### Hyperparameter Tuning
- **Linear Interpolation**: Grid search over lambda combinations
- **Stupid Backoff**: Grid search over alpha values
- **Optimization**: Minimize perplexity on validation data

## Assignment Requirements Checklist

- [x] MLE Models (N=1,2,3,4) with zero probability handling
- [x] Add-1 Smoothing implementation
- [x] Linear Interpolation with lambda tuning
- [x] Stupid Backoff with alpha tuning
- [x] Perplexity evaluation on test data
- [x] Hyperparameter tuning using validation data
- [x] Text generation from best model (5+ samples)
- [x] Comprehensive analysis and reporting
- [x] Preprocessing and vocabulary decisions documentation
- [x] N-gram order impact analysis
- [x] Smoothing strategy comparison
- [x] Generated text quality assessment
- [x] Well-commented, modular, runnable code
- [x] Detailed README with execution instructions
- [x] Report template for submission

## Module Documentation

### Core Modules
- **`utils.py`**: Common utility functions (logging, math, sampling)
- **`preprocessing.py`**: Text preprocessing and vocabulary management
- **`vocabulary.py`**: Advanced vocabulary handling and statistics
- **`ngram_counts.py`**: N-gram counting and statistics
- **`models.py`**: All N-gram model implementations
- **`eval.py`**: Model evaluation and hyperparameter tuning
- **`generate.py`**: Text generation with multiple methods

### Main Script
- **`run_train_eval.py`**: Complete evaluation pipeline

## Troubleshooting

### Common Issues

1. **Memory Issues with Large N-grams**
   - Reduce vocabulary size by filtering rare words
   - Use more aggressive smoothing

2. **Infinite Perplexity**
   - Expected for unsmoothed MLE models
   - Indicates necessity of smoothing

3. **Poor Text Generation Quality**
   - Try different sampling strategies
   - Adjust temperature parameters

### Performance Tips

1. **Faster Training**: Use smaller vocabulary for initial experiments
2. **Better Results**: Tune hyperparameters more extensively
3. **Memory Efficiency**: Use sparse data structures for large vocabularies

## License

This project is created for educational purposes as part of an NLP assignment.

## Contact

For questions or issues, please refer to the code comments or create an issue in the project repository.
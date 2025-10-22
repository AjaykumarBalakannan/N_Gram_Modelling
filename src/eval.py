"""
Model Evaluation for N-gram Language Models
==========================================

Handles evaluation, hyperparameter tuning, and comparison of N-gram models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import itertools

from .models import MLEModel, AddOneModel, LinearInterpolationModel, StupidBackoffModel
from .utils import format_perplexity, print_progress_bar


class ModelEvaluator:
    """Handles evaluation of N-gram models."""
    
    def __init__(self, train_data: List[List[str]], 
                 dev_data: List[List[str]], 
                 test_data: List[List[str]]):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.results = {}
        self.best_models = {}
    
    def evaluate_mle_models(self) -> Dict[str, float]:
        """Evaluate MLE models for N=1,2,3,4."""
        print("Evaluating MLE models...")
        results = {}
        
        for n in [1, 2, 3, 4]:
            print(f"Training {n}-gram MLE model...")
            model = MLEModel(n)
            model.train(self.train_data)
            
            # Evaluate on validation data
            val_perplexity = model.perplexity(self.dev_data)
            results[f'MLE_{n}gram_val'] = val_perplexity
            
            # Evaluate on test data
            test_perplexity = model.perplexity(self.test_data)
            results[f'MLE_{n}gram'] = test_perplexity
            
            print(f"{n}-gram MLE validation perplexity: {format_perplexity(val_perplexity)}")
            print(f"{n}-gram MLE test perplexity: {format_perplexity(test_perplexity)}")
        
        self.results.update(results)
        return results
    
    def evaluate_add1_model(self) -> Dict[str, float]:
        """Evaluate Add-1 smoothing model."""
        print("Evaluating Add-1 smoothing model...")
        
        model = AddOneModel(3)  # Trigram model
        model.train(self.train_data)
        
        # Evaluate on validation data
        val_perplexity = model.perplexity(self.dev_data)
        # Evaluate on test data
        test_perplexity = model.perplexity(self.test_data)
        
        results = {
            'Add1_3gram_val': val_perplexity,
            'Add1_3gram': test_perplexity
        }
        
        print(f"Add-1 3-gram validation perplexity: {format_perplexity(val_perplexity)}")
        print(f"Add-1 3-gram test perplexity: {format_perplexity(test_perplexity)}")
        
        self.results.update(results)
        self.best_models['Add1_3gram'] = model
        return results
    
    def tune_linear_interpolation(self) -> Tuple[Dict[str, float], List[float]]:
        """Tune lambda weights for linear interpolation using dev data."""
        print("Tuning linear interpolation weights...")
        
        # Try different lambda combinations
        lambda_combinations = [
            [0.1, 0.3, 0.6],  # Favor trigram
            [0.2, 0.3, 0.5],  # Balanced
            [0.3, 0.3, 0.4],  # More unigram
            [0.1, 0.2, 0.7],  # Strong trigram
            [0.4, 0.3, 0.3],  # More unigram
        ]
        
        best_perplexity = float('inf')
        best_lambdas = None
        best_model = None
        dev_results = {}
        
        for lambdas in tqdm(lambda_combinations, desc="Testing lambda combinations"):
            try:
                model = LinearInterpolationModel(lambdas)
                model.train(self.train_data)
                
                perplexity = model.perplexity(self.dev_data)
                dev_results[str(lambdas)] = perplexity
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_lambdas = lambdas
                    best_model = model
                    
            except Exception as e:
                print(f"Error with lambdas {lambdas}: {e}")
                continue
        
        print(f"Best lambda combination: {best_lambdas}")
        print(f"Best dev perplexity: {format_perplexity(best_perplexity)}")
        
        # Evaluate best model on test data
        if best_model is not None:
            test_perplexity = best_model.perplexity(self.test_data)
        else:
            # If no model worked, create a default one
            best_lambdas = [0.1, 0.3, 0.6]  # Default values
            best_model = LinearInterpolationModel(best_lambdas)
            best_model.train(self.train_data)
            test_perplexity = best_model.perplexity(self.test_data)
        
        results = {
            'LinearInterp_3gram': test_perplexity,
            'LinearInterp_dev': best_perplexity
        }
        
        print(f"Linear interpolation test perplexity: {format_perplexity(test_perplexity)}")
        
        self.results.update(results)
        self.best_models['LinearInterp_3gram'] = best_model
        return results, best_lambdas
    
    def tune_stupid_backoff(self) -> Tuple[Dict[str, float], float]:
        """Tune alpha parameter for Stupid backoff using dev data."""
        print("Tuning Stupid backoff alpha parameter...")
        
        # Try different alpha values
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        best_perplexity = float('inf')
        best_alpha = None
        best_model = None
        dev_results = {}
        
        for alpha in tqdm(alpha_values, desc="Testing alpha values"):
            try:
                model = StupidBackoffModel(alpha)
                model.train(self.train_data)
                
                perplexity = model.perplexity(self.dev_data)
                dev_results[alpha] = perplexity
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_alpha = alpha
                    best_model = model
                    
            except Exception as e:
                print(f"Error with alpha {alpha}: {e}")
                continue
        
        print(f"Best alpha: {best_alpha}")
        print(f"Best dev perplexity: {format_perplexity(best_perplexity)}")
        
        # Evaluate best model on test data
        if best_model is not None:
            test_perplexity = best_model.perplexity(self.test_data)
        else:
            # If no model worked, create a default one
            best_alpha = 0.4  # Default value
            best_model = StupidBackoffModel(best_alpha)
            best_model.train(self.train_data)
            test_perplexity = best_model.perplexity(self.test_data)
        
        results = {
            'StupidBackoff_3gram': test_perplexity,
            'StupidBackoff_dev': best_perplexity
        }
        
        print(f"Stupid backoff test perplexity: {format_perplexity(test_perplexity)}")
        
        self.results.update(results)
        self.best_models['StupidBackoff_3gram'] = best_model
        return results, best_alpha
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model based on test perplexity."""
        # Filter out dev results
        test_results = {k: v for k, v in self.results.items() if 'dev' not in k}
        
        if not test_results:
            raise ValueError("No test results available")
        
        # Find model with lowest perplexity (excluding infinite values)
        finite_results = {k: v for k, v in test_results.items() if v != float('inf')}
        
        if not finite_results:
            raise ValueError("All models have infinite perplexity")
        
        best_model_name = min(finite_results.keys(), key=lambda k: finite_results[k])
        best_perplexity = finite_results[best_model_name]
        
        print(f"Best model: {best_model_name} (perplexity: {format_perplexity(best_perplexity)})")
        
        # Get the actual model
        if best_model_name in self.best_models:
            return best_model_name, self.best_models[best_model_name]
        else:
            # For MLE models, we need to retrain
            if best_model_name.startswith('MLE_'):
                n = int(best_model_name.split('_')[1].replace('gram', ''))
                model = MLEModel(n)
                model.train(self.train_data)
                return best_model_name, model
            else:
                raise ValueError(f"Model {best_model_name} not found in best_models")
    
    def print_results_table(self) -> None:
        """Print a formatted results table."""
        print("\n" + "="*80)
        print("N-GRAM LANGUAGE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Separate validation and test results
        val_results = {k: v for k, v in self.results.items() if '_val' in k}
        test_results = {k: v for k, v in self.results.items() if '_val' not in k and 'dev' not in k}
        
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<25s} {'Validation':<15s} {'Test':<15s} {'Improvement':<15s}")
        print("-" * 80)
        
        # Create comparison table
        for model_name in sorted(test_results.keys()):
            val_name = model_name + '_val'
            test_perplexity = test_results[model_name]
            val_perplexity = val_results.get(val_name, float('inf'))
            
            # Calculate improvement (negative means test is worse)
            if val_perplexity != float('inf') and test_perplexity != float('inf'):
                improvement = ((val_perplexity - test_perplexity) / val_perplexity) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{model_name:<25s} {format_perplexity(val_perplexity):<15s} {format_perplexity(test_perplexity):<15s} {improvement_str:<15s}")
        
        print("="*80)
        
        # Additional summary
        print("\nSummary:")
        print("-" * 40)
        finite_test_results = {k: v for k, v in test_results.items() if v != float('inf')}
        if finite_test_results:
            best_model = min(finite_test_results.keys(), key=lambda k: finite_test_results[k])
            best_perplexity = finite_test_results[best_model]
            print(f"Best Test Model: {best_model} (perplexity: {format_perplexity(best_perplexity)})")
        
        finite_val_results = {k: v for k, v in val_results.items() if v != float('inf')}
        if finite_val_results:
            best_val_model = min(finite_val_results.keys(), key=lambda k: val_results[k])
            best_val_perplexity = val_results[best_val_model]
            print(f"Best Validation Model: {best_val_model} (perplexity: {format_perplexity(best_val_perplexity)})")
        
        print("="*80)
    
    def save_results(self, filename: str = "results.csv") -> None:
        """Save results to CSV file."""
        df = pd.DataFrame(list(self.results.items()), columns=['Model', 'Perplexity'])
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get model comparison as DataFrame."""
        test_results = {k: v for k, v in self.results.items() if '_val' not in k and 'dev' not in k}
        val_results = {k: v for k, v in self.results.items() if '_val' in k}
        
        data = []
        for model_name in test_results.keys():
            val_name = model_name + '_val'
            test_perplexity = test_results[model_name]
            val_perplexity = val_results.get(val_name, float('inf'))
            
            data.append({
                'Model': model_name,
                'Validation_Perplexity': val_perplexity,
                'Test_Perplexity': test_perplexity,
                'Validation_Formatted': format_perplexity(val_perplexity),
                'Test_Formatted': format_perplexity(test_perplexity)
            })
        
        return pd.DataFrame(data).sort_values('Test_Perplexity')
    
    def create_perplexity_comparison_plot(self, save_path: str = "perplexity_comparison.png") -> None:
        """Create bar plot comparing model perplexities."""
        df = self.get_model_comparison()
        
        # Filter out infinite values for plotting
        plot_df = df[df['Test_Perplexity'] != float('inf')].copy()
        
        if plot_df.empty:
            print("No finite perplexity values to plot")
            return
        
        # Set up the plot
        plt.figure(figsize=(14, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Test set perplexities
        models = plot_df['Model'].tolist()
        test_perplexities = plot_df['Test_Perplexity'].tolist()
        
        bars1 = ax1.bar(range(len(models)), test_perplexities, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Test Set Perplexity Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, test_perplexities)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_perplexities)*0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Validation vs Test comparison
        val_perplexities = plot_df['Validation_Perplexity'].tolist()
        val_perplexities = [v if v != float('inf') else 0 for v in val_perplexities]  # Replace inf with 0 for plotting
        
        x = np.arange(len(models))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, val_perplexities, width, label='Validation', color='lightcoral', alpha=0.7)
        bars3 = ax2.bar(x + width/2, test_perplexities, width, label='Test', color='lightgreen', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Validation vs Test Perplexity Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, val_perplexities):
            if val > 0:  # Only label non-zero values
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_perplexities + test_perplexities)*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars3, test_perplexities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_perplexities + test_perplexities)*0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Perplexity comparison plot saved to {save_path}")
    
    def create_ngram_order_plot(self, save_path: str = "ngram_order_analysis.png") -> None:
        """Create plot showing impact of N-gram order."""
        # Extract MLE results
        mle_results = {k: v for k, v in self.results.items() if k.startswith('MLE_') and '_val' not in k and 'dev' not in k}
        
        if not mle_results:
            print("No MLE results found for N-gram order analysis")
            return
        
        # Extract N-gram orders and perplexities
        orders = []
        perplexities = []
        
        for model_name, perplexity in mle_results.items():
            if 'MLE_' in model_name and 'gram' in model_name:
                n = int(model_name.split('_')[1].replace('gram', ''))
                orders.append(n)
                perplexities.append(perplexity if perplexity != float('inf') else None)
        
        if not orders:
            print("No valid MLE results found")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Filter out None values for plotting
        plot_orders = [o for o, p in zip(orders, perplexities) if p is not None]
        plot_perplexities = [p for p in perplexities if p is not None]
        
        if plot_orders:
            plt.plot(plot_orders, plot_perplexities, 'bo-', linewidth=2, markersize=8, label='MLE Models')
            plt.scatter(plot_orders, plot_perplexities, s=100, zorder=5)
            
            # Add value labels
            for i, (order, perplexity) in enumerate(zip(plot_orders, plot_perplexities)):
                plt.annotate(f'{perplexity:.1f}', (order, perplexity), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # Mark infinite values
        inf_orders = [o for o, p in zip(orders, perplexities) if p is None]
        if inf_orders:
            plt.scatter(inf_orders, [max(plot_perplexities) * 1.1] * len(inf_orders), 
                       s=100, marker='x', color='red', label='Infinite Perplexity')
            for order in inf_orders:
                plt.annotate('INF', (order, max(plot_perplexities) * 1.1), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('N-gram Order')
        plt.ylabel('Perplexity')
        plt.title('Impact of N-gram Order on Perplexity (MLE Models)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Use log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"N-gram order analysis plot saved to {save_path}")
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Run complete evaluation of all models."""
        print("Starting comprehensive model evaluation...")
        
        # Evaluate all models
        mle_results = self.evaluate_mle_models()
        add1_results = self.evaluate_add1_model()
        linear_results, best_lambdas = self.tune_linear_interpolation()
        backoff_results, best_alpha = self.tune_stupid_backoff()
        
        # Print results
        self.print_results_table()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_perplexity_comparison_plot()
        self.create_ngram_order_plot()
        
        # Get best model
        best_model_name, best_model = self.get_best_model()
        
        return {
            'results': self.results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_lambdas': best_lambdas,
            'best_alpha': best_alpha,
            'comparison_df': self.get_model_comparison()
        }

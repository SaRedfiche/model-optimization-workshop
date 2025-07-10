#!/usr/bin/env python
"""
Model Optimization Workshop: Model Pruning

This script performs the same operations as the 03_pruning.ipynb notebook.
It demonstrates how to prune a Hugging Face model to reduce its size.
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Machine learning
import torch
import torch.nn.utils.prune as prune
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

def setup_environment():
    """Set up the environment and return configuration."""
    # Load workshop configuration if available
    try:
        with open('workshop_config.json', 'r') as f:
            workshop_config = json.load(f)
        
        # Use configuration values
        base_model = workshop_config.get("base_model", "distilbert-base-uncased-finetuned-sst-2-english")
        task = workshop_config.get("task", "sequence-classification")
    except FileNotFoundError:
        # Default values if config not found
        base_model = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sequence-classification"
        print("Workshop configuration not found. Using default values.")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return {
        "base_model": base_model,
        "task": task,
        "device": device
    }

def load_model(config):
    """Load the model and tokenizer."""
    base_model = config["base_model"]
    device = config["device"]
    
    print(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    
    # Move model to device
    model = model.to(device)
    
    return model, tokenizer

def evaluate_model(model, tokenizer, config):
    """Evaluate the model on sample inputs."""
    device = config["device"]
    
    # Create a sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Test with a few examples
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Completely disappointed with the quality.",
        "It's okay, not great but not bad either.",
        "The customer service was excellent but the product was mediocre."
    ]
    
    # Run inference
    results = sentiment_pipeline(test_texts)
    
    # Display results
    for text, result in zip(test_texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})\n")
    
    return test_texts, results

def get_model_size(model):
    """Get the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def measure_inference_time(model, tokenizer, test_texts, config, num_runs=100, warmup_runs=10):
    """Measure the inference time of the model."""
    device = config["device"]
    
    # Tokenize inputs
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(**inputs)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(**inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # Convert to milliseconds

def apply_pruning(model, pruning_method, amount):
    """Apply pruning to the model."""
    print(f"Applying {pruning_method} pruning with amount {amount}...")
    
    # Get all Linear layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if pruning_method == "l1_unstructured":
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif pruning_method == "random_unstructured":
                prune.random_unstructured(module, name='weight', amount=amount)
            elif pruning_method == "ln_structured":
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    print("Pruning applied successfully.")
    return model

def make_pruning_permanent(model):
    """Make the pruning permanent by removing the pruning reparameterization."""
    print("Making pruning permanent...")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    
    print("Pruning made permanent.")
    return model

def save_model(model, tokenizer, pruning_method, amount):
    """Save the pruned model."""
    output_dir = f"pruned_models/{pruning_method}_{int(amount * 100)}percent"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    return output_dir

def compare_models(original_size, original_time, pruned_sizes, pruned_times, pruning_methods, amounts):
    """Compare the original and pruned models."""
    # Create labels for pruned models
    pruned_labels = [f"{method} {int(amount * 100)}%" for method, amount in zip(pruning_methods, amounts)]
    
    # Prepare data for plotting
    model_names = ["Original"] + pruned_labels
    sizes = [original_size] + pruned_sizes
    times = [original_time] + pruned_times
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot model sizes
    ax1.bar(model_names, sizes)
    ax1.set_title("Model Size Comparison")
    ax1.set_ylabel("Size (MB)")
    ax1.set_ylim(0, max(sizes) * 1.1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add size reduction percentages
    for i, size in enumerate(sizes[1:], 1):
        reduction = (original_size - size) / original_size * 100
        ax1.text(i, size + 0.1, f"-{reduction:.1f}%", ha='center')
    
    # Plot inference times
    ax2.bar(model_names, times)
    ax2.set_title("Inference Time Comparison")
    ax2.set_ylabel("Time (ms)")
    ax2.set_ylim(0, max(times) * 1.1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add time reduction percentages
    for i, time_val in enumerate(times[1:], 1):
        reduction = (original_time - time_val) / original_time * 100
        ax2.text(i, time_val + 0.1, f"-{reduction:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("pruning_comparison.png")
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prune a model to reduce its size")
    parser.add_argument("--method", choices=["l1_unstructured", "random_unstructured", "ln_structured"], 
                        default="l1_unstructured", help="Pruning method to use")
    parser.add_argument("--amount", type=float, default=0.3, 
                        help="Amount of weights to prune (0.0 to 1.0)")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare multiple pruning methods")
    return parser.parse_args()

def main():
    """Main function to run the pruning script."""
    args = parse_args()
    
    # Set up environment
    config = setup_environment()
    
    # Load model
    model, tokenizer = load_model(config)
    
    # Evaluate original model
    test_texts, _ = evaluate_model(model, tokenizer, config)
    
    # Get original model size and inference time
    original_size = get_model_size(model)
    original_time = measure_inference_time(model, tokenizer, test_texts, config)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original inference time: {original_time:.2f} ms")
    
    if args.compare:
        # Compare multiple pruning methods and amounts
        pruning_methods = ["l1_unstructured", "random_unstructured", "ln_structured"]
        amounts = [0.1, 0.3, 0.5]
        
        pruned_sizes = []
        pruned_times = []
        pruned_methods = []
        pruned_amounts = []
        
        for method in pruning_methods:
            for amount in amounts:
                # Load a fresh model for each pruning configuration
                model, tokenizer = load_model(config)
                
                # Apply pruning
                model = apply_pruning(model, method, amount)
                
                # Make pruning permanent
                model = make_pruning_permanent(model)
                
                # Get pruned model size and inference time
                pruned_size = get_model_size(model)
                pruned_time = measure_inference_time(model, tokenizer, test_texts, config)
                
                print(f"{method} {int(amount * 100)}% pruning:")
                print(f"  Size: {pruned_size:.2f} MB ({(original_size - pruned_size) / original_size * 100:.1f}% reduction)")
                print(f"  Inference time: {pruned_time:.2f} ms ({(original_time - pruned_time) / original_time * 100:.1f}% reduction)")
                
                pruned_sizes.append(pruned_size)
                pruned_times.append(pruned_time)
                pruned_methods.append(method)
                pruned_amounts.append(amount)
                
                # Save the model
                save_model(model, tokenizer, method, amount)
        
        # Compare models
        compare_models(original_size, original_time, pruned_sizes, pruned_times, pruned_methods, pruned_amounts)
    else:
        # Apply pruning with specified method and amount
        model = apply_pruning(model, args.method, args.amount)
        
        # Make pruning permanent
        model = make_pruning_permanent(model)
        
        # Get pruned model size and inference time
        pruned_size = get_model_size(model)
        pruned_time = measure_inference_time(model, tokenizer, test_texts, config)
        
        print(f"Pruned model size: {pruned_size:.2f} MB ({(original_size - pruned_size) / original_size * 100:.1f}% reduction)")
        print(f"Pruned inference time: {pruned_time:.2f} ms ({(original_time - pruned_time) / original_time * 100:.1f}% reduction)")
        
        # Evaluate pruned model
        print("\nEvaluating pruned model:")
        evaluate_model(model, tokenizer, config)
        
        # Save the model
        save_model(model, tokenizer, args.method, args.amount)

if __name__ == "__main__":
    main()

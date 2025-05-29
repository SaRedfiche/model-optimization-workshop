#!/usr/bin/env python
# coding: utf-8

"""
WANDA Pruning Script for SageMaker Processing

This script implements the WANDA (Weight ANalysis for Deep leArning) pruning technique,
which considers both weight magnitudes and activation statistics when deciding which weights to prune.
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM
import psutil
import gc


def parse_args():
    """Parse arguments for the pruning script."""
    parser = argparse.ArgumentParser(description="WANDA Pruning for Transformer Models")
    parser.add_argument(
        "--model-info-path",
        type=str,
        required=True,
        help="Path to the model info JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the pruned model and metrics",
    )
    parser.add_argument(
        "--pruning-amount",
        type=float,
        default=0.3,
        help="Amount of weights to prune (0.0 to 1.0)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=32,
        help="Number of calibration samples to use for activation statistics",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_info):
    """Load model and tokenizer based on model info."""
    model_key = list(model_info.keys())[0]
    info = model_info[model_key]
    
    model_name = info["model_name"]
    task = info.get("task", "text-classification")
    
    print(f"Loading model: {model_name} for task: {task}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on task
    if task == "text-classification" or task == "sentiment-analysis":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "token-classification" or task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task == "masked-lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return model, tokenizer, model_key, task


def generate_calibration_data(tokenizer, task, num_samples=32):
    """Generate calibration data for collecting activation statistics."""
    calibration_texts = [
        "The movie was excellent and I enjoyed every minute of it.",
        "I didn't like the book at all, it was boring.",
        "The restaurant had amazing food but poor service.",
        "The new smartphone has impressive features and battery life.",
        "The concert was disappointing and not worth the ticket price.",
        "This product exceeded my expectations in every way.",
        "The hotel room was clean, spacious, and had a great view.",
        "The customer service representative was unhelpful and rude.",
        "I highly recommend this software for its ease of use.",
        "The weather was perfect for our vacation at the beach."
    ]
    
    # Repeat or truncate to get the desired number of samples
    if len(calibration_texts) < num_samples:
        calibration_texts = calibration_texts * (num_samples // len(calibration_texts) + 1)
    calibration_texts = calibration_texts[:num_samples]
    
    # Tokenize based on task
    if task == "text-classification" or task == "sentiment-analysis":
        inputs = tokenizer(calibration_texts, padding=True, truncation=True, return_tensors="pt")
    elif task == "token-classification" or task == "ner":
        inputs = tokenizer(calibration_texts, padding=True, truncation=True, return_tensors="pt")
    elif task == "question-answering":
        questions = ["What is the main point?"] * num_samples
        inputs = tokenizer(questions, calibration_texts, padding=True, truncation=True, return_tensors="pt")
    elif task == "masked-lm":
        masked_texts = [text.replace("was", "[MASK]") for text in calibration_texts]
        inputs = tokenizer(masked_texts, padding=True, truncation=True, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return inputs


def collect_activation_statistics(model, inputs):
    """Collect activation statistics for WANDA pruning."""
    activations = {}
    hooks = []
    
    # Register forward hooks to collect activations
    def hook_fn(name):
        def hook(module, input, output):
            # Store the input activations
            if isinstance(input, tuple) and len(input) > 0:
                # Take the first element if input is a tuple
                act = input[0].detach().abs().mean(dim=0)
                if name in activations:
                    activations[name] = torch.max(activations[name], act)
                else:
                    activations[name] = act
        return hook
    
    # Register hooks for linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass to collect activations
    with torch.no_grad():
        if 'input_ids' in inputs:
            _ = model(**inputs)
        else:
            raise ValueError("Unexpected input format")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def wanda_pruning(model, activations, pruning_amount):
    """Apply WANDA pruning to the model."""
    total_params = 0
    pruned_params = 0
    
    # Apply pruning to each linear layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the weight tensor
            weight = module.weight.data
            
            # Get activation statistics for this layer
            if name in activations:
                act_stats = activations[name]
                
                # Expand dimensions to match weight shape
                if act_stats.dim() == 1:
                    act_stats = act_stats.unsqueeze(0).expand(weight.size(0), -1)
                
                # Calculate importance scores (weight magnitude × activation magnitude)
                importance = weight.abs() * act_stats
            else:
                # Fallback to simple magnitude pruning if no activation stats
                importance = weight.abs()
            
            # Determine threshold for pruning
            threshold = torch.quantile(importance.view(-1), pruning_amount)
            
            # Create pruning mask
            mask = importance > threshold
            
            # Apply mask to weights
            module.weight.data = module.weight.data * mask.float()
            
            # Count parameters
            total_params += weight.numel()
            pruned_params += (mask == 0).sum().item()
    
    sparsity = pruned_params / total_params if total_params > 0 else 0
    print(f"Applied WANDA pruning with {pruning_amount:.2f} target. Achieved sparsity: {sparsity:.4f}")
    
    return sparsity


def measure_model_size(model):
    """Measure the size of the model in MB."""
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb


def measure_inference_time(model, inputs, num_runs=10):
    """Measure inference time in milliseconds."""
    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(**inputs)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms


def measure_memory_usage(model, inputs):
    """Measure peak memory usage during inference in MB."""
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Get baseline memory usage
    baseline = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    # Run inference
    with torch.no_grad():
        _ = model(**inputs)
    
    # Get peak memory usage
    peak = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    return peak - baseline


def main():
    """Main function to run WANDA pruning."""
    args = parse_args()
    
    # Load model info
    with open(args.model_info_path, "r") as f:
        model_info = json.load(f)
    
    # Load model and tokenizer
    model, tokenizer, model_key, task = load_model_and_tokenizer(model_info)
    
    # Generate calibration data
    inputs = generate_calibration_data(tokenizer, task, args.calibration_samples)
    
    # Measure baseline metrics
    baseline_size = measure_model_size(model)
    baseline_time = measure_inference_time(model, inputs)
    baseline_memory = measure_memory_usage(model, inputs)
    
    print(f"Baseline metrics - Size: {baseline_size:.2f} MB, Inference time: {baseline_time:.2f} ms")
    
    # Collect activation statistics
    print("Collecting activation statistics...")
    activations = collect_activation_statistics(model, inputs)
    
    # Apply WANDA pruning
    print(f"Applying WANDA pruning with amount {args.pruning_amount}...")
    sparsity = wanda_pruning(model, activations, args.pruning_amount)
    
    # Measure pruned metrics
    pruned_size = measure_model_size(model)
    pruned_time = measure_inference_time(model, inputs)
    pruned_memory = measure_memory_usage(model, inputs)
    
    # Calculate improvements
    size_reduction = (baseline_size - pruned_size) / baseline_size * 100
    time_improvement = (baseline_time - pruned_time) / baseline_time * 100
    memory_reduction = (baseline_memory - pruned_memory) / baseline_memory * 100
    
    print(f"Pruned metrics - Size: {pruned_size:.2f} MB, Inference time: {pruned_time:.2f} ms")
    print(f"Improvements - Size: {size_reduction:.2f}%, Time: {time_improvement:.2f}%")
    
    # Save pruned model
    output_model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    # Save metrics
    metrics = {
        model_key: {
            "model_name": model_info[model_key]["model_name"],
            "pruning_method": "wanda",
            "pruning_amount": args.pruning_amount,
            "model_size": pruned_size,
            "inference_time": pruned_time,
            "memory_usage": pruned_memory,
            "size_reduction": size_reduction,
            "time_improvement": time_improvement,
            "memory_reduction": memory_reduction,
            "sparsity": sparsity * 100,  # Convert to percentage
            "parameter_reduction": sparsity * 100  # Same as sparsity for now
        }
    }
    
    with open(os.path.join(args.output_dir, "wanda-pruned-metrics.json"), "w") as f:  # Using hyphens instead of underscores
        json.dump(metrics, f, indent=2)
    
    print("WANDA pruning completed successfully!")


if __name__ == "__main__":
    main()

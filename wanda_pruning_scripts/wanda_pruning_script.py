#!/usr/bin/env python
# coding: utf-8

"""
WANDA Pruning Script for SageMaker Processing

This script implements the WANDA (Weight ANalysis for Deep leArning) pruning technique,
which considers both weight magnitudes and activation statistics when deciding which weights to prune.
Metrics collection and analysis are handled in the notebook.
"""

import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM
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
        help="Directory to save the pruned model",
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
    if task == "text-classification" or task == "sequence-classification" or task == "sentiment-analysis":
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
    if task == "text-classification" or task == "sentiment-analysis" or task == "sequence-classification":
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
                    # Make sure dimensions match before taking max
                    if activations[name].shape == act.shape:
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
                
                # Check if dimensions match
                if act_stats.dim() == 1 and act_stats.size(0) == weight.size(1):
                    # Expand dimensions to match weight shape
                    act_stats = act_stats.unsqueeze(0).expand(weight.size(0), -1)
                    
                    # Calculate importance scores (weight magnitude × activation magnitude)
                    importance = weight.abs() * act_stats
                else:
                    # Fallback to simple magnitude pruning if dimensions don't match
                    print(f"Warning: Activation shape {act_stats.shape} doesn't match weight shape {weight.shape} for {name}. Using magnitude pruning instead.")
                    importance = weight.abs()
            else:
                # Fallback to simple magnitude pruning if no activation stats
                print(f"Warning: No activation statistics found for {name}. Using magnitude pruning instead.")
                importance = weight.abs()
            
            # Determine threshold for pruning - handle large tensors
            try:
                # Try using torch.quantile first (faster)
                threshold = torch.quantile(importance.view(-1), pruning_amount)
            except RuntimeError:
                # If tensor is too large, use a different approach
                print(f"Warning: Tensor too large for quantile. Using alternative method for {name}.")
                flattened = importance.view(-1).detach().cpu().numpy()
                k = int(flattened.size * pruning_amount)
                threshold = float(np.partition(flattened, k)[k])
                threshold = torch.tensor(threshold, device=importance.device)
            
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
    
    # Collect activation statistics
    print("Collecting activation statistics...")
    activations = collect_activation_statistics(model, inputs)
    
    # Apply WANDA pruning
    print(f"Applying WANDA pruning with amount {args.pruning_amount}...")
    sparsity = wanda_pruning(model, activations, args.pruning_amount)
    
    # Save pruned model
    output_model_dir = os.path.join(args.output_dir, f"{model_key}_pruned")
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    print("WANDA pruning completed successfully!")


if __name__ == "__main__":
    main()

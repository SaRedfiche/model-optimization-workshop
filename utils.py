"""
Utility functions for the Model Optimization Workshop.
This file contains common functions used across multiple notebooks.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import psutil
import boto3
from typing import Dict, List, Tuple, Optional

def measure_inference_time(model, inputs, num_runs=100, warmup_runs=10):
    """Measure inference time for a model.
    
    Args:
        model: The model to measure
        inputs: Input tensors for the model
        num_runs: Number of inference runs to average over
        warmup_runs: Number of initial runs to discard (for GPU warmup)
        
    Returns:
        Average inference time in milliseconds
    """
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

def get_model_size(model):
    """Get model size in MB.
    
    Args:
        model: The model to measure
        
    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

# Note: We're no longer using measure_memory_usage as model_size is a good proxy

def plot_comparison(metrics, title, ylabel, higher_is_better=False):
    """Plot comparison of metrics across models.
    
    Args:
        metrics: Dictionary mapping model names to metric values
        title: Plot title
        ylabel: Y-axis label
        higher_is_better: Whether higher values are better
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by metric value
    sorted_items = sorted(metrics.items(), key=lambda x: x[1], reverse=higher_is_better)
    models = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create bar chart
    sns.barplot(x=models, y=values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()

def estimate_monthly_cost(metrics, requests_per_second=10, hours_per_day=24, days_per_month=30):
    """Estimate monthly cost for a model.
    
    Args:
        metrics: Dictionary with model metrics
        requests_per_second: Average number of requests per second
        hours_per_day: Hours of operation per day
        days_per_month: Days of operation per month
        
    Returns:
        Dictionary with cost estimates
    """
    # Cost parameters (approximate)
    instance_cost_per_hour = 0.5  # $0.50 per hour for ml.c5.xlarge
    storage_cost_per_gb_month = 0.023  # $0.023 per GB-month for S3 Standard
    
    # Calculate monthly requests
    monthly_requests = requests_per_second * 3600 * hours_per_day * days_per_month
    
    # Calculate compute cost based on inference time
    inference_time_seconds = metrics["inference_time"] / 1000  # Convert ms to seconds
    compute_hours = (inference_time_seconds * monthly_requests) / 3600
    compute_cost = compute_hours * instance_cost_per_hour
    
    # Calculate storage cost
    storage_gb = metrics["model_size"] / 1024  # Convert MB to GB
    storage_cost = storage_gb * storage_cost_per_gb_month
    
    # Total cost
    total_cost = compute_cost + storage_cost
    
    return {
        "compute_cost": compute_cost,
        "storage_cost": storage_cost,
        "total_cost": total_cost
    }

def load_model_and_tokenizer(model_name, task):
    """Load model and tokenizer based on task.
    
    Args:
        model_name: Name of the model to load
        task: Task type (sequence-classification, token-classification, etc.)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
    from transformers import AutoModelForMaskedLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if task == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task == "masked-lm":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return model, tokenizer

def prepare_inputs(task, tokenizer, sample_input):
    """Prepare inputs for different model tasks.
    
    Args:
        task: Task type (sequence-classification, token-classification, etc.)
        tokenizer: Tokenizer to use
        sample_input: Sample input text or dictionary
        
    Returns:
        Dictionary of input tensors
    """
    if task == "sequence-classification":
        inputs = tokenizer(sample_input, return_tensors="pt")
    elif task == "token-classification":
        inputs = tokenizer(sample_input, return_tensors="pt")
    elif task == "question-answering":
        inputs = tokenizer(
            sample_input["question"],
            sample_input["context"],
            return_tensors="pt"
        )
    elif task == "masked-lm":
        inputs = tokenizer(sample_input, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return inputs

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

def measure_memory_usage(model, inputs):
    """Measure peak memory usage during inference.
    
    Args:
        model: The model to measure
        inputs: Input tensors for the model
        
    Returns:
        Memory usage in megabytes
    """
    # Record baseline memory
    baseline = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    
    # Run inference
    _ = model(**inputs)
    
    # Record peak memory
    peak = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    
    return peak - baseline

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
    
    # Create bar plot
    bars = plt.bar(models, values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt

def calculate_cost_savings(baseline_metrics, optimized_metrics, requests_per_month=1000000):
    """Calculate cost savings from model optimization.
    
    Args:
        baseline_metrics: Dictionary with baseline model metrics
        optimized_metrics: Dictionary with optimized model metrics
        requests_per_month: Number of inference requests per month
        
    Returns:
        Dictionary with cost savings information
    """
    # Assumptions
    compute_cost_per_hour = 0.5  # $0.5 per hour for compute
    storage_cost_per_gb_month = 0.023  # $0.023 per GB-month for S3
    
    # Calculate baseline costs
    baseline_inference_time_hours = (baseline_metrics['inference_time'] * requests_per_month) / (1000 * 60 * 60)
    baseline_compute_cost = baseline_inference_time_hours * compute_cost_per_hour
    baseline_storage_cost = (baseline_metrics['model_size'] / 1024) * storage_cost_per_gb_month
    baseline_total_cost = baseline_compute_cost + baseline_storage_cost
    
    # Calculate optimized costs
    optimized_inference_time_hours = (optimized_metrics['inference_time'] * requests_per_month) / (1000 * 60 * 60)
    optimized_compute_cost = optimized_inference_time_hours * compute_cost_per_hour
    optimized_storage_cost = (optimized_metrics['model_size'] / 1024) * storage_cost_per_gb_month
    optimized_total_cost = optimized_compute_cost + optimized_storage_cost
    
    # Calculate savings
    savings = {
        'compute_savings': baseline_compute_cost - optimized_compute_cost,
        'storage_savings': baseline_storage_cost - optimized_storage_cost,
        'total_savings': baseline_total_cost - optimized_total_cost,
        'savings_percentage': (baseline_total_cost - optimized_total_cost) / baseline_total_cost * 100
    }
    
    return savings

def estimate_monthly_cost(model_metrics, requests_per_month=1000000):
    """Estimate monthly cost for running a model in production.
    
    Args:
        model_metrics: Dictionary with model metrics
        requests_per_month: Number of inference requests per month
        
    Returns:
        Dictionary with cost information
    """
    # Assumptions
    compute_cost_per_hour = 0.5  # $0.5 per hour for compute (e.g., ml.g4dn.xlarge)
    storage_cost_per_gb_month = 0.023  # $0.023 per GB-month for S3
    
    # Calculate compute cost
    inference_time_hours = (model_metrics["inference_time"] * requests_per_month) / (1000 * 60 * 60)
    compute_cost = inference_time_hours * compute_cost_per_hour
    
    # Calculate storage cost
    storage_cost = (model_metrics["model_size"] / 1024) * storage_cost_per_gb_month
    
    # Total cost
    total_cost = compute_cost + storage_cost
    
    return {
        "compute_cost": compute_cost,
        "storage_cost": storage_cost,
        "total_cost": total_cost
    }

def load_model_and_tokenizer(model_path, task, device=None):
    """Load model and tokenizer from local path.
    
    Args:
        model_path: Path to the model directory
        task: Task type (sequence-classification, token-classification, etc.)
        device: Device to load the model on (defaults to GPU if available)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForMaskedLM
    )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model based on task
    if task == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif task == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    elif task == "masked-lm":
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, tokenizer

def prepare_inputs(task, sample_input, tokenizer, device):
    """Prepare inputs for the model based on task.
    
    Args:
        task: Task type (sequence-classification, token-classification, etc.)
        sample_input: Input text or dictionary
        tokenizer: Tokenizer to use
        device: Device to place tensors on
        
    Returns:
        Dictionary of input tensors
    """
    if task == "sequence-classification" or task == "token-classification":
        inputs = tokenizer(sample_input, return_tensors="pt")
    elif task == "question-answering":
        inputs = tokenizer(sample_input["question"], sample_input["context"], return_tensors="pt")
    elif task == "masked-lm":
        inputs = tokenizer(sample_input, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def package_model_for_sagemaker(model_path, s3_bucket, model_name):
    """Package a model for SageMaker deployment.
    
    Args:
        model_path: Path to the model directory
        s3_bucket: S3 bucket to upload to
        model_name: Name for the model
        
    Returns:
        S3 URI for the packaged model
    """
    import tarfile
    import tempfile
    import shutil
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy model files to temporary directory
    for item in os.listdir(model_path):
        s = os.path.join(model_path, item)
        d = os.path.join(model_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    # Create tar.gz file
    tar_path = os.path.join(temp_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname="")
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_key = f"models/{model_name.replace('/', '-')}/model.tar.gz"
    s3_client.upload_file(tar_path, s3_bucket, s3_key)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return f"s3://{s3_bucket}/{s3_key}"

def plot_metric_comparison(df, metric, title, ylabel, higher_is_better=False):
    """Plot comparison for a specific metric.
    
    Args:
        df: DataFrame with comparison data
        metric: Column name for the metric to plot
        title: Plot title
        ylabel: Y-axis label
        higher_is_better: Whether higher values are better
        
    Returns:
        None (displays plot)
    """
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar plot
    sns.barplot(x="Model", y=metric, hue="Type", data=df)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

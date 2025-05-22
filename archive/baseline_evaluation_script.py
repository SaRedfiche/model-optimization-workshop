"""
Baseline evaluation script for model optimization workshop.
This script evaluates baseline performance metrics for transformer models.
"""

import sys
import subprocess

# Install required packages
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "datasets"])
print("Packages installed successfully.")

import os
import json
import torch
import argparse
import numpy as np
import logging
import traceback
import sys
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_num_parameters(model):
    """Calculate number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, inputs, num_runs=10):
    """Measure average inference time over multiple runs."""
    # Warm-up run
    with torch.no_grad():
        model(**inputs)
    
    # Measure inference time
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    inference_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            start_event.record()
            with torch.no_grad():
                model(**inputs)
            end_event.record()
            torch.cuda.synchronize()
            inference_times.append(start_event.elapsed_time(end_event))
        else:
            start_time = time.time()
            with torch.no_grad():
                model(**inputs)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return sum(inference_times) / len(inference_times)

def measure_memory_usage(model, inputs):
    """Measure peak memory usage during inference."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            model(**inputs)
        
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    else:
        # For CPU, use a rough estimate based on model size
        memory_usage = get_model_size(model) * 2  # Rough estimate
    
    return memory_usage

def prepare_sample_inputs(model_name, task, tokenizer, device):
    """Prepare sample inputs for the model based on its task."""
    if task == "sequence-classification":
        text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task == "masked-lm":
        text = "The [MASK] is a large language model trained by OpenAI."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def evaluate_model(model_key, model_info, output_dir):
    """Evaluate baseline performance metrics for a model."""
    logger.info(f"Evaluating model: {model_key}")
    
    model_name = model_info["model_name"]
    task = model_info["task"]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on task
    logger.info(f"Loading model: {model_name} for task: {task}")
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
    
    model = model.to(device)
    model.eval()
    
    # Prepare sample inputs
    inputs = prepare_sample_inputs(model_name, task, tokenizer, device)
    
    # Measure model size
    model_size = get_model_size(model)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Measure number of parameters
    num_parameters = get_num_parameters(model)
    logger.info(f"Number of parameters: {num_parameters:,}")
    
    # Measure inference time
    inference_time = measure_inference_time(model, inputs)
    logger.info(f"Average inference time: {inference_time:.2f} ms")
    
    # Measure memory usage
    memory_usage = measure_memory_usage(model, inputs)
    logger.info(f"Memory usage: {memory_usage:.2f} MB")
    
    # Save metrics
    metrics = {
        "model_name": model_name,
        "task": task,
        "model_size": round(model_size, 2),
        "num_parameters": num_parameters,
        "inference_time": round(inference_time, 2),
        "memory_usage": round(memory_usage, 2)
    }
    
    return metrics

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Baseline evaluation script")
parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics")
args = parser.parse_args()

try:
    # Load model info
    logger.info(f"Loading model info from {args.model_info_path}")
    with open(args.model_info_path, "r") as f:
        model_info = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each model
    all_metrics = {}
    for model_key, info in model_info.items():
        try:
            metrics = evaluate_model(model_key, info, args.output_dir)
            all_metrics[model_key] = metrics
        except Exception as e:
            logger.error(f"Error evaluating model {model_key}: {e}")
            logger.error(traceback.format_exc())
    
    # Save all metrics to a single file
    metrics_path = os.path.join(args.output_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Saved baseline metrics to {metrics_path}")

except Exception as e:
    logger.error(f"Error in baseline evaluation: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

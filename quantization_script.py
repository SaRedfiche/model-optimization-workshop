#!/usr/bin/env python3
"""
Quantization script for model optimization workshop.
This script applies quantization to transformer models and measures performance metrics.
"""

import sys
import subprocess

# Install required packages
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "datasets", "torch"])
print("Packages installed successfully.")

import os
import json
import torch
import argparse
import numpy as np
import logging
import traceback
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
    inference_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
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

def prepare_sample_inputs(model_name, task, tokenizer, device):
    """Prepare sample inputs for the model based on its task."""
    if task == "text-classification":
        text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task == "fill-mask":
        text = "The [MASK] is a large language model trained by OpenAI."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def apply_dynamic_quantization(model, bits=8):
    """Apply dynamic quantization to the model."""
    logger.info(f"Applying dynamic quantization with {bits} bits")
    
    # Check if quantization is supported for this model
    if not hasattr(torch.quantization, 'quantize_dynamic'):
        logger.warning("Dynamic quantization not supported in this PyTorch version")
        return model
    
    try:
        # For PyTorch >= 1.6.0
        if bits == 8:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            logger.warning(f"Bit width {bits} not supported for dynamic quantization, using 8-bit")
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        return quantized_model
    except Exception as e:
        logger.error(f"Error during dynamic quantization: {e}")
        logger.error(traceback.format_exc())
        return model

def apply_static_quantization(model, bits=8):
    """Apply static quantization to the model."""
    logger.info(f"Static quantization with {bits} bits is not fully implemented for transformer models")
    logger.info("Falling back to dynamic quantization")
    return apply_dynamic_quantization(model, bits)

def apply_quantization_aware_training(model, bits=8):
    """Apply quantization-aware training to the model."""
    logger.info(f"Quantization-aware training with {bits} bits is not implemented in this script")
    logger.info("Falling back to dynamic quantization")
    return apply_dynamic_quantization(model, bits)

def save_checkpoint(output_dir, model_key, step, message):
    """Save a checkpoint with progress information."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_key}_step_{step}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "model_key": model_key,
            "step": step,
            "message": message,
            "timestamp": time.time()
        }, f)
    
    logger.info(f"Saved checkpoint: {checkpoint_file}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Quantization script")
parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics and models")
parser.add_argument("--quantization-method", type=str, default="dynamic", 
                    choices=["dynamic", "static", "qat"],
                    help="Quantization method to use")
parser.add_argument("--quantization-bits", type=int, default=8, 
                    choices=[8, 16],
                    help="Number of bits for quantization")
args = parser.parse_args()

# Load model info
logger.info(f"Loading model info from {args.model_info_path}")
with open(args.model_info_path, "r") as f:
    model_info = json.load(f)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Process each model
all_metrics = {}
for model_key, info in model_info.items():
    try:
        logger.info(f"Processing model: {model_key}")
        save_checkpoint(args.output_dir, model_key, 1, "Starting quantization process")
        
        model_name = info["model_name"]
        task = info["task"]
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model based on task
        logger.info(f"Loading model: {model_name} for task: {task}")
        save_checkpoint(args.output_dir, model_key, 2, f"Loading model {model_name}")
        
        if task == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "token-classification":
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        elif task == "question-answering":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "fill-mask":
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        model = model.to(device)
        model.eval()
        
        # Prepare sample inputs
        inputs = prepare_sample_inputs(model_name, task, tokenizer, device)
        
        # Measure baseline metrics before quantization
        baseline_size = get_model_size(model)
        baseline_params = get_num_parameters(model)
        baseline_inference_time = measure_inference_time(model, inputs)
        
        logger.info(f"Baseline model size: {baseline_size:.2f} MB")
        logger.info(f"Baseline parameters: {baseline_params:,}")
        logger.info(f"Baseline inference time: {baseline_inference_time:.2f} ms")
        
        save_checkpoint(args.output_dir, model_key, 3, "Applying quantization")
        
        # Apply quantization based on method
        if args.quantization_method == "dynamic":
            quantized_model = apply_dynamic_quantization(model, args.quantization_bits)
        elif args.quantization_method == "static":
            quantized_model = apply_static_quantization(model, args.quantization_bits)
        elif args.quantization_method == "qat":
            quantized_model = apply_quantization_aware_training(model, args.quantization_bits)
        else:
            raise ValueError(f"Unsupported quantization method: {args.quantization_method}")
        
        # Measure metrics after quantization
        quantized_size = get_model_size(quantized_model)
        quantized_params = get_num_parameters(quantized_model)
        quantized_inference_time = measure_inference_time(quantized_model, inputs)
        
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Quantized parameters: {quantized_params:,}")
        logger.info(f"Quantized inference time: {quantized_inference_time:.2f} ms")
        
        # Calculate improvements
        size_reduction = (baseline_size - quantized_size) / baseline_size * 100
        time_improvement = (baseline_inference_time - quantized_inference_time) / baseline_inference_time * 100
        
        logger.info(f"Size reduction: {size_reduction:.2f}%")
        logger.info(f"Inference time improvement: {time_improvement:.2f}%")
        
        save_checkpoint(args.output_dir, model_key, 4, "Saving quantized model")
        
        # Save metrics
        metrics = {
            "model_name": model_name,
            "task": task,
            "quantization_method": args.quantization_method,
            "quantization_bits": args.quantization_bits,
            "baseline_size_mb": round(baseline_size, 2),
            "baseline_parameters": baseline_params,
            "baseline_inference_time_ms": round(baseline_inference_time, 2),
            "quantized_size_mb": round(quantized_size, 2),
            "quantized_parameters": quantized_params,
            "quantized_inference_time_ms": round(quantized_inference_time, 2),
            "size_reduction_percent": round(size_reduction, 2),
            "time_improvement_percent": round(time_improvement, 2)
        }
        
        all_metrics[model_key] = metrics
        
        # Save quantized model
        model_dir = os.path.join(args.output_dir, f"{model_key}_quantized")
        os.makedirs(model_dir, exist_ok=True)
        quantized_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Saved quantized model to {model_dir}")
        
        save_checkpoint(args.output_dir, model_key, 5, "Quantization complete")
        
    except Exception as e:
        logger.error(f"Error processing model {model_key}: {e}")
        logger.error(traceback.format_exc())
        save_checkpoint(args.output_dir, model_key, "error", f"Error: {str(e)}")

# Save all metrics to a single file
metrics_path = os.path.join(args.output_dir, "quantization_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)

logger.info(f"Saved quantization metrics to {metrics_path}")

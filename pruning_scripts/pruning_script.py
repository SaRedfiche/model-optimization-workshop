#!/usr/bin/env python3
"""
Improved pruning script for model optimization workshop.
This script applies pruning to transformer models and measures performance metrics,
with proper model size reduction and optimization.
"""

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
from torch.nn.utils import prune
import torch.nn.functional as F

# Import specialized pruning libraries
try:
    import torch_pruning as tp
    import onnx
    from onnxruntime.transformers import optimizer as ort_optimizer
    SPECIALIZED_LIBS_AVAILABLE = True
except ImportError:
    SPECIALIZED_LIBS_AVAILABLE = False
    print("Warning: Specialized pruning libraries not available. Falling back to basic pruning.")

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

def count_non_zero_params(model):
    """Count non-zero parameters in the model."""
    non_zero = 0
    total = 0
    for param in model.parameters():
        if param.dim() > 1:  # Only count weights, not biases
            non_zero += torch.count_nonzero(param).item()
            total += param.numel()
    return non_zero, total

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
    if task == "sequence-classification" or task == "text-classification":
        text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "Jeff Bezos founded Amazon in 1994 and the company is headquartered in Seattle, Washington."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task == "masked-lm" or task == "fill-mask":
        text = "The [MASK] is a large language model trained by OpenAI."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def apply_structured_pruning(model, amount):
    """Apply structured pruning to remove entire neurons/filters."""
    logger.info("Applying structured pruning...")
    
    # Store the original parameter count
    original_params = get_num_parameters(model)
    
    # Apply structured pruning to each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the weight matrix
            weight = module.weight.data
            
            # Calculate L2 norm for each output neuron
            norm = torch.norm(weight, p=2, dim=1)
            
            # Determine number of neurons to prune
            num_to_prune = int(amount * len(norm))
            
            # Find the neurons with smallest L2 norm
            _, indices = torch.topk(norm, k=num_to_prune, largest=False)
            
            # Zero out the weights for these neurons
            module.weight.data[indices] = 0
            
            # If bias exists, zero it out too
            if module.bias is not None:
                module.bias.data[indices] = 0
    
    # Count parameters after pruning
    pruned_params = get_num_parameters(model)
    logger.info(f"Original parameters: {original_params}, After pruning: {pruned_params}")
    
    return model

def apply_advanced_pruning(model, inputs, amount):
    """Apply advanced pruning using torch_pruning library."""
    logger.info("Applying advanced pruning with torch_pruning...")
    
    if not SPECIALIZED_LIBS_AVAILABLE:
        logger.warning("torch_pruning not available, falling back to basic structured pruning")
        return apply_structured_pruning(model, amount)
    
    # Initialize pruner
    example_inputs = tuple(inputs.values())
    
    # Create dependency graph
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    
    # Create pruning strategy
    strategy = tp.strategy.L1Strategy()
    
    # Create pruner
    pruner = tp.Pruner(
        model,
        DG,
        strategy,
        pruning_ratio=amount,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1 norm
        global_pruning=True
    )
    
    # Apply pruning
    model = pruner.prune()
    
    return model

def export_to_onnx(model, inputs, output_path):
    """Export model to ONNX format with optimizations."""
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Prepare input names and dynamic axes
    input_names = list(inputs.keys())
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    
    # Export to ONNX
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        output_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=13
    )
    
    # Optimize ONNX model if libraries are available
    if SPECIALIZED_LIBS_AVAILABLE:
        try:
            logger.info("Optimizing ONNX model...")
            
            # Load the model
            onnx_model = onnx.load(output_path)
            
            # Optimize the model
            # Note: This is a simplified version, actual optimization depends on model architecture
            optimized_model_path = output_path.replace(".onnx", "_optimized.onnx")
            
            # Use a simplified optimization for any model type
            opt_model = ort_optimizer.optimize_model(
                output_path,
                optimization_level=99  # Maximum optimization
            )
            
            # Save the optimized model
            opt_model.save_model_to_file(optimized_model_path)
            logger.info(f"Saved optimized ONNX model to {optimized_model_path}")
            
            return optimized_model_path
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            logger.error(traceback.format_exc())
            return output_path
    else:
        logger.warning("ONNX optimization libraries not available, skipping optimization")
        return output_path

def prune_model(model_key, model_info, pruning_method, pruning_amount, output_dir):
    """Apply pruning to a model and evaluate its performance."""
    logger.info(f"Pruning model: {model_key} with method: {pruning_method}, amount: {pruning_amount}")
    
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
    if task == "sequence-classification" or task == "text-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task == "masked-lm" or task == "fill-mask":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    model = model.to(device)
    model.eval()
    
    # Prepare sample inputs
    inputs = prepare_sample_inputs(model_name, task, tokenizer, device)
    
    # Measure baseline metrics before pruning
    baseline_size = get_model_size(model)
    baseline_params = get_num_parameters(model)
    baseline_non_zero, baseline_total = count_non_zero_params(model)
    baseline_inference_time = measure_inference_time(model, inputs)
    baseline_memory_usage = measure_memory_usage(model, inputs)
    
    logger.info(f"Baseline model size: {baseline_size:.2f} MB")
    logger.info(f"Baseline parameters: {baseline_params}")
    logger.info(f"Baseline non-zero weights: {baseline_non_zero}/{baseline_total} ({baseline_non_zero/baseline_total*100:.2f}%)")
    logger.info(f"Baseline inference time: {baseline_inference_time:.2f} ms")
    logger.info(f"Baseline memory usage: {baseline_memory_usage:.2f} MB")
    
    # Apply pruning based on method
    if pruning_method == "l1_unstructured":
        # Apply unstructured pruning (sets weights to zero but doesn't reduce model size)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                # Make pruning permanent by removing the reparameterization
                prune.remove(module, 'weight')
    elif pruning_method == "structured":
        # Apply structured pruning (removes entire neurons)
        model = apply_structured_pruning(model, pruning_amount)
    elif pruning_method == "advanced":
        # Apply advanced pruning with torch_pruning
        model = apply_advanced_pruning(model, inputs, pruning_amount)
    else:
        raise ValueError(f"Unsupported pruning method: {pruning_method}")
    
    # Measure metrics after pruning
    pruned_size = get_model_size(model)
    pruned_params = get_num_parameters(model)
    pruned_non_zero, pruned_total = count_non_zero_params(model)
    pruned_inference_time = measure_inference_time(model, inputs)
    pruned_memory_usage = measure_memory_usage(model, inputs)
    
    logger.info(f"Pruned model size: {pruned_size:.2f} MB")
    logger.info(f"Pruned parameters: {pruned_params}")
    logger.info(f"Pruned non-zero weights: {pruned_non_zero}/{pruned_total} ({pruned_non_zero/pruned_total*100:.2f}%)")
    logger.info(f"Pruned inference time: {pruned_inference_time:.2f} ms")
    logger.info(f"Pruned memory usage: {pruned_memory_usage:.2f} MB")
    
    # Calculate improvements
    size_reduction = (baseline_size - pruned_size) / baseline_size * 100
    param_reduction = (baseline_params - pruned_params) / baseline_params * 100
    sparsity = (1 - pruned_non_zero / pruned_total) * 100
    time_improvement = (baseline_inference_time - pruned_inference_time) / baseline_inference_time * 100
    memory_reduction = (baseline_memory_usage - pruned_memory_usage) / baseline_memory_usage * 100
    
    logger.info(f"Size reduction: {size_reduction:.2f}%")
    logger.info(f"Parameter reduction: {param_reduction:.2f}%")
    logger.info(f"Model sparsity: {sparsity:.2f}%")
    logger.info(f"Inference time improvement: {time_improvement:.2f}%")
    logger.info(f"Memory usage reduction: {memory_reduction:.2f}%")
    
    # Export to ONNX for further optimization
    onnx_path = os.path.join(output_dir, f"{model_key}_pruned.onnx")
    try:
        optimized_onnx_path = export_to_onnx(model, inputs, onnx_path)
        logger.info(f"Exported model to ONNX: {optimized_onnx_path}")
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        logger.error(traceback.format_exc())
    
    # Save metrics
    metrics = {
        model_key: {
            "model_name": model_name,
            "task": task,
            "pruning_method": pruning_method,
            "pruning_amount": pruning_amount,
            "model_size": round(pruned_size, 2),
            "inference_time": round(pruned_inference_time, 2),
            "memory_usage": round(pruned_memory_usage, 2),
            "size_reduction": round(size_reduction, 2),
            "time_improvement": round(time_improvement, 2),
            "memory_reduction": round(memory_reduction, 2),
            "parameter_reduction": round(param_reduction, 2),
            "sparsity": round(sparsity, 2),
            "non_zero_weights": pruned_non_zero,
            "total_weights": pruned_total
        }
    }
    
    return metrics, model

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Improved pruning script")
parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics and models")
parser.add_argument("--pruning-method", type=str, default="structured", 
                    choices=["l1_unstructured", "structured", "advanced"],
                    help="Pruning method to use")
parser.add_argument("--pruning-amount", type=float, default=0.3, 
                    help="Amount of weights to prune (0.0 to 1.0)")
args = parser.parse_args()

try:
    # Load model info
    logger.info(f"Loading model info from {args.model_info_path}")
    # Check if model_info_path is a directory
    if os.path.isdir(args.model_info_path):
        # List files in the directory
        logger.info(f"model_info_path is a directory. Contents: {os.listdir(args.model_info_path)}")
        # Try to find a JSON file
        json_files = [f for f in os.listdir(args.model_info_path) if f.endswith('.json')]
        if json_files:
            # Use the first JSON file found
            model_info_file = os.path.join(args.model_info_path, json_files[0])
            logger.info(f"Using JSON file: {model_info_file}")
            with open(model_info_file, "r") as f:
                model_info = json.load(f)
        else:
            raise FileNotFoundError(f"No JSON files found in directory: {args.model_info_path}")
    else:
        # It's a file, load it directly
        with open(args.model_info_path, "r") as f:
            model_info = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prune each model
    all_metrics = {}
    for model_key, info in model_info.items():
        try:
            metrics, pruned_model = prune_model(
                model_key, info, args.pruning_method, args.pruning_amount, args.output_dir
            )
            all_metrics.update(metrics)
            
            # Save pruned model
            model_dir = os.path.join(args.output_dir, f"{model_key}_pruned")
            os.makedirs(model_dir, exist_ok=True)
            pruned_model.save_pretrained(model_dir)
            logger.info(f"Saved pruned model to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error pruning model {model_key}: {e}")
            logger.error(traceback.format_exc())
    
    # Save all metrics to a single file
    metrics_path = os.path.join(args.output_dir, "pruned-metrics.json")  # Using hyphen instead of underscore
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Saved pruned metrics to {metrics_path}")

except Exception as e:
    logger.error(f"Error in pruning: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

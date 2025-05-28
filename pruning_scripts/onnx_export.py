#!/usr/bin/env python3
"""
ONNX export module for optimized models.
This module provides functions to export PyTorch models to ONNX format with optimizations.
"""

import os
import torch
import onnx
import logging
import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    from onnxruntime.transformers import optimizer
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    logger.warning("ONNX Runtime not available. ONNX optimization will be limited.")

def prepare_dummy_inputs(model_name, task, tokenizer, device):
    """Prepare dummy inputs for ONNX export based on the task."""
    if task in ["sequence-classification", "text-classification"]:
        text = "This is a sample text for ONNX export."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "token-classification":
        text = "John Smith works at Amazon in Seattle."
        inputs = tokenizer(text, return_tensors="pt")
    elif task == "question-answering":
        question = "What is machine learning?"
        context = "Machine learning is a field of AI."
        inputs = tokenizer(question, context, return_tensors="pt")
    elif task in ["masked-lm", "fill-mask"]:
        text = "The [MASK] is a large language model."
        inputs = tokenizer(text, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Move inputs to the appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def export_to_onnx(model, inputs, output_path, task_type):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        inputs: Sample inputs for tracing
        output_path: Path to save the ONNX model
        task_type: Type of task (classification, qa, etc.)
        
    Returns:
        Path to the exported ONNX model
    """
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Prepare input names and dynamic axes
    input_names = list(inputs.keys())
    output_names = ["logits"]
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes.update({"logits": {0: "batch_size"}})
    
    # Set model to evaluation mode
    model.eval()
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
    
    logger.info(f"Model exported to {output_path}")
    
    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")
    
    return output_path

def optimize_onnx_model(onnx_path, task_type):
    """
    Optimize ONNX model using ONNX Runtime.
    
    Args:
        onnx_path: Path to the ONNX model
        task_type: Type of task (classification, qa, etc.)
        
    Returns:
        Path to the optimized ONNX model
    """
    if not ONNX_RUNTIME_AVAILABLE:
        logger.warning("ONNX Runtime not available. Skipping optimization.")
        return onnx_path
    
    logger.info(f"Optimizing ONNX model: {onnx_path}")
    
    # Define optimization options based on task type
    if task_type in ["sequence-classification", "text-classification", "token-classification"]:
        model_type = "bert"
    elif task_type == "question-answering":
        model_type = "bert_qa"
    elif task_type in ["masked-lm", "fill-mask"]:
        model_type = "bert_mlm"
    else:
        model_type = "bert"  # Default to BERT
    
    # Create optimized model path
    optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
    
    try:
        # Optimize the model
        opt_model = optimizer.optimize_model(
            onnx_path,
            model_type=model_type,
            num_heads=12,  # Typical for BERT-base models
            hidden_size=768,  # Typical for BERT-base models
            optimization_options=optimizer.OptimizationOptions(
                enable_gelu=True,
                enable_layer_norm=True,
                enable_attention=True,
                enable_skip_layer_norm=True,
                enable_embed_layer_norm=True,
                enable_bias_skip_layer_norm=True,
                enable_bias_gelu=True,
                enable_gelu_approximation=True,
                enable_gemm_fast_gelu=True,
                enable_shape_inference=True,
                enable_math=True,
                enable_cast=True
            )
        )
        
        # Save the optimized model
        opt_model.save_model_to_file(optimized_path)
        logger.info(f"Optimized model saved to {optimized_path}")
        
        return optimized_path
    except Exception as e:
        logger.error(f"Error optimizing ONNX model: {e}")
        return onnx_path

def quantize_onnx_model(onnx_path):
    """
    Quantize ONNX model to int8.
    
    Args:
        onnx_path: Path to the ONNX model
        
    Returns:
        Path to the quantized ONNX model
    """
    if not ONNX_RUNTIME_AVAILABLE:
        logger.warning("ONNX Runtime not available. Skipping quantization.")
        return onnx_path
    
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    logger.info(f"Quantizing ONNX model: {onnx_path}")
    
    # Create quantized model path
    quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
    
    try:
        # Quantize the model
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QInt8
        )
        
        logger.info(f"Quantized model saved to {quantized_path}")
        return quantized_path
    except Exception as e:
        logger.error(f"Error quantizing ONNX model: {e}")
        return onnx_path

def export_and_optimize_model(model, tokenizer, model_name, task, output_dir):
    """
    Export, optimize, and quantize a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        tokenizer: Tokenizer for the model
        model_name: Name of the model
        task: Task type
        output_dir: Directory to save the models
        
    Returns:
        Dictionary with paths to the exported models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare dummy inputs
    inputs = prepare_dummy_inputs(model_name, task, tokenizer, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}.onnx")
    exported_path = export_to_onnx(model, inputs, onnx_path, task)
    
    # Optimize ONNX model
    optimized_path = optimize_onnx_model(exported_path, task)
    
    # Quantize ONNX model
    quantized_path = quantize_onnx_model(optimized_path)
    
    return {
        "original": exported_path,
        "optimized": optimized_path,
        "quantized": quantized_path
    }

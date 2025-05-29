#!/usr/bin/env python3
"""
Quantization script for SageMaker Processing jobs.
This script quantizes a Hugging Face model using ONNX and saves the quantized model.
"""

import os
import sys
import logging
import argparse
import shutil
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description='Quantize a Hugging Face model')
parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/input/model',
                    help='Directory containing the model to quantize')
parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output',
                    help='Directory to save the quantized model')
parser.add_argument('--quantization-approach', type=str, default='dynamic',
                    choices=['dynamic', 'static'],
                    help='Quantization approach to use')
parser.add_argument('--bits', type=int, default=8,
                    choices=[8],
                    help='Bit precision for quantization')

args = parser.parse_args()

logger.info(f"Starting quantization with the following parameters:")
logger.info(f"Model directory: {args.model_dir}")
logger.info(f"Output directory: {args.output_dir}")
logger.info(f"Quantization approach: {args.quantization_approach}")
logger.info(f"Bits: {args.bits}")

try:
    # Import required libraries
    logger.info("Importing required libraries...")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import onnx
    import onnxruntime
    import numpy as np
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory {args.model_dir} does not exist")
    
    # List contents of model directory
    logger.info(f"Contents of model directory: {os.listdir(args.model_dir)}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a temporary directory for ONNX export
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Export model to ONNX
    logger.info("Exporting model to ONNX format...")
    
    # Create dummy input for tracing
    batch_size = 1
    sequence_length = 128
    input_shape = (batch_size, sequence_length)
    
    # Determine input features based on model architecture
    logger.info(f"Model architecture: {model.config.model_type}")
    
    # Default input names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Create dummy inputs
    dummy_inputs = {
        "input_ids": torch.ones(input_shape, dtype=torch.long),
        "attention_mask": torch.ones(input_shape, dtype=torch.long),
    }
    
    # Add token_type_ids for models that use it (like BERT, but not DistilBERT)
    # Check if the model architecture is known to use token_type_ids
    if model.config.model_type.lower() in ["bert", "electra", "albert"]:
        input_names.append("token_type_ids")
        dummy_inputs["token_type_ids"] = torch.zeros(input_shape, dtype=torch.long)
    
    # Export the model to ONNX
    onnx_path = os.path.join(temp_dir, "model.onnx")
    
    # Create dynamic axes dictionary
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
    
    # Add token_type_ids dynamic axes if it's in the inputs
    if "token_type_ids" in input_names:
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
    
    # Export to ONNX
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=12,
        do_constant_folding=True
    )
    
    logger.info(f"Model exported to ONNX at {onnx_path}")
    
    # Verify the ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification successful")
    
    # Quantize the model
    logger.info(f"Quantizing model using {args.quantization_approach} approach...")
    
    # Import the quantization tools
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    
    # Define the quantized model path
    quantized_model_path = os.path.join(temp_dir, "model_quantized.onnx")
    
    # Perform quantization
    if args.quantization_approach == "dynamic":
        quantize_dynamic(
            onnx_path,
            quantized_model_path,
            weight_type=QuantType.QInt8
        )
    else:  # static quantization
        # For static quantization, we would need calibration data
        # Since we don't have calibration data, we'll use dynamic quantization as a fallback
        logger.warning("Static quantization requires calibration data. Using dynamic quantization as fallback.")
        quantize_dynamic(
            onnx_path,
            quantized_model_path,
            weight_type=QuantType.QInt8
        )
    
    logger.info(f"Model quantized and saved to {quantized_model_path}")
    
    # Copy the quantized model to the output directory
    logger.info("Copying quantized model to output directory...")
    shutil.copy(quantized_model_path, os.path.join(args.output_dir, "model.onnx"))
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)
    
    # Save model configuration
    logger.info("Saving model configuration...")
    model.config.save_pretrained(args.output_dir)
    
    # Create a config file for the quantized model
    quantization_config = {
        "quantization_approach": args.quantization_approach,
        "bits": args.bits,
        "original_model": model.config.model_type,
        "framework": "onnx",
        "quantization_library": "onnxruntime"
    }
    
    with open(os.path.join(args.output_dir, "quantization-config.json"), "w") as f:  # Using hyphen instead of underscore
        json.dump(quantization_config, f, indent=2)
    
    # Clean up temporary directory
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    # Verify output
    logger.info(f"Contents of output directory: {os.listdir(args.output_dir)}")
    logger.info("Quantization completed successfully!")

except Exception as e:
    logger.error(f"Error during quantization: {str(e)}", exc_info=True)
    sys.exit(1)

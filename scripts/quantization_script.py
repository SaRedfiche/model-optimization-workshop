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
                    choices=['dynamic', 'static', 'aware_training'],
                    help='Quantization approach to use')
parser.add_argument('--bits', type=int, default=8,
                    choices=[8, 4],
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
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    
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
    temp_onnx_dir = os.path.join(args.output_dir, "temp_onnx")
    os.makedirs(temp_onnx_dir, exist_ok=True)
    
    # Export model to ONNX
    logger.info("Exporting model to ONNX format...")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        args.model_dir, 
        export=True,
        provider="CPUExecutionProvider"
    )
    ort_model.save_pretrained(temp_onnx_dir)
    
    # Configure quantization
    logger.info(f"Configuring {args.bits}-bit {args.quantization_approach} quantization...")
    
    # Use simpler quantization configuration to avoid LRScheduler dependency
    if args.quantization_approach == "dynamic":
        from optimum.onnxruntime.configuration import OnnxQuantizationConfig
        quantization_config = OnnxQuantizationConfig(
            is_static=False,
            format="QOperator" if args.bits == 8 else "QDQ",
            mode="IntegerOps",
            activations_dtype="uint8",
            weights_dtype="int8" if args.bits == 8 else "int4",
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul", "Attention"]
        )
    elif args.quantization_approach == "static":
        from optimum.onnxruntime.configuration import OnnxQuantizationConfig
        quantization_config = OnnxQuantizationConfig(
            is_static=True,
            format="QOperator" if args.bits == 8 else "QDQ",
            mode="IntegerOps",
            activations_dtype="uint8",
            weights_dtype="int8" if args.bits == 8 else "int4",
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul", "Attention"]
        )
    else:  # aware_training - fallback to dynamic as QAT requires more setup
        logger.warning("QAT requires more setup, falling back to dynamic quantization")
        from optimum.onnxruntime.configuration import OnnxQuantizationConfig
        quantization_config = OnnxQuantizationConfig(
            is_static=False,
            format="QOperator" if args.bits == 8 else "QDQ",
            mode="IntegerOps",
            activations_dtype="uint8",
            weights_dtype="int8" if args.bits == 8 else "int4",
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul", "Attention"]
        )
    
    # Create quantizer
    logger.info("Creating quantizer...")
    quantizer = ORTQuantizer.from_pretrained(temp_onnx_dir)
    
    # Quantize model
    logger.info("Quantizing model...")
    quantizer.quantize(
        save_dir=args.output_dir,
        quantization_config=quantization_config
    )
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)
    
    # Clean up temporary directory
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_onnx_dir)
    
    # Verify output
    logger.info(f"Contents of output directory: {os.listdir(args.output_dir)}")
    logger.info("Quantization completed successfully!")

except Exception as e:
    logger.error(f"Error during quantization: {str(e)}", exc_info=True)
    sys.exit(1)

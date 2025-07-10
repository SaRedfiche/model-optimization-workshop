#!/usr/bin/env python3
"""
Quantization script for SageMaker Processing Jobs.
This script quantizes a Hugging Face model using ONNX Runtime and Optimum.
"""

import argparse
import os
import sys
import json
import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantize a Hugging Face model')
    parser.add_argument('--quantization-approach', type=str, default='dynamic',
                       choices=['dynamic', 'static'], 
                       help='Quantization approach to use')
    parser.add_argument('--bits', type=int, default=8,
                       choices=[8, 16], 
                       help='Number of bits for quantization')
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input/model',
                       help='Input directory containing the model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output',
                       help='Output directory for the quantized model')
    
    return parser.parse_args()

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    
    # Install optimum and onnxruntime
    os.system("pip install optimum[onnxruntime] onnx onnxruntime")
    
    logger.info("Dependencies installed successfully")

def quantize_model(args):
    """Quantize the model using Optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer
    import torch
    
    logger.info(f"Starting quantization with approach: {args.quantization_approach}, bits: {args.bits}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate input directory exists and contains model files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    model_files = list(input_path.glob("**/*.bin")) + list(input_path.glob("**/*.safetensors"))
    config_files = list(input_path.glob("**/config.json"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files (.bin or .safetensors) found in {input_path}")
    
    if not config_files:
        raise FileNotFoundError(f"No config.json found in {input_path}")
    
    model_path = str(input_path)
    logger.info(f"Using model from: {model_path}")
    
    # Load the model and tokenizer - fail if this doesn't work
    logger.info("Loading model and tokenizer...")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path, 
        from_transformers=True,
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded successfully")
    
    # Configure quantization
    if args.quantization_approach == "dynamic":
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False, 
            per_channel=False
        )
    else:
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=True, 
            per_channel=False
        )
    
    # Create quantizer
    quantizer = ORTQuantizer.from_pretrained(model)
    
    # Apply quantization
    logger.info("Applying quantization...")
    quantized_model_path = os.path.join(args.output_dir, "quantized_model")
    quantizer.quantize(
        save_dir=quantized_model_path,
        quantization_config=quantization_config
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(quantized_model_path)
    
    # Create model info file
    model_info = {
        "quantization_approach": args.quantization_approach,
        "bits": args.bits,
        "original_model": model_path,
        "quantized_model_path": quantized_model_path
    }
    
    with open(os.path.join(args.output_dir, "quantization_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Quantization completed successfully. Output saved to: {quantized_model_path}")

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting quantization process...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quantization approach: {args.quantization_approach}")
    logger.info(f"Bits: {args.bits}")
    
    # Install dependencies
    install_dependencies()
    
    # Quantize model
    quantize_model(args)
    
    logger.info("Quantization process completed successfully!")

if __name__ == "__main__":
    main()

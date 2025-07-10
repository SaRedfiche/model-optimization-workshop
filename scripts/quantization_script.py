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
import tarfile
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
    
    # Install from requirements file with Python 3.11 compatible versions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "quantization_requirements.txt")
    
    if os.path.exists(requirements_file):
        logger.info(f"Installing from {requirements_file}")
        os.system(f"pip install -r {requirements_file}")
    else:
        # Fallback to specific versions if requirements file not found
        logger.info("Requirements file not found, installing specific versions...")
        os.system("pip install 'optimum>=1.21.0,<2.0.0' 'onnx>=1.16.0,<2.0.0' 'onnxruntime>=1.18.0,<2.0.0'")
    
    logger.info("Dependencies installed successfully")

def quantize_model(args):
    """Quantize the model using Optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    import torch
    
    logger.info(f"Starting quantization with approach: {args.quantization_approach}, bits: {args.bits}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    # Check if we have a compressed model file (model.tar.gz)
    tar_files = list(input_path.glob("**/*.tar.gz"))
    if tar_files:
        logger.info(f"Found compressed model file: {tar_files[0]}")
        # Extract the tar.gz file
        extract_path = input_path / "extracted_model"
        os.makedirs(extract_path, exist_ok=True)
        
        with tarfile.open(tar_files[0], "r:gz") as tar:
            tar.extractall(extract_path)
        
        # Update model path to the extracted directory
        model_path = str(extract_path)
        logger.info(f"Extracted model to: {model_path}")
    else:
        # Check for direct model files
        model_files = list(input_path.glob("**/*.bin")) + list(input_path.glob("**/*.safetensors"))
        config_files = list(input_path.glob("**/config.json"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files (.bin, .safetensors, or .tar.gz) found in {input_path}")
        
        if not config_files:
            raise FileNotFoundError(f"No config.json found in {input_path}")
        
        model_path = str(input_path)
        logger.info(f"Using model from: {model_path}")
    
    # Check if this is a base model or already a classification model
    logger.info("Checking model configuration...")
    config = AutoConfig.from_pretrained(model_path)
    
    # If it's a base model (no num_labels), convert it to sequence classification
    if not hasattr(config, 'num_labels') or config.num_labels is None:
        logger.info("Base model detected - converting to sequence classification model")
        
        # Load as sequence classification model with 2 labels (binary classification)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save the converted model to a temporary directory
        temp_model_path = os.path.join(args.output_dir, "temp_classification_model")
        os.makedirs(temp_model_path, exist_ok=True)
        model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        
        # Now load with ORT - using correct API without from_transformers parameter
        logger.info("Loading converted model with ONNX Runtime...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            temp_model_path, 
            export=True
        )
        
        # Clean up temp directory
        shutil.rmtree(temp_model_path)
        
    else:
        logger.info("Classification model detected - loading directly")
        # Load the model and tokenizer directly - using correct API without from_transformers parameter
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
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
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    
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

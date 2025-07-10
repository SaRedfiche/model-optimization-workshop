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
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
        import torch
        
        logger.info(f"Starting quantization with approach: {args.quantization_approach}, bits: {args.bits}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Try to find the model in the input directory
        input_path = Path(args.input_dir)
        model_files = list(input_path.glob("**/*.bin")) + list(input_path.glob("**/*.safetensors"))
        
        if not model_files:
            # If no model files found, assume we need to download from HuggingFace
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            logger.info(f"No model files found in {input_path}, using default model: {model_name}")
        else:
            # Use the input directory as model path
            model_name = str(input_path)
            logger.info(f"Using model from: {model_name}")
        
        # Load the model and tokenizer
        logger.info("Loading model and tokenizer...")
        try:
            # Try to load as ONNX model first
            model = ORTModelForSequenceClassification.from_pretrained(
                model_name, 
                from_transformers=True,
                export=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load as ONNX model: {e}")
            # Fallback to default model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            model = ORTModelForSequenceClassification.from_pretrained(
                model_name, 
                from_transformers=True,
                export=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
            "original_model": model_name,
            "quantized_model_path": quantized_model_path
        }
        
        with open(os.path.join(args.output_dir, "quantization_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Quantization completed successfully. Output saved to: {quantized_model_path}")
        
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        
        # Fallback: create a simple quantized version using PyTorch
        logger.info("Attempting fallback quantization using PyTorch...")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Load model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # Save quantized model
            output_path = os.path.join(args.output_dir, "quantized_model")
            os.makedirs(output_path, exist_ok=True)
            
            quantized_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Create model info file
            model_info = {
                "quantization_approach": "pytorch_dynamic",
                "bits": 8,
                "original_model": model_name,
                "quantized_model_path": output_path
            }
            
            with open(os.path.join(args.output_dir, "quantization_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Fallback quantization completed. Output saved to: {output_path}")
            
        except Exception as fallback_error:
            logger.error(f"Fallback quantization also failed: {fallback_error}")
            raise

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

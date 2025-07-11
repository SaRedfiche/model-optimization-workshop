#!/usr/bin/env python3
"""
Pruning script for SageMaker Processing Jobs.
This script applies structured and unstructured pruning to Hugging Face models.
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
    parser = argparse.ArgumentParser(description='Prune a Hugging Face model')
    parser.add_argument('--pruning-approach', type=str, default='unstructured',
                       choices=['structured', 'unstructured'], 
                       help='Pruning approach to use')
    parser.add_argument('--sparsity', type=float, default=0.5,
                       help='Sparsity level (0.0 to 1.0)')
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input/model',
                       help='Input directory containing the model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output',
                       help='Output directory for the pruned model')
    
    return parser.parse_args()

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    
    # Install from requirements file with Python 3.10 compatible versions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "pruning_requirements.txt")
    
    if os.path.exists(requirements_file):
        logger.info(f"Installing from {requirements_file}")
        os.system(f"pip install -r {requirements_file}")
    else:
        # Fallback to specific versions if requirements file not found
        logger.info("Requirements file not found, installing specific versions...")
        os.system("pip install 'torch>=2.4.0,<3.0.0' 'transformers>=4.44.0,<5.0.0' 'numpy>=1.26.0,<2.0.0' 'torch-pruning>=1.4.0,<2.0.0'")
    
    logger.info("Dependencies installed successfully")

def prune_model(args):
    """Prune the model using PyTorch pruning utilities."""
    import torch
    import torch.nn.utils.prune as prune
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    logger.info(f"Starting pruning with approach: {args.pruning_approach}, sparsity: {args.sparsity}")
    
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
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded successfully")
    
    # Apply pruning
    logger.info(f"Applying {args.pruning_approach} pruning with {args.sparsity} sparsity...")
    
    if args.pruning_approach == "unstructured":
        # Apply unstructured pruning to linear layers
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=args.sparsity,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
            
    else:  # structured pruning
        # Apply structured pruning (remove entire neurons/channels)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                # Calculate number of neurons to prune
                num_neurons = module.weight.size(0)
                num_to_prune = int(num_neurons * args.sparsity)
                
                if num_to_prune > 0:
                    # Apply structured pruning
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=num_to_prune, 
                        n=2, 
                        dim=0
                    )
                    # Make pruning permanent
                    prune.remove(module, 'weight')
    
    # Calculate sparsity
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    actual_sparsity = zero_params / total_params
    logger.info(f"Achieved sparsity: {actual_sparsity:.4f}")
    
    # Save pruned model directly to output directory (flatten structure)
    logger.info("Saving pruned model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Create model info file
    model_info = {
        "pruning_approach": args.pruning_approach,
        "target_sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "original_model": model_path,
        "pruned_model_path": args.output_dir,
        "total_parameters": total_params,
        "zero_parameters": zero_params
    }
    
    with open(os.path.join(args.output_dir, "pruning_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Pruning completed successfully. Output saved to: {args.output_dir}")

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting pruning process...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Pruning approach: {args.pruning_approach}")
    logger.info(f"Sparsity: {args.sparsity}")
    
    # Install dependencies
    install_dependencies()
    
    # Prune model
    prune_model(args)
    
    logger.info("Pruning process completed successfully!")

if __name__ == "__main__":
    main()

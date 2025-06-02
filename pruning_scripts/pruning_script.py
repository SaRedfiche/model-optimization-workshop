#!/usr/bin/env python3
"""
Pruning script for model optimization workshop.
This script applies pruning to transformer models and saves the pruned model.
"""

import os
import json
import torch
import argparse
import logging
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM
from torch.nn.utils import prune

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def apply_structured_pruning(model, amount):
    """Apply structured pruning to remove entire neurons/filters."""
    logger.info("Applying structured pruning...")
    
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
    
    return model

def main():
    """Main function to run pruning."""
    parser = argparse.ArgumentParser(description="Pruning script")
    parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for pruned models")
    parser.add_argument("--pruning-method", type=str, default="structured", help="Pruning method to use")
    parser.add_argument("--pruning-amount", type=float, default=0.3, help="Amount of weights to prune (0.0 to 1.0)")
    args = parser.parse_args()
    
    try:
        # Load model info
        logger.info(f"Loading model info from {args.model_info_path}")
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
        
        # Process each model
        for model_key, info in model_info.items():
            try:
                model_name = info["model_name"]
                task = info.get("task", "text-classification")
                
                logger.info(f"Processing model: {model_name} for task: {task}")
                
                # Load tokenizer
                logger.info(f"Loading tokenizer: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load model based on task
                logger.info(f"Loading model: {model_name}")
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
                
                # Apply pruning
                logger.info(f"Applying {args.pruning_method} pruning with amount {args.pruning_amount}")
                if args.pruning_method == "structured":
                    model = apply_structured_pruning(model, args.pruning_amount)
                else:
                    raise ValueError(f"Unsupported pruning method: {args.pruning_method}")
                
                # Save pruned model
                pruned_model_dir = os.path.join(args.output_dir, f"{model_key}_pruned")
                os.makedirs(pruned_model_dir, exist_ok=True)
                model.save_pretrained(pruned_model_dir)
                tokenizer.save_pretrained(pruned_model_dir)
                logger.info(f"Saved pruned model to {pruned_model_dir}")
                
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        logger.error(f"Error in pruning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

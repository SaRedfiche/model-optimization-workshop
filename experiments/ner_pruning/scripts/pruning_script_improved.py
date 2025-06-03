#!/usr/bin/env python3
"""
Improved pruning script for model optimization workshop.
This script applies pruning to transformer models and saves the pruned model.
Added better error handling and debugging for token-classification models.
"""

import os
import json
import torch
import argparse
import logging
import sys
import traceback
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
    
    pruned_params = 0
    total_params = 0
    
    # Apply structured pruning to each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the weight matrix
            weight = module.weight.data
            
            # Calculate L2 norm for each output neuron
            norm = torch.norm(weight, p=2, dim=1)
            
            # Determine number of neurons to prune
            num_to_prune = int(amount * len(norm))
            pruned_params += num_to_prune
            total_params += len(norm)
            
            logger.info(f"Layer {name}: pruning {num_to_prune} out of {len(norm)} neurons")
            
            # Find the neurons with smallest L2 norm
            _, indices = torch.topk(norm, k=num_to_prune, largest=False)
            
            # Zero out the weights for these neurons
            module.weight.data[indices] = 0
            
            # If bias exists, zero it out too
            if module.bias is not None:
                module.bias.data[indices] = 0
    
    logger.info(f"Pruned {pruned_params} out of {total_params} neurons ({pruned_params/total_params*100:.2f}%)")
    return model

def main():
    """Main function to run pruning."""
    parser = argparse.ArgumentParser(description="Pruning script")
    parser.add_argument("--model-info-path", type=str, required=True, help="Path to model info JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for pruned models")
    parser.add_argument("--pruning-method", type=str, default="structured", help="Pruning method to use")
    parser.add_argument("--pruning-amount", type=float, default=0.3, help="Amount of weights to prune (0.0 to 1.0)")
    parser.add_argument("--debug", type=str, default="false", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set debug mode
    debug_mode = args.debug.lower() == "true"
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
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
        
        logger.info(f"Model info loaded: {json.dumps(model_info, indent=2)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created: {args.output_dir}")
        
        # Process each model
        for model_key, info in model_info.items():
            try:
                model_name = info["model_name"]
                task = info.get("task", "text-classification")
                
                logger.info(f"Processing model: {model_name} for task: {task}")
                
                # Load tokenizer
                logger.info(f"Loading tokenizer: {model_name}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    logger.info(f"Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading tokenizer: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Load model based on task
                logger.info(f"Loading model: {model_name}")
                try:
                    if task == "sequence-classification" or task == "text-classification":
                        model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    elif task == "token-classification":
                        logger.info("Loading token classification model...")
                        model = AutoModelForTokenClassification.from_pretrained(model_name)
                        logger.info("Token classification model loaded successfully")
                        # Print model structure for debugging
                        if debug_mode:
                            logger.debug(f"Model structure: {model}")
                            logger.debug(f"Model config: {model.config}")
                    elif task == "question-answering":
                        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                    elif task == "masked-lm" or task == "fill-mask":
                        model = AutoModelForMaskedLM.from_pretrained(model_name)
                    else:
                        raise ValueError(f"Unsupported task: {task}")
                    
                    logger.info(f"Model loaded successfully")
                    
                    # Print model size
                    model_size = sum(p.numel() for p in model.parameters())
                    logger.info(f"Model size: {model_size:,} parameters")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Apply pruning
                logger.info(f"Applying {args.pruning_method} pruning with amount {args.pruning_amount}")
                try:
                    if args.pruning_method == "structured":
                        model = apply_structured_pruning(model, args.pruning_amount)
                    else:
                        raise ValueError(f"Unsupported pruning method: {args.pruning_method}")
                    
                    logger.info(f"Pruning applied successfully")
                except Exception as e:
                    logger.error(f"Error applying pruning: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Save pruned model
                try:
                    pruned_model_dir = os.path.join(args.output_dir, f"{model_key}_pruned")
                    os.makedirs(pruned_model_dir, exist_ok=True)
                    logger.info(f"Saving pruned model to {pruned_model_dir}")
                    
                    model.save_pretrained(pruned_model_dir)
                    tokenizer.save_pretrained(pruned_model_dir)
                    
                    # Verify the saved files
                    saved_files = os.listdir(pruned_model_dir)
                    logger.info(f"Saved files: {saved_files}")
                    
                    # Also save as ONNX for compatibility with the notebook
                    try:
                        import torch.onnx
                        
                        # Create dummy input based on model type
                        batch_size = 1
                        seq_length = 128
                        dummy_input = torch.ones(batch_size, seq_length, dtype=torch.long)
                        
                        # Export to ONNX
                        onnx_path = os.path.join(args.output_dir, f"{model_key}_pruned.onnx")
                        torch.onnx.export(
                            model,
                            (dummy_input,),
                            onnx_path,
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'}}
                        )
                        logger.info(f"Saved ONNX model to {onnx_path}")
                    except Exception as e:
                        logger.error(f"Error saving ONNX model: {e}")
                        logger.error(traceback.format_exc())
                    
                    logger.info(f"Pruned model saved successfully")
                except Exception as e:
                    logger.error(f"Error saving pruned model: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error in pruning: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

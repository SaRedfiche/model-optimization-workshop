"""
Pruning script for model optimization workshop.
This script applies pruning to transformer models and measures performance metrics.
"""

import os
import json
import torch
import argparse
import numpy as np
import logging
import traceback
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM
from torch.nn.utils import prune

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pruning")

def load_model_and_tokenizer(model_name, task):
    """Load model and tokenizer based on task."""
    logger.info(f"Loading model {model_name} for task {task}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "token-classification":
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        elif task == "question-answering":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "masked-lm":
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        logger.info(f"Successfully loaded model and tokenizer")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        logger.error(traceback.format_exc())
        raise

def prepare_inputs(task, tokenizer, sample_input):
    """Prepare inputs for different model tasks."""
    logger.info(f"Preparing inputs for task {task}")
    try:
        if task == "sequence-classification":
            inputs = tokenizer(sample_input, return_tensors="pt")
        elif task == "token-classification":
            inputs = tokenizer(sample_input, return_tensors="pt")
        elif task == "question-answering":
            inputs = tokenizer(
                sample_input["question"],
                sample_input["context"],
                return_tensors="pt"
            )
        elif task == "masked-lm":
            inputs = tokenizer(sample_input, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        logger.info(f"Successfully prepared inputs")
        return inputs
    except Exception as e:
        logger.error(f"Error preparing inputs: {e}")
        logger.error(traceback.format_exc())
        raise

def measure_inference_time(model, inputs, num_runs=10):
    """Measure inference time for a model."""
    logger.info(f"Measuring inference time over {num_runs} runs")
    try:
        # Warm-up run
        with torch.no_grad():
            _ = model(**inputs)
        
        # Check if CUDA is available for timing
        if torch.cuda.is_available():
            # Measure inference time with CUDA events
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            timings = []
            with torch.no_grad():
                for i in range(num_runs):
                    start_time.record()
                    _ = model(**inputs)
                    end_time.record()
                    torch.cuda.synchronize()
                    timings.append(start_time.elapsed_time(end_time))
                    if i % 5 == 0:
                        logger.info(f"Completed {i+1}/{num_runs} inference runs")
        else:
            # Fallback to CPU timing
            import time
            timings = []
            with torch.no_grad():
                for i in range(num_runs):
                    start_time = time.time()
                    _ = model(**inputs)
                    end_time = time.time()
                    timings.append((end_time - start_time) * 1000)  # Convert to ms
                    if i % 5 == 0:
                        logger.info(f"Completed {i+1}/{num_runs} inference runs")
        
        avg_time = sum(timings) / len(timings)
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        return avg_time
    except Exception as e:
        logger.error(f"Error measuring inference time: {e}")
        logger.error(traceback.format_exc())
        raise

def get_model_size(model):
    """Get model size in MB."""
    try:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        logger.info(f"Model size: {size_mb:.2f} MB")
        return size_mb
    except Exception as e:
        logger.error(f"Error calculating model size: {e}")
        logger.error(traceback.format_exc())
        raise

def apply_pruning(model, pruning_method="l1_unstructured", amount=0.3):
    """Apply pruning to a model."""
    logger.info(f"Applying {pruning_method} pruning with amount {amount}")
    try:
        # Count prunable parameters before pruning
        prunable_params = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prunable_params += module.weight.numel()
        
        logger.info(f"Found {prunable_params} prunable parameters")
        
        # Apply pruning
        pruned_params = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    if pruning_method == "l1_unstructured":
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    elif pruning_method == "random_unstructured":
                        prune.random_unstructured(module, name='weight', amount=amount)
                    elif pruning_method == "ln_structured":
                        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                    else:
                        logger.warning(f"Unsupported pruning method: {pruning_method}, using l1_unstructured")
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    
                    # Make pruning permanent
                    prune.remove(module, 'weight')
                    pruned_params += int(module.weight.numel() * amount)
                except Exception as e:
                    logger.error(f"Error pruning module {name}: {e}")
                    logger.error(traceback.format_exc())
        
        logger.info(f"Pruned approximately {pruned_params} parameters ({pruned_params/prunable_params*100:.2f}% of prunable parameters)")
        return model
    except Exception as e:
        logger.error(f"Error applying pruning: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Returning original model")
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--pruning-method', type=str, default='l1_unstructured')
    parser.add_argument('--pruning-amount', type=float, default=0.3)
    args = parser.parse_args()
    
    logger.info(f"Starting pruning with arguments: {args}")
    
    try:
        # Check if output directory exists
        if not os.path.exists(args.output_dir):
            logger.info(f"Creating output directory: {args.output_dir}")
            os.makedirs(args.output_dir)
        
        # Check if model info file exists
        if not os.path.exists(args.model_info_path):
            logger.error(f"Model info file not found: {args.model_info_path}")
            raise FileNotFoundError(f"Model info file not found: {args.model_info_path}")
        
        # Load model info
        logger.info(f"Loading model info from {args.model_info_path}")
        with open(args.model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Process each model
        pruned_metrics = {}
        for model_key, info in model_info.items():
            logger.info(f"Processing {model_key}: {info['model_name']}")
            
            try:
                # Load model and tokenizer
                model, tokenizer = load_model_and_tokenizer(info['model_name'], info['task'])
                
                # Define sample input
                if info['task'] == 'sequence-classification':
                    sample_input = "This is a sample input for sentiment analysis."
                elif info['task'] == 'token-classification':
                    sample_input = "John Smith works at Microsoft in Seattle."
                elif info['task'] == 'question-answering':
                    sample_input = {
                        "question": "What is machine learning?",
                        "context": "Machine learning is a branch of artificial intelligence."
                    }
                elif info['task'] == 'masked-lm':
                    sample_input = "The [MASK] is a large language model."
                
                # Prepare inputs
                inputs = prepare_inputs(info['task'], tokenizer, sample_input)
                
                # Get original model size and inference time
                logger.info("Measuring metrics for original model")
                original_size = get_model_size(model)
                
                # Check available memory before moving to GPU
                if torch.cuda.is_available():
                    logger.info(f"CUDA available. Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                    logger.info(f"Current allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    logger.info(f"Current cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                    
                    # Move to GPU if available
                    try:
                        logger.info("Moving model to GPU")
                        model = model.to('cuda')
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        logger.info("Successfully moved model and inputs to GPU")
                    except RuntimeError as e:
                        logger.error(f"Error moving to GPU, likely out of memory: {e}")
                        logger.error(traceback.format_exc())
                        logger.info("Falling back to CPU")
                        # Ensure everything is on CPU
                        model = model.to('cpu')
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                original_inference_time = measure_inference_time(model, inputs)
                
                # Apply pruning
                pruned_model = apply_pruning(
                    model, 
                    pruning_method=args.pruning_method,
                    amount=args.pruning_amount
                )
                
                # Measure metrics for pruned model
                logger.info("Measuring metrics for pruned model")
                pruned_size = get_model_size(pruned_model)
                pruned_inference_time = measure_inference_time(pruned_model, inputs)
                num_parameters = sum(p.numel() for p in pruned_model.parameters())
                
                # Save pruned model
                output_dir = os.path.join(args.output_dir, model_key)
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Saving pruned model to {output_dir}")
                try:
                    pruned_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Successfully saved pruned model")
                except Exception as e:
                    logger.error(f"Error saving pruned model: {e}")
                    logger.error(traceback.format_exc())
                    # Try alternative saving method
                    try:
                        logger.info("Trying alternative saving method")
                        torch.save(pruned_model.state_dict(), os.path.join(output_dir, "pruned_model.pt"))
                        logger.info("Successfully saved pruned model state dict")
                    except Exception as e2:
                        logger.error(f"Error saving pruned model state dict: {e2}")
                        logger.error(traceback.format_exc())
                
                # Calculate improvements
                size_reduction = (original_size - pruned_size) / original_size * 100 if original_size > 0 else 0
                speedup = original_inference_time / pruned_inference_time if pruned_inference_time > 0 else 0
                
                # Save metrics
                pruned_metrics[model_key] = {
                    "model_key": model_key,
                    "model_name": info['model_name'],
                    "task": info['task'],
                    "pruning_method": args.pruning_method,
                    "pruning_amount": args.pruning_amount,
                    "model_size": pruned_size,
                    "original_size": original_size,
                    "inference_time": pruned_inference_time,
                    "original_inference_time": original_inference_time,
                    "num_parameters": num_parameters,
                    "size_reduction": size_reduction,
                    "speedup": speedup
                }
                
                logger.info(f"Successfully processed {model_key}")
                logger.info(f"Size reduction: {size_reduction:.2f}%, Speedup: {speedup:.2f}x")
                
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {e}")
                logger.error(traceback.format_exc())
                logger.info(f"Skipping model {model_key}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, 'pruned_metrics.json')
        logger.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(pruned_metrics, f, indent=2)
        
        logger.info("Pruning completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        # Exit with error code
        sys.exit(1)

if __name__ == '__main__':
    main()

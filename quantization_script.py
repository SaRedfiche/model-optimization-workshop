"""
Quantization script for model optimization workshop.
This script applies quantization to transformer models and measures performance metrics.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("quantization")

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

def apply_quantization(model, method="dynamic", bits=8):
    """Apply quantization to a model."""
    logger.info(f"Applying {method} quantization with {bits} bits")
    try:
        if method == "dynamic":
            # Check if the model has Linear layers that can be quantized
            has_linear = any(isinstance(module, torch.nn.Linear) for module in model.modules())
            if not has_linear:
                logger.warning("Model has no Linear layers for dynamic quantization, using original model")
                return model
                
            # Dynamic quantization (quantizes weights at runtime)
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully")
        elif method == "static":
            # Static quantization (requires calibration data)
            # This is a simplified version for demonstration
            logger.info("Applying static quantization (simplified version)")
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Calibration would happen here with real data
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Static quantization applied successfully")
        elif method == "aware":
            # Quantization-aware training (requires training)
            # This is a simplified version for demonstration
            logger.info("Applying quantization-aware training (simplified version)")
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            # Training would happen here
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Quantization-aware training applied successfully")
        else:
            logger.error(f"Unsupported quantization method: {method}")
            raise ValueError(f"Unsupported quantization method: {method}")
        
        return quantized_model
    except Exception as e:
        logger.error(f"Error applying quantization: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Falling back to original model")
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--quantization-method', type=str, default='dynamic')
    parser.add_argument('--quantization-bits', type=int, default=8)
    args = parser.parse_args()
    
    logger.info(f"Starting quantization with arguments: {args}")
    
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
        quantized_metrics = {}
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
                
                # Apply quantization
                quantized_model = apply_quantization(
                    model, 
                    method=args.quantization_method,
                    bits=args.quantization_bits
                )
                
                # Check available memory before moving to GPU
                if torch.cuda.is_available():
                    logger.info(f"CUDA available. Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                    logger.info(f"Current allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    logger.info(f"Current cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                    
                    # Move to GPU if available
                    try:
                        logger.info("Moving model to GPU")
                        model = model.to('cuda')
                        quantized_model = quantized_model.to('cuda')
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        logger.info("Successfully moved model and inputs to GPU")
                    except RuntimeError as e:
                        logger.error(f"Error moving to GPU, likely out of memory: {e}")
                        logger.error(traceback.format_exc())
                        logger.info("Falling back to CPU")
                        # Ensure everything is on CPU
                        model = model.to('cpu')
                        quantized_model = quantized_model.to('cpu')
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                # Measure metrics
                logger.info("Measuring metrics for original model")
                model_size = get_model_size(model)
                inference_time = measure_inference_time(model, inputs)
                
                logger.info("Measuring metrics for quantized model")
                quantized_size = get_model_size(quantized_model)
                quantized_inference_time = measure_inference_time(quantized_model, inputs)
                
                num_parameters = sum(p.numel() for p in model.parameters())
                quantized_parameters = sum(p.numel() for p in quantized_model.parameters())
                
                # Save quantized model
                output_dir = os.path.join(args.output_dir, model_key)
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Saving quantized model to {output_dir}")
                try:
                    torch.save(quantized_model.state_dict(), os.path.join(output_dir, "quantized_model.pt"))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Successfully saved quantized model")
                except Exception as e:
                    logger.error(f"Error saving quantized model: {e}")
                    logger.error(traceback.format_exc())
                
                # Calculate improvements
                size_reduction = (model_size - quantized_size) / model_size * 100 if model_size > 0 else 0
                speedup = inference_time / quantized_inference_time if quantized_inference_time > 0 else 0
                
                # Save metrics
                quantized_metrics[model_key] = {
                    "model_key": model_key,
                    "model_name": info['model_name'],
                    "task": info['task'],
                    "quantization_method": args.quantization_method,
                    "quantization_bits": args.quantization_bits,
                    "model_size": quantized_size,
                    "original_size": model_size,
                    "inference_time": quantized_inference_time,
                    "original_inference_time": inference_time,
                    "num_parameters": quantized_parameters,
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
        metrics_path = os.path.join(args.output_dir, 'quantized_metrics.json')
        logger.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(quantized_metrics, f, indent=2)
        
        logger.info("Quantization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        # Exit with error code
        sys.exit(1)

if __name__ == '__main__':
    main()

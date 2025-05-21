"""
Quantization script for model optimization workshop.
This script applies quantization to transformer models and measures performance metrics.
"""

import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM

def load_model_and_tokenizer(model_name, task):
    """Load model and tokenizer based on task."""
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
    
    return model, tokenizer

def prepare_inputs(task, tokenizer, sample_input):
    """Prepare inputs for different model tasks."""
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
    
    return inputs

def measure_inference_time(model, inputs, num_runs=10):
    """Measure inference time for a model."""
    # Warm-up run
    with torch.no_grad():
        _ = model(**inputs)
    
    # Measure inference time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time.record()
            _ = model(**inputs)
            end_time.record()
            torch.cuda.synchronize()
            timings.append(start_time.elapsed_time(end_time))
    
    return sum(timings) / len(timings)

def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def apply_quantization(model, method="dynamic", bits=8):
    """Apply quantization to a model."""
    if method == "dynamic":
        # Dynamic quantization (quantizes weights at runtime)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif method == "static":
        # Static quantization (requires calibration data)
        # This is a simplified version for demonstration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # Calibration would happen here with real data
        quantized_model = torch.quantization.convert(model, inplace=False)
    elif method == "aware":
        # Quantization-aware training (requires training)
        # This is a simplified version for demonstration
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        # Training would happen here
        quantized_model = torch.quantization.convert(model, inplace=False)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
    
    return quantized_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--quantization-method', type=str, default='dynamic')
    parser.add_argument('--quantization-bits', type=int, default=8)
    args = parser.parse_args()
    
    # Load model info
    with open(args.model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Process each model
    quantized_metrics = {}
    for model_key, info in model_info.items():
        print(f"Processing {model_key}: {info['model_name']}")
        
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
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            quantized_model = quantized_model.to('cuda')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Measure metrics
        model_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)
        inference_time = measure_inference_time(model, inputs)
        quantized_inference_time = measure_inference_time(quantized_model, inputs)
        num_parameters = sum(p.numel() for p in model.parameters())
        quantized_parameters = sum(p.numel() for p in quantized_model.parameters())
        
        # Save quantized model
        output_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(quantized_model.state_dict(), os.path.join(output_dir, "quantized_model.pt"))
        tokenizer.save_pretrained(output_dir)
        
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
            "size_reduction": (model_size - quantized_size) / model_size * 100,
            "speedup": inference_time / quantized_inference_time
        }
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'quantized_metrics.json'), 'w') as f:
        json.dump(quantized_metrics, f, indent=2)

if __name__ == '__main__':
    main()

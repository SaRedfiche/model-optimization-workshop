"""
Baseline evaluation script for model optimization workshop.
This script evaluates baseline performance metrics for transformer models.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load model info
    with open(args.model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Process each model
    baseline_metrics = {}
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
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Measure metrics
        model_size = get_model_size(model)
        inference_time = measure_inference_time(model, inputs)
        num_parameters = sum(p.numel() for p in model.parameters())
        
        # Save model for future use
        output_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save metrics
        baseline_metrics[model_key] = {
            "model_key": model_key,
            "model_name": info['model_name'],
            "task": info['task'],
            "model_size": model_size,
            "inference_time": inference_time,
            "num_parameters": num_parameters
        }
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=2)

if __name__ == '__main__':
    main()

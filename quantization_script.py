"""
Script for running quantization on a SageMaker Processing instance.
This script is designed to be run as a SageMaker Processing job.
"""

import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM
import torch.quantization
import boto3
import time

# Try to import Intel optimizations
try:
    from optimum.intel import INCQuantizer, INCModelForSequenceClassification
    from optimum.intel import INCModelForTokenClassification, INCModelForQuestionAnswering
    intel_optimum_available = True
    print("Intel Neural Compressor (optimum-intel) is available")
except ImportError:
    intel_optimum_available = False
    print("Intel Neural Compressor (optimum-intel) is not available. Will use PyTorch quantization only.")

def download_model_from_s3(s3_uri, local_path):
    """Download model from S3 to local path."""
    # Parse S3 URI
    s3_parts = s3_uri.replace("s3://", "").split("/")
    bucket = s3_parts[0]
    prefix = "/".join(s3_parts[1:])
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # List objects in the prefix
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    # Download each file
    os.makedirs(local_path, exist_ok=True)
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = os.path.basename(key)
        if filename:  # Skip directory entries
            local_file = os.path.join(local_path, filename)
            s3_client.download_file(bucket, key, local_file)
    
    print(f"Downloaded model from {s3_uri} to {local_path}")

def upload_model_to_s3(local_path, s3_uri):
    """Upload model from local path to S3."""
    # Parse S3 URI
    s3_parts = s3_uri.replace("s3://", "").split("/")
    bucket = s3_parts[0]
    prefix = "/".join(s3_parts[1:])
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Upload all files in the directory
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = f"{prefix}/{relative_path}"
            s3_client.upload_file(local_file, bucket, s3_key)
    
    print(f"Uploaded model from {local_path} to {s3_uri}")

def quantize_model_dynamic(model_path, task, output_path):
    """Apply dynamic quantization to a model."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare sample input based on task
    if task == "sequence-classification":
        sample_input = "This is a sample input for sequence classification."
    elif task == "token-classification":
        sample_input = "John Smith works at Amazon in Seattle."
    elif task == "question-answering":
        sample_input = {
            "question": "What is machine learning?",
            "context": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        }
    elif task == "masked-lm":
        sample_input = "The [MASK] is a large language model trained by OpenAI."
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Prepare inputs
    if task == "sequence-classification" or task == "token-classification":
        inputs = tokenizer(sample_input, return_tensors="pt")
    elif task == "question-answering":
        inputs = tokenizer(sample_input["question"], sample_input["context"], return_tensors="pt")
    elif task == "masked-lm":
        inputs = tokenizer(sample_input, return_tensors="pt")
    
    # Check if we can use Intel Neural Compressor for better CPU performance
    if intel_optimum_available:
        try:
            print("Using Intel Neural Compressor for quantization...")
            
            # Use Intel Neural Compressor for quantization
            quantizer = INCQuantizer.from_pretrained(model_path)
            
            # Apply dynamic quantization
            quantized_model = quantizer.quantize(save_directory=output_path, quantization_approach="dynamic")
            
            print(f"Model quantized with Intel Neural Compressor and saved to {output_path}")
            return
            
        except Exception as e:
            print(f"Intel Neural Compressor failed: {e}. Falling back to PyTorch quantization.")
    
    # Fall back to PyTorch quantization
    print("Using PyTorch dynamic quantization...")
    
    # Load model based on task
    if task == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif task == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif task == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    elif task == "masked-lm":
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    
    # Apply dynamic quantization
    model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_path)
    
    print(f"Model quantized with PyTorch and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Quantize a model using SageMaker Processing")
    parser.add_argument("--model-info-path", type=str, default="/opt/ml/processing/input/model_info/model_info.json")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output/quantized_model")
    
    args = parser.parse_args()
    
    # Load model info
    with open(args.model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Process each model
    for model_key, info in model_info.items():
        print(f"Processing model: {model_key}")
        
        # Download model from S3
        local_model_path = f"/tmp/models/{model_key}"
        download_model_from_s3(info["s3_uri"], local_model_path)
        
        # Create output path
        local_output_path = f"/tmp/quantized/{model_key}"
        
        # Quantize model
        quantize_model_dynamic(local_model_path, info["task"], local_output_path)
        
        # Upload quantized model to output location
        output_path = os.path.join(args.output_dir, model_key)
        upload_model_to_s3(local_output_path, output_path)
        
        print(f"Completed quantization for {model_key}")

if __name__ == "__main__":
    main()

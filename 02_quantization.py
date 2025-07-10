#!/usr/bin/env python
"""
Model Optimization Workshop: Model Quantization with SageMaker Processing

This script performs the same operations as the 02_quantization.ipynb notebook.
It demonstrates how to quantize a Hugging Face model using SageMaker Processing jobs.
"""

import os
import json
import time
import argparse

# AWS
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor

def setup_environment():
    """Set up the environment and return configuration."""
    # Load workshop configuration if available
    try:
        with open('workshop_config.json', 'r') as f:
            workshop_config = json.load(f)
        
        # Use configuration values
        base_model = workshop_config.get("base_model", "distilbert-base-uncased-finetuned-sst-2-english")
        task = workshop_config.get("task", "sequence-classification")
    except FileNotFoundError:
        # Default values if config not found
        base_model = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sequence-classification"
        print("Workshop configuration not found. Using default values.")
    
    # Set up SageMaker session
    try:
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        region = boto3.session.Session().region_name
        bucket = sagemaker_session.default_bucket()
        prefix = "quantization-workshop"
    except Exception as e:
        print(f"Error setting up SageMaker session: {e}")
        # Provide fallback values for testing
        sagemaker_session = None
        role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
        region = "us-west-2"
        bucket = "example-bucket"
        prefix = "quantization-workshop"
    
    print(f"SageMaker session established in region: {region}")
    print(f"Using S3 bucket: {bucket}")
    print(f"Using S3 prefix: {prefix}")
    
    return {
        "base_model": base_model,
        "task": task,
        "sagemaker_session": sagemaker_session,
        "role": role,
        "region": region,
        "bucket": bucket,
        "prefix": prefix
    }

def prepare_quantization_script():
    """Prepare the quantization script."""
    # Check if the script exists
    if not os.path.exists("scripts/quantization_script.py"):
        print("Quantization script not found. Creating directory...")
        os.makedirs("scripts", exist_ok=True)
        
        # Create the quantization script
        script_content = """
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import main_export

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--task", type=str, default="sequence-classification")
    parser.add_argument("--quantization_approach", type=str, default="dynamic")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Quantizing model: {args.model_id}")
    print(f"Task: {args.task}")
    print(f"Quantization approach: {args.quantization_approach}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export the model to ONNX
    print("Exporting model to ONNX format...")
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    main_export(
        args.model_id,
        output=onnx_path,
        task=args.task,
        opset=13
    )
    print(f"Model exported to: {onnx_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    
    # Create quantizer
    print("Creating quantizer...")
    quantizer = ORTQuantizer.from_pretrained(model)
    
    # Define quantization configuration
    if args.quantization_approach == "dynamic":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    elif args.quantization_approach == "static":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)
    else:
        raise ValueError(f"Unsupported quantization approach: {args.quantization_approach}")
    
    # Quantize the model
    print("Quantizing model...")
    quantizer.quantize(
        quantization_config=qconfig,
        save_dir=args.output_dir,
    )
    
    # Save the tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Quantized model and tokenizer saved to: {args.output_dir}")
    print("Quantization completed successfully!")

if __name__ == "__main__":
    main()
"""
        
        with open("scripts/quantization_script.py", "w") as f:
            f.write(script_content)
        
        print("Quantization script created successfully.")
    else:
        print("Quantization script already exists.")

def run_quantization_job(config, quantization_approach="dynamic"):
    """Run the quantization job."""
    sagemaker_session = config["sagemaker_session"]
    role = config["role"]
    bucket = config["bucket"]
    prefix = config["prefix"]
    base_model = config["base_model"]
    task = config["task"]
    
    if sagemaker_session is None:
        print("SageMaker session not available. Skipping quantization job.")
        return None
    
    # Create a PyTorch processor for running the quantization script
    pytorch_processor = PyTorchProcessor(
        framework_version="1.13.1",
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        base_job_name="quantization-job",
        sagemaker_session=sagemaker_session
    )
    
    # Define the output path
    output_path = f"s3://{bucket}/{prefix}/output"
    
    # Run the processing job
    job_name = f"quantization-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    print(f"Starting quantization job: {job_name}")
    
    pytorch_processor.run(
        code="quantization_script.py",
        source_dir="scripts",
        outputs=[
            ProcessingOutput(
                output_name="quantized_model",
                source="/opt/ml/processing/output",
                destination=output_path
            )
        ],
        arguments=[
            "--model_id", base_model,
            "--task", task,
            "--quantization_approach", quantization_approach
        ],
        job_name=job_name
    )
    
    print(f"Quantization job completed: {job_name}")
    print(f"Quantized model saved to: {output_path}")
    
    return {
        "job_name": job_name,
        "output_path": output_path
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantize a model using SageMaker Processing")
    parser.add_argument("--approach", choices=["dynamic", "static"], default="dynamic",
                        help="Quantization approach: dynamic or static")
    parser.add_argument("--skip-job", action="store_true", help="Skip the quantization job")
    return parser.parse_args()

def main():
    """Main function to run the quantization script."""
    args = parse_args()
    
    # Set up environment
    config = setup_environment()
    
    # Prepare quantization script
    prepare_quantization_script()
    
    # Run quantization job
    if not args.skip_job:
        job_info = run_quantization_job(config, args.approach)
        if job_info:
            print("\nQuantization Summary:")
            print(f"Job Name: {job_info['job_name']}")
            print(f"Output Path: {job_info['output_path']}")
    else:
        print("Skipping quantization job as requested.")

if __name__ == "__main__":
    main()

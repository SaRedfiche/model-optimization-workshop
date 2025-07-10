#!/usr/bin/env python
"""
Model Optimization Workshop: Introduction and Setup

This script performs the same operations as the 01_introduction_and_setup.ipynb notebook.
It sets up the environment and creates the workshop configuration.
"""

import os
import sys
import time
import json

# Data processing
import numpy as np
import pandas as pd

# Machine learning
import torch

# AWS
import boto3
import sagemaker

def main():
    """Main function to run the introduction and setup script."""
    # Print versions of key packages
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Boto3 version: {boto3.__version__}")
    print(f"SageMaker version: {sagemaker.__version__}")

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    # Set up SageMaker session
    try:
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        region = boto3.session.Session().region_name
        bucket = sagemaker_session.default_bucket()
        prefix = "model-optimization-workshop"

        print(f"SageMaker session established in region: {region}")
        print(f"Using S3 bucket: {bucket}")
        print(f"Using S3 prefix: {prefix}")
    except Exception as e:
        print(f"Error setting up SageMaker session: {e}")
        # Provide fallback values for testing
        region = "us-west-2"  # Default region
        bucket = "example-bucket"  # Example bucket name
        prefix = "model-optimization-workshop"  # Default prefix
        role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"  # Example role

    # Create workshop configuration
    workshop_config = {
        "base_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sequence-classification",
        "s3_bucket": bucket,
        "s3_prefix": prefix,
        "region": region,
        "role": role,
        "device_type": "GPU" if torch.cuda.is_available() else "CPU",
        "created_at": time.strftime("%Y-%m-%d-%H-%M-%S")
    }

    # Save configuration to file
    with open("workshop_config.json", "w") as f:
        json.dump(workshop_config, f, indent=2)

    print("Workshop configuration saved to workshop_config.json")
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()
